"""Pure-tensor GDPO-style advantage aggregation.

Standalone helpers that depend only on torch — kept separate from
``grpo.py`` (which transitively imports transformers / deepspeed / slyme)
so unit tests can exercise the algorithm on CPU-only environments without
loading the model stack.

References:
- NVIDIA GDPO (arxiv 2601.05242, Jan 2026): per-reward group-norm + batch-whiten
- ReMemR1 (arxiv 2509.23040): mean-only group-norm precedent (grpo_use_adv=False)
- Memex(RL) (arxiv 2603.04257): soft-trigger compress reward → overflow_pen
"""
from __future__ import annotations

from typing import Dict, List

import torch


# Reward keys (must match the column order of rewards_masks emitted by
# calc_rewards in grpo.py).
#
# v11.3: split recall signal into three independent columns so per-reward
# group-norm can normalise each separately. The previous mixed
# `recall_quality = outcome × (0.5×query_quality + 0.5×hit_rate)` collapsed
# two signals with different variance profiles into one column, which
# under-weighted the smaller-magnitude one after normalisation.
#
#   recall_quality   — JSON well-formed + no answer leakage (always applicable
#                      when recall fired)
#   recall_hit_rate  — |returned ∩ support| / |support| (only when both sides
#                      known; sparse)
#   range_tightness  — encourages a narrow + accurate time_range (sparse;
#                      only when recall used time_range AND hit_rate≥thresh)
REWARD_DICT_KEYS: tuple = (
    "format", "correctness", "timing", "silent_quality",
    "recall_quality", "recall_hit_rate", "range_tightness", "overflow_pen",
)

# Per-reward weights AFTER per-reward group-norm (so weights control relative
# pull, not magnitude). recall_quality + recall_hit_rate + range_tightness
# sum to 0.15 — slightly bumped from the old 0.10 for `recall_quality` because
# the three sub-signals together represent a more important objective post-v9.4.2
# (multimodal recall + dual schema). The 0.05 came out of a balanced pull from
# format / timing / silent_quality — see plan in fuzzy-plotting-valiant.md.
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "correctness":     0.30,   # primary outcome
    "silent_quality":  0.18,   # was 0.20
    "timing":          0.18,   # was 0.20
    "recall_quality":  0.05,   # query JSON format + leakage
    "recall_hit_rate": 0.07,   # |returned ∩ support| / |support|
    "range_tightness": 0.03,   # narrow + accurate time_range
    "format":          0.09,   # was 0.10
    "overflow_pen":    0.10,   # sparse compress-timing (Memex pattern)
}


# ===========================================================================
# v12.0 reward design — agentic reframing + timing-bucket reward.
# ===========================================================================
# 5 components (vs v11's 8) per design doc §5 + survey verdict in
# docs/v12.0_protocol_migration_design.md §11. Drops:
#   - recall_quality / recall_hit_rate / range_tightness → folded into
#     `outcome` (incorrect recall → wrong answer → outcome=0)
#   - silent_quality / num_responses → folded into `timing` (silent through
#     visibility window = -0.5; answer in window = +1)
#   - overflow_pen → replaced by compress_quality on the compress turn itself
#
# What's NEW: explicit `timing` component with bucket structure
# (early=-1, on_time=+1, late_partial=+0.5, missed=-0.5). NO prior released
# streaming-video paper or agentic RL paper has this — confirmed by the
# v12.0 survey. ThinkStream is the first.
#
# Multi-level GRPO advantage aggregation (ReMemR1 ICLR'26 pattern):
#   final_advantage = α · outcome_advantage + (1−α) · state_advantage
#   outcome_advantage   = GRPO-norm(correctness, group_by=video_uid)
#   state_advantage     = GRPO-norm(timing + format + compress_q − spam,
#                                   group_by=(video_uid, chunk_idx))
#   default α = 0.7 (ReMemR1 default 0.8 is HotpotQA — ThinkStream has
#   stronger per-step signal so we skew toward state).
#
# IMPORTANT — spam is ADDITIVE, NOT MULTIPLICATIVE.
# DeepEyesV2's `(1 - search_penalty) * acc` shape under-penalises when
# acc=0 (spam free) and over-penalises when acc=1 (already getting full
# reward). Linear additive `−spam_w * spam_score` decouples cleanly.
V12_REWARD_DICT_KEYS: tuple = (
    "outcome",          # 0/1 final answer correctness (LLM-judge or rule)
    "timing",           # bucketed timing reward (-1 early / +1 on / +0.5 late / -0.5 missed)
    "format",           # 0/1 — tags balanced, JSON parses, exactly one terminal
    "spam",             # >=0 — penalty for excess tool calls (additive)
    "compress_quality", # 0..1 — only on compress turns, ROUGE/coverage vs gold summary
    "recall_quality",   # v12.2 — chunk-level hit_rate vs support_chunks gold;
                        #         -0.5 if recall fired but retrieved empty;
                        #         -0.3 if query leaks gold answer.
                        #         Mask=0 when sample didn't recall or no support_chunks.
    "silent_quality",   # v12.2 — closes silent/response error modes:
                        #         +0.3 correct silence, -0.6 hallucinate, -0.6 missed.
                        #         Mask=1.0 always (every chunk has a silent/respond decision).
)

V12_DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "outcome":          1.0,    # primary signal
    "timing":           0.3,    # explicit time penalty/bonus
    "format":           0.1,    # weak — let outcome carry it
    "spam":            -0.2,    # NEGATIVE — additive penalty (sign in weight,
                                #            so r = sum(w * x) is the formula)
    "compress_quality": 0.2,    # only contributes when chunk has compress action
    "recall_quality":   0.3,    # v12.2 — chunk-level hit_rate; mid-weight (0.3 matches
                                #         timing) since recall is sparse and informative
    "silent_quality":   0.2,    # v12.2 — silent/response error correction; 0.2 ⇒ swing
                                #         ±0.12 same order as outcome×0.5 (v11 parity)
}

# Multi-level advantage mixing coefficient. final_adv = α·outcome + (1-α)·state
V12_ADVANTAGE_MIX_ALPHA: float = 0.7


def per_reward_group_norm(
    reward_col: torch.Tensor,
    mask_col: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Mean-only group-norm with mask handling.

    Args:
        reward_col: [B] = [N_samples * G] raw reward for one component.
        mask_col:   [B] applicability mask (1.0 = apply, 0.0 = skip).
        group_size: G — number of rollouts per sample.

    Returns:
        [B] per-sample mean-zero advantage (within each group of G).
        Masked-out rows produce 0 advantage. Empty groups (all rows masked)
        also produce 0 (no NaN leak).

    Why mean-only (not (x-μ)/σ): bimodal reward distributions (e.g. when
    half the group failed format and got -1.0) make σ unstable and amplify
    noise. ReMemR1's training script uses ``grpo_use_adv=False`` for the
    same reason.
    """
    B = reward_col.shape[0]
    assert B % group_size == 0, (
        f"reward_col len {B} not divisible by group_size={group_size}"
    )
    N = B // group_size

    # NaN-mask before nanmean so masked rows are excluded from the group
    # mean computation.
    masked = torch.where(
        mask_col > 0,
        reward_col,
        torch.full_like(reward_col, float("nan")),
    )
    grouped = masked.view(N, group_size)                              # [N, G]
    g_mean = torch.nanmean(grouped, dim=1, keepdim=True)              # [N, 1]
    # Empty groups → nanmean returns NaN; replace with 0 so subtraction
    # below produces NaN-only rows that nan_to_num converts to 0.
    g_mean = torch.nan_to_num(g_mean, nan=0.0)
    adv = (grouped - g_mean).flatten()                                # [B]
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    return adv


def aggregate_gdpo(
    rewards_per_func: Dict[str, torch.Tensor],
    rewards_masks: torch.Tensor,
    group_size: int,
    weights: Dict[str, float] = None,
    keys: List[str] = None,
):
    """Full GDPO-style aggregation: per-reward group-norm → weighted sum → batch-whiten.

    Args:
        rewards_per_func: {key: [B]} per-component raw rewards.
        rewards_masks: [B, num_rewards] applicability mask, columns
            ordered by ``keys`` (defaults to ``REWARD_DICT_KEYS``).
        group_size: G — rollouts per sample.
        weights: per-reward weights dict; defaults to ``DEFAULT_REWARD_WEIGHTS``.
        keys: column ordering; defaults to ``REWARD_DICT_KEYS``.

    Returns:
        Tuple of:
        - advantages: [B] post-batch-whiten advantage scalars
        - diag: dict of per-reward and total diagnostic stats for logging
    """
    keys = list(keys or REWARD_DICT_KEYS)
    weights = weights or DEFAULT_REWARD_WEIGHTS

    rewards_cols = torch.stack(
        [rewards_per_func[k].float() for k in keys], dim=1
    )  # [B, R]
    masks_cols = rewards_masks.float()
    if masks_cols.shape != rewards_cols.shape:
        raise ValueError(
            f"rewards_masks shape {tuple(masks_cols.shape)} != "
            f"rewards shape {tuple(rewards_cols.shape)}"
        )

    # Step 1: per-reward group-norm
    adv_per_reward = torch.zeros_like(rewards_cols)
    diag: Dict[str, float] = {}
    for ki, k in enumerate(keys):
        adv_k = per_reward_group_norm(
            rewards_cols[:, ki], masks_cols[:, ki], group_size
        )
        adv_per_reward[:, ki] = adv_k
        diag[f"adv_{k}_mean"] = adv_k.mean().item()
        diag[f"adv_{k}_std"] = adv_k.std().item() if adv_k.numel() > 1 else 0.0
        diag[f"mask_{k}_rate"] = masks_cols[:, ki].mean().item()

    # Step 2: weighted sum across rewards
    w = torch.tensor(
        [weights[k] for k in keys], dtype=torch.float32, device=adv_per_reward.device
    )
    weighted = (adv_per_reward * w.unsqueeze(0)).sum(dim=1)            # [B]

    # Step 3: batch-wide whiten — keeps advantage scale stable across runs
    # with different reward weights or new components added.
    if weighted.numel() > 1:
        b_mean = weighted.mean()
        b_std = weighted.std()
        adv = (weighted - b_mean) / (b_std + 1e-4)
    else:
        adv = weighted - weighted.mean()

    diag["adv_total_mean"] = adv.mean().item()
    diag["adv_total_var"] = adv.var().item() if adv.numel() > 1 else 0.0

    return adv, diag


def aggregate_grpo(
    rewards_per_func: Dict[str, torch.Tensor],
    rewards_masks: torch.Tensor,
    group_size: int,
    weights: Dict[str, float] = None,
    keys: List[str] = None,
):
    """Vanilla GRPO aggregation: weighted scalar reward → group (z-)norm.

    v11.4 alternative to ``aggregate_gdpo`` for ablation / when bimodal
    component distributions don't matter. Standard DeepSeekMath GRPO
    formulation (arxiv 2402.03300):

      R_i = Σ_k w_k * r_k(i) * mask_k(i)         # weighted scalar per rollout
      μ_g = mean of R within each group of G
      σ_g = std  of R within each group of G
      A_i = (R_i − μ_g) / (σ_g + ε)              # group z-norm

    Differences from aggregate_gdpo:
      - Components are aggregated BEFORE normalization (lossy: a small
        but well-discriminated reward gets drowned by a large noisy one).
      - Single group-level z-norm (uses std, not mean-only) — assumes
        intra-group variance is non-degenerate, which holds for the
        outcome-driven primary reward (correctness) but not for sparse
        signals with high mask-rate.
      - No batch-whiten — advantage scale floats with reward weights.

    When to use which:
      GDPO (default): when you care about each component pulling the
        policy independently (the explicit goal of v11/v11.3 design;
        every sparse signal has its own group-norm so its weight stays
        meaningful even if it's bimodal).
      GRPO (this fn): cleaner when the reward is dominated by a single
        outcome (e.g. correctness >> 0.7 weight); also the standard
        baseline most papers compare against, useful for ablations.

    Args / Returns: same shape as aggregate_gdpo.
    """
    keys = list(keys or REWARD_DICT_KEYS)
    weights = weights or DEFAULT_REWARD_WEIGHTS

    rewards_cols = torch.stack(
        [rewards_per_func[k].float() for k in keys], dim=1
    )  # [B, R]
    masks_cols = rewards_masks.float()
    if masks_cols.shape != rewards_cols.shape:
        raise ValueError(
            f"rewards_masks shape {tuple(masks_cols.shape)} != "
            f"rewards shape {tuple(rewards_cols.shape)}"
        )
    w = torch.tensor(
        [weights[k] for k in keys], dtype=torch.float32, device=rewards_cols.device
    )

    # Step 1: weighted sum across components (mask-gated).
    # Σ_k w_k * r_k * mask_k — gives a single scalar reward per rollout.
    weighted_per_row = (rewards_cols * masks_cols * w.unsqueeze(0)).sum(dim=1)  # [B]

    # Step 2: per-group z-norm.
    B = weighted_per_row.shape[0]
    if B % group_size != 0:
        raise ValueError(
            f"weighted_per_row len {B} not divisible by group_size={group_size}"
        )
    N = B // group_size
    grouped = weighted_per_row.view(N, group_size)              # [N, G]
    g_mean = grouped.mean(dim=1, keepdim=True)                  # [N, 1]
    g_std = grouped.std(dim=1, keepdim=True, unbiased=False)    # [N, 1]
    adv = ((grouped - g_mean) / (g_std + 1e-4)).flatten()       # [B]
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

    # Diagnostics: per-component pre-aggregation stats so we can still
    # see which signals drove the scalar advantage.
    diag: Dict[str, float] = {}
    for ki, k in enumerate(keys):
        col = rewards_cols[:, ki]
        msk = masks_cols[:, ki]
        applied = col[msk > 0]
        diag[f"reward_{k}_mean"] = applied.mean().item() if applied.numel() else 0.0
        diag[f"reward_{k}_std"] = (
            applied.std().item() if applied.numel() > 1 else 0.0
        )
        diag[f"mask_{k}_rate"] = msk.mean().item()
    diag["adv_total_mean"] = adv.mean().item()
    diag["adv_total_var"] = adv.var().item() if adv.numel() > 1 else 0.0

    return adv, diag


def aggregate_advantages(
    rewards_per_func: Dict[str, torch.Tensor],
    rewards_masks: torch.Tensor,
    group_size: int,
    *,
    mode: str = "gdpo",
    weights: Dict[str, float] = None,
    keys: List[str] = None,
):
    """Dispatch to aggregate_gdpo or aggregate_grpo based on mode flag.

    ``mode == "gdpo"`` (default): per-reward group-norm + weighted sum +
    batch-whiten. Best when sparse signals have meaningful but bimodal
    distributions — each component pulls advantage independently.

    ``mode == "grpo"``: weighted sum first → single z-norm per group.
    Standard DeepSeekMath formulation; cleaner ablation baseline.
    """
    if mode == "gdpo":
        return aggregate_gdpo(
            rewards_per_func, rewards_masks, group_size,
            weights=weights, keys=keys,
        )
    if mode == "grpo":
        return aggregate_grpo(
            rewards_per_func, rewards_masks, group_size,
            weights=weights, keys=keys,
        )
    raise ValueError(f"unknown advantage mode: {mode!r} (choose 'gdpo' | 'grpo')")
