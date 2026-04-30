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
# v12.3 (Apr 2026) — REWARD STACK SIMPLIFICATION
#
# Audit driven by user pushback (chunk-level support_chunks gold loses
# generalization; many families e.g. CR3/CR6/CR7 have no support_chunks,
# their reward signal would die under mask=0). Industry verification:
#
#   DeepEyesV2 (arXiv:2511.05271, Nov 2025) — final reward = 0.8·acc + 0.2·format.
#     Tool-specific reward variables EXIST in code but are NOT used in the
#     final score (verl/utils/reward_score/deepeyesv2.py:193). Paper quote:
#     "relies only on two simple rewards, accuracy and format, without
#     complex reward engineering" — explicit retreat from the original
#     DeepEyes complex conditional tool reward.
#   ReTool, MemAgent — outcome only.
#   ReMemR1 — counterfactual delta (recall_with - recall_without word-overlap),
#     NOT chunk-level gold annotations.
#   NeurIPS 2025 / ICLR 2026 (TIPS, Multi-Turn Reasoning) — turn-level reward
#     IS the trend, but DERIVED from outcome via advantage propagation, NOT
#     from gold tool targets.
#
# Result: drop recall_quality + compress_quality from the production reward
# stack. Their functions remain in v12_rewards.py for legacy callers, but
# V12_REWARD_DICT_KEYS / V12_DEFAULT_REWARD_WEIGHTS no longer reference them.
# Tool-decision credit assignment now flows via:
#   1. outcome reward → GRPO group-norm propagates advantage to all chunks
#      (rollouts whose recall/compress decisions led to correct answer get
#       above-mean advantage; others get below-mean → policy learns)
#   2. silent_quality + timing — streaming-specific signals (no industry
#      analog); these are NOT chunk-level credit assignment, they're
#      additional outcome dimensions ("when to talk", not just "what to say")
#
V12_REWARD_DICT_KEYS: tuple = (
    "outcome",          # 0/1 per-question correctness; dominant signal
    "timing",           # bucketed timing reward (-1 early / +1 on / +0.5 late / -0.5 missed)
    "format",           # 0/1 — tags balanced, JSON parses, exactly one terminal
    "spam",             # >=0 — penalty for excess tool calls (additive)
    "silent_quality",   # streaming-specific: +0.3 correct silence, -0.6 hallucinate, -0.6 missed
)

_V12_PRODUCTION_WEIGHTS: Dict[str, float] = {
    "outcome":          1.0,    # primary signal — DeepEyesV2 0.8 ↑ to 1.0 (drop format weight)
    "timing":           0.3,    # streaming bonus/penalty
    "format":           0.1,    # weak — gate-like; per DeepEyesV2 0.2 but lower since
                                # silent_quality + timing already shape the streaming policy
    "spam":            -0.2,    # NEGATIVE — over-budget tool penalty (additive)
    "silent_quality":   0.2,    # streaming-specific decision quality (no industry analog)
}


def _load_reward_weights() -> Dict[str, float]:
    """Resolve reward weights, honoring THINKSTREAM_REWARD_WEIGHTS_PATH override.

    v12.6: ablation_runner.py writes a JSON override per ablation; this
    function picks it up at module-import time. Falls back to production
    weights if env var is unset or file is missing/malformed.
    """
    import json as _json
    import os as _os
    p = _os.environ.get("THINKSTREAM_REWARD_WEIGHTS_PATH")
    if p:
        try:
            with open(p) as _f:
                override = _json.load(_f)
            if isinstance(override, dict):
                merged = dict(_V12_PRODUCTION_WEIGHTS)
                merged.update({
                    k: float(v) for k, v in override.items()
                    if k in _V12_PRODUCTION_WEIGHTS
                })
                return merged
        except (OSError, ValueError):
            pass
    return dict(_V12_PRODUCTION_WEIGHTS)


V12_DEFAULT_REWARD_WEIGHTS: Dict[str, float] = _load_reward_weights()

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
            ordered by ``keys`` (defaults to ``V12_REWARD_DICT_KEYS``).
        group_size: G — rollouts per sample.
        weights: per-reward weights dict; defaults to ``V12_DEFAULT_REWARD_WEIGHTS``.
        keys: column ordering; defaults to ``V12_REWARD_DICT_KEYS``.

    Returns:
        Tuple of:
        - advantages: [B] post-batch-whiten advantage scalars
        - diag: dict of per-reward and total diagnostic stats for logging
    """
    keys = list(keys or V12_REWARD_DICT_KEYS)
    weights = weights or V12_DEFAULT_REWARD_WEIGHTS

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
    keys = list(keys or V12_REWARD_DICT_KEYS)
    weights = weights or V12_DEFAULT_REWARD_WEIGHTS

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
