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


# v11 reward keys (must match the column order of rewards_masks emitted by
# calc_rewards in grpo.py).
REWARD_DICT_KEYS: tuple = (
    "format", "correctness", "timing", "silent_quality",
    "recall_quality", "overflow_pen",
)

# Per-reward weights AFTER per-reward group-norm (so weights control relative
# pull, not magnitude). See plan rationale in
# /Users/hzh/.claude/plans/fuzzy-plotting-valiant.md.
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "correctness":    0.30,   # primary outcome
    "silent_quality": 0.20,   # streaming-specific, never gated
    "timing":         0.20,   # RL's main lever
    "recall_quality": 0.10,   # recall-query format + leakage
    "format":         0.10,   # soft component (was hard gate)
    "overflow_pen":   0.10,   # sparse compress-timing signal (Memex pattern)
}


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
