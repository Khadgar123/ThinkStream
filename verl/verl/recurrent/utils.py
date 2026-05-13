"""Small recurrent-rollout helpers shared by trainer integrations.

The semantics match the MemAgent/ReMemR1 recurrent interface: an expanded
rollout batch may contain several action rows per original trajectory;
``sample_index`` maps each action row back to its trajectory row, and exactly
one ``final_mask`` row per trajectory carries the answer that is scored.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto


def reverse_indices(tensor: torch.Tensor) -> torch.Tensor:
    """Return indices that order a unique integer tensor by its values."""

    tensor = tensor.long()
    unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    if len(unique) != len(tensor):
        raise ValueError("sample_index for final rows must be unique")
    indices = torch.scatter_reduce(
        torch.zeros_like(unique, dtype=torch.long, device=tensor.device),
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(0), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    return indices


def final_batch(
    batch: DataProto,
    final_mask: torch.Tensor | None = None,
    sample_index: torch.Tensor | None = None,
) -> DataProto:
    """Select final action rows and restore original trajectory order."""

    if final_mask is None:
        final_mask = batch.batch["final_mask"]
    if sample_index is None:
        sample_index = batch.batch["sample_index"]
    final_mask = final_mask.bool()
    sample_index = sample_index.long()

    final_row_idx = torch.where(final_mask)[0]
    if final_row_idx.numel() == 0:
        raise ValueError("recurrent rollout has no final rows")

    final_sample_index = sample_index[final_row_idx]
    expected = int(torch.unique(sample_index).numel()) if sample_index.numel() else 0
    unique_final, inverse = torch.unique(
        final_sample_index,
        sorted=True,
        return_inverse=True,
    )
    if unique_final.numel() != expected:
        raise ValueError(
            "recurrent rollout must contain exactly one final row per trajectory: "
            f"got {unique_final.numel()}, expected {expected}"
        )

    selected = torch.empty(
        unique_final.numel(),
        dtype=final_row_idx.dtype,
        device=final_row_idx.device,
    )
    selected.scatter_reduce_(
        0,
        inverse,
        final_row_idx,
        reduce="amin",
        include_self=False,
    )
    return batch[selected]


def compute_1D_grpo_advantage(
    token_level_rewards: torch.Tensor,
    index: np.ndarray | list[Any] | torch.Tensor,
    epsilon: float = 1e-6,
    use_adv: bool = True,
) -> torch.Tensor:
    """Compute one scalar GRPO-style advantage per trajectory.

    This is the ReMemR1 convention used for recurrent rollouts. Rewards are
    grouped by prompt uid before being broadcast back to all action rows.
    Singleton groups keep their raw score by using mean=0 and std=1.
    """

    scores = (
        token_level_rewards.sum(dim=-1).clone()
        if token_level_rewards.dim() > 1
        else token_level_rewards.clone()
    ).to(dtype=torch.float32)
    n_items = scores.shape[0]
    if len(index) != n_items:
        raise ValueError(f"index length {len(index)} != rewards length {n_items}")

    def _key(i: int) -> Any:
        value = index[i]
        if isinstance(value, torch.Tensor):
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    id2mean: dict[Any, torch.Tensor] = {}
    id2std: dict[Any, torch.Tensor] = {}

    with torch.no_grad():
        for i in range(n_items):
            id2score[_key(i)].append(scores[i])
        for key, values in id2score.items():
            if len(values) == 1:
                id2mean[key] = torch.tensor(0.0, device=scores.device)
                if use_adv:
                    id2std[key] = torch.tensor(1.0, device=scores.device)
            else:
                stacked = torch.stack(values)
                id2mean[key] = stacked.mean()
                if use_adv:
                    id2std[key] = stacked.std()

        output = torch.zeros_like(scores)
        for i in range(n_items):
            key = _key(i)
            if use_adv:
                output[i] = (scores[i] - id2mean[key]) / (id2std[key] + epsilon)
            else:
                output[i] = scores[i] - id2mean[key]
    return output
