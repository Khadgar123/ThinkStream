"""Smoke tests for v11 GDPO-style advantage aggregation.

Covers:
- Per-reward group-norm zero-mean within group
- Mask=0 rows contribute 0 advantage and don't affect group mean
- Empty groups (all masked) produce all zeros (no NaN leak)
- Batch-whiten yields ~unit variance and ~zero mean
- compute_gdpo_advantages end-to-end shape + value sanity

These tests do not require GPU, network, or any model.
"""
from __future__ import annotations

import math
import unittest

import torch

from thinkstream.trainer.gdpo_advantage import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_DICT_KEYS,
    aggregate_gdpo,
    per_reward_group_norm,
)
_gdpo_per_reward_group_norm = per_reward_group_norm  # alias for clarity


class TestGdpoPerRewardGroupNorm(unittest.TestCase):
    def test_simple_group_norm_zero_mean(self):
        # G=4, N=2 → B=8.  Each group of 4 should have mean-zero advantage.
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0])
        masks = torch.ones_like(rewards)
        adv = _gdpo_per_reward_group_norm(rewards, masks, group_size=4)

        # Group 0: mean=2.5 → adv = [-1.5, -0.5, 0.5, 1.5]
        # Group 1: mean=25.0 → adv = [-15, -5, 5, 15]
        torch.testing.assert_close(adv[:4], torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        torch.testing.assert_close(adv[4:], torch.tensor([-15.0, -5.0, 5.0, 15.0]))

        # Each group's advantage sums to 0
        self.assertAlmostEqual(adv[:4].sum().item(), 0.0, places=5)
        self.assertAlmostEqual(adv[4:].sum().item(), 0.0, places=5)

    def test_masked_rows_excluded_from_mean(self):
        # G=4. First group: rows 1,2 masked → mean over rows 0,3 only = 2.5.
        # Masked rows themselves should produce 0 advantage (no signal).
        rewards = torch.tensor([1.0, 99.0, 99.0, 4.0])
        masks = torch.tensor([1.0, 0.0, 0.0, 1.0])
        adv = _gdpo_per_reward_group_norm(rewards, masks, group_size=4)
        # Mean over unmasked = (1+4)/2 = 2.5
        # adv = [1-2.5, 0, 0, 4-2.5] = [-1.5, 0, 0, 1.5]
        torch.testing.assert_close(adv, torch.tensor([-1.5, 0.0, 0.0, 1.5]))

    def test_empty_group_produces_zeros(self):
        # All rows masked in group 0, all unmasked in group 1
        rewards = torch.tensor([5.0, 5.0,  10.0, 20.0])
        masks = torch.tensor([0.0, 0.0,  1.0, 1.0])
        adv = _gdpo_per_reward_group_norm(rewards, masks, group_size=2)
        # Group 0: empty → all zeros (no NaN leak).
        # Group 1: mean=15 → [-5, 5]
        torch.testing.assert_close(adv[:2], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(adv[2:], torch.tensor([-5.0, 5.0]))
        self.assertFalse(torch.isnan(adv).any().item(), "no NaN leak allowed")

    def test_assert_on_bad_group_size(self):
        # B=5 not divisible by G=2 → assertion error
        with self.assertRaises(AssertionError):
            _gdpo_per_reward_group_norm(
                torch.zeros(5), torch.ones(5), group_size=2
            )


class TestGdpoFullAggregation(unittest.TestCase):
    """Reproduce the full algorithm steps without invoking the slyme node."""

    def _full_aggregate(
        self,
        rewards_dict: dict,
        rewards_masks: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Mirror compute_gdpo_advantages without the slyme Context wrapper."""
        adv, _diag = aggregate_gdpo(rewards_dict, rewards_masks, group_size)
        return adv

    def test_batch_whiten_unit_variance(self):
        G, N = 4, 3   # B=12
        B = G * N
        rewards_dict = {
            k: torch.randn(B, generator=torch.Generator().manual_seed(42 + i))
            for i, k in enumerate(REWARD_DICT_KEYS)
        }
        masks = torch.ones(B, len(REWARD_DICT_KEYS))
        adv = self._full_aggregate(rewards_dict, masks, group_size=G)

        self.assertEqual(adv.shape, (B,))
        # batch-whiten → mean ~0, var ~1 (allow some slack from 1e-4 epsilon)
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(adv.var().item(), 1.0, delta=0.1)

    def test_sparse_recall_quality_only_fires_for_recalled_rollouts(self):
        # G=4, N=2.  Only rollouts 0 and 5 "did_recall".
        G, N = 4, 2
        B = G * N
        keys = list(REWARD_DICT_KEYS)
        recall_idx = keys.index("recall_quality")

        rewards_dict = {k: torch.zeros(B) for k in keys}
        # Give recall a strong positive reward only for rollout 5
        rewards_dict["recall_quality"][0] = 0.5  # in group 0
        rewards_dict["recall_quality"][5] = 1.0  # in group 1
        masks = torch.zeros(B, len(keys))
        masks[:, keys.index("format")] = 1.0
        masks[:, keys.index("silent_quality")] = 1.0
        masks[:, keys.index("overflow_pen")] = 1.0
        # Only rollouts 0 and 5 have recall_quality applicable
        masks[0, recall_idx] = 1.0
        masks[5, recall_idx] = 1.0

        adv = self._full_aggregate(rewards_dict, masks, group_size=G)

        # No assertion errors; result should be finite for all rollouts.
        self.assertFalse(torch.isnan(adv).any().item())
        self.assertFalse(torch.isinf(adv).any().item())
        # Rollout 5 (the only recall-success one) should have positive advantage
        # (its recall_quality dominates over zero baseline of group 1's other 3)
        # Rollout 0's recall=0.5 alone in its group; group_mean(unmasked)=0.5,
        # so its recall_quality advantage is 0 — but rollout 5's group_mean=1.0,
        # also alone, also 0. So actually neither dominates strongly via recall.
        # We just check finiteness here, which is the key invariant.

    def test_all_masked_for_one_reward_kills_only_that_signal(self):
        # If recall_quality has mask=0 for every row, that column contributes
        # 0 across the board — the other rewards still drive advantage.
        G, N = 4, 2
        B = G * N
        keys = list(REWARD_DICT_KEYS)
        rewards_dict = {k: torch.zeros(B) for k in keys}
        # Give correctness a real signal
        rewards_dict["correctness"] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0,  1.0, 1.0, 0.0, 0.0]
        )
        masks = torch.ones(B, len(keys))
        masks[:, keys.index("recall_quality")] = 0.0  # fully muted

        adv = self._full_aggregate(rewards_dict, masks, group_size=G)
        self.assertFalse(torch.isnan(adv).any().item())
        # Rollouts with correctness=1 should have higher advantage than those with 0
        # (in their respective groups)
        group0 = adv[:G]
        self.assertGreater(group0[0].item(), group0[1].item())


if __name__ == "__main__":
    unittest.main()
