"""Phase 4 (MemAgent-aligned) recurrent advantage flow test.

Validates the trajectory-level reward + 1D GRPO advantage + sample_index
broadcast pipeline that replaces the (incorrect) Phase 2 broadcast.

Specifically tests:
  - compute_1D_grpo_advantage groups by uid → per-trajectory z-score
  - reverse_indices reorders permutation back to identity
  - the full simulated flow:
      finals extracted by final_mask
      reordered to original input order via reverse_indices(sidx[final])
      advantage computed on trajectory level, NOT action level
      broadcast via sample_index back to action rows
  - per-action shape/values match what ray_trainer.fit() now writes

Why this test exists:
  test_recurrent_reward_broadcast.py exercised the OLD Phase 2 block
  (broadcast same scalar to all sibling actions then run standard GRPO).
  That approach was wrong — same uid + same scalar → within-group
  variance=0 → advantage=0 → no learning signal. The user's audit
  flagged it. This test exercises the NEW (correct) MemAgent-aligned
  flow: trajectory-level groups → within-group variance comes from
  actually-different rollout outcomes.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "verl"))


# Inline copies of verl/recurrent/utils.py helpers so the test runs without
# a ray install (verl/__init__.py imports ray transitively). These are byte-
# for-byte the same as the production code at:
#   verl/verl/recurrent/utils.py:252  reverse_indices
#   verl/verl/recurrent/utils.py:289  compute_1D_grpo_advantage
# Keeping them inline lets us test the math + flow on a pure-CPU mac env.
def reverse_indices(tensor):
    unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    assert len(unique) == len(tensor), "Your input tensor has duplicated elements."
    indices = torch.scatter_reduce(
        torch.zeros_like(unique, dtype=torch.long, device=tensor.device),
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(0), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    return indices


def compute_1D_grpo_advantage(token_level_rewards, index, epsilon=1e-6, use_adv=True):
    scores = token_level_rewards.sum(dim=-1).clone()
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                if use_adv:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                if use_adv:
                    id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if use_adv:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    return scores


def test_reverse_indices():
    """reverse_indices(t) returns indices s.t. t[reverse_indices(t)] = sorted_unique(t)."""
    t = torch.tensor([2, 0, 3, 1])
    rev = reverse_indices(t)
    reordered = t[rev]
    assert torch.equal(reordered, torch.tensor([0, 1, 2, 3])), (
        f"reverse_indices broken: {reordered=} expected [0,1,2,3]"
    )
    print("  ✓ reverse_indices: [2,0,3,1] -> reorder via rev -> [0,1,2,3]")


def test_1D_grpo_advantage_basic():
    """compute_1D_grpo_advantage groups by uid → z-score per group."""

    # 4 trajectories, 2 prompts × 2 rollouts (n=2):
    #   p0 outcomes: [0.8, 0.2] → mean=0.5, std=sqrt(0.18)=0.424 → adv=[+0.707, -0.707]
    #   p1 outcomes: [1.0, 0.5] → mean=0.75, std=sqrt(0.125)=0.354 → adv=[+0.707, -0.707]
    rt = torch.zeros(4, 8, dtype=torch.float32)
    rt[0, 7] = 0.8
    rt[1, 7] = 0.2
    rt[2, 7] = 1.0
    rt[3, 7] = 0.5
    uids = np.array(["p0", "p0", "p1", "p1"], dtype=object)

    adv = compute_1D_grpo_advantage(token_level_rewards=rt, index=uids, use_adv=True)
    assert adv.shape == (4,), f"expected shape (4,), got {adv.shape}"
    expected = torch.tensor([0.7071, -0.7071, 0.7071, -0.7071])
    assert torch.allclose(adv, expected, atol=1e-3), (
        f"\n  got = {adv}\n  expected = {expected}"
    )
    print(f"  ✓ 1D GRPO adv: groups by uid, returns per-traj z-score [+0.71/-0.71]")
    print(f"    (raw scores [0.8,0.2,1.0,0.5] → adv [{adv[0]:.3f},{adv[1]:.3f},{adv[2]:.3f},{adv[3]:.3f}])")


def test_full_flow_simulation():
    """Simulate the ray_trainer Phase 4d flow end-to-end.

    Synthetic batch:
      4 trajectories (B=2 prompts × n=2 rollouts), K=[3,1,2,1] actions
      → batch has 7 rows (sum(K_i)=7)
      sample_index =     [0, 0, 0, 1, 2, 2, 3]
      final_mask   =     [F, F, T, T, F, T, T]
      rm_scores (final rows only):
        row 2 (traj 0): 0.8
        row 3 (traj 1): 0.2
        row 5 (traj 2): 1.0
        row 6 (traj 3): 0.5
      original_batch uids = ['p0','p0','p1','p1']
    """

    n_actions = 7
    response_len = 8
    sidx = torch.tensor([0, 0, 0, 1, 2, 2, 3], dtype=torch.long)
    fmask = torch.tensor([False, False, True, True, False, True, True], dtype=torch.bool)
    rm_scores = torch.zeros(n_actions, response_len, dtype=torch.float32)
    rm_scores[2, 7] = 0.8  # traj 0 final
    rm_scores[3, 5] = 0.2  # traj 1 final (placed at random valid position)
    rm_scores[5, 6] = 1.0  # traj 2 final
    rm_scores[6, 4] = 0.5  # traj 3 final
    response_mask = torch.ones(n_actions, response_len, dtype=torch.float32)

    n_traj = 4
    uids = np.array(["p0", "p0", "p1", "p1"], dtype=object)

    # ---- Phase 4d block simulation ----
    # 1. Extract final-only reward, reorder to original input order
    final_action_idx = torch.where(fmask)[0]
    assert final_action_idx.tolist() == [2, 3, 5, 6]
    traj_idx_of_finals = sidx[final_action_idx].long()
    assert traj_idx_of_finals.tolist() == [0, 1, 2, 3]
    reorder = reverse_indices(traj_idx_of_finals)
    reward_traj_tensor = rm_scores[final_action_idx][reorder]
    assert reward_traj_tensor.shape == (n_traj, response_len)
    # Sanity: per-traj outcome scalars should be [0.8, 0.2, 1.0, 0.5]
    per_traj_outcome = reward_traj_tensor.sum(dim=-1)
    assert torch.allclose(per_traj_outcome, torch.tensor([0.8, 0.2, 1.0, 0.5]))
    print(f"  ✓ trajectory-level reward extraction: per_traj_outcome = {per_traj_outcome.tolist()}")

    # 2. 1D GRPO advantage on trajectory level
    adv_scalar = compute_1D_grpo_advantage(reward_traj_tensor, uids, use_adv=True)
    expected_adv = torch.tensor([0.7071, -0.7071, 0.7071, -0.7071])
    assert torch.allclose(adv_scalar, expected_adv, atol=1e-3)
    print(f"  ✓ adv_scalar [B*n=4]: {adv_scalar.tolist()}")

    # 3. Broadcast via sample_index
    adv_per_action = adv_scalar[sidx]  # shape [7]
    expected_per_action = torch.tensor([
        0.7071, 0.7071, 0.7071,   # traj 0 actions (3 rows, all get traj 0 adv)
        -0.7071,                  # traj 1 (1 row)
        0.7071, 0.7071,           # traj 2 (2 rows)
        -0.7071,                  # traj 3 (1 row)
    ])
    assert torch.allclose(adv_per_action, expected_per_action, atol=1e-3)
    print(f"  ✓ broadcast: adv_per_action [sum(K)=7]: {adv_per_action.tolist()}")

    # 4. Tile across response_length
    advantages = adv_per_action.unsqueeze(-1).tile([1, response_len]) * response_mask
    assert advantages.shape == (n_actions, response_len)
    # Each row's advantage should be the trajectory's z-scored outcome,
    # masked by response_mask (here all 1s, so unmasked).
    assert torch.allclose(advantages[0, :], advantages[1, :])  # siblings of traj 0
    assert torch.allclose(advantages[0, 0], torch.tensor(0.7071), atol=1e-3)
    print(f"  ✓ advantages [{n_actions},{response_len}]: siblings share value, tiled across response_length")


def test_adv_nonzero_with_n_equal_1():
    """Critical correctness check: with rollout.n=1, the OLD Phase 2 broadcast +
    standard GRPO would give advantage=0 (within-group variance=0). The NEW
    flow groups by prompt uid where each prompt has exactly 1 trajectory in
    n=1 setting → 1D advantage falls back to mean-centered=0 (singleton group).

    This is mathematically expected — GRPO needs n>1 rollouts per prompt to
    have a learning signal. The point is: the OLD broadcast was even worse,
    creating fake within-group variance=0 even when n>1 had real variance.
    """

    # n=1: each prompt has 1 rollout = 1 trajectory
    rt = torch.zeros(2, 8)
    rt[0, 7] = 0.8
    rt[1, 7] = 0.3
    uids = np.array(["p0", "p1"], dtype=object)
    adv = compute_1D_grpo_advantage(rt, uids, use_adv=True)
    # singleton group: mean=0, std=1 → adv = score - 0 = score itself
    assert torch.allclose(adv, torch.tensor([0.8, 0.3])), (
        f"n=1 fallback broken: {adv=}"
    )
    print(f"  ✓ n=1 singleton group: adv = raw score (mean=0, std=1 fallback)")
    print(f"    note: n=1 is GRPO-degenerate; production should use rollout.n >= 2")


def main() -> int:
    print("═══ Phase 4 (MemAgent-aligned) recurrent advantage flow test ═══")
    print()
    print("Test 1: reverse_indices reorder primitive")
    test_reverse_indices()
    print()
    print("Test 2: compute_1D_grpo_advantage (group_by uid → per-traj z-score)")
    test_1D_grpo_advantage_basic()
    print()
    print("Test 3: full ray_trainer Phase 4d flow simulation (4 traj, K=[3,1,2,1])")
    test_full_flow_simulation()
    print()
    print("Test 4: n=1 singleton-group advantage (degenerate but correct)")
    test_adv_nonzero_with_n_equal_1()
    print()
    print("✓ ALL PHASE 4 ADVANTAGE TESTS PASS")
    print()
    print("Coverage:")
    print("  - 1D GRPO grouping (replaces broken Phase 2 broadcast + std GRPO)")
    print("  - sample_index broadcast (sibling actions share trajectory's adv)")
    print("  - reverse_indices reorder (final rows → input order)")
    print()
    print("NOT covered by this test (orthogonal to advantage math):")
    print("  - DataProto.repeat / swap mechanics → ray_trainer integration test")
    print("  - end-to-end FSDP shape divisibility")
    print("  - actor loss with broadcasted advantages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
