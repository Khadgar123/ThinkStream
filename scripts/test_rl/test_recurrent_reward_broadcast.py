"""DEPRECATED — Phase 2 reward broadcast was REMOVED in Phase 4d.

This test originally exercised the v12.14 Phase 2 block that broadcast
each trajectory's final-action reward to all sibling actions' last
response token, then handed the per-action reward tensor to standard
GRPO `compute_advantage`. The user's audit (see also docs commit) showed
this was wrong credit assignment:

  - With rollout.n=1: same uid + same broadcasted scalar across all
    sibling actions → within-group variance=0 → advantage=0 → no
    learning signal.
  - With rollout.n>1: long trajectories (more action rows) get over-
    weighted in the per-uid mean/std statistics, biasing GRPO.

The MemAgent / ReMemR1 pattern is correct and is now implemented in
ray_trainer.fit() under the comment marker
`# v12.14 Option B Phase 4d (MemAgent-aligned)`:

  1. Save `original_batch` (B*n rows, has uid) BEFORE batch swap.
  2. After swap (batch = expanded sum(K_i) rows), keep per-action
     reward unchanged in batch.batch["rm_scores"] / "token_level_scores".
  3. Extract trajectory-final reward subset → reorder to input order.
  4. Compute 1D GRPO advantage on trajectory level (B*n rows).
  5. Broadcast advantage scalar via `sample_index` back to action rows.
  6. Tile across response_length, mask by response_mask.

The new flow is unit-tested in:

    scripts/test_rl/test_phase4_recurrent_advantage.py

This stub remains so the test runner doesn't fail with "module not
found" for anyone still referencing the old name.
"""
from __future__ import annotations


def main() -> int:
    print("═══ test_recurrent_reward_broadcast.py: DEPRECATED ═══")
    print()
    print("The Phase 2 reward broadcast block this test exercised was")
    print("removed in Phase 4d (MemAgent-aligned trajectory-level")
    print("advantage). See module docstring for context.")
    print()
    print("Run the replacement test:")
    print("  python scripts/test_rl/test_phase4_recurrent_advantage.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
