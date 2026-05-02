"""Option B Phase 2: ray_trainer recurrent reward broadcast unit test.

Verifies the new reward-broadcast block in ray_trainer.fit() (the
"v12.14 Option B Phase 2" comment) does the right thing:

  - When final_mask is all True → no-op (legacy stitched path).
  - When final_mask has False rows → each trajectory's final-action
    reward is broadcast to every sibling action's last valid response
    token. Sibling rewards are zero'd out before broadcast.

This test extracts the relevant block from ray_trainer.py via AST
manipulation and exercises it with synthetic tensors. We do this
instead of full ray_trainer instantiation because the latter pulls in
ray / vllm / FSDP runtime that won't import on the mac test env.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    src = (PROJECT_ROOT / "verl/verl/trainer/ppo/ray_trainer.py").read_text()

    # Locate the Phase-2 block (we marked it with a unique comment).
    marker_start = "# v12.14 Option B Phase 2: recurrent reward broadcast."
    marker_end = "# Operating Mode Selection:"
    s = src.find(marker_start)
    e = src.find(marker_end, s)
    assert s > 0 and e > s, "Phase 2 block not found in ray_trainer.py"
    block = src[s:e]
    print("═══ Phase 2 block found ═══")
    print(textwrap.indent(block.split("\n", 1)[0], "  "))

    # Run the block with synthetic batch.
    # Synthetic scenario:
    #   - 2 trajectories
    #   - traj 0 = 3 actions (action_0/1/2; action_2 is final, outcome 1.0)
    #   - traj 1 = 1 action (it's the final, outcome 0.5)
    # batch layout (5 rows):
    #   row 0: traj0 action0 (final=False, raw_reward=0.0)
    #   row 1: traj0 action1 (final=False, raw_reward=0.0)
    #   row 2: traj0 action2 (final=True,  raw_reward=1.0)
    #   row 3: traj1 action0 (final=True,  raw_reward=0.5)
    sidx = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    fmask = torch.tensor([False, False, True, True], dtype=torch.bool)
    n_rows = 4
    response_len = 8
    prompt_len = 4

    # reward_tensor: only finals have score. Place at last valid token (we'll
    # use len 6/7/8/5 valid response tokens per row to test position math).
    reward_tensor = torch.zeros(n_rows, response_len, dtype=torch.float32)
    # Final rows: place outcome at their last valid token
    reward_tensor[2, 7] = 1.0    # traj0 final, len 8
    reward_tensor[3, 4] = 0.5    # traj1 final, len 5

    # Construct a synthetic batch dict matching what the block reads
    class FakeBatch:
        def __init__(self):
            self.batch = {
                "sample_index": sidx,
                "final_mask": fmask,
                "prompts": torch.zeros(n_rows, prompt_len, dtype=torch.long),
                "attention_mask": torch.cat([
                    torch.ones(n_rows, prompt_len, dtype=torch.long),
                    # Per-row response valid lengths: [6, 7, 8, 5]
                    torch.tensor([
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 0, 0],
                    ], dtype=torch.long),
                ], dim=1),
            }

    batch = FakeBatch()
    metrics = {}

    # Execute the block via exec — supply the namespace it expects.
    exec_ns = {
        "batch": batch,
        "reward_tensor": reward_tensor,
        "metrics": metrics,
        "torch": torch,
    }
    # The marker line itself is at indent 0 (it follows
    # `reward_tensor, reward_extra_infos_dict = extract_reward(batch)`
    # which is the indented line in ray_trainer; the substring slice
    # captured the comment but cut off its leading spaces). The actual
    # code lines below the marker are at 20-space indent (5 levels: class
    # → method → for epoch → for batch → with marked_timer). Dedent by
    # stripping 20 spaces from non-empty lines that have ≥20 leading
    # spaces; first marker comment line stays at col 0.
    lines = block.split("\n")
    indent_len = 20
    stripped = []
    for line in lines:
        if line.startswith(" " * indent_len):
            stripped.append(line[indent_len:])
        elif not line.strip():
            stripped.append("")
        else:
            # Comment-only first line at col 0, keep
            stripped.append(line)
    code = "\n".join(stripped)
    exec(code, exec_ns)
    new_reward = exec_ns["reward_tensor"]

    print()
    print("═══ Verify broadcast result ═══")
    print(f"  shape: {new_reward.shape}")
    print(f"  per-row sum: {new_reward.sum(dim=-1).tolist()}")
    print(f"  argmax positions: {new_reward.argmax(dim=-1).tolist()}")

    # Assertions:
    # - traj0 (rows 0, 1, 2) all should have outcome 1.0 placed at their
    #   own last valid token: positions 5 (len 6), 6 (len 7), 7 (len 8)
    # - traj1 (row 3) outcome 0.5 at position 4 (len 5)
    expected = torch.zeros(n_rows, response_len, dtype=torch.float32)
    expected[0, 5] = 1.0  # traj0 sibling, broadcast to row 0's last token
    expected[1, 6] = 1.0  # traj0 sibling, broadcast to row 1's last token
    expected[2, 7] = 1.0  # traj0 final unchanged (its own outcome)
    expected[3, 4] = 0.5  # traj1 final unchanged

    assert torch.allclose(new_reward, expected), (
        f"\n  got = {new_reward}\n  expected = {expected}"
    )
    print(f"  ✓ broadcast correct: traj0 outcome 1.0 → rows 0,1,2; traj1 → row 3")

    # Telemetry checks
    assert "recurrent/n_trajectories" in metrics
    assert metrics["recurrent/n_trajectories"] == 2.0
    assert metrics["recurrent/n_actions_total"] == 4.0
    assert metrics["recurrent/avg_actions_per_traj"] == 2.0
    print(f"  ✓ telemetry: {metrics}")

    # Test 2: backward compat — final_mask all True → no-op
    print()
    print("═══ Backward compat (final_mask all True) ═══")
    sidx2 = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    fmask2 = torch.tensor([True, True, True, True], dtype=torch.bool)
    rt2 = torch.zeros(4, response_len, dtype=torch.float32)
    rt2[0, 7] = 0.1; rt2[1, 7] = 0.2; rt2[2, 7] = 0.3; rt2[3, 7] = 0.4

    class FakeBatch2:
        def __init__(self):
            self.batch = {
                "sample_index": sidx2,
                "final_mask": fmask2,
                "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
                "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
            }
    batch2 = FakeBatch2()
    metrics2 = {}
    ns2 = {"batch": batch2, "reward_tensor": rt2, "metrics": metrics2, "torch": torch}
    rt2_orig = rt2.clone()
    exec(code, ns2)
    rt2_after = ns2["reward_tensor"]
    assert torch.equal(rt2_after, rt2_orig), "all-final should be a no-op!"
    assert "recurrent/n_trajectories" not in metrics2
    print(f"  ✓ all-final is a no-op (reward_tensor unchanged, no telemetry)")

    print()
    print("✓ ALL OPTION B PHASE 2 BROADCAST TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
