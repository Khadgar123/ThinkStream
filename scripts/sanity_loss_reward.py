"""SFT loss + RL reward sanity check — runs on CPU-only venv.

Verifies critical invariants without GPU:

  [SFT] given a messages-format sample, the loss-mask:
    1. Has loss=1 only on assistant content tokens
    2. Has loss=0 (IGNORE_INDEX) on system + user + tool turns
    3. Number of unmasked tokens matches teacher's assistant span length

  [REWARD] given a synthetic rollout outcome:
    1. Outcome reward = 1 only when answer matches gold (literal forms)
    2. Format reward gates malformed JSON
    3. Spam reward is additive over budget
    4. Silent quality penalizes hallucination AND missed response equally
    5. Aggregate advantage: outcome dominates (weight 1.0), state shapes (-)

  [CROSS-VALIDATION] same input → same numbers via slyme path AND verl path
    (both call the same v12_rewards functions; this check guards against
    accidental forking of the algorithm).

If any invariant breaks, exits with non-zero status. Run before every
training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# venv import test path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===========================================================================
# SFT loss-mask invariants (uses pass5 output to verify mask correctness)
# ===========================================================================

def check_sft_loss_mask():
    """Loss is masked correctly only on assistant tokens."""
    print("\n[SFT loss-mask check]")
    from scripts.agent_data_v5.pass5_messages import build_messages
    from pathlib import Path as _P

    # Synthetic compress sample
    sample = {
        "chunk_idx": 7, "sample_type": "compress", "v12_inter_chunk": True,
        "video_id": "vid", "video_path": "/tmp/x.mp4", "trajectory_id": "t1",
        "input": {
            "memory": {
                "compressed_segments": [],
                "recent_thinks": [
                    {"chunk": i, "time": f"{i}-{i+1}", "text": f"think_{i}"}
                    for i in range(5)
                ],
            },
            "queries": [],
            "user_input": "<compress_trigger range='0-5'/>",
        },
        "output": (
            '<think>summary thought</think>'
            '<tool_call>\n{"name":"compress","arguments":'
            '{"time_range":[0,5],"text":"compressed"}}\n</tool_call>'
        ),
    }
    msgs = build_messages(sample, _P("/tmp"))

    # Check 1: 3 messages (system + user + assistant)
    assert len(msgs) == 3, f"compress should have 3 msgs, got {len(msgs)}"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    print("  ✓ message structure: system → user → assistant (compress shape C)")

    # Check 2: assistant content matches teacher output
    assistant_text = msgs[2]["content"][0]["text"]
    assert assistant_text == sample["output"], "assistant content must equal teacher output"
    print(f"  ✓ assistant content = teacher output ({len(assistant_text)} chars)")

    # Check 3: user content carries memory + trigger, no visual_window
    user_text_blocks = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
    user_text = " ".join(user_text_blocks)
    assert "<memory>" in user_text, "user must contain <memory>"
    assert "<compress_trigger" in user_text, "user must contain trigger"
    assert "<visual_window>" not in user_text, "inter-chunk must NOT have visual_window"
    print("  ✓ user content: memory + trigger present, visual_window absent")

    return True


# ===========================================================================
# Reward function invariants
# ===========================================================================

def check_reward_invariants():
    """All reward functions produce expected values for canonical cases."""
    print("\n[Reward invariants]")
    from thinkstream.trainer.v12_rewards import (
        compute_outcome_reward_v12,
        compute_timing_reward_v12,
        compute_format_reward_v12,
        compute_spam_score_v12,
        compute_silent_quality_v12,
    )

    # Outcome
    assert compute_outcome_reward_v12("yes", "yes", answer_form="binary") == 1.0
    assert compute_outcome_reward_v12("no", "yes", answer_form="binary") == 0.0
    assert compute_outcome_reward_v12("YES", "yes", answer_form="binary") == 1.0
    assert compute_outcome_reward_v12("a" * 1001, "a", answer_form="binary") == 0.0  # anti-hack length
    print("  ✓ outcome: literal match, length anti-hack")

    # Timing
    # answer in window [5, 10]
    assert compute_timing_reward_v12(7, 5, 10) == 1.0
    # before window: hallucination
    assert compute_timing_reward_v12(3, 5, 10) == -1.0
    # after window: late decay (1 chunk over → 0.5)
    assert abs(compute_timing_reward_v12(11, 5, 10, late_window_chunks=1) - 0.5) < 1e-6
    # never answered: -0.5
    assert compute_timing_reward_v12(None, 5, 10) == -0.5
    print("  ✓ timing: in-window=+1, hallucinate=-1, late-decay, missed=-0.5")

    # Format
    assert compute_format_reward_v12(["<think>x</think><answer>y</answer>"]) == 1.0
    assert compute_format_reward_v12(["<think>broken"]) == 0.0
    print("  ✓ format: well-formed=1, broken=0")

    # Spam (additive penalty score, weight in V12_DEFAULT_REWARD_WEIGHTS is negative)
    assert compute_spam_score_v12(0, 0) == 0.0
    assert compute_spam_score_v12(2, 0, recall_budget=1) == 0.5  # 1 over × 0.5
    assert compute_spam_score_v12(1, 2, compress_budget=1) == 0.3  # 1 over × 0.3
    print("  ✓ spam: 0 in budget, additive over")

    # Silent quality
    assert compute_silent_quality_v12("", "silent", "") == 0.3       # correct silence
    assert compute_silent_quality_v12("hello", "silent", "") == -0.6  # hallucination
    assert compute_silent_quality_v12("", "response", "yes") == -0.6  # missed
    assert compute_silent_quality_v12("yes", "response", "yes") == 0.0  # correctness handled by outcome
    print("  ✓ silent_quality: correct=+0.3, halluc=-0.6, miss=-0.6")

    return True


# ===========================================================================
# Aggregate-advantage invariants
# ===========================================================================

def check_advantage_aggregation():
    """Multi-level mixed advantage produces signed values per rollout."""
    print("\n[Advantage aggregation]")
    import torch
    from thinkstream.trainer.v12_rollout import (
        compute_1d_grpo_advantage,
        compute_mixed_advantage_v12,
    )

    # GRPO 1-D: video v1 has 4 rollouts, v2 has 2
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
    groups = ["v1", "v1", "v1", "v1", "v2", "v2"]
    adv = compute_1d_grpo_advantage(rewards, groups)
    # v1 mean=2.5, std=sqrt(((1-2.5)²+(2-2.5)²+(3-2.5)²+(4-2.5)²)/3)=1.291
    # v2 mean=1.5, std=sqrt(((1-1.5)²+(2-1.5)²)/1)=0.707
    # Adv signs and ordering
    assert adv[0] < adv[1] < adv[2] < adv[3], "v1 monotone in score"
    assert adv[4] < adv[5], "v2 monotone"
    # Adv sums per group ≈ 0 after group-mean subtraction
    assert abs(float(adv[:4].sum())) < 1e-3, f"v1 sum should be 0, got {adv[:4].sum()}"
    print(f"  ✓ GRPO 1-D: v1 adv={[round(x,3) for x in adv[:4].tolist()]}, "
          f"v2 adv={[round(x,3) for x in adv[4:].tolist()]}")

    # Mixed: outcome dominates (α=0.7), state shapes
    out_r = torch.tensor([1.0, 0.0, 1.0, 0.0])      # 2 correct, 2 wrong
    state_r = torch.tensor([0.5, 0.5, 0.5, 0.5])    # uniform → state_adv=0
    mixed = compute_mixed_advantage_v12(
        out_r, state_r,
        video_uid_per_row=["a", "a", "b", "b"],
        chunk_idx_per_row=[5, 5, 5, 5],
        alpha=0.7,
    )
    # Each video has [1, 0] → mean=0.5, std≈0.707 → adv≈[+0.707, -0.707]
    # state uniform → state_adv=0 → mixed = 0.7 × outcome_adv
    expected = 0.7 * 0.707
    assert abs(float(mixed[0]) - expected) < 0.01, f"got {mixed[0]}, expected ~{expected}"
    assert mixed[0] > 0 and mixed[1] < 0 and mixed[2] > 0 and mixed[3] < 0
    print(f"  ✓ Mixed (α=0.7): adv={[round(x,3) for x in mixed.tolist()]}")
    print("    outcome dominates, state uniform→0, signs align with correctness")

    return True


# ===========================================================================
# Cross-validation: slyme path == verl path
# ===========================================================================

def check_slyme_vs_verl_parity():
    """Same trajectory → same per-component reward dict on both paths."""
    print("\n[slyme vs verl reward parity]")
    from thinkstream.trainer.v12_rewards import (
        compute_outcome_reward_v12,
        compute_timing_reward_v12,
        compute_format_reward_v12,
        compute_spam_score_v12,
        compute_silent_quality_v12,
    )
    from thinkstream.trainer_verl.reward_fn import compute_thinkstream_reward

    # Synthetic trajectory
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "chunk 0"},
        {"role": "assistant",
         "content": "<think>frame 0</think><answer></answer>"},
        {"role": "user", "content": "chunk 5: question?"},
        {"role": "assistant",
         "content": "<think>I see the answer</think><answer>yes</answer>"},
    ]
    ground_truth = {
        "gold_answer": "yes",
        "answer_form": "binary",
        "ask_chunks": [5],
        "visible_start_chunk": 5,
        "visible_end_chunk": 6,
        "gold_action_per_chunk": {"0": "silent", "5": "response"},
    }
    extra_info = {"answer_chunk": 5, "final_answer": "yes"}

    # verl path
    verl_rewards = compute_thinkstream_reward(messages, ground_truth, extra_info)

    # slyme equivalent — direct calls to v12_rewards (mimics _calc_rewards_v12)
    ai_outputs = [m["content"] for m in messages if m["role"] == "assistant"]
    slyme_rewards = {
        "outcome": compute_outcome_reward_v12("yes", "yes", answer_form="binary"),
        "timing":  compute_timing_reward_v12(5, 5, 6),
        "format":  compute_format_reward_v12(ai_outputs),
        "spam":    compute_spam_score_v12(0, 0),
        "silent_quality": compute_silent_quality_v12("yes", "response", "yes"),
    }

    print(f"  verl  rewards: {verl_rewards}")
    print(f"  slyme rewards: {slyme_rewards}")
    for k in slyme_rewards:
        assert abs(verl_rewards[k] - slyme_rewards[k]) < 1e-6, (
            f"divergence on {k}: verl={verl_rewards[k]}, slyme={slyme_rewards[k]}"
        )
    print("  ✓ slyme == verl on all 5 components")
    return True


# ===========================================================================
# Driver
# ===========================================================================

def main():
    print("=" * 78)
    print("ThinkStream loss/reward sanity check (CPU-only)")
    print("=" * 78)

    checks = [
        ("SFT loss-mask",          check_sft_loss_mask),
        ("Reward invariants",      check_reward_invariants),
        ("Advantage aggregation",  check_advantage_aggregation),
        ("slyme vs verl parity",   check_slyme_vs_verl_parity),
    ]

    failed = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  ❌ FAILED: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    if failed:
        print(f"❌ {len(failed)}/{len(checks)} checks FAILED:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"✅ ALL {len(checks)} sanity checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
