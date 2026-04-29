"""v12.0 RL reward function smoke tests.

Tests the 5 v12 reward components from thinkstream/trainer/grpo.py:
- outcome (binary correctness with anti-hacking)
- timing (bucket: early/-1, on-time/+1, late_partial/+0.5, missed/-0.5)
- format (binary: all turns parse cleanly)
- spam (additive penalty for excess tool calls)
- compress_quality (range_iou + text_match for compress turns)

Plus the multi-level GRPO advantage aggregation.

Run: python tests/test_v12_rewards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def test_outcome_v12():
    from thinkstream.trainer.v12_rewards import compute_outcome_reward_v12 as f

    # Literal exact match
    assert f("red", "red", answer_form="literal") == 1.0
    assert f("Red", "red", answer_form="literal") == 1.0  # case-insensitive
    assert f("blue", "red", answer_form="literal") == 0.0

    # No answer
    assert f(None, "red") == 0.0
    assert f("red", "") == 0.0

    # Anti-hacking length cap
    assert f("a" * 1001, "a" * 1001) == 0.0

    # Descriptive: fuzzy substring fallback
    assert f("the answer is red apple", "red apple") == 1.0
    assert f("blueberry", "red") == 0.0

    # Custom judge
    assert f("answer", "gold", judge_fn=lambda a, b: 0.7) == 0.7

    print("✓ outcome_v12")


def test_timing_v12():
    from thinkstream.trainer.v12_rewards import compute_timing_reward_v12 as f

    # On-time → +1
    assert f(answer_chunk=5, visible_start_chunk=3, visible_end_chunk=7) == 1.0
    assert f(answer_chunk=3, visible_start_chunk=3, visible_end_chunk=7) == 1.0  # at start
    assert f(answer_chunk=7, visible_start_chunk=3, visible_end_chunk=7) == 1.0  # at end

    # Hallucination (early) → -1
    assert f(answer_chunk=2, visible_start_chunk=3, visible_end_chunk=7) == -1.0
    assert f(answer_chunk=0, visible_start_chunk=3, visible_end_chunk=7) == -1.0

    # Missed (silent through window) → -0.5
    assert f(answer_chunk=None, visible_start_chunk=3, visible_end_chunk=7) == -0.5

    # Late partial → linear decay 1.0 → 0.5
    r = f(answer_chunk=8, visible_start_chunk=3, visible_end_chunk=7, late_window_chunks=2)
    assert abs(r - 0.75) < 1e-6, r  # delay=1, half-decay = 1.0 - 0.5*(1/2) = 0.75
    r = f(answer_chunk=9, visible_start_chunk=3, visible_end_chunk=7, late_window_chunks=2)
    assert abs(r - 0.5) < 1e-6, r  # delay=2 = full window, score = 0.5

    # Way late → 0
    assert f(answer_chunk=20, visible_start_chunk=3, visible_end_chunk=7, late_window_chunks=2) == 0.0

    # No visibility window → neutral
    assert f(answer_chunk=5, visible_start_chunk=None, visible_end_chunk=None) == 0.0

    # visible_end=None (open window) → on-time
    assert f(answer_chunk=10, visible_start_chunk=3, visible_end_chunk=None) == 1.0

    print("✓ timing_v12 (early=-1, on=+1, late=decay, missed=-0.5)")


def test_format_v12():
    from thinkstream.trainer.v12_rewards import compute_format_reward_v12 as f

    # All turns parse
    outs = [
        "<think>x</think><tool_call>\n{\"name\":\"recall\",\"arguments\":{\"query\":\"q\",\"time_range\":\"1-5\"}}\n</tool_call>",
        "<think>y</think><answer>red</answer>",
    ]
    assert f(outs) == 1.0

    # Single turn parses
    assert f(["<think>x</think><answer></answer>"]) == 1.0

    # One bad → all fails
    bad = ["<think>x</think><answer>red</answer>", "<think>y</think>"]  # second has no terminal
    assert f(bad) == 0.0

    # Bad JSON
    assert f(["<think>x</think><tool_call>not json</tool_call>"]) == 0.0

    # Empty
    assert f([]) == 0.0

    print("✓ format_v12")


def test_spam_v12():
    from thinkstream.trainer.v12_rewards import compute_spam_score_v12 as f

    # Within budget: 0
    assert f(n_recall_calls=1, n_compress_calls=1) == 0.0
    assert f(n_recall_calls=0, n_compress_calls=0) == 0.0

    # Excess recall: 0.5 each
    assert f(n_recall_calls=2, n_compress_calls=1) == 0.5
    assert f(n_recall_calls=3, n_compress_calls=1) == 1.0

    # Excess compress: 0.3 each
    assert f(n_recall_calls=1, n_compress_calls=2) == 0.3
    assert f(n_recall_calls=1, n_compress_calls=3) == 0.6

    # Both excess
    assert f(n_recall_calls=2, n_compress_calls=2) == 0.8

    # Custom budgets
    assert f(n_recall_calls=2, n_compress_calls=0, recall_budget=2) == 0.0

    print("✓ spam_v12")


def test_compress_quality_v12():
    from thinkstream.trainer.v12_rewards import compute_compress_quality_v12 as f

    # Perfect: same range, identical text
    r = f(
        summary_text="chef cooks pasta",
        summary_range=[4, 12],
        gold_summary_text="chef cooks pasta",
        gold_range=[4, 12],
    )
    assert abs(r - 1.0) < 1e-6, r  # iou=1.0, text=1.0 (all gold tokens covered)

    # Disjoint range
    r = f(
        summary_text="chef cooks pasta",
        summary_range=[4, 8],
        gold_summary_text="chef cooks pasta",
        gold_range=[10, 14],
    )
    # iou = 0/8 = 0, text = 1.0 → score = 0.5
    assert abs(r - 0.5) < 1e-6, r

    # Partial range overlap
    r = f(
        summary_text="chef cooks",
        summary_range=[4, 10],
        gold_summary_text="chef cooks pasta tomato",
        gold_range=[6, 12],
    )
    # iou = 4/8 = 0.5, text = 2/4 = 0.5 → score = 0.5
    assert abs(r - 0.5) < 1e-6, r

    # No data
    assert f(None, None, None, None) == 0.0
    assert f("text", None, None, None) == 0.0

    print("✓ compress_quality_v12")


def test_v12_advantage_aggregation():
    """Test multi-level GRPO advantage: 2 videos × 4 rollouts each, 3 chunks per rollout."""
    from thinkstream.trainer.v12_rewards import aggregate_v12_advantages

    # Setup: 2 videos × 4 rollouts × 3 chunks = 24 rows
    n_video = 2
    group_size = 4
    n_chunks = 3
    B = n_video * group_size * n_chunks

    chunk_to_video_uid = torch.tensor(
        [v for v in range(n_video) for _ in range(group_size * n_chunks)]
    )
    chunk_idx_per_row = torch.tensor(
        [c for _ in range(n_video) for _ in range(group_size) for c in range(n_chunks)]
    )

    # All-correct outcomes for video 0, all-wrong for video 1
    outcome = torch.zeros(B)
    outcome[:group_size * n_chunks] = 1.0  # video 0: all 1.0
    # video 1: all 0.0 (default)

    # Timing: rollout 0 of each video gets +1, others 0
    timing = torch.zeros(B)
    for v in range(n_video):
        for c in range(n_chunks):
            row = v * group_size * n_chunks + 0 * n_chunks + c  # rollout 0
            timing[row] = 1.0

    rewards = {"outcome": outcome, "timing": timing,
               "format": torch.ones(B), "spam": torch.zeros(B),
               "compress_quality": torch.zeros(B)}
    masks = {k: torch.ones(B) for k in rewards}
    masks["compress_quality"] = torch.zeros(B)  # no compress in test

    adv = aggregate_v12_advantages(
        rewards, masks, chunk_to_video_uid, chunk_idx_per_row,
        group_size=group_size, alpha=0.7,
    )

    assert adv.shape == (B,)

    # All chunks of video 0 should have positive outcome advantage (above
    # video-0 group mean of 1.0 — wait, video 0 group is uniform 1.0 so
    # outcome_adv per video-0 chunk = 0). Hmm let me reconsider.
    # outcome_adv groups by video uid: video 0 has all rollouts = 1.0, so
    # group mean = 1.0, so each chunk's outcome_adv = 0.
    # Video 1 same: all 0.0 → group mean 0.0 → adv 0.
    # So outcome_adv is all 0 in this test. That's fine.

    # State_advantage: timing on rollout 0 of each video should be > rollout
    # 1/2/3 (which had timing=0). So adv[rollout_0_chunks] > adv[rollout_1_chunks].
    # Rollout 0 of video 0: rows 0,1,2 — timing=1, others=0. State sum = 1*0.3 + 1*0.1 = 0.4
    # Rollout 1-3 of video 0: rows 3..11 — timing=0, format=1. State = 0.1
    # Per-chunk-position group: 4 rollouts at chunk 0 → values [0.4, 0.1, 0.1, 0.1], mean=0.175
    # Rollout 0 chunk 0 state_adv = 0.4 - 0.175 = 0.225
    # Rollout 1 chunk 0 state_adv = 0.1 - 0.175 = -0.075
    # final_adv = 0.7*0 + 0.3*0.225 = 0.0675 (rollout 0)
    # final_adv = 0.7*0 + 0.3*-0.075 = -0.0225 (rollout 1)
    rollout_0_chunk_0 = adv[0].item()
    rollout_1_chunk_0 = adv[n_chunks].item()
    assert rollout_0_chunk_0 > rollout_1_chunk_0, (
        f"rollout 0 should have higher advantage than rollout 1: "
        f"{rollout_0_chunk_0} vs {rollout_1_chunk_0}"
    )
    assert abs(rollout_0_chunk_0 - 0.0675) < 1e-3, rollout_0_chunk_0

    print("✓ v12 multi-level advantage aggregation")


def test_v12_reward_keys_match():
    """V12_REWARD_DICT_KEYS and V12_DEFAULT_REWARD_WEIGHTS must agree."""
    from thinkstream.trainer.gdpo_advantage import (
        V12_REWARD_DICT_KEYS, V12_DEFAULT_REWARD_WEIGHTS,
    )
    keys_set = set(V12_REWARD_DICT_KEYS)
    weights_set = set(V12_DEFAULT_REWARD_WEIGHTS.keys())
    assert keys_set == weights_set, (
        f"key/weight mismatch: keys-only={keys_set - weights_set}, "
        f"weights-only={weights_set - keys_set}"
    )
    # spam weight must be negative (additive penalty)
    assert V12_DEFAULT_REWARD_WEIGHTS["spam"] < 0
    print("✓ v12 reward keys + weights consistency")


if __name__ == "__main__":
    test_v12_reward_keys_match()
    test_outcome_v12()
    test_timing_v12()
    test_format_v12()
    test_spam_v12()
    test_compress_quality_v12()
    test_v12_advantage_aggregation()
    print("\n✅ all v12.0 reward smoke tests passed")
