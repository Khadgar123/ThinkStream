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


def test_silent_quality_v12():
    """v12.2 silent_quality: closes silent/response error modes."""
    from thinkstream.trainer.v12_rewards import compute_silent_quality_v12 as f

    # Should be silent + WAS silent → +0.3
    assert f(None, "silent", "") == 0.3
    assert f("", "silent", "") == 0.3
    assert f("   ", "silent", "") == 0.3  # whitespace-only counts as silent

    # Should be silent + HALLUCINATED → -0.6
    assert f("red apron", "silent", "") == -0.6
    assert f("red", "silent", "") == -0.6

    # Should respond + WAS silent → -0.6 (missed)
    assert f(None, "response", "red apron") == -0.6
    assert f("", "response", "red apron") == -0.6
    # Same for recall_response
    assert f(None, "recall_response", "yes") == -0.6

    # Should respond + DID respond → 0.0 (correctness handled by outcome)
    assert f("red apron", "response", "red apron") == 0.0
    assert f("blue apron", "response", "red apron") == 0.0  # wrong but answered

    # Compress / recall_query → neutral (other rewards handle these)
    assert f(None, "compress", "") == 0.0
    assert f(None, "recall_query", "") == 0.0

    # Empty/unknown gold_action with no gold_answer → treat as silent
    assert f(None, "", "") == 0.3
    assert f("hallucinated", "", "") == -0.6

    print("✓ silent_quality_v12")


def test_trajectory_outcome_v124_single_question():
    """v12.4 trajectory outcome with single question — equivalent to v12.3
    single-question outcome semantics."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f

    rollout = [
        {"chunk_idx": 0, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 1, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 5, "kind": "answer", "answer_text": "red apron"},
    ]
    questions = [{
        "card_id": "c1", "gold_answer": "red apron",
        "answer_form": "literal", "ask_chunks": [5],
    }]
    res = f(rollout, questions)
    assert res["outcome"] == 1.0, res
    assert res["n_questions"] == 1
    assert res["n_answered"] == 1
    assert res["n_correct"] == 1
    print("✓ trajectory_outcome_v124 single-question correct")


def test_trajectory_outcome_v124_multi_question_mixed():
    """v12.4 trajectory with 3 questions — answered correctly, wrong, missed."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f

    rollout = [
        {"chunk_idx": 0, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 5, "kind": "answer", "answer_text": "red apron"},   # correct (Q1)
        {"chunk_idx": 12, "kind": "answer", "answer_text": "blue"},        # wrong (Q2)
        # Q3 ask=28 — model stays silent in entire window 28..33
        {"chunk_idx": 28, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 30, "kind": "answer", "answer_text": ""},
    ]
    questions = [
        {"card_id": "Q1", "gold_answer": "red apron",
         "answer_form": "literal", "ask_chunks": [5]},
        {"card_id": "Q2", "gold_answer": "yes",
         "answer_form": "binary", "ask_chunks": [12]},
        {"card_id": "Q3", "gold_answer": "3",
         "answer_form": "number", "ask_chunks": [28]},
    ]
    res = f(rollout, questions)
    # outcome = (1 + 0 + 0) / 3 = 0.333
    assert abs(res["outcome"] - 1/3) < 1e-6, res
    assert res["n_questions"] == 3
    assert res["n_answered"] == 2     # Q1 and Q2 answered (Q3 silent)
    assert res["n_correct"] == 1
    assert res["per_q_outcomes"] == [1.0, 0.0, 0.0]
    print(f"✓ trajectory_outcome_v124 multi-question (1 correct of 3) = {res['outcome']:.3f}")


def test_trajectory_outcome_v124_empty_questions():
    """Trajectory with zero questions (base-only) → outcome=0, no crash."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f
    res = f([{"chunk_idx": 0, "kind": "answer", "answer_text": ""}], [])
    assert res["outcome"] == 0.0
    assert res["n_questions"] == 0
    print("✓ trajectory_outcome_v124 empty-questions")


def test_trajectory_outcome_v124_multi_response_per_ask():
    """v12.4 multi-response: F7-style card with 4 ask_chunks. Model answers
    at 3 of 4 → outcome = 0.75 (not 1.0 as a fused-window would give)."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f

    rollout = [
        # Question's ask_chunks = [40, 41, 42, 45]
        {"chunk_idx": 40, "kind": "answer", "answer_text": "Yes"},   # ask 40 ✓
        {"chunk_idx": 41, "kind": "answer", "answer_text": "Yes"},   # ask 41 ✓
        {"chunk_idx": 42, "kind": "answer", "answer_text": ""},      # ask 42 ✗ (silent)
        # ask 45 — model answers at chunk 46 (within window)
        {"chunk_idx": 45, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 46, "kind": "answer", "answer_text": "Yes"},   # ask 45 ✓ (late within window)
    ]
    questions = [{
        "card_id": "F7_001", "gold_answer": "Yes",
        "answer_form": "binary", "ask_chunks": [40, 41, 42, 45],
    }]
    res = f(rollout, questions, answer_window_chunks=5)
    # Per-ask scores: 1, 1, 0, 1 → mean 0.75
    assert abs(res["outcome"] - 0.75) < 1e-6, res
    assert res["n_questions"] == 1
    # n_answered: 1 (ask40) + 1 (ask41) + 0 (ask42 silent) + 1 (ask45 via 46) = 3
    assert res["n_answered"] == 3
    assert res["n_correct"] == 3
    print(f"✓ trajectory_outcome_v124 multi-response 3/4 = {res['outcome']:.3f}")


def test_trajectory_outcome_v124_multi_response_no_overlap():
    """v12.4 — a late answer at ask_2 must NOT also count for ask_1's window."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f

    rollout = [
        # ask_chunks=[10, 12], window_chunks=5
        # Model is silent at chunk 10 (ask_1).
        # Model answers at chunk 12 (ask_2) — should ONLY count for ask_2,
        # not also for ask_1 which had window [10, 11] (next ask − 1).
        {"chunk_idx": 10, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 11, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 12, "kind": "answer", "answer_text": "Yes"},
    ]
    questions = [{
        "card_id": "M1", "gold_answer": "Yes",
        "answer_form": "binary", "ask_chunks": [10, 12],
    }]
    res = f(rollout, questions, answer_window_chunks=5)
    # ask_1 (chunk 10): window [10, 11] (since next ask=12, so window_end=11)
    #   → no answer found → score 0
    # ask_2 (chunk 12): window [12, 17] → answer "Yes" → score 1
    # mean: 0.5
    assert abs(res["outcome"] - 0.5) < 1e-6, (
        f"expected 0.5 (only ask_2 satisfied), got {res['outcome']:.3f}; "
        f"per_q_outcomes={res.get('per_q_outcomes')}"
    )
    assert res["n_answered"] == 1
    assert res["n_correct"] == 1
    print(f"✓ trajectory_outcome_v124 non-overlap windows = {res['outcome']:.3f}")


def test_trajectory_outcome_v124_single_response_unchanged():
    """v12.4 — single-response cards (91% of questions) should behave the same
    as v12.3 semantics: window = [ask, ask + answer_window]."""
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12 as f

    rollout = [
        {"chunk_idx": 4, "kind": "answer", "answer_text": ""},
        {"chunk_idx": 5, "kind": "answer", "answer_text": "red apron"},
    ]
    questions = [{
        "card_id": "F1", "gold_answer": "red apron",
        "answer_form": "literal", "ask_chunks": [5],
    }]
    res = f(rollout, questions, answer_window_chunks=5)
    assert res["outcome"] == 1.0
    print("✓ trajectory_outcome_v124 single-response unchanged")


def test_per_chunk_silent_quality_v124():
    """v12.4 per-chunk silent_quality from gold_action_per_chunk map."""
    from thinkstream.trainer.v12_rewards import compute_per_chunk_silent_quality_v12 as f

    rollout = [
        {"chunk_idx": 0, "kind": "answer", "answer_text": ""},        # gold=silent ✓ +0.3
        {"chunk_idx": 1, "kind": "answer", "answer_text": "talky"},   # gold=silent ✗ -0.6
        {"chunk_idx": 5, "kind": "answer", "answer_text": "red"},     # gold=response ✓ 0.0
        {"chunk_idx": 6, "kind": "answer", "answer_text": ""},        # gold=response ✗ -0.6
        {"chunk_idx": 9, "kind": "compress", "answer_text": None},    # gold=compress neutral
    ]
    gold_map = {
        "0": "silent", "1": "silent", "5": "response",
        "6": "response", "9": "compress",
    }
    res = f(rollout, gold_map)
    # 4 scored chunks; sum = +0.3 - 0.6 + 0 - 0.6 = -0.9; mean = -0.225
    assert res["n_chunks_scored"] == 4, res
    assert abs(res["silent_quality"] - (-0.225)) < 1e-6, res
    assert res["n_correct_silent"] == 1
    assert res["n_hallucinate"] == 1
    assert res["n_missed"] == 1
    print(f"✓ per_chunk_silent_quality_v124 = {res['silent_quality']:.3f}")


def test_per_chunk_silent_quality_perfect_silence():
    """All-silent rollout where gold matches → mean = +0.3."""
    from thinkstream.trainer.v12_rewards import compute_per_chunk_silent_quality_v12 as f

    rollout = [
        {"chunk_idx": i, "kind": "answer", "answer_text": ""} for i in range(5)
    ]
    gold_map = {str(i): "silent" for i in range(5)}
    res = f(rollout, gold_map)
    assert res["silent_quality"] == 0.3
    assert res["n_correct_silent"] == 5
    print(f"✓ per_chunk_silent_quality_v124 perfect = {res['silent_quality']}")


def test_silent_quality_v12_complements_outcome():
    """Verify silent_quality fills the reward gap that outcome alone misses.

    Scenario the audit identified: gold_action='silent', gold_answer='',
    model emits a hallucinated response. With ONLY outcome+timing rewards,
    this scores 0 (both masked). With silent_quality added, it scores -0.6.
    """
    from thinkstream.trainer.v12_rewards import (
        compute_silent_quality_v12 as f_silent,
        compute_outcome_reward_v12 as f_outcome,
    )

    # Hallucinate-when-should-be-silent
    outcome = f_outcome("hallucinated answer", "", answer_form="literal")
    silent = f_silent("hallucinated answer", "silent", "")
    # Pre-v12.2: outcome=0 (no gold) → 0 total reward (BUG)
    assert outcome == 0.0
    # Post-v12.2: silent_quality=-0.6 → caller now penalizes correctly
    assert silent == -0.6

    # Silent-when-should-respond
    outcome2 = f_outcome(None, "red apron", answer_form="literal")
    silent2 = f_silent(None, "response", "red apron")
    # Pre-v12.2: outcome=0 (no answer to score), timing=-0.5×0.3=-0.15 only
    assert outcome2 == 0.0
    # Post-v12.2: silent_quality=-0.6 strengthens the signal
    assert silent2 == -0.6

    print("✓ silent_quality_v12 closes outcome gap")


if __name__ == "__main__":
    test_v12_reward_keys_match()
    test_outcome_v12()
    test_timing_v12()
    test_format_v12()
    test_spam_v12()
    test_compress_quality_v12()
    test_recall_quality_v12()
    test_silent_quality_v12()
    test_trajectory_outcome_v124_single_question()
    test_trajectory_outcome_v124_multi_question_mixed()
    test_trajectory_outcome_v124_empty_questions()
    test_trajectory_outcome_v124_multi_response_per_ask()
    test_trajectory_outcome_v124_multi_response_no_overlap()
    test_trajectory_outcome_v124_single_response_unchanged()
    test_per_chunk_silent_quality_v124()
    test_per_chunk_silent_quality_perfect_silence()
    test_silent_quality_v12_complements_outcome()
    test_v12_advantage_aggregation()
    print("\n✅ all v12.0 reward smoke tests passed")
