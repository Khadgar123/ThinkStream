"""v12.0 RL reward function smoke tests.

Tests the v12 reward components from thinkstream/trainer/rewards.py:
- outcome (binary correctness with anti-hacking)
- answer_decision (answer/no-answer timing decision)
- timing (bucket: early/-1, on-time/+1, late_partial/+0.5, missed/-0.5)
- format (binary: all turns parse cleanly)

Plus the multi-level GRPO advantage aggregation.

Run: python tests/test_rewards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "verl"))

import torch


def test_outcome_v12():
    from thinkstream.trainer.rewards import compute_outcome_reward as f

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

    # Form-aware fallback: MCQ should still score by letter/correct_option even
    # if legacy rows are missing the options list.
    assert f("C", "blue", answer_form="multiple_choice", correct_option="C") == 1.0
    assert f("A", "blue", answer_form="multiple_choice", correct_option="C") == 0.0
    assert f(
        "red apple",
        "red apple",
        answer_form="multiple_choice",
        correct_option="",
        options=[],
    ) == 1.0

    # Other generated answer forms.
    assert f("Yes.", "yes", answer_form="binary") == 1.0
    assert f("$12.00", "12", answer_form="number") == 1.0
    assert f("the red apron", "red apron", answer_form="short_exact") == 1.0

    # Custom judge
    assert f("answer", "gold", judge_fn=lambda a, b: 0.7) == 0.7

    print("✓ outcome_v12")


def test_timing_v12():
    from thinkstream.trainer.rewards import compute_timing_reward as f

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


def test_answer_decision_v12():
    from thinkstream.trainer.rewards import compute_answer_decision_reward as f

    assert f(answer_chunk=5, visible_start_chunk=5, visible_end_chunk=5) == 1.0
    assert f(answer_chunk=4, visible_start_chunk=5, visible_end_chunk=5) == -1.0
    assert f(answer_chunk=None, visible_start_chunk=5, visible_end_chunk=5) == -1.0
    assert f(answer_chunk=None, visible_start_chunk=None, visible_end_chunk=None) == 0.0
    assert f(
        answer_chunk=2,
        visible_start_chunk=None,
        visible_end_chunk=None,
        has_answer=True,
    ) == -1.0

    print("✓ answer_decision_v12 (slot/event reward, normal silence neutral)")


def test_format_v12():
    from thinkstream.trainer.rewards import compute_format_reward as f

    # All turns parse
    outs = [
        "<think>x</think><tool_call>\n{\"name\":\"recall\",\"arguments\":{\"start_time\":1,\"end_time\":5}}\n</tool_call>",
        "<think>y</think></Response> red",
    ]
    assert f(outs) == 1.0

    # Single turn parses
    assert f(["<think>x</think></Silence>"]) == 1.0

    # One bad → all fails
    bad = ["<think>x</think></Response> red", "<think>y</think>"]  # second has no terminal
    assert f(bad) == 0.0

    # Bad JSON
    assert f(["<think>x</think><tool_call>not json</tool_call>"]) == 0.0

    # Bad tool schemas
    assert f([
        '<think>x</think><tool_call>{"name":"recall","arguments":{"start_time":1}}</tool_call>'
    ]) == 0.0
    assert f([
        '<think>x</think><tool_call>{"name":"compress","arguments":{"time_range":"1-5","text":"s"}}</tool_call>'
    ]) == 0.0

    # Extra text outside the protocol skeleton is not valid format.
    assert f(["<think>x</think></Silence>\nextra"]) == 0.0

    # Empty
    assert f([]) == 0.0

    print("✓ format_v12")


def test_v12_advantage_aggregation():
    """Test multi-level GRPO advantage: 2 videos × 4 rollouts each, 3 chunks per rollout."""
    from thinkstream.trainer.rewards import aggregate_advantages

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

    # Answer-decision: rollout 0 of each video gets +1, others 0
    answer_decision = torch.zeros(B)
    for v in range(n_video):
        for c in range(n_chunks):
            row = v * group_size * n_chunks + 0 * n_chunks + c  # rollout 0
            answer_decision[row] = 1.0

    rewards = {"outcome": outcome, "answer_decision": answer_decision,
               "format": torch.ones(B),
               "compress_quality": torch.zeros(B)}
    masks = {k: torch.ones(B) for k in rewards}
    masks["compress_quality"] = torch.zeros(B)  # no compress in test

    adv = aggregate_advantages(
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

    # State_advantage: answer_decision on rollout 0 of each video should be
    # > rollout 1/2/3. So adv[rollout_0_chunks] > adv[rollout_1_chunks].
    # Rollout 0 of video 0: rows 0,1,2 — answer_decision=1, others=0.
    # State sum = 1*0.3 + 1*0.1 = 0.4
    # Rollout 1-3 of video 0: rows 3..11 — answer_decision=0, format=1.
    # State = 0.1
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
    print("✓ v12 reward keys + weights consistency")


def test_silent_quality_v12():
    """v12.2 silent_quality: closes silent/response error modes."""
    from thinkstream.trainer.rewards import compute_silent_quality as f

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
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f

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
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f

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
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f
    res = f([{"chunk_idx": 0, "kind": "answer", "answer_text": ""}], [])
    assert res["outcome"] == 0.0
    assert res["n_questions"] == 0
    print("✓ trajectory_outcome_v124 empty-questions")


def test_trajectory_outcome_v124_multi_response_per_ask():
    """v12.4 multi-response: F7-style card with 4 ask_chunks. Model answers
    at 3 of 4 → outcome = 0.75 (not 1.0 as a fused-window would give)."""
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f

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
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f

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
    from thinkstream.trainer.rewards import compute_trajectory_outcome as f

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
    from thinkstream.trainer.rewards import compute_per_chunk_silent_quality as f

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
    from thinkstream.trainer.rewards import compute_per_chunk_silent_quality as f

    rollout = [
        {"chunk_idx": i, "kind": "answer", "answer_text": ""} for i in range(5)
    ]
    gold_map = {str(i): "silent" for i in range(5)}
    res = f(rollout, gold_map)
    assert res["silent_quality"] == 0.3
    assert res["n_correct_silent"] == 5
    print(f"✓ per_chunk_silent_quality_v124 perfect = {res['silent_quality']}")


def test_recall_silent_requires_recall_check():
    from thinkstream.trainer.rewards import compute_per_chunk_silent_quality as f

    recall_ok = f(
        [{"chunk_idx": 5, "kind": "recall", "answer_text": None}],
        {"5": "recall_silent"},
    )
    skipped_recall = f(
        [{"chunk_idx": 5, "kind": "answer", "answer_text": ""}],
        {"5": "recall_silent"},
    )
    hallucinated = f(
        [{"chunk_idx": 5, "kind": "answer", "answer_text": "red cup"}],
        {"5": "recall_silent"},
    )

    assert recall_ok["silent_quality"] == 0.3
    assert skipped_recall["silent_quality"] == 0.0
    assert hallucinated["silent_quality"] == -0.6


def test_recipe_reward_gates_positive_auxiliary_on_correct_answer():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    parts = {
        "outcome": 0.0,
        "answer_decision": 1.0,
        "format": 1.0,
        "timing": 1.0,
        "silent_quality": 0.3,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 0.0
    assert score == 0.0


def test_recipe_reward_keeps_negative_auxiliary_when_answer_wrong():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    parts = {
        "outcome": 0.0,
        "answer_decision": -1.0,
        "format": 1.0,
        "timing": -1.0,
        "silent_quality": -0.6,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 0.0
    assert abs(score - (-0.3)) < 1e-6


def test_recipe_reward_allows_auxiliary_when_answer_correct():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    parts = {
        "outcome": 1.0,
        "answer_decision": 1.0,
        "format": 1.0,
        "timing": 1.0,
        "silent_quality": 0.3,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 1.0
    assert abs(score - 1.4) < 1e-6


def test_recipe_reward_includes_compress_quality_in_global_scalar():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "compress_quality": 0.1,
    }
    parts = {
        "outcome": 1.0,
        "compress_quality": 0.8,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 1.0
    assert abs(score - 1.08) < 1e-6


def test_recipe_reward_gates_positive_compress_quality_when_answer_wrong():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "compress_quality": 0.1,
    }
    parts = {
        "outcome": 0.0,
        "compress_quality": 1.0,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 0.0
    assert score == 0.0


def test_recipe_reward_scales_auxiliary_on_partial_outcome():
    from thinkstream.rl.thinkstream import _combine_reward_parts

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    parts = {
        "outcome": 0.5,
        "answer_decision": 1.0,
        "format": 1.0,
        "timing": 1.0,
        "silent_quality": 0.3,
    }

    score, gate = _combine_reward_parts(weights, parts)
    assert gate == 0.5
    # 0.5 outcome + 0.5 * (0.3 answer_decision + 0.1 format)
    assert abs(score - 0.7) < 1e-6


def test_recipe_multi_q_reward_gates_each_question_independently():
    from thinkstream.rl.thinkstream import _compute_score_multi_q
    from thinkstream.trainer.rewards import (
        compute_timing_reward,
        compute_answer_decision_reward,
        compute_silent_quality,
    )

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    rewards = {
        "outcome": lambda *a, **k: 0.0,
        "timing": compute_timing_reward,
        "answer_decision": compute_answer_decision_reward,
        "format": lambda chunks: 1.0,
        "silent_quality": compute_silent_quality,
    }
    questions = [
        {
            "gold_answer": "red",
            "answer_form": "short_exact",
            "ask_chunk": 5,
            "ask_chunks": [5],
            "answer_chunks": [5],
        },
        {
            "gold_answer": "yes",
            "answer_form": "binary",
            "ask_chunk": 12,
            "ask_chunks": [12],
            "answer_chunks": [12],
        },
        {
            "gold_answer": "3",
            "answer_form": "number",
            "ask_chunk": 20,
            "ask_chunks": [20],
            "answer_chunks": [20],
        },
    ]
    extra = {
        "ts_per_q_answer_chunk": [5, 12, -1],
        "ts_per_q_answer_text": ["red", "no", ""],
        "ts_per_q_answers": [[], [], []],
    }

    res = _compute_score_multi_q(
        rewards,
        weights,
        questions,
        extra,
        "<think>ok</think></Response> red",
    )

    assert abs(res["outcome"] - (1 / 3)) < 1e-6, res
    # Raw timing is answer-weighted; here each question has one answer slot.
    assert abs(res["timing"] - 0.5) < 1e-6, res
    # answer_decision does not award dense normal silence; missed slots are -1.
    assert abs(res["answer_decision"] - (1 / 3)) < 1e-6, res
    assert abs(res["outcome_gate"] - (1 / 3)) < 1e-6, res
    # Score:
    # Q1: 1 outcome + 0.3 answer_decision = 1.3
    # Q2: wrong but timely, positive answer_decision is gated to 0
    # Q3: missed answer, answer_decision -0.3 applies
    # Mean question score = (1.3 + 0 - 0.3) / 3, plus format 0.1 * 1/3.
    expected = ((1.3 + 0.0 - 0.3) / 3.0) + (0.1 / 3.0)
    assert abs(res["score"] - expected) < 1e-6, res


def test_recipe_multi_q_rewards_recall_labeled_answer_when_recalled_and_correct():
    from thinkstream.rl.thinkstream import _compute_score_multi_q
    from thinkstream.trainer.rewards import (
        compute_timing_reward,
        compute_answer_decision_reward,
        compute_silent_quality,
    )

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    rewards = {
        "outcome": lambda *a, **k: 0.0,
        "timing": compute_timing_reward,
        "answer_decision": compute_answer_decision_reward,
        "format": lambda chunks: 1.0,
        "silent_quality": compute_silent_quality,
    }
    questions = [{
        "gold_answer": "red",
        "answer_form": "short_exact",
        "ask_chunk": 5,
        "ask_chunks": [5],
        "answer_chunks": [5],
    }]
    extra = {
        "gold_action_per_chunk": {"5": "recall"},
        "ts_chunk_kinds": ["recall"],
        "ts_chunk_turn_kinds": ["recall"],
        "ts_chunk_video_indices": [5],
        "ts_chunk_event_indices": [5],
        "ts_chunk_asst_texts": [
            '<think>need history</think><tool_call>{"name":"recall","arguments":{"start_time":0,"end_time":3}}</tool_call>',
        ],
        "ts_chunk_action_space_errors": [""],
        "ts_per_q_answer_chunk": [5],
        "ts_per_q_answer_text": ["red"],
        "ts_per_q_answers": [[]],
    }

    res = _compute_score_multi_q(
        rewards,
        weights,
        questions,
        extra,
        "<think>ok</think></Response> red",
    )

    assert res["recall_answer_labeled"] == 1.0, res
    assert res["recall_answer_used"] == 1.0, res
    assert res["recall_answer_success"] == 1.0, res
    assert res["recall_answer"] == 1.0, res
    # 1.0 outcome + 0.3 answer_decision + 0.2 recall_answer + 0.1 format.
    assert abs(res["score"] - 1.6) < 1e-6, res


def test_recipe_multi_q_aggregates_by_expected_answer_slot():
    from thinkstream.rl.thinkstream import _compute_score_multi_q
    from thinkstream.trainer.rewards import (
        compute_timing_reward,
        compute_answer_decision_reward,
        compute_silent_quality,
    )

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    rewards = {
        "outcome": lambda *a, **k: 0.0,
        "timing": compute_timing_reward,
        "answer_decision": compute_answer_decision_reward,
        "format": lambda chunks: 1.0,
        "silent_quality": compute_silent_quality,
    }
    questions = [
        {
            "gold_answer": "",
            "answer_form": "short_exact",
            "ask_chunks": [5],
            "answer_chunks": [5, 10],
            "per_emit_answers": [
                {"chunk": 5, "value": "red"},
                {"chunk": 10, "value": "blue"},
            ],
        },
        {
            "gold_answer": "yes",
            "answer_form": "binary",
            "ask_chunks": [20],
            "answer_chunks": [20],
        },
    ]
    extra = {
        "ts_per_q_answer_chunk": [-1, 20],
        "ts_per_q_answer_text": ["", "no"],
        "ts_per_q_answers": [
            [
                {"chunk": 5, "text": "red"},
                {"chunk": 10, "text": "wrong"},
            ],
            [{"chunk": 20, "text": "no"}],
        ],
    }

    res = _compute_score_multi_q(
        rewards,
        weights,
        questions,
        extra,
        "<think>ok</think></Response> no",
    )

    assert res["n_questions"] == 2.0, res
    assert res["n_answers"] == 3.0, res
    # Q1 has two answer slots with outcomes [1, 0], Q2 has one wrong answer.
    assert abs(res["outcome"] - (1.0 / 3.0)) < 1e-6, res
    assert abs(res["answer_decision"] - 1.0) < 1e-6, res
    assert abs(res["outcome_gate"] - (1.0 / 3.0)) < 1e-6, res
    # Per-answer score: (correct 1.3 + wrong 0 + wrong 0) / 3,
    # plus trajectory format 0.1 gated by the answer-weighted outcome gate.
    expected = (1.3 / 3.0) + (0.1 / 3.0)
    assert abs(res["score"] - expected) < 1e-6, res


def test_recipe_multi_q_ignores_future_questions_past_rollout_horizon():
    from thinkstream.rl.thinkstream import _compute_score_multi_q
    from thinkstream.trainer.rewards import (
        compute_timing_reward,
        compute_answer_decision_reward,
        compute_silent_quality,
    )

    weights = {
        "outcome": 1.0,
        "answer_decision": 0.3,
        "format": 0.1,
    }
    rewards = {
        "outcome": lambda *a, **k: 0.0,
        "timing": compute_timing_reward,
        "answer_decision": compute_answer_decision_reward,
        "format": lambda chunks: 1.0,
        "silent_quality": compute_silent_quality,
    }
    questions = [
        {
            "gold_answer": "red",
            "answer_form": "short_exact",
            "ask_chunks": [5],
            "answer_chunks": [5],
        },
        {
            "gold_answer": "blue",
            "answer_form": "short_exact",
            "ask_chunks": [90],
            "answer_chunks": [90],
        },
    ]
    extra = {
        "ts_chunk_video_indices": [0, 1, 2, 3, 4, 5],
        "ts_per_q_answer_chunk": [5, -1],
        "ts_per_q_answer_text": ["red", ""],
        "ts_per_q_answers": [[], []],
    }

    res = _compute_score_multi_q(
        rewards,
        weights,
        questions,
        extra,
        "<think>ok</think></Response> red",
    )

    assert res["n_questions"] == 1.0, res
    assert res["n_questions_total"] == 2.0, res
    assert res["n_questions_excluded_future"] == 1.0, res
    assert res["n_answered"] == 1.0, res
    assert res["outcome"] == 1.0, res


def test_recipe_action_shaping_scores_system_compress_only():
    from thinkstream.rl.thinkstream import _per_chunk_action_avg

    extra = {
        "ts_chunk_kinds": ["answer", "compress", "answer"],
        "ts_chunk_asst_texts": [
            "<think>x</think></Response> A",
            (
                "<think>x</think><tool_call>"
                '{"name":"compress","arguments":{"time_range":[0,1],"text":"x"}}'
                "</tool_call>"
            ),
            "<think>x</think></Response> B",
        ],
        "ts_chunk_video_indices": [0, -1, 2],
        "ts_chunk_turn_kinds": ["streaming", "compress", "streaming"],
    }
    # Initial RL keeps compression monitor-only. The offline compress label at
    # chunk 0 and the live system compress turn are neutral; only the response
    # action contributes.
    assert _per_chunk_action_avg(extra, {"0": "compress", "2": "response"}) == 0.1

    bad_extra = dict(extra)
    bad_extra["ts_chunk_kinds"] = ["answer", "answer", "answer"]
    assert _per_chunk_action_avg(bad_extra, {"0": "compress", "2": "response"}) == 0.1
    # Explicit opt-in restores the old live-compress shaping for ablations.
    assert (
        _per_chunk_action_avg(
            bad_extra,
            {"0": "compress", "2": "response"},
            score_compress=True,
        )
        == 0.025
    )


def test_rl_recall_start_end_runtime_accepts_history_and_rejects_future():
    from thinkstream.rl.thinkstream import _tool_time_range_runtime_ok

    assert _tool_time_range_runtime_ok(
        "recall",
        {"start_time": 20, "end_time": 21},
        current_chunk=31,
    )
    assert _tool_time_range_runtime_ok(
        "recall",
        {"start_time": 30, "end_time": 30},
        current_chunk=31,
    )
    assert not _tool_time_range_runtime_ok(
        "recall",
        {"start_time": 31, "end_time": 31},
        current_chunk=31,
    )
    assert not _tool_time_range_runtime_ok(
        "recall",
        {"start_time": 31, "end_time": 99},
        current_chunk=31,
    )


def test_segment_runtime_bounds_keep_late_absolute_chunks():
    from thinkstream.rl.streaming_agent_loop import _runtime_chunk_bounds

    # Full-video mode keeps the historical prefix cap semantics.
    assert _runtime_chunk_bounds(
        max_chunks=64,
        n_chunks_dataset=200,
        latest_ask_chunk=140,
    ) == (0, 64)

    # Segment mode uses absolute source-video chunk ids. MAX_CHUNKS is not an
    # absolute time ceiling, otherwise a late segment would collapse to chunk 63
    # and all current/recall frame paths would be resolved against the wrong
    # part of the video.
    assert _runtime_chunk_bounds(
        max_chunks=64,
        n_chunks_dataset=200,
        segment_start_chunk=132,
        segment_end_chunk=178,
        latest_ask_chunk=140,
    ) == (132, 179)

    # Correctness-first windows that exceed segment_max_chunks should still be
    # honored by runtime once the dataset has materialized them.
    assert _runtime_chunk_bounds(
        max_chunks=64,
        n_chunks_dataset=200,
        segment_start_chunk=0,
        segment_end_chunk=198,
    ) == (0, 199)


def test_segment_recall_direct_frame_range_uses_absolute_past_chunks():
    from thinkstream.rl.streaming_agent_loop import _recall_chunks_from_time_range

    assert _recall_chunks_from_time_range(
        (20, 24),
        chunk_sec=1.0,
        current_chunk=132,
    ) == [20, 21, 22, 23, 24]
    assert _recall_chunks_from_time_range(
        (130, 140),
        chunk_sec=1.0,
        current_chunk=132,
    ) == [130, 131]


def _single_q_dataset_stub():
    from thinkstream.rl.thinkstream import CustomRLHFDataset

    ds = object.__new__(CustomRLHFDataset)
    ds.segment_pre_context = 8
    ds.segment_post_context = 8
    ds.segment_max_chunks = 64
    ds.segment_require_recall_archive = True
    return ds


def test_rl_episode_mode_segment_alias():
    from thinkstream.rl.thinkstream import CustomRLHFDataset

    assert CustomRLHFDataset._normalize_episode_mode("segment") == "single_question"
    assert CustomRLHFDataset._normalize_episode_mode("single-question") == "single_question"
    assert CustomRLHFDataset._normalize_episode_mode("full-video") == "full"


def test_rl_dataset_recovers_legacy_offline_compress_boundaries():
    from thinkstream.rl.thinkstream import (
        _merge_offline_compress_chunks,
        _strip_offline_compress_actions,
    )

    legacy_gold = {
        "12": "silent",
        "32": "compress",
        "63": "response",
        "bad": "compress",
    }

    assert _strip_offline_compress_actions(legacy_gold) == {
        "12": "silent",
        "63": "response",
    }
    assert _merge_offline_compress_chunks(None, legacy_gold) == [32]
    assert _merge_offline_compress_chunks([8, "32"], legacy_gold) == [8, 32]
    assert (
        _merge_offline_compress_chunks([8], legacy_gold, start_chunk=20, end_chunk=40)
        == [32]
    )


def test_rl_compress_output_transfers_student_memory_state():
    from thinkstream.trainer.rollout import VideoTrajectoryState, default_update_state

    state = VideoTrajectoryState(video_uid="video-a", chunk_idx=32)
    state.compressed_summaries = [{"time_range": [0, 8], "text": "old teacher-ish memory"}]
    state.recent_thinks = [
        {"chunk": 28, "text": "student saw a red cup"},
        {"chunk": 29, "text": "student saw the cup move"},
    ]

    student_output = (
        "<think>Compact my visible notes before continuing.</think>"
        "  <m t=\"0-7\">Student summary of the first scene.</m>\n"
        "  <m t=\"8-15\">Student summary of the second scene.</m>\n"
        "  <m t=\"16-23\">Student summary of the third scene.</m>\n"
        "  <m t=\"24-31\">Student summary of the latest scene.</m>\n"
    )

    new_state = default_update_state(state, student_output, chunk_idx=32)

    assert new_state.n_compress_calls == 1
    assert [seg["text"] for seg in new_state.compressed_summaries] == [
        "Student summary of the first scene.",
        "Student summary of the second scene.",
        "Student summary of the third scene.",
        "Student summary of the latest scene.",
    ]
    assert new_state.recent_thinks == []
    assert state.recent_thinks, "default_update_state must not mutate the caller state"


def test_single_question_window_boundaries_and_scalar_chunks():
    ds = _single_q_dataset_stub()

    assert ds._safe_int_list(5) == [5]
    assert ds._safe_int_list(None) == []
    assert ds._safe_int_list(["2", "bad", -1]) == [2, -1]

    q = {
        "ask_chunk": "40",
        "answer_chunks": 45,
        "support_chunks": [10, 12, 99, -2],
    }
    # start anchors around the ask chunk, end covers the farthest valid target.
    assert ds._window_for_question(q, n_chunks=60) == (32, 53)

    long_q = {
        "ask_chunks": [140],
        "answer_chunks": [170],
        "support_chunks": [20],
    }
    # Earlier support is represented through student prefix state; online
    # segment starts near the ask chunk.
    assert ds._window_for_question(long_q, n_chunks=200) == (132, 178)

    capped_q = dict(long_q)
    capped_q["answer_chunks"] = [190]
    # If the ask-to-answer tail exceeds max_chunks, keep the answer tail.
    assert ds._window_for_question(capped_q, n_chunks=200) == (135, 198)

    far_q = {
        "ask_chunk": 0,
        "answer_chunks": [190],
        "support_chunks": [180],
    }
    # Correctness wins over the max window: do not cut away the query ask
    # chunk, otherwise the segment would never receive the question.
    assert ds._window_for_question(far_q, n_chunks=200) == (0, 198)

    empty_q = {"ask_chunks": [], "answer_chunks": [], "support_chunks": []}
    assert ds._window_for_question(empty_q, n_chunks=0) == (0, 0)


def test_single_question_segment_uses_student_snapshot_or_rolls_from_zero():
    ds = _single_q_dataset_stub()
    q = {
        "question": "What color is the cup?",
        "gold_answer": "red",
        "answer_form": "short_exact",
        "ask_chunks": [40],
        "answer_chunks": [45],
        "support_chunks": [12],
    }
    row = {
        "index": "7",
        "video_id": "v1",
        "video_path": "v1.mp4",
        "n_chunks": 80,
        "extra_info": {
            "n_chunks": 80,
            "questions": [q],
            "gold_action_per_chunk": {
                "12": "silent",
                "30": "silent",
                "32": "compress",
                "40": "response",
            },
            "student_cache_meta": {
                "source": "pass2_student_rollout",
                "checkpoint": "ckpt-a",
                "global_step": 12,
            },
            "student_think_archive": [
                {"chunk": 12, "time": "24-26", "text": "saw a red cup"},
                {"chunk": 31, "time": "62-64", "text": "future after snapshot"},
            ],
            "student_state_by_chunk": {
                "30": {
                    "compressed_segments": [{
                        "time_range": [0, 20],
                        "text": "student summary",
                        "source_chunks": [0, 1],
                    }],
                    "recent_thinks": [{"chunk": 29, "time": "58-60", "text": "student think"}],
                }
            },
        },
        "reward_model": {},
    }

    materialized = ds._materialize_single_question_row(row, 0)
    extra = materialized["extra_info"]
    assert extra["segment_planned_start_chunk"] == 32
    # Nearest earlier student snapshot is used, so chunks 30..31 are simulated
    # before the planned question window. This preserves compression/memory state.
    assert extra["segment_start_chunk"] == 30
    assert extra["segment_end_chunk"] == 53
    assert extra["question_idx"] == 0
    assert extra["question_index"] == 0
    assert extra["initial_student_state_source"] == "student_state_by_chunk"
    assert extra["initial_student_state"]["compressed_segments"][0]["text"] == "student summary"
    assert [x["chunk"] for x in extra["initial_student_state"]["think_archive"]] == [12]
    assert extra["initial_student_state_checkpoint"] == "ckpt-a"
    assert extra["initial_student_state_global_step"] == 12
    assert extra["offline_compress_chunks"] == [32]
    assert "12" not in extra["gold_action_per_chunk"]
    assert "30" in extra["gold_action_per_chunk"]
    assert extra["gold_action_per_chunk"]["32"] == "silent"
    assert "40" in extra["gold_action_per_chunk"]

    bad_cache_row = dict(row)
    bad_cache_row["extra_info"] = dict(row["extra_info"])
    bad_cache_row["extra_info"].pop("student_think_archive", None)
    bad_cache_row["extra_info"]["student_state_by_chunk"] = {
        "32": {"recent_thinks": [{"chunk": 31, "text": "no archive"}]},
    }
    bad_materialized = ds._materialize_single_question_row(bad_cache_row, 0)
    bad_extra = bad_materialized["extra_info"]
    # Without a recall archive, correctness wins over speed: run prefix from 0
    # and let the live student trajectory trigger compression naturally.
    assert bad_extra["segment_start_chunk"] == 0
    assert bad_extra["segment_prefix_source"] == "missing_cache_rollout_from_zero"
    assert bad_extra["initial_student_state_missing"] is True


def test_silent_quality_v12_complements_outcome():
    """Verify silent_quality fills the reward gap that outcome alone misses.

    Scenario the audit identified: gold_action='silent', gold_answer='',
    model emits a hallucinated response. With ONLY outcome+timing rewards,
    this scores 0 (both masked). With silent_quality added, it scores -0.6.
    """
    from thinkstream.trainer.rewards import (
        compute_silent_quality as f_silent,
        compute_outcome_reward as f_outcome,
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
    test_answer_decision_v12()
    test_format_v12()
    test_silent_quality_v12()
    test_trajectory_outcome_v124_single_question()
    test_trajectory_outcome_v124_multi_question_mixed()
    test_trajectory_outcome_v124_empty_questions()
    test_trajectory_outcome_v124_multi_response_per_ask()
    test_trajectory_outcome_v124_multi_response_no_overlap()
    test_trajectory_outcome_v124_single_response_unchanged()
    test_per_chunk_silent_quality_v124()
    test_per_chunk_silent_quality_perfect_silence()
    test_recall_silent_requires_recall_check()
    test_recipe_reward_gates_positive_auxiliary_on_correct_answer()
    test_recipe_reward_keeps_negative_auxiliary_when_answer_wrong()
    test_recipe_reward_allows_auxiliary_when_answer_correct()
    test_recipe_reward_scales_auxiliary_on_partial_outcome()
    test_recipe_multi_q_reward_gates_each_question_independently()
    test_recipe_multi_q_aggregates_by_expected_answer_slot()
    test_recipe_multi_q_ignores_future_questions_past_rollout_horizon()
    test_recipe_action_shaping_scores_system_compress_only()
    test_rl_episode_mode_segment_alias()
    test_rl_dataset_recovers_legacy_offline_compress_boundaries()
    test_rl_compress_output_transfers_student_memory_state()
    test_single_question_window_boundaries_and_scalar_chunks()
    test_single_question_segment_uses_student_snapshot_or_rolls_from_zero()
    test_silent_quality_v12_complements_outcome()
    test_v12_advantage_aggregation()
    print("\n✅ all v12.0 reward smoke tests passed")
