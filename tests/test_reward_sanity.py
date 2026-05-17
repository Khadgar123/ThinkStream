"""Reward sanity checks for the production streaming-RL reward stack.

The initial RL objective is intentionally small:

  outcome + answer_decision + format

Positive auxiliary rewards are gated by answer correctness in the scorer, so
wrong answers cannot earn reward just by being timely or well-formatted. This
file checks ordering invariants for that exact scalar objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkstream.trainer.gdpo_advantage import (
    V12_DEFAULT_REWARD_WEIGHTS,
    V12_REWARD_DICT_KEYS,
)
from thinkstream.trainer.rewards import (
    compute_answer_decision_reward,
    compute_format_reward,
    compute_outcome_reward,
)


def _weighted_score(
    *,
    final_answer,
    gold_answer,
    answer_form,
    gold_action,
    answer_chunk,
    visible_start_chunk,
    visible_end_chunk,
    chunk_texts,
):
    """Mirror thinkstream.rl.thinkstream._combine_reward_parts."""
    del gold_action
    w = V12_DEFAULT_REWARD_WEIGHTS

    if (
        final_answer
        and visible_start_chunk is not None
        and answer_chunk is not None
        and answer_chunk < visible_start_chunk
    ):
        outcome = 0.0
    else:
        outcome = compute_outcome_reward(
            final_answer,
            gold_answer,
            answer_form=answer_form,
        )

    answer_decision = compute_answer_decision_reward(
        answer_chunk=answer_chunk,
        visible_start_chunk=visible_start_chunk,
        visible_end_chunk=visible_end_chunk,
        has_answer=bool(final_answer),
    )
    fmt = compute_format_reward(chunk_texts)

    gate = max(0.0, min(1.0, float(outcome)))
    score = w["outcome"] * outcome
    for key, value in (
        ("answer_decision", answer_decision),
        ("format", fmt),
    ):
        weighted = w[key] * value
        score += gate * weighted if weighted > 0 else weighted
    return score


def test_reward_keys_minimal():
    expected = {"outcome", "answer_decision", "format"}
    assert set(V12_REWARD_DICT_KEYS) == expected
    assert set(V12_DEFAULT_REWARD_WEIGHTS) == expected


def test_correct_on_time_is_top_score():
    score = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>spotted it</think></Response> red apron"],
    )
    assert score > 1.3


def test_hallucinate_is_strictly_worse_than_silence():
    silent_score = _weighted_score(
        final_answer=None,
        gold_answer="",
        answer_form="",
        gold_action="silent",
        answer_chunk=None,
        visible_start_chunk=None,
        visible_end_chunk=None,
        chunk_texts=["<think>nothing yet</think></Silence>"],
    )
    hallucinate_score = _weighted_score(
        final_answer="hallucinated",
        gold_answer="",
        answer_form="",
        gold_action="silent",
        answer_chunk=2,
        visible_start_chunk=None,
        visible_end_chunk=None,
        chunk_texts=["<think>guessing</think></Response> hallucinated"],
    )
    assert silent_score == 0.0
    assert hallucinate_score < silent_score
    assert hallucinate_score < 0


def test_missed_response_is_penalized():
    score = _weighted_score(
        final_answer=None,
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=None,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>didn't see</think></Silence>"],
    )
    assert score <= -0.3


def test_wrong_answer_on_time_beats_silent_missed():
    wrong_on_time = _weighted_score(
        final_answer="blue apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>guess</think></Response> blue apron"],
    )
    silent_missed = _weighted_score(
        final_answer=None,
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=None,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>nothing</think></Silence>"],
    )
    assert wrong_on_time == 0.0
    assert wrong_on_time > silent_missed


def test_correct_beats_wrong_on_time():
    correct = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>seen</think></Response> red apron"],
    )
    wrong_on_time = _weighted_score(
        final_answer="blue apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>guess</think></Response> blue apron"],
    )
    assert correct - wrong_on_time >= 1.0


def test_early_correct_text_is_not_credited():
    early = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=2,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>jumping</think></Response> red apron"],
    )
    correct_on_time = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>seen</think></Response> red apron"],
    )
    assert early < correct_on_time
    assert early <= -0.3


def test_format_requires_think_block():
    assert compute_format_reward(["</Response> answer"]) == 0.0
    assert compute_format_reward([
        "<think>seen</think></Response> answer"
    ]) == 1.0


def main():
    tests = [
        test_reward_keys_minimal,
        test_correct_on_time_is_top_score,
        test_correct_beats_wrong_on_time,
        test_hallucinate_is_strictly_worse_than_silence,
        test_missed_response_is_penalized,
        test_wrong_answer_on_time_beats_silent_missed,
        test_early_correct_text_is_not_credited,
        test_format_requires_think_block,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} sanity tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
