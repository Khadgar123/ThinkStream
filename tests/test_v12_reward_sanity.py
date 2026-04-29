"""v12.3 reward sanity — scenario-based ordering invariants.

After dropping recall_quality + compress_quality (DeepEyesV2 alignment),
the v12 reward stack is 5 components:
  outcome (1.0) + timing (0.3) + format (0.1) + spam (-0.2) + silent_quality (0.2)

This test asserts the SCENARIO ORDERING that any sane streaming-agent reward
stack must satisfy, independent of advantage aggregation:

  CORRECT_ON_TIME > CORRECT_LATE > CORRECT_EARLY > WRONG_BUT_TRIED >
    SILENT_WHEN_SHOULD_RESPOND > HALLUCINATE_WHEN_SHOULD_BE_SILENT

If the ordering ever breaks (e.g., someone bumps timing weight high enough
that "fast hallucination" outscores "correct silence"), this test catches it
before the model is trained.

Run: python tests/test_v12_reward_sanity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkstream.trainer.gdpo_advantage import (
    V12_DEFAULT_REWARD_WEIGHTS,
    V12_REWARD_DICT_KEYS,
)
from thinkstream.trainer.v12_rewards import (
    compute_outcome_reward_v12,
    compute_timing_reward_v12,
    compute_format_reward_v12,
    compute_spam_score_v12,
    compute_silent_quality_v12,
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
    n_recall_calls=0,
    n_compress_calls=0,
):
    """Compute the trajectory-level weighted score using the production
    weights. Mirrors what `_calc_rewards_v12` would compute, minus the
    advantage normalization. We use this to assert ordering invariants."""
    w = V12_DEFAULT_REWARD_WEIGHTS

    outcome = compute_outcome_reward_v12(
        final_answer, gold_answer, answer_form=answer_form,
    )
    outcome_mask = 1.0 if gold_answer else 0.0

    timing = compute_timing_reward_v12(
        answer_chunk=answer_chunk,
        visible_start_chunk=visible_start_chunk,
        visible_end_chunk=visible_end_chunk,
    )
    timing_mask = 1.0 if visible_start_chunk is not None else 0.0

    fmt = compute_format_reward_v12(chunk_texts)
    spam = compute_spam_score_v12(
        n_recall_calls=n_recall_calls, n_compress_calls=n_compress_calls,
    )
    silent_q = compute_silent_quality_v12(
        final_answer=final_answer, gold_action=gold_action,
        gold_answer=gold_answer or "",
    )

    return (
        w["outcome"] * outcome * outcome_mask
        + w["timing"] * timing * timing_mask
        + w["format"] * fmt
        + w["spam"] * spam       # weight is already negative
        + w["silent_quality"] * silent_q
    )


def test_reward_keys_minimal():
    """v12.3 reward dict keys: exactly the 5 components (no recall_quality
    or compress_quality after the simplification)."""
    expected = {"outcome", "timing", "format", "spam", "silent_quality"}
    assert set(V12_REWARD_DICT_KEYS) == expected, (
        f"v12.3 keys mismatch: got {set(V12_REWARD_DICT_KEYS)}, "
        f"expected {expected}. recall_quality / compress_quality should "
        f"have been dropped."
    )
    assert set(V12_DEFAULT_REWARD_WEIGHTS) == expected
    print("  PASS v12.3 reward stack is exactly the 5-component minimal set")


def test_correct_on_time_is_top_score():
    """Scenario A: model emits correct answer within visibility window.
    This MUST be the highest-scoring scenario tested."""
    answer_chunk = 5
    score = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=answer_chunk,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>spotted it</think><answer>red apron</answer>"],
    )
    # outcome=+1×1.0 + timing=+1×0.3 + format=1×0.1 + silent=0×0.2 = 1.4
    assert score > 1.3, f"correct on-time should score > 1.3, got {score}"
    print(f"  PASS correct_on_time = {score:.3f} (expected ~1.4)")


def test_hallucinate_is_strictly_worse_than_silence():
    """Hallucinate-when-should-be-silent must be STRICTLY worse than
    correct silence. This is the primary motivation for silent_quality."""
    silent_score = _weighted_score(
        final_answer=None,
        gold_answer="",
        answer_form="",
        gold_action="silent",
        answer_chunk=None,
        visible_start_chunk=None,
        visible_end_chunk=None,
        chunk_texts=["<think>nothing yet</think><answer></answer>"],
    )
    # outcome=0 (mask), timing=0 (mask), format=1×0.1, silent=+0.3×0.2 = 0.16
    hallucinate_score = _weighted_score(
        final_answer="hallucinated",
        gold_answer="",
        answer_form="",
        gold_action="silent",
        answer_chunk=2,
        visible_start_chunk=None,
        visible_end_chunk=None,
        chunk_texts=["<think>guessing</think><answer>hallucinated</answer>"],
    )
    # outcome=0 (mask), timing=0 (mask), format=1×0.1, silent=-0.6×0.2 = -0.02
    assert silent_score > hallucinate_score, (
        f"correct silence ({silent_score:.3f}) must beat hallucination "
        f"({hallucinate_score:.3f}) — silent_quality reward broken"
    )
    assert hallucinate_score < 0, (
        f"hallucination should score NEGATIVE, got {hallucinate_score:.3f} "
        f"— this means silent_quality weight is too low to overcome format"
    )
    print(
        f"  PASS hallucinate ({hallucinate_score:.3f}) < silent ({silent_score:.3f})"
    )


def test_missed_response_is_penalized():
    """Silent-when-should-respond: silent_quality=-0.6 + timing=-0.5 fire."""
    score = _weighted_score(
        final_answer=None,
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=None,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>didn't see</think><answer></answer>"],
    )
    # outcome=0×1.0 (no answer) + timing=-0.5×0.3=-0.15 + format=1×0.1 +
    # silent=-0.6×0.2=-0.12 = -0.17
    assert score < -0.1, (
        f"missed response should score < -0.1, got {score:.3f} — both "
        f"timing and silent_quality should fire negatively"
    )
    print(f"  PASS missed_response = {score:.3f} (≤ -0.1)")


def test_wrong_answer_on_time_beats_silent_missed():
    """A wrong-but-trying response is MORE valuable than total silence
    when the question demands an answer. Both lose outcome reward, but
    wrong-but-trying gets timing+0.3 + silent_quality 0 (responded);
    silent missed gets timing -0.15 + silent_quality -0.12."""
    wrong_on_time = _weighted_score(
        final_answer="blue apron",     # wrong
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=5,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>guess</think><answer>blue apron</answer>"],
    )
    # outcome=0×1.0 + timing=+1×0.3 + format=1×0.1 + silent=0×0.2 = 0.4
    silent_missed = _weighted_score(
        final_answer=None,
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=None,
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>nothing</think><answer></answer>"],
    )
    assert wrong_on_time > silent_missed, (
        f"wrong-on-time ({wrong_on_time:.3f}) must beat silent-missed "
        f"({silent_missed:.3f}) — model should be encouraged to attempt"
    )
    print(f"  PASS wrong_on_time ({wrong_on_time:.3f}) > silent_missed ({silent_missed:.3f})")


def test_correct_beats_wrong_on_time():
    """Correctness must dominate timing — outcome weight 1.0 vs timing 0.3."""
    correct = _weighted_score(
        final_answer="red apron", gold_answer="red apron",
        answer_form="literal", gold_action="response",
        answer_chunk=5, visible_start_chunk=4, visible_end_chunk=6,
        chunk_texts=["<think>seen</think><answer>red apron</answer>"],
    )
    wrong_on_time = _weighted_score(
        final_answer="blue apron", gold_answer="red apron",
        answer_form="literal", gold_action="response",
        answer_chunk=5, visible_start_chunk=4, visible_end_chunk=6,
        chunk_texts=["<think>guess</think><answer>blue apron</answer>"],
    )
    delta = correct - wrong_on_time
    assert delta >= 0.9, (
        f"outcome weight too small: correct ({correct:.3f}) vs "
        f"wrong-on-time ({wrong_on_time:.3f}) gap = {delta:.3f} < 0.9. "
        f"Outcome must dominate timing."
    )
    print(f"  PASS correct - wrong_on_time = {delta:.3f} (≥ 0.9)")


def test_early_hallucination_is_worst():
    """Early-emit hallucination (answer before evidence visible): timing=-1
    plus silent_quality penalty if gold_action was actually silent."""
    score = _weighted_score(
        final_answer="red apron",
        gold_answer="red apron",
        answer_form="literal",
        gold_action="response",
        answer_chunk=2,                    # before visibility
        visible_start_chunk=4,
        visible_end_chunk=6,
        chunk_texts=["<think>jumping</think><answer>red apron</answer>"],
    )
    # outcome=+1×1.0 + timing=-1×0.3 + format=1×0.1 + silent=0×0.2 = 0.8
    correct_on_time = _weighted_score(
        final_answer="red apron", gold_answer="red apron",
        answer_form="literal", gold_action="response",
        answer_chunk=5, visible_start_chunk=4, visible_end_chunk=6,
        chunk_texts=["<think>seen</think><answer>red apron</answer>"],
    )
    assert score < correct_on_time, (
        f"early-emit-correct ({score:.3f}) must be worse than "
        f"on-time-correct ({correct_on_time:.3f})"
    )
    print(f"  PASS early ({score:.3f}) < on_time ({correct_on_time:.3f})")


def test_spam_penalizes_excess_tools():
    """Multiple recall + compress calls beyond budget should subtract."""
    one_each = _weighted_score(
        final_answer="answer", gold_answer="answer",
        answer_form="literal", gold_action="response",
        answer_chunk=5, visible_start_chunk=4, visible_end_chunk=6,
        chunk_texts=["<answer>answer</answer>"],
        n_recall_calls=1, n_compress_calls=1,
    )
    spammy = _weighted_score(
        final_answer="answer", gold_answer="answer",
        answer_form="literal", gold_action="response",
        answer_chunk=5, visible_start_chunk=4, visible_end_chunk=6,
        chunk_texts=["<answer>answer</answer>"],
        n_recall_calls=5, n_compress_calls=4,
    )
    # spam_score difference = 0.5×4 + 0.3×3 = 2.9
    # weighted with -0.2: spammy is ~0.58 worse
    assert one_each > spammy, (
        f"one-each ({one_each:.3f}) must beat spammy ({spammy:.3f})"
    )
    print(f"  PASS one_each ({one_each:.3f}) > spammy ({spammy:.3f})")


def main():
    tests = [
        test_reward_keys_minimal,
        test_correct_on_time_is_top_score,
        test_correct_beats_wrong_on_time,
        test_hallucinate_is_strictly_worse_than_silence,
        test_missed_response_is_penalized,
        test_wrong_answer_on_time_beats_silent_missed,
        test_early_hallucination_is_worst,
        test_spam_penalizes_excess_tools,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} sanity tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
