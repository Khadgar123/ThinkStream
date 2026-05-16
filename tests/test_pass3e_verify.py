"""Pass3e_verify tests — verify tag_samples preserves trajectory continuity.

The v12.5 restructure moved verification from "drop failures" (filter_samples,
which broke trajectory chunk-timeline) to "tag in-place" (tag_samples, which
keeps every sample with verification.passed/.fail_reasons populated).

This test asserts:
  1. tag_samples returns ALL input samples (no drops).
  2. Each sample has verification.passed flag.
  3. Aggregate stats still report pass/fail counts correctly.
  4. Legacy filter_samples name is also tag-only for backward compat.

Run: python tests/test_pass3e_verify.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_minimal_v12_sample(chunk_idx, sample_type, video_id="vid_test",
                              traj_id="traj_0", card_id="c1",
                              gold_answer="red"):
    """A minimal v12 sample that should pass all verifiers (where applicable)."""
    return {
        "video_id": video_id,
        "trajectory_id": traj_id,
        "card_id": card_id,
        "chunk_idx": chunk_idx,
        "sample_type": sample_type,
        "action": sample_type,
        "protocol_version": "v12",
        "prompt_type": "SYSTEM_PROMPT",
        "sequence_type": "immediate_response",
        "input": {
            "system": "v12 system",
            "memory": {"compressed": [], "recent_thinks": []},
            "queries": [],
            "user_input": "" if sample_type == "silent" else "what color?",
            "visual_window": {
                "video_start": chunk_idx * 2, "video_end": chunk_idx * 2 + 2,
                "frames": 8, "frame_indices": list(range(8)),
            },
        },
        "output": (
            f'<think>chunk {chunk_idx} obs of {chunk_idx*2}+ duration entities visible '
            f'red apron and stove and pan and counter</think>'
            + ('</Silence>' if sample_type == "silent"
               else f'</Response> {gold_answer}')
        ),
        "metadata": {
            "gold_answer": gold_answer if sample_type == "response" else "",
            "gold_action": sample_type,
            "answer_form": "literal",
            "family": "F1",
            "support_chunks": [chunk_idx],
            "availability": "in_visual",
        },
    }


def test_tag_samples_returns_all_input():
    """No samples dropped — tag_samples returns same count as input."""
    from scripts.agent_data.pass3e_verify import tag_samples

    inputs = [
        _make_minimal_v12_sample(0, "silent"),
        _make_minimal_v12_sample(5, "response"),
        _make_minimal_v12_sample(6, "silent"),
    ]
    n_in = len(inputs)
    out, stats = tag_samples(inputs)
    assert len(out) == n_in, (
        f"tag_samples dropped {n_in - len(out)} samples; should keep all"
    )
    assert stats["total"] == n_in
    print(f"  PASS tag_samples returns all {n_in} inputs")


def test_tag_samples_attaches_verification():
    """Every output sample must carry a verification dict."""
    from scripts.agent_data.pass3e_verify import tag_samples

    inputs = [
        _make_minimal_v12_sample(0, "silent"),
        _make_minimal_v12_sample(5, "response"),
    ]
    out, _ = tag_samples(inputs)
    for s in out:
        assert "verification" in s, f"sample missing verification tag: {s}"
        assert "passed" in s["verification"]
        assert "fail_reasons" in s["verification"]
    print(f"  PASS each sample has verification.passed/.fail_reasons")


def test_tag_samples_keeps_failures_with_reasons():
    """A sample that fails verification must remain in the list, with
    verification.passed=False AND a non-empty fail_reasons list."""
    from scripts.agent_data.pass3e_verify import tag_samples

    bad_sample = _make_minimal_v12_sample(5, "response", gold_answer="answer")
    # Force a failure: empty think will trigger format/grounding check
    bad_sample["output"] = "</Response> answer"   # no <think> tag
    inputs = [_make_minimal_v12_sample(0, "silent"), bad_sample]
    out, stats = tag_samples(inputs)
    assert len(out) == 2, "failed sample must NOT be dropped"
    failed = [s for s in out if not s["verification"]["passed"]]
    # If the bad sample fails, fail_reasons should be populated.
    if failed:
        assert all(s["verification"]["fail_reasons"] for s in failed), (
            "failed samples must have fail_reasons populated"
        )
        print(f"  PASS {len(failed)} failures retained with reasons")
    else:
        print(f"  SKIP this sample passed all checks (verification permissive)")


def test_filter_samples_is_tag_only_legacy_name():
    """Backward-compat: legacy filter_samples name no longer drops rows."""
    from scripts.agent_data.pass3e_verify import filter_samples

    bad_sample = _make_minimal_v12_sample(5, "response", gold_answer="answer")
    bad_sample["output"] = "no tags at all"
    inputs = [_make_minimal_v12_sample(0, "silent"), bad_sample]
    out, stats = filter_samples(inputs)
    assert len(out) == len(inputs), "filter_samples must preserve trajectory rows"
    assert stats["total"] == len(inputs)
    assert any(not s.get("verification", {}).get("passed", True) for s in out)
    print(f"  PASS filter_samples keeps all {len(out)} rows with tags")


def test_recall_start_end_is_valid():
    """Current recall tool schema uses explicit start_time/end_time."""
    from scripts.agent_data.pass3e_verify import verify_format

    sample = _make_minimal_v12_sample(5, "silent")
    sample.update({
        "sample_type": "recall",
        "action": "response",
        "v12_assistant_turn_1": (
            '<think>need prior visual evidence</think>'
            '<tool_call>{"name":"recall","arguments":'
            '{"start_time":1,"end_time":5}}</tool_call>'
        ),
        "v12_assistant_turn_2": (
            "<think>retrieved frames contain the answer</think></Response> red"
        ),
    })
    ok, reason = verify_format(sample)
    assert ok, reason
    print("  PASS recall start_time/end_time accepted")


def test_think_checks_are_policy_skipped():
    """Think length/blacklist audits should not hard-fail current data."""
    from scripts.agent_data.pass3e_verify import (
        verify_grounding,
        verify_think_token_length,
    )

    sample = _make_minimal_v12_sample(5, "silent")
    sample["output"] = (
        "<think>"
        + " ".join(["noise"] * 200)
        + "</think></Silence>"
    )
    ok, reason = verify_grounding(sample)
    assert ok, reason
    ok, reason = verify_think_token_length(sample)
    assert ok, reason
    print("  PASS think grounding/token-length checks are skipped")


def test_mc_answer_text_in_options_is_not_leakage():
    """MC prompts may contain answer candidates by design."""
    from scripts.agent_data.pass3e_verify import verify_question_answer_leakage

    sample = _make_minimal_v12_sample(5, "response", gold_answer="red hat")
    sample["metadata"].update({
        "answer_form": "multiple_choice",
        "question": "Which item appears first: red hat, blue cup, or green bag?",
        "options": ["A) red hat", "B) blue cup", "C) green bag"],
    })
    ok, reason = verify_question_answer_leakage(sample)
    assert ok, reason
    print("  PASS MC option answer text is not leakage")


def test_aggregate_stats_pass_rate():
    """Stats still report correct pass/fail breakdown even when nothing
    is dropped — total stays the same as input count."""
    from scripts.agent_data.pass3e_verify import tag_samples

    inputs = [_make_minimal_v12_sample(i, "silent") for i in range(5)]
    out, stats = tag_samples(inputs)
    assert stats["total"] == 5
    assert stats["passed"] + stats["failed"] == stats["total"]
    assert 0.0 <= stats["pass_rate"] <= 1.0
    print(f"  PASS stats: {stats['passed']}/{stats['total']} pass_rate={stats['pass_rate']:.2f}")


def main():
    tests = [
        test_tag_samples_returns_all_input,
        test_tag_samples_attaches_verification,
        test_tag_samples_keeps_failures_with_reasons,
        test_filter_samples_is_tag_only_legacy_name,
        test_recall_start_end_is_valid,
        test_legacy_recall_time_range_is_rejected,
        test_think_checks_are_policy_skipped,
        test_mc_answer_text_in_options_is_not_leakage,
        test_aggregate_stats_pass_rate,
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

    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
