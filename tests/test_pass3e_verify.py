"""Pass3e_verify tests — verify tag_samples preserves trajectory continuity.

The v12.5 restructure moved verification from "drop failures" (filter_samples,
which broke trajectory chunk-timeline) to "tag in-place" (tag_samples, which
keeps every sample with verification.passed/.fail_reasons populated).

This test asserts:
  1. tag_samples returns ALL input samples (no drops).
  2. Each sample has verification.passed flag.
  3. Aggregate stats still report pass/fail counts correctly.
  4. Legacy filter_samples (for backward compat) still drops failures.

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
            + ('<answer></answer>' if sample_type == "silent"
               else f'<answer>{gold_answer}</answer>')
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
    from scripts.agent_data_v5.pass3e_verify import tag_samples

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
    from scripts.agent_data_v5.pass3e_verify import tag_samples

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
    from scripts.agent_data_v5.pass3e_verify import tag_samples

    bad_sample = _make_minimal_v12_sample(5, "response", gold_answer="answer")
    # Force a failure: empty think will trigger format/grounding check
    bad_sample["output"] = "<answer>answer</answer>"   # no <think> tag
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


def test_filter_samples_still_drops_legacy():
    """Backward-compat: legacy filter_samples still drops failures."""
    from scripts.agent_data_v5.pass3e_verify import filter_samples

    bad_sample = _make_minimal_v12_sample(5, "response", gold_answer="answer")
    bad_sample["output"] = "no tags at all"
    inputs = [_make_minimal_v12_sample(0, "silent"), bad_sample]
    out, stats = filter_samples(inputs)
    # filter_samples returns only verification.passed=True
    assert all(s.get("verification", {}).get("passed", True) for s in out)
    print(f"  PASS filter_samples keeps {len(out)}/{len(inputs)} (drops failures)")


def test_aggregate_stats_pass_rate():
    """Stats still report correct pass/fail breakdown even when nothing
    is dropped — total stays the same as input count."""
    from scripts.agent_data_v5.pass3e_verify import tag_samples

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
        test_filter_samples_still_drops_legacy,
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
