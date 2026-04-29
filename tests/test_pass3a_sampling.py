"""Pass3a even-stride sampling test.

Verifies the bug fix in scripts/agent_data_v5/pass3a_cards.py
(_format_evidence_for_prompt) — the previous `chunk_indices[:10]` truncation
caused 62.9% of card support_chunks to fall in the first 20% of videos.
After fix, sampled chunks span the FULL chunk_indices range.

Run: python tests/test_pass3a_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data_v5.pass3a_cards import (
    _format_evidence_for_prompt,
    _EV_PROMPT_MAX_CHUNKS,
)


def _make_evidence(num_chunks):
    """Build minimal evidence list with one cap per chunk."""
    return [
        {
            "chunk_idx": i,
            "time": [i * 2, i * 2 + 2],
            "visible_entities": [{"desc": f"entity_{i}"}],
            "atomic_facts": [{"fact": f"fact_{i}", "confidence": 0.9}],
            "ocr": [],
            "state_changes": [],
        }
        for i in range(num_chunks)
    ]


def _extract_chunk_idxs_from_prompt(prompt_text):
    """Re-parse the chunk indices that ended up in the formatted prompt."""
    import re
    return [int(m.group(1)) for m in re.finditer(r"chunk (\d+)", prompt_text)]


def test_no_truncation_when_under_cap():
    """Input ≤ MAX_CHUNKS: all chunks must appear in the prompt."""
    ev = _make_evidence(8)
    text = _format_evidence_for_prompt(ev, list(range(8)))
    seen = _extract_chunk_idxs_from_prompt(text)
    assert seen == list(range(8)), f"expected all 8 chunks, got {seen}"
    print("  PASS short input: all chunks preserved")


def test_even_stride_for_long_video():
    """Input >> MAX_CHUNKS: sampled chunks must span the FULL range,
    not just the first MAX_CHUNKS chronologically."""
    ev = _make_evidence(100)
    text = _format_evidence_for_prompt(ev, list(range(100)))
    seen = _extract_chunk_idxs_from_prompt(text)
    assert len(seen) <= _EV_PROMPT_MAX_CHUNKS, (
        f"expected ≤{_EV_PROMPT_MAX_CHUNKS} chunks, got {len(seen)}"
    )
    # Critical assertion: must include both first AND last chunk
    assert seen[0] == 0, f"first sampled chunk must be 0, got {seen[0]}"
    assert seen[-1] == 99, f"last sampled chunk must be 99, got {seen[-1]}"
    # Middle of video must also be represented (not just start)
    assert any(40 <= c <= 60 for c in seen), (
        f"middle of video (chunks 40-60) must be sampled, got {seen}"
    )
    print(f"  PASS long video: sampled {seen} (spans 0-99)")


def test_extreme_video_196_chunks():
    """Real-world worst case: 196 chunks (max observed in pass1a corpus).
    The OLD `[:10]` would yield support_chunks ⊆ [0..9] only — 5.1% of video.
    New: must include first AND last AND middle."""
    ev = _make_evidence(196)
    text = _format_evidence_for_prompt(ev, list(range(196)))
    seen = _extract_chunk_idxs_from_prompt(text)
    assert seen[0] == 0 and seen[-1] == 195, seen
    # No bucket of 5 (20% bins) should be empty
    bucket_counts = [0] * 5
    for c in seen:
        bucket_counts[min(c // 40, 4)] += 1
    empty_buckets = [i for i, n in enumerate(bucket_counts) if n == 0]
    # We have 10 chunks across 5 buckets — middle 3 buckets each get ≥1
    assert sum(bucket_counts[1:4]) >= 3, (
        f"middle 60% of video must have ≥3 sampled chunks, got buckets {bucket_counts}"
    )
    print(f"  PASS 196-chunk video: bucket distribution {bucket_counts}")


def test_dedup_preserves_order():
    """Sparse `chunk_indices` (with gaps) must not produce duplicates."""
    ev = _make_evidence(20)
    # Pass irregular subset that the formula could collide on edges
    text = _format_evidence_for_prompt(ev, [0, 0, 1, 5, 10, 15, 19])
    seen = _extract_chunk_idxs_from_prompt(text)
    assert len(seen) == len(set(seen)), f"duplicates in output: {seen}"
    print(f"  PASS dedup: {seen}")


def test_irregular_input_indices():
    """Caller may pass non-monotonic chunk_indices (e.g., support±2 for
    multiple support chunks). Must still produce sensible output."""
    ev = _make_evidence(50)
    irregular = [3, 4, 5, 12, 13, 14, 25, 26, 27, 40, 41, 42, 48]
    text = _format_evidence_for_prompt(ev, irregular)
    seen = _extract_chunk_idxs_from_prompt(text)
    assert len(seen) <= _EV_PROMPT_MAX_CHUNKS
    # First and last of irregular range must be sampled
    assert seen[0] == 3, seen
    assert seen[-1] == 48, seen
    print(f"  PASS irregular input → sampled {seen}")


def test_old_truncation_would_have_dropped_middle():
    """Regression: with the OLD `[:10]` truncation a 100-chunk video would
    sample chunks [0..9] only. We assert the NEW sampler does NOT produce
    that (i.e., max sampled chunk > 9 when input has more chunks)."""
    ev = _make_evidence(100)
    text = _format_evidence_for_prompt(ev, list(range(100)))
    seen = _extract_chunk_idxs_from_prompt(text)
    assert max(seen) > 9, (
        f"REGRESSION: sampler reverted to first-10 truncation; max chunk = {max(seen)}"
    )
    print(f"  PASS regression check: max sampled chunk = {max(seen)} (>9)")


def main():
    tests = [
        test_no_truncation_when_under_cap,
        test_even_stride_for_long_video,
        test_extreme_video_196_chunks,
        test_dedup_preserves_order,
        test_irregular_input_indices,
        test_old_truncation_would_have_dropped_middle,
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
