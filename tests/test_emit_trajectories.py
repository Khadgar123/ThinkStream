"""Sanity test for emit_trajectories.py.

Verifies that:
  - Each trajectory row preserves all samples from a single (video, traj_id)
  - Samples are ordered by chunk_idx
  - Metadata is lifted from the first sample
  - Stats counts are consistent
  - No trajectory is empty
  - Splits are disjoint (no video appears in two splits)

Read-only test: operates on already-emitted trajectory files; does NOT
re-run emit_trajectories. Skipped if files don't exist (in CI).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_FINAL = Path(__file__).resolve().parents[1] / "data" / "agent_v5" / "final"
TRAJ_FILES = [
    "train_sft_trajectories.jsonl",
    "train_rl_trajectories.jsonl",
    "val_trajectories.jsonl",
    "test_trajectories.jsonl",
]


def _load_trajectories(fname):
    fp = DATA_FINAL / fname
    if not fp.exists():
        return None
    with fp.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def test_chunk_idx_ordered():
    """Every trajectory's samples must be sorted by chunk_idx ascending."""
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            print(f"  SKIP {fname} (file missing)")
            continue
        for row in rows:
            chunks = [int(s["chunk_idx"]) for s in row["samples"]]
            assert chunks == sorted(chunks), (
                f"{fname} traj={row['trajectory_id']} chunks not ordered: {chunks[:10]}"
            )
        print(f"  PASS {fname}: chunk_idx order ({len(rows)} trajectories)")


def test_no_empty_trajectory():
    """Each trajectory row must contain at least one sample."""
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        for row in rows:
            n = len(row["samples"])
            assert n > 0, f"{fname} traj={row['trajectory_id']} is empty"
            assert row["stats"]["n_samples"] == n, (
                f"{fname} traj={row['trajectory_id']} stats.n_samples={row['stats']['n_samples']} "
                f"≠ len(samples)={n}"
            )
        print(f"  PASS {fname}: no empty trajectories")


def test_metadata_consistency_v124():
    """Each sample's metadata must match SOME question in the trajectory's
    questions list (v12.4: multi-card trajectories — different chunks may
    own different cards, so per-sample metadata varies; but every real
    card sample must correspond to one of the questions)."""
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        unmatched = 0
        for row in rows:
            qs_by_card = {q["card_id"]: q for q in row.get("questions", [])}
            for s in row["samples"]:
                cid = s.get("card_id", "")
                s_meta = s.get("metadata") or {}
                # Skip base/silent samples without a gold_answer
                if not s_meta.get("gold_answer") or not cid:
                    continue
                q = qs_by_card.get(cid)
                if q is None:
                    unmatched += 1
                    continue
                # Sample.metadata must agree with its question
                if s_meta.get("family") and q.get("family") and (
                    s_meta["family"] != q["family"]
                ):
                    unmatched += 1
        assert unmatched == 0, (
            f"{fname}: {unmatched} samples have card_id not in trajectory.questions "
            f"or family mismatch with their card"
        )
        print(f"  PASS {fname}: per-sample metadata consistent with questions list "
              f"({len(rows)} trajectories)")


def test_splits_disjoint_at_video_level():
    """No video_id appears in more than one split."""
    seen = {}  # video_id -> split_name
    overlaps = []
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        for row in rows:
            vid = row["video_id"]
            if vid in seen and seen[vid] != fname:
                overlaps.append((vid, seen[vid], fname))
            else:
                seen[vid] = fname
    assert not overlaps, f"video overlap across splits: {overlaps[:3]}"
    print(f"  PASS splits disjoint at video level ({len(seen)} unique videos)")


def test_protocol_version_consistent():
    """All samples within one trajectory must share the same protocol_version."""
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        bad = 0
        for row in rows:
            traj_proto = row.get("protocol_version", "v12")
            for s in row["samples"]:
                if s.get("protocol_version", "v12") != traj_proto:
                    bad += 1
        assert bad == 0, f"{fname}: {bad} samples with mismatched protocol_version"
        print(f"  PASS {fname}: protocol_version consistent")


def test_sample_count_matches_pipeline_stats():
    """Total trajectory-grouped samples should match pipeline_stats.passed."""
    stats_path = DATA_FINAL / "pipeline_stats.json"
    if not stats_path.exists():
        print("  SKIP pipeline_stats.json missing")
        return
    with stats_path.open() as f:
        stats = json.load(f)
    target = stats.get("passed")  # 47,289 in current run
    if not target:
        return
    total = 0
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            print(f"  SKIP {fname} missing — cannot verify total")
            return
        total += sum(r["stats"]["n_samples"] for r in rows)
    # Allow ±1% tolerance — emit may skip videos with no verified file.
    delta = abs(total - target)
    assert delta < target * 0.02, (
        f"trajectory total {total} differs from pass4-passed {target} by {delta} (>2%)"
    )
    print(f"  PASS total samples = {total} (pipeline_stats.passed = {target})")


def test_questions_field_present_v124():
    """v12.4 — every trajectory must have a `questions` list (may be empty
    for base-only trajectories) and `gold_action_per_chunk` dict."""
    seen_multi_q = 0
    seen_per_chunk_map = 0
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        for row in rows:
            assert "questions" in row, (
                f"{fname} traj={row.get('trajectory_id')}: missing v12.4 'questions' field"
            )
            assert "gold_action_per_chunk" in row, (
                f"{fname} traj={row.get('trajectory_id')}: missing v12.4 'gold_action_per_chunk'"
            )
            # Each question must have card_id + gold_answer + ask_chunks
            for q in row["questions"]:
                assert "card_id" in q
                assert "gold_answer" in q
                assert "ask_chunks" in q
                assert isinstance(q["ask_chunks"], list)
            if len(row["questions"]) >= 2:
                seen_multi_q += 1
            if row["gold_action_per_chunk"]:
                seen_per_chunk_map += 1
    assert seen_multi_q > 0, (
        "expected at least some trajectories with ≥2 questions"
    )
    assert seen_per_chunk_map > 0, (
        "expected gold_action_per_chunk to be populated"
    )
    print(f"  PASS v12.4 questions+gold_action_per_chunk "
          f"(multi_q={seen_multi_q}, with_per_chunk_map={seen_per_chunk_map})")


def test_gold_action_consistent_with_samples():
    """gold_action_per_chunk must reflect the highest-priority sample_type
    on each chunk (response > recall_response > recall_query > compress > silent)."""
    priority = {
        "response": 0, "recall_response": 1, "recall_query": 2,
        "compress": 3, "recall_silent": 4, "silent": 5,
    }
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        for row in rows:
            map_ = row.get("gold_action_per_chunk") or {}
            # Build expected map from samples
            expected = {}
            for s in row["samples"]:
                ci = str(s.get("chunk_idx", 0))
                st = s.get("sample_type", "silent")
                if ci not in expected or priority.get(st, 99) < priority.get(expected[ci], 99):
                    expected[ci] = st
            for ci, st in expected.items():
                assert map_.get(ci) == st, (
                    f"{fname} traj={row['trajectory_id']} chunk={ci}: "
                    f"map says {map_.get(ci)} but samples imply {st}"
                )
        print(f"  PASS {fname}: gold_action_per_chunk matches sample priorities")


def main():
    tests = [
        test_chunk_idx_ordered,
        test_no_empty_trajectory,
        test_metadata_consistency_v124,
        test_splits_disjoint_at_video_level,
        test_protocol_version_consistent,
        test_sample_count_matches_pipeline_stats,
        test_questions_field_present_v124,
        test_gold_action_consistent_with_samples,
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
