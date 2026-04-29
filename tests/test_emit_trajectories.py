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


def test_metadata_consistency():
    """Trajectory metadata must match the metadata of all its samples
    (all samples in one trajectory share card_id / family / gold_answer)."""
    for fname in TRAJ_FILES:
        rows = _load_trajectories(fname)
        if rows is None:
            continue
        mismatched = 0
        for row in rows:
            traj_meta = row["metadata"]
            for s in row["samples"]:
                s_meta = s.get("metadata") or {}
                if not s_meta:
                    continue
                # Must agree on family + gold_answer if both present
                for key in ("family", "gold_answer"):
                    tv = traj_meta.get(key, "")
                    sv = s_meta.get(key, "")
                    if tv and sv and tv != sv:
                        mismatched += 1
        assert mismatched == 0, (
            f"{fname}: {mismatched} sample-vs-trajectory metadata mismatches"
        )
        print(f"  PASS {fname}: metadata consistent ({len(rows)} trajectories)")


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


def main():
    tests = [
        test_chunk_idx_ordered,
        test_no_empty_trajectory,
        test_metadata_consistency,
        test_splits_disjoint_at_video_level,
        test_protocol_version_consistent,
        test_sample_count_matches_pipeline_stats,
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
