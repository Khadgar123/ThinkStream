"""Emit trajectory-grouped jsonl files for v12 RL rollout + streaming benchmark.

WHY (per user feedback 2026-04-29):
  Existing data/agent_v5/final/{train_rl,val,test}.jsonl emit FLAT per-step
  samples after applying MAX_SAMPLES_PER_VIDEO=15 round-robin truncation.
  That destroys two properties RL + benchmark eval depend on:

  1. **Trajectory continuity** — RL chunk-level rollout (v12.2 ChunkLevelRolloutLoop)
     plays back chunks 0..N sequentially. Sparse single-step samples force
     the agent to "imagine" middle chunks at inference time → train→eval
     distribution shift.
  2. **No information loss** — pass4 already filters quality (88% pass);
     the additional 15-sample/video cap then drops 90% MORE. For benchmark
     and RL, all 47k verified samples should be available, organized by
     trajectory.

WHAT THIS SCRIPT DOES:
  - Reuses the EXISTING per-video split (read from train_sft.jsonl /
    train_rl.jsonl / val.jsonl / test.jsonl which already have video_id).
  - For each video in each split, loads data/agent_v5/verified/{video}.json
    (47,174 samples total — pre-cap, post-quality-filter).
  - Groups samples by (video_id, trajectory_id), preserves chunk order.
  - Emits trajectory-grouped jsonl: one row = one trajectory.

OUTPUT FILES (data/agent_v5/final/):
  train_sft_trajectories.jsonl    — SFT-side trajectories (109 videos)
  train_rl_trajectories.jsonl     — RL-side trajectories (109 videos)
  val_trajectories.jsonl          — val-side trajectories (47 videos)
  test_trajectories.jsonl         — test-side trajectories (47 videos)
  trajectories_manifest.json      — per-split stats + video lists

EACH ROW (jsonl):
  {
    "video_id": str,
    "trajectory_id": str,
    "video_path": str,
    "card_id": str,                  # Common card from first sample
    "metadata": {                    # Trajectory-level metadata (from first sample)
      "family": str,
      "gold_action": str,
      "gold_answer": str,
      "answer_form": str,
      "support_chunks": [int, ...],
      "gold_compress_chunks": [int, ...],
      "availability": str,
    },
    "samples": [                     # Full per-step samples, ordered by chunk_idx
      {"chunk_idx": int, "sample_type": str, "input": {...},
       "output": str, "verification": {...}, ...},
      ...
    ],
    "stats": {
      "n_samples": int,
      "chunk_idx_min": int,
      "chunk_idx_max": int,
      "n_chunks_covered": int,
      "actions": {action: count, ...},
    }
  }

NON-DESTRUCTIVE: This script does NOT modify the existing
{train,train_sft,train_rl,val,test}.jsonl files. Adds new files alongside.

Usage:
    python -m scripts.agent_data_v5.emit_trajectories
    AGENT_DATA_DIR=/cluster/path/data/agent_v5\\
        python -m scripts.agent_data_v5.emit_trajectories
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "agent_v5"


def _read_video_ids(jsonl_path: Path) -> Set[str]:
    """Extract unique video_ids from a flat per-step jsonl split."""
    if not jsonl_path.exists():
        return set()
    vids: Set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = row.get("video_id")
            if vid:
                vids.add(vid)
    return vids


def _load_verified_samples(verified_dir: Path, video_id: str) -> List[Dict]:
    """Load all samples for one video from verified/{video}.json.

    Returns the `samples` list (post-pass4-quality-filter, pre-density-cap).
    Each sample carries verification status. Returns [] if file missing.
    """
    fp = verified_dir / f"{video_id}.json"
    if not fp.exists():
        logger.warning(f"  verified/{video_id}.json missing; skipping")
        return []
    try:
        with fp.open() as f:
            blob = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"  verified/{video_id}.json malformed: {e}")
        return []
    if not isinstance(blob, dict) or "samples" not in blob:
        logger.warning(f"  verified/{video_id}.json missing 'samples' key")
        return []
    return blob["samples"] or []


def _build_trajectory_record(
    video_id: str,
    trajectory_id: str,
    samples: List[Dict],
) -> Dict:
    """Bundle one trajectory's samples into a single row.

    Sorts samples by chunk_idx, then by sample_type so that within-chunk
    multi-turn (e.g., recall_query before recall_response) keeps order.
    Lifts trajectory-level metadata from the first sample (all samples in
    a trajectory share the same card_id / family / gold_answer).
    """
    sorted_samples = sorted(
        samples,
        key=lambda s: (
            int(s.get("chunk_idx", 0)),
            # Stable secondary sort for multi-turn within a chunk:
            # query before response, silent last.
            {"recall_query": 0, "recall_response": 1, "response": 2,
             "compress": 3, "silent": 4, "recall_silent": 5}.get(
                s.get("sample_type", "silent"), 6
            ),
        ),
    )

    if not sorted_samples:
        return {
            "video_id": video_id,
            "trajectory_id": trajectory_id,
            "samples": [],
            "stats": {"n_samples": 0},
        }

    first = sorted_samples[0]
    meta = first.get("metadata") or {}

    # Stats
    chunks = [int(s.get("chunk_idx", 0)) for s in sorted_samples]
    actions: Dict[str, int] = defaultdict(int)
    for s in sorted_samples:
        actions[s.get("sample_type", "?")] += 1

    return {
        "video_id": video_id,
        "trajectory_id": trajectory_id,
        "video_path": first.get("video_path", ""),
        "card_id": first.get("card_id", ""),
        "protocol_version": first.get("protocol_version", "v12"),
        "metadata": {
            "family": meta.get("family", ""),
            "gold_action": meta.get("gold_action", ""),
            "gold_answer": meta.get("gold_answer", ""),
            "canonical_answer": meta.get("canonical_answer", ""),
            "answer_form": meta.get("answer_form", ""),
            "availability": meta.get("availability", ""),
            "support_chunks": meta.get("support_chunks", []),
            "gold_compress_chunks": meta.get("gold_compress_chunks", []),
        },
        "samples": sorted_samples,
        "stats": {
            "n_samples": len(sorted_samples),
            "chunk_idx_min": min(chunks),
            "chunk_idx_max": max(chunks),
            "n_chunks_covered": len(set(chunks)),
            "actions": dict(actions),
        },
    }


def _emit_split(
    split_name: str,
    video_ids: Set[str],
    verified_dir: Path,
    out_path: Path,
) -> Dict:
    """Build trajectory-grouped jsonl for one split. Returns stats."""
    n_videos = 0
    n_trajectories = 0
    n_samples = 0
    n_videos_no_data = 0
    action_totals: Dict[str, int] = defaultdict(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f:
        for vid in sorted(video_ids):
            samples = _load_verified_samples(verified_dir, vid)
            if not samples:
                n_videos_no_data += 1
                continue
            n_videos += 1
            # Group by trajectory_id within this video
            by_traj: Dict[str, List[Dict]] = defaultdict(list)
            for s in samples:
                tid = s.get("trajectory_id") or "unknown_traj"
                by_traj[tid].append(s)
            for tid, tsamples in sorted(by_traj.items()):
                rec = _build_trajectory_record(vid, tid, tsamples)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_trajectories += 1
                n_samples += rec["stats"]["n_samples"]
                for act, c in rec["stats"]["actions"].items():
                    action_totals[act] += c

    stats = {
        "split": split_name,
        "videos_in_split": len(video_ids),
        "videos_with_data": n_videos,
        "videos_missing": n_videos_no_data,
        "trajectories": n_trajectories,
        "samples": n_samples,
        "actions": dict(action_totals),
        "output_path": str(out_path),
    }
    logger.info(
        f"  [{split_name}] {n_videos}/{len(video_ids)} videos, "
        f"{n_trajectories} trajs, {n_samples} samples → {out_path.name}"
    )
    return stats


def emit_all(data_dir: Path) -> Dict:
    """Emit all trajectory-grouped split files. Returns manifest dict."""
    verified_dir = data_dir / "verified"
    final_dir = data_dir / "final"

    if not verified_dir.exists():
        raise FileNotFoundError(
            f"{verified_dir} missing — run pass4 first."
        )
    if not final_dir.exists():
        raise FileNotFoundError(
            f"{final_dir} missing — run pipeline.py first."
        )

    splits = {
        "train_sft": _read_video_ids(final_dir / "train_sft.jsonl"),
        "train_rl":  _read_video_ids(final_dir / "train_rl.jsonl"),
        "val":       _read_video_ids(final_dir / "val.jsonl"),
        "test":      _read_video_ids(final_dir / "test.jsonl"),
    }

    logger.info(
        f"Read split assignment: SFT={len(splits['train_sft'])}, "
        f"RL={len(splits['train_rl'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

    # Sanity: must be disjoint
    seen: Set[str] = set()
    for name, vids in splits.items():
        overlap = vids & seen
        if overlap:
            raise ValueError(
                f"split {name} overlaps prior splits on {len(overlap)} videos"
            )
        seen |= vids

    all_stats: Dict[str, Dict] = {}
    for split_name, video_ids in splits.items():
        out_path = final_dir / f"{split_name}_trajectories.jsonl"
        all_stats[split_name] = _emit_split(
            split_name, video_ids, verified_dir, out_path,
        )

    manifest = {
        "generated_by": "emit_trajectories.py",
        "source_verified_dir": str(verified_dir),
        "splits": all_stats,
        "totals": {
            "videos": sum(s["videos_with_data"] for s in all_stats.values()),
            "trajectories": sum(s["trajectories"] for s in all_stats.values()),
            "samples": sum(s["samples"] for s in all_stats.values()),
        },
    }
    manifest_path = final_dir / "trajectories_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AGENT_DATA_DIR", str(DEFAULT_DATA_DIR))),
        help="data/agent_v5 root (contains verified/ and final/).",
    )
    args = ap.parse_args(argv)

    manifest = emit_all(args.data_dir)
    print(f"\nTotals: "
          f"{manifest['totals']['videos']} videos, "
          f"{manifest['totals']['trajectories']} trajectories, "
          f"{manifest['totals']['samples']} samples")


if __name__ == "__main__":
    main()
