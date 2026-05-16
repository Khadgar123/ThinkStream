#!/usr/bin/env python3
"""Propagate regenerated compact-memory rollouts into training artifacts.

The compact-memory regeneration step updates pass2 ``rollout/*.json``.  Pass4
and pass5 do not read rollout directly; pass4 reads ``verified/*.json`` and
pass5 reads pass4 trajectory files.  This script syncs affected videos from
the fixed rollout into ``samples_3c``, ``verified``, and the flat final split
files without changing video splits or question assignments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.retrofit_compact_memory import (  # noqa: E402
    _json_load,
    migrate_sample_file,
)


FINAL_SPLITS = ("train", "train_sft", "train_rl", "val", "test")
COMPRESS_FIELDS = (
    "output",
    "memory_update_input",
    "gold_caption",
    "gold_compress_chunks",
    "gold_memory_entries",
    "memory_update_mode",
    "user_input",
    "recall_result",
)
COMPRESS_META_FIELDS = (
    "gold_compress_chunks",
    "gold_memory_entries",
    "memory_update_mode",
    "memory_update_input",
    "task_type",
    "gold_action",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _backup(path: Path, backup_root: Path, seen: Set[Path]) -> Path:
    path = path.resolve()
    if path in seen:
        return backup_root / path.relative_to(PROJECT_ROOT)
    dst = backup_root / path.relative_to(PROJECT_ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    seen.add(path)
    return dst


def _targets_from_reports(reports: Sequence[Path]) -> Dict[Path, Set[str]]:
    out: Dict[Path, Set[str]] = defaultdict(set)
    for report_path in reports:
        report = _json_load(report_path)
        for item in report.get("applied_files") or []:
            path = Path(str(item.get("path") or ""))
            video_id = str(item.get("video_id") or path.stem)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            batch_root = path.parent.parent
            out[batch_root].add(video_id)
    return out


def _sample_key(sample: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(sample.get("trajectory_id") or ""),
        int(sample.get("chunk_idx") or 0),
        str(sample.get("sample_type") or ""),
        str(sample.get("action") or ""),
        str(sample.get("card_id") or ""),
        str(sample.get("sequence_type") or ""),
        str(sample.get("base_role") or ""),
    )


def _sample_sort_key(sample: Dict[str, Any]) -> Tuple[int, int, str, str]:
    order = {
        "compress": -1,
        "recall_query": 0,
        "recall": 1,
        "recall_response": 1,
        "response": 2,
        "silent": 4,
        "recall_silent": 5,
    }
    return (
        int(sample.get("chunk_idx") or 0),
        order.get(str(sample.get("sample_type") or ""), 6),
        str(sample.get("trajectory_id") or ""),
        str(sample.get("card_id") or ""),
    )


def _load_verified_index(batch_root: Path, video_ids: Set[str]) -> Dict[str, Dict[Tuple[Any, ...], Deque[Dict[str, Any]]]]:
    by_video: Dict[str, Dict[Tuple[Any, ...], Deque[Dict[str, Any]]]] = {}
    for video_id in sorted(video_ids):
        path = batch_root / "verified" / f"{video_id}.json"
        if not path.exists():
            continue
        blob = _json_load(path)
        samples = blob.get("samples") if isinstance(blob, dict) else blob
        buckets: Dict[Tuple[Any, ...], Deque[Dict[str, Any]]] = defaultdict(deque)
        for sample in sorted(samples or [], key=_sample_sort_key):
            buckets[_sample_key(sample)].append(sample)
        by_video[video_id] = buckets
    return by_video


def _update_final_row(row: Dict[str, Any], updated: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(row)
    if isinstance(out.get("input"), dict) and isinstance(updated.get("input"), dict):
        out["input"] = deepcopy(out["input"])
        if isinstance(updated["input"].get("memory"), dict):
            out["input"]["memory"] = deepcopy(updated["input"]["memory"])
        if out.get("sample_type") == "compress":
            out["input"]["memory_update_input"] = str(
                updated.get("memory_update_input")
                or updated["input"].get("memory_update_input")
                or ""
            ).strip()

    if out.get("sample_type") == "compress":
        for field in COMPRESS_FIELDS:
            if field in updated:
                out[field] = deepcopy(updated[field])
        meta = dict(out.get("metadata") or {})
        updated_meta = updated.get("metadata") or {}
        for field in COMPRESS_META_FIELDS:
            if field in updated_meta:
                meta[field] = deepcopy(updated_meta[field])
            elif field in updated:
                meta[field] = deepcopy(updated[field])
        if out.get("memory_update_input"):
            meta["memory_update_input"] = out["memory_update_input"]
        out["metadata"] = meta
    return out


def _sync_sample_files(batch_root: Path, video_ids: Set[str], backup_root: Path, seen: Set[Path]) -> Dict[str, Any]:
    stats = {
        "videos": 0,
        "samples_3c_files": 0,
        "verified_files": 0,
        "missing": [],
        "sample_stats": {},
    }
    for video_id in sorted(video_ids):
        rollout_path = batch_root / "rollout" / f"{video_id}.json"
        if not rollout_path.exists():
            stats["missing"].append(str(rollout_path))
            continue
        rollout = _json_load(rollout_path)
        stats["videos"] += 1
        for dirname, verified_shape in (("samples_3c", False), ("verified", True)):
            sample_path = batch_root / dirname / f"{video_id}.json"
            if not sample_path.exists():
                stats["missing"].append(str(sample_path))
                continue
            _backup(sample_path, backup_root, seen)
            migrated = migrate_sample_file(
                sample_path,
                rollout,
                sample_path,
                verified_shape=verified_shape,
            )
            stats[f"{dirname}_files"] += 1
            stats["sample_stats"][f"{dirname}/{video_id}"] = migrated
    return stats


def _sync_final_files(batch_root: Path, video_ids: Set[str], backup_root: Path, seen: Set[Path]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"splits": {}, "missing_matches": []}
    for split in FINAL_SPLITS:
        path = batch_root / "final" / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = _load_jsonl(path)
        present_videos = {str(r.get("video_id") or "") for r in rows} & video_ids
        if not present_videos:
            stats["splits"][split] = {"rows": len(rows), "updated_rows": 0}
            continue
        index = _load_verified_index(batch_root, present_videos)
        updated_rows: List[Dict[str, Any]] = []
        n_updated = 0
        n_compress = 0
        for row in rows:
            video_id = str(row.get("video_id") or "")
            if video_id not in present_videos:
                updated_rows.append(row)
                continue
            bucket = index.get(video_id, {}).get(_sample_key(row))
            if not bucket:
                stats["missing_matches"].append({
                    "split": split,
                    "video_id": video_id,
                    "key": list(_sample_key(row)),
                })
                updated_rows.append(row)
                continue
            updated = bucket.popleft()
            new_row = _update_final_row(row, updated)
            updated_rows.append(new_row)
            n_updated += 1
            if row.get("sample_type") == "compress":
                n_compress += 1
        _backup(path, backup_root, seen)
        _write_jsonl(path, updated_rows)
        stats["splits"][split] = {
            "rows": len(rows),
            "updated_rows": n_updated,
            "updated_compress_rows": n_compress,
            "videos": len(present_videos),
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    reports = [
        p if p.is_absolute() else PROJECT_ROOT / p
        for p in (Path(x).expanduser() for x in args.reports)
    ]
    targets = _targets_from_reports(reports)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "data" / "agent_v5" / "audits" / f"training_compact_sync_backup_{stamp}"
    seen: Set[Path] = set()
    all_stats: Dict[str, Any] = {
        "generated_at": stamp,
        "reports": [str(p) for p in reports],
        "backup_root": str(backup_root),
        "batches": {},
    }
    for batch_root, video_ids in sorted(targets.items(), key=lambda kv: str(kv[0])):
        sample_stats = _sync_sample_files(batch_root, video_ids, backup_root, seen)
        final_stats = _sync_final_files(batch_root, video_ids, backup_root, seen)
        all_stats["batches"][str(batch_root)] = {
            "target_videos": len(video_ids),
            "sample_sync": sample_stats,
            "final_sync": final_stats,
        }

    report_out = Path(args.report_out) if args.report_out else PROJECT_ROOT / "data" / "agent_v5" / "audits" / f"sync_compact_rollout_to_training_{stamp}.json"
    if not report_out.is_absolute():
        report_out = PROJECT_ROOT / report_out
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(report_out),
        "backup_root": str(backup_root),
        "batches": {
            str(k): {
                "target_videos": v["target_videos"],
                "sample_sync": {
                    "videos": v["sample_sync"]["videos"],
                    "samples_3c_files": v["sample_sync"]["samples_3c_files"],
                    "verified_files": v["sample_sync"]["verified_files"],
                    "missing": len(v["sample_sync"]["missing"]),
                },
                "final_splits": v["final_sync"]["splits"],
                "missing_final_matches": len(v["final_sync"]["missing_matches"]),
            }
            for k, v in all_stats["batches"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
