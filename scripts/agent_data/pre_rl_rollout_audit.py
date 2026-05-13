#!/usr/bin/env python
"""Compatibility wrapper for the current pre-RL rollout audit.

The old pre-RL audit used a separate simulated/vLLM rollout path. That path is
retired for current correctness checks. Executing this module now forwards to
``scripts/agent_data/run_rl_recurrent_audit.sh``, which runs the same verl
true-KV recurrent AgentLoop used by RL training and test.

The lightweight parquet/JSONL loader helpers remain here for tests and small
inspection scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def _plain(value: Any) -> Any:
    """Convert pandas/pyarrow/numpy wrappers to plain Python containers."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _read_jsonl(path: Path, limit: int = 0) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".gz":
        import gzip

        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_parquet_trajectory_rows(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    """Load the same multi-Q parquet rows used by RL validation/training."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("pandas is required to read parquet audit input") from exc

    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    rows: List[Dict[str, Any]] = []
    for raw in df.to_dict("records"):
        row = _plain(raw)
        extra = row.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            reward_model = {}
        gt_raw = reward_model.get("ground_truth")
        gt: Dict[str, Any] = {}
        if isinstance(gt_raw, str):
            try:
                parsed = json.loads(gt_raw)
                if isinstance(parsed, dict):
                    gt = parsed
            except json.JSONDecodeError:
                gt = {}
        elif isinstance(gt_raw, dict):
            gt = gt_raw

        questions = extra.get("questions") or gt.get("questions") or []
        gold_action = (
            extra.get("gold_action_per_chunk")
            or gt.get("gold_action_per_chunk")
            or row.get("gold_action_per_chunk")
            or {}
        )
        offline_compress = (
            extra.get("offline_compress_chunks")
            or gt.get("offline_compress_chunks")
            or []
        )
        out = {
            "trajectory_id": str(extra.get("index") or row.get("video_id") or ""),
            "video_id": str(row.get("video_id") or extra.get("video_id") or extra.get("index") or ""),
            "video_path": str(row.get("video_path") or extra.get("video_path") or ""),
            "questions": list(questions or []),
            "gold_action_per_chunk": dict(gold_action or {}),
            "offline_compress_chunks": list(offline_compress or []),
            "stats": {"n_chunks_covered": int(row.get("n_chunks") or extra.get("n_chunks") or 0)},
        }
        for key in ("segment_start_chunk", "segment_end_chunk", "source_video_path", "ovo_split_meta"):
            if key in extra:
                out[key] = extra[key]
        rows.append(out)
    return rows


def _load_source_rows(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    if path.suffix == ".parquet":
        return _load_parquet_trajectory_rows(path, limit=limit)
    return list(_read_jsonl(path, limit=limit))


def _trajectory_max_chunk(row: Dict[str, Any]) -> int:
    candidates: List[int] = []
    stats = row.get("stats") or {}
    if isinstance(stats, dict):
        for key in ("chunk_idx_max", "n_chunks_covered", "n_chunks"):
            if key in stats:
                try:
                    value = int(stats[key])
                    candidates.append(value - 1 if key.startswith("n_") else value)
                except (TypeError, ValueError):
                    pass
    for key in ("n_chunks", "segment_end_chunk"):
        if key in row:
            try:
                value = int(row[key])
                candidates.append(value - 1 if key == "n_chunks" else value)
            except (TypeError, ValueError):
                pass
    for ck in (row.get("gold_action_per_chunk") or {}).keys():
        try:
            candidates.append(int(ck))
        except (TypeError, ValueError):
            pass
    for ck in row.get("offline_compress_chunks") or []:
        try:
            candidates.append(int(ck))
        except (TypeError, ValueError):
            pass
    for q in row.get("questions") or []:
        if not isinstance(q, dict):
            continue
        for ck in list(q.get("ask_chunks") or []) + list(q.get("answer_chunks") or []):
            try:
                candidates.append(int(ck))
            except (TypeError, ValueError):
                pass
    return max(candidates) if candidates else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="RL multi-Q parquet or trajectory JSONL")
    ap.add_argument("--ckpt", required=True, help="HF checkpoint to validate")
    ap.add_argument("--frames-root", default="")
    ap.add_argument("--out", default="", help="Compatibility alias; parent dir becomes --out-dir")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-questions-per-traj", default="16")
    args, _unknown = ap.parse_known_args()

    out_dir = args.out_dir
    if not out_dir and args.out:
        out_dir = str(Path(args.out).with_suffix(""))
    if not out_dir:
        out_dir = "output/rl_recurrent_audit"

    cmd = [
        "bash",
        str(ROOT / "scripts/agent_data/run_rl_recurrent_audit.sh"),
        "--ckpt",
        args.ckpt,
        "--source",
        args.source,
        "--out-dir",
        out_dir,
        "--max-questions-per-traj",
        str(args.max_questions_per_traj),
    ]
    if args.frames_root:
        cmd.extend(["--frames-root", args.frames_root])
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
