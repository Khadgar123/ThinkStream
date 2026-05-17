#!/usr/bin/env python3
"""Sample ThinkStream trajectory JSONL rows and rebuild the paired parquet."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(jsonl_path: Path, parquet_path: Path, *, max_questions_per_traj: int) -> int:
    import pandas as pd

    from scripts.agent_data.build_verl_parquet import _iter_rows_multi_q

    rows = list(
        _iter_rows_multi_q(
            jsonl_path,
            max_questions_per_traj=max(1, int(max_questions_per_traj)),
            frame_protocol="video_meta",
            render_layout="standard_query_last",
            include_student_cache=False,
        )
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return len(rows)


def _span(row: Dict[str, Any]) -> int:
    return int(row.get("segment_end_chunk") or 0) - int(row.get("segment_start_chunk") or 0) + 1


def _terminal(row: Dict[str, Any]) -> int:
    return int(row.get("segment_end_chunk") or 0)


def _question_task(row: Dict[str, Any]) -> str:
    questions = row.get("questions") or []
    if not questions:
        return "unknown"
    q = questions[0]
    if not isinstance(q, dict):
        return "unknown"
    return str(
        q.get("ovo_task")
        or q.get("streamingbench_task_type")
        or q.get("family")
        or q.get("family_name")
        or "unknown"
    )


def _select(rows: List[Dict[str, Any]], *, mode: str, limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    if mode == "head":
        return rows[:limit]
    if mode == "longest":
        return sorted(rows, key=lambda r: (_span(r), _terminal(r)), reverse=True)[:limit]
    if mode == "latest":
        return sorted(rows, key=lambda r: (_terminal(r), _span(r)), reverse=True)[:limit]
    if mode == "task_round_robin":
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(_question_task(row), []).append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: (_terminal(r), _span(r)))
        selected: List[Dict[str, Any]] = []
        while len(selected) < limit and any(buckets.values()):
            for task in sorted(buckets):
                bucket = buckets[task]
                if bucket:
                    selected.append(bucket.pop(0))
                    if len(selected) >= limit:
                        break
        return selected
    if mode == "random":
        rng = random.Random(seed)
        return rng.sample(rows, limit)
    raise ValueError(f"unknown sample mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-parquet", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument(
        "--mode",
        default="task_round_robin",
        choices=["head", "longest", "latest", "task_round_robin", "random"],
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-questions-per-trajectory", type=int, default=16)
    args = ap.parse_args()

    rows = _select(
        _read_jsonl(args.in_jsonl),
        mode=args.mode,
        limit=int(args.limit),
        seed=int(args.seed),
    )
    _write_jsonl(args.out_jsonl, rows)
    parquet_rows = _write_parquet(
        args.out_jsonl,
        args.out_parquet,
        max_questions_per_traj=int(args.max_questions_per_trajectory),
    )
    spans = [_span(r) for r in rows]
    summary = {
        "source": str(args.in_jsonl),
        "mode": args.mode,
        "limit": int(args.limit),
        "trajectories": len(rows),
        "parquet_rows": parquet_rows,
        "tasks": {task: sum(1 for r in rows if _question_task(r) == task) for task in sorted({_question_task(r) for r in rows})},
        "min_span": min(spans, default=0),
        "max_span": max(spans, default=0),
        "avg_span": (sum(spans) / len(spans)) if spans else 0.0,
        "max_segment_end_chunk": max((_terminal(r) for r in rows), default=0),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
