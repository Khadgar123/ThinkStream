#!/usr/bin/env python3
"""Legacy helper: create paired no-split prefix rows from split benchmark rows.

The input is a ThinkStream trajectory JSONL. Each output row keeps the same
question payload and gold answer schedule, but starts from chunk 0 and runs
until the last answer/ask chunk plus post-context. This gives a paired baseline
for "accumulate KV from the beginning" without changing the benchmark question
set.

Current benchmark eval should prefer each benchmark builder's native
``--split-policy continuous_prefix``. This helper cannot recover questions that
were already dropped by an upstream short-window split.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ints(values: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _question_terminal_chunk(q: Dict[str, Any]) -> int:
    chunks: List[int] = []
    chunks.extend(_ints(q.get("answer_chunks") or []))
    chunks.extend(_ints(q.get("ask_chunks") or []))
    chunks.extend(_ints([q.get("ask_chunk")]))
    chunks.extend(_ints([q.get("open_until")]))
    for item in q.get("per_emit_answers") or []:
        if isinstance(item, dict):
            chunks.extend(_ints([item.get("chunk")]))
    return max(chunks, default=0)


def _trajectory_terminal_chunk(row: Dict[str, Any], *, post_context_chunks: int) -> int:
    chunks = [_question_terminal_chunk(q) for q in row.get("questions") or []]
    for raw in (row.get("gold_action_per_chunk") or {}).keys():
        chunks.extend(_ints([raw]))
    end = max(chunks, default=int(row.get("segment_end_chunk") or 0))
    return max(0, int(end) + max(0, int(post_context_chunks)))


def _as_task(q: Dict[str, Any]) -> str:
    return str(
        q.get("ovo_task")
        or q.get("streamingbench_task_type")
        or q.get("family")
        or q.get("family_name")
        or "unknown"
    )


def make_unsplit(
    rows: Iterable[Dict[str, Any]],
    *,
    post_context_chunks: int,
    track_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        new_row = copy.deepcopy(row)
        old_meta = dict(new_row.get("ovo_split_meta") or {})
        old_start = int(new_row.get("segment_start_chunk") or 0)
        old_end = int(new_row.get("segment_end_chunk") or 0)
        new_end = _trajectory_terminal_chunk(
            new_row,
            post_context_chunks=post_context_chunks,
        )
        old_tid = str(new_row.get("trajectory_id") or f"traj_{i:06d}")
        new_tid = f"{old_tid}__{track_name}"
        new_row["trajectory_id"] = new_tid
        new_row["video_id"] = new_tid
        new_row["segment_start_chunk"] = 0
        new_row["segment_end_chunk"] = int(new_end)
        new_row["stats"] = {
            **dict(new_row.get("stats") or {}),
            "n_chunks_covered": int(new_end) + 1,
            "chunk_idx_max": int(new_end),
            "n_questions": len(new_row.get("questions") or []),
        }
        new_row["ovo_split_meta"] = {
            **old_meta,
            "source_segment_start_chunk": old_start,
            "source_segment_end_chunk": old_end,
            "segment_start_chunk": 0,
            "segment_end_chunk": int(new_end),
            "split_policy": "unsplit_prefix",
            "benchmark_track": track_name,
            "unsplit_prefix": True,
            "post_context_chunks": int(post_context_chunks),
        }
        out.append(new_row)
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(obj)
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


def _summary(rows: List[Dict[str, Any]], *, parquet_rows: int | None = None) -> Dict[str, Any]:
    spans = [
        int(r.get("segment_end_chunk") or 0) - int(r.get("segment_start_chunk") or 0) + 1
        for r in rows
    ]
    tasks = Counter()
    for row in rows:
        for q in row.get("questions") or []:
            if isinstance(q, dict):
                tasks[_as_task(q)] += 1
    out: Dict[str, Any] = {
        "trajectories": len(rows),
        "questions": int(sum(len(r.get("questions") or []) for r in rows)),
        "tasks": dict(tasks),
        "min_span": min(spans, default=0),
        "max_span": max(spans, default=0),
        "avg_span": (sum(spans) / len(spans)) if spans else 0.0,
        "max_segment_end_chunk": max((int(r.get("segment_end_chunk") or 0) for r in rows), default=0),
        "split_policy": "unsplit_prefix",
    }
    if parquet_rows is not None:
        out["parquet_rows"] = int(parquet_rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-parquet", type=Path, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--post-context-chunks", type=int, default=2)
    ap.add_argument("--track-name", default="unsplit_prefix")
    ap.add_argument("--max-questions-per-trajectory", type=int, default=16)
    args = ap.parse_args()

    source = _read_jsonl(args.in_jsonl)
    rows = make_unsplit(
        source,
        post_context_chunks=int(args.post_context_chunks),
        track_name=str(args.track_name),
    )
    _write_jsonl(args.out_jsonl, rows)
    parquet_rows: int | None = None
    if args.out_parquet:
        parquet_rows = _write_parquet(
            args.out_jsonl,
            args.out_parquet,
            max_questions_per_traj=int(args.max_questions_per_trajectory),
        )
    summary = _summary(rows, parquet_rows=parquet_rows)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
