#!/usr/bin/env python3
"""Precompute fixed StreamingBench eval windows.

The manifest is the stable input for later rollout. It freezes row ordering,
resolved source video path, 25-45s online window, timestamped frame paths, and
recall-video block frame paths so the evaluator does not redo CSV/video/window
resolution on every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.streamingbench.base_vllm import (  # noqa: E402
    VideoResolver,
    _sample_uniform,
    frame_index_for_time,
    load_rows,
    row_to_manifest,
)


def _block_to_json(block: tuple[float, float, list[Path]]) -> dict[str, Any]:
    start, end, frames = block
    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "frame_paths": [str(p) for p in frames],
        "n_frames": len(frames),
    }


def _frame_dir_for(video: Path, args: argparse.Namespace) -> Path:
    rel = video.parent.relative_to(args.video_root)
    return args.frames_root / rel


def _frames_for_dir(frame_dir: Path, cache: dict[Path, list[Path]]) -> list[Path]:
    if frame_dir not in cache:
        cache[frame_dir] = sorted(frame_dir.glob("frame_*.jpg"))
    return cache[frame_dir]


def _frame_window_from_cache(
    video: Path,
    args: argparse.Namespace,
    cache: dict[Path, list[Path]],
    *,
    start: float,
    end: float,
    max_frames: int = 0,
) -> list[Path]:
    frames = _frames_for_dir(_frame_dir_for(video, args), cache)
    if not frames:
        return []
    first = frame_index_for_time(max(0.0, start), args.fps)
    last = max(first, frame_index_for_time(max(start, end), args.fps) - 1)
    selected = [
        p for p in frames
        if first <= int(p.stem.rsplit("_", 1)[-1]) <= last
    ]
    if max_frames:
        selected = _sample_uniform(selected, max_frames)
    return selected


def _recall_blocks_from_cache(
    video: Path,
    args: argparse.Namespace,
    cache: dict[Path, list[Path]],
    *,
    start: float,
    end: float,
) -> list[tuple[float, float, list[Path]]]:
    n_blocks = max(1, int(args.recall_video_blocks))
    span = max(0.001, end - start)
    out: list[tuple[float, float, list[Path]]] = []
    for i in range(n_blocks):
        b_start = start + span * i / n_blocks
        b_end = start + span * (i + 1) / n_blocks
        frames = _frame_window_from_cache(
            video,
            args,
            cache,
            start=b_start,
            end=b_end,
            max_frames=max(1, int(args.recall_block_frames)),
        )
        if frames:
            out.append((b_start, b_end, frames))
    return out


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(args.csv_dir, sample_per_csv=args.sample_per_csv, seed=args.seed)
    if args.limit:
        rows = rows[: int(args.limit)]
    resolver = VideoResolver(args.video_root)

    items: list[dict[str, Any]] = []
    missing_video: list[str] = []
    missing_frames: list[str] = []
    span_outside_contract: list[str] = []
    frame_cache: dict[Path, list[Path]] = {}
    for row in rows:
        video = resolver.resolve(row)
        if video is None:
            missing_video.append(row.question_id)
            continue

        start = max(0.0, float(row.timestamp_sec) - float(args.window_sec))
        end = float(row.timestamp_sec)
        span = end - start
        if args.require_strict_span and not (25.0 <= span <= 45.0):
            span_outside_contract.append(row.question_id)
            continue
        frame_paths = _frame_window_from_cache(
            video,
            args,
            frame_cache,
            start=start,
            end=end,
            max_frames=int(args.max_frames_per_request or 0),
        )
        if not frame_paths:
            missing_frames.append(row.question_id)
            if args.require_frames:
                continue

        recall_blocks = _recall_blocks_from_cache(
            video,
            args,
            frame_cache,
            start=start,
            end=end,
        )
        item = {
            **row_to_manifest(row),
            "video": str(video),
            "window_sec": float(args.window_sec),
            "fps": float(args.fps),
            "window_start": round(start, 3),
            "window_end": round(end, 3),
            "span_sec": round(span, 3),
            "frame_paths": [str(p) for p in frame_paths],
            "n_frames": len(frame_paths),
            "recall_image_frame_paths": [
                str(p) for p in _sample_uniform(frame_paths, max(1, int(args.recall_image_frames)))
            ],
            "recall_video_blocks": [_block_to_json(block) for block in recall_blocks],
        }
        items.append(item)

    spans = [float(x["span_sec"]) for x in items]
    summary = {
        "csv_dir": str(args.csv_dir),
        "video_root": str(args.video_root),
        "frames_root": str(args.frames_root),
        "rows_loaded": len(rows),
        "rows_manifest": len(items),
        "missing_video": len(missing_video),
        "missing_frames": len(missing_frames),
        "span_outside_25_45": len(span_outside_contract),
        "window_sec": float(args.window_sec),
        "fps": float(args.fps),
        "min_span_sec": min(spans) if spans else 0,
        "max_span_sec": max(spans) if spans else 0,
        "all_spans_inside_25_45": all(25.0 <= s <= 45.0 for s in spans),
        "recall_image_frames": int(args.recall_image_frames),
        "recall_video_blocks": int(args.recall_video_blocks),
        "recall_block_frames": int(args.recall_block_frames),
    }
    return items, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/StreamingBench"))
    parser.add_argument("--video-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/extracted"))
    parser.add_argument("--frames-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/frames_fps2"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=32.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames-per-request", type=int, default=0)
    parser.add_argument("--recall-image-frames", type=int, default=8)
    parser.add_argument("--recall-video-blocks", type=int, default=4)
    parser.add_argument("--recall-block-frames", type=int, default=8)
    parser.add_argument("--sample-per-csv", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-frames", action="store_true")
    parser.add_argument("--allow-outside-25-45", dest="require_strict_span", action="store_false")
    parser.set_defaults(require_strict_span=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (25.0 <= float(args.window_sec) <= 45.0):
        raise SystemExit(f"--window-sec must stay inside 25-45s, got {args.window_sec}")
    items, summary = build_manifest(args)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_path = args.summary_out or args.out_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "out_jsonl": str(args.out_jsonl), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
