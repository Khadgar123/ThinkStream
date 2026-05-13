#!/usr/bin/env python3
"""Run a staged stream-memory visual input matrix.

Each configuration runs:
1. stream: first segment as normal multi-turn current-captioning.
2. prefill_continue: re-prefill selected first-segment memory, then continue.
3. prefill_history_qa: ask about a historical timestamp from initialized memory.
4. prefill_time_range_qa: ask one or more historical time-range questions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PIXEL_PRESETS: dict[str, tuple[int, int]] = {
    "low": (32768, 65536),
    "mid": (65536, 100000),
    "high": (130000, 220000),
}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_probe_cmd(args: argparse.Namespace, mode: str, out_dir: Path, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "scripts/audit/vllm_prefill_memory_caption_probe.py",
        "--mode",
        mode,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--jsonl",
        args.jsonl,
        "--video-id",
        args.video_id,
        "--start-time",
        str(args.start_time),
        "--end-time",
        str(args.end_time),
        "--visual-format",
        args.visual_format,
        "--visual-fps",
        str(args.visual_fps),
        "--memory-lookback",
        str(args.memory_lookback),
        "--min-pixels",
        str(args.min_pixels),
        "--max-pixels",
        str(args.max_pixels),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--wait-server-sec",
        str(args.wait_server_sec),
        "--out-dir",
        str(out_dir),
        *extra,
    ]


def metric_row(config_name: str, phase: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": config_name,
        "phase": phase,
        "mode": summary.get("mode"),
        "visual_format": summary.get("visual_format"),
        "visual_fps": summary.get("visual_fps"),
        "frames_per_turn": summary.get("frames_per_turn"),
        "min_pixels": summary.get("min_pixels"),
        "max_pixels": summary.get("max_pixels"),
    }
    if "results" in summary:
        results = summary["results"]
        last_usage = results[-1].get("usage", {}) if results else {}
        row.update({
            "turns": summary.get("turns"),
            "avg_current_keyword_recall": summary.get("avg_current_keyword_recall"),
            "avg_previous_keyword_recall": summary.get("avg_previous_keyword_recall"),
            "old_copy_risk_count": summary.get("old_copy_risk_count"),
            "unique_pred_count": summary.get("unique_pred_count"),
            "top_repeat_count": summary.get("top_repeat_count"),
            "last_prompt_tokens": last_usage.get("prompt_tokens"),
            "last_total_tokens": last_usage.get("total_tokens"),
        })
    else:
        row.update({
            "history_check_time": summary.get("history_check_time"),
            "history_gold_keyword_recall": summary.get("history_gold_keyword_recall"),
            "history_prefill_keyword_recall": summary.get("history_prefill_keyword_recall"),
            "time_query": summary.get("time_query"),
            "expected_range": summary.get("expected_range"),
            "predicted_range": summary.get("predicted_range"),
            "time_range_hit": summary.get("hit"),
            "time_range_iou": summary.get("iou"),
            "prompt_tokens": (summary.get("usage") or {}).get("prompt_tokens"),
            "total_tokens": (summary.get("usage") or {}).get("total_tokens"),
        })
    return row


def run_one(args: argparse.Namespace, repo: Path, matrix_jsonl: Path) -> dict[str, Any]:
    config_name = (
        f"{args.visual_format}_fps{args.visual_fps:g}_"
        f"{args.pixel_preset}_{args.min_pixels}_{args.max_pixels}"
    )
    config_dir = Path(args.out_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    common_prefill = [
        "--prefill-start",
        str(args.prefill_start),
        "--prefill-end",
        str(args.prefill_end),
        "--prefill-text-mode",
        args.prefill_text_mode,
        "--prefill-text-count",
        str(args.prefill_text_count),
        "--prefill-visual-mode",
        args.prefill_visual_mode,
        "--prefill-visual-count",
        str(args.prefill_visual_count),
        "--prefill-summary-count",
        str(args.prefill_summary_count),
        "--prefill-summary-style",
        args.prefill_summary_style,
        "--prefill-layout",
        args.prefill_layout,
    ]

    stream_dir = config_dir / f"stream_{args.start_time}_{args.end_time}"
    stream_summary_path = stream_dir / "summary.json"
    if not (args.resume and stream_summary_path.exists()):
        run_cmd(build_probe_cmd(args, "stream", stream_dir, []), repo)
    stream_summary = load_summary(stream_summary_path)
    append_jsonl(matrix_jsonl, metric_row(config_name, "stream", stream_summary))

    continue_dir = config_dir / f"continue_{args.continue_start}_{args.continue_end}"
    continue_summary_path = continue_dir / "summary.json"
    if not (args.resume and continue_summary_path.exists()):
        old_start, old_end = args.start_time, args.end_time
        args.start_time, args.end_time = args.continue_start, args.continue_end
        try:
            run_cmd(
                build_probe_cmd(
                    args,
                    "prefill_continue",
                    continue_dir,
                    [
                        "--prefill-summary",
                        str(stream_summary_path),
                        *common_prefill,
                    ],
                ),
                repo,
            )
        finally:
            args.start_time, args.end_time = old_start, old_end
    continue_summary = load_summary(continue_summary_path)
    append_jsonl(matrix_jsonl, metric_row(config_name, "prefill_continue", continue_summary))

    for history_time in args.history_check_times:
        hist_dir = config_dir / f"history_qa_t{history_time}"
        hist_summary_path = hist_dir / "summary.json"
        if not (args.resume and hist_summary_path.exists()):
            run_cmd(
                build_probe_cmd(
                    args,
                    "prefill_history_qa",
                    hist_dir,
                    [
                        "--prefill-summary",
                        str(stream_summary_path),
                        "--history-check-time",
                        str(history_time),
                        *common_prefill,
                    ],
                ),
                repo,
            )
        hist_summary = load_summary(hist_summary_path)
        append_jsonl(matrix_jsonl, metric_row(config_name, f"history_qa_t{history_time}", hist_summary))

    for time_query in args.time_queries:
        range_dir = config_dir / f"time_range_{time_query}"
        range_summary_path = range_dir / "summary.json"
        if not (args.resume and range_summary_path.exists()):
            run_cmd(
                build_probe_cmd(
                    args,
                    "prefill_time_range_qa",
                    range_dir,
                    [
                        "--prefill-summary",
                        str(stream_summary_path),
                        "--time-query",
                        time_query,
                        *common_prefill,
                    ],
                ),
                repo,
            )
        range_summary = load_summary(range_summary_path)
        append_jsonl(matrix_jsonl, metric_row(config_name, f"time_range_{time_query}", range_summary))

    return {"config": config_name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18114/v1")
    parser.add_argument("--model", default="qwen3vl2b-stream-visual-matrix")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--video-id", default="Making_Vanilla_Banana_Bread")
    parser.add_argument("--out-dir", default="output/vllm_stream_visual_matrix/matrix")
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--end-time", type=int, default=59)
    parser.add_argument("--continue-start", type=int, default=60)
    parser.add_argument("--continue-end", type=int, default=89)
    parser.add_argument("--prefill-start", type=int, default=0)
    parser.add_argument("--prefill-end", type=int, default=59)
    parser.add_argument("--visual-format", choices=["image", "video"], required=True)
    parser.add_argument("--visual-fps", type=float, choices=[1.0, 2.0], required=True)
    parser.add_argument("--pixel-preset", choices=sorted(PIXEL_PRESETS), required=True)
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--prefill-text-mode", choices=["none", "all", "recent", "uniform", "uniform_recent", "query_top"], default="uniform_recent")
    parser.add_argument("--prefill-text-count", type=int, default=16)
    parser.add_argument("--prefill-visual-mode", choices=["none", "all", "recent", "uniform", "uniform_recent", "query_top"], default="uniform_recent")
    parser.add_argument("--prefill-visual-count", type=int, default=4)
    parser.add_argument("--prefill-summary-count", type=int, default=8)
    parser.add_argument("--prefill-summary-style", choices=["keywords", "snippets"], default="snippets")
    parser.add_argument("--prefill-layout", choices=["text_then_visual", "visual_then_text", "text_only", "visual_only"], default="text_then_visual")
    parser.add_argument("--history-check-times", type=int, nargs="+", default=[2, 50])
    parser.add_argument("--time-queries", nargs="+", default=["electric_mixer", "flour_measuring"])
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.min_pixels, args.max_pixels = PIXEL_PRESETS[args.pixel_preset]
    return args


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_jsonl = out_dir / "matrix_metrics.jsonl"
    run_one(args, repo, matrix_jsonl)


if __name__ == "__main__":
    main()
