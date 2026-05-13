#!/usr/bin/env python3
"""Base-model visual carrier matrix for streaming multi-turn probes.

Compares:
  image  - two pre-extracted frames as timestamped image_url items
  video  - two pre-extracted frames packed into one data:video/jpeg video_url
  mp4    - two pre-extracted frames encoded as a tiny mp4 video_url

Each run streams a first segment, re-prefills that segment as memory, continues
on the next segment, and asks simple historical caption QA for selected times.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import vllm_prefill_memory_caption_probe as probe


PIXEL_PROFILES = {
    "low": (32768, 65536),
    "mid": (65536, 100000),
    "legacy": (130000, 220000),
    "runtime": (200704, 401408),
}


def csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def metric_from_stream(config: Dict[str, Any], phase: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    results = summary.get("results", [])
    usage = results[-1].get("usage", {}) if results else {}
    return {
        **config,
        "phase": phase,
        "turns": summary.get("turns"),
        "avg_current_keyword_recall": summary.get("avg_current_keyword_recall"),
        "avg_previous_keyword_recall": summary.get("avg_previous_keyword_recall"),
        "old_copy_risk_count": summary.get("old_copy_risk_count"),
        "unique_pred_count": summary.get("unique_pred_count"),
        "top_repeat_count": summary.get("top_repeat_count"),
        "last_prompt_tokens": usage.get("prompt_tokens"),
        "last_total_tokens": usage.get("total_tokens"),
    }


def metric_from_history(config: Dict[str, Any], history_time: int, summary: Dict[str, Any]) -> Dict[str, Any]:
    usage = summary.get("usage", {})
    return {
        **config,
        "phase": f"history_qa_t{history_time}",
        "history_check_time": history_time,
        "history_gold_keyword_recall": summary.get("history_gold_keyword_recall"),
        "history_prefill_keyword_recall": summary.get("history_prefill_keyword_recall"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_quiet(log_path: Path, fn, *args):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        return fn(*args)


def make_probe_args(args: argparse.Namespace, video_id: str, visual_format: str, pixel_name: str, out_dir: Path) -> SimpleNamespace:
    min_pixels, max_pixels = PIXEL_PROFILES[pixel_name]
    return SimpleNamespace(
        mode="stream",
        base_url=args.base_url,
        model=args.model,
        jsonl=args.jsonl,
        video_id=video_id,
        start_time=args.start_time,
        end_time=args.start_time + args.segment_len - 1,
        prefill_start=args.start_time,
        prefill_end=args.start_time + args.segment_len - 1,
        prefill_summary="",
        prefill_text_mode=args.prefill_text_mode,
        prefill_text_count=args.prefill_text_count,
        prefill_visual_mode=args.prefill_visual_mode,
        prefill_visual_count=args.prefill_visual_count,
        prefill_summary_count=args.prefill_summary_count,
        prefill_summary_start=None,
        prefill_summary_end=None,
        prefill_summary_style=args.prefill_summary_style,
        prefill_layout=args.prefill_layout,
        history_check_time=args.start_time,
        time_query="flour_measuring",
        frames_per_turn=2 if args.visual_fps == 2 else 1,
        visual_format=visual_format,
        visual_fps=float(args.visual_fps),
        memory_lookback=args.memory_lookback,
        isolated_turns=False,
        out_dir=str(out_dir),
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        wait_server_sec=args.wait_server_sec,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def run_one(args: argparse.Namespace, video_id: str, visual_format: str, pixel_name: str, metrics_path: Path) -> None:
    config_name = f"{video_id}__{visual_format}_fps{args.visual_fps:g}_{pixel_name}"
    out_dir = Path(args.out_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pa = make_probe_args(args, video_id, visual_format, pixel_name, out_dir / "stream")
    config = {
        "video_id": video_id,
        "config": config_name,
        "visual_format": visual_format,
        "visual_fps": args.visual_fps,
        "pixel_profile": pixel_name,
        "min_pixels": pa.min_pixels,
        "max_pixels": pa.max_pixels,
        "segment": [pa.start_time, pa.end_time],
        "continue_segment": [pa.end_time + 1, pa.end_time + args.continue_len],
    }

    stream_summary_path = Path(pa.out_dir) / "summary.json"
    if args.resume and stream_summary_path.exists():
        stream_summary = json.loads(stream_summary_path.read_text(encoding="utf-8"))
    else:
        pa.mode = "stream"
        stream_summary = run_quiet(out_dir / "stream.log", probe.run_stream, pa)
    append_jsonl(metrics_path, metric_from_stream(config, "stream", stream_summary))

    cont_dir = out_dir / "continue"
    pc = make_probe_args(args, video_id, visual_format, pixel_name, cont_dir)
    pc.mode = "prefill_continue"
    pc.start_time = pa.end_time + 1
    pc.end_time = pa.end_time + args.continue_len
    pc.prefill_summary = str(stream_summary_path)
    cont_summary_path = cont_dir / "summary.json"
    if args.resume and cont_summary_path.exists():
        cont_summary = json.loads(cont_summary_path.read_text(encoding="utf-8"))
    else:
        cont_summary = run_quiet(out_dir / "continue.log", probe.run_prefill_continue, pc)
    append_jsonl(metrics_path, metric_from_stream(config, "prefill_continue", cont_summary))

    for history_time in args.history_times:
        if history_time < pa.start_time or history_time > pa.end_time:
            continue
        hist_dir = out_dir / f"history_t{history_time}"
        ph = make_probe_args(args, video_id, visual_format, pixel_name, hist_dir)
        ph.mode = "prefill_history_qa"
        ph.prefill_summary = str(stream_summary_path)
        ph.history_check_time = history_time
        hist_summary_path = hist_dir / "summary.json"
        if args.resume and hist_summary_path.exists():
            hist_summary = json.loads(hist_summary_path.read_text(encoding="utf-8"))
        else:
            hist_summary = run_quiet(out_dir / f"history_t{history_time}.log", probe.run_prefill_history_qa, ph)
        append_jsonl(metrics_path, metric_from_history(config, history_time, hist_summary))


def parse_history_times(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18114/v1")
    parser.add_argument("--model", default="qwen3vl2b-stream-visual-matrix")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--videos", default="surveil_112")
    parser.add_argument("--formats", default="image,video,mp4")
    parser.add_argument("--pixel-profiles", default="low,mid,legacy")
    parser.add_argument("--visual-fps", type=float, choices=[1.0, 2.0], default=2.0)
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--segment-len", type=int, default=20)
    parser.add_argument("--continue-len", type=int, default=10)
    parser.add_argument("--history-times", type=parse_history_times, default=[2, 15])
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--prefill-text-mode", default="uniform_recent")
    parser.add_argument("--prefill-text-count", type=int, default=12)
    parser.add_argument("--prefill-visual-mode", default="uniform_recent")
    parser.add_argument("--prefill-visual-count", type=int, default=4)
    parser.add_argument("--prefill-summary-count", type=int, default=6)
    parser.add_argument("--prefill-summary-style", default="snippets")
    parser.add_argument("--prefill-layout", default="text_then_visual")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--out-dir", default="output/base_visual_carrier_matrix")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "matrix_metrics.jsonl"
    if not args.resume and metrics_path.exists():
        metrics_path.unlink()
    probe.wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    for video_id in csv(args.videos):
        for visual_format in csv(args.formats):
            for pixel_name in csv(args.pixel_profiles):
                if pixel_name not in PIXEL_PROFILES:
                    raise ValueError(f"unknown pixel profile: {pixel_name}")
                run_one(args, video_id, visual_format, pixel_name, metrics_path)


if __name__ == "__main__":
    main()
