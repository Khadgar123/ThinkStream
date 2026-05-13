#!/usr/bin/env python3
"""Estimate rendered multi-turn SFT trajectory sequence lengths."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from transformers import AutoTokenizer


VISUAL_TOKENS_PER_FRAME_RUNTIME = 235


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "n": float(len(values)),
        "min": float(min(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and "text" in item
        )
    return ""


def count_media_frames(messages: list[dict[str, Any]]) -> tuple[int, int, Counter[int]]:
    blocks = 0
    frames = 0
    hist: Counter[int] = Counter()
    for msg in messages:
        content = msg.get("content") or []
        if isinstance(content, str):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video" or item.get("video"):
                blocks += 1
                video = item.get("video")
                n = len(video) if isinstance(video, list) else int(item.get("nframes") or 0)
                frames += n
                hist[n] += 1
            elif item.get("type") == "image" or item.get("image") or item.get("image_url"):
                blocks += 1
                frames += 1
                hist[1] += 1
    return blocks, frames, hist


def scan(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    per_row: defaultdict[str, list[float]] = defaultdict(list)
    by_type: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    type_counts: Counter[str] = Counter()
    frame_hist: Counter[int] = Counter()
    threshold_counts: Counter[str] = Counter()
    per_file: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    n_rows = 0

    for path in args.sft_jsonl:
        file_rows = 0
        for line_no, sample in enumerate(iter_jsonl(path), 1):
            if args.max_rows and n_rows >= args.max_rows:
                break
            messages = sample.get("messages") or []
            tools = sample.get("tools")
            kwargs = {"tokenize": True, "add_generation_prompt": False}
            if tools is not None:
                kwargs["tools"] = tools
            try:
                ids = tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - audit should keep going.
                errors.append({
                    "path": str(path),
                    "line": line_no,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                })
                continue

            blocks, frames, hist = count_media_frames(messages)
            assistant_text = 0
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_text += len(tokenizer.encode(content_text(msg.get("content")), add_special_tokens=False))

            seq_est = len(ids) + frames * args.visual_tokens_per_frame - ids.count(video_pad_id)
            if seq_est <= args.max_sample_tokens:
                threshold_counts["within_max_sample_tokens"] += 1
            else:
                threshold_counts["over_max_sample_tokens"] += 1
            kind = str(sample.get("trajectory_type") or sample.get("loss_class") or sample.get("sample_type") or "")
            type_counts[kind] += 1
            frame_hist.update(hist)
            metrics = {
                "seq_est": seq_est,
                "raw_template_tokens": len(ids),
                "frames": frames,
                "video_blocks": blocks,
                "messages": len(messages),
                "assistant_text_tokens": assistant_text,
            }
            for key, value in metrics.items():
                per_row[key].append(float(value))
                by_type[kind][key].append(float(value))
            n_rows += 1
            file_rows += 1

        per_file.append({"path": str(path), "rows": file_rows})
        if args.max_rows and n_rows >= args.max_rows:
            break

    return {
        "path": [str(path) for path in args.sft_jsonl],
        "per_file": per_file,
        "model": str(args.model),
        "rows": n_rows,
        "type_counts": dict(type_counts),
        "frame_hist": dict(sorted(frame_hist.items())),
        "errors": errors[: args.max_errors],
        "assumptions": {
            "visual_tokens_per_frame": args.visual_tokens_per_frame,
            "max_sample_tokens": args.max_sample_tokens,
        },
        "threshold_counts": dict(threshold_counts),
        "per_row": {key: stats(values) for key, values in per_row.items()},
        "per_row_by_type": {
            kind: {key: stats(values) for key, values in values_by_metric.items()}
            for kind, values_by_metric in sorted(by_type.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model",
        default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--visual-tokens-per-frame", type=int, default=VISUAL_TOKENS_PER_FRAME_RUNTIME)
    parser.add_argument("--max-sample-tokens", type=int, default=16384)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(scan(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
