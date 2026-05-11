#!/usr/bin/env python
"""Measure real Qwen-VL token budgets for timeline/image-pad rows.

The script samples rendered ShareGPT rows, runs the same SFT processor path as
training, and reports expanded sequence lengths plus image_pad/video_pad counts.
It is meant for sizing visual pixels, compression thresholds, and max lengths.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

IGNORE_INDEX = -100
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sample_rows(rows: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    chosen: Dict[str, Dict[str, Any]] = {}

    def add(row: Dict[str, Any]) -> None:
        key = str(row.get("sample_id") or id(row))
        chosen[key] = row

    # Always include long cheap-estimate rows.
    for row in sorted(rows, key=lambda r: int(r.get("num_tokens", 0) or 0), reverse=True)[: max(8, limit // 4)]:
        add(row)

    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        meta = row.get("meta") or row.get("metadata") or {}
        typ = str(row.get("sample_type") or meta.get("sample_type") or "unknown")
        by_type[typ].append(row)
    per_type = max(2, limit // max(1, len(by_type)) // 2)
    for group in by_type.values():
        for row in group[:per_type]:
            add(row)
        if len(group) > per_type:
            for row in rng.sample(group, min(per_type, len(group))):
                add(row)

    remaining = [r for r in rows if str(r.get("sample_id") or id(r)) not in chosen]
    if len(chosen) < limit and remaining:
        for row in rng.sample(remaining, min(limit - len(chosen), len(remaining))):
            add(row)
    return list(chosen.values())[:limit]


def msg_text(messages: List[Dict[str, Any]], role: str | None = None) -> str:
    parts: List[str] = []
    for msg in messages:
        if role is not None and msg.get("role") != role:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "".join(parts)


def token_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def text_part_counts(tokenizer, messages: List[Dict[str, Any]]) -> Dict[str, int]:
    system_text = msg_text(messages, "system")
    user_text = msg_text(messages, "user")
    assistant_text = msg_text(messages, "assistant")
    parts = {
        "system_text": system_text,
        "memory_text": "".join(re.findall(r"<memory>.*?</memory>", user_text, re.DOTALL)),
        "active_query_text": "".join(re.findall(r"<active_query>.*?</active_query>", user_text, re.DOTALL)),
        "response_history_text": "".join(re.findall(r"<response_history>.*?</response_history>", user_text, re.DOTALL)),
        "recall_text": "".join(re.findall(r"<recalled_frames>.*?</recalled_frames>|<recall_result>.*?</recall_result>", user_text, re.DOTALL)),
        "user_input_text": "".join(re.findall(r"<user_input>.*?</user_input>", user_text, re.DOTALL)),
        "visual_tag_text": "".join(re.findall(r"</?(?:VISUAL_CHUNK|RECALLED_CHUNK)[^>]*>", user_text)),
        "assistant_text": assistant_text,
    }
    return {name: token_len(tokenizer, text) for name, text in parts.items()}


def quantiles(values: List[int]) -> Dict[str, int]:
    if not values:
        return {}
    xs = sorted(values)
    def q(p: float) -> int:
        idx = min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))
        return int(xs[idx])
    return {
        "min": int(xs[0]),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": int(xs[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--base-path", type=Path, default=Path("."))
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--min-pixels", type=int, default=90_000)
    ap.add_argument("--max-pixels", type=int, default=160_000)
    args = ap.parse_args()

    from scripts.eval.processor_loader import load_processor_for_checkpoint
    from thinkstream.sft.args import DataArguments
    from thinkstream.sft.data_processor import preprocess_per_timestep, update_processor_pixels

    processor = load_processor_for_checkpoint(args.processor)
    data_args = DataArguments()
    data_args.min_pixels = int(args.min_pixels)
    data_args.max_pixels = int(args.max_pixels)
    processor = update_processor_pixels(processor, data_args)
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    image_pad_id = vocab.get("<|image_pad|>")
    video_pad_id = vocab.get("<|video_pad|>")

    rows = list(iter_rows(args.jsonl))
    rows = sample_rows(rows, args.limit, args.seed)

    metrics: Dict[str, List[int]] = collections.defaultdict(list)
    worst: List[Dict[str, Any]] = []
    by_type = collections.Counter()
    failures: List[str] = []
    for row in rows:
        meta = row.get("meta") or row.get("metadata") or {}
        typ = str(row.get("sample_type") or meta.get("sample_type") or "unknown")
        by_type[typ] += 1
        row = dict(row)
        row.setdefault("data_path", str(args.base_path))
        try:
            out = preprocess_per_timestep(row, processor, data_args)
        except Exception as exc:
            failures.append(f"{row.get('sample_id')}: {type(exc).__name__}: {exc}")
            continue
        input_ids = out["input_ids"][0].tolist()
        labels = out["labels"][0].tolist()
        label_tokens = sum(1 for x in labels if x != IGNORE_INDEX)
        image_tokens = input_ids.count(image_pad_id) if image_pad_id is not None else 0
        video_tokens = input_ids.count(video_pad_id) if video_pad_id is not None else 0
        total = len(input_ids)
        prompt_context = total - label_tokens

        metrics["total_train_seq"].append(total)
        metrics["prompt_context"].append(prompt_context)
        metrics["assistant_output"].append(label_tokens)
        metrics["image_pad_tokens"].append(image_tokens)
        metrics["video_pad_tokens"].append(video_tokens)
        metrics["non_visual_tokens"].append(total - image_tokens - video_tokens)
        for name, value in text_part_counts(tok, row.get("messages") or []).items():
            metrics[name].append(value)
        worst.append({
            "sample_id": row.get("sample_id"),
            "type": typ,
            "chunk_idx": row.get("chunk_idx"),
            "total": total,
            "prompt_context": prompt_context,
            "assistant_output": label_tokens,
            "image_pad_tokens": image_tokens,
            "video_pad_tokens": video_tokens,
        })

    report = {
        "jsonl": str(args.jsonl),
        "sampled_rows": len(rows),
        "processed_rows": len(metrics.get("total_train_seq", [])),
        "by_type": dict(by_type),
        "quantiles": {k: quantiles(v) for k, v in sorted(metrics.items())},
        "worst_total": sorted(worst, key=lambda x: x["total"], reverse=True)[:10],
        "failures": failures[:20],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
