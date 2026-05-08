#!/usr/bin/env python
"""Audit ThinkStream timeline/interleaved rendered message files.

This is intentionally structural and image-free: it walks ShareGPT JSONL
messages, checks timeline tags, validates simple temporal boundaries, and
reports any fallback to the legacy visual_window layout.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


AGENT_CHUNK_SEC = 1
FRAMES_PER_CHUNK = 2


LEGACY_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("visual_window", re.compile(r"<visual_window|</visual_window>|visual_window")),
    ("frame_id", re.compile(r"\bF_\d{3,}\b")),
    ("role_attr", re.compile(r"\brole=\"")),
    ("frame_tag", re.compile(r"<frame\b")),
    ("covers_time", re.compile(r"covers_time=")),
    ("key_times", re.compile(r"\bkey_times\b")),
    ("bad_visual_time_range", re.compile(r"<VISUAL_CHUNK[^>]*time_range")),
]


TAG_PAIRS = [
    ("memory", "<memory>", "</memory>"),
    ("visual_chunk", "<VISUAL_CHUNK", "</VISUAL_CHUNK>"),
    ("recalled_chunk", "<RECALLED_CHUNK", "</RECALLED_CHUNK>"),
    ("summary", "<SUMMARY", "</SUMMARY>"),
    ("memory_think", "<MEMORY_THINK", "</MEMORY_THINK>"),
]


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                yield path, lineno, json.loads(line)


def content_text(messages: List[Dict[str, Any]], *, role: str | None = None) -> str:
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
    return "\n".join(parts)


def media_count(item: Dict[str, Any]) -> int:
    typ = item.get("type")
    if typ == "image" or item.get("image") or item.get("image_url"):
        return 1
    if typ == "video":
        video = item.get("video")
        if isinstance(video, list):
            return len(video)
        return 1 if video else 0
    return 0


def row_chunk(row: Dict[str, Any]) -> int:
    meta = row.get("metadata") or row.get("meta") or {}
    for obj in (row, meta):
        if obj.get("chunk_idx") is not None:
            return int(obj.get("chunk_idx") or 0)
        if obj.get("current_chunk") is not None:
            return int(obj.get("current_chunk") or 0)
    return 0


def row_sample_type(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") or row.get("meta") or {}
    return str(row.get("sample_type") or meta.get("sample_type") or "unknown")


def is_inter_chunk(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") or row.get("meta") or {}
    return bool(row.get("v12_inter_chunk") or meta.get("v12_inter_chunk"))


def audit_media_grouping(messages: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    memory_depth = 0
    open_tag: str | None = None
    open_media = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                if open_tag:
                    open_media += media_count(item)
                continue
            text = str(item.get("text", ""))
            if "<memory>" in text:
                memory_depth += text.count("<memory>")
            if "<VISUAL_CHUNK" in text:
                if memory_depth:
                    errors.append("visual_chunk_open_inside_memory")
                if open_tag:
                    errors.append(f"nested_media_tag:{open_tag}")
                open_tag = "visual"
                open_media = 0
            if "<RECALLED_CHUNK" in text:
                if memory_depth:
                    errors.append("recalled_chunk_open_inside_memory")
                if open_tag:
                    errors.append(f"nested_media_tag:{open_tag}")
                open_tag = "recalled"
                open_media = 0
            if "</VISUAL_CHUNK>" in text:
                if open_tag != "visual":
                    errors.append("visual_chunk_close_without_open")
                elif open_media != FRAMES_PER_CHUNK:
                    errors.append(f"visual_chunk_media_count:{open_media}")
                open_tag = None
                open_media = 0
            if "</RECALLED_CHUNK>" in text:
                if open_tag != "recalled":
                    errors.append("recalled_chunk_close_without_open")
                elif open_media <= 0 or open_media % FRAMES_PER_CHUNK != 0:
                    errors.append(f"recalled_chunk_media_count:{open_media}")
                open_tag = None
                open_media = 0
            if "</memory>" in text:
                memory_depth -= text.count("</memory>")
                if memory_depth < 0:
                    errors.append("memory_close_without_open")
                    memory_depth = 0
    if open_tag:
        errors.append(f"unclosed_media_tag:{open_tag}")
    if memory_depth:
        errors.append("unclosed_memory")
    return errors


def audit_row(row: Dict[str, Any]) -> List[str]:
    messages = row.get("messages") or []
    text = content_text(messages)
    user_text = content_text(messages, role="user")
    errors: List[str] = []
    for name, pattern in LEGACY_PATTERNS:
        if pattern.search(text):
            errors.append(f"legacy:{name}")
    for name, start, end in TAG_PAIRS:
        start_count = user_text.count(start)
        end_count = user_text.count(end)
        if start_count != end_count:
            errors.append(f"unbalanced:{name}:{start_count}!={end_count}")
    errors.extend(audit_media_grouping(messages))

    chunk = row_chunk(row)
    current_t = chunk * AGENT_CHUNK_SEC
    visual_times = [int(x) for x in re.findall(r"<VISUAL_CHUNK time=\"(\d+)\"", user_text)]
    if any(t > current_t for t in visual_times):
        errors.append("future_visual_chunk")
    summary_ranges = re.findall(r"<SUMMARY time_range=\"([^\"]+)\"", user_text)
    for raw in summary_ranges:
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*", raw)
        if not m:
            errors.append(f"bad_summary_time_range:{raw}")
            continue
        end = float(m.group(2))
        if end > current_t + AGENT_CHUNK_SEC:
            errors.append(f"future_summary:{raw}@chunk{chunk}")
    think_times = [int(float(x)) for x in re.findall(r"<MEMORY_THINK time=\"([0-9.]+)\"", user_text)]
    if any(t > current_t for t in think_times):
        errors.append("future_memory_think")
    if is_inter_chunk(row) and visual_times:
        errors.append("compress_turn_has_visual")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--max-errors", type=int, default=40)
    args = ap.parse_args()

    by_type = collections.Counter()
    by_error = collections.Counter()
    examples: List[str] = []
    total = 0
    for path, lineno, row in iter_jsonl(args.jsonl):
        total += 1
        by_type[row_sample_type(row)] += 1
        errs = audit_row(row)
        for err in errs:
            by_error[err] += 1
        if errs and len(examples) < args.max_errors:
            examples.append(
                f"{path}:{lineno} sample_id={row.get('sample_id')} "
                f"type={row_sample_type(row)} errors={errs}"
            )

    print(json.dumps({
        "rows": total,
        "by_type": dict(by_type),
        "error_counts": dict(by_error),
        "n_error_rows_shown": len(examples),
    }, indent=2, ensure_ascii=False))
    for item in examples:
        print(item)
    if by_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
