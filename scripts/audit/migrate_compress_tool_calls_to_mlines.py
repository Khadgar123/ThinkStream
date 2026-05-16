#!/usr/bin/env python3
"""Rewrite obsolete compress tool_call outputs to compact-memory <m> lines.

Only the compress tool-call shape is migrated:

  <tool_call>{"name":"compress","arguments":{"time_range":[0, 24],"text":"..."}}</tool_call>

Recall tool calls are left untouched. The script recursively transforms string
fields in JSONL records so it covers ``output`` as well as nested message
content fields.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_BEFORE_TOOL_RE = re.compile(
    r"\s*<think>.*?</think>\s*<tool_call>\s*(\{.*?\})\s*</tool_call>\s*",
    re.DOTALL,
)


def _format_time_value(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "0"
    if x.is_integer():
        return str(int(x))
    return f"{x:g}"


def _tool_to_mline(tool_obj: Any) -> str | None:
    if not isinstance(tool_obj, dict) or tool_obj.get("name") != "compress":
        return None
    args = tool_obj.get("arguments") or {}
    if not isinstance(args, dict):
        return None
    tr = args.get("time_range") or []
    if isinstance(tr, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", tr)
        tr = nums[:2]
    if not isinstance(tr, (list, tuple)) or len(tr) < 2:
        return None
    start = _format_time_value(tr[0])
    end = _format_time_value(tr[1])
    text = html.escape(str(args.get("text") or args.get("summary") or "").strip(), quote=False)
    if not text:
        return None
    return f'<m t="{start}-{end}">{text}</m>'


def _replace_match(match: re.Match[str]) -> str:
    try:
        tool_obj = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return match.group(0)
    mline = _tool_to_mline(tool_obj)
    return mline if mline is not None else match.group(0)


def rewrite_text(value: str) -> tuple[str, int]:
    if "<tool_call>" not in value or '"compress"' not in value:
        return value, 0
    before = value
    value = THINK_BEFORE_TOOL_RE.sub(_replace_match, value)
    value = TOOL_CALL_RE.sub(_replace_match, value)
    changed = int(value != before)
    return value, changed


def rewrite_obj(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return rewrite_text(value)
    if isinstance(value, list):
        changed = 0
        out = []
        for item in value:
            new_item, n = rewrite_obj(item)
            out.append(new_item)
            changed += n
        return out, changed
    if isinstance(value, dict):
        changed = 0
        out = {}
        for k, item in value.items():
            new_item, n = rewrite_obj(item)
            out[k] = new_item
            changed += n
        return out, changed
    return value, 0


def iter_jsonl(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file() and root.suffix == ".jsonl":
            yield root
        elif root.is_dir():
            yield from root.rglob("*.jsonl")


def rewrite_file(path: Path, *, write: bool) -> tuple[int, int]:
    changed_records = 0
    changed_fields = 0
    out_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                out_lines.append(line)
                continue
            obj = json.loads(line)
            new_obj, n = rewrite_obj(obj)
            if n:
                changed_records += 1
                changed_fields += n
                line = json.dumps(new_obj, ensure_ascii=False) + "\n"
            out_lines.append(line)
    if write and changed_records:
        path.write_text("".join(out_lines), encoding="utf-8")
    return changed_records, changed_fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    total_files = total_records = total_fields = 0
    for path in iter_jsonl(args.paths):
        records, fields = rewrite_file(path, write=args.write)
        if records:
            total_files += 1
            total_records += records
            total_fields += fields
            print(f"{path}: records={records} fields={fields}")
    mode = "wrote" if args.write else "dry-run"
    print(f"{mode}: files={total_files} records={total_records} fields={total_fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
