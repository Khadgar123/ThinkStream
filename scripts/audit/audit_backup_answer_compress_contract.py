#!/usr/bin/env python3
"""Audit answer surfaces and compact-memory rows in a ThinkStream data root."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


M_LINE_RE = re.compile(r'<m\s+t="([^"]+)"\s*>(.*?)</m>', re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
LETTER_RE = re.compile(r"^[A-Z]$")
LETTER_TEXT_RE = re.compile(r"^[A-Z]\)\s+\S")
INT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def norm_answer_kind(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty"
    lower = text.lower()
    if lower in {"yes", "no"}:
        return "yes_no"
    if LETTER_RE.fullmatch(text):
        return "letter_only"
    if LETTER_TEXT_RE.match(text):
        return "letter_plus_text"
    if INT_RE.fullmatch(text):
        return "number"
    if len(text.split()) <= 6:
        return "short_text"
    return "long_text"


def parse_t(tag: str) -> tuple[int, int] | None:
    tag = str(tag or "").strip()
    if re.fullmatch(r"\d+", tag):
        v = int(tag)
        return v, v
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tag)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def option_text(question: dict[str, Any]) -> str:
    options = list(question.get("options") or [])
    correct = str(question.get("correct_option") or "").strip().upper()
    if len(correct) != 1:
        return ""
    idx = ord(correct) - ord("A")
    if idx < 0 or idx >= len(options):
        return ""
    raw = str(options[idx])
    m = re.match(r"^\s*(?:\(([A-Z])\)|([A-Z])[\).:])\s*(.*)\s*$", raw, re.DOTALL)
    return (m.group(3) if m else raw).strip()


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def audit_question(q: dict[str, Any], counters: Counter, examples: dict[str, list[str]]) -> None:
    form = str(q.get("answer_form") or "unknown")
    counters[f"answer_form/{form}"] += 1
    counters[f"answer_style/{q.get('answer_style') or 'missing'}"] += 1
    instruction = str(q.get("answer_instruction") or "")
    if "one letter only" in instruction:
        counters["instruction/one_letter_only"] += 1
    if "letter plus option text" in instruction:
        counters["instruction/letter_plus_text"] += 1

    values: list[Any] = []
    for emit in q.get("per_emit_answers") or []:
        if isinstance(emit, dict):
            values.append(emit.get("value"))
    for key in ("sft_answer", "gold_answer", "canonical_answer", "correct_answer_text"):
        if q.get(key):
            values.append(q.get(key))
    for value in values:
        kind = norm_answer_kind(value)
        counters[f"answer_surface/{kind}"] += 1
        if len(examples.setdefault(kind, [])) < 5:
            examples[kind].append(str(value))

    if form == "multiple_choice":
        text = option_text(q)
        correct = str(q.get("correct_option") or "").strip().upper()
        if correct and text:
            counters["mc/has_option_text"] += 1
            target = f"{correct}) {text}"
            if target in {str(x.get("value") or "").strip() for x in q.get("per_emit_answers") or [] if isinstance(x, dict)}:
                counters["mc/per_emit_letter_plus_text"] += 1
            else:
                counters["mc/per_emit_not_letter_plus_text"] += 1
        else:
            counters["mc/missing_option_text"] += 1


def audit_compress_text(
    text: str,
    *,
    prefix: str,
    expected_chunks: Any,
    counters: Counter,
    examples: dict[str, list[str]],
) -> None:
    counters[f"{prefix}/rows"] += 1
    lines = M_LINE_RE.findall(text)
    counters[f"{prefix}/line_count/{len(lines)}"] += 1
    if 4 <= len(lines) <= 6:
        counters[f"{prefix}/line_count_4_6"] += 1
    if not lines:
        counters[f"{prefix}/no_m_lines"] += 1
        key = f"{prefix}_no_m"
        if len(examples.setdefault(key, [])) < 5:
            examples[key].append(text[:240].replace("\n", "\\n"))
        return

    expected = sorted(
        int(x) for x in (expected_chunks or [])
        if isinstance(x, int) or str(x).isdigit()
    )
    parsed_ranges: list[tuple[int, int]] = []
    for tag, body in lines:
        tr = parse_t(tag)
        if tr is None:
            counters[f"{prefix}/t_other"] += 1
        elif tr[0] == tr[1]:
            counters[f"{prefix}/t_single"] += 1
            parsed_ranges.append(tr)
        else:
            counters[f"{prefix}/t_range"] += 1
            parsed_ranges.append(tr)
        if str(body or "").strip():
            counters[f"{prefix}/nonempty_body"] += 1

    if expected and parsed_ranges:
        got_start = min(a for a, _ in parsed_ranges)
        got_end = max(b for _, b in parsed_ranges)
        exp_start, exp_end = min(expected), max(expected)
        if got_start <= exp_start and got_end >= exp_end:
            counters[f"{prefix}/covers_expected"] += 1
        else:
            counters[f"{prefix}/does_not_cover_expected"] += 1
            key = f"{prefix}_bad_range"
            if len(examples.setdefault(key, [])) < 8:
                examples[key].append(
                    f"expected={exp_start}-{exp_end} tags={[tag for tag, _ in lines]} "
                    f"text={text[:220].replace(chr(10), ' ')}"
                )


def audit_compress(sample: dict[str, Any], counters: Counter, examples: dict[str, list[str]]) -> None:
    if sample.get("sample_type") != "compress" and sample.get("action") != "compress":
        return
    text = str(sample.get("gold_caption") or sample.get("output") or "")
    audit_compress_text(
        text,
        prefix="source_compress",
        expected_chunks=sample.get("gold_compress_chunks"),
        counters=counters,
        examples=examples,
    )
    output = str(sample.get("output") or "")
    match = TOOL_CALL_RE.search(output)
    if not match:
        counters["source_compress_toolcall/missing"] += 1
        return
    try:
        tool = json.loads(match.group(1))
    except json.JSONDecodeError:
        counters["source_compress_toolcall/bad_json"] += 1
        return
    if tool.get("name") != "compress":
        counters["source_compress_toolcall/not_compress"] += 1
        return
    args = tool.get("arguments") or {}
    tr = args.get("time_range")
    body = str(args.get("text") or args.get("summary") or "").strip()
    counters["source_compress_toolcall/rows"] += 1
    if body:
        counters["source_compress_toolcall/has_text"] += 1
    if isinstance(tr, list) and len(tr) >= 2:
        counters["source_compress_toolcall/has_time_range"] += 1
        expected = sorted(
            int(x) for x in (sample.get("gold_compress_chunks") or [])
            if isinstance(x, int) or str(x).isdigit()
        )
        if expected:
            try:
                got_start, got_end = int(tr[0]), int(tr[1])
            except (TypeError, ValueError):
                counters["source_compress_toolcall/bad_time_range"] += 1
            else:
                exp_start, exp_end = min(expected), max(expected)
                # The legacy tool_call convention used [start, end_exclusive]
                # while <m> uses an inclusive t="start-end" range.
                if got_start <= exp_start and got_end >= exp_end:
                    counters["source_compress_toolcall/covers_expected"] += 1
                else:
                    counters["source_compress_toolcall/does_not_cover_expected"] += 1
                    key = "source_compress_toolcall_bad_range"
                    if len(examples.setdefault(key, [])) < 8:
                        examples[key].append(
                            f"expected={exp_start}-{exp_end} tool_range={tr} "
                            f"text={body[:220].replace(chr(10), ' ')}"
                        )
    else:
        counters["source_compress_toolcall/missing_time_range"] += 1


def audit_rendered_row(row: dict[str, Any], counters: Counter, examples: dict[str, list[str]]) -> None:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        return

    all_text = "\n".join(
        content_text(msg.get("content"))
        for msg in messages
        if isinstance(msg, dict)
    )
    if "one letter only" in all_text:
        counters["rendered_instruction/one_letter_only"] += 1
    if "letter plus option text" in all_text:
        counters["rendered_instruction/letter_plus_text"] += 1

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = content_text(msg.get("content"))
        for match in re.finditer(r"</Response>\s*(.*?)\s*$", text, re.DOTALL):
            value = match.group(1).strip()
            kind = norm_answer_kind(value)
            counters[f"rendered_answer_surface/{kind}"] += 1
            if len(examples.setdefault(f"rendered_{kind}", [])) < 5:
                examples[f"rendered_{kind}"].append(value)

    if row.get("trajectory_type") == "compact_memory_update" or row.get("sample_type") == "compress":
        assistant_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_text = content_text(msg.get("content"))
                break
        event = row.get("compress_event") or {}
        expected = event.get("source_chunks") if isinstance(event, dict) else None
        audit_compress_text(
            assistant_text,
            prefix="rendered_compress",
            expected_chunks=expected,
            counters=counters,
            examples=examples,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="data/batch1_4_high_concurrency_backup_20260515_2057",
    )
    args = parser.parse_args()
    root = Path(args.root)
    counters: Counter = Counter()
    examples: dict[str, list[str]] = {}
    seen_questions: set[str] = set()

    traj_paths = sorted(root.glob("batch*/final/*_trajectories.jsonl"))
    for path in traj_paths:
        # train_rl/train_sft can duplicate the same underlying trajectories.
        split_key = path.name.replace("_trajectories.jsonl", "")
        if split_key == "train_rl":
            continue
        counters["files/trajectory_jsonl"] += 1
        for row in iter_jsonl(path):
            for q in row.get("questions") or []:
                if not isinstance(q, dict):
                    continue
                key = str(q.get("card_id") or "") or json.dumps(q, sort_keys=True)
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                audit_question(q, counters, examples)
            for sample in row.get("samples") or []:
                if isinstance(sample, dict):
                    audit_compress(sample, counters, examples)

    rendered_paths = sorted(root.glob("batch*/rendered/trajectory/*_trajectory.jsonl"))
    for path in rendered_paths:
        counters["files/rendered_trajectory_jsonl"] += 1
        for row in iter_jsonl(path):
            audit_rendered_row(row, counters, examples)

    print(f"root={root}")
    print(f"unique_questions={len(seen_questions)}")
    for key in sorted(counters):
        print(f"{key}={counters[key]}")
    if examples:
        print("examples:")
        for key in sorted(examples):
            print(f"  {key}:")
            for item in examples[key]:
                print(f"    - {item}")


if __name__ == "__main__":
    main()
