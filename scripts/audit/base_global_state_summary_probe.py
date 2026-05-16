#!/usr/bin/env python3
"""Probe whether the base VL model can self-compress streaming KV into state memory.

This script first runs a no-<response> streaming caption session over a historical
range, asks the same accumulated conversation to emit a compact
<global_state_summary>, then tests whether that summary can replace ordinary
historical think text for later streaming and historical time-range QA.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


def call_with_max_tokens(args: argparse.Namespace, messages: List[Dict], max_tokens: int) -> Dict:
    old = args.max_tokens
    args.max_tokens = max_tokens
    try:
        return probe.call_model(args, messages)
    finally:
        args.max_tokens = old


def caption_system() -> Dict:
    return {
        "role": "system",
        "content": (
            "You are a streaming video captioning and memory agent. Each current turn may contain "
            "<memory>, <current_vision>, and <active_query>. Caption only the newest <current_vision> chunk when asked. "
            "Use <memory> and <recent_vision_memory> only as historical context, never as the current caption."
        ),
    }


def memory_system() -> Dict:
    return {
        "role": "system",
        "content": (
            "You are a streaming video memory agent. Historical memory may contain compact state summaries "
            "and recent observations with exact <t=...> seconds. For time localization questions, answer "
            "only from the initialized history and output exactly one JSON object inside <answer>."
        ),
    }


def parse_global_summary(raw: str) -> str:
    match = re.search(r"<global_state_summary>.*?</global_state_summary>", raw, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    entries = re.findall(r"<m\s+t=[^>]+>.*?</m>", raw, flags=re.DOTALL)
    if entries:
        return "<global_state_summary>\n  " + "\n  ".join(entry.strip() for entry in entries) + "\n</global_state_summary>"
    return "<global_state_summary>\n</global_state_summary>"


def parse_summary_entries(summary_text: str) -> List[Dict]:
    entries = []
    for match in re.finditer(r"<m\s+t=\"([^\"]+)\"\s*>(.*?)</m>", summary_text, flags=re.DOTALL):
        entries.append({
            "time": match.group(1).strip(),
            "text": re.sub(r"\s+", " ", match.group(2)).strip(),
        })
    return entries


def build_summary_request() -> List[Dict]:
    return [{"type": "text", "text": "\n".join([
        "<active_query>",
        (
            "  <q>From all previous streaming observations in this conversation, write a compact "
            "global_state_summary for the historical video segment. Output exactly this XML shape:\n"
            "  <global_state_summary>\n"
            "    <m t=\"start-end\">short state memory</m>\n"
            "  </global_state_summary>\n"
            "Requirements: produce 4 to 6 entries; use accurate second ranges from the previous turns; "
            "summarize task progress, event state, and important object state; do not write dense visual "
            "captions; do not describe clothing/background unless essential; do not predict future actions; "
            "each entry must be <=25 English words.</q>"
        ),
        "</active_query>",
    ])}]


def build_init_user(
    summary_text: str,
    recent_items: List[Dict],
    recent_visual_rows: List[Dict],
    frames_per_turn: int,
) -> List[Dict]:
    lines = [summary_text]
    lines.append("<recent_memory>")
    if recent_items:
        for item in recent_items:
            caption = item["pred_caption"].replace("\n", " ")[:500]
            lines.append(f'  <m t="{item["current_time"]}">Recent observation: {caption}</m>')
    else:
        lines.append('  <m t="none">No recent ordinary observation text is provided.</m>')
    lines.append("</recent_memory>")
    content: List[Dict] = [{"type": "text", "text": "\n".join(lines)}]
    if recent_visual_rows:
        content.append({"type": "text", "text": "<recent_vision_memory>"})
        for row in recent_visual_rows:
            probe.add_vision(content, row, frames_per_turn, "RECENT HISTORICAL visual memory only; do not caption as current")
        content.append({"type": "text", "text": "</recent_vision_memory>"})
    content.append({"type": "text", "text": "\n".join([
        "<historical_query>",
        (
            "  <q>Historical initialization only: store this compact state summary and any recent memory. "
            "Future current-caption turns must still focus on the newest <current_vision> chunk. "
            "<recent_vision_memory> is context only and must not be described as the current frame. "
            "For historical questions, use the initialized memory and its exact time tags.</q>"
        ),
        "</historical_query>",
    ])})
    return content


def stream_history(args: argparse.Namespace) -> tuple[List[Dict], List[Dict], List[Dict]]:
    rows = probe.load_rows(Path(args.jsonl), args.video_id, args.prefill_start, args.prefill_end)
    messages: List[Dict] = [caption_system()]
    out_rows = []
    for idx, row in enumerate(rows):
        messages.append({
            "role": "user",
            "content": probe.build_stream_user(row, out_rows, args.frames_per_turn, args.memory_lookback),
        })
        response = probe.call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        previous_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = probe.result_row(row, raw, previous_gold, len(messages), response.get("usage", {}))
        out_rows.append(out)
        messages.append({"role": "assistant", "content": raw})
        print(json.dumps({"stage": "history_stream", **out}, ensure_ascii=False), flush=True)
    return rows, out_rows, messages


def ask_global_summary(args: argparse.Namespace, messages: List[Dict]) -> tuple[str, str, Dict]:
    messages.append({"role": "user", "content": build_summary_request()})
    response = call_with_max_tokens(args, messages, args.summary_max_tokens)
    raw = response["choices"][0]["message"].get("content", "")
    summary_text = parse_global_summary(raw)
    messages.append({"role": "assistant", "content": raw})
    return raw, summary_text, response.get("usage", {})


def continue_stream(
    args: argparse.Namespace,
    case_name: str,
    base_messages: List[Dict],
    seed_memory: List[Dict],
    explicit_lookback: int,
) -> Dict:
    rows = probe.load_rows(Path(args.jsonl), args.video_id, args.start_time, args.end_time)
    messages = deepcopy(base_messages)
    out_rows = []
    for idx, row in enumerate(rows):
        generated_for_memory = seed_memory + out_rows
        messages.append({
            "role": "user",
            "content": probe.build_stream_user(row, generated_for_memory, args.frames_per_turn, explicit_lookback),
        })
        response = probe.call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        previous_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = probe.result_row(row, raw, previous_gold, len(messages), response.get("usage", {}))
        out_rows.append(out)
        messages.append({"role": "assistant", "content": raw})
        print(json.dumps({"stage": "continue", "case": case_name, **out}, ensure_ascii=False), flush=True)
    captions = [r["pred_caption"] for r in out_rows]
    return {
        "case": case_name,
        "turns": len(out_rows),
        "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in out_rows) / len(out_rows),
        "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in out_rows) / len(out_rows),
        "old_copy_risk_count": sum(1 for r in out_rows if r["old_copy_risk"]),
        "unique_pred_count": len(set(captions)),
        "top_repeat_count": max((captions.count(c) for c in set(captions)), default=0),
        "results": out_rows,
    }


def ask_time_range_from_memory(args: argparse.Namespace, case_name: str, init_user: List[Dict]) -> Dict:
    rows = []
    for query_name, query_spec in probe.TIME_RANGE_QUERIES.items():
        messages = [memory_system(), {"role": "user", "content": init_user}]
        messages.append({"role": "user", "content": [{"type": "text", "text": "\n".join([
            "<active_query>",
            (
                f'  <q>From the initialized historical memory only, locate this past event: '
                f"{query_spec['question']} Return the most specific historical time range in seconds. "
                'Use format exactly: <answer>{"time_range":"start-end","evidence":"short reason"}</answer>.</q>'
            ),
            "</active_query>",
        ])}]})
        response = call_with_max_tokens(args, messages, args.qa_max_tokens)
        raw = response["choices"][0]["message"].get("content", "")
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else raw.strip()
        predicted = probe.parse_time_range(answer)
        expected = query_spec["expected_range"]
        row = {
            "case": case_name,
            "time_query": query_name,
            "question": query_spec["question"],
            "expected_range": expected,
            "predicted_range": predicted,
            "answer": answer,
            "raw": raw,
            **probe.range_overlap(predicted, expected),
            "usage": response.get("usage", {}),
        }
        rows.append(row)
        print(json.dumps({"stage": "history_qa", **row}, ensure_ascii=False), flush=True)
    return {
        "case": case_name,
        "n": len(rows),
        "hit_rate": sum(1 for row in rows if row["hit"]) / len(rows),
        "mean_iou": sum(row["iou"] for row in rows) / len(rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--model", default="qwen3vl2b-prefill-matrix")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--video-id", default="Making_Vanilla_Banana_Bread")
    parser.add_argument("--prefill-start", type=int, default=0)
    parser.add_argument("--prefill-end", type=int, default=60)
    parser.add_argument("--start-time", type=int, default=61)
    parser.add_argument("--end-time", type=int, default=90)
    parser.add_argument("--frames-per-turn", type=int, default=1)
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--recent-text-sec", type=int, default=8)
    parser.add_argument("--recent-visual-sec", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--summary-max-tokens", type=int, default=320)
    parser.add_argument("--qa-max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=100000)
    args = parser.parse_args()

    probe.wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_rows, history_outputs, kv_messages = stream_history(args)
    raw_summary, summary_text, summary_usage = ask_global_summary(args, kv_messages)
    summary_entries = parse_summary_entries(summary_text)

    recent_text = history_outputs[-args.recent_text_sec:] if args.recent_text_sec > 0 else []
    recent_visual_times = set(range(args.prefill_end - args.recent_visual_sec + 1, args.prefill_end + 1)) if args.recent_visual_sec > 0 else set()
    recent_visual_rows = [row for row in history_rows if row["current_time"] in recent_visual_times]

    init_summary_only = build_init_user(summary_text, [], [], args.frames_per_turn)
    init_summary_recent_text = build_init_user(summary_text, recent_text, [], args.frames_per_turn)
    init_summary_recent_text_visual = build_init_user(summary_text, recent_text, recent_visual_rows, args.frames_per_turn)
    init_recent_text_only = build_init_user("<global_state_summary>\n</global_state_summary>", recent_text, [], args.frames_per_turn)

    cases = []
    cases.append(continue_stream(
        args,
        "full_kv_plus_self_summary_no_explicit_old_text",
        kv_messages,
        [],
        0,
    ))
    cases.append(continue_stream(
        args,
        "reset_self_summary_only",
        [caption_system(), {"role": "user", "content": init_summary_only}],
        [],
        0,
    ))
    cases.append(continue_stream(
        args,
        "reset_self_summary_plus_recent_text",
        [caption_system(), {"role": "user", "content": init_summary_recent_text}],
        [],
        0,
    ))
    if recent_visual_rows:
        cases.append(continue_stream(
            args,
            "reset_self_summary_plus_recent_text_visual",
            [caption_system(), {"role": "user", "content": init_summary_recent_text_visual}],
            [],
            0,
        ))
    cases.append(continue_stream(
        args,
        "reset_recent_text_only_no_summary",
        [caption_system(), {"role": "user", "content": init_recent_text_only}],
        [],
        0,
    ))

    qa_cases = [
        ask_time_range_from_memory(args, "self_summary_only", init_summary_only),
        ask_time_range_from_memory(args, "self_summary_plus_recent_text", init_summary_recent_text),
        ask_time_range_from_memory(args, "recent_text_only_no_summary", init_recent_text_only),
    ]

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "jsonl": args.jsonl,
        "video_id": args.video_id,
        "prefill_start": args.prefill_start,
        "prefill_end": args.prefill_end,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "frames_per_turn": args.frames_per_turn,
        "recent_text_sec": args.recent_text_sec,
        "recent_visual_sec": args.recent_visual_sec,
        "raw_self_summary": raw_summary,
        "self_summary": summary_text,
        "self_summary_entries": summary_entries,
        "summary_usage": summary_usage,
        "history_stream": {
            "turns": len(history_outputs),
            "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in history_outputs) / len(history_outputs),
            "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in history_outputs) / len(history_outputs),
            "old_copy_risk_count": sum(1 for r in history_outputs if r["old_copy_risk"]),
        },
        "continue_cases": cases,
        "history_qa_cases": qa_cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "self_summary": summary_text,
        "continue_cases": [{k: v for k, v in case.items() if k != "results"} for case in cases],
        "history_qa_cases": [{k: v for k, v in case.items() if k != "results"} for case in qa_cases],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
