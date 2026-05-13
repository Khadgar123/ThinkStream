#!/usr/bin/env python3
"""Probe iterative segment memory for training-free streaming.

This focuses on the third-segment question:
after segment 2, should the next summary be generated from only the latest
segment observations, or from previous summary plus latest observations?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import base_global_state_summary_probe as base_probe  # noqa: E402
from scripts.audit import end_to_end_prefill_memory_matrix as e2e  # noqa: E402
from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


def add_init_assistant(messages: List[Dict], mode: str) -> None:
    if mode == "empty":
        messages.append({"role": "assistant", "content": ""})
    elif mode == "loaded":
        messages.append({"role": "assistant", "content": "<memory_loaded/>"})


def stream_rows(args: argparse.Namespace, rows: List[Dict], messages: List[Dict], lookback: int) -> Tuple[List[Dict], List[Dict]]:
    outputs: List[Dict] = []
    for idx, row in enumerate(rows):
        memory_source = outputs if lookback > 0 else []
        messages.append({
            "role": "user",
            "content": probe.build_stream_user(row, memory_source, args.frames_per_turn, lookback),
        })
        response = probe.call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        prev_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = probe.result_row(row, raw, prev_gold, len(messages), response.get("usage", {}))
        outputs.append(out)
        messages.append({"role": "assistant", "content": raw})
        print(json.dumps({
            "stage": "stream",
            "t": row["current_time"],
            "cur": out["current_keyword_recall"],
        }, ensure_ascii=False), flush=True)
    return outputs, messages


def output_metrics(outputs: List[Dict]) -> Dict:
    captions = [item["pred_caption"] for item in outputs]
    return {
        "turns": len(outputs),
        "avg_current_keyword_recall": sum(x["current_keyword_recall"] for x in outputs) / len(outputs) if outputs else 0.0,
        "avg_previous_keyword_recall": sum(x["previous_keyword_recall"] for x in outputs) / len(outputs) if outputs else 0.0,
        "old_copy_risk_count": sum(1 for x in outputs if x["old_copy_risk"]),
        "unique_pred_count": len(set(captions)),
        "top_repeat_count": max((captions.count(c) for c in set(captions)), default=0),
        "max_prompt_tokens": max((x.get("usage", {}).get("prompt_tokens", 0) for x in outputs), default=0),
    }


def summarize_segment(
    args: argparse.Namespace,
    outputs: List[Dict],
    rows: List[Dict],
    start: int,
    end_exclusive: int,
    slot_count: int,
) -> Dict:
    return e2e.generate_summary(
        args,
        "explicit_text",
        [],
        outputs,
        rows,
        0,
        slot_count,
        start,
        end_exclusive,
    )


def merge_summaries(*summary_texts: str) -> str:
    entries: List[Dict] = []
    for text in summary_texts:
        entries.extend(base_probe.parse_summary_entries(text))
    lines = ["<global_state_summary>"]
    for entry in entries:
        clean = re.sub(r"\s+", " ", entry["text"]).strip()
        lines.append(f'  <m t="{entry["time"]}">{clean}</m>')
    lines.append("</global_state_summary>")
    return "\n".join(lines)


def rolling_summary(
    args: argparse.Namespace,
    previous_summary: str,
    outputs: List[Dict],
    start: int,
    end_exclusive: int,
    slot_count: int,
) -> Dict:
    expected_slots = e2e.slot_labels(0, end_exclusive, slot_count)
    lines = [
        "<required_slots>",
    ]
    for label in expected_slots:
        lines.append(f'  <slot t="{label}"/>')
    lines.extend([
        "</required_slots>",
        "<previous_global_state_summary>",
        previous_summary,
        "</previous_global_state_summary>",
        "<new_segment_observations>",
    ])
    for item in outputs:
        text = item["pred_caption"].replace("\n", " ")[:260]
        lines.append(f'  <m t="{item["current_time"]}">{text}</m>')
    lines.extend([
        "</new_segment_observations>",
        "<instruction>",
        "Update the memory for all history so far.",
        "Use previous summary as historical context and the new observations as the latest segment.",
        "Keep task progress, events, object state, and time ranges.",
        "Do not copy dense captions. Do not predict future actions.",
        f"CRITICAL: Output exactly {slot_count} <m> entries.",
        "The t attribute of each <m> must exactly match required_slots, in the same order.",
        "Do not merge slots. Do not omit quiet/repeated slots.",
        "Output exactly this XML shape:",
        "<global_state_summary>",
    ])
    for label in expected_slots:
        lines.append(f'  <m t="{label}">short state memory, <=20 words</m>')
    lines.extend([
        "</global_state_summary>",
        "</instruction>",
    ])
    messages = [{
        "role": "system",
        "content": "You are a precise video memory compressor for iterative streaming.",
    }, {
        "role": "user",
        "content": [{"type": "text", "text": "\n".join(lines)}],
    }]
    attempts = []
    response = base_probe.call_with_max_tokens(args, messages, args.summary_max_tokens)
    raw = response["choices"][0]["message"].get("content", "")
    summary_text = base_probe.parse_global_summary(raw)
    ok, expected, got = e2e.summary_matches_required_slots(summary_text, 0, end_exclusive, slot_count)
    attempts.append({
        "raw": raw,
        "summary": summary_text,
        "usage": response.get("usage", {}),
        "valid_slots": ok,
        "expected_slots": expected,
        "got_slots": got,
    })
    for _ in range(args.summary_retries):
        if ok:
            break
        repair_messages = deepcopy(messages)
        repair_messages.append({"role": "assistant", "content": raw})
        repair_messages.append({"role": "user", "content": [{"type": "text", "text": "\n".join([
            "<repair_instruction>",
            "The previous summary is INVALID.",
            f"Expected t slots exactly: {', '.join(expected)}",
            f"Got t slots: {', '.join(got) if got else 'none'}",
            "Rewrite the summary using exactly the expected slots, one <m> per slot, in order.",
            "Use the original observations above. Do not add explanations.",
            "</repair_instruction>",
        ])}]})
        response = base_probe.call_with_max_tokens(args, repair_messages, args.summary_max_tokens)
        raw = response["choices"][0]["message"].get("content", "")
        summary_text = base_probe.parse_global_summary(raw)
        ok, expected, got = e2e.summary_matches_required_slots(summary_text, 0, end_exclusive, slot_count)
        attempts.append({
            "raw": raw,
            "summary": summary_text,
            "usage": response.get("usage", {}),
            "valid_slots": ok,
            "expected_slots": expected,
            "got_slots": got,
        })
    return {
        "source": "previous_summary_plus_current_segment",
        "slot_count": slot_count,
        "raw": raw,
        "summary": summary_text,
        "entries": base_probe.parse_summary_entries(summary_text),
        "usage": response.get("usage", {}),
        "valid_slots": ok,
        "expected_slots": expected,
        "got_slots": got,
        "attempts": attempts,
    }


def build_prefill_messages(args: argparse.Namespace, summary_text: str, recent_outputs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    init_user = e2e.build_init_user(summary_text, recent_outputs, [], args.frames_per_turn)
    messages = [e2e.caption_system(), {"role": "user", "content": init_user}]
    add_init_assistant(messages, args.init_assistant_mode)
    return messages, init_user


def run(args: argparse.Namespace) -> Dict:
    all_rows = e2e.rows_by_video(Path(args.jsonl))
    video_rows = all_rows[args.video_id]
    s = args.segment_len
    rows0 = e2e.contiguous_rows(video_rows, 0, s)
    rows1 = e2e.contiguous_rows(video_rows, s, 2 * s)
    rows2 = e2e.contiguous_rows(video_rows, 2 * s, 3 * s)

    print(json.dumps({"stage": "segment0_start", "video_id": args.video_id, "segment_len": s}, ensure_ascii=False), flush=True)
    outputs0, messages0 = stream_rows(args, rows0, [e2e.caption_system()], args.memory_lookback)
    summary0 = summarize_segment(args, outputs0, rows0, 0, s, args.summary_slot_count)
    recent0 = outputs0[-args.recent_text_sec:] if args.recent_text_sec > 0 else []

    print(json.dumps({"stage": "segment1_start"}, ensure_ascii=False), flush=True)
    messages1, init1 = build_prefill_messages(args, summary0["summary"], recent0)
    outputs1, _ = stream_rows(args, rows1, messages1, 0)

    summary1_segment_only = summarize_segment(args, outputs1, rows1, s, 2 * s, args.summary_slot_count)
    chain_summary = merge_summaries(summary0["summary"], summary1_segment_only["summary"])
    summary1_with_prior = rolling_summary(
        args,
        summary0["summary"],
        outputs1,
        s,
        2 * s,
        args.summary_slot_count * 2,
    )
    recent1 = outputs1[-args.recent_text_sec:] if args.recent_text_sec > 0 else []

    variants = []
    for name, summary_text in [
        ("append_segment_only_summary", chain_summary),
        ("recompress_previous_summary_plus_current", summary1_with_prior["summary"]),
    ]:
        print(json.dumps({"stage": "segment2_start", "variant": name}, ensure_ascii=False), flush=True)
        messages2, init2 = build_prefill_messages(args, summary_text, recent1)
        outputs2, _ = stream_rows(args, rows2, messages2, 0)
        variants.append({
            "variant": name,
            "summary": summary_text,
            "summary_entries": base_probe.parse_summary_entries(summary_text),
            "init_prompt_items": len(init2),
            "segment2_metrics": output_metrics(outputs2),
            "segment2_results": outputs2,
        })

    return {
        "video_id": args.video_id,
        "segment_len": args.segment_len,
        "recent_text_sec": args.recent_text_sec,
        "init_assistant_mode": args.init_assistant_mode,
        "segment0_metrics": output_metrics(outputs0),
        "segment1_metrics": output_metrics(outputs1),
        "summary0": summary0,
        "summary1_segment_only": summary1_segment_only,
        "summary1_with_prior": summary1_with_prior,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--model", default="qwen3vl2b-prefill-matrix")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--video-id", default="Making_Vanilla_Banana_Bread")
    parser.add_argument("--segment-len", type=int, default=30)
    parser.add_argument("--summary-slot-count", type=int, default=4)
    parser.add_argument("--recent-text-sec", type=int, default=8)
    parser.add_argument("--init-assistant-mode", choices=["none", "empty", "loaded"], default="empty")
    parser.add_argument("--frames-per-turn", type=int, default=1)
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--summary-max-tokens", type=int, default=500)
    parser.add_argument("--summary-retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=100000)
    args = parser.parse_args()

    probe.wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    result = run(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', args.video_id)}_seg{args.segment_len}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_path": str(out_path),
        "segment0": result["segment0_metrics"],
        "segment1": result["segment1_metrics"],
        "variants": [
            {
                "variant": v["variant"],
                "segment2_metrics": v["segment2_metrics"],
                "entries": len(v["summary_entries"]),
            }
            for v in result["variants"]
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
