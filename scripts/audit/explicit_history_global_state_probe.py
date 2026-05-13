#!/usr/bin/env python3
"""Generate global state summaries from explicit history text plus sampled frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import base_global_state_summary_probe as base_probe  # noqa: E402
from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


def load_generated_history(args: argparse.Namespace, rows: List[Dict]) -> List[Dict]:
    by_time: Dict[int, Dict] = {}
    for path_str in args.history_summary_json:
        path = Path(path_str)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("results", []):
            current_time = item.get("current_time")
            if current_time is None:
                continue
            by_time[int(current_time)] = item

    history = []
    missing = []
    for row in rows:
        t = row["current_time"]
        item = by_time.get(t)
        if item:
            caption = item.get("pred_caption") or probe.extract_think(item.get("raw", ""))
        elif args.history_source == "gold":
            caption = row["gold_caption"]
        else:
            missing.append(t)
            continue
        history.append({
            "current_time": t,
            "pred_caption": caption,
            "gold_caption": row["gold_caption"],
        })
    if missing:
        raise RuntimeError(f"missing generated history for times: {missing[:30]}")
    return history


def select_uniform_rows(rows: List[Dict], count: int) -> List[Dict]:
    if count <= 0:
        return []
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[-1]]
    positions = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    selected = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            selected.append(rows[pos])
            seen.add(pos)
    return selected


def build_explicit_summary_prompt(
    history_items: List[Dict],
    frame_rows: List[Dict],
    frames_per_turn: int,
) -> List[Dict]:
    lines = [
        "<historical_think>",
    ]
    for item in history_items:
        text = item["pred_caption"].replace("\n", " ")[:360]
        lines.append(f'  <m t="{item["current_time"]}">{text}</m>')
    lines.append("</historical_think>")
    lines.append("<instruction>")
    lines.append("Compress these observations into a compact historical state summary.")
    lines.append("Keep only task progress, event, object state, and time range.")
    lines.append("Do not describe background/person clothing unless essential.")
    lines.append("Do not predict future actions.")
    lines.append("Use 4 to 6 entries. Each entry must be <=25 English words.")
    lines.append("Output exactly:")
    lines.append("<global_state_summary>")
    lines.append('  <m t="start-end">short state memory</m>')
    lines.append("</global_state_summary>")
    lines.append("</instruction>")
    content: List[Dict] = [{"type": "text", "text": "\n".join(lines)}]
    if frame_rows:
        content.append({"type": "text", "text": "<historical_visual_samples>"})
        for row in frame_rows:
            probe.add_vision(content, row, frames_per_turn, "Uniformly sampled historical video chunk")
        content.append({"type": "text", "text": "</historical_visual_samples>"})
    return content


def generate_summary(
    args: argparse.Namespace,
    case_name: str,
    history_items: List[Dict],
    frame_rows: List[Dict],
) -> Dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise video memory compressor. Use explicit historical think text and optional "
                "sampled video frames to produce compact state memory with accurate time tags."
            ),
        },
        {
            "role": "user",
            "content": build_explicit_summary_prompt(history_items, frame_rows, args.frames_per_turn),
        },
    ]
    response = base_probe.call_with_max_tokens(args, messages, args.summary_max_tokens)
    raw = response["choices"][0]["message"].get("content", "")
    summary_text = base_probe.parse_global_summary(raw)
    return {
        "case": case_name,
        "raw": raw,
        "summary": summary_text,
        "entries": base_probe.parse_summary_entries(summary_text),
        "frame_times": [row["current_time"] for row in frame_rows],
        "usage": response.get("usage", {}),
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
    parser.add_argument("--history-source", choices=["generated", "gold"], default="generated")
    parser.add_argument("--history-summary-json", action="append", default=[
        "output/vllm_latest_frame_probe/noresp_stream_0_59/summary.json",
        "output/vllm_latest_frame_probe/noresp_stream_60_120/summary.json",
    ])
    parser.add_argument("--uniform-frame-counts", default="0,8,12")
    parser.add_argument("--frames-per-turn", type=int, default=1)
    parser.add_argument("--recent-text-sec", type=int, default=8)
    parser.add_argument("--recent-visual-sec", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--summary-max-tokens", type=int, default=360)
    parser.add_argument("--qa-max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=100000)
    args = parser.parse_args()

    probe.wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefill_rows = probe.load_rows(Path(args.jsonl), args.video_id, args.prefill_start, args.prefill_end)
    history_items = load_generated_history(args, prefill_rows)
    frame_counts = [int(x) for x in args.uniform_frame_counts.split(",") if x.strip()]

    recent_text = history_items[-args.recent_text_sec:] if args.recent_text_sec > 0 else []
    recent_visual_times = set(range(args.prefill_end - args.recent_visual_sec + 1, args.prefill_end + 1)) if args.recent_visual_sec > 0 else set()
    recent_visual_rows = [row for row in prefill_rows if row["current_time"] in recent_visual_times]

    summaries = []
    continue_cases = []
    qa_cases = []
    for count in frame_counts:
        frame_rows = select_uniform_rows(prefill_rows, count)
        case = f"explicit_think_uniform{count}frames"
        summary = generate_summary(args, case, history_items, frame_rows)
        summaries.append(summary)
        init_summary_only = base_probe.build_init_user(summary["summary"], [], [], args.frames_per_turn)
        init_summary_recent_text = base_probe.build_init_user(summary["summary"], recent_text, [], args.frames_per_turn)
        init_summary_recent_text_visual = base_probe.build_init_user(summary["summary"], recent_text, recent_visual_rows, args.frames_per_turn)
        continue_cases.append(base_probe.continue_stream(
            args,
            f"{case}_summary_only",
            [base_probe.caption_system(), {"role": "user", "content": init_summary_only}],
            [],
            0,
        ))
        continue_cases.append(base_probe.continue_stream(
            args,
            f"{case}_summary_plus_recent_text",
            [base_probe.caption_system(), {"role": "user", "content": init_summary_recent_text}],
            [],
            0,
        ))
        continue_cases.append(base_probe.continue_stream(
            args,
            f"{case}_summary_plus_recent_text_visual",
            [base_probe.caption_system(), {"role": "user", "content": init_summary_recent_text_visual}],
            [],
            0,
        ))
        qa_cases.append(base_probe.ask_time_range_from_memory(args, f"{case}_summary_only", init_summary_only))
        qa_cases.append(base_probe.ask_time_range_from_memory(args, f"{case}_summary_plus_recent_text", init_summary_recent_text))

    result = {
        "model": args.model,
        "base_url": args.base_url,
        "jsonl": args.jsonl,
        "video_id": args.video_id,
        "prefill_start": args.prefill_start,
        "prefill_end": args.prefill_end,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "history_source": args.history_source,
        "history_summary_json": args.history_summary_json,
        "recent_text_sec": args.recent_text_sec,
        "recent_visual_sec": args.recent_visual_sec,
        "summaries": summaries,
        "continue_cases": continue_cases,
        "history_qa_cases": qa_cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summaries": [{k: v for k, v in item.items() if k not in {"raw"}} for item in summaries],
        "continue_cases": [{k: v for k, v in item.items() if k != "results"} for item in continue_cases],
        "history_qa_cases": [{k: v for k, v in item.items() if k != "results"} for item in qa_cases],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
