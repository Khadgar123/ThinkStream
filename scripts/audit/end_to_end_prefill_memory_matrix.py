#!/usr/bin/env python3
"""End-to-end streaming -> summary -> re-prefill -> continue/QA matrix.

The model is responsible for all memory artifacts:
1. stream the first segment and emit per-step observations;
2. generate historical localization QA items for that first segment;
3. generate a compact global_state_summary from either accumulated KV or explicit
   first-segment observations plus optional sampled frames;
4. reset, prefill selected memory, continue the next segment, and answer the
   first-segment QA questions with time ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import base_global_state_summary_probe as base_probe  # noqa: E402
from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


def row_time(row: Dict) -> Optional[int]:
    try:
        visual_window, video = probe.visual_window_and_video(row)
    except Exception:
        return None
    if not video:
        return None
    t = visual_window.get("current_time")
    return int(t) if t is not None else None


def rows_by_video(path: Path) -> Dict[str, Dict[int, Dict]]:
    result: Dict[str, Dict[int, Dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if row.get("sample_type") not in {"silent", "response"}:
                continue
            t = row_time(row)
            if t is None:
                continue
            video_id = row.get("video_id")
            if video_id is None:
                continue
            if video_id not in result:
                result[video_id] = {}
            if t not in result[video_id]:
                visual_window, video = probe.visual_window_and_video(row)
                result[video_id][t] = {
                    "line_no": line_no,
                    "current_time": t,
                    "video": video,
                    "gold_caption": probe.assistant_think(row),
                    "sample_type": row.get("sample_type"),
                    "chunk_idx": row.get("chunk_idx"),
                }
    return result


def contiguous_rows(video_rows: Dict[int, Dict], start: int, end_exclusive: int) -> List[Dict]:
    missing = [t for t in range(start, end_exclusive) if t not in video_rows]
    if missing:
        raise ValueError(f"missing times {missing[:20]}")
    return [video_rows[t] for t in range(start, end_exclusive)]


def parse_range_value(value: str) -> Optional[List[int]]:
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    if len(nums) >= 2:
        return [min(nums[0], nums[1]), max(nums[0], nums[1])]
    if len(nums) == 1:
        return [nums[0], nums[0]]
    return None


def parse_json_object(text: str) -> Optional[Dict]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        pass
    raw = raw.replace("</answer>", "").replace("<answer>", "")
    try:
        return json.loads(raw)
    except Exception:
        return None


def select_uniform(items: List[Dict], count: int) -> List[Dict]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[-1]]
    selected: List[Dict] = []
    seen = set()
    for i in range(count):
        pos = round(i * (len(items) - 1) / (count - 1))
        if pos not in seen:
            selected.append(items[pos])
            seen.add(pos)
    return selected


def make_slots(start: int, end_exclusive: int, slot_count: int) -> List[Tuple[int, int]]:
    total = end_exclusive - start
    slots = []
    for i in range(slot_count):
        s = start + math.floor(i * total / slot_count)
        e = start + math.floor((i + 1) * total / slot_count) - 1
        slots.append((s, max(s, e)))
    return slots


def slot_labels(start: int, end_exclusive: int, slot_count: int) -> List[str]:
    return [f"{s}-{e}" for s, e in make_slots(start, end_exclusive, slot_count)]


def summary_matches_required_slots(summary_text: str, start: int, end_exclusive: int, slot_count: int) -> Tuple[bool, List[str], List[str]]:
    expected = slot_labels(start, end_exclusive, slot_count)
    got = [entry["time"] for entry in base_probe.parse_summary_entries(summary_text)]
    return got == expected, expected, got


def caption_system() -> Dict:
    return {
        "role": "system",
        "content": (
            "You are a streaming video captioning and memory agent. Caption only the newest "
            "<current_vision> chunk when asked. Historical text or visual memory is context only."
        ),
    }


def memory_system() -> Dict:
    return {
        "role": "system",
        "content": (
            "You are a video memory QA agent. Use initialized historical memory and exact time tags. "
            "For time localization questions, output exactly one JSON object inside <answer>."
        ),
    }


def stream_segment(args: argparse.Namespace, rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    messages: List[Dict] = [caption_system()]
    outputs: List[Dict] = []
    for idx, row in enumerate(rows):
        messages.append({
            "role": "user",
            "content": probe.build_stream_user(row, outputs, args.frames_per_turn, args.memory_lookback),
        })
        response = probe.call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        prev_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = probe.result_row(row, raw, prev_gold, len(messages), response.get("usage", {}))
        outputs.append(out)
        messages.append({"role": "assistant", "content": raw})
        print(json.dumps({"stage": "stream", "t": row["current_time"], "cur": out["current_keyword_recall"]}, ensure_ascii=False), flush=True)
    return outputs, messages


def build_generated_qa_prompt(question_count: int, start: int, end_exclusive: int) -> List[Dict]:
    return [{"type": "text", "text": "\n".join([
        "<active_query>",
        (
            f"  <q>Create {question_count} historical time-localization QA items for the segment "
            f"t={start}-{end_exclusive - 1}. Use only events visible in the previous streaming observations. "
            "Choose distinct events across early/middle/late parts. Each question must be answerable by a "
            "specific time range in seconds. Do not choose generic background or repeated whole-video states. "
            "Output exactly JSON inside <answer>: "
            '{"questions":[{"id":"q1","question":"When ...?","expected_range":"start-end","evidence":"short visible evidence"}]}'
            "</q>"
        ),
        "</active_query>",
    ])}]


def generate_qa_set(args: argparse.Namespace, messages: List[Dict], start: int, end_exclusive: int) -> Dict:
    qa_messages = deepcopy(messages)
    qa_messages.append({"role": "user", "content": build_generated_qa_prompt(args.qa_count, start, end_exclusive)})
    response = base_probe.call_with_max_tokens(args, qa_messages, args.qa_gen_max_tokens)
    raw = response["choices"][0]["message"].get("content", "")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
    answer = match.group(1).strip() if match else raw
    data = parse_json_object(answer)
    if data is None:
        try:
            parsed = json.loads(answer)
        except Exception:
            parsed = []
        data = {"questions": parsed} if isinstance(parsed, list) else {"questions": []}
    elif isinstance(data, list):
        data = {"questions": data}
    questions = []
    for i, item in enumerate(data.get("questions", []), start=1):
        expected = parse_range_value(item.get("expected_range", ""))
        if expected is None:
            continue
        questions.append({
            "id": str(item.get("id") or f"q{i}"),
            "question": str(item.get("question") or "").strip(),
            "expected_range": expected,
            "evidence": str(item.get("evidence") or "").strip(),
        })
    return {"raw": raw, "questions": questions, "usage": response.get("usage", {})}


def answer_qa_from_messages(args: argparse.Namespace, case: str, init_messages: List[Dict], questions: List[Dict]) -> Dict:
    rows = []
    for item in questions:
        messages = deepcopy(init_messages)
        messages.append({"role": "user", "content": [{"type": "text", "text": "\n".join([
            "<active_query>",
            (
                f'  <q>From initialized historical memory only, answer this time-localization question: '
                f"{item['question']} Return the most specific historical time range in seconds. "
                'Use format exactly: <answer>{"time_range":"start-end","evidence":"short reason"}</answer>.</q>'
            ),
            "</active_query>",
        ])}]})
        response = base_probe.call_with_max_tokens(args, messages, args.qa_answer_max_tokens)
        raw = response["choices"][0]["message"].get("content", "")
        match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
        answer = match.group(1).strip() if match else raw
        data = parse_json_object(answer) or {}
        predicted = parse_range_value(data.get("time_range", answer))
        overlap = probe.range_overlap(predicted, item["expected_range"])
        rows.append({
            "case": case,
            **item,
            "predicted_range": predicted,
            "answer": answer,
            "raw": raw,
            **overlap,
            "usage": response.get("usage", {}),
        })
    return {
        "case": case,
        "n": len(rows),
        "hit_rate": sum(1 for row in rows if row["hit"]) / len(rows) if rows else 0.0,
        "mean_iou": sum(row["iou"] for row in rows) / len(rows) if rows else 0.0,
        "results": rows,
    }


def build_summary_prompt(
    source: str,
    stream_outputs: List[Dict],
    segment_rows: List[Dict],
    frame_count: int,
    slot_count: int,
    start: int,
    end_exclusive: int,
    frames_per_turn: int,
) -> List[Dict]:
    slots = make_slots(start, end_exclusive, slot_count)
    labels = [f"{s}-{e}" for s, e in slots]
    lines = ["<required_slots>"]
    for label in labels:
        lines.append(f'  <slot t="{label}"/>')
    lines.append("</required_slots>")
    lines.append("<historical_think>")
    for item in stream_outputs:
        text = item["pred_caption"].replace("\n", " ")[:260]
        lines.append(f'  <m t="{item["current_time"]}">{text}</m>')
    lines.append("</historical_think>")
    lines.append("<instruction>")
    lines.append("Compress these observations into compact historical state memory.")
    lines.append("Keep only task progress, event, object state, and time range.")
    lines.append("Do not describe background/person clothing unless essential.")
    lines.append("Do not predict future actions.")
    lines.append("Use low-anchor abstract state wording, not dense captions.")
    lines.append(f"CRITICAL: Output exactly {slot_count} <m> entries, one for every required slot.")
    lines.append("The t attribute of each <m> must exactly match the required slot labels, in the same order.")
    lines.append("Do not merge slots. Do not omit quiet/repeated slots. If a slot repeats prior state, still write that slot's state.")
    lines.append("Output exactly:")
    lines.append("<global_state_summary>")
    for label in labels:
        lines.append(f'  <m t="{label}">short state memory, <=20 words</m>')
    lines.append("</global_state_summary>")
    lines.append("</instruction>")
    content: List[Dict] = [{"type": "text", "text": "\n".join(lines)}]
    if source == "explicit_text_frames" and frame_count > 0:
        content.append({"type": "text", "text": "<historical_visual_samples>"})
        for row in select_uniform(segment_rows, frame_count):
            probe.add_vision(content, row, frames_per_turn, "HISTORICAL sampled frame for summary only")
        content.append({"type": "text", "text": "</historical_visual_samples>"})
    return content


def generate_summary(
    args: argparse.Namespace,
    source: str,
    stream_messages: List[Dict],
    stream_outputs: List[Dict],
    segment_rows: List[Dict],
    frame_count: int,
    slot_count: int,
    start: int,
    end_exclusive: int,
) -> Dict:
    if source == "kv":
        messages = deepcopy(stream_messages)
        messages.append({"role": "user", "content": [{"type": "text", "text": "\n".join([
            "<active_query>",
            (
                "  <q>From all previous streaming observations in this conversation, produce compact "
                "global_state_summary. Use low-anchor abstract state wording, not dense captions. "
                f"Output exactly {slot_count} entries covering t={start}-{end_exclusive - 1}; no extra entries. "
                "Output XML <global_state_summary><m t=\"start-end\">...</m></global_state_summary>.</q>"
            ),
            "</active_query>",
        ])}]})
    elif source == "kv_explicit_text":
        messages = deepcopy(stream_messages)
        content = build_summary_prompt(
            source,
            stream_outputs,
            segment_rows,
            0,
            slot_count,
            start,
            end_exclusive,
            args.frames_per_turn,
        )
        content.insert(0, {
            "type": "text",
            "text": (
                "Use the previous streaming conversation as context, but use the explicit "
                "<historical_think> time tags below as the authoritative index. Compress them into "
                "compact global state memory without copying dense captions."
            ),
        })
        messages.append({"role": "user", "content": content})
    else:
        messages = [{
            "role": "system",
            "content": (
                "You are a precise video memory compressor. Use explicit historical think text and optional "
                "sampled frames to produce compact state memory with accurate time tags."
            ),
        }, {
            "role": "user",
            "content": build_summary_prompt(
                source,
                stream_outputs,
                segment_rows,
                frame_count,
                slot_count,
                start,
                end_exclusive,
                args.frames_per_turn,
            ),
        }]
    attempts = []
    response = base_probe.call_with_max_tokens(args, messages, args.summary_max_tokens)
    raw = response["choices"][0]["message"].get("content", "")
    summary_text = base_probe.parse_global_summary(raw)
    ok, expected_slots, got_slots = summary_matches_required_slots(summary_text, start, end_exclusive, slot_count)
    attempts.append({
        "raw": raw,
        "summary": summary_text,
        "usage": response.get("usage", {}),
        "valid_slots": ok,
        "expected_slots": expected_slots,
        "got_slots": got_slots,
    })
    retries = getattr(args, "summary_retries", 1)
    for _ in range(retries):
        if ok:
            break
        repair_messages = deepcopy(messages)
        repair_messages.append({"role": "assistant", "content": raw})
        repair_messages.append({"role": "user", "content": [{"type": "text", "text": "\n".join([
            "<repair_instruction>",
            "The previous summary is INVALID.",
            f"Expected t slots exactly: {', '.join(expected_slots)}",
            f"Got t slots: {', '.join(got_slots) if got_slots else 'none'}",
            "Rewrite the summary using exactly the expected slots, one <m> per slot, in order.",
            "Use the original historical observations already provided above. Do not add explanations.",
            "</repair_instruction>",
        ])}]})
        response = base_probe.call_with_max_tokens(args, repair_messages, args.summary_max_tokens)
        raw = response["choices"][0]["message"].get("content", "")
        summary_text = base_probe.parse_global_summary(raw)
        ok, expected_slots, got_slots = summary_matches_required_slots(summary_text, start, end_exclusive, slot_count)
        attempts.append({
            "raw": raw,
            "summary": summary_text,
            "usage": response.get("usage", {}),
            "valid_slots": ok,
            "expected_slots": expected_slots,
            "got_slots": got_slots,
        })
    return {
        "source": source,
        "frame_count": frame_count,
        "slot_count": slot_count,
        "raw": raw,
        "summary": summary_text,
        "entries": base_probe.parse_summary_entries(summary_text),
        "usage": response.get("usage", {}),
        "valid_slots": ok,
        "expected_slots": expected_slots,
        "got_slots": got_slots,
        "attempts": attempts,
    }


def build_init_user(
    summary_text: str,
    recent_text: List[Dict],
    recent_visual_rows: List[Dict],
    frames_per_turn: int,
) -> List[Dict]:
    lines = [summary_text, "<recent_memory>"]
    if recent_text:
        for item in recent_text:
            text = item["pred_caption"].replace("\n", " ")[:420]
            lines.append(f'  <m t="{item["current_time"]}">Recent observation: {text}</m>')
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
            "  <q>Historical initialization only. Store this summary and recent context. Future "
            "current-caption turns must describe only <current_vision>. Historical visual memory is context only.</q>"
        ),
        "</historical_query>",
    ])})
    return content


def continue_after_prefill(args: argparse.Namespace, case: str, init_user: List[Dict], rows: List[Dict]) -> Dict:
    messages = [caption_system(), {"role": "user", "content": init_user}]
    if args.init_assistant_mode == "empty":
        messages.append({"role": "assistant", "content": ""})
    elif args.init_assistant_mode == "loaded":
        messages.append({"role": "assistant", "content": "<memory_loaded/>"})
    outputs = []
    for idx, row in enumerate(rows):
        messages.append({
            "role": "user",
            "content": probe.build_stream_user(row, [], args.frames_per_turn, 0),
        })
        response = probe.call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        prev_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = probe.result_row(row, raw, prev_gold, len(messages), response.get("usage", {}))
        outputs.append(out)
        messages.append({"role": "assistant", "content": raw})
    captions = [r["pred_caption"] for r in outputs]
    return {
        "case": case,
        "turns": len(outputs),
        "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in outputs) / len(outputs) if outputs else 0.0,
        "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in outputs) / len(outputs) if outputs else 0.0,
        "old_copy_risk_count": sum(1 for r in outputs if r["old_copy_risk"]),
        "unique_pred_count": len(set(captions)),
        "top_repeat_count": max((captions.count(c) for c in set(captions)), default=0),
        "results": outputs,
    }


def run_case(args: argparse.Namespace, video_id: str, video_rows: Dict[int, Dict], segment_len: int) -> Optional[Dict]:
    first_start = 0
    first_end = segment_len
    second_start = segment_len
    second_end = segment_len + args.continue_len
    try:
        first_rows = contiguous_rows(video_rows, first_start, first_end)
        second_rows = contiguous_rows(video_rows, second_start, second_end)
    except ValueError as exc:
        return {"video_id": video_id, "segment_len": segment_len, "skipped": str(exc)}

    print(json.dumps({"stage": "case_start", "video_id": video_id, "segment_len": segment_len}, ensure_ascii=False), flush=True)
    stream_outputs, stream_messages = stream_segment(args, first_rows)
    qa_gen = generate_qa_set(args, stream_messages, first_start, first_end)
    first_kv_qa = answer_qa_from_messages(args, "first_segment_kv", deepcopy(stream_messages), qa_gen["questions"])

    recent_text = stream_outputs[-args.recent_text_sec:] if args.recent_text_sec > 0 else []
    recent_visual_times = set(range(first_end - args.recent_visual_sec, first_end)) if args.recent_visual_sec > 0 else set()
    recent_visual_rows = [row for row in first_rows if row["current_time"] in recent_visual_times]

    summaries = []
    continue_cases = []
    reqa_cases = []
    for source in args.summary_sources:
        frame_counts = args.summary_frame_counts if source == "explicit_text_frames" else [0]
        for frame_count in frame_counts:
            for slot_count in args.summary_slot_counts:
                summary = generate_summary(
                    args,
                    source,
                    stream_messages,
                    stream_outputs,
                    first_rows,
                    frame_count,
                    slot_count,
                    first_start,
                    first_end,
                )
                summaries.append(summary)
                variants = {
                    "summary_only": ([], []),
                    f"summary_recent_text{args.recent_text_sec}": (recent_text, []),
                }
                if args.recent_visual_sec > 0:
                    variants[f"summary_recent_text{args.recent_text_sec}_visual{args.recent_visual_sec}"] = (
                        recent_text,
                        recent_visual_rows,
                    )
                for variant, (text_items, visual_items) in variants.items():
                    case = f"{source}_f{frame_count}_s{slot_count}_{variant}"
                    init_user = build_init_user(summary["summary"], text_items, visual_items, args.frames_per_turn)
                    continue_cases.append(continue_after_prefill(args, case, init_user, second_rows))
                    init_messages = [memory_system(), {"role": "user", "content": init_user}]
                    if args.init_assistant_mode == "empty":
                        init_messages.append({"role": "assistant", "content": ""})
                    elif args.init_assistant_mode == "loaded":
                        init_messages.append({"role": "assistant", "content": "<memory_loaded/>"})
                    reqa_cases.append(answer_qa_from_messages(args, case, init_messages, qa_gen["questions"]))

    return {
        "video_id": video_id,
        "segment_len": segment_len,
        "continue_len": args.continue_len,
        "first_segment": {
            "turns": len(stream_outputs),
            "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in stream_outputs) / len(stream_outputs),
            "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in stream_outputs) / len(stream_outputs),
            "old_copy_risk_count": sum(1 for r in stream_outputs if r["old_copy_risk"]),
        },
        "qa_generated": qa_gen,
        "first_segment_kv_qa": first_kv_qa,
        "summaries": summaries,
        "continue_cases": continue_cases,
        "re_prefill_qa_cases": reqa_cases,
    }


def parse_csv_ints(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--model", default="qwen3vl2b-prefill-matrix")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--video-ids", default="surveil_112,HxfwLkoj2gs,Making_Vanilla_Banana_Bread")
    parser.add_argument("--segment-lens", default="60,90,120")
    parser.add_argument("--continue-len", type=int, default=30)
    parser.add_argument("--summary-sources", default="explicit_text,explicit_text_frames,kv")
    parser.add_argument("--summary-frame-counts", default="4,8")
    parser.add_argument("--summary-slot-counts", default="4,6")
    parser.add_argument("--recent-text-sec", type=int, default=8)
    parser.add_argument("--recent-visual-sec", type=int, default=8)
    parser.add_argument("--init-assistant-mode", choices=["none", "empty", "loaded"], default="none")
    parser.add_argument("--qa-count", type=int, default=4)
    parser.add_argument("--frames-per-turn", type=int, default=1)
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--summary-max-tokens", type=int, default=360)
    parser.add_argument("--summary-retries", type=int, default=1)
    parser.add_argument("--qa-gen-max-tokens", type=int, default=420)
    parser.add_argument("--qa-answer-max-tokens", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=100000)
    args = parser.parse_args()

    args.summary_sources = parse_csv(args.summary_sources)
    args.summary_frame_counts = parse_csv_ints(args.summary_frame_counts)
    args.summary_slot_counts = parse_csv_ints(args.summary_slot_counts)
    video_ids = parse_csv(args.video_ids)
    segment_lens = parse_csv_ints(args.segment_lens)

    probe.wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    all_rows = rows_by_video(Path(args.jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video_id in video_ids:
        if video_id not in all_rows:
            results.append({"video_id": video_id, "skipped": "not found"})
            continue
        for segment_len in segment_lens:
            result = run_case(args, video_id, all_rows[video_id], segment_len)
            results.append(result)
            safe_video = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)
            (out_dir / f"{safe_video}_seg{segment_len}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"args": vars(args), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "cases": [
            {
                "video_id": r.get("video_id"),
                "segment_len": r.get("segment_len"),
                "skipped": r.get("skipped"),
                "qa_count": len((r.get("qa_generated") or {}).get("questions", [])) if r else 0,
                "num_summaries": len(r.get("summaries", [])) if r and not r.get("skipped") else 0,
                "num_continue_cases": len(r.get("continue_cases", [])) if r and not r.get("skipped") else 0,
            }
            for r in results if r
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
