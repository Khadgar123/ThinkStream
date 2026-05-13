#!/usr/bin/env python3
"""Probe iterative text-memory compression with a base Qwen3-VL model.

This intentionally tests only the text compression step:
segment 0 captions -> summary_0, then summary_0 + segment 1 captions
-> summary_0_1.  The output is meant to show whether a bounded 4-6 entry
summary can preserve enough history for the next segment.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "also", "with", "without",
    "there", "their", "these", "those", "this", "that", "from", "into", "onto", "over",
    "under", "while", "where", "which", "being", "been", "have", "has", "had", "does",
    "display", "displays", "show", "shows", "shown", "scene", "video", "frame", "frames",
    "current", "visible", "appears", "appearing", "using", "wearing", "holding", "object",
    "person", "people", "image", "close", "view", "background", "foreground", "left",
    "right", "center", "central", "likely", "text", "visible", "caption", "chunk",
    "seconds", "second", "observation", "observations",
}


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def assistant_think(row: Dict[str, Any]) -> str:
    text = text_from_content(row["messages"][-1].get("content", ""))
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def visual_window_and_video(row: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    user_content = row["messages"][1]["content"]
    text = text_from_content(user_content)
    match = re.search(r"<visual_window>\s*(\{.*?\})\s*</visual_window>", text, flags=re.DOTALL)
    visual_window = json.loads(match.group(1)) if match else {}
    video = []
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "video":
            video = item.get("video") or []
            break
    return visual_window, video


def iter_caption_rows(path: Path, video_id: str, start: int, end: int) -> Iterable[Dict[str, Any]]:
    by_time: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if row.get("video_id") != video_id:
                continue
            if row.get("sample_type") not in {"silent", "response"}:
                continue
            visual_window, video = visual_window_and_video(row)
            current_time = visual_window.get("current_time")
            if current_time is None or not video:
                continue
            current_time = int(current_time)
            if not (start <= current_time <= end):
                continue
            if current_time in by_time:
                continue
            caption = assistant_think(row)
            if not caption:
                continue
            by_time[current_time] = {
                "line_no": line_no,
                "video_id": video_id,
                "t": current_time,
                "caption": caption,
                "sample_type": row.get("sample_type"),
                "chunk_idx": row.get("chunk_idx"),
            }
    missing = [t for t in range(start, end + 1) if t not in by_time]
    if missing:
        raise RuntimeError(f"missing captions for times: {missing[:40]}")
    for t in range(start, end + 1):
        yield by_time[t]


def keywords(text: str, limit: int = 160) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text.lower())
    out: List[str] = []
    seen = set()
    for word in words:
        word = word.strip("'")
        if word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= limit:
            break
    return out


def keyword_recall(prediction: str, references: List[str]) -> Dict[str, Any]:
    ref_words = keywords("\n".join(references), limit=300)
    if not ref_words:
        return {"recall": 0.0, "hits": [], "misses": []}
    pred_words = set(keywords(prediction, limit=300))
    hits = [w for w in ref_words if w in pred_words]
    misses = [w for w in ref_words if w not in pred_words]
    return {
        "recall": len(hits) / len(ref_words),
        "hits": hits[:80],
        "misses": misses[:80],
        "ref_keyword_count": len(ref_words),
    }


def format_caption_block(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(f'  <m t="{r["t"]}">{r["caption"]}</m>' for r in rows)


def build_messages(previous_summary: str, rows: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, str]]:
    previous = previous_summary.strip() or '  <m t="none">No prior summary.</m>'
    prompt = f"""Instruction:
Compress these streaming observations into one bounded historical state summary.
Keep only task progress, event changes, object state, and useful time ranges.
Merge adjacent seconds that describe the same event.
Preserve important earlier events from <previous_summary> if they remain useful.
Do not copy every caption. Do not describe clothing/background unless essential.
Do not predict future actions. Do not answer any question.
Output exactly 4-6 entries in this XML-like format:
<global_state_summary>
  <m t="a-b">short event/state.</m>
</global_state_summary>

<previous_summary>
{previous}
</previous_summary>

<new_observations t="{start}-{end}">
{format_caption_block(rows)}
</new_observations>"""
    return [
        {
            "role": "system",
            "content": "You compress video-observation memory for a streaming video agent.",
        },
        {"role": "user", "content": prompt},
    ]


def generate_summary(model, processor, messages: List[Dict[str, str]], max_new_tokens: int) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    latency = time.time() - t0
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "new_tokens": int(new_ids.shape[-1]),
        "latency_sec": latency,
        "output": output,
    }


def load_model(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
        ).eval()
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        ).eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl"))
    parser.add_argument("--video-id", default="HxfwLkoj2gs")
    parser.add_argument("--model", default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-sec", type=int, default=30)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("output/base_summary_iteration_probe/iter30_hxfw.json"))
    args = parser.parse_args()

    if args.segments < 2:
        raise ValueError("--segments must be >= 2")
    start = 0
    end = args.segment_sec * args.segments - 1
    rows = list(iter_caption_rows(args.jsonl, args.video_id, start, end))
    model, processor = load_model(args.model, args.device)

    summaries: List[Dict[str, Any]] = []
    previous_summary = ""
    for seg_idx in range(args.segments):
        seg_start = seg_idx * args.segment_sec
        seg_end = seg_start + args.segment_sec - 1
        seg_rows = [r for r in rows if seg_start <= r["t"] <= seg_end]
        messages = build_messages(previous_summary, seg_rows, seg_start, seg_end)
        gen = generate_summary(model, processor, messages, args.max_new_tokens)
        previous_summary = gen["output"]
        seen_rows = [r for r in rows if 0 <= r["t"] <= seg_end]
        summaries.append({
            "segment_index": seg_idx,
            "segment_range": [seg_start, seg_end],
            "input_caption_count": len(seg_rows),
            "previous_summary_input": messages[1]["content"].split("<previous_summary>", 1)[1].split("</previous_summary>", 1)[0].strip(),
            **gen,
            "coverage_all_seen": keyword_recall(gen["output"], [r["caption"] for r in seen_rows]),
            "coverage_current_segment": keyword_recall(gen["output"], [r["caption"] for r in seg_rows]),
        })

    result = {
        "video_id": args.video_id,
        "jsonl": str(args.jsonl),
        "model": args.model,
        "segment_sec": args.segment_sec,
        "segments": args.segments,
        "caption_rows": rows,
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "video_id": args.video_id,
        "summaries": [
            {
                "segment_range": s["segment_range"],
                "prompt_tokens": s["prompt_tokens"],
                "new_tokens": s["new_tokens"],
                "latency_sec": round(s["latency_sec"], 2),
                "coverage_all_seen": round(s["coverage_all_seen"]["recall"], 3),
                "coverage_current_segment": round(s["coverage_current_segment"]["recall"], 3),
                "output": s["output"],
            }
            for s in summaries
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
