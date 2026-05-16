#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinkstream.data.schema import COMPACT_MEMORY_SYSTEM_PROMPT


MEM_RE = re.compile(r"(?is)^\s*<MEM>.*</MEM>\s*$")
M_RE = re.compile(r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>', re.I | re.S)
C_RE = re.compile(r'<c\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</c>', re.I | re.S)
MEM_BLOCK_RE = re.compile(r"(?is)<MEM>.*?</MEM>")
OLD_TAIL_RE = re.compile(
    r"(?is)\n+Covered latest span:.*?Return NEW_MEMORY\.\s*$"
)


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def parse_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for match in M_RE.finditer(text or ""):
        a = int(match.group(1))
        b = int(match.group(2) or a)
        if b < a:
            a, b = b, a
        ranges.append((a, b))
    return ranges


def parse_caption_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for match in C_RE.finditer(text or ""):
        a = int(match.group(1))
        b = int(match.group(2) or a)
        if b < a:
            a, b = b, a
        ranges.append((a, b))
    return ranges


def bare_m_only(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    cleaned = M_RE.sub("", stripped)
    return cleaned.strip() == "" and "<MEM" not in stripped and "</MEM>" not in stripped


def replace_old_memory(user_text: str, new_mem: str) -> str:
    if not new_mem.strip():
        return user_text
    match = MEM_BLOCK_RE.search(user_text)
    if not match:
        return user_text
    return user_text[: match.start()] + new_mem.strip() + user_text[match.end() :]


def sanitize_user_text(user_text: str) -> str:
    """Remove old conflicting output instructions from historical SFT rows."""
    text = OLD_TAIL_RE.sub("", user_text).rstrip()
    return (
        text
        + "\n\nReturn only 4-6 chronological <m t=\"...\">...</m> lines. "
        + "Do not output NEW_MEMORY, markdown, prose, analysis, or any text outside the <m> lines."
    )


def load_compact_rows(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            if (
                row.get("trajectory_type") == "compact_memory_update"
                or row.get("sample_type") == "compress"
                or row.get("loss_class") == "compress"
            ):
                row["_line_no"] = line_no
                rows.append(row)
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def choose_sequences(rows: list[dict[str, Any]], *, nseq: int, rounds: int) -> list[list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row.get("video_id") or "")].append(row)
    candidates = []
    for video_id, items in by_video.items():
        if len(items) >= rounds:
            items = sorted(items, key=lambda r: int(r.get("chunk_idx") or r.get("chunk_start") or r["_line_no"]))
            candidates.append((video_id, items[:rounds]))
    candidates.sort(key=lambda x: x[0])
    return [items for _, items in candidates[:nseq]]


@torch.inference_mode()
def generate(model, processor, user_text: str, max_new_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": COMPACT_MEMORY_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
    start = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new_ids = out[:, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "new_tokens": int(new_ids.shape[-1]),
        "latency_sec": round(time.time() - start, 3),
        "text": text,
    }


def load_model(model_path: str, device: str, attn: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=attn,
    ).eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--nseq", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scan-rows", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sanitize-user", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_compact_rows(args.jsonl, max_rows=args.scan_rows)
    sequences = choose_sequences(rows, nseq=args.nseq, rounds=args.rounds)
    if not sequences:
        raise RuntimeError("no compact row sequences found")

    model, processor = load_model(args.model, args.device, args.attn_implementation)
    results = []
    for seq_idx, seq in enumerate(sequences):
        previous_generated = ""
        seq_result = {
            "seq_idx": seq_idx,
            "video_id": seq[0].get("video_id"),
            "rounds": [],
        }
        carried_previous_ranges: list[tuple[int, int]] = []
        for round_idx, row in enumerate(seq):
            user_text = text_from_content((row.get("messages") or [None, {"content": ""}])[1].get("content"))
            if args.sanitize_user:
                user_text = sanitize_user_text(user_text)
            if previous_generated:
                user_text = replace_old_memory(user_text, previous_generated)
            gold = ""
            for message in row.get("messages") or []:
                if message.get("role") == "assistant":
                    gold = str(message.get("content") or "")
            old_ranges = parse_ranges(user_text.split("NEW_CAPTIONS:", 1)[0])
            new_ranges = parse_caption_ranges(user_text.split("NEW_CAPTIONS:", 1)[-1])
            gen = generate(model, processor, user_text, args.max_new_tokens)
            text = gen["text"]
            ranges = parse_ranges(text)
            valid_mem = bool(MEM_RE.match(text))
            valid_bare_m = bare_m_only(text) and 4 <= len(ranges) <= 6
            m_count = len(ranges)
            covers_old = bool(old_ranges) and any(a <= b2 and b >= a2 for a, b in ranges for a2, b2 in old_ranges)
            covers_new = bool(new_ranges) and any(a <= b2 and b >= a2 for a, b in ranges for a2, b2 in new_ranges)
            carries_previous_generated = (
                not carried_previous_ranges
                or any(
                    a <= b2 and b >= a2
                    for a, b in ranges
                    for a2, b2 in carried_previous_ranges
                )
            )
            seq_result["rounds"].append({
                "line_no": row.get("_line_no"),
                "chunk_idx": row.get("chunk_idx"),
                "gold_valid_mem": bool(MEM_RE.match(gold)),
                "gold_m_count": len(parse_ranges(gold)),
                "valid_mem": valid_mem,
                "valid_bare_m": valid_bare_m,
                "m_count": m_count,
                "m_count_4_6": 4 <= m_count <= 6,
                "covers_old": covers_old,
                "covers_new": covers_new,
                "carries_previous_generated": carries_previous_generated,
                "old_ranges": old_ranges[:12],
                "new_ranges": new_ranges[:12],
                "out_ranges": ranges[:12],
                **gen,
            })
            previous_generated = text
            carried_previous_ranges = ranges
        results.append(seq_result)

    summary = {
        "total_sequences": len(results),
        "total_rounds": sum(len(r["rounds"]) for r in results),
        "valid_mem": sum(x["valid_mem"] for r in results for x in r["rounds"]),
        "valid_bare_m": sum(x["valid_bare_m"] for r in results for x in r["rounds"]),
        "m_count_4_6": sum(x["m_count_4_6"] for r in results for x in r["rounds"]),
        "covers_old": sum(x["covers_old"] for r in results for x in r["rounds"]),
        "covers_new": sum(x["covers_new"] for r in results for x in r["rounds"]),
        "carries_previous_generated": sum(x["carries_previous_generated"] for r in results for x in r["rounds"]),
    }
    payload = {
        "model": args.model,
        "jsonl": str(args.jsonl),
        "prompt": COMPACT_MEMORY_SYSTEM_PROMPT,
        "sanitize_user": args.sanitize_user,
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
