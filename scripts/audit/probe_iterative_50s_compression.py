#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinkstream.data.schema import COMPACT_MEMORY_SYSTEM_PROMPT


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
M_RE = re.compile(r'<m\s+t="[^"]+"\s*>.*?</m>', re.S | re.I)


def _extract_think(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _load_thinks(jsonl: Path, video_id: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("video_id") != video_id:
                continue
            try:
                chunk = int(row.get("chunk_idx"))
            except Exception:
                continue
            think = _extract_think(str(row.get("output") or ""))
            if think and chunk not in out:
                out[chunk] = think
    return out


def _build_user_text(memory: str, captions: list[tuple[int, str]]) -> str:
    old_memory = memory.strip() or "(empty)"
    cap_lines = [
        f'<c t="{chunk}-{chunk + 1}">{text}</c>'
        for chunk, text in captions
        if text.strip()
    ]
    new_captions = "\n".join(cap_lines) if cap_lines else "(empty)"
    return (
        "OLD_MEMORY:\n"
        f"{old_memory}\n\n"
        "NEW_CAPTIONS:\n"
        f"{new_captions}\n\n"
        'Return only 4-6 chronological <m t="...">...</m> lines. '
        "Do not output NEW_MEMORY, Markdown, prose, analysis, or any text outside the <m> lines."
    )


@torch.inference_mode()
def _generate_compact(
    *,
    model: Any,
    processor: Any,
    user_text: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": COMPACT_MEMORY_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new_ids = output[:, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_ids[0], skip_special_tokens=False).strip()
    m_lines = M_RE.findall(text)
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "new_tokens": int(new_ids.shape[1]),
        "latency_sec": round(time.time() - t0, 3),
        "text": text,
        "m_lines": m_lines,
        "m_count": len(m_lines),
        "valid_4_6": 4 <= len(m_lines) <= 6 and "".join(m_lines).strip() == text.replace("<|im_end|>", "").strip(),
        "hit_cap": int(new_ids.shape[1]) >= max_new_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/qwen3vl8b-sft-b1_8_10-compact-noqa-v1-recallsidecar-len49k-r1")
    ap.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch2/final/train_rl.jsonl"))
    ap.add_argument("--video-id", default="v_Nb87GFizCB8")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--segment-sec", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-seconds", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--out", type=Path, default=Path("output/iter50_compress_probe/results.json"))
    args = ap.parse_args()

    thinks = _load_thinks(args.jsonl, args.video_id)
    if not thinks:
        raise RuntimeError("no think rows found")
    max_chunk = max(thinks)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    memory = ""
    rounds = []
    for round_i in range(args.rounds):
        start = round_i * args.segment_sec
        end = min(start + args.segment_sec, args.max_seconds, max_chunk + 1)
        captions = [(i, thinks[i]) for i in range(start, end) if i in thinks]
        user_text = _build_user_text(memory, captions)
        gen = _generate_compact(
            model=model,
            processor=processor,
            user_text=user_text,
            max_new_tokens=args.max_new_tokens,
        )
        if gen["m_lines"]:
            memory = "\n".join(gen["m_lines"])
        else:
            memory = gen["text"]
        rec = {
            "round": round_i + 1,
            "window": [start, end],
            "caption_count": len(captions),
            "memory_chars_after": len(memory),
            **gen,
        }
        rounds.append(rec)
        print(
            f"round={round_i + 1} window={start}-{end} captions={len(captions)} "
            f"m={gen['m_count']} valid={gen['valid_4_6']} hit_cap={gen['hit_cap']} "
            f"mem_chars={len(memory)}"
        )

    result = {
        "jsonl": str(args.jsonl),
        "video_id": args.video_id,
        "available_chunks": [min(thinks), max_chunk],
        "segment_sec": args.segment_sec,
        "rounds_requested": args.rounds,
        "max_seconds": args.max_seconds,
        "rounds": rounds,
        "final_memory": memory,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
