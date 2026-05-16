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

from scripts.agent_data.pass5_messages import build_messages
from thinkstream.data.agent_protocol import tools_for_turn


def _load_rows(jsonl: Path, video_id: str, targets: list[tuple[str, int, str | None]]) -> dict[str, tuple[int, dict[str, Any]]]:
    rows: dict[str, tuple[int, dict[str, Any]]] = {}
    with jsonl.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            if row.get("video_id") != video_id:
                continue
            for name, chunk, sample_type in targets:
                if int(row.get("chunk_idx") or -1) != chunk:
                    continue
                if sample_type is not None and row.get("sample_type") != sample_type:
                    continue
                rows[name] = (line_no, row)
    return rows


def _video_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content") or []
        if isinstance(content, str):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            meta = item.get("video_metadata")
            if isinstance(meta, dict):
                out.append({k: v for k, v in meta.items() if k != "do_sample_frames"})
    return out


@torch.inference_mode()
def _generate(
    *,
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    line_no: int,
    name: str,
    data_root: Path,
    source_offset: int,
    stream_max_new_tokens: int,
    compress_max_new_tokens: int,
) -> dict[str, Any]:
    messages = build_messages(
        sample,
        data_root,
        data_dir=data_root,
        frame_protocol="video_meta",
    )
    prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
    is_compress = sample.get("sample_type") == "compress" or sample.get("action") == "compress"

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "do_sample_frames": False,
    }
    if not is_compress:
        template_kwargs["tools"] = tools_for_turn("streaming")
    vmeta = _video_metadata(prompt_messages)
    if vmeta:
        template_kwargs["video_metadata"] = vmeta

    inputs = processor.apply_chat_template(prompt_messages, **template_kwargs)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=compress_max_new_tokens if is_compress else stream_max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new_ids = output[:, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_ids[0], skip_special_tokens=False).strip()
    think_match = re.search(r"<think>(.*?)</think>", text, re.S)
    return {
        "name": name,
        "line_no": line_no,
        "video_id": sample.get("video_id"),
        "chunk_idx": sample.get("chunk_idx"),
        "source_time_start": source_offset + int(sample.get("chunk_idx") or 0),
        "sample_type": sample.get("sample_type"),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "new_tokens": int(new_ids.shape[1]),
        "latency_sec": round(time.time() - t0, 3),
        "gold": str(sample.get("output") or ""),
        "generated": text,
        "think": think_match.group(1).strip() if think_match else "",
        "m_lines": re.findall(r'<m\s+t="[^"]+"\s*>.*?</m>', text, flags=re.S | re.I),
        "has_old_silent": "<silent>" in text,
        "has_new_silence": "</Silence>" in text,
        "has_response": "</Response>" in text,
        "has_tool": "<tool_call>" in text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/qwen3vl8b-sft-b1_8_10-compact-noqa-v1-recallsidecar-len49k-r1")
    ap.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch8/final/val.jsonl"))
    ap.add_argument("--data-root", type=Path, default=Path("data/agent_v5/batch8"))
    ap.add_argument("--video-id", default="J6X6sbqEUHM_660.0_810.0")
    ap.add_argument("--source-offset", type=int, default=660)
    ap.add_argument("--stream-chunk", type=int, default=120)
    ap.add_argument("--compress-chunk", type=int, default=125)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stream-max-new-tokens", type=int, default=256)
    ap.add_argument("--compress-max-new-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("output/after500_prompt_probe/results.json"))
    args = ap.parse_args()

    targets = [
        ("stream_after500", args.stream_chunk, None),
        ("compress_after500", args.compress_chunk, "compress"),
    ]
    rows = _load_rows(args.jsonl, args.video_id, targets)
    missing = [name for name, _, _ in targets if name not in rows]
    if missing:
        raise RuntimeError(f"missing target rows: {missing}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    results = []
    for name, _, _ in targets:
        line_no, sample = rows[name]
        result = _generate(
            model=model,
            processor=processor,
            sample=sample,
            line_no=line_no,
            name=name,
            data_root=args.data_root,
            source_offset=args.source_offset,
            stream_max_new_tokens=args.stream_max_new_tokens,
            compress_max_new_tokens=args.compress_max_new_tokens,
        )
        results.append(result)
        print(f"--- {name} line={line_no} chunk={result['chunk_idx']} source_t={result['source_time_start']} ---")
        print(result["generated"][:2000])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
