#!/usr/bin/env python3
"""Run a small pass5 trajectory through StreamingWindowInferenceEngine.

This is a local HF smoke path for debugging recurrent KV behavior. It is not a
vLLM audit: each turn feeds only the new user chunk and keeps generated
assistant tokens in the same streaming KV cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.local_hf_kv_visual_probe import (
    load_model_for_stream,
    make_engine,
    normalize_video_inputs,
)
from thinkstream.data.stream_data_processor import compute_position_ids
from thinkstream.models.agent_loop import _parse_agent_output


def _resolve_msg(msg: Dict[str, Any], *, frames_root: Path, video_id: str) -> Dict[str, Any]:
    msg = copy.deepcopy(msg)
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    for item in content:
        if not (isinstance(item, dict) and item.get("type") == "video"):
            continue
        video = item.get("video")
        if not isinstance(video, list):
            continue
        fixed = []
        for frame in video:
            frame_path = Path(str(frame))
            if frame_path.is_absolute():
                fixed.append(str(frame_path))
            else:
                fixed.append(str((frames_root / video_id / frame_path).resolve()))
        item["video"] = fixed
    return msg


def _load_first_trajectory(path: Path, min_messages: int) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("trajectory_type") == "from_start" and len(row.get("messages") or []) >= min_messages:
                return row
    raise RuntimeError(f"no from_start trajectory with >= {min_messages} messages in {path}")


def _user_chunk(message: Dict[str, Any], default: int) -> int:
    content = message.get("content")
    if isinstance(content, list):
        text = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = str(content)
    match = re.search(r"<t=(\d+)>", text)
    return int(match.group(1)) if match else default


def _assistant_pairs(messages: List[Dict[str, Any]], limit: int) -> List[tuple[int, int]]:
    pairs: List[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "system":
            i += 1
            continue
        if (
            messages[i].get("role") == "user"
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "assistant"
        ):
            pairs.append((i, i + 1))
            i += 2
            if len(pairs) >= limit:
                break
        else:
            i += 1
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-type", default="qwen3vl")
    ap.add_argument("--pixel-profile", default="low")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-len", type=int, default=49152)
    ap.add_argument("--kv-window", type=int, default=8)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    load_args = SimpleNamespace(
        model=args.model,
        model_type=args.model_type,
        pixel_profile=args.pixel_profile,
        max_len=args.max_len,
        kv_window=args.kv_window,
    )
    model, processor = load_model_for_stream(load_args, device)
    engine = make_engine(load_args, model, processor, device)

    row = _load_first_trajectory(Path(args.jsonl), min_messages=args.turns * 2)
    messages = row["messages"]
    video_id = str(row.get("video_id") or "")
    tools = row.get("tools") or None
    pairs = _assistant_pairs(messages, args.turns)

    engine.reset()
    results: List[Dict[str, Any]] = []
    frames_root = Path(args.frames_root)
    for turn_no, (user_idx, assistant_idx) in enumerate(pairs):
        if turn_no == 0 and messages and messages[0].get("role") == "system":
            turn_messages = [
                _resolve_msg(messages[0], frames_root=frames_root, video_id=video_id),
                _resolve_msg(messages[user_idx], frames_root=frames_root, video_id=video_id),
            ]
        else:
            turn_messages = [
                _resolve_msg(messages[user_idx], frames_root=frames_root, video_id=video_id)
            ]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        text = processor.apply_chat_template(turn_messages, **template_kwargs)
        _, video_inputs, video_kwargs = process_vision_info(
            turn_messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadata = normalize_video_inputs(video_inputs)
        proc_kwargs = {
            "text": [text],
            "videos": videos,
            "return_tensors": "pt",
            **(video_kwargs or {}),
        }
        if video_metadata is not None:
            proc_kwargs["video_metadata"] = video_metadata
        inputs = processor(**proc_kwargs)
        inputs_for_rope = dict(inputs)
        inputs_for_rope["video_chunk_size"] = 1.0
        inputs["position_ids"] = compute_position_ids(
            inputs_for_rope,
            processor,
            args.model_type,
        )
        keep = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": (
                inputs.get("attention_mask").to(device)
                if inputs.get("attention_mask") is not None
                else None
            ),
            "position_ids": inputs["position_ids"].to(device),
            "pixel_values_videos": (
                inputs.get("pixel_values_videos").to(device)
                if inputs.get("pixel_values_videos") is not None
                else None
            ),
            "video_grid_thw": (
                inputs.get("video_grid_thw").to(device)
                if inputs.get("video_grid_thw") is not None
                else None
            ),
        }

        t0 = time.time()
        generated = engine.generate(
            **keep,
            max_new_tokens=args.max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )[0]
        latency = time.time() - t0
        pred_text = processor.tokenizer.decode(
            generated.tolist(),
            skip_special_tokens=False,
        ).strip()
        gold_text = messages[assistant_idx].get("content") or ""
        pred = _parse_agent_output(pred_text)
        gold = _parse_agent_output(gold_text)
        results.append({
            "turn": turn_no,
            "chunk": _user_chunk(messages[user_idx], turn_no),
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": int(generated.numel()),
            "latency_sec": latency,
            "tokens_per_sec": float(generated.numel()) / max(latency, 1e-6),
            "gold_action": gold.get("action"),
            "pred_action": pred.get("action"),
            "action_match": pred.get("action") == gold.get("action"),
            "format_error": pred.get("format_error", ""),
            "cache_len": int(engine.decoder.cache.cache_seqlens[0, 0].item()),
            "window_count": int(engine._window_count[0].item()),
            "window_starts": engine._window_starts[
                0, : int(engine._window_count[0].item())
            ].detach().cpu().tolist(),
            "window_ends": engine._window_ends[
                0, : int(engine._window_count[0].item())
            ].detach().cpu().tolist(),
            "pred_text": pred_text[:1000],
            "gold_text": gold_text[:500],
        })

    summary = {
        "mode": "true_kv_action_smoke",
        "checkpoint": args.model,
        "source": args.jsonl,
        "video_id": video_id,
        "trajectory_type": row.get("trajectory_type"),
        "turns": len(results),
        "action_match_rate": sum(r["action_match"] for r in results) / max(1, len(results)),
        "format_error_count": sum(1 for r in results if r["format_error"]),
        "avg_latency_sec": sum(r["latency_sec"] for r in results) / max(1, len(results)),
        "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in results) / max(1, len(results)),
        "total_generated_tokens": sum(r["generated_tokens"] for r in results),
        "total_decode_sec": sum(r["latency_sec"] for r in results),
        "last_cache_len": results[-1]["cache_len"] if results else None,
        "last_window_count": results[-1]["window_count"] if results else None,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    for row in results:
        keys = [
            "turn",
            "chunk",
            "gold_action",
            "pred_action",
            "action_match",
            "format_error",
            "generated_tokens",
            "latency_sec",
            "tokens_per_sec",
            "cache_len",
            "window_count",
            "window_starts",
            "window_ends",
        ]
        print(json.dumps({k: row[k] for k in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
