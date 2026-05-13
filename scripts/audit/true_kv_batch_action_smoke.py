#!/usr/bin/env python3
"""Batched true-KV trajectory smoke for StreamingWindowInferenceEngine.

This exercises the local HF recurrent-KV path, not vLLM.  A fixed set of
trajectory slots is advanced one assistant turn at a time through a single
StreamingWindowInferenceEngine with batch_size > 1, so video-window eviction
and per-slot cache lengths are real.
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
from typing import Any, Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.local_hf_kv_visual_probe import (  # noqa: E402
    load_model_for_stream,
    make_engine,
    normalize_video_inputs,
)
from thinkstream.data.stream_data_processor import compute_position_ids  # noqa: E402
from thinkstream.models.agent_loop import _parse_agent_output  # noqa: E402


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
            elif str(frame).startswith("data/"):
                fixed.append(str((REPO_ROOT / frame_path).resolve()))
            else:
                fixed.append(str((frames_root / video_id / frame_path).resolve()))
        item["video"] = fixed
    return msg


def _assistant_pairs(messages: List[Dict[str, Any]], limit: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
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


def _load_trajectories(
    path: Path,
    *,
    batch_size: int,
    turns: int,
    skip: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matched = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("trajectory_type") not in {None, "from_start"}:
                continue
            pairs = _assistant_pairs(row.get("messages") or [], turns)
            if len(pairs) < turns:
                continue
            if matched < skip:
                matched += 1
                continue
            row["_assistant_pairs"] = pairs
            rows.append(row)
            if len(rows) >= batch_size:
                break
    if len(rows) < batch_size:
        raise RuntimeError(
            f"only found {len(rows)} trajectories with >= {turns} user->assistant turns in {path}"
        )
    return rows


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


def _gold_action(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        name = (((tool_calls[0] or {}).get("function") or {}).get("name") or "").strip()
        if name:
            return "recall" if name == "recall" else name
        return "tool_call"
    parsed = _parse_agent_output(str(message.get("content") or ""))
    return str(parsed.get("action") or "")


def _prepare_batch(
    *,
    processor: Any,
    trajectories: List[Dict[str, Any]],
    turn_no: int,
    frames_root: Path,
) -> Tuple[Dict[str, Any], List[int], List[str]]:
    texts: List[str] = []
    videos_all: List[Any] = []
    video_metadata_all: List[Dict[str, Any]] = []
    chunks: List[int] = []
    gold_actions: List[str] = []
    common_video_kwargs: Dict[str, Any] = {}

    for row in trajectories:
        messages = row["messages"]
        user_idx, assistant_idx = row["_assistant_pairs"][turn_no]
        video_id = str(row.get("video_id") or "")
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
        tools = row.get("tools") or None
        if tools:
            template_kwargs["tools"] = tools
        texts.append(processor.apply_chat_template(turn_messages, **template_kwargs))

        _, video_inputs, video_kwargs = process_vision_info(
            turn_messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        for key, value in (video_kwargs or {}).items():
            common_video_kwargs.setdefault(key, value)
        videos, video_metadata = normalize_video_inputs(video_inputs)
        if videos:
            videos_all.extend(videos)
        if video_metadata:
            video_metadata_all.extend(video_metadata)
        chunks.append(_user_chunk(messages[user_idx], turn_no))
        gold_actions.append(_gold_action(messages[assistant_idx]))

    proc_kwargs: Dict[str, Any] = {
        "text": texts,
        "videos": videos_all or None,
        "return_tensors": "pt",
        "padding": True,
        **common_video_kwargs,
    }
    if video_metadata_all:
        proc_kwargs["video_metadata"] = video_metadata_all
    inputs = processor(**proc_kwargs)
    inputs_for_rope = dict(inputs)
    inputs_for_rope["video_chunk_size"] = 1.0
    inputs["position_ids"] = compute_position_ids(inputs_for_rope, processor, "qwen3vl")
    return inputs, chunks, gold_actions


def _to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    keys = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values_videos",
        "video_grid_thw",
    ]
    return {
        key: (inputs.get(key).to(device) if inputs.get(key) is not None else None)
        for key in keys
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-type", default="qwen3vl")
    ap.add_argument("--pixel-profile", default="low")
    ap.add_argument("--max-len", type=int, default=49152)
    ap.add_argument("--kv-window", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--skip-trajectories", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    load_args = SimpleNamespace(
        model=args.model,
        model_type=args.model_type,
        pixel_profile=args.pixel_profile,
        max_len=args.max_len,
        kv_window=args.kv_window,
        batch_size=args.batch_size,
    )
    model, processor = load_model_for_stream(load_args, device)
    # Right padding is required for batched streaming prefill: cache length
    # rewind drops pad tokens from the tail, not from the left side.
    processor.tokenizer.padding_side = "right"
    engine = make_engine(load_args, model, processor, device)

    trajectories = _load_trajectories(
        Path(args.jsonl),
        batch_size=args.batch_size,
        turns=args.turns,
        skip=args.skip_trajectories,
    )
    frames_root = Path(args.frames_root)
    engine.reset()

    results: List[Dict[str, Any]] = []
    total_generated = 0
    total_decode_sec = 0.0
    for turn_no in range(args.turns):
        inputs, chunks, gold_actions = _prepare_batch(
            processor=processor,
            trajectories=trajectories,
            turn_no=turn_no,
            frames_root=frames_root,
        )
        keep = _to_device(inputs, device)
        t0 = time.time()
        generated_list = engine.generate(
            **keep,
            max_new_tokens=args.max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )
        latency = time.time() - t0
        gen_tokens = sum(int(t.numel()) for t in generated_list)
        total_generated += gen_tokens
        total_decode_sec += latency
        step_rows: List[Dict[str, Any]] = []
        for slot, generated in enumerate(generated_list):
            pred_text = processor.tokenizer.decode(
                generated.tolist(),
                skip_special_tokens=False,
            ).strip()
            pred = _parse_agent_output(pred_text)
            pred_action = str(pred.get("action") or "")
            step_rows.append({
                "slot": slot,
                "video_id": trajectories[slot].get("video_id"),
                "turn": turn_no,
                "chunk": chunks[slot],
                "prompt_tokens_padded": int(inputs["input_ids"].shape[-1]),
                "prompt_tokens_valid": int(inputs["attention_mask"][slot].sum().item()),
                "generated_tokens": int(generated.numel()),
                "gold_action": gold_actions[slot],
                "pred_action": pred_action,
                "action_match": pred_action == gold_actions[slot],
                "format_error": pred.get("format_error", ""),
                "cache_len": int(engine.decoder.cache.cache_seqlens[0, slot].item()),
                "window_count": int(engine._window_count[slot].item()),
                "window_starts": engine._window_starts[
                    slot, : int(engine._window_count[slot].item())
                ].detach().cpu().tolist(),
                "window_ends": engine._window_ends[
                    slot, : int(engine._window_count[slot].item())
                ].detach().cpu().tolist(),
                "pred_text": pred_text[:800],
            })
        results.append({
            "turn": turn_no,
            "latency_sec": latency,
            "generated_tokens": gen_tokens,
            "tokens_per_sec": float(gen_tokens) / max(latency, 1e-6),
            "rows": step_rows,
        })
        print(json.dumps({
            "turn": turn_no,
            "latency_sec": round(latency, 4),
            "generated_tokens": gen_tokens,
            "tokens_per_sec": round(float(gen_tokens) / max(latency, 1e-6), 2),
            "action_match": f"{sum(r['action_match'] for r in step_rows)}/{len(step_rows)}",
            "window_counts": [r["window_count"] for r in step_rows],
            "cache_lens": [r["cache_len"] for r in step_rows],
        }, ensure_ascii=False), flush=True)

    flat = [r for step in results for r in step["rows"]]
    summary = {
        "mode": "true_kv_batch_action_smoke",
        "checkpoint": args.model,
        "source": args.jsonl,
        "batch_size": args.batch_size,
        "turns": args.turns,
        "kv_window": args.kv_window,
        "max_len": args.max_len,
        "max_new_tokens": args.max_new_tokens,
        "videos": [r.get("video_id") for r in trajectories],
        "action_match_rate": sum(r["action_match"] for r in flat) / max(1, len(flat)),
        "format_error_count": sum(1 for r in flat if r["format_error"]),
        "total_generated_tokens": total_generated,
        "total_decode_sec": total_decode_sec,
        "overall_tokens_per_sec": float(total_generated) / max(total_decode_sec, 1e-6),
        "avg_step_latency_sec": total_decode_sec / max(1, len(results)),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
