#!/usr/bin/env python3
"""Local HF/StreamingWindow visual carrier audit.

This intentionally does not call vLLM/OpenAI-compatible APIs.  It has two
paths:

* processor: inspect local qwen-vl-utils/HF processor behavior for image
  items vs video frame lists at several frame counts and pixel budgets.
* stream: run a short multi-turn caption stream through ThinkStream's local
  StreamingWindowInferenceEngine so old video KV blocks are evicted by the
  same code path used by recurrent rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thinkstream.data.agent_protocol import infer_video_metadata
from thinkstream.data.stream_data_processor import compute_position_ids
from thinkstream.models import MODEL_CLS, get_text_config


PIXEL_PROFILES: Dict[str, Tuple[int, int]] = {
    "low": (32_768, 65_536),
    "mid": (65_536, 100_352),
    "legacy": (130_000, 220_000),
    "runtime": (256 * 28 * 28, 512 * 28 * 28),
}

STOPWORDS = {
    "about", "above", "after", "again", "along", "also", "with", "there",
    "their", "these", "those", "this", "that", "from", "into", "video",
    "frame", "frames", "current", "visible", "appears", "wearing", "holding",
    "person", "people", "image", "scene", "caption", "chunk", "background",
    "foreground", "left", "right", "center",
}


def iter_video_items(messages: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                yield item


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content if item.get("type") == "text"
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
    video: List[str] = []
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "video":
            video = list(item.get("video") or [])
            break
    return visual_window, video


def load_rows(path: Path, video_id: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
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
            if not (start_time <= current_time <= end_time):
                continue
            by_time[current_time] = {
                "line_no": line_no,
                "video_id": video_id,
                "current_time": current_time,
                "video": video,
                "gold_caption": assistant_think(row),
                "sample_type": row.get("sample_type"),
                "chunk_idx": row.get("chunk_idx"),
            }
    missing = [t for t in range(start_time, end_time + 1) if t not in by_time]
    if missing:
        raise RuntimeError(f"missing current_time values: {missing[:20]}")
    return [by_time[t] for t in sorted(by_time)]


def keywords(text: str, limit: int = 80) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", str(text).lower())
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


def keyword_recall(prediction: str, reference: str) -> float:
    ref = keywords(reference)
    if not ref:
        return 0.0
    pred = set(keywords(prediction, limit=160))
    return len([w for w in ref if w in pred]) / len(ref)


def set_processor_pixels(processor: Any, min_pixels: int, max_pixels: int) -> None:
    for proc in (getattr(processor, "image_processor", None), getattr(processor, "video_processor", None)):
        if proc is None:
            continue
        if hasattr(proc, "min_pixels"):
            proc.min_pixels = min_pixels
        if hasattr(proc, "max_pixels"):
            proc.max_pixels = max_pixels
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            proc.size["shortest_edge"] = min_pixels
            proc.size["longest_edge"] = max_pixels


def tensor_shape(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def normalize_video_inputs(video_inputs: Any) -> tuple[List[Any] | None, List[Dict[str, Any]] | None]:
    if video_inputs is None:
        return None, None
    videos = []
    metas = []
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            videos.append(item[0])
            metas.append(item[1])
        else:
            videos.append(item)
            metas.append({})
    return videos, metas


def processor_probe(args: argparse.Namespace) -> None:
    min_pixels, max_pixels = PIXEL_PROFILES[args.pixel_profile]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    set_processor_pixels(processor, min_pixels, max_pixels)
    rows = load_rows(Path(args.jsonl), args.video_id, args.start_time, args.start_time)
    frame_pool = rows[0]["video"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for frame_count in args.frame_counts:
            frames = frame_pool[-frame_count:]
            if len(frames) < frame_count:
                frames = (frame_pool * frame_count)[:frame_count]
            meta = infer_video_metadata(frames, fps=float(args.fps), total_num_frames=len(frames))
            cases: Dict[str, List[Dict[str, Any]]] = {}
            image_content: List[Dict[str, Any]] = []
            for frame in frames:
                image_content.append({"type": "image", "image": frame, "min_pixels": min_pixels, "max_pixels": max_pixels})
            cases["image_items_inline"] = [{"role": "user", "content": image_content + [{"type": "text", "text": "Caption current visual."}]}]
            cases["video_frame_list_inline"] = [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": frames,
                    "video_metadata": {**meta, "do_sample_frames": False},
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }, {"type": "text", "text": "Caption current visual."}],
            }]
            cases["video_frame_list_processor_only"] = [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": frames,
                    "video_metadata": {**meta, "do_sample_frames": False},
                }, {"type": "text", "text": "Caption current visual."}],
            }]

            for name, messages in cases.items():
                row: Dict[str, Any] = {
                    "mode": "processor",
                    "model": args.model,
                    "video_id": args.video_id,
                    "carrier": name,
                    "pixel_profile": args.pixel_profile,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "frame_count": frame_count,
                }
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                    videos, video_metadata = normalize_video_inputs(video_inputs)
                    proc_kwargs = dict(
                        text=[text],
                        images=image_inputs,
                        videos=videos,
                        return_tensors="pt",
                        **(video_kwargs or {}),
                    )
                    if video_metadata is not None:
                        proc_kwargs["video_metadata"] = video_metadata
                    encoded = processor(**proc_kwargs)
                    row.update({
                        "ok": True,
                        "input_tokens": int(encoded["input_ids"].shape[-1]),
                        "image_grid_thw": encoded.get("image_grid_thw").tolist() if encoded.get("image_grid_thw") is not None else None,
                        "video_grid_thw": encoded.get("video_grid_thw").tolist() if encoded.get("video_grid_thw") is not None else None,
                        "pixel_values_shape": tensor_shape(encoded.get("pixel_values")),
                        "pixel_values_videos_shape": tensor_shape(encoded.get("pixel_values_videos")),
                        "rendered_image_tokens": text.count("<|image_pad|>"),
                        "rendered_video_tokens": text.count("<|video_pad|>"),
                    })
                except Exception as exc:  # noqa: BLE001
                    row.update({"ok": False, "error": repr(exc)})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stream_prompt(
    processor: Any,
    frames: List[str],
    current_time: int,
    *,
    first: bool,
    system_prompt: str,
    fps: float,
    min_pixels: int,
    max_pixels: int,
    prompt_style: str,
) -> tuple[str, List[Dict[str, Any]]]:
    metadata = infer_video_metadata(frames, fps=fps, total_num_frames=max(2, len(frames)))
    video_item = {
        "type": "video",
        "video": frames,
        "video_metadata": {**metadata, "do_sample_frames": False},
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    query_text = (
        f'<query><q t="{current_time}">Caption only the newest current video chunk. '
        "Do not copy earlier captions. Output one concise English sentence.</q></query>"
    )
    if prompt_style == "schema":
        user_content = [
            {"type": "text", "text": f"<t={current_time}>"},
            video_item,
            {"type": "text", "text": "\n" + query_text},
        ]
    else:
        user_content = [
            video_item,
            {"type": "text", "text": "\n" + query_text},
        ]
    messages: List[Dict[str, Any]]
    if first:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    else:
        messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, user_content


def load_model_for_stream(args: argparse.Namespace, device: str):
    model = MODEL_CLS[args.model_type].from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    min_pixels, max_pixels = PIXEL_PROFILES[args.pixel_profile]
    set_processor_pixels(processor, min_pixels, max_pixels)
    model.config.text_config._attn_implementation = "flash_attention_2_infer"
    model.eval()
    return model, processor


def make_engine(args: argparse.Namespace, model: Any, processor: Any, device: str):
    from thinkstream.models.inference import StreamingWindowInferenceEngine

    text_config = get_text_config(model.config)
    num_heads = int(getattr(text_config, "num_attention_heads"))
    head_dim = int(getattr(text_config, "head_dim", getattr(text_config, "hidden_size") // num_heads))
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return StreamingWindowInferenceEngine(
        model=model,
        batch_size=int(getattr(args, "batch_size", 1)),
        max_len=args.max_len,
        num_hidden_layers=int(getattr(text_config, "num_hidden_layers")),
        num_key_value_heads=int(getattr(text_config, "num_key_value_heads", num_heads)),
        head_dim=head_dim,
        vocab_size=int(
            getattr(text_config, "vocab_size", None)
            or getattr(model.config, "vocab_size", None)
            or len(processor.tokenizer)
        ),
        pad_token_id=int(processor.tokenizer.pad_token_id or eos_id),
        eos_token_ids=[int(eos_id)],
        video_token_id=int(video_token_id),
        video_flex_window_size=int(args.kv_window),
        dtype=torch.bfloat16,
        device=device,
    )


def select_turn_frames(row: Dict[str, Any], frames_per_turn: int) -> List[str]:
    frames = list(row["video"][-max(1, frames_per_turn):])
    if frames_per_turn == 1:
        return frames[:1]
    if len(frames) == 1:
        frames = [frames[0], frames[0]]
    return frames


def stream_probe(args: argparse.Namespace) -> None:
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    min_pixels, max_pixels = PIXEL_PROFILES[args.pixel_profile]
    model, processor = load_model_for_stream(args, device)
    engine = make_engine(args, model, processor, device)
    system_prompt = (
        "You are a streaming video captioning agent. Each user turn contains only the newest current video chunk. "
        "Caption the current visual chunk, not historical KV."
    )

    def run_one_video(video_id: str) -> Dict[str, Any]:
        with torch.inference_mode():
            engine.reset()
        rows = load_rows(Path(args.jsonl), video_id, args.start_time, args.end_time)
        out_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            frames = select_turn_frames(row, args.frames_per_turn)
            if len(frames) == 1 and args.duplicate_single_video_frame:
                frames = [frames[0], frames[0]]
            text, user_content = build_stream_prompt(
                processor,
                frames,
                int(row["current_time"]),
                first=(idx == 0),
                system_prompt=system_prompt,
                fps=float(args.fps),
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                prompt_style=args.prompt_style,
            )
            media_messages = [{"role": "user", "content": user_content}]
            _, video_inputs, video_kwargs = process_vision_info(
                media_messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadata = normalize_video_inputs(video_inputs)
            proc_kwargs = dict(
                text=[text],
                videos=videos,
                return_tensors="pt",
                **(video_kwargs or {}),
            )
            if video_metadata is not None:
                proc_kwargs["video_metadata"] = video_metadata
            inputs = processor(**proc_kwargs)
            inputs_for_rope = dict(inputs)
            inputs_for_rope["video_chunk_size"] = 1.0
            inputs["position_ids"] = compute_position_ids(inputs_for_rope, processor, args.model_type)
            keep = {
                "input_ids": inputs["input_ids"].to(device),
                "attention_mask": inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
                "position_ids": inputs["position_ids"].to(device),
                "pixel_values_videos": inputs.get("pixel_values_videos").to(device) if inputs.get("pixel_values_videos") is not None else None,
                "video_grid_thw": inputs.get("video_grid_thw").to(device) if inputs.get("video_grid_thw") is not None else None,
            }
            t_gen = time.time()
            generated = engine.generate(
                **keep,
                max_new_tokens=args.max_new_tokens,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                repetition_penalty=1.0,
            )[0]
            latency_sec = time.time() - t_gen
            raw = processor.tokenizer.decode(generated.tolist(), skip_special_tokens=True).strip()
            current = keyword_recall(raw, row["gold_caption"])
            previous = keyword_recall(raw, rows[idx - 1]["gold_caption"]) if idx else 0.0
            out_rows.append({
                "mode": "stream",
                "model": args.model,
                "video_id": video_id,
                "pixel_profile": args.pixel_profile,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "frames_per_turn": args.frames_per_turn,
                "duplicate_single_video_frame": args.duplicate_single_video_frame,
                "prompt_style": args.prompt_style,
                "kv_window": args.kv_window,
                "current_time": row["current_time"],
                "gold_caption": row["gold_caption"],
                "pred_caption": raw,
                "current_keyword_recall": current,
                "previous_keyword_recall": previous,
                "old_copy_risk": bool(idx and previous > current + 0.10),
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "generated_tokens": int(generated.numel()),
                "latency_sec": latency_sec,
                "tokens_per_sec": float(generated.numel()) / max(latency_sec, 1e-6),
                "video_grid_thw": inputs.get("video_grid_thw").tolist() if inputs.get("video_grid_thw") is not None else None,
                "pixel_values_videos_shape": tensor_shape(inputs.get("pixel_values_videos")),
                "cache_len": int(engine.decoder.cache.cache_seqlens[0, 0].item()),
                "window_count": int(engine._window_count[0].item()),
                "window_starts": engine._window_starts[0, : int(engine._window_count[0].item())].detach().cpu().tolist(),
                "window_ends": engine._window_ends[0, : int(engine._window_count[0].item())].detach().cpu().tolist(),
            })
        return {
            "mode": "stream_summary",
            "model": args.model,
            "video_id": video_id,
            "pixel_profile": args.pixel_profile,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "frames_per_turn": args.frames_per_turn,
            "duplicate_single_video_frame": args.duplicate_single_video_frame,
            "prompt_style": args.prompt_style,
            "kv_window": args.kv_window,
            "turns": len(out_rows),
            "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in out_rows) / max(1, len(out_rows)),
            "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in out_rows) / max(1, len(out_rows)),
            "old_copy_risk_count": sum(1 for r in out_rows if r["old_copy_risk"]),
            "last_cache_len": out_rows[-1]["cache_len"] if out_rows else None,
            "results": out_rows,
        }

    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()] if args.video_ids else [args.video_id]
    if len(video_ids) > 1:
        per_video = [run_one_video(video_id) for video_id in video_ids]
        all_turns = [r for summary in per_video for r in summary["results"]]
        summary = {
            "mode": "stream_multivideo_summary",
            "model": args.model,
            "video_ids": video_ids,
            "pixel_profile": args.pixel_profile,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "frames_per_turn": args.frames_per_turn,
            "duplicate_single_video_frame": args.duplicate_single_video_frame,
            "prompt_style": args.prompt_style,
            "kv_window": args.kv_window,
            "videos": len(per_video),
            "turns": len(all_turns),
            "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in all_turns) / max(1, len(all_turns)),
            "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in all_turns) / max(1, len(all_turns)),
            "old_copy_risk_count": sum(1 for r in all_turns if r["old_copy_risk"]),
            "avg_last_cache_len": sum(s["last_cache_len"] or 0 for s in per_video) / max(1, len(per_video)),
            "per_video": per_video,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, ensure_ascii=False))
        return

    rows = load_rows(Path(args.jsonl), args.video_id, args.start_time, args.end_time)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        frames = select_turn_frames(row, args.frames_per_turn)
        if len(frames) == 1 and args.duplicate_single_video_frame:
            frames = [frames[0], frames[0]]
        text, user_content = build_stream_prompt(
            processor,
            frames,
            int(row["current_time"]),
            first=(idx == 0),
            system_prompt=system_prompt,
            fps=float(args.fps),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            prompt_style=args.prompt_style,
        )
        media_messages = [{"role": "user", "content": user_content}]
        _, video_inputs, video_kwargs = process_vision_info(
            media_messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadata = normalize_video_inputs(video_inputs)
        proc_kwargs = dict(
            text=[text],
            videos=videos,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        if video_metadata is not None:
            proc_kwargs["video_metadata"] = video_metadata
        inputs = processor(**proc_kwargs)
        inputs_for_rope = dict(inputs)
        inputs_for_rope["video_chunk_size"] = 1.0
        inputs["position_ids"] = compute_position_ids(inputs_for_rope, processor, args.model_type)
        keep = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
            "position_ids": inputs["position_ids"].to(device),
            "pixel_values_videos": inputs.get("pixel_values_videos").to(device) if inputs.get("pixel_values_videos") is not None else None,
            "video_grid_thw": inputs.get("video_grid_thw").to(device) if inputs.get("video_grid_thw") is not None else None,
        }
        t_gen = time.time()
        generated = engine.generate(
            **keep,
            max_new_tokens=args.max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )[0]
        latency_sec = time.time() - t_gen
        raw = processor.tokenizer.decode(generated.tolist(), skip_special_tokens=True).strip()
        current = keyword_recall(raw, row["gold_caption"])
        previous = keyword_recall(raw, rows[idx - 1]["gold_caption"]) if idx else 0.0
        out_rows.append({
            "mode": "stream",
            "model": args.model,
            "video_id": args.video_id,
            "pixel_profile": args.pixel_profile,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "frames_per_turn": args.frames_per_turn,
            "duplicate_single_video_frame": args.duplicate_single_video_frame,
            "prompt_style": args.prompt_style,
            "kv_window": args.kv_window,
            "current_time": row["current_time"],
            "gold_caption": row["gold_caption"],
            "pred_caption": raw,
            "current_keyword_recall": current,
            "previous_keyword_recall": previous,
            "old_copy_risk": bool(idx and previous > current + 0.10),
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": int(generated.numel()),
            "latency_sec": latency_sec,
            "tokens_per_sec": float(generated.numel()) / max(latency_sec, 1e-6),
            "video_grid_thw": inputs.get("video_grid_thw").tolist() if inputs.get("video_grid_thw") is not None else None,
            "pixel_values_videos_shape": tensor_shape(inputs.get("pixel_values_videos")),
            "cache_len": int(engine.decoder.cache.cache_seqlens[0, 0].item()),
            "window_count": int(engine._window_count[0].item()),
            "window_starts": engine._window_starts[0, : int(engine._window_count[0].item())].detach().cpu().tolist(),
            "window_ends": engine._window_ends[0, : int(engine._window_count[0].item())].detach().cpu().tolist(),
        })
    summary = {
        "mode": "stream_summary",
        "model": args.model,
        "video_id": args.video_id,
        "pixel_profile": args.pixel_profile,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "frames_per_turn": args.frames_per_turn,
        "duplicate_single_video_frame": args.duplicate_single_video_frame,
        "prompt_style": args.prompt_style,
        "kv_window": args.kv_window,
        "turns": len(out_rows),
        "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in out_rows) / max(1, len(out_rows)),
        "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in out_rows) / max(1, len(out_rows)),
        "old_copy_risk_count": sum(1 for r in out_rows if r["old_copy_risk"]),
        "last_cache_len": out_rows[-1]["cache_len"] if out_rows else None,
        "results": out_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


def parse_csv_ints(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["processor", "stream"], required=True)
    p.add_argument("--model", default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct")
    p.add_argument("--model-type", default="qwen3vl")
    p.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    p.add_argument("--video-id", default="HxfwLkoj2gs")
    p.add_argument("--video-ids", default="", help="Comma-separated video ids for one-model multi-video stream runs.")
    p.add_argument("--start-time", type=int, default=0)
    p.add_argument("--end-time", type=int, default=9)
    p.add_argument("--pixel-profile", choices=sorted(PIXEL_PROFILES), default="mid")
    p.add_argument("--frame-counts", type=parse_csv_ints, default=[1, 2, 4])
    p.add_argument("--frames-per-turn", type=int, default=2)
    p.add_argument("--duplicate-single-video-frame", action="store_true")
    p.add_argument("--prompt-style", choices=["simple", "schema"], default="simple")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--kv-window", type=int, default=4)
    p.add_argument("--max-len", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.mode == "processor":
        processor_probe(args)
    else:
        stream_probe(args)


if __name__ == "__main__":
    main()
