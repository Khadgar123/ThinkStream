#!/usr/bin/env python3
"""Verify Qwen3-VL image/video input paths for ThinkStream.

This script intentionally separates two questions:
1. What does vLLM OpenAI HTTP accept?
2. Does the local HF/Qwen processor used by training honor type=video frame
   lists and run-level pixel budgets?
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


PIXEL_PROFILES = {
    "low": (32768, 65536),
    "mid": (65536, 100000),
    "legacy": (130000, 220000),
    "runtime": (200704, 401408),
}


def image_data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def video_jpeg_data_uri(paths: List[Path]) -> str:
    return "data:video/jpeg;base64," + ",".join(
        base64.b64encode(path.read_bytes()).decode("ascii") for path in paths
    )


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "ok": True,
                "status": resp.status,
                "body": json.loads(resp.read().decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return {"ok": False, "status": exc.code, "body": parsed}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "body": repr(exc)}


def load_one_row(jsonl: Path, video_id: str, time_sec: int) -> Dict[str, Any]:
    rows = probe.load_rows(jsonl, video_id, time_sec, time_sec)
    return rows[0]


def base_messages_text(t: int) -> tuple[Dict[str, Any], str, str]:
    system = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": "You are a streaming video captioning agent. Caption only the current 1-second chunk.",
        }],
    }
    prefix = f'<current_chunk t="{t}-{t + 1}" fps="2">'
    suffix = (
        "</current_chunk>\n"
        "<active_query>Describe only the attached current 1-second chunk. "
        "Output one concise English sentence.</active_query>"
    )
    return system, prefix, suffix


def build_http_messages(row: Dict[str, Any], carrier: str, min_pixels: int, max_pixels: int) -> List[Dict[str, Any]]:
    t = int(row["current_time"])
    frames = [Path(p) for p in row["video"][-2:]]
    system, prefix, suffix = base_messages_text(t)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prefix}]
    if carrier == "image_url":
        for idx, frame in enumerate(frames):
            content.append({"type": "text", "text": f'<frame ts="{t + idx * 0.5:.1f}">'} )
            content.append({"type": "image_url", "image_url": {"url": image_data_uri(frame)}})
    elif carrier == "video_url_jpeg":
        content.append({
            "type": "video_url",
            "video_url": {"url": video_jpeg_data_uri(frames)},
        })
    elif carrier == "type_video":
        content.append({
            "type": "video",
            "video": [str(p) for p in frames],
            "video_metadata": {
                "fps": 2.0,
                "frames_indices": [2 * t, 2 * t + 1],
                "total_num_frames": max(2 * t + 2, 2),
                "do_sample_frames": False,
            },
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })
    else:
        raise ValueError(f"unknown carrier: {carrier}")
    content.append({"type": "text", "text": suffix})
    return [system, {"role": "user", "content": content}]


def run_http_check(args: argparse.Namespace, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for carrier in args.http_carriers:
        for profile in args.pixel_profiles:
            min_pixels, max_pixels = PIXEL_PROFILES[profile]
            payload: Dict[str, Any] = {
                "model": args.model,
                "messages": build_http_messages(row, carrier, min_pixels, max_pixels),
                "temperature": 0.0,
                "max_tokens": 80,
                "mm_processor_kwargs": {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
            }
            if carrier == "video_url_jpeg":
                payload["media_io_kwargs"] = {
                    "video": {
                        "fps": 2.0,
                        "frames_indices": [2 * int(row["current_time"]), 2 * int(row["current_time"]) + 1],
                        "total_num_frames": max(2 * int(row["current_time"]) + 2, 2),
                        "do_sample_frames": False,
                    }
                }
            resp = post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout)
            body = resp["body"]
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            content = ""
            if isinstance(body, dict) and body.get("choices"):
                content = body["choices"][0]["message"].get("content") or ""
            results.append({
                "layer": "vllm_http",
                "carrier": carrier,
                "profile": profile,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "ok": resp["ok"],
                "status": resp["status"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "content": content,
                "error": "" if resp["ok"] else str(body)[:500],
            })
    return results


def summarize_tensor(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return list(shape)
    if isinstance(value, list):
        return [summarize_tensor(v) for v in value[:3]]
    return str(type(value))


def set_processor_pixels(processor: Any, min_pixels: int, max_pixels: int) -> None:
    """Match ThinkStream SFT/eval run-level processor resolution control."""
    for attr in ("image_processor", "video_processor"):
        proc = getattr(processor, attr, None)
        if proc is None:
            continue
        if hasattr(proc, "min_pixels"):
            proc.min_pixels = min_pixels
        if hasattr(proc, "max_pixels"):
            proc.max_pixels = max_pixels
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            proc.size["shortest_edge"] = min_pixels
            proc.size["longest_edge"] = max_pixels


def build_processor_messages(row: Dict[str, Any], carrier: str, min_pixels: int, max_pixels: int) -> List[Dict[str, Any]]:
    t = int(row["current_time"])
    frames = [str(p) for p in row["video"][-2:]]
    system, prefix, suffix = base_messages_text(t)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prefix}]
    if carrier == "type_video":
        content.append({
            "type": "video",
            "video": frames,
            "video_metadata": {
                "fps": 2.0,
                "frames_indices": [2 * t, 2 * t + 1],
                "total_num_frames": max(2 * t + 2, 2),
                "do_sample_frames": False,
            },
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })
    elif carrier == "type_image":
        for idx, frame in enumerate(frames):
            content.append({"type": "text", "text": f'<frame ts="{t + idx * 0.5:.1f}">'} )
            content.append({
                "type": "image",
                "image": frame,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            })
    else:
        raise ValueError(f"unknown processor carrier: {carrier}")
    content.append({"type": "text", "text": suffix})
    return [system, {"role": "user", "content": content}]


def collect_video_metadata(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                meta = item.get("video_metadata")
                if isinstance(meta, dict):
                    metadata.append({k: v for k, v in meta.items() if k != "do_sample_frames"})
    return metadata


def run_processor_check(args: argparse.Namespace, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    try:
        from transformers import AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return [{
            "layer": "hf_processor",
            "ok": False,
            "error": f"cannot import processor utilities: {exc!r}",
        }]
    try:
        processor = AutoProcessor.from_pretrained(args.processor_model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        return [{
            "layer": "hf_processor",
            "ok": False,
            "error": f"cannot load processor: {exc!r}",
        }]
    for carrier in args.processor_carriers:
        for profile in args.pixel_profiles:
            min_pixels, max_pixels = PIXEL_PROFILES[profile]
            set_processor_pixels(processor, min_pixels, max_pixels)
            messages = build_processor_messages(row, carrier, min_pixels, max_pixels)
            try:
                template_kwargs = dict(
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    do_sample_frames=False,
                    size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                )
                video_metadata = collect_video_metadata(messages)
                if video_metadata:
                    template_kwargs["video_metadata"] = video_metadata
                rendered = processor.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
                keys = sorted(rendered.keys())
                input_ids = rendered.get("input_ids")
                video_grid = rendered.get("video_grid_thw")
                image_grid = rendered.get("image_grid_thw")
                pixel_values = rendered.get("pixel_values")
                pixel_values_videos = rendered.get("pixel_values_videos")
                video_tokens = None
                if video_grid is not None:
                    video_tokens = int(video_grid.prod().item())
                image_tokens = None
                if image_grid is not None:
                    image_tokens = int(image_grid.prod(dim=1).sum().item())
                results.append({
                    "layer": "hf_processor",
                    "carrier": carrier,
                    "profile": profile,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "ok": True,
                    "keys": keys,
                    "input_ids_shape": summarize_tensor(input_ids),
                    "input_token_count": int(input_ids.shape[-1]) if input_ids is not None else None,
                    "video_grid_thw": video_grid.tolist() if video_grid is not None else None,
                    "image_grid_thw": image_grid.tolist() if image_grid is not None else None,
                    "video_grid_product": video_tokens,
                    "image_grid_product": image_tokens,
                    "pixel_values_shape": summarize_tensor(pixel_values),
                    "pixel_values_videos_shape": summarize_tensor(pixel_values_videos),
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "layer": "hf_processor",
                    "carrier": carrier,
                    "profile": profile,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "ok": False,
                    "error": repr(exc),
                })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl"))
    parser.add_argument("--video-id", default="HxfwLkoj2gs")
    parser.add_argument("--time-sec", type=int, default=25)
    parser.add_argument("--base-url", default="http://127.0.0.1:18114/v1")
    parser.add_argument("--model", default="qwen3vl2b-stream-visual-matrix")
    parser.add_argument("--processor-model-path", default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--pixel-profiles", nargs="+", default=["low", "mid", "legacy", "runtime"])
    parser.add_argument("--http-carriers", nargs="+", default=["image_url", "video_url_jpeg", "type_video"])
    parser.add_argument("--processor-carriers", nargs="+", default=["type_image", "type_video"])
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument("--skip-processor", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("output/base_visual_carrier_matrix/video_input_path_verification.jsonl"))
    args = parser.parse_args()
    row = load_one_row(args.jsonl, args.video_id, args.time_sec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if not args.skip_http:
        rows.extend(run_http_check(args, row))
    if not args.skip_processor:
        rows.extend(run_processor_check(args, row))
    with args.out.open("w", encoding="utf-8") as f:
        for result in rows:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
