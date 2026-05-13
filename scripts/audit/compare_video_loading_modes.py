#!/usr/bin/env python3
"""Compare Qwen3-VL raw-mp4 vs pre-extracted-frame video loading.

This is a processor-level audit.  It does not load model weights or use GPU.
It answers whether two input protocols produce the same selected frame count,
timestamp metadata, and visual token grid.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from transformers import AutoProcessor


def iter_video_items(messages: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                yield item


def load_sample(path: Path, line_index: int | None, min_frames: int) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line_index is not None and i != line_index:
                continue
            row = json.loads(line)
            for item in iter_video_items(row["messages"]):
                video = item.get("video")
                if isinstance(video, list) and len(video) >= min_frames:
                    return i, row, item
            if line_index is not None:
                break
    raise RuntimeError(f"No sample with >= {min_frames} pre-extracted frames found in {path}")


def make_mp4(frames: List[str], out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq_dir = out_path.parent / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for old in seq_dir.glob("frame_*.jpg"):
        old.unlink()
    for i, frame in enumerate(frames, start=1):
        link = seq_dir / f"frame_{i:06d}.jpg"
        os.symlink(Path(frame).resolve(), link)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(seq_dir / "frame_%06d.jpg"),
        "-frames:v",
        str(len(frames)),
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def probe_video(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    streams = json.loads(out).get("streams") or []
    return streams[0] if streams else {}


def strip_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep the same prompt prefix, but remove the gold assistant answer so this
    # audit reflects generation-time input.
    return [m for m in messages if m.get("role") != "assistant"]


def replace_video_item(messages: List[Dict[str, Any]], new_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    replaced = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(dict(msg))
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video" and not replaced:
                new_content.append(new_item)
                replaced = True
            else:
                new_content.append(item)
        out.append({**msg, "content": new_content})
    if not replaced:
        raise RuntimeError("No video item was replaced")
    return out


def metadata_without_flag(meta: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    return {k: v for k, v in meta.items() if k != "do_sample_frames"}


def setup_processor(
    model: str,
    *,
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
) -> Any:
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = image_min_pixels
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = image_max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = image_min_pixels
            ip.size["longest_edge"] = image_max_pixels
    vp = getattr(processor, "video_processor", None)
    if vp is not None:
        if hasattr(vp, "min_pixels"):
            vp.min_pixels = video_min_pixels
        if hasattr(vp, "max_pixels"):
            vp.max_pixels = video_max_pixels
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = video_min_pixels
            vp.size["longest_edge"] = video_max_pixels
    return processor


def run_processor(
    processor: Any,
    messages: List[Dict[str, Any]],
    video_metadata: List[Dict[str, Any]] | None,
    *,
    do_sample_frames: bool,
    processor_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    processor_kwargs = processor_kwargs or {}
    kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        do_sample_frames=do_sample_frames,
        **processor_kwargs,
    )
    if video_metadata:
        kwargs["video_metadata"] = video_metadata
    inputs = processor.apply_chat_template(messages, **kwargs)

    text_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        do_sample_frames=do_sample_frames,
        **processor_kwargs,
    )
    if video_metadata:
        text_kwargs["video_metadata"] = video_metadata
    rendered = processor.apply_chat_template(messages, **text_kwargs)

    def tensor_shape(key: str) -> Any:
        value = inputs.get(key)
        if value is None:
            return None
        return list(value.shape)

    grid = inputs.get("video_grid_thw")
    image_grid = inputs.get("image_grid_thw")
    return {
        "input_ids": tensor_shape("input_ids"),
        "attention_mask": tensor_shape("attention_mask"),
        "pixel_values": tensor_shape("pixel_values"),
        "pixel_values_videos": tensor_shape("pixel_values_videos"),
        "video_grid_thw": grid.tolist() if grid is not None else None,
        "image_grid_thw": image_grid.tolist() if image_grid is not None else None,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "rendered_video_token_count": rendered.count("<|video_pad|>"),
        "rendered_image_token_count": rendered.count("<|image_pad|>"),
        "rendered_second_markers": rendered.count("seconds"),
        "rendered_frame_tag_count": rendered.count("<frame "),
        "rendered_prefix": rendered[:600],
    }


def make_ts_image_item_sequence(
    frames: List[str],
    metadata: Dict[str, Any],
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> List[Dict[str, Any]]:
    fps = float(metadata.get("fps") or 2.0)
    indices = list(metadata.get("frames_indices") or range(len(frames)))
    content: List[Dict[str, Any]] = []
    for offset, frame in enumerate(frames):
        frame_idx = int(indices[offset]) if offset < len(indices) else offset
        content.append({
            "type": "text",
            "text": f'<frame ts="{frame_idx / fps:.1f}" role="latest chunk" />',
        })
        image_item: Dict[str, Any] = {"type": "image", "image": frame}
        if min_pixels is not None:
            image_item["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_item["max_pixels"] = max_pixels
        content.append(image_item)
    return content


def replace_video_item_with_sequence(messages: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    replaced = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(dict(msg))
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video" and not replaced:
                new_content.extend(new_items)
                replaced = True
            else:
                new_content.append(item)
        out.append({**msg, "content": new_content})
    if not replaced:
        raise RuntimeError("No video item was replaced")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl"))
    parser.add_argument("--line-index", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--model", default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--out-dir", type=Path, default=Path("output/video_loading_compare"))
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=200704)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    args = parser.parse_args()

    line_idx, row, video_item = load_sample(args.jsonl, args.line_index, args.min_frames)
    frames = list(video_item["video"])
    source_meta = dict(video_item.get("video_metadata") or {})
    source_fps = float(source_meta.get("fps") or 2.0)

    sample_dir = args.out_dir / f"{row['video_id']}_line{line_idx}_chunk{row['chunk_idx']}"
    mp4_path = sample_dir / f"frames_as_{source_fps:g}fps.mp4"
    make_mp4(frames, mp4_path, source_fps)

    image_min_pixels = int(args.image_min_pixels or args.video_min_pixels)
    image_max_pixels = int(args.image_max_pixels or args.video_max_pixels)
    processor = setup_processor(
        args.model,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
    )
    base_messages = strip_assistant(row["messages"])

    modes: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]] | None, bool, Dict[str, Any], Dict[str, Any]]] = {}

    pre_item = dict(video_item)
    pre_meta = metadata_without_flag(pre_item.get("video_metadata"))
    modes["preframes_all_do_sample_false"] = (
        replace_video_item(base_messages, pre_item),
        [pre_meta] if pre_meta else None,
        False,
        {},
        {"frames": len(frames), "metadata": pre_meta, "item_minmax": [pre_item.get("min_pixels"), pre_item.get("max_pixels")]},
    )

    sampled_frames = frames[::2]
    sampled_indices = list(source_meta.get("frames_indices") or range(len(frames)))[::2]
    sampled_meta = {
        "fps": source_fps,
        "frames_indices": sampled_indices,
        "total_num_frames": int(source_meta.get("total_num_frames") or len(frames)),
    }
    sampled_item = {
        **pre_item,
        "video": sampled_frames,
        "video_metadata": {**sampled_meta, "do_sample_frames": False},
    }
    modes["preframes_every_other_do_sample_false"] = (
        replace_video_item(base_messages, sampled_item),
        [sampled_meta],
        False,
        {},
        {"frames": len(sampled_frames), "metadata": sampled_meta, "item_minmax": [sampled_item.get("min_pixels"), sampled_item.get("max_pixels")]},
    )

    ts_items = make_ts_image_item_sequence(
        frames,
        pre_meta or {},
        min_pixels=video_item.get("min_pixels"),
        max_pixels=video_item.get("max_pixels"),
    )
    modes["ts_image_all"] = (
        replace_video_item_with_sequence(base_messages, ts_items),
        None,
        False,
        {},
        {"frames": len(frames), "metadata": pre_meta, "item_minmax": [video_item.get("min_pixels"), video_item.get("max_pixels")]},
    )

    ts_sampled_items = make_ts_image_item_sequence(
        sampled_frames,
        sampled_meta,
        min_pixels=video_item.get("min_pixels"),
        max_pixels=video_item.get("max_pixels"),
    )
    modes["ts_image_every_other"] = (
        replace_video_item_with_sequence(base_messages, ts_sampled_items),
        None,
        False,
        {},
        {"frames": len(sampled_frames), "metadata": sampled_meta, "item_minmax": [video_item.get("min_pixels"), video_item.get("max_pixels")]},
    )

    for fps in (source_fps, source_fps / 2.0):
        raw_item = {
            "type": "video",
            "video": str(mp4_path),
            "fps": fps,
            "min_pixels": video_item.get("min_pixels"),
            "max_pixels": video_item.get("max_pixels"),
        }
        modes[f"raw_mp4_fps_{fps:g}"] = (
            replace_video_item(base_messages, raw_item),
            None,
            True,
            {"fps": fps},
            {"mp4": str(mp4_path), "fps": fps, "item_minmax": [raw_item.get("min_pixels"), raw_item.get("max_pixels")]},
        )

    results: Dict[str, Any] = {
        "sample": {
            "jsonl": str(args.jsonl),
            "line_index": line_idx,
            "video_id": row["video_id"],
            "chunk_idx": row["chunk_idx"],
            "frames": len(frames),
            "source_metadata": source_meta,
            "mp4": str(mp4_path),
            "mp4_probe": probe_video(mp4_path),
            "processor_image_size": getattr(getattr(processor, "image_processor", None), "size", None),
            "processor_video_size": getattr(getattr(processor, "video_processor", None), "size", None),
        },
        "modes": {},
    }

    for name, (messages, video_metadata, do_sample_frames, processor_kwargs, extra) in modes.items():
        try:
            results["modes"][name] = {
                **extra,
                **run_processor(
                    processor,
                    messages,
                    video_metadata,
                    do_sample_frames=do_sample_frames,
                    processor_kwargs=processor_kwargs,
                ),
            }
        except Exception as exc:
            results["modes"][name] = {**extra, "error": repr(exc)}

    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / "compare.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
