#!/usr/bin/env python3
"""Compare real video_url chunks across metadata, fps server, and pixel caps."""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit import end_to_end_prefill_memory_matrix as matrix  # noqa: E402
from scripts.audit import vllm_prefill_memory_caption_probe as probe  # noqa: E402


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_pixel_profiles(args: argparse.Namespace) -> List[Dict]:
    if args.mm_profiles_file:
        data = json.loads(Path(args.mm_profiles_file).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.items()
        else:
            items = [(item["name"], item["mm_processor_kwargs"]) for item in data]
        return [
            {
                "name": str(name),
                "min_pixels": 0,
                "max_pixels": 0,
                "mm_processor_kwargs": dict(mm_kwargs),
            }
            for name, mm_kwargs in items
        ]
    if args.pixel_profiles:
        profiles = []
        for spec in parse_csv(args.pixel_profiles):
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"pixel profile must be name:min:max, got {spec}")
            profiles.append({
                "name": parts[0],
                "min_pixels": int(parts[1]),
                "max_pixels": int(parts[2]),
                "mm_processor_kwargs": None,
            })
        return profiles
    return [
        {
            "name": f"px{max_pixels}",
            "min_pixels": min(args.min_pixels, max_pixels),
            "max_pixels": max_pixels,
            "mm_processor_kwargs": None,
        }
        for max_pixels in parse_csv_ints(args.pixel_caps)
    ]


def video_data_uri(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image_size(path: Path) -> List[int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return [int(image.width), int(image.height)]
    except Exception:
        return [0, 0]


def make_video_chunk(
    row: Dict,
    out_dir: Path,
    chunk_source_frames: int,
    source_fps: float,
    scale_height: int,
) -> Path:
    frames = [Path(p) for p in row["video"][-chunk_source_frames:]]
    if not frames:
        raise ValueError(f"row t={row['current_time']} has no frames")
    scale_label = f"h{scale_height}" if scale_height else "native"
    stem = f"{row.get('video_id', 'video')}_t{row['current_time']}_n{len(frames)}_{scale_label}"
    out_path = out_dir / f"{stem}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / f"{stem}_frames"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        link = seq_dir / f"frame_{idx:06d}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.symlink(frame.resolve(), link)
        except OSError:
            link.write_bytes(frame.read_bytes())
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(source_fps),
        "-i",
        str(seq_dir / "frame_%06d.jpg"),
        "-frames:v",
        str(len(frames)),
    ]
    if scale_height > 0:
        cmd.extend(["-vf", f"scale=-2:{scale_height}"])
    cmd.extend(["-r", str(source_fps), "-pix_fmt", "yuv420p", str(out_path)])
    subprocess.run(cmd, check=True)
    return out_path


def build_messages(
    row: Dict,
    video_path: Path,
    layout: str,
    fps_label: float,
    source_fps: float,
    chunk_source_frames: int,
) -> List[Dict]:
    t = int(row["current_time"])
    frame_paths = [Path(p) for p in row["video"][-chunk_source_frames:]]
    dims = image_size(frame_paths[-1]) if frame_paths else [0, 0]
    metadata = {
        "video_id": row.get("video_id"),
        "current_time_sec": t,
        "chunk_time_range_sec": [max(0.0, t + 1.0 - chunk_source_frames / source_fps), t + 1.0],
        "source_frame_fps": source_fps,
        "server_video_sample_fps": fps_label,
        "frames_in_encoded_chunk": len(frame_paths),
        "last_frame_resolution": dims,
        "semantics": "The attached video_url is only the current video chunk, not historical memory.",
    }
    if layout == "metadata":
        prefix = "\n".join([
            "<video_metadata>",
            json.dumps(metadata, ensure_ascii=False),
            "</video_metadata>",
            f'<current_videochunk t="{t}">',
        ])
    elif layout == "plain":
        prefix = f'<current_videochunk t="{t}">'
    else:
        raise ValueError(f"unknown layout: {layout}")
    suffix = "\n".join([
        "</current_videochunk>",
        "<query>",
        (
            f'  <q t="{t}">Describe only the attached current video chunk in one concise English observation. '
            "Use the video content as the visual source. Do not describe older memory or answer QA. "
            "Output exactly <think>current chunk observation</think><answer></answer></q>"
        ),
        "</query>",
    ])
    return [
        {
            "role": "system",
            "content": (
                "You are a streaming video captioning agent. Caption only the newest current video chunk. "
                "If metadata is present, use it only for timing and chunk identity."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {"type": "video_url", "video_url": {"url": video_data_uri(video_path)}},
                {"type": "text", "text": suffix},
            ],
        },
    ]


def build_payload(
    args: argparse.Namespace,
    messages: List[Dict],
    min_pixels: int,
    max_pixels: int,
    mm_processor_kwargs: Dict | None = None,
) -> Dict:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "mm_processor_kwargs": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }
    if mm_processor_kwargs is not None:
        payload["mm_processor_kwargs"] = mm_processor_kwargs
    elif args.mm_processor_kwargs_json:
        payload["mm_processor_kwargs"] = json.loads(args.mm_processor_kwargs_json)
    return payload


def call_model(
    args: argparse.Namespace,
    messages: List[Dict],
    min_pixels: int,
    max_pixels: int,
    mm_processor_kwargs: Dict | None = None,
) -> Dict:
    payload = build_payload(args, messages, min_pixels, max_pixels, mm_processor_kwargs)
    return probe.post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize(results: List[Dict]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in results:
        groups[(row["layout"], row["pixel_profile"], row["min_pixels"], row["max_pixels"])].append(row)
    summary = []
    for (layout, pixel_profile, min_pixels, max_pixels), rows in sorted(groups.items()):
        summary.append({
            "layout": layout,
            "pixel_profile": pixel_profile,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "n": len(rows),
            "avg_keyword_recall": mean(r["keyword_recall"] for r in rows),
            "avg_prompt_tokens": mean(r["usage"].get("prompt_tokens", 0) for r in rows),
            "avg_completion_tokens": mean(r["usage"].get("completion_tokens", 0) for r in rows),
            "format_valid_rate": mean(1.0 if r["format_valid"] else 0.0 for r in rows),
        })
    return summary


def result_from_response(task: Dict, response: Dict) -> Dict:
    if "choices" not in response:
        raw = json.dumps(response, ensure_ascii=False)
        pred = ""
        error = raw[:500]
    else:
        raw = response["choices"][0]["message"].get("content", "")
        pred = probe.extract_think(raw)
        error = ""
    return {
        "video_id": task["video_id"],
        "current_time": task["current_time"],
        "layout": task["layout"],
        "pixel_profile": task["pixel_profile"],
        "min_pixels": task["min_pixels"],
        "max_pixels": task["max_pixels"],
        "encoded_video": task["encoded_video"],
        "gold_caption": task["gold_caption"],
        "pred_caption": pred,
        "raw": raw,
        "keyword_recall": probe.keyword_recall(pred, task["gold_caption"]),
        "format_valid": "<think>" in raw and "</think>" in raw,
        "usage": response.get("usage", {}),
        "error": error,
    }


def write_summary(args: argparse.Namespace, result_rows: List[Dict], config: Dict) -> None:
    summary = {
        "config": config,
        "summary": summarize(result_rows),
        "results": result_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_curl_config(args: argparse.Namespace, tasks: List[Dict]) -> None:
    request_dir = args.out_dir / "requests"
    response_dir = args.out_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"
    config_lines: List[str] = []
    for idx, task in enumerate(tasks):
        request_path = request_dir / f"{task['request_id']}.json"
        response_path = response_dir / f"{task['request_id']}.json"
        request_path.write_text(json.dumps(task["payload"], ensure_ascii=False), encoding="utf-8")
        task["request_path"] = str(request_path)
        task["response_path"] = str(response_path)
        if idx:
            config_lines.append("next")
        config_lines.extend([
            "silent",
            "show-error",
            "fail-with-body",
            f"max-time = {args.timeout}",
            'request = "POST"',
            'header = "Content-Type: application/json"',
            f'url = "{chat_url}"',
            f'data = "@{request_path.resolve()}"',
            f'output = "{response_path.resolve()}"',
        ])
    (args.out_dir / "curl.cfg").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    manifest = [{k: v for k, v in task.items() if k != "payload"} for task in tasks]
    (args.out_dir / "manifest.json").write_text(
        json.dumps({"tasks": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_responses(args: argparse.Namespace, config: Dict) -> None:
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_rows: List[Dict] = []
    for task in manifest["tasks"]:
        response_path = Path(task["response_path"])
        if response_path.exists() and response_path.stat().st_size > 0:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        else:
            response = {"error": f"missing response file: {response_path}"}
        result = result_from_response(task, response)
        result_rows.append(result)
        print(json.dumps({
            "t": result["current_time"],
            "layout": result["layout"],
            "pixel_profile": result["pixel_profile"],
            "min_pixels": result["min_pixels"],
            "max_pixels": result["max_pixels"],
            "recall": result["keyword_recall"],
            "prompt_tokens": result["usage"].get("prompt_tokens"),
            "error": result["error"][:80],
            "pred": result["pred_caption"][:140],
        }, ensure_ascii=False), flush=True)
    write_summary(args, result_rows, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--model", default="qwen3vl2b-video500")
    parser.add_argument("--video-id")
    parser.add_argument("--times", default="0,30,60,90,120")
    parser.add_argument("--layouts", default="plain,metadata")
    parser.add_argument("--pixel-caps", default="65536,100352,200704,401408")
    parser.add_argument(
        "--pixel-profiles",
        default="low:65536:100352,legacy:130000:220000,runtime:200704:401408",
        help="Comma separated name:min:max profiles. Overrides --pixel-caps.",
    )
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--source-fps", type=float, default=2.0)
    parser.add_argument("--expected-sample-fps", type=float, default=1.0)
    parser.add_argument("--chunk-source-frames", type=int, default=2)
    parser.add_argument("--scale-height", type=int, default=0)
    parser.add_argument("--mm-processor-kwargs-json", default="")
    parser.add_argument("--mm-profiles-file")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--emit-requests-only", action="store_true")
    parser.add_argument("--collect-responses-only", action="store_true")
    args = parser.parse_args()

    config_path = args.out_dir / "config.json"
    if args.collect_responses_only:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        collect_responses(args, config)
        return
    if not args.video_id:
        raise SystemExit("--video-id is required unless --collect-responses-only is set")

    rows_by_video = matrix.rows_by_video(args.jsonl)
    if args.video_id not in rows_by_video:
        raise SystemExit(f"video_id not found in {args.jsonl}: {args.video_id}")
    rows = rows_by_video[args.video_id]
    selected_times = parse_csv_ints(args.times)
    missing = [t for t in selected_times if t not in rows]
    if missing:
        raise SystemExit(f"missing requested times for {args.video_id}: {missing}")
    pixel_profiles = parse_pixel_profiles(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out_dir / "chunks"
    config = {
        "jsonl": str(args.jsonl),
        "base_url": args.base_url,
        "model": args.model,
        "video_id": args.video_id,
        "times": selected_times,
        "layouts": parse_csv(args.layouts),
        "pixel_profiles": pixel_profiles,
        "source_fps": args.source_fps,
        "expected_sample_fps": args.expected_sample_fps,
        "chunk_source_frames": args.chunk_source_frames,
        "scale_height": args.scale_height,
        "mm_processor_kwargs_json": args.mm_processor_kwargs_json,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    tasks: List[Dict] = []
    result_rows: List[Dict] = []
    for t in selected_times:
        row = dict(rows[t])
        row["video_id"] = args.video_id
        video_path = make_video_chunk(
            row,
            chunks_dir,
            args.chunk_source_frames,
            args.source_fps,
            args.scale_height,
        )
        for layout in parse_csv(args.layouts):
            for profile in pixel_profiles:
                messages = build_messages(
                    row,
                    video_path,
                    layout=layout,
                    fps_label=args.expected_sample_fps,
                    source_fps=args.source_fps,
                    chunk_source_frames=args.chunk_source_frames,
                )
                payload = build_payload(
                    args,
                    messages,
                    min_pixels=profile["min_pixels"],
                    max_pixels=profile["max_pixels"],
                    mm_processor_kwargs=profile.get("mm_processor_kwargs"),
                )
                task = {
                    "request_id": f"{args.video_id}_t{t}_{layout}_{profile['name']}",
                    "video_id": args.video_id,
                    "current_time": t,
                    "layout": layout,
                    "pixel_profile": profile["name"],
                    "min_pixels": profile["min_pixels"],
                    "max_pixels": profile["max_pixels"],
                    "mm_processor_kwargs": profile.get("mm_processor_kwargs"),
                    "encoded_video": str(video_path),
                    "gold_caption": row["gold_caption"],
                    "payload": payload,
                }
                tasks.append(task)
                if args.emit_requests_only:
                    continue
                response = call_model(
                    args,
                    messages,
                    min_pixels=profile["min_pixels"],
                    max_pixels=profile["max_pixels"],
                    mm_processor_kwargs=profile.get("mm_processor_kwargs"),
                )
                result = result_from_response(task, response)
                result_rows.append(result)
                print(json.dumps({
                    "t": t,
                    "layout": layout,
                    "pixel_profile": profile["name"],
                    "min_pixels": profile["min_pixels"],
                    "max_pixels": profile["max_pixels"],
                    "recall": result["keyword_recall"],
                    "prompt_tokens": result["usage"].get("prompt_tokens"),
                    "pred": result["pred_caption"][:140],
                }, ensure_ascii=False), flush=True)

    if args.emit_requests_only:
        write_curl_config(args, tasks)
        print(json.dumps({
            "status": "requests_emitted",
            "tasks": len(tasks),
            "curl_config": str(args.out_dir / "curl.cfg"),
            "manifest": str(args.out_dir / "manifest.json"),
        }, ensure_ascii=False), flush=True)
        return
    write_summary(args, result_rows, config)


if __name__ == "__main__":
    main()
