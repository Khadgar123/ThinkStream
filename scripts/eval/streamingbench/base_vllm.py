#!/usr/bin/env python3
"""StreamingBench base-model eval through vLLM OpenAI video_url requests.

This is a deliberately small runner for quick ablations while data is still
being built. It evaluates multiple-choice StreamingBench CSV rows by showing
the model only a recent online video window ending at the question timestamp.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx


QUESTION_RE = re.compile(r"^(?P<family>.+?)_sample_(?P<sample>\d+)_")
TIME_RE = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)$")
LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


@dataclass(frozen=True)
class Row:
    idx: int
    csv_name: str
    question_id: str
    family: str
    sample_id: int
    task_type: str
    question: str
    timestamp_sec: float
    answer: str
    options: list[str]
    temporal_clue_type: str
    frames_required: str


def parse_time(value: str) -> float:
    text = str(value or "").strip()
    match = TIME_RE.match(text)
    if not match:
        raise ValueError(f"bad time_stamp={value!r}")
    h = int(match.group("h") or 0)
    m = int(match.group("m") or 0)
    s = int(match.group("s") or 0)
    return float(h * 3600 + m * 60 + s)


def parse_options(value: str) -> list[str]:
    import ast

    try:
        parsed = ast.literal_eval(str(value or ""))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def load_rows(csv_dir: Path, *, sample_per_csv: int | None, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        if sample_per_csv and len(raw_rows) > sample_per_csv:
            raw_rows = rng.sample(raw_rows, sample_per_csv)
        for raw in raw_rows:
            options = parse_options(raw.get("options", ""))
            answer = str(raw.get("answer") or "").strip()
            if not options or answer not in {"A", "B", "C", "D", "E"}:
                continue
            qid = str(raw.get("question_id") or "").strip()
            match = QUESTION_RE.match(qid)
            if not match:
                continue
            rows.append(Row(
                idx=len(rows),
                csv_name=csv_path.name,
                question_id=qid,
                family=match.group("family"),
                sample_id=int(match.group("sample")),
                task_type=str(raw.get("task_type") or "").strip(),
                question=str(raw.get("question") or "").strip(),
                timestamp_sec=parse_time(str(raw.get("time_stamp") or "")),
                answer=answer,
                options=options,
                temporal_clue_type=str(raw.get("temporal_clue_type") or "").strip(),
                frames_required=str(raw.get("frames_required") or "").strip(),
            ))
    return rows


def row_to_manifest(row: Row) -> dict[str, Any]:
    return {
        "idx": row.idx,
        "csv": row.csv_name,
        "question_id": row.question_id,
        "family": row.family,
        "sample_id": row.sample_id,
        "task_type": row.task_type,
        "question": row.question,
        "timestamp_sec": row.timestamp_sec,
        "answer": row.answer,
        "options": row.options,
        "temporal_clue_type": row.temporal_clue_type,
        "frames_required": row.frames_required,
    }


def row_from_manifest(item: dict[str, Any]) -> Row:
    return Row(
        idx=int(item.get("idx", 0)),
        csv_name=str(item.get("csv") or item.get("csv_name") or ""),
        question_id=str(item.get("question_id") or ""),
        family=str(item.get("family") or ""),
        sample_id=int(item.get("sample_id") or 0),
        task_type=str(item.get("task_type") or ""),
        question=str(item.get("question") or ""),
        timestamp_sec=float(item.get("timestamp_sec") or 0.0),
        answer=str(item.get("answer") or ""),
        options=[str(x) for x in (item.get("options") or [])],
        temporal_clue_type=str(item.get("temporal_clue_type") or ""),
        frames_required=str(item.get("frames_required") or ""),
    )


def load_split_manifest(path: Path) -> list[tuple[Row, Path, dict[str, Any]]]:
    resolved: list[tuple[Row, Path, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            video = Path(str(item.get("video") or item.get("video_path") or ""))
            resolved.append((row_from_manifest(item), video, item))
    return resolved


def option_prompt(row: Row) -> str:
    opts = "\n".join(row.options)
    return (
        f"Question: {row.question}\n"
        f"Options:\n{opts}\n"
        "Answer with exactly one letter: A, B, C, D, or E."
    )


class VideoResolver:
    def __init__(self, root: Path):
        self.root = root
        self._by_sample: dict[int, list[Path]] = {}
        for path in root.glob("**/sample_*/video.mp4"):
            match = re.search(r"sample_(\d+)", str(path))
            if match:
                self._by_sample.setdefault(int(match.group(1)), []).append(path)

    def resolve(self, row: Row) -> Path | None:
        candidates = self._by_sample.get(row.sample_id, [])
        if not candidates:
            return None
        family_norm = normalize(row.family)
        best: tuple[int, Path] | None = None
        for path in candidates:
            parent_text = " ".join(part for part in path.parts if "sample_" not in part)
            parent_norm = normalize(parent_text)
            score = 0
            if family_norm and family_norm in parent_norm:
                score += 1000
            for token in re.findall(r"[A-Za-z]+", row.family):
                if normalize(token) in parent_norm:
                    score += 1
            if best is None or score > best[0]:
                best = (score, path)
        if best and best[0] > 0:
            return best[1]
        return candidates[0] if len(candidates) == 1 else None


def build_messages(row: Row, video_path: Path, *, window_sec: float, fps: float) -> list[dict[str, Any]]:
    start = max(0.0, row.timestamp_sec - window_sec)
    end = row.timestamp_sec
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a streaming video understanding assistant. "
                    "You see only the current online visual window ending at "
                    "the question timestamp. Choose the best option letter."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"<online_window start=\"{start:.1f}\" end=\"{end:.1f}\" "
                        f"chunk_sec=\"1\" frames_per_chunk=\"{fps:g}\">"
                    ),
                },
                {"type": "video_url", "video_url": {"url": video_path.resolve().as_uri()}},
                {"type": "text", "text": "</online_window>\n" + option_prompt(row)},
            ],
        },
    ]


def frame_index_for_time(t: float, fps: float) -> int:
    return max(1, int(t * fps) + 1)


def resolve_frame_window(row: Row, source_video: Path, args: argparse.Namespace) -> list[Path]:
    if args.frames_root is None:
        return []
    rel = source_video.parent.relative_to(args.video_root)
    frame_dir = args.frames_root / rel
    start = max(0.0, row.timestamp_sec - float(args.window_sec))
    end = max(start, row.timestamp_sec)
    first = frame_index_for_time(start, args.fps)
    last = max(first, frame_index_for_time(end, args.fps) - 1)
    frames = [frame_dir / f"frame_{i:06d}.jpg" for i in range(first, last + 1)]
    frames = [p for p in frames if p.exists()]
    if args.max_frames_per_request and len(frames) > args.max_frames_per_request:
        if args.max_frames_per_request == 1:
            frames = [frames[-1]]
        else:
            step = (len(frames) - 1) / (args.max_frames_per_request - 1)
            frames = [frames[round(i * step)] for i in range(args.max_frames_per_request)]
    return frames


def _manifest_paths(item: dict[str, Any], key: str) -> list[Path]:
    values = item.get(key) or []
    if not isinstance(values, list):
        return []
    return [Path(str(x)) for x in values if str(x)]


def _sample_uniform(items: list[Path], n: int) -> list[Path]:
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / float(n - 1)
    return [items[round(i * step)] for i in range(n)]


def resolve_frame_range(
    source_video: Path,
    args: argparse.Namespace,
    *,
    start: float,
    end: float,
    max_frames: int = 0,
) -> list[Path]:
    if args.frames_root is None:
        return []
    rel = source_video.parent.relative_to(args.video_root)
    frame_dir = args.frames_root / rel
    first = frame_index_for_time(max(0.0, start), args.fps)
    last = max(first, frame_index_for_time(max(start, end), args.fps) - 1)
    frames = [frame_dir / f"frame_{i:06d}.jpg" for i in range(first, last + 1)]
    frames = [p for p in frames if p.exists()]
    return _sample_uniform(frames, max_frames) if max_frames else frames


def jpeg_video_data_uri(frames: list[Path]) -> str:
    payload = ",".join(base64.b64encode(path.read_bytes()).decode("ascii") for path in frames)
    return "data:video/jpeg;base64," + payload


def image_data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_frame_messages(row: Row, frames: list[Path], *, window_sec: float, fps: float) -> list[dict[str, Any]]:
    start = max(0.0, row.timestamp_sec - window_sec)
    end = row.timestamp_sec
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a streaming video understanding assistant. "
                    "You see only the current online visual window ending at "
                    "the question timestamp. Choose the best option letter."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"<online_window start=\"{start:.1f}\" end=\"{end:.1f}\" "
                        f"chunk_sec=\"1\" frames_per_chunk=\"{fps:g}\" "
                        f"preextracted_frames=\"{len(frames)}\">"
                    ),
                },
                {"type": "video_url", "video_url": {"url": jpeg_video_data_uri(frames)}},
                {"type": "text", "text": "</online_window>\n" + option_prompt(row)},
            ],
        },
    ]


def build_image_messages(row: Row, frames: list[Path], *, window_sec: float, fps: float) -> list[dict[str, Any]]:
    start = max(0.0, row.timestamp_sec - window_sec)
    end = row.timestamp_sec
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"<online_window start=\"{start:.1f}\" end=\"{end:.1f}\" "
            f"chunk_sec=\"1\" frames_per_chunk=\"{fps:g}\" "
            f"preextracted_frames=\"{len(frames)}\">"
        ),
    }]
    for i, frame in enumerate(frames):
        ts = start + (i / max(float(fps), 1e-6))
        content.append({"type": "text", "text": f'<frame ts="{ts:.1f}" />'})
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(frame)}})
    content.append({"type": "text", "text": "</online_window>\n" + option_prompt(row)})
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a streaming video understanding assistant. "
                    "You see timestamped frames from the current online visual "
                    "window ending at the question timestamp. Choose the best "
                    "option letter."
                ),
            }],
        },
        {"role": "user", "content": content},
    ]


def build_recall_image_messages(
    row: Row,
    frames: list[Path],
    *,
    window_sec: float,
    fps: float,
) -> list[dict[str, Any]]:
    start = max(0.0, row.timestamp_sec - window_sec)
    end = row.timestamp_sec
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"<recall_result visual=\"timestamped_images\" start=\"{start:.1f}\" "
            f"end=\"{end:.1f}\" sampled_frames=\"{len(frames)}\">"
        ),
    }]
    denom = max(len(frames) - 1, 1)
    for i, frame in enumerate(frames):
        ts = start + ((end - start) * i / denom)
        content.append({"type": "text", "text": f'<frame ts="{ts:.1f}" />'})
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(frame)}})
    content.append({"type": "text", "text": "</recall_result>\n" + option_prompt(row)})
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You answer after a recall tool returned historical visual "
                    "evidence as timestamped images. Use the timestamps as real "
                    "video time and choose the best option letter."
                ),
            }],
        },
        {"role": "user", "content": content},
    ]


def build_recall_video_block_messages(
    row: Row,
    blocks: list[tuple[float, float, list[Path]]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": f'<recall_result visual="video_blocks" blocks="{len(blocks)}">',
    }]
    for i, (start, end, frames) in enumerate(blocks):
        content.append({
            "type": "text",
            "text": (
                f'<recall_block index="{i}" start="{start:.1f}" '
                f'end="{end:.1f}" frames="{len(frames)}">'
            ),
        })
        content.append({"type": "video_url", "video_url": {"url": jpeg_video_data_uri(frames)}})
        content.append({"type": "text", "text": "</recall_block>"})
    content.append({"type": "text", "text": "</recall_result>\n" + option_prompt(row)})
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You answer after a recall tool returned historical visual "
                    "evidence as timestamped video blocks. Use the block "
                    "timestamps as real video time and choose the best option letter."
                ),
            }],
        },
        {"role": "user", "content": content},
    ]


def resolve_recall_video_blocks(
    row: Row,
    source_video: Path,
    args: argparse.Namespace,
) -> list[tuple[float, float, list[Path]]]:
    start = max(0.0, row.timestamp_sec - float(args.window_sec))
    end = max(start, row.timestamp_sec)
    n_blocks = max(1, int(args.recall_video_blocks))
    out: list[tuple[float, float, list[Path]]] = []
    span = max(0.001, end - start)
    for i in range(n_blocks):
        b_start = start + span * i / n_blocks
        b_end = start + span * (i + 1) / n_blocks
        frames = resolve_frame_range(
            source_video,
            args,
            start=b_start,
            end=b_end,
            max_frames=max(1, int(args.recall_block_frames)),
        )
        if frames:
            out.append((b_start, b_end, frames))
    return out


def _sync_post_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.post(url, json=body)
    return resp.status_code, resp.json()


def clip_path_for(row: Row, source: Path, args: argparse.Namespace) -> Path:
    start = max(0.0, row.timestamp_sec - float(args.window_sec))
    end = float(row.timestamp_sec)
    family = normalize(row.family) or "family"
    source_key = normalize(str(source.parent.parent.name or source.parent.name))[:40]
    start_tag = f"{start:.1f}".replace(".", "p")
    end_tag = f"{end:.1f}".replace(".", "p")
    fps_tag = f"{args.fps:g}".replace(".", "p")
    name = f"{family}_{source_key}_sample{row.sample_id}_t{start_tag}_{end_tag}_fps{fps_tag}.mp4"
    return args.clip_cache_dir / name


def ensure_clip(row: Row, source: Path, args: argparse.Namespace) -> Path:
    out = clip_path_for(row, source, args)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, row.timestamp_sec - float(args.window_sec))
    duration = max(0.5, float(row.timestamp_sec) - start)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"fps={float(args.fps):g}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    return out


def extract_answer(text: str) -> str | None:
    stripped = str(text or "").strip()
    if stripped[:1].upper() in {"A", "B", "C", "D", "E"}:
        return stripped[:1].upper()
    match = LETTER_RE.search(stripped)
    return match.group(1).upper() if match else None


async def call_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    row: Row,
    video_path: Path,
    args: argparse.Namespace,
    manifest_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    frame_sources = {"frames_video", "frames_image", "recall_images"}
    manifest_item = manifest_item or {}
    frame_paths = (
        _manifest_paths(manifest_item, "frame_paths")
        if args.visual_source in frame_sources and manifest_item
        else resolve_frame_window(row, video_path, args)
        if args.visual_source in frame_sources
        else []
    )
    if args.visual_source == "recall_images" and frame_paths:
        frame_paths = (
            _manifest_paths(manifest_item, "recall_image_frame_paths")
            if manifest_item.get("recall_image_frame_paths")
            else _sample_uniform(frame_paths, max(1, int(args.recall_image_frames)))
        )
    if args.visual_source in frame_sources and not frame_paths:
        return {
            "idx": row.idx,
            "csv": row.csv_name,
            "question_id": row.question_id,
            "family": row.family,
            "sample_id": row.sample_id,
            "task_type": row.task_type,
            "timestamp_sec": row.timestamp_sec,
            "temporal_clue_type": row.temporal_clue_type,
            "frames_required": row.frames_required,
            "video": str(video_path),
            "ok": False,
            "answer": row.answer,
            "pred": None,
            "correct": False,
            "raw": "",
            "elapsed_sec": round(time.time() - t0, 3),
            "error": "no pre-extracted frames for window",
        }
    clip_path: Path | None = None
    recall_blocks: list[tuple[float, float, list[Path]]] = []
    if args.visual_source == "recall_video_blocks":
        raw_blocks = manifest_item.get("recall_video_blocks") or []
        if raw_blocks:
            recall_blocks = [
                (
                    float(block.get("start", 0.0)),
                    float(block.get("end", 0.0)),
                    [Path(str(p)) for p in (block.get("frame_paths") or block.get("frames") or [])],
                )
                for block in raw_blocks
                if block.get("frame_paths") or block.get("frames")
            ]
        else:
            recall_blocks = resolve_recall_video_blocks(row, video_path, args)
        if not recall_blocks:
            return {
                "idx": row.idx,
                "question_id": row.question_id,
                "ok": False,
                "answer": row.answer,
                "pred": None,
                "correct": False,
                "raw": "",
                "elapsed_sec": round(time.time() - t0, 3),
                "error": "no pre-extracted frames for recall video blocks",
            }
        messages = build_recall_video_block_messages(row, recall_blocks)
        visual_source = "recall_video_blocks_data_video_jpeg"
    elif frame_paths and args.visual_source == "frames_video":
        messages = build_frame_messages(row, frame_paths, window_sec=args.window_sec, fps=args.fps)
        visual_source = "frames_data_video_jpeg"
    elif frame_paths and args.visual_source == "recall_images":
        messages = build_recall_image_messages(row, frame_paths, window_sec=args.window_sec, fps=args.fps)
        visual_source = "recall_timestamped_image_url"
    elif frame_paths and args.visual_source == "frames_image":
        messages = build_image_messages(row, frame_paths, window_sec=args.window_sec, fps=args.fps)
        visual_source = "frames_image_url"
    else:
        clip_path = ensure_clip(row, video_path, args)
        messages = build_messages(row, clip_path, window_sec=args.window_sec, fps=args.fps)
        visual_source = "clip_file_video_url"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "min_pixels": int(args.min_pixels),
            "max_pixels": int(args.max_pixels),
        },
    }
    url = f"{endpoint.rstrip('/')}/chat/completions"
    err: Exception | None = None
    try:
        resp = await client.post(url, json=body)
        ok = resp.status_code == 200
        data = resp.json()
    except Exception as first_exc:  # noqa: BLE001
        err = first_exc
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(args.timeout),
                trust_env=False,
            ) as retry_client:
                resp = await retry_client.post(url, json=body)
            ok = resp.status_code == 200
            data = resp.json()
        except Exception as retry_exc:  # noqa: BLE001
            try:
                status_code, data = await asyncio.to_thread(
                    _sync_post_json,
                    url,
                    body,
                    float(args.timeout),
                )
                ok = status_code == 200
                resp = SimpleNamespace(status_code=status_code)
            except Exception as sync_exc:  # noqa: BLE001
                err = sync_exc
            else:
                err = None
        else:
            err = None
    if err is not None:
        return {
            "idx": row.idx,
            "question_id": row.question_id,
            "ok": False,
            "endpoint": endpoint,
            "error": repr(err),
            "elapsed_sec": round(time.time() - t0, 3),
        }
    content = ""
    usage = {}
    if ok:
        choice = data.get("choices", [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
    pred = extract_answer(content)
    return {
        "idx": row.idx,
        "csv": row.csv_name,
        "question_id": row.question_id,
        "family": row.family,
        "sample_id": row.sample_id,
        "task_type": row.task_type,
        "timestamp_sec": row.timestamp_sec,
        "temporal_clue_type": row.temporal_clue_type,
        "frames_required": row.frames_required,
        "video": str(video_path),
        "clip": str(clip_path) if clip_path is not None else "",
        "visual_source": visual_source,
        "frames": [str(p) for p in frame_paths],
        "recall_blocks": [
            {
                "start": round(s, 3),
                "end": round(e, 3),
                "frames": [str(p) for p in fs],
            }
            for s, e, fs in recall_blocks
        ],
        "n_frames": len(frame_paths),
        "n_recall_block_frames": sum(len(fs) for _s, _e, fs in recall_blocks),
        "ok": ok,
        "status_code": resp.status_code,
        "answer": row.answer,
        "pred": pred,
        "correct": bool(pred == row.answer),
        "raw": content[:500],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_sec": round(time.time() - t0, 3),
        "error": "" if ok else str(data)[:500],
    }


async def run(args: argparse.Namespace) -> None:
    if args.split_manifest:
        resolved = load_split_manifest(args.split_manifest)
        rows = [row for row, _path, _item in resolved]
        missing = []
    else:
        rows = load_rows(args.csv_dir, sample_per_csv=args.sample_per_csv, seed=args.seed)
        resolver = VideoResolver(args.video_root)
        resolved = []
        missing = []
        for row in rows:
            path = resolver.resolve(row)
            if path is None:
                missing.append(row.question_id)
            else:
                resolved.append((row, path, {}))
    if args.limit:
        resolved = resolved[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if len(models) == 1:
        models = models * len(endpoints)
    if len(models) != len(endpoints):
        raise ValueError("--models must have one value or match --endpoints")

    meta = {
        "csv_dir": str(args.csv_dir),
        "video_root": str(args.video_root),
        "rows_loaded": len(rows),
        "rows_resolved": len(resolved),
        "missing_video": len(missing),
        "endpoints": endpoints,
        "models": models,
        "carrier": "image_url" if args.visual_source == "frames_image" else "video_url",
        "visual_source": args.visual_source,
        "requested_type": "OpenAI HTTP accepts video_url; frames_image uses timestamped image_url fallback",
        "chunk_sec": 1,
        "frames_per_chunk": args.fps,
        "visual_window": args.window_sec,
        "kv_window": args.window_sec,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "clip_cache_dir": str(args.clip_cache_dir),
        "frames_root": str(args.frames_root) if args.frames_root else None,
        "max_frames_per_request": args.max_frames_per_request,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(64, args.concurrency * 2))
    out_path = args.out_dir / "predictions.jsonl"
    done = 0
    correct = 0
    with out_path.open("w", encoding="utf-8") as f:
        if int(args.concurrency) <= 1:
            async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
                for i, item in enumerate(resolved):
                    row = await call_one(
                        client,
                        endpoints[i % len(endpoints)],
                        models[i % len(models)],
                        item[0],
                        item[1],
                        args,
                        item[2],
                    )
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    correct += int(bool(row.get("correct")))
                    if done % args.log_every == 0:
                        print(f"done={done}/{len(resolved)} acc={correct / max(done, 1):.3f}", flush=True)
        else:
            sem = asyncio.Semaphore(args.concurrency)
            async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
                async def bounded(i: int, item: tuple[Row, Path, dict[str, Any]]) -> dict[str, Any]:
                    async with sem:
                        endpoint = endpoints[i % len(endpoints)]
                        model = models[i % len(models)]
                        return await call_one(client, endpoint, model, item[0], item[1], args, item[2])

                tasks = [
                    asyncio.create_task(bounded(i, item))
                    for i, item in enumerate(resolved)
                ]
                for coro in asyncio.as_completed(tasks):
                    row = await coro
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    correct += int(bool(row.get("correct")))
                    if done % args.log_every == 0:
                        print(f"done={done}/{len(tasks)} acc={correct / max(done, 1):.3f}", flush=True)

    rows_out = [json.loads(line) for line in (args.out_dir / "predictions.jsonl").read_text().splitlines()]
    ok_rows = [r for r in rows_out if r.get("ok")]
    summary = {
        **meta,
        "completed": len(rows_out),
        "ok": len(ok_rows),
        "accuracy": sum(1 for r in ok_rows if r.get("correct")) / max(len(ok_rows), 1),
        "by_csv": {},
        "by_task_type": {},
    }
    for key_name, out_key in [("csv", "by_csv"), ("task_type", "by_task_type")]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in ok_rows:
            groups.setdefault(str(row.get(key_name) or ""), []).append(row)
        summary[out_key] = {
            key: {
                "n": len(vals),
                "acc": sum(1 for v in vals if v.get("correct")) / max(len(vals), 1),
            }
            for key, vals in sorted(groups.items())
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/StreamingBench"))
    parser.add_argument("--video-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/streamingbench_base_vllm"))
    parser.add_argument("--clip-cache-dir", type=Path, default=Path("output/streamingbench_base_vllm/clips"))
    parser.add_argument("--frames-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/frames_fps2"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Precomputed StreamingBench JSONL manifest with fixed windows/frames.",
    )
    parser.add_argument(
        "--visual-source",
        choices=["frames_video", "frames_image", "clip", "recall_images", "recall_video_blocks"],
        default="frames_image",
    )
    parser.add_argument("--recall-image-frames", type=int, default=8)
    parser.add_argument("--recall-video-blocks", type=int, default=4)
    parser.add_argument("--recall-block-frames", type=int, default=8)
    parser.add_argument("--max-frames-per-request", type=int, default=0)
    parser.add_argument("--endpoints", required=True, help="Comma-separated OpenAI base URLs, e.g. http://127.0.0.1:18100/v1,...")
    parser.add_argument("--models", required=True, help="One model name or comma-separated names matching endpoints")
    parser.add_argument("--window-sec", type=float, default=32.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--sample-per-csv", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
