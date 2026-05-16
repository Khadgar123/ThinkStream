#!/usr/bin/env python3
"""High-concurrency frame pre-extraction for StreamingBench videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def discover_videos(video_root: Path) -> list[Path]:
    return sorted(video_root.glob("**/sample_*/video.mp4"))


def rel_video_id(video_root: Path, video_path: Path) -> str:
    rel = video_path.parent.relative_to(video_root)
    return "/".join(rel.parts)


def count_frames(frame_dir: Path) -> int:
    return sum(1 for _ in frame_dir.glob("frame_*.jpg"))


def ffprobe_video(path: Path) -> dict[str, Any]:
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
    try:
        out = subprocess.check_output(cmd, text=True)
        streams = json.loads(out).get("streams") or []
        return streams[0] if streams else {}
    except Exception as exc:  # noqa: BLE001
        return {"probe_error": repr(exc)}


def extract_one(video_root: Path, frames_root: Path, video_path: Path, fps: float, jpeg_q: int, force: bool) -> dict[str, Any]:
    started = time.time()
    vid = rel_video_id(video_root, video_path)
    out_dir = frames_root / vid
    marker = out_dir / ".extract_done"
    meta_path = out_dir / ".frame_meta.json"
    if marker.exists() and not force:
        n = count_frames(out_dir)
        return {
            "video_id": vid,
            "video_path": str(video_path),
            "frame_dir": str(out_dir),
            "n_frames": n,
            "fps": fps,
            "frames_per_chunk": int(fps),
            "chunk_sec": 1,
            "ok": n > 0,
            "skipped": True,
            "elapsed_sec": round(time.time() - started, 3),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    if marker.exists():
        marker.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        "-q:v",
        str(jpeg_q),
        str(out_dir / "frame_%06d.jpg"),
    ]
    status: dict[str, Any] = {
        "video_id": vid,
        "video_path": str(video_path),
        "frame_dir": str(out_dir),
        "fps": fps,
        "frames_per_chunk": int(fps),
        "chunk_sec": 1,
        "skipped": False,
    }
    try:
        subprocess.run(cmd, check=True)
        n = count_frames(out_dir)
        probe = ffprobe_video(video_path)
        status.update({
            "n_frames": n,
            "num_chunks": n // int(fps),
            "ok": n > 0,
            "source_probe": probe,
        })
        meta_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if n > 0:
            marker.write_text("done\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        status.update({"n_frames": count_frames(out_dir), "ok": False, "error": repr(exc)})
    status["elapsed_sec"] = round(time.time() - started, 3)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/extracted"))
    parser.add_argument("--frames-root", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/frames_fps2"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--jpeg-q", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status-out", type=Path, default=None)
    args = parser.parse_args()

    videos = discover_videos(args.video_root)
    if args.limit:
        videos = videos[: args.limit]
    args.frames_root.mkdir(parents=True, exist_ok=True)
    status_out = args.status_out or (args.frames_root / "preextract_status.jsonl")
    summary_out = args.frames_root / "preextract_summary.json"

    print(
        f"Pre-extract StreamingBench frames: videos={len(videos)} "
        f"fps={args.fps:g} workers={args.workers} out={args.frames_root}",
        flush=True,
    )
    statuses: list[dict[str, Any]] = []
    with status_out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [
            pool.submit(extract_one, args.video_root, args.frames_root, video, args.fps, args.jpeg_q, args.force)
            for video in videos
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()
            statuses.append(status)
            f.write(json.dumps(status, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(futs):
                ok = sum(1 for s in statuses if s.get("ok"))
                skipped = sum(1 for s in statuses if s.get("skipped"))
                frames = sum(int(s.get("n_frames") or 0) for s in statuses)
                print(f"[{i}/{len(futs)}] ok={ok} skipped={skipped} frames={frames}", flush=True)

    summary = {
        "video_root": str(args.video_root),
        "frames_root": str(args.frames_root),
        "videos": len(videos),
        "ok": sum(1 for s in statuses if s.get("ok")),
        "failed": sum(1 for s in statuses if not s.get("ok")),
        "skipped": sum(1 for s in statuses if s.get("skipped")),
        "total_frames": sum(int(s.get("n_frames") or 0) for s in statuses),
        "fps": args.fps,
        "frames_per_chunk": int(args.fps),
        "chunk_sec": 1,
        "workers": args.workers,
        "status_out": str(status_out),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
