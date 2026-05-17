#!/usr/bin/env python3
"""Re-extract an OVO frame-cache subtree at a fixed FPS."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


def _video_sort_key(path: Path) -> tuple[int, str]:
    return (len(path.stem), path.stem)


def _extract_one(video_path: Path, out_dir: Path, fps: int) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        pattern,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    dropped: List[str] = []
    if len(frames) % 2 == 1:
        last = frames[-1]
        dropped.append(last.name)
        last.unlink()
        frames = frames[:-1]

    ok = proc.returncode == 0 and bool(frames)
    if ok:
        (out_dir / ".fps").write_text(str(fps))
        (out_dir / ".frame_norm.json").write_text(
            json.dumps(
                {
                    "tail_policy": "drop",
                    "dropped_tail_frames": dropped,
                    "padded_tail_frames": [],
                },
                indent=2,
            )
        )
    return {
        "video_id": out_dir.name,
        "video_path": str(video_path),
        "ok": ok,
        "frames": len(frames),
        "chunks": len(frames) // 2,
        "seconds": round(time.time() - t0, 2),
        "returncode": proc.returncode,
        "error_tail": (proc.stderr or "")[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--delete-existing", action="store_true")
    args = parser.parse_args()

    videos = sorted(args.source_dir.glob("*.mp4"), key=_video_sort_key)
    if not videos:
        raise SystemExit(f"no mp4 files found under {args.source_dir}")

    if args.delete_existing and args.frames_root.exists():
        shutil.rmtree(args.frames_root)
    args.frames_root.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"extracting {len(videos)} videos to {args.frames_root} "
        f"fps={args.fps} workers={args.workers}",
        flush=True,
    )

    statuses: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {
            pool.submit(_extract_one, video, args.frames_root / video.stem, args.fps): video
            for video in videos
        }
        for i, fut in enumerate(as_completed(future_map), 1):
            status = fut.result()
            statuses.append(status)
            print(
                "[{}/{}] {} ok={} frames={} chunks={} sec={}".format(
                    i,
                    len(videos),
                    status["video_id"],
                    status["ok"],
                    status["frames"],
                    status["chunks"],
                    status["seconds"],
                ),
                flush=True,
            )

    summary = {
        "source_dir": str(args.source_dir),
        "frames_root": str(args.frames_root),
        "videos": len(videos),
        "ok": sum(1 for s in statuses if s["ok"]),
        "failed": sum(1 for s in statuses if not s["ok"]),
        "total_frames": sum(int(s["frames"]) for s in statuses),
        "fps": int(args.fps),
        "workers": int(args.workers),
    }
    args.summary.write_text(
        json.dumps(
            {"summary": summary, "statuses": sorted(statuses, key=lambda s: s["video_id"])},
            indent=2,
            ensure_ascii=False,
        )
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"WROTE {args.summary}", flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
