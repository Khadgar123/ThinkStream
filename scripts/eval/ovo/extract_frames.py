#!/usr/bin/env python
"""Extract 1fps frames for all videos referenced by OVO benchmark.

Writes frames to <video_root>/frames/<video_stem>/frame_000001.jpg
Skips videos that already have frames extracted.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path


def extract_one(args):
    video_path, out_dir = args
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*.jpg")):
        return (str(video_path), "skipped")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "fps=1,scale='min(640,iw)':-2",
        "-q:v", "2",
        str(out_dir / "frame_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (str(video_path), "ok")
    except Exception as e:
        return (str(video_path), f"error: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--out_dir", required=True, help="Root directory for extracted frames")
    p.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    args = p.parse_args()

    video_root = Path(args.video_root)
    out_root = Path(args.out_dir)

    with open(args.benchmark_json) as f:
        data = json.load(f)

    unique_videos = set()
    for s in data:
        v = s.get("video")
        if v:
            vp = video_root / v
            if vp.exists():
                unique_videos.add(str(vp))

    print(f"Found {len(unique_videos)} unique videos to extract.")

    tasks = []
    for vp in sorted(unique_videos):
        vp = Path(vp)
        rel = vp.relative_to(video_root)
        stem = rel.with_suffix("")
        out_dir = out_root / stem
        tasks.append((vp, out_dir))

    with Pool(args.workers) as pool:
        results = pool.map(extract_one, tasks)

    ok = sum(1 for _, r in results if r == "ok")
    skipped = sum(1 for _, r in results if r == "skipped")
    errors = [r for _, r in results if r.startswith("error")]
    print(f"Done: ok={ok}, skipped={skipped}, errors={len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
