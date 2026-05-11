"""Pre-extract batch video frames under the configured data root.

This is the frame-only front half of pipeline.py. It writes only batch metadata,
frames/<video_id>/frame_*.jpg, .fps markers, and an extraction audit. Downstream
pipeline runs then reuse the extracted frames instead of recomputing them.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from .config import (
    AUDIT_DIR,
    DATA_ROOT,
    FPS,
    FRAMES_PER_CHUNK,
    ensure_dirs,
)
from .pipeline import (
    _load_videos_jsonl,
    _write_batch_manifest,
    _write_jsonl,
    extract_frames,
)

logger = logging.getLogger(__name__)


def _extract_one(
    video: Dict,
    frames_root: Path,
    fps: int,
    tail_policy: str,
) -> Tuple[Dict, Dict]:
    vid = str(video["video_id"])
    frames = extract_frames(
        str(video["video_path"]),
        frames_root / vid,
        fps=fps,
        frames_per_chunk=FRAMES_PER_CHUNK,
        tail_policy=tail_policy,
    )
    norm_path = frames_root / vid / ".frame_norm.json"
    norm = {}
    if norm_path.exists():
        try:
            norm = json.loads(norm_path.read_text())
        except Exception:
            norm = {}
    num_chunks = len(frames) // FRAMES_PER_CHUNK
    status = {
        "video_id": vid,
        "video_path": str(video.get("video_path", "")),
        "n_frames": len(frames),
        "num_chunks": num_chunks,
        "fps": fps,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "frame_tail_policy": norm.get("tail_policy", "drop"),
        "dropped_tail_frames": norm.get("dropped_tail_frames", []),
        "padded_tail_frames": norm.get("padded_tail_frames", []),
        "ok": num_chunks > 0,
    }
    out_video = dict(video)
    out_video["num_chunks"] = num_chunks
    return out_video, status


def preextract(
    videos_jsonl: Path,
    *,
    num_videos: int,
    workers: int,
    fps: int,
    tail_policy: str,
) -> Dict:
    ensure_dirs()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frames_root = DATA_ROOT / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    videos = _load_videos_jsonl(str(videos_jsonl), limit=num_videos)
    _write_jsonl(DATA_ROOT / "video_registry.jsonl", videos)
    _write_batch_manifest(videos, source=str(videos_jsonl), seed=0)

    logger.info(
        "Pre-extracting %d videos to %s at fps=%s with workers=%d",
        len(videos),
        frames_root,
        fps,
        workers,
    )

    valid_videos: List[Dict] = []
    statuses: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {
            pool.submit(_extract_one, video, frames_root, fps, tail_policy): video
            for video in videos
        }
        for i, fut in enumerate(as_completed(future_map), 1):
            video = future_map[fut]
            vid = str(video.get("video_id", ""))
            try:
                out_video, status = fut.result()
            except Exception as exc:
                status = {
                    "video_id": vid,
                    "video_path": str(video.get("video_path", "")),
                    "n_frames": 0,
                    "num_chunks": 0,
                    "fps": fps,
                    "ok": False,
                    "error": repr(exc),
                }
                logger.exception("[%s] frame extraction failed", vid)
            else:
                if status["ok"]:
                    valid_videos.append(out_video)
                logger.info(
                    "[%d/%d] %s frames=%d chunks=%d ok=%s",
                    i,
                    len(videos),
                    status["video_id"],
                    status["n_frames"],
                    status["num_chunks"],
                    status["ok"],
                )
            statuses.append(status)

    summary = {
        "data_root": str(DATA_ROOT),
        "frames_root": str(frames_root),
        "videos_requested": len(videos),
        "videos_ok": sum(1 for s in statuses if s.get("ok")),
        "videos_failed": sum(1 for s in statuses if not s.get("ok")),
        "total_frames": sum(int(s.get("n_frames") or 0) for s in statuses),
        "fps": fps,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "frame_tail_policy": tail_policy,
        "workers": workers,
    }
    (AUDIT_DIR / "frame_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    with (AUDIT_DIR / "frame_extraction_status.jsonl").open("w") as f:
        for status in sorted(statuses, key=lambda x: x.get("video_id", "")):
            f.write(json.dumps(status, ensure_ascii=False) + "\n")

    if valid_videos:
        _write_jsonl(DATA_ROOT / "selected_videos_with_frames.jsonl", valid_videos)
    logger.info("Frame extraction summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-jsonl",
        type=Path,
        default=DATA_ROOT.parent / "batch2_videos.jsonl",
    )
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--tail-policy",
        choices=["drop", "pad_duplicate", "keep"],
        default="drop",
        help="How to handle extracted tail frames that cannot form a full "
             "FRAMES_PER_CHUNK chunk. drop preserves timestamp correctness; "
             "pad_duplicate keeps the partial tail by duplicating the last "
             "frame; keep leaves the raw odd count.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    preextract(
        args.videos_jsonl,
        num_videos=args.num_videos,
        workers=args.workers,
        fps=args.fps,
        tail_policy=args.tail_policy,
    )


if __name__ == "__main__":
    main()
