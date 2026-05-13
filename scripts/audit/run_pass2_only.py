#!/usr/bin/env python3
"""Run pass2 rollout only for an existing batch root.

The main pipeline continues into pass3, so this helper is for cases where
pass1a/1b are already present and only rollout/compression needs to be rebuilt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.config import FRAMES_PER_CHUNK
from scripts.agent_data.pass2_rollout import (
    compute_compression_stats,
    run_pass2_single_video,
    save_rollout,
)
from scripts.agent_data.audit_pass2_stale import audit_rollouts
from scripts.agent_data.cache_version import STAGE_VERSIONS, write_stage_version
from scripts.agent_data_pipeline.vllm_client import VLLMClient


LOGGER = logging.getLogger("run_pass2_only")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _frame_paths(batch_root: Path, video_id: str) -> List[str]:
    paths = sorted((batch_root / "frames" / video_id).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"missing frames for {video_id}")
    return [str(p) for p in paths]


async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    batch_root = args.batch_root
    videos_path = args.videos_jsonl or batch_root / "selected_videos.jsonl"
    videos = _read_jsonl(videos_path)
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    output_dir = args.output_dir or batch_root / "rollout"
    audit_dir = batch_root / "audits"
    chunk_log_path = audit_dir / "pass2_chunks.jsonl"
    audit_dir.mkdir(parents=True, exist_ok=True)
    chunk_log_path.write_text("", encoding="utf-8")

    client = VLLMClient(
        args.api_base,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )
    semaphore = asyncio.Semaphore(max(1, int(args.video_concurrent)))
    rollout_map: Dict[str, Dict[str, Any]] = {}
    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    failed: Dict[str, str] = {}
    started = time.time()

    async def _one(idx: int, video: Dict[str, Any]) -> None:
        video_id = str(video["video_id"])
        try:
            evidence = _read_json(batch_root / "evidence_1b" / f"{video_id}.json")
            frames = _frame_paths(batch_root, video_id)
            num_chunks = len(frames) // FRAMES_PER_CHUNK
            async with semaphore:
                rollout = await run_pass2_single_video(
                    video_id=video_id,
                    frame_paths=frames,
                    num_chunks=num_chunks,
                    client=client,
                    evidence=evidence,
                    chunk_log_path=chunk_log_path,
                )
            save_rollout(video_id, rollout, output_dir=output_dir)
            rollout_map[video_id] = rollout
            evidence_map[video_id] = evidence
            LOGGER.info(
                "pass2 %d/%d %s: thinks=%d compressions=%d",
                idx + 1,
                len(videos),
                video_id,
                len(rollout.get("thinks") or []),
                len(rollout.get("compression_events") or []),
            )
        except Exception as exc:
            failed[video_id] = repr(exc)
            LOGGER.exception("pass2 failed for %s: %s", video_id, exc)

    await asyncio.gather(*[_one(i, v) for i, v in enumerate(videos)])

    comp_stats = compute_compression_stats(rollout_map)
    stale_report = audit_rollouts(rollout_map, evidence_map)
    _write_json(audit_dir / "compression_stats.json", comp_stats)
    _write_json(audit_dir / "pass2_stale_audit.json", stale_report)
    if not failed and output_dir == batch_root / "rollout":
        write_stage_version("2")
    report = {
        "batch_root": str(batch_root),
        "videos_requested": len(videos),
        "videos_ok": len(rollout_map),
        "failed": failed,
        "output_dir": str(output_dir),
        "chunk_log": str(chunk_log_path),
        "elapsed_sec": round(time.time() - started, 1),
        "stage_version_2": STAGE_VERSIONS.get("2"),
        "compression_stats": comp_stats,
        "stale_totals": stale_report.get("totals", {}),
    }
    _write_json(audit_dir / "pass2_only_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--videos-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrent", type=int, default=1024)
    parser.add_argument("--video-concurrent", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=5400.0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    report = asyncio.run(main_async(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
