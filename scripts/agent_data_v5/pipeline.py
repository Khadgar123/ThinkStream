"""
Main Pipeline Orchestrator for Agent Data v5.0

Usage:
    python -m scripts.agent_data_v5.pipeline run \
        --api_base http://AMD_IP:8000/v1 \
        --video_root /path/to/videos \
        --num_videos 300

    python -m scripts.agent_data_v5.pipeline stress_test \
        --api_base http://AMD_IP:8000/v1
"""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from .config import (
    ALL_DIRS,
    DATA_ROOT,
    EVIDENCE_DIR,
    FINAL_DIR,
    PASS_CONFIG,
    PHASE_CONFIG,
    ROLLOUT_DIR,
    SAMPLES_DIR,
    TASKS_DIR,
    VLLM_MODEL,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Assignment
# ---------------------------------------------------------------------------


def assign_phase(sample: Dict) -> str:
    """Assign training phase based on sample type and task complexity."""
    sample_type = sample.get("sample_type", "")
    task_type = sample.get("metadata", {}).get("task_type", "")

    if sample_type == "silent":
        return "1"  # Phase 1: protocol alignment
    elif sample_type == "response":
        if "compress" in task_type:
            return "C1"  # Phase C1: response from compressed memory
        elif "unanswerable" in task_type:
            return "2"  # Phase 2
        return "1"  # Phase 1: basic response
    elif sample_type == "compress":
        return "C1"  # Phase C1: compression training
    elif sample_type in ("recall_query", "recall_response"):
        if "compress_recall" in task_type:
            return "C1"  # Phase C1: recall from compressed
        return "2"  # Phase 2: recall training
    return "5"  # Phase 5: mixed


# ---------------------------------------------------------------------------
# Video Selection
# ---------------------------------------------------------------------------


def select_videos(
    video_root: str,
    num_videos: int = 300,
    min_duration: int = 60,
    seed: int = 42,
) -> List[Dict]:
    """Select videos for data construction.

    Prefers: >120s, diverse content, high annotation density.
    Falls back to existing video_registry if available.
    """
    registry_path = DATA_ROOT / "video_registry.jsonl"
    if registry_path.exists():
        videos = []
        with open(registry_path, "r") as f:
            for line in f:
                videos.append(json.loads(line))
        logger.info(f"Loaded {len(videos)} videos from registry.")
        return videos[:num_videos]

    # Scan video_root for .mp4 files
    import subprocess
    video_root = Path(video_root)
    videos = []

    for vpath in sorted(video_root.rglob("*.mp4")):
        # Get duration via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", str(vpath)],
                capture_output=True, text=True, timeout=10,
            )
            duration = float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            continue

        if duration >= min_duration:
            videos.append({
                "video_id": vpath.stem,
                "video_path": str(vpath),
                "duration_sec": duration,
            })

    # Sort by duration (prefer longer videos), take top N
    random.seed(seed)
    random.shuffle(videos)
    videos.sort(key=lambda x: x["duration_sec"], reverse=True)
    selected = videos[:num_videos]

    # Save registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        for v in selected:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    logger.info(f"Selected {len(selected)} videos (min {min_duration}s)")
    return selected


def extract_frames(video_path: str, output_dir: Path, fps: int = 1) -> List[str]:
    """Extract frames from video at given fps.

    Returns list of frame file paths in order.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%06d.jpg")

    # Check if already extracted
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        return [str(p) for p in existing]

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        "-y", pattern,
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning(f"Frame extraction failed for {video_path}: {e}")
        return []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return [str(p) for p in frames]


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    api_base: str,
    model: str,
    video_root: str,
    num_videos: int = 300,
    seed: int = 42,
    skip_pass: List[int] = None,
):
    """Run the full 5-pass pipeline."""
    from scripts.agent_data_pipeline.vllm_client import VLLMClient

    ensure_dirs()
    skip_pass = skip_pass or []

    # --- Setup ---
    client = VLLMClient(
        api_base=api_base,
        model=model,
        max_concurrent=PASS_CONFIG["pass1_evidence"]["concurrent_videos"],
    )

    # --- Video selection ---
    videos = select_videos(video_root, num_videos, seed=seed)
    logger.info(f"Pipeline starting with {len(videos)} videos")

    # --- Extract frames ---
    frames_dir = DATA_ROOT / "frames"
    video_frames = {}
    for v in videos:
        v_frames_dir = frames_dir / v["video_id"]
        frames = extract_frames(v["video_path"], v_frames_dir, fps=1)
        video_frames[v["video_id"]] = frames
        num_chunks = len(frames) // 2  # 2 frames per chunk (1fps, 2s chunks)
        v["num_chunks"] = num_chunks

    logger.info(f"Frame extraction complete. {sum(len(f) for f in video_frames.values())} total frames.")

    # =================================================================
    # PASS 1: Teacher Evidence Graph
    # =================================================================
    if 1 not in skip_pass:
        from .pass1_evidence import load_evidence, run_pass1_single_video, save_evidence

        logger.info("=" * 60)
        logger.info("PASS 1: Teacher Evidence Graph")
        logger.info("=" * 60)

        # Process videos in parallel (16 concurrent)
        semaphore = asyncio.Semaphore(PASS_CONFIG["pass1_evidence"]["concurrent_videos"])

        async def process_video_pass1(video):
            vid = video["video_id"]
            # Check cache
            cached = load_evidence(vid)
            if cached:
                logger.info(f"  [{vid}] Using cached evidence ({len(cached)} chunks)")
                return vid, cached

            async with semaphore:
                captions = await run_pass1_single_video(
                    video_id=vid,
                    frame_paths=video_frames[vid],
                    num_chunks=video["num_chunks"],
                    client=client,
                )
                save_evidence(vid, captions)
                logger.info(f"  [{vid}] Evidence complete: {len(captions)} chunks")
                return vid, captions

        results = await asyncio.gather(*[process_video_pass1(v) for v in videos])
        evidence_map = {vid: caps for vid, caps in results}
        logger.info(f"Pass 1 complete: {len(evidence_map)} videos")
    else:
        from .pass1_evidence import load_evidence
        evidence_map = {}
        for v in videos:
            cached = load_evidence(v["video_id"])
            if cached:
                evidence_map[v["video_id"]] = cached

    # =================================================================
    # PASS 2: Question-blind Streaming Rollout
    # =================================================================
    if 2 not in skip_pass:
        from .pass2_rollout import load_rollout, run_pass2_single_video, save_rollout

        logger.info("=" * 60)
        logger.info("PASS 2: Question-blind Streaming Rollout")
        logger.info("=" * 60)

        semaphore = asyncio.Semaphore(PASS_CONFIG["pass2_rollout"]["concurrent_videos"])

        async def process_video_pass2(video):
            vid = video["video_id"]
            cached = load_rollout(vid)
            if cached:
                logger.info(f"  [{vid}] Using cached rollout")
                return vid, cached

            async with semaphore:
                rollout = await run_pass2_single_video(
                    video_id=vid,
                    frame_paths=video_frames[vid],
                    num_chunks=video["num_chunks"],
                    client=client,
                )
                save_rollout(vid, rollout)
                logger.info(
                    f"  [{vid}] Rollout complete: {len(rollout['observations'])} obs, "
                    f"{len(rollout['compression_events'])} compressions"
                )
                return vid, rollout

        results = await asyncio.gather(*[process_video_pass2(v) for v in videos])
        rollout_map = {vid: roll for vid, roll in results}
        logger.info(f"Pass 2 complete: {len(rollout_map)} videos")
    else:
        from .pass2_rollout import load_rollout
        rollout_map = {}
        for v in videos:
            cached = load_rollout(v["video_id"])
            if cached:
                rollout_map[v["video_id"]] = cached

    # =================================================================
    # PASS 3: Task Planning
    # =================================================================
    if 3 not in skip_pass:
        from .pass3_tasks import run_pass3

        logger.info("=" * 60)
        logger.info("PASS 3: Task Planning")
        logger.info("=" * 60)

        all_tasks = {}
        for v in videos:
            vid = v["video_id"]
            if vid not in evidence_map or vid not in rollout_map:
                continue

            tasks = await run_pass3(vid, evidence_map[vid], rollout_map[vid], client)
            all_tasks[vid] = tasks

            total = sum(len(t) for t in tasks.values())
            logger.info(f"  [{vid}] Tasks mined: {total}")

        # Save tasks
        tasks_path = TASKS_DIR / "all_tasks.json"
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        with open(tasks_path, "w") as f:
            json.dump(all_tasks, f, ensure_ascii=False)

        logger.info(f"Pass 3 complete: {sum(sum(len(t) for t in v.values()) for v in all_tasks.values())} total tasks")
    else:
        tasks_path = TASKS_DIR / "all_tasks.json"
        with open(tasks_path, "r") as f:
            all_tasks = json.load(f)

    # =================================================================
    # PASS 4: Question-aware Forks
    # =================================================================
    if 4 not in skip_pass:
        from .pass4_forks import (
            build_compress_sample,
            build_recall_sample,
            build_response_sample,
            build_silent_sample,
        )

        logger.info("=" * 60)
        logger.info("PASS 4: Question-aware Forks (Sample Generation)")
        logger.info("=" * 60)

        all_samples = []

        for v in videos:
            vid = v["video_id"]
            if vid not in all_tasks or vid not in rollout_map:
                continue

            rollout = rollout_map[vid]
            tasks = all_tasks[vid]
            observations = rollout["observations"]
            snapshots = rollout["snapshots"]

            # 4a: Silent samples (subset, not all chunks)
            # Sample ~20% of silent chunks for training
            silent_chunks = list(range(rollout["num_chunks"]))
            selected_silent = random.sample(
                silent_chunks, min(len(silent_chunks) // 5, 15)
            )
            for chunk_idx in selected_silent:
                snapshot = snapshots.get(chunk_idx, snapshots.get(str(chunk_idx)))
                if snapshot and chunk_idx < len(observations):
                    sample = build_silent_sample(
                        chunk_idx, observations[chunk_idx]["observation"],
                        snapshot, video_frames.get(vid, []),
                    )
                    sample["video_id"] = vid
                    all_samples.append(sample)

            # 4b: Compress samples
            for event in rollout["compression_events"]:
                chunk_idx = event["trigger_chunk"]
                snapshot = snapshots.get(chunk_idx, snapshots.get(str(chunk_idx)))
                if snapshot and chunk_idx < len(observations):
                    sample = build_compress_sample(
                        chunk_idx, observations[chunk_idx]["observation"],
                        snapshot, event,
                    )
                    sample["video_id"] = vid
                    all_samples.append(sample)

            # 4c: Response samples
            for task_type in ["response_from_frames", "compress_response", "unanswerable"]:
                for task in tasks.get(task_type, []):
                    if not task.get("question"):
                        continue
                    ask_chunk = task["ask_chunk"]
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot:
                        sample = await build_response_sample(
                            task, snapshot, observations, client, vid
                        )
                        if sample:
                            sample["video_id"] = vid
                            all_samples.append(sample)

            # 4d: Recall samples
            for task_type in ["recall", "compress_recall"]:
                for task in tasks.get(task_type, []):
                    if not task.get("question"):
                        continue
                    ask_chunk = task["ask_chunk"]
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot:
                        samples = await build_recall_sample(
                            task, snapshot, observations, client, vid
                        )
                        if samples:
                            for s in samples:
                                s["video_id"] = vid
                            all_samples.extend(samples)

        # Assign sample_id and phase to all samples
        for i, s in enumerate(all_samples):
            vid = s.get("video_id", "unk")
            chunk = s.get("chunk_idx", 0)
            stype = s.get("sample_type", "unk")
            s["sample_id"] = f"{vid}_t{chunk}_{stype}_{i}"
            s["phase"] = assign_phase(s)

        # Save all samples
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        samples_path = SAMPLES_DIR / "all_samples.jsonl"
        with open(samples_path, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"Pass 4 complete: {len(all_samples)} samples generated")
    else:
        samples_path = SAMPLES_DIR / "all_samples.jsonl"
        all_samples = []
        with open(samples_path, "r") as f:
            for line in f:
                all_samples.append(json.loads(line))

    # =================================================================
    # PASS 5: Verify + Filter
    # =================================================================
    from .pass5_verify import filter_samples

    logger.info("=" * 60)
    logger.info("PASS 5: Verify + Filter")
    logger.info("=" * 60)

    passed_samples, stats = filter_samples(all_samples)
    logger.info(f"Verification: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1%})")
    logger.info(f"Fail reasons: {stats['fail_reasons']}")
    logger.info(f"Difficulty distribution: {stats['difficulty_distribution']}")

    # --- Final output: split by phase ---
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Split by video for train/val/test
    video_ids = list(set(s.get("video_id", "") for s in passed_samples))
    random.seed(seed)
    random.shuffle(video_ids)
    n = len(video_ids)
    train_vids = set(video_ids[:int(n * 0.8)])
    val_vids = set(video_ids[int(n * 0.8):int(n * 0.9)])
    test_vids = set(video_ids[int(n * 0.9):])

    train_samples = [s for s in passed_samples if s.get("video_id") in train_vids]
    val_samples = [s for s in passed_samples if s.get("video_id") in val_vids]
    test_samples = [s for s in passed_samples if s.get("video_id") in test_vids]

    # Save
    for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = FINAL_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} samples → {path}")

    # Save stats
    stats_path = FINAL_DIR / "pipeline_stats.json"
    stats["train_count"] = len(train_samples)
    stats["val_count"] = len(val_samples)
    stats["test_count"] = len(test_samples)
    stats["split_by_video"] = True
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total samples: {len(passed_samples)}")
    logger.info(f"Output: {FINAL_DIR}")
    logger.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Agent Data Pipeline v5.0")
    subparsers = parser.add_subparsers(dest="command")

    # Run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--api_base", required=True)
    run_parser.add_argument("--model", default=VLLM_MODEL)
    run_parser.add_argument("--video_root", required=True)
    run_parser.add_argument("--num_videos", type=int, default=300)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--skip_pass", type=int, nargs="*", default=[])

    # Stress test
    st_parser = subparsers.add_parser("stress_test", help="Test vLLM endpoint")
    st_parser.add_argument("--api_base", required=True)
    st_parser.add_argument("--model", default=VLLM_MODEL)
    st_parser.add_argument("--num_requests", type=int, default=10)
    st_parser.add_argument("--max_concurrent", type=int, default=8)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run":
        asyncio.run(run_pipeline(
            api_base=args.api_base,
            model=args.model,
            video_root=args.video_root,
            num_videos=args.num_videos,
            seed=args.seed,
            skip_pass=args.skip_pass,
        ))
    elif args.command == "stress_test":
        from scripts.agent_data_pipeline.vllm_client import stress_test
        asyncio.run(stress_test(
            api_base=args.api_base,
            model=args.model,
            num_requests=args.num_requests,
            max_concurrent=args.max_concurrent,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
