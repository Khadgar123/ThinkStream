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
    AGENT_CHUNK_SEC,
    ALL_DIRS,
    AUDIT_DIR,
    DATA_ROOT,
    EVIDENCE_DIR,
    FINAL_DIR,
    PASS_CONFIG,
    PHASE_CONFIG,
    ROLLOUT_DIR,
    SAMPLES_DIR,
    TASKS_DIR,
    VLLM_MODEL,
    estimated_request_tokens,
    safe_concurrency_for_pass,
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
    phase_override = sample.get("metadata", {}).get("phase", "")

    # Explicit phase from metadata (e.g., C2 compress samples)
    if phase_override in ("C1", "C2"):
        return phase_override

    if sample_type == "silent":
        return "1"  # Phase 1: protocol alignment
    elif sample_type == "response":
        if "compress" in task_type:
            return "C1"  # Phase C1: response from compressed memory
        elif "unanswerable" in task_type:
            return "2"  # Phase 2
        elif "response_from_memory" in task_type:
            return "2"  # Phase 2: memory-based response
        elif "pending" in task_type:
            return "2"  # Phase 2: pending event response
        return "1"  # Phase 1: basic response (from frames)
    elif sample_type == "compress":
        if "merge_compress" in task_type:
            return "C1"  # Phase C1: second-level compression
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

    # Stratified sampling: group by parent directory (dataset source), then
    # take proportionally from each group, preferring longer videos within group.
    # This avoids all selected videos coming from the longest-duration dataset.
    from collections import defaultdict
    groups = defaultdict(list)
    for v in videos:
        parent = Path(v["video_path"]).parent.name
        groups[parent].append(v)

    # Sort each group by duration descending
    for g in groups.values():
        g.sort(key=lambda x: x["duration_sec"], reverse=True)

    # Round-robin proportional selection
    random.seed(seed)
    group_keys = sorted(groups.keys())
    random.shuffle(group_keys)
    selected = []
    remaining = num_videos
    per_group = max(1, num_videos // max(len(group_keys), 1))
    for key in group_keys:
        take = min(per_group, len(groups[key]), remaining)
        selected.extend(groups[key][:take])
        remaining -= take
        if remaining <= 0:
            break
    # Fill remaining from largest groups
    if remaining > 0:
        all_remaining = [v for g in group_keys for v in groups[g] if v not in selected]
        all_remaining.sort(key=lambda x: x["duration_sec"], reverse=True)
        selected.extend(all_remaining[:remaining])

    random.shuffle(selected)
    selected = selected[:num_videos]

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
    random.seed(seed)  # Seed early for reproducibility across all passes

    from scripts.agent_data_pipeline.vllm_client import VLLMClient

    ensure_dirs()
    skip_pass = skip_pass or []

    # --- Setup ---
    # Client cap = max safe concurrency across all passes; per-pass semaphores
    # enforce tighter limits.
    client = VLLMClient(
        api_base=api_base,
        model=model,
        max_concurrent=max(
            safe_concurrency_for_pass("pass1_evidence"),
            safe_concurrency_for_pass("pass2_rollout"),
            safe_concurrency_for_pass("pass3_tasks"),
            safe_concurrency_for_pass("pass4_forks"),
        ),
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

        pass1_conc = safe_concurrency_for_pass("pass1_evidence")
        logger.info(f"PASS 1 safe concurrency={pass1_conc}")
        semaphore = asyncio.Semaphore(pass1_conc)

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
                    frame_paths=video_frames.get(vid, []),
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

        pass2_conc = safe_concurrency_for_pass("pass2_rollout")
        logger.info(f"PASS 2 safe concurrency={pass2_conc}")
        semaphore = asyncio.Semaphore(pass2_conc)

        async def process_video_pass2(video):
            vid = video["video_id"]
            cached = load_rollout(vid)
            if cached:
                logger.info(f"  [{vid}] Using cached rollout")
                return vid, cached

            async with semaphore:
                rollout = await run_pass2_single_video(
                    video_id=vid,
                    frame_paths=video_frames.get(vid, []),
                    num_chunks=video["num_chunks"],
                    client=client,
                    evidence=evidence_map.get(vid),
                )
                save_rollout(vid, rollout)
                logger.info(
                    f"  [{vid}] Rollout complete: {len(rollout['thinks'])} thinks, "
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
        from .pass3_tasks import audit_task_coverage, run_pass3

        logger.info("=" * 60)
        logger.info("PASS 3: Task Planning")
        logger.info("=" * 60)

        pass3_conc = safe_concurrency_for_pass("pass3_tasks")
        logger.info(f"PASS 3 safe concurrency={pass3_conc}")
        semaphore = asyncio.Semaphore(pass3_conc)

        async def process_video_pass3(video):
            vid = video["video_id"]
            if vid not in evidence_map or vid not in rollout_map:
                return vid, None
            async with semaphore:
                tasks = await run_pass3(vid, evidence_map[vid], rollout_map[vid], client)
                audit = audit_task_coverage(vid, tasks, rollout_map[vid])
                tasks["_coverage_audit"] = audit
                total = sum(
                    len(t) for k, t in tasks.items()
                    if not k.startswith("_") and isinstance(t, list)
                )
                if not audit["passed"]:
                    logger.warning(
                        f"  [{vid}] Task audit issues: "
                        f"missing={audit['missing_expected_task_types']}, "
                        f"leakage={len(audit['question_answer_leakage'])}, "
                        f"minimality={len(audit['action_minimality_risks'])}"
                    )
                logger.info(f"  [{vid}] Tasks mined: {total}")
                return vid, tasks

        results = await asyncio.gather(*[process_video_pass3(v) for v in videos])
        all_tasks = {vid: tasks for vid, tasks in results if tasks is not None}

        # Save tasks and coverage audit
        tasks_path = TASKS_DIR / "all_tasks.json"
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        with open(tasks_path, "w") as f:
            json.dump(all_tasks, f, ensure_ascii=False)

        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_report = {
            vid: tasks.get("_coverage_audit", {})
            for vid, tasks in all_tasks.items()
        }
        with open(AUDIT_DIR / "task_coverage_report.json", "w") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

        total_tasks = sum(
            sum(len(t) for k, t in v.items() if not k.startswith("_") and isinstance(t, list))
            for v in all_tasks.values()
        )
        logger.info(f"Pass 3 complete: {total_tasks} total tasks")
    else:
        tasks_path = TASKS_DIR / "all_tasks.json"
        if tasks_path.exists():
            with open(tasks_path, "r") as f:
                all_tasks = json.load(f)
        else:
            logger.warning(f"No cached tasks at {tasks_path}")
            all_tasks = {}

    # =================================================================
    # PASS 4: Question-aware Forks
    # =================================================================
    if 4 not in skip_pass:
        from .pass4_forks import (
            build_compress_sample,
            build_compress_sample_c2,
            build_merge_compress_sample,
            build_pending_samples,
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
            observations = rollout.get("thinks", rollout.get("observations", []))
            snapshots = rollout["snapshots"]

            # --- Action priority enforcement (constraint #8) ---
            # Collect chunks that have user-interaction tasks so we can
            # skip compress/silent at those chunks. Priority order:
            #   post-recall > user response/recall > pending > compress > silent
            interaction_chunks = set()
            for task_type_key in ["response_from_frames", "response_from_memory",
                                   "compress_response", "unanswerable",
                                   "recall", "compress_recall"]:
                for task in tasks.get(task_type_key, []):
                    if task.get("question"):
                        interaction_chunks.add(task.get("ask_chunk"))
            for task in tasks.get("pending", []):
                if task.get("question"):
                    interaction_chunks.add(task.get("ask_chunk"))
                    # trigger_chunk also gets a response
                    interaction_chunks.add(task.get("trigger_chunk"))

            # 4a: Silent samples (subset, not all chunks)
            # Sample ~20% of silent chunks for training
            # Skip chunks that have user-interaction tasks (priority rule)
            silent_eligible = [c for c in range(rollout["num_chunks"])
                               if c not in interaction_chunks]
            selected_silent = random.sample(
                silent_eligible, min(len(silent_eligible) // 5, 15)
            ) if silent_eligible else []
            for chunk_idx in selected_silent:
                snapshot = snapshots.get(chunk_idx, snapshots.get(str(chunk_idx)))
                if snapshot and chunk_idx < len(observations):
                    sample = build_silent_sample(
                        chunk_idx, observations[chunk_idx]["think"],
                        snapshot, video_frames.get(vid, []),
                    )
                    sample["video_id"] = vid
                    all_samples.append(sample)

            # 4b: Compress samples (C1 + C2)
            # Skip chunks that have user-interaction tasks (user > compress)
            for event in rollout["compression_events"]:
                chunk_idx = event["trigger_chunk"]
                if chunk_idx in interaction_chunks:
                    continue  # User interaction takes priority over compress
                snapshot = snapshots.get(chunk_idx, snapshots.get(str(chunk_idx)))
                if snapshot and chunk_idx < len(observations):
                    # C1: system-specified range
                    sample_c1 = build_compress_sample(
                        chunk_idx, observations[chunk_idx]["think"],
                        snapshot, event, frame_paths=video_frames.get(vid, []),
                    )
                    sample_c1["video_id"] = vid
                    all_samples.append(sample_c1)

                    # C2: no range specified, model self-selects
                    sample_c2 = build_compress_sample_c2(
                        chunk_idx, observations[chunk_idx]["think"],
                        snapshot, event, frame_paths=video_frames.get(vid, []),
                    )
                    sample_c2["video_id"] = vid
                    all_samples.append(sample_c2)

            # 4b2: Merge-compress samples (second-level compression)
            # Generate from rollout when compressed_segments would exceed limit
            final_mem = rollout.get("final_memory", {})
            comp_segs = final_mem.get("compressed_segments", [])
            if len(comp_segs) >= 2:
                # Create merge samples from adjacent segment pairs
                for seg_idx in range(len(comp_segs) - 1):
                    seg_a = comp_segs[seg_idx]
                    seg_b = comp_segs[seg_idx + 1]
                    # Use a chunk near the midpoint of the video
                    merge_chunk = min(rollout["num_chunks"] - 1,
                                      seg_b["time_range"][1] // AGENT_CHUNK_SEC)
                    merge_snapshot = snapshots.get(merge_chunk, snapshots.get(str(merge_chunk)))
                    if merge_snapshot and merge_chunk < len(observations):
                        combined = f'{seg_a["text"]} {seg_b["text"]}'
                        words = combined.split()
                        merged_text = " ".join(words[:200]) if len(words) > 200 else combined
                        merge_sample = build_merge_compress_sample(
                            merge_snapshot, seg_a, seg_b, merged_text,
                            merge_chunk, observations[merge_chunk]["think"],
                            frame_paths=video_frames.get(vid, []),
                        )
                        merge_sample["video_id"] = vid
                        all_samples.append(merge_sample)

            # 4c: Response samples
            for task_type in ["response_from_frames", "response_from_memory", "compress_response", "unanswerable"]:
                for task in tasks.get(task_type, []):
                    if not task.get("question"):
                        continue
                    ask_chunk = task["ask_chunk"]
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot:
                        sample = await build_response_sample(
                            task, snapshot, observations, client, vid,
                            frame_paths=video_frames.get(vid, []),
                        )
                        if sample:
                            sample["video_id"] = vid
                            all_samples.append(sample)

            # 4d: Pending/event-watch samples
            for task in tasks.get("pending", []):
                if not task.get("question"):
                    continue
                samples = await build_pending_samples(
                    task, snapshots, observations, client, vid,
                    frame_paths=video_frames.get(vid, []),
                )
                if samples:
                    for s in samples:
                        s["video_id"] = vid
                    all_samples.extend(samples)

            # 4e: Recall samples
            for task_type in ["recall", "compress_recall"]:
                for task in tasks.get(task_type, []):
                    if not task.get("question"):
                        continue
                    ask_chunk = task["ask_chunk"]
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot:
                        samples = await build_recall_sample(
                            task, snapshot, observations, client, vid,
                            frame_paths=video_frames.get(vid, []),
                        )
                        if samples:
                            for s in samples:
                                s["video_id"] = vid
                            all_samples.extend(samples)

        # Assign sample_id, phase, and video_path to all samples
        vid_path_map = {v["video_id"]: v.get("video_path", "") for v in videos}
        for i, s in enumerate(all_samples):
            vid = s.get("video_id", "unk")
            chunk = s.get("chunk_idx", 0)
            stype = s.get("sample_type", "unk")
            s["sample_id"] = f"{vid}_t{chunk}_{stype}_{i}"
            s["phase"] = assign_phase(s)
            s["video_path"] = vid_path_map.get(vid, "")

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
        if samples_path.exists():
            with open(samples_path, "r") as f:
                for line in f:
                    all_samples.append(json.loads(line))
        else:
            logger.warning(f"No cached samples at {samples_path}, Pass 5 will run on empty set")

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
