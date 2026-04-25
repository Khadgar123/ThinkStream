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

from .progress import ProgressTracker
from .config import (
    AGENT_CHUNK_SEC,
    ALL_DIRS,
    AUDIT_DIR,
    DATA_ROOT,
    FINAL_DIR,
    MAX_SAMPLES_PER_VIDEO,
    PASS_CONFIG,
    PHASE_CONFIG,
    ROLLOUT_DIR,
    VISUAL_WINDOW_CHUNKS,
    VLLM_MODEL,
    safe_concurrency_for_pass,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Assignment
# ---------------------------------------------------------------------------


def assign_phase(sample: Dict) -> str:
    """Assign training phase based on sample_type and sequence_type.

    v8.0: uses sample_type (silent/response/recall_query/recall_response/recall_silent)
    and sequence_type (immediate_response/recall_success/recall_fail_then_found/
    event_watch/multi_response) instead of old task_type.
    """
    sample_type = sample.get("sample_type", "")
    sequence_type = sample.get("sequence_type", "")
    prompt_type = sample.get("prompt_type", "")

    if sample_type == "compress":
        return "C1"

    if sample_type in ("recall_query", "recall_response", "recall_silent"):
        return "2"  # Phase 2: recall training

    if sample_type == "silent":
        if sequence_type in ("event_watch", "multi_response"):
            return "2"  # Phase 2: query-aware silent
        return "1"  # Phase 1: basic silent

    if sample_type == "response":
        if sequence_type in ("recall_fail_then_found",):
            return "2"  # Phase 2: recovery after recall fail
        if sequence_type in ("event_watch", "multi_response"):
            return "2"  # Phase 2: query-triggered response
        return "1"  # Phase 1: basic response

    return "5"  # Phase 5: mixed / unclassified


# ---------------------------------------------------------------------------
# Video Selection
# ---------------------------------------------------------------------------


def select_videos(
    video_root: str,
    num_videos: int = 300,
    min_duration: int = 60,
    max_duration: int = 400,
    seed: int = 42,
    catalog_csv: str = None,
) -> List[Dict]:
    """Select videos for data construction.

    Duration mix strategy (not just "longer is better"):
    - 60-120s  (30%): simple memory, learn basic think+response
    - 120-240s (60%): main force, all task types, moderate memory complexity
    - 240-400s (10%): deep compression (multi-merge), complex memory states

    Stratified by dataset source for content diversity.

    Sources (in order of preference):
    1. Existing registry file (cached from previous run)
    2. CSV catalog (pre-scanned, fast)
    3. Filesystem scan (slow fallback)
    """
    registry_path = DATA_ROOT / "video_registry.jsonl"
    if registry_path.exists():
        videos = []
        with open(registry_path, "r") as f:
            for line in f:
                videos.append(json.loads(line))
        # Validate registry against current parameters
        valid = [v for v in videos
                 if min_duration <= v.get("duration_sec", 0) <= max_duration]
        if len(valid) >= num_videos:
            logger.info(f"Loaded {len(valid)} videos from registry "
                        f"({len(videos) - len(valid)} filtered by duration).")
            return valid[:num_videos]
        else:
            logger.warning(
                f"Registry has {len(valid)} valid videos (need {num_videos}), "
                f"re-selecting from catalog."
            )

    # --- Source: CSV catalog (fast) ---
    if catalog_csv is None:
        # Auto-detect catalog in project data dir
        for candidate in [
            DATA_ROOT.parent / "video_catalog_30s_plus.csv",
            Path(video_root) / "video_catalog_30s_plus.csv",
        ]:
            if candidate.exists():
                catalog_csv = str(candidate)
                break

    videos = []
    if catalog_csv and Path(catalog_csv).exists():
        import csv
        with open(catalog_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                duration = float(row["duration_sec"])
                if min_duration <= duration <= max_duration:
                    videos.append({
                        "video_id": Path(row["video_path"]).stem,
                        "video_path": row["video_path"],
                        "duration_sec": duration,
                        "dataset": row.get("dataset", "unknown"),
                    })
        logger.info(f"Loaded {len(videos)} videos from catalog (duration {min_duration}-{max_duration}s)")
    else:
        # --- Source: Filesystem scan (slow fallback) ---
        import subprocess
        video_root = Path(video_root)
        for vpath in sorted(video_root.rglob("*.mp4")):
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries",
                     "format=duration", "-of", "csv=p=0", str(vpath)],
                    capture_output=True, text=True, timeout=10,
                )
                duration = float(result.stdout.strip())
            except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                continue
            if min_duration <= duration <= max_duration:
                videos.append({
                    "video_id": vpath.stem,
                    "video_path": str(vpath),
                    "duration_sec": duration,
                    "dataset": vpath.parent.name,
                })
        logger.info(f"Scanned {len(videos)} videos from {video_root}")

    if not videos:
        logger.error(f"No videos found with duration {min_duration}-{max_duration}s")
        return []

    # --- Duration-stratified selection ---
    # Split videos into duration buckets, then sample proportionally
    short = [v for v in videos if v["duration_sec"] < 120]          # 60-120s
    medium = [v for v in videos if 120 <= v["duration_sec"] < 240]  # 120-240s
    long = [v for v in videos if v["duration_sec"] >= 240]          # 240-400s

    # Target mix: 30% short, 60% medium, 10% long
    n_short = min(int(num_videos * 0.3), len(short))
    n_long = min(int(num_videos * 0.1), len(long))
    n_medium = min(num_videos - n_short - n_long, len(medium))
    # Fill any shortfall from medium
    n_medium += num_videos - n_short - n_medium - n_long

    logger.info(
        f"Duration mix target: {n_short} short (60-120s) + "
        f"{n_medium} medium (120-240s) + {n_long} long (240-400s)"
    )

    def _stratified_sample(pool, n, seed_val):
        """Sample n videos from pool, stratified by dataset source."""
        from collections import defaultdict
        groups = defaultdict(list)
        for v in pool:
            groups[v.get("dataset", "unknown")].append(v)
        # Shuffle within groups
        rng = random.Random(seed_val)
        for g in groups.values():
            rng.shuffle(g)
        # Round-robin across groups
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        result = []
        per_group = max(1, n // max(len(group_keys), 1))
        for key in group_keys:
            take = min(per_group, len(groups[key]), n - len(result))
            result.extend(groups[key][:take])
            if len(result) >= n:
                break
        if len(result) < n:
            remaining = [v for k in group_keys for v in groups[k] if v not in result]
            rng.shuffle(remaining)
            result.extend(remaining[:n - len(result)])
        return result[:n]

    random.seed(seed)
    selected = (
        _stratified_sample(short, n_short, seed)
        + _stratified_sample(medium, n_medium, seed + 1)
        + _stratified_sample(long, n_long, seed + 2)
    )

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
            safe_concurrency_for_pass("pass1a"),
            safe_concurrency_for_pass("pass1b"),
            safe_concurrency_for_pass("pass2_rollout"),
            safe_concurrency_for_pass("pass3a"),
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
    # =================================================================
    # PASS 1-A: Independent Chunk Annotation
    # =================================================================
    if 1 not in skip_pass:
        from .pass1a_evidence import load_1a, run_pass1a, save_1a

        logger.info("=" * 60)
        logger.info("PASS 1-A: Independent Chunk Annotation")
        logger.info("=" * 60)

        pass1a_conc = safe_concurrency_for_pass("pass1a")
        chunk_semaphore = asyncio.Semaphore(pass1a_conc)
        uncached_1a = [v for v in videos if not load_1a(v["video_id"])]
        tracker_1a = ProgressTracker("pass1a", len(uncached_1a), AUDIT_DIR)

        async def process_video_1a(video):
            vid = video["video_id"]
            cached = load_1a(vid)
            if cached:
                return vid, cached
            captions = await run_pass1a(
                video_id=vid,
                frame_paths=video_frames.get(vid, []),
                num_chunks=video["num_chunks"],
                client=client,
                semaphore=chunk_semaphore,
            )
            save_1a(vid, captions)
            n_ok = sum(1 for c in captions if c.get("parse_success"))
            await tracker_1a.record(success=n_ok > 0, video_id=vid, chunks=len(captions), parsed=n_ok)
            return vid, captions

        results = await asyncio.gather(*[process_video_1a(v) for v in videos])
        evidence_1a_map = {vid: caps for vid, caps in results}
        tracker_1a.summary()
    else:
        from .pass1a_evidence import load_1a
        evidence_1a_map = {}
        for v in videos:
            cached = load_1a(v["video_id"])
            if cached:
                evidence_1a_map[v["video_id"]] = cached

    # =================================================================
    # PASS 1-B: Entity Alignment + State Change Detection
    # =================================================================
    if 1 not in skip_pass:
        from .pass1b_enrich import load_1b, run_pass1b, save_1b

        logger.info("=" * 60)
        logger.info("PASS 1-B: Entity Alignment + State Changes")
        logger.info("=" * 60)

        pass1b_conc = safe_concurrency_for_pass("pass1b")
        pass1b_semaphore = asyncio.Semaphore(pass1b_conc)
        logger.info(f"PASS 1-B: {len(evidence_1a_map)} videos, concurrency={pass1b_conc}")

        uncached_1b = [v for v in videos if not load_1b(v["video_id"]) and v["video_id"] in evidence_1a_map]
        tracker_1b = ProgressTracker("pass1b", len(uncached_1b), AUDIT_DIR)

        async def process_video_1b(video):
            vid = video["video_id"]
            cached = load_1b(vid)
            if cached:
                return vid, cached
            if vid not in evidence_1a_map:
                return vid, None
            enriched = await run_pass1b(
                evidence=evidence_1a_map[vid],
                client=client,
                video_id=vid,
                semaphore=pass1b_semaphore,
            )
            save_1b(vid, enriched)
            n_sc = sum(1 for c in enriched if c.get("state_changes"))
            await tracker_1b.record(success=True, video_id=vid, state_changes=n_sc)
            return vid, enriched

        results = await asyncio.gather(*[process_video_1b(v) for v in videos])
        evidence_map = {vid: ev for vid, ev in results if ev is not None}
        tracker_1b.summary()
    else:
        from .pass1b_enrich import load_1b
        evidence_map = {}
        for v in videos:
            cached = load_1b(v["video_id"])
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

        uncached_p2 = [v for v in videos if not load_rollout(v["video_id"])]
        tracker_p2 = ProgressTracker("pass2", len(uncached_p2), AUDIT_DIR)
        # Per-chunk debug log: tail -f to see each chunk's think + memory state
        pass2_chunk_log = AUDIT_DIR / "pass2_chunks.jsonl"
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        pass2_chunk_log.write_text("")  # truncate on new run
        logger.info(f"PASS 2: per-chunk debug → tail -f {pass2_chunk_log}")

        async def process_video_pass2(video):
            vid = video["video_id"]
            cached = load_rollout(vid)
            if cached:
                return vid, cached

            async with semaphore:
                rollout = await run_pass2_single_video(
                    video_id=vid,
                    frame_paths=video_frames.get(vid, []),
                    num_chunks=video["num_chunks"],
                    client=client,
                    evidence=evidence_map.get(vid),
                    chunk_log_path=pass2_chunk_log,
                )
                save_rollout(vid, rollout)
                await tracker_p2.record(
                    success=True, video_id=vid,
                    thinks=len(rollout["thinks"]),
                    compressions=len(rollout["compression_events"]),
                )
                return vid, rollout

        results = await asyncio.gather(*[process_video_pass2(v) for v in videos])
        rollout_map = {vid: roll for vid, roll in results}
        tracker_p2.summary()

        # --- Compression statistics ---
        from .pass2_rollout import compute_compression_stats
        comp_stats = compute_compression_stats(rollout_map)
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_DIR / "compression_stats.json", "w") as f:
            json.dump(comp_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Compression stats saved to {AUDIT_DIR / 'compression_stats.json'}")
    else:
        from .pass2_rollout import load_rollout
        rollout_map = {}
        for v in videos:
            cached = load_rollout(v["video_id"])
            if cached:
                rollout_map[v["video_id"]] = cached

    # =================================================================
    # PASS 3-A: Task Card Generation + Verification
    # =================================================================
    if 3 not in skip_pass:
        from .pass3a_cards import generate_cards, verify_cards, save_cards, load_cards

        logger.info("=" * 60)
        logger.info("PASS 3-A: Task Card Generation")
        logger.info("=" * 60)

        pass3a_conc = safe_concurrency_for_pass("pass3a")
        semaphore_3a = asyncio.Semaphore(pass3a_conc)

        uncached_3a = [v for v in videos if not load_cards(v["video_id"]) and v["video_id"] in evidence_map]
        tracker_3a = ProgressTracker("pass3a", len(uncached_3a), AUDIT_DIR)

        async def process_video_3a(video):
            vid = video["video_id"]
            cached = load_cards(vid)
            if cached:
                return vid, cached
            if vid not in evidence_map:
                return vid, []
            async with semaphore_3a:
                cards = await generate_cards(vid, evidence_map[vid], client)
                # Verify each card independently (high concurrency within)
                cards = await verify_cards(vid, cards, evidence_map[vid], client)
                save_cards(vid, cards)
                await tracker_3a.record(success=len(cards) > 0, video_id=vid, n_cards=len(cards))
                return vid, cards

        results = await asyncio.gather(*[process_video_3a(v) for v in videos])
        cards_map = {vid: cards for vid, cards in results}
        tracker_3a.summary()
    else:
        from .pass3a_cards import load_cards
        cards_map = {}
        for v in videos:
            cached = load_cards(v["video_id"])
            if cached:
                cards_map[v["video_id"]] = cached

    # =================================================================
    # PASS 3-B: Placement + Trajectory Planning (LLM visibility)
    # =================================================================
    if 3 not in skip_pass:
        from .pass3b_placement import (
            compute_all_placements, plan_trajectories,
            save_placements, load_placements,
        )

        logger.info("=" * 60)
        logger.info("PASS 3-B: Placement + Trajectory Planning (LLM visibility)")
        logger.info("=" * 60)

        trajectories_map = {}

        async def process_video_3b(video):
            vid = video["video_id"]
            cached = load_placements(vid)
            if cached:
                return vid, cached
            if vid not in cards_map or vid not in rollout_map or vid not in evidence_map:
                return vid, None
            # LLM visibility checks inside (independent per card, high concurrency)
            placements = await compute_all_placements(
                cards_map[vid], rollout_map[vid], evidence_map[vid],
                client=client, video_id=vid,
            )
            vid_cards = {c["card_id"]: c for c in cards_map[vid]}
            nc = rollout_map[vid]["num_chunks"]
            trajectories = plan_trajectories(
                placements, cards_map=vid_cards,
                num_chunks=nc, evidence=evidence_map[vid])
            data = {"placements": placements, "trajectories": trajectories}
            save_placements(vid, data)
            logger.info(f"  [{vid}] 3-B: {len(placements)} placements → {len(trajectories)} trajectories")
            return vid, data

        results = await asyncio.gather(*[process_video_3b(v) for v in videos])
        for vid, data in results:
            if data is not None:
                trajectories_map[vid] = data

        logger.info(f"Pass 3-B complete: {sum(len(d.get('trajectories',[])) for d in trajectories_map.values())} trajectories")
    else:
        from .pass3b_placement import load_placements
        trajectories_map = {}
        for v in videos:
            cached = load_placements(v["video_id"])
            if cached:
                trajectories_map[v["video_id"]] = cached

    # =================================================================
    # PASS 3-C: Trajectory Sample Generation
    # =================================================================
    if 3 not in skip_pass:
        from .pass3c_samples import generate_trajectory_samples, save_samples

        logger.info("=" * 60)
        logger.info("PASS 3-C: Trajectory Sample Generation")
        logger.info("=" * 60)

        all_samples = []
        uncached_3c = [v for v in videos if v["video_id"] in trajectories_map]
        tracker_3c = ProgressTracker("pass3c", len(uncached_3c), AUDIT_DIR)

        async def process_video_3c(video):
            vid = video["video_id"]
            if vid not in trajectories_map or vid not in rollout_map or vid not in evidence_map:
                return vid, []
            traj_data = trajectories_map[vid]
            trajectories = traj_data.get("trajectories", [])
            if not trajectories:
                return vid, []

            vid_cards = {c["card_id"]: c for c in cards_map.get(vid, [])}

            # Trajectories within a video are independent (each starts
            # with empty queries_state) — run them concurrently.
            # Only placements WITHIN a trajectory must be sequential.
            traj_tasks = [
                generate_trajectory_samples(
                    trajectory=traj,
                    cards_map=vid_cards,
                    rollout=rollout_map[vid],
                    evidence=evidence_map[vid],
                    client=client,
                    video_id=vid,
                )
                for traj in trajectories
            ]
            traj_results = await asyncio.gather(*traj_tasks, return_exceptions=True)

            vid_samples = []
            for result in traj_results:
                if isinstance(result, Exception):
                    logger.error(f"  [{vid}] 3-C trajectory failed: {result}")
                    continue
                vid_samples.extend(result)

            for s in vid_samples:
                s["video_id"] = vid
                s["video_path"] = video.get("video_path", "")

            save_samples(vid, vid_samples)
            await tracker_3c.record(success=len(vid_samples) > 0, video_id=vid, n_samples=len(vid_samples))
            return vid, vid_samples

        # Videos are independent — run them all concurrently.
        # Client semaphore limits actual API calls in flight.
        results_3c = await asyncio.gather(*[process_video_3c(v) for v in videos])
        for vid, vid_samples in results_3c:
            all_samples.extend(vid_samples)

        tracker_3c.summary()
    else:
        from .pass3c_samples import load_samples
        all_samples = []
        for v in videos:
            cached = load_samples(v["video_id"])
            if cached:
                all_samples.extend(cached)

    # =================================================================
    # PASS 4: Verify + Filter
    # =================================================================
    from .pass4_verify import filter_samples, save_verified

    logger.info("=" * 60)
    logger.info("PASS 4: Verify + Filter")
    logger.info("=" * 60)

    passed_samples, stats = filter_samples(all_samples)
    logger.info(f"Verification: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1%})")
    logger.info(f"Fail reasons: {stats['fail_reasons']}")
    logger.info(f"Action dist: {stats['action_distribution']}")
    logger.info(f"Difficulty dist: {stats['difficulty_distribution']}")
    logger.info(f"Trajectory check failures: {stats['trajectory_check_failures']}/{stats['trajectories']}")

    # Save verified samples per video
    verified_by_vid = {}
    for s in passed_samples:
        vid = s.get("video_id", "unknown")
        verified_by_vid.setdefault(vid, []).append(s)
    for vid, vid_samples in verified_by_vid.items():
        save_verified(vid, vid_samples, {"video_id": vid, "count": len(vid_samples)})

    # =================================================================
    # RENDER: Convert verified samples into SFT-ready format
    # =================================================================
    from .render_samples import render_video_samples

    logger.info("=" * 60)
    logger.info("RENDER: Building SFT-ready samples")
    logger.info("=" * 60)

    sft_samples = []
    per_video_stats = {}  # {vid: {count, families, seq_types, actions}}

    for vid, vid_samples in verified_by_vid.items():
        if vid not in rollout_map:
            logger.warning(f"  [{vid}] no rollout for render, skipping")
            continue
        v_info = next((v for v in videos if v["video_id"] == vid), {})
        video_path = v_info.get("video_path", "")
        vid_cards = {c["card_id"]: c for c in cards_map.get(vid, [])}
        rendered = render_video_samples(
            vid_samples, rollout_map[vid], video_path, vid, vid_cards)

        # Enforce MAX_SAMPLES_PER_VIDEO cap
        if MAX_SAMPLES_PER_VIDEO > 0 and len(rendered) > MAX_SAMPLES_PER_VIDEO:
            logger.warning(
                f"  [{vid}] {len(rendered)} samples exceeds cap {MAX_SAMPLES_PER_VIDEO}, "
                f"truncating (keeping diverse sample types)")
            # Prioritize: response > recall > compress > silent
            priority = {"response": 0, "recall_query": 1, "recall_response": 1,
                        "compress": 2, "silent": 3, "recall_silent": 3}
            rendered.sort(key=lambda s: (
                priority.get(s.get("sample_type", "silent"), 4),
                s.get("chunk_idx", 0),
            ))
            rendered = rendered[:MAX_SAMPLES_PER_VIDEO]

        # Collect per-video distribution stats
        vid_families = {}
        vid_seq_types = {}
        vid_actions = {}
        for s in rendered:
            fam = s.get("metadata", {}).get("family", "?")
            vid_families[fam] = vid_families.get(fam, 0) + 1
            seq = s.get("sequence_type", "?")
            vid_seq_types[seq] = vid_seq_types.get(seq, 0) + 1
            act = s.get("action", "?")
            vid_actions[act] = vid_actions.get(act, 0) + 1

        per_video_stats[vid] = {
            "count": len(rendered),
            "families": vid_families,
            "sequence_types": vid_seq_types,
            "actions": vid_actions,
        }

        sft_samples.extend(rendered)

    logger.info(f"Rendered {len(sft_samples)} SFT samples from {len(verified_by_vid)} videos")

    # --- Distribution audit ---
    sample_counts = [v["count"] for v in per_video_stats.values()]
    if sample_counts:
        min_c, max_c = min(sample_counts), max(sample_counts)
        avg_c = sum(sample_counts) / len(sample_counts)
        logger.info(f"Per-video samples: min={min_c}, max={max_c}, avg={avg_c:.1f}")

        # Warn on extreme skew
        for vid, vs in per_video_stats.items():
            if vs["count"] < 3:
                logger.warning(f"  [{vid}] only {vs['count']} samples — underrepresented")
            if vs["count"] > MAX_SAMPLES_PER_VIDEO * 0.9 and MAX_SAMPLES_PER_VIDEO > 0:
                logger.warning(f"  [{vid}] {vs['count']} samples — near cap")

    # Global family distribution
    global_families = {}
    global_seq_types = {}
    global_base_roles = {}
    for s in sft_samples:
        fam = s.get("metadata", {}).get("family", "")
        if fam:
            global_families[fam] = global_families.get(fam, 0) + 1
        seq = s.get("sequence_type", "")
        global_seq_types[seq] = global_seq_types.get(seq, 0) + 1
        br = s.get("base_role", "")
        if br:
            global_base_roles[br] = global_base_roles.get(br, 0) + 1

    logger.info(f"Global family dist: {dict(sorted(global_families.items()))}")
    logger.info(f"Global seq_type dist: {dict(sorted(global_seq_types.items()))}")
    logger.info(f"Global base_role dist: {dict(sorted(global_base_roles.items()))}")

    # Warn if any expected family has <1% representation
    total_with_family = sum(global_families.values()) or 1
    for fam in ["F1", "F2", "E1", "E2", "S1"]:
        fam_count = global_families.get(fam, 0)
        fam_pct = fam_count / total_with_family * 100
        if fam_pct < 1.0:
            logger.warning(f"  Family {fam} underrepresented: {fam_count} ({fam_pct:.1f}%)")

    # Assign sample_id and phase AFTER render (render creates new dicts)
    for i, s in enumerate(sft_samples):
        s["sample_id"] = f"{s.get('video_id', 'unk')}_{s.get('action', 'unk')}_{i}"
        s["phase"] = assign_phase(s)

    # Use rendered samples for final output
    passed_samples = sft_samples

    # --- Final output ---
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Split by video for train/val/test (80/10/10)
    video_ids = list(set(s.get("video_id", "") for s in passed_samples))
    video_ids = [v for v in video_ids if v]  # remove empty
    random.seed(seed)
    random.shuffle(video_ids)
    n = len(video_ids)
    train_vids = set(video_ids[:int(n * 0.8)])
    val_vids = set(video_ids[int(n * 0.8):int(n * 0.9)])
    test_vids = set(video_ids[int(n * 0.9):])

    train_samples = [s for s in passed_samples if s.get("video_id") in train_vids]
    val_samples = [s for s in passed_samples if s.get("video_id") in val_vids]
    test_samples = [s for s in passed_samples if s.get("video_id") in test_vids]

    # Save all-in-one splits
    for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = FINAL_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} samples → {path}")

    # Save phase-specific train files (for SFT data_list.py)
    phase_map = {
        "1": "phase1_train.jsonl",
        "2": "phase2_train.jsonl",
        "C1": "c1_train.jsonl",
        "C2": "c2_train.jsonl",
    }
    phase_counts = {}
    for phase_key, filename in phase_map.items():
        phase_data = [s for s in train_samples if s.get("phase") == phase_key]
        path = FINAL_DIR / filename
        with open(path, "w") as f:
            for s in phase_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        phase_counts[phase_key] = len(phase_data)
        logger.info(f"  phase {phase_key}: {len(phase_data)} train samples → {path}")

    # Phase 5 = ALL train samples (mixed training, not a separate phase)
    p5_path = FINAL_DIR / "phase5_train.jsonl"
    with open(p5_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    phase_counts["5"] = len(train_samples)
    logger.info(f"  phase 5 (mixed): {len(train_samples)} train samples → {p5_path}")

    # Save comprehensive stats
    stats_path = FINAL_DIR / "pipeline_stats.json"
    stats["train_count"] = len(train_samples)
    stats["val_count"] = len(val_samples)
    stats["test_count"] = len(test_samples)
    stats["phase_counts"] = phase_counts
    stats["split_by_video"] = True
    stats["global_family_distribution"] = global_families
    stats["global_sequence_type_distribution"] = global_seq_types
    stats["global_base_role_distribution"] = global_base_roles
    stats["per_video_sample_counts"] = {
        "min": min(sample_counts) if sample_counts else 0,
        "max": max(sample_counts) if sample_counts else 0,
        "avg": round(sum(sample_counts) / len(sample_counts), 1) if sample_counts else 0,
        "total_videos": len(sample_counts),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save per-video distribution audit
    audit_path = AUDIT_DIR / "per_video_distribution.json"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(per_video_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Per-video distribution audit → {audit_path}")

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
