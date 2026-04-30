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
import os
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
    """Assign a per-category label used for diagnostic file splits.

    v11 (2026-04-27): training is single-stage SFT on the merged
    phase5_train.jsonl. The per-category labels below are NOT a training
    curriculum; they only feed `phase{1,2,C1,5}_train.jsonl` for
    per-category eval and ablation. The old "C2" label (model-self-pick
    range) was removed when SFT collapsed C1+C2 into one stage that
    always uses teacher gold range; range exploration moved to RL via
    `overflow_pen` reward.
    """
    sample_type = sample.get("sample_type", "")
    sequence_type = sample.get("sequence_type", "")
    prompt_type = sample.get("prompt_type", "")

    if sample_type == "compress":
        return "C1"  # diagnostic label: compress-trained samples

    if sample_type in ("recall_query", "recall_response", "recall_silent"):
        return "2"  # diagnostic label: recall-trained samples

    if sample_type == "silent":
        if sequence_type in ("event_watch", "multi_response"):
            return "2"  # query-aware silent
        return "1"  # basic silent

    if sample_type == "response":
        if sequence_type in ("recall_fail_then_found",):
            return "2"  # recovery after recall fail
        if sequence_type in ("event_watch", "multi_response"):
            return "2"  # query-triggered response
        return "1"  # basic response

    return "5"  # mixed / unclassified


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
    # v12.1 batch2: when THINKSTREAM_BATCH=batch2, extend the duration
    # ceiling to capture longer procedural / multi-event content (gives
    # more CR2/F5/F6 generation material). Also pick videos that aren't
    # already in the registry to keep batch2 disjoint from batch1.
    if os.environ.get("THINKSTREAM_BATCH", "").lower() == "batch2":
        max_duration = max(max_duration, 600)
        logger.info(
            f"BATCH2: extending max_duration {max_duration}s to capture "
            f"longer multi-event content for CR2/F5/F6 yield."
        )
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
    existing: List[Dict] = []
    existing_ids: set = set()
    if registry_path.exists():
        with open(registry_path, "r") as f:
            for line in f:
                v = json.loads(line)
                if min_duration <= v.get("duration_sec", 0) <= max_duration:
                    existing.append(v)
                    existing_ids.add(v["video_id"])
        if len(existing) >= num_videos:
            # Cache hit — return the prefix.
            logger.info(f"Loaded {len(existing)} videos from registry "
                        f"(cap to {num_videos}).")
            return existing[:num_videos]
        # Cache miss for the requested num_videos. Instead of throwing the
        # existing selection away (the old behavior), KEEP it and top up
        # from the catalog with NEW video_ids only. This lets you grow the
        # dataset across batches without re-picking the same videos:
        #   batch 1: --num_videos 312  → registry has 312
        #   batch 2: --num_videos 712  → keeps the 312, picks 400 new
        # The pipeline's per-stage cache (evidence_1a/, task_cards/, ...)
        # then hits for batch 1 and runs fresh for batch 2 only.
        logger.info(
            f"Registry has {len(existing)} valid videos, want {num_videos} — "
            f"will keep existing and top up {num_videos - len(existing)} new."
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

    # --- Cross-batch dedup ---
    # If the registry already had `existing` videos selected by a prior run,
    # exclude them from the candidate pool so we top up with NEW ids only.
    if existing_ids:
        before = len(videos)
        videos = [v for v in videos if v["video_id"] not in existing_ids]
        logger.info(
            f"Excluded {before - len(videos)} prior-batch videos from catalog "
            f"({len(videos)} candidates remain)."
        )

    # How many we still need to reach `num_videos`.
    n_needed = max(0, num_videos - len(existing))
    if n_needed == 0:
        return existing[:num_videos]
    if len(videos) < n_needed:
        logger.warning(
            f"Only {len(videos)} NEW candidates available, need {n_needed}. "
            f"Will return {len(existing) + len(videos)} videos total."
        )

    # --- Duration-stratified selection on the NEW pool ---
    # Split candidates into duration buckets, then sample proportionally
    # to the *needed* count (not original num_videos).
    short = [v for v in videos if v["duration_sec"] < 120]          # 60-120s
    medium = [v for v in videos if 120 <= v["duration_sec"] < 240]  # 120-240s
    long = [v for v in videos if v["duration_sec"] >= 240]          # 240-400s

    # Target mix: 30% short, 60% medium, 10% long, applied to n_needed.
    n_short = min(int(n_needed * 0.3), len(short))
    n_long = min(int(n_needed * 0.1), len(long))
    n_medium = min(n_needed - n_short - n_long, len(medium))
    # Fill any shortfall from medium
    n_medium += n_needed - n_short - n_medium - n_long

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
    new_selected = (
        _stratified_sample(short, n_short, seed)
        + _stratified_sample(medium, n_medium, seed + 1)
        + _stratified_sample(long, n_long, seed + 2)
    )

    # If duration-stratified quotas couldn't hit n_needed (e.g. the short
    # bucket was exhausted by an earlier batch and only medium/long
    # candidates remain), top up from any remaining videos. Without this,
    # cross-batch top-ups silently under-deliver after the first run.
    if len(new_selected) < n_needed:
        chosen = {v["video_id"] for v in new_selected}
        leftover = [v for v in videos if v["video_id"] not in chosen]
        rng = random.Random(seed + 3)
        rng.shuffle(leftover)
        shortfall = n_needed - len(new_selected)
        new_selected.extend(leftover[:shortfall])
        if shortfall > len(leftover):
            logger.warning(
                f"Catalog exhausted: only {len(new_selected)} new videos "
                f"available (wanted {n_needed})."
            )

    random.shuffle(new_selected)
    new_selected = new_selected[:n_needed]

    # Final list = prior batches' videos + this batch's new pick.
    # Order: existing first (so per-stage iteration matches batch order).
    selected = existing + new_selected

    # Save union registry. Pipeline cache (evidence_1a/, task_cards/, ...)
    # already keys off video_id, so cached batch-1 stages auto-hit and
    # only the new batch-2 ids need fresh API calls.
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        for v in selected:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    logger.info(
        f"Registry: {len(existing)} existing + {len(new_selected)} new "
        f"= {len(selected)} videos (min {min_duration}s)"
    )
    return selected


def extract_frames(video_path: str, output_dir: Path, fps: int = 1) -> List[str]:
    """Extract frames from video at given fps.

    Returns list of frame file paths in order.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%06d.jpg")

    # Check if already extracted at the requested fps.
    # Without fps validation, stale frames (e.g. 1fps cached on server)
    # would be silently reused even when config demands 2fps,
    # causing half the chunks to be lost (pass1 "serious bug").
    fps_marker = output_dir / ".fps"
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and fps_marker.exists() and fps_marker.read_text().strip() == str(fps):
        return [str(p) for p in existing]
    # Stale fps or no marker → purge and re-extract.
    for f in output_dir.glob("frame_*.jpg"):
        f.unlink()

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
    fps_marker.write_text(str(fps))
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

    # ── Per-pass VLLMClients (v11) ──
    # Each outer pass owns a dedicated client with its own semaphore — no
    # shared 1024-cap client, no semaphore-swap hacks. Each client keeps
    # the default 5400s (90min) timeout; do NOT shorten (it's the safety
    # net that prevented the orphan-cascade we hit at high concurrency).
    # Concurrency values come from PASS_CONFIG (see config.py).
    _client_kwargs = dict(api_base=api_base, model=model, timeout=5400.0)
    client_1a = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass1a"),
    )
    client_1b = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass1b"),
    )
    client_2 = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass2_rollout"),
    )
    client_3a = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass3a"),
    )
    client_3b = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass3b_visibility"),
    )
    client_3c = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass3c"),
    )
    logger.info(
        "VLLMClient caps: 1a=%d 1b=%d 2=%d 3a=%d 3b=%d 3c=%d (timeout=5400s)",
        client_1a.max_concurrent, client_1b.max_concurrent,
        client_2.max_concurrent, client_3a.max_concurrent,
        client_3b.max_concurrent, client_3c.max_concurrent,
    )

    # --- Video selection ---
    videos = select_videos(video_root, num_videos, seed=seed)
    logger.info(f"Pipeline starting with {len(videos)} videos")

    # --- Extract frames ---
    # v12.5 (2026-04-29): fps=2 + FRAMES_PER_CHUNK=2 → 1s/chunk (was fps=1 → 2s/chunk).
    from scripts.agent_data_v5.config import FPS, FRAMES_PER_CHUNK
    frames_dir = DATA_ROOT / "frames"
    video_frames = {}
    for v in videos:
        v_frames_dir = frames_dir / v["video_id"]
        frames = extract_frames(v["video_path"], v_frames_dir, fps=FPS)
        video_frames[v["video_id"]] = frames
        num_chunks = len(frames) // FRAMES_PER_CHUNK
        v["num_chunks"] = num_chunks

    logger.info(f"Frame extraction complete. {sum(len(f) for f in video_frames.values())} total frames.")

    # =================================================================
    # PASS 1 + 2: Streaming pipeline (1a → 1b → 2 per video, videos overlap)
    # =================================================================
    # When both pass 1 and 2 are needed, run them concurrently at video
    # granularity — a video enters pass2 as soon as its pass1b finishes,
    # without waiting for the whole batch to complete pass1.
    # This doubles effective wall-time concurrency from 128/256 to 1024.
    # -----------------------------------------------------------------
    run_1 = 1 not in skip_pass
    run_2 = 2 not in skip_pass

    if run_1:
        from .pass1a_evidence import load_1a, run_pass1a, save_1a
        from .pass1b_enrich import load_1b, run_pass1b, save_1b

        logger.info("=" * 60)
        logger.info("PASS 1-A: Independent Chunk Annotation")
        logger.info("=" * 60)

        uncached_1a = [v for v in videos if not load_1a(v["video_id"])]
        tracker_1a = ProgressTracker("pass1a", len(uncached_1a), AUDIT_DIR)
        VIDEO_CONCURRENCY_1A = 16
        video_semaphore_1a = asyncio.Semaphore(VIDEO_CONCURRENCY_1A)
        logger.info(f"PASS 1-A: {len(uncached_1a)} uncached videos, video_concurrency={VIDEO_CONCURRENCY_1A}, chunk_concurrency={client_1a.max_concurrent}")

        logger.info("=" * 60)
        logger.info("PASS 1-B: Entity Alignment + State Changes")
        logger.info("=" * 60)

        VIDEO_CONCURRENCY_1B = 16
        video_semaphore_1b = asyncio.Semaphore(VIDEO_CONCURRENCY_1B)
        tracker_1b = ProgressTracker("pass1b", len(videos), AUDIT_DIR)
        logger.info(f"PASS 1-B: video_concurrency={VIDEO_CONCURRENCY_1B}, chunk_concurrency={client_1b.max_concurrent}")
    else:
        from .pass1a_evidence import load_1a
        from .pass1b_enrich import load_1b
        evidence_1a_map = {}
        evidence_map = {}
        for v in videos:
            cached = load_1a(v["video_id"])
            if cached:
                evidence_1a_map[v["video_id"]] = cached
            cached = load_1b(v["video_id"])
            if cached:
                evidence_map[v["video_id"]] = cached

    if run_2:
        from .pass2_rollout import load_rollout, run_pass2_single_video, save_rollout

        logger.info("=" * 60)
        logger.info("PASS 2: Question-blind Streaming Rollout")
        logger.info("=" * 60)

        logger.info(f"PASS 2 safe concurrency={client_2.max_concurrent}")

        uncached_p2 = [v for v in videos if not load_rollout(v["video_id"])]
        tracker_p2 = ProgressTracker("pass2", len(uncached_p2), AUDIT_DIR)
        pass2_chunk_log = AUDIT_DIR / "pass2_chunks.jsonl"
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        pass2_chunk_log.write_text("")  # truncate on new run
        logger.info(f"PASS 2: per-chunk debug → tail -f {pass2_chunk_log}")
    else:
        from .pass2_rollout import load_rollout
        rollout_map = {}
        for v in videos:
            cached = load_rollout(v["video_id"])
            if cached:
                rollout_map[v["video_id"]] = cached

    if run_1 or run_2:
        async def process_video_pipeline(video):
            vid = video["video_id"]
            caps = None
            ev = None
            roll = None

            # --- Pass 1a ---
            if run_1:
                cached = load_1a(vid)
                if cached:
                    caps = cached
                else:
                    async with video_semaphore_1a:
                        caps = await run_pass1a(
                            video_id=vid,
                            frame_paths=video_frames.get(vid, []),
                            num_chunks=video["num_chunks"],
                            client=client_1a,
                        )
                    save_1a(vid, caps)
                    n_ok = sum(1 for c in caps if c.get("parse_success"))
                    await tracker_1a.record(success=n_ok > 0, video_id=vid, chunks=len(caps), parsed=n_ok)

            # --- Pass 1b ---
            if run_1 and caps:
                cached = load_1b(vid)
                if cached:
                    ev = cached
                else:
                    async with video_semaphore_1b:
                        ev = await run_pass1b(
                            evidence=caps,
                            client=client_1b,
                            video_id=vid,
                        )
                    save_1b(vid, ev)
                    n_sc = sum(1 for c in ev if c.get("state_changes"))
                    await tracker_1b.record(success=True, video_id=vid, state_changes=n_sc)

            # --- Pass 2 ---
            if run_2:
                cached = load_rollout(vid)
                if cached:
                    roll = cached
                else:
                    rollout = await run_pass2_single_video(
                        video_id=vid,
                        frame_paths=video_frames.get(vid, []),
                        num_chunks=video["num_chunks"],
                        client=client_2,
                        evidence=ev,
                        chunk_log_path=pass2_chunk_log,
                    )
                    save_rollout(vid, rollout)
                    await tracker_p2.record(
                        success=True, video_id=vid,
                        thinks=len(rollout["thinks"]),
                        compressions=len(rollout["compression_events"]),
                    )
                    roll = rollout

            return vid, caps, ev, roll

        results = await asyncio.gather(*[process_video_pipeline(v) for v in videos])

        if run_1:
            evidence_1a_map = {vid: caps for vid, caps, ev, roll in results}
            evidence_map = {vid: ev for vid, caps, ev, roll in results if ev is not None}
            tracker_1a.summary()
            tracker_1b.summary()
            from .cache_version import write_stage_version
            write_stage_version("1a")
            write_stage_version("1b")

        if run_2:
            rollout_map = {vid: roll for vid, caps, ev, roll in results if roll is not None}
            tracker_p2.summary()
            from .cache_version import write_stage_version
            write_stage_version("2")

            # --- Compression statistics ---
            from .pass2_rollout import compute_compression_stats
            comp_stats = compute_compression_stats(rollout_map)
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_DIR / "compression_stats.json", "w") as f:
                json.dump(comp_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Compression stats saved to {AUDIT_DIR / 'compression_stats.json'}")

    # =================================================================
    # PASS 3-A: Task Card Generation + Verification
    # =================================================================
    if 3 not in skip_pass:
        from .pass3a_cards import generate_cards, verify_cards, save_cards, load_cards

        logger.info("=" * 60)
        logger.info("PASS 3-A: Task Card Generation")
        logger.info("=" * 60)

        # Two-level concurrency for 3-A:
        #   - video_semaphore_3a limits how many videos enter at once
        #   - client_3a.semaphore (inside _call_one) limits actual API calls.
        # DO NOT reuse client_3a.semaphore here: generate_cards uses
        # asyncio.gather to fire per-family tasks, each of which acquires
        # client_3a.semaphore. If the outer video task already holds the same
        # semaphore, all family tasks deadlock (resource exhaustion — every
        # permit is held by a video task waiting on its own children).
        VIDEO_CONCURRENCY_3A = 8
        video_semaphore_3a = asyncio.Semaphore(VIDEO_CONCURRENCY_3A)

        uncached_3a = [v for v in videos if not load_cards(v["video_id"]) and v["video_id"] in evidence_map]
        tracker_3a = ProgressTracker("pass3a", len(uncached_3a), AUDIT_DIR)
        logger.info(f"PASS 3-A: {len(uncached_3a)} uncached videos, video_concurrency={VIDEO_CONCURRENCY_3A}, family_concurrency={client_3a.max_concurrent}")

        async def process_video_3a(video):
            vid = video["video_id"]
            cached = load_cards(vid)
            if cached:
                return vid, cached
            if vid not in evidence_map:
                return vid, []
            async with video_semaphore_3a:
                cards = await generate_cards(vid, evidence_map[vid], client_3a)
                # Verify each card independently (still bound by client_3a cap).
                cards = await verify_cards(vid, cards, evidence_map[vid], client_3a)
                save_cards(vid, cards)
                await tracker_3a.record(success=len(cards) > 0, video_id=vid, n_cards=len(cards))
                return vid, cards

        results = await asyncio.gather(*[process_video_3a(v) for v in videos])
        cards_map = {vid: cards for vid, cards in results}
        tracker_3a.summary()
        from .cache_version import write_stage_version
        write_stage_version("3a")
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
                client=client_3b, video_id=vid,
            )
            vid_cards = {c["card_id"]: c for c in cards_map[vid]}
            nc = rollout_map[vid]["num_chunks"]
            # Stable per-video seed so two runs with the same global `seed`
            # arg produce identical trajectories without coupling videos.
            traj_seed = seed * 10_000 + (hash(vid) & 0xFFFFFF)
            trajectories = plan_trajectories(
                placements, cards_map=vid_cards,
                num_chunks=nc, evidence=evidence_map[vid],
                seed=traj_seed)
            data = {"placements": placements, "trajectories": trajectories}
            save_placements(vid, data)
            logger.info(f"  [{vid}] 3-B: {len(placements)} placements → {len(trajectories)} trajectories")
            return vid, data

        results = await asyncio.gather(*[process_video_3b(v) for v in videos])
        for vid, data in results:
            if data is not None:
                trajectories_map[vid] = data

        logger.info(f"Pass 3-B complete: {sum(len(d.get('trajectories',[])) for d in trajectories_map.values())} trajectories")
        from .cache_version import write_stage_version
        write_stage_version("3b")
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
        from .pass3c_samples import generate_trajectory_samples, save_samples, load_samples

        logger.info("=" * 60)
        logger.info("PASS 3-C: Trajectory Sample Generation")
        logger.info("=" * 60)

        all_samples = []
        uncached_3c = [v for v in videos if v["video_id"] in trajectories_map]
        tracker_3c = ProgressTracker("pass3c", len(uncached_3c), AUDIT_DIR)

        # client_3c is dedicated to pass 3-C (no semaphore swap needed).
        logger.info(f"PASS 3-C: client_3c.max_concurrent={client_3c.max_concurrent}")

        async def process_video_3c(video):
            vid = video["video_id"]
            if vid not in trajectories_map or vid not in rollout_map or vid not in evidence_map:
                return vid, []

            # Use cached samples when available (avoids regenerating on re-runs)
            cached = load_samples(vid)
            if cached:
                for s in cached:
                    s.setdefault("video_id", vid)
                    s.setdefault("video_path", video.get("video_path", ""))
                await tracker_3c.record(success=len(cached) > 0, video_id=vid, n_samples=len(cached))
                return vid, cached

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
                    client=client_3c,
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

        # Videos are independent — run them all concurrently. client_3c's
        # internal semaphore limits actual API calls in flight.
        results_3c = await asyncio.gather(*[process_video_3c(v) for v in videos])
        for vid, vid_samples in results_3c:
            all_samples.extend(vid_samples)

        tracker_3c.summary()
        from .cache_version import write_stage_version
        write_stage_version("3c")
    else:
        from .pass3c_samples import load_samples
        all_samples = []
        for v in videos:
            cached = load_samples(v["video_id"])
            if cached:
                all_samples.extend(cached)

    # =================================================================
    # RENDER: Convert raw samples into SFT-ready format (BEFORE Pass4)
    # =================================================================
    # Render MUST happen before Pass4 because Pass4's semantic checks
    # depend on fields that render creates:
    #   - metadata.gold_answer (question-answer leakage check)
    #   - metadata.support_chunks (recall evidence reachability)
    #   - input.memory (compression ratio/provenance/retention checks)
    from .render_samples import render_video_samples

    logger.info("=" * 60)
    logger.info("RENDER: Building SFT-ready samples")
    logger.info("=" * 60)

    # Group raw samples by video for rendering
    raw_by_vid = {}
    for s in all_samples:
        vid = s.get("video_id", "unknown")
        raw_by_vid.setdefault(vid, []).append(s)

    rendered_samples = []
    for vid, vid_samples in raw_by_vid.items():
        if vid not in rollout_map:
            logger.warning(f"  [{vid}] no rollout for render, skipping")
            continue
        v_info = next((v for v in videos if v["video_id"] == vid), {})
        video_path = v_info.get("video_path", "")
        vid_cards = {c["card_id"]: c for c in cards_map.get(vid, [])}
        rendered = render_video_samples(
            vid_samples, rollout_map[vid], video_path, vid, vid_cards,
            all_frame_paths=video_frames.get(vid, []))
        rendered_samples.extend(rendered)

    logger.info(f"Rendered {len(rendered_samples)} samples from {len(raw_by_vid)} videos")

    # =================================================================
    # PASS 3-E: Verify + TAG (no drops — preserves trajectory continuity)
    #
    # v12.5 (2026-04-29): renamed from "PASS 4 Verify+Filter". The old
    # filter step dropped failures (~12% of samples), creating gaps in
    # the chunk timeline that downstream RL rollout couldn't replay.
    # New step tags every sample with verification.passed/.fail_reasons
    # but keeps all samples in the trajectory. Filtering decisions move
    # to consumer side (e.g. SFT trainer can weight by pass/fail).
    # =================================================================
    from .pass3e_verify import tag_samples, save_verified

    logger.info("=" * 60)
    logger.info("PASS 3-E: Verify + Tag (rendered samples — no drops)")
    logger.info("=" * 60)

    # v9.5: pass evidence_map so verify_support_chunks_have_evidence fires
    tagged_samples, stats = tag_samples(
        rendered_samples, evidence_map=evidence_map,
    )
    logger.info(f"Verification: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1%}) — "
                f"all {stats['total']} retained as tagged samples")
    logger.info(f"Fail reasons: {stats['fail_reasons']}")
    logger.info(f"Action dist: {stats['action_distribution']}")
    logger.info(f"Difficulty dist: {stats['difficulty_distribution']}")
    logger.info(f"Trajectory check failures: {stats['trajectory_check_failures']}/{stats['trajectories']}")

    # Save ALL tagged samples per video (not just passed)
    verified_by_vid = {}
    for s in tagged_samples:
        vid = s.get("video_id", "unknown")
        verified_by_vid.setdefault(vid, []).append(s)
    for vid, vid_samples in verified_by_vid.items():
        save_verified(vid, vid_samples, {"video_id": vid, "count": len(vid_samples)})

    # Carry forward as `passed_samples` for naming compat with old caps/split
    # logic below (the variable name is misleading post-v12.5 but keeping it
    # avoids touching ~250 lines of downstream code).
    passed_samples = tagged_samples

    # =================================================================
    # POST-FILTER: Enforce caps + distribution audit
    # =================================================================
    sft_samples = []
    per_video_stats = {}

    for vid, vid_samples in verified_by_vid.items():
        # Enforce MAX_SAMPLES_PER_VIDEO cap, family-aware.
        #
        # Old policy was a flat priority sort (response > recall > compress >
        # silent) and a global truncate. That works for action balance but
        # silently squeezes out tail families: F5/F6/N1 samples often live
        # *as silent* (event_watch wait_silent / pre-trigger silent), so any
        # video with 50+ non-silent samples lost all its tail-family signal
        # at the cap.
        #
        # New policy: round-robin across (family, action) buckets, pulling
        # one sample from each non-empty bucket each pass until the cap is
        # filled. Within a bucket, sort by chunk_idx so picks span the
        # timeline. Compress samples (which carry no family — sequence_type
        # is "base") get bucketed by action alone.
        if MAX_SAMPLES_PER_VIDEO > 0 and len(vid_samples) > MAX_SAMPLES_PER_VIDEO:
            from collections import defaultdict, OrderedDict

            # Bucket key: (family, action). Empty family for compress/base.
            buckets: "OrderedDict[tuple, list]" = OrderedDict()
            # Action priority for tie-breaking when buckets are equally
            # full: keep the high-information actions over silent.
            action_prio = {"response": 0, "recall_query": 1,
                           "recall_response": 1, "compress": 2,
                           "silent": 3, "recall_silent": 3}
            for s in vid_samples:
                fam = s.get("metadata", {}).get("family", "") or ""
                act = s.get("sample_type", "silent")
                buckets.setdefault((fam, act), []).append(s)

            # Order each bucket by chunk_idx so picks are temporally diverse.
            for k in buckets:
                buckets[k].sort(key=lambda s: s.get("chunk_idx", 0))

            # Round-robin draw, prioritizing action class on tie.
            kept: list = []
            bucket_keys = sorted(
                buckets.keys(),
                key=lambda k: (action_prio.get(k[1], 4), k[0]),
            )
            cursors = {k: 0 for k in bucket_keys}
            cap = MAX_SAMPLES_PER_VIDEO
            while len(kept) < cap:
                progressed = False
                for k in bucket_keys:
                    if cursors[k] < len(buckets[k]):
                        kept.append(buckets[k][cursors[k]])
                        cursors[k] += 1
                        progressed = True
                        if len(kept) >= cap:
                            break
                if not progressed:
                    break  # all buckets drained

            logger.warning(
                f"  [{vid}] {len(vid_samples)} samples > cap {cap}, "
                f"family-aware truncate to {len(kept)} "
                f"({len(buckets)} buckets)")
            vid_samples = kept

        # Collect per-video distribution stats
        vid_families = {}
        vid_seq_types = {}
        vid_actions = {}
        for s in vid_samples:
            fam = s.get("metadata", {}).get("family", "?")
            vid_families[fam] = vid_families.get(fam, 0) + 1
            seq = s.get("sequence_type", "?")
            vid_seq_types[seq] = vid_seq_types.get(seq, 0) + 1
            act = s.get("action", "?")
            vid_actions[act] = vid_actions.get(act, 0) + 1

        per_video_stats[vid] = {
            "count": len(vid_samples),
            "families": vid_families,
            "sequence_types": vid_seq_types,
            "actions": vid_actions,
        }

        sft_samples.extend(vid_samples)

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

    # Warn if any expected family has <1% representation.
    # v9.4: include reasoning families (CR1-4); 0.5% floor is more lenient
    # because reasoning cards are scarcer per video by design.
    total_with_family = sum(global_families.values()) or 1
    for fam, floor_pct in [
        ("F1", 1.0), ("F2", 1.0), ("E1", 1.0), ("E2", 1.0), ("S1", 1.0),
        ("CR1", 0.5), ("CR2", 0.5), ("CR3", 0.3), ("CR4", 0.5),
    ]:
        fam_count = global_families.get(fam, 0)
        fam_pct = fam_count / total_with_family * 100
        if fam_pct < floor_pct:
            logger.warning(
                f"  Family {fam} underrepresented: {fam_count} ({fam_pct:.1f}% < {floor_pct}%)"
            )

    # Assign sample_id and phase AFTER render (render creates new dicts)
    for i, s in enumerate(sft_samples):
        s["sample_id"] = f"{s.get('video_id', 'unk')}_{s.get('action', 'unk')}_{i}"
        s["phase"] = assign_phase(s)

    # PASS 3-D (IFD + submodular selection) was deleted — never ran in
    # production (PASS3D_TARGET=0 default). pass4 trajectory emission keeps
    # all samples instead. Recover from git history if needed.
    passed_samples = sft_samples

    # --- Final output ---
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # v12.0 SPLIT OVERHAUL — split-aware density + RL parity.
    # v11.5 was 80/10/10 train/val/test with 80/20 SFT/RL inside train.
    # With v12 density caps (~15 samples/video, was 32), v11.5 split would
    # leave RL with 746 samples ≈ 93 GRPO groups @ group_size=8 — too few
    # for stable trajectory-level credit assignment.
    #
    # v12.0:
    #   70/15/15 train/val/test  (more held-out for stable eval)
    #   Within train: 50/50 SFT/RL  (RL needs as much volume as SFT)
    #   val/test: thin to BENCHMARK density (~1 traj/video, ≤3 Q)
    #             so eval distribution matches OVO-Bench / StreamingBench
    #             rather than train-time density (avoids train→eval shift).
    #
    # Projected v12 final corpus on 311-video batch1 (avg 15 samples/video):
    #   SFT  train: 109 videos × 15 = ~1635 samples (vs v11.5 9900: 6× smaller)
    #   RL   train: 109 videos × 15 = ~1635 samples  (~204 GRPO groups @ G=8)
    #   val:  47 videos × ~5 thinned = ~235 samples (1 traj × 3 Q each)
    #   test: 47 videos × ~5 thinned = ~235 samples
    #
    # If batch1 SFT corpus < 2000 ends up too small, options:
    #   (a) generate batch2 (more videos, NOT denser per-video)
    #   (b) bump MAX_SAMPLES_PER_VIDEO 15 → 20 (mild density relaxation)
    #   (c) keep MAX_TRAJECTORIES_PER_VIDEO=5 but ease MAX_QUESTIONS=3→4
    video_ids = list(set(s.get("video_id", "") for s in passed_samples))
    video_ids = [v for v in video_ids if v]
    random.seed(seed)
    random.shuffle(video_ids)
    n = len(video_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_vids = set(video_ids[:train_end])
    val_vids = set(video_ids[train_end:val_end])
    test_vids = set(video_ids[val_end:])

    # Sub-split train 50/50 into SFT and RL (was 80/20). v12 needs RL parity
    # because GRPO trajectory-level credit assignment needs ≥150 unique
    # prompt groups. With G=8 rollouts, that's 1200+ samples minimum.
    train_vid_list = video_ids[:train_end]
    sft_split = int(len(train_vid_list) * 0.50)
    sft_train_vids = set(train_vid_list[:sft_split])
    rl_train_vids = set(train_vid_list[sft_split:])
    assert sft_train_vids.isdisjoint(rl_train_vids), \
        "SFT and RL train video sets must be disjoint"

    train_samples = [s for s in passed_samples if s.get("video_id") in train_vids]
    val_samples = [s for s in passed_samples if s.get("video_id") in val_vids]
    test_samples = [s for s in passed_samples if s.get("video_id") in test_vids]
    train_sft_samples = [s for s in train_samples if s.get("video_id") in sft_train_vids]
    train_rl_samples = [s for s in train_samples if s.get("video_id") in rl_train_vids]

    # v12.0 EVAL DENSITY THINNING — keep val/test at benchmark density (~1
    # q/min) so eval is a fair proxy for OVO-Bench / StreamingBench scores.
    # Thin to FIRST trajectory only per video (ascending trajectory_id).
    # This drops ~50-60% of val/test samples but keeps ALL silent/observation
    # context needed to evaluate timing.
    def _thin_to_first_trajectory(samples: list) -> list:
        from collections import defaultdict
        by_vid = defaultdict(list)
        for s in samples:
            by_vid[s.get("video_id", "")].append(s)
        kept = []
        for vid, vsamps in by_vid.items():
            traj_ids = sorted({s.get("trajectory_id", "0") for s in vsamps})
            if not traj_ids:
                kept.extend(vsamps)
                continue
            first_traj = traj_ids[0]
            # Keep samples from first trajectory + ALL silent samples without
            # a trajectory_id (base silents preserve the silent-decision
            # eval signal).
            for s in vsamps:
                tid = s.get("trajectory_id", "")
                if not tid or tid == first_traj or s.get("sequence_type") == "base":
                    kept.append(s)
        return kept

    val_samples = _thin_to_first_trajectory(val_samples)
    test_samples = _thin_to_first_trajectory(test_samples)
    logger.info(
        f"  eval-density thinned: val={len(val_samples)}, test={len(test_samples)} "
        f"(1 traj/video + base silents)"
    )

    splits_to_save = [
        ("train", train_samples),                # full union (backward compat)
        ("train_sft", train_sft_samples),        # SFT-only
        ("train_rl", train_rl_samples),          # RL-only (held out from SFT)
        ("val", val_samples),
        ("test", test_samples),
    ]
    for split_name, split_data in splits_to_save:
        path = FINAL_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} samples → {path}")

    # Per-category diagnostic split files (NOT a training curriculum).
    # v11 production trains on phase5_train.jsonl (= all train samples).
    # The 1/2/C1 splits exist only for per-category ablation eval.
    # "C2" was removed in v11 (was always empty: assign_phase never
    # returned "C2"; model-self-pick range moved to RL stage).
    phase_map = {
        "1": "phase1_train.jsonl",
        "2": "phase2_train.jsonl",
        "C1": "c1_train.jsonl",
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

    # Phase 5 = ALL train samples (the production SFT dataset).
    p5_path = FINAL_DIR / "phase5_train.jsonl"
    with open(p5_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    phase_counts["5"] = len(train_samples)
    logger.info(f"  phase 5 (mixed): {len(train_samples)} train samples → {p5_path}")

    # Save comprehensive stats
    stats_path = FINAL_DIR / "pipeline_stats.json"
    stats["train_count"] = len(train_samples)
    stats["train_sft_count"] = len(train_sft_samples)
    stats["train_rl_count"] = len(train_rl_samples)
    stats["val_count"] = len(val_samples)
    stats["test_count"] = len(test_samples)
    stats["video_counts"] = {
        "train": len(train_vids),
        "train_sft": len(sft_train_vids),
        "train_rl": len(rl_train_vids),
        "val": len(val_vids),
        "test": len(test_vids),
    }
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
    run_parser.add_argument(
        "--force_rerun_from",
        choices=["1a", "1b", "2", "3a", "3b", "3c", "4"],
        default=None,
        help="Delete cache for this stage and all downstream stages, forcing regeneration.",
    )

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
        if getattr(args, "force_rerun_from", None):
            from .cache_version import invalidate_stage_and_downstream
            logger.warning(
                f"--force_rerun_from {args.force_rerun_from} → "
                f"clearing cache from this stage downstream"
            )
            invalidate_stage_and_downstream(args.force_rerun_from)
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
