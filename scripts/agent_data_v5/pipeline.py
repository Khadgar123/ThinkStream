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
    MAX_COMPRESSED_SEGMENTS,
    MAX_SAMPLES_PER_VIDEO,
    PASS_CONFIG,
    PHASE_CONFIG,
    ROLLOUT_DIR,
    SAMPLES_DIR,
    TASKS_DIR,
    VISUAL_WINDOW_CHUNKS,
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
    elif sample_type in ("recall_query", "recall_response",
                         "proactive_recall_query", "proactive_recall_silent"):
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
            safe_concurrency_for_pass("pass3_tasks"),
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
        total_chunks = sum(v["num_chunks"] for v in videos)
        logger.info(f"PASS 1-A: {total_chunks} chunks, concurrency={pass1a_conc}")

        async def process_video_1a(video):
            vid = video["video_id"]
            cached = load_1a(vid)
            if cached:
                logger.info(f"  [{vid}] 1-A cached ({len(cached)} chunks)")
                return vid, cached
            captions = await run_pass1a(
                video_id=vid,
                frame_paths=video_frames.get(vid, []),
                num_chunks=video["num_chunks"],
                client=client,
                semaphore=chunk_semaphore,
            )
            save_1a(vid, captions)
            return vid, captions

        results = await asyncio.gather(*[process_video_1a(v) for v in videos])
        evidence_1a_map = {vid: caps for vid, caps in results}
        logger.info(f"Pass 1-A complete: {len(evidence_1a_map)} videos")
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

        async def process_video_1b(video):
            vid = video["video_id"]
            cached = load_1b(vid)
            if cached:
                logger.info(f"  [{vid}] 1-B cached")
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
            return vid, enriched

        results = await asyncio.gather(*[process_video_1b(v) for v in videos])
        evidence_map = {vid: ev for vid, ev in results if ev is not None}
        logger.info(f"Pass 1-B complete: {len(evidence_map)} videos")
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
                tasks = await run_pass3(vid, evidence_map[vid], rollout_map[vid], client,
                                        frame_paths=video_frames.get(vid))
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
            build_per_timestep_messages,
            simulate_recall_result,
            _generate_response_text,
            _generate_recall_texts,
        )
        from .pass3_tasks import extract_keywords, keyword_overlap

        logger.info("=" * 60)
        logger.info("PASS 4: Per-Timestep Sample Generation (single-turn messages)")
        logger.info("=" * 60)

        all_samples = []

        for v in videos:
            vid = v["video_id"]
            vpath = v.get("video_path", "")
            if vid not in all_tasks or vid not in rollout_map:
                continue

            rollout = rollout_map[vid]
            tasks = all_tasks[vid]
            observations = rollout.get("thinks", rollout.get("observations", []))
            snapshots = rollout["snapshots"]
            compression_events = rollout["compression_events"]

            # --- Build task/compress/pending lookup tables ---
            compress_at = {e["trigger_chunk"]: e for e in compression_events}
            task_at = {}  # chunk_idx -> (task_type, task)
            skip_task_keys = {"_", "compress", "pending"}  # handled separately
            # Shuffle task type iteration order to avoid bias when multiple
            # types compete for the same chunk (first-come-first-served).
            task_type_keys = [k for k in tasks if isinstance(tasks[k], list)
                              and not any(k.startswith(s) for s in skip_task_keys)]
            random.shuffle(task_type_keys)
            for task_type_key in task_type_keys:
                for task in tasks[task_type_key]:
                    if task.get("question") and task.get("ask_chunk") not in task_at:
                        task_at[task["ask_chunk"]] = (task_type_key, task)

            # Pending tasks: active from ask_chunk to trigger_chunk
            pending_active = {}  # chunk_idx -> pending task
            for pt in tasks.get("pending", []):
                if pt.get("question") and pt.get("ask_chunk") is not None:
                    ask = pt["ask_chunk"]
                    trigger = pt["trigger_chunk"]
                    for c in range(ask, trigger + 1):
                        if c not in pending_active:
                            pending_active[c] = pt

            interaction_chunks = set(task_at.keys())
            vid_samples = []  # collect all, then budget-downsample

            for chunk_idx in range(rollout["num_chunks"]):
                if chunk_idx >= len(observations):
                    continue
                snapshot = snapshots.get(chunk_idx, snapshots.get(str(chunk_idx)))
                if not snapshot:
                    continue

                think_text = observations[chunk_idx]["think"]
                has_question = chunk_idx in task_at
                has_compress = chunk_idx in compress_at and chunk_idx not in interaction_chunks
                is_pending_start = (chunk_idx in pending_active
                                    and chunk_idx == pending_active[chunk_idx].get("ask_chunk"))
                is_pending_trigger = (chunk_idx in pending_active
                                      and chunk_idx == pending_active[chunk_idx].get("trigger_chunk"))
                is_pending_mid = (chunk_idx in pending_active
                                  and not is_pending_start and not is_pending_trigger)

                # ── Determine action + build output + suffix ──
                if has_question:
                    task_type, task = task_at[chunk_idx]
                    vid_frames = video_frames.get(vid)
                    if task["gold_action"] == "response":
                        resp = await _generate_response_text(task, snapshots, observations, client, vid,
                                                              frame_paths=vid_frames)
                        if not resp:
                            continue
                        output = (f"<think>{think_text}</think>"
                                  f"<action>response</action><response>{resp}</response>")
                        sample = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, output,
                            user_text_suffix=task["question"],
                        )
                        sample["sample_type"] = "response"
                        sample["video_id"] = vid
                        sample["metadata"] = {"task_type": task_type, "gold_action": "response",
                                              "gold_answer": task.get("gold_answer", "")}
                        vid_samples.append(sample)

                    elif task["gold_action"] == "recall":
                        query_json, resp, recall_result = await _generate_recall_texts(
                            task, snapshots, observations, client, vid,
                            frame_paths=vid_frames)
                        if not query_json:
                            continue
                        # Sample 1: recall query
                        output1 = (f"<think>{think_text}</think>"
                                   f"<action>recall</action>"
                                   f'<query>{json.dumps(query_json, ensure_ascii=False)}</query>')
                        s1 = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, output1,
                            user_text_suffix=task["question"],
                        )
                        s1["sample_type"] = "recall_query"
                        s1["video_id"] = vid
                        s1["metadata"] = {"task_type": task_type, "gold_action": "recall",
                                          "gold_answer": task.get("gold_answer", "")}
                        vid_samples.append(s1)

                        # Sample 2: post-recall response (no think)
                        # The answer IS in the past (Pass3 verified), so model
                        # should always attempt to respond:
                        # - recall succeeded → confident response
                        # - recall failed → uncertain response (answer exists
                        #   but retrieval missed it)
                        is_failed = recall_result.get("noise_level") in ("distractor", "failure")
                        if is_failed:
                            resp = "I could not find enough evidence to answer confidently."
                        recall_suffix = (
                            f'<pending since="{chunk_idx * AGENT_CHUNK_SEC}">{task["question"]}</pending>\n'
                            f'<recall_result>{recall_result.get("text_content", "")}</recall_result>'
                        )
                        recalled_frames = None
                        if recall_result.get("returned_chunks"):
                            rc = recall_result["returned_chunks"]
                            recalled_frames = {
                                "time_range": [rc[0] * AGENT_CHUNK_SEC, (rc[-1] + 1) * AGENT_CHUNK_SEC],
                                "n_frames": len(rc) * 2,
                            }

                        output2 = f"<action>response</action><response>{resp or ''}</response>"
                        s2 = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, output2,
                            user_text_suffix=recall_suffix,
                            recalled_frames=recalled_frames,
                        )
                        s2["sample_type"] = "recall_response"
                        s2["video_id"] = vid
                        s2["metadata"] = {"task_type": task_type, "gold_action": "response",
                                          "recall_noise": recall_result.get("noise_level")}
                        vid_samples.append(s2)

                elif is_pending_start:
                    # User asks event-watch question → model outputs silent
                    pt = pending_active[chunk_idx]
                    output = f"<think>{think_text}</think><action>silent</action>"
                    sample = build_per_timestep_messages(
                        snapshot, chunk_idx, vpath, output,
                        user_text_suffix=pt["question"],
                    )
                    sample["sample_type"] = "silent"
                    sample["video_id"] = vid
                    sample["metadata"] = {"task_type": "pending_start",
                                          "pending_question": pt["question"]}
                    vid_samples.append(sample)

                elif is_pending_trigger:
                    # Event happened → model responds to pending question
                    pt = pending_active[chunk_idx]
                    resp_text = pt.get("event", "Event observed.")
                    # Inject pending into snapshot so model sees it
                    snap_with_pending = dict(snapshot)
                    snap_with_pending["pending_questions"] = [{
                        "question": pt["question"],
                        "since_chunk": pt["ask_chunk"],
                    }]
                    output = (f"<think>{think_text}</think>"
                              f"<action>response</action>"
                              f"<response>{resp_text}</response>")
                    sample = build_per_timestep_messages(
                        snap_with_pending, chunk_idx, vpath, output,
                    )
                    sample["sample_type"] = "response"
                    sample["video_id"] = vid
                    sample["metadata"] = {"task_type": "pending_response",
                                          "gold_action": "response",
                                          "pending_question": pt["question"],
                                          "event": pt.get("event", "")}
                    vid_samples.append(sample)

                elif is_pending_mid and chunk_idx == pending_active[chunk_idx].get("mid_chunk"):
                    # Mid-point silent with pending visible
                    pt = pending_active[chunk_idx]
                    snap_with_pending = dict(snapshot)
                    snap_with_pending["pending_questions"] = [{
                        "question": pt["question"],
                        "since_chunk": pt["ask_chunk"],
                    }]
                    output = f"<think>{think_text}</think><action>silent</action>"
                    sample = build_per_timestep_messages(
                        snap_with_pending, chunk_idx, vpath, output,
                    )
                    sample["sample_type"] = "silent"
                    sample["video_id"] = vid
                    sample["metadata"] = {"task_type": "pending_silent",
                                          "pending_question": pt["question"]}
                    vid_samples.append(sample)

                elif has_compress:
                    event = compress_at[chunk_idx]
                    summary = event["summary"]
                    tr = summary.get("time_range", [0, 0])
                    compress_output = (f"<think>{think_text}</think>"
                                       f"<action>compress</action>"
                                       f'<summary>{json.dumps(summary, ensure_ascii=False)}</summary>')
                    compress_meta_base = {
                        "gold_action": "compress",
                        "compressed_range": tr,
                        "compressed_chunks": event.get("compressed_thinks_chunks", []),
                        "has_visual_context": event.get("has_visual_context", False),
                    }

                    # C1: system trigger with specified range
                    c1_trigger = f'<compress_trigger range="{tr[0]}-{tr[1]}"/>'
                    c1 = build_per_timestep_messages(
                        snapshot, chunk_idx, vpath, compress_output,
                        user_text_suffix=c1_trigger,
                    )
                    c1["sample_type"] = "compress"
                    c1["video_id"] = vid
                    c1["metadata"] = {**compress_meta_base, "phase": "C1"}
                    vid_samples.append(c1)

                    # C2: system trigger WITHOUT range
                    c2_trigger = "<compress_trigger/>"
                    c2 = build_per_timestep_messages(
                        snapshot, chunk_idx, vpath, compress_output,
                        user_text_suffix=c2_trigger,
                    )
                    c2["sample_type"] = "compress"
                    c2["video_id"] = vid
                    c2["metadata"] = {**compress_meta_base, "phase": "C2"}
                    vid_samples.append(c2)

                    # Merge compress: if this compression would push segments
                    # over MAX_COMPRESSED_SEGMENTS, the system merges the two
                    # oldest. Generate a training sample for this merge.
                    n_segs = len(snapshot.get("compressed_segments", []))
                    if n_segs >= MAX_COMPRESSED_SEGMENTS:
                        segs = snapshot["compressed_segments"]
                        seg_a, seg_b = segs[0], segs[1]
                        tr_a = seg_a.get("time_range", [0, 0])
                        tr_b = seg_b.get("time_range", [0, 0])
                        merged_text = f'{seg_a["text"]} {seg_b["text"]}'
                        # Truncate merged text (same as rollout logic)
                        words = merged_text.split()
                        if len(words) > 60:
                            merged_text = " ".join(words[:60])
                        merged_summary = {
                            "time_range": [tr_a[0], tr_b[1]],
                            "text": merged_text,
                        }
                        merge_trigger = (
                            f'<merge_compress_trigger segments='
                            f'"{tr_a[0]}-{tr_a[1]},{tr_b[0]}-{tr_b[1]}"/>'
                        )
                        merge_output = (
                            f"<think>{think_text}</think>"
                            f"<action>compress</action>"
                            f'<summary>{json.dumps(merged_summary, ensure_ascii=False)}</summary>'
                        )
                        mc = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, merge_output,
                            user_text_suffix=merge_trigger,
                        )
                        mc["sample_type"] = "compress"
                        mc["video_id"] = vid
                        mc["metadata"] = {
                            "gold_action": "compress",
                            "task_type": "merge_compress",
                            "phase": "C1",
                            "compressed_range": [tr_a[0], tr_b[1]],
                            "source_segments": [tr_a, tr_b],
                        }
                        vid_samples.append(mc)


                elif (not has_question and not has_compress
                      and snapshot.get("compressed_segments")
                      and chunk_idx > VISUAL_WINDOW_CHUNKS + 5):
                    # Proactive recall: no user question, but current think
                    # has keyword overlap with a compressed segment → model
                    # should recall to reconnect with historical context.
                    #
                    # Trigger condition is VERIFIABLE (keyword overlap), not
                    # random. Small volume (~few per video) seeds the behavior
                    # so RL can refine timing; without this, RL would need to
                    # discover proactive recall from scratch.
                    think_kw = extract_keywords(think_text)
                    best_seg = None
                    best_overlap = 0.0
                    for seg in snapshot["compressed_segments"]:
                        ov = keyword_overlap(seg["text"], think_kw)
                        if ov > best_overlap:
                            best_overlap = ov
                            best_seg = seg
                    # Threshold: at least 30% keyword overlap to be meaningful
                    if best_seg and best_overlap >= 0.3:
                        target_range = best_seg.get("time_range", [0, 0])
                        ev_start = int(target_range[0] // AGENT_CHUNK_SEC)
                        ev_end = int(target_range[1] // AGENT_CHUNK_SEC)
                        evidence_chunks = list(range(ev_start, ev_end))

                        query_json = {
                            "query": " ".join(think_kw[:5]),
                            "time_range": f"{int(target_range[0])}-{int(target_range[1])}",
                        }

                        # Sample 1: proactive recall query
                        output_pr = (f"<think>{think_text}</think>"
                                     f"<action>recall</action>"
                                     f'<query>{json.dumps(query_json, ensure_ascii=False)}</query>')
                        s_pr = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, output_pr,
                        )
                        s_pr["sample_type"] = "proactive_recall_query"
                        s_pr["video_id"] = vid
                        s_pr["metadata"] = {"task_type": "proactive_recall",
                                            "gold_action": "recall",
                                            "trigger": "keyword_overlap",
                                            "overlap": round(best_overlap, 2),
                                            "target_range": target_range}
                        vid_samples.append(s_pr)

                        # Sample 2: post-recall silent (no user to answer)
                        recall_result = simulate_recall_result(
                            {"ask_chunk": chunk_idx, "gold_answer": "",
                             "evidence_chunks": evidence_chunks},
                            snapshot, observations, chunk_idx, query_json,
                        )
                        recall_suffix = (
                            f'<recall_result>{recall_result.get("text_content", "")}</recall_result>'
                        )
                        recalled_frames = None
                        if recall_result.get("returned_chunks"):
                            rc = recall_result["returned_chunks"]
                            recalled_frames = {
                                "time_range": [rc[0] * AGENT_CHUNK_SEC, (rc[-1] + 1) * AGENT_CHUNK_SEC],
                                "n_frames": len(rc) * 2,
                            }
                        output_pr2 = "<action>silent</action>"
                        s_pr2 = build_per_timestep_messages(
                            snapshot, chunk_idx, vpath, output_pr2,
                            user_text_suffix=recall_suffix,
                            recalled_frames=recalled_frames,
                        )
                        s_pr2["sample_type"] = "proactive_recall_silent"
                        s_pr2["video_id"] = vid
                        s_pr2["metadata"] = {"task_type": "proactive_recall",
                                             "gold_action": "silent",
                                             "recall_noise": recall_result.get("noise_level"),
                                             "target_range": target_range}
                        vid_samples.append(s_pr2)
                    # else: overlap < 0.3, fall through to silent

                else:
                    # Silent — subsample ~20%
                    if random.random() > 0.2:
                        continue
                    output = f"<think>{think_text}</think><action>silent</action>"
                    sample = build_per_timestep_messages(
                        snapshot, chunk_idx, vpath, output,
                    )
                    sample["sample_type"] = "silent"
                    sample["video_id"] = vid
                    vid_samples.append(sample)

            # --- Per-video budgeted downsample ---
            # Collect all, then cap with type-stratified selection so no
            # single type (especially compress) dominates.
            if len(vid_samples) > MAX_SAMPLES_PER_VIDEO:
                from collections import defaultdict
                by_type = defaultdict(list)
                for s in vid_samples:
                    by_type[s.get("sample_type", "unknown")].append(s)
                n_types = max(len(by_type), 1)
                per_type_budget = MAX_SAMPLES_PER_VIDEO // n_types
                kept = set()  # track indices for O(1) lookup
                kept_list = []
                for stype, samples_of_type in by_type.items():
                    take = min(len(samples_of_type), per_type_budget)
                    if take < len(samples_of_type):
                        # Evenly spaced selection across the list (timeline order)
                        step = len(samples_of_type) / take
                        indices = [int(i * step) for i in range(take)]
                        selected = [samples_of_type[j] for j in indices]
                    else:
                        selected = samples_of_type
                    for s in selected:
                        kept.add(id(s))
                        kept_list.append(s)
                # Fill remaining budget
                surplus = MAX_SAMPLES_PER_VIDEO - len(kept_list)
                if surplus > 0:
                    remaining = [s for s in vid_samples if id(s) not in kept]
                    random.shuffle(remaining)
                    kept_list.extend(remaining[:surplus])
                vid_samples = kept_list
                logger.info(f"  [{vid}] {len(vid_samples)} samples "
                            f"(capped from {sum(len(v) for v in by_type.values())}, "
                            f"{n_types} types)")
            else:
                logger.info(f"  [{vid}] {len(vid_samples)} samples")
            all_samples.extend(vid_samples)

        # Assign sample_id, phase, and estimated token count
        for i, s in enumerate(all_samples):
            stype = s.get("sample_type", "unk")
            s["sample_id"] = f"{s.get('video_path', 'unk').split('/')[-1].replace('.mp4','')}_{stype}_{i}"
            s["phase"] = assign_phase(s)
            # Estimate text token count (chars / 3.5 for mixed EN/CJK).
            # Vision tokens are added by the processor and not counted here;
            # this estimate is for the max_sample_tokens pre-filter in SFT.
            text_len = 0
            for msg in s.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_len += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_len += len(item.get("text", ""))
            s["num_tokens"] = int(text_len / 3.5)

        # Save
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        samples_path = SAMPLES_DIR / "all_samples.jsonl"
        with open(samples_path, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"Pass 4 complete: {len(all_samples)} per-timestep samples")
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

    # Save stats
    stats_path = FINAL_DIR / "pipeline_stats.json"
    stats["train_count"] = len(train_samples)
    stats["val_count"] = len(val_samples)
    stats["test_count"] = len(test_samples)
    stats["phase_counts"] = phase_counts
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
