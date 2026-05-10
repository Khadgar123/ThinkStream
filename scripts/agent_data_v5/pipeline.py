"""
Main Pipeline Orchestrator for Agent Data v5.0

Usage:
    python -m scripts.agent_data_v5.pipeline run \
        --api_base http://AMD_IP:8000/v1 \
        --video_root /path/to/videos \
        --num_videos 300

    THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
    python -m scripts.agent_data_v5.pipeline run \
        --api_base http://AMD_IP:8000/v1 \
        --videos_jsonl data/agent_v5/batch2_videos.jsonl \
        --num_videos 500

    python -m scripts.agent_data_v5.pipeline stress_test \
        --api_base http://AMD_IP:8000/v1
"""

import argparse
import asyncio
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .progress import ProgressTracker
from .stable_hash import stable_seed
from .config import (
    AGENT_CHUNK_SEC,
    ALL_DIRS,
    AUDIT_DIR,
    BATCH_ID,
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

CANONICAL_FRAME_PROTOCOL = "video_meta"
CANONICAL_RENDER_LAYOUT = "standard_query_last"
CANONICAL_RENDER_DIRNAME = f"{CANONICAL_FRAME_PROTOCOL}_{CANONICAL_RENDER_LAYOUT}"


def _require_stage_cache(
    label: str,
    cache_map: Dict[str, object],
    videos: List[Dict],
) -> None:
    """Fail fast when a skipped stage lacks valid per-video cache.

    A skipped pass means the user expects current on-disk caches to be used.
    Continuing with partial caches silently shrinks the corpus and can produce
    empty final files, so missing/stale caches are treated as hard errors.
    """
    expected = [str(v.get("video_id", "")) for v in videos if v.get("video_id")]
    missing = [vid for vid in expected if vid not in cache_map]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"{label}: missing or stale cache for {len(missing)}/"
            f"{len(expected)} selected videos: {preview}{suffix}"
        )
    if expected and not cache_map:
        raise RuntimeError(f"{label}: no valid cache entries loaded")


def _require_nonempty(label: str, items) -> None:
    if not items:
        raise RuntimeError(f"{label}: empty output; aborting to avoid bad final data")


_SPLIT_CORE_WEIGHTS = {
    # All splits should see the same question distribution because RL/eval/test
    # measure benchmark-like ability, while SFT still needs enough questions to
    # learn the response semantics behind each action.
    "questions": 3.0,
    "non_mcq_questions": 8.0,
    "recall_questions": 7.0,
    "multi_questions": 4.0,
    "response_rows": 1.5,
    "recall_rows": 2.5,
    "silent_rows": 0.15,
}

_SPLIT_DYNAMIC_PREFIX_WEIGHTS = {
    "q_family:": 2.5,
    "q_category:": 1.5,
    "q_form:": 5.0,
    "q_availability:": 2.0,
    "q_type:": 2.0,
    "q_family_form:": 2.0,
    "q_recall_family:": 3.0,
    "q_recall_form:": 3.0,
}

_SPLIT_SFT_ACTION_PREFIX_WEIGHTS = {
    "sft_action:recall|": 4.0,
    "sft_action:response|": 2.0,
    "sft_action:silent|": 1.0,
    "sft_silent:": 1.0,
}

_SFT_COMPRESS_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_SFT_COMPRESS_TARGET_FRACTION", "0.18")
)
_SPLIT_SWAP_CANDIDATE_LIMIT = int(
    os.environ.get("THINKSTREAM_SPLIT_SWAP_CANDIDATES", "24")
)


def _split_feature_weight(feature: str, bucket: str) -> float:
    """Return the split-balancing weight for a profile feature.

    Question features are shared by every split. SFT gets extra behavior-state
    features because it supervises boundary actions directly. Compress is mostly
    a system-maintenance action for RL/eval/test, so it is only strongly balanced
    into SFT.
    """
    if feature == "compress_rows":
        return 18.0 if bucket == "train_sft" else 0.15
    if feature in _SPLIT_CORE_WEIGHTS:
        return _SPLIT_CORE_WEIGHTS[feature]
    for prefix, weight in _SPLIT_DYNAMIC_PREFIX_WEIGHTS.items():
        if feature.startswith(prefix):
            return weight
    if bucket == "train_sft":
        for prefix, weight in _SPLIT_SFT_ACTION_PREFIX_WEIGHTS.items():
            if feature.startswith(prefix):
                return weight
    return 0.0


def _video_split_profiles(samples: List[Dict]) -> Dict[str, Counter]:
    """Build per-video profiles for split balancing from flat pass3c samples."""
    profiles: Dict[str, Counter] = defaultdict(Counter)
    seen_questions: Dict[str, set] = defaultdict(set)
    recall_questions: Dict[str, set] = defaultdict(set)
    question_meta: Dict[str, Dict[tuple, Dict]] = defaultdict(dict)

    for s in samples:
        vid = str(s.get("video_id") or "")
        if not vid:
            continue
        action = str(s.get("sample_type") or s.get("action") or "")
        profiles[vid]["rows"] += 1
        if action:
            profiles[vid][f"{action}_rows"] += 1

        meta = s.get("metadata") or {}
        answer_form = str(meta.get("answer_form") or "unknown")
        question_type = str(meta.get("question_type") or "unknown")
        availability = str(
            meta.get("availability")
            or s.get("sequence_type")
            or "unknown"
        )
        family = str(meta.get("family") or "unknown")
        category = str(meta.get("category") or "unknown")

        if action in {"recall", "response", "silent"} and meta.get("question"):
            profiles[vid][f"sft_action:{action}|family:{family}"] += 1
            profiles[vid][f"sft_action:{action}|form:{answer_form}"] += 1
            profiles[vid][f"sft_action:{action}|availability:{availability}"] += 1
            if action == "silent":
                base_role = str(s.get("base_role") or "active")
                profiles[vid][f"sft_silent:{base_role}|{availability}|{family}|{answer_form}"] += 1

        card_id = str(s.get("card_id") or meta.get("card_id") or "")
        if not card_id:
            continue
        qkey = (str(s.get("trajectory_id") or ""), card_id)
        if action == "recall":
            recall_questions[vid].add(qkey)
        if qkey in seen_questions[vid]:
            continue
        seen_questions[vid].add(qkey)
        question_meta[vid][qkey] = meta

        profiles[vid]["questions"] += 1
        profiles[vid][f"q_family:{family}"] += 1
        profiles[vid][f"q_category:{category}"] += 1
        profiles[vid][f"q_form:{answer_form}"] += 1
        profiles[vid][f"q_availability:{availability}"] += 1
        profiles[vid][f"q_type:{question_type}"] += 1
        profiles[vid][f"q_family_form:{family}|{answer_form}"] += 1
        if answer_form != "multiple_choice":
            profiles[vid]["non_mcq_questions"] += 1
        if question_type == "multi_emit" or availability == "multi_response":
            profiles[vid]["multi_questions"] += 1

    for vid, qkeys in recall_questions.items():
        profiles[vid]["recall_questions"] = len(qkeys)
        for qkey in qkeys:
            meta = question_meta.get(vid, {}).get(qkey) or {}
            family = str(meta.get("family") or "unknown")
            answer_form = str(meta.get("answer_form") or "unknown")
            profiles[vid][f"q_recall_family:{family}"] += 1
            profiles[vid][f"q_recall_form:{answer_form}"] += 1
    return profiles


def _balanced_video_buckets(
    video_ids: List[str],
    samples: List[Dict],
    *,
    seed: int,
) -> Tuple[Dict[str, set], Dict[str, Dict[str, float]]]:
    """Split videos while balancing question/action profiles across splits."""
    n = len(video_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    sft_target = int(train_end * 0.50)
    targets = {
        "train_sft": sft_target,
        "train_rl": train_end - sft_target,
        "val": val_end - train_end,
        "test": n - val_end,
    }
    profiles = _video_split_profiles(samples)
    global_profile = Counter()
    for vid in video_ids:
        global_profile.update(profiles.get(vid, Counter()))

    rng = random.Random(seed)
    order = list(video_ids)
    rng.shuffle(order)

    def profile_weight(vid: str) -> float:
        p = profiles.get(vid, Counter())
        return (
            2.0 * p.get("questions", 0)
            + 4.0 * p.get("non_mcq_questions", 0)
            + 4.0 * p.get("recall_questions", 0)
            + 2.0 * p.get("multi_questions", 0)
            + 3.0 * p.get("compress_rows", 0)
        )

    order.sort(key=profile_weight, reverse=True)

    buckets: Dict[str, List[str]] = {name: [] for name in targets}
    bucket_profiles: Dict[str, Counter] = {name: Counter() for name in targets}
    balance_features = [
        feat for feat, value in global_profile.items()
        if value and any(_split_feature_weight(feat, name) > 0 for name in targets)
    ]
    target_profiles = {
        name: Counter({
            feat: global_profile.get(feat, 0) * (size / max(n, 1))
            for feat in balance_features
        })
        for name, size in targets.items()
    }
    if global_profile.get("compress_rows", 0):
        target_profiles["train_sft"]["compress_rows"] = (
            global_profile["compress_rows"] * _SFT_COMPRESS_TARGET_FRACTION
        )

    def profile_error(bucket: str, projected: Counter) -> float:
        target = target_profiles[bucket]
        err = 0.0
        for feat in balance_features:
            weight = _split_feature_weight(feat, bucket)
            if weight <= 0:
                continue
            denom = max(float(target.get(feat, 0.0)), 1.0)
            diff = float(projected.get(feat, 0.0)) - float(target.get(feat, 0.0))
            if bucket == "train_sft" and feat in {"compress_rows", "recall_rows"} and diff < 0:
                weight *= 3.0
            err += weight * (diff / denom) ** 2
        return err

    def score(bucket: str, vid: str) -> float:
        projected = bucket_profiles[bucket] + profiles.get(vid, Counter())
        err = profile_error(bucket, projected)
        fill = (len(buckets[bucket]) + 1) / max(targets[bucket], 1)
        return err + 0.10 * fill

    # SFT is the only split with direct compress-action supervision. Seed it
    # with compress-rich videos before the general benchmark-style balancing so
    # rare memory-maintenance behavior is not accidentally concentrated in
    # RL/eval/test, where it is mostly a system-side runtime event.
    sft_compress_target = float(target_profiles["train_sft"].get("compress_rows", 0.0))
    if sft_compress_target > 0:
        max_preseed = max(1, int(targets["train_sft"] * 0.25))
        preseeded = set()
        for vid in sorted(
            order,
            key=lambda v: (
                -profiles.get(v, Counter()).get("compress_rows", 0),
                -profiles.get(v, Counter()).get("questions", 0),
                -profile_weight(v),
            ),
        ):
            if len(buckets["train_sft"]) >= max_preseed:
                break
            if profiles.get(vid, Counter()).get("compress_rows", 0) <= 0:
                break
            buckets["train_sft"].append(vid)
            bucket_profiles["train_sft"].update(profiles.get(vid, Counter()))
            preseeded.add(vid)
            if bucket_profiles["train_sft"].get("compress_rows", 0) >= sft_compress_target:
                break
        if preseeded:
            order = [vid for vid in order if vid not in preseeded]

    for vid in order:
        candidates = [
            name for name, size in targets.items()
            if len(buckets[name]) < size
        ]
        if not candidates:
            break
        best = min(candidates, key=lambda name: (score(name, vid), name))
        buckets[best].append(vid)
        bucket_profiles[best].update(profiles.get(vid, Counter()))

    # Greedy assignment can get trapped when compress-heavy videos also carry
    # skewed question profiles. A few deterministic pair-swap passes make the
    # final split match both benchmark-like question distribution and SFT
    # behavior coverage without changing split sizes.
    def swap_candidates(bucket: str) -> List[str]:
        vids = list(buckets[bucket])
        if len(vids) <= _SPLIT_SWAP_CANDIDATE_LIMIT:
            return vids
        picked: List[str] = []
        if bucket == "train_sft":
            # Prefer low-compress SFT videos as possible outgoing swaps.
            picked.extend(sorted(
                vids,
                key=lambda v: (
                    profiles.get(v, Counter()).get("compress_rows", 0),
                    profile_weight(v),
                ),
            )[:_SPLIT_SWAP_CANDIDATE_LIMIT // 2])
        else:
            # Prefer compress-rich non-SFT videos as possible incoming swaps.
            picked.extend(sorted(
                vids,
                key=lambda v: (
                    -profiles.get(v, Counter()).get("compress_rows", 0),
                    -profile_weight(v),
                ),
            )[:_SPLIT_SWAP_CANDIDATE_LIMIT // 2])
        picked.extend(sorted(vids, key=profile_weight, reverse=True))
        out: List[str] = []
        seen = set()
        for vid in picked:
            if vid in seen:
                continue
            seen.add(vid)
            out.append(vid)
            if len(out) >= _SPLIT_SWAP_CANDIDATE_LIMIT:
                break
        return out

    for _pass in range(1):
        improved = False
        bucket_names = list(targets)
        for i, left in enumerate(bucket_names):
            for right in bucket_names[i + 1:]:
                old_pair = (
                    profile_error(left, bucket_profiles[left])
                    + profile_error(right, bucket_profiles[right])
                )
                left_vids = swap_candidates(left)
                right_vids = swap_candidates(right)
                best_gain = 0.0
                best_swap = None
                for lv in left_vids:
                    lp = profiles.get(lv, Counter())
                    for rv in right_vids:
                        rp = profiles.get(rv, Counter())
                        new_left = bucket_profiles[left] - lp + rp
                        new_right = bucket_profiles[right] - rp + lp
                        new_pair = (
                            profile_error(left, new_left)
                            + profile_error(right, new_right)
                        )
                        gain = old_pair - new_pair
                        if gain > best_gain + 1e-9:
                            best_gain = gain
                            best_swap = (lv, rv, new_left, new_right)
                if best_swap is None:
                    continue
                lv, rv, new_left, new_right = best_swap
                buckets[left].remove(lv)
                buckets[right].remove(rv)
                buckets[left].append(rv)
                buckets[right].append(lv)
                bucket_profiles[left] = new_left
                bucket_profiles[right] = new_right
                improved = True
        if not improved:
            break

    audit: Dict[str, Dict[str, float]] = {}
    for name, vids in buckets.items():
        p = bucket_profiles[name]
        q = max(float(p.get("questions", 0)), 1.0)
        rows = max(float(
            p.get("silent_rows", 0)
            + p.get("response_rows", 0)
            + p.get("recall_rows", 0)
            + p.get("compress_rows", 0)
        ), 1.0)
        audit[name] = {
            "videos": float(len(vids)),
            "questions": float(p.get("questions", 0)),
            "rows": float(rows),
            "response_rows": float(p.get("response_rows", 0)),
            "recall_rows": float(p.get("recall_rows", 0)),
            "compress_rows": float(p.get("compress_rows", 0)),
            "non_mcq_question_pct": round(p.get("non_mcq_questions", 0) / q * 100, 2),
            "recall_question_pct": round(p.get("recall_questions", 0) / q * 100, 2),
            "multi_question_pct": round(p.get("multi_questions", 0) / q * 100, 2),
            "recall_row_pct": round(p.get("recall_rows", 0) / rows * 100, 2),
            "compress_row_pct": round(p.get("compress_rows", 0) / rows * 100, 2),
            "split_score": round(profile_error(name, p), 4),
        }
    return {name: set(vs) for name, vs in buckets.items()}, audit


def _write_quality_audit(path: Path, label: str) -> None:
    """Write a distribution/quality audit report for a generated JSONL file."""
    if not path.exists():
        logger.warning("Quality audit skipped for %s: file missing: %s", label, path)
        return
    from .audit_distribution import audit_jsonl, diagnose
    report = audit_jsonl(path)
    flags = diagnose(report)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIT_DIR / f"{label}_quality_report.json"
    out_path.write_text(json.dumps(
        {"report": report, "flags": flags},
        indent=2, ensure_ascii=False,
    ))
    logger.info("Quality audit written: %s", out_path)
    for flag in flags:
        logger.warning("  [%s audit] %s", label, flag)
    blockers = [f for f in flags if str(f).startswith("BLOCKER")]
    if blockers:
        raise RuntimeError(f"{label} quality audit blockers: {blockers}")


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_videos_jsonl(path: str, limit: int = 0) -> List[Dict]:
    """Load an explicit batch video list.

    Required fields per row: video_id, video_path. duration_sec and dataset
    are preserved when present. This keeps selection outside the expensive
    pass pipeline when a batch has already been balanced/validated.
    """
    rows: List[Dict] = []
    src = Path(path)
    with src.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("video_id") or not row.get("video_path"):
                raise ValueError(
                    f"{src}:{line_no}: each row needs video_id and video_path"
                )
            rows.append({
                "video_id": str(row["video_id"]),
                "video_path": str(row["video_path"]),
                "duration_sec": float(row.get("duration_sec", 0) or 0),
                "dataset": row.get("dataset", "unknown"),
            })
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No videos loaded from {src}")
    return rows


def _write_batch_manifest(videos: List[Dict], *, source: str, seed: int) -> None:
    """Write a stable manifest describing this batch root."""
    selected_path = DATA_ROOT / "selected_videos.jsonl"
    _write_jsonl(selected_path, videos)

    manifest = {
        "batch_id": BATCH_ID,
        "data_root": str(DATA_ROOT),
        "source": source,
        "seed": seed,
        "n_videos": len(videos),
        "selected_videos": str(selected_path),
        "frames_dir": str(DATA_ROOT / "frames"),
        "stage_dirs": {
            "pass1a": str(DATA_ROOT / "evidence_1a"),
            "pass1b": str(DATA_ROOT / "evidence_1b"),
            "pass2": str(DATA_ROOT / "rollout"),
            "pass3a": str(DATA_ROOT / "task_cards"),
            "pass3b": str(DATA_ROOT / "placements"),
            "pass3c": str(DATA_ROOT / "samples_3c"),
            "pass3e": str(DATA_ROOT / "verified"),
            "pass4_pass5_final": str(FINAL_DIR),
            "audits": str(AUDIT_DIR),
        },
        "final_files": {
            "sft_messages": str(FINAL_DIR / "train_sft_messages.jsonl"),
            "rl_trajectories": str(FINAL_DIR / "train_rl_trajectories.jsonl"),
            "val_trajectories": str(FINAL_DIR / "val_trajectories.jsonl"),
            "test_trajectories": str(FINAL_DIR / "test_trajectories.jsonl"),
            "val_messages": str(FINAL_DIR / "val_messages.jsonl"),
            "test_messages": str(FINAL_DIR / "test_messages.jsonl"),
            "dataset_info": str(FINAL_DIR / "dataset_info.json"),
        },
        "rendered_dirs": {
            CANONICAL_RENDER_DIRNAME: str(DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME),
        },
        "derived_files": {
            "sft_messages": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "train_sft_messages.jsonl"
            ),
            "val_messages": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "val_messages.jsonl"
            ),
            "test_messages": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "test_messages.jsonl"
            ),
            "train_parquet": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "train_rl_multi_q.parquet"
            ),
            "val_parquet": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "val_rl_multi_q.parquet"
            ),
            "test_parquet": str(
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "test_rl_multi_q.parquet"
            ),
        },
    }
    (DATA_ROOT / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Phase Assignment
# ---------------------------------------------------------------------------


def assign_phase(sample: Dict) -> str:
    """Assign a per-category label used for diagnostic file splits.

    The label is retained as metadata for audit/debugging. It is no longer
    emitted as separate phase{1,2,C1,5}_train files; canonical training
    outputs are train_sft/train_rl plus trajectory/message/parquet renders.
    """
    sample_type = sample.get("sample_type", "")
    sequence_type = sample.get("sequence_type", "")
    prompt_type = sample.get("prompt_type", "")

    if sample_type == "compress":
        return "C1"  # diagnostic label: compress-trained samples

    if sample_type in ("recall", "recall_query", "recall_response", "recall_silent"):
        return "2"  # diagnostic label: recall-trained samples

    if sample_type == "silent":
        if sequence_type in ("event_watch", "multi_response", "memory_response"):
            return "2"  # query-aware silent
        return "1"  # basic silent

    if sample_type == "response":
        if sequence_type in ("recall_fail_then_found",):
            return "2"  # recovery after recall fail
        if sequence_type in ("event_watch", "multi_response", "memory_response"):
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
    _write_jsonl(registry_path, selected)

    logger.info(
        f"Registry: {len(existing)} existing + {len(new_selected)} new "
        f"= {len(selected)} videos (min {min_duration}s)"
    )
    return selected


def _normalize_frame_tail(
    output_dir: Path,
    *,
    frames_per_chunk: Optional[int],
    tail_policy: str,
) -> List[Path]:
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames or not frames_per_chunk:
        return frames

    fpc = int(frames_per_chunk)
    if fpc <= 1:
        return frames

    remainder = len(frames) % fpc
    dropped: List[str] = []
    padded: List[str] = []
    policy = str(tail_policy or "drop").strip().lower()

    if remainder:
        if policy in {"drop", "trim", "truncate"}:
            for p in frames[-remainder:]:
                dropped.append(p.name)
                p.unlink()
        elif policy in {"pad", "pad_duplicate", "duplicate"}:
            import shutil

            last = frames[-1]
            missing = fpc - remainder
            start_no = int(last.stem.split("_", 1)[1])
            for offset in range(1, missing + 1):
                dst = output_dir / f"frame_{start_no + offset:06d}.jpg"
                shutil.copy2(last, dst)
                padded.append(dst.name)
        elif policy in {"keep", "none"}:
            pass
        else:
            raise ValueError(
                f"Unsupported frame tail policy {tail_policy!r}; expected "
                "drop, pad_duplicate, or keep"
            )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    (output_dir / ".frame_norm.json").write_text(json.dumps({
        "frames_per_chunk": fpc,
        "tail_policy": policy,
        "n_frames": len(frames),
        "num_chunks": len(frames) // fpc,
        "dropped_tail_frames": dropped,
        "padded_tail_frames": padded,
    }, indent=2, ensure_ascii=False))
    return frames


def extract_frames(
    video_path: str,
    output_dir: Path,
    fps: int = 1,
    *,
    frames_per_chunk: Optional[int] = None,
    tail_policy: str = "drop",
) -> List[str]:
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
        frames = _normalize_frame_tail(
            output_dir,
            frames_per_chunk=frames_per_chunk,
            tail_policy=tail_policy,
        )
        return [str(p) for p in frames]
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

    frames = _normalize_frame_tail(
        output_dir,
        frames_per_chunk=frames_per_chunk,
        tail_policy=tail_policy,
    )
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
    videos_jsonl: str = None,
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
    client_3c = VLLMClient(
        **_client_kwargs,
        max_concurrent=safe_concurrency_for_pass("pass3c"),
    )
    logger.info(
        "VLLMClient caps: 1a=%d 1b=%d 2=%d 3a=%d 3c=%d (timeout=5400s)",
        client_1a.max_concurrent, client_1b.max_concurrent,
        client_2.max_concurrent, client_3a.max_concurrent,
        client_3c.max_concurrent,
    )

    # --- Video selection / explicit batch input ---
    if videos_jsonl:
        videos = _load_videos_jsonl(videos_jsonl, limit=num_videos)
        video_source = str(Path(videos_jsonl))
        _write_jsonl(DATA_ROOT / "video_registry.jsonl", videos)
        logger.info(
            "Loaded explicit batch list: %s (%d videos)",
            video_source,
            len(videos),
        )
    else:
        videos = select_videos(video_root, num_videos, seed=seed)
        video_source = "select_videos"
    _write_batch_manifest(videos, source=video_source, seed=seed)
    logger.info(f"Pipeline starting with {len(videos)} videos")

    # --- Extract frames ---
    # v12.5 (2026-04-29): fps=2 + FRAMES_PER_CHUNK=2 → 1s/chunk (was fps=1 → 2s/chunk).
    from scripts.agent_data_v5.config import FPS, FRAMES_PER_CHUNK
    frames_dir = DATA_ROOT / "frames"
    frame_tail_policy = os.environ.get("THINKSTREAM_FRAME_TAIL_POLICY", "drop")
    video_frames = {}
    valid_videos = []
    skipped_zero_chunk = []
    for v in videos:
        v_frames_dir = frames_dir / v["video_id"]
        frames = extract_frames(
            v["video_path"],
            v_frames_dir,
            fps=FPS,
            frames_per_chunk=FRAMES_PER_CHUNK,
            tail_policy=frame_tail_policy,
        )
        num_chunks = len(frames) // FRAMES_PER_CHUNK
        if num_chunks <= 0:
            skipped_zero_chunk.append({
                "video_id": v["video_id"],
                "video_path": v.get("video_path", ""),
                "n_frames": len(frames),
                "frames_per_chunk": FRAMES_PER_CHUNK,
            })
            logger.error(
                "  [%s] skipped: extracted %d frames < FRAMES_PER_CHUNK=%d",
                v["video_id"], len(frames), FRAMES_PER_CHUNK,
            )
            continue
        video_frames[v["video_id"]] = frames
        v["num_chunks"] = num_chunks
        valid_videos.append(v)

    videos = valid_videos
    if skipped_zero_chunk:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        skipped_path = AUDIT_DIR / "skipped_zero_chunk_videos.json"
        skipped_path.write_text(json.dumps(
            skipped_zero_chunk, indent=2, ensure_ascii=False
        ))
        logger.error(
            "Skipped %d zero-chunk videos; details saved to %s",
            len(skipped_zero_chunk), skipped_path,
        )
    if not videos:
        raise RuntimeError(
            "No valid videos after frame extraction; every selected video had "
            "fewer frames than one chunk."
        )

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
    evidence_1a_map: Dict[str, list] = {}
    evidence_map: Dict[str, list] = {}

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
        for v in videos:
            cached = load_1a(v["video_id"])
            if cached:
                evidence_1a_map[v["video_id"]] = cached
            cached = load_1b(v["video_id"])
            if cached:
                evidence_map[v["video_id"]] = cached
        _require_stage_cache("PASS 1-B (--skip_pass 1)", evidence_map, videos)

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
        if 3 not in skip_pass:
            _require_stage_cache("PASS 2 (--skip_pass 2)", rollout_map, videos)

    if run_1 or run_2:
        # v12.11 (2026-05-01): refactor — replace per-video chained pipeline
        # with three sequential WAVES (1a → 1b → 2), each fully parallel
        # across all videos.
        #
        # Why: the prior `process_video_pipeline` chained 1a → 1b → 2 per
        # video and gathered all videos. That meant the SHARED vLLM endpoint
        # saw a mix of (pass1a long-prefill / pass1b very-long-prefill /
        # pass2 short-fast) requests at the same time. vLLM's batching
        # scheduler prioritized short requests → pass1b's long sequences
        # got starved → wall-clock dominated by tail of pass1b. User's
        # measurement: chained > 8h vs. waves ~4h on 320 batch1 videos.
        #
        # Trade-off: waves require all 320 to finish stage N before stage
        # N+1 starts. The slowest video in stage N gates stage N+1. But
        # this is FAR better than vLLM internal starvation, because each
        # wave runs with uniform request length distribution → vLLM can
        # batch optimally without head-of-line blocking from short tasks.
        #
        # Each wave still uses its dedicated client + semaphore (defined
        # above), so per-pass concurrency caps are unchanged.

        rollout_per_video: Dict[str, dict] = {}

        # ─── Wave 1: Pass 1a (per-chunk evidence) ───────────────────────
        if run_1:
            async def _do_pass1a(video):
                vid = video["video_id"]
                cached = load_1a(vid)
                if cached:
                    return vid, cached
                async with video_semaphore_1a:
                    caps = await run_pass1a(
                        video_id=vid,
                        frame_paths=video_frames.get(vid, []),
                        num_chunks=video["num_chunks"],
                        client=client_1a,
                    )
                n_ok = sum(1 for c in caps if c.get("parse_success"))
                if n_ok <= 0:
                    await tracker_1a.record(
                        success=False, video_id=vid,
                        chunks=len(caps), parsed=n_ok,
                    )
                    logger.error(
                        "  [%s] PASS 1-A produced zero parsed chunks; "
                        "not using this video cache",
                        vid,
                    )
                    return vid, None
                save_1a(vid, caps)
                await tracker_1a.record(
                    success=n_ok > 0, video_id=vid,
                    chunks=len(caps), parsed=n_ok,
                )
                return vid, caps

            logger.info("=" * 60)
            logger.info(f"WAVE 1: Pass 1a — {len(videos)} videos in parallel "
                        f"(video_concurrency={VIDEO_CONCURRENCY_1A})")
            logger.info("=" * 60)
            wave1_results = await asyncio.gather(*[_do_pass1a(v) for v in videos])
            failed_1a = [vid for vid, caps in wave1_results if not caps]
            if failed_1a:
                preview = ", ".join(failed_1a[:10])
                suffix = "..." if len(failed_1a) > 10 else ""
                raise RuntimeError(
                    "PASS 1-A failed with zero parsed chunks for "
                    f"{len(failed_1a)}/{len(videos)} videos: {preview}{suffix}"
                )
            evidence_1a_map = {vid: caps for vid, caps in wave1_results}
            tracker_1a.summary()

        # ─── Wave 2: Pass 1b (entity link + state changes) ───────────────
        if run_1:
            async def _do_pass1b(video):
                vid = video["video_id"]
                caps = evidence_1a_map.get(vid)
                if not caps:
                    # 1a failed for this video → skip 1b
                    return vid, None
                cached = load_1b(vid)
                if cached:
                    return vid, cached
                async with video_semaphore_1b:
                    ev = await run_pass1b(
                        evidence=caps,
                        client=client_1b,
                        video_id=vid,
                    )
                save_1b(vid, ev)
                n_sc = sum(1 for c in ev if c.get("state_changes"))
                await tracker_1b.record(
                    success=True, video_id=vid, state_changes=n_sc,
                )
                return vid, ev

            logger.info("=" * 60)
            logger.info(f"WAVE 2: Pass 1b — {len(videos)} videos in parallel "
                        f"(video_concurrency={VIDEO_CONCURRENCY_1B})")
            logger.info("=" * 60)
            wave2_results = await asyncio.gather(*[_do_pass1b(v) for v in videos])
            evidence_map = {vid: ev for vid, ev in wave2_results if ev is not None}
            tracker_1b.summary()
            from .cache_version import write_stage_version
            write_stage_version("1a")
            write_stage_version("1b")
        elif run_2:
            # --skip_pass 1 + run pass2: still need 1b evidence on disk for
            # pass2's compression boundary scoring (state_change-aware range
            # selection in score_range_for_compression).
            for v in videos:
                ev = load_1b(v["video_id"])
                if ev is not None:
                    evidence_map[v["video_id"]] = ev
            _require_stage_cache("PASS 1-B (--skip_pass 1)", evidence_map, videos)

        # ─── Wave 3: Pass 2 (streaming rollout) ───────────────────────────
        if run_2:
            async def _do_pass2(video):
                vid = video["video_id"]
                cached = load_rollout(vid)
                if cached:
                    return vid, cached
                rollout = await run_pass2_single_video(
                    video_id=vid,
                    frame_paths=video_frames.get(vid, []),
                    num_chunks=video["num_chunks"],
                    client=client_2,
                    # pass2 uses evidence only for QC/repair triggers and
                    # compression boundary scoring. Prefer enriched 1b, but
                    # fall back to 1a so videos whose 1b failed do not run
                    # completely blind to obvious stale-repeat drift.
                    evidence=evidence_map.get(vid) or evidence_1a_map.get(vid),
                    chunk_log_path=pass2_chunk_log,
                )
                n_thinks = len(rollout.get("thinks") or [])
                if n_thinks <= 0:
                    await tracker_p2.record(
                        success=False, video_id=vid,
                        thinks=0,
                        compressions=len(rollout.get("compression_events") or []),
                    )
                    logger.error(
                        "  [%s] PASS 2 produced no thinks; not saving cache",
                        vid,
                    )
                    return vid, None
                save_rollout(vid, rollout)
                await tracker_p2.record(
                    success=n_thinks > 0, video_id=vid,
                    thinks=n_thinks,
                    compressions=len(rollout.get("compression_events") or []),
                )
                return vid, rollout

            logger.info("=" * 60)
            logger.info(f"WAVE 3: Pass 2 — {len(videos)} videos in parallel "
                        f"(chunk_concurrency={client_2.max_concurrent})")
            logger.info("=" * 60)
            wave3_results = await asyncio.gather(*[_do_pass2(v) for v in videos])
            rollout_per_video = {vid: r for vid, r in wave3_results if r is not None}

        if run_2:
            # v12.11: rollout_map now sourced from rollout_per_video (wave 3
            # output) instead of the legacy 4-tuple-from-process_video_pipeline.
            rollout_map = dict(rollout_per_video)
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

            # Diagnostic only: pass2 think text is reused directly in SFT/RL
            # trajectories, so catch the high-risk failure mode where the model
            # repeats stale memory for a long span while pass1 evidence changes.
            from .audit_pass2_stale import audit_rollouts as audit_pass2_rollouts
            pass2_audit_evidence = {
                vid: evidence_map.get(vid) or evidence_1a_map.get(vid)
                for vid in rollout_map
            }
            stale_report = audit_pass2_rollouts(rollout_map, pass2_audit_evidence)
            stale_path = AUDIT_DIR / "pass2_stale_audit.json"
            stale_path.write_text(json.dumps(
                stale_report, indent=2, ensure_ascii=False,
            ))
            stale_rate = stale_report.get("totals", {}).get(
                "hard_stale_video_rate", 0.0,
            )
            logger.info("Pass2 stale audit saved to %s", stale_path)
            if stale_rate >= 0.10:
                logger.warning(
                    "Pass2 stale audit flagged %.1f%% videos. Inspect %s before "
                    "training; this is usually visual-window grounding drift.",
                    stale_rate * 100.0,
                    stale_path,
                )

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
        VIDEO_CONCURRENCY_3A = max(
            1, int(os.environ.get("THINKSTREAM_PASS3A_VIDEO_CONCURRENT", "8") or 8)
        )
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
        _require_nonempty(
            "PASS 3-A task cards",
            [c for cards in cards_map.values() for c in (cards or [])],
        )
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
        _require_stage_cache("PASS 3-A (--skip_pass 3)", cards_map, videos)

    # =================================================================
    # PASS 3-B: Placement + Trajectory Planning (programmatic)
    # =================================================================
    if 3 not in skip_pass:
        from .pass3b_placement import (
            compute_all_placements, plan_trajectories,
            save_placements, load_placements,
        )

        logger.info("=" * 60)
        logger.info("PASS 3-B: Placement + Trajectory Planning (programmatic)")
        logger.info("=" * 60)

        trajectories_map = {}

        async def process_video_3b(video):
            vid = video["video_id"]
            cached = load_placements(vid)
            if cached:
                return vid, cached
            if vid not in cards_map or vid not in rollout_map or vid not in evidence_map:
                return vid, None
            # Programmatic placement; the client argument is kept only for
            # backward-compatible function signatures.
            placements = await compute_all_placements(
                cards_map[vid], rollout_map[vid], evidence_map[vid],
                client=None, video_id=vid,
            )
            vid_cards = {c["card_id"]: c for c in cards_map[vid]}
            nc = rollout_map[vid]["num_chunks"]
            traj_seed = stable_seed(seed * 10_000, vid, modulo=0x1000000)
            trajectories = plan_trajectories(
                placements, cards_map=vid_cards,
                num_chunks=nc, evidence=evidence_map[vid],
                rollout=rollout_map[vid], video_id=vid,
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
        _require_nonempty(
            "PASS 3-B trajectories",
            [t for d in trajectories_map.values() for t in d.get("trajectories", [])],
        )
        from .cache_version import write_stage_version
        write_stage_version("3b")
    else:
        from .pass3b_placement import load_placements
        trajectories_map = {}
        for v in videos:
            cached = load_placements(v["video_id"])
            if cached:
                trajectories_map[v["video_id"]] = cached
        _require_stage_cache("PASS 3-B (--skip_pass 3)", trajectories_map, videos)
        _require_nonempty(
            "PASS 3-B trajectories (--skip_pass 3)",
            [t for d in trajectories_map.values() for t in d.get("trajectories", [])],
        )

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
            traj_errors = []
            for result in traj_results:
                if isinstance(result, Exception):
                    logger.error(f"  [{vid}] 3-C trajectory failed: {result}")
                    traj_errors.append(result)
                    continue
                vid_samples.extend(result)
            if traj_errors:
                preview = "; ".join(str(e) for e in traj_errors[:3])
                suffix = "..." if len(traj_errors) > 3 else ""
                raise RuntimeError(
                    f"[{vid}] PASS 3-C failed for {len(traj_errors)}/"
                    f"{len(trajectories)} trajectories: {preview}{suffix}"
                )

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

        _require_nonempty("PASS 3-C samples", all_samples)
        sample_vids = {
            str(s.get("video_id") or "")
            for s in all_samples
            if s.get("video_id")
        }
        missing_sample_vids = [
            str(v.get("video_id"))
            for v in uncached_3c
            if str(v.get("video_id")) not in sample_vids
        ]
        if missing_sample_vids:
            preview = ", ".join(missing_sample_vids[:10])
            suffix = "..." if len(missing_sample_vids) > 10 else ""
            raise RuntimeError(
                "PASS 3-C generated no samples for "
                f"{len(missing_sample_vids)}/{len(uncached_3c)} selected videos: "
                f"{preview}{suffix}"
            )
        tracker_3c.summary()
        from .cache_version import write_stage_version
        write_stage_version("3c")
    else:
        from .pass3c_samples import load_samples
        all_samples = []
        for v in videos:
            cached = load_samples(v["video_id"])
            if cached:
                for s in cached:
                    s.setdefault("video_id", v["video_id"])
                    s.setdefault("video_path", v.get("video_path", ""))
                all_samples.extend(cached)
        sample_vids = {
            str(s.get("video_id") or "")
            for s in all_samples
            if s.get("video_id")
        }
        missing_sample_vids = [
            str(v.get("video_id"))
            for v in videos
            if str(v.get("video_id")) not in sample_vids
        ]
        if missing_sample_vids:
            preview = ", ".join(missing_sample_vids[:10])
            suffix = "..." if len(missing_sample_vids) > 10 else ""
            raise RuntimeError(
                "PASS 3-C (--skip_pass 3): missing or stale samples for "
                f"{len(missing_sample_vids)}/{len(videos)} selected videos: "
                f"{preview}{suffix}"
            )
        _require_nonempty("PASS 3-C samples (--skip_pass 3)", all_samples)

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
    _require_nonempty("RENDER samples", rendered_samples)
    rendered_vids = {
        str(s.get("video_id") or "")
        for s in rendered_samples
        if s.get("video_id")
    }
    missing_render_vids = [
        vid for vid in sorted(raw_by_vid)
        if vid and vid not in rendered_vids
    ]
    if missing_render_vids:
        preview = ", ".join(missing_render_vids[:10])
        suffix = "..." if len(missing_render_vids) > 10 else ""
        raise RuntimeError(
            "RENDER produced no samples for "
            f"{len(missing_render_vids)}/{len(raw_by_vid)} videos: "
            f"{preview}{suffix}"
        )

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
    _require_nonempty("PASS 3-E tagged samples", tagged_samples)

    # Save ALL tagged samples per video (not just passed)
    verified_by_vid = {}
    for s in tagged_samples:
        vid = s.get("video_id", "unknown")
        verified_by_vid.setdefault(vid, []).append(s)
    for vid, vid_samples in verified_by_vid.items():
        save_verified(vid, vid_samples, {"video_id": vid, "count": len(vid_samples)})

    # v12.11 review-fix (2026-05-01): stamp verifier stage version. Audit-5
    # bumped STAGE_VERSIONS["4"] to v12.11 but no caller wrote the marker
    # → existing v12.5-stamped caches were never re-validated as up-to-date.
    from .cache_version import write_stage_version as _write_stage_version_4
    _write_stage_version_4("4")

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
            action_prio = {"response": 0, "recall": 1, "recall_query": 1,
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
    _require_nonempty("POST-FILTER SFT samples", sft_samples)

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
    global_categories = {}
    global_seq_types = {}
    global_base_roles = {}
    for s in sft_samples:
        meta = s.get("metadata", {})
        fam = meta.get("family", "")
        if fam:
            global_families[fam] = global_families.get(fam, 0) + 1
        cat = meta.get("category", "")
        if cat:
            global_categories[cat] = global_categories.get(cat, 0) + 1
        seq = s.get("sequence_type", "")
        global_seq_types[seq] = global_seq_types.get(seq, 0) + 1
        br = s.get("base_role", "")
        if br:
            global_base_roles[br] = global_base_roles.get(br, 0) + 1

    logger.info(f"Global family dist: {dict(sorted(global_families.items()))}")
    logger.info(f"Global category dist: {dict(sorted(global_categories.items()))}")
    logger.info(f"Global seq_type dist: {dict(sorted(global_seq_types.items()))}")
    logger.info(f"Global base_role dist: {dict(sorted(global_base_roles.items()))}")

    # Warn if any expected v12 taxonomy family has very low representation.
    # Multi-emit families are allowed a lower floor because each unique card
    # expands into several response rows and is capped by design.
    total_with_family = sum(global_families.values()) or 1
    for fam, floor_pct in [
        ("N1", 1.0), ("P1", 1.0), ("HLD1", 0.8),
        ("CR1", 1.0), ("CR2", 1.0), ("CR3", 0.8), ("CR4", 1.0),
        ("CR5", 1.0), ("CR7", 0.8), ("E2", 1.0), ("F6", 0.5),
        ("F7", 0.5), ("R1", 0.8), ("ACR1", 0.8), ("STU1", 0.5),
        ("OJR1", 0.5), ("C1", 0.5), ("F5", 0.3), ("PN1", 0.3),
        ("M1", 0.5),
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
    # v12.13 (2026-05-02): use sorted() to make split deterministic across
    # runs. `list(set(...))` ordering depends on PYTHONHASHSEED (random by
    # default in Python 3) — even with seed=42 fixed for shuffle, the
    # input list order varies, so SFT/RL/val/test assignments are NOT
    # reproducible.
    video_ids = sorted({s.get("video_id", "") for s in passed_samples})
    video_ids = [v for v in video_ids if v]
    split_buckets, split_balance_audit = _balanced_video_buckets(
        video_ids,
        passed_samples,
        seed=seed,
    )
    sft_train_vids = split_buckets["train_sft"]
    rl_train_vids = split_buckets["train_rl"]
    val_vids = split_buckets["val"]
    test_vids = split_buckets["test"]
    train_vids = sft_train_vids | rl_train_vids
    logger.info("Split balance audit: %s", split_balance_audit)
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
            # eval signal). Never thin recall/compress rows: tool-use and
            # memory-maintenance boundary cases are too sparse to discard.
            for s in vsamps:
                tid = s.get("trajectory_id", "")
                sample_type = str(s.get("sample_type") or s.get("action") or "")
                if (
                    not tid
                    or tid == first_traj
                    or s.get("sequence_type") == "base"
                    or sample_type in {"recall", "compress"}
                ):
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

    _write_quality_audit(FINAL_DIR / "train_sft.jsonl", "train_sft")
    _write_quality_audit(FINAL_DIR / "train_rl.jsonl", "train_rl")

    phase_counts = Counter(str(s.get("phase", "")) for s in train_samples)
    logger.info(
        "  legacy phase train files are retired; phase metadata counts: %s",
        dict(phase_counts),
    )

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
    stats["phase_counts"] = dict(phase_counts)
    stats["legacy_phase_files_emitted"] = False
    stats["split_by_video"] = True
    stats["split_balance_audit"] = split_balance_audit
    stats["global_family_distribution"] = global_families
    stats["global_category_distribution"] = global_categories
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

    # v12.6: chain pass4 (trajectory grouping) + pass5_messages (ShareGPT
    # conversion) so a single pipeline invocation produces every file the
    # downstream trainers actually default to:
    #   - SFT default reads train_sft_messages.jsonl  (pass5 output)
    #   - RL  default reads train_rl_trajectories.jsonl (pass4 output)
    # Without this chain, users had to remember two extra commands. Skipping
    # if SKIP_PASS45 env is set (useful when only re-running pass1-3).
    if not os.environ.get("SKIP_PASS45"):
        import sys as _sys
        _argv_backup = _sys.argv

        logger.info("=" * 60)
        logger.info("PASS 4: trajectory grouping (per-video → per-trajectory rows)")
        logger.info("=" * 60)
        # v12.6 fix: isolate sys.argv so pass4's argparser doesn't see the
        # outer pipeline's `run --api_base ...` flags. Without this swap,
        # pass4.main() raises SystemExit('unrecognized arguments...') which
        # the broad try/except below silently swallows → pass4 never runs →
        # downstream pass5 has no train_*_trajectories.jsonl input → SFT
        # default dataset stays missing. Match pass5's argv-isolation idiom.
        pass4_ok = False
        try:
            from scripts.agent_data_v5 import pass4 as _pass4_mod
            _sys.argv = ["pass4", "--data-dir", str(DATA_ROOT)]
            try:
                _pass4_mod.main()
                pass4_ok = True
            finally:
                _sys.argv = _argv_backup
        except SystemExit as _e:
            logger.warning(f"pass4 SystemExit (rc={_e.code}); inspect logs above")
        except Exception as e:
            logger.error(f"pass4 failed: {e}; SFT/RL default datasets may be missing")
            _sys.argv = _argv_backup
        if pass4_ok:
            _write_quality_audit(
                FINAL_DIR / "train_rl_trajectories.jsonl",
                "train_rl_trajectories",
            )

        # v12.14: MC option letters must not carry a dataset-level prior.
        # Pass3A LLM generations can skew correct_option heavily toward A
        # even when the answer text is valid. Rebalance after pass4 and before
        # rendering pass5 variants so flat samples, trajectory rows, and both
        # messages-format protocols stay in sync.
        try:
            from scripts.agent_data_v5 import rebalance_mc_options as _mc_mod

            mapping = _mc_mod.build_mapping(FINAL_DIR)
            changed = _mc_mod.apply_mapping(FINAL_DIR, mapping)
            validation = _mc_mod.validate_mapping(FINAL_DIR, mapping)
            report = _mc_mod.summarize_mapping(mapping)
            report["changed_by_file"] = changed
            report["validation"] = {
                "rows_checked": validation["rows_checked"],
                "n_errors": validation["n_errors"],
                "errors_preview": validation["errors"][:20],
            }
            report_path = AUDIT_DIR / "mc_rebalance_report.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            if validation["errors"]:
                raise RuntimeError(
                    "MC option rebalance validation failed:\n"
                    + "\n".join(validation["errors"][:20])
                )
            logger.info(
                "MC option rebalance: %d questions; report -> %s",
                report.get("n_questions", 0),
                report_path,
            )

            _write_quality_audit(FINAL_DIR / "train_sft.jsonl", "train_sft")
            _write_quality_audit(FINAL_DIR / "train_rl.jsonl", "train_rl")
            if (FINAL_DIR / "train_rl_trajectories.jsonl").exists():
                _write_quality_audit(
                    FINAL_DIR / "train_rl_trajectories.jsonl",
                    "train_rl_trajectories",
                )
        except Exception as e:
            logger.error(f"MC option rebalance failed: {e}; inspect MC balance audit")
            raise

        logger.info("=" * 60)
        logger.info("PASS 5: messages-format conversion (LLaMA-Factory ShareGPT)")
        logger.info("=" * 60)
        pass5_ok = True
        try:
            from scripts.agent_data_v5 import pass5_messages as _pass5_mod

            # Render the single canonical protocol used by SFT/RL/eval.
            # Archived standard/ts_image variants are intentionally not
            # generated by the main pipeline anymore.
            pass5_jobs = [
                (
                    CANONICAL_RENDER_DIRNAME,
                    DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME,
                    CANONICAL_FRAME_PROTOCOL,
                    CANONICAL_RENDER_LAYOUT,
                ),
            ]
            for label, output_dir, frame_protocol, render_layout in pass5_jobs:
                logger.info(
                    "PASS 5 render: %s → %s (frame_protocol=%s, render_layout=%s)",
                    label,
                    output_dir,
                    frame_protocol,
                    render_layout,
                )
                _sys.argv = [
                    "pass5_messages",
                    "--input", "traj",
                    "--final-dir", str(FINAL_DIR),
                    "--output-dir", str(output_dir),
                    "--frame-protocol", frame_protocol,
                    "--render-layout", render_layout,
                ]
                try:
                    _pass5_mod.main()
                finally:
                    _sys.argv = _argv_backup

            # v12.11 review-fix: stamp pass5 stage version on success.
            from .cache_version import write_stage_version as _write_v5
            _write_v5("5")
        except SystemExit as _e:
            pass5_ok = False
            logger.warning(f"pass5_messages SystemExit (rc={_e.code}); inspect logs above")
            _sys.argv = _argv_backup
        except Exception as e:
            pass5_ok = False
            logger.error(f"pass5_messages failed: {e}; SFT/eval default datasets may be missing")
            _sys.argv = _argv_backup

        logger.info("=" * 60)
        logger.info("RL PARQUET: build canonical %s", CANONICAL_RENDER_DIRNAME)
        logger.info("=" * 60)
        try:
            from scripts.agent_data_v5 import build_verl_parquet as _parquet_mod

            parquet_jobs = [
                (
                    "canonical train",
                    FINAL_DIR / "train_rl_trajectories.jsonl",
                    DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "train_rl_multi_q.parquet",
                    CANONICAL_FRAME_PROTOCOL,
                    CANONICAL_RENDER_LAYOUT,
                ),
                (
                    "canonical val",
                    FINAL_DIR / "val_trajectories.jsonl",
                    DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "val_rl_multi_q.parquet",
                    CANONICAL_FRAME_PROTOCOL,
                    CANONICAL_RENDER_LAYOUT,
                ),
                (
                    "canonical test",
                    FINAL_DIR / "test_trajectories.jsonl",
                    DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME / "test_rl_multi_q.parquet",
                    CANONICAL_FRAME_PROTOCOL,
                    CANONICAL_RENDER_LAYOUT,
                ),
            ]
            for label, src, out_path, frame_protocol, render_layout in parquet_jobs:
                if not src.exists():
                    logger.warning("RL parquet skipped (%s): missing %s", label, src)
                    continue
                logger.info(
                    "RL parquet: %s → %s (frame_protocol=%s, render_layout=%s)",
                    src.name,
                    out_path,
                    frame_protocol,
                    render_layout,
                )
                _sys.argv = [
                    "build_verl_parquet",
                    "--jsonl", str(src),
                    "--out", str(out_path),
                    "--multi_q",
                    "--frame-protocol", frame_protocol,
                    "--render-layout", render_layout,
                ]
                try:
                    rc = _parquet_mod.main()
                finally:
                    _sys.argv = _argv_backup
                if rc:
                    logger.error("RL parquet build failed (%s) with rc=%s", label, rc)
        except SystemExit as _e:
            logger.warning(f"build_verl_parquet SystemExit (rc={_e.code}); inspect logs above")
            _sys.argv = _argv_backup
        except Exception as e:
            logger.error(f"build_verl_parquet failed: {e}; RL default parquet may be missing")
            _sys.argv = _argv_backup

        if pass5_ok:
            logger.info(
                "Rendered canonical protocol: %s",
                DATA_ROOT / "rendered" / CANONICAL_RENDER_DIRNAME,
            )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total samples: {len(passed_samples)}")
    logger.info(f"Output: {FINAL_DIR}")
    logger.info("Next:")
    logger.info("  SFT: bash scripts/sft_per_timestep.sh")
    logger.info("  RL:  LLM=<sft_ckpt> bash scripts/grpo_train_verl.sh")
    logger.info("  Eval/test: bash scripts/eval/ovo/run_sft_full.sh or run_rl_full.sh")
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
    run_parser.add_argument(
        "--video_root",
        default="",
        help="Root used for catalog/filesystem selection. Optional with --videos_jsonl.",
    )
    run_parser.add_argument("--num_videos", type=int, default=300)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--videos_jsonl",
        default=None,
        help=(
            "Optional explicit batch list with video_id/video_path rows. "
            "Use this for pre-balanced candidate batches."
        ),
    )
    run_parser.add_argument("--skip_pass", type=int, nargs="*", default=[])
    run_parser.add_argument(
        "--force_rerun_from",
        choices=["1a", "1b", "2", "3a", "3b", "3c", "4", "5"],
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
        if not args.video_root and not args.videos_jsonl:
            run_parser.error("either --video_root or --videos_jsonl is required")
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
            videos_jsonl=args.videos_jsonl,
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
