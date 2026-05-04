"""Select batch-2 videos from a validated candidate pool.

This selector is intentionally task-aware instead of sampling from the raw
pool distribution. Streamo-Instruct mixes several streaming task families
(narration, event/action captioning, grounding, time-sensitive QA, offline
QA) and uses sources such as Koala, LLaVA-Video, ActivityNet, QVHighlight,
YouCook2, DiDeMo, and COIN. ThinkStream additionally needs longer one-video
trajectories for recall/compression, so we keep Streamo's source coverage but
shift the duration mix upward.

Default input:
    /Users/hzh/Downloads/candidate_pool_validated.jsonl

Default outputs:
    data/agent_v5/batch2_videos.jsonl
    data/agent_v5/batch2_selection_report.json

Multiple batches:
    python scripts/select_batch2.py --num-batches 5

This writes batch2..batch6, 500 videos each. Within a multi-batch run, each
later batch excludes earlier batches from the same run. Existing output files
are overwritten, not used as exclusions, so the command is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = Path("/Users/hzh/Downloads/candidate_pool_validated.jsonl")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "agent_v5" / "batch2_videos.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "agent_v5" / "batch2_selection_report.json"
DEFAULT_MULTI_REPORT = PROJECT_ROOT / "data" / "agent_v5" / "batch_selection_summary.json"

SEED = 42
TARGET = 500
MIN_DURATION = 30.0
MAX_DURATION = 600.0


# Functional quotas for one-video/one-trajectory construction. The labels map
# each selected video to the capability it is meant to exercise; downstream
# passes can ignore these extra fields.
STRATA: List[Dict[str, Any]] = [
    # Procedural / step-structured videos: action caption, event caption,
    # ASI-like ordering, state changes, object permanence, and future
    # prediction.
    {"group": "how_to_step", "quota": 40, "kind": "procedural_step"},
    {"group": "how_to_caption", "quota": 30, "kind": "procedural_step"},
    {"group": "Koala_raw", "quota": 40, "kind": "procedural_longform"},
    {"group": "Koala", "quota": 20, "kind": "procedural_short_clip"},
    {"group": "VideoMind/coin", "quota": 35, "kind": "procedural_benchmark"},
    {"group": "VideoMind/youcook2", "quota": 30, "kind": "procedural_benchmark"},
    # Temporal localization / grounding sources.
    {"group": "VideoMind/qvhighlights", "quota": 40, "kind": "temporal_grounding"},
    {"group": "VideoMind/didemo", "quota": 25, "kind": "temporal_grounding"},
    {"group": "VideoMind/queryd", "quota": 20, "kind": "temporal_grounding"},
    {"group": "VideoMind/hirest", "quota": 30, "kind": "temporal_grounding"},
    # Natural QA / open-world context.
    {"group": "LLaVA-Video/youtube", "quota": 45, "kind": "open_world_qa"},
    {"group": "LLaVA-Video/academic", "quota": 25, "kind": "offline_video_qa"},
    # Activity / action-recognition sources for repeated action, entity
    # tracking, and "what happens next" supervision.
    #
    # LLaVA nextqa/activitynetqa and tarsier ActivityNet/Charades largely
    # duplicate VideoMind video_ids in this candidate pool. Since ThinkStream
    # stage caches are keyed by video_id, choose one canonical copy instead of
    # mixing duplicates from two directory trees.
    {"group": "VideoMind/activitynet", "quota": 50, "kind": "activity_event"},
    {"group": "VideoMind/charades_sta", "quota": 20, "kind": "activity_action"},
    # Short dynamic clips are useful for precise response timing and
    # time-sensitive QA; keep them present but capped.
    {"group": "tarsier2/VATEX", "quota": 35, "kind": "short_dynamic_scene"},
    {"group": "VideoMind/nextqa", "quota": 15, "kind": "short_dynamic_qa"},
]

assert sum(s["quota"] for s in STRATA) == TARGET

# Streamo-Instruct's >=30s videos are roughly 28/32/30/9 over
# 30-60/60-120/120-240/240+ seconds. ThinkStream needs more long trajectories
# to train memory compression and recall, so batch2 intentionally shifts to
# longer clips while bounding cost at 600s.
DURATION_TARGETS = {
    "30-60": 75,
    "60-120": 125,
    "120-240": 175,
    "240-600": 125,
}

VALID_CODECS = {"h264", "hevc", "h265", "vp9", "av1"}


def duration_bucket(duration: float) -> str:
    if duration < 60:
        return "30-60"
    if duration < 120:
        return "60-120"
    if duration < 240:
        return "120-240"
    if duration <= 600:
        return "240-600"
    return "600+"


def canonical_group(row: Dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "")
    subdataset = str(row.get("subdataset") or "")

    if dataset in {"how_to_step", "how_to_caption", "Koala_raw", "Koala"}:
        return dataset
    if dataset == "VideoMind-Dataset":
        return f"VideoMind/{subdataset}"
    if dataset == "tarsier2_unzip":
        return f"tarsier2/{subdataset}"
    if dataset == "LLaVA-Video-178K":
        if "youtube" in subdataset:
            return "LLaVA-Video/youtube"
        if "academic" in subdataset:
            return "LLaVA-Video/academic"
        if "nextqa" in subdataset:
            return "LLaVA-Video/nextqa"
        if "activitynetqa" in subdataset:
            return "LLaVA-Video/activitynetqa"
        return "LLaVA-Video/other"
    return f"{dataset}/{subdataset}"


def is_valid_codec(codec: Any) -> bool:
    c = str(codec or "").lower()
    return c in VALID_CODECS or c.startswith("h26")


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def read_existing(exclude_files: List[Path]) -> Tuple[set, set]:
    paths = set()
    ids = set()
    for path in exclude_files:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            if row.get("video_path"):
                paths.add(str(row["video_path"]))
            if row.get("video_id"):
                ids.add(str(row["video_id"]))
    return paths, ids


def default_exclude_files() -> List[Path]:
    return [
        PROJECT_ROOT / "data" / "selected_videos.jsonl",
        PROJECT_ROOT / "data" / "agent_v5_test" / "video_registry.jsonl",
        PROJECT_ROOT / "data" / "agent_v5" / "video_registry.jsonl",
    ]


def row_score(row: Dict[str, Any], rng: random.Random) -> Tuple[int, float, float, str]:
    """Lower score is better for deterministic within-bucket selection."""
    duration = parse_float(row.get("duration_sec"))
    bucket = duration_bucket(duration)
    # Prefer non-Streamo overlap when the source has enough alternatives.
    streamo_penalty = 1 if row.get("is_streamo") else 0
    # Keep very long videos inside the long bucket but prefer sub-5min clips
    # before the 5-10min tail to control generation cost.
    long_penalty = max(0.0, duration - 300.0) / 300.0 if bucket == "240-600" else 0.0
    return (streamo_penalty, long_penalty, rng.random(), str(row.get("video_path") or ""))


def load_pool(
    candidate_path: Path,
    exclude_paths: set,
    exclude_ids: set,
    seed: int,
    min_duration: float,
    max_duration: float,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Any]]:
    rng = random.Random(seed)
    target_groups = {s["group"] for s in STRATA}
    pool: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    report = {
        "candidate_rows": 0,
        "available_rows": 0,
        "skipped": Counter(),
        "candidate_by_group": Counter(),
        "available_by_group": Counter(),
        "available_by_bucket": Counter(),
        "available_by_dataset": Counter(),
    }
    seen_ids = set()
    seen_paths = set()

    for raw in iter_jsonl(candidate_path):
        report["candidate_rows"] += 1
        row = dict(raw)
        group = canonical_group(row)
        report["candidate_by_group"][group] += 1

        if group not in target_groups:
            report["skipped"]["non_target_group"] += 1
            continue
        video_path = str(row.get("video_path") or "")
        video_id = str(row.get("video_id") or Path(video_path).stem)
        if not video_path or not video_id:
            report["skipped"]["missing_id_or_path"] += 1
            continue
        if video_path in exclude_paths or video_id in exclude_ids or row.get("is_thinkstream"):
            report["skipped"]["existing_or_thinkstream"] += 1
            continue
        if video_path in seen_paths or video_id in seen_ids:
            report["skipped"]["duplicate_candidate"] += 1
            continue

        duration = parse_float(row.get("duration_sec"), -1.0)
        if duration < min_duration or duration > max_duration:
            report["skipped"]["duration_out_of_range"] += 1
            continue
        if row.get("known_ffmpeg_fail"):
            report["skipped"]["known_ffmpeg_fail"] += 1
            continue
        width = parse_int(row.get("width"))
        height = parse_int(row.get("height"))
        if width and height and width * height < 320 * 240:
            report["skipped"]["resolution_low"] += 1
            continue
        if not is_valid_codec(row.get("codec")):
            report["skipped"]["bad_codec"] += 1
            continue

        seen_paths.add(video_path)
        seen_ids.add(video_id)
        bucket = duration_bucket(duration)
        clean = {
            "video_id": video_id,
            "video_path": video_path,
            "duration_sec": round(duration, 3),
            "dataset": row.get("dataset", "unknown"),
            "subdataset": row.get("subdataset", ""),
            "source_group": group,
            "duration_bucket": bucket,
            "is_streamo": bool(row.get("is_streamo")),
        }
        pool[group][bucket].append(clean)
        report["available_rows"] += 1
        report["available_by_group"][group] += 1
        report["available_by_bucket"][bucket] += 1
        report["available_by_dataset"][clean["dataset"]] += 1

    for group_buckets in pool.values():
        for bucket, rows in group_buckets.items():
            rows.sort(key=lambda r: row_score(r, rng))

    return pool, report


def choose_bucket(
    buckets: Dict[str, List[Dict[str, Any]]],
    selected_by_bucket: Counter,
) -> Optional[str]:
    available = [b for b in DURATION_TARGETS if buckets.get(b)]
    if not available:
        return None

    def deficit(bucket: str) -> Tuple[float, int]:
        target = DURATION_TARGETS[bucket]
        current = selected_by_bucket[bucket]
        return ((target - current) / max(target, 1), target - current)

    return max(available, key=deficit)


def sample_group(
    group: str,
    quota: int,
    kind: str,
    buckets: Dict[str, List[Dict[str, Any]]],
    selected_by_bucket: Counter,
) -> List[Dict[str, Any]]:
    selected = []
    for _ in range(quota):
        bucket = choose_bucket(buckets, selected_by_bucket)
        if bucket is None:
            break
        row = buckets[bucket].pop(0)
        row["selection_kind"] = kind
        row["selection_quota_group"] = group
        selected.append(row)
        selected_by_bucket[bucket] += 1
    return selected


def top_up(
    selected: List[Dict[str, Any]],
    pool: Dict[str, Dict[str, List[Dict[str, Any]]]],
    selected_by_bucket: Counter,
    target: int,
) -> List[Dict[str, Any]]:
    if len(selected) >= target:
        return selected[:target]
    remaining_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for group, buckets in pool.items():
        for bucket, rows in buckets.items():
            for row in rows:
                row["selection_kind"] = row.get("selection_kind", "topup")
                row["selection_quota_group"] = row.get("selection_quota_group", group)
                remaining_by_bucket[bucket].append(row)

    while len(selected) < target:
        bucket = choose_bucket(remaining_by_bucket, selected_by_bucket)
        if bucket is None:
            break
        row = remaining_by_bucket[bucket].pop(0)
        selected.append(row)
        selected_by_bucket[bucket] += 1
    return selected


def select_batch2(
    candidate_path: Path,
    output_path: Path,
    report_path: Path,
    target: int,
    seed: int,
    min_duration: float,
    max_duration: float,
    exclude_files: List[Path],
) -> List[Dict[str, Any]]:
    output_resolved = output_path.resolve()
    exclude_files = [
        p for p in exclude_files
        if not p.exists() or p.resolve() != output_resolved
    ]
    exclude_paths, exclude_ids = read_existing(exclude_files)
    pool, load_report = load_pool(
        candidate_path=candidate_path,
        exclude_paths=exclude_paths,
        exclude_ids=exclude_ids,
        seed=seed,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    selected: List[Dict[str, Any]] = []
    selected_by_bucket: Counter = Counter()
    stratum_shortfalls: Dict[str, int] = {}
    for stratum in STRATA:
        group = stratum["group"]
        quota = int(stratum["quota"])
        picked = sample_group(
            group=group,
            quota=quota,
            kind=stratum["kind"],
            buckets=pool[group],
            selected_by_bucket=selected_by_bucket,
        )
        selected.extend(picked)
        if len(picked) < quota:
            stratum_shortfalls[group] = quota - len(picked)

    selected = top_up(selected, pool, selected_by_bucket, target)
    if len(selected) != target:
        raise RuntimeError(f"selected {len(selected)} videos, expected {target}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_ids = [r["video_id"] for r in selected]
    selected_paths = [r["video_path"] for r in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("duplicate video_id in selected batch")
    if len(selected_paths) != len(set(selected_paths)):
        raise RuntimeError("duplicate video_path in selected batch")

    report = {
        "candidate_path": str(candidate_path),
        "output_path": str(output_path),
        "target": target,
        "seed": seed,
        "duration_range_sec": [min_duration, max_duration],
        "streamo_paper_takeaways": {
            "task_families": [
                "real_time_narration",
                "event_caption",
                "action_caption",
                "event_grounding",
                "time_sensitive_qa",
                "offline_qa",
            ],
            "sources": [
                "Koala",
                "LLaVA-Video",
                "ActivityNet",
                "QVHighlight",
                "YouCook2",
                "HACS",
                "EgoTimeQA",
                "DiDeMo",
                "COIN",
            ],
            "streamo_duration_distribution": {
                "0-30": 68273,
                "30-60": 19153,
                "60-120": 21834,
                "120-240": 20529,
                "240+": 6086,
            },
        },
        "thinkstream_batch2_policy": {
            "reason": (
                "Keep Streamo-like source/task coverage but use more 120-600s "
                "videos because ThinkStream trains one video as one trajectory "
                "and needs recall/compression supervision."
            ),
            "target_duration_buckets": DURATION_TARGETS,
            "strata": STRATA,
        },
        "excluded_existing_files": [str(p) for p in exclude_files if p.exists()],
        "excluded_existing_paths": len(exclude_paths),
        "excluded_existing_ids": len(exclude_ids),
        "candidate_stats": counter_report(load_report),
        "selection_stats": summarize(selected),
        "stratum_shortfalls": stratum_shortfalls,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def write_combined_jsonl(path: Path, batches: List[Tuple[int, List[Dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for batch_no, rows in batches:
            for row in rows:
                out = dict(row)
                out["batch"] = batch_no
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def adaptive_q_count_from_duration(duration: float) -> int:
    return max(6, min(14, int(duration) // 12))


def scale_estimate(rows: List[Dict[str, Any]], usable_rate: float = 0.90) -> Dict[str, Any]:
    """Estimate trainable data volume from selected videos.

    Calibration:
    - v2_simulation.json: 34,470 timestep samples / 251 videos = 137.3 rows/video.
    - batch2-batchN selected videos average about 150s, so 0.9 rows per second
      is the matching conservative estimate after patrol downsampling.
    """
    n = len(rows)
    usable_videos = int(round(n * usable_rate))
    total_seconds = sum(parse_float(r.get("duration_sec")) for r in rows)
    sft_rows_all = int(round(total_seconds * usable_rate * 0.90))
    avg_questions = (
        sum(adaptive_q_count_from_duration(parse_float(r.get("duration_sec"))) for r in rows)
        / max(n, 1)
    )
    train_videos = int(usable_videos * 0.70)
    sft_videos = int(train_videos * 0.50)
    rl_videos = train_videos - sft_videos
    val_videos = int(usable_videos * 0.15)
    test_videos = usable_videos - train_videos - val_videos

    return {
        "selected_videos": n,
        "usable_rate_assumption": usable_rate,
        "usable_videos_est": usable_videos,
        "total_hours_selected": round(total_seconds / 3600.0, 3),
        "avg_duration_sec": round(total_seconds / max(n, 1), 1),
        "avg_questions_per_trajectory_est": round(avg_questions, 2),
        "split_videos_est": {
            "train": train_videos,
            "train_sft": sft_videos,
            "train_rl": rl_videos,
            "val": val_videos,
            "test": test_videos,
        },
        "sft_timestep_rows_est": {
            "all_verified": sft_rows_all,
            "train_sft": int(round(sft_rows_all * 0.35)),
            "train_rl_timestep_sidecar": int(round(sft_rows_all * 0.35)),
            "val": int(round(sft_rows_all * 0.15)),
            "test": int(round(sft_rows_all * 0.15)),
        },
        "rl_prompt_groups_est": rl_videos,
        "rl_rollouts_per_epoch_est_g8": rl_videos * 8,
        "rl_questions_est": int(round(rl_videos * avg_questions)),
        "benchmark_question_instances_est": int(round((val_videos + test_videos) * avg_questions)),
    }


def select_multiple_batches(
    candidate_path: Path,
    output_dir: Path,
    start_batch: int,
    num_batches: int,
    batch_size: int,
    seed: int,
    min_duration: float,
    max_duration: float,
    base_exclude_files: List[Path],
    summary_path: Path,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    batches: List[Tuple[int, List[Dict[str, Any]]]] = []
    generated_files: List[Path] = []
    for offset in range(num_batches):
        batch_no = start_batch + offset
        output_path = output_dir / f"batch{batch_no}_videos.jsonl"
        report_path = output_dir / f"batch{batch_no}_selection_report.json"
        selected = select_batch2(
            candidate_path=candidate_path,
            output_path=output_path,
            report_path=report_path,
            target=batch_size,
            seed=seed + offset,
            min_duration=min_duration,
            max_duration=max_duration,
            exclude_files=base_exclude_files + generated_files,
        )
        for row in selected:
            row["batch"] = batch_no
        batches.append((batch_no, selected))
        generated_files.append(output_path)

    all_rows = [row for _, rows in batches for row in rows]
    combined_path = output_dir / f"batch{start_batch}_to_batch{start_batch + num_batches - 1}_videos.jsonl"
    write_combined_jsonl(combined_path, batches)
    summary = {
        "candidate_path": str(candidate_path),
        "start_batch": start_batch,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "combined_output": str(combined_path),
        "batches": {
            f"batch{batch_no}": {
                "count": len(rows),
                "output": str(output_dir / f"batch{batch_no}_videos.jsonl"),
                "report": str(output_dir / f"batch{batch_no}_selection_report.json"),
                "selection_stats": summarize(rows),
                "scale_estimate": scale_estimate(rows),
            }
            for batch_no, rows in batches
        },
        "combined_selection_stats": summarize(all_rows),
        "combined_scale_estimate": scale_estimate(all_rows),
        "recommended_project_scale": project_scale_recommendation(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return batches


def project_scale_recommendation() -> Dict[str, Any]:
    return {
        "minimum_clean_videos": 1000,
        "recommended_clean_videos": 2000,
        "optimal_clean_videos": 2500,
        "scale_ablation_ceiling_videos": 3000,
        "reasoning": [
            "Pipeline split is by video: 70% train, then train is split 50/50 into SFT and RL.",
            "At 2000 clean videos this yields about 700 SFT videos, 700 RL videos, and 300+300 val/test videos.",
            "RL needs at least 150 unique prompt groups; 2000 videos gives roughly 700 groups, enough for stable GRPO-style training.",
            "A 300-video validation/test side matches the scale of Streamo-Bench and is large enough for per-task slices.",
            "2500 videos gives a reserve for benchmark-hard subsets and scale ablation without changing per-video density.",
        ],
    }


def counter_report(report: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in report.items():
        if isinstance(value, Counter):
            out[key] = dict(value.most_common())
        else:
            out[key] = value
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = sorted(parse_float(r.get("duration_sec")) for r in rows)
    n = len(durations)
    return {
        "count": n,
        "duration_sec": {
            "min": durations[0],
            "p50": durations[n // 2],
            "p90": durations[int(n * 0.90)],
            "p95": durations[int(n * 0.95)],
            "max": durations[-1],
            "total_hours": round(sum(durations) / 3600.0, 3),
        },
        "by_duration_bucket": dict(Counter(r["duration_bucket"] for r in rows).most_common()),
        "by_dataset": dict(Counter(r["dataset"] for r in rows).most_common()),
        "by_source_group": dict(Counter(r["source_group"] for r in rows).most_common()),
        "by_selection_kind": dict(Counter(r["selection_kind"] for r in rows).most_common()),
        "streamo_overlap": {
            "true": sum(1 for r in rows if r.get("is_streamo")),
            "false": sum(1 for r in rows if not r.get("is_streamo")),
        },
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    stats = summarize(rows)
    dur = stats["duration_sec"]
    print(f"selected: {stats['count']} videos")
    print(
        "duration: "
        f"min={dur['min']:.1f}s p50={dur['p50']:.1f}s "
        f"p90={dur['p90']:.1f}s p95={dur['p95']:.1f}s "
        f"max={dur['max']:.1f}s total={dur['total_hours']:.2f}h"
    )
    print("duration buckets:")
    for bucket in DURATION_TARGETS:
        print(f"  {bucket:8s} {stats['by_duration_bucket'].get(bucket, 0):3d}")
    print("selection kinds:")
    for kind, count in stats["by_selection_kind"].items():
        print(f"  {kind:24s} {count:3d}")
    print("source groups:")
    for group, count in stats["by_source_group"].items():
        print(f"  {group:32s} {count:3d}")
    print(f"streamo overlap: {stats['streamo_overlap']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "agent_v5")
    parser.add_argument("--summary", type=Path, default=DEFAULT_MULTI_REPORT)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--batch-size", type=int, default=TARGET)
    parser.add_argument("--start-batch", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL files whose video_id/video_path should be excluded.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not exclude data/selected_videos.jsonl and local registries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_files = [] if args.no_default_excludes else default_exclude_files()
    exclude_files.extend(args.exclude_jsonl or [])
    if args.num_batches > 1:
        batches = select_multiple_batches(
            candidate_path=args.candidate_pool,
            output_dir=args.output_dir,
            start_batch=args.start_batch,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            seed=args.seed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            base_exclude_files=exclude_files,
            summary_path=args.summary,
        )
        all_rows = [row for _, rows in batches for row in rows]
        print_summary(all_rows)
        for batch_no, rows in batches:
            print(f"batch{batch_no}: {len(rows)} videos")
        print(f"summary: {args.summary}")
        return

    selected = select_batch2(
        candidate_path=args.candidate_pool,
        output_path=args.output,
        report_path=args.report,
        target=args.target,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        exclude_files=exclude_files,
    )
    print_summary(selected)
    print(f"wrote: {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
