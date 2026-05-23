#!/usr/bin/env python3
"""Curate visually clear agent_v5 trajectory examples.

The exporter builds a compact showcase package:
- ranked trajectory/video selections with visual quality metrics
- representative resized frames
- all questions for each selected trajectory
- chunk-level thinks and compact action trajectory
- dataset-level quality summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


TRAJECTORY_RE = re.compile(
    r'^\{"video_id": "(?P<video_id>(?:[^"\\]|\\.)*)", '
    r'"trajectory_id": "(?P<trajectory_id>(?:[^"\\]|\\.)*)", '
    r'"video_path": "(?P<video_path>(?:[^"\\]|\\.)*)"'
)


QUESTION_FIELDS = [
    "category",
    "family",
    "family_name",
    "task_family",
    "slot_subtype",
    "temporal_bucket",
    "benchmark_task",
    "answer_form",
    "question_type",
    "availability",
    "support_policy",
    "required_answer_mode",
    "question",
    "correct_answer_text",
    "correct_option",
    "options",
    "accepted_answers",
    "ask_chunk",
    "ask_time_sec",
    "ask_time_norm",
    "support_chunks",
    "answer_chunks",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_json(value: str, default: Any) -> Any:
    if value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        slug = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return slug[:max_len]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def frame_path(frame_dir: Path, frame_idx: int) -> Path:
    return frame_dir / f"frame_{frame_idx:06d}.jpg"


def nearest_frame(frame_dir: Path, frame_idx: int, frame_count: int) -> Path | None:
    if frame_count <= 0:
        return None
    idx = max(1, min(frame_count, frame_idx))
    for delta in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
        candidate = idx + delta
        if 1 <= candidate <= frame_count:
            path = frame_path(frame_dir, candidate)
            if path.exists():
                return path
    return None


def metric_for_frame(path: Path) -> dict[str, float] | None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((384, 384))
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "dark_fraction": float(np.mean(gray < 45)),
        "overexposed_fraction": float(np.mean(gray > 245)),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {
            "metric_frame_count": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "saturation": 0.0,
            "dark_fraction": 1.0,
            "overexposed_fraction": 1.0,
            "visual_score": 0.0,
        }
    agg = {
        "metric_frame_count": float(len(metrics)),
        "brightness": float(np.mean([m["brightness"] for m in metrics])),
        "contrast": float(np.mean([m["contrast"] for m in metrics])),
        "sharpness": float(np.mean([m["sharpness"] for m in metrics])),
        "saturation": float(np.mean([m["saturation"] for m in metrics])),
        "dark_fraction": float(np.mean([m["dark_fraction"] for m in metrics])),
        "overexposed_fraction": float(np.mean([m["overexposed_fraction"] for m in metrics])),
    }
    brightness_score = clamp(1.0 - abs(agg["brightness"] - 155.0) / 95.0)
    sharpness_score = clamp(math.log1p(agg["sharpness"]) / math.log1p(950.0))
    contrast_score = clamp(agg["contrast"] / 70.0)
    saturation_score = clamp(agg["saturation"] / 85.0)
    exposure_score = clamp(1.0 - agg["dark_fraction"] * 1.4 - agg["overexposed_fraction"] * 1.2)
    agg["visual_score"] = round(
        (
            0.34 * brightness_score
            + 0.28 * sharpness_score
            + 0.16 * contrast_score
            + 0.12 * saturation_score
            + 0.10 * exposure_score
        )
        * exposure_score,
        6,
    )
    return agg


def score_trajectory(row: dict[str, str], samples: int) -> dict[str, Any]:
    frame_dir = Path(row["frame_dir"])
    frame_count = safe_int(row["frame_count"])
    fps = safe_float(row["frame_fps"], 2.0)
    duration = safe_float(row["video_duration_sec"])
    if frame_count <= 0 or not frame_dir.is_dir():
        return {**row, **aggregate_metrics([]), "overall_score": 0.0, "quality_pass": False}
    if duration <= 0:
        duration = frame_count / max(fps, 1.0)
    sample_times = np.linspace(0.08, 0.92, samples)
    metrics: list[dict[str, float]] = []
    for frac in sample_times:
        frame_idx = int(round(duration * frac * fps)) + 1
        path = nearest_frame(frame_dir, frame_idx, frame_count)
        if path:
            metric = metric_for_frame(path)
            if metric:
                metrics.append(metric)
    agg = aggregate_metrics(metrics)
    n_questions = safe_int(row["n_questions"])
    duration_score = 1.0 if 45 <= duration <= 300 else clamp(1.0 - abs(duration - 160.0) / 280.0)
    question_score = clamp(n_questions / 18.0)
    selection_ratio_score = clamp(safe_float(row.get("questions_per_min")) / 8.0)
    overall = round(
        0.72 * agg["visual_score"]
        + 0.12 * duration_score
        + 0.10 * question_score
        + 0.06 * selection_ratio_score,
        6,
    )
    quality_pass = (
        agg["metric_frame_count"] >= max(4, samples // 2)
        and 80 <= agg["brightness"] <= 220
        and agg["dark_fraction"] <= 0.35
        and agg["overexposed_fraction"] <= 0.25
        and agg["visual_score"] >= 0.45
        and n_questions >= 8
        and 40 <= duration <= 330
    )
    return {
        **row,
        **agg,
        "overall_score": overall,
        "quality_pass": quality_pass,
    }


def trajectory_prior(row: dict[str, str]) -> float:
    duration = safe_float(row.get("video_duration_sec"))
    n_questions = safe_int(row.get("n_questions"))
    questions_per_min = safe_float(row.get("questions_per_min"))
    duration_score = 1.0 if 45 <= duration <= 300 else clamp(1.0 - abs(duration - 160.0) / 280.0)
    question_score = clamp(n_questions / 20.0)
    density_score = clamp(questions_per_min / 8.0)
    return round(0.45 * duration_score + 0.35 * question_score + 0.20 * density_score, 6)


def preselect_trajectories(
    rows: list[dict[str, str]],
    per_source: int,
    total: int,
) -> list[dict[str, str]]:
    filtered = [
        row
        for row in rows
        if safe_int(row.get("frame_count")) > 0
        and safe_int(row.get("n_questions")) >= 8
        and 40 <= safe_float(row.get("video_duration_sec")) <= 330
    ]
    for row in filtered:
        row["prior_score"] = trajectory_prior(row)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        by_source[row["source_dataset"]].append(row)
    pool: list[dict[str, str]] = []
    for source_rows in by_source.values():
        pool.extend(sorted(source_rows, key=lambda r: safe_float(r["prior_score"]), reverse=True)[:per_source])
    pool = sorted(pool, key=lambda r: safe_float(r["prior_score"]), reverse=True)[:total]
    return pool


def score_job(args: tuple[dict[str, str], int]) -> dict[str, Any]:
    row, samples = args
    return score_trajectory(row, samples)


def select_diverse(rows: list[dict[str, Any]], count: int, source_cap: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: safe_float(r["overall_score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    by_source: Counter[str] = Counter()
    seen_videos: set[str] = set()
    for row in sorted_rows:
        source = row["source_dataset"]
        if row["video_id"] in seen_videos:
            continue
        if by_source[source] >= source_cap:
            continue
        selected.append(row)
        by_source[source] += 1
        seen_videos.add(row["video_id"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        for row in sorted_rows:
            if row["video_id"] in seen_videos:
                continue
            selected.append(row)
            seen_videos.add(row["video_id"])
            if len(selected) >= count:
                break
    return selected


def parse_line_header(line: str) -> tuple[str, str, str]:
    match = TRAJECTORY_RE.match(line)
    if not match:
        raise ValueError("could not parse trajectory line header")
    return (
        json.loads(f'"{match.group("video_id")}"'),
        json.loads(f'"{match.group("trajectory_id")}"'),
        json.loads(f'"{match.group("video_path")}"'),
    )


def load_trajectory_object(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = root / row["batch"] / "final" / f'{row["split"]}_trajectories.jsonl'
    target_line = safe_int(row["line_no"])
    with path.open(encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            if idx != target_line:
                continue
            obj = json.loads(line)
            if obj.get("video_id") != row["video_id"] or obj.get("trajectory_id") != row["trajectory_id"]:
                raise ValueError(f"trajectory mismatch at {path}:{idx}")
            return obj
    raise FileNotFoundError(f"missing trajectory line {target_line} in {path}")


def load_rollout_thinks(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / row["batch"] / "rollout" / f'{row["video_id"]}.json'
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("thinks") or []


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    inp = sample.get("input") or {}
    meta = sample.get("metadata") or {}
    keep_meta = {
        key: meta.get(key)
        for key in [
            "card_id",
            "category",
            "family",
            "benchmark_task",
            "answer_form",
            "question_type",
            "availability",
            "ask_chunk",
            "question",
            "correct_answer_text",
            "correct_option",
            "sft_answer",
            "gold_answer",
        ]
        if key in meta
    }
    return {
        "chunk_idx": sample.get("chunk_idx"),
        "sample_type": sample.get("sample_type"),
        "action": sample.get("action"),
        "sequence_type": sample.get("sequence_type"),
        "card_id": sample.get("card_id"),
        "user_input": inp.get("user_input") if isinstance(inp, dict) else "",
        "visual_window": inp.get("visual_window") if isinstance(inp, dict) else {},
        "open_query_count": len(inp.get("queries") or []) if isinstance(inp, dict) else 0,
        "output": sample.get("output", ""),
        "metadata": keep_meta,
    }


def choose_frame_chunks(questions: list[dict[str, Any]], n_chunks: int, limit: int) -> list[int]:
    chunks: set[int] = set()
    for q in questions:
        if isinstance(q.get("ask_chunk"), int):
            chunks.add(q["ask_chunk"])
        for key in ["answer_chunks", "expected_answer_chunks", "support_chunks"]:
            vals = q.get(key) or []
            if vals:
                chunks.add(int(vals[-1]))
    ordered = sorted(c for c in chunks if 0 <= c < n_chunks)
    if len(ordered) > limit:
        idxs = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
        ordered = [ordered[i] for i in idxs]
    fill = [int(round(x)) for x in np.linspace(0, max(0, n_chunks - 1), limit)]
    for chunk in fill:
        if len(ordered) >= limit:
            break
        if chunk not in ordered:
            ordered.append(chunk)
    return sorted(set(ordered))[:limit]


def copy_resized_frame(src: Path, dst: Path, max_side: int, quality: int) -> dict[str, Any]:
    img = Image.open(src).convert("RGB")
    original_size = img.size
    img.thumbnail((max_side, max_side))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=quality, optimize=True)
    return {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "export_width": img.size[0],
        "export_height": img.size[1],
    }


def source_dataset_summary(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row["source_dataset"]].append(row)
    rows = []
    for source, items in grouped.items():
        passes = [r for r in items if r["quality_pass"]]
        rows.append(
            {
                "source_dataset": source,
                "trajectory_count": len(items),
                "quality_pass_count": len(passes),
                "quality_pass_rate": round(len(passes) / len(items), 6) if items else 0,
                "mean_overall_score": round(float(np.mean([safe_float(r["overall_score"]) for r in items])), 6),
                "mean_visual_score": round(float(np.mean([safe_float(r["visual_score"]) for r in items])), 6),
                "mean_brightness": round(float(np.mean([safe_float(r["brightness"]) for r in items])), 3),
                "mean_sharpness": round(float(np.mean([safe_float(r["sharpness"]) for r in items])), 3),
                "mean_duration_sec": round(float(np.mean([safe_float(r["video_duration_sec"]) for r in items])), 3),
                "mean_questions": round(float(np.mean([safe_float(r["n_questions"]) for r in items])), 3),
            }
        )
    return sorted(rows, key=lambda r: r["mean_overall_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/agent_v5"))
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("data/agent_v5/statistics/batch1_10_structured"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agent_v5/curated/batch1_10_showcase_100"),
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--score-frame-samples", type=int, default=8)
    parser.add_argument("--export-frame-chunks", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--source-cap", type=int, default=28)
    parser.add_argument("--preselect-per-source", type=int, default=240)
    parser.add_argument("--preselect-total", type=int, default=1400)
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    trajectories = load_csv(args.stats_dir / "trajectories.csv")
    candidate_pool = preselect_trajectories(
        trajectories,
        per_source=args.preselect_per_source,
        total=args.preselect_total,
    )
    print(
        f"Scoring {len(candidate_pool)} preselected trajectories "
        f"from {len(trajectories)} total...",
        flush=True,
    )
    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            scored = list(pool.imap_unordered(score_job, [(row, args.score_frame_samples) for row in candidate_pool]))
    else:
        scored = [score_trajectory(row, args.score_frame_samples) for row in candidate_pool]
    quality_rows = [row for row in scored if row["quality_pass"]]
    selected = select_diverse(quality_rows, args.count, args.source_cap)
    if len(selected) < args.count:
        raise RuntimeError(f"only selected {len(selected)} trajectories")

    if args.output.exists():
        shutil.rmtree(args.output)
    (args.output / "frames").mkdir(parents=True)

    questions_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_csv(args.stats_dir / "questions.csv"):
        key = (row["batch"], row["split"], row["video_id"], row["trajectory_id"])
        questions_by_key[key].append(row)

    dataset_summary = source_dataset_summary(scored)
    write_csv(
        args.output / "dataset_quality_summary.csv",
        dataset_summary,
        [
            "source_dataset",
            "trajectory_count",
            "quality_pass_count",
            "quality_pass_rate",
            "mean_overall_score",
            "mean_visual_score",
            "mean_brightness",
            "mean_sharpness",
            "mean_duration_sec",
            "mean_questions",
        ],
    )

    selected_rows: list[dict[str, Any]] = []
    selected_questions: list[dict[str, Any]] = []
    frames_manifest: list[dict[str, Any]] = []
    gallery_items: list[str] = []

    timeline_path = args.output / "trajectory_timelines.jsonl"
    with timeline_path.open("w", encoding="utf-8") as timeline_f:
        for rank, row in enumerate(selected, start=1):
            key = (row["batch"], row["split"], row["video_id"], row["trajectory_id"])
            traj_obj = load_trajectory_object(args.root, row)
            rollout_thinks = load_rollout_thinks(args.root, row)
            n_chunks = safe_int(row["n_chunks_covered"])
            frame_dir = Path(row["frame_dir"])
            frame_count = safe_int(row["frame_count"])
            fps = safe_float(row["frame_fps"], 2.0)
            out_slug = f"{rank:03d}_{safe_slug(row['video_id'])}"
            out_frame_dir = args.output / "frames" / out_slug

            frame_chunks = choose_frame_chunks(
                traj_obj.get("questions") or [], n_chunks, args.export_frame_chunks
            )
            copied_frames: list[dict[str, Any]] = []
            for chunk in frame_chunks:
                frame_idx = int(round(chunk * fps)) + 1
                src = nearest_frame(frame_dir, frame_idx, frame_count)
                if not src:
                    continue
                dst = out_frame_dir / f"chunk_{chunk:04d}_frame_{frame_idx:06d}.jpg"
                size_info = copy_resized_frame(src, dst, args.max_side, args.jpeg_quality)
                rel_dst = dst.relative_to(args.output)
                rel_src = src
                frame_row = {
                    "rank": rank,
                    "video_id": row["video_id"],
                    "batch": row["batch"],
                    "split": row["split"],
                    "trajectory_id": row["trajectory_id"],
                    "chunk_idx": chunk,
                    "time_sec": round(chunk, 3),
                    "source_frame_path": str(rel_src),
                    "export_frame_path": str(rel_dst),
                    **size_info,
                }
                copied_frames.append(frame_row)
                frames_manifest.append(frame_row)

            compact_questions = traj_obj.get("questions") or []
            compact_samples = [compact_sample(sample) for sample in traj_obj.get("samples") or []]
            action_counts = Counter(sample.get("action") for sample in traj_obj.get("samples") or [])
            selected_question_rows = questions_by_key[key]
            for q_idx, qrow in enumerate(selected_question_rows):
                selected_questions.append({"rank": rank, **qrow})

            selected_row = {
                "rank": rank,
                "video_id": row["video_id"],
                "batch": row["batch"],
                "split": row["split"],
                "trajectory_id": row["trajectory_id"],
                "source_dataset": row["source_dataset"],
                "source_relpath": row["source_relpath"],
                "n_questions": row["n_questions"],
                "n_chunks_covered": row["n_chunks_covered"],
                "video_duration_sec": row["video_duration_sec"],
                "questions_per_min": row["questions_per_min"],
                "overall_score": row["overall_score"],
                "visual_score": row["visual_score"],
                "brightness": row["brightness"],
                "contrast": row["contrast"],
                "sharpness": row["sharpness"],
                "saturation": row["saturation"],
                "dark_fraction": row["dark_fraction"],
                "overexposed_fraction": row["overexposed_fraction"],
                "export_frame_count": len(copied_frames),
                "frame_dir": row["frame_dir"],
                "video_path": row["video_path"],
                "action_counts": dict(action_counts),
                "top_frame": copied_frames[0]["export_frame_path"] if copied_frames else "",
            }
            selected_rows.append(selected_row)

            timeline_obj = {
                "rank": rank,
                "selection": selected_row,
                "questions": compact_questions,
                "samples": compact_samples,
                "rollout_thinks": rollout_thinks,
                "exported_frames": copied_frames,
                "stats": traj_obj.get("stats", {}),
            }
            timeline_f.write(json.dumps(timeline_obj, ensure_ascii=False) + "\n")

            first_frame = copied_frames[0]["export_frame_path"] if copied_frames else ""
            first_questions = selected_question_rows[:3]
            q_html = "".join(
                f"<li><b>{html.escape(q.get('benchmark_task', ''))}</b>: "
                f"{html.escape(q.get('question', '')[:220])}</li>"
                for q in first_questions
            )
            gallery_items.append(
                f"<section><h2>#{rank:03d} {html.escape(row['video_id'])}</h2>"
                f"<p>{html.escape(row['source_dataset'])} | score {row['overall_score']} | "
                f"{row['n_questions']} questions | {row['video_duration_sec']}s</p>"
                f"<img src=\"{html.escape(first_frame)}\" loading=\"lazy\" />"
                f"<ol>{q_html}</ol></section>"
            )

    selected_fields = [
        "rank",
        "video_id",
        "batch",
        "split",
        "trajectory_id",
        "source_dataset",
        "source_relpath",
        "n_questions",
        "n_chunks_covered",
        "video_duration_sec",
        "questions_per_min",
        "overall_score",
        "visual_score",
        "brightness",
        "contrast",
        "sharpness",
        "saturation",
        "dark_fraction",
        "overexposed_fraction",
        "export_frame_count",
        "top_frame",
        "action_counts",
        "frame_dir",
        "video_path",
    ]
    write_csv(args.output / "selected_trajectories.csv", selected_rows, selected_fields)
    question_fields = ["rank", *load_csv(args.stats_dir / "questions.csv")[0].keys()]
    write_csv(args.output / "selected_questions.csv", selected_questions, question_fields)
    write_csv(
        args.output / "frames_manifest.csv",
        frames_manifest,
        [
            "rank",
            "video_id",
            "batch",
            "split",
            "trajectory_id",
            "chunk_idx",
            "time_sec",
            "source_frame_path",
            "export_frame_path",
            "original_width",
            "original_height",
            "export_width",
            "export_height",
        ],
    )

    source_counts = Counter(row["source_dataset"] for row in selected_rows)
    category_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    for q in selected_questions:
        category_counts[q.get("category", "")] += 1
        benchmark_counts[q.get("benchmark_task", "")] += 1

    manifest = {
        "name": "agent_v5_batch1_10_showcase_100",
        "source_root": str(args.root),
        "stats_dir": str(args.stats_dir),
        "selection_count": len(selected_rows),
        "export_frame_chunks_per_trajectory": args.export_frame_chunks,
        "frame_max_side": args.max_side,
        "jpeg_quality": args.jpeg_quality,
        "criteria": {
            "quality_pass": "brightness 80-220, dark_fraction <= 0.35, overexposed_fraction <= 0.25, visual_score >= 0.45, n_questions >= 8, duration 40-330s",
            "ranking": "0.72*visual_score + 0.12*duration_score + 0.10*question_score + 0.06*question_density_score",
            "diversity": f"at most {args.source_cap} trajectories from one source before refill",
        },
        "selected_source_counts": dict(source_counts),
        "selected_category_counts": dict(category_counts),
        "selected_benchmark_task_counts": dict(benchmark_counts),
        "files": {
            "selected_trajectories_csv": "selected_trajectories.csv",
            "selected_questions_csv": "selected_questions.csv",
            "trajectory_timelines_jsonl": "trajectory_timelines.jsonl",
            "frames_manifest_csv": "frames_manifest.csv",
            "dataset_quality_summary_csv": "dataset_quality_summary.csv",
            "gallery_html": "gallery.html",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    html_doc = """<!doctype html>
<html><head><meta charset="utf-8"><title>agent_v5 showcase 100</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f2328}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}
section{background:white;border:1px solid #ddd;border-radius:8px;padding:12px}
img{width:100%;height:auto;border-radius:4px;background:#eee}
h1{margin-top:0} h2{font-size:16px;margin:8px 0} p,li{font-size:13px;line-height:1.4}
ol{padding-left:18px}
</style></head><body><h1>agent_v5 batch1-10 showcase 100</h1><main>
"""
    html_doc += "\n".join(gallery_items)
    html_doc += "\n</main></body></html>\n"
    (args.output / "gallery.html").write_text(html_doc, encoding="utf-8")

    readme = f"""# agent_v5 batch1-10 showcase 100

This package contains 100 curated final trajectories selected for bright, clear representative frames and useful question coverage.

Contents:
- `selected_trajectories.csv`: selected trajectory metadata and visual-quality scores.
- `selected_questions.csv`: all actual questions in the selected trajectories.
- `trajectory_timelines.jsonl`: compact per-trajectory records with questions, chunk-level samples, rollout thinks, and exported frame metadata.
- `frames/`: resized representative frames copied from the original frame directories.
- `frames_manifest.csv`: mapping from exported frames back to original frame files and chunks.
- `dataset_quality_summary.csv`: source-dataset quality analysis used before selection.
- `gallery.html`: lightweight visual browser for the selected examples.

Frame export:
- {args.export_frame_chunks} representative chunk frames per trajectory.
- JPEG max side {args.max_side}, quality {args.jpeg_quality}.
- Original frame paths are kept in `frames_manifest.csv`.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote showcase to {args.output}")


if __name__ == "__main__":
    main()
