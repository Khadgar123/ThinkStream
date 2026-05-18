#!/usr/bin/env python3
"""Build StreamingBench rows for the ThinkStream recurrent RL evaluator.

This converts mjuicem/StreamingBench CSV rows into the same multi-question
trajectory JSONL/parquet schema used by scripts/eval/ovo/run_rl_recurrent_eval.sh.
It intentionally includes Proactive Output rows, which the vLLM base runner
skips because they are not multiple-choice questions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkstream.eval.prompt_contract import build_streaming_query_meta  # noqa: E402


TIME_RE = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)$")
QUESTION_RE = re.compile(r"^(?P<family>.+?)_sample_(?P<sample>\d+)_")
SAMPLE_RE = re.compile(r"sample_(?P<sample>\d+)")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SPLIT_POLICIES = {"continuous_prefix", "strict_window"}
CURRENT_SPLIT_POLICY = "continuous_prefix"
TASK_ALIASES = {
    "OP": "Object Perception",
    "CR": "Causal Reasoning",
    "CS": "Clips Summarize",
    "ATP": "Attribute Perception",
    "EU": "Event Understanding",
    "TR": "Text-Rich Understanding",
    "PR": "Prospective Reasoning",
    "SU": "Spatial Understanding",
    "ACP": "Action Perception",
    "CT": "Counting",
}


@dataclass(frozen=True)
class BenchRow:
    idx: int
    csv_name: str
    question_id: str
    family: str
    sample_id: int
    task_type: str
    question: str
    ask_sec: int
    answer_sec: int
    answer: str
    options: List[str]
    temporal_clue_type: str
    frames_required: str
    source_video: Path
    video_rel: str


def _parse_time(value: str) -> int:
    text = str(value or "").strip()
    match = TIME_RE.match(text)
    if not match:
        raise ValueError(f"bad time value: {value!r}")
    h = int(match.group("h") or 0)
    m = int(match.group("m") or 0)
    s = int(match.group("s") or 0)
    return h * 3600 + m * 60 + s


def _parse_options(value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(str(value or ""))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _parse_task_filter(value: str) -> Optional[set[str]]:
    raw_items = [x.strip() for x in str(value or "").split(",") if x.strip()]
    if not raw_items:
        return None
    out: set[str] = set()
    unknown: List[str] = []
    for item in raw_items:
        key = item.upper()
        if key in TASK_ALIASES:
            out.add(TASK_ALIASES[key])
        elif item in TASK_ALIASES.values():
            out.add(item)
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(
            "unknown StreamingBench task filter(s): "
            + ", ".join(sorted(unknown))
            + f"; known aliases: {', '.join(sorted(TASK_ALIASES))}"
        )
    return out


def _question_family_sample(qid: str, task_type: str, fallback_idx: int) -> Tuple[str, int]:
    match = QUESTION_RE.match(str(qid or ""))
    if match:
        return match.group("family"), int(match.group("sample"))
    nums = re.findall(r"\d+", str(qid or ""))
    sample_id = int(nums[-1]) if nums else fallback_idx
    return task_type or "StreamingBench", sample_id


class VideoResolver:
    def __init__(self, video_root: Path):
        self.video_root = video_root
        self._by_sample: Dict[int, List[Path]] = defaultdict(list)
        for path in sorted(video_root.glob("**/sample_*/*.mp4")):
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            match = SAMPLE_RE.search(str(path))
            if match:
                self._by_sample[int(match.group("sample"))].append(path)

    def resolve(self, *, family: str, task_type: str, sample_id: int) -> Optional[Path]:
        candidates = self._by_sample.get(int(sample_id), [])
        if not candidates:
            return None
        family_norm = _normalize(family)
        task_norm = _normalize(task_type)
        best: Optional[Tuple[int, Path]] = None
        for path in candidates:
            parent_norm = _normalize(" ".join(path.parts))
            score = 0
            if family_norm and family_norm in parent_norm:
                score += 1000
            if task_norm and task_norm in parent_norm:
                score += 500
            for token in re.findall(r"[A-Za-z]+", family + " " + task_type):
                if _normalize(token) in parent_norm:
                    score += 1
            if best is None or score > best[0]:
                best = (score, path)
        if best and best[0] > 0:
            return best[1]
        return candidates[0] if len(candidates) == 1 else None


def _frame_dir_exists(frames_root: Path, video_rel: str) -> bool:
    vp = Path(video_rel)
    candidates = [
        frames_root / vp.parent,
        frames_root / vp.with_suffix(""),
        frames_root / vp.stem,
        frames_root / vp.with_suffix("").name,
    ]
    return any(c.exists() and any(c.glob("frame_*.jpg")) for c in candidates)


def load_rows(csv_dir: Path, video_root: Path, frames_root: Path) -> Tuple[List[BenchRow], Dict[str, Any]]:
    resolver = VideoResolver(video_root)
    rows: List[BenchRow] = []
    skipped = Counter()
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw_i, raw in enumerate(reader):
                task_type = str(raw.get("task_type") or "").strip()
                qid = str(raw.get("question_id") or "").strip()
                family, sample_id = _question_family_sample(qid, task_type, raw_i + 1)
                try:
                    ask_sec = _parse_time(str(raw.get("time_stamp") or ""))
                except ValueError:
                    skipped["bad_time"] += 1
                    continue
                answer_sec = ask_sec
                answer = str(raw.get("answer") or "").strip()
                options = _parse_options(str(raw.get("options") or ""))
                if not answer:
                    answer = str(raw.get("ground_truth_output") or "").strip()
                    try:
                        answer_sec = _parse_time(str(raw.get("ground_truth_time_stamp") or ""))
                    except ValueError:
                        answer_sec = ask_sec
                if not answer:
                    skipped["missing_answer"] += 1
                    continue
                video = resolver.resolve(family=family, task_type=task_type, sample_id=sample_id)
                if video is None:
                    skipped["missing_video"] += 1
                    continue
                try:
                    video_rel = video.relative_to(video_root).as_posix()
                except ValueError:
                    video_rel = video.as_posix()
                if not _frame_dir_exists(frames_root, video_rel):
                    skipped["missing_frames"] += 1
                    continue
                if options and answer not in set(LETTERS[: len(options)]):
                    skipped["bad_mc_answer"] += 1
                    continue
                rows.append(BenchRow(
                    idx=len(rows),
                    csv_name=csv_path.name,
                    question_id=qid,
                    family=family,
                    sample_id=sample_id,
                    task_type=task_type,
                    question=str(raw.get("question") or "").strip(),
                    ask_sec=int(ask_sec),
                    answer_sec=int(max(ask_sec, answer_sec)),
                    answer=answer,
                    options=options,
                    temporal_clue_type=str(raw.get("temporal_clue_type") or "").strip(),
                    frames_required=str(raw.get("frames_required") or "").strip(),
                    source_video=video,
                    video_rel=video_rel,
                ))
    return rows, {"skipped": dict(skipped)}


def _strip_option_label(option: str) -> str:
    return re.sub(r"^\s*[A-E][\).:]\s*", "", str(option or "")).strip()


def _answer_form(row: BenchRow) -> str:
    if row.options:
        return "multiple_choice"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", row.answer.strip()):
        return "number"
    return "short_exact"


def _support_start(row: BenchRow, window_sec: int) -> int:
    if row.temporal_clue_type == "Concurrent":
        return row.ask_sec
    if row.temporal_clue_type == "Subsequent":
        return row.ask_sec
    return max(0, row.ask_sec - int(window_sec))


def _question_payload(row: BenchRow, *, support_window_sec: int) -> Dict[str, Any]:
    form = _answer_form(row)
    q: Dict[str, Any] = {
        "card_id": row.question_id or f"streamingbench:{row.idx}",
        "family": row.task_type,
        "family_name": row.task_type,
        "category": row.csv_name.removesuffix(".csv"),
        "skill": "streamingbench",
        "question": row.question,
        "ask_chunk": int(row.ask_sec),
        "ask_chunks": [int(row.ask_sec)],
        "answer_chunks": [int(row.answer_sec)],
        "per_emit_answers": [{"chunk": int(row.answer_sec), "value": row.answer}],
        "gold_answer": row.answer,
        "answer_form": form,
        "support_chunks": list(range(_support_start(row, support_window_sec), row.answer_sec + 1)),
        "streamingbench_csv": row.csv_name,
        "streamingbench_question_id": row.question_id,
        "streamingbench_family": row.family,
        "streamingbench_sample_id": row.sample_id,
        "streamingbench_task_type": row.task_type,
        "streamingbench_temporal_clue_type": row.temporal_clue_type,
        "streamingbench_frames_required": row.frames_required,
        "streamingbench_ask_sec": row.ask_sec,
        "streamingbench_answer_sec": row.answer_sec,
        "open_until": int(row.answer_sec),
    }
    if row.options:
        q["options"] = list(row.options)
        q["correct_option"] = row.answer
        idx = LETTERS.index(row.answer) if row.answer in LETTERS else -1
        if 0 <= idx < len(row.options):
            q["correct_answer_text"] = _strip_option_label(row.options[idx])
    meta = build_streaming_query_meta(q, answer_form=form)
    for key in ("answer_instruction", "answer_style"):
        if key in meta:
            q[key] = meta[key]
    return q


def _bounds(row: BenchRow, *, support_window_sec: int, min_span: int, max_span: int, post_context: int) -> Tuple[int, int, bool]:
    min_start = _support_start(row, support_window_sec)
    end = int(row.answer_sec) + int(post_context)
    if row.temporal_clue_type == "Subsequent":
        start = max(0, row.ask_sec - 1)
    else:
        start = max(0, min(min_start, end - max_span + 1))
    if start == row.ask_sec and start > 0:
        start -= 1
    if end - start + 1 < min_span:
        end = start + min_span - 1
    return int(start), int(end), bool(end - start + 1 > max_span)


def _trajectory(row: BenchRow, *, support_window_sec: int, min_span: int, max_span: int, post_context: int) -> Dict[str, Any]:
    q = _question_payload(row, support_window_sec=support_window_sec)
    start, end, span_exceeded = _bounds(
        row,
        support_window_sec=support_window_sec,
        min_span=min_span,
        max_span=max_span,
        post_context=post_context,
    )
    tid = (
        f"streamingbench__{Path(row.video_rel).with_suffix('').as_posix().replace('/', '__')}"
        f"__{row.idx:05d}"
    )
    return {
        "trajectory_id": tid,
        "video_id": tid,
        "source_video_path": row.video_rel,
        "video_path": row.video_rel,
        "segment_start_chunk": start,
        "segment_end_chunk": end,
        "questions": [q],
        "gold_action_per_chunk": {str(int(row.answer_sec)): "response"},
        "offline_compress_chunks": [],
        "samples": [],
        "stats": {
            "n_chunks_covered": end + 1,
            "chunk_idx_max": end,
            "n_questions": 1,
        },
        "ovo_split_meta": {
            "benchmark": "StreamingBench",
            "source_csv": row.csv_name,
            "task_type": row.task_type,
            "temporal_clue_type": row.temporal_clue_type,
            "frames_required": row.frames_required,
            "sample_id": row.sample_id,
            "question_id": row.question_id,
            "segment_start_chunk": start,
            "segment_end_chunk": end,
            "span_exceeded_soft_limit": span_exceeded,
            "benchmark_track": "strict_window_with_future_wait",
        },
    }


def _question_window(q: Dict[str, Any]) -> Tuple[int, int]:
    starts: List[int] = []
    ends: List[int] = []
    for raw in q.get("ask_chunks") or [q.get("ask_chunk")]:
        try:
            starts.append(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in q.get("answer_chunks") or [q.get("open_until")]:
        try:
            ends.append(int(raw))
        except (TypeError, ValueError):
            continue
    start = min(starts) if starts else 0
    end = max(ends) if ends else start
    return start, max(start, end)


def _windows_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _question_support_intervals(q: Dict[str, Any]) -> List[Tuple[int, int]]:
    chunks: List[int] = []
    for raw in q.get("support_chunks") or []:
        try:
            chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    return [(min(chunks), max(chunks))] if chunks else []


def _question_anchor_points(q: Dict[str, Any]) -> List[int]:
    points: List[int] = []
    for key in ("ask_chunks", "answer_chunks"):
        for raw in q.get(key) or []:
            try:
                points.append(int(raw))
            except (TypeError, ValueError):
                continue
    for key in ("ask_chunk", "open_until"):
        try:
            points.append(int(q[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _compress_boundary_penalty(boundary: int, questions: List[Dict[str, Any]]) -> int:
    """Lower is better. Avoid splitting answer-active and evidence windows."""
    penalty = 0
    b = int(boundary)
    for q in questions:
        active_start, active_end = _question_window(q)
        if active_start <= b <= active_end:
            penalty += 1_000_000_000
        elif active_start - 2 <= b <= active_end + 2:
            penalty += 5_000_000

        for support_start, support_end in _question_support_intervals(q):
            # Boundary b is between b-1 and b. s < b <= e splits evidence.
            if support_start < b <= support_end:
                penalty += 100_000_000
            elif support_start - 2 <= b <= support_end + 2:
                penalty += 1_000_000

        for point in _question_anchor_points(q):
            dist = abs(b - point)
            if dist == 0:
                penalty += 10_000_000
            elif dist <= 2:
                penalty += 5_000_000
            elif dist <= 5:
                penalty += 2_000_000
            elif dist <= 10:
                penalty += 1_000_000
            elif dist <= 15:
                penalty += 100_000
            elif dist <= 20:
                penalty += 10_000
    return penalty


def _plan_offline_compress_chunks(
    questions: List[Dict[str, Any]],
    *,
    segment_start: int,
    segment_end: int,
    min_chunks: int,
    max_chunks: int,
) -> List[int]:
    """Plan compact-memory boundaries inside a complete continuous trajectory."""
    min_chunks = max(1, int(min_chunks))
    max_chunks = max(min_chunks, int(max_chunks))
    start = max(0, int(segment_start))
    end_exclusive = max(start, int(segment_end) + 1)
    cursor = start
    boundaries: List[int] = []
    while end_exclusive - cursor > max_chunks:
        lo = cursor + min_chunks
        hi = min(cursor + max_chunks, end_exclusive - 1)
        if lo > hi:
            break
        boundary = min(
            range(lo, hi + 1),
            key=lambda b: (_compress_boundary_penalty(b, questions), -int(b)),
        )
        boundaries.append(int(boundary))
        cursor = int(boundary)
    return boundaries


def _pack_non_overlapping_questions(
    questions: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    windows_by_group: List[List[Tuple[int, int]]] = []
    ordered = sorted(
        questions,
        key=lambda q: (_question_window(q)[0], _question_window(q)[1], str(q.get("card_id") or "")),
    )
    for q in ordered:
        window = _question_window(q)
        placed = False
        for gi, existing in enumerate(windows_by_group):
            if any(_windows_overlap(window, other) for other in existing):
                continue
            groups[gi].append(q)
            existing.append(window)
            placed = True
            break
        if not placed:
            groups.append([q])
            windows_by_group.append([window])
    return groups


def _continuous_prefix_trajectories(
    rows: List[BenchRow],
    *,
    support_window_sec: int,
    post_context: int,
    offline_compress_min_chunks: int,
    offline_compress_max_chunks: int,
) -> List[Dict[str, Any]]:
    by_video: Dict[str, List[BenchRow]] = defaultdict(list)
    for row in rows:
        by_video[row.video_rel].append(row)

    trajectories: List[Dict[str, Any]] = []
    for video_rel, video_rows in sorted(by_video.items()):
        questions = [
            _question_payload(row, support_window_sec=support_window_sec)
            for row in sorted(video_rows, key=lambda r: (r.ask_sec, r.answer_sec, r.question_id))
        ]
        groups = _pack_non_overlapping_questions(questions)
        for group_i, group in enumerate(groups):
            end = max((_question_window(q)[1] for q in group), default=0) + max(0, int(post_context))
            offline_compress_chunks = _plan_offline_compress_chunks(
                group,
                segment_start=0,
                segment_end=int(end),
                min_chunks=offline_compress_min_chunks,
                max_chunks=offline_compress_max_chunks,
            )
            tid = (
                f"streamingbench__{Path(video_rel).with_suffix('').as_posix().replace('/', '__')}"
                f"__prefix{group_i:03d}"
            )
            gold_action: Dict[str, str] = {}
            for q in group:
                for ck in q.get("answer_chunks") or []:
                    try:
                        gold_action[str(int(ck))] = "response"
                    except (TypeError, ValueError):
                        continue
            trajectories.append({
                "trajectory_id": tid,
                "video_id": tid,
                "source_video_path": video_rel,
                "video_path": video_rel,
                "segment_start_chunk": 0,
                "segment_end_chunk": int(end),
                "questions": group,
                "gold_action_per_chunk": gold_action,
                "offline_compress_chunks": offline_compress_chunks,
                "samples": [],
                "stats": {
                    "n_chunks_covered": int(end) + 1,
                    "chunk_idx_max": int(end),
                    "n_questions": len(group),
                    "n_offline_compress_chunks": len(offline_compress_chunks),
                },
                "ovo_split_meta": {
                    "benchmark": "StreamingBench",
                    "source_video_path": video_rel,
                    "source_csv": sorted({str(q.get("streamingbench_csv") or "") for q in group}),
                    "tasks": sorted({str(q.get("streamingbench_task_type") or q.get("family") or "") for q in group}),
                    "sample_ids": sorted({str(q.get("streamingbench_sample_id") or "") for q in group}),
                    "question_ids": [str(q.get("streamingbench_question_id") or q.get("card_id") or "") for q in group],
                    "segment_start_chunk": 0,
                    "segment_end_chunk": int(end),
                    "split_policy": "continuous_prefix",
                    "benchmark_track": "continuous_prefix",
                    "post_context_chunks": int(post_context),
                    "offline_compress_policy": "planned_avoid_answer_support_windows",
                    "offline_compress_min_chunks": int(offline_compress_min_chunks),
                    "offline_compress_max_chunks": int(offline_compress_max_chunks),
                },
            })
    return trajectories


def _sample_rows(rows: List[BenchRow], *, per_task: int, seed: int, limit: int) -> List[BenchRow]:
    rng = random.Random(seed)
    by_task: Dict[str, List[BenchRow]] = defaultdict(list)
    for row in rows:
        by_task[row.task_type].append(row)
    selected: List[BenchRow] = []
    for task in sorted(by_task):
        candidates = sorted(
            by_task[task],
            key=lambda r: (
                r.answer_sec - r.ask_sec,
                r.csv_name,
                r.sample_id,
                r.question_id,
            ),
        )
        if per_task > 0 and len(candidates) > per_task:
            head = candidates[: max(per_task * 3, per_task)]
            candidates = rng.sample(head, per_task)
        selected.extend(candidates[:per_task] if per_task > 0 else candidates)
    selected = sorted(selected, key=lambda r: (r.task_type, r.csv_name, r.sample_id, r.question_id))
    return selected[:limit] if limit > 0 else selected


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(jsonl_path: Path, parquet_path: Path, *, max_questions_per_traj: int) -> int:
    import pandas as pd
    from scripts.agent_data.build_verl_parquet import _iter_rows_multi_q

    rows = list(
        _iter_rows_multi_q(
            jsonl_path,
            max_questions_per_traj=max_questions_per_traj,
            frame_protocol="video_meta",
            render_layout="standard_query_last",
            include_student_cache=False,
        )
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return len(rows)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rows, load_summary = load_rows(args.csv_dir, args.video_root, args.frames_root)
    task_filter = _parse_task_filter(args.tasks)
    if task_filter:
        rows = [row for row in rows if row.task_type in task_filter]
    selected = _sample_rows(
        rows,
        per_task=max(0, int(args.sample_per_task_type)),
        seed=int(args.seed),
        limit=max(0, int(args.limit)),
    )
    split_policy = str(args.split_policy)
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(f"split_policy must be one of {sorted(SPLIT_POLICIES)}, got {split_policy!r}")
    if split_policy == "continuous_prefix":
        trajectories = _continuous_prefix_trajectories(
            selected,
            support_window_sec=int(args.support_window_sec),
            post_context=int(args.post_context_chunks),
            offline_compress_min_chunks=int(args.offline_compress_min_chunks),
            offline_compress_max_chunks=int(args.offline_compress_max_chunks),
        )
    else:
        trajectories = [
            _trajectory(
                row,
                support_window_sec=int(args.support_window_sec),
                min_span=int(args.min_span_chunks),
                max_span=int(args.max_span_chunks),
                post_context=int(args.post_context_chunks),
            )
            for row in selected
        ]
    out_jsonl = Path(args.out_jsonl)
    _write_jsonl(out_jsonl, trajectories)

    summary: Dict[str, Any] = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "trajectories": len(trajectories),
        "questions": sum(len(t.get("questions") or []) for t in trajectories),
        "tasks": dict(Counter(r.task_type for r in selected)),
        "csv": dict(Counter(r.csv_name for r in selected)),
        "temporal_clue_type": dict(Counter(r.temporal_clue_type for r in selected)),
        "answer_form": dict(Counter(_answer_form(r) for r in selected)),
        "span_over_max": sum(
            1 for t in trajectories if (t.get("ovo_split_meta") or {}).get("span_exceeded_soft_limit")
        ),
        "max_span": max(
            (
                int(t["segment_end_chunk"]) - int(t["segment_start_chunk"]) + 1
                for t in trajectories
            ),
            default=0,
        ),
        "sample_per_task_type": int(args.sample_per_task_type),
        "limit": int(args.limit),
        "requested_tasks": sorted(task_filter) if task_filter else [],
        "split_policy": split_policy,
        "continuous_prefix": split_policy == "continuous_prefix",
        "max_questions_per_trajectory": (
            None if split_policy == "continuous_prefix" else int(args.max_questions_per_trajectory)
        ),
        "requested_max_questions_per_trajectory": int(args.max_questions_per_trajectory),
        "offline_compress_policy": (
            "planned_avoid_answer_support_windows"
            if split_policy == "continuous_prefix"
            else ""
        ),
        "offline_compress_min_chunks": int(args.offline_compress_min_chunks),
        "offline_compress_max_chunks": int(args.offline_compress_max_chunks),
        "offline_compress_events": sum(len(t.get("offline_compress_chunks") or []) for t in trajectories),
        **load_summary,
    }
    if args.out_parquet:
        parquet_max_questions = max(
            (len(t.get("questions") or []) for t in trajectories),
            default=1,
        ) if split_policy == "continuous_prefix" else max(1, int(args.max_questions_per_trajectory))
        summary["parquet_max_questions_per_traj"] = int(parquet_max_questions)
        summary["parquet_rows"] = _write_parquet(
            out_jsonl,
            Path(args.out_parquet),
            max_questions_per_traj=int(parquet_max_questions),
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-parquet", default="")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--sample-per-task-type", type=int, default=0)
    ap.add_argument(
        "--tasks",
        default="",
        help=(
            "Comma-separated task names or aliases. Aliases: "
            "OP, CR, CS, ATP, EU, TR, PR, SU, ACP, CT."
        ),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--support-window-sec", type=int, default=32)
    ap.add_argument("--min-span-chunks", type=int, default=25)
    ap.add_argument("--max-span-chunks", type=int, default=45)
    ap.add_argument("--post-context-chunks", type=int, default=2)
    ap.add_argument("--offline-compress-min-chunks", type=int, default=25)
    ap.add_argument("--offline-compress-max-chunks", type=int, default=45)
    ap.add_argument("--max-questions-per-trajectory", type=int, default=16)
    ap.add_argument(
        "--split-policy",
        default=CURRENT_SPLIT_POLICY,
        choices=sorted(SPLIT_POLICIES),
        help=(
            "continuous_prefix=group questions by source video and run from "
            "chunk 0 to the last answer slot; strict_window=legacy one-question "
            "25-45 second window."
        ),
    )
    args = ap.parse_args()

    summary = build(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_out:
        p = Path(args.summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
