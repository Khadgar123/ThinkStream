#!/usr/bin/env python
"""Render OVO-Bench into ThinkStream RL multi-Q trajectory rows.

The output JSONL uses the same shape as ``final/*_trajectories.jsonl`` consumed
by ``scripts.agent_data.build_verl_parquet``. It is intended for eval/monitoring
through the verl recurrent rollout path, so OVO no longer needs a separate
hand-written streaming runner for correctness checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkstream.eval.prompt_contract import (  # noqa: E402
    build_streaming_query_meta,
    label_mc_options,
)

RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
BT_TASKS = {"EPM", "ASI", "HLD"}
FT_TASKS = {"REC", "SSR", "CRR"}
ALL_TASKS = RT_TASKS | BT_TASKS | FT_TASKS
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class QuestionUnit:
    video_path: str
    task: str
    sample_id: str
    unit_id: str
    question: Dict[str, Any]
    interval_start: int
    interval_end: int
    score_until: int
    support_chunks: List[int]


def _time_to_chunk(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _interval_from_times(start: Any, end: Any = None) -> Tuple[int, int]:
    a = _time_to_chunk(start)
    b = _time_to_chunk(start if end is None else end)
    if b < a:
        a, b = b, a
    return a, b


def _chunks_from_intervals(intervals: Iterable[Tuple[int, int]]) -> List[int]:
    chunks: List[int] = []
    for start, end in intervals:
        chunks.extend(range(int(start), int(end) + 1))
    return sorted(set(chunks))


def _support_intervals(sample: Dict[str, Any], task: Optional[str] = None) -> List[Tuple[int, int]]:
    task = task or str(sample.get("task") or "")
    if task == "REC":
        return [
            _interval_from_times(s, e)
            for s, e in zip(sample.get("start_times", []), sample.get("end_times", []))
        ]
    if task == "SSR":
        return [
            _interval_from_times(s, e)
            for s, e in zip(sample.get("start_time", []), sample.get("end_time", []))
        ]
    if task == "CRR" and sample.get("clue_time") is not None:
        c = _time_to_chunk(sample["clue_time"])
        return [(c, c)]
    if sample.get("realtime") is not None:
        c = _time_to_chunk(sample["realtime"])
        return [(c, c)]
    return []


def _category_for_task(task: str) -> str:
    if task in RT_TASKS:
        return "RT"
    if task in BT_TASKS:
        return "BT"
    if task in FT_TASKS:
        return "FT"
    return "unknown"


def _correct_letter(sample: Dict[str, Any]) -> str:
    try:
        idx = int(sample.get("gt"))
    except (TypeError, ValueError):
        return ""
    return LETTERS[idx] if 0 <= idx < len(LETTERS) else ""


def _correct_option_text(sample: Dict[str, Any]) -> str:
    options = list(sample.get("options") or [])
    try:
        idx = int(sample.get("gt"))
    except (TypeError, ValueError):
        return ""
    if 0 <= idx < len(options):
        return str(options[idx])
    return ""


def _rec_question(sample: Dict[str, Any]) -> str:
    activity = sample.get("activity", "perform the action")
    return (
        "You're watching a video where people may perform a certain action "
        "repetitively. The performer is referred to as 'they'.\n"
        f"How many times have they {activity} so far?\n"
        "Your response type should be INT, for example, 0/1/2/3."
    )


def _ssr_question(step_text: str) -> str:
    return (
        "You're watching a tutorial video which contains a sequence of steps. "
        "The following is one step from the procedure:\n\n"
        f"{step_text}\n\n"
        "Your task is to decide: Is the person in the video currently carrying "
        "out this step?\nReturn \"Yes\" if they are; return \"No\" if not."
    )


def _crr_question(sample: Dict[str, Any]) -> str:
    return (
        f"{sample.get('question', '')}\n"
        "Return \"Yes\" if the action described has happened in the visible "
        "video so far; otherwise return \"No\"."
    )


def _base_question_payload(
    *,
    sample: Dict[str, Any],
    task: str,
    question: str,
    answer_form: str,
    ask_chunks: List[int],
    answer_chunks: List[int],
    gold_answer: str,
    per_emit_answers: Optional[List[Dict[str, Any]]] = None,
    options: Optional[List[str]] = None,
    correct_option: str = "",
    correct_answer_text: str = "",
    probe_index: Optional[int] = None,
    probe_type: Optional[int] = None,
    realtime: Optional[Any] = None,
) -> Dict[str, Any]:
    q: Dict[str, Any] = {
        "card_id": str(sample.get("id", "")) if probe_index is None else f"{sample.get('id')}:{probe_index}",
        "family": task,
        "family_name": task,
        "category": _category_for_task(task),
        "skill": "ovobench",
        "question": question,
        "ask_chunk": int(ask_chunks[0]) if ask_chunks else -1,
        "ask_chunks": sorted(set(int(x) for x in ask_chunks)),
        "answer_chunks": sorted(set(int(x) for x in answer_chunks)),
        "per_emit_answers": list(per_emit_answers or []),
        "gold_answer": gold_answer,
        "answer_form": answer_form,
        "support_chunks": _chunks_from_intervals(_support_intervals(sample, task)),
        "ovo_task": task,
        "ovo_category": _category_for_task(task),
        "ovo_sample_id": sample.get("id"),
        "ovo_probe_index": probe_index,
        "ovo_probe_type": probe_type,
        "ovo_realtime": realtime,
        "ovo_ask_time": sample.get("ask_time"),
        "ovo_clue_time": sample.get("clue_time"),
        "ovo_support_intervals": _support_intervals(sample, task),
    }
    if options is not None:
        q["options"] = list(options)
    if correct_option:
        q["correct_option"] = correct_option
    if correct_answer_text:
        q["correct_answer_text"] = correct_answer_text
    meta = build_streaming_query_meta(q, answer_form=answer_form)
    for key in ("answer_instruction", "answer_style"):
        if key in meta:
            q[key] = meta[key]
    if answer_chunks:
        q["open_until"] = max(answer_chunks)
    return q


def _units_for_sample(sample: Dict[str, Any], *, scoring: str) -> List[QuestionUnit]:
    task = str(sample.get("task") or "")
    video_path = str(sample.get("video") or "")
    sample_id = str(sample.get("id", ""))
    if not video_path or task not in ALL_TASKS:
        return []

    if task in RT_TASKS or task in BT_TASKS:
        ask = _time_to_chunk(sample.get("realtime"))
        extra = 60 if scoring == "lenient" else 2
        correct = _correct_letter(sample)
        options = label_mc_options(sample.get("options") or [], style="paren")
        q = _base_question_payload(
            sample=sample,
            task=task,
            question=str(sample.get("question") or ""),
            answer_form="multiple_choice",
            ask_chunks=[ask],
            answer_chunks=[ask],
            gold_answer=correct,
            options=options,
            correct_option=correct,
            correct_answer_text=_correct_option_text(sample),
            realtime=sample.get("realtime"),
        )
        return [QuestionUnit(
            video_path=video_path,
            task=task,
            sample_id=sample_id,
            unit_id=sample_id,
            question=q,
            interval_start=min([ask] + q["support_chunks"]) if q["support_chunks"] else ask,
            interval_end=ask,
            score_until=ask + extra,
            support_chunks=q["support_chunks"],
        )]

    if task == "REC":
        probes = list(sample.get("test_info") or [])
        chunks = sorted(set(_time_to_chunk(p.get("realtime")) for p in probes))
        per_emit = [
            {"chunk": _time_to_chunk(p.get("realtime")), "value": str(int(p.get("count", 0)))}
            for p in probes
        ]
        q = _base_question_payload(
            sample=sample,
            task=task,
            question=_rec_question(sample),
            answer_form="number",
            ask_chunks=[0],
            answer_chunks=chunks,
            gold_answer=str(int(probes[-1].get("count", 0))) if probes else "",
            per_emit_answers=per_emit,
        )
        end = max(chunks or [0])
        return [QuestionUnit(
            video_path=video_path,
            task=task,
            sample_id=sample_id,
            unit_id=sample_id,
            question=q,
            interval_start=0,
            interval_end=end,
            score_until=end,
            support_chunks=q["support_chunks"],
        )]

    if task == "CRR":
        probes = list(sample.get("test_info") or [])
        ask = _time_to_chunk(sample.get("ask_time"))
        answer_chunks = [_time_to_chunk(p.get("realtime")) for p in probes]
        per_emit = [
            {
                "chunk": _time_to_chunk(p.get("realtime")),
                "value": "Yes" if int(p.get("type", 0) or 0) == 1 else "No",
            }
            for p in probes
        ]
        q = _base_question_payload(
            sample=sample,
            task=task,
            question=_crr_question(sample),
            answer_form="binary",
            ask_chunks=[ask],
            answer_chunks=answer_chunks,
            gold_answer="Yes",
            per_emit_answers=per_emit,
        )
        end = max(answer_chunks or [ask])
        return [QuestionUnit(
            video_path=video_path,
            task=task,
            sample_id=sample_id,
            unit_id=sample_id,
            question=q,
            interval_start=min([ask] + q["support_chunks"]) if q["support_chunks"] else ask,
            interval_end=end,
            score_until=end,
            support_chunks=q["support_chunks"],
        )]

    if task == "SSR":
        units: List[QuestionUnit] = []
        for probe_i, probe in enumerate(sample.get("test_info") or []):
            chunk = _time_to_chunk(probe.get("realtime"))
            gold = "Yes" if int(probe.get("type", 0) or 0) == 1 else "No"
            q = _base_question_payload(
                sample=sample,
                task=task,
                question=_ssr_question(str(probe.get("step") or "")),
                answer_form="binary",
                ask_chunks=[chunk],
                answer_chunks=[chunk],
                gold_answer=gold,
                per_emit_answers=[{"chunk": chunk, "value": gold}],
                probe_index=probe_i,
                probe_type=int(probe.get("type", 0) or 0),
                realtime=probe.get("realtime"),
            )
            units.append(QuestionUnit(
                video_path=video_path,
                task=task,
                sample_id=sample_id,
                unit_id=f"{sample_id}:{probe_i}",
                question=q,
                interval_start=min([chunk] + q["support_chunks"]) if q["support_chunks"] else chunk,
                interval_end=chunk,
                score_until=chunk,
                support_chunks=q["support_chunks"],
            ))
        return units

    return []


def _overlaps(a: QuestionUnit, b: QuestionUnit) -> bool:
    return not (a.interval_end < b.interval_start or b.interval_end < a.interval_start)


def _pack_units(
    units: List[QuestionUnit],
    *,
    max_questions_per_trajectory: int,
    max_span_chunks: int,
    pre_context_chunks: int,
    post_context_chunks: int,
) -> List[List[QuestionUnit]]:
    groups: List[List[QuestionUnit]] = []
    for unit in sorted(units, key=lambda u: (u.interval_start, u.interval_end, u.unit_id)):
        best_i: Optional[int] = None
        best_growth: Optional[int] = None
        for i, group in enumerate(groups):
            if len(group) >= max_questions_per_trajectory:
                continue
            if any(_overlaps(unit, other) for other in group):
                continue
            starts = [u.interval_start for u in group] + [unit.interval_start]
            ends = [max(u.interval_end, u.score_until) for u in group] + [
                max(unit.interval_end, unit.score_until)
            ]
            seg_start = max(0, min(starts) - pre_context_chunks)
            seg_end = max(ends) + post_context_chunks
            span = seg_end - seg_start + 1
            if max_span_chunks > 0 and span > max_span_chunks:
                continue
            growth = span - (
                max(max(u.interval_end, u.score_until) for u in group)
                - max(0, min(u.interval_start for u in group) - pre_context_chunks)
                + 1
            )
            if best_growth is None or growth < best_growth:
                best_i = i
                best_growth = growth
        if best_i is None:
            groups.append([unit])
        else:
            groups[best_i].append(unit)
    return groups


def _segment_bounds(
    group: List[QuestionUnit],
    *,
    max_span_chunks: int,
    pre_context_chunks: int,
    post_context_chunks: int,
) -> Tuple[int, int, bool]:
    min_start = min(u.interval_start for u in group)
    max_end = max(max(u.interval_end, u.score_until) for u in group)
    start = max(0, min_start - pre_context_chunks)
    end = max_end + post_context_chunks
    exceeded = max_span_chunks > 0 and (end - start + 1) > max_span_chunks
    if exceeded:
        shifted = max(0, end - max_span_chunks + 1)
        if shifted <= min_start:
            start = shifted
            exceeded = False
    return start, end, exceeded


def _trajectory_id(video_path: str, group_index: int) -> str:
    stem = Path(video_path).with_suffix("").as_posix().strip("/").replace("/", "__")
    return f"ovo__{stem}__seg{group_index:03d}"


def _trajectory_from_group(
    video_path: str,
    group: List[QuestionUnit],
    group_index: int,
    *,
    max_span_chunks: int,
    pre_context_chunks: int,
    post_context_chunks: int,
) -> Dict[str, Any]:
    segment_start, segment_end, span_exceeded = _segment_bounds(
        group,
        max_span_chunks=max_span_chunks,
        pre_context_chunks=pre_context_chunks,
        post_context_chunks=post_context_chunks,
    )
    questions = [u.question for u in sorted(group, key=lambda u: (u.question["ask_chunk"], u.unit_id))]
    gold_action: Dict[str, str] = {}
    for q in questions:
        for ck in q.get("answer_chunks") or []:
            try:
                ci = int(ck)
            except (TypeError, ValueError):
                continue
            gold_action[str(ci)] = "response"
    tid = _trajectory_id(video_path, group_index)
    return {
        "trajectory_id": tid,
        "video_id": tid,
        "source_video_path": video_path,
        "video_path": video_path,
        "segment_start_chunk": int(segment_start),
        "segment_end_chunk": int(segment_end),
        "questions": questions,
        "gold_action_per_chunk": gold_action,
        "offline_compress_chunks": [],
        "samples": [],
        "stats": {
            "n_chunks_covered": int(segment_end) + 1,
            "chunk_idx_max": int(segment_end),
            "n_questions": len(questions),
        },
        "ovo_split_meta": {
            "group_index": int(group_index),
            "source_video_path": video_path,
            "tasks": sorted(set(u.task for u in group)),
            "sample_ids": [u.sample_id for u in group],
            "unit_ids": [u.unit_id for u in group],
            "segment_start_chunk": int(segment_start),
            "segment_end_chunk": int(segment_end),
            "max_span_chunks": int(max_span_chunks),
            "span_exceeded_soft_limit": bool(span_exceeded),
        },
    }


def build_trajectories(
    samples: List[Dict[str, Any]],
    *,
    tasks: Optional[set[str]] = None,
    scoring: str = "strict",
    max_questions_per_trajectory: int = 16,
    max_span_chunks: int = 512,
    pre_context_chunks: int = 64,
    post_context_chunks: int = 2,
    pack_across_tasks: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_bucket: Dict[Tuple[str, str], List[QuestionUnit]] = defaultdict(list)
    task_counts: Counter = Counter()
    for sample in samples:
        task = str(sample.get("task") or "")
        if tasks and task not in tasks:
            continue
        units = _units_for_sample(sample, scoring=scoring)
        for unit in units:
            pack_key = "*" if pack_across_tasks else unit.task
            by_bucket[(unit.video_path, pack_key)].append(unit)
            task_counts[unit.task] += 1

    trajectories: List[Dict[str, Any]] = []
    span_exceeded = 0
    group_counter_by_video: Counter = Counter()
    for video_path, _pack_key in sorted(by_bucket):
        groups = _pack_units(
            by_bucket[(video_path, _pack_key)],
            max_questions_per_trajectory=max_questions_per_trajectory,
            max_span_chunks=max_span_chunks,
            pre_context_chunks=pre_context_chunks,
            post_context_chunks=post_context_chunks,
        )
        for group in groups:
            group_i = int(group_counter_by_video[video_path])
            group_counter_by_video[video_path] += 1
            traj = _trajectory_from_group(
                video_path,
                group,
                group_i,
                max_span_chunks=max_span_chunks,
                pre_context_chunks=pre_context_chunks,
                post_context_chunks=post_context_chunks,
            )
            if (traj.get("ovo_split_meta") or {}).get("span_exceeded_soft_limit"):
                span_exceeded += 1
            trajectories.append(traj)

    summary = {
        "input_samples": len(samples),
        "videos": len(set(video_path for video_path, _ in by_bucket)),
        "trajectories": len(trajectories),
        "questions": sum(len(t.get("questions") or []) for t in trajectories),
        "tasks": dict(task_counts),
        "max_questions_per_trajectory": max_questions_per_trajectory,
        "max_span_chunks": max_span_chunks,
        "pre_context_chunks": pre_context_chunks,
        "post_context_chunks": post_context_chunks,
        "pack_across_tasks": bool(pack_across_tasks),
        "span_exceeded_soft_limit": span_exceeded,
    }
    return trajectories, summary


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-json", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-parquet", default="")
    ap.add_argument("--tasks", default="", help="Comma-separated OVO task subset.")
    ap.add_argument("--scoring", default="strict", choices=["strict", "lenient"])
    ap.add_argument("--max-questions-per-trajectory", type=int, default=16)
    ap.add_argument("--max-span-chunks", type=int, default=512)
    ap.add_argument("--pre-context-chunks", type=int, default=64)
    ap.add_argument("--post-context-chunks", type=int, default=2)
    ap.add_argument(
        "--pack-across-tasks",
        action="store_true",
        help=(
            "Allow one trajectory to contain different OVO task families. "
            "Default keeps task families separate for cleaner per-task eval."
        ),
    )
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    with Path(args.benchmark_json).open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise SystemExit("benchmark JSON must be a list")
    task_filter = {
        t.strip() for t in str(args.tasks).split(",") if t.strip()
    } or None
    trajectories, summary = build_trajectories(
        samples,
        tasks=task_filter,
        scoring=args.scoring,
        max_questions_per_trajectory=max(1, int(args.max_questions_per_trajectory)),
        max_span_chunks=max(0, int(args.max_span_chunks)),
        pre_context_chunks=max(0, int(args.pre_context_chunks)),
        post_context_chunks=max(0, int(args.post_context_chunks)),
        pack_across_tasks=bool(args.pack_across_tasks),
    )
    out_jsonl = Path(args.out_jsonl)
    _write_jsonl(out_jsonl, trajectories)
    if args.out_parquet:
        summary["parquet_rows"] = _write_parquet(
            out_jsonl,
            Path(args.out_parquet),
            max_questions_per_traj=max(1, int(args.max_questions_per_trajectory)),
        )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_out:
        p = Path(args.summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
