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
SPLIT_POLICIES = {
    "continuous_prefix",
    "strict25_45",
    "strict25_45_stateful",
}
STRICT_BENCHMARK_POLICIES = {"strict25_45", "strict25_45_stateful"}
CURRENT_SPLIT_POLICY = "continuous_prefix"


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
        "Return one integer only, such as 0, 1, 2, or 3."
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
        count_by_chunk: Dict[int, int] = {}
        for probe in probes:
            chunk = _time_to_chunk(probe.get("realtime"))
            count = int(probe.get("count", 0))
            count_by_chunk[chunk] = max(count_by_chunk.get(chunk, count), count)
        chunks = sorted(count_by_chunk)
        per_emit = [
            {"chunk": chunk, "value": str(count_by_chunk[chunk])}
            for chunk in chunks
        ]
        ask_chunk = chunks[0] if chunks else 0
        q = _base_question_payload(
            sample=sample,
            task=task,
            question=_rec_question(sample),
            answer_form="number",
            ask_chunks=[ask_chunk],
            answer_chunks=chunks,
            gold_answer=str(count_by_chunk[chunks[-1]]) if chunks else "",
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


def _segment_bounds_strict25_45(
    group: List[QuestionUnit],
    *,
    min_span_chunks: int,
    max_span_chunks: int,
    post_context_chunks: int,
) -> Tuple[int, int, bool]:
    min_start = max(0, min(u.interval_start for u in group))
    max_end = max(max(u.interval_end, u.score_until) for u in group)
    end = max_end + post_context_chunks
    min_span_chunks = max(1, int(min_span_chunks))
    max_span_chunks = max(min_span_chunks, int(max_span_chunks))

    required_span = end - min_start + 1
    if required_span > max_span_chunks:
        return min_start, end, True

    target_span = max_span_chunks
    start = max(0, end - target_span + 1)
    if start > min_start:
        start = min_start
    if end - start + 1 < min_span_chunks:
        end = start + min_span_chunks - 1
    starts_on_question = any(
        int((u.question.get("ask_chunks") or [u.question.get("ask_chunk", -1)])[0]) == start
        for u in group
    )
    if starts_on_question and start > 0:
        if end - start + 1 < max_span_chunks:
            start -= 1
        elif end > max_end:
            start -= 1
            end -= 1
    return start, end, False


def _segment_bounds_continuous_prefix(
    group: List[QuestionUnit],
    *,
    post_context_chunks: int,
) -> Tuple[int, int, bool]:
    """Run from source-video chunk 0 through the group's last answer slot."""
    max_end = max(max(u.interval_end, u.score_until) for u in group)
    return 0, max(0, int(max_end) + max(0, int(post_context_chunks))), False


def _unit_active_window(unit: QuestionUnit) -> Tuple[int, int]:
    q = unit.question
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
    start = min(starts) if starts else int(unit.interval_start)
    end = max(ends) if ends else int(max(unit.interval_end, unit.score_until))
    return start, max(start, end)


def _windows_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _question_active_window(q: Dict[str, Any]) -> Tuple[int, int]:
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


def _question_support_intervals(q: Dict[str, Any]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    for raw in q.get("ovo_support_intervals") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            start = int(raw[0])
            end = int(raw[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        intervals.append((start, end))
    chunks: List[int] = []
    for raw in q.get("support_chunks") or []:
        try:
            chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    if chunks:
        intervals.append((min(chunks), max(chunks)))
    return intervals


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
        active_start, active_end = _question_active_window(q)
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
    """Plan compact-memory boundaries inside a complete continuous trajectory.

    Boundaries are the first visual chunk after a compressed raw range. The
    runtime compresses accumulated recent_thinks before processing that chunk,
    then re-prefills and continues at the same chunk.
    """
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
        candidates = range(lo, hi + 1)
        # Main key: avoid answer/evidence windows. Tie-break later boundaries
        # to keep updates closer to 45 chunks and reduce compression frequency.
        boundary = min(
            candidates,
            key=lambda b: (_compress_boundary_penalty(b, questions), -int(b)),
        )
        boundaries.append(int(boundary))
        cursor = int(boundary)
    return boundaries


def _continuous_prefix_groups(
    units: List[QuestionUnit],
) -> List[List[QuestionUnit]]:
    """Pack one source video into non-overlapping continuous-prefix tracks.

    The runtime prompt renders a single active query. If two benchmark
    questions are open at the same chunk, putting them in one trajectory would
    hide one of them. Greedy sibling tracks preserve continuous KV within each
    track while avoiding active-query collisions.
    """
    groups: List[List[QuestionUnit]] = []
    windows_by_group: List[List[Tuple[int, int]]] = []
    ordered = sorted(
        units,
        key=lambda u: (_unit_active_window(u)[0], _unit_active_window(u)[1], u.task, u.unit_id),
    )
    for unit in ordered:
        window = _unit_active_window(unit)
        placed = False
        for gi, existing in enumerate(windows_by_group):
            if any(_windows_overlap(window, other) for other in existing):
                continue
            groups[gi].append(unit)
            existing.append(window)
            placed = True
            break
        if not placed:
            groups.append([unit])
            windows_by_group.append([window])
    return groups


def _trajectory_id(video_path: str, group_index: int) -> str:
    stem = Path(video_path).with_suffix("").as_posix().strip("/").replace("/", "__")
    return f"ovo__{stem}__seg{group_index:03d}"


def _trajectory_from_group(
    video_path: str,
    group: List[QuestionUnit],
    group_index: int,
    *,
    split_policy: str,
    max_span_chunks: int,
    pre_context_chunks: int,
    post_context_chunks: int,
    short_min_span_chunks: int,
    short_max_span_chunks: int,
    offline_compress_min_chunks: int,
    offline_compress_max_chunks: int,
) -> Dict[str, Any]:
    if split_policy in {"strict25_45", "strict25_45_stateful"}:
        segment_start, segment_end, span_exceeded = _segment_bounds_strict25_45(
            group,
            min_span_chunks=short_min_span_chunks,
            max_span_chunks=short_max_span_chunks,
            post_context_chunks=post_context_chunks,
        )
    elif split_policy == "continuous_prefix":
        segment_start, segment_end, span_exceeded = _segment_bounds_continuous_prefix(
            group,
            post_context_chunks=post_context_chunks,
        )
    else:
        segment_start, segment_end, span_exceeded = _segment_bounds(
            group,
            max_span_chunks=max_span_chunks,
            pre_context_chunks=pre_context_chunks,
            post_context_chunks=post_context_chunks,
        )
    questions = [u.question for u in sorted(group, key=lambda u: (u.question["ask_chunk"], u.unit_id))]
    offline_compress_chunks = (
        _plan_offline_compress_chunks(
            questions,
            segment_start=int(segment_start),
            segment_end=int(segment_end),
            min_chunks=offline_compress_min_chunks,
            max_chunks=offline_compress_max_chunks,
        )
        if split_policy == "continuous_prefix"
        else []
    )
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
        "offline_compress_chunks": offline_compress_chunks,
        "samples": [],
        "stats": {
            "n_chunks_covered": int(segment_end) + 1,
            "chunk_idx_max": int(segment_end),
            "n_questions": len(questions),
            "n_offline_compress_chunks": len(offline_compress_chunks),
        },
        "ovo_split_meta": {
            "group_index": int(group_index),
            "source_video_path": video_path,
            "tasks": sorted(set(u.task for u in group)),
            "sample_ids": [u.sample_id for u in group],
            "unit_ids": [u.unit_id for u in group],
            "segment_start_chunk": int(segment_start),
            "segment_end_chunk": int(segment_end),
            "split_policy": split_policy,
            "benchmark_track": (
                "continuous_prefix"
                if split_policy == "continuous_prefix"
                else (
                    "strict_window"
                    if split_policy == "strict25_45"
                    else "stateful_multi_emit_reprefill"
                )
            ),
            "max_span_chunks": int(max_span_chunks),
            "short_min_span_chunks": int(short_min_span_chunks),
            "short_max_span_chunks": int(short_max_span_chunks),
            "span_exceeded_soft_limit": bool(span_exceeded),
            "offline_compress_policy": (
                "planned_avoid_answer_support_windows"
                if split_policy == "continuous_prefix"
                else ""
            ),
            "offline_compress_min_chunks": int(offline_compress_min_chunks),
            "offline_compress_max_chunks": int(offline_compress_max_chunks),
        },
    }


def _balanced_ranges(start: int, end: int, *, min_span: int, max_span: int) -> List[Tuple[int, int]]:
    length = int(end) - int(start) + 1
    if length <= 0:
        return []
    min_span = max(1, int(min_span))
    max_span = max(min_span, int(max_span))
    if length <= max_span:
        return [(int(start), int(end))]

    # Long benchmark rows are cut into non-overlapping recurrent parts. If the
    # final tail would be shorter than min_span, merge it into the previous part
    # instead of creating a too-short re-prefill segment; this can reasonably
    # produce a part slightly above max_span (e.g. 49s for a 25-45 target).
    n_parts = (length + max_span - 1) // max_span
    while n_parts > 1 and (length / n_parts) < min_span:
        n_parts -= 1
    base = length // n_parts
    rem = length % n_parts
    ranges: List[Tuple[int, int]] = []
    cur = int(start)
    for i in range(n_parts):
        part_len = base + (1 if i < rem else 0)
        part_end = cur + part_len - 1
        ranges.append((cur, part_end))
        cur = part_end + 1
    return ranges


def _filter_question_for_stateful_segment(
    q: Dict[str, Any],
    *,
    segment_start: int,
    segment_end: int,
    part_index: int,
    total_parts: int,
) -> Optional[Dict[str, Any]]:
    answer_chunks: List[int] = []
    for raw in q.get("answer_chunks") or []:
        try:
            ck = int(raw)
        except (TypeError, ValueError):
            continue
        if segment_start <= ck <= segment_end:
            answer_chunks.append(ck)
    answer_chunks = sorted(set(answer_chunks))
    if not answer_chunks:
        return None

    per_emit_by_chunk: Dict[int, Dict[str, Any]] = {}
    answer_form = str(q.get("answer_form") or "")
    for item in q.get("per_emit_answers") or []:
        if not isinstance(item, dict):
            continue
        try:
            ck = int(item.get("chunk"))
        except (TypeError, ValueError):
            continue
        if ck in answer_chunks:
            out = dict(item)
            out["chunk"] = ck
            if answer_form == "number" and ck in per_emit_by_chunk:
                try:
                    prev = int(per_emit_by_chunk[ck].get("value", 0))
                    cur = int(out.get("value", 0))
                except (TypeError, ValueError):
                    cur = 0
                    prev = 0
                if cur >= prev:
                    per_emit_by_chunk[ck] = out
            else:
                per_emit_by_chunk[ck] = out
    per_emit = [per_emit_by_chunk[ck] for ck in sorted(per_emit_by_chunk)]

    q_out = dict(q)
    # Re-open a long multi-emit question at the beginning of each scored part.
    # Only cumulative count tasks receive earlier emitted answers as
    # response_history; independent status/probe tasks must be judged from the
    # current visible state.
    ask_chunk = int(segment_start)
    q_out["ask_chunk"] = ask_chunk
    q_out["ask_chunks"] = [ask_chunk]
    q_out["answer_chunks"] = answer_chunks
    q_out["per_emit_answers"] = per_emit
    family = str(q.get("family") or q.get("ovo_task") or q.get("task") or "").upper()
    answer_form = str(q.get("answer_form") or "").strip().lower()
    uses_response_history = family == "REC" or answer_form == "number"
    if uses_response_history:
        initial_history: List[Dict[str, Any]] = []
        for item in q.get("per_emit_answers") or []:
            if not isinstance(item, dict):
                continue
            try:
                ck = int(item.get("chunk"))
            except (TypeError, ValueError):
                continue
            if ck >= int(segment_start):
                continue
            initial_history.append({
                "chunk": ck,
                "time": ck,
                "text": str(item.get("value", "")),
                "expected_chunk": ck,
                "counts_for_completion": True,
                "stateful_seed": True,
            })
        if initial_history:
            q_out["initial_response_history"] = initial_history
            q_out["stateful_seed_response_history"] = initial_history
            q_out["response_history_policy"] = "include"
    support_chunks: List[int] = []
    for raw in q.get("support_chunks") or []:
        try:
            ck = int(raw)
        except (TypeError, ValueError):
            continue
        if segment_start <= ck <= segment_end:
            support_chunks.append(ck)
    q_out["support_chunks"] = sorted(set(support_chunks))
    q_out["open_until"] = max(answer_chunks)
    q_out["stateful_split_parent_card_id"] = q.get("card_id", "")
    q_out["stateful_split_part_index"] = int(part_index)
    q_out["stateful_split_total_parts"] = int(total_parts)
    q_out["stateful_split_original_ask_chunks"] = list(q.get("ask_chunks") or [])
    q_out["stateful_split_original_answer_chunks"] = list(q.get("answer_chunks") or [])
    q_out["stateful_split_original_per_emit_answers"] = list(q.get("per_emit_answers") or [])
    return q_out


def _split_stateful_trajectory(
    traj: Dict[str, Any],
    *,
    min_span_chunks: int,
    max_span_chunks: int,
) -> List[Dict[str, Any]]:
    start = int(traj.get("segment_start_chunk", 0))
    end = int(traj.get("segment_end_chunk", start))
    if end - start + 1 <= max_span_chunks:
        meta = dict(traj.get("ovo_split_meta") or {})
        meta["stateful_split"] = False
        traj["ovo_split_meta"] = meta
        return [traj]
    ranges = _balanced_ranges(
        start,
        end,
        min_span=min_span_chunks,
        max_span=max_span_chunks,
    )
    if len(ranges) <= 1:
        meta = dict(traj.get("ovo_split_meta") or {})
        meta["stateful_split"] = False
        traj["ovo_split_meta"] = meta
        return [traj]

    base_id = str(traj.get("trajectory_id") or traj.get("video_id") or "trajectory")
    parent_questions = list(traj.get("questions") or [])
    source_answer_chunks: List[int] = []
    for q in parent_questions:
        for raw in q.get("answer_chunks") or []:
            try:
                source_answer_chunks.append(int(raw))
            except (TypeError, ValueError):
                continue

    parts: List[Dict[str, Any]] = []
    assigned_answer_chunks_by_card: Dict[str, set[int]] = defaultdict(set)
    for part_i, (seg_start, seg_end) in enumerate(ranges):
        if seg_start > start:
            starts_on_answer = any(
                int(raw) == int(seg_start)
                for q in parent_questions
                for raw in (q.get("answer_chunks") or [])
                if str(raw).lstrip("-").isdigit()
            )
            if starts_on_answer and (seg_end - seg_start + 1) < max_span_chunks:
                seg_start -= 1
        part_questions: List[Dict[str, Any]] = []
        for q in parent_questions:
            q_part = _filter_question_for_stateful_segment(
                q,
                segment_start=seg_start,
                segment_end=seg_end,
                part_index=part_i,
                total_parts=len(ranges),
            )
            if q_part is not None:
                card_key = str(q.get("card_id") or q.get("question") or id(q))
                assigned = assigned_answer_chunks_by_card[card_key]
                fresh_answers = [
                    int(ck) for ck in (q_part.get("answer_chunks") or [])
                    if int(ck) not in assigned
                ]
                if not fresh_answers:
                    continue
                fresh_set = set(fresh_answers)
                q_part["answer_chunks"] = fresh_answers
                q_part["open_until"] = max(fresh_answers)
                q_part["per_emit_answers"] = [
                    item for item in (q_part.get("per_emit_answers") or [])
                    if int(item.get("chunk", -1)) in fresh_set
                ]
                assigned.update(fresh_answers)
                part_questions.append(q_part)

        gold_action = {}
        for q in part_questions:
            for ck in q.get("answer_chunks") or []:
                gold_action[str(int(ck))] = "response"

        part = dict(traj)
        part_id = f"{base_id}__part{part_i:03d}"
        part["trajectory_id"] = part_id
        part["video_id"] = part_id
        part["segment_start_chunk"] = int(seg_start)
        part["segment_end_chunk"] = int(seg_end)
        part["questions"] = part_questions
        part["gold_action_per_chunk"] = gold_action
        part["stats"] = {
            **dict(traj.get("stats") or {}),
            "n_chunks_covered": int(seg_end) + 1,
            "chunk_idx_max": int(seg_end),
            "n_questions": len(part_questions),
        }
        part["ovo_split_meta"] = {
            **dict(traj.get("ovo_split_meta") or {}),
            "split_policy": "strict25_45_stateful",
            "stateful_split": True,
            "stateful_split_parent_id": base_id,
            "stateful_split_part_index": int(part_i),
            "stateful_split_total_parts": int(len(ranges)),
            "stateful_split_role": "scored" if part_questions else "context_only",
            "stateful_split_source_start_chunk": int(start),
            "stateful_split_source_end_chunk": int(end),
            "stateful_split_source_answer_chunks": sorted(set(source_answer_chunks)),
            "segment_start_chunk": int(seg_start),
            "segment_end_chunk": int(seg_end),
            "span_exceeded_soft_limit": False,
        }
        parts.append(part)
    return parts


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
    split_policy: str = CURRENT_SPLIT_POLICY,
    short_min_span_chunks: int = 20,
    short_max_span_chunks: int = 40,
    offline_compress_min_chunks: int = 25,
    offline_compress_max_chunks: int = 45,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(
            f"split_policy must be one of {sorted(SPLIT_POLICIES)}, got {split_policy!r}"
        )
    effective_pack_across_tasks = bool(pack_across_tasks or split_policy == "continuous_prefix")
    by_bucket: Dict[Tuple[str, str], List[QuestionUnit]] = defaultdict(list)
    task_counts: Counter = Counter()
    for sample in samples:
        task = str(sample.get("task") or "")
        if tasks and task not in tasks:
            continue
        units = _units_for_sample(sample, scoring=scoring)
        for unit in units:
            pack_key = "*" if effective_pack_across_tasks else unit.task
            by_bucket[(unit.video_path, pack_key)].append(unit)
            task_counts[unit.task] += 1

    trajectories: List[Dict[str, Any]] = []
    span_exceeded = 0
    strict_excluded_span_exceeded = 0
    group_counter_by_video: Counter = Counter()
    for video_path, _pack_key in sorted(by_bucket):
        units = by_bucket[(video_path, _pack_key)]
        if split_policy in STRICT_BENCHMARK_POLICIES:
            short_min_span_chunks = 25
            short_max_span_chunks = 45

        if split_policy == "continuous_prefix":
            groups = _continuous_prefix_groups(units)
        else:
            groups = [[unit] for unit in sorted(
                units,
                key=lambda u: (u.interval_start, u.interval_end, u.unit_id),
            )]
        for group in groups:
            group_i = int(group_counter_by_video[video_path])
            group_counter_by_video[video_path] += 1
            traj = _trajectory_from_group(
                video_path,
                group,
                group_i,
                split_policy=split_policy,
                max_span_chunks=max_span_chunks,
                pre_context_chunks=pre_context_chunks,
                post_context_chunks=post_context_chunks,
                short_min_span_chunks=short_min_span_chunks,
                short_max_span_chunks=short_max_span_chunks,
                offline_compress_min_chunks=offline_compress_min_chunks,
                offline_compress_max_chunks=offline_compress_max_chunks,
            )
            if (traj.get("ovo_split_meta") or {}).get("span_exceeded_soft_limit"):
                span_exceeded += 1
                if split_policy == "strict25_45":
                    strict_excluded_span_exceeded += 1
                    continue
            if split_policy == "strict25_45_stateful":
                split_parts = _split_stateful_trajectory(
                    traj,
                    min_span_chunks=short_min_span_chunks,
                    max_span_chunks=short_max_span_chunks,
                )
                trajectories.extend(split_parts)
            else:
                trajectories.append(traj)

    stateful_context_rows = sum(
        1 for t in trajectories if not (t.get("questions") or [])
    )
    stateful_split_parents = len({
        (t.get("ovo_split_meta") or {}).get("stateful_split_parent_id")
        for t in trajectories
        if (t.get("ovo_split_meta") or {}).get("stateful_split")
    })
    stateful_max_part_span = max(
        (
            int(t.get("segment_end_chunk", 0)) - int(t.get("segment_start_chunk", 0)) + 1
            for t in trajectories
        ),
        default=0,
    )
    stateful_over_45_rows = sum(
        1
        for t in trajectories
        if int(t.get("segment_end_chunk", 0)) - int(t.get("segment_start_chunk", 0)) + 1
        > int(short_max_span_chunks)
    )
    summary = {
        "input_samples": len(samples),
        "videos": len(set(video_path for video_path, _ in by_bucket)),
        "trajectories": len(trajectories),
        "questions": sum(len(t.get("questions") or []) for t in trajectories),
        "source_question_units": int(sum(task_counts.values())),
        "tasks": dict(task_counts),
        "max_questions_per_trajectory": (
            None if split_policy == "continuous_prefix" else max_questions_per_trajectory
        ),
        "requested_max_questions_per_trajectory": max_questions_per_trajectory,
        "max_span_chunks": max_span_chunks,
        "pre_context_chunks": pre_context_chunks,
        "post_context_chunks": post_context_chunks,
        "pack_across_tasks": bool(effective_pack_across_tasks),
        "requested_pack_across_tasks": bool(pack_across_tasks),
        "split_policy": split_policy,
        "benchmark_tracks": [
            "strict25_45",
            "strict25_45_stateful",
        ] if split_policy in STRICT_BENCHMARK_POLICIES else [],
        "short_min_span_chunks": int(short_min_span_chunks),
        "short_max_span_chunks": int(short_max_span_chunks),
        "span_exceeded_soft_limit": span_exceeded,
        "strict_excluded_span_exceeded": strict_excluded_span_exceeded,
        "stateful_context_rows": stateful_context_rows,
        "stateful_scored_rows": len(trajectories) - stateful_context_rows,
        "stateful_split_parent_rows": stateful_split_parents,
        "stateful_max_part_span": stateful_max_part_span,
        "stateful_over_45_rows": stateful_over_45_rows,
        "continuous_prefix": split_policy == "continuous_prefix",
        "offline_compress_policy": (
            "planned_avoid_answer_support_windows"
            if split_policy == "continuous_prefix"
            else ""
        ),
        "offline_compress_min_chunks": int(offline_compress_min_chunks),
        "offline_compress_max_chunks": int(offline_compress_max_chunks),
        "offline_compress_events": sum(len(t.get("offline_compress_chunks") or []) for t in trajectories),
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
        "--split-policy",
        default=CURRENT_SPLIT_POLICY,
        choices=sorted(SPLIT_POLICIES),
        help=(
            "continuous_prefix=source-video prefix tracks from chunk 0 with "
            "non-overlapping questions placed on the timeline; "
            "strict25_45=one question per row in a strict 25-45s window; "
            "strict25_45_stateful=split long multi-emit questions into "
            "25-45s scored/context parts for compress and re-prefill evaluation."
        ),
    )
    ap.add_argument("--short-min-span-chunks", type=int, default=20)
    ap.add_argument("--short-max-span-chunks", type=int, default=40)
    ap.add_argument("--offline-compress-min-chunks", type=int, default=25)
    ap.add_argument("--offline-compress-max-chunks", type=int, default=45)
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
        split_policy=args.split_policy,
        short_min_span_chunks=max(1, int(args.short_min_span_chunks)),
        short_max_span_chunks=max(1, int(args.short_max_span_chunks)),
        offline_compress_min_chunks=max(1, int(args.offline_compress_min_chunks)),
        offline_compress_max_chunks=max(1, int(args.offline_compress_max_chunks)),
    )
    out_jsonl = Path(args.out_jsonl)
    _write_jsonl(out_jsonl, trajectories)
    if args.out_parquet:
        parquet_max_questions = max(
            (len(t.get("questions") or []) for t in trajectories),
            default=1,
        ) if args.split_policy == "continuous_prefix" else max(1, int(args.max_questions_per_trajectory))
        summary["parquet_max_questions_per_traj"] = int(parquet_max_questions)
        summary["parquet_rows"] = _write_parquet(
            out_jsonl,
            Path(args.out_parquet),
            max_questions_per_traj=int(parquet_max_questions),
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
