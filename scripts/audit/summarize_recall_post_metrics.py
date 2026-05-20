#!/usr/bin/env python3
"""Recompute recall/post-recall monitors from rollout audit JSONL.

This is a sidecar tool: it does not touch the running trainer. When the audit
record omits support_chunks, pass the source parquet(s) so the script can merge
them back by video_id/card_id.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _int_chunks(value: Any) -> List[int]:
    value = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, dict):
        return []
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    out: List[int] = []
    for raw in values:
        try:
            out.append(int(float(raw)))
        except (TypeError, ValueError):
            continue
    return sorted(set(c for c in out if c >= 0))


def _support_chunks(q: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for key in ("support_chunks", "evidence_chunks", "grounding_chunks", "grounding_frames"):
        out.extend(_int_chunks(q.get(key)))
    key_chunks = q.get("key_chunks")
    if isinstance(key_chunks, dict):
        for key in ("support", "supports", "evidence", "grounding"):
            out.extend(_int_chunks(key_chunks.get(key)))
    return sorted(set(out))


def _text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _load_obj(value: Any) -> Dict[str, Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_question_lookup(parquets: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not parquets:
        return lookup
    import pandas as pd

    for parquet in parquets:
        df = pd.read_parquet(parquet, columns=["video_id", "extra_info"])
        for row in df.itertuples(index=False):
            video_id = str(getattr(row, "video_id", "") or "")
            extra = _load_obj(getattr(row, "extra_info", None))
            for raw_q in _safe_list(extra.get("questions")):
                if not isinstance(raw_q, dict):
                    continue
                q = _jsonable(raw_q)
                card_id = str(q.get("card_id") or q.get("id") or q.get("qid") or "")
                if video_id and card_id:
                    lookup[(video_id, card_id)] = q
    return lookup


def _merge_question_support(
    video_id: str,
    questions: List[Dict[str, Any]],
    lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in questions:
        merged = dict(q)
        card_id = str(q.get("card_id") or q.get("id") or q.get("qid") or "")
        source = lookup.get((video_id, card_id), {})
        if not _support_chunks(merged) and source:
            merged["support_chunks"] = _support_chunks(source)
        if "options" not in merged and "options" in source:
            merged["options"] = _jsonable(source.get("options"))
        out.append(merged)
    return out


def _answer_chunks(q: Dict[str, Any]) -> List[int]:
    chunks = _int_chunks(q.get("answer_chunks"))
    for item in _safe_list(q.get("per_emit_answers")):
        if isinstance(item, dict):
            chunks.extend(_int_chunks(item.get("chunk")))
    return sorted(set(chunks))


def _recall_label_chunks(q: Dict[str, Any], gold_action: Dict[str, str]) -> List[int]:
    recall_actions = {"recall", "recall_silent"}
    ask_chunks = _int_chunks(q.get("ask_chunks")) or _int_chunks(q.get("ask_chunk"))
    direct = [
        c for c in ask_chunks
        if str(gold_action.get(str(c), "")).strip() in recall_actions
    ]
    if direct:
        return sorted(set(direct))
    bounds = ask_chunks + _answer_chunks(q)
    if not bounds:
        return []
    lo, hi = min(bounds), max(bounds)
    out: List[int] = []
    for key, action in gold_action.items():
        if str(action).strip() not in recall_actions:
            continue
        try:
            chunk = int(key)
        except (TypeError, ValueError):
            continue
        if lo <= chunk <= hi:
            out.append(chunk)
    return sorted(set(out))


def _range_hits(raw_range: Any, support: set[int], chunk_sec: float = 1.0) -> set[int]:
    if not isinstance(raw_range, dict):
        return set()
    try:
        start = float(raw_range.get("start_time"))
        end = float(raw_range.get("end_time"))
    except (TypeError, ValueError):
        return set()
    if end < start:
        return set()
    hits: set[int] = set()
    for chunk in support:
        c0 = float(chunk) * chunk_sec
        c1 = c0 + chunk_sec
        if c0 <= end and c1 > start:
            hits.add(chunk)
    return hits


def _score_answer(text: str, q: Dict[str, Any], expected_chunk: Any) -> float:
    from thinkstream.trainer.outcome_match import score_outcome_by_form

    per_emit = _safe_list(q.get("per_emit_answers"))
    chunk_gold: Dict[int, str] = {}
    for item in per_emit:
        if not isinstance(item, dict):
            continue
        chunks = _int_chunks(item.get("chunk"))
        if chunks:
            chunk_gold[chunks[0]] = str(item.get("value", q.get("gold_answer", "") or ""))
    chunks = _int_chunks(expected_chunk)
    gold = chunk_gold.get(chunks[0], str(q.get("gold_answer", "") or "")) if chunks else str(q.get("gold_answer", "") or "")
    return float(score_outcome_by_form(
        text,
        options=_safe_list(q.get("options")),
        correct_option=q.get("correct_option", ""),
        gold_answer=gold,
        answer_form=q.get("answer_form", "") or "",
    ))


def _row_metrics(row: Dict[str, Any], lookup: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, float]:
    video_id = str(row.get("video_id") or row.get("index") or "")
    questions = _merge_question_support(
        video_id,
        [q for q in _safe_list(row.get("questions")) if isinstance(q, dict)],
        lookup,
    )
    gold = _load_obj(row.get("ground_truth")).get("gold_action_per_chunk") or {}
    gold_action = {str(k): str(v) for k, v in _load_obj(gold).items()} if not isinstance(gold, dict) else {str(k): str(v) for k, v in gold.items()}

    answer_events: List[Dict[str, Any]] = []
    for q_idx, raw_events in enumerate(_safe_list(row.get("per_q_answers"))[:len(questions)]):
        q = questions[q_idx]
        for ev in _safe_list(raw_events):
            if not isinstance(ev, dict):
                continue
            text = str(ev.get("text", "") or "").strip()
            if not text:
                continue
            chunks = _int_chunks(ev.get("chunk"))
            expected = ev.get("expected_chunk", chunks[0] if chunks else -1)
            outcome = _score_answer(text, q, expected)
            answer_events.append({
                "chunk": chunks[0] if chunks else -1,
                "text_key": _text_key(text),
                "outcome": outcome,
            })

    used_answer_indices: set[int] = set()
    post_turns = 0
    post_outcomes: List[float] = []
    recall_turns: List[Dict[str, Any]] = []
    for turn in _safe_list(row.get("turns")):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("kind") or "") == "recall":
            recall_turns.append(turn)
        if str(turn.get("turn_kind") or "") not in {"post_recall", "recall_response"}:
            continue
        post_turns += 1
        answer_text = str(turn.get("answer_text") or "").strip()
        if not answer_text:
            continue
        chunks = _int_chunks(turn.get("event_chunk")) or _int_chunks(turn.get("video_chunk"))
        chunk = chunks[0] if chunks else -1
        key = _text_key(answer_text)
        matched: Optional[int] = None
        for idx, ev in enumerate(answer_events):
            if idx in used_answer_indices:
                continue
            if int(ev.get("chunk", -1)) == chunk and str(ev.get("text_key", "")) == key:
                matched = idx
                break
        if matched is None:
            same_chunk = [
                idx for idx, ev in enumerate(answer_events)
                if idx not in used_answer_indices and int(ev.get("chunk", -1)) == chunk
            ]
            if len(same_chunk) == 1:
                matched = same_chunk[0]
        if matched is None:
            post_outcomes.append(0.0)
        else:
            used_answer_indices.add(matched)
            post_outcomes.append(float(answer_events[matched].get("outcome", 0.0)))

    support_targets: List[Dict[str, Any]] = []
    for q in questions:
        support = set(_support_chunks(q))
        labels = _recall_label_chunks(q, gold_action)
        if not support or not labels:
            continue
        answer_bounds = _answer_chunks(q)
        support_targets.append({
            "labels": set(labels),
            "start": min(labels),
            "end": max(answer_bounds + labels),
            "support": support,
        })

    support_seen = 0
    request_hit = 0
    returned_hit = 0
    request_cover: List[float] = []
    returned_cover: List[float] = []
    for turn in recall_turns:
        chunks = _int_chunks(turn.get("event_chunk")) or _int_chunks(turn.get("video_chunk"))
        recall_chunk = chunks[0] if chunks else -1
        exact: set[int] = set()
        window: set[int] = set()
        for target in support_targets:
            support = set(target["support"])
            if recall_chunk in set(target["labels"]):
                exact.update(support)
            elif int(target["start"]) <= recall_chunk <= int(target["end"]):
                window.update(support)
        support = exact or window
        if not support:
            continue
        support_seen += 1
        req_hits = _range_hits(turn.get("requested_time_range"), support)
        ret_hits = set(_int_chunks(turn.get("returned_chunks"))) & support
        request_hit += int(bool(req_hits))
        returned_hit += int(bool(ret_hits))
        request_cover.append(float(len(req_hits)) / float(len(support)))
        returned_cover.append(float(len(ret_hits)) / float(len(support)))

    return {
        "rows": 1.0,
        "recall_call_count": float(len(recall_turns)),
        "post_recall_turn_count": float(post_turns),
        "post_recall_current_answer_count": float(len(post_outcomes)),
        "post_recall_current_outcome_sum": float(sum(post_outcomes)),
        "post_recall_current_outcome_traj_sum": _mean(post_outcomes),
        "post_recall_logged_outcome": float((_load_obj(row.get("reward")).get("post_recall_outcome_mean") or 0.0)),
        "recall_support_seen": float(support_seen),
        "recall_support_request_hit": float(request_hit),
        "recall_support_returned_hit": float(returned_hit),
        "recall_support_request_cover_sum": float(sum(request_cover)),
        "recall_support_returned_cover_sum": float(sum(returned_cover)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--parquet", action="append", default=[], type=Path)
    parser.add_argument(
        "--train-log",
        type=Path,
        help="Optional train.log; when paired with --by-step, groups audit rows by rollout step.",
    )
    parser.add_argument("--by-step", action="store_true")
    args = parser.parse_args()

    lookup = _load_question_lookup(args.parquet)

    boundaries: List[Tuple[int, float]] = []
    if args.by_step:
        if args.train_log is None:
            raise SystemExit("--by-step requires --train-log")
        pat = re.compile(
            r"INFO:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+):"
            r"streaming update_weights loaded.*'global_steps': (\d+)"
        )
        for line in args.train_log.read_text(errors="ignore").splitlines():
            match = pat.search(line)
            if not match:
                continue
            dt = _datetime.datetime.strptime(
                f"{match.group(1)}.{match.group(2)}",
                "%Y-%m-%d %H:%M:%S.%f",
            )
            boundaries.append((int(match.group(3)), dt.timestamp()))
        boundaries.sort()

    def _step_for(ts: float) -> str:
        if not boundaries:
            return "all"
        for step, boundary_ts in boundaries:
            if step <= 0:
                continue
            if ts <= boundary_ts:
                return str(step)
        return f"{boundaries[-1][0] + 1}*"

    grouped: Dict[str, Dict[str, float]] = {}
    with args.audit.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _step_for(float(row.get("ts", 0.0) or 0.0)) if args.by_step else "all"
            totals = grouped.setdefault(key, {})
            metrics = _row_metrics(row, lookup)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)

    def _summarize(totals: Dict[str, float]) -> Dict[str, Any]:
        rows = totals.get("rows", 0.0)
        post_answers = totals.get("post_recall_current_answer_count", 0.0)
        post_turns = totals.get("post_recall_turn_count", 0.0)
        support_seen = totals.get("recall_support_seen", 0.0)
        return {
            "rows": int(rows),
            "recall_call_count": totals.get("recall_call_count", 0.0),
            "post_recall_turn_count": post_turns,
            "post_recall_current_answer_count": post_answers,
            "post_recall_current_answer_rate": post_answers / post_turns if post_turns else 0.0,
            "post_recall_current_outcome_mean": (
                totals.get("post_recall_current_outcome_traj_sum", 0.0) / rows
                if rows else 0.0
            ),
            "post_recall_current_outcome_answer_weighted": (
                totals.get("post_recall_current_outcome_sum", 0.0) / post_answers
                if post_answers else 0.0
            ),
            "post_recall_logged_outcome_mean": (
                totals.get("post_recall_logged_outcome", 0.0) / rows if rows else 0.0
            ),
            "recall_support_seen": support_seen,
            "recall_support_request_hit_rate": (
                totals.get("recall_support_request_hit", 0.0) / support_seen
                if support_seen else 0.0
            ),
            "recall_support_returned_hit_rate": (
                totals.get("recall_support_returned_hit", 0.0) / support_seen
                if support_seen else 0.0
            ),
            "recall_support_request_cover_mean": (
                totals.get("recall_support_request_cover_sum", 0.0) / support_seen
                if support_seen else 0.0
            ),
            "recall_support_returned_cover_mean": (
                totals.get("recall_support_returned_cover_sum", 0.0) / support_seen
                if support_seen else 0.0
            ),
        }

    if args.by_step:
        out = {key: _summarize(grouped[key]) for key in sorted(
            grouped,
            key=lambda x: (int(x.rstrip("*")), x.endswith("*")),
        )}
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    totals = grouped.get("all", {})
    rows = totals.get("rows", 0.0)
    post_answers = totals.get("post_recall_current_answer_count", 0.0)
    post_turns = totals.get("post_recall_turn_count", 0.0)
    support_seen = totals.get("recall_support_seen", 0.0)
    out = {
        "rows": int(rows),
        "recall_call_count": totals.get("recall_call_count", 0.0),
        "post_recall_turn_count": post_turns,
        "post_recall_current_answer_count": post_answers,
        "post_recall_current_answer_rate": post_answers / post_turns if post_turns else 0.0,
        "post_recall_current_outcome_mean": (
            totals.get("post_recall_current_outcome_traj_sum", 0.0) / rows
            if rows else 0.0
        ),
        "post_recall_current_outcome_answer_weighted": (
            totals.get("post_recall_current_outcome_sum", 0.0) / post_answers
            if post_answers else 0.0
        ),
        "post_recall_logged_outcome_mean": (
            totals.get("post_recall_logged_outcome", 0.0) / rows if rows else 0.0
        ),
        "recall_support_seen": support_seen,
        "recall_support_request_hit_rate": (
            totals.get("recall_support_request_hit", 0.0) / support_seen
            if support_seen else 0.0
        ),
        "recall_support_returned_hit_rate": (
            totals.get("recall_support_returned_hit", 0.0) / support_seen
            if support_seen else 0.0
        ),
        "recall_support_request_cover_mean": (
            totals.get("recall_support_request_cover_sum", 0.0) / support_seen
            if support_seen else 0.0
        ),
        "recall_support_returned_cover_mean": (
            totals.get("recall_support_returned_cover_sum", 0.0) / support_seen
            if support_seen else 0.0
        ),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
