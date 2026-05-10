#!/usr/bin/env python3
"""Lightweight monitor for ThinkStream recurrent RL runs.

Reads a verl train.log plus reward-time rollout audit JSONL and prints the
signals that are easiest to miss in wandb: recurrent action counts, recall
behavior vs question/support labels, answer accuracy/timing, compression/action
parse health, and tool time-range validity.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


STEP_RE = re.compile(r"\bstep:(?P<step>\d+)\b(?P<body>.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+):(?:np\.\w+\()?([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return "n/a"
    return f"{100.0 * num / den:.1f}%"


def _norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"^[a-d]\)\s*", "", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_mc_letter(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    m = re.match(r"^\s*([A-Za-z])(?:\b|\)|\.|:)", text)
    return m.group(1).upper() if m else ""


def _is_correct_answer(answer: str, question: dict[str, Any]) -> bool:
    answer = answer or ""
    answer_norm = _norm_text(answer)
    if not answer_norm:
        return False

    correct_option = str(question.get("correct_option") or "").strip().upper()
    answer_form = str(question.get("answer_form") or "").strip().lower()
    if correct_option:
        letter = _extract_mc_letter(answer)
        if letter and letter == correct_option:
            return True
        # Some rows request text_only for MC; accepted answers covers both.

    accepted = _safe_list(question.get("accepted_answers"))
    gold = question.get("gold_answer") or question.get("correct_answer_text")
    if gold:
        accepted.append(gold)
    if correct_option:
        accepted.append(correct_option)

    accepted_norm = [_norm_text(x) for x in accepted if _norm_text(x)]
    if answer_norm in accepted_norm:
        return True
    # Descriptive answers are often short phrases; allow containment in either
    # direction after normalization, but do not do this for one-letter MC.
    if answer_form != "multiple_choice":
        for target in accepted_norm:
            if len(target) >= 3 and (target in answer_norm or answer_norm in target):
                return True
    return False


def _question_key(question: dict[str, Any], idx: int) -> str:
    return str(question.get("card_id") or question.get("question") or idx)


def _question_support(question: dict[str, Any]) -> set[int]:
    support = set()
    for x in _safe_list(question.get("support_chunks")):
        iv = _safe_int(x)
        if iv >= 0:
            support.add(iv)
    return support


def _question_ask(question: dict[str, Any]) -> int:
    asks = [_safe_int(x) for x in _safe_list(question.get("ask_chunks"))]
    asks = [x for x in asks if x >= 0]
    if asks:
        return min(asks)
    return _safe_int(question.get("ask_chunk"))


def _question_answer_deadline(question: dict[str, Any]) -> int:
    chunks = [_safe_int(x) for x in _safe_list(question.get("answer_chunks"))]
    chunks = [x for x in chunks if x >= 0]
    return max(chunks) if chunks else _question_ask(question)


def _range_to_chunks(time_range: Any) -> set[int]:
    start = end = None
    if isinstance(time_range, str):
        m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*", time_range)
        if m:
            start = _safe_float(m.group(1), math.nan)
            end = _safe_float(m.group(2), math.nan)
    elif isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        start = _safe_float(time_range[0], math.nan)
        end = _safe_float(time_range[1], math.nan)
    if start is None or end is None or math.isnan(start) or math.isnan(end) or end <= start:
        return set()
    lo = max(0, int(math.floor(start)))
    hi = max(lo, int(math.ceil(end)) - 1)
    return set(range(lo, hi + 1))


def _load_jsonl_tail(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if limit is None or limit <= 0:
        lines: Iterable[str] = path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        dq: deque[str] = deque(maxlen=limit)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                dq.append(line)
        lines = dq
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _parse_train_steps(path: Path, tail_lines: int = 2000) -> list[dict[str, float]]:
    if not path.exists():
        return []
    dq: deque[str] = deque(maxlen=tail_lines)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            dq.append(line)
    steps: list[dict[str, float]] = []
    for line in dq:
        m = STEP_RE.search(line)
        if not m:
            continue
        rec: dict[str, float] = {"step": float(m.group("step"))}
        for key, value in METRIC_RE.findall(m.group("body")):
            rec[key] = _safe_float(value)
        if "recurrent/n_trajectories" in rec:
            steps.append(rec)
    return steps


def _audit_questions(row: dict[str, Any]) -> list[dict[str, Any]]:
    gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
    questions = _safe_list(gt.get("questions"))
    if questions:
        return [q for q in questions if isinstance(q, dict)]
    return [q for q in _safe_list(row.get("questions")) if isinstance(q, dict)]


def _answer_events(row: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    raw = _safe_list(row.get("per_q_answers"))
    for idx, question in enumerate(questions):
        key = _question_key(question, idx)
        events = raw[idx] if idx < len(raw) else []
        out[key] = [ev for ev in _safe_list(events) if isinstance(ev, dict)]
    return out


def _turn_tool_args(turn: dict[str, Any]) -> dict[str, Any]:
    args = turn.get("tool_args")
    return args if isinstance(args, dict) else {}


def summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter()
    by_form = defaultdict(Counter)
    recall_support_jaccards: list[float] = []
    recall_support_coverages: list[float] = []
    recall_range_lens: list[int] = []
    recall_active_support_hits = 0
    recall_any_support_hits = 0
    recall_runtime_ok = 0
    recall_total = 0
    q_with_prior_recall = Counter()
    q_without_prior_recall = Counter()
    q_with_support_recall = Counter()
    reward_scores: list[float] = []
    chunk_unit_counts: list[int] = []
    turn_row_counts: list[int] = []
    unique_event_chunk_counts: list[int] = []

    for row in rows:
        c["records"] += 1
        reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
        reward_scores.append(_safe_float(reward.get("score")))
        counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}

        questions = _audit_questions(row)
        answer_by_q = _answer_events(row, questions)
        turns = [t for t in _safe_list(row.get("turns")) if isinstance(t, dict)]
        chunk_unit_counts.append(
            _safe_int(counts.get("action_units_used"), _safe_int(counts.get("chunks_used"), 0))
        )
        turn_row_counts.append(
            _safe_int(
                counts.get("action_rows_used"),
                _safe_int(counts.get("turns_used"), _safe_int(counts.get("n_actions_in_traj"), len(turns))),
            )
        )
        unique_events = {
            _safe_int(t.get("event_chunk"), _safe_int(t.get("video_chunk")))
            for t in turns
        }
        unique_event_chunk_counts.append(len({x for x in unique_events if x >= 0}))
        recall_turns = []
        compress_turns = []

        gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
        gold_action = gt.get("gold_action_per_chunk") if isinstance(gt.get("gold_action_per_chunk"), dict) else {}
        gold_recall_chunks = {
            _safe_int(k)
            for k, v in gold_action.items()
            if str(v) in {"recall", "recall_silent", "recall_response"}
        }
        gold_recall_chunks = {x for x in gold_recall_chunks if x >= 0}
        c["gold_recall_chunks"] += len(gold_recall_chunks)

        for turn in turns:
            kind = str(turn.get("kind") or "")
            turn_kind = str(turn.get("turn_kind") or "")
            format_error = str(turn.get("format_error") or "").strip()
            action_error = str(turn.get("action_space_error") or "").strip()
            if format_error:
                c["format_errors"] += 1
                if "json" in format_error.lower():
                    c["json_parse_errors"] += 1
            if action_error:
                c["action_space_errors"] += 1
            if kind == "unknown":
                c["unknown_turns"] += 1
            if turn_kind == "post_recall":
                c["post_recall_turns"] += 1
                if format_error or action_error or kind == "unknown":
                    c["post_recall_parse_errors"] += 1
            if kind == "recall":
                recall_turns.append(turn)
            if kind == "compress" or turn_kind == "compress":
                compress_turns.append(turn)

        c["turns"] += len(turns)
        c["recall_turns"] += len(recall_turns)
        c["compress_turns"] += len(compress_turns)
        c["traj_with_recall"] += int(bool(recall_turns))
        c["traj_with_compress"] += int(bool(compress_turns))

        for turn in compress_turns:
            if str(turn.get("kind") or "") == "compress":
                c["compress_kind_ok"] += 1
            if not turn.get("format_error") and not str(turn.get("action_space_error") or "").strip():
                c["compress_parse_ok"] += 1
            if turn.get("time_range_runtime_ok") is True:
                c["compress_time_ok"] += 1

        question_by_key = {_question_key(q, i): q for i, q in enumerate(questions)}
        q_recall_before_answer: dict[str, bool] = {k: False for k in question_by_key}
        q_support_recall_before_answer: dict[str, bool] = {k: False for k in question_by_key}

        for turn in recall_turns:
            recall_total += 1
            if turn.get("time_range_runtime_ok") is True:
                recall_runtime_ok += 1
            event_chunk = _safe_int(turn.get("event_chunk"), _safe_int(turn.get("video_chunk")))
            args = _turn_tool_args(turn)
            chunks = _range_to_chunks(args.get("time_range"))
            recall_range_lens.append(len(chunks))
            active_questions = []
            for idx, q in enumerate(questions):
                ask = _question_ask(q)
                deadline = _question_answer_deadline(q)
                if ask >= 0 and ask <= event_chunk <= max(deadline, event_chunk):
                    active_questions.append((idx, q))
            if not active_questions:
                active_questions = list(enumerate(questions))

            active_support = set()
            any_support = set()
            for q in questions:
                any_support |= _question_support(q)
            for _, q in active_questions:
                active_support |= _question_support(q)

            active_hit = bool(chunks & active_support) if chunks and active_support else False
            any_hit = bool(chunks & any_support) if chunks and any_support else False
            recall_active_support_hits += int(active_hit)
            recall_any_support_hits += int(any_hit)
            if chunks and active_support:
                recall_support_jaccards.append(len(chunks & active_support) / len(chunks | active_support))
                recall_support_coverages.append(len(chunks & active_support) / max(1, len(active_support)))

            for idx, q in active_questions:
                key = _question_key(q, idx)
                q_recall_before_answer[key] = True
                if active_hit:
                    q_support_recall_before_answer[key] = True

        for idx, q in enumerate(questions):
            key = _question_key(q, idx)
            form = str(q.get("answer_form") or "unknown")
            c["questions"] += 1
            by_form[form]["questions"] += 1
            events = answer_by_q.get(key, [])
            answered = bool(events)
            c["answered_questions"] += int(answered)
            by_form[form]["answered"] += int(answered)
            correct = False
            for ev in events:
                text = str(ev.get("text") or "")
                timing = str(ev.get("timing") or "unknown")
                c[f"answer_timing/{timing}"] += 1
                by_form[form][f"timing/{timing}"] += 1
                if ev.get("counts_for_completion") is False:
                    c["non_counted_answers"] += 1
                if _is_correct_answer(text, q):
                    correct = True
            c["correct_questions"] += int(correct)
            by_form[form]["correct"] += int(correct)
            bucket = q_with_prior_recall if q_recall_before_answer.get(key) else q_without_prior_recall
            bucket["questions"] += 1
            bucket["answered"] += int(answered)
            bucket["correct"] += int(correct)
            if q_support_recall_before_answer.get(key):
                q_with_support_recall["questions"] += 1
                q_with_support_recall["answered"] += int(answered)
                q_with_support_recall["correct"] += int(correct)

    return {
        "counts": c,
        "by_form": by_form,
        "reward_mean": mean(reward_scores) if reward_scores else 0.0,
        "chunk_units_mean": mean(chunk_unit_counts) if chunk_unit_counts else 0.0,
        "turn_rows_mean": mean(turn_row_counts) if turn_row_counts else 0.0,
        "unique_event_chunks_mean": mean(unique_event_chunk_counts) if unique_event_chunk_counts else 0.0,
        "recall_total": recall_total,
        "recall_runtime_ok": recall_runtime_ok,
        "recall_active_support_hits": recall_active_support_hits,
        "recall_any_support_hits": recall_any_support_hits,
        "recall_range_len_mean": mean(recall_range_lens) if recall_range_lens else 0.0,
        "recall_support_jaccard_mean": mean(recall_support_jaccards) if recall_support_jaccards else 0.0,
        "recall_support_coverage_mean": mean(recall_support_coverages) if recall_support_coverages else 0.0,
        "q_with_prior_recall": q_with_prior_recall,
        "q_without_prior_recall": q_without_prior_recall,
        "q_with_support_recall": q_with_support_recall,
    }


def print_report(train_steps: list[dict[str, float]], audit_rows: list[dict[str, Any]]) -> None:
    if train_steps:
        last = train_steps[-1]
        recent = train_steps[-5:]
        avg_units = [
            x.get("recurrent/avg_action_units_per_traj", x.get("recurrent/avg_actions_per_traj", 0.0))
            for x in recent
        ]
        avg_rows = [
            x.get("recurrent/avg_subturn_rows_per_traj", x.get("recurrent/avg_actions_per_traj", 0.0))
            for x in recent
        ]
        expanded = [x.get("recurrent/expanded_rows", 0.0) for x in recent]
        print("== train.log ==")
        print(
            f"last_step={int(last['step'])} "
            f"avg_action_units_per_traj="
            f"{last.get('recurrent/avg_action_units_per_traj', last.get('recurrent/avg_actions_per_traj', 0.0)):.2f} "
            f"avg_subturn_rows_per_traj="
            f"{last.get('recurrent/avg_subturn_rows_per_traj', last.get('recurrent/avg_actions_per_traj', 0.0)):.2f} "
            f"expanded_rows={last.get('recurrent/expanded_rows', 0.0):.0f} "
            f"n_traj={last.get('recurrent/n_trajectories', 0.0):.0f} "
            f"pad={last.get('recurrent/pad_size', 0.0):.0f} "
            f"reward_mean={last.get('recurrent/traj_reward_mean', 0.0):.4f}"
        )
        print(
            f"recent5 action_units={mean(avg_units):.2f} "
            f"subturn_rows={mean(avg_rows):.2f} "
            f"expanded_rows={mean(expanded):.1f}"
        )
    else:
        print("== train.log ==\nno recurrent step metrics found")

    s = summarize_audit(audit_rows)
    c = s["counts"]
    print("\n== rollout audit ==")
    print(
        f"records={c['records']} reward_mean={s['reward_mean']:.4f} "
        f"chunk_units_mean={s['chunk_units_mean']:.1f} "
        f"turn_rows_mean={s['turn_rows_mean']:.1f} "
        f"unique_event_chunks_mean={s['unique_event_chunks_mean']:.1f}"
    )
    print(
        f"questions={c['questions']} answered={c['answered_questions']} "
        f"({_pct(c['answered_questions'], c['questions'])}) "
        f"correct={c['correct_questions']} ({_pct(c['correct_questions'], c['questions'])}) "
        f"correct/answered={_pct(c['correct_questions'], c['answered_questions'])}"
    )
    print(
        f"recall_turns={c['recall_turns']} traj_with_recall={c['traj_with_recall']} "
        f"({_pct(c['traj_with_recall'], c['records'])}) "
        f"recall/question={c['recall_turns'] / max(1, c['questions']):.3f} "
        f"gold_recall_chunks={c['gold_recall_chunks']}"
    )
    print(
        f"recall_runtime_ok={_pct(s['recall_runtime_ok'], s['recall_total'])} "
        f"support_hit_active={_pct(s['recall_active_support_hits'], s['recall_total'])} "
        f"support_hit_any={_pct(s['recall_any_support_hits'], s['recall_total'])} "
        f"range_len_mean={s['recall_range_len_mean']:.1f} "
        f"support_jaccard_mean={s['recall_support_jaccard_mean']:.3f} "
        f"support_coverage_mean={s['recall_support_coverage_mean']:.3f}"
    )

    for label, bucket in [
        ("q_with_prior_recall", s["q_with_prior_recall"]),
        ("q_with_support_recall", s["q_with_support_recall"]),
        ("q_without_prior_recall", s["q_without_prior_recall"]),
    ]:
        print(
            f"{label}: n={bucket['questions']} answered={_pct(bucket['answered'], bucket['questions'])} "
            f"correct={_pct(bucket['correct'], bucket['questions'])} "
            f"correct/answered={_pct(bucket['correct'], bucket['answered'])}"
        )

    timing = {k.split("/", 1)[1]: v for k, v in c.items() if k.startswith("answer_timing/")}
    if timing:
        timing_str = " ".join(f"{k}={v}" for k, v in sorted(timing.items()))
    else:
        timing_str = "none"
    print(f"answer_timing: {timing_str}; non_counted={c['non_counted_answers']}")

    print(
        f"compress_turns={c['compress_turns']} traj_with_compress={c['traj_with_compress']} "
        f"kind_ok={_pct(c['compress_kind_ok'], c['compress_turns'])} "
        f"parse_ok={_pct(c['compress_parse_ok'], c['compress_turns'])} "
        f"time_ok={_pct(c['compress_time_ok'], c['compress_turns'])}"
    )
    print(
        f"parse/action health: turns={c['turns']} format_errors={c['format_errors']} "
        f"({_pct(c['format_errors'], c['turns'])}) action_space_errors={c['action_space_errors']} "
        f"({_pct(c['action_space_errors'], c['turns'])}) unknown_turns={c['unknown_turns']} "
        f"json_parse_errors={c['json_parse_errors']} "
        f"post_recall_errors={c['post_recall_parse_errors']}/{c['post_recall_turns']}"
    )

    if s["by_form"]:
        print("\nby answer_form:")
        for form, bc in sorted(s["by_form"].items()):
            print(
                f"  {form}: n={bc['questions']} answered={_pct(bc['answered'], bc['questions'])} "
                f"correct={_pct(bc['correct'], bc['questions'])} "
                f"correct/answered={_pct(bc['correct'], bc['answered'])}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, help="Run directory containing train.log and audit/rl_rollout_samples.jsonl")
    ap.add_argument("--train-log", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--tail-audit", type=int, default=500, help="Number of audit records to summarize; <=0 means all")
    ap.add_argument("--tail-log-lines", type=int, default=4000)
    args = ap.parse_args()

    train_log = args.train_log
    audit_path = args.audit
    if args.run_dir:
        train_log = train_log or args.run_dir / "train.log"
        audit_path = audit_path or args.run_dir / "audit" / "rl_rollout_samples.jsonl"
    if train_log is None or audit_path is None:
        ap.error("provide --run-dir or both --train-log and --audit")

    train_steps = _parse_train_steps(train_log, args.tail_log_lines)
    audit_rows = _load_jsonl_tail(audit_path, None if args.tail_audit <= 0 else args.tail_audit)
    print_report(train_steps, audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
