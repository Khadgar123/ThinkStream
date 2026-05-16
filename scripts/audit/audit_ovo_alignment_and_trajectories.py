#!/usr/bin/env python3
"""Audit OVO alignment and ThinkStream trajectory distributions.

This is intentionally read-only. It summarizes:
- OVO task/question/option distribution.
- Existing final trajectory question/action timing.
- Family uniqueness and support-position buckets.
- Rendered-message protocol issues when rendered trajectory rows exist.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    def q(p: float) -> float:
        idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
        return round(float(xs[idx]), 2)
    return {
        "n": len(xs),
        "mean": round(float(mean(xs)), 2),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": round(float(xs[-1]), 2),
    }


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _question_bucket(text: str) -> str:
    q = " ".join(str(text or "").lower().split())
    if q.startswith(("what color", "what colour")):
        return "what_color"
    if q.startswith("what text") or "what text" in q:
        return "what_text_ocr"
    if q.startswith("where") or "where " in q[:12]:
        return "where_location"
    if q.startswith("who") or " who " in q[:20]:
        return "who_identity"
    if q.startswith("how many"):
        return "how_many_count"
    if q.startswith(("did ", "is ", "was ", "were ", "does ", "do ")):
        return "yes_no_status"
    if "before" in q:
        return "before_temporal"
    if "after" in q:
        return "after_temporal"
    if "while" in q or "when" in q:
        return "while_when"
    return "other"


def audit_ovo(path: Path, sample_n: int = 3) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = Counter()
    option_counts = Counter()
    buckets = Counter()
    answer_space = defaultdict(Counter)
    realtime_by_task: Dict[str, List[int]] = defaultdict(list)
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in data:
        task = str(row.get("task") or "")
        tasks[task] += 1
        opts = row.get("options") or []
        option_counts[len(opts)] += 1
        bucket = _question_bucket(row.get("question", ""))
        buckets[bucket] += 1
        ans = str(row.get("answer") or "").strip()
        if ans:
            answer_space[task][ans] += 1
        try:
            realtime_by_task[task].append(int(row.get("realtime")))
        except Exception:
            pass
        if len(samples[task]) < sample_n:
            samples[task].append({
                "id": row.get("id"),
                "task": task,
                "realtime": row.get("realtime"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "options": opts,
            })
    return {
        "n": len(data),
        "task_counts": dict(tasks.most_common()),
        "task_pct": {k: _pct(v, len(data)) for k, v in tasks.most_common()},
        "option_count_pct": {str(k): _pct(v, len(data)) for k, v in sorted(option_counts.items())},
        "question_bucket_pct": {k: _pct(v, len(data)) for k, v in buckets.most_common()},
        "realtime_by_task": {k: _percentiles(v) for k, v in sorted(realtime_by_task.items())},
        "top_answers_by_task": {
            k: dict(v.most_common(8)) for k, v in sorted(answer_space.items())
        },
        "samples": dict(samples),
    }


def _support_bucket(q: Dict[str, Any], visual_window: int = 8) -> str:
    ask = q.get("ask_chunk", q.get("ask_time", 0))
    try:
        ask_i = int(float(ask))
    except Exception:
        ask_i = 0
    supports = []
    for c in q.get("support_chunks") or q.get("grounding_frames") or []:
        try:
            supports.append(int(c))
        except Exception:
            pass
    if not supports:
        return "no_support"
    mn, mx = min(supports), max(supports)
    if mn > ask_i:
        return "future"
    if mx < ask_i - visual_window:
        return "historical"
    if mn <= ask_i <= mx:
        return "current_overlap"
    if mx <= ask_i and mx >= ask_i - visual_window:
        return "recent_window"
    return "mixed"


def _longest_silent_run(actions: Dict[str, str]) -> int:
    if not actions:
        return 0
    pairs = []
    for k, v in actions.items():
        try:
            pairs.append((int(k), str(v)))
        except Exception:
            continue
    if not pairs:
        return 0
    pairs.sort()
    best = cur = 0
    prev = None
    for c, action in pairs:
        if action == "silent" and (prev is None or c == prev + 1):
            cur += 1
        elif action == "silent":
            cur = 1
        else:
            cur = 0
        best = max(best, cur)
        prev = c
    return best


def audit_final(root: Path, batches: str) -> Dict[str, Any]:
    batch_ids: List[int] = []
    for part in batches.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            batch_ids.extend(range(int(a), int(b) + 1))
        elif part:
            batch_ids.append(int(part))
    files: List[Tuple[str, Path]] = []
    for b in batch_ids:
        final = root / f"batch{b}" / "final"
        for split in ("train_sft", "train_rl", "val", "test"):
            p = final / f"{split}_trajectories.jsonl"
            if p.exists():
                files.append((split, p))

    split_rows = Counter()
    split_questions = Counter()
    split_actions = Counter()
    family = Counter()
    family_answer_rows = Counter()
    answer_form = Counter()
    qtype = Counter()
    ours = Counter()
    support_bucket = Counter()
    availability = Counter()
    question_bucket = Counter()
    question_intervals: List[int] = []
    q_per_traj: List[int] = []
    longest_silent: List[int] = []
    action_counts = Counter()
    recall_query_samples = []
    bad_mc = []
    old_protocol_rows = Counter()
    samples_out = []

    for split, path in files:
        for row in _load_jsonl(path):
            split_rows[split] += 1
            questions = row.get("questions") or []
            q_per_traj.append(len(questions))
            asks = []
            for q in questions:
                split_questions[split] += 1
                fam = str(q.get("family") or (row.get("metadata") or {}).get("family") or "")
                family[fam] += 1
                answer_form[str(q.get("answer_form") or "")] += 1
                qtype[str(q.get("question_type") or "")] += 1
                ours[str(bool(q.get("ours_unique")))] += 1
                availability[str(q.get("availability") or "")] += 1
                support_bucket[_support_bucket(q)] += 1
                question_bucket[_question_bucket(q.get("question", ""))] += 1
                try:
                    asks.append(int(q.get("ask_chunk", q.get("ask_time", 0))))
                except Exception:
                    pass
                for e in q.get("per_emit_answers") or []:
                    family_answer_rows[fam] += 1
                if q.get("answer_form") == "multiple_choice":
                    opts = [str(x).strip() for x in q.get("options") or [] if str(x).strip()]
                    corr = str(q.get("correct_option") or "").strip()
                    sft = str(q.get("sft_answer") or "").strip()
                    if corr and opts and not any(o.startswith(corr + ")") for o in opts):
                        bad_mc.append({"video_id": row.get("video_id"), "card_id": q.get("card_id"), "reason": "correct_option_missing", "q": q.get("question")})
                    if corr and sft and corr != sft:
                        bad_mc.append({"video_id": row.get("video_id"), "card_id": q.get("card_id"), "reason": "sft_answer_mismatch", "corr": corr, "sft": sft})
            asks = sorted(set(asks))
            question_intervals.extend(b - a for a, b in zip(asks, asks[1:]))
            actions = row.get("gold_action_per_chunk") or {}
            longest_silent.append(_longest_silent_run(actions))
            for a in actions.values():
                action_counts[str(a)] += 1
                split_actions[f"{split}:{a}"] += 1
            if len(samples_out) < 6:
                samples_out.append({
                    "split": split,
                    "video_id": row.get("video_id"),
                    "n_questions": len(questions),
                    "families": [q.get("family") for q in questions],
                    "asks": asks,
                    "actions": dict(Counter(actions.values())),
                    "longest_silent": _longest_silent_run(actions),
                    "first_questions": [
                        {
                            "family": q.get("family"),
                            "ask": q.get("ask_chunk"),
                            "support": q.get("support_chunks"),
                            "availability": q.get("availability"),
                            "question": q.get("question"),
                            "answer_form": q.get("answer_form"),
                        }
                        for q in questions[:3]
                    ],
                })
            for sample in row.get("samples") or []:
                out = str(sample.get("output") or "")
                if "<answer>" in out or "</answer>" in out:
                    old_protocol_rows["answer_tag"] += 1
                if "<response>" in out or "</response>" in out:
                    old_protocol_rows["response_tag"] += 1
                if ("<" + "action>") in out or ("</" + "action>") in out:
                    old_protocol_rows["action_tag"] += 1
                if "<MEM>" in out or "</MEM>" in out:
                    old_protocol_rows["mem_tag"] += 1
                rq = sample.get("recall_query") or (sample.get("metadata") or {}).get("recall_query")
                rr = sample.get("recall_result")
                if (rq or rr) and len(recall_query_samples) < 8:
                    recall_query_samples.append({
                        "video_id": row.get("video_id"),
                        "chunk_idx": sample.get("chunk_idx"),
                        "family": (sample.get("metadata") or {}).get("family"),
                        "question": (sample.get("metadata") or {}).get("question"),
                        "recall_query": rq,
                        "recall_result_time_range": (rr or {}).get("time_range") if isinstance(rr, dict) else None,
                    })

    total_q = sum(family.values())
    total_actions = sum(action_counts.values())
    return {
        "files": [str(p) for _, p in files],
        "split_rows": dict(split_rows),
        "split_row_pct": {k: _pct(v, sum(split_rows.values())) for k, v in split_rows.items()},
        "split_questions": dict(split_questions),
        "questions_per_trajectory": _percentiles(q_per_traj),
        "question_interval_chunks": _percentiles(question_intervals),
        "longest_silent_run": _percentiles(longest_silent),
        "action_counts": dict(action_counts.most_common()),
        "action_pct": {k: _pct(v, total_actions) for k, v in action_counts.most_common()},
        "family_counts": dict(family.most_common()),
        "family_pct": {k: _pct(v, total_q) for k, v in family.most_common()},
        "family_answer_rows": dict(family_answer_rows.most_common()),
        "answer_form_pct": {k: _pct(v, total_q) for k, v in answer_form.most_common()},
        "question_type_pct": {k: _pct(v, total_q) for k, v in qtype.most_common()},
        "ours_unique_pct": {k: _pct(v, total_q) for k, v in ours.most_common()},
        "availability_pct": {k: _pct(v, total_q) for k, v in availability.most_common()},
        "support_bucket_pct": {k: _pct(v, total_q) for k, v in support_bucket.most_common()},
        "question_bucket_pct": {k: _pct(v, total_q) for k, v in question_bucket.most_common()},
        "old_protocol_rows": dict(old_protocol_rows),
        "mc_consistency_issues": bad_mc[:20],
        "recall_query_samples": recall_query_samples,
        "trajectory_samples": samples_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/agent_v5")
    ap.add_argument("--batches", default="1-11")
    ap.add_argument("--ovo-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {
        "ovo": audit_ovo(Path(args.ovo_json)),
        "final_trajectories": audit_final(Path(args.root), args.batches),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "ovo_n": result["ovo"]["n"],
        "final_split_rows": result["final_trajectories"]["split_rows"],
        "final_questions_per_trajectory": result["final_trajectories"]["questions_per_trajectory"],
        "final_action_pct": result["final_trajectories"]["action_pct"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
