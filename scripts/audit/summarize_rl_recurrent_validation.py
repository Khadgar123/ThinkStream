#!/usr/bin/env python
"""Summarize verl recurrent validation dumps.

Input is the JSONL written by ``trainer.validation_data_dir`` in VAL_ONLY runs.
The rows already contain the reward components produced by
``thinkstream.rl.thinkstream.compute_score``; this script only aggregates them
by trajectory, task, and category. It is intentionally tied to the RL recurrent
rollout path rather than the legacy OVO HF/vLLM runners.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRIC_KEYS = (
    "score",
    "outcome",
    "answer_decision",
    "timing",
    "format",
    "spam",
    "silent_quality",
    "outcome_gate",
    "trajectory_all_correct",
    "trajectory_mean_correct",
    "per_q_outcome_min",
    "per_q_outcome_max",
    "per_q_reward_min",
    "per_q_reward_max",
    "per_chunk_action_avg",
    "recall_align_rate",
    "recall_runtime_ok_rate",
    "action_space",
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_gt(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    return {}


def _questions_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    gt = _parse_gt(row.get("gts"))
    qs = gt.get("questions") or []
    if isinstance(qs, list):
        return [q for q in qs if isinstance(q, dict)]
    return []


def _task_for_questions(questions: List[Dict[str, Any]]) -> str:
    if not questions:
        return "unknown"
    q = questions[0]
    return str(q.get("ovo_task") or q.get("family") or q.get("family_name") or "unknown")


def _category_for_questions(questions: List[Dict[str, Any]]) -> str:
    if not questions:
        return "unknown"
    q = questions[0]
    return str(q.get("ovo_category") or q.get("category") or "unknown")


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _new_bucket() -> Dict[str, Any]:
    return {
        "trajectories": 0,
        "questions": 0,
        "answered": 0.0,
        "metric_values": defaultdict(list),
        "metric_weighted_sum": defaultdict(float),
        "metric_weight": defaultdict(float),
    }


def _add(bucket: Dict[str, Any], row: Dict[str, Any], n_questions: int) -> None:
    weight = max(1, n_questions)
    bucket["trajectories"] += 1
    bucket["questions"] += weight
    bucket["answered"] += _as_float(row.get("n_answered"), 0.0)
    for key in METRIC_KEYS:
        if key not in row:
            continue
        value = _as_float(row.get(key), 0.0)
        bucket["metric_values"][key].append(value)
        bucket["metric_weighted_sum"][key] += value * weight
        bucket["metric_weight"][key] += weight
    if "score" not in row and "reward" in row:
        value = _as_float(row.get("reward"), 0.0)
        bucket["metric_values"]["score"].append(value)
        bucket["metric_weighted_sum"]["score"] += value * weight
        bucket["metric_weight"]["score"] += weight


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "trajectories": int(bucket["trajectories"]),
        "questions": int(bucket["questions"]),
        "answered_rate": (
            float(bucket["answered"]) / float(bucket["questions"])
            if bucket["questions"] else 0.0
        ),
    }
    for key, values in bucket["metric_values"].items():
        out[f"{key}_mean"] = _mean([float(v) for v in values])
        denom = float(bucket["metric_weight"].get(key, 0.0))
        if denom:
            out[f"{key}_question_weighted"] = (
                float(bucket["metric_weighted_sum"][key]) / denom
            )
    return out


def summarize(
    generations: Path,
    *,
    build_summary: Optional[Path] = None,
) -> Dict[str, Any]:
    overall = _new_bucket()
    by_task: Dict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    by_category: Dict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    task_category: Dict[str, str] = {}
    n_rows = 0

    for row in _read_jsonl(generations):
        n_rows += 1
        questions = _questions_from_row(row)
        n_questions = _as_int(row.get("n_questions"), len(questions) or 1)
        task = _task_for_questions(questions)
        category = _category_for_questions(questions)
        task_category.setdefault(task, category)
        _add(overall, row, n_questions)
        _add(by_task[task], row, n_questions)
        _add(by_category[category], row, n_questions)

    by_task_final = {task: _finalize_bucket(bucket) for task, bucket in sorted(by_task.items())}
    by_category_final = {
        category: _finalize_bucket(bucket)
        for category, bucket in sorted(by_category.items())
    }
    for bucket in by_category_final.values():
        bucket["avg"] = bucket.get("trajectory_mean_correct_question_weighted", 0.0)

    # OVO-style category average over task means, while still retaining the
    # question-weighted aggregate above.
    category_task_mean: Dict[str, float] = {}
    for category in sorted(set(task_category.values())):
        vals = [
            by_task_final[task].get("trajectory_mean_correct_question_weighted")
            for task, cat in task_category.items()
            if cat == category
        ]
        vals_f = [float(v) for v in vals if v is not None]
        category_task_mean[category] = _mean(vals_f) or 0.0
    overall_task_means = [
        float(v.get("trajectory_mean_correct_question_weighted", 0.0))
        for v in by_task_final.values()
    ]

    build_payload: Dict[str, Any] = {}
    if build_summary and build_summary.exists():
        build_payload = json.loads(build_summary.read_text(encoding="utf-8"))

    overall_final = _finalize_bucket(overall)
    overall_final["avg"] = overall_final.get("trajectory_mean_correct_question_weighted", 0.0)
    core = {
        "source": {
            "generations": str(generations),
            "build_summary": str(build_summary) if build_summary else "",
            "rows": n_rows,
        },
        "build": build_payload,
        "overall": {
            **overall_final,
            "task_macro_trajectory_mean_correct": _mean(overall_task_means) or 0.0,
        },
        "category": by_category_final,
        "category_task_macro": category_task_mean,
        "by_task": by_task_final,
        "health": {
            "answer": {
                "content_acc": overall_final.get("trajectory_mean_correct_question_weighted", 0.0),
                "all_correct_rate": overall_final.get("trajectory_all_correct_mean", 0.0),
                "answered_rate": overall_final.get("answered_rate", 0.0),
                "answer_decision": overall_final.get("answer_decision_question_weighted", 0.0),
            },
            "format_runtime": {
                "format": overall_final.get("format_mean", 0.0),
                "action_space": overall_final.get("action_space_mean", 0.0),
                "per_chunk_action_avg": overall_final.get("per_chunk_action_avg_mean", None),
            },
            "recall": {
                "align_rate": overall_final.get("recall_align_rate_mean", None),
                "runtime_ok_rate": overall_final.get("recall_runtime_ok_rate_mean", None),
            },
        },
    }
    return {**core, "summary": core}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--build-summary", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = summarize(
        Path(args.generations),
        build_summary=Path(args.build_summary) if args.build_summary else None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out.get("overall", {}), ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
