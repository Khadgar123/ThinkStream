"""Legacy v12 streaming benchmark dry-run helpers.

The production OVO/SFT/RL eval path now lives under ``scripts/eval/ovo``.
This module is retained for older unit tests and for cheap trajectory-shape
sanity checks without launching model inference.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_trajectories(path: Path) -> List[Dict]:
    with _open_jsonl(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def enrich_question_text(trajectory: Dict) -> List[Dict]:
    samples_by_card = defaultdict(list)
    for sample in trajectory.get("samples", []):
        card_id = sample.get("card_id")
        if card_id:
            samples_by_card[card_id].append(sample)

    enriched = []
    for question in trajectory.get("questions", []):
        text = ""
        for sample in samples_by_card.get(question.get("card_id"), []):
            if sample.get("sample_type") in {"response", "recall_response"}:
                text = (sample.get("input") or {}).get("user_input") or ""
                if text:
                    break
        enriched.append({**question, "question_text": text or "<implicit>"})
    return enriched


def compute_chunk_window(trajectory: Dict, *, post_window: int = 5) -> int:
    max_ask = 0
    for question in trajectory.get("questions") or []:
        for ask_chunk in question.get("ask_chunks") or []:
            max_ask = max(max_ask, int(ask_chunk))
    if not max_ask:
        max_ask = int((trajectory.get("stats") or {}).get("chunk_idx_max", 0))
    return max(1, max_ask + int(post_window))


def compute_benchmark_segment(
    trajectory: Dict,
    *,
    min_span_chunks: int = 25,
    max_span_chunks: int = 45,
    post_window: int = 2,
) -> Dict[str, int | bool]:
    """Return the canonical benchmark window for dry-run/test trajectories.

    The scored question is kept away from the start boundary whenever there is
    any earlier video context available. Extremely early questions are the only
    unavoidable exception.
    """
    asks: list[int] = []
    answers: list[int] = []
    for question in trajectory.get("questions") or []:
        asks.extend(int(x) for x in question.get("ask_chunks") or [])
        answers.extend(int(x) for x in question.get("answer_chunks") or [])
    anchor = min(asks or answers or [0])
    last_needed = max(answers or asks or [anchor])
    min_span = max(1, int(min_span_chunks))
    max_span = max(min_span, int(max_span_chunks))
    end = max(last_needed + int(post_window), anchor + int(post_window))
    start = max(0, end - max_span + 1)
    if start >= anchor and anchor > 0:
        start = anchor - 1
        end = max(end, start + min_span - 1)
    if end - start + 1 < min_span:
        end = start + min_span - 1
    if end - start + 1 > max_span:
        start = max(0, end - max_span + 1)
    return {
        "segment_start_chunk": int(start),
        "segment_end_chunk": int(end),
        "span_chunks": int(end - start + 1),
        "question_on_start_boundary": bool(anchor == start),
    }


def run_streaming_eval(
    trajectory: Dict,
    step_fn: Callable[[int, Optional[str]], Dict],
    *,
    post_window: int = 5,
) -> Dict:
    q_at_chunk: Dict[int, str] = {}
    for question in trajectory.get("questions") or []:
        text = question.get("question_text") or question.get("question") or ""
        for ask_chunk in question.get("ask_chunks") or []:
            q_at_chunk[int(ask_chunk)] = text

    n_chunks = compute_chunk_window(trajectory, post_window=post_window)
    chunk_outputs = []
    for chunk_idx in range(n_chunks):
        out = dict(step_fn(chunk_idx, q_at_chunk.get(chunk_idx)))
        out["chunk_idx"] = chunk_idx
        chunk_outputs.append(out)
    return {
        "video_id": trajectory["video_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "chunk_outputs": chunk_outputs,
        "n_chunks": n_chunks,
    }


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _answer_matches(answer: str, gold: str) -> bool:
    answer_n = _norm(answer)
    gold_n = _norm(gold)
    return bool(answer_n and gold_n and (answer_n == gold_n or gold_n in answer_n))


def score_trajectory(
    trajectory: Dict,
    chunk_outputs: List[Dict],
    *,
    answer_window_chunks: int = 5,
) -> Dict:
    questions = trajectory.get("questions") or []
    n_answered = 0
    n_correct = 0
    n_missed = 0
    per_q = []
    per_family: Dict[str, List[float]] = defaultdict(list)

    for question in questions:
        gold = question.get("gold_answer") or question.get("canonical_answer") or ""
        ask_chunks = [int(x) for x in question.get("ask_chunks") or []]
        start = min(ask_chunks) if ask_chunks else 0
        end = start + int(answer_window_chunks)
        answers = [
            out.get("answer_text")
            for out in chunk_outputs
            if start <= int(out.get("chunk_idx", 0)) <= end and out.get("answer_text")
        ]
        answered = bool(answers)
        correct = any(_answer_matches(ans, gold) for ans in answers)
        n_answered += int(answered)
        n_correct += int(correct)
        n_missed += int(not answered)
        score = 1.0 if correct else 0.0
        per_q.append(score)
        per_family[question.get("family", "?")].append(score)

    gold_action_per_chunk = {
        int(k): v for k, v in (trajectory.get("gold_action_per_chunk") or {}).items()
    }
    silent_total = 0.0
    n_hallucinate = 0
    for out in chunk_outputs:
        idx = int(out.get("chunk_idx", 0))
        should_respond = str(gold_action_per_chunk.get(idx, "silent")) != "silent"
        answered = bool(out.get("answer_text"))
        if should_respond and not answered:
            silent_total -= 0.6
        elif not should_respond and not answered:
            silent_total += 0.3
        elif not should_respond and answered:
            n_hallucinate += 1
            silent_total -= 0.6
    silent_quality = silent_total / max(len(chunk_outputs), 1)

    return {
        "video_id": trajectory["video_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "outcome": sum(per_q) / max(len(per_q), 1),
        "n_questions": len(questions),
        "n_answered": n_answered,
        "n_correct": n_correct,
        "silent_quality": silent_quality,
        "n_correct_silent": sum(
            1
            for out in chunk_outputs
            if str(gold_action_per_chunk.get(int(out.get("chunk_idx", 0)), "silent")) == "silent"
            and not out.get("answer_text")
        ),
        "n_hallucinate": n_hallucinate,
        "n_missed": n_missed,
        "per_family": dict(per_family),
    }


def aggregate(per_traj_scores: List[Dict]) -> Dict:
    if not per_traj_scores:
        return {}
    n_traj = len(per_traj_scores)
    n_questions = sum(item["n_questions"] for item in per_traj_scores)
    family_scores: Dict[str, List[float]] = defaultdict(list)
    for item in per_traj_scores:
        for family, scores in item.get("per_family", {}).items():
            family_scores[family].extend(scores)
    return {
        "n_trajectories": n_traj,
        "n_questions": n_questions,
        "n_answered": sum(item["n_answered"] for item in per_traj_scores),
        "n_correct": sum(item["n_correct"] for item in per_traj_scores),
        "answer_rate": sum(item["n_answered"] for item in per_traj_scores) / max(n_questions, 1),
        "outcome_acc": (
            sum(item["outcome"] * item["n_questions"] for item in per_traj_scores)
            / max(n_questions, 1)
        ),
        "silent_quality_avg": sum(item["silent_quality"] for item in per_traj_scores) / n_traj,
        "n_hallucinate_total": sum(item["n_hallucinate"] for item in per_traj_scores),
        "n_missed_total": sum(item["n_missed"] for item in per_traj_scores),
        "per_family": {
            family: {"mean": sum(scores) / len(scores), "n": len(scores)}
            for family, scores in family_scores.items()
            if scores
        },
    }


def make_dry_run_step_fn():
    def _stub(chunk_idx: int, user_question: Optional[str]) -> Dict:
        return {
            "chunk_idx": chunk_idx,
            "kind": "answer",
            "answer_text": None,
            "tool_call": None,
        }

    return _stub


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("output/eval_v12"))
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    rows = load_trajectories(args.trajectories)
    if args.max_trajectories:
        rows = rows[: args.max_trajectories]
    if not args.dry_run:
        raise NotImplementedError("Only --dry_run is retained for the legacy v12 benchmark.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_fn = make_dry_run_step_fn()
    scores = []
    for row in rows:
        row = {**row, "questions": enrich_question_text(row)}
        result = run_streaming_eval(row, step_fn)
        scores.append(score_trajectory(row, result["chunk_outputs"]))
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
