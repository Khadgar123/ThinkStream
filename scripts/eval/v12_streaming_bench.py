"""v12 streaming benchmark eval — runs trajectory-aware streaming eval.

Loads val/test_trajectories.jsonl(.gz), enriches each question with its
text from the corresponding response sample, runs the v12 streaming
agent through every chunk of every trajectory, then scores per-question
outcome + per-chunk silent_quality + per-trajectory aggregate.

Decoupled from vLLM: the rollout `step_fn` is injected. For production,
wire to `thinkstream.eval.streaming_vllm.streaming_predict_mcq_vllm` (which
does the actual vLLM batched generation). For unit tests, pass a stub
that returns canned outputs.

Usage:
    python -m scripts.eval.v12_streaming_bench \\
        --trajectories data/agent_v5/final/test_trajectories.jsonl.gz \\
        --output_dir output/eval_test_v12 \\
        --dry_run        # just enrich + report stats; no model inference
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _open_jsonl(path: Path):
    """Open .jsonl or .jsonl.gz transparently."""
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def load_trajectories(path: Path) -> List[Dict]:
    """Load trajectory file (handles .jsonl + .jsonl.gz)."""
    with _open_jsonl(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def enrich_question_text(trajectory: Dict) -> List[Dict]:
    """For each question in trajectory.questions, extract its actual
    question text from the corresponding response sample's
    `input.user_input` field. Returns the questions list with an added
    `question_text` field.

    Some cards (E2 event_watch) have empty user_input — the question is
    implicit ("describe the most salient event so far"). We label those
    as `<implicit>` rather than failing.
    """
    samples_by_card = defaultdict(list)
    for s in trajectory.get("samples", []):
        cid = s.get("card_id")
        if cid:
            samples_by_card[cid].append(s)

    enriched = []
    for q in trajectory.get("questions", []):
        cid = q["card_id"]
        text = ""
        for s in samples_by_card.get(cid, []):
            if s.get("sample_type") in ("response", "recall_response", "recall"):
                text = (s.get("input") or {}).get("user_input") or ""
                if text:
                    break
        if not text:
            text = "<implicit>"
        enriched.append({**q, "question_text": text})
    return enriched


def compute_chunk_window(
    trajectory: Dict, *, post_window: int = 5,
) -> int:
    """Rollout length: max ask_chunk + post_window, capped at trajectory's
    chunk_idx_max + 1 (don't roll past the video)."""
    max_ask = 0
    for q in trajectory.get("questions") or []:
        for ac in q.get("ask_chunks") or []:
            max_ask = max(max_ask, int(ac))
    if not max_ask:
        max_ask = trajectory.get("stats", {}).get("chunk_idx_max", 0)
    return max(1, max_ask + post_window)


def run_streaming_eval(
    trajectory: Dict,
    step_fn: Callable[[int, Optional[str]], Dict],
    *,
    post_window: int = 5,
) -> Dict:
    """Run streaming agent through one trajectory, returning per-chunk
    parsed outputs. `step_fn(chunk_idx, user_question) -> parsed_output`.

    parsed_output expected shape:
      {"chunk_idx": int, "kind": "answer"|"recall"|"compress"|"unknown",
       "answer_text": Optional[str], "tool_call": Optional[dict]}
    """
    questions = trajectory.get("questions") or []
    # Build chunk → question_text map (from enriched questions, if present)
    q_at_chunk: Dict[int, str] = {}
    for q in questions:
        text = q.get("question_text") or q.get("question") or ""
        for ac in q.get("ask_chunks") or []:
            q_at_chunk[int(ac)] = text

    num_chunks = compute_chunk_window(trajectory, post_window=post_window)
    chunk_outputs: List[Dict] = []
    for chunk_idx in range(num_chunks):
        out = step_fn(chunk_idx, q_at_chunk.get(chunk_idx))
        out["chunk_idx"] = chunk_idx
        chunk_outputs.append(out)
    return {
        "video_id": trajectory["video_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "chunk_outputs": chunk_outputs,
        "n_chunks": num_chunks,
    }


def score_trajectory(
    trajectory: Dict,
    chunk_outputs: List[Dict],
    *,
    answer_window_chunks: int = 5,
) -> Dict:
    """Score one trajectory's rollout via v12.4 reward functions."""
    # Lazy imports to avoid loading transformers in dry-run mode.
    from thinkstream.trainer.v12_rewards import (
        compute_trajectory_outcome_v12,
        compute_per_chunk_silent_quality_v12,
    )

    questions = trajectory.get("questions") or []
    gold_action_per_chunk = trajectory.get("gold_action_per_chunk") or {}

    outcome = compute_trajectory_outcome_v12(
        rollout_chunk_outputs=chunk_outputs,
        trajectory_questions=questions,
        answer_window_chunks=answer_window_chunks,
    )
    silent = compute_per_chunk_silent_quality_v12(
        rollout_chunk_outputs=chunk_outputs,
        gold_action_per_chunk=gold_action_per_chunk,
    )
    # Per-family breakdown
    per_family: Dict[str, List[float]] = defaultdict(list)
    for q, score in zip(questions, outcome.get("per_q_outcomes", [])):
        per_family[q.get("family", "?")].append(score)
    return {
        "video_id": trajectory["video_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "outcome": outcome["outcome"],
        "n_questions": outcome["n_questions"],
        "n_answered": outcome["n_answered"],
        "n_correct": outcome["n_correct"],
        "silent_quality": silent["silent_quality"],
        "n_correct_silent": silent["n_correct_silent"],
        "n_hallucinate": silent["n_hallucinate"],
        "n_missed": silent["n_missed"],
        "per_family": dict(per_family),
    }


def aggregate(per_traj_scores: List[Dict]) -> Dict:
    """Aggregate across all trajectories: overall, per-family."""
    if not per_traj_scores:
        return {}
    n_traj = len(per_traj_scores)
    n_q = sum(s["n_questions"] for s in per_traj_scores)
    n_corr = sum(s["n_correct"] for s in per_traj_scores)
    n_ans = sum(s["n_answered"] for s in per_traj_scores)
    overall_outcome = (
        sum(s["outcome"] * s["n_questions"] for s in per_traj_scores)
        / max(n_q, 1)
    )
    overall_silent = sum(s["silent_quality"] for s in per_traj_scores) / n_traj
    fam_acc: Dict[str, List[float]] = defaultdict(list)
    for s in per_traj_scores:
        for fam, scores in s["per_family"].items():
            fam_acc[fam].extend(scores)
    fam_summary = {
        fam: {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "n": len(scores),
        }
        for fam, scores in fam_acc.items()
    }
    return {
        "n_trajectories": n_traj,
        "n_questions": n_q,
        "n_answered": n_ans,
        "n_correct": n_corr,
        "answer_rate": n_ans / max(n_q, 1),
        "outcome_acc": overall_outcome,
        "silent_quality_avg": overall_silent,
        "n_hallucinate_total": sum(s["n_hallucinate"] for s in per_traj_scores),
        "n_missed_total": sum(s["n_missed"] for s in per_traj_scores),
        "per_family": fam_summary,
    }


def make_dry_run_step_fn():
    """Stub step_fn for testing — emits silent everywhere."""
    def _stub(chunk_idx: int, user_question: Optional[str]) -> Dict:
        return {
            "chunk_idx": chunk_idx,
            "kind": "answer",
            "answer_text": None,
            "tool_call": None,
        }
    return _stub


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--trajectories", required=True, type=Path,
        help="Path to *_trajectories.jsonl[.gz]",
    )
    ap.add_argument(
        "--output_dir", type=Path, default=Path("output/eval_v12"),
        help="Output directory for per-trajectory scores + aggregate JSON",
    )
    ap.add_argument(
        "--max_trajectories", type=int, default=0,
        help="Limit to first N trajectories (0 = all)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Skip model inference; emit silent at every chunk (sanity check)",
    )
    ap.add_argument(
        "--answer_window", type=int, default=5,
        help="Per-ask answer window (chunks past ask_chunk before timeout)",
    )
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trajectories from {args.trajectories}")
    trajectories = load_trajectories(args.trajectories)
    if args.max_trajectories > 0:
        trajectories = trajectories[:args.max_trajectories]
    logger.info(f"  {len(trajectories)} trajectories loaded")

    if args.dry_run:
        step_fn = make_dry_run_step_fn()
        logger.info("DRY RUN — using silent stub step_fn (no model)")
    else:
        raise NotImplementedError(
            "Production step_fn (vLLM streaming wired to streaming_vllm.py) "
            "is not yet implemented. Pass --dry_run for now."
        )

    per_traj = []
    for i, traj in enumerate(trajectories):
        # Enrich questions with their text from samples
        traj = {**traj, "questions": enrich_question_text(traj)}
        rollout = run_streaming_eval(traj, step_fn,
                                      post_window=args.answer_window)
        score = score_trajectory(traj, rollout["chunk_outputs"],
                                  answer_window_chunks=args.answer_window)
        per_traj.append(score)
        if (i + 1) % 50 == 0:
            logger.info(f"  scored {i+1}/{len(trajectories)}")

    agg = aggregate(per_traj)
    logger.info(f"\n=== Aggregate ({len(per_traj)} trajectories) ===")
    logger.info(f"  n_questions:        {agg['n_questions']}")
    logger.info(f"  answer_rate:        {agg['answer_rate']:.1%}")
    logger.info(f"  outcome_acc:        {agg['outcome_acc']:.3f}")
    logger.info(f"  silent_quality_avg: {agg['silent_quality_avg']:.3f}")
    logger.info(f"  n_hallucinate:      {agg['n_hallucinate_total']}")
    logger.info(f"  n_missed:           {agg['n_missed_total']}")
    logger.info(f"\n=== Per-family ===")
    for fam, stat in sorted(agg["per_family"].items(), key=lambda x: -x[1]["n"]):
        logger.info(f"  {fam:5s}: acc={stat['mean']:.3f}  n={stat['n']}")

    out_per_traj = args.output_dir / "per_trajectory_scores.jsonl"
    with out_per_traj.open("w") as f:
        for s in per_traj:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    out_agg = args.output_dir / "aggregate.json"
    with out_agg.open("w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults written to:")
    logger.info(f"  {out_per_traj}")
    logger.info(f"  {out_agg}")


if __name__ == "__main__":
    main()
