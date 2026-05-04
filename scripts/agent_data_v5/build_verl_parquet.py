"""Flatten ThinkStream pass4 trajectory JSONL into a verl RLHFDataset parquet.

verl's `RLHFDataset` (and our `recipe/thinkstream/CustomRLHFDataset` subclass)
reads parquet — one row per RL training sample. Our pass4 emits one
trajectory per line with N nested questions:

    {"video_id":..., "video_path":...,
     "questions":[{"question":..., "gold_answer":..., "answer_form":...,
                   "ask_chunks":[..]}, ...],
     "gold_action_per_chunk":{...}, "stats":{...}}

This script flattens (video × question) → one parquet row with the columns
the recipe expects:

    prompt              List[Dict]           # list of {role,content} dicts
    video_id            str
    video_path          str
    question            str
    gold_answer         str
    answer_form         str
    ask_chunks          List[int]
    gold_action_per_chunk  Dict[str,str]
    n_chunks            int
    extra_info          Dict                 # passthrough metadata
    reward_model        Dict                 # verl convention: {"ground_truth": str, "style": str}

Usage:
    python -m scripts.agent_data_v5.build_verl_parquet \\
        --jsonl data/agent_v5/final/train_rl_trajectories.jsonl \\
        --out   data/agent_v5/final/train_rl.parquet
"""
from __future__ import annotations

import argparse
import json
import gzip
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

# verl/recipe expects pyarrow for parquet round-trip; pandas is a thin shim.
import pandas as pd

# Resolve repo root so we can import the v12 system prompt without
# relying on PYTHONPATH being preset.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinkstream.data.agent_protocol import SYSTEM_PROMPT_V12  # noqa: E402


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _iter_rows(jsonl_path: Path, max_questions_per_traj: int) -> Iterator[Dict[str, Any]]:
    with _open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = traj.get("video_id") or traj.get("trajectory_id") or ""
            video_path = traj.get("video_path", "")
            gold_action = traj.get("gold_action_per_chunk", {}) or {}
            n_chunks = int((traj.get("stats") or {}).get("n_chunks_covered", 0))
            questions = (traj.get("questions") or [])[:max_questions_per_traj]

            if not questions:
                continue

            for q_idx, q in enumerate(questions):
                question = q.get("question", "")
                gold_answer = q.get("gold_answer", "")
                answer_form = q.get("answer_form", "")
                options = list(q.get("options") or [])
                correct_option = q.get("correct_option", "")
                ask_chunks = list(q.get("ask_chunks") or [])

                # ── Per-question gold_action_per_chunk (P1.9 fix).
                # The trajectory-level gold_action carries actions for
                # ALL questions' chunks. When this row is one (video,
                # question) pair, only the chunks within THIS question's
                # answerable range should be scored — otherwise question A's
                # rollout gets penalised for not emitting question B's
                # response at chunk where question B was supposed to fire.
                # Strategy: keep gold_action ONLY for chunks within
                # [min(ask_chunks), max(ask_chunks)] (the question's
                # answerable window); for chunks outside that range we
                # treat the gold as "silent" so off-question chunks
                # don't penalise correct silent behaviour.
                q_gold_action: Dict[str, str] = {}
                if ask_chunks:
                    q_lo, q_hi = min(ask_chunks), max(ask_chunks)
                    for ck, gold in (gold_action or {}).items():
                        try:
                            ck_int = int(ck)
                        except (TypeError, ValueError):
                            continue
                        if q_lo <= ck_int <= q_hi:
                            # In range — keep the original action.
                            q_gold_action[ck] = gold
                        else:
                            # Out of range — silent is the correct
                            # action for this question at this chunk.
                            q_gold_action[ck] = "silent"
                else:
                    # No ask_chunks → treat as no actionable supervision.
                    q_gold_action = {ck: "silent" for ck in (gold_action or {}).keys()}

                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT_V12},
                    {"role": "user", "content": question},
                ]

                yield {
                    "prompt": prompt,
                    "video_id": video_id,
                    "video_path": video_path,
                    "question": question,
                    "options": options,
                    "correct_option": correct_option,
                    "correct_answer_text": q.get("correct_answer_text", ""),
                    "accepted_answers": list(q.get("accepted_answers") or []),
                    "answer_style": q.get("answer_style", ""),
                    "answer_instruction": q.get("answer_instruction", ""),
                    "gold_answer": gold_answer,
                    "answer_form": answer_form,
                    "answer_chunks": list(q.get("answer_chunks") or []),
                    "per_emit_answers": list(q.get("per_emit_answers") or []),
                    "ask_chunks": ask_chunks,
                    "gold_action_per_chunk": q_gold_action,
                    "n_chunks": n_chunks,
                    "extra_info": {
                        "index": f"{video_id}#{q_idx}",
                        "video_id": video_id,
                        "question_idx": q_idx,
                        "card_id": q.get("card_id", ""),
                        "family": q.get("family", ""),
                        "family_name": q.get("family_name", ""),
                        "category": q.get("category", ""),
                        "skill": q.get("skill", ""),
                        "ours_unique": bool(q.get("ours_unique", False)),
                        "options": options,
                        "correct_option": correct_option,
                        "answer_instruction": q.get("answer_instruction", ""),
                        "support_chunks": list(q.get("support_chunks") or []),
                        "answer_chunks": list(q.get("answer_chunks") or []),
                        "per_emit_answers": list(q.get("per_emit_answers") or []),
                    },
                    # verl convention: reward_model.ground_truth is what the
                    # reward function receives as `ground_truth`. Use a dict
                    # so we can pass the full bundle, not just a string.
                    "reward_model": {
                        "ground_truth": json.dumps({
                            "gold_answer": gold_answer,
                            "answer_form": answer_form,
                            "options": options,
                            "correct_option": correct_option,
                            "ask_chunks": ask_chunks,
                            "answer_chunks": list(q.get("answer_chunks") or []),
                            "per_emit_answers": list(q.get("per_emit_answers") or []),
                            "visible_start_chunk": min(ask_chunks) if ask_chunks else None,
                            "visible_end_chunk":   max(ask_chunks) if ask_chunks else None,
                            "gold_action_per_chunk": q_gold_action,
                        }, ensure_ascii=False),
                        "style": "thinkstream_v12",
                    },
                    "data_source": "thinkstream_v12_streaming",
                }


def _iter_rows_multi_q(
    jsonl_path: Path, max_questions_per_traj: int
) -> Iterator[Dict[str, Any]]:
    """Multi-Q trajectory rows: 1 video → 1 row containing ALL questions.

    This shape matches OVOBench's eval form (one video, many MCQ time-points)
    and the actual semantics of streaming-video agents — memory state +
    compress / recall decisions are SHARED across questions in one video,
    so flattening to (video, question) duplicates the visual rollout N
    times and discards the joint-supervision signal that compress / recall
    need to learn from.

    Schema per row:
      questions: List[Dict] — full pass4 question list (card_id, family,
                              ask_chunk, options, correct_option,
                              gold_answer, answer_form, per_emit_answers, ...)
      gold_action_per_chunk: Dict[str, str] — full per-chunk gold action map
                                              (NOT clipped to one question's window)
      reward_model.ground_truth: JSON-encoded list of per-question targets
                                 + the full gold_action map.

    The streaming agent loop reads `extra_info.questions` and injects each
    question's text into <user_input> at its `ask_chunk`; compute_score
    then evaluates each Q independently and aggregates (mean by default).
    """
    with _open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = traj.get("video_id") or traj.get("trajectory_id") or ""
            video_path = traj.get("video_path", "")
            gold_action = traj.get("gold_action_per_chunk", {}) or {}
            n_chunks = int((traj.get("stats") or {}).get("n_chunks_covered", 0))
            questions = (traj.get("questions") or [])[:max_questions_per_traj]

            if not questions:
                continue

            # Per-question target bundle — what compute_score scores against.
            q_targets: List[Dict[str, Any]] = []
            all_ask_chunks: List[int] = []
            for q in questions:
                ask_chunks = list(q.get("ask_chunks") or [])
                if not ask_chunks and q.get("ask_chunk", -1) >= 0:
                    ask_chunks = [int(q["ask_chunk"])]
                all_ask_chunks.extend(ask_chunks)
                q_targets.append({
                    "card_id": q.get("card_id", ""),
                    "family": q.get("family", ""),
                    "question": q.get("question", ""),
                    "options": list(q.get("options") or []),
                    "correct_option": q.get("correct_option", ""),
                    "correct_answer_text": q.get("correct_answer_text", ""),
                    "accepted_answers": list(q.get("accepted_answers") or []),
                    "answer_style": q.get("answer_style", ""),
                    "answer_instruction": q.get("answer_instruction", ""),
                    "gold_answer": q.get("gold_answer", ""),
                    "answer_form": q.get("answer_form", ""),
                    "ask_chunk": int(q.get("ask_chunk", -1)),
                    "ask_chunks": ask_chunks,
                    "answer_chunks": list(q.get("answer_chunks") or []),
                    "per_emit_answers": list(q.get("per_emit_answers") or []),
                    "support_chunks": list(q.get("support_chunks") or []),
                    "family_name": q.get("family_name", ""),
                    "category": q.get("category", ""),
                    "skill": q.get("skill", ""),
                    "ours_unique": bool(q.get("ours_unique", False)),
                })

            # System-level prompt only — actual question text is injected
            # by the agent loop at each ask_chunk. Streamed multi-Q agent
            # gets a generic role description here, not a single question.
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT_V12},
                {"role": "user", "content": (
                    "You are a streaming-video agent. You will receive video "
                    "frames in chunks and questions at specific time points. "
                    "Maintain a memory of what you observe and answer each "
                    "question when it is asked."
                )},
            ]

            yield {
                "prompt": prompt,
                "video_id": video_id,
                "video_path": video_path,
                "n_chunks": n_chunks,
                "n_questions": len(questions),
                "extra_info": {
                    "index": video_id,
                    "video_id": video_id,
                    "questions": q_targets,
                    "gold_action_per_chunk": gold_action,
                    "all_ask_chunks": sorted(set(all_ask_chunks)),
                },
                "reward_model": {
                    "ground_truth": json.dumps({
                        "questions": q_targets,
                        "gold_action_per_chunk": gold_action,
                    }, ensure_ascii=False),
                    "style": "thinkstream_v12_multi_q",
                },
                "data_source": "thinkstream_v12_streaming_multi_q",
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="pass4 trajectory JSONL[.gz]")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument(
        "--max_questions_per_traj",
        type=int,
        default=16,
        help="cap per video. Default 16 covers OVOBench worst case (max 16 Q/video).",
    )
    ap.add_argument(
        "--multi_q",
        action="store_true",
        help=(
            "Multi-Q trajectory mode: 1 video = 1 parquet row containing all "
            "questions. Aligns RL training with OVOBench eval form (one video, "
            "many MCQ time-points). Default OFF for backward compat with the "
            "(video, question) flatten path."
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = (
        _iter_rows_multi_q(in_path, max_questions_per_traj=args.max_questions_per_traj)
        if args.multi_q
        else _iter_rows(in_path, max_questions_per_traj=args.max_questions_per_traj)
    )
    rows: List[Dict[str, Any]] = list(iterator)
    if not rows:
        print(f"[build_verl_parquet] no rows produced from {in_path}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    shape_label = "video" if args.multi_q else "(video,question)"
    print(
        f"[build_verl_parquet] {in_path.name}: {len(rows)} {shape_label} rows "
        f"→ {out_path} ({out_path.stat().st_size/1024:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
