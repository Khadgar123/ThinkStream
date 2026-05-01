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
                    "gold_answer": gold_answer,
                    "answer_form": answer_form,
                    "ask_chunks": ask_chunks,
                    "gold_action_per_chunk": q_gold_action,
                    "n_chunks": n_chunks,
                    "extra_info": {
                        "index": f"{video_id}#{q_idx}",
                        "video_id": video_id,
                        "question_idx": q_idx,
                        "card_id": q.get("card_id", ""),
                        "family": q.get("family", ""),
                        "support_chunks": list(q.get("support_chunks") or []),
                    },
                    # verl convention: reward_model.ground_truth is what the
                    # reward function receives as `ground_truth`. Use a dict
                    # so we can pass the full bundle, not just a string.
                    "reward_model": {
                        "ground_truth": json.dumps({
                            "gold_answer": gold_answer,
                            "answer_form": answer_form,
                            "ask_chunks": ask_chunks,
                            "visible_start_chunk": min(ask_chunks) if ask_chunks else None,
                            "visible_end_chunk":   max(ask_chunks) if ask_chunks else None,
                            "gold_action_per_chunk": q_gold_action,
                        }, ensure_ascii=False),
                        "style": "thinkstream_v12",
                    },
                    "data_source": "thinkstream_v12_streaming",
                }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="pass4 trajectory JSONL[.gz]")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument(
        "--max_questions_per_traj",
        type=int,
        default=5,
        help="cap per video; pass4 typically emits ≤5 (default: 5)",
    )
    args = ap.parse_args()

    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = list(
        _iter_rows(in_path, max_questions_per_traj=args.max_questions_per_traj)
    )
    if not rows:
        print(f"[build_verl_parquet] no rows produced from {in_path}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(
        f"[build_verl_parquet] {in_path.name}: {len(rows)} (video,question) rows "
        f"→ {out_path} ({out_path.stat().st_size/1024:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
