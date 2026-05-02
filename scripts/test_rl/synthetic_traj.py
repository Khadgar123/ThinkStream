"""
Synthetic pass4-format trajectory generator for mini RL test (v12.13).

Produces a JSONL file matching pass4 train_rl_trajectories.jsonl schema with
ALL mechanism types covered, including a long-video case (320 chunks). Used
to validate verl recipe + reward functions without waiting for real pass4
data.

Mechanism coverage (one trajectory per mechanism):
  - direct (realtime):           ask=10, answer=10                  (single MC)
  - silent_then_response (forward): ask=5,  answer=25  (lead 20)    (single short_exact)
  - recall_demo (backward):      ask=80, answer=80 (via recall)     (single MC)
  - multi_emit (F5 counting):    emits at 10/20/30 → "1","2","3"    (number)
  - multi_emit (F7 SSR):         emits at 6-14 → No before 10, Yes after (binary)
  - LONG video (320 chunks):    forward at ask=10, answer=300       (stress test)

Each row matches verl recipe expectations:
  {
    "video_id": str, "trajectory_id": str, "video_path": str,
    "samples": [...],            # per-chunk samples
    "questions": [
      {
        "card_id", "family", "question", "options", "correct_option",
        "answer_form", "gold_answer", "canonical_answer",
        "ask_chunk", "ask_chunks", "answer_chunks", "per_emit_answers",
        "support_chunks", "gold_compress_chunks", "availability",
      },
      ...
    ],
    "gold_action_per_chunk": {"0":"silent","5":"silent",...,"25":"response",...},
    "stats": {"n_samples":N, ...}
  }

Usage:
    python -m scripts.test_rl.synthetic_traj \\
        --out data/test_rl/synthetic_trajectories.jsonl \\
        --frames-root data/test_rl/synthetic_frames

If --frames-root is set, also generates dummy 1280×720 JPGs (random pixels)
matching the chunk frame numbering convention.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data_v5.config import (  # noqa: E402
    AGENT_CHUNK_SEC, FRAMES_PER_CHUNK,
)


def _build_question(
    card_id: str, family: str, question: str,
    answer_form: str,
    ask_chunk: int,
    answer_chunks: List[int],
    gold_answer: str,
    per_emit_answers: List[Dict] = None,
    options: List[str] = None,
    correct_option: str = "",
    availability: str = "direct_response",
) -> Dict[str, Any]:
    """Build a questions[*] entry matching pass4 schema (v12.13)."""
    return {
        "card_id": card_id,
        "family": family,
        "gold_answer": gold_answer,
        "canonical_answer": gold_answer,
        "answer_form": answer_form,
        "availability": availability,
        "support_chunks": list(answer_chunks),
        "gold_compress_chunks": [],
        "ask_chunk": ask_chunk,
        "ask_chunks": [ask_chunk],
        "answer_chunks": list(answer_chunks),
        "per_emit_answers": per_emit_answers or [
            {"chunk": c, "value": gold_answer} for c in answer_chunks
        ],
        "question": question,
        "options": options or [],
        "correct_option": correct_option or "",
    }


def _build_per_chunk_samples(
    num_chunks: int,
    questions: List[Dict],
    video_id: str,
    trajectory_id: str,
    video_path: str,
) -> List[Dict]:
    """Per-chunk sample list (compatible with pass4.samples).

    For each chunk:
      - if chunk in answer_chunks of any question → sample_type=response
      - else → sample_type=silent (patrol / wait)
    """
    samples = []
    answer_chunks_set = set()
    for q in questions:
        for ac in q.get("answer_chunks", []):
            answer_chunks_set.add(int(ac))

    for c in range(num_chunks):
        is_answer = c in answer_chunks_set
        # Find which question's answer this chunk is (if any)
        gold_answer_for_chunk = ""
        card_id_for_chunk = ""
        if is_answer:
            for q in questions:
                if c in q.get("answer_chunks", []):
                    card_id_for_chunk = q["card_id"]
                    # multi_emit: use per_emit_answers
                    per_emit = q.get("per_emit_answers", [])
                    for e in per_emit:
                        if e["chunk"] == c:
                            gold_answer_for_chunk = e["value"]
                            break
                    if not gold_answer_for_chunk:
                        gold_answer_for_chunk = q.get("gold_answer", "")
                    break

        if is_answer:
            output = (
                f"<think>I see the answer at chunk {c}.</think>"
                f"<answer>{gold_answer_for_chunk}</answer>"
            )
            sample_type = "response"
        else:
            output = "<think>Nothing notable.</think><answer></answer>"
            sample_type = "silent"

        samples.append({
            "chunk_idx": c,
            "sample_type": sample_type,
            "sample_id": f"{trajectory_id}_chunk{c:04d}",
            "card_id": card_id_for_chunk,
            "trajectory_id": trajectory_id,
            "video_id": video_id,
            "video_path": video_path,
            "v12_inter_chunk": False,
            "input": {
                "memory": {"compressed_segments": [], "recent_thinks": []},
                "queries": [],
                "user_input": "",
                "visual_window": {
                    "video_start": max(0, c - 15) * AGENT_CHUNK_SEC,
                    "video_end": (c + 1) * AGENT_CHUNK_SEC,
                    "frames": (min(c + 1, 16)) * FRAMES_PER_CHUNK,
                },
            },
            "output": output,
            "metadata": {
                "gold_action": sample_type,
                "gold_answer": gold_answer_for_chunk,
                "answer_form": "",
                "family": "",
                "ask_chunk": -1,
                "question": "",
                "options": [],
                "correct_option": "",
                "per_emit_answers": [],
            },
            "verification": {"passed": True, "fail_reasons": []},
        })

    return samples


def build_gold_action_map(num_chunks: int, questions: List[Dict]) -> Dict[str, str]:
    """Build gold_action_per_chunk dict for verl reward.

    Maps str(chunk_idx) → "silent" / "response" / "recall" / "compress".
    """
    out = {str(c): "silent" for c in range(num_chunks)}
    for q in questions:
        for ac in q.get("answer_chunks", []):
            out[str(int(ac))] = "response"
        # ask_chunk that is not also an answer_chunk → silent_then_response
        # (model SHOULD remain silent at ask_chunk for forward cards)
    return out


def _trajectory_row(
    video_id: str,
    trajectory_id: str,
    num_chunks: int,
    questions: List[Dict],
    frames_root: str = "",
) -> Dict[str, Any]:
    video_path = f"{video_id}.mp4"
    samples = _build_per_chunk_samples(num_chunks, questions, video_id,
                                         trajectory_id, video_path)
    return {
        "video_id": video_id,
        "trajectory_id": trajectory_id,
        "video_path": video_path,
        "samples": samples,
        "questions": questions,
        "gold_action_per_chunk": build_gold_action_map(num_chunks, questions),
        "stats": {
            "n_samples": len(samples),
            "n_chunks": num_chunks,
            "n_questions": len(questions),
            "actions": {"silent": sum(1 for s in samples if s["sample_type"] == "silent"),
                        "response": sum(1 for s in samples if s["sample_type"] == "response")},
        },
    }


# ----------------------------------------------------------------------------
# Mechanism builders
# ----------------------------------------------------------------------------
def make_direct(video_id: str = "v_direct") -> Dict:
    """Direct (realtime) MC: ask=answer same chunk."""
    questions = [_build_question(
        card_id=f"{video_id}_CR3_001",
        family="CR3",
        question="What is on the table at chunk 10?",
        answer_form="multiple_choice",
        ask_chunk=10,
        answer_chunks=[10],
        gold_answer="A",
        options=["A) red apple", "B) blue cup", "C) green book", "D) yellow pencil"],
        correct_option="A",
        availability="direct_response",
    )]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=30, questions=questions)


def make_forward(video_id: str = "v_forward") -> Dict:
    """Forward (silent_then_response): ask=5, answer=25 (lead 20)."""
    questions = [_build_question(
        card_id=f"{video_id}_E2_001",
        family="E2",
        question="When the chef adds salt, what color is the bowl?",
        answer_form="short_exact",
        ask_chunk=5,
        answer_chunks=[25],
        gold_answer="red",
        availability="silent_then_response",
    )]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=40, questions=questions)


def make_backward_recall(video_id: str = "v_backward") -> Dict:
    """Backward (recall_demo): ask=80, answer=80 via recall."""
    questions = [_build_question(
        card_id=f"{video_id}_CR1_001",
        family="CR1",
        question="What was on the counter at chunk 5?",
        answer_form="multiple_choice",
        ask_chunk=80,
        answer_chunks=[80],
        gold_answer="B",
        options=["A) bowl", "B) knife", "C) spoon", "D) cup"],
        correct_option="B",
        availability="backward_recall_demo",
    )]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=100, questions=questions)


def make_multi_emit_counting(video_id: str = "v_f5") -> Dict:
    """F5 counting: emits at 10/20/30 with cumulative counts."""
    answer_chunks = [10, 20, 30]
    per_emit = [{"chunk": c, "value": str(i + 1)}
                for i, c in enumerate(answer_chunks)]
    questions = [_build_question(
        card_id=f"{video_id}_F5_001",
        family="F5",
        question='How many times does "person waves" happen so far?',
        answer_form="number",
        ask_chunk=5,
        answer_chunks=answer_chunks,
        gold_answer="3",
        per_emit_answers=per_emit,
        availability="multi_emit_counting",
    )]
    # Multi-emit ask_chunks should match ask_chunk (single ask, multi answer)
    questions[0]["ask_chunks"] = [5]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=40, questions=questions)


def make_multi_emit_ssr(video_id: str = "v_f7") -> Dict:
    """F7 SSR: state changes at chunk 10; Yes/No emits at 6-14."""
    answer_chunks = list(range(6, 15))
    per_emit = []
    for c in answer_chunks:
        per_emit.append({"chunk": c, "value": "No" if c < 10 else "Yes"})
    questions = [_build_question(
        card_id=f"{video_id}_F7_001",
        family="F7",
        question='Has "the door opens" happened by now?',
        answer_form="binary",
        ask_chunk=6,
        answer_chunks=answer_chunks,
        gold_answer="Yes",
        per_emit_answers=per_emit,
        availability="multi_emit_ssr",
    )]
    questions[0]["ask_chunks"] = [6]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=20, questions=questions)


def make_long_forward(video_id: str = "v_long") -> Dict:
    """Long forward: ask=10, answer=300 in a 320-chunk video (stress)."""
    questions = [_build_question(
        card_id=f"{video_id}_E2_long",
        family="E2",
        question="When the very last action happens at the end, what is shown?",
        answer_form="short_exact",
        ask_chunk=10,
        answer_chunks=[300],
        gold_answer="finale",
        availability="silent_then_response",
    )]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=320, questions=questions)


def make_multi_card(video_id: str = "v_multi_card") -> Dict:
    """Multiple cards in one trajectory (mixes mechanisms)."""
    questions = [
        _build_question(  # direct
            card_id=f"{video_id}_CR3_a",
            family="CR3", question="What's at chunk 10?",
            answer_form="multiple_choice", ask_chunk=10, answer_chunks=[10],
            gold_answer="C", options=["A) X", "B) Y", "C) Z", "D) W"],
            correct_option="C", availability="direct",
        ),
        _build_question(  # forward
            card_id=f"{video_id}_E2_b",
            family="E2", question="When the event happens, color?",
            answer_form="short_exact", ask_chunk=20, answer_chunks=[55],
            gold_answer="green", availability="silent_then_response",
        ),
        _build_question(  # backward recall
            card_id=f"{video_id}_CR1_c",
            family="CR1", question="What was at chunk 15?",
            answer_form="binary", ask_chunk=80, answer_chunks=[80],
            gold_answer="No", availability="backward_recall",
        ),
    ]
    # Adjust ask_chunks to match ask_chunk for single-ask cards
    for q in questions:
        q["ask_chunks"] = [q["ask_chunk"]]
    return _trajectory_row(video_id, f"{video_id}_traj0", num_chunks=120, questions=questions)


# ----------------------------------------------------------------------------
# Stub frame generation (optional — for visual rollout test)
# ----------------------------------------------------------------------------
def _generate_stub_frames(frames_root: Path, video_id: str, num_chunks: int) -> None:
    """Drop random 1280×720 JPGs that match pass1a's frame_NNNNNN.jpg convention."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.warning("PIL/numpy not available — skipping frame generation")
        return
    out_dir = frames_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = num_chunks * FRAMES_PER_CHUNK
    arr = (np.random.RandomState(0).rand(720, 1280, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    for i in range(1, n_frames + 1):
        p = out_dir / f"frame_{i:06d}.jpg"
        if not p.exists():
            img.save(p, quality=80)
    # Mark fps stamp (matches pass1a/extract_frames convention)
    (out_dir / ".fps").write_text("2")
    logger.info(f"  generated {n_frames} stub frames at {out_dir}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/test_rl/synthetic_trajectories.jsonl")
    ap.add_argument("--frames-root", default="",
                     help="If set, also write stub frames to this directory.")
    ap.add_argument("--include-long", action="store_true", default=True,
                     help="Include 320-chunk long video (slow stub frames).")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    builders = [
        make_direct,
        make_forward,
        make_backward_recall,
        make_multi_emit_counting,
        make_multi_emit_ssr,
        make_multi_card,
    ]
    if args.include_long:
        builders.append(make_long_forward)

    rows = [b() for b in builders]

    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows)} synthetic trajectories → {out}")
    for row in rows:
        n_q = len(row["questions"])
        nc = row["stats"]["n_chunks"]
        mechs = ", ".join({q["availability"] for q in row["questions"]})
        logger.info(f"  {row['video_id']:<20} chunks={nc:>4}, n_q={n_q}, mech=[{mechs}]")

    if args.frames_root:
        frames_root = Path(args.frames_root)
        for row in rows:
            _generate_stub_frames(frames_root, row["video_id"], row["stats"]["n_chunks"])
        logger.info(f"Stub frames generated under {frames_root}")


if __name__ == "__main__":
    main()
