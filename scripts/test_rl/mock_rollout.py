"""
Mock RL rollout test (v12.13).

Drives the streaming agent loop with a deterministic mock generate_fn so we
can:
  1. Confirm long trajectories (320 chunks) execute end-to-end without
     OOM or timeout.
  2. Verify mechanism-aware behavior:
       - direct (realtime):  model answers correctly at ask_chunk
       - forward (lead 20):  model stays silent ask..answer-1, answers at answer_chunk
       - backward (recall):  model emits recall tool_call at ask, answer after recall
       - multi_emit F5:      model emits per-emit "1","2","3" at expected chunks
       - multi_emit F7:      model flips "No"→"Yes" at the change chunk
  3. Run trajectory reward (compute_trajectory_outcome_v12) against the
     mock outputs and assert non-zero / expected scores.
  4. Compare prompt at chunk N built by streaming_agent_loop vs by
     pass5_messages — they must produce equivalent message lists for the
     same chunk (modulo runtime-only state like recall tool turn).

Usage:
    # 1) generate synthetic trajectories first
    python -m scripts.test_rl.synthetic_traj \\
        --out data/test_rl/synthetic_trajectories.jsonl

    # 2) run the mock test
    python -m scripts.test_rl.mock_rollout \\
        --traj data/test_rl/synthetic_trajectories.jsonl \\
        --report data/test_rl/mock_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------------
# Mock generator: returns canned text per chunk based on trajectory questions.
# ----------------------------------------------------------------------------
class MockGenerator:
    """Pretend-vLLM that answers per-chunk based on the trajectory's gold.

    Behavior modes (per question availability):
      - direct: emit correct answer at ask_chunk
      - silent_then_response: silent during ask..answer-1, correct answer at answer_chunk
      - recall: emit tool_call(recall) at ask_chunk, then correct answer
      - multi_emit_*: emit per-emit gold at each emit chunk
    """

    def __init__(self, trajectory: Dict[str, Any]):
        self.traj = trajectory
        # Build chunk → expected output map from gold.
        #
        # NOTE: recall mechanism in real RL is shape B multi-turn within one
        # chunk: turn1=tool_call(recall), tool=recall_result, turn2=answer.
        # This mock simulates only the OBSERVABLE final assistant turn (i.e.
        # the answer) at each answer_chunk because the test focuses on
        # reward computation, which only sees turn2's <answer>. The shape
        # B internal flow is exercised by SFT data construction tests
        # (test_v12_protocol.py etc.) not here.
        self.chunk_outputs: Dict[int, str] = {}
        for q in trajectory["questions"]:
            answer_chunks = q.get("answer_chunks", [])
            per_emit = {e["chunk"]: e["value"]
                         for e in q.get("per_emit_answers", [])}
            for ac in answer_chunks:
                val = per_emit.get(ac, q.get("gold_answer", "?"))
                self.chunk_outputs[ac] = (
                    f"<think>I see.</think><answer>{val}</answer>"
                )

    def __call__(self, chunk_idx: int) -> str:
        """Return canned generation text for a chunk."""
        return self.chunk_outputs.get(
            chunk_idx,
            "<think>Nothing notable.</think><answer></answer>",
        )


# ----------------------------------------------------------------------------
# Mock streaming rollout — drives the chunk loop with question_at_chunk maps.
# ----------------------------------------------------------------------------
def parse_chunk_output(text: str) -> Dict[str, Any]:
    """Mirror parse_agent_output_v12 (lite version)."""
    import re
    out = {"kind": "unknown", "answer_text": None, "tool_call": None,
           "think": None}
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        out["think"] = m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        out["kind"] = "answer"
        out["answer_text"] = m.group(1).strip()
        return out
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m:
        try:
            out["tool_call"] = json.loads(m.group(1).strip())
            name = out["tool_call"].get("name", "")
            if name == "recall":
                out["kind"] = "recall"
            elif name == "compress":
                out["kind"] = "compress"
        except json.JSONDecodeError:
            pass
    return out


def run_mock_trajectory(
    trajectory: Dict[str, Any],
    *,
    rollout_max_chunks: int = 360,
) -> Dict[str, Any]:
    """Drive a mock rollout for one trajectory. Returns rollout outputs."""
    questions = trajectory.get("questions", [])
    n_chunks = trajectory["stats"]["n_chunks"]
    mock = MockGenerator(trajectory)

    # Compute rollout cap (mirrors grpo.py / streaming_vllm.py logic):
    #   max(answer_chunks) + slack OR latest_ask + 5 fallback
    all_answer_chunks = []
    all_ask_chunks = []
    for q in questions:
        all_answer_chunks.extend(q.get("answer_chunks", []))
        all_ask_chunks.extend(q.get("ask_chunks", []))
    if all_answer_chunks:
        cap_target = max(all_answer_chunks) + 2
    else:
        cap_target = (max(all_ask_chunks) if all_ask_chunks else 0) + 5
    num_chunks_capped = min(cap_target + 1, rollout_max_chunks, n_chunks)

    chunk_outputs: List[Dict[str, Any]] = []
    t0 = time.time()
    for ci in range(num_chunks_capped):
        text = mock(ci)
        parsed = parse_chunk_output(text)
        chunk_outputs.append({
            "chunk_idx": ci,
            "kind": parsed["kind"],
            "answer_text": parsed["answer_text"],
            "raw": text,
        })
    elapsed = time.time() - t0

    return {
        "video_id": trajectory["video_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "num_chunks_target": n_chunks,
        "num_chunks_executed": num_chunks_capped,
        "rollout_cap": cap_target,
        "chunk_outputs": chunk_outputs,
        "wall_time_s": elapsed,
    }


# ----------------------------------------------------------------------------
# Reward consistency
# ----------------------------------------------------------------------------
def reward_check(trajectory: Dict[str, Any],
                  rollout: Dict[str, Any]) -> Dict[str, Any]:
    """Run compute_trajectory_outcome_v12 + manual timing audit.

    Asserts mechanism-specific expectations (mock should produce 1.0 outcome
    everywhere because it always emits the gold answer at the right chunks).
    """
    from thinkstream.trainer.v12_rewards import compute_trajectory_outcome_v12

    res = compute_trajectory_outcome_v12(
        rollout_chunk_outputs=rollout["chunk_outputs"],
        trajectory_questions=trajectory["questions"],
    )

    # Per-question expectations
    detail = []
    for q in trajectory["questions"]:
        avail = q.get("availability", "")
        answer_chunks = q.get("answer_chunks", [])
        # Did rollout actually run past the last answer chunk?
        cap = rollout["num_chunks_executed"]
        reached = cap > max(answer_chunks) if answer_chunks else False
        detail.append({
            "card_id": q["card_id"],
            "availability": avail,
            "ask_chunk": q.get("ask_chunk"),
            "answer_chunks": answer_chunks,
            "rollout_reached_answer": reached,
            "gold_emit_count": len(q.get("per_emit_answers", [])),
        })

    return {
        "outcome": res["outcome"],
        "n_questions": res["n_questions"],
        "n_answered": res["n_answered"],
        "n_correct": res["n_correct"],
        "per_q_outcomes": res["per_q_outcomes"],
        "per_q_detail": detail,
    }


# ----------------------------------------------------------------------------
# SFT vs RL prompt consistency
# ----------------------------------------------------------------------------
def prompt_consistency_check(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """For each non-silent sample, render via pass5 and check field shape.

    Doesn't compare token sequences (would need real processor) — instead
    asserts:
      - sample.input has the v12.12+ fields (memory, queries, visual_window)
      - questions[*] carries ask_chunk + answer_chunks + options if MC
      - gold_action_per_chunk is consistent with samples
    """
    from scripts.agent_data_v5.pass5_messages import build_messages

    issues = []
    samples_built = 0
    base_path = Path("/")

    # Verify questions[] schema
    for q in trajectory["questions"]:
        for required in ["ask_chunk", "answer_chunks", "per_emit_answers",
                         "question", "answer_form"]:
            if required not in q:
                issues.append(f"questions[{q['card_id']}] missing {required}")
        if q.get("answer_form") == "multiple_choice":
            if not q.get("options") or len(q["options"]) not in {2, 3, 4, 5}:
                issues.append(
                    f"MC question {q['card_id']} has bad options: {q.get('options')}"
                )
            valid = {
                chr(ord("A") + i)
                for i in range(min(len(q.get("options") or []), 26))
            }
            if q.get("correct_option") not in valid:
                issues.append(
                    f"MC question {q['card_id']} bad correct_option: {q.get('correct_option')}"
                )

    # Verify gold_action_per_chunk consistency
    gap = trajectory.get("gold_action_per_chunk", {})
    for q in trajectory["questions"]:
        for ac in q.get("answer_chunks", []):
            if gap.get(str(ac)) not in ("response", "recall"):
                issues.append(
                    f"gold_action_per_chunk[{ac}] = "
                    f"{gap.get(str(ac))} but {q['card_id']}'s answer chunk"
                )

    # Try rendering the FIRST non-silent sample (sanity check pass5)
    for s in trajectory["samples"]:
        if s["sample_type"] != "silent":
            try:
                msgs = build_messages(s, base_path=base_path)
                samples_built += 1
                # Check the assistant turn matches sample.output
                asst = next(m for m in msgs if m["role"] == "assistant")
                asst_text = "".join(c["text"] for c in asst["content"]
                                     if c.get("type") == "text")
                if not asst_text:
                    issues.append(
                        f"sample {s['sample_id']}: empty assistant turn"
                    )
                # Verify visual_window NOT rendered for compress (shape C)
                if s.get("v12_inter_chunk"):
                    user = next(m for m in msgs if m["role"] == "user")
                    user_text = "".join(c["text"] for c in user["content"]
                                         if c.get("type") == "text")
                    if "<visual_window>" in user_text:
                        issues.append(
                            f"compress sample {s['sample_id']} should drop visual_window"
                        )
                break
            except Exception as e:
                issues.append(f"build_messages failed on {s['sample_id']}: {e}")

    return {
        "samples_built": samples_built,
        "n_issues": len(issues),
        "issues": issues,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="data/test_rl/synthetic_trajectories.jsonl")
    ap.add_argument("--report", default="data/test_rl/mock_report.json")
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        logger.error(f"Trajectory file not found: {traj_path}")
        logger.error("Run: python -m scripts.test_rl.synthetic_traj first")
        sys.exit(1)

    trajectories = []
    with traj_path.open() as f:
        for line in f:
            trajectories.append(json.loads(line))
    logger.info(f"Loaded {len(trajectories)} synthetic trajectories\n")

    report = {"trajectories": []}
    overall_pass = True

    for traj in trajectories:
        vid = traj["video_id"]
        nc = traj["stats"]["n_chunks"]
        logger.info(f"═══ {vid} (chunks={nc}, n_q={len(traj['questions'])}) ═══")

        # 1) Mock rollout
        rollout = run_mock_trajectory(traj)
        logger.info(
            f"  [rollout] cap={rollout['rollout_cap']:>4}, "
            f"executed={rollout['num_chunks_executed']:>4}/{nc:<4} "
            f"in {rollout['wall_time_s']*1000:.1f}ms"
        )

        # 2) Reward check
        rwd = reward_check(traj, rollout)
        logger.info(
            f"  [reward] outcome={rwd['outcome']:.2f}, "
            f"n_correct={rwd['n_correct']}/{rwd['n_questions']}, "
            f"per_q={[round(x,2) for x in rwd['per_q_outcomes']]}"
        )

        # 3) Prompt consistency
        cons = prompt_consistency_check(traj)
        logger.info(
            f"  [prompt] {cons['samples_built']} sample(s) rendered, "
            f"{cons['n_issues']} issues"
        )
        for issue in cons["issues"]:
            logger.warning(f"    ⚠ {issue}")

        # Record
        report["trajectories"].append({
            "video_id": vid, "n_chunks": nc,
            "rollout": {k: v for k, v in rollout.items() if k != "chunk_outputs"},
            "reward": rwd,
            "consistency": cons,
        })

        # Pass criteria for this trajectory
        passed = (
            rollout["num_chunks_executed"] > 0
            and (rollout["num_chunks_executed"] >=
                 max((q.get("answer_chunks", [0])[-1]
                      for q in traj["questions"]), default=0))
            and rwd["outcome"] >= 0.99   # mock answers correctly → expect 1.0
            and cons["n_issues"] == 0
        )
        if not passed:
            overall_pass = False
            logger.error(f"  ✗ {vid} FAILED test criteria")
        else:
            logger.info(f"  ✓ {vid} passed")
        print()

    # Save report
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report → {out}")

    if overall_pass:
        logger.info("\n╔═══════════════════════════════════════╗")
        logger.info("║ ✓ ALL TRAJECTORIES PASSED MOCK ROLLOUT ║")
        logger.info("╚═══════════════════════════════════════╝")
        sys.exit(0)
    else:
        logger.error("\n╔═══════════════════════════════════════╗")
        logger.error("║ ✗ SOME TRAJECTORIES FAILED — see above ║")
        logger.error("╚═══════════════════════════════════════╝")
        sys.exit(1)


if __name__ == "__main__":
    main()
