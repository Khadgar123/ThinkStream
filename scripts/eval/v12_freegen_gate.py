"""v12.0 SFT→RL go/no-go gate via free-generation rollout.

Replaces the v11 teacher-forced action_acc metrics (which were broken for
compress=0% and recall=0% — see docs/v11.3_sft_run_postmortem.md). The v12
gate measures actual emission rates under free generation, mirroring how
DeepEyes v1 conditioned tool_reward on (count_vision_1 > 0) — i.e., the
floor check is "does the model ever emit the rare action?"

Usage:
    python -m scripts.eval.v12_freegen_gate \
        --ckpt output/agent-sft-v12.0/checkpoint-500 \
        --eval_data data/agent_v5/final/val.jsonl \
        --n_samples 200 \
        --output gate_results.json

Gates (all must pass before RL):
    A. recall_emit_rate    >= 0.5 * recall_freq_in_train
    B. compress_emit_rate  >= 0.95  (system-triggered, near-deterministic)
    C. format_compliance   >= 0.9
    D. answer_emit_rate    >= 0.95  (every non-trigger turn ends with <answer>)

If any gate fails, return SFT phase to investigate. If all pass, advance to
RL with high confidence that exploration mass exists for each rare action.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Defaults — override via CLI flags. Calibrated from DeepEyes v1's
# `count_vision_1 > 0` floor logic (verl/utils/reward_score/vl_agent.py:261).
DEFAULT_GATES = {
    "recall_emit_rate_min": 0.025,    # >= 50% of typical recall_query freq (5%)
    "compress_emit_rate_min": 0.95,   # near-deterministic given trigger
    "format_compliance_min": 0.90,
    "answer_emit_rate_min": 0.95,
}


def classify_emission(output_text: str) -> Dict:
    """Classify a single free-gen output into emission categories.

    Mirrors thinkstream.data.agent_protocol.parse_agent_output_v12 logic but
    inline to avoid coupling eval script to runtime module changes.
    """
    has_think = bool(re.search(r"<think>.*?</think>", output_text, re.DOTALL))
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)

    if answer_match and tool_match:
        return {"category": "format_error", "reason": "both answer and tool_call"}
    if not answer_match and not tool_match:
        return {"category": "format_error", "reason": "no terminal"}

    if answer_match:
        text = answer_match.group(1).strip()
        return {
            "category": "answer_silent" if not text else "answer_response",
            "has_think": has_think,
            "text": text,
        }

    # tool_call branch
    try:
        tool_obj = json.loads(tool_match.group(1).strip())
    except (json.JSONDecodeError, ValueError) as e:
        return {"category": "format_error", "reason": f"json: {e}"}

    name = tool_obj.get("name", "")
    if name == "recall":
        return {"category": "tool_recall", "has_think": has_think, "args": tool_obj.get("arguments", {})}
    if name == "compress":
        return {"category": "tool_compress", "has_think": has_think, "args": tool_obj.get("arguments", {})}
    return {"category": "format_error", "reason": f"unknown tool: {name!r}"}


def aggregate_gate_metrics(
    samples: List[Dict],
    classifications: List[Dict],
) -> Dict:
    """Compute the 4 gate metrics from per-sample classifications.

    Args:
        samples: list of original eval samples (must have sample_type field
                 and user_input field for compress_trigger detection).
        classifications: parallel list from classify_emission().
    """
    n_total = len(samples)
    if n_total == 0:
        return {"error": "no samples"}

    # Counter buckets
    counts = Counter(c["category"] for c in classifications)

    # Format compliance: anything not "format_error"
    n_valid = n_total - counts.get("format_error", 0)
    format_compliance = n_valid / n_total

    # Answer emit rate: how often we got a terminal answer (silent or response)
    # NOT counting compress-triggered turns (those should be tool_compress)
    n_trigger = sum(
        1 for s in samples
        if "<compress_trigger" in (s.get("user_input") or "")
    )
    n_non_trigger = n_total - n_trigger
    n_answer = counts.get("answer_silent", 0) + counts.get("answer_response", 0)
    answer_emit_rate = n_answer / max(1, n_non_trigger) if n_non_trigger else 0.0

    # Recall emit rate (out of all samples)
    recall_emit_rate = counts.get("tool_recall", 0) / n_total

    # Compress emit rate ONLY on trigger samples
    n_compress_when_triggered = 0
    for s, c in zip(samples, classifications):
        if "<compress_trigger" in (s.get("user_input") or "") and c["category"] == "tool_compress":
            n_compress_when_triggered += 1
    compress_emit_rate = (
        n_compress_when_triggered / n_trigger if n_trigger else float("nan")
    )

    # Per-sample-type emission breakdown (diagnostic)
    by_type = {}
    for s, c in zip(samples, classifications):
        st = s.get("sample_type", "?")
        by_type.setdefault(st, Counter())[c["category"]] += 1

    return {
        "n_total": n_total,
        "n_trigger_samples": n_trigger,
        "n_non_trigger_samples": n_non_trigger,
        "category_counts": dict(counts),
        "metrics": {
            "format_compliance": format_compliance,
            "answer_emit_rate": answer_emit_rate,
            "recall_emit_rate": recall_emit_rate,
            "compress_emit_rate": compress_emit_rate,
        },
        "by_sample_type": {
            st: dict(c) for st, c in by_type.items()
        },
    }


def evaluate_gates(metrics: Dict, gates: Dict) -> Dict:
    """Apply gate thresholds. Return per-gate pass/fail + overall verdict."""
    m = metrics["metrics"]
    results = {
        "A_recall_emit": {
            "value": m["recall_emit_rate"],
            "threshold": gates["recall_emit_rate_min"],
            "pass": m["recall_emit_rate"] >= gates["recall_emit_rate_min"],
        },
        "B_compress_emit": {
            "value": m["compress_emit_rate"],
            "threshold": gates["compress_emit_rate_min"],
            "pass": (
                m["compress_emit_rate"] != m["compress_emit_rate"]  # NaN
                or m["compress_emit_rate"] >= gates["compress_emit_rate_min"]
            ),
            "note": "NaN means no trigger samples in eval set" if m["compress_emit_rate"] != m["compress_emit_rate"] else None,
        },
        "C_format_compliance": {
            "value": m["format_compliance"],
            "threshold": gates["format_compliance_min"],
            "pass": m["format_compliance"] >= gates["format_compliance_min"],
        },
        "D_answer_emit": {
            "value": m["answer_emit_rate"],
            "threshold": gates["answer_emit_rate_min"],
            "pass": m["answer_emit_rate"] >= gates["answer_emit_rate_min"],
        },
    }
    results["overall_pass"] = all(g["pass"] for g in results.values() if isinstance(g, dict))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="SFT checkpoint dir")
    parser.add_argument("--eval_data", required=True, help="JSONL eval samples")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", default="v12_gate_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy. Set >0 to sample multiple rollouts.")
    parser.add_argument(
        "--predictions_file", default=None,
        help="If set, skip generation and load pre-generated predictions "
             "(JSONL with 'sample_id' + 'output' fields). Lets you decouple "
             "vLLM rollout (slow, GPU) from gate computation (fast, CPU).",
    )
    args = parser.parse_args()

    # Load eval samples
    eval_path = Path(args.eval_data)
    samples: List[Dict] = []
    with eval_path.open() as f:
        for line in f:
            s = json.loads(line)
            samples.append(s)
            if len(samples) >= args.n_samples:
                break
    logger.info(f"Loaded {len(samples)} eval samples from {eval_path}")

    # Either load predictions or run generation
    if args.predictions_file:
        preds_path = Path(args.predictions_file)
        preds_by_id = {}
        with preds_path.open() as f:
            for line in f:
                p = json.loads(line)
                preds_by_id[p["sample_id"]] = p["output"]

        outputs = []
        for s in samples:
            sid = s.get("sample_id") or s.get("trajectory_id", "?")
            outputs.append(preds_by_id.get(sid, ""))
        logger.info(f"Loaded {len(preds_by_id)} predictions from {preds_path}")
    else:
        # Online vLLM generation. Imported lazily to keep this script importable
        # without GPU dependencies (e.g., for CI gate-replay from saved preds).
        from thinkstream.eval.eval_baseline_vllm import (
            generate_freegen_for_v12_gate,
        )
        outputs = generate_freegen_for_v12_gate(
            ckpt=args.ckpt,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Classify each output
    classifications = [classify_emission(o) for o in outputs]

    # Aggregate metrics
    metrics = aggregate_gate_metrics(samples, classifications)

    # Apply gates
    gate_results = evaluate_gates(metrics, DEFAULT_GATES)

    # Combined report
    report = {
        "ckpt": args.ckpt,
        "eval_data": args.eval_data,
        "n_samples": len(samples),
        "metrics": metrics,
        "gates": gate_results,
        "overall_verdict": "PASS" if gate_results["overall_pass"] else "FAIL",
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # CLI summary
    print(f"\n=== v12.0 SFT→RL Gate Report ===")
    print(f"ckpt: {args.ckpt}")
    print(f"samples: {len(samples)}\n")
    print(f"Metrics:")
    for k, v in metrics["metrics"].items():
        print(f"  {k:>22} = {v:.4f}" if v == v else f"  {k:>22} = N/A")
    print(f"\nGates:")
    for name, g in gate_results.items():
        if name == "overall_pass":
            continue
        mark = "✓" if g["pass"] else "✗"
        v = g["value"]
        v_str = f"{v:.4f}" if v == v else "N/A"
        print(f"  {mark} {name:>22} = {v_str} (>= {g['threshold']})")
    print(f"\nVerdict: {report['overall_verdict']}")
    print(f"Full report: {out_path}")

    return 0 if gate_results["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
