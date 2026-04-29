"""v12.0 streaming eval entry — OVO-Bench / VideoMME / our val.jsonl
via pluggable adapters.

Architecture:
    [JSON file]  →  Adapter  →  StreamingEvalRunner  →  Per-item score
       items        format     vLLM rollout +              + per-task
                    + score    chunk-by-chunk              aggregation
                    rules      decoding

Why a new entry script (not modify eval_baseline_vllm.py):
- eval_baseline_vllm.py is 1000+ lines of v11-specific logic
- v12 has different message format (apply_chat_template + tools=)
- v12 has different parsing (<answer>/<tool_call> vs <action>/<response>)
- Cleaner to fork rather than gate every line by protocol_version

Usage:
    python -m scripts.eval.v12_ovo_eval \
        --ckpt output/agent-sft-v12.0/checkpoint-500 \
        --dataset /Users/hzh/Downloads/ovo_bench_new.json \
        --dataset-format ovo_bench \
        --output ovo_results.json \
        --max-items 200

For our val:
    python -m scripts.eval.v12_ovo_eval \
        --ckpt ... \
        --dataset data/agent_v5/final/val.jsonl \
        --dataset-format our_val \
        --output our_val_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.adapters import get_adapter, _extract_answer_text


def load_dataset(path: Path, fmt: str) -> List[Dict]:
    """Load eval items. JSON list or JSONL depending on file extension."""
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    return json.loads(path.read_text())


def run_streaming_inference_for_item(
    *,
    item: Dict,
    adapter,
    model_engine,           # vLLM engine — caller responsibility
    processor,
    chunk_sec: float = 2.0,
    max_chunks: int = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> List[Dict]:
    """Run chunk-by-chunk streaming inference for ONE eval item.

    Stub implementation — caller plugs in the actual vLLM engine. The
    contract is: build user_content per chunk via the adapter, run
    apply_chat_template (with tools=TOOLS_SCHEMA for v12), decode
    one assistant turn, append to outputs.

    For OVO-Bench: chunk count derived from `realtime + late_window`.
    For our_val: chunk count from item.num_chunks.

    Returns list of {"chunk_idx": int, "text": str}.
    """
    raise NotImplementedError(
        "Plug in vLLM engine here. The current scaffold only handles"
        " adapter logic + scoring. See scripts/eval/eval_baseline_vllm.py"
        " for the v11 inference loop to fork."
    )


def score_offline_predictions(
    *,
    items: List[Dict],
    predictions: List[List[Dict]],
    adapter,
) -> List[Dict]:
    """Score N items given N pre-computed prediction lists (CPU-only path).

    Each predictions[i] is a list of {"chunk_idx", "text"} for item i.
    Useful for replaying generations from a JSONL file without re-running
    vLLM (e.g., CI gate, ablation comparisons).
    """
    if len(items) != len(predictions):
        raise ValueError(
            f"items ({len(items)}) and predictions ({len(predictions)}) "
            f"length mismatch"
        )
    results = []
    for item, preds in zip(items, predictions):
        result = adapter.score(item, preds)
        result["item_id"] = item.get("id")
        results.append(result)
    return results


def aggregate(results: List[Dict]) -> Dict:
    """Per-task and overall accuracy + delay distribution."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    correct = sum(1 for r in results if r.get("correct"))
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    delays = []
    fmt_counts: Counter = Counter()

    for r in results:
        task = r.get("task") or "_unknown"
        by_task[task]["total"] += 1
        if r.get("correct"):
            by_task[task]["correct"] += 1
        if r.get("delay_chunks") is not None:
            delays.append(r["delay_chunks"])
        if r.get("fmt"):
            fmt_counts[r["fmt"]] += 1

    delay_stats = {}
    if delays:
        delays.sort()
        delay_stats = {
            "p50": delays[len(delays) // 2],
            "p90": delays[int(len(delays) * 0.9)],
            "max": max(delays),
            "mean": sum(delays) / len(delays),
            "n_with_answer": len(delays),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "by_task": {
            t: {**c, "accuracy": c["correct"] / c["total"]}
            for t, c in by_task.items()
        },
        "delay_chunks": delay_stats,
        "format_dist": dict(fmt_counts),
        "no_answer_rate": fmt_counts.get("no_answer", 0) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="SFT checkpoint dir")
    parser.add_argument("--dataset", required=True, help="JSON or JSONL eval items")
    parser.add_argument(
        "--dataset-format", required=True,
        choices=["ovo_bench", "our_val", "mc_no_timing"],
    )
    parser.add_argument("--output", default="v12_eval_results.json")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--predictions-file", default=None,
        help="JSONL with per-item predictions (skip vLLM, score-only). "
             "Format: one line per item, JSON array of {'chunk_idx','text'}.",
    )
    args = parser.parse_args()

    # Load
    items = load_dataset(Path(args.dataset), args.dataset_format)
    if args.max_items:
        items = items[: args.max_items]
    logger.info(f"Loaded {len(items)} items from {args.dataset}")

    adapter = get_adapter(args.dataset_format)

    # Predictions: either offline-replay or online vLLM
    if args.predictions_file:
        preds_path = Path(args.predictions_file)
        with preds_path.open() as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        if len(predictions) != len(items):
            raise ValueError(
                f"Predictions file has {len(predictions)} items but "
                f"dataset has {len(items)}. Trim --max-items to match."
            )
        results = score_offline_predictions(
            items=items, predictions=predictions, adapter=adapter,
        )
    else:
        # Lazy-import vLLM only when needed (keeps the script CPU-importable
        # for unit tests of adapters/scoring).
        raise NotImplementedError(
            "Online vLLM inference path requires the streaming engine "
            "(eval_baseline_vllm.py infrastructure). For now, generate "
            "predictions externally and pass --predictions-file. "
            "The adapter + scoring layers are CI-tested and ready to "
            "consume offline predictions."
        )

    # Aggregate + write
    summary = aggregate(results)
    out = {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "dataset_format": args.dataset_format,
        "n_items": len(items),
        "summary": summary,
        "per_item": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\n=== v12 OVO-Bench Eval ===")
    print(f"  ckpt:     {args.ckpt}")
    print(f"  dataset:  {args.dataset} ({args.dataset_format})")
    print(f"  total:    {summary['total']}")
    print(f"  accuracy: {summary['accuracy']:.4f}")
    if summary.get("by_task"):
        print(f"  per task:")
        for t, c in sorted(summary["by_task"].items(), key=lambda x: -x[1]["total"]):
            print(f"    {t:>5}: {c['correct']:>4}/{c['total']:>4} = {c['accuracy']:.4f}")
    if summary.get("delay_chunks"):
        d = summary["delay_chunks"]
        print(f"  delay (chunks):  p50={d['p50']} p90={d['p90']} max={d['max']}")
    print(f"  no_answer_rate: {summary['no_answer_rate']:.4f}")
    print(f"  format_dist: {summary['format_dist']}")
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
