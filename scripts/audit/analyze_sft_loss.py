#!/usr/bin/env python
"""L1: per-class SFT loss analyzer (offline).

Reads <run>/audit/sft_sample.jsonl (per-sample loss + metadata written by
WeightedSFTTrainer) and aggregates per (sample_type, sequence_type,
base_role) bucket. Use during training (in-progress) or after.

Why this exists:
    eval/loss is a single scalar averaged over the whole eval set.
    Per-class breakdown answers "is response class harder than silent?"
    "is warmup base undertrained?" — questions you can't see in wandb.

Usage:
    python scripts/audit/analyze_sft_loss.py --audit_dir output/agent-sft/audit
    python scripts/audit/analyze_sft_loss.py --audit_dir output/agent-sft/audit --last_steps 200
    python scripts/audit/analyze_sft_loss.py --audit_dir output/agent-sft/audit \\
        --group_by sample_type,base_role --csv loss_breakdown.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audit_dir", required=True, help="Path to <run>/audit/")
    p.add_argument(
        "--group_by",
        default="sample_type",
        help="Comma-separated keys from {sample_type, sequence_type, "
        "base_role, action}. Default: sample_type",
    )
    p.add_argument(
        "--last_steps",
        type=int,
        default=None,
        help="Only aggregate records from the last N optimizer steps "
        "(useful for in-progress monitoring — captures recent behavior "
        "rather than the whole trajectory).",
    )
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    args = p.parse_args()

    sample_path = Path(args.audit_dir) / "sft_sample.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"sft_sample.jsonl not found at {sample_path}")

    keys = [k.strip() for k in args.group_by.split(",") if k.strip()]

    # First pass for last_steps cutoff (avoid loading whole file twice if absent)
    cutoff = 0
    if args.last_steps:
        max_step = 0
        with open(sample_path) as f:
            for line in f:
                try:
                    max_step = max(max_step, json.loads(line).get("step", 0))
                except json.JSONDecodeError:
                    continue
        cutoff = max(0, max_step - args.last_steps)

    buckets: dict = defaultdict(list)
    weights_sum: dict = defaultdict(float)
    n_total = 0

    with open(sample_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("step", 0) < cutoff:
                continue
            n_total += 1
            key = tuple(r.get(k) or "_" for k in keys)
            buckets[key].append(float(r["loss"]))
            weights_sum[key] += float(r.get("weight", 1.0))

    if n_total == 0:
        print("No records found. Is training running / has it logged any samples yet?")
        return

    print(f"Audit dir : {args.audit_dir}")
    print(f"Records   : {n_total}" + (f" (last {args.last_steps} steps, ≥ step {cutoff})" if args.last_steps else ""))
    print(f"Group by  : {keys}")
    print()
    header_keys = "  ".join(f"{k:<20}" for k in keys)
    print(f"{header_keys}  {'n':>7}  {'mean':>8}  {'p50':>8}  {'p90':>8}  {'mean_w':>8}")
    print("-" * (24 * len(keys) + 50))

    rows = []
    for key in sorted(buckets.keys()):
        losses = sorted(buckets[key])
        n = len(losses)
        mean = sum(losses) / n
        p50 = losses[n // 2]
        p90 = losses[min(n - 1, int(n * 0.9))]
        mean_w = weights_sum[key] / n
        key_cells = "  ".join(f"{str(k):<20}" for k in key)
        print(f"{key_cells}  {n:>7}  {mean:>8.4f}  {p50:>8.4f}  {p90:>8.4f}  {mean_w:>8.3f}")
        rows.append((*key, n, mean, p50, p90, mean_w))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*keys, "n", "mean_loss", "p50_loss", "p90_loss", "mean_weight"])
            w.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
