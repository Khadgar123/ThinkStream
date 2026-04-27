#!/usr/bin/env python
"""Monitor and aggregate parallel RT/BT eval results in real-time."""
import glob
import json
import os
import time
import sys
from pathlib import Path


def parse_log(log_path):
    """Extract progress and accuracy from a log file."""
    total = None
    done = 0
    last_rate = 0
    task_accs = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if "Evaluating" in line and "samples" in line:
                # Evaluating 297 RT/BT samples across tasks: ['EPM']
                try:
                    total = int(line.split("Evaluating ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "samples/min" in line:
                done = int(line.split("/")[0].strip("[]"))
                try:
                    last_rate = float(line.split()[-2])
                except (IndexError, ValueError):
                    pass
            if line.startswith("  ") and ":" in line and "=" in line:
                #   EPM: 44/297 = 0.148
                try:
                    task = line.split(":")[0].strip()
                    parts = line.split("=")
                    acc = float(parts[-1].strip())
                    task_accs[task] = acc
                except (IndexError, ValueError):
                    pass
    return total, done, last_rate, task_accs


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "output/agent-sft/checkpoint-100/eval/ovo_rtbt_parallel"
    log_files = sorted(glob.glob(f"{base_dir}/*.log"))

    if not log_files:
        print(f"No log files found in {base_dir}")
        return

    total_samples = 0
    total_done = 0
    all_task_accs = {}

    print(f"{'GPU':<20} {'Total':>6} {'Done':>6} {'%':>6} {'Rate':>12} {'Tasks'}")
    print("-" * 70)
    for log_path in log_files:
        name = Path(log_path).stem
        total, done, rate, task_accs = parse_log(log_path)
        total_samples += total or 0
        total_done += done
        for task, acc in task_accs.items():
            all_task_accs[task] = acc
        status = f"{name:<20} {total or '?':>6} {done:>6} {100*done//max(1,total or 1):>5}% {rate:>10.1f}/min {','.join(task_accs.keys())}"
        print(status)

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_samples:>6} {total_done:>6} {100*total_done//max(1,total_samples):>5}%")
    if all_task_accs:
        print("\nCompleted tasks:")
        for task, acc in sorted(all_task_accs.items()):
            print(f"  {task}: {acc:.3f}")


if __name__ == "__main__":
    main()
