#!/usr/bin/env python
"""Merge sharded outputs from scripts/eval/ovo/base.py."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def aggregate(results):
    by_task = defaultdict(lambda: {
        "n": 0,
        "correct": 0,
        "strict_correct": 0,
        "targeted_correct": 0,
    })
    for result in results:
        for probe in result.get("probes", []):
            by_task[result["task"]]["n"] += 1
            if "correct" in probe:
                correct = bool(probe.get("correct"))
            elif "strict_correct" in probe:
                correct = bool(probe.get("strict_correct"))
            else:
                correct = False
            strict = bool(probe.get("strict_correct", correct))
            targeted = bool(probe.get("targeted_correct", correct))
            by_task[result["task"]]["correct"] += int(correct)
            by_task[result["task"]]["strict_correct"] += int(strict)
            by_task[result["task"]]["targeted_correct"] += int(targeted)

    per_task = {}
    for task, value in by_task.items():
        acc = value["correct"] / max(value["n"], 1)
        per_task[task] = {
            "n": value["n"],
            "acc": acc,
            "strict_acc": value["strict_correct"] / max(value["n"], 1),
            "targeted_acc": value["targeted_correct"] / max(value["n"], 1),
        }

    rt_tasks = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
    bt_tasks = {"EPM", "ASI", "HLD"}
    ft_tasks = {"REC", "SSR", "CRR"}

    def cat_avg(tasks, key="acc"):
        accs = [per_task[t][key] for t in tasks if t in per_task]
        return {"avg": sum(accs) / max(len(accs), 1), "n_tasks": len(accs)}

    rt = cat_avg(rt_tasks)
    bt = cat_avg(bt_tasks)
    ft = cat_avg(ft_tasks)
    rt_strict = cat_avg(rt_tasks, "strict_acc")
    bt_strict = cat_avg(bt_tasks, "strict_acc")
    ft_strict = cat_avg(ft_tasks, "strict_acc")
    rt_targeted = cat_avg(rt_tasks, "targeted_acc")
    bt_targeted = cat_avg(bt_tasks, "targeted_acc")
    ft_targeted = cat_avg(ft_tasks, "targeted_acc")
    active = [v for v in (rt, bt, ft) if v["n_tasks"] > 0]
    overall = sum(v["avg"] for v in active) / max(len(active), 1)
    active_strict = [v for v in (rt_strict, bt_strict, ft_strict) if v["n_tasks"] > 0]
    active_targeted = [
        v for v in (rt_targeted, bt_targeted, ft_targeted)
        if v["n_tasks"] > 0
    ]
    return {
        "per_task": dict(sorted(per_task.items())),
        "category": {
            "RT": {**rt, "strict_acc": rt_strict["avg"], "targeted_acc": rt_targeted["avg"]},
            "BT": {**bt, "strict_acc": bt_strict["avg"], "targeted_acc": bt_targeted["avg"]},
            "FT": {**ft, "strict_acc": ft_strict["avg"], "targeted_acc": ft_targeted["avg"]},
        },
        "overall": overall,
        "overall_strict": (
            sum(v["avg"] for v in active_strict) / max(len(active_strict), 1)
        ),
        "overall_targeted": (
            sum(v["avg"] for v in active_targeted) / max(len(active_targeted), 1)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    samples = []
    configs = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        configs.append({k: obj.get(k) for k in (
            "ckpt", "mode", "benchmark_json", "video_root", "frames_root",
            "max_frames", "visual_window_sec", "fps", "scoring", "profile",
            "frame_protocol", "num_shards", "shard_index",
        )})
        samples.extend(obj.get("samples", []))

    summary = aggregate(samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "merged_from": args.inputs,
            "shard_configs": configs,
            "n_samples": len(samples),
            "summary": summary,
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)

    print(f"merged {len(args.inputs)} shards, {len(samples)} samples -> {out}")
    print(
        f"overall={summary['overall']:.4f} "
        f"strict={summary.get('overall_strict', 0.0):.4f} "
        f"target={summary.get('overall_targeted', 0.0):.4f}"
    )
    for task, value in summary["per_task"].items():
        print(
            f"{task}: n={value['n']} acc={value['acc']:.4f} "
            f"strict={value.get('strict_acc', 0.0):.4f} "
            f"target={value.get('targeted_acc', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
