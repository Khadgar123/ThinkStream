#!/usr/bin/env python
"""Aggregate multiple parallel eval_full.py outputs into a single report.

Deduplicates by (task, sample_id) so overlapping parallel runs are safe.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_all(json_paths):
    all_results = []
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        all_results.extend(data.get("samples", []))
    return all_results


def deduplicate(results):
    seen = set()
    out = []
    for r in results:
        key = (r.get("task"), r.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def aggregate(results):
    by_task = defaultdict(lambda: {"n": 0, "correct": 0,
                                    "fp_n": 0, "fp": 0,
                                    "type0_n": 0, "type0_strict": 0, "type0_lenient": 0,
                                    "type1_n": 0, "type1_strict": 0, "type1_lenient": 0})
    for r in results:
        task = r["task"]
        for p in r["probes"]:
            by_task[task]["n"] += 1
            if "correct" in p:
                by_task[task]["correct"] += int(p["correct"])
            else:
                by_task[task]["correct"] += int(p["strict_correct"])
                if p.get("type") == 0:
                    by_task[task]["type0_n"] += 1
                    by_task[task]["type0_strict"] += int(p["strict_correct"])
                    by_task[task]["type0_lenient"] += int(p["lenient_correct"])
                    by_task[task]["fp"] += int(p.get("false_positive", False))
                    by_task[task]["fp_n"] += 1
                elif p.get("type") == 1:
                    by_task[task]["type1_n"] += 1
                    by_task[task]["type1_strict"] += int(p["strict_correct"])
                    by_task[task]["type1_lenient"] += int(p["lenient_correct"])

    def cat_avg(tasks):
        accs = []
        for t in tasks:
            v = by_task.get(t)
            if v and v["n"] > 0:
                accs.append(v["correct"] / v["n"])
        return sum(accs) / len(accs) if accs else 0.0, len(accs)

    RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
    BT_TASKS = {"EPM", "ASI", "HLD"}
    FT_TASKS = {"REC", "SSR", "CRR"}

    rt_avg, rt_n = cat_avg(RT_TASKS)
    bt_avg, bt_n = cat_avg(BT_TASKS)
    ft_avg, ft_n = cat_avg(FT_TASKS)
    overall = (rt_avg + bt_avg + ft_avg) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))

    return {
        "by_task": dict(by_task),
        "category": {
            "RT": {"avg": rt_avg, "n_tasks": rt_n},
            "BT": {"avg": bt_avg, "n_tasks": bt_n},
            "FT": {"avg": ft_avg, "n_tasks": ft_n},
        },
        "overall": overall,
    }


def print_report(agg):
    print()
    print(f"{'task':<8}  {'n':>6}  {'acc':>7}  {'fp_rate':>9}  notes")
    print("-" * 70)
    for task in sorted(agg["by_task"]):
        v = agg["by_task"][task]
        if v["n"] == 0:
            continue
        acc = v["correct"] / v["n"]
        fp = (v["fp"] / v["fp_n"]) if v["fp_n"] > 0 else 0.0
        notes = ""
        if task in ("CRR", "SSR"):
            t0 = v["type0_n"]
            t1 = v["type1_n"]
            t0a = (v["type0_strict"] / t0) if t0 else 0.0
            t1a = (v["type1_strict"] / t1) if t1 else 0.0
            notes = f"t0_strict={t0a:.3f}({t0}) t1_strict={t1a:.3f}({t1})"
        print(f"{task:<8}  {v['n']:>6}  {acc:>7.3f}  {fp:>9.3f}  {notes}")

    print()
    print(f"Real-Time Visual Perception (RT): {agg['category']['RT']['avg']:.3f} "
          f"({agg['category']['RT']['n_tasks']} tasks)")
    print(f"Backward Tracing (BT):            {agg['category']['BT']['avg']:.3f} "
          f"({agg['category']['BT']['n_tasks']} tasks)")
    print(f"Forward Active Responding (FT):   {agg['category']['FT']['avg']:.3f} "
          f"({agg['category']['FT']['n_tasks']} tasks)")
    print(f"OVERALL (mean of categories):     {agg['overall']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsons", nargs="+", help="Parallel eval output JSONs")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = load_all(args.jsons)
    results = deduplicate(results)
    agg = aggregate(results)
    print_report(agg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_sources": len(args.jsons),
            "n_samples": len(results),
            "summary": agg,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
