#!/usr/bin/env python
"""Diagnose RT/BT eval errors: format vs timing vs content."""
import argparse
import json
from collections import defaultdict


def diagnose_sample(sample, bench_sample):
    """Classify a single prediction error.

    Categories:
    - silent: model output <action>silent</action> instead of response
    - no_action_tag: output lacks <action> tag entirely
    - no_response_tag: has <action>response</action> but no <response> payload
    - letter_mismatch: output a letter that doesn't match GT
    - text_mismatch: output text that matched wrong option
    - text_unmatch: output text that didn't match any option
    - correct: pred == gt
    """
    raw = sample.get("raw_output", "")
    parsed = sample.get("parsed_response", "")
    pred = sample.get("pred")
    gt = sample.get("gt")
    options = bench_sample.get("options", []) if bench_sample else []

    if pred == gt:
        return "correct"

    # Check for silent
    if "<action>silent</action>" in raw:
        return "silent"

    # Check for action tag
    if "<action>" not in raw:
        return "no_action_tag"

    # Has action tag but not response
    if "<action>response</action>" not in raw:
        # Could be recall or compress
        if "<action>recall</action>" in raw:
            return "recall_instead"
        if "<action>compress</action>" in raw:
            return "compress_instead"
        return "no_response_action"

    # Has response action but no response payload
    if "<response>" not in raw:
        return "no_response_tag"

    # Has response payload
    if pred is None:
        return "text_unmatch"

    # Has pred letter but wrong
    if pred != gt:
        # Check if model output was a single letter
        if parsed.strip() and parsed.strip()[0].upper() in "ABCDE" and len(parsed.strip()) == 1:
            return "letter_mismatch"
        return "text_mismatch"

    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_json", help="Path to eval_sft_rtbt.py output JSON")
    p.add_argument("--benchmark_json", help="Original OVO benchmark JSON")
    args = p.parse_args()

    with open(args.result_json) as f:
        result = json.load(f)

    bench_map = {}
    if args.benchmark_json:
        with open(args.benchmark_json) as f:
            bench = json.load(f)
        bench_map = {(s.get("task"), s.get("id")): s for s in bench}

    # Overall stats
    total = len(result["samples"])
    correct = sum(1 for s in result["samples"] if s["correct"])
    print(f"Total: {total}  Correct: {correct}  Accuracy: {correct/total:.3f}")
    print()

    # Error diagnosis by category
    cats = defaultdict(lambda: {"count": 0, "examples": []})
    for s in result["samples"]:
        key = (s.get("task"), s.get("id"))
        bench_s = bench_map.get(key)
        cat = diagnose_sample(s, bench_s)
        cats[cat]["count"] += 1
        if len(cats[cat]["examples"]) < 3:
            cats[cat]["examples"].append(s)

    print("Error breakdown:")
    for cat in sorted(cats, key=lambda c: -cats[c]["count"]):
        info = cats[cat]
        print(f"  {cat:<20s}: {info['count']:4d} ({100*info['count']/total:5.1f}%)")
        for ex in info["examples"]:
            print(f"    - id={ex['id']} gt={ex['gt']} pred={ex['pred']!s:<4s} parsed={ex['parsed_response'][:60]}")
    print()

    # By task
    by_task = defaultdict(lambda: {"n": 0, "correct": 0, "cats": defaultdict(int)})
    for s in result["samples"]:
        key = (s.get("task"), s.get("id"))
        bench_s = bench_map.get(key)
        cat = diagnose_sample(s, bench_s)
        t = s["task"]
        by_task[t]["n"] += 1
        by_task[t]["correct"] += int(s["correct"])
        by_task[t]["cats"][cat] += 1

    print("Per-task breakdown:")
    for task in sorted(by_task):
        info = by_task[task]
        acc = info["correct"] / info["n"]
        print(f"  {task}: {info['correct']}/{info['n']} = {acc:.3f}")
        for cat, cnt in sorted(info["cats"].items(), key=lambda x: -x[1]):
            print(f"      {cat:<20s}: {cnt}")


if __name__ == "__main__":
    main()
