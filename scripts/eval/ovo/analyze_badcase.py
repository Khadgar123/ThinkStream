#!/usr/bin/env python
"""Deep badcase analysis for RT/BT single-step eval.

Analyzes:
- Chunk index distribution vs accuracy
- Action type distribution (response / silent / recall / compress / malformed)
- Badcase patterns: what did the model think vs what did it answer
- Option match quality scores
"""
import argparse
import json
import re
from collections import defaultdict, Counter


def get_action(raw_output):
    """Extract action type from raw model output."""
    if not raw_output:
        return "empty"
    m = re.search(r"<action>(.*?)</action>", raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return "no_action_tag"


def get_think(raw_output):
    """Extract think content."""
    m = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def get_response_text(raw_output):
    """Extract response payload."""
    m = re.search(r"<response>(.*?)</response>", raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_json")
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--top_n", type=int, default=10, help="Show top N badcases")
    p.add_argument("--task", default=None, help="Filter to one task")
    args = p.parse_args()

    with open(args.result_json) as f:
        result = json.load(f)

    with open(args.benchmark_json) as f:
        bench = json.load(f)
    bench_map = {(s["task"], s["id"]): s for s in bench}

    samples = result.get("samples", [])
    if args.task:
        samples = [s for s in samples if s["task"] == args.task]

    total = len(samples)
    if total == 0:
        print("No samples.")
        return

    correct = sum(1 for s in samples if s["correct"])
    print(f"Total: {total}  Correct: {correct}  Accuracy: {correct/total:.3f}")
    print()

    # ── Action type distribution ──
    actions = Counter()
    for s in samples:
        actions[get_action(s.get("raw_output", ""))] += 1
    print("Action distribution:")
    for act, cnt in actions.most_common():
        print(f"  {act:<20s}: {cnt:4d} ({100*cnt/total:.1f}%)")
    print()

    # ── Chunk index distribution ──
    chunk_buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for s in samples:
        chunk = s.get("ask_chunk", 0)
        # bucket by 25-chunk ranges
        bucket = f"{chunk//25*25}-{(chunk//25+1)*25}"
        chunk_buckets[bucket]["n"] += 1
        chunk_buckets[bucket]["correct"] += int(s["correct"])

    print("Accuracy by chunk index (time) buckets:")
    for bucket in sorted(chunk_buckets, key=lambda b: int(b.split("-")[0])):
        info = chunk_buckets[bucket]
        acc = info["correct"] / info["n"]
        print(f"  chunks {bucket:>10s}: {info['correct']:3d}/{info['n']:3d} = {acc:.3f}")
    print()

    # ── Badcase analysis ──
    badcases = [s for s in samples if not s["correct"]]
    print(f"Top {args.top_n} badcases (by option match quality):")
    for s in badcases[:args.top_n]:
        b = bench_map.get((s["task"], s["id"]))
        if not b:
            continue
        think = get_think(s.get("raw_output", ""))
        action = get_action(s.get("raw_output", ""))
        resp = get_response_text(s.get("raw_output", ""))
        print(f"  Task={s['task']} id={s['id']} chunk={s['ask_chunk']} action={action}")
        print(f"    Q: {b['question']}")
        print(f"    Options: {b['options']}")
        print(f"    GT: {b['options'][b['gt']]} ({s['gt']}) | Pred: {s['pred']}")
        print(f"    Response: {resp[:120]}")
        print(f"    Think: {think[:200]}...")
        print()

    # ── Correct case analysis (to see patterns) ──
    goodcases = [s for s in samples if s["correct"]]
    if goodcases:
        print(f"Top {min(args.top_n, len(goodcases))} correct cases:")
        for s in goodcases[:args.top_n]:
            b = bench_map.get((s["task"], s["id"]))
            if not b:
                continue
            think = get_think(s.get("raw_output", ""))
            resp = get_response_text(s.get("raw_output", ""))
            print(f"  Task={s['task']} id={s['id']} chunk={s['ask_chunk']}")
            print(f"    Q: {b['question']}")
            print(f"    GT: {b['options'][b['gt']]} | Response: {resp[:120]}")
            print(f"    Think: {think[:200]}...")
            print()


if __name__ == "__main__":
    main()
