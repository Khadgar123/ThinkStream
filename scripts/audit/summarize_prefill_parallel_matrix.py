#!/usr/bin/env python3
"""Summarize prefill memory matrix probe outputs."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"WARN failed to read {path}: {exc}", file=sys.stderr)
        return None


def summarize_time_range(root: Path) -> None:
    rows = []
    for path in sorted((root / "time_range").glob("*/*/summary.json")):
        data = load_json(path)
        if not data:
            continue
        config = path.parents[1].name
        query = path.parent.name
        rows.append({
            "config": config,
            "query": query,
            "hit": bool(data.get("hit")),
            "iou": float(data.get("iou", 0.0)),
            "predicted": data.get("predicted_range"),
            "expected": data.get("expected_range"),
            "answer": data.get("answer", "").replace("\n", " ")[:180],
            "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
        })

    print("TIME_RANGE_AGG")
    if not rows:
        print("no time_range summaries")
        return
    by_config = defaultdict(list)
    by_query = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
        by_query[row["query"]].append(row)
    for config, items in sorted(by_config.items()):
        hit_rate = sum(item["hit"] for item in items) / len(items)
        mean_iou = sum(item["iou"] for item in items) / len(items)
        print(f"{config}\tn={len(items)}\thit={hit_rate:.3f}\tmean_iou={mean_iou:.3f}")
    print("")
    print("TIME_RANGE_BY_QUERY")
    for query, items in sorted(by_query.items()):
        hit_rate = sum(item["hit"] for item in items) / len(items)
        mean_iou = sum(item["iou"] for item in items) / len(items)
        print(f"{query}\tn={len(items)}\thit={hit_rate:.3f}\tmean_iou={mean_iou:.3f}")
    print("")
    print("TIME_RANGE_CASES")
    for row in rows:
        print(
            f"{row['config']}\t{row['query']}\thit={int(row['hit'])}\tiou={row['iou']:.3f}"
            f"\tpred={row['predicted']}\texp={row['expected']}\tprompt={row['prompt_tokens']}\t{row['answer']}"
        )


def summarize_continue(root: Path) -> None:
    rows = []
    for path in sorted((root / "continue").glob("*/summary.json")):
        data = load_json(path)
        if not data:
            continue
        rows.append({
            "config": path.parent.name,
            "turns": int(data.get("turns", 0)),
            "avg_current": float(data.get("avg_current_keyword_recall", 0.0)),
            "avg_previous": float(data.get("avg_previous_keyword_recall", 0.0)),
            "old_copy": int(data.get("old_copy_risk_count", 0)),
            "unique": int(data.get("unique_pred_count", 0)),
            "top_repeat": int(data.get("top_repeat_count", 0)),
            "last_prompt_tokens": ((data.get("results") or [{}])[-1].get("usage") or {}).get("prompt_tokens") if data.get("results") else None,
        })

    print("")
    print("CONTINUE_AGG")
    if not rows:
        print("no continue summaries")
        return
    for row in sorted(rows, key=lambda r: (-r["avg_current"], r["old_copy"], -r["unique"])):
        print(
            f"{row['config']}\tturns={row['turns']}\tcur={row['avg_current']:.3f}"
            f"\tprev={row['avg_previous']:.3f}\told_copy={row['old_copy']}"
            f"\tunique={row['unique']}\ttop_repeat={row['top_repeat']}\tprompt={row['last_prompt_tokens']}"
        )


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/vllm_latest_frame_probe/prefill_matrix_8gpu")
    summarize_time_range(root)
    summarize_continue(root)


if __name__ == "__main__":
    main()
