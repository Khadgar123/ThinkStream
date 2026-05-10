#!/usr/bin/env python
"""Flatten an OVO base-eval merged JSON into summaries and review files."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _load_benchmark(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return {}
    if text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {str(row.get("id")): row for row in rows}


def _det_key(*parts):
    h = hashlib.md5("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h


def _reason(probe):
    if probe.get("correct"):
        return "correct"
    raw = str(probe.get("raw") or "").strip()
    pred = probe.get("pred")
    if pred is None or pred == "":
        return "no_parse" if raw else "empty_output"
    return "wrong_answer"


def _probe_context(sample_meta, probe_idx):
    if not sample_meta:
        return {}
    out = {
        "question": sample_meta.get("question"),
        "options": sample_meta.get("options"),
        "activity": sample_meta.get("activity"),
    }
    test_info = sample_meta.get("test_info")
    if isinstance(test_info, list) and 0 <= probe_idx < len(test_info):
        probe_meta = test_info[probe_idx]
        if isinstance(probe_meta, dict):
            out["probe_meta"] = {
                k: probe_meta.get(k)
                for k in ("realtime", "type", "count", "step")
                if k in probe_meta
            }
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_json")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--benchmark_json", default=None)
    ap.add_argument("--samples_per_bucket", type=int, default=8)
    ap.add_argument("--raw_chars", type=int, default=800)
    args = ap.parse_args()

    merged_path = Path(args.merged_json)
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_dir = Path(args.out_dir) if args.out_dir else merged_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {}
    shard_configs = data.get("shard_configs") or []
    if shard_configs:
        cfg = dict(shard_configs[0])
    benchmark_json = args.benchmark_json or cfg.get("benchmark_json")
    benchmark = _load_benchmark(benchmark_json)

    run_name = out_dir.name
    flat = []
    for sample in data.get("samples", []):
        sid = str(sample.get("id"))
        meta = benchmark.get(sid, {})
        for probe_idx, probe in enumerate(sample.get("probes") or []):
            row = {
                "run": run_name,
                "ckpt": cfg.get("ckpt"),
                "mode": cfg.get("mode"),
                "max_frames": cfg.get("max_frames"),
                "fps": cfg.get("fps"),
                "task": sample.get("task"),
                "id": sid,
                "probe_idx": probe_idx,
                "realtime": probe.get("realtime"),
                "gt": probe.get("gt"),
                "pred": probe.get("pred"),
                "correct": bool(probe.get("correct")),
                "reason": _reason(probe),
                "raw": str(probe.get("raw") or "")[: args.raw_chars],
            }
            row.update(_probe_context(meta, probe_idx))
            flat.append(row)

    summary = data.get("summary", {})
    compact = {
        "run": run_name,
        "ckpt": cfg.get("ckpt"),
        "mode": cfg.get("mode"),
        "max_frames": cfg.get("max_frames"),
        "fps": cfg.get("fps"),
        "visual_window_sec": cfg.get("visual_window_sec"),
        "n_samples": data.get("n_samples"),
        "n_probes": len(flat),
        "overall": summary.get("overall"),
        "overall_strict": summary.get("overall_strict"),
        "overall_targeted": summary.get("overall_targeted"),
        "category": summary.get("category"),
        "per_task": summary.get("per_task"),
    }
    with open(out_dir / "summary_compact.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    with open(out_dir / "probes.jsonl", "w", encoding="utf-8") as f:
        for row in flat:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bad = [row for row in flat if not row["correct"]]
    with open(out_dir / "bad_cases.jsonl", "w", encoding="utf-8") as f:
        for row in bad:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    buckets = defaultdict(list)
    for row in flat:
        buckets[(row["task"], row["correct"])].append(row)
    sampled = []
    for key, rows in sorted(buckets.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        rows = sorted(
            rows,
            key=lambda r: _det_key(run_name, r["task"], r["id"], r["probe_idx"]),
        )
        sampled.extend(rows[: args.samples_per_bucket])
    with open(out_dir / "sample_cases.jsonl", "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"{run_name}\toverall={compact['overall']}\t"
        f"probes={len(flat)}\tbad={len(bad)}"
    )


if __name__ == "__main__":
    main()
