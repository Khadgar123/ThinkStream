#!/usr/bin/env python3
"""Summarize verl RL metrics for long-running ThinkStream jobs.

The verl "file" logger writes JSONL records as {"step": int, "data": {...}}.
This monitor keeps a plotting-friendly 10-step snapshot while preserving the
raw logger file untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _flatten_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    step = record.get("step")
    data = record.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    out: Dict[str, Any] = {"step": step}
    has_val = any(str(k).startswith(("val-core/", "val-aux/")) for k in data)
    has_train = any(str(k).startswith(("training/", "actor/", "critic/", "reward/", "recurrent/")) for k in data)
    out["kind"] = "val" if has_val and not has_train else "train_val" if has_val else "train"
    for key, value in data.items():
        if _is_number(value):
            out[str(key)] = float(value)
    for key, value in list(out.items()):
        if "/outcome/" in key:
            out[key.replace("/outcome/", "/acc/")] = value
    return out


def _audit_summary(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {}
    reward_keys = [
        "score",
        "outcome",
        "answer_decision",
        "format",
        "n_questions",
        "n_questions_total",
        "n_questions_excluded_future",
        "n_answered",
        "n_answered_total",
        "horizon_chunk",
    ]
    out: Dict[str, Any] = {"audit/records": len(rows)}
    for key in reward_keys:
        vals: List[float] = []
        for row in rows:
            reward = row.get("reward") or {}
            if isinstance(reward, dict) and _is_number(reward.get(key)):
                vals.append(float(reward[key]))
        if vals:
            out[f"audit/{key}/mean"] = sum(vals) / len(vals)
            out[f"audit/{key}/min"] = min(vals)
            out[f"audit/{key}/max"] = max(vals)
    if "audit/outcome/mean" in out:
        out["audit/acc/mean"] = out["audit/outcome/mean"]
        out["audit/acc/min"] = out.get("audit/outcome/min")
        out["audit/acc/max"] = out.get("audit/outcome/max")
    completion_rates: List[float] = []
    for row in rows:
        reward = row.get("reward") or {}
        if not isinstance(reward, dict):
            continue
        n_answered = reward.get("n_answered")
        n_questions = reward.get("n_questions")
        if _is_number(n_answered) and _is_number(n_questions) and float(n_questions) > 0:
            completion_rates.append(float(n_answered) / float(n_questions))
    if completion_rates:
        out["audit/answer_completion_rate/mean"] = sum(completion_rates) / len(completion_rates)
        out["audit/answer_completion_rate/min"] = min(completion_rates)
        out["audit/answer_completion_rate/max"] = max(completion_rates)
    return out


def _select_rows(records: Iterable[Dict[str, Any]], interval_steps: int) -> List[Dict[str, Any]]:
    latest_by_step_kind: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for record in records:
        flat = _flatten_metrics(record)
        step_raw = flat.get("step")
        if not isinstance(step_raw, int):
            continue
        is_interval = interval_steps <= 1 or step_raw % interval_steps == 0
        is_val = str(flat.get("kind", "")).startswith("val") or "val" in str(flat.get("kind", ""))
        if not is_interval and not is_val:
            continue
        latest_by_step_kind[(step_raw, str(flat.get("kind", "train")))] = flat
    return [latest_by_step_kind[k] for k in sorted(latest_by_step_kind)]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    preferred = ["step", "kind", "time"]
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _append_gpu_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return
    ts = time.time()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time", "gpu", "memory_used_mib", "utilization_gpu_pct", "temperature_c", "power_w"])
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                writer.writerow([ts] + parts[:5])


def summarize_once(args: argparse.Namespace) -> int:
    rows = _select_rows(_read_jsonl(args.metrics_jsonl), args.interval_steps)
    audit = _audit_summary(args.audit_jsonl) if args.audit_jsonl else {}
    now = time.time()
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        merged = {"time": now, **row}
        merged.update(audit)
        enriched.append(merged)
    _write_jsonl(args.out_jsonl, enriched)
    _write_csv(args.out_csv, enriched)
    if args.gpu_csv:
        _append_gpu_csv(args.gpu_csv)
    return len(enriched)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path)
    parser.add_argument("--interval-steps", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    while True:
        summarize_once(args)
        if not args.follow:
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
