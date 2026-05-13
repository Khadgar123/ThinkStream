#!/usr/bin/env python3
"""Parallel exact-ish RL token-budget scan over rendered batch trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit.rl_batch_sequence_budget import (
    VISUAL_TOKENS_PER_FRAME_RUNTIME,
    action_items_for_sample,
    content_text,
    infer_batch_root,
    iter_jsonl,
    prompt_len_estimate,
)
from scripts.agent_data.pass5_messages import PROJECT_ROOT, build_messages


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "n": float(len(values)),
        "min": float(min(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def scan_file(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    path_str, cfg = payload
    path = Path(path_str)
    batch_root = infer_batch_root(path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)

    action_counts: Counter[str] = Counter()
    frame_hist: Counter[int] = Counter()
    per_action: defaultdict[str, list[float]] = defaultdict(list)
    by_kind: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_traj: defaultdict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    n_traj = 0
    n_samples = 0

    for line_no, traj in enumerate(iter_jsonl(path), 1):
        n_traj += 1
        traj_totals = Counter()
        for sample in traj.get("samples") or []:
            n_samples += 1
            item = deepcopy(sample)
            item.setdefault("video_id", traj.get("video_id", ""))
            item.setdefault("video_path", traj.get("video_path", ""))
            item.setdefault("trajectory_id", traj.get("trajectory_id", ""))
            try:
                messages = build_messages(
                    item,
                    PROJECT_ROOT,
                    data_dir=batch_root,
                    frame_protocol="video_meta",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "path": path_str,
                    "line": line_no,
                    "chunk_idx": sample.get("chunk_idx"),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                })
                continue

            for kind, prompt_messages, response_message, tools in action_items_for_sample(messages, item):
                try:
                    prompt_est, prompt_raw, prompt_frames, prompt_blocks, hist = prompt_len_estimate(
                        tokenizer,
                        prompt_messages,
                        tools=tools,
                        visual_tokens_per_frame=cfg["visual_tokens_per_frame"],
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append({
                        "path": path_str,
                        "line": line_no,
                        "chunk_idx": sample.get("chunk_idx"),
                        "error": f"tokenize {type(exc).__name__}: {str(exc)[:200]}",
                    })
                    continue

                response_len = len(
                    tokenizer.encode(
                        content_text(response_message.get("content")),
                        add_special_tokens=False,
                    )
                )
                actual_seq = prompt_est + response_len
                dense_seq = prompt_est + cfg["response_pad_length"]

                action_counts[kind] += 1
                frame_hist.update(hist)
                metrics = {
                    "prompt_est": prompt_est,
                    "prompt_raw": prompt_raw,
                    "prompt_frames": prompt_frames,
                    "prompt_blocks": prompt_blocks,
                    "response_text": response_len,
                    "seq_actual_est": actual_seq,
                    "seq_dense_padded": dense_seq,
                }
                for key, value in metrics.items():
                    value_f = float(value)
                    per_action[key].append(value_f)
                    by_kind[kind][key].append(value_f)
                    traj_totals[key] += value_f
                traj_totals["actions"] += 1.0

        for key, value in traj_totals.items():
            per_traj[key].append(float(value))

    return {
        "path": path_str,
        "batch_root": str(batch_root),
        "trajectories": n_traj,
        "raw_samples": n_samples,
        "action_counts": dict(action_counts),
        "frame_hist": dict(frame_hist),
        "per_action": {key: values for key, values in per_action.items()},
        "per_action_by_kind": {
            kind: {key: values for key, values in values_by_metric.items()}
            for kind, values_by_metric in by_kind.items()
        },
        "per_trajectory_group": {key: values for key, values in per_traj.items()},
        "errors": errors,
    }


def merge_results(args: argparse.Namespace, parts: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    frame_hist: Counter[int] = Counter()
    per_action: defaultdict[str, list[float]] = defaultdict(list)
    by_kind: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_traj: defaultdict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    n_traj = 0
    n_samples = 0

    for part in sorted(parts, key=lambda item: item["path"]):
        per_file.append({
            "path": part["path"],
            "batch_root": part["batch_root"],
            "trajectories": part["trajectories"],
            "raw_samples": part["raw_samples"],
        })
        n_traj += int(part["trajectories"])
        n_samples += int(part["raw_samples"])
        action_counts.update(part["action_counts"])
        frame_hist.update({int(k): v for k, v in part["frame_hist"].items()})
        errors.extend(part["errors"])
        for key, values in part["per_action"].items():
            per_action[key].extend(values)
        for kind, values_by_metric in part["per_action_by_kind"].items():
            for key, values in values_by_metric.items():
                by_kind[kind][key].extend(values)
        for key, values in part["per_trajectory_group"].items():
            per_traj[key].extend(values)

    traj_stats = {key: stats(values) for key, values in per_traj.items()}
    groups_per_step = args.batch_size * args.group_size
    per_step = {
        "groups_per_step": groups_per_step,
        "actual_tokens": {
            key: value * groups_per_step
            for key, value in traj_stats.get("seq_actual_est", {}).items()
            if key != "n"
        },
        "dense_padded_tokens": {
            key: value * groups_per_step
            for key, value in traj_stats.get("seq_dense_padded", {}).items()
            if key != "n"
        },
        "action_rows": {
            key: value * groups_per_step
            for key, value in traj_stats.get("actions", {}).items()
            if key != "n"
        },
    }
    token_cap = max(args.ppo_token_budget_per_gpu, 1)
    per_gpu_mean_tokens = per_step["actual_tokens"].get("mean", 0.0) / max(args.world_size, 1)
    per_step["actor_microbatches_per_gpu_mean"] = math.ceil(per_gpu_mean_tokens / token_cap)
    per_step["actor_peak_valid_tokens_per_gpu_cap"] = token_cap

    return {
        "path": [str(path) for path in args.train_jsonl],
        "per_file": per_file,
        "model": str(args.model),
        "trajectories": n_traj,
        "raw_samples": n_samples,
        "action_counts": dict(action_counts),
        "frame_hist": dict(sorted(frame_hist.items())),
        "errors": errors[: args.max_errors],
        "assumptions": {
            "ordinary_streaming_frames_per_turn": 2,
            "visual_tokens_per_frame": args.visual_tokens_per_frame,
            "response_pad_length": args.response_pad_length,
            "batch_size_video_rows": args.batch_size,
            "group_size_rollouts_per_video": args.group_size,
            "world_size": args.world_size,
            "ppo_token_budget_per_gpu": args.ppo_token_budget_per_gpu,
        },
        "per_action": {key: stats(values) for key, values in per_action.items()},
        "per_action_by_kind": {
            kind: {key: stats(values) for key, values in values_by_metric.items()}
            for kind, values_by_metric in sorted(by_kind.items())
        },
        "per_trajectory_group": traj_stats,
        "per_train_step_BxG": per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model",
        default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--response-pad-length", type=int, default=4096)
    parser.add_argument("--ppo-token-budget-per-gpu", type=int, default=65536)
    parser.add_argument("--visual-tokens-per-frame", type=int, default=VISUAL_TOKENS_PER_FRAME_RUNTIME)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()

    cfg = {
        "model": str(args.model),
        "response_pad_length": args.response_pad_length,
        "visual_tokens_per_frame": args.visual_tokens_per_frame,
    }
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(scan_file, (str(path), cfg))
            for path in args.train_jsonl
        ]
        parts = [future.result() for future in as_completed(futures)]

    print(json.dumps(merge_results(args, parts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
