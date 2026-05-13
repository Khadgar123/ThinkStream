#!/usr/bin/env python3
"""Estimate current ThinkStream recurrent-RL batch/sequence token budgets.

This intentionally uses the current pass5/RL prompt contract:
ordinary streaming turns carry only the current chunk's two frames.  It does
not trust legacy ``input.visual_window.frames`` counters, which may describe
old cumulative windows in raw pass4 records.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.agent_data.pass5_messages import PROJECT_ROOT, build_messages
from thinkstream.data.agent_protocol import tools_for_turn


VISUAL_TOKENS_PER_FRAME_RUNTIME = 235


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
            "n": 0,
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


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and "text" in item
        )
    return ""


def count_media_frames(messages: list[dict[str, Any]]) -> tuple[int, int, Counter[int]]:
    blocks = 0
    frames = 0
    hist: Counter[int] = Counter()
    for msg in messages:
        content = msg.get("content") or []
        if isinstance(content, str):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video" or item.get("video"):
                blocks += 1
                video = item.get("video")
                n = len(video) if isinstance(video, list) else int(item.get("nframes") or 0)
                frames += n
                hist[n] += 1
            elif item.get("type") == "image" or item.get("image") or item.get("image_url"):
                blocks += 1
                frames += 1
                hist[1] += 1
    return blocks, frames, hist


def prompt_len_estimate(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: Any,
    visual_tokens_per_frame: int,
) -> tuple[int, int, int, int, Counter[int]]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        tools=tools,
        add_generation_prompt=True,
    )
    video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    blocks, frames, hist = count_media_frames(messages)
    # The rendered template already has one <|video_pad|> per video block.
    # Processor expansion replaces that with many visual tokens.
    prompt_len = len(ids) + frames * visual_tokens_per_frame - ids.count(video_pad_id)
    return prompt_len, len(ids), frames, blocks, hist


def action_items_for_sample(messages: list[dict[str, Any]], sample: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], dict[str, Any], Any]]:
    sample_kind = str(sample.get("sample_type") or sample.get("action") or "")
    if (
        len(messages) >= 5
        and messages[2].get("role") == "assistant"
        and messages[3].get("role") == "user"
    ):
        return [
            ("recall_query", messages[:2], messages[2], tools_for_turn("streaming")),
            ("post_recall", messages[:4], messages[4], tools_for_turn("post_recall")),
        ]

    inter_chunk = sample_kind == "compress" or str(sample.get("action")) == "compress"
    kind = "compress" if inter_chunk else sample_kind
    return [
        (
            kind,
            messages[:-1],
            messages[-1],
            tools_for_turn("compress" if inter_chunk else "streaming"),
        )
    ]


def infer_batch_root(train_jsonl: Path) -> Path:
    parent = train_jsonl.parent
    if parent.name.startswith("final"):
        return parent.parent
    if parent.name.startswith("trajectory") and parent.parent.name == "rendered":
        return parent.parent.parent
    return parent


def scan_trajectories(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    action_counts: Counter[str] = Counter()
    frame_hist: Counter[int] = Counter()
    per_action: defaultdict[str, list[float]] = defaultdict(list)
    by_kind: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_traj: defaultdict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    n_traj = 0
    n_samples = 0

    for train_jsonl in args.train_jsonl:
        batch_root = infer_batch_root(train_jsonl)
        file_traj = 0
        file_samples = 0
        for line_no, traj in enumerate(iter_jsonl(train_jsonl), 1):
            if args.max_trajectories and n_traj >= args.max_trajectories:
                break
            n_traj += 1
            file_traj += 1
            traj_totals = Counter()
            for sample in traj.get("samples") or []:
                n_samples += 1
                file_samples += 1
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
                except Exception as exc:  # noqa: BLE001 - audit should keep going.
                    errors.append({
                        "path": str(train_jsonl),
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
                            visual_tokens_per_frame=args.visual_tokens_per_frame,
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors.append({
                            "path": str(train_jsonl),
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
                    dense_seq = prompt_est + args.response_pad_length

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
                        per_action[key].append(float(value))
                        by_kind[kind][key].append(float(value))
                        traj_totals[key] += float(value)
                    traj_totals["actions"] += 1.0

            for key, value in traj_totals.items():
                per_traj[key].append(float(value))

        per_file.append({
            "path": str(train_jsonl),
            "batch_root": str(batch_root),
            "trajectories": file_traj,
            "raw_samples": file_samples,
        })
        if args.max_trajectories and n_traj >= args.max_trajectories:
            break

    action_stats = {key: stats(values) for key, values in per_action.items()}
    traj_stats = {key: stats(values) for key, values in per_traj.items()}
    by_kind_stats = {
        kind: {key: stats(values) for key, values in values_by_metric.items()}
        for kind, values_by_metric in sorted(by_kind.items())
    }

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
        "per_action": action_stats,
        "per_action_by_kind": by_kind_stats,
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--response-pad-length", type=int, default=4096)
    parser.add_argument("--ppo-token-budget-per-gpu", type=int, default=65536)
    parser.add_argument("--visual-tokens-per-frame", type=int, default=VISUAL_TOKENS_PER_FRAME_RUNTIME)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()

    print(json.dumps(scan_trajectories(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
