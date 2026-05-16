#!/usr/bin/env python
"""Batch-vLLM DAgger rollout for ThinkStream step-2 SFT data.

This script reads full-video trajectory rows, rolls out the current student
policy in chunk lockstep, then labels the student-visited states with the
existing teacher action/output at the same video chunk. It writes ShareGPT
messages compatible with the normal SFT loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.agent_data.pass5_messages import (
    _accepted_answers_for_question,
    _mc_letter_text,
    _mc_target_for_question,
    _normalise_assistant_output,
    build_sft_rows,
)
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    build_recall_result_user_content,
    canonical_answer_instruction,
    normalize_frame_protocol,
    normalize_render_layout,
    query_is_complete,
)
from thinkstream.eval.streaming_vllm import streaming_vllm_rollout
from thinkstream.eval.vllm_engine import init_vllm_engine
from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS
from thinkstream.sft.args import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


DEFAULT_SCHEME = (
    Path("data/agent_v5")
    / "batch12345678_bank_scheme_bal_sft1120_dagger720_rl1360_val240_test240_v1259"
)


def _dataset_info_entry(file_name: str) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "videos": "videos"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _merge_dataset_info(out_dir: Path, split_stem: str) -> None:
    path = out_dir / "dataset_info.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[f"thinkstream_{split_stem}"] = _dataset_info_entry(
        f"{split_stem}_messages.jsonl"
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path, limit: int = 0) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _select_shard(
    rows: List[Dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def _trajectory_for_rollout(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Use video_id.mp4 so streaming_vllm resolves frames_root/<video_id>/."""
    out = deepcopy(traj)
    vid = str(out.get("video_id") or out.get("trajectory_id") or "")
    out["_original_video_path"] = out.get("video_path", "")
    out["video_path"] = f"{vid}.mp4" if vid else str(out.get("video_path", ""))
    out["data_path"] = ""
    return out


def _question_map(traj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        q.get("card_id"): q
        for q in traj.get("questions", [])
        if isinstance(q, dict) and q.get("card_id")
    }


def _patch_sample_from_question(
    sample: Dict[str, Any],
    q_by_card: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sample = deepcopy(sample)
    card_id = sample.get("card_id") or (sample.get("metadata") or {}).get("card_id")
    q = q_by_card.get(card_id)
    if not q:
        return sample

    sample["_trajectory_question"] = q
    meta = dict(sample.get("metadata") or {})
    for key in (
        "card_id",
        "question",
        "options",
        "correct_option",
        "answer_form",
        "answer_style",
        "answer_instruction",
        "question_type",
        "family",
        "family_name",
        "category",
        "skill",
        "ours_unique",
        "availability",
        "support_chunks",
        "gold_compress_chunks",
        "ask_chunk",
        "per_emit_answers",
    ):
        if key in q:
            value = q.get(key)
            meta[key] = list(value) if isinstance(value, list) else value
    if q.get("answer_form") == "multiple_choice":
        meta["answer_style"] = "letter_plus_text"
    instruction = canonical_answer_instruction(meta)
    if instruction:
        meta["answer_instruction"] = instruction
    if q.get("answer_form") == "multiple_choice":
        _letter, correct_text = _mc_letter_text(q)
        meta["correct_answer_text"] = correct_text
        meta["canonical_answer"] = correct_text or q.get("canonical_answer", "")
        meta["gold_answer"] = correct_text or q.get("gold_answer", "")
        meta["accepted_answers"] = _accepted_answers_for_question(q)
        target = _mc_target_for_question(q, sample.get("chunk_idx"))
        if target:
            meta["sft_answer"] = target
    sample["metadata"] = meta
    return sample


def _user_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
    return "".join(parts)


def _is_compress_prompt(messages: List[Dict[str, Any]]) -> bool:
    return "<compress_trigger" in _user_text(messages)


def _is_compress_sample(sample: Dict[str, Any]) -> bool:
    return bool(sample.get("inter_chunk")) or sample.get("action") == "compress"


def _query_is_open_local(q: Dict[str, Any]) -> bool:
    status = str(q.get("status", "")).strip().lower()
    if status in {"open", "pending", "active"}:
        return True
    if status in {"answered", "closed", "done", "replaced"}:
        return False
    if not q.get("answers"):
        return True
    return not query_is_complete(q)


def _compress_failure_reason(
    student_action: str,
    prefix_diag: Dict[str, Any],
    action_space_error: str,
) -> str:
    label = str(prefix_diag.get("label") or "")
    if student_action == "compress":
        return "ok"
    if label == "good_prefix_likely_truncated":
        return "length_cap_or_truncated"
    if label in {"good_json_missing_tool_close", "missing_tool_close", "text_not_closed"}:
        return "parse_incomplete_or_missing_close"
    if label.startswith("json_"):
        return "json_parse_error"
    if label in {
        "no_tool_call",
        "tool_open_no_json",
        "json_no_compress_name",
        "compress_name_no_arguments",
        "arguments_no_time_range_pair",
        "time_range_no_text_key",
    }:
        return f"format_error:{label}"
    if action_space_error:
        return f"action_space_error:{action_space_error}"
    if student_action in {"response", "recall", "silent", "unknown", ""}:
        return f"wrong_action:{student_action or 'empty'}"
    return f"invalid_or_parse_error:{student_action}"


def _update_query_stats(
    stats: Counter,
    *,
    step_messages: List[Dict[str, Any]],
    compress_prompt: bool,
    student_action: str,
    queries_before: List[Dict[str, Any]],
    queries_after: List[Dict[str, Any]],
) -> None:
    prompt_text = _user_text(step_messages)
    has_active_query = "<active_query>" in prompt_text
    has_response_history = "<response_history>" in prompt_text
    open_before = [q for q in queries_before if _query_is_open_local(q)]
    open_after = [q for q in queries_after if _query_is_open_local(q)]
    answered_after = [
        q for q in queries_after
        if str(q.get("status", "")).strip().lower() == "answered"
    ]
    if has_active_query:
        stats["query_prompt_active"] += 1
    if has_response_history:
        stats["query_prompt_response_history"] += 1
    if has_active_query and not open_before:
        stats["query_active_prompt_without_open_query"] += 1
    if open_before and not has_active_query and not compress_prompt:
        stats["query_open_not_rendered"] += 1
    if compress_prompt and open_before:
        stats["query_suppressed_during_system_compress"] += 1
    if student_action == "response":
        stats["student_response_steps"] += 1
        if answered_after:
            stats["student_response_completed_query"] += 1
        if open_after:
            stats["student_response_query_still_open"] += 1


def _teacher_messages(
    sample: Dict[str, Any],
    step_messages: List[Dict[str, Any]],
    *,
    frame_protocol: str,
    render_layout: str,
    min_pixels: int,
    max_pixels: int,
) -> List[Dict[str, Any]]:
    messages = deepcopy(step_messages)
    if sample.get("v12_assistant_turn_1") and sample.get("v12_assistant_turn_2"):
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })
        tool_user_content = build_recall_result_user_content(
            sample.get("recalled_frames"),
            sample.get("recall_result") or {},
            frame_protocol=frame_protocol,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            render_layout=render_layout,
        )
        messages.append({
            "role": "tool",
            "tool_call_id": "recall",
            "content": tool_user_content,
        })
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": _normalise_assistant_output(
                    sample, sample["v12_assistant_turn_2"]
                ),
            }],
        })
    else:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": _normalise_assistant_output(sample)}],
        })
    return messages


def _teacher_action(sample: Dict[str, Any]) -> str:
    if sample.get("sample_type") == "recall":
        return "recall"
    return str(sample.get("action") or sample.get("sample_type") or "")


def _row_sample_id(row: Dict[str, Any], sample: Dict[str, Any], gen_idx: int) -> str:
    base = (
        row.get("sample_id")
        or sample.get("sample_id")
        or f"{sample.get('video_id','')}:{sample.get('trajectory_id','')}:{sample.get('chunk_idx', -1)}"
    )
    subtype = row.get("sft_subtype") or row.get("sample_type") or ""
    return f"{base}:dagger:g{gen_idx}:{subtype}"


def _extract_rows_from_rollout(
    *,
    source_traj: Dict[str, Any],
    rollout_result: Dict[str, Any],
    ckpt: str,
    frame_protocol: str,
    render_layout: str,
    min_pixels: int,
    max_pixels: int,
    keep_only_mismatch: bool,
) -> tuple[List[Dict[str, Any]], Counter, List[Dict[str, Any]]]:
    stats: Counter = Counter()
    q_by_card = _question_map(source_traj)
    samples_by_chunk = {
        int(s.get("chunk_idx")): s
        for s in source_traj.get("samples", [])
        if s.get("chunk_idx") is not None
    }
    rows: List[Dict[str, Any]] = []
    compress_audits: List[Dict[str, Any]] = []

    for cr in rollout_result.get("chunk_results", []):
        step_messages_list = cr.get("dagger_step_messages") or cr.get("step_messages") or []
        raw_outputs = cr.get("raw_outputs") or []
        first_actions = cr.get("first_actions") or cr.get("actions") or []
        turn_kinds = cr.get("turn_kinds") or []
        chunk_indices = cr.get("chunk_indices") or []
        compress_prefix_diags = cr.get("compress_prefix_diagnostics") or []
        compress_trigger_diags = cr.get("compress_trigger_diagnostics") or []
        action_space_errors = cr.get("action_space_errors") or []
        invalid_actions = cr.get("invalid_actions") or []
        queries_before_list = cr.get("queries_before") or []
        queries_after_list = cr.get("queries_after") or []
        group_size = max(
            len(step_messages_list),
            len(first_actions),
            len(chunk_indices),
            len(raw_outputs),
        )
        for gen_idx in range(group_size):
            step_messages = (
                step_messages_list[gen_idx]
                if gen_idx < len(step_messages_list)
                else None
            )
            if not step_messages:
                stats["skip_no_step_messages"] += 1
                continue
            chunk_idx = (
                int(chunk_indices[gen_idx])
                if gen_idx < len(chunk_indices) and int(chunk_indices[gen_idx]) >= 0
                else int(cr.get("chunk_idx", -1))
            )
            compress_prompt = _is_compress_prompt(step_messages)
            turn_kind = (
                str(turn_kinds[gen_idx])
                if gen_idx < len(turn_kinds)
                else ("compress" if compress_prompt else "streaming")
            )
            student_action = (
                str(first_actions[gen_idx])
                if gen_idx < len(first_actions)
                else ""
            )
            prefix_diag = (
                compress_prefix_diags[gen_idx]
                if gen_idx < len(compress_prefix_diags)
                and isinstance(compress_prefix_diags[gen_idx], dict)
                else {}
            )
            trigger_diag = (
                compress_trigger_diags[gen_idx]
                if gen_idx < len(compress_trigger_diags)
                and isinstance(compress_trigger_diags[gen_idx], dict)
                else {}
            )
            action_space_error = (
                str(action_space_errors[gen_idx])
                if gen_idx < len(action_space_errors)
                else ""
            )
            invalid_action = (
                str(invalid_actions[gen_idx])
                if gen_idx < len(invalid_actions)
                else ""
            )
            queries_before = (
                queries_before_list[gen_idx]
                if gen_idx < len(queries_before_list)
                and isinstance(queries_before_list[gen_idx], list)
                else []
            )
            queries_after = (
                queries_after_list[gen_idx]
                if gen_idx < len(queries_after_list)
                and isinstance(queries_after_list[gen_idx], list)
                else []
            )
            _update_query_stats(
                stats,
                step_messages=step_messages,
                compress_prompt=compress_prompt,
                student_action=student_action,
                queries_before=queries_before,
                queries_after=queries_after,
            )

            # System compression is an online environment event. It should be
            # audited against the student's action, not aligned to teacher
            # gold compress chunks from the source trajectory.
            if compress_prompt or turn_kind == "compress":
                stats["system_compress_required"] += 1
                stats[f"system_compress_student_action:{student_action}"] += 1
                reason = _compress_failure_reason(
                    student_action,
                    prefix_diag,
                    action_space_error,
                )
                stats[f"system_compress_reason:{reason}"] += 1
                if student_action == "compress":
                    stats["system_compress_success"] += 1
                else:
                    stats["system_compress_failure"] += 1
                    compress_audits.append({
                        "video_id": source_traj.get("video_id"),
                        "trajectory_id": source_traj.get("trajectory_id"),
                        "chunk_idx": chunk_idx,
                        "gen_idx": gen_idx,
                        "turn_kind": turn_kind,
                        "student_action": student_action,
                        "invalid_action": invalid_action,
                        "failure_reason": reason,
                        "action_space_error": action_space_error,
                        "compress_trigger_diagnostic": trigger_diag,
                        "compress_prefix_diagnostic": prefix_diag,
                        "raw_output": (
                            str(raw_outputs[gen_idx])
                            if gen_idx < len(raw_outputs)
                            else ""
                        ),
                    })
                continue

            sample0 = samples_by_chunk.get(chunk_idx)
            if not sample0:
                stats["skip_no_teacher_sample"] += 1
                continue
            if _is_compress_sample(sample0):
                stats["skip_teacher_gold_compress_ignored"] += 1
                continue

            sample = _patch_sample_from_question(sample0, q_by_card)
            teacher_action = _teacher_action(sample)
            if keep_only_mismatch and student_action == teacher_action:
                stats["skip_student_matches_teacher"] += 1
                continue

            messages = _teacher_messages(
                sample,
                step_messages,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            for row in build_sft_rows(
                sample,
                messages,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
            ):
                row["render_layout"] = render_layout
                row["sample_id"] = _row_sample_id(row, sample, gen_idx)
                meta = dict(row.get("metadata") or {})
                meta.update({
                    "render_layout": render_layout,
                    "dagger": True,
                    "dagger_policy_checkpoint": ckpt,
                    "dagger_gen_idx": gen_idx,
                    "dagger_chunk_idx": chunk_idx,
                    "dagger_turn_kind": turn_kind,
                    "dagger_student_action": student_action,
                    "dagger_teacher_action": teacher_action,
                    "dagger_student_output": (
                        str(raw_outputs[gen_idx])
                        if gen_idx < len(raw_outputs)
                        else ""
                    ),
                })
                row["metadata"] = meta
                rows.append(row)
                stats["rows"] += 1
                stats[f"rows_by_sample_type:{row.get('sample_type','')}"] += 1
                stats[f"rows_by_loss_class:{row.get('loss_class','')}"] += 1

            stats["teacher_samples_used"] += 1
            stats[f"teacher_action:{teacher_action}"] += 1
            stats[f"student_action:{student_action}"] += 1
            if student_action == teacher_action:
                stats["student_teacher_action_match"] += 1
            else:
                stats["student_teacher_action_mismatch"] += 1
    return rows, stats, compress_audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SCHEME / "final" / "train_sft_dagger_source_trajectories.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_SCHEME
            / "rendered"
            / "video_meta_standard_query_last"
            / "train_sft_dagger_messages.jsonl"
        ),
    )
    parser.add_argument("--stats-out", default="")
    parser.add_argument("--compress-audit-out", default="")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--rollouts-per-video", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--compress-max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--vllm-max-images-per-prompt", type=int, default=64)
    parser.add_argument("--vllm-max-videos-per-prompt", type=int, default=2)
    parser.add_argument("--vllm-mm-processor-cache-gb", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--frame-protocol", default="video_meta", choices=["video_meta"])
    parser.add_argument(
        "--render-layout",
        default="standard_query_last",
        choices=["standard_query_last"],
    )
    parser.add_argument("--disable-recall", action="store_true")
    parser.add_argument("--only-mismatch", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    stats_out = Path(args.stats_out) if args.stats_out else out.with_suffix(".stats.json")
    compress_audit_out = (
        Path(args.compress_audit_out)
        if args.compress_audit_out
        else out.with_suffix(".compress_audit.jsonl")
    )
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = normalize_render_layout(args.render_layout)
    scheme_root = source.parent.parent if source.parent.name == "final" else source.parent
    frames_root = Path(args.frames_root) if args.frames_root else scheme_root / "frames"

    all_trajectories = list(_read_jsonl(source, args.limit_videos))
    trajectories = _select_shard(
        all_trajectories,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not trajectories:
        raise SystemExit(f"no trajectories loaded from {source}")

    print(
        f"Loaded {len(trajectories)}/{len(all_trajectories)} DAgger source "
        f"trajectories from {source}; shard={args.shard_index}/{args.num_shards}; "
        f"frames_root={frames_root}"
    )
    print(f"Loading processor/checkpoint: {args.ckpt}")
    processor = load_processor_for_checkpoint(args.ckpt, trust_remote_code=True)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    llm = init_vllm_engine(
        args.ckpt,
        tensor_parallel_size=(args.tensor_parallel_size or None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_images_per_prompt=args.vllm_max_images_per_prompt,
        max_videos_per_prompt=args.vllm_max_videos_per_prompt,
        mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
        enable_prefix_caching=True,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    compress_audit_out.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    t0 = time.time()
    with (
        out.open("w", encoding="utf-8") as f_out,
        compress_audit_out.open("w", encoding="utf-8") as f_audit,
    ):
        for start in range(0, len(trajectories), args.rollout_batch_size):
            batch = trajectories[start:start + args.rollout_batch_size]
            step_inputs = [_trajectory_for_rollout(t) for t in batch]
            max_batch_chunk = max(
                (
                    max((int(s.get("chunk_idx", 0)) for s in t.get("samples", [])), default=0)
                    for t in batch
                ),
                default=0,
            )
            rollout_max_chunks = min(args.max_chunks, max_batch_chunk + 1)
            print(
                f"[{start}:{start + len(batch)}] rollout "
                f"B={len(batch)} S={args.rollouts_per_video} max_chunks={rollout_max_chunks}"
            )
            rollout_results = streaming_vllm_rollout(
                step_inputs,
                llm,
                processor,
                processor.tokenizer,
                group_size=args.rollouts_per_video,
                max_new_tokens=args.max_new_tokens,
                compress_max_new_tokens=args.compress_max_new_tokens,
                rollout_max_chunks=rollout_max_chunks,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                frames_root=str(frames_root),
                video_root=None,
                enable_recall=not args.disable_recall,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
            )
            for source_traj, rollout_result in zip(batch, rollout_results):
                rows, row_stats, compress_audits = _extract_rows_from_rollout(
                    source_traj=source_traj,
                    rollout_result=rollout_result,
                    ckpt=args.ckpt,
                    frame_protocol=frame_protocol,
                    render_layout=render_layout,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    keep_only_mismatch=args.only_mismatch,
                )
                stats.update(row_stats)
                stats["videos_processed"] += 1
                for row in rows:
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                for audit in compress_audits:
                    f_audit.write(json.dumps(audit, ensure_ascii=False) + "\n")
            f_out.flush()
            f_audit.flush()

    elapsed = time.time() - t0
    stats["elapsed_sec"] = round(elapsed, 3)
    stats["source_trajectories"] = len(trajectories)
    stats["source_trajectories_total"] = len(all_trajectories)
    stats["num_shards"] = args.num_shards
    stats["shard_index"] = args.shard_index
    stats["rollout_batch_size"] = args.rollout_batch_size
    stats["rollouts_per_video"] = args.rollouts_per_video
    stats["output"] = str(out)
    stats["compress_audit_output"] = str(compress_audit_out)
    stats["checkpoint"] = args.ckpt
    stats_out.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    _merge_dataset_info(out.parent, "train_sft_dagger")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"Wrote {stats.get('rows', 0)} DAgger SFT rows -> {out}")
    print(f"Wrote stats -> {stats_out}")


if __name__ == "__main__":
    main()
