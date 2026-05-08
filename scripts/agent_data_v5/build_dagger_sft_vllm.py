#!/usr/bin/env python
"""Build DAgger SFT messages with a vLLM-batched on-policy rollout.

This is intentionally separate from verl training: DAgger produces
ShareGPT/messages JSONL rows, while verl produces PPO/GRPO batches.  The
rollout state machine mirrors thinkstream.eval.streaming_vllm:

* each trajectory owns an independent MemoryState;
* live trajectories are batched into one vLLM generate call;
* system compress is an inter-chunk turn and does not consume the video chunk;
* recall is a same-chunk two-turn tool call.

Use --correction-only for the stage-2 SFT dataset. It emits only states where
the policy made a targetable mistake such as repeated think, missed compress,
missed recall, over-recall, missed response, wrong response, or early answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from scripts.agent_data_v5.build_dagger_sft import (
    DEFAULT_DAGGER_CORRECTION_REASONS,
    _answer_chunks,
    _answer_matches,
    _content_text,
    _default_batch_root,
    _emit_dagger_row,
    _apply_oracle_compress_recovery,
    _gold_answer,
    _gold_compress_summary,
    _group_by_chunk,
    _iter_trajectory_rows,
    _new_question,
    _normalise_range,
    _policy_answer,
    _prompt_has_compress_trigger,
    _propagate_sample_fields,
    _question_text,
    _question_meta,
    _parse_reason_set,
    _record_compress_diag_stats,
    _resolve_path,
    _resolve_video_path,
    finalize_dagger_stats,
)
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    build_compress_trigger_user_input,
    build_recalled_frames_metadata,
    build_recall_result_user_content,
    diagnose_compress_output_v12,
    has_compress_trigger,
    normalize_frame_protocol,
    normalize_render_layout,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
    action_space_error_for_turn,
    tools_for_turn,
)
from thinkstream.model.agent_loop import (
    COMPRESS_RANGE_MIN,
    COMPRESS_TOKEN_THRESHOLD,
    MemoryState,
    RECENT_THINKS_TOKEN_BUDGET,
    _parse_agent_output,
    build_single_step_messages,
    select_compress_range_by_tokens,
)
from thinkstream.model.retrieval import BM25Retriever
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


@dataclass
class DaggerRunner:
    idx: int
    traj_i: int
    video_path: str
    samples_by_chunk: Dict[int, List[Tuple[str, Dict[str, Any]]]]
    question_at_chunk: Dict[int, str]
    question_meta_at_chunk: Dict[int, Dict[str, Any]]
    max_chunk: int
    memory: MemoryState
    frames_root: Optional[str]
    video_root: Optional[str]
    min_pixels: int
    max_pixels: int
    frame_protocol: str
    render_layout: str
    current_chunk: int = 0
    done: bool = False
    error: Optional[str] = None
    _last_trigger: bool = False
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    retriever: Optional[BM25Retriever] = field(default_factory=BM25Retriever)
    emitted: set[str] = field(default_factory=set)
    compress_retries: Dict[int, int] = field(default_factory=dict)
    pending_missed_responses: List[Dict[str, Any]] = field(default_factory=list)

    # Fields consumed by _prepare_step_messages for legacy fallback.
    raw_sample: Dict[str, Any] = field(default_factory=dict)
    query: Optional[str] = None
    ask_chunk: int = 0
    gen_idx: int = 0
    sample_idx: int = 0
    max_chunks: int = 0

    def _record_answer_to_memory(self, answer_text: str, chunk_idx: int) -> None:
        if not answer_text:
            return
        response_time = chunk_idx * AGENT_CHUNK_SEC
        for q in reversed(self.memory.queries):
            status = str(q.get("status", "")).strip().lower()
            if status in {"open", "pending", "active"} or (
                not status and not q.get("answers")
            ):
                self.memory.answer_query(q["question"], answer_text, response_time)
                break


def _sample_key(chunk_idx: int, pos: int, sample: Dict[str, Any]) -> str:
    sid = sample.get("sample_id") or sample.get("id") or ""
    return f"{chunk_idx}:{pos}:{sample.get('sample_type','')}:{sid}"


def _trajectory_question_maps(traj: Dict[str, Any], samples: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]]]:
    q_at: Dict[int, str] = {}
    q_meta: Dict[int, Dict[str, Any]] = {}
    if isinstance(traj.get("questions"), list):
        for q in traj["questions"]:
            text = q.get("question") or q.get("gold_answer") or ""
            meta = {
                "options": list(q.get("options") or []),
                "answer_form": q.get("answer_form", ""),
                "answer_style": q.get("answer_style", ""),
                "answer_instruction": q.get("answer_instruction", ""),
                "answer_chunks": list(q.get("answer_chunks") or []),
                "per_emit_answers": list(q.get("per_emit_answers") or []),
            }
            ans_chunks = [int(x) for x in q.get("answer_chunks") or []]
            if ans_chunks:
                meta["open_until"] = max(ans_chunks) * AGENT_CHUNK_SEC
            for c in q.get("ask_chunks") or []:
                ci = int(c)
                q_at[ci] = text
                q_meta[ci] = meta
    if q_at:
        return q_at, q_meta

    for chunk_idx, chunk_samples in _group_by_chunk(samples).items():
        for s in chunk_samples:
            q = _new_question(s)
            if q:
                q_at[int(chunk_idx)] = q
                q_meta[int(chunk_idx)] = _question_meta(s)
                break
    return q_at, q_meta


def _runner_max_chunk(traj: Dict[str, Any], samples: List[Dict[str, Any]]) -> int:
    vals = [int(s.get("chunk_idx", 0)) for s in samples]
    for q in traj.get("questions") or []:
        vals.extend(int(c) for c in q.get("ask_chunks") or [])
        vals.extend(int(c) for c in q.get("answer_chunks") or [])
    return max(vals) if vals else 0


def _make_runners(
    trajectories: Path,
    *,
    data_dir: Path,
    frames_root: str,
    video_root: Optional[str],
    tokenizer,
    frame_protocol: str,
    min_pixels: int,
    max_pixels: int,
    max_trajectories: int,
    num_shards: int,
    shard_index: int,
    rollout_all_chunks: bool,
    render_layout: str,
) -> Tuple[List[DaggerRunner], Dict[str, Any]]:
    runners: List[DaggerRunner] = []
    stats = {
        "trajectories_seen": 0,
        "trajectories_used": 0,
        "skipped": {},
    }
    for traj_i, traj in enumerate(_iter_trajectory_rows(trajectories)):
        if max_trajectories and stats["trajectories_used"] >= max_trajectories:
            break
        stats["trajectories_seen"] += 1
        if num_shards > 1 and (traj_i % num_shards) != shard_index:
            continue

        samples = _propagate_sample_fields(traj)
        if not samples:
            continue
        video_path = _resolve_video_path(samples[0].get("video_path", ""), video_root)
        if not video_path or not Path(video_path).exists():
            stats["skipped"]["missing_video"] = stats["skipped"].get("missing_video", 0) + 1
            continue

        grouped_raw = _group_by_chunk(samples)
        samples_by_chunk: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
        for chunk_idx, chunk_samples in grouped_raw.items():
            samples_by_chunk[int(chunk_idx)] = [
                (_sample_key(int(chunk_idx), i, s), s)
                for i, s in enumerate(chunk_samples)
            ]

        q_at, q_meta = _trajectory_question_maps(traj, samples)
        max_chunk = _runner_max_chunk(traj, samples)
        if not rollout_all_chunks:
            # Debug/ablation mode: only visit target chunks, preserving the
            # older HF builder's cheaper but less realistic state distribution.
            max_chunk = max(samples_by_chunk) if samples_by_chunk else max_chunk
        stats["trajectories_used"] += 1
        runners.append(DaggerRunner(
            idx=len(runners),
            traj_i=traj_i,
            video_path=video_path,
            samples_by_chunk=samples_by_chunk,
            question_at_chunk=q_at,
            question_meta_at_chunk=q_meta,
            max_chunk=max_chunk,
            memory=MemoryState(tokenizer=tokenizer),
            frames_root=frames_root,
            video_root=video_root,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            frame_protocol=frame_protocol,
            render_layout=render_layout,
            raw_sample=traj,
            max_chunks=max_chunk + 1,
        ))
    return runners, stats


def _chunk_entries(runner: DaggerRunner) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]]]:
    entries = [
        (k, s) for k, s in runner.samples_by_chunk.get(runner.current_chunk, [])
        if k not in runner.emitted
    ]
    compress = [(k, s) for k, s in entries if str(s.get("sample_type", "")) == "compress"]
    visual = [(k, s) for k, s in entries if str(s.get("sample_type", "")) != "compress"]
    return compress, visual


def _select_rollout_batch(
    runners: List[DaggerRunner],
    *,
    start_idx: int,
    batch_size: int,
    scheduler: str,
) -> Tuple[List[DaggerRunner], int]:
    """Select live runners without pinning rollout to the first long videos.

    The legacy scheduler used ``live[:batch_size]``, which completes the first
    few trajectories before later trajectories get any on-policy state.  The
    round-robin scheduler keeps each trajectory's chunk order intact while
    spreading early rollout coverage across the whole split.
    """
    if batch_size <= 0:
        batch_size = 1
    if scheduler == "head":
        return [r for r in runners if not r.done][:batch_size], start_idx

    n = len(runners)
    if n == 0:
        return [], 0
    batch: List[DaggerRunner] = []
    idx = start_idx % n
    scanned = 0
    while scanned < n and len(batch) < batch_size:
        r = runners[idx]
        if not r.done:
            batch.append(r)
        idx = (idx + 1) % n
        scanned += 1
    return batch, idx


def _prepare_forced_compress_messages(runner: DaggerRunner) -> List[Dict[str, Any]]:
    """Build a system compress turn at the current chunk.

    Gold compress samples mark memory-management states.  The trigger is a
    system event, so DAgger should still be able to ask the policy to compress
    when the gold target reaches this chunk even if the policy-induced memory
    token count has not naturally crossed the runtime threshold.
    """
    chunk_idx = runner.current_chunk
    runner._last_trigger = True
    runner._last_turn_kind = "compress"
    return build_single_step_messages(
        runner.memory.snapshot(chunk_idx),
        chunk_idx,
        runner.video_path,
        user_input=build_compress_trigger_user_input(),
        queries=runner.memory.queries,
        min_pixels=runner.min_pixels,
        max_pixels=runner.max_pixels,
        frame_paths=None,
        frame_protocol=runner.frame_protocol,
        inter_chunk=True,
        render_layout=runner.render_layout,
    )


def _bump_nested_counter(stats: Dict[str, Any], name: str, key: str, subkey: str, n: int = 1) -> None:
    bucket = stats.setdefault(name, {})
    inner = bucket.setdefault(str(key or "unknown"), {})
    inner[str(subkey or "unknown")] = inner.get(str(subkey or "unknown"), 0) + int(n)


def _chunks_covered_by_range(
    recent_thinks: List[Dict[str, Any]],
    time_range: Optional[List[int]],
) -> List[int]:
    if not time_range:
        return []
    start, end = int(time_range[0]), int(time_range[1])
    out: List[int] = []
    for item in recent_thinks:
        try:
            chunk = int(item.get("chunk"))
        except (TypeError, ValueError):
            continue
        cs = chunk * AGENT_CHUNK_SEC
        ce = cs + AGENT_CHUNK_SEC
        if start <= cs and ce <= end:
            out.append(chunk)
    return sorted(set(out))


def _validate_compress_payload(
    result: Dict[str, Any],
    recent_thinks: List[Dict[str, Any]],
) -> Tuple[str, Optional[List[int]], List[int]]:
    if str(result.get("action") or "") != "compress":
        diag = result.get("compress_prefix_diagnostic") or {}
        label = str(diag.get("label") or "unknown")
        if diag.get("likely_truncated"):
            return "json_truncated", None, []
        if diag.get("tool_call_open"):
            return f"malformed_tool:{label}", None, []
        return f"not_compress_action:{result.get('action') or 'unknown'}", None, []

    summary = ((result.get("payload") or {}).get("summary") or {})
    pred_range = _normalise_range(summary.get("time_range"))
    if pred_range is None:
        return "bad_range_schema", None, []
    text = summary.get("text")
    if not isinstance(text, str) or not text.strip():
        return "empty_summary_text", pred_range, []
    chunks = _chunks_covered_by_range(recent_thinks, pred_range)
    if not chunks:
        return "range_no_recent_chunks", pred_range, []
    if len(chunks) < COMPRESS_RANGE_MIN:
        return "range_too_small", pred_range, chunks
    return "ok", pred_range, chunks


def _apply_compress_turn_output(
    runner: DaggerRunner,
    output_text: str,
    tokenizer,
    *,
    pre_recent_thinks: List[Dict[str, Any]],
    pre_memory_tokens: int,
    compress_budget: int,
) -> None:
    """Apply a compress turn only after validating the emitted tool call.

    The generic rollout helper eventually falls back inside MemoryState.compress
    when a range covers no chunks.  For DAgger diagnostics that hides the real
    failure.  This path keeps bad compress outputs from mutating memory, records
    the exact reason, and lets the outer loop choose a recovery mode.
    """
    chunk_idx = runner.current_chunk
    parsed = _parse_agent_output(output_text)
    action = parsed.get("action") or "unknown"
    action_error = action_space_error_for_turn(action, "compress")
    if action_error:
        parsed["action_space_error"] = action_error
        parsed["invalid_action"] = action
        action = "invalid"

    result = {
        "chunk_idx": chunk_idx,
        "action": action,
        "think": parsed.get("think", ""),
        "payload": parsed.get("payload", {}),
        "raw_output": output_text,
        "compress_prefix_diagnostic": diagnose_compress_output_v12(output_text),
        "action_space_error": parsed.get("action_space_error", ""),
        "invalid_action": parsed.get("invalid_action", ""),
        "generated_tokens": tokenizer.encode(output_text, add_special_tokens=False),
        "compress_budget": compress_budget,
        "recall_returned_chunks": [],
        "window_start": chunk_idx * int(AGENT_CHUNK_SEC),
        "window_end": (chunk_idx + 1) * int(AGENT_CHUNK_SEC),
        "step_messages": None,
    }
    if parsed.get("format_error"):
        result["format_error"] = parsed.get("format_error")

    status, pred_range, chunks = _validate_compress_payload(result, pre_recent_thinks)
    applied = False
    if status == "ok" and pred_range is not None and chunks:
        summary = dict(((result.get("payload") or {}).get("summary") or {}))
        summary["time_range"] = pred_range
        summary["text"] = str(summary.get("text") or "").strip()
        runner.memory.compress(summary, compressed_chunks=chunks)
        applied = True

    post_tokens = runner.memory.count_recent_tokens()
    result.update({
        "memory_token_count": post_tokens,
        "pre_compress_memory_token_count": pre_memory_tokens,
        "post_compress_memory_token_count": post_tokens,
        "compress_memory_token_delta": int(pre_memory_tokens) - int(post_tokens),
        "compress_runtime_error": "ok" if applied else status,
        "compress_predicted_range": pred_range,
        "compress_applied": applied,
        "compress_applied_chunks": chunks if applied else [],
    })
    runner.chunk_results.append(result)


def _extractive_compress_summary(memory: MemoryState) -> Optional[Tuple[Dict[str, Any], List[int]]]:
    n = select_compress_range_by_tokens(
        memory.recent_thinks,
        token_count_fn=memory._token_count,
    )
    if n <= 0:
        return None
    chosen = list(memory.recent_thinks[:n])
    chunks = [int(t.get("chunk")) for t in chosen if t.get("chunk") is not None]
    if not chunks:
        return None
    start = min(chunks) * AGENT_CHUNK_SEC
    end = (max(chunks) + 1) * AGENT_CHUNK_SEC
    pieces = []
    for item in chosen:
        text = str(item.get("text") or "").strip()
        if text:
            pieces.append(f"[{item.get('time', '')}] {text}")
    summary_text = " ".join(pieces).strip()
    if not summary_text:
        summary_text = f"Earlier observations from {start}-{end}s were compressed."
    tokenizer = getattr(memory, "_tokenizer", None)
    if tokenizer is not None:
        ids = tokenizer.encode(summary_text, add_special_tokens=False)
        if len(ids) > 280:
            summary_text = tokenizer.decode(ids[:280])
    else:
        summary_text = summary_text[:1200]
    return {"time_range": [int(start), int(end)], "text": summary_text}, chunks


def _apply_extractive_compress_recovery(
    memory: MemoryState,
    stats: Dict[str, Any],
    *,
    source: str,
    failure_reason: str,
) -> Optional[Dict[str, Any]]:
    recovered = _extractive_compress_summary(memory)
    if recovered is None:
        stats["skipped"]["extractive_compress_recovery_unavailable"] = (
            stats["skipped"].get("extractive_compress_recovery_unavailable", 0) + 1
        )
        return None
    summary, chunks = recovered
    memory.compress(summary, compressed_chunks=chunks)
    stats["extractive_compress_recoveries"] = stats.get("extractive_compress_recoveries", 0) + 1
    _bump_nested_counter(stats, "compress_recovery_by_source", source, "extractive")
    _bump_nested_counter(stats, "compress_recovery_by_reason", failure_reason, "extractive")
    return {
        "mode": "extractive",
        "source": source,
        "failure_reason": failure_reason,
        "time_range": summary.get("time_range"),
        "chunks": chunks,
    }


def _apply_compress_recovery(
    runner: DaggerRunner,
    compress_entries: List[Tuple[str, Dict[str, Any]]],
    stats: Dict[str, Any],
    *,
    source: str,
    failure_reason: str,
    allow_oracle: bool,
) -> Optional[Dict[str, Any]]:
    if allow_oracle and compress_entries:
        before = stats.get("oracle_compress_recoveries", 0)
        if _apply_oracle_compress_recovery(
            runner.memory,
            [s for _, s in compress_entries],
            stats,
        ):
            _bump_nested_counter(stats, "compress_recovery_by_source", source, "gold")
            _bump_nested_counter(stats, "compress_recovery_by_reason", failure_reason, "gold")
            summary = _gold_compress_summary(compress_entries[0][1])
            return {
                "mode": "gold",
                "source": source,
                "failure_reason": failure_reason,
                "time_range": (summary or {}).get("time_range"),
                "oracle_recovery_index": stats.get("oracle_compress_recoveries", before),
            }
    return _apply_extractive_compress_recovery(
        runner.memory,
        stats,
        source=source,
        failure_reason=failure_reason,
    )


def _compact_counter(raw: Dict[str, Any], *, limit: int = 5) -> Dict[str, Any]:
    items = sorted(
        ((str(k), v) for k, v in (raw or {}).items()),
        key=lambda kv: (-int(kv[1]), kv[0]),
    )
    return {k: v for k, v in items[:limit]}


def _update_scalar_stats(bucket: Dict[str, Any], value: int) -> None:
    value = int(value)
    bucket["count"] = int(bucket.get("count", 0) or 0) + 1
    bucket["sum"] = int(bucket.get("sum", 0) or 0) + value
    bucket["min"] = value if "min" not in bucket else min(int(bucket["min"]), value)
    bucket["max"] = value if "max" not in bucket else max(int(bucket["max"]), value)
    bucket["mean"] = bucket["sum"] / max(bucket["count"], 1)


def _gold_compress_miss_reason(runner: DaggerRunner) -> str:
    tokens = runner.memory.count_recent_tokens()
    if tokens < COMPRESS_TOKEN_THRESHOLD:
        return "below_token_threshold"
    if len(runner.memory.recent_thinks) < COMPRESS_RANGE_MIN:
        return "below_min_recent_thinks"
    n = select_compress_range_by_tokens(
        runner.memory.recent_thinks,
        token_count_fn=runner.memory._token_count,
    )
    if n <= 0:
        return "no_actionable_compress_range"
    return "unknown"


def _record_gold_compress_state(
    stats: Dict[str, Any],
    runner: DaggerRunner,
    *,
    target_count: int,
    natural_trigger: bool,
    forced_trigger: bool,
) -> None:
    stats["gold_compress_targets_seen"] = (
        stats.get("gold_compress_targets_seen", 0) + int(target_count)
    )
    stats["gold_compress_chunks_seen"] = stats.get("gold_compress_chunks_seen", 0) + 1
    token_bucket = stats.setdefault("gold_compress_memory_tokens", {})
    recent_bucket = stats.setdefault("gold_compress_recent_thinks", {})
    _update_scalar_stats(token_bucket, runner.memory.count_recent_tokens())
    _update_scalar_stats(recent_bucket, len(runner.memory.recent_thinks))
    if natural_trigger:
        stats["gold_compress_natural_trigger_chunks"] = (
            stats.get("gold_compress_natural_trigger_chunks", 0) + 1
        )
        stats["gold_compress_natural_trigger_targets"] = (
            stats.get("gold_compress_natural_trigger_targets", 0) + int(target_count)
        )
        return
    stats["gold_compress_without_natural_trigger_chunks"] = (
        stats.get("gold_compress_without_natural_trigger_chunks", 0) + 1
    )
    stats["gold_compress_without_natural_trigger_targets"] = (
        stats.get("gold_compress_without_natural_trigger_targets", 0) + int(target_count)
    )
    reason = _gold_compress_miss_reason(runner)
    miss_bucket = stats.setdefault("gold_compress_without_trigger_reason", {})
    miss_bucket[reason] = miss_bucket.get(reason, 0) + int(target_count)
    if forced_trigger:
        stats["gold_compress_forced_trigger_chunks"] = (
            stats.get("gold_compress_forced_trigger_chunks", 0) + 1
        )
        stats["gold_compress_forced_trigger_targets"] = (
            stats.get("gold_compress_forced_trigger_targets", 0) + int(target_count)
        )


def _record_compress_turn_result(
    stats: Dict[str, Any],
    result: Dict[str, Any],
    *,
    source: str,
) -> None:
    action = str(result.get("action") or "unknown")
    by_source = stats.setdefault("compress_turn_action_by_source", {})
    action_bucket = by_source.setdefault(source, {})
    action_bucket[action] = action_bucket.get(action, 0) + 1
    runtime_error = str(result.get("compress_runtime_error") or "unknown")
    _bump_nested_counter(stats, "compress_runtime_error_by_source", source, runtime_error)
    if result.get("compress_applied"):
        applied_bucket = stats.setdefault("compress_applied_by_source", {})
        applied_bucket[source] = applied_bucket.get(source, 0) + 1
    if result.get("format_ok"):
        ok_bucket = stats.setdefault("compress_turn_format_ok_by_source", {})
        ok_bucket[source] = ok_bucket.get(source, 0) + 1
    if action == "compress":
        payload = result.get("payload") or {}
        summary = payload.get("summary") or {}
        tr = summary.get("time_range")
        text = summary.get("text")
        if isinstance(tr, list) and len(tr) == 2 and isinstance(text, str) and text.strip():
            eff_bucket = stats.setdefault("compress_turn_effective_by_source", {})
            eff_bucket[source] = eff_bucket.get(source, 0) + 1
    examples = stats.setdefault("compress_turn_examples", [])
    if len(examples) < 12:
        examples.append({
            "chunk_idx": result.get("chunk_idx"),
            "source": source,
            "action": action,
            "format_ok": bool(result.get("format_ok")),
            "pre_memory_tokens": result.get("pre_compress_memory_token_count"),
            "pre_recent_thinks": result.get("pre_compress_recent_thinks"),
            "post_memory_tokens": result.get("post_compress_memory_token_count"),
            "memory_token_delta": result.get("compress_memory_token_delta"),
            "runtime_error": runtime_error,
            "compress_applied": bool(result.get("compress_applied")),
            "compress_applied_chunks": result.get("compress_applied_chunks") or [],
            "payload": result.get("payload") or {},
            "diagnostic": result.get("compress_prefix_diagnostic") or {},
            "raw_output": str(result.get("raw_output") or result.get("raw") or "")[:1200],
        })


def _inc_stat(stats: Dict[str, Any], key: str, n: int = 1) -> None:
    stats[key] = int(stats.get(key, 0) or 0) + int(n)


def _bucket_stat(stats: Dict[str, Any], key: str, item: str, n: int = 1) -> None:
    bucket = stats.setdefault(key, {})
    item = str(item or "unknown")
    bucket[item] = int(bucket.get(item, 0) or 0) + int(n)


def _record_missed_response_example(
    stats: Dict[str, Any],
    name: str,
    item: Dict[str, Any],
    *,
    limit: int = 12,
) -> None:
    examples = stats.setdefault(name, [])
    if len(examples) < limit:
        examples.append(item)


def _close_pending_missed_responses(
    runner: DaggerRunner,
    stats: Dict[str, Any],
    *,
    reason: str,
    close_chunk: Optional[int] = None,
) -> None:
    for pending in runner.pending_missed_responses:
        if pending.get("closed"):
            continue
        pending["closed"] = True
        pending["close_reason"] = reason
        if close_chunk is not None:
            pending["close_chunk"] = int(close_chunk)
        if not pending.get("resolved"):
            _inc_stat(stats, "missed_response_unresolved")
            _inc_stat(stats, "response_event_missed_unresolved")
            _bucket_stat(stats, "missed_response_unresolved_by_reason", reason)


def _match_pending_missed_response(
    pending_items: List[Dict[str, Any]],
    answer_text: str,
) -> Optional[Dict[str, Any]]:
    open_items = [
        p for p in pending_items
        if not p.get("closed") and not p.get("resolved")
    ]
    if not open_items:
        return None
    for item in reversed(open_items):
        gold = str(item.get("gold_answer") or "").strip()
        if gold and _answer_matches(gold, answer_text):
            return item
    return open_items[-1]


def _sample_has_response_history(sample: Dict[str, Any]) -> bool:
    queries = sample.get("queries")
    if not isinstance(queries, list):
        queries = ((sample.get("input") or {}).get("queries") or [])
    for q in queries or []:
        if isinstance(q, dict) and q.get("answers"):
            return True
    return False


def _future_answer_chunks(sample: Dict[str, Any], chunk_idx: int) -> List[int]:
    return [c for c in _answer_chunks(sample) if int(c) > int(chunk_idx)]


def _record_response_timing_example(
    stats: Dict[str, Any],
    name: str,
    item: Dict[str, Any],
    *,
    limit: int = 12,
) -> None:
    examples = stats.setdefault(name, [])
    if len(examples) < limit:
        examples.append(item)


def _record_response_timing_stats(
    *,
    runner: DaggerRunner,
    visual_entries: List[Tuple[str, Dict[str, Any]]],
    result: Dict[str, Any],
    stats: Dict[str, Any],
) -> None:
    chunk_idx = int(result.get("chunk_idx", runner.current_chunk) or 0)
    if chunk_idx in (runner.question_at_chunk or {}):
        _close_pending_missed_responses(
            runner,
            stats,
            reason="new_query",
            close_chunk=chunk_idx,
        )

    final_action = str(result.get("final_action") or result.get("action") or "unknown")
    policy_answer = _policy_answer(result)
    response_targets = [
        (key, sample)
        for key, sample in visual_entries
        if str(sample.get("action") or sample.get("sample_type") or "") == "response"
    ]
    active_future = [
        (key, sample)
        for key, sample in visual_entries
        if _future_answer_chunks(sample, chunk_idx)
    ]
    if response_targets:
        _inc_stat(stats, "response_timing_gold_targets")
        _inc_stat(stats, "response_event_gold_total")
    if final_action == "response" and policy_answer:
        if response_targets:
            key, sample = response_targets[0]
            gold_answer = _gold_answer(sample)
            matched = bool(_answer_matches(gold_answer, policy_answer))
            _inc_stat(stats, "response_on_time")
            _inc_stat(stats, "response_on_time_correct" if matched else "response_on_time_wrong")
            _inc_stat(stats, "response_event_matched_correct_on_time" if matched else "response_event_matched_wrong_on_time")
            _record_response_timing_example(
                stats,
                "response_on_time_examples",
                {
                    "traj_i": runner.traj_i,
                    "video_path": runner.video_path,
                    "chunk": chunk_idx,
                    "gold_answer": gold_answer,
                    "policy_answer": policy_answer[:240],
                    "matched": matched,
                    "question": _question_text(sample)[:240],
                },
            )
        elif active_future:
            key, sample = active_future[0]
            future = _future_answer_chunks(sample, chunk_idx)
            expected = min(future) if future else None
            lead = int(expected) - chunk_idx if expected is not None else None
            gold_answer = _gold_answer(sample)
            matched = bool(_answer_matches(gold_answer, policy_answer))
            _inc_stat(stats, "response_early")
            _inc_stat(stats, "response_event_false_positive_early")
            if lead is not None:
                _bucket_stat(stats, "response_early_lead_chunks", str(lead))
            _inc_stat(stats, "response_early_correct_text" if matched else "response_early_wrong_text")
            _record_response_timing_example(
                stats,
                "response_early_examples",
                {
                    "traj_i": runner.traj_i,
                    "video_path": runner.video_path,
                    "chunk": chunk_idx,
                    "expected_chunk": expected,
                    "lead_chunks": lead,
                    "gold_answer": gold_answer,
                    "policy_answer": policy_answer[:240],
                    "matched": matched,
                    "question": _question_text(sample)[:240],
                },
            )
        else:
            _inc_stat(stats, "response_over_emit")
            _inc_stat(stats, "response_event_false_positive_over")
            _record_response_timing_example(
                stats,
                "response_over_emit_examples",
                {
                    "traj_i": runner.traj_i,
                    "video_path": runner.video_path,
                    "chunk": chunk_idx,
                    "policy_answer": policy_answer[:240],
                    "action": final_action,
                },
            )
    elif final_action == "silent":
        if active_future:
            if any(_sample_has_response_history(sample) for _, sample in active_future):
                _inc_stat(stats, "silent_after_partial_ok")
            else:
                _inc_stat(stats, "silent_before_answer_ok")

    if final_action == "response" and policy_answer:
        pending = _match_pending_missed_response(
            runner.pending_missed_responses,
            policy_answer,
        )
        if pending is not None and chunk_idx > int(pending.get("miss_chunk", chunk_idx)):
            delay = chunk_idx - int(pending.get("miss_chunk", chunk_idx))
            _inc_stat(stats, "response_late")
            _inc_stat(stats, "missed_response_late_attempts")
            _bucket_stat(stats, "response_late_delay_chunks", str(delay))
            _bucket_stat(stats, "missed_response_late_delay_chunks", str(delay))
            matched = bool(_answer_matches(str(pending.get("gold_answer") or ""), policy_answer))
            if matched:
                _inc_stat(stats, "response_late_correct")
                _inc_stat(stats, "response_event_matched_correct_late")
                _inc_stat(stats, "missed_response_late_correct")
                pending["resolved"] = True
                pending["closed"] = True
                pending["close_reason"] = "late_correct_response"
            else:
                _inc_stat(stats, "response_late_wrong")
                _inc_stat(stats, "response_event_matched_wrong_late")
                _inc_stat(stats, "missed_response_late_wrong")
            _record_missed_response_example(
                stats,
                "missed_response_late_examples",
                {
                    "traj_i": runner.traj_i,
                    "video_path": runner.video_path,
                    "miss_chunk": pending.get("miss_chunk"),
                    "answer_chunk": chunk_idx,
                    "delay_chunks": delay,
                    "gold_answer": pending.get("gold_answer"),
                    "policy_answer": policy_answer[:240],
                    "matched": matched,
                    "question": str(pending.get("question") or "")[:240],
                },
            )

    for key, sample in visual_entries:
        if str(sample.get("action") or sample.get("sample_type") or "") != "response":
            continue
        if final_action == "response" and policy_answer:
            continue
        gold_answer = _gold_answer(sample)
        chunks = _answer_chunks(sample)
        pending = {
            "key": key,
            "traj_i": runner.traj_i,
            "miss_chunk": chunk_idx,
            "answer_chunks": chunks,
            "gold_answer": gold_answer,
            "question": _question_text(sample),
            "policy_action": final_action,
            "first_action": str(result.get("action") or "unknown"),
            "format_ok": bool(result.get("format_ok", True)),
        }
        runner.pending_missed_responses.append(pending)
        _inc_stat(stats, "response_missed")
        _inc_stat(stats, "response_event_missed_initial")
        _inc_stat(stats, "missed_response_timing_targets")
        _bucket_stat(stats, "missed_response_timing_policy_action", final_action)
        _record_missed_response_example(
            stats,
            "missed_response_timing_examples",
            {
                "traj_i": runner.traj_i,
                "video_path": runner.video_path,
                "miss_chunk": chunk_idx,
                "answer_chunks": chunks,
                "gold_answer": gold_answer,
                "policy_action": final_action,
                "question": pending["question"][:240],
            },
        )


def _result_format_ok(result: Dict[str, Any]) -> bool:
    action = str(result.get("action") or "")
    if action in {"unknown", "invalid"}:
        return False
    if result.get("action_space_error") or result.get("format_error"):
        return False
    if not str(result.get("think") or "").strip():
        return False
    payload = result.get("payload") or {}
    if action == "silent":
        return True
    if action == "response":
        return bool(str(payload.get("response") or "").strip())
    if action == "recall":
        return isinstance(payload.get("query"), dict)
    if action == "compress":
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            return False
        return (
            _normalise_range(summary.get("time_range")) is not None
            and bool(str(summary.get("text") or "").strip())
        )
    return False


def _compress_diag_summary(stats: Dict[str, Any]) -> str:
    parts = [
        f"forced={stats.get('forced_gold_compress_turns', 0)}",
        f"forced_targets={stats.get('forced_gold_compress_targets', 0)}",
    ]
    if stats.get("gold_compress_targets_seen"):
        parts.append(
            "gold="
            f"targets:{stats.get('gold_compress_targets_seen', 0)},"
            f"natural:{stats.get('gold_compress_natural_trigger_targets', 0)},"
            f"miss:{stats.get('gold_compress_without_natural_trigger_targets', 0)},"
            f"miss_reasons:{_compact_counter(stats.get('gold_compress_without_trigger_reason') or {}, limit=3)}"
        )
    if stats.get("compress_turn_action_by_source"):
        parts.append(f"actions:{stats.get('compress_turn_action_by_source')}")
    if stats.get("compress_runtime_error_by_source"):
        parts.append(f"runtime_errors:{stats.get('compress_runtime_error_by_source')}")
    if stats.get("compress_recovery_by_source"):
        parts.append(f"recoveries:{stats.get('compress_recovery_by_source')}")
    if stats.get("recall_prepare_error_by_reason"):
        parts.append(f"recall_prepare:{stats.get('recall_prepare_error_by_reason')}")
    if stats.get("recall_retrieval_by_status"):
        parts.append(f"recall_retrieval:{stats.get('recall_retrieval_by_status')}")
    if stats.get("gold_recall_targets_seen"):
        parts.append(
            "gold_recall="
            f"targets:{stats.get('gold_recall_targets_seen', 0)},"
            f"policy_recall:{stats.get('gold_recall_policy_recall', 0)},"
            f"retrieval:{_compact_counter(stats.get('gold_recall_retrieval_status') or {}, limit=4)},"
            f"range:{_compact_counter(stats.get('gold_recall_query_time_range_status') or {}, limit=3)},"
            f"post:{_compact_counter(stats.get('gold_recall_post_action') or {}, limit=3)}"
        )
    if stats.get("policy_recall_on_non_recall_targets"):
        parts.append(
            "over_recall_targets="
            f"{stats.get('policy_recall_on_non_recall_targets', 0)},"
            f"gold:{_compact_counter(stats.get('policy_recall_on_non_recall_by_gold_type') or {}, limit=3)}"
        )
    if stats.get("post_recall_blocked_by_reason"):
        parts.append(f"post_recall_blocked:{stats.get('post_recall_blocked_by_reason')}")
    targets = int(stats.get("dagger_targets_seen", 0) or 0)
    if targets:
        first_acc = (stats.get("dagger_first_action_correct", 0) or 0) / max(targets, 1)
        final_acc = (stats.get("dagger_final_action_correct", 0) or 0) / max(targets, 1)
        format_ok = (stats.get("dagger_format_ok_targets", 0) or 0) / max(targets, 1)
        parts.append(
            "target_eval="
            f"targets:{targets},first_acc:{first_acc:.3f},"
            f"final_acc:{final_acc:.3f},format_ok:{format_ok:.3f},"
            f"errors:{_compact_counter(stats.get('dagger_primary_error_type') or {}, limit=5)}"
        )
    answer_targets = int(stats.get("dagger_answer_targets_seen", 0) or 0)
    if answer_targets:
        answer_acc = (stats.get("dagger_answer_correct", 0) or 0) / max(answer_targets, 1)
        emit_rate = (
            (stats.get("dagger_answer_correct", 0) or 0)
            + (stats.get("dagger_answer_wrong", 0) or 0)
        ) / max(answer_targets, 1)
        parts.append(
            "answer_eval="
            f"targets:{answer_targets},acc:{answer_acc:.3f},emit:{emit_rate:.3f},"
            f"by:{_compact_counter(stats.get('dagger_answer_eval') or {}, limit=5)}"
        )
    missed = int(stats.get("missed_response_timing_targets", 0) or 0)
    if missed:
        parts.append(
            "missed_timing="
            f"targets:{missed},attempts:{stats.get('missed_response_late_attempts', 0)},"
            f"correct:{stats.get('missed_response_late_correct', 0)},"
            f"wrong:{stats.get('missed_response_late_wrong', 0)},"
            f"unresolved:{stats.get('missed_response_unresolved', 0)},"
            f"delay:{_compact_counter(stats.get('missed_response_late_delay_chunks') or {}, limit=5)}"
        )
    if stats.get("response_timing_gold_targets") or stats.get("response_early") or stats.get("response_over_emit"):
        parts.append(
            "response_timing="
            f"gold:{stats.get('response_timing_gold_targets', 0)},"
            f"on:{stats.get('response_on_time', 0)},"
            f"early:{stats.get('response_early', 0)},"
            f"late:{stats.get('response_late', 0)},"
            f"miss:{stats.get('response_missed', 0)},"
            f"over:{stats.get('response_over_emit', 0)},"
            f"silent_wait:{stats.get('silent_before_answer_ok', 0)},"
            f"silent_partial:{stats.get('silent_after_partial_ok', 0)},"
            f"lead:{_compact_counter(stats.get('response_early_lead_chunks') or {}, limit=3)},"
            f"delay:{_compact_counter(stats.get('response_late_delay_chunks') or {}, limit=3)}"
        )
    event_gold = int(stats.get("response_event_gold_total", 0) or 0)
    if event_gold or stats.get("response_event_false_positive_early") or stats.get("response_event_false_positive_over"):
        matched_correct = (
            int(stats.get("response_event_matched_correct_on_time", 0) or 0)
            + int(stats.get("response_event_matched_correct_late", 0) or 0)
        )
        matched_wrong = (
            int(stats.get("response_event_matched_wrong_on_time", 0) or 0)
            + int(stats.get("response_event_matched_wrong_late", 0) or 0)
        )
        parts.append(
            "response_events="
            f"gold:{event_gold},tp:{matched_correct},wrong:{matched_wrong},"
            f"miss0:{stats.get('response_event_missed_initial', 0)},"
            f"unres:{stats.get('response_event_missed_unresolved', 0)},"
            f"fp_early:{stats.get('response_event_false_positive_early', 0)},"
            f"fp_over:{stats.get('response_event_false_positive_over', 0)}"
        )
    for key, label in [
        ("compress_prefix_by_turn", "turn"),
        ("compress_prefix_by_target", "target"),
    ]:
        bucket = stats.get(key) or {}
        total = int(bucket.get("total", 0) or 0)
        if not total:
            continue
        parts.append(
            f"{label}=total:{total},json:{bucket.get('json_complete', 0)},"
            f"trunc:{bucket.get('likely_truncated', 0)},"
            f"labels:{_compact_counter(bucket.get('by_label') or {}, limit=4)}"
        )
    reasons = stats.get("by_correction_reason") or {}
    selected = stats.get("by_selected_correction_reason") or {}
    compress_reasons = {
        k: reasons.get(k, 0)
        for k in [
            "missed_compress",
            "bad_compress_json",
            "bad_compress_range_schema",
            "bad_compress_range",
            "empty_compress_summary",
            "compress_good_prefix_truncated",
            "compress_good_prefix_unparsed",
            "compress_bad_prefix_after_tool_open",
        ]
        if reasons.get(k, 0)
    }
    selected_compress = {
        k: selected.get(k, 0)
        for k in [
            "missed_compress",
            "bad_compress_json",
            "bad_compress_range_schema",
            "bad_compress_range",
            "empty_compress_summary",
        ]
        if selected.get(k, 0)
    }
    if compress_reasons:
        parts.append(f"compress_reasons={compress_reasons}")
    if selected_compress:
        parts.append(f"selected_compress={selected_compress}")
    return " ".join(parts)


def _emit_entries(
    *,
    runner: DaggerRunner,
    entries: List[Tuple[str, Dict[str, Any]]],
    onpolicy_prompt: List[Dict[str, Any]],
    result: Dict[str, Any],
    fout,
    stats: Dict[str, Any],
    ckpt: str,
    data_dir: Path,
    frame_protocol: str,
    include_failed_targets: bool,
    sample_types: set[str],
    correction_only: bool,
    correction_reasons: set[str],
) -> int:
    rows_written = 0
    for key, sample in entries:
        wrote = _emit_dagger_row(
            sample=sample,
            onpolicy_prompt=onpolicy_prompt,
            result=result,
            fout=fout,
            stats=stats,
            ckpt=ckpt,
            data_dir=data_dir,
            frame_protocol=frame_protocol,
            include_failed_targets=include_failed_targets,
            sample_types=sample_types,
            correction_only=correction_only,
            correction_reasons=correction_reasons,
        )
        if wrote:
            runner.emitted.add(key)
            rows_written += 1
    return rows_written


def _build_recall_messages(
    runner: DaggerRunner,
    first_messages: List[Dict[str, Any]],
    first_text: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], Optional[Dict[str, Any]]]:
    parsed = _parse_agent_output(first_text)
    query = parsed.get("payload", {}).get("query", {})
    if runner.retriever is None:
        return None, {"error_reason": "missing_retriever"}, None
    if not isinstance(query, dict) or not str(query.get("query") or "").strip():
        return None, {"error_reason": "empty_recall_query"}, None

    think_text = runner.chunk_results[-1].get("think", "")
    if think_text:
        try:
            runner.retriever.index_chunk(runner.current_chunk, runner.video_path, think_text)
        except Exception:
            pass

    recall_result = runner.retriever(query, runner.memory.retrieval_archive)
    returned_chunks = select_recall_chunks(recall_result.get("returned_chunks", []))
    recall_result["returned_chunks"] = returned_chunks
    runner.chunk_results[-1]["recall_returned_chunks"] = list(returned_chunks)

    recalled_frames = None
    if returned_chunks and recall_result.get("source") == "historical_frames":
        from thinkstream.eval.streaming_vllm import _resolve_chunk_frame_paths

        rf_paths: List[str] = []
        frame_chunks: List[int] = []
        for rc in returned_chunks:
            cf = _resolve_chunk_frame_paths(
                runner.video_path, rc, runner.frames_root, runner.video_root,
            )
            if cf:
                frame_chunks.append(rc)
                rf_paths.extend(cf)
        recalled_frames = build_recalled_frames_metadata(
            frame_chunks or returned_chunks,
            rf_paths,
            chunk_sec=AGENT_CHUNK_SEC,
            frames_per_chunk=FRAMES_PER_CHUNK,
        )

    messages = deepcopy(first_messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt_for_frame_protocol(
                    runner.frame_protocol,
                    prompt_kind="post_recall",
                    render_layout=runner.render_layout,
                ),
            }],
        }
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": first_text}],
    })
    tool_user_content: List[Dict[str, Any]] = build_recall_result_user_content(
        recalled_frames,
        recall_result,
        frame_protocol=runner.frame_protocol,
        min_pixels=runner.min_pixels,
        max_pixels=runner.max_pixels,
        render_layout=runner.render_layout,
    )
    messages.append({"role": "user", "content": tool_user_content})
    return messages, recall_result, recalled_frames


def build_dagger_vllm(
    *,
    ckpt: str,
    trajectories: Path,
    out: Path,
    data_dir: Path,
    frames_root: str,
    video_root: Optional[str],
    frame_protocol: str,
    render_layout: str,
    sample_types: set[str],
    include_failed_targets: bool,
    correction_only: bool,
    correction_reasons: set[str],
    max_trajectories: int,
    max_rows: int,
    max_steps: int,
    num_shards: int,
    shard_index: int,
    rollout_batch_size: int,
    max_new_tokens: int,
    compress_max_new_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_images_per_prompt: int,
    max_videos_per_prompt: int,
    mm_processor_cache_gb: Optional[int],
    max_compress_turns_per_chunk: int,
    oracle_compress_recovery: bool,
    rollout_all_chunks: bool,
    rollout_scheduler: str,
    force_gold_compress_trigger: bool,
    log_every_steps: int,
) -> Dict[str, Any]:
    # Heavy rollout imports are intentionally lazy so `--help` and py_compile
    # work in shells that do not have CUDA/flash-attn loaded.
    from thinkstream.eval.streaming_vllm import (
        _apply_rollout_output,
        _prepare_step_messages,
    )
    from thinkstream.eval.vllm_engine import (
        generate_with_turn_sampling,
        init_vllm_engine,
        make_sampling_params,
        prepare_vllm_input,
    )

    frame_protocol = normalize_frame_protocol(frame_protocol)
    render_layout = normalize_render_layout(render_layout)
    processor = load_processor_for_checkpoint(ckpt)
    data_args = DataArguments()
    data_args.min_pixels = int(
        os.environ.get("IMAGE_MIN_PIXELS", os.environ.get("MIN_PIXELS", data_args.min_pixels))
    )
    data_args.max_pixels = int(
        os.environ.get("IMAGE_MAX_PIXELS", os.environ.get("MAX_PIXELS", data_args.max_pixels))
    )
    processor = update_processor_pixels(processor, data_args)
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt,
        model_max_length=max_model_len,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_tokens(
        [
            t for t in processor.tokenizer.get_added_vocab().keys()
            if t not in tokenizer.get_vocab()
        ],
        special_tokens=True,
    )
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    llm = init_vllm_engine(
        ckpt,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_images_per_prompt=max_images_per_prompt,
        max_videos_per_prompt=max_videos_per_prompt,
        enable_prefix_caching=True,
        mm_processor_cache_gb=mm_processor_cache_gb,
    )
    sampling_params = make_sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        repetition_penalty=1.05,
    )
    compress_sampling_params = make_sampling_params(
        max_new_tokens=compress_max_new_tokens,
        temperature=0.0,
        top_k=1,
        repetition_penalty=1.05,
    )

    runners, base_stats = _make_runners(
        trajectories,
        data_dir=data_dir,
        frames_root=frames_root,
        video_root=video_root,
        tokenizer=tokenizer,
        frame_protocol=frame_protocol,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
        max_trajectories=max_trajectories,
        num_shards=num_shards,
        shard_index=shard_index,
        rollout_all_chunks=rollout_all_chunks,
        render_layout=render_layout,
    )
    stats: Dict[str, Any] = {
        **base_stats,
        "steps": 0,
        "rows": 0,
        "by_type": {},
        "by_correction_reason": {},
        "by_selected_correction_reason": {},
        "rollout_scheduler": rollout_scheduler,
        "force_gold_compress_trigger": bool(force_gold_compress_trigger),
        "step_errors": 0,
        "policy_compress_turns": 0,
        "forced_gold_compress_turns": 0,
        "forced_gold_compress_targets": 0,
        "visual_retries_after_compress": 0,
        "recall_step2_blocked": 0,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    next_runner_idx = 0
    with out.open("w") as fout:
        while True:
            live = [r for r in runners if not r.done]
            if not live:
                break
            batch, next_runner_idx = _select_rollout_batch(
                runners,
                start_idx=next_runner_idx,
                batch_size=rollout_batch_size,
                scheduler=rollout_scheduler,
            )
            active: List[Tuple[DaggerRunner, List[Dict[str, Any]], Dict[str, Any]]] = []
            for r in batch:
                if r.current_chunk > r.max_chunk:
                    r.done = True
                    continue
                try:
                    messages = _prepare_step_messages(r)
                    compress_entries, _ = _chunk_entries(r)
                    natural_compress = _prompt_has_compress_trigger(messages)
                    pre_memory_tokens = r.memory.count_recent_tokens()
                    pre_recent_thinks = len(r.memory.recent_thinks)
                    forced_compress = False
                    if (
                        force_gold_compress_trigger
                        and compress_entries
                        and "compress" in sample_types
                        and not natural_compress
                    ):
                        messages = _prepare_forced_compress_messages(r)
                        forced_compress = True
                        stats["forced_gold_compress_turns"] += 1
                        stats["forced_gold_compress_targets"] += len(compress_entries)
                    if compress_entries:
                        _record_gold_compress_state(
                            stats,
                            r,
                            target_count=len(compress_entries),
                            natural_trigger=natural_compress,
                            forced_trigger=forced_compress,
                        )
                except Exception as exc:
                    r.done = True
                    r.error = f"prepare:{type(exc).__name__}:{exc}"
                    stats["step_errors"] += 1
                    stats["skipped"]["prepare_error"] = stats["skipped"].get("prepare_error", 0) + 1
                    continue
                active.append((
                    r,
                    deepcopy(messages),
                    {
                        "forced_compress": forced_compress,
                        "natural_compress": natural_compress,
                        "pre_memory_tokens": pre_memory_tokens,
                        "pre_recent_thinks": pre_recent_thinks,
                        "pre_recent_thinks_snapshot": deepcopy(r.memory.recent_thinks),
                    },
                ))
            if not active:
                continue

            try:
                turn_kinds = [
                    "compress" if _prompt_has_compress_trigger(m) else "streaming"
                    for _, m, _ in active
                ]
                vllm_inputs = [
                    prepare_vllm_input(
                        m,
                        processor,
                        tools=tools_for_turn(turn_kind),
                    )
                    for (_, m, _), turn_kind in zip(active, turn_kinds)
                ]
                outputs = generate_with_turn_sampling(
                    llm,
                    vllm_inputs,
                    turn_kinds,
                    sampling_params,
                    {"compress": compress_sampling_params},
                )
            except Exception as exc:
                stats["step_errors"] += len(active)
                stats["skipped"]["generate_error"] = stats["skipped"].get("generate_error", 0) + len(active)
                for r, _, _ in active:
                    r.done = True
                    r.error = f"generate:{type(exc).__name__}:{exc}"
                continue

            recall_batch: List[Tuple[DaggerRunner, List[Dict[str, Any]], str, Dict[str, Any]]] = []
            pending: List[Tuple[DaggerRunner, List[Dict[str, Any]], Dict[str, Any], bool]] = []
            for (r, messages, meta), out_item in zip(active, outputs):
                text = out_item.outputs[0].text
                prompt_is_compress = _prompt_has_compress_trigger(messages)
                try:
                    if prompt_is_compress:
                        _apply_compress_turn_output(
                            r,
                            text,
                            tokenizer,
                            pre_recent_thinks=list(meta.get("pre_recent_thinks_snapshot") or []),
                            pre_memory_tokens=int(meta.get("pre_memory_tokens") or 0),
                            compress_budget=RECENT_THINKS_TOKEN_BUDGET,
                        )
                    else:
                        _apply_rollout_output(
                            r,
                            text,
                            tokenizer,
                            compress_budget=RECENT_THINKS_TOKEN_BUDGET,
                        )
                    result = r.chunk_results[-1]
                    result["format_ok"] = _result_format_ok(result)
                    forced_compress = bool(meta.get("forced_compress"))
                    result["forced_compress_trigger"] = forced_compress
                    result["compress_trigger_source"] = (
                        "forced_gold"
                        if forced_compress
                        else ("natural" if prompt_is_compress else "")
                    )
                    result["pre_compress_memory_token_count"] = (
                        meta.get("pre_memory_tokens")
                        if prompt_is_compress
                        else None
                    )
                    result["pre_compress_recent_thinks"] = (
                        meta.get("pre_recent_thinks")
                        if prompt_is_compress
                        else None
                    )
                    if prompt_is_compress:
                        result["post_compress_memory_token_count"] = r.memory.count_recent_tokens()
                        result["compress_memory_token_delta"] = (
                            int(meta.get("pre_memory_tokens") or 0)
                            - int(result.get("post_compress_memory_token_count") or 0)
                        )
                    stats["steps"] += 1
                except Exception as exc:
                    if not prompt_is_compress:
                        r.done = True
                        r.error = f"apply:{type(exc).__name__}:{exc}"
                        stats["step_errors"] += 1
                        stats["skipped"]["apply_error"] = stats["skipped"].get("apply_error", 0) + 1
                        continue
                    result = {
                        "chunk_idx": r.current_chunk,
                        "action": "invalid",
                        "think": "",
                        "payload": {},
                        "raw_output": text,
                        "compress_prefix_diagnostic": diagnose_compress_output_v12(text),
                        "action_space_error": f"apply_error:{type(exc).__name__}",
                        "invalid_action": "apply_error",
                        "generated_tokens": tokenizer.encode(text, add_special_tokens=False),
                        "memory_token_count": r.memory.count_recent_tokens(),
                        "compress_budget": RECENT_THINKS_TOKEN_BUDGET,
                        "recall_returned_chunks": [],
                        "forced_compress_trigger": bool(meta.get("forced_compress")),
                        "compress_trigger_source": (
                            "forced_gold"
                            if bool(meta.get("forced_compress"))
                            else "natural"
                        ),
                        "pre_compress_memory_token_count": meta.get("pre_memory_tokens"),
                        "pre_compress_recent_thinks": meta.get("pre_recent_thinks"),
                        "post_compress_memory_token_count": r.memory.count_recent_tokens(),
                        "compress_memory_token_delta": 0,
                        "compress_runtime_error": f"apply_error:{type(exc).__name__}",
                        "compress_applied": False,
                        "compress_applied_chunks": [],
                        "format_ok": False,
                    }
                    r.chunk_results.append(result)
                    stats["steps"] += 1
                pending.append((r, messages, result, prompt_is_compress))
                if result.get("action") == "recall":
                    recall_batch.append((r, messages, text, result))

            if recall_batch:
                recall_active = []
                for r, messages, first_text, result in recall_batch:
                    stats["recall_first_turns"] = stats.get("recall_first_turns", 0) + 1
                    try:
                        recall_messages, recall_result, _ = _build_recall_messages(
                            r, messages, first_text,
                        )
                    except Exception as exc:
                        r.error = f"recall_prepare:{type(exc).__name__}:{exc}"
                        result["recall_prepare_error_reason"] = f"exception:{type(exc).__name__}"
                        stats["recall_prepare_errors"] = stats.get("recall_prepare_errors", 0) + 1
                        reason_bucket = stats.setdefault("recall_prepare_error_by_reason", {})
                        reason_bucket[result["recall_prepare_error_reason"]] = (
                            reason_bucket.get(result["recall_prepare_error_reason"], 0) + 1
                        )
                        continue
                    if recall_messages is None:
                        reason = str((recall_result or {}).get("error_reason") or "unknown")
                        result["recall_prepare_error_reason"] = reason
                        stats["recall_prepare_errors"] = stats.get("recall_prepare_errors", 0) + 1
                        reason_bucket = stats.setdefault("recall_prepare_error_by_reason", {})
                        reason_bucket[reason] = reason_bucket.get(reason, 0) + 1
                        continue
                    result["recall_messages"] = deepcopy(recall_messages)
                    result["recall_result"] = recall_result
                    returned = list((recall_result or {}).get("returned_chunks") or [])
                    result["recall_returned_chunks"] = returned
                    retrieval_status = "nonempty" if returned else "empty"
                    bucket = stats.setdefault("recall_retrieval_by_status", {})
                    bucket[retrieval_status] = bucket.get(retrieval_status, 0) + 1
                    recall_active.append((r, result, recall_messages, recall_result))
                if recall_active:
                    try:
                        rc_inputs = [
                            prepare_vllm_input(
                                m,
                                processor,
                                tools=tools_for_turn("post_recall"),
                            )
                            for _, _, m, _ in recall_active
                        ]
                        rc_outputs = llm.generate(rc_inputs, sampling_params=sampling_params)
                    except Exception as exc:
                        for r, _, _, _ in recall_active:
                            r.error = f"recall_generate:{type(exc).__name__}:{exc}"
                    else:
                        for (r, result, _, recall_result), rc_out in zip(recall_active, rc_outputs):
                            rc_text = rc_out.outputs[0].text
                            rc_parsed = _parse_agent_output(rc_text)
                            recall_action_error = action_space_error_for_turn(
                                rc_parsed.get("action", ""),
                                "post_recall",
                            )
                            if recall_action_error or rc_parsed.get("action") in ("recall", "compress"):
                                stats["recall_step2_blocked"] += 1
                                block_reason = (
                                    recall_action_error
                                    or f"disallowed_action:{rc_parsed.get('action') or 'unknown'}"
                                )
                                block_bucket = stats.setdefault("post_recall_blocked_by_reason", {})
                                block_bucket[block_reason] = block_bucket.get(block_reason, 0) + 1
                                result["recall_step2_blocked"] = {
                                    "action": rc_parsed.get("action"),
                                    "action_space_error": recall_action_error,
                                    "reason": block_reason,
                                    "raw_output": rc_text,
                                }
                                rc_parsed = {
                                    "action": "silent",
                                    "payload": {},
                                    "raw_output": "",
                                }
                            result["recall_step2_raw_text"] = rc_text
                            result["recall_result"] = recall_result
                            post_action = str(rc_parsed.get("action") or "unknown")
                            post_bucket = stats.setdefault("post_recall_action", {})
                            post_bucket[post_action] = post_bucket.get(post_action, 0) + 1
                            if rc_parsed.get("action") in ("response", "silent"):
                                result["final_action"] = rc_parsed.get("action")
                                result["final_payload"] = rc_parsed.get("payload", {})
                            if rc_parsed.get("action") == "response":
                                ans = rc_parsed.get("payload", {}).get("response", "")
                                r._record_answer_to_memory(ans, r.current_chunk)

            for r, messages, result, prompt_is_compress in pending:
                compress_entries, visual_entries = _chunk_entries(r)
                if prompt_is_compress:
                    stats["policy_compress_turns"] += 1
                    _record_compress_diag_stats(
                        stats, result, prefix="compress_prefix_by_turn",
                    )
                    _record_compress_turn_result(
                        stats,
                        result,
                        source=str(result.get("compress_trigger_source") or "natural"),
                    )
                    compress_rows = _emit_entries(
                        runner=r,
                        entries=compress_entries,
                        onpolicy_prompt=messages,
                        result=result,
                        fout=fout,
                        stats=stats,
                        ckpt=ckpt,
                        data_dir=data_dir,
                        frame_protocol=frame_protocol,
                        include_failed_targets=include_failed_targets,
                        sample_types=sample_types,
                        correction_only=correction_only,
                        correction_reasons=correction_reasons,
                    )
                    if (
                        result.get("forced_compress_trigger")
                        and compress_entries
                        and compress_rows == 0
                        and result.get("action") == "compress"
                        and result.get("compress_applied")
                    ):
                        for key, _ in compress_entries:
                            r.emitted.add(key)
                        stats["skipped"]["correct_forced_compress_no_correction"] = (
                            stats["skipped"].get("correct_forced_compress_no_correction", 0)
                            + len(compress_entries)
                        )
                    # Compress is an inter-chunk management turn.  A successful
                    # or recovered compress should retry the same video chunk so
                    # the policy can observe it under the updated memory state.
                    n = r.compress_retries.get(r.current_chunk, 0) + 1
                    r.compress_retries[r.current_chunk] = n
                    if result.get("compress_applied"):
                        if n <= max_compress_turns_per_chunk:
                            stats["visual_retries_after_compress"] += 1
                            continue
                        stats["skipped"]["too_many_policy_compress_turns"] = (
                            stats["skipped"].get("too_many_policy_compress_turns", 0)
                            + max(1, len(visual_entries))
                        )
                    else:
                        failure_reason = str(result.get("compress_runtime_error") or result.get("action") or "unknown")
                        stats["skipped"]["policy_failed_compress_before_visual"] = (
                            stats["skipped"].get("policy_failed_compress_before_visual", 0)
                            + max(1, len(visual_entries))
                        )
                        key = f"policy_failed_compress_before_visual:{failure_reason}"
                        stats["skipped"][key] = stats["skipped"].get(key, 0) + max(1, len(visual_entries))
                        if n <= max_compress_turns_per_chunk:
                            recovery = _apply_compress_recovery(
                                r,
                                compress_entries,
                                stats,
                                source=str(result.get("compress_trigger_source") or "natural"),
                                failure_reason=failure_reason,
                                allow_oracle=oracle_compress_recovery,
                            )
                            if recovery:
                                result["compress_recovery"] = recovery
                                result["post_compress_memory_token_count"] = r.memory.count_recent_tokens()
                                result["compress_memory_token_delta"] = (
                                    int(result.get("pre_compress_memory_token_count") or 0)
                                    - int(result.get("post_compress_memory_token_count") or 0)
                                )
                                stats["visual_retries_after_compress_recovery"] = (
                                    stats.get("visual_retries_after_compress_recovery", 0) + 1
                                )
                                continue
                    r.current_chunk += 1
                else:
                    _record_response_timing_stats(
                        runner=r,
                        visual_entries=visual_entries,
                        result=result,
                        stats=stats,
                    )
                    _emit_entries(
                        runner=r,
                        entries=visual_entries,
                        onpolicy_prompt=messages,
                        result=result,
                        fout=fout,
                        stats=stats,
                        ckpt=ckpt,
                        data_dir=data_dir,
                        frame_protocol=frame_protocol,
                        include_failed_targets=include_failed_targets,
                        sample_types=sample_types,
                        correction_only=correction_only,
                        correction_reasons=correction_reasons,
                    )
                    if compress_entries:
                        stats["skipped"]["compress_target_without_trigger"] = (
                            stats["skipped"].get("compress_target_without_trigger", 0)
                            + len(compress_entries)
                        )
                    r.current_chunk += 1

                if r.current_chunk > r.max_chunk:
                    r.done = True

            if max_rows and stats["rows"] >= max_rows:
                for r in runners:
                    r.done = True
            if max_steps and stats["steps"] >= max_steps:
                for r in runners:
                    r.done = True
            if log_every_steps and stats["steps"] % log_every_steps < rollout_batch_size:
                rate = stats["steps"] / max(time.time() - t0, 1e-6)
                print(
                    f"[steps={stats['steps']}] rows={stats['rows']} "
                    f"live={sum(1 for r in runners if not r.done)} "
                    f"compress_turns={stats['policy_compress_turns']} "
                    f"rate={rate:.3f} step/s skipped={stats['skipped']} "
                    f"compress_diag={_compress_diag_summary(stats)}",
                    flush=True,
                )
                fout.flush()

    for r in runners:
        _close_pending_missed_responses(
            r,
            stats,
            reason="rollout_end",
            close_chunk=r.current_chunk,
        )
    finalize_dagger_stats(stats)
    stats["out"] = str(out)
    stats["elapsed_sec"] = round(time.time() - t0, 3)
    return stats


def main() -> None:
    batch_root = _default_batch_root(None)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--trajectories", default=str(batch_root / "final" / "train_sft_trajectories.jsonl"))
    p.add_argument("--out", default=None)
    p.add_argument("--data-dir", default=str(batch_root))
    p.add_argument("--frames-root", default=str(batch_root / "frames"))
    p.add_argument("--video-root", default=None)
    p.add_argument("--frame-protocol", default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"))
    p.add_argument("--render-layout", default=os.environ.get("THINKSTREAM_RENDER_LAYOUT", "standard"))
    p.add_argument("--sample-types", default="silent,response,recall,compress")
    p.add_argument("--include-failed-targets", action="store_true")
    p.add_argument(
        "--correction-only",
        action="store_true",
        help="Emit only rows whose rollout matches selected correction reasons.",
    )
    p.add_argument(
        "--correction-reasons",
        default=",".join(sorted(DEFAULT_DAGGER_CORRECTION_REASONS)),
        help="Comma-separated correction reasons; 'default' or 'all' are accepted.",
    )
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--rollout-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--compress-max-new-tokens",
        type=int,
        default=512,
        help="Generation budget for compression turns. Normal streaming/recall turns use --max-new-tokens.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-images-per-prompt", type=int, default=64)
    p.add_argument("--max-videos-per-prompt", type=int, default=2)
    p.add_argument(
        "--mm-processor-cache-gb",
        type=int,
        default=None,
        help="vLLM CPU multimodal preprocessor cache GB. Defaults to auto/env.",
    )
    p.add_argument("--max-compress-turns-per-chunk", type=int, default=2)
    p.add_argument(
        "--no-oracle-compress-recovery",
        action="store_true",
        help=(
            "Disable DAgger recovery that applies the gold compress target "
            "after a missed system-compress turn before retrying the same visual chunk."
        ),
    )
    p.add_argument("--target-chunks-only", action="store_true")
    p.add_argument(
        "--rollout-scheduler",
        choices=["round_robin", "head"],
        default="round_robin",
        help="Runner scheduling policy. 'head' preserves the legacy live[:batch] behavior.",
    )
    p.add_argument(
        "--no-force-gold-compress-trigger",
        action="store_true",
        help=(
            "Do not synthesize a system compress turn when a gold compress "
            "target reaches the current chunk but runtime memory has not "
            "naturally crossed the compression threshold."
        ),
    )
    p.add_argument("--log-every-steps", type=int, default=20)
    args = p.parse_args()

    sample_types = {x.strip() for x in args.sample_types.split(",") if x.strip()}
    correction_reasons = _parse_reason_set(args.correction_reasons)
    render_layout = normalize_render_layout(args.render_layout)
    os.environ["THINKSTREAM_RENDER_LAYOUT"] = render_layout
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    out_path = (
        _resolve_path(args.out)
        if args.out
        else batch_root / "rendered" / (
            args.frame_protocol
            if render_layout == "standard"
            else f"{args.frame_protocol}_{render_layout}"
        ) / "train_sft_dagger_messages.jsonl"
    )

    stats = build_dagger_vllm(
        ckpt=args.ckpt,
        trajectories=_resolve_path(args.trajectories),
        out=out_path,
        data_dir=_resolve_path(args.data_dir),
        frames_root=str(_resolve_path(args.frames_root)),
        video_root=str(_resolve_path(args.video_root)) if args.video_root else None,
        frame_protocol=normalize_frame_protocol(args.frame_protocol),
        render_layout=render_layout,
        sample_types=sample_types,
        include_failed_targets=args.include_failed_targets,
        correction_only=args.correction_only,
        correction_reasons=correction_reasons,
        max_trajectories=args.max_trajectories,
        max_rows=args.max_rows,
        max_steps=args.max_steps,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        compress_max_new_tokens=args.compress_max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_images_per_prompt=args.max_images_per_prompt,
        max_videos_per_prompt=args.max_videos_per_prompt,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        max_compress_turns_per_chunk=args.max_compress_turns_per_chunk,
        oracle_compress_recovery=not args.no_oracle_compress_recovery,
        rollout_all_chunks=not args.target_chunks_only,
        rollout_scheduler=args.rollout_scheduler,
        force_gold_compress_trigger=not args.no_force_gold_compress_trigger,
        log_every_steps=args.log_every_steps,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
