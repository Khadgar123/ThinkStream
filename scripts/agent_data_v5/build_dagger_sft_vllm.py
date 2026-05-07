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
missed recall, missed response, wrong response, or early answer.
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
    _content_text,
    _default_batch_root,
    _emit_dagger_row,
    _group_by_chunk,
    _iter_trajectory_rows,
    _new_question,
    _prompt_has_compress_trigger,
    _propagate_sample_fields,
    _question_meta,
    _parse_reason_set,
    _resolve_path,
    _resolve_video_path,
)
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    append_visual_frames,
    build_recalled_frames_metadata,
    has_compress_trigger,
    normalize_frame_protocol,
    select_recall_chunks,
    tools_for_turn,
)
from thinkstream.model.agent_loop import (
    MemoryState,
    RECENT_THINKS_TOKEN_BUDGET,
    _parse_agent_output,
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
    current_chunk: int = 0
    done: bool = False
    error: Optional[str] = None
    _last_trigger: bool = False
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    retriever: Optional[BM25Retriever] = field(default_factory=BM25Retriever)
    emitted: set[str] = field(default_factory=set)
    compress_retries: Dict[int, int] = field(default_factory=dict)

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
            if status in {"open", "pending", "active"} or not q.get("answers"):
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
    max_trajectories: int,
    num_shards: int,
    shard_index: int,
    rollout_all_chunks: bool,
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
            min_pixels=130_000,
            max_pixels=220_000,
            frame_protocol=frame_protocol,
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
) -> None:
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


def _build_recall_messages(
    runner: DaggerRunner,
    first_messages: List[Dict[str, Any]],
    first_text: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], Optional[Dict[str, Any]]]:
    parsed = _parse_agent_output(first_text)
    query = parsed.get("payload", {}).get("query", {})
    if not query or runner.retriever is None:
        return None, {}, None

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

    messages = list(first_messages)
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": first_text}],
    })
    tool_user_content: List[Dict[str, Any]] = []
    if recalled_frames:
        rf_header = json.dumps({
            "time_range": recalled_frames["time_range"],
            "source": recalled_frames.get("source", "historical_frames"),
            "n_frames": recalled_frames["n_frames"],
        })
        tool_user_content.append({
            "type": "text",
            "text": f"<recalled_frames>{rf_header}</recalled_frames>",
        })
        if recalled_frames.get("frame_paths"):
            tr_start, tr_end = recalled_frames["time_range"]
            append_visual_frames(
                tool_user_content,
                recalled_frames["frame_paths"],
                frame_protocol=runner.frame_protocol,
                fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
                start_frame_index=int(tr_start) * FRAMES_PER_CHUNK,
                total_num_frames=int(tr_end) * FRAMES_PER_CHUNK,
                context_label="recalled frame",
                min_pixels=runner.min_pixels,
                max_pixels=runner.max_pixels,
            )
    rr_json = json.dumps({
        "source": recall_result.get("source", ""),
        "time": recall_result.get("time", ""),
        "text": recall_result.get("text_content", recall_result.get("text", "")),
    }, ensure_ascii=False)
    tool_user_content.append({
        "type": "text",
        "text": f"<recall_result>{rr_json}</recall_result>",
    })
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
    sample_types: set[str],
    include_failed_targets: bool,
    correction_only: bool,
    correction_reasons: set[str],
    max_trajectories: int,
    max_rows: int,
    num_shards: int,
    shard_index: int,
    rollout_batch_size: int,
    max_new_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_images_per_prompt: int,
    max_videos_per_prompt: int,
    max_compress_turns_per_chunk: int,
    rollout_all_chunks: bool,
    log_every_steps: int,
) -> Dict[str, Any]:
    # Heavy rollout imports are intentionally lazy so `--help` and py_compile
    # work in shells that do not have CUDA/flash-attn loaded.
    from thinkstream.eval.streaming_vllm import (
        _apply_rollout_output,
        _prepare_step_messages,
    )
    from thinkstream.eval.vllm_engine import (
        init_vllm_engine,
        make_sampling_params,
        prepare_vllm_input,
    )

    frame_protocol = normalize_frame_protocol(frame_protocol)
    processor = load_processor_for_checkpoint(ckpt)
    processor = update_processor_pixels(processor, DataArguments())
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
    )
    sampling_params = make_sampling_params(
        max_new_tokens=max_new_tokens,
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
        max_trajectories=max_trajectories,
        num_shards=num_shards,
        shard_index=shard_index,
        rollout_all_chunks=rollout_all_chunks,
    )
    stats: Dict[str, Any] = {
        **base_stats,
        "steps": 0,
        "rows": 0,
        "by_type": {},
        "by_correction_reason": {},
        "by_selected_correction_reason": {},
        "step_errors": 0,
        "policy_compress_turns": 0,
        "visual_retries_after_compress": 0,
        "recall_step2_blocked": 0,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fout:
        while True:
            live = [r for r in runners if not r.done]
            if not live:
                break
            batch = live[:rollout_batch_size]
            active: List[Tuple[DaggerRunner, List[Dict[str, Any]]]] = []
            for r in batch:
                if r.current_chunk > r.max_chunk:
                    r.done = True
                    continue
                try:
                    messages = _prepare_step_messages(r)
                except Exception as exc:
                    r.done = True
                    r.error = f"prepare:{type(exc).__name__}:{exc}"
                    stats["step_errors"] += 1
                    stats["skipped"]["prepare_error"] = stats["skipped"].get("prepare_error", 0) + 1
                    continue
                active.append((r, deepcopy(messages)))
            if not active:
                continue

            try:
                vllm_inputs = [
                    prepare_vllm_input(
                        m,
                        processor,
                        tools=tools_for_turn(
                            "compress" if _prompt_has_compress_trigger(m) else "streaming"
                        ),
                    )
                    for _, m in active
                ]
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            except Exception as exc:
                stats["step_errors"] += len(active)
                stats["skipped"]["generate_error"] = stats["skipped"].get("generate_error", 0) + len(active)
                for r, _ in active:
                    r.done = True
                    r.error = f"generate:{type(exc).__name__}:{exc}"
                continue

            recall_batch: List[Tuple[DaggerRunner, List[Dict[str, Any]], str, Dict[str, Any]]] = []
            pending: List[Tuple[DaggerRunner, List[Dict[str, Any]], Dict[str, Any], bool]] = []
            for (r, messages), out_item in zip(active, outputs):
                text = out_item.outputs[0].text
                prompt_is_compress = _prompt_has_compress_trigger(messages)
                try:
                    _apply_rollout_output(
                        r,
                        text,
                        tokenizer,
                        compress_budget=RECENT_THINKS_TOKEN_BUDGET,
                    )
                    result = r.chunk_results[-1]
                    result["format_ok"] = result.get("action") != "unknown"
                    stats["steps"] += 1
                except Exception as exc:
                    r.done = True
                    r.error = f"apply:{type(exc).__name__}:{exc}"
                    stats["step_errors"] += 1
                    stats["skipped"]["apply_error"] = stats["skipped"].get("apply_error", 0) + 1
                    continue
                pending.append((r, messages, result, prompt_is_compress))
                if result.get("action") == "recall":
                    recall_batch.append((r, messages, text, result))

            if recall_batch:
                recall_active = []
                for r, messages, first_text, result in recall_batch:
                    try:
                        recall_messages, recall_result, _ = _build_recall_messages(
                            r, messages, first_text,
                        )
                    except Exception as exc:
                        r.error = f"recall_prepare:{type(exc).__name__}:{exc}"
                        continue
                    if recall_messages:
                        result["recall_messages"] = deepcopy(recall_messages)
                        result["recall_result"] = recall_result
                        recall_active.append((r, result, recall_messages, recall_result))
                if recall_active:
                    try:
                        rc_inputs = [
                            prepare_vllm_input(
                                m,
                                processor,
                                tools=tools_for_turn("recall_response"),
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
                            if rc_parsed.get("action") in ("recall", "compress"):
                                stats["recall_step2_blocked"] += 1
                                result["recall_step2_blocked"] = {
                                    "action": rc_parsed.get("action"),
                                    "raw_output": rc_text,
                                }
                                rc_parsed = {
                                    "action": "silent",
                                    "payload": {},
                                    "raw_output": "",
                                }
                            result["recall_step2_raw_text"] = rc_text
                            result["recall_result"] = recall_result
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
                    _emit_entries(
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
                    if visual_entries or not compress_entries:
                        if result.get("action") == "compress":
                            n = r.compress_retries.get(r.current_chunk, 0) + 1
                            r.compress_retries[r.current_chunk] = n
                            if n <= max_compress_turns_per_chunk:
                                stats["visual_retries_after_compress"] += 1
                                continue
                            stats["skipped"]["too_many_policy_compress_turns"] = (
                                stats["skipped"].get("too_many_policy_compress_turns", 0)
                                + len(visual_entries)
                            )
                        elif visual_entries:
                            stats["skipped"]["policy_failed_compress_before_visual"] = (
                                stats["skipped"].get("policy_failed_compress_before_visual", 0)
                                + len(visual_entries)
                            )
                    r.current_chunk += 1
                else:
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
            if log_every_steps and stats["steps"] % log_every_steps < rollout_batch_size:
                rate = stats["steps"] / max(time.time() - t0, 1e-6)
                print(
                    f"[steps={stats['steps']}] rows={stats['rows']} "
                    f"live={sum(1 for r in runners if not r.done)} "
                    f"compress_turns={stats['policy_compress_turns']} "
                    f"rate={rate:.3f} step/s skipped={stats['skipped']}",
                    flush=True,
                )
                fout.flush()

    stats["out"] = str(out)
    stats["elapsed_sec"] = round(time.time() - t0, 3)
    return stats


def main() -> None:
    batch_root = _default_batch_root(None)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--trajectories", default=str(batch_root / "final" / "train_sft_trajectories.jsonl"))
    p.add_argument("--out", default=str(batch_root / "rendered" / "video_meta" / "train_sft_dagger_messages.jsonl"))
    p.add_argument("--data-dir", default=str(batch_root))
    p.add_argument("--frames-root", default=str(batch_root / "frames"))
    p.add_argument("--video-root", default=None)
    p.add_argument("--frame-protocol", default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"))
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
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--rollout-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-images-per-prompt", type=int, default=64)
    p.add_argument("--max-videos-per-prompt", type=int, default=2)
    p.add_argument("--max-compress-turns-per-chunk", type=int, default=2)
    p.add_argument("--target-chunks-only", action="store_true")
    p.add_argument("--log-every-steps", type=int, default=20)
    args = p.parse_args()

    sample_types = {x.strip() for x in args.sample_types.split(",") if x.strip()}
    correction_reasons = _parse_reason_set(args.correction_reasons)
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    stats = build_dagger_vllm(
        ckpt=args.ckpt,
        trajectories=_resolve_path(args.trajectories),
        out=_resolve_path(args.out),
        data_dir=_resolve_path(args.data_dir),
        frames_root=str(_resolve_path(args.frames_root)),
        video_root=str(_resolve_path(args.video_root)) if args.video_root else None,
        frame_protocol=normalize_frame_protocol(args.frame_protocol),
        sample_types=sample_types,
        include_failed_targets=args.include_failed_targets,
        correction_only=args.correction_only,
        correction_reasons=correction_reasons,
        max_trajectories=args.max_trajectories,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_images_per_prompt=args.max_images_per_prompt,
        max_videos_per_prompt=args.max_videos_per_prompt,
        max_compress_turns_per_chunk=args.max_compress_turns_per_chunk,
        rollout_all_chunks=not args.target_chunks_only,
        log_every_steps=args.log_every_steps,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
