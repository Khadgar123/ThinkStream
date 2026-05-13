"""Chunk-lockstep streaming eval via vLLM.

Replaces mcq_predict_streaming's per-sample sequential loop with a
chunk-aligned cross-sample batch. At each chunk_idx all live samples
build their per-step prompt, the batch is submitted to vLLM in one
generate() call, then each sample's MemoryState is advanced
independently. Samples that emit <action>response</action> are removed
from the live set.

Eval-mode constraints (matches mcq_predict_streaming + agent_loop semantics):
- allow_recall=False: no recall second-pass per step → exactly one
  generate per chunk per sample.
- compress_mode="system": system inserts <compress_trigger> when the
  memory threshold fires; the model only writes the summary.
- Each sample's prompt is rebuilt from scratch per chunk (no KV reuse),
  so vLLM batching is safe without prefix-cache invariants.
"""

import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    VISUAL_WINDOW_CHUNKS,
    action_space_error_for_turn,
    build_recalled_frames_metadata,
    build_recall_result_metadata,
    build_recall_result_user_content,
    canonical_answer_instruction,
    diagnose_compress_output,
    normalize_frame_protocol,
    normalize_render_layout,
    query_is_complete,
    resolve_chunk_frame_paths,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
    tools_for_turn,
)
from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS
from thinkstream.models.agent_loop import (
    COMPRESS_RANGE_MAX,
    COMPRESS_TOKEN_THRESHOLD,
    COMPRESS_RANGE_MIN,
    MemoryState,
    _parse_agent_output,
    build_single_step_messages,
)
from thinkstream.eval.prompt_contract import build_streaming_query_meta

from eval_baseline import DebugLogger, setup_eval_logging
from vllm_engine import (
    generate_with_turn_sampling,
    make_sampling_params,
    prepare_vllm_input,
)


@dataclass
class _SampleRunner:
    """Per-sample state for chunk-lockstep eval."""
    idx: int
    datum: Dict
    video_path: str
    query: str
    ask_chunk: int
    num_chunks: int
    memory: MemoryState
    options: List[str]
    frames_root: Optional[str]
    video_root: Optional[str]
    min_pixels: int
    max_pixels: int
    frame_protocol: str = field(
        default_factory=lambda: normalize_frame_protocol(None)
    )
    render_layout: str = field(
        default_factory=lambda: normalize_render_layout(None)
    )
    current_chunk: int = 0
    done: bool = False
    answer_text: Optional[str] = None
    pred_idx: Optional[int] = None
    error: Optional[str] = None
    _last_turn_kind: str = "streaming"
    # last_compress_trigger: True when system injected a trigger this step
    # so caller can skip user_question on the same step.
    _last_trigger: bool = False
    _last_action: str = "unknown"
    _last_compress_trigger_diagnostic: Dict[str, Any] = field(default_factory=dict)
    _last_compress_prefix_diagnostic: Dict[str, Any] = field(default_factory=dict)
    chunks_generated: int = 0
    question_at_chunk: Dict[int, str] = field(default_factory=dict)
    question_meta_at_chunk: Dict[int, Dict] = field(default_factory=dict)


def _resolve_preextracted_frame_dir(
    video_path: str,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Optional[Path]:
    if not frames_root:
        return None
    root = Path(frames_root)
    vp = Path(video_path)
    candidates: List[Path] = []
    if video_root:
        try:
            rel = vp.relative_to(Path(video_root))
            candidates.append(root / rel.with_suffix(""))
        except ValueError:
            if not vp.is_absolute():
                candidates.append(root / vp.with_suffix(""))
    elif not vp.is_absolute():
        candidates.append(root / vp.with_suffix(""))
    candidates.append(root / vp.stem)
    if any(root.glob("frame_*.jpg")):
        candidates.append(root)
    for frame_dir in candidates:
        if frame_dir.exists():
            return frame_dir
    return None


def _resolve_frame_paths(
    video_path: str,
    chunk_idx: int,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Optional[List[str]]:
    """Resolve the current 1s chunk's pre-extracted JPEGs.

    Returns None if frames_root not configured or insufficient frames found
    (caller falls back to online video decode).
    """
    frame_dir = _resolve_preextracted_frame_dir(video_path, frames_root, video_root)
    if frame_dir is None:
        return None

    paths = resolve_chunk_frame_paths(
        frame_dir,
        chunk_idx,
        frames_per_chunk=FRAMES_PER_CHUNK,
    )
    if len(paths) < FRAMES_PER_CHUNK:
        return None
    return paths


def _resolve_chunk_frame_paths(
    video_path: str,
    chunk_idx: int,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> List[str]:
    """Resolve exactly one chunk's pre-extracted frames."""
    frame_dir = _resolve_preextracted_frame_dir(video_path, frames_root, video_root)
    if frame_dir is None:
        return []
    return resolve_chunk_frame_paths(
        frame_dir,
        chunk_idx,
        frames_per_chunk=FRAMES_PER_CHUNK,
    )


def _compress_trigger_diagnostic(memory: MemoryState) -> Dict[str, Any]:
    tokens = int(memory.count_recent_tokens())
    n_recent = len(memory.recent_thinks)
    triggered = bool(
        n_recent >= COMPRESS_RANGE_MIN
        and (
            tokens >= COMPRESS_TOKEN_THRESHOLD
            or n_recent >= COMPRESS_RANGE_MAX
        )
    )
    if triggered:
        reason = "triggered"
    elif tokens < COMPRESS_TOKEN_THRESHOLD:
        reason = "under_token_threshold"
    elif n_recent < COMPRESS_RANGE_MIN:
        reason = "under_min_thinks"
    else:
        reason = "no_selectable_range"
    return {
        "triggered": triggered,
        "reason": reason,
        "recent_tokens": tokens,
        "token_threshold": int(COMPRESS_TOKEN_THRESHOLD),
        "recent_thinks": n_recent,
        "range_min": int(COMPRESS_RANGE_MIN),
        "range_max": int(COMPRESS_RANGE_MAX),
        "selected_range_n": n_recent if triggered else 0,
        "selected_range_chunks": memory.chunks_for_items(memory.recent_thinks) if triggered else [],
    }


def _maybe_compress_trigger(memory: MemoryState, chunk_idx: int) -> str:
    """Return <compress_trigger/> if memory threshold fires, else "".

    v11.3: range was selected by select_compress_range_by_tokens.
    v12.12 (2026-05-02): trigger emits NO range. Model must derive the
    range from <memory> contents and emit it inside the assistant
    tool_call. This matches pass3c_samples._compress_sample (no range
    in the SFT input) and the eventual RL upgrade where the trigger
    itself is removed (model decides when AND what to compress).
    """
    return "<compress_trigger/>" if _compress_trigger_diagnostic(memory)["triggered"] else ""


def _compress_trigger_for_runner(runner: Any, chunk_idx: int) -> Dict[str, Any]:
    """Use offline pass2 boundaries when the row provides them.

    The RL training rollout consumes the same boundaries to avoid spending
    rollout time recomputing token thresholds. Eval/pre-RL audit should use
    the identical trigger source so answer timing and compression turns are
    measured under the same conversation structure.
    """
    runtime_diag = _compress_trigger_diagnostic(runner.memory)
    trigger_source = str(
        getattr(runner, "compress_trigger_source", "runtime_memory_threshold")
        or "runtime_memory_threshold"
    )
    if trigger_source != "offline_pass2_boundaries":
        diag = dict(runtime_diag)
        diag["trigger_source"] = "runtime_memory_threshold"
        return diag

    pending = getattr(runner, "offline_compress_pending", None)
    if pending is None:
        pending = set()
        setattr(runner, "offline_compress_pending", pending)
    triggered = chunk_idx in pending
    if triggered:
        pending.discard(chunk_idx)
    diag = dict(runtime_diag)
    diag.update({
        "triggered": bool(triggered),
        "reason": "offline_pass2_boundary" if triggered else "not_offline_boundary",
        "trigger_source": "offline_pass2_boundaries",
        "offline_compress_chunks_remaining": sorted(int(x) for x in pending),
    })
    return diag


def _prepare_step_messages(runner: _SampleRunner) -> List[Dict]:
    """Replicates StreamingAgentLoop.step() up to but not including generate."""
    chunk_idx = runner.current_chunk
    snapshot = runner.memory.snapshot(chunk_idx)

    # v12.6 #15: trajectory schema support — look up the per-chunk question
    # from the precomputed map. Falls back to legacy single-question
    # behavior (runner.query at runner.ask_chunk) when the map is empty.
    # v12.13 fix (P0-1): runner.question_meta_at_chunk carries options +
    # answer_form for the question at each ask_chunk. MemoryState.add_query
    # stores them so format_queries_block renders MC Options for active
    # queries. Falls back to {} for legacy runners without the field.
    q_at_chunk = getattr(runner, "question_at_chunk", None) or {}
    q_meta_at_chunk = getattr(runner, "question_meta_at_chunk", None) or {}
    if q_at_chunk:
        user_question = q_at_chunk.get(chunk_idx)
    else:
        user_question = runner.query if chunk_idx == runner.ask_chunk else None
    if user_question:
        ask_time = chunk_idx * AGENT_CHUNK_SEC
        already = any(
            q["question"] == user_question and q.get("ask_time") == ask_time
            for q in runner.memory.queries
        )
        if not already:
            meta = q_meta_at_chunk.get(chunk_idx) or {}
            runner.memory.add_query(
                user_question, ask_time,
                options=meta.get("options"),
                answer_form=meta.get("answer_form"),
                answer_style=meta.get("answer_style"),
                answer_instruction=meta.get("answer_instruction"),
                answer_chunks=meta.get("answer_chunks"),
                per_emit_answers=meta.get("per_emit_answers"),
                open_until=meta.get("open_until"),
            )

    runner._last_compress_trigger_diagnostic = _compress_trigger_for_runner(
        runner, chunk_idx
    )
    compress_trigger = (
        "<compress_trigger/>"
        if runner._last_compress_trigger_diagnostic.get("triggered")
        else ""
    )
    runner._last_trigger = bool(compress_trigger)

    if compress_trigger:
        user_input = compress_trigger
    elif user_question:
        user_input = user_question
    else:
        user_input = ""

    # Memory-compaction turns are text-only inter-chunk actions: suppress
    # visual_window, query/recalled-answer context, and expose compress-only
    # instructions/tools.
    is_inter_chunk = bool(compress_trigger)
    runner._last_turn_kind = "compress" if is_inter_chunk else "streaming"

    frame_paths = _resolve_frame_paths(
        runner.video_path, chunk_idx, runner.frames_root, runner.video_root,
    )

    return build_single_step_messages(
        snapshot,
        chunk_idx,
        runner.video_path,
        user_input=user_input,
        queries=runner.memory.queries,
        min_pixels=runner.min_pixels,
        max_pixels=runner.max_pixels,
        frame_paths=frame_paths,
        frame_protocol=getattr(runner, "frame_protocol", None),
        inter_chunk=is_inter_chunk,
        render_layout=getattr(runner, "render_layout", None),
    )


def _apply_step_output(runner: _SampleRunner, output_text: str) -> str:
    """Replicates the post-generate state update in StreamingAgentLoop.step().

    Eval mode → recall path is dead (allow_recall=False at sampler level
    AND we ignore <action>recall</action> if it slips through).
    """
    parsed = _parse_agent_output(output_text)
    chunk_idx = runner.current_chunk

    action = parsed.get("action") or "unknown"
    action_error = action_space_error_for_turn(
        action,
        getattr(runner, "_last_turn_kind", "streaming"),
    )
    if action_error:
        parsed["action_space_error"] = action_error
        parsed["invalid_action"] = action
        action = "invalid"
    runner._last_action = action
    runner._last_compress_prefix_diagnostic = (
        diagnose_compress_output(output_text)
        if getattr(runner, "_last_turn_kind", "streaming") == "compress"
        else {}
    )
    if (
        parsed.get("think")
        and action != "compress"
        and getattr(runner, "_last_turn_kind", "streaming") != "compress"
    ):
        merge_event = runner.memory.add_think(chunk_idx, parsed["think"])
        runner._last_memory_merge_event = merge_event
    else:
        runner._last_memory_merge_event = None
    if action == "compress":
        entries = parsed["payload"].get("memory_entries") or []
        if entries:
            runner.memory.replace_with_compact_memory(entries)
        else:
            summary = parsed["payload"].get("summary", {})
            if summary and "time_range" in summary:
                compressed_chunks = runner.memory.chunks_in_time_range(
                    summary["time_range"]
                )
                runner.memory.compress(summary, compressed_chunks=compressed_chunks)
    elif action == "response":
        answer_text = parsed["payload"].get("response", "")
        if answer_text:
            response_time = chunk_idx * AGENT_CHUNK_SEC
            # Attach to most-recent active query (mirrors _record_answer).
            attached = False
            complete = False
            for q in reversed(runner.memory.queries):
                status = str(q.get("status", "")).strip().lower()
                if status in {"open", "pending", "active"} or (
                    not status and not q.get("answers")
                ):
                    runner.memory.answer_query(q["question"], answer_text, response_time)
                    attached = True
                    complete = query_is_complete(q)
                    break
            if attached and complete:
                runner.answer_text = answer_text
                runner.done = True
    return action


def _option_match(answer_text: str, options: List[str]) -> int:
    """Map free-text answer to an option index. v12.13 (2026-05-02):
    routes through the SHARED MCQ matcher (same one RL reward uses) so
    eval matches what the model was trained to maximize.

    Old behavior used local startswith / head / substring chain; that
    diverged from RL's _match_mcq_answer (which has the "B" in "table"
    false-positive guard) and caused train/eval reward gap.

    Strategy: try each option's index as the candidate "correct" answer;
    return the first one the shared matcher accepts. Falls back to
    random if none match (preserves the original API contract).
    """
    from thinkstream.trainer.outcome_match import match_mcq_answer
    if not options:
        return 0
    for i in range(len(options)):
        if match_mcq_answer(answer_text, options, i):
            return i
    return random.randint(0, len(options) - 1)


def _build_runners(
    dataset,
    options: List[str],
    question_prefix: str,
    question_postfix: str,
    frames_per_chunk: int,
    max_chunks: int,
    min_pixels: int,
    max_pixels: int,
    frames_root: Optional[str],
    video_root: Optional[str],
    tokenizer=None,
    frame_protocol: Optional[str] = None,
    render_layout: Optional[str] = None,
) -> List[_SampleRunner]:
    frame_protocol = normalize_frame_protocol(frame_protocol)
    render_layout = normalize_render_layout(render_layout)
    runners: List[_SampleRunner] = []
    for i in range(len(dataset)):
        idx = i
        datum = dataset.datums[i]
        try:
            video_end = datum.get("video_end")
            video_start = datum.get("video_start", 0.0)
            if video_end is None:
                raise ValueError("missing video_end")

            num_chunks = max(1, int((video_end - video_start) / AGENT_CHUNK_SEC))
            num_chunks = min(num_chunks, max_chunks)
            ask_chunk = max(0, num_chunks - 1)

            query = str(datum.get("question", ""))
            question_at_chunk = {ask_chunk: query} if query else {}
            question_meta_at_chunk = (
                {ask_chunk: build_streaming_query_meta(datum)}
                if query else {}
            )

            video_path = os.path.join(dataset.data_dir, datum["video"])

            runners.append(_SampleRunner(
                idx=idx,
                datum=datum,
                video_path=video_path,
                query=query,
                ask_chunk=ask_chunk,
                num_chunks=num_chunks,
                memory=MemoryState(tokenizer=tokenizer),
                options=options,
                frames_root=frames_root,
                video_root=video_root,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
                question_at_chunk=question_at_chunk,
                question_meta_at_chunk=question_meta_at_chunk,
            ))
        except Exception as e:
            runners.append(_SampleRunner(
                idx=i,
                datum=datum,
                video_path="",
                query="",
                ask_chunk=0,
                num_chunks=0,
                memory=MemoryState(tokenizer=tokenizer),
                options=options,
                frames_root=frames_root,
                video_root=video_root,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
                done=True,
                error=str(e),
            ))
    return runners


def streaming_predict_mcq_vllm(
    llm,
    processor,
    dataset,
    options: List[str],
    *,
    question_prefix: str = "",
    question_postfix: str = "\nAnswer with a single letter.",
    max_new_tokens: int = 256,
    compress_max_new_tokens: int = 512,
    frames_per_chunk: int = 8,
    max_chunks: int = 30,
    min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    frames_root: Optional[str] = None,
    video_root: Optional[str] = None,
    temperature: float = 0.0,
    repetition_penalty: float = 1.1,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    frame_protocol: Optional[str] = None,
    render_layout: Optional[str] = None,
):
    """Chunk-lockstep streaming MCQ eval via vLLM batched generate.

    All samples advance one chunk per orchestration round. At each round
    every live sample contributes one prompt to a single llm.generate()
    call, then each parses its own output and advances state. Samples
    that emit a final <answer> are removed from the live set.

    Returns: (predictions, datums) — sorted by original dataset index.
    """
    if debug_dir is None:
        debug_dir = os.path.join(getattr(dataset, "data_dir", "."), "debug")
    log = setup_eval_logging(
        os.path.join(debug_dir, "streaming_eval_vllm.log"), rank=0
    )

    dbg = DebugLogger(
        os.path.join(debug_dir, "streaming_debug_vllm.jsonl"),
        enabled=debug, rank=0,
    )

    tokenizer = processor.tokenizer
    frame_protocol = normalize_frame_protocol(frame_protocol)
    render_layout = normalize_render_layout(render_layout)
    runners = _build_runners(
        dataset, options, question_prefix, question_postfix,
        frames_per_chunk, max_chunks, min_pixels, max_pixels,
        frames_root, video_root, tokenizer=tokenizer,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
    )

    log.info(
        f"vLLM streaming eval: {len(runners)} samples, "
        f"max_chunks={max_chunks}, frames_per_chunk={frames_per_chunk}, "
        f"frame_protocol={frame_protocol}, render_layout={render_layout}"
    )

    # repetition_penalty>1.0 is critical for think generation — the v11.2
    # SFT ckpt collapses to 280-300 token repetitive boilerplate at greedy
    # decode without it, even though SFT training data caps think at 130.
    sampling_params = make_sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=1 if temperature == 0.0 else -1,
        repetition_penalty=repetition_penalty,
    )
    compress_sampling_params = make_sampling_params(
        max_new_tokens=compress_max_new_tokens,
        temperature=temperature,
        top_k=1 if temperature == 0.0 else -1,
        repetition_penalty=repetition_penalty,
    )

    # ── Chunk-lockstep loop ──
    t0 = time.time()
    pbar = tqdm.tqdm(total=max_chunks, desc="Chunks")
    n_total_calls = 0
    for chunk_idx in range(max_chunks):
        live = [r for r in runners if not r.done and r.current_chunk == chunk_idx]
        if not live:
            # All remaining runners are either done or behind (impossible by
            # lockstep invariant), so we're finished.
            break

        # Phase A: build per-sample messages
        messages_list: List[List[Dict]] = []
        for r in live:
            try:
                messages_list.append(_prepare_step_messages(r))
            except Exception as e:
                r.error = f"prepare:{e}"
                r.done = True
                messages_list.append(None)

        # Filter out runners whose prepare failed
        active_pairs = [
            (r, m) for r, m in zip(live, messages_list) if m is not None
        ]
        if not active_pairs:
            pbar.update(1)
            continue

        # Phase B: build vLLM inputs
        try:
            turn_kinds = [
                getattr(r, "_last_turn_kind", "streaming")
                for r, _ in active_pairs
            ]
            vllm_inputs = [
                prepare_vllm_input(
                    m,
                    processor,
                    tools=tools_for_turn(turn_kind),
                )
                for (_, m), turn_kind in zip(active_pairs, turn_kinds)
            ]
        except Exception as e:
            log.error(f"vLLM input prep failed at chunk {chunk_idx}: {e}", exc_info=True)
            for r, _ in active_pairs:
                r.error = f"prep_input:{e}"
                r.done = True
            pbar.update(1)
            continue

        # Phase C: batched generate
        outputs = generate_with_turn_sampling(
            llm,
            vllm_inputs,
            turn_kinds,
            sampling_params,
            {"compress": compress_sampling_params},
        )
        n_total_calls += len(outputs)

        # Phase D: apply outputs
        for (r, _), out in zip(active_pairs, outputs):
            try:
                text = out.outputs[0].text
                action = _apply_step_output(r, text)
                r.chunks_generated += 1
                if debug:
                    dbg.log({
                        "idx": r.idx, "chunk_idx": chunk_idx,
                        "video": r.datum.get("video"),
                        "output": text,
                        "memory_thinks": len(r.memory.recent_thinks),
                        "memory_compressed": len(r.memory.compressed_segments),
                        "compress_trigger": r._last_trigger,
                        "compress_prefix_diagnostic": r._last_compress_prefix_diagnostic,
                        "action": action,
                    })
            except Exception as e:
                r.error = f"apply:{e}"
                r.done = True

        # Advance chunk pointer for all runners that participated this round
        for r, _ in active_pairs:
            if not r.done:
                if r._last_action == "compress" and r._last_trigger:
                    continue
                r.current_chunk += 1
                if r.current_chunk >= r.num_chunks:
                    # Reached end without ever emitting <answer> (v12 response); mark done.
                    r.done = True

        pbar.update(1)
        pbar.set_postfix({
            "live": sum(1 for r in runners if not r.done),
            "answered": sum(1 for r in runners if r.answer_text is not None),
        })
    pbar.close()

    elapsed = time.time() - t0
    log.info(
        f"streaming eval done in {elapsed:.1f}s — "
        f"{n_total_calls} vLLM requests across "
        f"{sum(r.chunks_generated for r in runners)} sample-chunks"
    )

    # ── Build predictions in dataset order ──
    predictions: List[int] = []
    datums_out: List[Dict] = []
    sorted_runners = sorted(runners, key=lambda r: r.idx)
    correct = 0
    parsed = 0
    for r in sorted_runners:
        if r.answer_text:
            pred = _option_match(r.answer_text, options)
            success = True
            parsed += 1
        else:
            pred = random.randint(0, len(options) - 1)
            success = False

        gt = r.datum.get("answer", "")
        is_correct = (options[pred] == gt) if success else False
        if is_correct:
            correct += 1

        log.info(
            f"[{r.idx}] {'✓' if is_correct else '✗'} "
            f"answer={r.answer_text!r} -> {options[pred]} "
            f"(gt={gt}, chunks={r.chunks_generated}, "
            f"err={r.error or '-'})"
        )

        predictions.append(pred)
        datums_out.append({
            **r.datum,
            "success": success,
            "generated_answer": r.answer_text or "",
            "chunks_generated": r.chunks_generated,
            "error": r.error,
        })

    if parsed > 0:
        log.info(
            f"Accuracy among answered: {correct}/{parsed} = {correct/parsed:.1%}"
        )
    log.info(
        f"Coverage: {parsed}/{len(runners)} samples emitted a response "
        f"({parsed/len(runners):.1%})"
    )

    dbg.close()
    return np.array(predictions), datums_out


# ───────────────────────────────────────────────────────────────────────
# RL rollout (v11.3): chunk-lockstep × group_size cross-sample batching
# ───────────────────────────────────────────────────────────────────────


@dataclass
class _RolloutRunner:
    """Per-(sample, gen_idx) state for RL rollout.

    Field names matching _SampleRunner where shared so _prepare_step_messages
    works on this type via duck typing (runner.current_chunk, .memory, .query,
    .ask_chunk, .video_path, .frames_root, .video_root, .min_pixels,
    .max_pixels, ._last_trigger).
    """
    sample_idx: int
    gen_idx: int
    raw_sample: Dict
    video_path: str
    query: Optional[str]
    ask_chunk: int
    max_chunks: int
    memory: MemoryState
    frames_root: Optional[str]
    video_root: Optional[str]
    min_pixels: int
    max_pixels: int
    frame_protocol: str = field(
        default_factory=lambda: normalize_frame_protocol(None)
    )
    render_layout: str = field(
        default_factory=lambda: normalize_render_layout(None)
    )
    current_chunk: int = 0
    done: bool = False
    error: Optional[str] = None
    _last_turn_kind: str = "streaming"
    _last_trigger: bool = False
    _last_compress_trigger_diagnostic: Dict[str, Any] = field(default_factory=dict)
    compress_trigger_source: str = "runtime_memory_threshold"
    offline_compress_pending: set[int] = field(default_factory=set)
    # Per-chunk results, shape matches grpo.py:736-758 contract.
    chunk_results: List[Dict] = field(default_factory=list)
    # v12.6 #15: trajectory schema support — each chunk may carry its own
    # question (multi-ask trajectories from pass4 train_rl_trajectories.jsonl).
    # Built once per runner from raw_sample["questions"]; lookup in
    # _prepare_step_messages.
    question_at_chunk: Dict[int, str] = field(default_factory=dict)
    # v12.13 fix (P0-1): per-chunk options + answer_form for MC queries
    question_meta_at_chunk: Dict[int, Dict] = field(default_factory=dict)
    # Per-runner retriever for recall tool execution (BM25 index per video).
    # None = recall second-pass disabled (vLLM legacy behavior pre-#15).
    retriever: Optional[object] = None

    def _record_answer_to_memory(self, answer_text: str, chunk_idx: int) -> None:
        """Mirror agent_loop.StreamingAgentLoop._record_answer.

        Attach the recall second-pass answer to the most-recent active query.
        """
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


def _runner_should_stop_after_response(runner: _RolloutRunner, chunk_idx: int) -> bool:
    """Return whether a response should terminate this eval rollout.

    Single-question eval can stop after a non-empty response. Multi-question
    and multi-emit trajectories must continue to the fixed answer horizon so
    later questions/emits can still be observed and scored.
    """
    if chunk_idx < int(getattr(runner, "ask_chunk", 0) or 0):
        return False
    q_at_chunk = getattr(runner, "question_at_chunk", None) or {}
    q_meta_at_chunk = getattr(runner, "question_meta_at_chunk", None) or {}
    if len(q_at_chunk) > 1:
        return False
    for meta in q_meta_at_chunk.values():
        answer_chunks = meta.get("answer_chunks") or []
        per_emit = meta.get("per_emit_answers") or []
        try:
            n_answer_chunks = len(answer_chunks)
        except TypeError:
            n_answer_chunks = 0
        try:
            n_per_emit = len(per_emit)
        except TypeError:
            n_per_emit = 0
        if max(n_answer_chunks, n_per_emit) > 1:
            return False
    return True


_USER_INPUT_RE = re.compile(r"<user_input>(.*?)</user_input>", re.DOTALL)


def _extract_user_question(raw_sample: Dict) -> Optional[str]:
    """Mirrors grpo.py:693-712 — pull user_question from new/legacy sample formats.

    Order: input.user_input → messages.<user_input> tag → conversations[role=user].
    Returns None when no question is present (silent-only sample).
    """
    inp = raw_sample.get("input")
    if isinstance(inp, dict) and inp.get("user_input"):
        return inp["user_input"]
    msgs = raw_sample.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        m = _USER_INPUT_RE.search(text)
                        if m:
                            return m.group(1)
    convs = raw_sample.get("conversations")
    if isinstance(convs, list):
        for c in convs:
            if c.get("role") == "user":
                return c.get("content", "")
    return None


def _extract_question_at_chunk_map(raw_sample: Dict) -> Dict[int, str]:
    """v12.6 #15: build {chunk_idx → question_text} for trajectory rows.

    Mirrors grpo.py:_extract_questions_at_chunks. Trajectory format
    (pass4 train_rl_trajectories.jsonl) carries questions[*].ask_chunks,
    multiple cards per trajectory each firing at one or more chunks.
    Falls back to single-question (input.user_input / conversations) for
    flat datasets — those rows produce a 1-entry map keyed by
    raw_sample.chunk_idx.
    """
    out: Dict[int, str] = {}
    # Schema A: trajectory (v12.5+)
    if (isinstance(raw_sample.get("questions"), list)
            and isinstance(raw_sample.get("gold_action_per_chunk"), dict)):
        # v12.13: options live ONLY in active-query state via format_queries_block.
        # user_input/question_at_chunk carries the bare question text.
        for q in raw_sample["questions"]:
            q_text = q.get("question") or q.get("gold_answer", "")
            for ac in q.get("ask_chunks") or []:
                out[int(ac)] = q_text
        return out
    # Schema B: flat single-question fallback
    single_q = _extract_user_question(raw_sample)
    if single_q is not None:
        ck = int(raw_sample.get("chunk_idx", 0))
        out[ck] = single_q
    return out


def _extract_offline_compress_chunks(raw_sample: Dict) -> List[int]:
    extra = raw_sample.get("extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    chunks = (
        raw_sample.get("offline_compress_chunks")
        or extra.get("offline_compress_chunks")
        or []
    )
    out: List[int] = []
    for ck in chunks:
        try:
            out.append(int(ck))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _apply_rollout_output(
    runner: _RolloutRunner, output_text: str, tokenizer, *,
    compress_budget: int,
    step_messages: Optional[List[Dict]] = None,
) -> None:
    """Per-chunk state advance + chunk_results append.

    Differs from _apply_step_output (eval) in two ways:
      1. Does NOT set runner.done on response — RL rolls a few chunks past
         ask_chunk so the model emits the full <think> + <answer> (v12 response)
         under post-answer pressure (matches grpo.py legacy rollout).
      2. Records the legacy per-chunk dict (grpo.py:736-758 contract):
         action, think, payload, raw_output, generated_tokens,
         memory_token_count, compress_budget, recall_returned_chunks,
         window_start, window_end.
    """
    chunk_idx = runner.current_chunk
    parsed = _parse_agent_output(output_text)
    queries_before = deepcopy(getattr(runner.memory, "queries", []))
    memory_merge_event = None

    action = parsed.get("action") or "unknown"
    action_error = action_space_error_for_turn(
        action,
        getattr(runner, "_last_turn_kind", "streaming"),
    )
    if action_error:
        parsed["action_space_error"] = action_error
        parsed["invalid_action"] = action
        action = "invalid"
    if (
        parsed.get("think")
        and action != "compress"
        and getattr(runner, "_last_turn_kind", "streaming") != "compress"
    ):
        memory_merge_event = runner.memory.add_think(chunk_idx, parsed["think"])
        if memory_merge_event:
            memory_merge_event = deepcopy(memory_merge_event)
            memory_merge_event["raw_output"] = output_text
    if action == "compress":
        summary = parsed.get("payload", {}).get("summary", {})
        if summary and "time_range" in summary:
            compressed_chunks = runner.memory.chunks_in_time_range(
                summary["time_range"]
            )
            runner.memory.compress(summary, compressed_chunks=compressed_chunks)
    elif action == "response":
        answer_text = parsed.get("payload", {}).get("response", "")
        if answer_text:
            response_time = chunk_idx * AGENT_CHUNK_SEC
            for q in reversed(runner.memory.queries):
                status = str(q.get("status", "")).strip().lower()
                if status in {"open", "pending", "active"} or (
                    not status and not q.get("answers")
                ):
                    runner.memory.answer_query(q["question"], answer_text, response_time)
                    break

    entry = {
        "chunk_idx": chunk_idx,
        "action": action,
        "think": parsed.get("think", ""),
        "payload": parsed.get("payload", {}),
        "raw_output": output_text,
        "compress_prefix_diagnostic": (
            diagnose_compress_output(output_text)
            if getattr(runner, "_last_turn_kind", "streaming") == "compress"
            else {}
        ),
        "compress_trigger_diagnostic": deepcopy(
            getattr(runner, "_last_compress_trigger_diagnostic", {})
        ),
        "action_space_error": parsed.get("action_space_error", ""),
        "invalid_action": parsed.get("invalid_action", ""),
        "generated_tokens": tokenizer.encode(output_text, add_special_tokens=False),
        "memory_token_count": runner.memory.count_recent_tokens(),
        "compress_budget": compress_budget,
        # vLLM rollout doesn't run the retriever — recall samples that need
        # hit-rate reward should pass --rollout_use_retriever (not yet wired)
        # or accept that recall_returned_chunks is empty (reward_masks gate
        # this column anyway).
        "recall_returned_chunks": [],
        "window_start": chunk_idx * int(AGENT_CHUNK_SEC),
        "window_end": (chunk_idx + 1) * int(AGENT_CHUNK_SEC),
        "step_messages": deepcopy(step_messages) if step_messages is not None else None,
        # DAgger needs the state before the student action, even when a
        # recall second pass later overwrites step_messages for RL loss replay.
        "dagger_step_messages": deepcopy(step_messages) if step_messages is not None else None,
        "turn_kind": getattr(runner, "_last_turn_kind", "streaming"),
        "first_action": action,
        "queries_before": queries_before,
        "queries_after": deepcopy(getattr(runner.memory, "queries", [])),
        "memory_merge_event": memory_merge_event,
    }
    runner.chunk_results.append(entry)


def streaming_vllm_rollout(
    step_inputs: List[Dict],
    llm,
    processor,
    tokenizer,
    *,
    group_size: int,
    max_new_tokens: int = 256,
    compress_max_new_tokens: int = 512,
    rollout_max_chunks: int = 30,
    rollout_extra_chunks: int = 5,
    min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    frames_root: Optional[str] = None,
    video_root: Optional[str] = None,
    compress_budget: Optional[int] = None,
    enable_recall: bool = True,
    frame_protocol: Optional[str] = None,
    render_layout: Optional[str] = None,
    memory_merge_similar_thinks: Optional[bool] = None,
    memory_merge_similarity_threshold: Optional[float] = None,
) -> List[Dict]:
    """vLLM-batched RL rollout matching grpo.py:617-803 output contract.

    For each raw_sample in step_inputs, runs G=group_size independent
    trajectories. All N×G runners advance in chunk-lockstep — at each
    chunk_idx, every live runner contributes one prompt to a single
    llm.generate() call so the GPU stays full. Per-runner MemoryState
    is maintained independently; the message format goes through the
    same build_single_step_messages used by SFT and eval, guaranteeing
    byte-identical prompts.

    Returns list of dicts with the legacy shape:
      [{"raw_sample": <dict>,
        "chunk_results": [{
          "chunk_idx": int, "window_start": int, "window_end": int,
          "generated_tokens": List[Tensor]  # len = group_size,
          "memory_token_count": List[int]   # len = group_size,
          "compress_budget":   List[int]    # len = group_size,
          "recall_returned_chunks": List[List[int]]  # len = group_size,
        }, ...]
       }, ...]

    so grpo.py downstream (reward calc, GDPO advantage, loss) is unchanged.
    """
    import torch as _torch
    from collections import defaultdict

    # Default budget pulled from agent_loop's RECENT_THINKS_TOKEN_BUDGET to
    # match the SFT/eval value without forcing callers to pass it.
    if compress_budget is None:
        from thinkstream.models.agent_loop import RECENT_THINKS_TOKEN_BUDGET
        compress_budget = RECENT_THINKS_TOKEN_BUDGET
    frame_protocol = normalize_frame_protocol(frame_protocol)
    render_layout = normalize_render_layout(render_layout)

    # ── Build N × G runners ──
    runners: List[_RolloutRunner] = []
    for s_idx, raw_sample in enumerate(step_inputs):
        data_path = raw_sample.get("data_path", "")
        rel_video = raw_sample.get("video_path", "")
        if data_path and rel_video and not Path(rel_video).is_absolute():
            video_path = str(Path(data_path) / rel_video)
        else:
            video_path = rel_video or ""

        # v12.6 #15: build per-chunk question map (trajectory schema) +
        # latest-firing ask_chunk for max_chunks bound. Trajectory rows
        # may have multiple ask_chunks across multiple cards; we extend
        # the rollout horizon past the LATEST one so each fires its
        # response window.
        q_at_chunk = _extract_question_at_chunk_map(raw_sample)
        # v12.13 fix (P0-1): build per-chunk meta map alongside question text
        # so MC options and lifecycle metadata propagate to runtime queries.
        q_meta_at_chunk: Dict[int, Dict] = {}
        # v12.13 fix (P0-3): track answer_chunks so rollout cap covers
        # forward / silent_then_response cards (lead 18-32 chunks).
        all_answer_chunks: List[int] = []
        if (isinstance(raw_sample.get("questions"), list)
                and isinstance(raw_sample.get("gold_action_per_chunk"), dict)):
            for q in raw_sample["questions"]:
                meta = {
                    "options": list(q.get("options") or []),
                    "answer_form": q.get("answer_form", ""),
                    "answer_style": (
                        "letter_only"
                        if q.get("answer_form") == "multiple_choice"
                        else q.get("answer_style", "")
                    ),
                    "answer_instruction": canonical_answer_instruction(q)
                    or q.get("answer_instruction", ""),
                    "answer_chunks": list(q.get("answer_chunks") or []),
                    "per_emit_answers": list(q.get("per_emit_answers") or []),
                }
                ans_chunks = [int(x) for x in q.get("answer_chunks") or []]
                if ans_chunks:
                    meta["open_until"] = max(ans_chunks) * AGENT_CHUNK_SEC
                for ac in q.get("ask_chunks") or []:
                    q_meta_at_chunk[int(ac)] = meta
                for ac in q.get("answer_chunks") or []:
                    all_answer_chunks.append(int(ac))
        if q_at_chunk:
            latest_ask = max(q_at_chunk.keys())
            # Legacy fields for back-compat: pick the canonical first ask_chunk
            ask_chunk = min(q_at_chunk.keys())
            question = q_at_chunk[ask_chunk]
        else:
            ask_chunk = int(raw_sample.get("chunk_idx", rollout_max_chunks - 1))
            latest_ask = ask_chunk
            question = None
        # v12.13: rollout cap = max(answer_chunks) + slack. Fallback to
        # latest_ask + rollout_extra_chunks for legacy trajectories without
        # answer_chunks.
        if all_answer_chunks:
            cap_target = max(all_answer_chunks) + 2  # slack
        else:
            cap_target = latest_ask + rollout_extra_chunks
        max_chunks_this = min(cap_target + 1, rollout_max_chunks)
        raw_extra = raw_sample.get("extra_info") or {}
        if not isinstance(raw_extra, dict):
            raw_extra = {}
        offline_compress_chunks = _extract_offline_compress_chunks(raw_sample)
        compress_trigger_source = str(
            raw_sample.get("compress_trigger_source")
            or raw_extra.get("compress_trigger_source")
            or (
                "offline_pass2_boundaries"
                if offline_compress_chunks
                else "runtime_memory_threshold"
            )
        )

        for g in range(group_size):
            # v12.6 #15: per-runner BM25 retriever for recall tool execution.
            # Each rollout has its own memory state → its own think archive →
            # its own retriever index. Only built if enable_recall=True.
            runner_retriever = None
            if enable_recall:
                from thinkstream.models.retrieval import BM25Retriever
                runner_retriever = BM25Retriever()
            runners.append(_RolloutRunner(
                sample_idx=s_idx,
                gen_idx=g,
                raw_sample=raw_sample,
                video_path=video_path,
                query=question,
                ask_chunk=ask_chunk,
                max_chunks=max_chunks_this,
                memory=MemoryState(
                    tokenizer=tokenizer,
                    merge_similar_thinks=memory_merge_similar_thinks,
                    merge_similarity_threshold=memory_merge_similarity_threshold,
                ),
                frames_root=frames_root,
                video_root=video_root,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
                question_at_chunk=q_at_chunk,
                question_meta_at_chunk=q_meta_at_chunk,    # v12.13 P0-1
                retriever=runner_retriever,
                compress_trigger_source=compress_trigger_source,
                offline_compress_pending=set(offline_compress_chunks),
            ))

    if not runners:
        return []

    sampling_params = make_sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    compress_sampling_params = make_sampling_params(
        max_new_tokens=compress_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # ── Chunk-lockstep loop ──
    max_global_chunk = max(r.max_chunks for r in runners)
    max_compress_turns = max(
        (len(getattr(r, "offline_compress_pending", set()) or set()) for r in runners),
        default=0,
    )
    for _ in range(max_global_chunk + max_compress_turns + 8):
        active = [r for r in runners if not r.done]
        if not active:
            break
        chunk_idx = min(r.current_chunk for r in active)
        live = [r for r in active if r.current_chunk == chunk_idx]

        # Phase A: build messages (reuse _prepare_step_messages via duck typing)
        messages_list: List[List[Dict]] = []
        live_active: List[_RolloutRunner] = []
        for r in live:
            try:
                messages_list.append(_prepare_step_messages(r))
                live_active.append(r)
            except Exception as e:
                r.error = f"prepare:{e}"
                r.done = True

        if not live_active:
            continue

        # Phase B: vLLM input + batch generate with turn-local tools.
        try:
            turn_kinds = [
                getattr(r, "_last_turn_kind", "streaming")
                for r in live_active
            ]
            vllm_inputs = [
                prepare_vllm_input(
                    m,
                    processor,
                    tools=tools_for_turn(turn_kind),
                )
                for m, turn_kind in zip(messages_list, turn_kinds)
            ]
        except Exception as e:
            for r in live_active:
                r.error = f"prep_input:{e}"
                r.done = True
            continue
        outputs = generate_with_turn_sampling(
            llm,
            vllm_inputs,
            turn_kinds,
            sampling_params,
            {"compress": compress_sampling_params},
        )

        # Phase C: apply outputs + collect runners that emitted recall
        recall_runners: List[Tuple[_RolloutRunner, List[Dict], str]] = []
        for r, out, msgs in zip(live_active, outputs, messages_list):
            try:
                text = out.outputs[0].text
                _apply_rollout_output(
                    r,
                    text,
                    tokenizer,
                    compress_budget=compress_budget,
                    step_messages=msgs,
                )
            except Exception as e:
                r.error = f"apply:{e}"
                r.done = True
                continue

            last_action = r.chunk_results[-1]["action"]
            # v12.6 #15: queue recall runners for second-pass; DO NOT advance
            # chunk_idx — recall is single-chunk multi-turn (shape B).
            if last_action == "recall" and r.retriever is not None and enable_recall:
                # Index this chunk's think into the per-runner retriever before
                # querying (matches HF agent_loop.step:799-805 ordering).
                think_text = r.chunk_results[-1].get("think", "")
                if think_text:
                    try:
                        r.retriever.index_chunk(chunk_idx, r.video_path, think_text)
                    except Exception:
                        pass
                recall_runners.append((r, msgs, text))
                continue

            if last_action == "compress" and r._last_trigger:
                # Compression is an inter-chunk memory-management turn.
                # It should fold recent memory, then retry the same video chunk.
                continue

            r.current_chunk += 1
            # Single-Q eval can stop after a response. Multi-Q and multi-emit
            # eval must keep rolling to the fixed answer horizon so later
            # questions/emits are not hidden by the first response.
            if r.current_chunk >= r.max_chunks:
                r.done = True
            elif last_action == "response" and _runner_should_stop_after_response(r, chunk_idx):
                r.done = True

        # ── Phase D: recall second-pass (batched vLLM generate) ──
        # For each recall-emitting runner, build the multi-turn prompt:
        #   [system, user(chunk N), assistant(recall tool_call),
        #    user(<recalled_frames> + <recall_result>)]
        # then generate the final answer turn. Mirrors agent_loop.step()'s
        # recall branch (lines 868-908) to maintain SFT/runtime parity.
        if recall_runners:
            recall_msgs_batch: List[List[Dict]] = []
            recall_meta: List[Tuple[_RolloutRunner, Dict, Optional[Dict], List[Dict]]] = []
            for r, first_msgs, first_text in recall_runners:
                try:
                    parsed = _parse_agent_output(first_text)
                    query = parsed.get("payload", {}).get("query", {})
                    if not query:
                        # Malformed recall — record empty result + advance
                        r.chunk_results[-1]["recall_returned_chunks"] = []
                        r.current_chunk += 1
                        if r.current_chunk >= r.max_chunks:
                            r.done = True
                        continue
                    raw_recall_result = r.retriever(query, r.memory.retrieval_archive)
                    returned_chunks = select_recall_chunks(
                        raw_recall_result.get("returned_chunks", [])
                    )
                    raw_recall_result["returned_chunks"] = returned_chunks
                    r.chunk_results[-1]["recall_returned_chunks"] = list(returned_chunks)

                    # Build recalled_frames metadata (matching shape B in pass5)
                    recalled_frames = None
                    if returned_chunks and raw_recall_result.get("source") == "historical_frames":
                        rf_paths: List[str] = []
                        frame_chunks: List[int] = []
                        for rc in returned_chunks:
                            cf = _resolve_chunk_frame_paths(
                                r.video_path, rc, r.frames_root, r.video_root,
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
                    recall_result = build_recall_result_metadata(
                        raw_recall_result,
                        recalled_frames,
                    )

                    # Multi-turn message construction: original prompt +
                    # assistant(first_text) + user(tool result + frames)
                    rc_msgs = deepcopy(first_msgs)
                    if rc_msgs and rc_msgs[0].get("role") == "system":
                        rc_msgs[0] = {
                            "role": "system",
                            "content": [{
                                "type": "text",
                                "text": system_prompt_for_frame_protocol(
                                    r.frame_protocol,
                                    prompt_kind="post_recall",
                                    render_layout=r.render_layout,
                                ),
                            }],
                        }
                    rc_msgs.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": first_text}],
                    })
                    tool_user_content = build_recall_result_user_content(
                        recalled_frames,
                        recall_result,
                        frame_protocol=r.frame_protocol,
                        min_pixels=r.min_pixels,
                        max_pixels=r.max_pixels,
                        render_layout=r.render_layout,
                    )
                    rc_msgs.append({
                        "role": "tool",
                        "tool_call_id": "recall",
                        "content": tool_user_content,
                    })

                    recall_msgs_batch.append(rc_msgs)
                    recall_meta.append((r, recall_result, recalled_frames, rc_msgs))
                except Exception as e:
                    r.error = f"recall_prep:{e}"
                    r.current_chunk += 1
                    if r.current_chunk >= r.max_chunks:
                        r.done = True

            if recall_msgs_batch:
                try:
                    rc_inputs = [
                        prepare_vllm_input(
                            m,
                            processor,
                            tools=tools_for_turn("post_recall"),
                        )
                        for m in recall_msgs_batch
                    ]
                    rc_outputs = llm.generate(rc_inputs, sampling_params=sampling_params)
                except Exception as e:
                    for r, _, _, _ in recall_meta:
                        r.error = f"recall_gen:{e}"
                        r.current_chunk += 1
                        if r.current_chunk >= r.max_chunks:
                            r.done = True
                else:
                    for (r, recall_result, recalled_frames, rc_msgs), rc_out in zip(
                        recall_meta, rc_outputs
                    ):
                        rc_action = ""
                        try:
                            rc_text = rc_out.outputs[0].text
                            rc_parsed = _parse_agent_output(rc_text)
                            rc_action = rc_parsed.get("action") or "unknown"
                            rc_action_error = action_space_error_for_turn(
                                rc_action,
                                "post_recall",
                            )
                            if rc_action_error:
                                rc_parsed["action_space_error"] = rc_action_error
                                rc_parsed["invalid_action"] = rc_action
                                rc_action = "invalid"
                            # Update memory with the final-answer turn
                            if rc_action == "response":
                                ans = rc_parsed.get("payload", {}).get("response", "")
                                if ans:
                                    r._record_answer_to_memory(ans, r.current_chunk)

                            # v12.6 #22 fix: FOLD the second-pass result into
                            # the SAME chunk_results entry rather than append
                            # a new list element. The downstream merger zips
                            # by list index ci; appending here would shift
                            # all subsequent chunks by 1 → reward/advantage
                            # alignment breaks.
                            #
                            # Both assistant turns (recall tool_call + final
                            # answer) are still trained: completion_mask
                            # rebuilds messages from step_messages (which
                            # carries the full multi-turn shape) and
                            # find_assistant_spans picks up BOTH spans →
                            # both contribute to GRPO logprob. The chunk's
                            # "primary action" for reward eval is the final
                            # answer turn, so raw_output/action are
                            # overwritten with the second-pass result;
                            # first-pass tokens are kept for diagnostics.
                            entry = r.chunk_results[-1]
                            first_pass_tokens = entry.get("generated_tokens", [])
                            entry["_recall_first_text"] = entry.get("raw_output", "")
                            entry["_recall_first_action"] = entry.get("action", "")
                            entry["_recall_first_payload"] = entry.get("payload", {})
                            entry["_recall_first_tokens"] = first_pass_tokens
                            entry["raw_output"] = rc_text
                            entry["action"] = rc_action
                            entry["think"] = rc_parsed.get("think", entry.get("think", ""))
                            entry["payload"] = rc_parsed.get("payload", {})
                            entry["step_messages"] = deepcopy(rc_msgs)
                            entry["action_space_error"] = rc_parsed.get("action_space_error", "")
                            entry["invalid_action"] = rc_parsed.get("invalid_action", "")
                            # v12.11 P0.6 fix (2026-05-01): generated_tokens
                            # MUST be ONLY the second-pass tokens. The
                            # previous concat (first + second) caused the
                            # loss-time merger to render an assistant turn
                            # containing BOTH <tool_call> and <answer>,
                            # which v12 parser flags as format_error. The
                            # first-pass tool_call already lives in
                            # step_messages; the loss reconstruction
                            # appends ONE final assistant turn = second-pass.
                            entry["generated_tokens"] = list(
                                tokenizer.encode(rc_text, add_special_tokens=False)
                            )
                            entry["memory_token_count"] = r.memory.count_recent_tokens()
                            entry["recall_returned_chunks"] = list(
                                recall_result.get("returned_chunks") or []
                            )
                            entry["recall_multiturn"] = True
                            entry["queries_after"] = deepcopy(
                                getattr(r.memory, "queries", [])
                            )
                        except Exception as e:
                            r.error = f"recall_apply:{e}"
                        r.current_chunk += 1
                        if r.current_chunk >= r.max_chunks:
                            r.done = True
                        elif rc_action == "response" and _runner_should_stop_after_response(r, chunk_idx):
                            r.done = True

    # ── Group runners back: per-sample list of G trajectories ──
    per_sample: Dict[int, List[_RolloutRunner]] = defaultdict(list)
    for r in runners:
        per_sample[r.sample_idx].append(r)

    all_rollout_results: List[Dict] = []
    for s_idx in range(len(step_inputs)):
        gens = sorted(per_sample[s_idx], key=lambda r: r.gen_idx)
        per_gen_results = [g.chunk_results for g in gens]
        max_chunks_seen = max((len(g) for g in per_gen_results), default=0)

        merged_chunk_results = []
        for ci in range(max_chunks_seen):
            merged = {
                "chunk_idx": ci,
                "window_start": ci * int(AGENT_CHUNK_SEC),
                "window_end": (ci + 1) * int(AGENT_CHUNK_SEC),
                "generated_tokens": [],
                "memory_token_count": [],
                "compress_budget": [],
                "recall_returned_chunks": [],
                # v12.11 audit-5 P1 #6 fix (2026-05-01): vLLM rollout merge
                # was dropping step_messages and recall_first_pass_text.
                # If/when vLLM-RL is wired up, per-chunk loss reconstruction
                # needs step_messages (loss-time prompt parity) and the
                # trajectory reward parser needs recall_first_pass_text
                # (n_recall counter). HF rollout merge already includes
                # these (grpo.py:711-758); aligning here keeps both
                # backends interchangeable.
                "step_messages": [],
                "dagger_step_messages": [],
                "recall_first_pass_text": [],
                "raw_outputs": [],
                "actions": [],
                "first_actions": [],
                "turn_kinds": [],
                "chunk_indices": [],
                "compress_prefix_diagnostics": [],
                "compress_trigger_diagnostics": [],
                "action_space_errors": [],
                "invalid_actions": [],
                "queries_before": [],
                "queries_after": [],
                "memory_merge_events": [],
            }
            for g_idx in range(group_size):
                if ci < len(per_gen_results[g_idx]):
                    cr_g = per_gen_results[g_idx][ci]
                    merged["generated_tokens"].append(
                        _torch.tensor(cr_g["generated_tokens"], dtype=_torch.long)
                    )
                    merged["memory_token_count"].append(int(cr_g["memory_token_count"]))
                    merged["compress_budget"].append(int(cr_g["compress_budget"]))
                    merged["recall_returned_chunks"].append(
                        list(cr_g["recall_returned_chunks"])
                    )
                    merged["step_messages"].append(cr_g.get("step_messages"))
                    merged["dagger_step_messages"].append(
                        cr_g.get("dagger_step_messages") or cr_g.get("step_messages")
                    )
                    merged["recall_first_pass_text"].append(
                        cr_g.get("_recall_first_text", "")
                    )
                    merged["raw_outputs"].append(cr_g.get("raw_output", ""))
                    merged["actions"].append(cr_g.get("action", ""))
                    merged["first_actions"].append(
                        cr_g.get("_recall_first_action")
                        or cr_g.get("first_action")
                        or cr_g.get("action", "")
                    )
                    merged["turn_kinds"].append(cr_g.get("turn_kind", "streaming"))
                    merged["chunk_indices"].append(int(cr_g.get("chunk_idx", ci)))
                    merged["compress_prefix_diagnostics"].append(
                        cr_g.get("compress_prefix_diagnostic") or {}
                    )
                    merged["compress_trigger_diagnostics"].append(
                        cr_g.get("compress_trigger_diagnostic") or {}
                    )
                    merged["action_space_errors"].append(
                        cr_g.get("action_space_error", "")
                    )
                    merged["invalid_actions"].append(cr_g.get("invalid_action", ""))
                    merged["queries_before"].append(cr_g.get("queries_before") or [])
                    merged["queries_after"].append(cr_g.get("queries_after") or [])
                    merged["memory_merge_events"].append(
                        cr_g.get("memory_merge_event")
                    )
                else:
                    # Pad: this gen finished early (response emitted past ask_chunk).
                    merged["generated_tokens"].append(_torch.tensor([], dtype=_torch.long))
                    merged["memory_token_count"].append(0)
                    merged["compress_budget"].append(0)
                    merged["recall_returned_chunks"].append([])
                    merged["step_messages"].append(None)
                    merged["dagger_step_messages"].append(None)
                    merged["recall_first_pass_text"].append("")
                    merged["raw_outputs"].append("")
                    merged["actions"].append("")
                    merged["first_actions"].append("")
                    merged["turn_kinds"].append("")
                    merged["chunk_indices"].append(-1)
                    merged["compress_prefix_diagnostics"].append({})
                    merged["compress_trigger_diagnostics"].append({})
                    merged["action_space_errors"].append("")
                    merged["invalid_actions"].append("")
                    merged["queries_before"].append([])
                    merged["queries_after"].append([])
                    merged["memory_merge_events"].append(None)
            merged_chunk_results.append(merged)

        all_rollout_results.append({
            "raw_sample": step_inputs[s_idx],
            "chunk_results": merged_chunk_results,
        })

    return all_rollout_results
