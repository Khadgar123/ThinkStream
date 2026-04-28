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
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    VISUAL_WINDOW_CHUNKS,
    parse_agent_output,
)
from thinkstream.model.agent_loop import (
    COMPRESS_RANGE_MIN,
    MemoryState,
    build_single_step_messages,
    select_compress_range_by_tokens,
)

from eval_baseline import DebugLogger, setup_eval_logging
from vllm_engine import make_sampling_params, prepare_vllm_input


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
    current_chunk: int = 0
    done: bool = False
    answer_text: Optional[str] = None
    pred_idx: Optional[int] = None
    error: Optional[str] = None
    # last_compress_trigger: True when system injected a trigger this step
    # so caller can skip user_question on the same step.
    _last_trigger: bool = False
    chunks_generated: int = 0


def _resolve_frame_paths(
    video_path: str,
    chunk_idx: int,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Optional[List[str]]:
    """Mirrors StreamingAgentLoop._get_frame_paths() — pre-extracted JPEGs.

    Returns None if frames_root not configured or insufficient frames found
    (caller falls back to online video decode).
    """
    if not frames_root:
        return None
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    vp = Path(video_path)
    if video_root:
        try:
            rel = vp.relative_to(Path(video_root))
            frame_dir = Path(frames_root) / rel.with_suffix("")
        except ValueError:
            frame_dir = Path(frames_root) / vp.with_suffix("")
    else:
        frame_dir = Path(frames_root) / vp.with_suffix("")

    if not frame_dir.exists():
        return None

    start_frame = int(video_start) + 1
    end_frame = int(video_end) + 1
    paths = []
    for i in range(start_frame, end_frame + 1):
        fp = frame_dir / f"frame_{i:06d}.jpg"
        if fp.exists():
            paths.append(str(fp))

    if len(paths) < max(1, n_frames // 2):
        return None
    return paths


def _maybe_compress_trigger(memory: MemoryState, chunk_idx: int) -> str:
    """Return <compress_trigger range="..."/> if memory threshold fires.

    v11.3: range size selected by select_compress_range_by_tokens so eval
    matches the post-fix agent_loop policy (variable range driven by token
    budget, was hardcoded to first 4 thinks).
    """
    if not memory.should_compress():
        return ""
    n = select_compress_range_by_tokens(
        memory.recent_thinks,
        token_count_fn=memory._token_count,
    )
    if n <= 0:
        return ""
    oldest = memory.recent_thinks[:n]
    chunks = [t["chunk"] for t in oldest]
    t_start = min(chunks) * AGENT_CHUNK_SEC
    t_end = (max(chunks) + 1) * AGENT_CHUNK_SEC
    return f'<compress_trigger range="{t_start}-{t_end}"/>'


def _prepare_step_messages(runner: _SampleRunner) -> List[Dict]:
    """Replicates StreamingAgentLoop.step() up to but not including generate."""
    chunk_idx = runner.current_chunk
    snapshot = runner.memory.snapshot(chunk_idx)

    user_question = runner.query if chunk_idx == runner.ask_chunk else None
    if user_question:
        ask_time = chunk_idx * AGENT_CHUNK_SEC
        already = any(
            q["question"] == user_question and q.get("ask_time") == ask_time
            for q in runner.memory.queries
        )
        if not already:
            runner.memory.add_query(user_question, ask_time)

    compress_trigger = _maybe_compress_trigger(runner.memory, chunk_idx)
    runner._last_trigger = bool(compress_trigger)

    if compress_trigger and not user_question:
        user_input = compress_trigger
    elif user_question:
        user_input = user_question
    else:
        user_input = ""

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
    )


def _apply_step_output(runner: _SampleRunner, output_text: str) -> None:
    """Replicates the post-generate state update in StreamingAgentLoop.step().

    Eval mode → recall path is dead (allow_recall=False at sampler level
    AND we ignore <action>recall</action> if it slips through).
    """
    parsed = parse_agent_output(output_text)
    chunk_idx = runner.current_chunk

    if parsed.get("think"):
        runner.memory.add_think(chunk_idx, parsed["think"])

    action = parsed.get("action")
    if action == "compress":
        summary = parsed["payload"].get("summary", {})
        if summary and "time_range" in summary:
            tr = summary["time_range"]
            compressed_chunks = []
            for t in runner.memory.recent_thinks:
                cs = t["chunk"] * AGENT_CHUNK_SEC
                ce = cs + AGENT_CHUNK_SEC
                if cs >= tr[0] and ce <= tr[1]:
                    compressed_chunks.append(t["chunk"])
            runner.memory.compress(summary, compressed_chunks=compressed_chunks)
    elif action == "response":
        answer_text = parsed["payload"].get("response", "")
        if answer_text:
            response_time = chunk_idx * AGENT_CHUNK_SEC
            # Attach to most-recent unanswered query (mirrors _record_answer).
            for q in reversed(runner.memory.queries):
                if not q.get("answers"):
                    runner.memory.answer_query(q["question"], answer_text, response_time)
                    break
            runner.answer_text = answer_text
            runner.done = True


def _option_match(answer_text: str, options: List[str]) -> int:
    """Map free-text answer to an option index. Same fallback chain as
    eval_baseline.parse_answer but operating on the agent <response>.
    """
    s = (answer_text or "").strip().upper()
    for i, opt in enumerate(options):
        if s.startswith(opt.upper()):
            return i
    if s:
        head = s[:1]
        for i, opt in enumerate(options):
            if opt.upper() == head:
                return i
        for i, opt in enumerate(options):
            if opt.upper() in s:
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
) -> List[_SampleRunner]:
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

            if datum.get("options"):
                query = (
                    question_prefix + datum["question"] + "\n"
                    + "\n".join(datum["options"]) + question_postfix
                )
            else:
                query = datum["question"]

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
    question_postfix: str = "\nPlease select the correct answer.",
    max_new_tokens: int = 256,
    frames_per_chunk: int = 8,
    max_chunks: int = 30,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    frames_root: Optional[str] = None,
    video_root: Optional[str] = None,
    temperature: float = 0.0,
    repetition_penalty: float = 1.1,
    debug: bool = False,
    debug_dir: Optional[str] = None,
):
    """Chunk-lockstep streaming MCQ eval via vLLM batched generate.

    All samples advance one chunk per orchestration round. At each round
    every live sample contributes one prompt to a single llm.generate()
    call, then each parses its own output and advances state. Samples
    that emit <action>response</action> are removed from the live set.

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
    runners = _build_runners(
        dataset, options, question_prefix, question_postfix,
        frames_per_chunk, max_chunks, min_pixels, max_pixels,
        frames_root, video_root, tokenizer=tokenizer,
    )

    log.info(
        f"vLLM streaming eval: {len(runners)} samples, "
        f"max_chunks={max_chunks}, frames_per_chunk={frames_per_chunk}"
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
            vllm_inputs = [
                prepare_vllm_input(m, processor) for _, m in active_pairs
            ]
        except Exception as e:
            log.error(f"vLLM input prep failed at chunk {chunk_idx}: {e}", exc_info=True)
            for r, _ in active_pairs:
                r.error = f"prep_input:{e}"
                r.done = True
            pbar.update(1)
            continue

        # Phase C: batched generate
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        n_total_calls += len(outputs)

        # Phase D: apply outputs
        for (r, _), out in zip(active_pairs, outputs):
            try:
                text = out.outputs[0].text
                _apply_step_output(r, text)
                r.chunks_generated += 1
                if debug:
                    dbg.log({
                        "idx": r.idx, "chunk_idx": chunk_idx,
                        "video": r.datum.get("video"),
                        "output": text,
                        "memory_thinks": len(r.memory.recent_thinks),
                        "memory_compressed": len(r.memory.compressed_segments),
                        "compress_trigger": r._last_trigger,
                    })
            except Exception as e:
                r.error = f"apply:{e}"
                r.done = True

        # Advance chunk pointer for all runners that participated this round
        for r, _ in active_pairs:
            if not r.done:
                r.current_chunk += 1
                if r.current_chunk >= r.num_chunks:
                    # Reached end without ever emitting <response>; mark done.
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
