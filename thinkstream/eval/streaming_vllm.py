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
    # v12 protocol: switches build_single_step_messages to SYSTEM_PROMPT_V12
    # and the caller's prepare_vllm_input to pass tools=TOOLS_SCHEMA.
    protocol_version: str = "v11"


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
        protocol_version=runner.protocol_version,
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
    protocol_version: str = "v11",
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
                protocol_version=protocol_version,
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
                protocol_version=protocol_version,
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
    protocol_version: str = "v11",
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

    # v12.0: hand TOOLS_SCHEMA to chat_template when protocol_version='v12'
    # so <tools>...</tools> renders in system prompt and the model can emit
    # <tool_call>{...}</tool_call>. v11 path keeps tools=None (legacy).
    tools_for_template = None
    if protocol_version == "v12":
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
        tools_for_template = TOOLS_SCHEMA
    dbg = DebugLogger(
        os.path.join(debug_dir, "streaming_debug_vllm.jsonl"),
        enabled=debug, rank=0,
    )

    tokenizer = processor.tokenizer
    runners = _build_runners(
        dataset, options, question_prefix, question_postfix,
        frames_per_chunk, max_chunks, min_pixels, max_pixels,
        frames_root, video_root, tokenizer=tokenizer,
        protocol_version=protocol_version,
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
                prepare_vllm_input(m, processor, tools=tools_for_template) for _, m in active_pairs
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
    current_chunk: int = 0
    done: bool = False
    error: Optional[str] = None
    _last_trigger: bool = False
    # Per-chunk results, shape matches grpo.py:736-758 contract.
    chunk_results: List[Dict] = field(default_factory=list)


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


def _apply_rollout_output(
    runner: _RolloutRunner, output_text: str, tokenizer, *,
    compress_budget: int,
) -> None:
    """Per-chunk state advance + chunk_results append.

    Differs from _apply_step_output (eval) in two ways:
      1. Does NOT set runner.done on response — RL rolls a few chunks past
         ask_chunk so the model emits the full <think><action><response>
         under post-answer pressure (matches grpo.py legacy rollout).
      2. Records the legacy per-chunk dict (grpo.py:736-758 contract):
         action, think, payload, raw_output, generated_tokens,
         memory_token_count, compress_budget, recall_returned_chunks,
         window_start, window_end.
    """
    chunk_idx = runner.current_chunk
    parsed = parse_agent_output(output_text)

    if parsed.get("think"):
        runner.memory.add_think(chunk_idx, parsed["think"])

    action = parsed.get("action") or "unknown"
    if action == "compress":
        summary = parsed.get("payload", {}).get("summary", {})
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
        answer_text = parsed.get("payload", {}).get("response", "")
        if answer_text:
            response_time = chunk_idx * AGENT_CHUNK_SEC
            for q in reversed(runner.memory.queries):
                if not q.get("answers"):
                    runner.memory.answer_query(q["question"], answer_text, response_time)
                    break

    runner.chunk_results.append({
        "chunk_idx": chunk_idx,
        "action": action,
        "think": parsed.get("think", ""),
        "payload": parsed.get("payload", {}),
        "raw_output": output_text,
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
    })


def streaming_vllm_rollout(
    step_inputs: List[Dict],
    llm,
    processor,
    tokenizer,
    *,
    group_size: int,
    max_new_tokens: int = 256,
    rollout_max_chunks: int = 30,
    rollout_extra_chunks: int = 5,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    frames_root: Optional[str] = None,
    video_root: Optional[str] = None,
    compress_budget: Optional[int] = None,
    protocol_version: str = "v11",
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
        from thinkstream.model.agent_loop import RECENT_THINKS_TOKEN_BUDGET
        compress_budget = RECENT_THINKS_TOKEN_BUDGET

    # ── Build N × G runners ──
    runners: List[_RolloutRunner] = []
    for s_idx, raw_sample in enumerate(step_inputs):
        data_path = raw_sample.get("data_path", "")
        rel_video = raw_sample.get("video_path", "")
        if data_path and rel_video and not Path(rel_video).is_absolute():
            video_path = str(Path(data_path) / rel_video)
        else:
            video_path = rel_video or ""
        ask_chunk = int(raw_sample.get("chunk_idx", rollout_max_chunks - 1))
        question = _extract_user_question(raw_sample)
        max_chunks_this = min(ask_chunk + rollout_extra_chunks, rollout_max_chunks)
        for g in range(group_size):
            runners.append(_RolloutRunner(
                sample_idx=s_idx,
                gen_idx=g,
                raw_sample=raw_sample,
                video_path=video_path,
                query=question,
                ask_chunk=ask_chunk,
                max_chunks=max_chunks_this,
                memory=MemoryState(tokenizer=tokenizer),
                frames_root=frames_root,
                video_root=video_root,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
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

    # ── Chunk-lockstep loop ──
    max_global_chunk = max(r.max_chunks for r in runners)
    for chunk_idx in range(max_global_chunk):
        live = [r for r in runners
                if not r.done and r.current_chunk == chunk_idx]
        if not live:
            break

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

        # Phase B: vLLM input + batch generate
        # v12.0: pass tools=TOOLS_SCHEMA to chat_template (so <tools> block
        # renders + model can emit <tool_call>). v11 keeps tools=None.
        _rollout_tools = None
        if protocol_version == "v12":
            from thinkstream.data.agent_protocol import TOOLS_SCHEMA
            _rollout_tools = TOOLS_SCHEMA
        try:
            vllm_inputs = [
                prepare_vllm_input(m, processor, tools=_rollout_tools)
                for m in messages_list
            ]
        except Exception as e:
            for r in live_active:
                r.error = f"prep_input:{e}"
                r.done = True
            continue
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

        # Phase C: apply outputs + advance state
        for r, out in zip(live_active, outputs):
            try:
                text = out.outputs[0].text
                _apply_rollout_output(r, text, tokenizer, compress_budget=compress_budget)
            except Exception as e:
                r.error = f"apply:{e}"
                r.done = True
                continue
            r.current_chunk += 1
            # RL stops only when (a) the runner reached max_chunks (a few
            # past ask_chunk) or (b) it emitted a response after ask_chunk.
            # Mirrors grpo.py:760-761 early-stop.
            last_action = r.chunk_results[-1]["action"]
            if r.current_chunk >= r.max_chunks:
                r.done = True
            elif last_action == "response" and chunk_idx >= r.ask_chunk:
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
                else:
                    # Pad: this gen finished early (response emitted past ask_chunk).
                    merged["generated_tokens"].append(_torch.tensor([], dtype=_torch.long))
                    merged["memory_token_count"].append(0)
                    merged["compress_budget"].append(0)
                    merged["recall_returned_chunks"].append([])
            merged_chunk_results.append(merged)

        all_rollout_results.append({
            "raw_sample": step_inputs[s_idx],
            "chunk_results": merged_chunk_results,
        })

    return all_rollout_results
