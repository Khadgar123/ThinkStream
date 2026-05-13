# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ThinkStream streaming-video agent loop for verl 0.4+.
#
# DESIGN — MemAgent-style recurrent rollout with true KV:
# ────────────────────────────────────────────────────────────────────────────
# Each ordinary video chunk appends one new chat user turn to an existing
# KV stream. The rendered visual carrier contains only the current 1s chunk
# (2 frames); StreamingWindowInferenceEngine owns the visual KV window and
# evicts old video blocks CASIA-style:
#
#     turn_chunk_N =
#       <|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>     ← common prefix
#       <|im_start|>user\n{question}\n<|im_end|>                ← common prefix
#       <|im_start|>user\n
#         <memory>...</memory>
#         <active_query>...</active_query> plus <response_history>...</response_history>
#                                             (while a query is live)
#         <visual_window>{current chunk header}</visual_window>
#         {video_meta for current chunk's 2 frames}
#         <user_input>...</user_input>     (question text or bare compress trigger)
#       <|im_end|>
#       <|im_start|>assistant\n
#     ...
#
# CACHE BEHAVIOUR (v12.13, 2026-05-02):
#   Two independent caches help streaming-video rollout. Each targets a
#   different bottleneck.
#
#   The rollout backend is the local HF/CASIA-style streaming engine, not
#   full-prompt vLLM. Full re-prefill would repeatedly append old text and
#   visual evidence under true KV, so it is disabled for RL correctness.
#
# PROMPT LAYOUT (must match SFT exactly — see
# thinkstream/data/agent_protocol.py:213-214 build_user_content):
#
#     prompt_chunk_t = [
#       system + user_q                        ← stable across chunks
#       <memory>                               ← monotonic append; SFT-first
#       (active_query + response_history)      ← optional while a query is live
#       <visual_window header>                 ← current chunk metadata
#       video_meta frame block                 ← current 2 frames only
#       <recall_result> (optional)             ← chunk-specific
#       <user_input>                           ← question or bare compress trigger
#     ]
#
# WINDOW MODE (THINKSTREAM_VISUAL_WINDOW_MODE):
#   Retained for data compatibility. In true-KV RL the prompt renderer receives
#   the resolved frame paths but build_user_content() consumes only the current
#   chunk's final 2 paths; the physical visual window is maintained by
#   StreamingWindowInferenceEngine.video_flex_window_size.
#
# SFT ALIGNMENT (the 5 things that must match pass5_messages.py):
#   1. Frame paths use 1-indexed numbering: frame_{ci*FPC + fi + 1:06d}.jpg
#   2. Pre-extracted frames render through the selected frame protocol:
#      `ts_image` uses frame-tag text + image items; `video_meta` uses one
#      Qwen video block with explicit video_metadata. Both carry real time.
#   3. Frame timestamps/metadata use frame_idx / fps, where
#      frame_idx = window_start*FPC + i.
#   4. Compress turn uses a compression-only system prompt plus bare
#      <compress_trigger/> (v12.12: no range). It carries memory only:
#      active_query, visual_window, and media carriers are suppressed so
#      SFT/RL/eval share the same text-only compression payload.
#   5. Recall result rendering as <recall_result>{...}</recall_result>
#      JSON dict (source/time/text).
#
# DEFERRED (must be addressed before claiming full deploy parity):
#   D1. Per-chunk attention reset / training-time per-chunk forward
#       (P0.4 / P0.3). At training time verl's actor sees the stitched
#       long sequence; user-block tokens have mask=0 so they don't
#       contribute to loss, but attention activations at assistant
#       positions still see prior chunks' user_blocks. Real fix is a
#       block-diagonal attention mask in actor forward, which is a verl
#       trainer change (not local to the recipe). Until then, the
#       gradient is correct for the loss but attention context differs
#       from rollout-time per-chunk independence.
#
# verl AgentLoopOutput stitching:
#   output.prompt_ids   = [system + user_q]  (the COMMON prefix)
#   output.response_ids = [user_block_0 + asst_0 + user_block_1 + asst_1 + ...]
#   output.response_mask = [0...0 | 1...1 | 0...0 | 1...1 | ...]
#   output.multi_modal_data["images"] = image payloads for timestamped frames
#
# Registered under `"thinkstream_streaming_agent"`.
from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from thinkstream.data.agent_protocol import (
    RECALL_RETURN_CHUNKS,
    build_recalled_frames_metadata,
    recall_time_string_for_chunks,
    resolve_chunk_frame_paths,
    select_recall_chunks,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Token budget fallback for compression. Current RL defaults to pass2's
# offline compact-memory boundaries when parquet provides them; this budget
# path remains for legacy parquet or explicit runtime-trigger ablations.
#
#   COMPACT_MEMORY_TEXT_TOKEN_BUDGET = 3200
#   COMPACT_MEMORY_MIN_NEW_CHUNKS    = 24
#   COMPACT_MEMORY_MAX_NEW_CHUNKS    = 36
#
# v12.13 (2026-05-03): RL was using a hardcoded 3200 + word-count×1.3
# estimate, which drifted from pass2 / eval (both use the real Qwen
# tokenizer). Diverging means RL's compress fires at a different memory
# state than what the model trained on under SFT, breaking SFT-RL
# distribution alignment. Now: import the constants directly and use
# self.tokenizer (already loaded by AgentLoopBase) for accurate counting.
try:
    from scripts.agent_data.config import (  # type: ignore
        COMPACT_MEMORY_TEXT_TOKEN_BUDGET,
        COMPACT_MEMORY_MIN_NEW_CHUNKS,
        COMPACT_MEMORY_MAX_NEW_CHUNKS,
    )
    RECENT_THINKS_TOKEN_BUDGET = COMPACT_MEMORY_TEXT_TOKEN_BUDGET
    COMPRESS_TOKEN_THRESHOLD = COMPACT_MEMORY_TEXT_TOKEN_BUDGET
    COMPRESS_RANGE_MIN = COMPACT_MEMORY_MIN_NEW_CHUNKS
    COMPRESS_RANGE_MAX = COMPACT_MEMORY_MAX_NEW_CHUNKS
except ImportError:
    # Fallback for envs that don't have THINKSTREAM_HOME on PYTHONPATH —
    # values mirror config.py but won't auto-update if config changes.
    RECENT_THINKS_TOKEN_BUDGET = 3200
    COMPRESS_TOKEN_THRESHOLD = 3200
    COMPRESS_RANGE_MIN = 24
    COMPRESS_RANGE_MAX = 36

# Backward-compat alias for any external import; new code should use
# COMPRESS_TOKEN_THRESHOLD directly.
DEFAULT_COMPRESS_TOKEN_THRESHOLD = COMPRESS_TOKEN_THRESHOLD


# Module-level placeholder so hydra's `_target_:
# thinkstream.rl.streaming_agent_loop.ThinkStreamStreamingAgentLoop`
# resolves. Populated by _register_streaming_agent_loop() at import time.
ThinkStreamStreamingAgentLoop: Any = None


def _resolve_frame_dir(video_path: str, frames_root: str) -> Optional[Path]:
    """Find the pre-extracted frame directory for a video."""
    if not video_path or not frames_root:
        return None
    root = Path(frames_root)
    if not root.exists():
        return None
    vp = Path(video_path)
    candidates = [root / vp.stem, root / vp.with_suffix("").name]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    if any(root.glob("frame_*.jpg")):
        return root
    return None


def _chunk_frame_paths(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    frames_per_chunk: int = 2,
) -> List[str]:
    """Return absolute frame_path strings for chunk `chunk_idx`.

    Frame numbering matches SFT (pass5_messages.py:134, pipeline.py:321):
      frame_{ci * FPC + fi + 1:06d}.jpg     (1-indexed!)
    Returns [] if any frame is missing — caller falls back to text-only.
    """
    frame_dir = _resolve_frame_dir(video_path, frames_root)
    if frame_dir is None:
        return []
    return resolve_chunk_frame_paths(
        frame_dir,
        chunk_idx,
        frames_per_chunk=frames_per_chunk,
    )


def _build_visual_window(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    visual_window_chunks: int,
    frames_per_chunk: int,
    chunk_sec: float = 1.0,
    mode: str = "sliding",
) -> Tuple[List[str], int, int]:
    """Build the current 1s visual chunk for chunk N.

    Returns:
      flat_paths:     current chunk frame paths only (2 frames)
      window_start_chunk, window_end_chunk
    """
    del visual_window_chunks, chunk_sec, mode
    cf = _chunk_frame_paths(video_path, frames_root, chunk_idx, frames_per_chunk)
    if not cf:
        return [], chunk_idx, chunk_idx
    return list(cf), chunk_idx, chunk_idx


# ---------------------------------------------------------------------------
# Recall retriever — BM25 over the raw per-chunk think archive.
# ---------------------------------------------------------------------------
def _retrieve_from_memory(
    state_think_archive: List[Dict[str, Any]],
    query_text: str,
    time_range: Optional[Tuple[float, float]] = None,
    top_k: int = RECALL_RETURN_CHUNKS,
    chunk_sec: float = 1.0,
) -> Dict[str, Any]:
    """Return a legacy recall_result dict used internally before metadata-only
    serialisation as <recall_result>...</recall_result> on the NEXT chunk's
    user message. (True intra-chunk multi-turn recall — assistant tool_call
    → user/tool recall_result+frames → assistant answer — is deferred.)
    """
    if not (query_text or "").strip():
        return {"source": "memory", "time": "", "text": "", "returned_chunks": []}
    archive: List[Dict[str, Any]] = []
    for entry in state_think_archive or []:
        text = entry.get("text", entry.get("think", "")) or ""
        if not text:
            continue
        try:
            chunk = int(entry.get("chunk", entry.get("chunk_idx", -1)))
        except (TypeError, ValueError):
            continue
        if chunk < 0:
            continue
        archive.append({
            "chunk": chunk,
            "time": entry.get("time") or (
                f"{int(chunk * chunk_sec)}-{int((chunk + 1) * chunk_sec)}"
            ),
            "text": text,
        })
    if not archive:
        tr_text = ""
        if time_range is not None:
            tr_text = f"{int(time_range[0])}-{int(time_range[1])}"
        return {
            "source": "memory",
            "time": tr_text,
            "text": "No relevant past observation in the requested time range.",
            "returned_chunks": [],
        }
    from thinkstream.models.agent_loop import bm25_retrieve
    query = {"query": query_text}
    if time_range is not None:
        query["time_range"] = [float(time_range[0]), float(time_range[1])]
    result = bm25_retrieve(query, archive, max_results=top_k)
    returned_chunks = select_recall_chunks(result.get("returned_chunks") or [])
    if not returned_chunks:
        tr_text = ""
        if time_range is not None:
            tr_text = f"{int(time_range[0])}-{int(time_range[1])}"
        return {
            "source": "memory",
            "time": tr_text,
            "text": "No relevant past observation in the requested time range.",
            "returned_chunks": [],
        }
    return {
        "source": "memory",
        "time": recall_time_string_for_chunks(returned_chunks),
        "text": result.get("text_content", ""),
        "returned_chunks": returned_chunks,
    }


def _count_recent_thinks_tokens(
    recent_thinks: List[Dict[str, Any]],
    tokenizer=None,
) -> int:
    """Count tokens in recent_thinks, matching pass2/eval EXACTLY.

    pass2 (scripts/agent_data/pass2_rollout.py:183) and eval
    (thinkstream/model/agent_loop.py:170) both use:
        tokenizer.encode(text, add_special_tokens=False)
    falling back to len(text)//4 when tokenizer is None.

    RL must use the same accounting or its compress trigger fires at a
    different memory state than what SFT data was generated under,
    diverging the train/RL/eval distribution.
    """
    total = 0
    for t in recent_thinks or []:
        text = t.get("text") if isinstance(t, dict) else str(t)
        if not text:
            continue
        if tokenizer is not None:
            try:
                total += len(tokenizer.encode(text, add_special_tokens=False))
                continue
            except Exception:
                pass  # fall through to char-based estimate
        total += len(text) // 4
    return total


def _count_visible_memory_tokens(
    compressed_summaries: List[Dict[str, Any]],
    recent_thinks: List[Dict[str, Any]],
    tokenizer=None,
) -> int:
    return (
        _count_recent_thinks_tokens(compressed_summaries, tokenizer=tokenizer)
        + _count_recent_thinks_tokens(recent_thinks, tokenizer=tokenizer)
    )


# Backward-compat alias for any external import.
def _estimate_recent_thinks_tokens(recent_thinks: List[Dict[str, Any]]) -> int:
    """DEPRECATED: kept as a no-tokenizer shim. Prefer
    _count_recent_thinks_tokens(recent_thinks, tokenizer=...)."""
    return _count_recent_thinks_tokens(recent_thinks, tokenizer=None)


# ---------------------------------------------------------------------------
# verl-side registration. Wrapped so the module is importable in dev
# environments lacking verl/ray/torch.
# ---------------------------------------------------------------------------
def _register_streaming_agent_loop():
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
    from verl.utils.profiler import simple_timer  # type: ignore
    from verl.workers.rollout.replica import TokenOutput  # type: ignore

    from thinkstream.data.agent_protocol import (  # type: ignore
        action_space_error_for_turn,
        append_query_answer_with_timing,
        build_recall_result_metadata,
        build_user_content,
        build_recall_result_user_content,
        decode_agent_output_tokens,
        normalize_frame_protocol,
        normalize_render_layout,
        parse_agent_output,
        format_memory_block,
        query_is_complete,
        query_expected_answer_chunks,
        query_completed_answer_chunks,
        system_prompt_for_frame_protocol,
        tools_for_turn,
    )
    from thinkstream.trainer.rollout import (  # type: ignore
        VideoTrajectoryState,
        default_update_state,
    )

    def _runtime_mm_processor_kwargs() -> Dict[str, int]:
        try:
            from scripts.agent_data.config import (
                RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
            )
        except ImportError:
            _RTKW = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
        return {
            "min_pixels": int(
                os.environ.get("IMAGE_MIN_PIXELS")
                or os.environ.get("MIN_PIXELS")
                or _RTKW["min_pixels"]
            ),
            "max_pixels": int(
                os.environ.get("IMAGE_MAX_PIXELS")
                or os.environ.get("MAX_PIXELS")
                or _RTKW["max_pixels"]
            ),
        }

    class ThinkStreamStreamingAgentLoop(AgentLoopBase):
        """MemAgent-style chunk-level rollout for streaming video."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            self.max_model_len = self.rollout_config.max_model_len or (self.prompt_length + self.response_length)
            # response_length is the stitched trajectory buffer used by verl's
            # loss tensors. A single chunk action must be much smaller. Keep
            # normal streaming/recall turns at the SFT/eval budget and give
            # compression enough room for the JSON summary without letting
            # ordinary turns drift into long repeated think loops.
            self.max_tokens_per_action = int(
                os.environ.get("THINKSTREAM_MAX_TOKENS_PER_ACTION", "256") or 256
            )
            self.max_tokens_per_compress_action = int(
                os.environ.get("THINKSTREAM_COMPRESS_MAX_TOKENS_PER_ACTION", "512")
                or 512
            )
            self.frame_protocol = normalize_frame_protocol(
                os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta")
            )
            self.render_layout = normalize_render_layout(
                os.environ.get("THINKSTREAM_RENDER_LAYOUT", "standard_query_last")
            )
            mt = self.rollout_config.multi_turn
            # v12.13 (2026-05-02): verl's MultiTurnConfig dataclass rejects
            # custom fields (max_turns / frames_root / frames_per_chunk /
            # visual_window_chunks / recall_stub_text). All of those are
            # now read from environment variables set by the launch script.
            # max_assistant_turns is the verl-native cap.
            self.max_chunks = int(
                getattr(mt, "max_assistant_turns", 0) or 360
            )
            self.frames_root = str(
                os.environ.get("THINKSTREAM_FRAMES_ROOT", "")
            )
            self.frames_per_chunk = int(
                os.environ.get("THINKSTREAM_FRAMES_PER_CHUNK", "2") or 2
            )
            self.visual_window_chunks = int(
                os.environ.get("THINKSTREAM_VISUAL_WINDOW_CHUNKS", "8") or 8
            )
            self.chunk_sec = float(
                os.environ.get("THINKSTREAM_CHUNK_SEC", "1.0") or 1.0
            )
            # Runtime-token fallback. Normal RL uses offline pass2 boundaries
            # from extra_info.offline_compress_chunks so query injection after
            # compression matches SFT/pass5.
            self.compress_token_threshold = int(
                os.environ.get(
                    "THINKSTREAM_COMPRESS_THRESHOLD",
                    str(COMPRESS_TOKEN_THRESHOLD),
                )
                or COMPRESS_TOKEN_THRESHOLD
            )
            trigger_source = str(
                os.environ.get(
                    "THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE",
                    "offline_pass2_boundaries",
                )
                or "offline_pass2_boundaries"
            ).strip().lower()
            if trigger_source not in {
                "offline_pass2_boundaries",
                "runtime_memory_threshold",
            }:
                trigger_source = "offline_pass2_boundaries"
            self.compress_trigger_source = trigger_source
            self.recall_stub_text = str(
                os.environ.get(
                    "THINKSTREAM_RECALL_STUB",
                    "(no relevant past observation found)",
                )
            )
            # v12.13: visual window mode (see _compute_window_start
            # docstring). Default "sliding" matches SFT
            # agent_protocol.py:275 + pass5_messages.py exactly. Opt in
            # to "expanding" only after regenerating SFT data with the
            # same THINKSTREAM_VISUAL_WINDOW_MODE env var.
            mode = str(
                os.environ.get("THINKSTREAM_VISUAL_WINDOW_MODE", "sliding")
            ).lower()
            if mode not in ("sliding", "expanding"):
                mode = "sliding"
            self.visual_window_mode = mode

            # v12.13 D1: max recall tool_call rounds per video chunk.
            # 0 = legacy behavior (recall result delivered next chunk, no
            # frames). 1 = one in-chunk multi-turn cycle (assistant emits
            # tool_call → system injects tool_response w/ frames →
            # assistant emits answer); matches SFT pass5 shape B exactly.
            # >=2 lets the model retry recall if first result was unhelpful
            # (rare in SFT data; default 1 to match training distribution).
            self.max_recall_per_chunk = int(
                os.environ.get("THINKSTREAM_MAX_RECALL_PER_CHUNK", "1") or 1
            )

            # v12.14 Option B Phase 3: dual-mode rollout output.
            #   "stitched" (default): one AgentLoopOutput per trajectory
            #                          (response_ids = all chunks
            #                          stitched). Backward compat with
            #                          existing trainer + reward path.
            #   "recurrent" (experimental, Phase 4 trainer wiring):
            #                list[AgentLoopOutput] — one per assistant
            #                action. AgentLoopWorker (Phase 1) flattens
            #                across the batch, ray_trainer (Phase 4d)
            #                computes 1D GRPO advantage on trajectory
            #                final rewards and broadcasts via
            #                sample_index back to action rows.
            mode = str(
                os.environ.get("THINKSTREAM_RECURRENT_MODE", "stitched")
            ).lower()
            if mode not in ("stitched", "recurrent"):
                mode = "stitched"
            self.recurrent_mode = mode
            engine = str(
                os.environ.get("THINKSTREAM_ROLLOUT_ENGINE", "streaming")
            ).strip().lower()
            if engine not in {"streaming", "hf_streaming", "local_streaming", "casia"}:
                logger.warning(
                    "Ignoring THINKSTREAM_ROLLOUT_ENGINE=%r; true-KV "
                    "streaming rollout is required.",
                    engine,
                )
            self.rollout_engine = "streaming"

        # -------------------------------------------------------------------
        # Per-chunk user-side text. Mirrors SFT/pass5 layout so train/RL
        # distributions line up. Pre-extracted frames are rendered through the
        # same late-bound frame protocol as pass5/SFT/eval.
        # -------------------------------------------------------------------
        def _format_user_input(
            self,
            chunk_idx: int,
            question: str,
            ask_chunks: List[int],
            triggered_questions: Optional[List[Dict[str, Any]]],
        ) -> Optional[str]:
            """Render the <user_input>...</user_input> string for this chunk.

            Two paths:
              - Multi-Q mode (triggered_questions non-empty): render the
                question(s) whose ask_chunk == chunk_idx, including the
                MCQ options when present so the model has the option text
                in the prompt at decision time.
              - Single-Q legacy mode (`question` set, ask_chunks given):
                fire the same single question whenever chunk_idx >=
                min(ask_chunks). Backward compat with the (video, question)
                flatten parquet shape.

            Returns None when nothing should be injected this chunk.
            """
            # Multi-Q path
            if triggered_questions:
                blocks: List[str] = []
                for q in triggered_questions:
                    qtxt = q.get("question", "") or ""
                    if qtxt:
                        blocks.append(qtxt)
                if blocks:
                    return "\n---\n".join(blocks)
                return None
            # Legacy single-Q path
            if question and ask_chunks and chunk_idx >= min(ask_chunks):
                return question
            return None

        def _build_chunk_user_content(
            self,
            *,
            state: "VideoTrajectoryState",
            chunk_idx: int,
            window_paths: List[str],
            window_start_chunk: int,
            window_end_chunk: int,
            question: str,
            ask_chunks: List[int],
            triggered_questions: Optional[List[Dict[str, Any]]],
            queries: Optional[List[Dict[str, Any]]],
            recall_result: Optional[Dict[str, Any]],
            compress_trigger_range: Optional[Tuple[int, int]],
            inter_chunk: bool,
        ) -> List[Dict[str, Any]]:
            """Build the user content list for chunk N.

            inter_chunk=True marks a compression turn. It is text-only:
            user_input carries the bare compress trigger and memory carries
            the compression target. Queries, recall-answer context, visual
            window, and frame carriers are suppressed to match pass5/runtime.

            Mirrors the shared SFT/runtime layout in
            thinkstream.data.agent_protocol.build_user_content EXACTLY:
              <user_input> → <memory> → <visual_window> + video_meta frames →
              <active_query> + <response_history> →
              <recalled_frames> + visual evidence → <recall_result> metadata

            Distribution alignment is the hard constraint. Do not rearrange
            text or visual blocks to chase prefix-cache hits; the true-KV
            engine receives only the current 2-frame video block here and
            maintains the physical visual window itself.
            """
            if compress_trigger_range is not None:
                user_input_text = "<compress_trigger/>"
            else:
                user_input_text = self._format_user_input(
                    chunk_idx, question, ask_chunks, triggered_questions,
                ) or ""
            try:
                mem_snapshot = {
                    "compressed_segments": state.compressed_summaries,
                    "compressed": state.compressed_summaries,
                    "recent_thinks": state.recent_thinks,
                }
                mem_text = format_memory_block(mem_snapshot)
            except Exception:
                mem_snapshot = {
                    "compressed_segments": [],
                    "compressed": [],
                    "recent_thinks": [],
                }
                mem_text = ""
            _RTKW = _runtime_mm_processor_kwargs()
            return build_user_content(
                mem_text,
                chunk_idx,
                "",
                user_input=user_input_text,
                queries=queries or [],
                recall_result=recall_result,
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
                frame_paths=window_paths,
                frame_protocol=self.frame_protocol,
                inter_chunk=inter_chunk,
                memory_snapshot=mem_snapshot,
                render_layout=self.render_layout,
            )
        async def _execute_recall(
            self, args: Dict[str, Any], state: "VideoTrajectoryState",
            *, video_path: str = "",
        ) -> Dict[str, Any]:
            """v12.13 D1: build the COMPLETE tool response for shape-B recall.

            Returns a dict matching what SFT pass5_messages.py:280-380 expects:
              {
                "recall_result": {source, returned_chunks, time, status}
                                 — metadata-only model-visible result
                "recalled_frames": {time_range, source, n_frames, frame_paths}
                                   or None on retrieval failure / no frames
                                   — same shape as pass3c rendering input
              }

            In normal v12.13+ mode these fields are rendered immediately as
            the same-chunk tool response. The prompt builder anchors recalled
            frame paths to HISTORICAL chunk timestamps (not the current
            chunk), so the model sees these as old frames from time T, not
            current visual evidence.
            """
            query = (args.get("query") or args.get("keywords") or
                     args.get("text") or "")
            time_range = args.get("time_range")
            tr_tuple: Optional[Tuple[float, float]] = None
            # Recall tool schema (agent_protocol.py:417) emits time_range
            # as "start-end" string (e.g. "10-30"); legacy eval code may
            # pass list/tuple [start, end]. Accept BOTH.
            if isinstance(time_range, str) and time_range.strip():
                m = re.match(r"\s*([\d.]+)\s*-\s*([\d.]+)\s*", time_range)
                if m:
                    try:
                        tr_tuple = (float(m.group(1)), float(m.group(2)))
                    except ValueError:
                        tr_tuple = None
            elif isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
                try:
                    tr_tuple = (float(time_range[0]), float(time_range[1]))
                except (TypeError, ValueError):
                    tr_tuple = None

            # ── Text retrieval (existing path) ──────────────────────────
            try:
                text_result = _retrieve_from_memory(
                    getattr(state, "think_archive", []) or [],
                    query_text=query,
                    time_range=tr_tuple,
                    chunk_sec=self.chunk_sec,
                )
            except Exception as e:
                logger.warning("recall retrieval failed: %s", e)
                text_result = {"source": "memory", "time": "", "text": "(retrieval error)"}

            # ── Historical frame extraction (D1) ───────────────────────
            # Text retrieval chooses candidate chunks first, then we cap to
            # top-K and render only those chunks. Do not expand the model's
            # requested time_range into an unbounded frame interval.
            selected_chunks = select_recall_chunks(
                text_result.get("returned_chunks") or []
            )
            recalled_frame_paths: List[str] = []
            frame_chunks: List[int] = []
            if selected_chunks and self.frames_root and video_path:
                for ci in selected_chunks:
                    cf = _chunk_frame_paths(
                        video_path, self.frames_root, ci,
                        self.frames_per_chunk,
                    )
                    if cf:
                        frame_chunks.append(ci)
                        recalled_frame_paths.extend(cf)

            time_text = recall_time_string_for_chunks(selected_chunks) or (
                text_result.get("time", "") or ""
            )
            success = bool(selected_chunks) or bool(text_result.get("text"))
            raw_recall_result = {
                "source": "historical_frames" if recalled_frame_paths else (
                    text_result.get("source", "memory") if success else "failure"
                ),
                "text_content": text_result.get("text", "")
                                if success else "No matching results found.",
                "text": text_result.get("text", "")
                        if success else "No matching results found.",
                "returned_chunks": selected_chunks,
                "time": time_text,
            }
            recalled_frames = build_recalled_frames_metadata(
                frame_chunks,
                recalled_frame_paths,
                chunk_sec=self.chunk_sec,
                frames_per_chunk=self.frames_per_chunk,
            ) if recalled_frame_paths else None
            recall_result = build_recall_result_metadata(
                raw_recall_result,
                recalled_frames,
            )
            return {
                "recall_result": recall_result,
                "recalled_frames": recalled_frames,
            }

        def _build_recall_tool_message(
            self, recall_payload: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Wrap _execute_recall's return into a chat message.

            Mirrors pass5_messages.py:280-380 ordering EXACTLY (the v12.11
            audit P0 fix order is the SFT contract):
              1. <recalled_frames>{json header}</recalled_frames> text
              2. protocol visual frames anchored to historical chunk timestamps
              3. <recall_result>{json}</recall_result> metadata

            Use role="tool" to match pass5/SFT. Qwen's chat template renders
            this as a user-wrapped <tool_response>...</tool_response> block,
            which keeps the post-recall prompt identical between SFT and
            rollout/eval.
            """
            rr = recall_payload.get("recall_result") or {}
            rf = recall_payload.get("recalled_frames")  # may be None
            _RTKW = _runtime_mm_processor_kwargs()
            content = build_recall_result_user_content(
                rf,
                rr,
                frame_protocol=self.frame_protocol,
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
                render_layout=self.render_layout,
            )
            return {"role": "tool", "tool_call_id": "recall", "content": content}

        def _check_compress_trigger(
            self,
            state: "VideoTrajectoryState",
            *,
            force_boundary: bool = False,
        ) -> Optional[Tuple[int, int]]:
            """Return raw new-caption span when compact memory should update."""
            if not state.recent_thinks:
                return None
            if force_boundary:
                chunks = [
                    int(t.get("chunk", -1))
                    for t in state.recent_thinks
                    if isinstance(t, dict) and t.get("chunk", -1) >= 0
                ]
                if not chunks:
                    return None
                return (min(chunks), max(chunks))
            if len(state.recent_thinks) < COMPRESS_RANGE_MIN:
                return None
            est = _count_visible_memory_tokens(
                state.compressed_summaries,
                state.recent_thinks, tokenizer=self.tokenizer,
            )
            if est < self.compress_token_threshold and len(state.recent_thinks) < COMPRESS_RANGE_MAX:
                return None
            chunks = [
                int(t.get("chunk", -1))
                for t in state.recent_thinks
                if isinstance(t, dict) and t.get("chunk", -1) >= 0
            ]
            if not chunks:
                return None
            return (min(chunks), max(chunks))

        @staticmethod
        def _as_plain_list(value: Any) -> List[Any]:
            if hasattr(value, "tolist"):
                value = value.tolist()
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            return []

        def _apply_initial_student_state(
            self,
            state: "VideoTrajectoryState",
            snapshot: Any,
        ) -> None:
            """Seed a segment rollout from a student-generated memory state.

            This is intentionally a pure state restore. It does not synthesize
            teacher memory. If no snapshot is provided, the segment starts with
            empty memory and should therefore normally start at chunk 0 or at
            the question ask chunk only for no-history ablations.
            """
            if hasattr(snapshot, "tolist"):
                snapshot = snapshot.tolist()
            if not isinstance(snapshot, dict):
                return

            compressed = (
                snapshot.get("compressed_segments")
                or snapshot.get("compressed")
                or snapshot.get("compressed_summaries")
                or []
            )
            recent = snapshot.get("recent_thinks") or []
            archive = (
                snapshot.get("think_archive")
                or snapshot.get("retrieval_archive")
                or snapshot.get("evidence_bank")
                or []
            )
            state.compressed_summaries = [
                dict(x) for x in self._as_plain_list(compressed)
                if isinstance(x, dict)
            ]
            state.recent_thinks = [
                dict(x) for x in self._as_plain_list(recent)
                if isinstance(x, dict)
            ]
            # If the cache does not carry a full archive, recent_thinks is the
            # best student-state fallback for recall over the restored segment.
            archive_list = self._as_plain_list(archive) or list(state.recent_thinks)
            state.think_archive = [
                dict(x) for x in archive_list if isinstance(x, dict)
            ]

        async def _generate_action_tokens(
            self,
            *,
            request_id: str,
            prompt_ids: List[int],
            new_prompt_ids: List[int],
            sampling_params: Dict[str, Any],
            image_data: Optional[List[Any]],
            video_data: Optional[List[Any]],
            turn_kind: str = "",
            chunk_idx: Optional[int] = None,
            stream_reset_before: bool = False,
            stream_isolated_turn: bool = False,
        ):
            """Generate one assistant action through the configured rollout backend.

            The default ``THINKSTREAM_ROLLOUT_ENGINE=streaming`` path is an
            adapter hook for CASIA-style local HF rollout: it receives the full
            prompt for logging/recovery plus ``new_prompt_ids`` for
            incremental re-prefill against an existing KV stream. The manager
            is expected to return an object with ``token_ids``, ``log_probs``
            and ``stop_reason`` attributes, or a dict/list carrying those
            fields. The legacy full-prompt vLLM path is disabled because it
            repeatedly appends old text under true KV and breaks recall
            deletion semantics.
            """
            generate_streaming = (
                getattr(self.server_manager, "generate_streaming", None)
                or getattr(self.server_manager, "generate_streaming_turn", None)
                or getattr(self.server_manager, "generate_turn", None)
            )
            if generate_streaming is None:
                raise RuntimeError(
                    "ThinkStream RL now requires a true-KV streaming rollout "
                    "backend exposing generate_streaming_turn/generate_turn; "
                    "full-prompt vLLM rollout is disabled for correctness."
                )
            out = generate_streaming(
                request_id=request_id,
                prompt_ids=prompt_ids,
                new_prompt_ids=new_prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                turn_kind=turn_kind,
                chunk_idx=chunk_idx,
                stream_reset_before=stream_reset_before,
                stream_isolated_turn=stream_isolated_turn,
            )
            if hasattr(out, "__await__"):
                out = await out
            if isinstance(out, list):
                if not out:
                    raise RuntimeError("streaming rollout backend returned no outputs")
                out = out[0]
            return out

        async def run(
            self, sampling_params: dict[str, Any], **kwargs,
        ) -> Union["AgentLoopOutput", List["AgentLoopOutput"]]:
            metrics: Dict[str, Any] = {}
            request_id = uuid4().hex

            extra_info = kwargs.get("extra_info") or {}
            video_id = extra_info.get("video_id") or extra_info.get("index", "")
            video_path = extra_info.get("video_path", "")
            n_chunks_dataset = int(extra_info.get("n_chunks") or 0)
            n_chunks = min(self.max_chunks, n_chunks_dataset) if n_chunks_dataset else self.max_chunks
            segment_start_chunk = int(extra_info.get("segment_start_chunk") or 0)
            segment_end_raw = extra_info.get("segment_end_chunk")
            segment_end_chunk: Optional[int] = None
            if segment_end_raw is not None:
                try:
                    segment_end_chunk = int(segment_end_raw)
                except (TypeError, ValueError):
                    segment_end_chunk = None
            question = extra_info.get("question", "")
            ask_chunks = list(extra_info.get("ask_chunks") or [])

            # Multi-Q trajectory mode (build_verl_parquet --multi_q): one
            # parquet row carries `questions` (List[Dict]). Each entry
            # has its own ask_chunk; the rollout fires the corresponding
            # question's text into <user_input> at exactly that chunk.
            # Per-Q answers + answered chunks are surfaced via
            # extra_fields so compute_score can score each Q independently.
            multi_q_list_raw = extra_info.get("questions") or []
            # Normalise: parquet round-trip can wrap dicts in numpy structs
            # whose iteration yields plain dicts but `isinstance dict` may
            # be False. Coerce defensively.
            multi_q_list: List[Dict[str, Any]] = []
            for q in multi_q_list_raw:
                if hasattr(q, "tolist"):
                    q = q.tolist()
                if not isinstance(q, dict):
                    continue
                multi_q_list.append({k: q[k] for k in q.keys()})

            # Build the per-chunk ask map: chunk_idx → list of question
            # indices that should fire at that chunk. Pre-compute once
            # rather than scanning N questions on every chunk iteration.
            ask_at_chunk: Dict[int, List[int]] = {}
            if multi_q_list:
                for q_idx, q in enumerate(multi_q_list):
                    aks = q.get("ask_chunks") or (
                        [int(q["ask_chunk"])] if int(q.get("ask_chunk", -1)) >= 0 else []
                    )
                    for ck in aks:
                        try:
                            ck_int = int(ck)
                        except (TypeError, ValueError):
                            continue
                        ask_at_chunk.setdefault(ck_int, []).append(q_idx)
                # Ensure rollout reaches at least the latest ask_chunk.
                if ask_at_chunk:
                    n_chunks = max(n_chunks, max(ask_at_chunk.keys()) + 1)
                    n_chunks = min(self.max_chunks, n_chunks)

            if segment_end_chunk is not None:
                segment_end_chunk = max(segment_start_chunk, segment_end_chunk)
                n_chunks = min(n_chunks, segment_end_chunk + 1)
            segment_start_chunk = max(0, min(segment_start_chunk, max(0, n_chunks - 1)))
            offline_compress_chunks: List[int] = []
            for raw in self._as_plain_list(extra_info.get("offline_compress_chunks")):
                try:
                    ci = int(raw)
                except (TypeError, ValueError):
                    continue
                if segment_start_chunk <= ci < n_chunks:
                    offline_compress_chunks.append(ci)
            offline_compress_pending = set(sorted(set(offline_compress_chunks)))
            use_offline_compress = (
                self.compress_trigger_source == "offline_pass2_boundaries"
            )

            # Per-Q answer tracking (multi-Q mode only). Indexed by q_idx;
            # captured when the corresponding chunk's assistant turn parses
            # to kind=="answer" (or any explicit <answer>...</answer>).
            per_q_answer_chunk: List[int] = [-1] * len(multi_q_list)
            per_q_answer_text: List[str] = [""] * len(multi_q_list)
            per_q_answers: List[List[Dict[str, Any]]] = [
                [] for _ in multi_q_list
            ]
            # Stack of question indices waiting to be assigned to the next
            # assistant turn (for chunks where multiple Qs fire — Q1's
            # spec says this won't happen now, but keep the queue for
            # robustness).
            pending_q_indices: List[int] = []
            query_log: List[Dict[str, Any]] = []
            query_log_idx_by_q: Dict[int, int] = {}

            def _question_complete(q_idx: int) -> bool:
                """Return True when this pending question has enough answers.

                Empty <answer></answer> is a per-chunk silent action, not a
                terminal event. Early answers are logged but do not satisfy an
                expected answer chunk; multi-emit cards need one countable
                answer per expected answer chunk.
                """
                if q_idx < 0 or q_idx >= len(multi_q_list):
                    return True
                qlog_i = query_log_idx_by_q.get(q_idx)
                if qlog_i is not None and 0 <= qlog_i < len(query_log):
                    return query_is_complete(query_log[qlog_i])
                q_obj = multi_q_list[q_idx]
                ans_ch = q_obj.get("answer_chunks") or []
                if hasattr(ans_ch, "tolist"):
                    ans_ch = ans_ch.tolist()
                expected: List[int] = []
                for x in ans_ch:
                    try:
                        expected.append(int(x))
                    except (TypeError, ValueError):
                        continue
                expected_n = max(1, len(expected))
                return len(per_q_answers[q_idx]) >= expected_n

            # ── Initial prompt: [system + user(question)]. Prefilled once
            # per true-KV stream; ordinary chunks append only their delta.
            initial_messages = list(kwargs["raw_prompt"])
            initial_mm = await self.process_vision_info(initial_messages)
            initial_images: List[Any] = list(initial_mm.get("images") or [])
            initial_videos: List[Any] = list(initial_mm.get("videos") or [])

            # P0.3 fix (post-review 2026-05-01): AgentLoopBase.apply_chat_template
            # hardcodes add_generation_prompt=True, so calling it on
            # [system, user_q] gives [sys, user_q, asst_prefix]. Then chunk_prompt_ids
            # = apply_chat_template([system, user_q, chunk_user]) gives
            # [sys, user_q, chunk_user, asst_prefix]. Slicing by len(initial_prompt_ids)
            # would chop the first len(asst_prefix) tokens of chunk_user.
            # Fix: tokenize initial WITHOUT the assistant prefix via the
            # raw tokenizer/processor so the slice prefix is exact.
            try:
                initial_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        initial_messages,
                        tools=tools_for_turn("streaming"),
                        add_generation_prompt=False,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                if hasattr(initial_prompt_ids, "tolist"):
                    initial_prompt_ids = initial_prompt_ids.tolist()
                initial_prompt_ids = list(initial_prompt_ids)
            except Exception:
                # Fallback: use the agent-loop helper (with asst prefix) and
                # recover the no-prefix length by tokenizing the prefix
                # marker itself.
                with_prefix = await self.apply_chat_template(
                    initial_messages, tools=tools_for_turn("streaming"),
                    images=initial_images if initial_images else None,
                    videos=initial_videos if initial_videos else None,
                )
                initial_prompt_ids = list(with_prefix)

            response_ids: List[int] = []
            response_mask: List[int] = []
            response_logprobs: List[float] = []
            any_logprobs_returned = False
            chunk_asst_spans: List[Tuple[int, int]] = []
            chunk_kinds: List[str] = []
            chunk_turn_kinds: List[str] = []
            chunk_action_space_errors: List[str] = []
            chunk_asst_texts: List[str] = []
            chunk_prompt_lens: List[int] = []
            chunk_response_lens: List[int] = []
            chunk_max_tokens: List[int] = []
            chunk_hit_max_tokens: List[bool] = []
            chunk_stop_reasons: List[str] = []
            chunk_recall_query_ranges: List[Any] = []
            chunk_recall_returned_chunks: List[List[int]] = []
            chunk_recall_result_sources: List[str] = []
            chunk_compress_expected_chunks: List[List[int]] = []
            chunk_compress_emitted_ranges: List[Any] = []
            budget_abort_events: List[Dict[str, Any]] = []
            # P1.7 fix (post-review 2026-05-01): chunk_kinds/spans/texts
            # are appended on EVERY assistant turn including inter-chunk
            # compress turns. Without a parallel video-chunk-index list,
            # compute_score's per-chunk action gold lookup
            # (`enumerate(chunk_kinds)` → chunk_idx) would right-shift after
            # each compress turn. Track the actual video chunk_idx per
            # turn here; compute_score uses this for `gold_action_per_chunk[
            # str(video_chunk_idx)]` lookup. -1 marks compress (system event).
            chunk_video_indices: List[int] = []
            # Actual video cursor at the time of each assistant turn. For
            # inter-chunk compression this is the chunk that will be retried
            # after compression succeeds. Reward/audit code can correlate the
            # system event with runtime state, but must not treat offline
            # gold_action_per_chunk["compress"] labels as the trigger.
            chunk_event_indices: List[int] = []

            # multi_modal_data accumulator: one (tensor, metadata) per
            # Accumulators for multi-modal payloads. `ts_image` contributes
            # images; `video_meta` contributes videos with explicit metadata.
            accumulated_images: List[Any] = list(initial_images)
            accumulated_videos: List[Any] = list(initial_videos)

            # v12.14 Phase 3: per-action tracking for recurrent mode.
            # Stitched mode reads from response_ids/response_mask above and
            # ignores these. Recurrent mode emits one AgentLoopOutput per
            # entry of these arrays at the end of run().
            per_action_prompt_ids: List[List[int]] = []
            per_action_response_ids: List[List[int]] = []
            per_action_response_mask: List[List[int]] = []
            per_action_response_logprobs: List[Optional[List[float]]] = []
            per_action_mm_data: List[Optional[Dict[str, Any]]] = []

            state = VideoTrajectoryState(
                video_uid=str(video_id),
                chunk_idx=segment_start_chunk,
            )
            self._apply_initial_student_state(
                state,
                extra_info.get("initial_student_state")
                or extra_info.get("initial_memory_snapshot")
                or extra_info.get("initial_state"),
            )
            recall_result_for_next: Optional[Dict[str, Any]] = None
            num_assistant_turns = 0
            n_chunks_with_frames = 0
            n_chunks_text_only = 0
            n_chunks_compress_inter = 0
            stream_reset_before_next_turn = False

            chunk_idx = segment_start_chunk
            while chunk_idx < n_chunks:
                if not state.is_active:
                    break
                state.chunk_idx = chunk_idx

                # ── Decide turn type: compress trigger fires BETWEEN
                # chunks. Compression turns are text-only memory management:
                # compression-only system prompt + memory-only user payload.
                # Offline pass2 boundaries are recorded on the first visual
                # chunk after the compacted raw range. Run the update before
                # that chunk's visual turn.
                offline_ready = (
                    use_offline_compress
                    and chunk_idx in offline_compress_pending
                )
                if offline_ready:
                    compress_range = self._check_compress_trigger(
                        state,
                        force_boundary=True,
                    )
                    # Consume the planned boundary once. A failed compression
                    # should be scored as a bad action, not retried forever
                    # before the first visual turn after the boundary.
                    offline_compress_pending.discard(chunk_idx)
                elif use_offline_compress:
                    compress_range = None
                else:
                    compress_range = self._check_compress_trigger(state)
                inter_chunk = compress_range is not None

                # ── Sliding visual window.
                window_paths: List[str] = []
                window_start_chunk = chunk_idx
                window_end_chunk = chunk_idx
                if self.frames_root and video_path:
                    window_paths, window_start_chunk, window_end_chunk = (
                        _build_visual_window(
                            video_path, self.frames_root, chunk_idx,
                            visual_window_chunks=self.visual_window_chunks,
                            frames_per_chunk=self.frames_per_chunk,
                            chunk_sec=self.chunk_sec,
                            mode=self.visual_window_mode,
                        )
                    )
                visual_injected = bool(window_paths) and not inter_chunk

                # ── Multi-Q: which questions fire at this chunk?
                triggered_qs_for_chunk: List[Dict[str, Any]] = []
                triggered_q_indices_for_chunk: List[int] = []
                if multi_q_list and not inter_chunk:
                    new_q_indices = [
                        int(q_idx)
                        for q_idx in ask_at_chunk.get(chunk_idx, [])
                        if int(q_idx) not in query_log_idx_by_q
                    ]
                    if new_q_indices:
                        # Do not close an older question just because a new
                        # question fires. Completion is defined by the
                        # question's expected answer chunks and countable
                        # answer events. This matters across compress
                        # boundaries and off-policy RL: an unfinished
                        # question may still be answered late, or remain
                        # unanswered and receive the no-answer timing penalty.
                        pending_q_indices = [
                            qi for qi in pending_q_indices
                            if not _question_complete(qi)
                        ]
                    for q_idx in ask_at_chunk.get(chunk_idx, []):
                        q_obj = multi_q_list[q_idx]
                        triggered_qs_for_chunk.append(q_obj)
                        triggered_q_indices_for_chunk.append(q_idx)
                        if q_idx not in query_log_idx_by_q:
                            query_log_idx_by_q[q_idx] = len(query_log)
                            ans_chunks_raw = q_obj.get("answer_chunks") or []
                            if hasattr(ans_chunks_raw, "tolist"):
                                ans_chunks_raw = ans_chunks_raw.tolist()
                            ans_chunks_int: List[int] = []
                            for x in ans_chunks_raw:
                                try:
                                    ans_chunks_int.append(int(x))
                                except (TypeError, ValueError):
                                    continue
                            query_log.append({
                                "question": q_obj.get("question", ""),
                                "options": list(q_obj.get("options") or []),
                                "answer_form": q_obj.get("answer_form", ""),
                                "answer_style": (
                                    "letter_only"
                                    if q_obj.get("answer_form") == "multiple_choice"
                                    else q_obj.get("answer_style", "")
                                ),
                                "answer_instruction": q_obj.get("answer_instruction", ""),
                                "answer_chunks": ans_chunks_int,
                                "per_emit_answers": list(q_obj.get("per_emit_answers") or []),
                                "ask_time": chunk_idx * self.chunk_sec,
                                "open_until": (
                                    max(ans_chunks_int) * self.chunk_sec
                                    if ans_chunks_int else chunk_idx * self.chunk_sec
                                ),
                                "status": "open",
                                "answers": [],
                            })
                    # Push triggered Qs into the pending queue so the
                    # NEXT assistant turn's <answer> is assigned to them.
                    for qi in triggered_q_indices_for_chunk:
                        if qi not in pending_q_indices:
                            pending_q_indices.append(qi)

                # ── Build user content + chunk_messages.
                user_content = self._build_chunk_user_content(
                    state=state,
                    chunk_idx=chunk_idx,
                    window_paths=window_paths,
                    window_start_chunk=window_start_chunk,
                    window_end_chunk=window_end_chunk,
                    question=question,
                    ask_chunks=ask_chunks,
                    triggered_questions=(
                        triggered_qs_for_chunk if multi_q_list else None
                    ),
                    queries=query_log,
                    recall_result=recall_result_for_next,
                    compress_trigger_range=compress_range,
                    inter_chunk=inter_chunk,
                )
                recall_result_for_next = None

                if inter_chunk:
                    # Compression is a turn-local system event, not a
                    # continuation of the ordinary streaming prompt. Match
                    # SFT/runtime exactly: one compression-only system prompt
                    # plus memory-only user payload. Do not append a second
                    # system prompt after the initial streaming raw_prompt.
                    chunk_messages = [{
                        "role": "system",
                        "content": system_prompt_for_frame_protocol(
                            self.frame_protocol,
                            inter_chunk=True,
                            render_layout=self.render_layout,
                        ),
                    }]
                else:
                    chunk_messages = list(initial_messages)
                chunk_messages.append({"role": "user", "content": user_content})
                # ── chunk-internal ready-loop (v12.13 D1):
                #
                # When `max_recall_per_chunk == 0` (legacy default), the
                # loop runs once and recall_result_for_next is delivered
                # to the NEXT chunk's user payload (D1 deferred behaviour).
                #
                # When `max_recall_per_chunk >= 1`, on a recall turn we
                # build the tool response (with historical frames + MROPE
                # anchored at recall time) and append it to chunk_messages,
                # then re-generate WITHIN THE SAME CHUNK. Mirrors SFT
                # pass5 shape B exactly: assistant→tool→assistant.
                # Repeats up to `max_recall_per_chunk` times before
                # falling through to "treat the next response as final".
                response_text = ""
                kind = "unknown"
                parsed: Dict[str, Any] = {}
                recall_rounds_this_chunk = 0
                last_prompt_len = len(initial_prompt_ids)
                inner_aborted = False
                while True:
                    turn_kind = (
                        "post_recall"
                        if recall_rounds_this_chunk > 0
                        else ("compress" if inter_chunk else "streaming")
                    )
                    # Resolve media only for the KV delta of this turn. On the
                    # post-recall round the ordinary current chunk and the
                    # assistant tool_call are already in KV; only the tool
                    # response's recalled frames are new.
                    if turn_kind == "post_recall":
                        chunk_mm_messages = (
                            [chunk_messages[-1]]
                            if chunk_messages and chunk_messages[-1].get("role") == "tool"
                            else []
                        )
                        template_mm_messages = (
                            chunk_messages[len(initial_messages):]
                            if not inter_chunk
                            else chunk_messages
                        )
                    else:
                        chunk_mm_messages = (
                            chunk_messages
                            if inter_chunk
                            else chunk_messages[len(initial_messages):]
                        )
                        template_mm_messages = chunk_mm_messages
                    chunk_extra_mm = await self.process_vision_info(
                        chunk_mm_messages,
                    )
                    chunk_images = chunk_extra_mm.get("images") or []
                    chunk_videos = chunk_extra_mm.get("videos") or []
                    if template_mm_messages is chunk_mm_messages:
                        template_images = chunk_images
                        template_videos = chunk_videos
                    else:
                        template_extra_mm = await self.process_vision_info(
                            template_mm_messages,
                        )
                        template_images = template_extra_mm.get("images") or []
                        template_videos = template_extra_mm.get("videos") or []

                    turn_tools = tools_for_turn(turn_kind)
                    template_messages = chunk_messages
                    prompt_images = (initial_images + template_images) if (
                        turn_kind != "post_recall" and template_images
                    ) else (
                        template_images if template_images else (
                            initial_images if (
                                turn_kind != "post_recall" and initial_images
                            ) else None
                        )
                    )
                    prompt_videos = (initial_videos + template_videos) if (
                        turn_kind != "post_recall" and template_videos
                    ) else (
                        template_videos if template_videos else (
                            initial_videos if (
                                turn_kind != "post_recall" and initial_videos
                            ) else None
                        )
                    )
                    chunk_prompt_ids = await self.apply_chat_template(
                        template_messages,
                        tools=turn_tools,
                        images=prompt_images,
                        videos=prompt_videos,
                    )

                    # Prompt-budget guard (single-chunk + tool round can
                    # exceed budget after frames are appended; skip out
                    # if so — outer loop ends rollout). Use the per-action
                    # generation cap here, not the stitched response buffer.
                    remaining_context = self.max_model_len - len(chunk_prompt_ids)
                    if remaining_context <= 0:
                        budget_abort_events.append({
                            "chunk": int(chunk_idx),
                            "turn_kind": turn_kind,
                            "reason": "prompt_exceeds_max_model_len",
                            "prompt_len": int(len(chunk_prompt_ids)),
                            "max_model_len": int(self.max_model_len),
                            "remaining_context": int(remaining_context),
                        })
                        inner_aborted = True
                        break
                    effective_last_prompt_len = (
                        0
                        if (
                            turn_kind == "compress"
                            or stream_reset_before_next_turn
                        )
                        else last_prompt_len
                    )
                    user_block_len = len(chunk_prompt_ids) - effective_last_prompt_len
                    if user_block_len < 0:
                        budget_abort_events.append({
                            "chunk": int(chunk_idx),
                            "turn_kind": turn_kind,
                            "reason": "negative_user_block_len",
                            "prompt_len": int(len(chunk_prompt_ids)),
                            "last_prompt_len": int(effective_last_prompt_len),
                        })
                        inner_aborted = True
                        break
                    if self.recurrent_mode == "recurrent":
                        # In recurrent mode `response_length` is the dense
                        # per-action tensor width used by verl after this
                        # loop emits one AgentLoopOutput per assistant turn.
                        # Do not spend it as a stitched trajectory-wide
                        # budget, or long videos stop after only a few
                        # chunks once accumulated user/context blocks fill
                        # the old stitched buffer.
                        remaining_response = self.response_length
                    else:
                        remaining_response = (
                            self.response_length
                            - len(response_mask)
                            - user_block_len
                            - 1
                        )
                    turn_max_tokens = (
                        self.max_tokens_per_compress_action
                        if turn_kind == "compress"
                        else self.max_tokens_per_action
                    )
                    max_tokens_this_turn = min(
                        turn_max_tokens,
                        remaining_context,
                        remaining_response,
                    )
                    if max_tokens_this_turn <= 0:
                        budget_abort_events.append({
                            "chunk": int(chunk_idx),
                            "turn_kind": turn_kind,
                            "reason": "no_response_budget",
                            "prompt_len": int(len(chunk_prompt_ids)),
                            "remaining_context": int(remaining_context),
                            "remaining_response": int(remaining_response),
                        })
                        inner_aborted = True
                        break

                    # ── Generate.
                    #
                    # True-KV rollout appends `new_prompt_ids` to the active
                    # stream and returns per-token log_probs for PPO. Compress
                    # turns are isolated text-only system events; recall
                    # post-turns use the next-turn sidecar KV policy.
                    with simple_timer("generate_sequences", metrics):
                        output = await self._generate_action_tokens(
                            request_id=request_id,
                            prompt_ids=chunk_prompt_ids,
                            new_prompt_ids=chunk_prompt_ids[effective_last_prompt_len:],
                            sampling_params={
                                **sampling_params,
                                "max_tokens": max_tokens_this_turn,
                                "recall_kv_policy": "next_turn",
                                "delete_previous_recall_toolcall_kv": (
                                    turn_kind == "post_recall"
                                ),
                            },
                            image_data=(
                                chunk_images if turn_kind == "post_recall"
                                else (
                                    (initial_images + chunk_images)
                                    if chunk_images
                                    else (initial_images if initial_images else None)
                                )
                            ),
                            video_data=(
                                chunk_videos if turn_kind == "post_recall"
                                else (
                                    (initial_videos + chunk_videos)
                                    if chunk_videos
                                    else (initial_videos if initial_videos else None)
                                )
                            ),
                            turn_kind=turn_kind,
                            chunk_idx=chunk_idx,
                            stream_reset_before=stream_reset_before_next_turn,
                            stream_isolated_turn=inter_chunk,
                        )
                    stream_reset_before_next_turn = False
                    if isinstance(output, dict):
                        assistant_ids = list(output.get("token_ids") or [])
                        output_log_probs = output.get("log_probs")
                        output_stop_reason = output.get("stop_reason", "")
                    else:
                        assistant_ids = list(getattr(output, "token_ids", []) or [])
                        output_log_probs = getattr(output, "log_probs", None)
                        output_stop_reason = getattr(output, "stop_reason", "")
                    output_log_probs_list = None
                    if output_log_probs is not None:
                        try:
                            if hasattr(output_log_probs, "tolist"):
                                output_log_probs_list = list(output_log_probs.tolist())
                            else:
                                output_log_probs_list = list(output_log_probs)
                        except TypeError:
                            output_log_probs_list = None
                    if not assistant_ids:
                        budget_abort_events.append({
                            "chunk": int(chunk_idx),
                            "turn_kind": turn_kind,
                            "reason": "empty_generation",
                            "prompt_len": int(len(chunk_prompt_ids)),
                            "max_tokens": int(max_tokens_this_turn),
                        })
                        inner_aborted = True
                        break

                    # ── Stitch (incremental relative to last_prompt_len).
                    user_block_ids = chunk_prompt_ids[effective_last_prompt_len:]
                    response_ids.extend(user_block_ids)
                    response_mask.extend([0] * len(user_block_ids))
                    response_logprobs.extend([0.0] * len(user_block_ids))

                    asst_start = len(response_ids)
                    response_ids.extend(assistant_ids)
                    response_mask.extend([1] * len(assistant_ids))

                    # v12.14 Phase 3: also record per-action standalone
                    # (prompt, response) so recurrent mode can emit one
                    # AgentLoopOutput per action. Each action's prompt is
                    # the FULL prompt at this round (chunk_prompt_ids,
                    # which already includes prior recall multi-turn
                    # context for round 2+); its response is just this
                    # round's assistant tokens.
                    per_action_prompt_ids.append(list(chunk_prompt_ids))
                    per_action_response_ids.append(list(assistant_ids))
                    per_action_response_mask.append([1] * len(assistant_ids))
                    if (
                        output_log_probs_list is not None
                        and len(output_log_probs_list) == len(assistant_ids)
                    ):
                        per_action_response_logprobs.append(list(output_log_probs_list))
                    else:
                        per_action_response_logprobs.append(None)
                    # mm payload for actor logprob/replay must match the FULL
                    # per-action prompt, not just the KV delta used by
                    # generation. In post-recall turns that prompt contains the
                    # current visual chunk plus recalled frames, while the
                    # true-KV generation call still receives only the new tool
                    # frames.
                    _ac_mm = None
                    if prompt_videos:
                        _ac_mm = {"videos": list(prompt_videos)}
                    if prompt_images:
                        _ac_mm = _ac_mm or {}
                        _ac_mm["images"] = list(prompt_images)
                    per_action_mm_data.append(_ac_mm)
                    if (
                        output_log_probs_list is not None
                        and len(output_log_probs_list) == len(assistant_ids)
                    ):
                        response_logprobs.extend(list(output_log_probs_list))
                        any_logprobs_returned = True
                    else:
                        response_logprobs.extend([0.0] * len(assistant_ids))
                    asst_end = len(response_ids)
                    chunk_asst_spans.append((asst_start, asst_end))
                    # -1 for compress (system inter-chunk turn).
                    chunk_video_indices.append(-1 if inter_chunk else chunk_idx)
                    chunk_event_indices.append(chunk_idx)
                    chunk_prompt_lens.append(int(len(chunk_prompt_ids)))
                    chunk_response_lens.append(int(len(assistant_ids)))
                    chunk_max_tokens.append(int(max_tokens_this_turn))
                    chunk_hit_max_tokens.append(
                        len(assistant_ids) >= int(max_tokens_this_turn)
                    )
                    chunk_stop_reasons.append(str(output_stop_reason or ""))

                    if visual_injected:
                        accumulated_images.extend(chunk_images)
                        accumulated_videos.extend(chunk_videos)
                        n_chunks_with_frames += 1
                    elif not inter_chunk:
                        n_chunks_text_only += 1
                    if inter_chunk:
                        n_chunks_compress_inter += 1
                    visual_injected = False  # only count once per chunk

                    num_assistant_turns += 1

                    # ── Decode + parse this turn.
                    response_text = decode_agent_output_tokens(
                        self.tokenizer,
                        assistant_ids,
                    )
                    parsed = parse_agent_output(
                        response_text,
                        allow_bare_answer=(turn_kind == "post_recall"),
                        allow_malformed_tool_call=(turn_kind == "compress"),
                    )
                    kind = parsed.get("kind", "unknown")
                    action_error = action_space_error_for_turn(kind, turn_kind)
                    if action_error:
                        parsed["action_space_error"] = action_error
                        parsed["invalid_kind"] = kind
                        kind = "invalid"
                    tool_args = (
                        (parsed.get("tool_call") or {}).get("arguments") or {}
                    )
                    chunk_kinds.append(kind)
                    chunk_turn_kinds.append(turn_kind)
                    chunk_action_space_errors.append(action_error)
                    chunk_asst_texts.append(response_text)
                    if kind == "recall":
                        chunk_recall_query_ranges.append(
                            tool_args.get("time_range", "")
                        )
                    else:
                        chunk_recall_query_ranges.append("")
                    chunk_recall_returned_chunks.append([])
                    chunk_recall_result_sources.append("")
                    if inter_chunk and compress_range is not None:
                        chunk_compress_expected_chunks.append(
                            list(range(int(compress_range[0]), int(compress_range[1]) + 1))
                        )
                    else:
                        chunk_compress_expected_chunks.append([])
                    if kind == "compress":
                        mem_entries = tool_args.get("memory_text")
                        chunk_compress_emitted_ranges.append(
                            "MEM" if mem_entries else tool_args.get("time_range")
                        )
                    else:
                        chunk_compress_emitted_ranges.append(None)

                    # ── Decide: stay in chunk for shape-B recall multi-
                    # turn, or break out and advance chunk_idx.
                    if (
                        kind == "recall"
                        and not inter_chunk
                        and recall_rounds_this_chunk < self.max_recall_per_chunk
                    ):
                        args = tool_args
                        recall_payload = await self._execute_recall(
                            args, state, video_path=video_path,
                        )
                        recall_result = recall_payload.get("recall_result") or {}
                        chunk_recall_returned_chunks[-1] = [
                            int(x) for x in (
                                recall_result.get("returned_chunks") or []
                            )
                            if isinstance(x, (int, float))
                        ]
                        chunk_recall_result_sources[-1] = str(
                            recall_result.get("source") or ""
                        )
                        # Bookkeep on state (accounting only — no truncation).
                        try:
                            state.n_recall_calls = int(getattr(state, "n_recall_calls", 0)) + 1
                        except Exception:
                            pass

                        # Append turn1 assistant text + tool message to
                        # chunk_messages so the NEXT round of apply_chat_template
                        # rebuilds the prompt with the tool response in place.
                        # (We can't reuse `assistant_ids` directly — the chat
                        # template adds <|im_start|>assistant prefix/suffix
                        # tokens we'd duplicate.)
                        chunk_messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": response_text}],
                        })
                        chunk_messages.append(self._build_recall_tool_message(recall_payload))

                        last_prompt_len = (
                            len(chunk_prompt_ids) + len(assistant_ids)
                        )
                        recall_rounds_this_chunk += 1
                        # Continue inner loop — generate turn 2 (the answer).
                        continue

                    # Non-recall turn (answer / silent / unknown / compress)
                    # OR exceeded recall budget → exit inner, advance chunk.
                    if kind == "recall" and recall_rounds_this_chunk >= self.max_recall_per_chunk:
                        # Exhausted recall budget within this chunk —
                        # legacy fallback: deliver result on next chunk.
                        args = tool_args
                        recall_payload = await self._execute_recall(
                            args, state, video_path=video_path,
                        )
                        recall_result_for_next = recall_payload.get("recall_result")
                        recall_result = recall_result_for_next or {}
                        chunk_recall_returned_chunks[-1] = [
                            int(x) for x in (
                                recall_result.get("returned_chunks") or []
                            )
                            if isinstance(x, (int, float))
                        ]
                        chunk_recall_result_sources[-1] = str(
                            recall_result.get("source") or ""
                        )
                    break

                if inner_aborted and num_assistant_turns == 0:
                    break

                # ── Multi-Q: assign this turn's NON-EMPTY <answer> to a
                # pending question.
                #
                # IMPORTANT — what pass3 SFT data actually looks like:
                # design.py:gold_action_at is a pure function of
                # (chunk_idx, ask_chunk, gold_emits, question_type) that
                # marks AT MOST ONE Q as `response` per chunk. So along a
                # SFT-aligned rollout, pure FIFO IS correct: when the model
                # emits <answer> at chunk N, exactly one pending Q is in
                # its `response_chunk == N` slot, and any earlier pending
                # Q has already been popped at its own response_chunk.
                #
                # The 3-stage rule below is a DEFENSIVE upgrade for RL
                # exploration where the model may answer at the wrong
                # chunk (off-policy from SFT distribution). In SFT-aligned
                # behavior all 3 stages converge on the same pending Q;
                # in off-policy behavior the answer_chunks-window match
                # rescues attribution that pure FIFO would silently
                # mis-route.
                #
                # Stages (deterministic):
                #   1. On-time/window match: pending Q whose unanswered
                #      answer chunk(s) bracket the current chunk_idx.
                #   2. Late match: closest unmatched expected chunk behind
                #      the current chunk.
                #   3. Early match: closest unmatched expected chunk ahead
                #      of the current chunk.
                #   4. LIFO fallback for malformed/no-answer-chunk metadata.
                if multi_q_list and pending_q_indices:
                    answer_str = (
                        parsed.get("answer_text") if kind == "answer" else None
                    )
                    if answer_str is not None:
                        answer_str = str(answer_str).strip()
                    if answer_str:

                        chosen_pos = None

                        def _unmatched_expected_chunks_for_q(qi: int) -> List[int]:
                            qlog_i = query_log_idx_by_q.get(qi)
                            if qlog_i is not None and 0 <= qlog_i < len(query_log):
                                qlog = query_log[qlog_i]
                                expected = query_expected_answer_chunks(qlog)
                                done = set(query_completed_answer_chunks(qlog))
                                return [c for c in expected if c not in done]
                            q_obj = multi_q_list[qi]
                            ans_ch = q_obj.get("answer_chunks") or []
                            if hasattr(ans_ch, "tolist"):
                                ans_ch = ans_ch.tolist()
                            out: List[int] = []
                            for raw_c in ans_ch:
                                try:
                                    out.append(int(raw_c))
                                except (TypeError, ValueError):
                                    continue
                            return sorted(set(out))

                        pending_expected: List[Tuple[int, int, List[int]]] = []
                        for pos, qi in enumerate(pending_q_indices):
                            expected = _unmatched_expected_chunks_for_q(qi)
                            pending_expected.append((pos, qi, expected))

                        # 1. On-time/window match: same as the reward window.
                        # For single-answer cards answer_chunks may denote a
                        # short acceptable interval; for multi-emit cards they
                        # are the exact expected emit chunks.
                        for pos, _qi, expected in pending_expected:
                            if expected and min(expected) <= chunk_idx <= max(expected):
                                chosen_pos = pos
                                break

                        # 2. Late answer: choose the unfinished question whose
                        # latest unmatched expected chunk is closest behind the
                        # current chunk. This lets a late answer repair an older
                        # pending question instead of being routed to the newest
                        # query by LIFO.
                        if chosen_pos is None:
                            late_candidates: List[Tuple[int, int]] = []
                            for pos, _qi, expected in pending_expected:
                                past = [c for c in expected if c <= chunk_idx]
                                if past:
                                    late_candidates.append((max(past), pos))
                            if late_candidates:
                                _expected_chunk, chosen_pos = max(late_candidates)

                        # 3. Early answer: if no question is due/past-due,
                        # attribute to the closest future answer slot so timing
                        # records an early event that does not complete it.
                        if chosen_pos is None:
                            early_candidates: List[Tuple[int, int]] = []
                            for pos, _qi, expected in pending_expected:
                                future = [c for c in expected if c > chunk_idx]
                                if future:
                                    early_candidates.append((min(future), pos))
                            if early_candidates:
                                _expected_chunk, chosen_pos = min(early_candidates)

                        # 4. Fallback for malformed/no-answer-chunk metadata:
                        # newest live query, matching the prompt renderer's
                        # single-active behavior.
                        if chosen_pos is None:
                            chosen_pos = len(pending_q_indices) - 1

                        q_idx = pending_q_indices[chosen_pos]
                        answer_event: Dict[str, Any] = {
                            "chunk": int(chunk_idx),
                            "text": answer_str,
                        }
                        if per_q_answer_chunk[q_idx] < 0:
                            per_q_answer_chunk[q_idx] = chunk_idx
                            per_q_answer_text[q_idx] = answer_str
                        qlog_i = query_log_idx_by_q.get(q_idx)
                        if qlog_i is not None and 0 <= qlog_i < len(query_log):
                            timing = append_query_answer_with_timing(
                                query_log[qlog_i],
                                answer_str,
                                chunk_idx * self.chunk_sec,
                                chunk_sec=self.chunk_sec,
                            )
                            answer_event.update({
                                "timing": timing.get("timing"),
                                "expected_chunk": timing.get("expected_chunk"),
                                "counts_for_completion": bool(
                                    timing.get("counts_for_completion")
                                ),
                            })
                            if timing.get("lead_chunks") is not None:
                                answer_event["lead_chunks"] = timing.get("lead_chunks")
                            if timing.get("delay_chunks") is not None:
                                answer_event["delay_chunks"] = timing.get("delay_chunks")
                            query_log[qlog_i]["status"] = (
                                "answered" if query_is_complete(query_log[qlog_i]) else "open"
                            )
                        per_q_answers[q_idx].append(answer_event)
                        if _question_complete(q_idx):
                            pending_q_indices.pop(chosen_pos)

                # default_update_state advances chunk_idx by +1 on
                # EVERY turn — including compress. For inter-chunk
                # compress turns we DON'T want to skip ahead in the
                # video timeline, so we revert state.chunk_idx after.
                # If the output violated the turn-local action space, do
                # not feed the raw text to default_update_state: a
                # compress-prompt <answer> must not terminate the rollout,
                # and a streaming-prompt compress tool_call must not mutate
                # memory.
                # `kind` / `parsed` / `response_text` here are from the
                # FINAL inner-loop turn (the chunk's terminating
                # answer/silent/compress, after any in-chunk recall
                # multi-turn rounds have completed).
                invalid_action_space = bool(parsed.get("action_space_error"))
                pre_chunk_idx = state.chunk_idx
                if invalid_action_space:
                    # Treat illegal action-space outputs as semantic no-ops.
                    # The local cursor advances below to keep rollout moving.
                    state.chunk_idx = chunk_idx + 1
                else:
                    state = default_update_state(state, response_text, chunk_idx)
                    if inter_chunk:
                        # System event — don't consume a video chunk.
                        state.chunk_idx = pre_chunk_idx
                        # The next turn is the first visual turn of the new
                        # compressed segment. A streaming backend must not
                        # continue from old visual KV; it should rebuild from
                        # the compressed text state and then append this
                        # chunk's fresh visual window.
                        stream_reset_before_next_turn = True
                    else:
                        # Append only visual-turn think to memory. post_recall
                        # reasoning is conditioned on retrieved evidence, not a
                        # fresh frame observation, so it must not become stale
                        # visual memory for later chunks.
                        think_text = parsed.get("think") or ""
                        if (
                            think_text
                            and turn_kind != "post_recall"
                            and kind in ("answer", "recall", "unknown")
                        ):
                            item = {"chunk": chunk_idx, "text": think_text}
                            state.recent_thinks.append(item)
                            state.think_archive.append(dict(item))
                        if multi_q_list:
                            all_questions_complete = all(
                                _question_complete(qi)
                                for qi in range(len(multi_q_list))
                            )
                            if not all_questions_complete:
                                # default_update_state is single-question
                                # oriented and marks any non-empty answer as done.
                                # Multi-Q and multi-emit trajectories must keep
                                # streaming until every question has enough answers
                                # or the fixed rollout horizon is reached.
                                state.is_active = True
                                state.is_done = False
                # NOTE: recall handling moved into the inner ready-loop
                # above (chunk-internal multi-turn). The legacy "deliver
                # recall_result on the next chunk" path is exercised only
                # when max_recall_per_chunk is exhausted within the chunk.

                if not state.is_active or state.is_done:
                    break
                if (
                    self.recurrent_mode != "recurrent"
                    and len(response_mask) >= self.response_length
                ):
                    break

                # Advance video chunk pointer ONLY on non-compress turns.
                if (not inter_chunk) or invalid_action_space:
                    chunk_idx += 1
                # On compress turn we re-enter the loop at the same
                # chunk_idx; default_update_state cleared the
                # compressed range from recent_thinks already, so the
                # next iteration's _check_compress_trigger should return
                # None (loop progresses).

            num_turns = num_assistant_turns + 1

            # One environment/action unit is one video chunk. Some chunks
            # produce multiple assistant generations internally (e.g. recall
            # tool_call + post-recall answer, or a compression system turn
            # before the chunk is retried). Keep those as trainable subturn
            # rows, but expose chunk-level unit accounting separately so
            # trainer metrics/audits do not confuse subturn rows with env
            # timesteps.
            unit_index_by_event: Dict[int, int] = {}
            unit_subturn_counts: Dict[int, int] = {}
            per_action_unit_indices: List[int] = []
            per_action_subturn_indices: List[int] = []
            for raw_event_idx in chunk_event_indices:
                try:
                    event_idx = int(raw_event_idx)
                except (TypeError, ValueError):
                    event_idx = -1
                if event_idx < 0:
                    event_idx = len(unit_index_by_event)
                if event_idx not in unit_index_by_event:
                    unit_index_by_event[event_idx] = len(unit_index_by_event)
                unit_idx = unit_index_by_event[event_idx]
                per_action_unit_indices.append(unit_idx)
                subturn_idx = unit_subturn_counts.get(unit_idx, 0)
                per_action_subturn_indices.append(subturn_idx)
                unit_subturn_counts[unit_idx] = subturn_idx + 1
            n_action_units = len(unit_index_by_event)

            # Common extra_fields content for both stitched and recurrent modes.
            common_extras = {
                "turn_scores": [],
                "tool_rewards": [],
                "ts_rollout_engine": self.rollout_engine,
                "ts_frame_protocol": self.frame_protocol,
                "ts_n_recall": float(state.n_recall_calls),
                "ts_n_compress": float(state.n_compress_calls),
                "ts_chunks_used": float(n_action_units),
                "ts_turns_used": float(num_assistant_turns),
                "ts_action_rows_used": float(len(per_action_prompt_ids)),
                "ts_action_units_used": float(n_action_units),
                "ts_chunks_with_frames": float(n_chunks_with_frames),
                "ts_chunks_text_only": float(n_chunks_text_only),
                "ts_chunks_compress_inter": float(n_chunks_compress_inter),
                "ts_answer_chunk": float(
                    state.final_answer_chunk
                    if state.final_answer_chunk is not None else -1
                ),
                "ts_final_answer": state.final_answer or "",
                "ts_chunk_asst_spans": chunk_asst_spans,
                "ts_chunk_kinds": chunk_kinds,
                "ts_chunk_turn_kinds": chunk_turn_kinds,
                "ts_chunk_action_space_errors": chunk_action_space_errors,
                "ts_chunk_asst_texts": chunk_asst_texts,
                "ts_chunk_video_indices": chunk_video_indices,
                "ts_chunk_event_indices": chunk_event_indices,
                "ts_chunk_prompt_lens": chunk_prompt_lens,
                "ts_chunk_response_lens": chunk_response_lens,
                "ts_chunk_max_tokens": chunk_max_tokens,
                "ts_chunk_hit_max_tokens": chunk_hit_max_tokens,
                "ts_chunk_stop_reasons": chunk_stop_reasons,
                "ts_budget_abort_events": budget_abort_events,
                "ts_recall_query_ranges": chunk_recall_query_ranges,
                "ts_recall_returned_chunks": chunk_recall_returned_chunks,
                "ts_recall_result_sources": chunk_recall_result_sources,
                "ts_compress_expected_chunks": chunk_compress_expected_chunks,
                "ts_compress_emitted_ranges": chunk_compress_emitted_ranges,
                "ts_compress_trigger_source": (
                    "offline_pass2_boundaries"
                    if use_offline_compress else "runtime_memory_threshold"
                ),
                "ts_offline_compress_chunks": offline_compress_chunks,
                "ts_per_q_answer_chunk": list(per_q_answer_chunk),
                "ts_per_q_answer_text": list(per_q_answer_text),
                "ts_per_q_answers": per_q_answers,
                "ts_n_questions": float(len(multi_q_list)),
            }

            # ──────────────────────────────────────────────────────
            # v12.14 Phase 3: dispatch on recurrent_mode
            # ──────────────────────────────────────────────────────
            if self.recurrent_mode == "recurrent":
                # Emit one AgentLoopOutput per assistant action. Phase 1's
                # AgentLoopWorker.generate_sequences flattens these and
                # tags sample_index + final_mask. Phase 4d's ray_trainer
                # extracts trajectory-level reward from final actions,
                # computes 1D GRPO advantage by uid, then broadcasts the
                # advantage back to all sibling action rows via sample_index.
                #
                # Per-action: prompt_ids = full-context prompt at that
                # round (includes prior recall multi-turn context for
                # inner round 2+). response_ids = JUST that round's
                # assistant tokens (mask all 1s — no user-block in the
                # per-action response since the prompt already absorbed it).
                outputs: List[AgentLoopOutput] = []
                n_actions = len(per_action_prompt_ids)
                if n_actions == 0:
                    # Pathological: rollout produced nothing. Emit one empty
                    # action so AgentLoopWorker doesn't crash on empty list.
                    outputs.append(AgentLoopOutput(
                        prompt_ids=initial_prompt_ids,
                        response_ids=[],
                        response_mask=[],
                        multi_modal_data=(
                            {
                                **({"images": initial_images} if initial_images else {}),
                                **({"videos": initial_videos} if initial_videos else {}),
                            }
                        ),
                        num_turns=1,
                        metrics=metrics,
                        extra_fields={
                            **common_extras,
                            "ts_action_index": 0,
                            "ts_n_actions_in_traj": 1,
                            "ts_action_is_final": True,
                            "ts_action_unit_index": 0,
                            "ts_n_action_units_in_traj": 1,
                            "ts_action_subturn_index": 0,
                            "ts_action_unit_is_final": True,
                        },
                    ))
                else:
                    for ai in range(n_actions):
                        a_mm = per_action_mm_data[ai] or {}
                        unit_idx = (
                            per_action_unit_indices[ai]
                            if ai < len(per_action_unit_indices) else ai
                        )
                        ext = {
                            **common_extras,
                            "ts_action_index": ai,
                            "ts_n_actions_in_traj": n_actions,
                            "ts_action_is_final": ai == n_actions - 1,
                            "ts_action_unit_index": unit_idx,
                            "ts_n_action_units_in_traj": n_action_units,
                            "ts_action_subturn_index": (
                                per_action_subturn_indices[ai]
                                if ai < len(per_action_subturn_indices) else 0
                            ),
                            "ts_action_unit_is_final": unit_idx == n_action_units - 1,
                            # video chunk this action belongs to (-1 for
                            # compress inter-chunk turns); useful for the
                            # trainer-side reward path.
                            "ts_action_chunk_idx": (
                                chunk_video_indices[ai]
                                if ai < len(chunk_video_indices) else -1
                            ),
                            "ts_action_event_chunk_idx": (
                                chunk_event_indices[ai]
                                if ai < len(chunk_event_indices) else -1
                            ),
                        }
                        outputs.append(AgentLoopOutput(
                            prompt_ids=per_action_prompt_ids[ai],
                            response_ids=per_action_response_ids[ai],
                            response_mask=per_action_response_mask[ai],
                            response_logprobs=per_action_response_logprobs[ai],
                            multi_modal_data=a_mm,
                            num_turns=1,  # one assistant turn per action
                            metrics=metrics,
                            extra_fields=ext,
                        ))
                release_streaming_session = getattr(
                    self.server_manager, "release_streaming_session", None
                )
                if release_streaming_session is not None:
                    maybe_awaitable = release_streaming_session(request_id)
                    if hasattr(maybe_awaitable, "__await__"):
                        await maybe_awaitable
                return outputs

            # ──────────────────────────────────────────────────────
            # Stitched mode (default, backward-compat)
            # ──────────────────────────────────────────────────────
            multi_modal_data: Dict[str, Any] = {}
            if accumulated_images:
                multi_modal_data["images"] = accumulated_images
            if accumulated_videos:
                multi_modal_data["videos"] = accumulated_videos

            output_obj = AgentLoopOutput(
                prompt_ids=initial_prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=(
                    response_logprobs[: self.response_length]
                    if any_logprobs_returned else None
                ),
                multi_modal_data=multi_modal_data,
                num_turns=num_turns,
                metrics=metrics,
                extra_fields={},
            )
            output_obj.extra_fields.update(common_extras)
            release_streaming_session = getattr(
                self.server_manager, "release_streaming_session", None
            )
            if release_streaming_session is not None:
                maybe_awaitable = release_streaming_session(request_id)
                if hasattr(maybe_awaitable, "__await__"):
                    await maybe_awaitable
            return output_obj

    # Manual registration with factory-function target so hydra can locate it.
    # The @register decorator stores subclass.__qualname__ which breaks for
    # local classes (contains '<locals>'). We override with the factory path.
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry  # type: ignore
    _agent_loop_registry["thinkstream_streaming_agent"] = {
        "_target_": "thinkstream.rl.streaming_agent_loop.make_thinkstream_streaming_agent_loop"
    }

    return ThinkStreamStreamingAgentLoop


# ---------------------------------------------------------------------------
# Factory for hydra instantiate (avoids local-class locate issue).
# ---------------------------------------------------------------------------
def make_thinkstream_streaming_agent_loop(**kwargs):
    """Return an instance of ThinkStreamStreamingAgentLoop.

    hydra.utils.instantiate can't locate a class defined inside another
    function (its __qualname__ contains '<locals>'). This factory lives at
    module level, so agent_loops.yaml can point `_target_` here; we call
    `_register_streaming_agent_loop()` to get the real class and instantiate.
    """
    cls = _register_streaming_agent_loop()
    return cls(**kwargs)


try:
    ThinkStreamStreamingAgentLoop = _register_streaming_agent_loop()
except Exception as e:  # noqa: BLE001
    logger.debug("ThinkStreamStreamingAgentLoop not registered: %s", e)
