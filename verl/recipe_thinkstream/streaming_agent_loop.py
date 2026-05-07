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
# DESIGN — MemAgent-style sliding window with per-chunk independent generate:
# ────────────────────────────────────────────────────────────────────────────
# Each chunk is one INDEPENDENT vLLM generate request whose prompt is freshly
# constructed every turn (mirrors SFT pass5_messages.py exactly):
#
#     prompt_chunk_N = [
#       <|im_start|>system\n{SYSTEM_PROMPT_V12}\n<|im_end|>     ← common prefix
#       <|im_start|>user\n{question}\n<|im_end|>                ← common prefix
#       <|im_start|>user\n
#         <memory>...</memory>
#         <active_query>...</active_query> plus <response_history>...</response_history>
#                                             (while a query is live)
#         <visual_window>{header}</visual_window>               ← turn-specific
#         {frame-tag text + image items for chunks[max(0,N-15)..N]}
#         <user_input>...</user_input>     (question text or bare compress trigger)
#       <|im_end|>
#       <|im_start|>assistant\n
#     ]
#
# CACHE BEHAVIOUR (v12.13, 2026-05-02):
#   Two independent caches help streaming-video rollout. Each targets a
#   different bottleneck.
#
#   1. vLLM mm_processor_cache (CPU, configured via
#      engine_kwargs.vllm.mm_processor_cache_gb=64 in run_thinkstream_grpo.sh):
#        Caches (PIL load + smart_resize + ViT-friendly tensor) per
#        (frame_path, min_pixels, max_pixels) key. Sliding window means
#        any single frame reappears in N=visual_window_chunks consecutive
#        chunks, so per-frame ViT prep is reused (N-1)/N ≈ 94% of the
#        time. THIS IS THE PRIMARY OPTIMIZATION for our streaming-video
#        workload — pass2 teacher rollout (--mm-processor-cache-gb 512)
#        empirically gets 93.8% mm-cache hit rate. RL inherits the same
#        mechanism as long as mm_processor_kwargs are byte-stable across
#        chunks (v12.12's RUNTIME_MM_PROCESSOR_KWARGS guarantees that).
#
#   2. vLLM enable_prefix_caching (GPU, thinkstream_grpo.yaml):
#        Reuses attention KV blocks for byte-identical prompt prefixes.
#        Across two chunks of one trajectory, the [system + user_q]
#        prefix (~600 tok) and SFT-aligned [<memory>'s leading thinks]
#        are byte-identical, so prefix cache saves the prefill there.
#        BUT the visual block (which dominates token count) is a
#        cache MISS under sliding window because frame token IDs shift
#        every chunk. We do NOT restructure the prompt to chase prefix-
#        cache hits on visuals — that's mm_processor_cache's job.
#
#   The "expanding" visual-window mode below is an OPT-IN experiment for
#   prefix-cache-on-visual workloads; default is "sliding" to preserve
#   SFT-RL distribution alignment with the existing data.
#
# PROMPT LAYOUT (must match SFT exactly — see
# thinkstream/data/agent_protocol.py:213-214 build_user_content):
#
#     prompt_chunk_t = [
#       system + user_q                        ← stable across chunks
#       <memory>                               ← monotonic append; SFT-first
#       (active_query + response_history)      ← optional while a query is live
#       <visual_window header>                 ← {start, end, frames, current_time}
#       frame-tag text + image items          ← sliding window (or expanding opt-in)
#       <recall_result> (optional)             ← chunk-specific
#       <user_input>                           ← question or bare compress trigger
#     ]
#
# WINDOW MODE (THINKSTREAM_VISUAL_WINDOW_MODE):
#   "sliding"  (default, matches SFT agent_protocol.py:275 and
#               pass5_messages.py): window = [max(0, chunk-VWC+1) .. chunk].
#               Frame token IDs shift left every chunk → 0% prefix-cache
#               hit on visuals; mm_processor_cache reuses the per-frame
#               ViT prep instead.
#   "expanding" (opt-in, requires re-running pass2/pass5 with the same
#                env var to keep SFT data in sync): window starts at the
#                segment boundary (chunk // VWC × VWC) and grows to chunk.
#                Within a segment frame IDs are byte-stable, so prefix
#                cache hits the visual block ~94% (15/16). Boundary
#                chunks lose recent visual context — re-verify SFT
#                quality after switching.
#
# SFT ALIGNMENT (the 5 things that must match pass5_messages.py):
#   1. Frame paths use 1-indexed numbering: frame_{ci*FPC + fi + 1:06d}.jpg
#   2. Pre-extracted frames render through the selected frame protocol:
#      `ts_image` uses frame-tag text + image items; `video_meta` uses one
#      Qwen video block with explicit video_metadata. Both carry real time.
#   3. Frame timestamps/metadata use frame_idx / fps, where
#      frame_idx = window_start*FPC + i.
#   4. Compress turn uses a compression-only system prompt plus bare
#      <compress_trigger/> (v12.12: no range). It still carries memory and
#      visual_window so SFT/RL/eval use one multimodal payload shape.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from thinkstream.data.agent_protocol import (
    RECALL_RETURN_CHUNKS,
    build_recalled_frames_metadata,
    recall_time_string_for_chunks,
    select_recall_chunks,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Token budget for compress system trigger — MUST match pass2 / SFT / eval
# exactly. Single source of truth is scripts/agent_data_v5/config.py:
#
#   RECENT_THINKS_TOKEN_BUDGET    = 4000   # total budget
#   COMPRESS_TRIGGER_RATIO        = 0.8    # fire at 80%
#   COMPRESS_TOKEN_THRESHOLD      = 3200   # = budget × ratio
#   COMPRESS_RANGE_MIN            = 8      # min items in a compress range
#
# v12.13 (2026-05-03): RL was using a hardcoded 3200 + word-count×1.3
# estimate, which drifted from pass2 / eval (both use the real Qwen
# tokenizer). Diverging means RL's compress fires at a different memory
# state than what the model trained on under SFT, breaking SFT-RL
# distribution alignment. Now: import the constants directly and use
# self.tokenizer (already loaded by AgentLoopBase) for accurate counting.
try:
    from scripts.agent_data_v5.config import (  # type: ignore
        RECENT_THINKS_TOKEN_BUDGET,
        COMPRESS_TOKEN_THRESHOLD,
        COMPRESS_RANGE_MIN,
    )
except ImportError:
    # Fallback for envs that don't have THINKSTREAM_HOME on PYTHONPATH —
    # values mirror config.py but won't auto-update if config changes.
    RECENT_THINKS_TOKEN_BUDGET = 4000
    COMPRESS_TOKEN_THRESHOLD = 3200
    COMPRESS_RANGE_MIN = 8

# Backward-compat alias for any external import; new code should use
# COMPRESS_TOKEN_THRESHOLD directly.
DEFAULT_COMPRESS_TOKEN_THRESHOLD = COMPRESS_TOKEN_THRESHOLD


# Module-level placeholder so hydra's `_target_:
# recipe_thinkstream.streaming_agent_loop.ThinkStreamStreamingAgentLoop`
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
    paths: List[str] = []
    for fi in range(frames_per_chunk):
        # 1-indexed file numbering — pass1a's ffmpeg `frame_%06d.jpg`
        # default starts at frame_000001. Older test harnesses with
        # 0-indexed dumps fall back to the next probe.
        idx_one = chunk_idx * frames_per_chunk + fi + 1
        candidates = [
            frame_dir / f"frame_{idx_one:06d}.jpg",
            frame_dir / f"frame_{idx_one:05d}.jpg",
            frame_dir / f"frame_{idx_one:04d}.jpg",
        ]
        # 0-indexed legacy fallback
        idx_zero = chunk_idx * frames_per_chunk + fi
        candidates.extend([
            frame_dir / f"frame_{idx_zero:06d}.jpg",
            frame_dir / f"frame_{idx_zero:05d}.jpg",
        ])
        chosen: Optional[Path] = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            return []
        paths.append(str(chosen))
    return paths


def _compute_window_start(
    chunk_idx: int,
    visual_window_chunks: int,
    mode: str = "sliding",
) -> int:
    """Compute the start chunk of the visual window for `chunk_idx`.

    mode="sliding"   (legacy / SFT-aligned): window_start = max(0, chunk-VWC+1)
                     The window slides 1 chunk per step. Frame token IDs in
                     the prompt shift left by `frames_per_chunk` every step
                     → vLLM prefix cache misses on the visual block.
    mode="expanding" (v12.13 / KV-friendly): window_start = (chunk // VWC) * VWC
                     The window is anchored at a segment boundary and grows
                     until it hits the next segment. Within a segment the
                     leading frames occupy IDENTICAL prompt positions across
                     chunks → vLLM prefix cache hits ~(VWC-1)/VWC of visual
                     tokens. Trade-off: at chunk_idx % VWC == 0 the window
                     contains only 1 chunk of frames; recent context is
                     thinner than `sliding` for the first chunk in each
                     segment. Best paired with SFT data regenerated under
                     the same mode (pass5_messages.py:147 needs the same
                     branch); otherwise the rollout-time visual context
                     differs from training-time context for boundary chunks.
    """
    if mode == "expanding":
        seg = max(1, int(visual_window_chunks))
        return (int(chunk_idx) // seg) * seg
    # sliding (default)
    return max(0, int(chunk_idx) - int(visual_window_chunks) + 1)


def _build_visual_window(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    visual_window_chunks: int,
    frames_per_chunk: int,
    chunk_sec: float = 1.0,
    mode: str = "sliding",
) -> Tuple[List[str], int, int]:
    """Build the visual window for chunk N.

    Returns:
      flat_paths:     all frame paths in window order (window_start..N)
      window_start_chunk, window_end_chunk

    `mode` selects the windowing strategy (see _compute_window_start).
    """
    window_start = _compute_window_start(chunk_idx, visual_window_chunks, mode)
    window_end = chunk_idx
    flat_paths: List[str] = []
    for c in range(window_start, window_end + 1):
        cf = _chunk_frame_paths(video_path, frames_root, c, frames_per_chunk)
        if not cf:
            return [], window_start, window_end
        flat_paths.extend(cf)
    return flat_paths, window_start, window_end


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
    """Return a recall_result dict (source/time/text) for inline JSON
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
    from thinkstream.model.agent_loop import bm25_retrieve
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

    pass2 (scripts/agent_data_v5/pass2_rollout.py:183) and eval
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
        append_visual_frames,
        normalize_frame_protocol,
        parse_agent_output_v12,
        format_memory_block,
        format_queries_block,
        format_user_input_block,
        system_prompt_for_frame_protocol,
        tools_for_turn,
    )
    from thinkstream.trainer.v12_rollout import (  # type: ignore
        VideoTrajectoryState,
        default_v12_update_state,
    )

    class ThinkStreamStreamingAgentLoop(AgentLoopBase):
        """MemAgent-style chunk-level rollout for streaming video."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            self.max_model_len = self.rollout_config.max_model_len or (self.prompt_length + self.response_length)
            # response_length is the stitched trajectory buffer used by verl's
            # loss tensors. A single chunk action must be much smaller; pass2
            # uses 1024 for observation and 4096 for compress. Use 4096 as a
            # safe unified cap so max_model_len checks do not confuse the
            # whole-trajectory buffer with one vLLM request.
            self.max_tokens_per_action = int(
                os.environ.get("THINKSTREAM_MAX_TOKENS_PER_ACTION", "4096") or 4096
            )
            self.frame_protocol = normalize_frame_protocol(
                os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "ts_image")
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
                os.environ.get("THINKSTREAM_VISUAL_WINDOW_CHUNKS", "16") or 16
            )
            self.chunk_sec = float(
                os.environ.get("THINKSTREAM_CHUNK_SEC", "1.0") or 1.0
            )
            # v12.13: default = COMPRESS_TOKEN_THRESHOLD imported from
            # scripts/agent_data_v5/config.py (3200 = 80% of 4000-tok
            # RECENT_THINKS_TOKEN_BUDGET). Matches pass2 / eval / SFT
            # exactly — diverging here means RL's compress fires at
            # different memory state than what the model trained on.
            self.compress_token_threshold = int(
                os.environ.get(
                    "THINKSTREAM_COMPRESS_THRESHOLD",
                    str(COMPRESS_TOKEN_THRESHOLD),
                )
                or COMPRESS_TOKEN_THRESHOLD
            )
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

            inter_chunk=True marks a compression turn. It still carries the
            same visual_window + memory shape as SFT/pass5; only query and
            recall-answer context are suppressed.

            Mirrors the shared SFT/runtime layout in
            thinkstream/data/agent_protocol.py:213-214 build_user_content
            EXACTLY:
              <user_input> → <memory> → (active_query + response_history) →
              <visual_window> + protocol visual frames → <recall_result> →
              ...

            Distribution alignment is the hard constraint. Per-frame ViT
            re-encoding cost is handled by vLLM's mm_processor_cache
            (engine_kwargs.vllm.mm_processor_cache_gb in run_thinkstream_grpo.sh),
            not by rearranging the text blocks.
            """
            content: List[Dict[str, Any]] = []

            # User input — either the question (when it fires) or the bare
            # compress_trigger system event. This comes before memory to match
            # SFT/pass5/build_user_content.
            if compress_trigger_range is not None:
                user_input_block = format_user_input_block(
                    "<compress_trigger/>",
                    inter_chunk=True,
                )
            else:
                user_input_text = self._format_user_input(
                    chunk_idx, question, ask_chunks, triggered_questions,
                )
                user_input_block = format_user_input_block(
                    user_input_text,
                    inter_chunk=False,
                )
            if user_input_block:
                content.append({
                    "type": "text",
                    "text": user_input_block.lstrip("\n"),
                })

            # ── Memory block ──
            # P0.5 fix (post-review 2026-05-01): format_memory_block reads
            # the dict under "compressed_segments" or legacy "compressed".
            # We were passing "compressed_summaries" → memory after compress
            # silently disappeared from the prompt. Pass under BOTH keys
            # so the legacy reader path works regardless of which alias
            # format_memory_block prefers.
            try:
                mem_text = format_memory_block({
                    "compressed_segments": state.compressed_summaries,
                    "compressed": state.compressed_summaries,
                    "recent_thinks": state.recent_thinks,
                })
            except Exception:
                mem_text = ""
            content.append({
                "type": "text",
                "text": f"\n<memory>\n{mem_text}\n</memory>" if content
                else f"<memory>\n{mem_text}\n</memory>",
            })

            # ── Active query block — same renderer as SFT/pass5. Data carries
            # structured question/options/answer_style fields; prompt text is
            # rendered here so eval adapters can reuse the same interface.
            try:
                queries_text = format_queries_block(queries or [])
            except Exception:
                queries_text = ""
            if queries_text and not inter_chunk:
                content.append({
                    "type": "text",
                    "text": f"\n{queries_text}",
                })

            # ── Visual window header + protocol-selected frame carrier
            # (after memory, matches SFT). Header layout copies
            # agent_protocol.py: keys
            # `start`, `end`, `frames`, `current_time` are all required —
            # SFT trained the model on this exact JSON shape, removing
            # any field would diverge train/RL distribution.
            vw_header = json.dumps({
                "start": window_start_chunk * self.chunk_sec,
                "end": (window_end_chunk + 1) * self.chunk_sec,
                "frames": len(window_paths),
                "current_time": [
                    chunk_idx * self.chunk_sec,
                    (chunk_idx + 1) * self.chunk_sec,
                ],
            })
            content.append({
                "type": "text",
                "text": f"\n<visual_window>{vw_header}</visual_window>",
            })
            if window_paths:
                # v12.22: runtime resize bounds are attached to each image.
                # v12.13: identical kwargs
                # across chunks → vLLM mm_processor_cache key is stable
                # (frame_path, min_pixels, max_pixels) so PIL+ViT
                # preprocessing is cached when the same frame recurs in
                # consecutive sliding windows. This is the ONLY visual-
                # token reuse mechanism we rely on; do not rearrange the
                # surrounding content blocks for prefix-cache purposes.
                try:
                    from scripts.agent_data_v5.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
                append_visual_frames(
                    content,
                    window_paths,
                    frame_protocol=self.frame_protocol,
                    fps=float(self.frames_per_chunk) / float(self.chunk_sec),
                    start_frame_index=window_start_chunk * self.frames_per_chunk,
                    total_num_frames=(chunk_idx + 1) * self.frames_per_chunk,
                    latest_start_frame_index=chunk_idx * self.frames_per_chunk,
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )

            # Recall result (single-turn legacy form — SFT shape A inline).
            # True shape-B intra-chunk multi-turn is a deferred follow-up.
            if recall_result is not None and not inter_chunk:
                rr_json = json.dumps({
                    "source": recall_result.get("source", ""),
                    "time": recall_result.get("time", ""),
                    "text": recall_result.get("text", ""),
                }, ensure_ascii=False)
                content.append({
                    "type": "text",
                    "text": f"\n<recall_result>{rr_json}</recall_result>",
                })

            return content

        async def _execute_recall(
            self, args: Dict[str, Any], state: "VideoTrajectoryState",
            *, video_path: str = "",
        ) -> Dict[str, Any]:
            """v12.13 D1: build the COMPLETE tool response for shape-B recall.

            Returns a dict matching what SFT pass5_messages.py:280-380 expects:
              {
                "recall_result": {source, text_content, returned_chunks, time}
                                 — same shape as pass3c_samples._recall_result_for
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
            recall_result = {
                "source": "historical_frames" if recalled_frame_paths else (
                    text_result.get("source", "memory") if success else "failure"
                ),
                # SFT (pass3c) stores under `text_content`; pass5 reads with
                # text_content fallback to `text`. Provide BOTH keys so
                # downstream renderers don't care which one they read.
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
              3. <recall_result>{json}</recall_result> text

            We use role="user" (not "tool") to match pass5's DeepEyesV2
            ShareGPT alignment — the chat_template renders both as
            <|im_start|>user\\n<tool_response>... so the on-the-wire
            token stream is identical, but role="user" plays nicer with
            verl's downstream chat-template handling.
            """
            rr = recall_payload.get("recall_result") or {}
            rf = recall_payload.get("recalled_frames")  # may be None
            content: List[Dict[str, Any]] = []

            # 1. <recalled_frames> header text (only when frames exist)
            if rf:
                rf_header = json.dumps({
                    "time_range": rf["time_range"],
                    "source": rf.get("source", "historical_frames"),
                    "n_frames": rf["n_frames"],
                })
                content.append({
                    "type": "text",
                    "text": f"<recalled_frames>{rf_header}</recalled_frames>",
                })
                # 2. historical frames via the active frame protocol.
                try:
                    from scripts.agent_data_v5.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
                tr_start, tr_end = rf["time_range"]
                append_visual_frames(
                    content,
                    rf["frame_paths"],
                    frame_protocol=self.frame_protocol,
                    fps=float(self.frames_per_chunk) / float(self.chunk_sec),
                    start_frame_index=int(tr_start) * self.frames_per_chunk,
                    total_num_frames=int(tr_end) * self.frames_per_chunk,
                    context_label="recalled frame",
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )

            # 3. <recall_result> text — pass5 puts this AFTER frames
            #    (v12.11 audit-5 P0 fix). Empty/failure case still emits
            #    the tag so the model can parse "no result found".
            rr_json = json.dumps({
                "source": rr.get("source", "failure"),
                "time": rr.get("time", ""),
                "text": rr.get("text_content", rr.get("text", "")),
            }, ensure_ascii=False)
            content.append({
                "type": "text",
                "text": (
                    f"\n<recall_result>{rr_json}</recall_result>"
                    if rf else
                    f"<recall_result>{rr_json}</recall_result>"
                ),
            })

            return {"role": "user", "content": content}

        def _check_compress_trigger(
            self, state: "VideoTrajectoryState"
        ) -> Optional[Tuple[int, int]]:
            """Return (start_chunk, end_chunk) range to compress, or None.

            Mirrors pass2 (scripts/agent_data_v5/pass2_rollout.py:199) and
            eval (thinkstream/model/agent_loop.py:181) MemoryState.should_compress():
              fires when recent_thinks tokens >= COMPRESS_TOKEN_THRESHOLD
              AND len(recent_thinks) >= COMPRESS_RANGE_MIN.

            Token counting uses self.tokenizer (the same Qwen tokenizer
            verl loaded for the actor) — NOT word-count×1.3 — so the
            trigger fires at the same memory state pass2/eval would.
            """
            if not state.recent_thinks:
                return None
            if len(state.recent_thinks) < COMPRESS_RANGE_MIN:
                # Match eval MemoryState.should_compress's range_min guard:
                # at least 8 thinks before compress is meaningful.
                return None
            est = _count_recent_thinks_tokens(
                state.recent_thinks, tokenizer=self.tokenizer,
            )
            if est < self.compress_token_threshold:
                return None
            chunks = [
                int(t.get("chunk", -1))
                for t in state.recent_thinks
                if isinstance(t, dict) and t.get("chunk", -1) >= 0
            ]
            if not chunks:
                return None
            return (min(chunks), max(chunks))

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
                terminal event. A question is removed from the pending queue
                only after a non-empty answer; multi-emit cards need one
                non-empty answer per expected answer chunk.
                """
                if q_idx < 0 or q_idx >= len(multi_q_list):
                    return True
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

            # ── Initial prompt: [system + user(question)]. Cached on vLLM
            # side; never re-prefilled across chunks.
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
            # P1.7 fix (post-review 2026-05-01): chunk_kinds/spans/texts
            # are appended on EVERY assistant turn including inter-chunk
            # compress turns. Without a parallel video-chunk-index list,
            # compute_score's per-chunk action gold lookup
            # (`enumerate(chunk_kinds)` → chunk_idx) would right-shift after
            # each compress turn. Track the actual video chunk_idx per
            # turn here; compute_score uses this for `gold_action_per_chunk[
            # str(video_chunk_idx)]` lookup. -1 marks compress (system event).
            chunk_video_indices: List[int] = []

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

            state = VideoTrajectoryState(video_uid=str(video_id), chunk_idx=0)
            recall_result_for_next: Optional[Dict[str, Any]] = None
            num_assistant_turns = 0
            n_chunks_with_frames = 0
            n_chunks_text_only = 0
            n_chunks_compress_inter = 0

            chunk_idx = 0
            while chunk_idx < n_chunks:
                if not state.is_active:
                    break

                # ── Decide turn type: compress trigger fires BETWEEN
                # chunks. The compress turn still receives the same memory +
                # visual-window payload shape; only the system prompt and
                # allowed action differ.
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
                visual_injected = bool(window_paths)

                # ── Multi-Q: which questions fire at this chunk?
                triggered_qs_for_chunk: List[Dict[str, Any]] = []
                triggered_q_indices_for_chunk: List[int] = []
                if multi_q_list and not inter_chunk:
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
                                "answer_style": q_obj.get("answer_style", ""),
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
                    pending_q_indices.extend(triggered_q_indices_for_chunk)

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

                chunk_messages = list(initial_messages)
                if inter_chunk:
                    # Preserve the stable raw_prompt prefix for vLLM prefix
                    # slicing, then add a turn-local system message that
                    # carries the compression-only policy. The old user-side
                    # memory-compaction rules are gone; <user_input> contains
                    # only the bare legacy trigger marker.
                    chunk_messages.append({
                        "role": "system",
                        "content": system_prompt_for_frame_protocol(
                            self.frame_protocol,
                            inter_chunk=True,
                        ),
                    })
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
                    # Resolve images/videos for the LATEST chunk_messages.
                    # On round 1 this is just the user payload. On round 2+
                    # it ALSO includes the appended assistant turn1 + tool
                    # message (recalled frames live in the tool message).
                    chunk_extra_mm = await self.process_vision_info(
                        chunk_messages[len(initial_messages):],
                    )
                    chunk_images = chunk_extra_mm.get("images") or []
                    chunk_videos = chunk_extra_mm.get("videos") or []

                    turn_kind = (
                        "recall_response"
                        if recall_rounds_this_chunk > 0
                        else ("compress" if inter_chunk else "streaming")
                    )
                    turn_tools = tools_for_turn(turn_kind)
                    chunk_prompt_ids = await self.apply_chat_template(
                        chunk_messages,
                        tools=turn_tools,
                        images=(initial_images + chunk_images) if chunk_images else (
                            initial_images if initial_images else None
                        ),
                        videos=(initial_videos + chunk_videos) if chunk_videos else (
                            initial_videos if initial_videos else None
                        ),
                    )

                    # Prompt-budget guard (single-chunk + tool round can
                    # exceed budget after frames are appended; skip out
                    # if so — outer loop ends rollout). Use the per-action
                    # generation cap here, not the stitched response buffer.
                    remaining_context = self.max_model_len - len(chunk_prompt_ids)
                    if remaining_context <= 0:
                        inner_aborted = True
                        break
                    user_block_len = len(chunk_prompt_ids) - last_prompt_len
                    if user_block_len < 0:
                        inner_aborted = True
                        break
                    remaining_response = self.response_length - len(response_mask) - user_block_len - 1
                    max_tokens_this_turn = min(
                        self.max_tokens_per_action,
                        remaining_context,
                        remaining_response,
                    )
                    if max_tokens_this_turn <= 0:
                        inner_aborted = True
                        break

                    # ── Generate. v12.13 cache breakdown:
                    # - prefix cache hits the [system + user_q + memory's
                    #   leading thinks] head + the unchanged earlier part
                    #   of the user payload across rounds.
                    # - mm_processor_cache reuses ViT prep for the ~94%
                    #   of frames that overlap the previous chunk's
                    #   sliding window (matches pass2's 93.8% hit).
                    with simple_timer("generate_sequences", metrics):
                        output: TokenOutput = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=chunk_prompt_ids,
                            sampling_params={
                                **sampling_params,
                                "max_tokens": max_tokens_this_turn,
                            },
                            image_data=(initial_images + chunk_images) if chunk_images else (
                                initial_images if initial_images else None
                            ),
                            video_data=(initial_videos + chunk_videos) if chunk_videos else (
                                initial_videos if initial_videos else None
                            ),
                        )
                    assistant_ids = list(output.token_ids)
                    if not assistant_ids:
                        inner_aborted = True
                        break

                    # ── Stitch (incremental relative to last_prompt_len).
                    user_block_ids = chunk_prompt_ids[last_prompt_len:]
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
                    if output.log_probs and len(output.log_probs) == len(assistant_ids):
                        per_action_response_logprobs.append(list(output.log_probs))
                    else:
                        per_action_response_logprobs.append(None)
                    # mm payload: stitched mode accumulates media globally;
                    # recurrent mode needs per-action attribution. The active
                    # frame protocol decides whether frames are images or
                    # video blocks with metadata.
                    _ac_mm = None
                    if chunk_videos:
                        _ac_mm = {"videos": list(chunk_videos)}
                    if chunk_images:
                        _ac_mm = _ac_mm or {}
                        _ac_mm["images"] = list(chunk_images)
                    per_action_mm_data.append(_ac_mm)
                    if output.log_probs and len(output.log_probs) == len(assistant_ids):
                        response_logprobs.extend(list(output.log_probs))
                        any_logprobs_returned = True
                    else:
                        response_logprobs.extend([0.0] * len(assistant_ids))
                    asst_end = len(response_ids)
                    chunk_asst_spans.append((asst_start, asst_end))
                    # -1 for compress (system inter-chunk turn).
                    chunk_video_indices.append(-1 if inter_chunk else chunk_idx)

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
                    response_text = self.tokenizer.decode(
                        assistant_ids, skip_special_tokens=True,
                    )
                    parsed = parse_agent_output_v12(response_text)
                    kind = parsed.get("kind", "unknown")
                    action_error = action_space_error_for_turn(kind, turn_kind)
                    if action_error:
                        parsed["action_space_error"] = action_error
                        parsed["invalid_kind"] = kind
                        kind = "invalid"
                    chunk_kinds.append(kind)
                    chunk_turn_kinds.append(turn_kind)
                    chunk_action_space_errors.append(action_error)
                    chunk_asst_texts.append(response_text)

                    # ── Decide: stay in chunk for shape-B recall multi-
                    # turn, or break out and advance chunk_idx.
                    if (
                        kind == "recall"
                        and not inter_chunk
                        and recall_rounds_this_chunk < self.max_recall_per_chunk
                    ):
                        args = (parsed.get("tool_call") or {}).get("arguments") or {}
                        recall_payload = await self._execute_recall(
                            args, state, video_path=video_path,
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
                        args = (parsed.get("tool_call") or {}).get("arguments") or {}
                        recall_payload = await self._execute_recall(
                            args, state, video_path=video_path,
                        )
                        recall_result_for_next = recall_payload.get("recall_result")
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
                #   1. answer_chunks window match: pending Q whose
                #      answer_chunks bracket the current chunk_idx.
                #   2. LIFO: most-recently-triggered pending Q.
                #   3. FIFO floor (implicit): single-pending case.
                if multi_q_list and pending_q_indices:
                    answer_str = (
                        parsed.get("answer_text") if kind == "answer" else None
                    )
                    if answer_str is not None:
                        answer_str = str(answer_str).strip()
                    if answer_str:

                        chosen_pos = None
                        # 1. answer_chunks window match
                        for pos, qi in enumerate(pending_q_indices):
                            q_obj = multi_q_list[qi]
                            ans_ch = q_obj.get("answer_chunks") or []
                            if hasattr(ans_ch, "tolist"):
                                ans_ch = ans_ch.tolist()
                            ans_ch_int = [
                                int(x) for x in ans_ch
                                if isinstance(x, (int, float))
                            ]
                            if ans_ch_int and (
                                min(ans_ch_int) <= chunk_idx <= max(ans_ch_int)
                            ):
                                chosen_pos = pos
                                break
                        # 2. LIFO (most-recent triggered)
                        if chosen_pos is None:
                            chosen_pos = len(pending_q_indices) - 1
                        # 3. FIFO is the implicit floor when 2 falls through
                        # (single pending Q → both LIFO/FIFO pick it).

                        q_idx = pending_q_indices[chosen_pos]
                        per_q_answers[q_idx].append({
                            "chunk": int(chunk_idx),
                            "text": answer_str,
                        })
                        if per_q_answer_chunk[q_idx] < 0:
                            per_q_answer_chunk[q_idx] = chunk_idx
                            per_q_answer_text[q_idx] = answer_str
                        qlog_i = query_log_idx_by_q.get(q_idx)
                        if qlog_i is not None and 0 <= qlog_i < len(query_log):
                            query_log[qlog_i].setdefault("answers", []).append({
                                "text": answer_str,
                                "time": chunk_idx * self.chunk_sec,
                            })
                            query_log[qlog_i]["status"] = (
                                "answered" if _question_complete(q_idx) else "open"
                            )
                        if _question_complete(q_idx):
                            pending_q_indices.pop(chosen_pos)

                # default_v12_update_state advances chunk_idx by +1 on
                # EVERY turn — including compress. For inter-chunk
                # compress turns we DON'T want to skip ahead in the
                # video timeline, so we revert state.chunk_idx after.
                # `kind` / `parsed` / `response_text` here are from the
                # FINAL inner-loop turn (the chunk's terminating
                # answer/silent/compress, after any in-chunk recall
                # multi-turn rounds have completed).
                invalid_action_space = bool(parsed.get("action_space_error"))
                pre_chunk_idx = state.chunk_idx
                state = default_v12_update_state(state, response_text, chunk_idx)
                if inter_chunk and not invalid_action_space:
                    # System event — don't consume a video chunk.
                    state.chunk_idx = pre_chunk_idx
                else:
                    # Append think to recent_thinks for next turn's memory.
                    think_text = parsed.get("think") or ""
                    if think_text and kind in ("answer", "recall", "unknown"):
                        item = {"chunk": chunk_idx, "text": think_text}
                        state.recent_thinks.append(item)
                        state.think_archive.append(dict(item))
                    if multi_q_list:
                        all_questions_complete = all(
                            _question_complete(qi)
                            for qi in range(len(multi_q_list))
                        )
                        if not all_questions_complete:
                            # default_v12_update_state is single-question
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
                if len(response_mask) >= self.response_length:
                    break

                # Advance video chunk pointer ONLY on non-compress turns.
                if (not inter_chunk) or invalid_action_space:
                    chunk_idx += 1
                # On compress turn we re-enter the loop at the same
                # chunk_idx; default_v12_update_state cleared the
                # compressed range from recent_thinks already, so the
                # next iteration's _check_compress_trigger should return
                # None (loop progresses).

            num_turns = num_assistant_turns + 1

            # Common extra_fields content for both stitched and recurrent modes.
            common_extras = {
                "turn_scores": [],
                "tool_rewards": [],
                "ts_frame_protocol": self.frame_protocol,
                "ts_n_recall": float(state.n_recall_calls),
                "ts_n_compress": float(state.n_compress_calls),
                "ts_chunks_used": float(num_assistant_turns),
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
                        },
                    ))
                else:
                    for ai in range(n_actions):
                        a_mm = per_action_mm_data[ai] or {}
                        ext = {
                            **common_extras,
                            "ts_action_index": ai,
                            "ts_n_actions_in_traj": n_actions,
                            # video chunk this action belongs to (-1 for
                            # compress inter-chunk turns); useful for the
                            # trainer-side reward path.
                            "ts_action_chunk_idx": (
                                chunk_video_indices[ai]
                                if ai < len(chunk_video_indices) else -1
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
            return output_obj

    # Manual registration with factory-function target so hydra can locate it.
    # The @register decorator stores subclass.__qualname__ which breaks for
    # local classes (contains '<locals>'). We override with the factory path.
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry  # type: ignore
    _agent_loop_registry["thinkstream_streaming_agent"] = {
        "_target_": "recipe_thinkstream.streaming_agent_loop.make_thinkstream_streaming_agent_loop"
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
