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
#         <visual_window>{header}</visual_window>               ← turn-specific
#         {video block: chunks[max(0,N-15)..N]'s frames + video_metadata}
#         <memory>...</memory>
#         <queries>...</queries>           (when ask_chunks fired)
#         <user_input>...</user_input>     (the question text)
#         OR <compress_trigger/>   (compress turn — system-injected, v12.12 no range)
#       <|im_end|>
#       <|im_start|>assistant\n
#     ]
#
# KV CACHE BEHAVIOUR (v12.12, 2026-05-01):
#   User content reordered to put MEMORY FIRST (stable monotonic prefix)
#   and visual_window AFTER. Across consecutive chunks of one trajectory:
#     prompt_chunk_t   = [system + user_q + <memory_at_t> + <queries_at_t>
#                         + <visual_window_t> + frames_t + <user_input_t>]
#     prompt_chunk_t+1 = [system + user_q + <memory_at_t+1> + ...]
#   memory_at_t+1 is memory_at_t with one extra appended think (modulo
#   periodic compression rewrites that invalidate the prefix and reset
#   the cache). vLLM's prefix cache therefore hits ~3-5K tokens of the
#   stable head every chunk instead of just the ~600-tok system block.
#
#   Sliding visual window still changes every turn (chunk 0's frames drop
#   out at turn 16) and remains a cache miss, but it's now LATER in the
#   sequence so the missed region is shorter.
#
# SFT ALIGNMENT (the 5 things that must match pass5_messages.py):
#   1. Frame paths use 1-indexed numbering: frame_{ci*FPC + fi + 1:06d}.jpg
#   2. Single video block per chunk (not multiple image blocks) — Qwen3-VL
#      needs a single <|video_pad|> with video_metadata for MROPE temporal
#      alignment.
#   3. video_metadata.frames_indices = [window_start*FPC + i for i in range(n)]
#      — drives Qwen3-VL's per-frame `<X.X seconds>` temporal MROPE token.
#   4. Compress turn uses bare <compress_trigger/> (v12.12: no range,
#      no visual_window). Model derives time_range from memory.
#   5. Recall result rendering as <recall_result>{...}</recall_result>
#      JSON dict (source/time/text).
#
# DEFERRED (must be addressed before claiming SFT-RL parity):
#   D1. Intra-chunk recall multi-turn shape (P0.6 from review).
#       SFT shape B is: assistant→tool(recall_result + recalled_frames)
#       →assistant within ONE chunk. Current loop puts recall_result on
#       the NEXT chunk's user message AND drops recalled_frames. Effect:
#       model trains on a different recall topology than SFT — recall is
#       still learnable but with one-chunk delay and missing visual
#       context. Fix needs: when kind=recall, immediately build a tool
#       turn with recall_result + recalled_frames video block, append to
#       prompt_ids with mask=0, generate again before advancing chunk_idx.
#   D2. Per-chunk attention reset / training-time per-chunk forward
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
#   output.multi_modal_data["videos"] = list of (video_tensor, metadata) per chunk
#
# Registered under `"thinkstream_streaming_agent"`.
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Token budget for compress system trigger (matches SFT default in
# scripts.agent_data_v5.config.RECENT_THINKS_TOKEN_BUDGET; we estimate
# tokens at ~1.3× whitespace word count to avoid pulling in tiktoken).
DEFAULT_COMPRESS_TOKEN_THRESHOLD = 3200


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


def _build_visual_window(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    visual_window_chunks: int,
    frames_per_chunk: int,
    chunk_sec: float = 1.0,
) -> Tuple[List[str], Dict[str, Any], int, int]:
    """Build the sliding visual window for chunk N.

    Returns:
      flat_paths:     all frame paths in window order (window_start..N)
      video_metadata: Qwen3-VL metadata dict (fps, frames_indices,
                      total_num_frames) — drives MROPE temporal anchor
      window_start_chunk, window_end_chunk

    Mirrors SFT's pass5_messages.py:140-161 exactly.
    """
    window_start = max(0, chunk_idx - visual_window_chunks + 1)
    window_end = chunk_idx
    flat_paths: List[str] = []
    for c in range(window_start, window_end + 1):
        cf = _chunk_frame_paths(video_path, frames_root, c, frames_per_chunk)
        if not cf:
            return [], {}, window_start, window_end
        flat_paths.extend(cf)
    n_frames = len(flat_paths)
    metadata = {
        "fps": float(frames_per_chunk) / float(chunk_sec),
        "frames_indices": [
            window_start * frames_per_chunk + i for i in range(n_frames)
        ],
        "total_num_frames": (chunk_idx + 1) * frames_per_chunk,
    }
    return flat_paths, metadata, window_start, window_end


# ---------------------------------------------------------------------------
# Recall retriever — keyword overlap over compressed_summaries + recent_thinks.
# ---------------------------------------------------------------------------
def _retrieve_from_memory(
    state_compressed: List[Dict[str, Any]],
    state_recent: List[Dict[str, Any]],
    query_text: str,
    time_range: Optional[Tuple[float, float]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Return a recall_result dict (source/time/text) for inline JSON
    serialisation as <recall_result>...</recall_result> on the NEXT chunk's
    user message. (True intra-chunk multi-turn recall — assistant tool_call
    → user/tool recall_result+frames → assistant answer — is deferred.)
    """
    qtext = (query_text or "").lower()
    keywords = [w for w in re.findall(r"[a-z0-9]+", qtext) if len(w) >= 3]
    if not keywords:
        return {"source": "memory", "time": "", "text": ""}
    candidates: List[Tuple[float, str, str]] = []
    for entry in state_compressed or []:
        text = entry.get("text", "") or ""
        if not text:
            continue
        if time_range is not None:
            tr = entry.get("time_range") or []
            if len(tr) >= 2:
                lo, hi = float(tr[0]), float(tr[1])
                qlo, qhi = float(time_range[0]), float(time_range[1])
                if hi < qlo or lo > qhi:
                    continue
        score = sum(1 for kw in keywords if kw in text.lower())
        if score == 0:
            continue
        tr_str = ""
        if entry.get("time_range"):
            tr_str = f"{entry['time_range'][0]}-{entry['time_range'][1]}s"
        candidates.append((score, tr_str, text))
    for entry in state_recent or []:
        text = entry.get("text", "") or ""
        if not text:
            continue
        chunk = entry.get("chunk", -1)
        if time_range is not None and chunk >= 0:
            qlo, qhi = float(time_range[0]), float(time_range[1])
            if not (qlo <= float(chunk) <= qhi):
                continue
        score = sum(1 for kw in keywords if kw in text.lower())
        if score == 0:
            continue
        candidates.append((score, f"chunk {chunk}", text))
    if not candidates:
        return {"source": "memory", "time": "", "text": "(no relevant past observation)"}
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_k]
    return {
        "source": "memory",
        "time": "; ".join(t for _, t, _ in top if t),
        "text": " | ".join(t for _, _, t in top),
    }


def _estimate_recent_thinks_tokens(recent_thinks: List[Dict[str, Any]]) -> int:
    """Rough word-based token count for the compress trigger threshold.
    Matches SFT's RECENT_THINKS_TOKEN_BUDGET semantics — a coarse upper
    bound is fine since vLLM has its own tokenizer for the actual text."""
    total_words = 0
    for t in recent_thinks or []:
        text = t.get("text") if isinstance(t, dict) else str(t)
        if text:
            total_words += len(text.split())
    return int(total_words * 1.3)  # ~1.3 tokens per word for English


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
        TOOLS_SCHEMA,
        parse_agent_output_v12,
        format_memory_block,
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
            self.compress_token_threshold = int(
                os.environ.get(
                    "THINKSTREAM_COMPRESS_THRESHOLD",
                    str(DEFAULT_COMPRESS_TOKEN_THRESHOLD),
                )
                or DEFAULT_COMPRESS_TOKEN_THRESHOLD
            )
            self.recall_stub_text = str(
                os.environ.get(
                    "THINKSTREAM_RECALL_STUB",
                    "(no relevant past observation found)",
                )
            )

        # -------------------------------------------------------------------
        # Per-chunk user-side text. Mirrors SFT's
        # build_per_timestep_messages_v12 layout so train/RL distributions
        # line up. The video block is rendered separately as a content
        # block of type "video" (Qwen3-VL <|video_pad|> with frames_indices).
        # -------------------------------------------------------------------
        def _build_chunk_user_content(
            self,
            *,
            state: "VideoTrajectoryState",
            chunk_idx: int,
            window_paths: List[str],
            window_metadata: Dict[str, Any],
            window_start_chunk: int,
            window_end_chunk: int,
            question: str,
            ask_chunks: List[int],
            recall_result: Optional[Dict[str, Any]],
            compress_trigger_range: Optional[Tuple[int, int]],
            inter_chunk: bool,
        ) -> List[Dict[str, Any]]:
            """Build the user content list for chunk N. inter_chunk=True
            (compress turn) skips the visual_window — matches SFT shape C.

            v12.12 (2026-05-01): order = memory → visual_window + frames →
            recall_result → user_input. Memory first means the stable
            monotonic prefix lands at the head of the user message and
            the vLLM async server's prefix cache reuses [system + user_q
            + memory_at_t-1] across chunks of one trajectory.
            """
            content: List[Dict[str, Any]] = []

            # Memory block (FIRST — stable monotonic prefix; always emitted,
            # even on compress turn).
            #
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
                "text": f"<memory>\n{mem_text}\n</memory>",
            })

            # Visual window header + video block (cache-miss boundary).
            if not inter_chunk:
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
                    # v12.12: runtime mm_processor_kwargs at video-item level
                    # so qwen-vl-utils.process_vision_info forwards them to
                    # vLLM as smart_resize bounds. Identical values to pass2 +
                    # SFT data_processor + agent_loop inference → student
                    # sees the same visual token sequence at training and
                    # rollout time.
                    try:
                        from scripts.agent_data_v5.config import (
                            RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                        )
                    except ImportError:
                        _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
                    content.append({
                        "type": "video",
                        "video": window_paths,
                        "min_pixels": _RTKW["min_pixels"],
                        "max_pixels": _RTKW["max_pixels"],
                        "video_metadata": window_metadata,
                    })

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

            # User input — either the question (when it fires) or the
            # compress_trigger system event. LAST so the monotonic prefix
            # above stays cache-friendly.
            #
            # v12.12 (2026-05-02): trigger is bare `<compress_trigger/>` —
            # NO range attribute (matches pass3c_samples._compress_sample SFT
            # input format). The model derives the time_range from <memory>
            # and emits it inside the assistant tool_call. The local variable
            # `compress_trigger_range` is still computed by
            # `_check_compress_trigger` so OOD-safety / telemetry code can
            # consult it, but it is NOT exposed to the model in the prompt.
            # When v12.13 RL goes fully model-driven this whole branch is
            # removed (no trigger at all).
            if compress_trigger_range is not None:
                content.append({
                    "type": "text",
                    "text": "\n<compress_trigger/>",
                })
            elif question and ask_chunks and chunk_idx >= min(ask_chunks):
                content.append({
                    "type": "text",
                    "text": f"\n<user_input>{question}</user_input>",
                })

            return content

        async def _execute_recall(
            self, args: Dict[str, Any], state: "VideoTrajectoryState"
        ) -> Dict[str, Any]:
            query = (args.get("query") or args.get("keywords") or
                     args.get("text") or "")
            time_range = args.get("time_range")
            tr_tuple: Optional[Tuple[float, float]] = None
            # Secondary fix (post-review 2026-05-01): recall tool schema
            # in agent_protocol.py:417 emits time_range as "start-end"
            # string (e.g. "10-30"), but earlier eval code passes
            # list/tuple [start, end]. Accept BOTH so the time-range
            # filter actually works regardless of which shape the model
            # learned during SFT.
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
            try:
                return _retrieve_from_memory(
                    state.compressed_summaries or [],
                    state.recent_thinks or [],
                    query_text=query,
                    time_range=tr_tuple,
                )
            except Exception as e:
                logger.warning("recall retrieval failed: %s", e)
                return {"source": "memory", "time": "", "text": "(retrieval error)"}

        def _check_compress_trigger(
            self, state: "VideoTrajectoryState"
        ) -> Optional[Tuple[int, int]]:
            """Return (start_chunk, end_chunk) range to compress, or None.
            Fires when recent_thinks token estimate exceeds threshold."""
            est = _estimate_recent_thinks_tokens(state.recent_thinks)
            if est < self.compress_token_threshold:
                return None
            if not state.recent_thinks:
                return None
            chunks = [
                int(t.get("chunk", -1))
                for t in state.recent_thinks
                if isinstance(t, dict) and t.get("chunk", -1) >= 0
            ]
            if not chunks:
                return None
            return (min(chunks), max(chunks))

        async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":
            metrics: Dict[str, Any] = {}
            request_id = uuid4().hex

            extra_info = kwargs.get("extra_info") or {}
            video_id = extra_info.get("video_id") or extra_info.get("index", "")
            video_path = extra_info.get("video_path", "")
            n_chunks_dataset = int(extra_info.get("n_chunks") or 0)
            n_chunks = min(self.max_chunks, n_chunks_dataset) if n_chunks_dataset else self.max_chunks
            question = extra_info.get("question", "")
            ask_chunks = list(extra_info.get("ask_chunks") or [])

            # ── Initial prompt: [system + user(question)]. Cached on vLLM
            # side; never re-prefilled across chunks.
            initial_messages = list(kwargs["raw_prompt"])
            initial_mm = await self.process_vision_info(initial_messages)
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
                        tools=TOOLS_SCHEMA,
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
                    initial_messages, tools=TOOLS_SCHEMA, images=None,
                    videos=initial_videos if initial_videos else None,
                )
                initial_prompt_ids = list(with_prefix)

            response_ids: List[int] = []
            response_mask: List[int] = []
            response_logprobs: List[float] = []
            any_logprobs_returned = False
            chunk_asst_spans: List[Tuple[int, int]] = []
            chunk_kinds: List[str] = []
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
            # chunk that injected a video block. NOT a flat per-frame
            # PIL list, so memory cost is O(n_chunks) tensors not
            # O(n_chunks × window × fpc) PIL images. (P1.11 fix.)
            accumulated_videos: List[Any] = list(initial_videos)

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
                # chunks (inter_chunk=True, no visual_window).
                compress_range = self._check_compress_trigger(state)
                inter_chunk = compress_range is not None

                # ── Sliding visual window (skipped for compress turns).
                window_paths: List[str] = []
                window_metadata: Dict[str, Any] = {}
                window_start_chunk = chunk_idx
                window_end_chunk = chunk_idx
                if not inter_chunk and self.frames_root and video_path:
                    window_paths, window_metadata, window_start_chunk, window_end_chunk = (
                        _build_visual_window(
                            video_path, self.frames_root, chunk_idx,
                            visual_window_chunks=self.visual_window_chunks,
                            frames_per_chunk=self.frames_per_chunk,
                            chunk_sec=self.chunk_sec,
                        )
                    )
                visual_injected = bool(window_paths) and not inter_chunk

                # ── Build user content + chunk_messages.
                user_content = self._build_chunk_user_content(
                    state=state,
                    chunk_idx=chunk_idx,
                    window_paths=window_paths,
                    window_metadata=window_metadata,
                    window_start_chunk=window_start_chunk,
                    window_end_chunk=window_end_chunk,
                    question=question,
                    ask_chunks=ask_chunks,
                    recall_result=recall_result_for_next,
                    compress_trigger_range=compress_range,
                    inter_chunk=inter_chunk,
                )
                recall_result_for_next = None

                chunk_messages = list(initial_messages) + [
                    {"role": "user", "content": user_content},
                ]

                # Resolve images/videos for THIS chunk (the new user msg
                # only — initial_messages were already processed once).
                chunk_extra_mm = await self.process_vision_info([chunk_messages[-1]])
                chunk_images = chunk_extra_mm.get("images") or []
                chunk_videos = chunk_extra_mm.get("videos") or []

                chunk_prompt_ids = await self.apply_chat_template(
                    chunk_messages,
                    tools=TOOLS_SCHEMA,
                    images=chunk_images if chunk_images else None,
                    videos=(initial_videos + chunk_videos) if chunk_videos else (
                        initial_videos if initial_videos else None
                    ),
                )

                # Prompt-budget guard.
                if len(chunk_prompt_ids) + self.response_length >= self.prompt_length:
                    break
                user_block_len = len(chunk_prompt_ids) - len(initial_prompt_ids)
                if user_block_len < 0:
                    break
                if len(response_mask) + user_block_len + 1 >= self.response_length:
                    break

                # ── Generate INDEPENDENTLY. v12.12: KV cache hits
                # initial_prompt_ids + memory_at_t-1 + queries_at_t-1
                # (memory now placed FIRST in user content). Visual window
                # remains a cache miss because the sliding window mutates
                # both ends each step (drops oldest frames, appends newest).
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=chunk_prompt_ids,
                        sampling_params=sampling_params,
                        image_data=chunk_images if chunk_images else None,
                        video_data=(initial_videos + chunk_videos) if chunk_videos else (
                            initial_videos if initial_videos else None
                        ),
                    )
                assistant_ids = list(output.token_ids)
                if not assistant_ids:
                    break

                # ── Stitch into verl's expected shape.
                user_block_ids = chunk_prompt_ids[len(initial_prompt_ids):]
                response_ids.extend(user_block_ids)
                response_mask.extend([0] * len(user_block_ids))
                response_logprobs.extend([0.0] * len(user_block_ids))

                asst_start = len(response_ids)
                response_ids.extend(assistant_ids)
                response_mask.extend([1] * len(assistant_ids))
                if output.log_probs and len(output.log_probs) == len(assistant_ids):
                    response_logprobs.extend(list(output.log_probs))
                    any_logprobs_returned = True
                else:
                    response_logprobs.extend([0.0] * len(assistant_ids))
                asst_end = len(response_ids)
                chunk_asst_spans.append((asst_start, asst_end))
                # P1.7: -1 for compress (system inter-chunk turn, not
                # mapped to any video chunk for action gold lookup).
                chunk_video_indices.append(-1 if inter_chunk else chunk_idx)

                if visual_injected:
                    accumulated_videos.extend(chunk_videos)
                    n_chunks_with_frames += 1
                elif inter_chunk:
                    n_chunks_compress_inter += 1
                else:
                    n_chunks_text_only += 1

                num_assistant_turns += 1

                # ── Decode + parse + state evolution.
                response_text = self.tokenizer.decode(
                    assistant_ids, skip_special_tokens=True,
                )
                parsed = parse_agent_output_v12(response_text)
                kind = parsed.get("kind", "unknown")
                chunk_kinds.append(kind)
                chunk_asst_texts.append(response_text)

                # default_v12_update_state advances chunk_idx by +1 on
                # EVERY turn — including compress. For inter-chunk
                # compress turns we DON'T want to skip ahead in the
                # video timeline, so we revert state.chunk_idx after.
                pre_chunk_idx = state.chunk_idx
                state = default_v12_update_state(state, response_text, chunk_idx)
                if inter_chunk:
                    # System event — don't consume a video chunk.
                    state.chunk_idx = pre_chunk_idx
                else:
                    # Append think to recent_thinks for next turn's memory.
                    think_text = parsed.get("think") or ""
                    if think_text and kind in ("answer", "recall", "unknown"):
                        state.recent_thinks.append({
                            "chunk": chunk_idx, "text": think_text,
                        })

                if kind == "recall":
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    recall_result_for_next = await self._execute_recall(args, state)

                if not state.is_active or state.is_done:
                    break
                if len(response_mask) >= self.response_length:
                    break

                # Advance video chunk pointer ONLY on non-compress turns.
                if not inter_chunk:
                    chunk_idx += 1
                # On compress turn we re-enter the loop at the same
                # chunk_idx; default_v12_update_state cleared the
                # compressed range from recent_thinks already, so the
                # next iteration's _check_compress_trigger should return
                # None (loop progresses).

            num_turns = num_assistant_turns + 1

            multi_modal_data: Dict[str, Any] = {}
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
            output_obj.extra_fields.update({
                "turn_scores": [],
                "tool_rewards": [],
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
                # Per-chunk metadata for compute_score's per-chunk action
                # shaping (folded into total `score` via GDPO α-mix).
                "ts_chunk_asst_spans": chunk_asst_spans,
                "ts_chunk_kinds": chunk_kinds,
                "ts_chunk_asst_texts": chunk_asst_texts,
                # P1.7: video chunk_idx per assistant turn (-1 for compress
                # inter-chunk turns). compute_score keys gold_action_per_chunk
                # by this, not by enumerate(chunk_kinds).
                "ts_chunk_video_indices": chunk_video_indices,
            })
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
