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
# constructed every turn:
#
#     prompt_chunk_N = [
#       <|im_start|>system\n{SYSTEM_PROMPT_V12}\n<|im_end|>     ← common prefix
#       <|im_start|>user\n{question}\n<|im_end|>                ← common prefix
#       <|im_start|>user\n
#         {visual_window: chunks[max(0,N-15)..N]'s frames}      ← turn-specific
#         {memory: state.compressed + state.recent_thinks}       ← turn-specific
#         {optional <recall_result>...}                          ← turn-specific
#         {optional <query>...}                                  ← turn-specific
#       <|im_end|>
#       <|im_start|>assistant\n
#     ]
#
# This matches what SFT trains on (build_per_timestep_messages_v12) — each
# chunk is its own conditioning context, no monotonic accumulation.
#
# KV CACHE BEHAVIOUR:
#   - The [system + user_q] prefix is byte-identical across all chunk turns
#     of one trajectory → vLLM's async server prefix cache hits it once and
#     serves it for free for the rest of the trajectory.
#   - The visual_window + memory portion changes every turn (sliding window
#     means chunk 0's frames drop out at turn 16) → that suffix is a cache
#     miss. We pay one prefill of ~16 chunks worth of vis tokens per turn
#     instead of monotonically growing the cache.
#   - Chunk 0's frames are NOT in turn 16+'s prompt → they're truly out of
#     the KV cache for those generates. This is the user's explicit ask.
#
# verl INTEGRATION:
#   verl's AgentLoopOutput expects ONE prompt_ids + ONE response_ids. We
#   stitch the per-chunk independent generates into that shape by:
#     output.prompt_ids   = [system + user_q]   (the COMMON prefix only)
#     output.response_ids = [user_block_0 + asst_0 + user_block_1 + asst_1 + ...]
#     output.response_mask = [0...0 | 1...1 | 0...0 | 1...1 | ...]
#                            ^user blocks  ^asst   ^user blocks ^asst
#   So at training time the actor sees ONE long sequence: system + user_q
#   followed by the concatenated chunk turns. This DIFFERS from rollout
#   topology (where each turn was an independent forward in vLLM) — at
#   training time the actor's attention DOES see prior chunks' user_blocks
#   when processing later chunks' tokens. This is a known SFT↔RL drift
#   inherent to mapping N independent rollout turns onto one verl sample.
#   Mitigations on the trainer side (per-chunk forward via attention mask
#   reset) can be added later — for now we accept the drift, since the
#   user-block portion has mask=0 and only assistant tokens contribute to
#   the loss.
#
# Registered under `"thinkstream_streaming_agent"` — that name is set by
# CustomRLHFDataset.__getitem__ in thinkstream.py so verl picks this loop
# up automatically for our rows.
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


# ---------------------------------------------------------------------------
# Frame loading — pure helpers, no verl dependency.
# ---------------------------------------------------------------------------
def _resolve_frame_dir(video_path: str, frames_root: str) -> Optional[Path]:
    """Find the pre-extracted frame directory for a video (matches the
    layout pass1a writes: frames_root/<video_stem>/frame_*.jpg)."""
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


def _load_chunk_frames(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    frames_per_chunk: int = 2,
) -> List[Any]:
    """Load `frames_per_chunk` PIL Images for chunk `chunk_idx`.
    Frame indexing: frame_dir / "frame_{i:06d}.jpg", chunk_idx maps to
    indices [chunk_idx*fpc, (chunk_idx+1)*fpc). Returns [] if the dir is
    missing or doesn't have enough frames — caller falls back to text-only.
    """
    try:
        from PIL import Image
    except ImportError:
        return []
    frame_dir = _resolve_frame_dir(video_path, frames_root)
    if frame_dir is None:
        return []
    start = chunk_idx * frames_per_chunk
    images: List[Any] = []
    for offset in range(frames_per_chunk):
        idx = start + offset
        for name in (f"frame_{idx:06d}.jpg", f"frame_{idx:05d}.jpg",
                     f"frame_{idx:04d}.jpg", f"frame_{idx}.jpg"):
            p = frame_dir / name
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception:
                    pass
                break
    if len(images) != frames_per_chunk:
        return []
    return images


def _load_visual_window(
    video_path: str,
    frames_root: str,
    chunk_idx: int,
    visual_window_chunks: int,
    frames_per_chunk: int,
) -> Tuple[List[Any], int, int]:
    """Load the sliding visual window covering chunks [start..chunk_idx]
    where start = max(0, chunk_idx - visual_window_chunks + 1).
    Returns (flat_frames, window_start_chunk, window_end_chunk).
    """
    start = max(0, chunk_idx - visual_window_chunks + 1)
    end = chunk_idx
    flat: List[Any] = []
    for c in range(start, end + 1):
        flat.extend(_load_chunk_frames(video_path, frames_root, c, frames_per_chunk))
    return flat, start, end


# ---------------------------------------------------------------------------
# Recall retriever — keyword overlap over compressed_summaries + recent_thinks.
# ---------------------------------------------------------------------------
def _retrieve_from_memory(
    state_compressed: List[Dict[str, Any]],
    state_recent: List[Dict[str, Any]],
    query_text: str,
    time_range: Optional[Tuple[float, float]] = None,
    top_k: int = 3,
) -> str:
    qtext = (query_text or "").lower()
    keywords = [w for w in re.findall(r"[a-z0-9]+", qtext) if len(w) >= 3]
    if not keywords:
        return "(empty query — nothing to retrieve)"
    candidates: List[Tuple[float, str]] = []
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
            tr_str = f"[{entry['time_range'][0]}-{entry['time_range'][1]}s] "
        candidates.append((score, f"{tr_str}{text}"))
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
        candidates.append((score, f"[chunk {chunk}] {text}"))
    if not candidates:
        return "(no relevant past observation found for these keywords)"
    candidates.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(text for _, text in candidates[:top_k])


# ---------------------------------------------------------------------------
# verl-side registration. Wrapped in a function so the module is importable
# in dev environments that lack verl/ray/torch.
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

    @register("thinkstream_streaming_agent")
    class ThinkStreamStreamingAgentLoop(AgentLoopBase):
        """MemAgent-style chunk-level rollout for streaming video.

        Each chunk = one independent vLLM generate request with a freshly
        constructed prompt that includes a sliding visual window. See the
        file header for the KV-cache reasoning.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            mt = self.rollout_config.multi_turn
            self.max_chunks = int(getattr(mt, "max_turns", 0) or 360)
            self.frames_root = str(
                getattr(mt, "frames_root", "") or
                os.environ.get("THINKSTREAM_FRAMES_ROOT", "")
            )
            self.frames_per_chunk = int(getattr(mt, "frames_per_chunk", 2) or 2)
            # Sliding visual window — must match SFT's VISUAL_WINDOW_CHUNKS
            # (16 in scripts.agent_data_v5.config) so RL's chunk N sees the
            # same frame distribution that SFT trained chunk N on.
            self.visual_window_chunks = int(
                getattr(mt, "visual_window_chunks", 16) or 16
            )
            self.recall_stub_text = getattr(
                mt, "recall_stub_text",
                "(no relevant past observation found)",
            )

        # -------------------------------------------------------------------
        # Per-chunk user-side text. Mirrors SFT's
        # build_per_timestep_messages_v12 layout so train/RL distributions
        # line up. Visual frames are inserted as image content blocks
        # ahead of this text in the same user message.
        # -------------------------------------------------------------------
        def _render_chunk_user_text(
            self,
            *,
            state: "VideoTrajectoryState",
            chunk_idx: int,
            window_start_chunk: int,
            window_end_chunk: int,
            question: str,
            ask_chunks: List[int],
            recall_result: Optional[str],
            compress_trigger: bool,
            visual_injected: bool,
        ) -> str:
            parts: List[str] = []
            marker = (
                f"<chunk_update idx={chunk_idx} t={chunk_idx}-{chunk_idx + 1}s "
                f"visual_window={window_start_chunk}-{window_end_chunk}/>"
            )
            if not visual_injected:
                marker += " (no_frames — frames_root unset or files missing)"
            parts.append(marker)
            try:
                mem_text = format_memory_block({
                    "compressed_summaries": state.compressed_summaries,
                    "recent_thinks": state.recent_thinks,
                })
            except Exception:
                mem_text = ""
            if mem_text:
                parts.append(f"<memory>\n{mem_text}\n</memory>")
            if question and ask_chunks and chunk_idx >= min(ask_chunks):
                parts.append(f"<query>{question}</query>")
            if recall_result is not None:
                parts.append(f"<recall_result>{recall_result}</recall_result>")
            if compress_trigger:
                parts.append(
                    "<compress_trigger range='recent'/>  "
                    "(memory token-budget exceeded; emit a compress tool_call this turn)"
                )
            return "\n".join(parts)

        async def _execute_recall(
            self, args: Dict[str, Any], state: "VideoTrajectoryState"
        ) -> str:
            query = (args.get("query") or args.get("keywords") or
                     args.get("text") or "")
            time_range = args.get("time_range")
            tr_tuple: Optional[Tuple[float, float]] = None
            if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
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
                    top_k=3,
                )
            except Exception as e:
                logger.warning("recall retrieval failed: %s", e)
                return self.recall_stub_text

        async def _execute_compress(
            self, args: Dict[str, Any], state: "VideoTrajectoryState"
        ) -> None:
            return None

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

            # ── COMMON PREFIX: [system + user(question)]. Tokenized once.
            # Every chunk's generate request shares this byte-identical
            # prefix, so vLLM's prefix cache serves it for the rest of the
            # trajectory at zero prefill cost.
            initial_messages = list(kwargs["raw_prompt"])
            initial_mm = await self.process_vision_info(initial_messages)
            initial_videos: List[Any] = list(initial_mm.get("videos") or [])
            initial_prompt_ids = await self.apply_chat_template(
                initial_messages,
                tools=TOOLS_SCHEMA,
                images=None,
                videos=initial_videos if initial_videos else None,
            )

            # Accumulators: everything AFTER initial_prompt_ids that verl
            # will treat as response_ids. user-block tokens get mask=0,
            # assistant tokens get mask=1.
            response_ids: List[int] = []
            response_mask: List[int] = []
            response_logprobs: List[float] = []
            any_logprobs_returned = False

            state = VideoTrajectoryState(video_uid=str(video_id), chunk_idx=0)
            recall_result_for_next: Optional[str] = None
            compress_trigger_for_next = False
            num_assistant_turns = 0
            n_chunks_with_frames = 0
            n_chunks_text_only = 0
            window_frames: List[Any] = []  # for multi_modal_data fallback

            for chunk_idx in range(n_chunks):
                if not state.is_active:
                    break

                # ── Sliding visual window: load frames for chunks
                # [max(0, chunk_idx - visual_window_chunks + 1) .. chunk_idx].
                # chunk 0's frames roll out of this window once chunk_idx
                # exceeds visual_window_chunks - 1. Those frames appear in
                # NO subsequent chunk's prompt → not in vLLM's KV for
                # those turns.
                window_frames = []
                window_start_chunk = chunk_idx
                window_end_chunk = chunk_idx
                if self.frames_root and video_path:
                    window_frames, window_start_chunk, window_end_chunk = _load_visual_window(
                        video_path, self.frames_root, chunk_idx,
                        visual_window_chunks=self.visual_window_chunks,
                        frames_per_chunk=self.frames_per_chunk,
                    )
                visual_injected = bool(window_frames)
                if visual_injected:
                    n_chunks_with_frames += 1
                else:
                    n_chunks_text_only += 1

                # ── Build chunk N's INDEPENDENT user message.
                user_text = self._render_chunk_user_text(
                    state=state,
                    chunk_idx=chunk_idx,
                    window_start_chunk=window_start_chunk,
                    window_end_chunk=window_end_chunk,
                    question=question,
                    ask_chunks=ask_chunks,
                    recall_result=recall_result_for_next,
                    compress_trigger=compress_trigger_for_next,
                    visual_injected=visual_injected,
                )
                recall_result_for_next = None
                compress_trigger_for_next = False

                if visual_injected:
                    user_content: List[Dict[str, Any]] = (
                        [{"type": "image"} for _ in window_frames]
                        + [{"type": "text", "text": user_text}]
                    )
                else:
                    user_content = [{"type": "text", "text": user_text}]

                chunk_messages = list(initial_messages) + [
                    {"role": "user", "content": user_content},
                ]

                # ── Tokenize the FULL chunk prompt (system + user_q + this
                # turn's user block). Independent each turn — no carry-over
                # of prior turns' user blocks. vLLM matches the
                # initial_prompt_ids prefix in its KV cache.
                chunk_prompt_ids = await self.apply_chat_template(
                    chunk_messages,
                    tools=TOOLS_SCHEMA,
                    images=window_frames if window_frames else None,
                    videos=initial_videos if initial_videos else None,
                )

                # Prompt-budget guard.
                if len(chunk_prompt_ids) + self.response_length >= self.prompt_length:
                    break
                user_block_len = len(chunk_prompt_ids) - len(initial_prompt_ids)
                if user_block_len < 0:
                    # apply_chat_template renormalised something unexpected — bail.
                    break
                if len(response_mask) + user_block_len + 1 >= self.response_length:
                    break

                # ── Generate INDEPENDENTLY. KV cache hits initial_prompt_ids
                # prefix; visual_window + memory portion is a cache miss
                # and gets prefilled fresh.
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=chunk_prompt_ids,
                        sampling_params=sampling_params,
                        image_data=window_frames if window_frames else None,
                        video_data=initial_videos if initial_videos else None,
                    )
                assistant_ids = list(output.token_ids)
                if not assistant_ids:
                    break

                # ── Stitch into verl's expected (prompt + response) shape.
                # The user-side block (everything in chunk_prompt_ids after
                # initial_prompt_ids) gets mask=0 — verl won't compute
                # actor loss on these. The assistant tokens get mask=1.
                user_block_ids = chunk_prompt_ids[len(initial_prompt_ids):]
                response_ids.extend(user_block_ids)
                response_mask.extend([0] * len(user_block_ids))
                response_logprobs.extend([0.0] * len(user_block_ids))

                response_ids.extend(assistant_ids)
                response_mask.extend([1] * len(assistant_ids))
                if output.log_probs and len(output.log_probs) == len(assistant_ids):
                    response_logprobs.extend(list(output.log_probs))
                    any_logprobs_returned = True
                else:
                    response_logprobs.extend([0.0] * len(assistant_ids))

                num_assistant_turns += 1

                # ── Decode + parse + state evolution.
                response_text = self.tokenizer.decode(
                    assistant_ids, skip_special_tokens=True,
                )
                parsed = parse_agent_output_v12(response_text)
                kind = parsed.get("kind", "unknown")

                state = default_v12_update_state(state, response_text, chunk_idx)
                # Append THIS turn's think to recent_thinks so the NEXT
                # chunk's memory snapshot includes it. default_v12_update_state
                # is a pure function shared with slyme — it intentionally
                # doesn't do this so we do it here.
                think_text = parsed.get("think") or ""
                if think_text and kind in ("answer", "recall", "unknown"):
                    state.recent_thinks.append({
                        "chunk": chunk_idx, "text": think_text,
                    })

                if kind == "recall":
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    recall_result_for_next = await self._execute_recall(args, state)
                elif kind == "compress":
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    await self._execute_compress(args, state)

                if not state.is_active or state.is_done:
                    break
                if len(response_mask) >= self.response_length:
                    break

            num_turns = num_assistant_turns + 1  # +1 for initial system+user

            # multi_modal_data: report the LATEST chunk's window frames.
            # Per-chunk slot reconstruction at training time isn't
            # supported by verl's actor forward — that's a known SFT-vs-RL
            # drift documented in the file header.
            multi_modal_data = {}
            if window_frames:
                multi_modal_data["images"] = window_frames
            if initial_videos:
                multi_modal_data["videos"] = initial_videos

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
                "ts_answer_chunk": float(
                    state.final_answer_chunk
                    if state.final_answer_chunk is not None else -1
                ),
                "ts_final_answer": state.final_answer or "",
            })
            return output_obj

    return ThinkStreamStreamingAgentLoop


try:
    _register_streaming_agent_loop()
except Exception as e:  # noqa: BLE001 — local dev tooling shouldn't crash.
    logger.debug("ThinkStreamStreamingAgentLoop not registered: %s", e)
