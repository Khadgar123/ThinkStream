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
# verl's stock multi-turn machinery is tool-call shaped (assistant emits
# <tool_call>, framework executes tool, returns <tool_response>, model
# continues). Streaming video doesn't fit that abstraction: each "turn" is
# the arrival of a NEW 1-second chunk, not a tool result. We model it as a
# custom AgentLoop that:
#
#   - Keeps a `VideoTrajectoryState` across chunks (memory, pending Qs,
#     final answer flag — same dataclass slyme uses)
#   - Each chunk:
#       (a) load the chunk's `frames_per_chunk` frames from disk
#       (b) render `<chunk_update>` + <memory> + <query> + <recall_result>
#       (c) append as a user turn (mask=0) with image content markers
#       (d) call server_manager.generate with ACCUMULATED image_data
#       (e) append assistant turn (mask=1)
#       (f) parse v12 protocol output and advance state
#   - Stops on: explicit non-empty <answer>, max_chunks, response budget,
#     prompt budget
#
# KV CACHE STRATEGY (important for cost/throughput):
#   We use *monotonic visual prefix accumulation*. Each chunk only APPENDS
#   new frames to the visual data — never rewrites or drops historical
#   frames. This means:
#     - prompt_ids prefix is identical across turns of the same trajectory
#     - vLLM async server's prefix cache hits the entire prefix
#     - per-turn prefill cost is O(new chunk tokens), not O(all prior tokens)
#   The trade-off: SFT trains on a sliding 16-chunk window, RL sees a
#   monotonic window. Distribution drifts after ~16 chunks. Acceptable for
#   v12.6 (acknowledged in the design doc); a future iteration can do
#   periodic prefix re-form to match SFT exactly at the cost of cache miss.
#
# Visual-budget guard: when the cumulative visual+text token count
# approaches `prompt_length`, new frames stop being injected (text-only
# chunk updates continue). This prevents prompt overflow while keeping the
# loop running to max_chunks.
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
# Frame loading — pure helpers, no verl dependency. Mirrors the layout that
# scripts/agent_data_v5/pass1a_evidence.py + thinkstream.data.stream_data_processor
# write: frames_root/<video_stem>/frame_*.jpg, sorted lexically = chronological.
# ---------------------------------------------------------------------------
def _resolve_frame_dir(video_path: str, frames_root: str) -> Optional[Path]:
    """Find the pre-extracted frame directory for a video.

    Looks under `frames_root/<video_stem>` first (flat layout used by
    pass1a). Falls back to `frames_root/<video_path_stem>` for the
    video-root-relative layout used by some splits.
    """
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
    # Last resort: check if frames_root IS the dir (single-video debug).
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

    Frame indexing matches pass1a's emit order:
        frame_dir / "frame_{i:06d}.jpg" for i in [0, n_frames)
    where chunk_idx maps to frames [chunk_idx*frames_per_chunk,
    (chunk_idx+1)*frames_per_chunk).
    Returns [] if the dir is missing or doesn't have enough frames — the
    loop falls back to text-only for that chunk.
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
        # Try common naming conventions in order.
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
        # Don't half-inject — Qwen3-VL chat template counts <|image_pad|>
        # against the image_data list size, mismatch is a hard error.
        return []
    return images


# ---------------------------------------------------------------------------
# Recall retriever — substring/keyword match over compressed_summaries +
# recent_thinks. Pure function so tests can hit it without verl/torch.
# ---------------------------------------------------------------------------
def _retrieve_from_memory(
    state_compressed: List[Dict[str, Any]],
    state_recent: List[Dict[str, Any]],
    query_text: str,
    time_range: Optional[Tuple[float, float]] = None,
    top_k: int = 3,
) -> str:
    """Retrieve up to `top_k` memory entries matching `query_text` and the
    optional `time_range` (seconds). Returns a serialized text block ready
    for `<recall_result>...</recall_result>`.

    Scoring: count of query keywords (lowercased, length≥3) appearing in
    the candidate's text. Time-range filter is intersection check on the
    candidate's chunk range. Returns the keyword-richest entries.
    """
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
            # 1s/chunk convention; drop entries outside requested range.
            if not (qlo <= float(chunk) <= qhi):
                continue
        score = sum(1 for kw in keywords if kw in text.lower())
        if score == 0:
            continue
        candidates.append((score, f"[chunk {chunk}] {text}"))

    if not candidates:
        return "(no relevant past observation found for these keywords)"
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [text for _, text in candidates[:top_k]]
    return "\n".join(top)


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
        """Chunk-level rollout. Each chunk = one user turn + one assistant turn.

        Model output shape per chunk:
            <think>40-60 tokens describing only what is newly visible</think>
            (then ONE of)
            <tool_call>{...}</tool_call>     # recall or compress
            <answer>response text</answer>   # break — final answer
            <answer></answer>                # silent, advance to next chunk

        Concatenated `response_ids` (assistant tokens mask=1, user
        chunk-update tokens mask=0) is decoded with skip_special_tokens=True
        and fed to compute_score, which splits on `<think>` to recover
        per-chunk outputs.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            mt = self.rollout_config.multi_turn
            self.max_chunks = int(getattr(mt, "max_turns", 0) or 360)
            # ThinkStream-specific knobs (see configs/thinkstream_grpo.yaml):
            self.frames_root = str(
                getattr(mt, "frames_root", "") or
                os.environ.get("THINKSTREAM_FRAMES_ROOT", "")
            )
            self.frames_per_chunk = int(getattr(mt, "frames_per_chunk", 2) or 2)
            # Visual budget — once cumulative visual tokens approach the
            # prompt budget, stop injecting frames (text-only chunk
            # updates continue). 200 ≈ Qwen3-VL vis tokens / image at
            # default spatial merge, conservative.
            self.vis_tokens_per_frame = int(getattr(mt, "vis_tokens_per_frame", 200) or 200)
            self.recall_stub_text = getattr(
                mt, "recall_stub_text",
                "(no relevant past observation found)",
            )

        # -------------------------------------------------------------------
        # Chunk → user-side text. Reuses the same protocol the SFT path uses
        # (see thinkstream/sft/data_processor.py:build_per_timestep_messages_v12).
        # -------------------------------------------------------------------
        def _render_chunk_user_block(
            self,
            *,
            state: "VideoTrajectoryState",
            chunk_idx: int,
            question: str,
            ask_chunks: List[int],
            recall_result: Optional[str],
            compress_trigger: bool,
            visual_injected: bool,
        ) -> str:
            parts: List[str] = []
            # Text marker — even when frames are injected, the marker
            # tells the model which chunk we're on (frames alone don't
            # carry an absolute timestamp at Qwen3-VL's resolution).
            marker = f"<chunk_update idx={chunk_idx} t={chunk_idx}-{chunk_idx + 1}s/>"
            if not visual_injected:
                marker += " (no_frames — visual budget exhausted)"
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

        # -------------------------------------------------------------------
        # Tool execution. Recall is a real keyword retriever over the
        # state's memory. Compress is purely state-side (handled by
        # default_v12_update_state).
        # -------------------------------------------------------------------
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

            # ── Initial prompt: system + user(question), no frames yet.
            messages = list(kwargs["raw_prompt"])
            initial_mm = await self.process_vision_info(messages)
            accumulated_images: List[Any] = list(initial_mm.get("images") or [])
            accumulated_videos: List[Any] = list(initial_mm.get("videos") or [])

            prompt_ids = await self.apply_chat_template(
                messages,
                tools=TOOLS_SCHEMA,
                images=accumulated_images if accumulated_images else None,
                videos=accumulated_videos if accumulated_videos else None,
            )

            initial_prompt_len = len(prompt_ids)
            response_mask: List[int] = []
            response_logprobs: List[float] = []
            any_logprobs_returned = False

            state = VideoTrajectoryState(video_uid=str(video_id), chunk_idx=0)
            recall_result_for_next: Optional[str] = None
            compress_trigger_for_next = False
            num_assistant_turns = 0
            n_chunks_with_frames = 0
            n_chunks_text_only = 0

            for chunk_idx in range(n_chunks):
                if not state.is_active:
                    break

                # ── Frame loading + visual budget gate.
                chunk_frames: List[Any] = []
                if self.frames_root and video_path:
                    chunk_frames = _load_chunk_frames(
                        video_path, self.frames_root, chunk_idx,
                        frames_per_chunk=self.frames_per_chunk,
                    )
                # Visual-budget guard: predict if injecting these frames
                # would push the prompt over budget. Reserve response_length
                # tokens for assistant generation + ~200 tokens slack.
                projected_visual_tokens = (
                    len(chunk_frames) * self.vis_tokens_per_frame
                )
                visual_injected = bool(chunk_frames) and (
                    len(prompt_ids) + projected_visual_tokens
                    + self.response_length + 200
                    < self.prompt_length
                )
                if not visual_injected:
                    chunk_frames = []
                    n_chunks_text_only += 1
                else:
                    n_chunks_with_frames += 1

                # ── Render chunk user block (text + optional image markers).
                user_text = self._render_chunk_user_block(
                    state=state,
                    chunk_idx=chunk_idx,
                    question=question,
                    ask_chunks=ask_chunks,
                    recall_result=recall_result_for_next,
                    compress_trigger=compress_trigger_for_next,
                    visual_injected=visual_injected,
                )
                recall_result_for_next = None
                compress_trigger_for_next = False

                if visual_injected:
                    # Multi-modal user content: image markers first, then
                    # text. Qwen3-VL chat template renders each {"type":"image"}
                    # as <|vision_start|><|image_pad|><|vision_end|>; image_data
                    # passed to apply_chat_template fills those slots.
                    user_content: List[Dict[str, Any]] = (
                        [{"type": "image"} for _ in chunk_frames]
                        + [{"type": "text", "text": user_text}]
                    )
                    user_msg = [{"role": "user", "content": user_content}]
                    user_ids = await self.apply_chat_template(
                        user_msg,
                        images=chunk_frames,
                        remove_system_prompt=True,
                    )
                    accumulated_images.extend(chunk_frames)
                else:
                    user_msg = [{"role": "user", "content": user_text}]
                    user_ids = await self.apply_chat_template(
                        user_msg, remove_system_prompt=True,
                    )

                # Prompt budget: bail before appending if we'd starve the
                # assistant generation.
                if (
                    len(prompt_ids) + len(user_ids) + self.response_length
                    >= self.prompt_length
                ):
                    break
                if len(response_mask) + len(user_ids) >= self.response_length:
                    break

                prompt_ids = prompt_ids + user_ids
                response_mask.extend([0] * len(user_ids))
                # Keep response_logprobs aligned to response_mask in lockstep —
                # placeholder zeros for user-block tokens, real logprobs (if
                # vLLM returns them) for assistant tokens. Avoids the sparse-
                # array bug where a missing logprobs in turn 1 silently
                # misaligned all subsequent turns.
                response_logprobs.extend([0.0] * len(user_ids))

                # ── Generate assistant turn.
                # Pass ACCUMULATED images so vLLM aligns image tokens in the
                # cached prefix. KV cache hits the prefix exactly because
                # prompt_ids are append-only.
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=accumulated_images if accumulated_images else None,
                        video_data=accumulated_videos if accumulated_videos else None,
                    )
                assistant_ids = list(output.token_ids)
                if not assistant_ids:
                    break
                prompt_ids = prompt_ids + assistant_ids
                response_mask.extend([1] * len(assistant_ids))
                if output.log_probs and len(output.log_probs) == len(assistant_ids):
                    response_logprobs.extend(list(output.log_probs))
                    any_logprobs_returned = True
                else:
                    response_logprobs.extend([0.0] * len(assistant_ids))
                num_assistant_turns += 1

                # ── Decode + parse.
                response_text = self.tokenizer.decode(
                    assistant_ids, skip_special_tokens=True,
                )
                parsed = parse_agent_output_v12(response_text)
                kind = parsed.get("kind", "unknown")

                # ── State transition (advance chunk_idx, mark final answer).
                # default_v12_update_state intentionally does NOT add the
                # think to recent_thinks (pure function shared with slyme
                # path), so we do it here per-loop. This keeps memory
                # rendering working between turns. Add it BEFORE compress
                # filtering would have run, so compress-on-this-turn
                # operates on the prior memory only — matches SFT shape
                # where compress summarizes already-existing recent_thinks.
                state = default_v12_update_state(state, response_text, chunk_idx)
                think_text = parsed.get("think") or ""
                if think_text and kind in ("answer", "recall", "unknown"):
                    # Don't append on compress turns — those don't carry a
                    # new observation think.
                    state.recent_thinks.append({
                        "chunk": chunk_idx, "text": think_text,
                    })

                # ── Tool execution: recall fills next chunk's user block.
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

            # +1 to count the initial system+user turn.
            num_turns = num_assistant_turns + 1
            response_ids_full = prompt_ids[initial_prompt_len:]
            multi_modal_data = {}
            if accumulated_images:
                multi_modal_data["images"] = accumulated_images
            if accumulated_videos:
                multi_modal_data["videos"] = accumulated_videos

            output_obj = AgentLoopOutput(
                prompt_ids=prompt_ids[:initial_prompt_len],
                response_ids=response_ids_full[: self.response_length],
                response_mask=response_mask[: self.response_length],
                # Only surface logprobs if vLLM actually returned them at
                # least once. Otherwise return None so verl recomputes
                # logprobs from the actor instead of trusting all-zero
                # placeholders.
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
                # ThinkStream-specific telemetry — wandb picks these up.
                "ts_n_recall": float(state.n_recall_calls),
                "ts_n_compress": float(state.n_compress_calls),
                "ts_chunks_used": float(num_assistant_turns),
                "ts_chunks_with_frames": float(n_chunks_with_frames),
                "ts_chunks_text_only": float(n_chunks_text_only),
                # Authoritative answer chunk: -1 if the rollout never
                # emitted a non-empty <answer>. compute_score reads this
                # to compute timing reward instead of inferring from
                # num_turns (which over-counts silent turns).
                "ts_answer_chunk": float(
                    state.final_answer_chunk
                    if state.final_answer_chunk is not None else -1
                ),
                "ts_final_answer": state.final_answer or "",
            })
            return output_obj

    return ThinkStreamStreamingAgentLoop


# Trigger registration on import. verl's main_ppo imports recipe modules
# eagerly (via custom_cls.path / custom_reward_function.path), so this
# runs in every Ray worker on first agent_name dispatch.
try:
    _register_streaming_agent_loop()
except Exception as e:  # noqa: BLE001 — local dev tooling shouldn't crash.
    logger.debug("ThinkStreamStreamingAgentLoop not registered: %s", e)
