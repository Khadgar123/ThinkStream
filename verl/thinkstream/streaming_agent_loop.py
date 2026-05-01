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
# This is the missing piece — verl's stock multi-turn machinery is
# tool-call shaped (assistant emits <tool_call>, framework executes tool,
# returns <tool_response>, model continues). Streaming video doesn't fit
# that abstraction: each "turn" is the arrival of a NEW 1-second chunk,
# not a tool result. We model it as a custom AgentLoop that:
#
#   - Keeps a `VideoTrajectoryState` across chunks (memory, pending Qs,
#     final answer flag — same dataclass slyme uses)
#   - Each chunk: render `<chunk_update>` text (optionally with frames),
#     append as a user turn (mask=0), call server_manager.generate, append
#     assistant turn (mask=1), parse the v12 protocol output, advance state
#   - Stops on: explicit non-empty <answer>, max_chunks reached,
#     prompt+response > response_length budget
#
# Registered under `"thinkstream_streaming_agent"` — that name is set by
# CustomRLHFDataset.__getitem__ in thinkstream.py so verl picks this loop
# up automatically for our rows.
#
# REMAINING GAPS (intentionally deferred — landed as a clear TODO):
#   1. Per-chunk visual frame injection
#      Streaming SFT trains on a sliding 16-chunk window of frames per
#      turn. To match that distribution at RL time, each chunk's user
#      block needs to inject NEW image tokens. Implementing this requires
#      either:
#        (a) handing image_data tensors to server_manager.generate()
#            mid-loop — supported by the API but every turn must re-prefill
#            the visual prefix (no kv cache savings on the visual side); or
#        (b) loading the full video once at the start and letting the
#            model temporally ground via text markers — degenerate but
#            simpler. Currently this loop runs TEXT-ONLY and relies on
#            the model's SFT priors.
#   2. Real recall retriever
#      `_execute_recall` returns a stub. The slyme path has a vector-store
#      retriever; wiring it here is decoupled from the loop topology so
#      I'm leaving a clear seam (`_execute_recall`, `_execute_compress`).
#   3. Per-chunk action gold loss
#      The reward function aggregates one trajectory-level score. To do
#      per-chunk action correctness (silent vs response per chunk), we'd
#      need a token-level reward broadcast — leave for a future PR.
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _register_streaming_agent_loop():
    """Lazy registration — verl + thinkstream imports happen at call time
    so this module is importable in dev environments that lack verl.

    verl 0.4 calls importlib on the recipe path before instantiating any
    loop, so this function runs once per worker on first agent_name match.
    """
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

        After max_chunks or first non-empty <answer>, loop terminates.
        Concatenated `response_ids` (assistant tokens with mask=1, user
        chunk-update tokens with mask=0) is fed to compute_score, which
        splits on `<think>` to recover per-chunk outputs.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            mt = self.rollout_config.multi_turn
            # Cap chunks: take min of (config max_turns, dataset n_chunks).
            self.max_chunks = int(getattr(mt, "max_turns", 0) or 360)
            # Optional knobs (added to the recipe yaml under multi_turn.*)
            self.recall_stub_text = getattr(
                mt, "recall_stub_text",
                "(no relevant past observation found)",
            )

        # -------------------------------------------------------------------
        # Chunk → user-side text. Reuses the same protocol the SFT path uses
        # (see thinkstream/sft/data_processor.py:build_per_timestep_messages_v12)
        # so train and RL distributions match.
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
        ) -> str:
            parts: List[str] = []
            # Visual placeholder — TODO replace with actual frame injection.
            parts.append(f"<chunk_update idx={chunk_idx} t={chunk_idx}-{chunk_idx + 1}s/>")
            # Memory snapshot (compressed_summaries + recent_thinks)
            try:
                mem_text = format_memory_block({
                    "compressed_summaries": state.compressed_summaries,
                    "recent_thinks": state.recent_thinks,
                })
            except Exception:
                mem_text = ""
            if mem_text:
                parts.append(f"<memory>\n{mem_text}\n</memory>")
            # Pending question from the dataset row, fired only at the
            # configured ask_chunks. After the first ask_chunk it's stale —
            # but we keep emitting until answered.
            if question and ask_chunks and chunk_idx >= min(ask_chunks):
                parts.append(f"<query>{question}</query>")
            # Recall result from the previous turn (Shape B inline form).
            if recall_result is not None:
                parts.append(f"<recall_result>{recall_result}</recall_result>")
            if compress_trigger:
                parts.append(
                    "<compress_trigger range='recent'/>  "
                    "(memory token-budget exceeded; emit a compress tool_call this turn)"
                )
            return "\n".join(parts)

        # -------------------------------------------------------------------
        # Tool execution stubs. Real wiring is decoupled — replace these
        # with retriever / summarizer calls when ready.
        # -------------------------------------------------------------------
        async def _execute_recall(
            self, args: Dict[str, Any], state: "VideoTrajectoryState"
        ) -> str:
            # TODO: call actual retriever over compressed_summaries +
            # recent_thinks. For now return a deterministic stub so the
            # protocol shape is correct and training can proceed.
            return self.recall_stub_text

        async def _execute_compress(
            self, args: Dict[str, Any], state: "VideoTrajectoryState"
        ) -> None:
            # State update already handled by default_v12_update_state.
            return None

        async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":
            metrics: Dict[str, Any] = {}
            request_id = uuid4().hex

            extra_info = kwargs.get("extra_info") or {}
            video_id = extra_info.get("video_id") or extra_info.get("index", "")
            n_chunks_dataset = int(extra_info.get("n_chunks") or 0)
            n_chunks = min(self.max_chunks, n_chunks_dataset) if n_chunks_dataset else self.max_chunks
            question = extra_info.get("question", "")
            ask_chunks = list(extra_info.get("ask_chunks") or [])

            # ── Initial prompt: system + user(question). No frames in this
            # pass — see file header for the frame-injection TODO.
            messages = list(kwargs["raw_prompt"])
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self.apply_chat_template(
                messages, tools=TOOLS_SCHEMA, images=images, videos=videos,
            )

            initial_prompt_len = len(prompt_ids)
            response_mask: List[int] = []
            response_logprobs: List[float] = []

            state = VideoTrajectoryState(video_uid=str(video_id), chunk_idx=0)
            recall_result_for_next: Optional[str] = None
            compress_trigger_for_next = False
            num_turns = 0

            for chunk_idx in range(n_chunks):
                if not state.is_active:
                    break

                # Render this chunk's user-side update.
                user_block = self._render_chunk_user_block(
                    state=state,
                    chunk_idx=chunk_idx,
                    question=question,
                    ask_chunks=ask_chunks,
                    recall_result=recall_result_for_next,
                    compress_trigger=compress_trigger_for_next,
                )
                recall_result_for_next = None
                compress_trigger_for_next = False

                user_msg = [{"role": "user", "content": user_block}]
                user_ids = await self.apply_chat_template(
                    user_msg, remove_system_prompt=True,
                )
                # Length budget — leave room for at least one assistant turn.
                if len(response_mask) + len(user_ids) >= self.response_length:
                    break
                prompt_ids = prompt_ids + user_ids
                response_mask.extend([0] * len(user_ids))
                if response_logprobs:
                    response_logprobs.extend([0.0] * len(user_ids))

                # Generate assistant turn.
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=images,
                        video_data=videos,
                    )
                assistant_ids = list(output.token_ids)
                if not assistant_ids:
                    break
                prompt_ids = prompt_ids + assistant_ids
                response_mask.extend([1] * len(assistant_ids))
                if output.log_probs:
                    if not response_logprobs:
                        response_logprobs.extend([0.0] * (len(response_mask) - len(assistant_ids)))
                    response_logprobs.extend(list(output.log_probs))
                num_turns += 1

                # Decode + parse this chunk's output.
                response_text = self.tokenizer.decode(
                    assistant_ids, skip_special_tokens=True,
                )
                parsed = parse_agent_output_v12(response_text)
                kind = parsed.get("kind", "unknown")

                # State transition.
                state = default_v12_update_state(state, response_text, chunk_idx)

                if kind == "recall":
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    recall_result_for_next = await self._execute_recall(args, state)
                elif kind == "compress":
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    await self._execute_compress(args, state)

                # Termination — either explicit final answer or response budget.
                if not state.is_active or state.is_done:
                    break
                if len(response_mask) >= self.response_length:
                    break

            num_turns += 1  # +1 for the initial system+user prompt turn

            response_ids_full = prompt_ids[initial_prompt_len:]
            output_obj = AgentLoopOutput(
                prompt_ids=prompt_ids[:initial_prompt_len],
                response_ids=response_ids_full[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                multi_modal_data={
                    "images": images if images else [],
                    "videos": videos if videos else [],
                },
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
                "ts_final_chunk": float(state.final_answer_chunk if state.final_answer_chunk is not None else -1),
                "ts_chunks_used": float(num_turns),
            })
            return output_obj

    return ThinkStreamStreamingAgentLoop


# Trigger registration on import. verl's main_ppo imports recipe modules
# eagerly, so this runs in every Ray worker before agent loop dispatch.
try:
    _register_streaming_agent_loop()
except Exception as e:  # noqa: BLE001 — local dev tooling shouldn't crash.
    logger.debug("ThinkStreamStreamingAgentLoop not registered: %s", e)
