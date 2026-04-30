"""verl multi-turn rollout config for ThinkStream chunk-level streaming.

Bridges verl's rollout abstraction (verl 0.4+ supports multi-turn natively
via `MultiTurnRollout`) to ThinkStream's per-chunk decision schema. Each
chunk = one turn from verl's perspective. State evolution between turns
is delegated to `default_v12_update_state` from the slyme path — same
pure function, framework-agnostic.

verl 0.4 reference:
  https://verl.readthedocs.io/en/latest/multiturn/rollout.html

Design choice: drive each chunk through verl's `interaction_step` callback
rather than verl's full multi-turn template. We need the per-chunk visual
window to slide and the memory text to update — verl's stock template
does monotonic conversation growth which doesn't match our state model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from thinkstream.trainer.v12_rollout import (
    ChunkLevelRolloutConfig,
    VideoTrajectoryState,
    default_v12_update_state,
    replicate_state_for_group,
)


@dataclass
class VerlMultiTurnConfig:
    """verl-side rollout config.

    Mirrors slyme `ChunkLevelRolloutConfig` so cross-validation can compare
    apples-to-apples. Defaults aligned with v12.5 (1s/chunk, 16s window).
    """
    # Group + horizon
    group_size: int = 8
    max_chunks_per_video: int = 360
    # Token budgets (match SFT cutoff_len = 16384)
    max_prompt_length: int = 16384
    max_response_length: int = 2048
    # Chunk visual cost (16 chunks × 2 frames × 128 vis tokens)
    chunk_visual_tokens: int = 4096
    # Sampling (matches slyme rollout defaults)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    # Multi-level advantage mixing
    advantage_alpha: float = 0.7


class VerlChunkLevelRollout:
    """Adapter that wraps verl's vLLM rollout into chunk-level streaming.

    verl provides `Worker.generate_sequences()` which generates n responses
    in parallel for a batch of prompts. We invoke it once per chunk: build
    per-rollout messages from the current state, generate G responses,
    update state, repeat until all rollouts terminate or `max_chunks` hit.

    Public methods mirror the slyme `ChunkLevelRolloutLoop` so reward /
    advantage code is identical:

      .start(batch)        — init G states per video
      .action()            — build per-chunk messages
      .step(responses)     — apply G responses, advance state
      .end()               — return per-rollout final states + chunk_idx
    """

    def __init__(
        self,
        config: VerlMultiTurnConfig,
        verl_rollout_worker: Any,
        build_messages_fn: Callable[[VideoTrajectoryState, Dict], List[Dict]],
        update_state_fn: Callable[
            [VideoTrajectoryState, str, int], VideoTrajectoryState
        ] = default_v12_update_state,
    ):
        """
        verl_rollout_worker: a verl Worker (typically a `vLLMRollout` instance).
                             Must expose `generate_sequences(prompts, n) -> List[str]`.
        build_messages_fn:  same signature as slyme path; renders a state
                             into LLaMA-Factory ShareGPT messages.
        update_state_fn:    same as slyme — pure function over state, response, chunk.
        """
        self.cfg = config
        self.worker = verl_rollout_worker
        self.build_messages_fn = build_messages_fn
        self.update_state_fn = update_state_fn

    def rollout_one_video(
        self,
        video_meta: Dict,
        initial_state: VideoTrajectoryState,
    ) -> Dict[str, Any]:
        """Run G parallel rollouts of a single video, chunk by chunk.

        Identical contract to slyme `ChunkLevelRolloutLoop.rollout_one_video`.
        """
        states = replicate_state_for_group(initial_state, self.cfg.group_size)
        responses_per_chunk: List[List[str]] = []
        messages_per_chunk: List[List[List[Dict]]] = []

        for chunk_step in range(self.cfg.max_chunks_per_video):
            active_indices = [i for i, s in enumerate(states) if s.is_active]
            if not active_indices:
                break

            chunk_msgs: List[List[Dict]] = []
            for i in active_indices:
                msgs = self.build_messages_fn(
                    states[i], {**video_meta, "chunk_idx": chunk_step},
                )
                chunk_msgs.append(msgs)

            # verl rollout call: batched vLLM generate, n=1 per prompt
            chunk_responses_active = self.worker.generate_sequences(
                prompts=chunk_msgs,
                n=1,
                max_new_tokens=self.cfg.max_response_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )

            full_responses = ["" for _ in range(self.cfg.group_size)]
            full_messages: List[List[Dict]] = [[] for _ in range(self.cfg.group_size)]
            for i_in_active, i in enumerate(active_indices):
                full_responses[i] = chunk_responses_active[i_in_active]
                full_messages[i] = chunk_msgs[i_in_active]
                states[i] = self.update_state_fn(
                    states[i], chunk_responses_active[i_in_active], chunk_step,
                )

            responses_per_chunk.append(full_responses)
            messages_per_chunk.append(full_messages)

        return {
            "video_uid": initial_state.video_uid,
            "states_per_rollout": states,
            "responses_per_chunk": responses_per_chunk,
            "messages_per_chunk": messages_per_chunk,
            "final_chunk_per_rollout": [s.chunk_idx for s in states],
        }
