"""Streaming rollout adapter for verl's AgentLoopBase.

Wraps :class:`thinkstream.models.inference.StreamingWindowInferenceEngine` in a
thin, verl-friendly interface so the per-chunk vllm ``server_manager.generate``
call inside ``thinkstream/rl/streaming_agent_loop.py`` can be swapped
for a sliding-window streaming rollout that:

1. Keeps KV continuity across assistant turns within a trajectory
   (multi-turn dialogue), instead of re-prefilling the full prompt every
   chunk.
2. Evicts old visual KV blocks via the CUDA Graph sliding window
   (``StreamingWindowInferenceEngine``).
3. Performs a partial reset at compress boundaries (``reset_to_prefix``)
   so the system prompt + carried-over memory summary stays in KV while
   the previous trajectory's content is dropped.
4. Returns per-token log-probabilities for the PPO actor
   (``return_log_probs``).
5. Optionally enforces a tool-call early stop
   (``tool_call_stop_sample``).

References:
- ``StreamingWindowInferenceEngine`` — sliding-window streaming inference
- MemAgent ``recurrent/generation_manager.py`` — sample_index / final_mask
  bookkeeping pattern for variable-turn trajectories (consumed downstream by
  verl's recurrent advantage code).
- DeepEyes ``verl/workers/agent/envs/`` — pattern for injecting custom
  environments / rollouts into verl's worker pool without touching the
  upstream actor code.

This module deliberately does NOT touch verl internals; it is a pure adapter
the recipe layer can call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Output of a single assistant turn under the streaming engine.

    Mirrors the subset of verl's ``TokenOutput`` that the recipe agent loop
    consumes (token_ids + per-token log_probs + stop_reason).
    """

    token_ids: List[int]
    log_probs: Optional[List[float]] = None
    stop_reason: str = ""
    # Diagnostics: turn-level cache footprint and timing — useful for
    # telemetry without leaking into the verl batch dict.
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


# Sentinel for "use the value the engine was initialized with / its primary EOS".
_UNSET = object()


class StreamingRolloutEngine:
    """Thin wrapper that exposes verl-friendly turn / boundary methods.

    The engine holds physical KV state; the recipe agent loop orchestrates
    the call sequence::

        engine = StreamingRolloutEngine(streaming_engine)
        # Trajectory 0 (from_start): prefill system + first user, generate
        result_0 = engine.generate_turn(
            input_ids=system_plus_user_0_ids,
            position_ids=system_plus_user_0_pos,
            sampling_params={...},
            pixel_values_videos=video_features_chunk_0,
            video_grid_thw=video_grid_thw_chunk_0,
        )
        # ... append next user turn within same trajectory (KV continues) ...
        result_1 = engine.generate_turn(
            input_ids=user_1_ids,
            position_ids=user_1_pos,
            ...
        )
        # Compress boundary → start a new trajectory keeping the system prefix
        engine.begin_trajectory(
            keep_lengths=prefix_len_per_batch_item,
            next_start_pos=prefix_end_pos,
            keep_window_count=None,  # drop old video windows
        )
        # ... next trajectory's first user turn ...
    """

    def __init__(self, engine):
        # Late import: keep this module importable in environments without
        # flash_attn (e.g. local CPU dev shell). Type-check via duck-typing.
        self.engine = engine

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        sampling_params: Optional[Dict[str, Any]] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_generations: int = 1,
        sample_callback: Optional[Callable] = None,
        sample_callback_kwargs: Optional[Dict[str, Any]] = None,
        return_log_probs: bool = True,
        turn_kind: str = "",
        recall_kv_policy: Optional[str] = None,
        delete_previous_assistant_kv: bool = False,
    ) -> List[TurnResult]:
        """Run one assistant turn, appending to the current KV state.

        ``input_ids`` carries ONLY the new tokens since the previous turn
        (i.e. the next user turn's content for multi-turn continuation; or
        the full ``system + user_0`` for a fresh trajectory after
        :meth:`begin_trajectory` or :meth:`reset`). Position IDs should be
        local (starting at 0); the engine shifts them by its internal
        ``next_start_pos`` automatically.

        Returns one :class:`TurnResult` per effective-batch row (input batch
        size × ``num_generations``), each carrying the assistant token list
        and optional log_probs.
        """
        sp = sampling_params or {}
        max_new_tokens = int(sp.get("max_new_tokens") or sp.get("max_tokens") or 128)
        top_k = int(sp.get("top_k", 50))
        top_p = float(sp.get("top_p", 0.95))
        temperature = float(sp.get("temperature", 1.0))
        repetition_penalty = float(sp.get("repetition_penalty", 1.0))

        callback_kwargs = dict(sample_callback_kwargs or {})
        if turn_kind:
            callback_kwargs.setdefault("turn_kind", turn_kind)
        out = self.engine.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            sample=sample_callback,
            sample_kwargs=callback_kwargs,
            return_log_probs=return_log_probs,
            turn_kind=turn_kind,
            recall_kv_policy=recall_kv_policy or sp.get("recall_kv_policy"),
            delete_previous_assistant_kv=(
                delete_previous_assistant_kv
                or bool(sp.get("delete_previous_assistant_kv", False))
                or bool(sp.get("delete_previous_recall_toolcall_kv", False))
            ),
        )
        if return_log_probs:
            tokens_list, log_probs_list = out
        else:
            tokens_list = out
            log_probs_list = [None] * len(tokens_list)

        results: List[TurnResult] = []
        for tokens, lps in zip(tokens_list, log_probs_list):
            tok_ids = tokens.tolist()
            stop = _infer_stop_reason(
                tok_ids,
                primary_eos_token_id=int(self.engine.primary_eos_token_id),
                max_new_tokens=max_new_tokens,
            )
            results.append(
                TurnResult(
                    token_ids=tok_ids,
                    log_probs=lps.tolist() if lps is not None else None,
                    stop_reason=stop,
                    diagnostics={
                        "num_new_tokens": len(tok_ids),
                        "max_new_tokens": max_new_tokens,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def begin_trajectory(
        self,
        *,
        keep_lengths: torch.Tensor,
        next_start_pos: torch.Tensor,
        keep_window_count: Optional[torch.Tensor] = None,
    ) -> None:
        """Compress / trajectory boundary: retain ``keep_lengths`` prefix
        positions in KV, drop the rest, optionally retain the first
        ``keep_window_count`` video windows.

        See :meth:`StreamingInferenceEngine.reset_to_prefix` and
        :meth:`StreamingWindowInferenceEngine.reset_to_prefix` for details.
        """
        # Windowed engine accepts keep_window_count; base engine does not.
        if hasattr(self.engine, "_window_count"):
            self.engine.reset_to_prefix(
                keep_lengths=keep_lengths,
                next_start_pos=next_start_pos,
                keep_window_count=keep_window_count,
            )
        else:
            self.engine.reset_to_prefix(
                keep_lengths=keep_lengths,
                next_start_pos=next_start_pos,
            )

    def reset(self) -> None:
        """Full reset — wipe all KV state and start a new independent stream."""
        self.engine.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_stop_reason(
    token_ids: List[int], *, primary_eos_token_id: int, max_new_tokens: int
) -> str:
    """Mirror verl's stop_reason taxonomy for a single sequence."""
    if not token_ids:
        return "empty"
    if token_ids[-1] == primary_eos_token_id:
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "length"
    return "other"
