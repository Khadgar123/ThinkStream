"""v12.2 chunk-level RL rollout — MemAgent-style recurrent loop for streaming video.

Direct port of MemAgent's `MemoryAgent` pattern (impls/memory.py:145-262)
adapted to ThinkStream's per-video, per-chunk decision schema.

Key idea (research-backed by industry survey, Jan 2026):
  240K monolithic trajectory = infeasible (KV cache 30 GB/seq × 8 rollouts
                                            = 240 GB on Qwen3-VL-7B).
  Solution: each chunk's decision is an INDEPENDENT rollout of ≤16K tokens.
            Trajectory-level outcome reward broadcasts across chunks via
            shared video_uid GRPO grouping (ReMemR1 ICLR'26 mixed advantage).

Architecture:
  ChunkLevelRolloutLoop.start(batch)            # init memory state per video
  while not done:
    msgs = loop.action()                        # build per-chunk messages
    outputs = vllm.generate(msgs, n=G)          # G rollouts per chunk
    loop.update(outputs)                        # update memory + track sample_idx
  loop.end() -> (final_mask, sample_index)

  ↓

  reward_v12 = calc_per_chunk_rewards(outputs)  # 5 components per chunk
  mixed_adv  = aggregate_v12_advantages(        # ReMemR1 mixed pattern
      rewards, masks, video_uid, chunk_idx, G, alpha=0.7
  )
  GRPO loss = -E[log_prob × mixed_adv]

References (file:line cited verbatim):
- /tmp/agent-sft-research/MemAgent/recurrent/impls/memory.py:175-237
  (action loop with active_mask + step counter + sample_index list)
- /tmp/agent-sft-research/ReMemR1/verl/trainer/ppo/ray_trainer.py:1288-1314
  (mixed outcome+state GRPO advantage)
- /tmp/agent-sft-research/ReMemR1/recurrent/impls/memory_revisit.py
  (state reward shaping)

This module is CPU-testable: the ChunkLevelRolloutLoop separates control
flow from generation. Tests inject a stub generator; real training uses
vLLM via thinkstream/eval/vllm_engine.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

import torch


@dataclass
class ChunkLevelRolloutConfig:
    """v12.5 chunk-level rollout config. Matches ReMemR1/MemAgent recurrent
    config conventions. Defaults aligned with 1s/chunk + 16s visual window +
    16k SFT cutoff_len.
    """
    group_size: int = 8                  # G = rollouts per (video, chunk)
    # v12.5: 5-minute video × 1 chunk/s = 300 chunks. Hard ceiling at 360
    # leaves slack for slightly-overlong videos. Streaming inference uses
    # the actual video duration; this is a safety cap to bound rollout cost.
    max_chunks_per_video: int = 360
    # v12.5: align with SFT cutoff_len. memory(≤4000) + visual_window(2048) +
    # queries(~300) + system+tools(~400) + recall(~500) + headroom.
    max_prompt_length: int = 16384
    max_response_length: int = 2048      # think + answer/tool_call
    # v12.5: 16 chunks × 2 frames × 128 vis tokens = 4096
    chunk_visual_tokens: int = 4096
    # Mixed advantage mixing (ReMemR1 default 0.8; ThinkStream uses 0.7
    # because per-step state reward is denser in long-horizon videos).
    advantage_alpha: float = 0.7
    # Compress trigger is driven by token-budget in MemoryState
    # (RECENT_THINKS_TOKEN_BUDGET=4000, threshold=3200), not by chunk count.
    # Kept here only for backward-compat with old rollout call sites.
    compress_trigger_every: int = 0  # 0 = disabled, use token-budget


@dataclass
class VideoTrajectoryState:
    """Per-video state carried across chunks (analog of MemAgent's
    `self.memory` array, but richer for streaming video).

    Each video gets one of these. We replicate it G times (group_size)
    so each rollout has its own state evolution.
    """
    video_uid: str
    chunk_idx: int = 0                   # current chunk position
    # Memory state — what the model has learned about the video so far
    compressed_summaries: List[Dict] = field(default_factory=list)
    recent_thinks: List[Dict] = field(default_factory=list)
    # Question/answer tracking
    pending_queries: List[Dict] = field(default_factory=list)  # {q, ask_chunk}
    answered_queries: List[Dict] = field(default_factory=list) # {q, a, ask, resp_chunk}
    # Tool usage tracking (for spam reward + diagnostic)
    n_recall_calls: int = 0
    n_compress_calls: int = 0
    # Final answer + when emitted (for outcome+timing reward)
    final_answer: Optional[str] = None
    final_answer_chunk: Optional[int] = None
    # Bookkeeping
    is_active: bool = True               # False after final answer or max chunks
    is_done: bool = False                # True after end signal


def replicate_state_for_group(
    initial_state: VideoTrajectoryState, group_size: int,
) -> List[VideoTrajectoryState]:
    """Make G independent copies of a video's initial state for GRPO group rollouts.

    Each rollout in the group evolves its memory independently — that's the
    GRPO baseline-of-rollouts that gives advantage signal.
    """
    from copy import deepcopy
    return [deepcopy(initial_state) for _ in range(group_size)]


class ChunkLevelRolloutLoop:
    """Recurrent loop that processes 1 video at a time, chunk-by-chunk.

    Mirrors MemAgent's MemoryAgent.{start, action, update, done, end} surface.
    Differences vs MemAgent:
      - Per-chunk visual tokens injected (MemAgent is text-only)
      - Per-chunk memory has structured fields (compressed/recent/queries)
      - Multiple termination conditions: emit answer / max chunks / explicit done
      - Tracks per-(video_uid, chunk_idx) reward components for mixed advantage

    Generation is delegated to a callable `generate_fn(messages, n) -> texts`,
    so tests can mock and production wires vLLM.
    """

    def __init__(
        self,
        config: ChunkLevelRolloutConfig,
        generate_fn: Callable[[List[Dict], int], List[str]],
        build_messages_fn: Callable[[VideoTrajectoryState, Dict], List[Dict]],
        update_state_fn: Callable[[VideoTrajectoryState, str, int], VideoTrajectoryState],
    ):
        """
        generate_fn(messages, n) -> list of n strings (G rollouts of one chunk).
            For real training, wraps vLLM.generate. For tests, returns canned strings.
        build_messages_fn(state, video_meta) -> messages list for chat_template.
            For v12, this includes <visual_window>, <memory>, <queries>,
            optional <compress_trigger>, and the user_input for any pending Q.
        update_state_fn(state, response_text, chunk_idx) -> updated state.
            Parses the response (parse_agent_output_v12), updates memory based
            on action kind. Pure function over input state.
        """
        self.cfg = config
        self.generate_fn = generate_fn
        self.build_messages_fn = build_messages_fn
        self.update_state_fn = update_state_fn

    def rollout_one_video(
        self,
        video_meta: Dict,
        initial_state: VideoTrajectoryState,
    ) -> Dict[str, Any]:
        """Run G parallel rollouts of a single video, chunk by chunk.

        Returns:
            {
              "video_uid": str,
              "states_per_rollout": [G VideoTrajectoryState],
              "responses_per_chunk": [n_chunks_used] of [G strings],
              "messages_per_chunk":  [n_chunks_used] of [G messages],
              "final_chunk_per_rollout": [G] int (where each rollout ended),
            }
        """
        states = replicate_state_for_group(initial_state, self.cfg.group_size)
        responses_per_chunk: List[List[str]] = []
        messages_per_chunk: List[List[List[Dict]]] = []

        for chunk_step in range(self.cfg.max_chunks_per_video):
            # active_mask follows MemAgent's pattern
            active_indices = [i for i, s in enumerate(states) if s.is_active]
            if not active_indices:
                break

            # Build per-active-rollout messages
            chunk_msgs: List[List[Dict]] = []
            for i in active_indices:
                # Each active state gets its own messages constructed from
                # its evolved memory state. video_meta provides the chunk's
                # raw visual + ground-truth pending question events.
                msgs = self.build_messages_fn(states[i], {
                    **video_meta, "chunk_idx": chunk_step,
                })
                chunk_msgs.append(msgs)

            # Generate one response per active rollout. Note: this is the
            # critical efficiency step — instead of one 240K-token rollout,
            # we do `len(active_indices)` independent ≤16K rollouts in
            # one batched vLLM call.
            chunk_responses_active = self.generate_fn(chunk_msgs, n=1)
            # generate_fn(messages_batch, n=1) returns a flat list, one per msg

            # Spread back to G slots (inactive get empty placeholder)
            chunk_responses_full: List[str] = ["" for _ in range(self.cfg.group_size)]
            chunk_messages_full: List[List[Dict]] = [[] for _ in range(self.cfg.group_size)]
            for i_in_active, i in enumerate(active_indices):
                chunk_responses_full[i] = chunk_responses_active[i_in_active]
                chunk_messages_full[i] = chunk_msgs[i_in_active]
                # Update memory state based on the response.
                states[i] = self.update_state_fn(
                    states[i], chunk_responses_active[i_in_active], chunk_step,
                )

            responses_per_chunk.append(chunk_responses_full)
            messages_per_chunk.append(chunk_messages_full)

        return {
            "video_uid": initial_state.video_uid,
            "states_per_rollout": states,
            "responses_per_chunk": responses_per_chunk,
            "messages_per_chunk": messages_per_chunk,
            "final_chunk_per_rollout": [s.chunk_idx for s in states],
        }


# ===========================================================================
# Default state-update logic for v12 protocol
# ===========================================================================


def default_v12_update_state(
    state: VideoTrajectoryState,
    response_text: str,
    chunk_idx: int,
) -> VideoTrajectoryState:
    """Apply a single rollout response to the state — v12 protocol-aware.

    Parses response via parse_agent_output_v12, updates fields based on kind:
      - answer (empty): silent at this chunk; no state change
      - answer (non-empty): final answer recorded, mark inactive
      - recall: increment counter, add tool result to memory (caller-provided)
      - compress: record summary, clear/reduce recent_thinks

    Pure function: returns a new state object (no mutation in caller's scope).
    """
    from copy import deepcopy
    from thinkstream.data.agent_protocol import parse_agent_output_v12

    new_state = deepcopy(state)
    new_state.chunk_idx = chunk_idx + 1   # advance to next chunk

    parsed = parse_agent_output_v12(response_text)
    kind = parsed.get("kind", "unknown")

    if kind == "answer":
        ans = parsed.get("answer_text") or ""
        if ans == "":
            # Silent — no behavioral change, just advance chunk
            pass
        else:
            # Final answer emitted. Mark inactive — this rollout is done.
            new_state.final_answer = ans
            new_state.final_answer_chunk = chunk_idx
            new_state.is_active = False
            new_state.is_done = True

            # Resolve any pending query that this answer addresses
            if new_state.pending_queries:
                pending = new_state.pending_queries[0]
                new_state.answered_queries.append({
                    "question": pending["question"],
                    "answer": ans,
                    "ask_chunk": pending["ask_chunk"],
                    "response_chunk": chunk_idx,
                })
                new_state.pending_queries = new_state.pending_queries[1:]

    elif kind == "recall":
        new_state.n_recall_calls += 1
        # NOTE: in real rollout, the system retrieval result should be
        # injected into memory (recent_thinks or a new "recall_result"
        # field) for the NEXT chunk's prompt. That side-effect is the
        # caller's responsibility (see eval/streaming_vllm.py). Here we
        # only track the call count for spam reward.

    elif kind == "compress":
        new_state.n_compress_calls += 1
        tool_call = parsed.get("tool_call") or {}
        args = tool_call.get("arguments") or {}
        new_state.compressed_summaries.append({
            "time_range": args.get("time_range", []),
            "text": args.get("text", ""),
            "from_chunk": chunk_idx,
        })
        # Drop the oldest recent_thinks that fall in the compressed range
        if args.get("time_range"):
            try:
                tr_start, tr_end = sorted(args["time_range"])[:2]
                new_state.recent_thinks = [
                    t for t in new_state.recent_thinks
                    if not (tr_start <= t.get("chunk", -1) <= tr_end)
                ]
            except (TypeError, ValueError):
                pass

    # else "unknown" — format error; treat as silent for state evolution
    return new_state


# ===========================================================================
# End-to-end mixed-advantage helper for v12 chunk-level RL
# ===========================================================================


def compute_1d_grpo_advantage(
    rewards: torch.Tensor,        # [N] OR [N, T] — if 2D, summed over last dim
    group_index: List[Any],        # [N] hashable group key per rollout
    *,
    epsilon: float = 1e-6,
    use_adv: bool = True,          # ReMemR1 line 252 function default
) -> torch.Tensor:
    """1-D GRPO advantage — EXACT port of ReMemR1 ray_trainer.py:249-288.

    Per-group computation matching ReMemR1 line-for-line:
      scores = rewards.sum(dim=-1) if 2D else rewards
      for each group g in id2score.keys():
        if len == 1:
          mean = 0.0 (NOT the sample's own value — preserves raw score
                      as advantage for singleton groups, which is
                      intentional per ReMemR1 convention)
          std  = 1.0 (only if use_adv)
        elif len > 1:
          mean = torch.mean(group_scores)
          std  = torch.std(group_scores)  # unbiased=True default (Bessel)
      adv[i] = (scores[i] - mean) / (std + epsilon)  if use_adv
      adv[i] = scores[i] - mean                       if not use_adv

    Returns: [N] scalar advantage. Default `use_adv=True` matches ReMemR1's
    function signature (line 252); the production trainer overrides via
    `self.config.algorithm.grpo_use_adv`.

    Differences from naive "subtract group mean":
      1. Singleton group: mean=0 (ReMemR1 convention), NOT the sample value.
      2. Std uses `torch.std` (unbiased=True/Bessel), not unbiased=False.
      3. Epsilon ADDED to denominator, not clamped.
    These three details affect numerics; matching them is critical for
    reproducing ReMemR1 advantage shape.
    """
    # Accept [N] scalar OR [N, T] token-level (sum over T to get [N])
    if rewards.dim() == 2:
        scores = rewards.sum(dim=-1)
    else:
        scores = rewards.clone()
    N = scores.shape[0]
    if len(group_index) != N:
        raise ValueError(
            f"group_index length {len(group_index)} != scores length {N}"
        )

    from collections import defaultdict
    id2score: Dict[Any, List[torch.Tensor]] = defaultdict(list)
    id2mean: Dict[Any, torch.Tensor] = {}
    id2std: Dict[Any, torch.Tensor] = {}

    # ReMemR1 wraps the entire compute in torch.no_grad (line 268). Replicate
    # so advantages flow into loss as detached scaling coefficients only.
    with torch.no_grad():
        for i in range(N):
            id2score[group_index[i]].append(scores[i])

        for idx, lst in id2score.items():
            if len(lst) == 1:
                # ReMemR1 lines 273-276 — singleton group: mean=0, std=1.
                # Advantage = (score - 0) / (1 + ε) = raw score. Preserves
                # full signal when no peer rollouts available; do NOT change.
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                if use_adv:
                    id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(lst) > 1:
                stacked = torch.stack(lst)
                id2mean[idx] = stacked.mean()
                if use_adv:
                    # ReMemR1 line 280: torch.std(torch.tensor([list])) wraps
                    # the list, producing [1, K]. std over the entire tensor
                    # with default unbiased=True (Bessel correction).
                    id2std[idx] = stacked.std()  # unbiased=True default
            else:
                raise ValueError(f"no score in group index: {idx}")

        out = torch.zeros_like(scores)
        for i in range(N):
            idx = group_index[i]
            m = id2mean[idx]
            if use_adv:
                s = id2std[idx]
                out[i] = (scores[i] - m) / (s + epsilon)
            else:
                out[i] = scores[i] - m
    return out


def compute_mixed_advantage_v12(
    outcome_reward: torch.Tensor,    # [N] per-rollout final outcome
    state_reward: torch.Tensor,      # [N] per-rollout per-chunk state shaping
    video_uid_per_row: List[str],    # [N] which video each rollout belongs to
    chunk_idx_per_row: List[int],    # [N] which chunk position each rollout is at
    *,
    alpha: float = 0.7,
    use_adv: bool = True,
) -> torch.Tensor:
    """v12 mixed advantage = α · outcome_adv + (1-α) · state_adv.

    Direct port of ReMemR1 ray_trainer.py:1288-1314 logic.

    outcome_adv: GRPO-norm grouped by video_uid (broadcasts trajectory-level
                 reward to all chunks of that trajectory's rollouts).
    state_adv:   GRPO-norm grouped by (video_uid, chunk_idx) (per-step credit
                 within rollouts that reached the same chunk position).

    α = 0.7 default (ReMemR1's 0.8 is HotpotQA; ThinkStream skews lower
    because per-step signal is denser in 30-chunk videos).
    """
    if outcome_reward.shape != state_reward.shape:
        raise ValueError(
            f"outcome and state reward shapes differ: "
            f"{outcome_reward.shape} vs {state_reward.shape}"
        )
    N = outcome_reward.shape[0]
    if len(video_uid_per_row) != N or len(chunk_idx_per_row) != N:
        raise ValueError(
            f"index length mismatch: video_uid={len(video_uid_per_row)}, "
            f"chunk_idx={len(chunk_idx_per_row)}, expected {N}"
        )

    outcome_adv = compute_1d_grpo_advantage(
        outcome_reward, video_uid_per_row, use_adv=use_adv,
    )
    step_index = [
        f"{uid}::{int(cid)}"
        for uid, cid in zip(video_uid_per_row, chunk_idx_per_row)
    ]
    state_adv = compute_1d_grpo_advantage(
        state_reward, step_index, use_adv=use_adv,
    )

    return alpha * outcome_adv + (1.0 - alpha) * state_adv


# ===========================================================================
# Adapter: ChunkLevelRolloutLoop output → _calc_rewards_v12 input shape
# ===========================================================================


def chunk_results_from_loop_result(
    loop_result: Dict[str, Any],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Translate `rollout_one_video()` output into the `chunk_results` list
    that `thinkstream.trainer.grpo._calc_rewards_v12` expects.

    Loop result shape:
      {
        "video_uid": str,
        "responses_per_chunk": [n_chunks][G] strings,
        ...
      }

    Reward path expects chunk_results = [
        {"chunk_idx": int, "generated_tokens": [G] token-id lists, ...},
        ...
    ]

    This adapter tokenizes each rollout's response per chunk so the existing
    decode-and-parse logic in `_calc_rewards_v12` works unchanged.
    """
    responses_per_chunk = loop_result["responses_per_chunk"]
    chunk_results: List[Dict[str, Any]] = []
    for chunk_idx, group_texts in enumerate(responses_per_chunk):
        tokens_per_g: List[List[int]] = []
        for text in group_texts:
            ids = tokenizer.encode(text, add_special_tokens=False) if text else []
            tokens_per_g.append(ids)
        chunk_results.append({
            "chunk_idx": chunk_idx,
            "generated_tokens": tokens_per_g,
        })
    return chunk_results
