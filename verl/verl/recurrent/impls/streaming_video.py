# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# v12.14 STREAMING-VIDEO RECURRENT AGENT (skeleton)
# ==================================================
#
# This file is the recurrent-RL counterpart to
# verl/recipe_thinkstream/streaming_agent_loop.py. It re-frames the
# trajectory so each chunk's (prompt, response) becomes ITS OWN
# training sample, removing the stitched-trajectory OOM ceiling.
#
# Status: SKELETON. Not wired into ray_trainer yet. Implement when:
#   - want >180-chunk trajectories (catalog 240-600s tier)
#   - observe train/eval reward gap >10% on OVOBench
#   - hit actor OOM at production batch sizes
#
# See docs/v12.14_recurrent_design.md for the full spec.
#
# ────────────────────────────────────────────────────────────────────────────
# Mapping: streaming_agent_loop.py → this file
#
#   stitched_run() outer chunk loop          → AsyncStreamingVideoAgent.rollout()
#   one chunk-internal multi-turn (D1)       → one AsyncOutput conversation
#   trajectory-final per-Q answer aggregation → final turn after all chunks
#   stitched response_ids/response_mask       → list of conversations + sample_index
#   per-Q answer attribution (LIFO/FIFO)      → tracked across conversations
#
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

from verl.recurrent.async_utils import ChatCompletionProxy
from verl.recurrent.interface import (
    AsyncRAgent, AsyncOutput, RConfig, RDataset, RRegister,
)
from verl.recurrent.utils import log_step, msg
from verl.protocol import DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class StreamingVideoConfig(RConfig):
    """Per-action / per-chunk budgets for streaming-video recurrent rollout.

    Compare with the stitched path's MAX_TURNS / max_response_length:
    here `max_chunks` is the upper bound on chunks per trajectory, but
    each chunk gets its own modest token budget (`max_chunk_response_length`)
    instead of sharing the trajectory-wide stitched cap.
    """
    # Chunk-level
    max_chunks: int = 360                # upper bound on video chunks
    max_chunk_prompt_length: int = 8192  # per-action prompt cap
    max_chunk_response_length: int = 512 # per-action response cap (one assistant turn)

    # Recall (D1 chunk-internal multi-turn)
    max_recall_per_chunk: int = 1
    max_recall_response_length: int = 512  # turn-2 answer after recall

    # Compress / memory
    compress_token_threshold: int = 3200   # mirrors config.py COMPRESS_TOKEN_THRESHOLD
    visual_window_chunks: int = 16
    frames_per_chunk: int = 2
    chunk_sec: float = 1.0

    # Visual mode
    visual_window_mode: str = "sliding"    # or "expanding" (opt-in)

    # Final aggregation turn (after all chunk turns)
    enable_final_turn: bool = False        # True = explicit per-Q summary turn
    max_final_response_length: int = 1024


# ===========================================================================
# Dataset
# ===========================================================================
class StreamingVideoDataset(RDataset):
    """Multi-Q parquet adapter for recurrent rollout.

    Reads the same parquet as the stitched path (CustomRLHFDataset in
    verl/recipe_thinkstream/thinkstream.py:178) but emits per-row state
    that the rollout will iterate over chunk-by-chunk:

      context_ids       — placeholder (not used; visual+memory rebuilt per chunk)
      video_id          — for frame loading
      video_path        — for frame loading
      questions         — multi-Q list (carried into rollout's chunk_messages)
      n_chunks          — total chunks for this video
      gold_action_per_chunk — per-chunk gold actions (silent/response/recall/compress)
    """

    def __init__(
        self,
        recurrent_config: StreamingVideoConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )
        self.recurrent_config = recurrent_config

    @override
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        # TODO: reuse CustomRLHFDataset.__getitem__'s multi-Q normalisation
        # logic (verl/recipe_thinkstream/thinkstream.py:216-305) so this
        # dataset and the stitched dataset stay in lockstep on data prep.
        # For now, leave as-is and let the agent's rollout() do field
        # extraction directly from row_dict.

        # Recurrent-required fields:
        row_dict["sample_uuid"] = str(uuid4())
        row_dict["index"] = row_dict.get("video_id", "")
        # context_ids placeholder — visual + memory are rebuilt per chunk
        # in rollout(), so a single context tokenisation is meaningless
        # here. Keep an empty tensor to satisfy the RDataset contract.
        row_dict["context_ids"] = torch.zeros(1, dtype=torch.long)
        row_dict["context_length"] = torch.tensor(1, dtype=torch.long)
        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
        return (
            ["context_ids", "context_length"],
            ["video_id", "video_path", "n_chunks", "extra_info",
             "reward_model", "data_source", "sample_uuid"],
        )


# ===========================================================================
# Agent
# ===========================================================================
class AsyncStreamingVideoAgent(AsyncRAgent):
    """Streaming-video recurrent agent.

    rollout(gen_item) produces an AsyncOutput where each conversation is
    ONE chunk's interaction:
      conversation_i = [
        {role: user,      content: chunk_i prompt (memory + visual + user_input)},
        {role: assistant, content: <think>X</think><answer or tool_call>...},
        # If chunk_i emitted recall (D1 multi-turn within chunk):
        {role: user,      content: <recalled_frames> + video + <recall_result>},
        {role: assistant, content: <think>Y</think><answer>Z</answer>},
      ]

    Chunks are independent samples (sample_index all == this row's idx).
    final_mask=True only on the LAST chunk's conversation (or on an
    explicit final aggregation turn if enable_final_turn=True).

    NOT YET IMPLEMENTED — this is a skeleton. The TODOs below mark the
    work to port from streaming_agent_loop.py.
    """

    def __init__(
        self,
        proxy: ChatCompletionProxy,
        tokenizer: PreTrainedTokenizer,
        config: RConfig,
        rollout_config: DictConfig,
    ):
        super().__init__(proxy, tokenizer, config, rollout_config)
        self.config: StreamingVideoConfig = config

    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """One trajectory → N independent chunk conversations + final mask.

        TODO (port from streaming_agent_loop.py):
          1. Initialize VideoTrajectoryState
          2. For chunk_idx in range(n_chunks):
              a. _check_compress_trigger → maybe a compress turn (no visual)
              b. _build_visual_window → sliding (or expanding) frames
              c. _build_chunk_user_content (memory + window + question)
              d. self.proxy.get_chat_completions(...)
              e. parse → if recall && rounds < max_recall_per_chunk:
                   - _execute_recall (with video_path → frames + MROPE)
                   - _build_recall_tool_message
                   - get_chat_completions again (turn 2: answer)
              f. attribute <answer> to a Q via answer_chunks/LIFO/FIFO
              g. update state, advance chunk_idx
          3. Optionally an explicit final_turn that asks the model to
             summarise all per-Q answers (enable_final_turn).
          4. Build sample_index = full(len(conversations), gen_item.idx)
             and final_mask = [F]*N + [T] (or [F]*(N-1) + [T] without
             explicit final).
        """
        timing_raw: Dict[str, float] = {}

        # Skeleton placeholder so the file imports and registers cleanly.
        # Real implementation needed when v12.14 is activated.
        raise NotImplementedError(
            "StreamingVideoAgent rollout not yet implemented. See TODO list "
            "above and docs/v12.14_recurrent_design.md for the full spec. "
            "Until then use the stitched path (recipe_thinkstream/streaming_agent_loop.py)."
        )


# ===========================================================================
# Registration entry point — verl loads via recurrent.path / recurrent.name
# ===========================================================================
REGISTER = RRegister(
    config_cls=StreamingVideoConfig,
    dataset_cls=StreamingVideoDataset,
    agent_cls=AsyncStreamingVideoAgent,
)
