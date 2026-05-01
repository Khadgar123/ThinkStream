# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ThinkStream-specific verl reward manager.
#
# Why a custom reward manager (vs verl's built-in NaiveRewardManager):
#   NaiveRewardManager places ONE scalar reward at the trajectory's last
#   assistant token. Streaming-video v12 needs per-chunk shaping (gold
#   silent/response/recall/compress per chunk_idx) plus the trajectory-
#   level outcome — so we need to broadcast a per-chunk reward vector to
#   each chunk's last assistant token, in addition to the outcome at
#   trajectory end. ThinkStreamPerChunkRewardManager does both.
#
# Wiring:
#   - The agent loop fills extra_fields["ts_chunk_asst_spans"] = list of
#     (start, end) slice indices into response_ids per chunk.
#   - compute_score (in thinkstream.py) returns {"score": outcome, ...,
#     "_per_chunk_action_rewards": [r0, r1, ...]} aligned to
#     ts_chunk_asst_spans.
#   - This reward_manager places r_i at response_mask span i's last
#     assistant token, and `score` at the trajectory's last assistant
#     token. (If span i's last token is also the trajectory's last token
#     the values are summed.)
#
# Registered as `"thinkstream_per_chunk"`. Set in yaml:
#   reward_model:
#     reward_manager: thinkstream_per_chunk
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _register_reward_manager():
    """Register on first import. Wrapped so module is importable in dev
    environments lacking verl/torch."""
    import torch  # type: ignore

    from verl import DataProto  # type: ignore
    from verl.utils.reward_score import default_compute_score  # type: ignore
    from verl.workers.reward_manager import register  # type: ignore
    from verl.workers.reward_manager.abstract import AbstractRewardManager  # type: ignore

    @register("thinkstream_per_chunk")
    class ThinkStreamPerChunkRewardManager(AbstractRewardManager):
        """Per-chunk + trajectory-end reward placement for streaming v12.

        Behaves like NaiveRewardManager except:
          - Reads `ts_chunk_asst_spans` from non_tensor_batch (from agent
            loop's extra_fields) — list of (start, end) per chunk.
          - Reads `_per_chunk_action_rewards` from compute_score's return
            dict — list of per-chunk shaping rewards.
          - Places each r_i at the span's last token (end-1), the
            trajectory outcome `score` at the trajectory's last valid
            token, summing if those positions coincide.
        """

        def __init__(self, tokenizer, num_examine, compute_score=None,
                     reward_fn_key="data_source") -> None:
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.compute_score = compute_score or default_compute_score
            self.reward_fn_key = reward_fn_key

        def __call__(self, data, return_dict=False):
            reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
            if reward_from_rm_scores is not None:
                return reward_from_rm_scores

            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32,
            )
            reward_extra_info: Dict[str, List[Any]] = defaultdict(list)
            already_print: Dict[str, int] = {}

            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().item()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                # Surface the agent loop's extra_fields (verl flattens
                # them to top-level columns of non_tensor_batch).
                for key in (
                    "ts_chunk_kinds", "ts_chunk_asst_texts",
                    "ts_chunk_asst_spans", "ts_answer_chunk",
                    "ts_final_answer", "ts_n_recall", "ts_n_compress",
                    "ts_chunks_used", "ts_chunks_with_frames",
                    "ts_chunks_text_only",
                ):
                    if key in data_item.non_tensor_batch:
                        extra_info[key] = data_item.non_tensor_batch[key]

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    outcome_reward = float(score.get("score", 0.0))
                    per_chunk = score.get("_per_chunk_action_rewards") or []
                    for key, value in score.items():
                        if key.startswith("_"):
                            # passthrough fields, not for wandb logging
                            continue
                        reward_extra_info[key].append(value)
                else:
                    outcome_reward = float(score)
                    per_chunk = []

                # ── Place per-chunk shaping rewards on each span's last token.
                spans: List[Tuple[int, int]] = list(extra_info.get("ts_chunk_asst_spans") or [])
                for span_i, span in enumerate(spans):
                    if span_i >= len(per_chunk):
                        break
                    if span is None or len(span) != 2:
                        continue
                    s, e = int(span[0]), int(span[1])
                    if e <= 0 or e > valid_response_length:
                        # Span extends past the (post-truncation) valid
                        # length; clamp to last valid token.
                        e = valid_response_length
                    if e <= 0:
                        continue
                    pos = e - 1
                    reward_tensor[i, pos] += float(per_chunk[span_i])

                # ── Outcome at trajectory end. Add (not overwrite) so that
                # if the last chunk also coincides we don't blow away its
                # per-chunk shaping.
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] += outcome_reward

                # Debug printing — match NaiveRewardManager's surface.
                if data_source not in already_print:
                    already_print[data_source] = 0
                if already_print[data_source] < self.num_examine:
                    already_print[data_source] += 1
                    print("[prompt]", prompt_str[:500])
                    print("[response]", response_str[:500])
                    print("[ground_truth]", ground_truth)
                    print("[outcome]", outcome_reward)
                    print("[per_chunk_rewards]", per_chunk[:8])
                    print("[chunks]", len(spans))

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            return reward_tensor

        @staticmethod
        def _extract_reward_from_rm_scores(data, return_dict):
            # Mirror NaiveRewardManager's behaviour: if the batch already
            # has rm_scores from a reward model, surface those directly.
            if "rm_scores" not in data.batch.keys():
                return None
            rm_scores = data.batch["rm_scores"]
            if return_dict:
                return {"reward_tensor": rm_scores, "reward_extra_info": {}}
            return rm_scores

    return ThinkStreamPerChunkRewardManager


try:
    _register_reward_manager()
except Exception as e:  # noqa: BLE001
    logger.debug("ThinkStreamPerChunkRewardManager not registered: %s", e)
