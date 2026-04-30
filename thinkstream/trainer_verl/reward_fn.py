"""verl-format reward function for ThinkStream v12 streaming agent.

Wraps the framework-agnostic helpers in `thinkstream.trainer.v12_rewards`
and `thinkstream.trainer.gdpo_advantage` into the signature verl expects:

    reward_fn(data_source, solution_str, ground_truth, extra_info) -> float

verl 0.4+ multi-turn signature (used here):

    reward_fn(messages, ground_truth, extra_info) -> Dict[str, float]
        # messages: full multi-turn conversation
        # ground_truth: gold_answer + ask_chunks + visibility info
        # extra_info: {video_uid, chunk_idx, recall_returned, n_recall, n_compress}
        # returns: per-component rewards dict (verl handles aggregation)

This file is PURE adapter — algorithm correctness lives in v12_rewards.py
and is exercised by tests/test_v12_rewards.py. Cross-validation against
the slyme path runs the same v12_rewards functions; if both paths produce
the same advantage vector for the same input, the orchestration layer is
not introducing bugs.
"""
from __future__ import annotations

import re
import json
from typing import Dict, List, Optional, Any

from thinkstream.trainer.v12_rewards import (
    compute_outcome_reward_v12,
    compute_timing_reward_v12,
    compute_format_reward_v12,
    compute_spam_score_v12,
    compute_silent_quality_v12,
)
from thinkstream.trainer.gdpo_advantage import (
    V12_REWARD_DICT_KEYS,
    V12_DEFAULT_REWARD_WEIGHTS,
)


def _parse_chunk_outputs(messages: List[Dict]) -> List[str]:
    """Extract per-chunk assistant texts from a verl-format message list."""
    return [
        m["content"]
        for m in messages
        if m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
    ]


def _count_tool_calls(assistant_outputs: List[str]) -> Dict[str, int]:
    """Tally recall / compress tool_call invocations across the trajectory."""
    n_recall = 0
    n_compress = 0
    for out in assistant_outputs:
        for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', out, re.S):
            try:
                tc = json.loads(m.group(1))
                name = tc.get("name", "")
                if name == "recall":
                    n_recall += 1
                elif name == "compress":
                    n_compress += 1
            except Exception:
                pass
    return {"recall": n_recall, "compress": n_compress}


def compute_thinkstream_reward(
    messages: List[Dict],
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Production reward function — verl-multiturn signature.

    Args:
      messages: full conversation including tool turns. Last assistant turn
                is the trajectory's final emission.
      ground_truth: {
          "gold_answer": str,                    # final answer to compare
          "answer_form": str,                    # "binary"|"number"|"mc"|"short_exact"|...
          "ask_chunks": List[int],               # chunks where question fires
          "visible_start_chunk": Optional[int],  # earliest answerable chunk
          "visible_end_chunk":   Optional[int],  # latest expected answer chunk
          "gold_action_per_chunk": Dict[str,str],# {chunk_idx → "silent"/"response"/...}
      }
      extra_info: {
          "answer_chunk":  Optional[int],   # which chunk emitted final answer
          "final_answer":  Optional[str],   # the answer text
      }

    Returns: dict with the 5 reward components — verl aggregates externally
             via the configured weights.
    """
    extra_info = extra_info or {}
    ai_outputs = _parse_chunk_outputs(messages)
    tool_counts = _count_tool_calls(ai_outputs)

    final_answer = extra_info.get("final_answer")
    answer_chunk = extra_info.get("answer_chunk")
    gold_answer = ground_truth.get("gold_answer", "")
    answer_form = ground_truth.get("answer_form", "")

    outcome = compute_outcome_reward_v12(
        final_answer, gold_answer, answer_form=answer_form,
    )
    timing = compute_timing_reward_v12(
        answer_chunk,
        ground_truth.get("visible_start_chunk"),
        ground_truth.get("visible_end_chunk"),
    )
    format_ok = compute_format_reward_v12(ai_outputs)
    spam = compute_spam_score_v12(
        n_recall_calls=tool_counts["recall"],
        n_compress_calls=tool_counts["compress"],
    )
    gold_action = ""
    if answer_chunk is not None:
        gold_action = ground_truth.get("gold_action_per_chunk", {}).get(
            str(answer_chunk), ""
        )
    silent_q = compute_silent_quality_v12(
        final_answer, gold_action, gold_answer,
    )

    return {
        "outcome":        outcome,
        "timing":         timing,
        "format":         format_ok,
        "spam":           spam,           # weight is NEGATIVE in V12_DEFAULT_REWARD_WEIGHTS
        "silent_quality": silent_q,
    }


def compute_thinkstream_reward_scalar(
    messages: List[Dict],
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Scalar fallback for verl single-reward APIs.

    Weighted sum using V12_DEFAULT_REWARD_WEIGHTS. Same signal as the
    dict-returning version; verl's GRPO advantage code does the group-norm
    after this point.
    """
    weights = weights or V12_DEFAULT_REWARD_WEIGHTS
    parts = compute_thinkstream_reward(messages, ground_truth, extra_info)
    return sum(weights.get(k, 0.0) * v for k, v in parts.items())
