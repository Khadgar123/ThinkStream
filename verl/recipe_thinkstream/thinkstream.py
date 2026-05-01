# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# verl recipe entry point for ThinkStream streaming-video GRPO.
#
# Mirrors recipe/deepeyes/deepeyes.py:
#   - CustomRLHFDataset reads ThinkStream pass5 trajectory JSONL
#     (train_rl_trajectories.jsonl) and emits verl-format rows.
#   - compute_score wraps thinkstream.trainer.v12_rewards into the
#     scalar (data_source, solution_str, ground_truth, extra_info) -> float
#     signature verl's main_ppo expects.
#
# The ThinkStream package is imported via PYTHONPATH (see
# run_thinkstream_grpo.sh — THINKSTREAM_HOME is prepended). We do not
# vendor ThinkStream code into this fork.
from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None  # noqa: N816  — only the dataset path needs PIL.

# Side-effect import: registers ThinkStreamStreamingAgentLoop under
# `thinkstream_streaming_agent` in verl's _agent_loop_registry. verl loads
# THIS file at training start (via custom_cls.path / custom_reward_function.path),
# so the agent loop is wired in automatically without any extra config.
#
# IMPORTANT: the recipe dir is named `recipe_thinkstream/` (NOT
# `thinkstream/`). Naming it `thinkstream/` would shadow the real
# ThinkStream Python package on PYTHONPATH. The relative import below
# only works when this module is loaded as part of the recipe_thinkstream
# package; the importlib fallback handles the spec_from_file_location
# path hydra uses without polluting sys.path.
def _side_effect_import(sibling_module: str, alias: str):
    """Import a sibling module of this recipe with explicit naming so we
    don't pollute sys.path. Triggers @register decorators."""
    try:
        # Works when this file is loaded as recipe_thinkstream.thinkstream
        import importlib
        importlib.import_module(f".{sibling_module}", package=__package__)
        return
    except Exception:
        pass
    try:
        import os as _os
        import importlib.util as _ilu
        _path = _os.path.join(_os.path.dirname(__file__), f"{sibling_module}.py")
        _spec = _ilu.spec_from_file_location(alias, _path)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
    except Exception:
        pass


# Register the sibling streaming_agent_loop module so its @register
# decorator runs and the `thinkstream_streaming_agent` AgentLoop key is
# wired into verl's _agent_loop_registry.
# (We use the built-in `naive` reward_manager from verl.experimental.
# reward_loop — no custom one needed; per-chunk shaping is folded into
# compute_score's returned `score`.)
_side_effect_import("streaming_agent_loop", "thinkstream_recipe_streaming_agent_loop")

# verl is imported lazily inside CustomRLHFDataset so that compute_score
# can be exercised without the full verl/ray runtime (e.g., in unit tests).

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of ThinkStream reward functions.
#
# Why lazy: verl spawns Ray workers; each one imports this module. If
# THINKSTREAM_HOME is not set on a worker we fall back to a degenerate
# scorer that returns 0.0 instead of crashing the whole job.
# ---------------------------------------------------------------------------
_ts_rewards = None
_ts_weights = None


def _load_thinkstream_rewards():
    global _ts_rewards, _ts_weights
    if _ts_rewards is not None:
        return _ts_rewards, _ts_weights
    try:
        from thinkstream.trainer.v12_rewards import (  # type: ignore
            compute_outcome_reward_v12,
            compute_timing_reward_v12,
            compute_format_reward_v12,
            compute_spam_score_v12,
            compute_silent_quality_v12,
        )
        from thinkstream.trainer.gdpo_advantage import (  # type: ignore
            V12_DEFAULT_REWARD_WEIGHTS,
        )
        _ts_rewards = {
            "outcome": compute_outcome_reward_v12,
            "timing": compute_timing_reward_v12,
            "format": compute_format_reward_v12,
            "spam": compute_spam_score_v12,
            "silent_quality": compute_silent_quality_v12,
        }
        _ts_weights = dict(V12_DEFAULT_REWARD_WEIGHTS)
    except Exception as e:
        logger.warning(
            "ThinkStream reward import failed (%s). "
            "Set THINKSTREAM_HOME and re-run. Returning 0.0 from compute_score.",
            e,
        )
        _ts_rewards = {}
        _ts_weights = {}
    return _ts_rewards, _ts_weights


# ---------------------------------------------------------------------------
# Trajectory JSONL loader (cached at process level).
# ---------------------------------------------------------------------------
_TRAJ_INDEX: Optional[Dict[str, Dict[str, Any]]] = None


def _load_traj_index() -> Dict[str, Dict[str, Any]]:
    """Index pass5 trajectories by video_id for ground_truth lookup at scoring time."""
    global _TRAJ_INDEX
    if _TRAJ_INDEX is not None:
        return _TRAJ_INDEX
    path = os.environ.get("THINKSTREAM_TRAJ_INDEX_PATH")
    if not path or not Path(path).exists():
        _TRAJ_INDEX = {}
        return _TRAJ_INDEX
    idx: Dict[str, Dict[str, Any]] = {}
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = row.get("video_id") or row.get("trajectory_id")
            if vid:
                idx[str(vid)] = row
    _TRAJ_INDEX = idx
    return idx


def _get_base_rlhf_dataset():
    from verl.utils.dataset.rl_dataset import RLHFDataset  # type: ignore
    return RLHFDataset


# Defer subclassing until import time of a verl-available environment.
# Hydra resolves `custom_cls.path/name` by importlib + getattr, which calls
# this module — at that point verl is on PYTHONPATH and the import succeeds.
try:
    _RLHFDataset = _get_base_rlhf_dataset()
except Exception:  # noqa: BLE001 — local dev without verl installed.
    _RLHFDataset = object


class CustomRLHFDataset(_RLHFDataset):  # type: ignore[misc, valid-type]
    """ThinkStream trajectory dataset for verl 0.4 agent-loop mode.

    verl 0.4 moved chat-template + tokenization into the AgentLoop side
    (see verl/utils/dataset/rl_dataset.py:359). The dataset MUST return
    only `raw_prompt` + a dummy tensor + the data_source / reward_model /
    extra_info passthroughs. Returning input_ids / attention_mask /
    position_ids would conflict with rollout's gen_batch_output union
    (see verl/trainer/ppo/ray_trainer.py:1411 — same-name keys collide).

    Each row is one (video, question) seed produced by
    scripts/agent_data_v5/build_verl_parquet.py. Columns expected:
      prompt:                 List[Dict] — [system, user(question)]
      video_id:               str
      video_path:             str
      question:               str
      gold_answer:            str
      answer_form:            str
      ask_chunks:             List[int]
      gold_action_per_chunk:  Dict[str, str]    (already filtered to this
                                                 question's chunk range)
      n_chunks:               int
      extra_info:             Dict
      reward_model:           {"ground_truth": JSON-string, "style": ...}
      data_source:            "thinkstream_v12_streaming"
    """

    def __getitem__(self, item):
        import torch  # type: ignore
        row_dict: dict = self.dataframe[item]

        # raw_prompt is the chat-format messages. Parquet stored it as
        # numpy array of {role, content}; ensure plain Python list-of-dict.
        prompt = row_dict.get(self.prompt_key, [])
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        # Each element may be a dict already; force plain dicts.
        prompt = [
            {"role": m.get("role"), "content": m.get("content")}
            if isinstance(m, dict) else m
            for m in prompt
        ]
        row_dict["raw_prompt"] = prompt

        # Dummy tensor — DataProto.batch can't be empty; verl removes
        # this constraint after the TensorDict migration but until then
        # we follow the upstream convention.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # Stash everything the reward fn needs into extra_info. The
        # streaming agent loop reads video_id/video_path/n_chunks from
        # here every chunk; compute_score reads gold_answer/ask_chunks/
        # gold_action_per_chunk for shaping.
        extra = row_dict.get("extra_info", {}) or {}
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        if not isinstance(extra, dict):
            extra = {}
        extra.update({
            "video_id": str(row_dict.get("video_id", "")),
            "video_path": str(row_dict.get("video_path", "")),
            "question": str(row_dict.get("question", "")),
            "gold_answer": str(row_dict.get("gold_answer", "")),
            "answer_form": str(row_dict.get("answer_form", "")),
            "ask_chunks": list(row_dict.get("ask_chunks") or []),
            "gold_action_per_chunk": dict(row_dict.get("gold_action_per_chunk") or {}),
            "n_chunks": int(row_dict.get("n_chunks") or 0),
        })
        row_dict["extra_info"] = extra
        row_dict["index"] = extra.get("index", str(row_dict.get("video_id", "")))
        row_dict["tools_kwargs"] = extra.get("tools_kwargs", {}) or {}
        row_dict["interaction_kwargs"] = extra.get("interaction_kwargs", {}) or {}

        # agent_name must match the @register key on our AgentLoop class.
        row_dict["agent_name"] = "thinkstream_streaming_agent"
        return row_dict


# ---------------------------------------------------------------------------
# Reward function — verl scalar signature.
# ---------------------------------------------------------------------------
def _split_assistant_chunks(solution_str: str) -> List[str]:
    """Split the concatenated assistant rollout into per-turn outputs.

    verl's reward_manager decodes responses with skip_special_tokens=True,
    so `<|im_end|>`/`<|im_start|>` are stripped before reaching us. Each
    v12 assistant turn opens with `<think>` and closes with either
    `</answer>` or `</tool_call>`. We split between turns by finding each
    `<think>` opening; the chunk extends to (but excludes) the next
    `<think>`. Degenerate output (no `<think>`) is returned as a single
    chunk so downstream code still has something to score.
    """
    s = solution_str.strip()
    if not s:
        return []
    if "<think>" not in s:
        return [s]
    starts = [m.start() for m in re.finditer(r"<think>", s)]
    chunks: List[str] = []
    # Anything before the first <think> (e.g., stray bos text) is dropped.
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(s)
        chunk = s[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _extract_final_answer(solution_str: str) -> Optional[str]:
    """The trajectory's final answer is the LAST <answer>...</answer>."""
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _count_tool_calls(solution_str: str) -> Dict[str, int]:
    n_recall = 0
    n_compress = 0
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", solution_str, re.S):
        try:
            tc = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = tc.get("name", "")
        if name == "recall":
            n_recall += 1
        elif name == "compress":
            n_compress += 1
    return {"recall": n_recall, "compress": n_compress}


def _coerce_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    """verl's reward_model.ground_truth comes in as either a JSON-encoded
    string (our parquet builder writes it that way to round-trip nested
    fields through pyarrow) or a dict (if a future config switches). Handle
    both."""
    if isinstance(ground_truth, dict):
        return dict(ground_truth)
    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"gold_answer": ground_truth}
    return {"gold_answer": str(ground_truth) if ground_truth is not None else ""}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Reward function called by verl's NaiveRewardManager.

    Returns a dict so verl logs per-component rewards to wandb:
        {"score": <total>, "outcome": ..., "timing": ..., "format": ...,
         "spam": ..., "silent_quality": ...}

    Aggregates the 5 v12 components with V12_DEFAULT_REWARD_WEIGHTS:
        outcome / timing / format / spam (NEGATIVE weight) / silent_quality.

    `ground_truth` is whatever the dataset wrote into the `reward_model`
    column; we accept either a JSON-encoded bundle (the parquet builder
    writes this) or a dict (a future config might pass directly).
    """
    rewards, weights = _load_thinkstream_rewards()
    if not rewards:
        return {"score": 0.0, "outcome": 0.0, "timing": 0.0,
                "format": 0.0, "spam": 0.0, "silent_quality": 0.0}

    extra = extra_info or {}

    # Fall back to the trajectory index if extra_info doesn't carry the bundle
    # (e.g., when verl strips dict columns down to scalars at parquet load).
    if not extra.get("gold_action_per_chunk") and extra.get("video_id"):
        idx = _load_traj_index()
        traj = idx.get(str(extra["video_id"]))
        if traj:
            extra.setdefault(
                "gold_action_per_chunk", traj.get("gold_action_per_chunk", {})
            )
            extra.setdefault("ask_chunks", [])

    gt = _coerce_ground_truth(ground_truth)
    gold_answer = gt.get("gold_answer", "") or extra.get("gold_answer", "")
    answer_form = gt.get("answer_form", "") or extra.get("answer_form", "")
    ask_chunks = gt.get("ask_chunks") or extra.get("ask_chunks") or []
    visible_start = gt.get("visible_start_chunk")
    visible_end = gt.get("visible_end_chunk")
    if visible_start is None and ask_chunks:
        visible_start = min(ask_chunks)
    if visible_end is None and ask_chunks:
        visible_end = max(ask_chunks)
    gold_action_per_chunk = (
        gt.get("gold_action_per_chunk")
        or extra.get("gold_action_per_chunk")
        or {}
    )

    chunks = _split_assistant_chunks(solution_str)
    final_answer_inferred = _extract_final_answer(solution_str)
    # Prefer the rollout-emitted answer text + chunk over decoding the
    # tokens. The agent loop has the authoritative state and writes both
    # into extra_fields (which verl funnels into extra_info via
    # rollout_reward_scores merge). num_turns is the LAST resort because
    # it counts ALL assistant turns including silent ones, so it drifts
    # from the actual answer chunk.
    final_answer: Optional[str]
    rollout_final_answer = extra.get("ts_final_answer")
    if isinstance(rollout_final_answer, str) and rollout_final_answer.strip():
        final_answer = rollout_final_answer.strip()
    else:
        final_answer = final_answer_inferred

    answer_chunk: Optional[int] = None
    rollout_answer_chunk = extra.get("ts_answer_chunk")
    try:
        rollout_answer_chunk_int = int(float(rollout_answer_chunk)) if rollout_answer_chunk is not None else -1
    except (TypeError, ValueError):
        rollout_answer_chunk_int = -1

    if rollout_answer_chunk_int >= 0:
        answer_chunk = rollout_answer_chunk_int
    elif final_answer is not None:
        # Fallback when rollout didn't surface the field (e.g., reward fn
        # invoked outside the agent loop). num_turns includes the initial
        # system+user turn at index 0, so answer_chunk = num_turns - 2 if
        # we have it; else scan chunks.
        n_turns = extra.get("num_turns")
        if isinstance(n_turns, int) and n_turns >= 2:
            answer_chunk = n_turns - 2
        else:
            for idx, chunk in enumerate(chunks):
                if re.search(r"<answer>(.+?)</answer>", chunk, re.DOTALL):
                    answer_chunk = idx
    tool_counts = _count_tool_calls(solution_str)

    parts: Dict[str, float] = {}
    try:
        parts["outcome"] = rewards["outcome"](
            final_answer, gold_answer, answer_form=answer_form
        )
        parts["timing"] = rewards["timing"](answer_chunk, visible_start, visible_end)
        parts["format"] = rewards["format"](chunks)
        parts["spam"] = rewards["spam"](
            n_recall_calls=tool_counts["recall"],
            n_compress_calls=tool_counts["compress"],
        )
        gold_action = ""
        if answer_chunk is not None:
            gold_action = (gold_action_per_chunk or {}).get(str(answer_chunk), "")
        parts["silent_quality"] = rewards["silent_quality"](
            final_answer, gold_action, gold_answer
        )
    except Exception as e:
        logger.warning("v12 reward component failed: %s", e)
        return {"score": 0.0, "outcome": 0.0, "timing": 0.0,
                "format": 0.0, "spam": 0.0, "silent_quality": 0.0}

    total = float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))

    # ── Per-chunk action reward (P1.5).
    # The streaming agent loop drops `ts_chunk_kinds` (a list[str] of
    # parsed action kinds for each chunk) and `ts_chunk_asst_texts` into
    # extra_fields → reward_manager surfaces them via extra_info. Compare
    # each chunk's parsed kind against gold_action_per_chunk[chunk_idx]
    # and emit a per-chunk shaping reward. This goes through to the
    # reward_manager which broadcasts to that chunk's last assistant
    # token position.
    chunk_kinds = extra.get("ts_chunk_kinds") or []
    chunk_texts = extra.get("ts_chunk_asst_texts") or []
    # P1.7 fix (post-review): the agent loop's chunk_kinds includes
    # inter-chunk compress turns that don't consume a video chunk_idx.
    # Use ts_chunk_video_indices (-1 = compress) as the authoritative
    # mapping from turn-index → video chunk_idx, instead of enumerate.
    chunk_vidx = extra.get("ts_chunk_video_indices") or []
    per_chunk_action: List[float] = []
    if chunk_kinds and gold_action_per_chunk:
        for turn_i, kind in enumerate(chunk_kinds):
            video_chunk_idx = (
                int(chunk_vidx[turn_i]) if turn_i < len(chunk_vidx) else turn_i
            )
            if video_chunk_idx < 0:
                # Compress inter-turn — system event, no video gold.
                per_chunk_action.append(0.0)
                continue
            gold_action = (gold_action_per_chunk or {}).get(str(video_chunk_idx), "")
            if not gold_action:
                per_chunk_action.append(0.0)
                continue
            # Map model output to canonical action label. (Use turn_i
            # for chunk_texts indexing since chunk_texts is parallel to
            # chunk_kinds, not to video_chunk_idx.)
            if kind == "answer":
                txt = chunk_texts[turn_i] if turn_i < len(chunk_texts) else ""
                m = re.search(r"<answer>(.*?)</answer>", txt, re.DOTALL)
                ans = m.group(1).strip() if m else ""
                model_action = "silent" if not ans else "response"
            elif kind == "recall":
                model_action = "recall"
            elif kind == "compress":
                model_action = "compress"
            else:
                model_action = "unknown"
            # Symmetric per-chunk shaping. Match → +0.1, mismatch → -0.05.
            # Calibrated so that 360 chunks of all-correct contributes at
            # most +36 to the total reward — comparable scale to outcome*1.0.
            if model_action == gold_action:
                per_chunk_action.append(0.1)
            else:
                per_chunk_action.append(-0.05)
        # Average state reward folded into the trajectory-level scalar so
        # plain GRPO (no token-level broadcast) still benefits.
        if per_chunk_action:
            state_avg = sum(per_chunk_action) / len(per_chunk_action)
            # GDPO mix (P1.4): α=0.7 outcome + (1-α)=0.3 state.
            alpha = float(extra.get("gdpo_alpha", 0.7))
            total = alpha * total + (1.0 - alpha) * state_avg
            parts["per_chunk_action_avg"] = state_avg

    # NaiveRewardManager places ONE scalar at the trajectory's last
    # assistant token (verl 0.4 reward_loop framework). Per-chunk shaping
    # has already been folded into `total` via the GDPO α-mix above —
    # we don't return a separate per-chunk vector because there's no
    # per-token broadcast hook in the new framework.
    return {"score": total, **{k: float(v) for k, v in parts.items()}}


if __name__ == "__main__":
    # Mirrors verl's reward_manager (skip_special_tokens=True) — there are
    # no <|im_*|> markers in solution_str.
    sample = (
        "<think>chunk 0 silent</think>"
        "<think>chunk 1 final</think><answer>yes</answer>"
    )
    gt = json.dumps({"gold_answer": "yes", "answer_form": "binary",
                     "ask_chunks": [1], "gold_action_per_chunk": {"1": "response"}})
    print("score:", compute_score("thinkstream_v12_streaming", sample, gt,
                                  {"num_turns": 2}))
