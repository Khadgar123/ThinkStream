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


# Register sibling modules whose @register decorators verl needs at runtime:
#   streaming_agent_loop → registers `thinkstream_streaming_agent` agent loop
#   reward_manager       → registers `thinkstream_per_chunk` reward manager
_side_effect_import("streaming_agent_loop", "thinkstream_recipe_streaming_agent_loop")
_side_effect_import("reward_manager", "thinkstream_recipe_reward_manager")

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
    """ThinkStream trajectory dataset.

    Each row is one (video, question) seed. The pass5 JSONL has shape:
        {
          "video_id": str,
          "video_path": str,
          "questions": [
              {"question": str, "gold_answer": str, "answer_form": str,
               "ask_chunks": [int], ...},
          ],
          "gold_action_per_chunk": {str: str},
          "stats": {"n_chunks_covered": int, ...},
        }

    We emit verl's expected dict with `prompt` (system + user), an `images`
    list (empty — frames are loaded per-chunk by the rollout adapter), and
    an `extra_info` bundle that the reward function reads back at scoring time.
    """

    def __getitem__(self, item):
        # Lazy verl imports — this method only runs in a verl runtime.
        import verl.utils.torch_functional as verl_F  # type: ignore
        from verl.utils.model import compute_position_id_with_mask  # type: ignore

        row_dict: dict = self.dataframe[item]

        # Pull the question out of the source row. We expect the upstream
        # parquet builder (scripts/agent_data_v5/build_verl_parquet.py) to
        # flatten one (video, question) into one parquet row with columns:
        # prompt, video_id, question, gold_answer, answer_form, ask_chunks,
        # gold_action_per_chunk, n_chunks.
        from thinkstream.data.agent_protocol import (  # type: ignore
            SYSTEM_PROMPT_V12,
        )

        question_text = row_dict.get("question", "")
        row_dict[self.prompt_key] = [
            {"role": "system", "content": SYSTEM_PROMPT_V12},
            {"role": "user", "content": question_text},
        ]

        images: List[Image.Image] = []
        row_dict_images = row_dict.get(self.image_key, None)
        if row_dict_images:
            images = [
                Image.open(io.BytesIO(image["bytes"])) for image in row_dict_images
            ]
        messages = self._build_messages(row_dict)

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.processor(
                text=[raw_prompt], images=images or None, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Qwen3-VL uses the Qwen2VL image processor internals; rope index
        # pathway matches deepeyes.
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} > {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # Stash everything the reward fn needs into extra_info.
        extra = row_dict.get("extra_info", {}) or {}
        extra.update(
            {
                "video_id": row_dict.get("video_id", ""),
                "video_path": row_dict.get("video_path", ""),
                "question": question_text,
                "gold_answer": row_dict.get("gold_answer", ""),
                "answer_form": row_dict.get("answer_form", ""),
                "ask_chunks": row_dict.get("ask_chunks", []),
                "gold_action_per_chunk": row_dict.get(
                    "gold_action_per_chunk", {}
                ),
                "n_chunks": row_dict.get("n_chunks", 0),
            }
        )
        row_dict["extra_info"] = extra
        row_dict["index"] = extra.get("index", row_dict.get("video_id", ""))
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
    per_chunk_action: List[float] = []
    if chunk_kinds and gold_action_per_chunk:
        for chunk_idx, kind in enumerate(chunk_kinds):
            gold_action = (gold_action_per_chunk or {}).get(str(chunk_idx), "")
            if not gold_action:
                per_chunk_action.append(0.0)
                continue
            # Map model output to canonical action label.
            if kind == "answer":
                txt = chunk_texts[chunk_idx] if chunk_idx < len(chunk_texts) else ""
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

    result = {"score": total, **{k: float(v) for k, v in parts.items()}}
    # The reward_manager will pick this up and broadcast to per-chunk
    # token positions. Keys prefixed with `_` are passthroughs that don't
    # show up as wandb scalars.
    if per_chunk_action:
        result["_per_chunk_action_rewards"] = per_chunk_action
    return result


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
