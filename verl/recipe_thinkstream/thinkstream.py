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

    def _build_messages(self, example: dict):
        """Return messages for prompt-length filtering (doc2len).

        Mirrors RLHFDataset._build_messages but without placeholder
        replacement — our parquet already stores full content dicts.
        """
        messages: list = example.get(self.prompt_key, [])
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        return messages

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

        # Stash everything the reward fn needs into extra_info. Two row
        # shapes are supported:
        #
        #   1. Multi-Q trajectory shape (build_verl_parquet --multi_q,
        #      data_source=thinkstream_v12_streaming_multi_q): row carries
        #      `extra_info.questions` (List[Dict]) — the streaming agent
        #      loop injects each question's text at its ask_chunk and
        #      compute_score scores all questions independently.
        #
        #   2. Legacy (video, question) flatten shape: row carries single
        #      question/gold_answer/ask_chunks fields at the row level.
        #      The agent loop reads `question` as the trajectory-wide
        #      user input and triggers it at min(ask_chunks).
        extra = row_dict.get("extra_info", {}) or {}
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        if not isinstance(extra, dict):
            extra = {}

        # Multi-Q rows already carry `questions` + `gold_action_per_chunk`
        # inside extra_info — preserve them. Single-Q rows fill from the
        # row-level columns.
        #
        # Avoid `bool(extra.get("questions"))` — pyarrow round-trip wraps
        # List[Dict] in numpy.ndarray whose multi-element __bool__ raises
        # ValueError. Use length check via a small helper that also handles
        # `None` and accidental scalars.
        def _qs_len(v):
            if v is None:
                return 0
            try:
                return len(v)
            except TypeError:
                return 0

        questions_raw = extra.get("questions")
        is_multi_q = "questions" in extra and _qs_len(questions_raw) > 0

        if is_multi_q:
            # Normalize questions list (parquet may have stored as np array
            # of object-dtype dicts; nested fields like options/ask_chunks
            # may also be ndarrays).
            if hasattr(questions_raw, "tolist"):
                questions_raw = questions_raw.tolist()
            normalized_qs: list = []
            for q in questions_raw:
                if hasattr(q, "tolist"):
                    q = q.tolist()
                if not isinstance(q, dict):
                    continue
                # Coerce nested ndarray fields to plain lists so downstream
                # `if x:` / `len(x)` checks don't blow up.
                clean = {}
                for k, val in q.items():
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    clean[k] = val
                normalized_qs.append(clean)
            extra["questions"] = normalized_qs

            gap = extra.get("gold_action_per_chunk")
            if hasattr(gap, "tolist"):
                gap = gap.tolist()
            extra["gold_action_per_chunk"] = dict(gap) if gap else {}

            extra.update({
                "video_id": str(row_dict.get("video_id", "")),
                "video_path": str(row_dict.get("video_path", "")),
                "n_chunks": int(row_dict.get("n_chunks") or 0),
            })
        else:
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


# ---------------------------------------------------------------------------
# Multi-Q answer matching — re-exported from thinkstream.trainer.outcome_match
# so RL / SFT eval / OVOBench eval all dispatch through the SAME matchers
# (avoids the classic train/eval reward gap where RL rewards "Yes." but
# eval judges it wrong because eval uses strict `lower() == lower()`).
# ---------------------------------------------------------------------------
try:
    from thinkstream.trainer.outcome_match import (  # type: ignore
        normalize_answer as _normalize_answer,
        match_mcq_answer as _match_mcq_answer,
        match_binary as _match_binary,
        match_number as _match_number,
        match_short_exact as _match_short_exact,
        match_descriptive as _match_descriptive,
        score_outcome_by_form as _score_outcome_by_form,
        binary_polarity as _binary_polarity,
        extract_first_number as _extract_first_number,
        strip_articles as _strip_articles,
    )
except ImportError:
    # Defensive fallback only — if THINKSTREAM_HOME isn't set the matchers
    # below would also fail to load. This branch keeps the module importable
    # for unrelated test paths (e.g., parquet schema check).
    _normalize_answer = _match_mcq_answer = _match_binary = _match_number = (
        _match_short_exact
    ) = _match_descriptive = _score_outcome_by_form = lambda *a, **k: 0.0
    _binary_polarity = _extract_first_number = _strip_articles = (
        lambda *a, **k: None
    )


def _safe_list(v: Any) -> list:
    """Coerce a value (which may be numpy array, list, tuple, or None)
    into a plain Python list. Avoids `value or []` which raises on
    multi-element numpy arrays (ambiguous truth value)."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return []


def _model_action_from_turn(kind: str, text: str) -> str:
    if kind == "answer":
        m = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
        ans = m.group(1).strip() if m else ""
        return "silent" if not ans else "response"
    if kind == "recall":
        return "recall"
    if kind == "compress":
        return "compress"
    return "unknown"


def _per_chunk_action_avg(
    extra: Dict[str, Any],
    gold_action_per_chunk: Dict[str, str],
) -> Optional[float]:
    """Small action-shaping signal aligned to turn-local rollout metadata."""
    chunk_kinds = _safe_list(extra.get("ts_chunk_kinds"))
    if not chunk_kinds or not gold_action_per_chunk:
        return None

    chunk_texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    chunk_vidx = _safe_list(extra.get("ts_chunk_video_indices"))
    chunk_events = _safe_list(extra.get("ts_chunk_event_indices"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    scores: List[float] = []
    recall_seen_for_chunk: set[int] = set()
    compress_seen_for_chunk: set[int] = set()

    for turn_i, kind_raw in enumerate(chunk_kinds):
        kind = str(kind_raw or "unknown")
        turn_kind = (
            str(turn_kinds[turn_i] or "")
            if turn_i < len(turn_kinds)
            else ""
        )
        if turn_kind == "recall_response":
            # The recall-response assistant turn is conditioned on the prior
            # tool result; answer correctness/timing scores it. The action
            # decision to train here is the preceding recall tool_call.
            continue

        try:
            video_chunk_idx = int(chunk_vidx[turn_i]) if turn_i < len(chunk_vidx) else turn_i
        except (TypeError, ValueError):
            video_chunk_idx = turn_i
        if video_chunk_idx < 0 and turn_kind == "compress":
            try:
                video_chunk_idx = int(chunk_events[turn_i])
            except (IndexError, TypeError, ValueError):
                video_chunk_idx = -1
        if video_chunk_idx < 0:
            continue

        gold_action = str(
            (gold_action_per_chunk or {}).get(str(video_chunk_idx), "")
        )
        if not gold_action:
            continue

        text = chunk_texts[turn_i] if turn_i < len(chunk_texts) else ""
        model_action = _model_action_from_turn(kind, str(text or ""))

        if gold_action == "compress":
            if turn_kind == "compress":
                if model_action == "compress":
                    compress_seen_for_chunk.add(video_chunk_idx)
                    scores.append(0.1)
                else:
                    scores.append(-0.05)
            elif video_chunk_idx not in compress_seen_for_chunk:
                scores.append(-0.05)
            continue

        if gold_action in {"recall", "recall_silent"}:
            if model_action == "recall":
                recall_seen_for_chunk.add(video_chunk_idx)
                scores.append(0.1)
            elif (
                gold_action == "recall_silent"
                and model_action == "silent"
                and video_chunk_idx in recall_seen_for_chunk
            ):
                scores.append(0.1)
            elif video_chunk_idx not in recall_seen_for_chunk:
                scores.append(-0.05)
            continue

        scores.append(0.1 if model_action == gold_action else -0.05)

    if not scores:
        return None
    return sum(scores) / len(scores)


def _outcome_gate(parts: Dict[str, float]) -> float:
    """Scale positive shaping rewards by answer correctness.

    Timing/format/silent/action positives should not rescue a wrong answer,
    but a partially correct multi-question rollout still needs learning
    signal. Default gate is therefore the clipped outcome in [0, 1], not the
    old all-or-nothing ``outcome >= 1`` threshold. Set
    THINKSTREAM_OUTCOME_GATE_MODE=hard to recover the legacy gate.
    """
    try:
        outcome = float(parts.get("outcome", 0.0))
    except (TypeError, ValueError):
        outcome = 0.0
    if outcome != outcome:  # NaN guard.
        outcome = 0.0
    outcome = max(0.0, min(1.0, outcome))

    mode = os.environ.get("THINKSTREAM_OUTCOME_GATE_MODE", "soft").strip().lower()
    if mode in {"hard", "threshold", "legacy"}:
        try:
            threshold = float(
                os.environ.get("THINKSTREAM_OUTCOME_GATE_THRESHOLD", "1.0")
            )
        except ValueError:
            threshold = 1.0
        return 1.0 if outcome >= threshold else 0.0
    return outcome


def _combine_reward_parts(
    weights: Dict[str, float],
    parts: Dict[str, float],
) -> tuple[float, float]:
    """Combine reward components with correctness-scaled positive auxiliaries.

    Negative penalties always apply. Positive non-outcome rewards are scaled
    by the outcome gate so partial correctness receives partial auxiliary
    credit while wrong answers receive none.
    """
    gate = _outcome_gate(parts)
    outcome_total = float(weights.get("outcome", 0.0) * parts.get("outcome", 0.0))
    aux_total = 0.0
    for key, value in parts.items():
        if key == "outcome":
            continue
        weighted = float(weights.get(key, 0.0) * value)
        if weighted > 0:
            aux_total += gate * weighted
        else:
            aux_total += weighted
    return outcome_total + aux_total, gate


def _combine_multi_q_reward_parts(
    weights: Dict[str, float],
    per_question_parts: List[Dict[str, float]],
    trajectory_parts: Dict[str, float],
) -> tuple[float, float, List[float]]:
    """Combine multi-question rewards at question granularity.

    ``per_question_parts`` contains outcome/timing/silent_quality for each
    question. Each question gates its own positive auxiliary rewards, then the
    question scores are averaged. Trajectory-level components such as format
    and spam are applied once: positive trajectory auxiliaries are scaled by
    the mean per-question gate, while negative penalties always apply.
    """
    if not per_question_parts:
        total, gate = _combine_reward_parts(weights, trajectory_parts)
        return total, gate, []

    per_question_scores: List[float] = []
    per_question_gates: List[float] = []
    for q_parts in per_question_parts:
        q_score, q_gate = _combine_reward_parts(weights, q_parts)
        per_question_scores.append(q_score)
        per_question_gates.append(q_gate)

    total = sum(per_question_scores) / len(per_question_scores)
    gate = sum(per_question_gates) / len(per_question_gates)

    for key, value in trajectory_parts.items():
        weighted = float(weights.get(key, 0.0) * value)
        if weighted > 0:
            total += gate * weighted
        else:
            total += weighted
    return total, gate, per_question_scores


def _score_one_question(
    rewards: Dict[str, Any],
    *,
    q: Dict[str, Any],
    model_answer: str,
    answered_chunk: int,
) -> Dict[str, float]:
    """Score a single question's outcome + timing + silent decision.

    Returns dict with keys: outcome, timing, silent_quality, answered.
    `answered`=1 if the model produced any answer text for this Q.
    """
    options = _safe_list(q.get("options"))
    correct_option = q.get("correct_option", "")
    gold_answer = q.get("gold_answer", "") or ""
    answer_form = q.get("answer_form", "") or ""
    try:
        ask_chunk = int(q.get("ask_chunk", -1))
    except (TypeError, ValueError):
        ask_chunk = -1
    ask_chunks = _safe_list(q.get("ask_chunks"))
    if not ask_chunks and ask_chunk >= 0:
        ask_chunks = [ask_chunk]
    ask_chunks_int: List[int] = []
    for x in ask_chunks:
        try:
            ask_chunks_int.append(int(x))
        except (TypeError, ValueError):
            continue
    # answer_chunks is the FULL answerable window (silent_then_response: ask=5,
    # answer=25 → window must include 25 or model gets penalised for late).
    # Audit P1.3: visible_window had to bracket ask_chunks AND answer_chunks,
    # otherwise pass4-style cards with (ask=20, answer=55) get scored as
    # late even when model answers correctly at 55.
    answer_chunks = _safe_list(q.get("answer_chunks"))
    answer_chunks_int: List[int] = []
    for x in answer_chunks:
        try:
            answer_chunks_int.append(int(x))
        except (TypeError, ValueError):
            continue
    window_marks = ask_chunks_int + answer_chunks_int
    visible_start = min(window_marks) if window_marks else None
    visible_end = max(window_marks) if window_marks else None

    answered = 1.0 if (model_answer or "").strip() else 0.0

    # Outcome — form-aware liberal matching. Dispatches to the right
    # matcher for each of pass3a's 5 answer_form values (multiple_choice
    # / binary / number / short_exact / descriptive). Falls back to
    # v12_rewards.compute_outcome_reward_v12 only if our dispatcher
    # raises (defensive — should be a no-op in practice).
    if not answered:
        outcome = 0.0
    else:
        try:
            outcome = _score_outcome_by_form(
                model_answer,
                options=options,
                correct_option=correct_option,
                gold_answer=gold_answer,
                answer_form=answer_form,
            )
        except Exception:
            try:
                outcome = float(rewards["outcome"](
                    model_answer, gold_answer, answer_form=answer_form,
                ))
            except Exception:
                outcome = 0.0

    # Timing — bucket the answered_chunk vs the visible window
    # (now bracketed by both ask_chunks AND answer_chunks).
    timing = float(rewards["timing"](
        answered_chunk if answered_chunk >= 0 else None,
        visible_start, visible_end,
    ))

    # Silent quality — audit P1.6: per-Q silent decision.
    # compute_silent_quality_v12 takes (final_answer, gold_action, gold_answer)
    # where gold_action is what the model SHOULD have done at this chunk:
    #   - if answered_chunk is within the window → gold_action="response"
    #     (model correctly took the response slot)
    #   - if answered_chunk < visible_start (early) → gold_action="silent"
    #     (model should have stayed silent at this point — penalise)
    #   - if not answered (silent throughout) → gold_action="silent" only
    #     when the question never had a visible window, otherwise "response"
    if visible_start is None or visible_end is None:
        gold_action_for_silent = "response"
    elif answered_chunk < 0:
        # Never answered — if there was a window, model should have responded.
        gold_action_for_silent = "response"
    elif visible_start <= answered_chunk <= visible_end:
        # Answered in window → correct response slot.
        gold_action_for_silent = "response"
    elif answered_chunk < visible_start:
        # Model answered too early — at THIS chunk the gold action is silent.
        gold_action_for_silent = "silent"
    else:
        # Late answer (past visible_end) — gold was response.
        gold_action_for_silent = "response"
    try:
        silent_q = float(rewards["silent_quality"](
            model_answer if answered else None,
            gold_action_for_silent,
            gold_answer,
        ))
    except Exception:
        silent_q = 0.0

    return {
        "outcome": outcome,
        "timing": timing,
        "silent_quality": silent_q,
        "answered": answered,
    }


def _score_one_question_events(
    rewards: Dict[str, Any],
    *,
    q: Dict[str, Any],
    answer_events: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Score a question from all attributed non-empty answer events.

    This is needed for multi-emit questions (F5/PN1/F7). Empty answers are
    not passed in: they are ordinary silent chunks and must not close or
    satisfy a pending question.
    """
    if not answer_events:
        return _score_one_question(
            rewards, q=q, model_answer="", answered_chunk=-1,
        )

    events: List[Dict[str, Any]] = []
    for e in answer_events:
        if hasattr(e, "tolist"):
            e = e.tolist()
        if not isinstance(e, dict):
            continue
        text = str(e.get("text", "")).strip()
        if not text:
            continue
        try:
            chunk = int(e.get("chunk", -1))
        except (TypeError, ValueError):
            chunk = -1
        events.append({"chunk": chunk, "text": text})
    events.sort(key=lambda x: int(x.get("chunk", -1)))
    if not events:
        return _score_one_question(
            rewards, q=q, model_answer="", answered_chunk=-1,
        )

    answer_chunks = _safe_list(q.get("answer_chunks"))
    answer_chunks_int: List[int] = []
    for x in answer_chunks:
        try:
            answer_chunks_int.append(int(x))
        except (TypeError, ValueError):
            continue
    answer_chunks_int = sorted(answer_chunks_int)
    per_emit = _safe_list(q.get("per_emit_answers"))
    is_multi = len(answer_chunks_int) > 1 or len(per_emit) > 1
    if not is_multi:
        first = events[0]
        return _score_one_question(
            rewards,
            q=q,
            model_answer=str(first.get("text", "")),
            answered_chunk=int(first.get("chunk", -1)),
        )

    options = _safe_list(q.get("options"))
    correct_option = q.get("correct_option", "")
    gold_default = q.get("gold_answer", "") or ""
    answer_form = q.get("answer_form", "") or ""
    chunk_gold = {
        int(e["chunk"]): str(e.get("value", gold_default))
        for e in per_emit
        if isinstance(e, dict) and e.get("chunk") is not None
    }
    target_chunks = sorted(chunk_gold.keys() or answer_chunks_int)
    if not target_chunks:
        first = events[0]
        return _score_one_question(
            rewards,
            q=q,
            model_answer=str(first.get("text", "")),
            answered_chunk=int(first.get("chunk", -1)),
        )

    slack = 2
    used_event_idx: set[int] = set()
    outcome_scores: List[float] = []
    timing_scores: List[float] = []
    silent_scores: List[float] = []
    for i, emit_chunk in enumerate(target_chunks):
        lo = emit_chunk - slack
        hi = emit_chunk + slack
        if i + 1 < len(target_chunks):
            hi = min(hi, target_chunks[i + 1] - 1)
        found_idx = None
        for ei, ev in enumerate(events):
            if ei in used_event_idx:
                continue
            ev_chunk = int(ev.get("chunk", -1))
            if lo <= ev_chunk <= hi:
                found_idx = ei
                break
        if found_idx is None:
            outcome_scores.append(0.0)
            timing_scores.append(float(rewards["timing"](None, emit_chunk, hi)))
            silent_scores.append(float(rewards["silent_quality"](
                None, "response", gold_default,
            )))
            continue
        used_event_idx.add(found_idx)
        ev = events[found_idx]
        model_answer = str(ev.get("text", ""))
        ev_chunk = int(ev.get("chunk", -1))
        gold_for_emit = chunk_gold.get(emit_chunk, gold_default)
        outcome_scores.append(float(_score_outcome_by_form(
            model_answer,
            options=options,
            correct_option=correct_option,
            gold_answer=gold_for_emit,
            answer_form=answer_form,
        )))
        timing_scores.append(float(rewards["timing"](ev_chunk, emit_chunk, hi)))
        silent_scores.append(float(rewards["silent_quality"](
            model_answer, "response", gold_for_emit,
        )))

    return {
        "outcome": sum(outcome_scores) / len(outcome_scores),
        "timing": sum(timing_scores) / len(timing_scores),
        "silent_quality": sum(silent_scores) / len(silent_scores),
        "answered": 1.0 if used_event_idx else 0.0,
    }


def _compute_score_multi_q(
    rewards: Dict[str, Any],
    weights: Dict[str, float],
    questions: List[Dict[str, Any]],
    extra: Dict[str, Any],
    solution_str: str,
) -> Dict[str, float]:
    """Score a multi-Q trajectory. Aggregate per-Q rewards by mean."""
    n_q = len(questions)
    if n_q == 0:
        return {"score": 0.0, "outcome": 0.0, "timing": 0.0,
                "format": 0.0, "spam": 0.0, "silent_quality": 0.0,
                "n_questions": 0.0, "n_answered": 0.0}

    # Per-Q answer attribution from the agent loop's extra_fields.
    per_q_chunk_raw = _safe_list(extra.get("ts_per_q_answer_chunk"))
    per_q_text_raw = _safe_list(extra.get("ts_per_q_answer_text"))
    per_q_answers_raw = _safe_list(extra.get("ts_per_q_answers"))
    per_q_chunk = list(per_q_chunk_raw) + [-1] * (n_q - len(per_q_chunk_raw))
    per_q_text = list(per_q_text_raw) + [""] * (n_q - len(per_q_text_raw))
    per_q_answers = list(per_q_answers_raw) + [[]] * (n_q - len(per_q_answers_raw))

    # Per-Q scoring.
    per_q_outcome: List[float] = []
    per_q_timing: List[float] = []
    per_q_silent: List[float] = []
    per_q_parts: List[Dict[str, float]] = []
    n_answered = 0
    for q_idx, q in enumerate(questions):
        answer_events = per_q_answers[q_idx]
        if hasattr(answer_events, "tolist"):
            answer_events = answer_events.tolist()
        if isinstance(answer_events, (list, tuple)) and answer_events:
            sub = _score_one_question_events(
                rewards, q=q, answer_events=list(answer_events),
            )
        else:
            sub = _score_one_question(
                rewards,
                q=q,
                model_answer=str(per_q_text[q_idx] or ""),
                answered_chunk=int(per_q_chunk[q_idx]),
            )
        per_q_outcome.append(sub["outcome"])
        per_q_timing.append(sub["timing"])
        per_q_silent.append(sub["silent_quality"])
        per_q_parts.append({
            "outcome": float(sub["outcome"]),
            "timing": float(sub["timing"]),
            "silent_quality": float(sub["silent_quality"]),
        })
        if sub["answered"] > 0:
            n_answered += 1

    # Trajectory-level aggregates.
    avg_outcome = sum(per_q_outcome) / n_q
    avg_timing = sum(per_q_timing) / n_q
    avg_silent = sum(per_q_silent) / n_q

    # Format + spam are trajectory-level (not per-Q).
    chunks = _split_assistant_chunks(solution_str)
    try:
        fmt = float(rewards["format"](chunks))
    except Exception:
        fmt = 0.0
    tool_counts = _count_tool_calls(solution_str)
    try:
        spam = float(rewards["spam"](
            n_recall_calls=tool_counts["recall"],
            n_compress_calls=tool_counts["compress"],
        ))
    except Exception:
        spam = 0.0

    parts = {
        "outcome": avg_outcome,
        "timing": avg_timing,
        "format": fmt,
        "spam": spam,
        "silent_quality": avg_silent,
    }
    total, gate, per_q_scores = _combine_multi_q_reward_parts(
        weights,
        per_q_parts,
        {"format": fmt, "spam": spam},
    )
    gold_action_per_chunk = extra.get("gold_action_per_chunk") or {}
    if not gold_action_per_chunk and extra.get("video_id"):
        traj = _load_traj_index().get(str(extra["video_id"]))
        if traj:
            gold_action_per_chunk = traj.get("gold_action_per_chunk", {}) or {}
    action_avg = _per_chunk_action_avg(extra, gold_action_per_chunk)
    if action_avg is not None:
        alpha = float(extra.get("gdpo_alpha", 0.7))
        gated_state = action_avg if action_avg <= 0 else gate * action_avg
        total = alpha * total + (1.0 - alpha) * gated_state
        parts["per_chunk_action_avg"] = float(action_avg)

    action_space_errors = [
        str(x) for x in _safe_list(extra.get("ts_chunk_action_space_errors"))
        if str(x or "").strip()
    ]
    n_action_turns = max(1, len(_safe_list(extra.get("ts_chunk_kinds"))))
    illegal_action_rate = len(action_space_errors) / n_action_turns
    if illegal_action_rate:
        total -= 0.2 * illegal_action_rate
        parts["action_space"] = -illegal_action_rate
    else:
        parts["action_space"] = 0.0

    return {
        "score": total,
        **{k: float(v) for k, v in parts.items()},
        "outcome_gate": float(gate),
        "n_questions": float(n_q),
        "n_answered": float(n_answered),
        "per_q_outcome_min": float(min(per_q_outcome)),
        "per_q_outcome_max": float(max(per_q_outcome)),
        "per_q_reward_min": float(min(per_q_scores)) if per_q_scores else 0.0,
        "per_q_reward_max": float(max(per_q_scores)) if per_q_scores else 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Reward function called by verl's NaiveRewardManager.

    Two modes, switched by data_source:
      - thinkstream_v12_streaming_multi_q: multi-Q trajectory; score every
        question independently, aggregate by mean. Aligns with OVOBench
        eval form.
      - thinkstream_v12_streaming (legacy): single (video, question)
        flatten; score one question with v12 5-component reward.

    Returns a dict so verl logs per-component rewards to wandb:
        {"score": <total>, "outcome": ..., "timing": ..., "format": ...,
         "spam": ..., "silent_quality": ...}
    """
    rewards, weights = _load_thinkstream_rewards()
    if not rewards:
        return {"score": 0.0, "outcome": 0.0, "timing": 0.0,
                "format": 0.0, "spam": 0.0, "silent_quality": 0.0}

    extra = extra_info or {}

    # ── Multi-Q dispatch ──
    # Parquet round-trip wraps List[Dict] columns in numpy.ndarray, which
    # raises ValueError on bool() when multi-element. Coerce to plain list
    # length-check explicitly before deciding the branch.
    def _list_len(x: Any) -> int:
        try:
            return len(x) if x is not None else 0
        except TypeError:
            return 0

    questions_in_extra = extra.get("questions")
    gt_dict_for_multi_q = _coerce_ground_truth(ground_truth)
    questions_in_gt = gt_dict_for_multi_q.get("questions")
    has_extra_qs = _list_len(questions_in_extra) > 0
    has_gt_qs = _list_len(questions_in_gt) > 0
    is_multi_q = (
        data_source == "thinkstream_v12_streaming_multi_q"
        or has_extra_qs
        or has_gt_qs
    )
    if is_multi_q:
        # Prefer extra_info.questions (live from dataset); fall back to
        # ground_truth's encoded copy (for cases where extra was stripped).
        questions = questions_in_extra if has_extra_qs else questions_in_gt
        # Coerce numpy-wrapped dicts to plain dicts.
        norm: List[Dict[str, Any]] = []
        for q in questions:
            if hasattr(q, "tolist"):
                q = q.tolist()
            if isinstance(q, dict):
                norm.append({k: q[k] for k in q.keys()})
        return _compute_score_multi_q(
            rewards, weights, norm, extra, solution_str,
        )

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
    options = gt.get("options") or extra.get("options") or []
    correct_option = (
        gt.get("correct_option")
        if gt.get("correct_option") is not None
        else extra.get("correct_option", "")
    )
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
            final_answer,
            gold_answer,
            answer_form=answer_form,
            options=options,
            correct_option=correct_option,
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

    total, gate = _combine_reward_parts(weights, parts)

    action_avg = _per_chunk_action_avg(extra, gold_action_per_chunk)
    if action_avg is not None:
        alpha = float(extra.get("gdpo_alpha", 0.7))
        gated_state = action_avg if action_avg <= 0 else gate * action_avg
        total = alpha * total + (1.0 - alpha) * gated_state
        parts["per_chunk_action_avg"] = float(action_avg)

    action_space_errors = [
        str(x) for x in _safe_list(extra.get("ts_chunk_action_space_errors"))
        if str(x or "").strip()
    ]
    n_action_turns = max(1, len(_safe_list(extra.get("ts_chunk_kinds"))))
    illegal_action_rate = len(action_space_errors) / n_action_turns
    if illegal_action_rate:
        total -= 0.2 * illegal_action_rate
        parts["action_space"] = -illegal_action_rate
    else:
        parts["action_space"] = 0.0

    # NaiveRewardManager places ONE scalar at the trajectory's last
    # assistant token (verl 0.4 reward_loop framework). Per-chunk shaping
    # has already been folded into `total` via the GDPO α-mix above —
    # we don't return a separate per-chunk vector because there's no
    # per-token broadcast hook in the new framework.
    return {
        "score": total,
        **{k: float(v) for k, v in parts.items()},
        "outcome_gate": float(gate),
    }


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
