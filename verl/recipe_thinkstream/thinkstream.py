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
        is_multi_q = "questions" in extra and extra.get("questions")

        if is_multi_q:
            # Normalize questions list (parquet may have stored as np array).
            questions = extra.get("questions") or []
            if hasattr(questions, "tolist"):
                questions = questions.tolist()
            normalized_qs: list = []
            for q in questions:
                if hasattr(q, "tolist"):
                    q = q.tolist()
                if not isinstance(q, dict):
                    continue
                normalized_qs.append({k: q[k] for k in q.keys()})
            extra["questions"] = normalized_qs

            gap = extra.get("gold_action_per_chunk") or {}
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
# Multi-Q answer matching — used by both compute_score multi-Q branch and
# the OVOBench eval. Liberal matching: option letter / option text / fuzzy.
# ---------------------------------------------------------------------------
def _normalize_answer(s: str) -> str:
    """Lowercase + strip + collapse whitespace + drop trailing punctuation."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.\,\!\?\:\;\)\]\}]+$", "", s)
    s = re.sub(r"^[\(\[\{]+", "", s)
    return s


def _match_mcq_answer(
    model_answer: str,
    options: List[str],
    correct_option: str,
    gold_answer: str = "",
) -> bool:
    """Liberal MCQ answer match. Accepts:
      - Option letter (A/B/C/D), case-insensitive, optional trailing
        punctuation. ``correct_option`` may be 0-indexed int or letter.
      - Full option text matching one of the options exactly (modulo
        whitespace + punctuation), then check that option is the correct one.
      - Direct match against gold_answer text (substring or equality).

    Returns True iff any of the above strategies says the model picked
    the correct option.
    """
    if not model_answer:
        return False
    ma = _normalize_answer(model_answer)
    if not ma:
        return False

    # Resolve the correct option's letter + text.
    correct_idx: Optional[int] = None
    if isinstance(correct_option, int):
        correct_idx = int(correct_option)
    elif isinstance(correct_option, str):
        co = correct_option.strip().upper()
        if len(co) == 1 and "A" <= co <= "Z":
            correct_idx = ord(co) - ord("A")
        else:
            try:
                correct_idx = int(co)
            except ValueError:
                correct_idx = None
    correct_letter = (
        chr(ord("A") + correct_idx) if correct_idx is not None
        and 0 <= correct_idx < 26 else ""
    ).lower()
    correct_text = ""
    if correct_idx is not None and 0 <= correct_idx < len(options):
        correct_text = _normalize_answer(options[correct_idx])

    # Strategy 1: leading character is the correct letter (e.g., "C",
    # "C.", "C)", "C: option text", "(C)").
    leading = ma.lstrip("([").lstrip()
    if leading and correct_letter and leading[0] == correct_letter:
        if len(leading) == 1 or not leading[1].isalpha():
            return True

    # Strategy 2: model output equals the correct option text, OR the
    # correct option text appears as a substring of the model output
    # (model answered "the answer is on the table" → still correct).
    # Critically, do NOT do `ma in correct_text` — "b" is a substring
    # of "table", which would let any single letter match any option
    # whose text contains it.
    if correct_text and (ma == correct_text or correct_text in ma):
        return True

    # Strategy 3: model output exactly matches one of the option texts —
    # must be the correct one to score. Also accept "long option text in
    # ma" (model wrote out the full chosen option), but require length ≥ 4
    # to avoid the same single-letter-in-text trap as Strategy 2.
    for i, opt in enumerate(options):
        on = _normalize_answer(opt)
        if on and (ma == on or (len(on) >= 4 and on in ma)):
            return i == correct_idx

    # Strategy 4: gold_answer text fallback (some datasets use free text).
    # Require gold_answer length ≥ 2 to avoid pathological single-char
    # gold-answer false positives ("a" in everything).
    if gold_answer:
        ga = _normalize_answer(gold_answer)
        if ga and len(ga) >= 2 and (ma == ga or ga in ma):
            return True
        # For very short gold answers (single chars / digits), require
        # exact match.
        if ga and len(ga) < 2 and ma == ga:
            return True

    return False


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


# ---------------------------------------------------------------------------
# Form-aware liberal outcome matchers.
#
# ThinkStream's pass3a emits 5 answer_form values:
#   multiple_choice / binary / number / short_exact / descriptive
#
# v12_rewards.compute_outcome_reward_v12 strict path is `fa.lower() == ga.lower()`,
# which is too rigid: "Yes." vs "yes", "100." vs "100", "the apple" vs "apple"
# all fail and silently zero the reward signal. The matchers below relax the
# comparison per-form while staying conservative enough to avoid reward
# hacking.
# ---------------------------------------------------------------------------
_BINARY_YES = {"yes", "y", "true", "1", "t", "是", "对"}
_BINARY_NO = {"no", "n", "false", "0", "f", "否", "不"}


def _binary_polarity(s: str) -> Optional[bool]:
    """Return True/False if the string is a binary yes/no, else None.
    Strips leading/trailing punctuation and looks at the first token."""
    if not s:
        return None
    norm = _normalize_answer(s)
    if not norm:
        return None
    # Take first whitespace-separated token, then strip its trailing punct.
    first = re.sub(r"[^\w]+$", "", norm.split()[0]) if norm.split() else ""
    if first in _BINARY_YES:
        return True
    if first in _BINARY_NO:
        return False
    # Some MCQ-style binaries store "A"/"B" as gold — leave those to MCQ.
    return None


def _match_binary(model_answer: str, gold_answer: str) -> bool:
    """yes/no/true/false matching with normalization."""
    ma_pol = _binary_polarity(model_answer)
    ga_pol = _binary_polarity(gold_answer)
    if ma_pol is None or ga_pol is None:
        # Fall through to short_exact normalize compare so we don't lose
        # weird gold values like "yeah" / "nope".
        return _normalize_answer(model_answer) == _normalize_answer(gold_answer)
    return ma_pol == ga_pol


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_first_number(s: str) -> Optional[float]:
    """Pull out the first numeric token from `s`. Returns None if none."""
    if not s:
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except (ValueError, TypeError):
        return None


def _match_number(model_answer: str, gold_answer: str, *, rel_tol: float = 0.0,
                  abs_tol: float = 1e-6) -> bool:
    """Numeric match with tolerance. Accepts trailing units / punctuation
    (e.g. "100." / "100 mph" / "$100"). Default tolerance is exact-equal
    on integers; bump rel_tol for noisy domains."""
    ma_n = _extract_first_number(model_answer)
    ga_n = _extract_first_number(gold_answer)
    if ma_n is None or ga_n is None:
        return False
    if abs(ma_n - ga_n) <= abs_tol:
        return True
    if rel_tol > 0 and ga_n != 0:
        return abs(ma_n - ga_n) / abs(ga_n) <= rel_tol
    return False


_STOPWORDS = {"the", "a", "an", "of", "is", "are", "was", "were"}


def _strip_articles(s: str) -> str:
    """Drop leading article-like stopwords. 'the apple' → 'apple'."""
    tokens = s.split()
    while tokens and tokens[0] in _STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _match_short_exact(model_answer: str, gold_answer: str) -> bool:
    """Entity-name / short-exact match: normalize, strip leading articles,
    accept either bidirectional substring (model wrote 'the apple' → still
    matches gold 'apple', and vice versa). Conservative: requires gold
    length ≥ 2 chars to avoid single-letter false positives."""
    ma = _strip_articles(_normalize_answer(model_answer))
    ga = _strip_articles(_normalize_answer(gold_answer))
    if not ma or not ga:
        return False
    if ma == ga:
        return True
    if len(ga) < 2:
        return ma == ga
    return ga in ma or ma in ga


def _match_descriptive(model_answer: str, gold_answer: str) -> bool:
    """Free-text descriptive answers — bidirectional substring with
    normalization. v12 fall-back path; consider plugging an LLM judge
    here for higher-fidelity scoring on production runs."""
    ma = _normalize_answer(model_answer)
    ga = _normalize_answer(gold_answer)
    if not ma or not ga:
        return False
    return ma == ga or ga in ma or ma in ga


def _score_outcome_by_form(
    model_answer: str,
    *,
    options: List[str],
    correct_option: Any,
    gold_answer: str,
    answer_form: str,
) -> float:
    """Form-aware liberal outcome scoring. Returns 1.0 for a match, 0.0 otherwise.

    Dispatches by `answer_form`:
      - multiple_choice / mc        → _match_mcq_answer (letter / text / gold fallback)
      - binary / yes_no             → _match_binary (yes/no/true/false polarity)
      - number / numeric            → _match_number (extract float, tolerance)
      - short_exact / entity / literal → _match_short_exact (article-stripped substring)
      - descriptive (default)       → _match_descriptive (bidirectional substring)

    The model answer must be non-empty for any match. Length > 1000 chars
    forces 0 (anti-spam, mirrors compute_outcome_reward_v12).
    """
    if not model_answer or not str(model_answer).strip():
        return 0.0
    if len(str(model_answer)) > 1000:
        return 0.0

    af = (answer_form or "").lower()

    if af in ("multiple_choice", "mc") and options:
        return 1.0 if _match_mcq_answer(
            model_answer, options, correct_option, gold_answer,
        ) else 0.0

    if af in ("binary", "yes_no"):
        return 1.0 if _match_binary(model_answer, gold_answer) else 0.0

    if af in ("number", "numeric"):
        return 1.0 if _match_number(model_answer, gold_answer) else 0.0

    if af in ("short_exact", "entity", "literal"):
        return 1.0 if _match_short_exact(model_answer, gold_answer) else 0.0

    # Default / descriptive
    return 1.0 if _match_descriptive(model_answer, gold_answer) else 0.0


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
    ask_chunks_int = [int(x) for x in ask_chunks if isinstance(x, (int, float))]
    visible_start = min(ask_chunks_int) if ask_chunks_int else None
    visible_end = max(ask_chunks_int) if ask_chunks_int else None

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

    # Timing — bucket the answered_chunk vs the visible window.
    timing = float(rewards["timing"](
        answered_chunk if answered_chunk >= 0 else None,
        visible_start, visible_end,
    ))

    # Silent quality — was the model silent before evidence and on-time
    # at evidence? compute_silent_quality_v12 takes (final_answer, gold_action,
    # gold_answer); for multi-Q we approximate gold_action as "response"
    # (we expect a response at this Q's ask_chunk).
    try:
        silent_q = float(rewards["silent_quality"](
            model_answer if answered else None,
            "response",
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
    per_q_chunk = list(per_q_chunk_raw) + [-1] * (n_q - len(per_q_chunk_raw))
    per_q_text = list(per_q_text_raw) + [""] * (n_q - len(per_q_text_raw))

    # Per-Q scoring.
    per_q_outcome: List[float] = []
    per_q_timing: List[float] = []
    per_q_silent: List[float] = []
    n_answered = 0
    for q_idx, q in enumerate(questions):
        sub = _score_one_question(
            rewards,
            q=q,
            model_answer=str(per_q_text[q_idx] or ""),
            answered_chunk=int(per_q_chunk[q_idx]),
        )
        per_q_outcome.append(sub["outcome"])
        per_q_timing.append(sub["timing"])
        per_q_silent.append(sub["silent_quality"])
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
    total = float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))

    return {
        "score": total,
        **{k: float(v) for k, v in parts.items()},
        "n_questions": float(n_q),
        "n_answered": float(n_answered),
        "per_q_outcome_min": float(min(per_q_outcome)),
        "per_q_outcome_max": float(max(per_q_outcome)),
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
