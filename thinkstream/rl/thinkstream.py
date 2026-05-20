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
#   - compute_score wraps thinkstream.trainer.rewards into the
#     scalar (data_source, solution_str, ground_truth, extra_info) -> float
#     signature verl's main_ppo expects.
#
# The ThinkStream package is imported via PYTHONPATH (see
# run_thinkstream_grpo.sh — THINKSTREAM_HOME is prepended). We do not
# vendor ThinkStream code into this fork.
from __future__ import annotations

import io
import html
import json
import logging
import os
import random
import re
import time
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
# IMPORTANT: the recipe dir is named `thinkstream/rl/` (NOT
# `thinkstream/`). Naming it `thinkstream/` would shadow the real
# ThinkStream Python package on PYTHONPATH. The relative import below
# only works when this module is loaded as part of the thinkstream/rl
# package; the importlib fallback handles the spec_from_file_location
# path hydra uses without polluting sys.path.
def _side_effect_import(sibling_module: str, alias: str):
    """Import a sibling module of this recipe with explicit naming so we
    don't pollute sys.path. Triggers @register decorators."""
    try:
        # Works when this file is loaded as thinkstream.rl.thinkstream
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
        from thinkstream.trainer.rewards import (  # type: ignore
            compute_outcome_reward,
            compute_timing_reward,
            compute_answer_decision_reward,
            compute_format_reward,
            compute_silent_quality,
        )
        from thinkstream.trainer.gdpo_advantage import (  # type: ignore
            V12_DEFAULT_REWARD_WEIGHTS,
        )
        _ts_rewards = {
            "outcome": compute_outcome_reward,
            "timing": compute_timing_reward,
            "answer_decision": compute_answer_decision_reward,
            "format": compute_format_reward,
            "silent_quality": compute_silent_quality,
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


def _strip_offline_compress_actions(gold_action: Any) -> Dict[str, str]:
    """Remove offline compress labels from RL/eval action shaping maps."""
    if hasattr(gold_action, "tolist"):
        gold_action = gold_action.tolist()
    if not isinstance(gold_action, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in gold_action.items():
        if v is None:
            continue
        action = str(v).strip()
        if not action or action.lower() == "none" or action == "compress":
            continue
        out[str(k)] = action
    return out


def _coerce_int_list(value: Any) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    out: List[int] = []
    for raw in raw_values:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _offline_compress_chunks_from_gold_action(gold_action: Any) -> List[int]:
    """Recover legacy offline compression triggers before action-map cleanup."""
    if hasattr(gold_action, "tolist"):
        gold_action = gold_action.tolist()
    if not isinstance(gold_action, dict):
        return []
    chunks: List[int] = []
    for k, v in gold_action.items():
        if str(v or "") != "compress":
            continue
        try:
            chunks.append(int(k))
        except (TypeError, ValueError):
            continue
    return chunks


def _merge_offline_compress_chunks(
    existing: Any,
    gold_action: Any,
    *,
    start_chunk: Optional[int] = None,
    end_chunk: Optional[int] = None,
) -> List[int]:
    """Merge new-style boundaries with legacy `gold_action=compress` labels."""
    chunks = _coerce_int_list(existing) + _offline_compress_chunks_from_gold_action(
        gold_action
    )
    if start_chunk is not None:
        chunks = [ci for ci in chunks if ci >= int(start_chunk)]
    if end_chunk is not None:
        chunks = [ci for ci in chunks if ci <= int(end_chunk)]
    return sorted(set(ci for ci in chunks if ci >= 0))


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
    scripts/agent_data/build_verl_parquet.py. Columns expected:
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

    @staticmethod
    def _normalize_episode_mode(value: Any) -> str:
        mode = str(value or "full").strip().lower().replace("-", "_")
        # This is a derived segment view over the same multi-Q parquet row,
        # not a separate single-step annotation source.
        if mode in {"segment", "single_question", "single_q", "question", "per_question"}:
            return "single_question"
        if mode in {"full", "full_video", "trajectory", "multi_q"}:
            return "full"
        return "full"

    def __init__(self, *args, **kwargs):
        self.thinkstream_episode_mode = self._normalize_episode_mode(
            os.environ.get("THINKSTREAM_RL_EPISODE_MODE", "full") or "full"
        )
        if self.thinkstream_episode_mode == "single_question":
            cfg = kwargs.get("config")
            if cfg is None and len(args) >= 3:
                cfg = args[2]
            if cfg is not None and _env_bool("THINKSTREAM_SINGLE_Q_FORCE_ORDER", True):
                for key, value in (("shuffle", False), ("dataloader_num_workers", 0)):
                    try:
                        cfg[key] = value
                    except Exception:
                        try:
                            setattr(cfg, key, value)
                        except Exception:
                            pass
        self.segment_pre_context = _env_int("THINKSTREAM_SEGMENT_PRE_CONTEXT", 8)
        self.segment_post_context = _env_int("THINKSTREAM_SEGMENT_POST_CONTEXT", 8)
        self.segment_max_chunks = _env_int("THINKSTREAM_SEGMENT_MAX_CHUNKS", 64)
        self.segment_require_recall_archive = _env_bool(
            "THINKSTREAM_SEGMENT_REQUIRE_RECALL_ARCHIVE", True,
        )
        self._question_episode_index: List[tuple[int, int]] = []
        super().__init__(*args, **kwargs)
        if self.thinkstream_episode_mode == "single_question":
            self._build_question_episode_index()

    def __len__(self):
        if (
            getattr(self, "thinkstream_episode_mode", "full") == "single_question"
            and getattr(self, "_question_episode_index", None)
        ):
            return len(self._question_episode_index)
        return super().__len__()

    @staticmethod
    def _drop_empty_video_items(messages: list[dict]) -> list[dict]:
        cleaned: list[dict] = []
        for message in messages or []:
            if not isinstance(message, dict):
                cleaned.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, list):
                cleaned.append(dict(message))
                continue
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                if item.get("type") == "video":
                    video = item.get("video")
                    if video in (None, "") or video == []:
                        continue
                new_content.append(dict(item))
            next_message = dict(message)
            next_message["content"] = new_content
            cleaned.append(next_message)
        return cleaned

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        from qwen_vl_utils import process_vision_info

        messages = cls._drop_empty_video_items(messages)
        explicit_video_metadata = []
        if str(os.environ.get("THINKSTREAM_PRESERVE_EXPLICIT_VIDEO_METADATA", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }:
            for message in messages or []:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "video":
                        continue
                    meta = item.get("video_metadata")
                    if isinstance(meta, dict):
                        cleaned_meta = dict(meta)
                        cleaned_meta.pop("do_sample_frames", None)
                        explicit_video_metadata.append(cleaned_meta)
        images, videos = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )
        if videos is not None and explicit_video_metadata:
            fixed_videos = []
            for i, video_item in enumerate(videos):
                if isinstance(video_item, tuple) and len(video_item) == 2:
                    video_tensor, video_meta = video_item
                else:
                    video_tensor, video_meta = video_item, None
                fixed_videos.append((
                    video_tensor,
                    explicit_video_metadata[i] if i < len(explicit_video_metadata) else video_meta,
                ))
            videos = fixed_videos
        return images, videos

    @staticmethod
    def _plain_list(value: Any) -> List[Any]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @classmethod
    def _normalize_questions(cls, questions_raw: Any) -> List[Dict[str, Any]]:
        questions_raw = cls._plain_list(questions_raw)
        out: List[Dict[str, Any]] = []
        for q in questions_raw:
            if hasattr(q, "tolist"):
                q = q.tolist()
            if not isinstance(q, dict):
                continue
            clean: Dict[str, Any] = {}
            for k, v in q.items():
                if hasattr(v, "tolist"):
                    v = v.tolist()
                clean[k] = v
            out.append(clean)
        return out

    @staticmethod
    def _safe_int_list(value: Any) -> List[int]:
        out: List[int] = []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value is None:
            values: List[Any] = []
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value]
        for x in values:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _valid_chunks(chunks: List[int], *, n_chunks: int) -> List[int]:
        if n_chunks <= 0:
            return [c for c in chunks if c >= 0]
        return [c for c in chunks if 0 <= c < n_chunks]

    def _build_question_episode_index(self) -> None:
        ordered: List[tuple[int, int, int, int]] = []
        for row_i in range(len(self.dataframe)):
            row = self.dataframe[row_i]
            extra = row.get("extra_info", {}) or {}
            if hasattr(extra, "tolist"):
                extra = extra.tolist()
            if not isinstance(extra, dict):
                continue
            n_chunks = int(row.get("n_chunks") or extra.get("n_chunks") or 0)
            questions = self._normalize_questions(extra.get("questions"))
            for q_i, q in enumerate(questions):
                segment_start, segment_end = self._window_for_question(
                    q, n_chunks=n_chunks,
                )
                ordered.append((row_i, segment_start, segment_end, q_i))
        ordered.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        self._question_episode_index = [(row_i, q_i) for row_i, _, _, q_i in ordered]
        print(
            "thinkstream single_question episodes: "
            f"{len(self._question_episode_index)} from {len(self.dataframe)} video rows"
        )

    def _window_for_question(
        self, q: Dict[str, Any], *, n_chunks: int,
    ) -> tuple[int, int]:
        n_chunks = max(0, int(n_chunks or 0))
        asks = self._valid_chunks(self._safe_int_list(q.get("ask_chunks")), n_chunks=n_chunks)
        if not asks and q.get("ask_chunk") is not None:
            asks = self._valid_chunks(self._safe_int_list(q.get("ask_chunk")), n_chunks=n_chunks)
        answers = self._valid_chunks(self._safe_int_list(q.get("answer_chunks")), n_chunks=n_chunks)
        supports = self._valid_chunks(self._safe_int_list(q.get("support_chunks")), n_chunks=n_chunks)
        probes = self._valid_chunks(
            self._safe_int_list(q.get("probe_chunks") or q.get("test_chunks")),
            n_chunks=n_chunks,
        )

        anchor_start = min(asks) if asks else 0
        targets = answers + supports + probes + asks
        anchor_end = max(targets) if targets else anchor_start
        start = max(0, anchor_start - max(0, self.segment_pre_context))
        end = min(
            max(0, n_chunks - 1),
            anchor_end + max(0, self.segment_post_context),
        )
        if self.segment_max_chunks > 0 and end - start + 1 > self.segment_max_chunks:
            capped_start = max(0, end - self.segment_max_chunks + 1)
            # Correctness first: the online segment must include the query
            # injection chunk. If the ask-to-answer/support span is longer
            # than max_chunks, allow a longer segment rather than creating an
            # answer-only rollout where the model never sees the question.
            if not asks or min(asks) >= capped_start:
                # Keep ask/probe tail in the online segment. Earlier support
                # must come from a student-generated initial_memory_snapshot.
                start = capped_start
        if end < start:
            end = start
        return start, end

    @classmethod
    def _snapshot_for_chunk_with_key(
        cls, snapshots: Any, chunk_idx: int,
    ) -> tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Pick a student-generated memory snapshot for the start of a segment.

        Snapshot key `chunk_idx` means "state before processing this chunk",
        matching MemoryState.snapshot(chunk_idx). If an exact key is missing,
        fall back to the latest earlier key; this lets cached prefix rollouts
        be sparse without ever inventing teacher history.
        """
        if hasattr(snapshots, "tolist"):
            snapshots = snapshots.tolist()
        if isinstance(snapshots, list):
            if 0 <= chunk_idx < len(snapshots) and isinstance(snapshots[chunk_idx], dict):
                return chunk_idx, dict(snapshots[chunk_idx])
            for idx in range(min(chunk_idx, len(snapshots) - 1), -1, -1):
                if isinstance(snapshots[idx], dict):
                    return idx, dict(snapshots[idx])
            return None, None
        if not isinstance(snapshots, dict):
            return None, None
        for key in (chunk_idx, str(chunk_idx)):
            value = snapshots.get(key)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, dict):
                return chunk_idx, dict(value)

        best_key: Optional[int] = None
        for key in snapshots.keys():
            try:
                k_int = int(key)
            except (TypeError, ValueError):
                continue
            if k_int <= chunk_idx and (best_key is None or k_int > best_key):
                best_key = k_int
        if best_key is None:
            return None, None
        value = snapshots.get(best_key, snapshots.get(str(best_key)))
        if hasattr(value, "tolist"):
            value = value.tolist()
        return (best_key, dict(value)) if isinstance(value, dict) else (None, None)

    @classmethod
    def _snapshot_for_chunk(cls, snapshots: Any, chunk_idx: int) -> Optional[Dict[str, Any]]:
        _, snapshot = cls._snapshot_for_chunk_with_key(snapshots, chunk_idx)
        return snapshot

    @classmethod
    def _snapshot_has_recall_archive(cls, snapshot: Dict[str, Any]) -> bool:
        archive = (
            snapshot.get("think_archive")
            or snapshot.get("retrieval_archive")
            or snapshot.get("evidence_bank")
            or []
        )
        return bool(cls._plain_list(archive))

    @classmethod
    def _student_archive_until(
        cls,
        extra: Dict[str, Any],
        snapshot_chunk: int,
    ) -> List[Dict[str, Any]]:
        """Return student-generated archive entries before a snapshot.

        Pass2 snapshots intentionally store only the visible memory. The
        sibling rollout cache also has per-chunk student thinks; combine those
        at materialization time so recall in a segment can search the same
        student-observation history without storing O(n^2) archives in parquet.
        """
        for key in ("student_think_archive", "pass2_thinks", "think_archive"):
            raw = extra.get(key)
            archive = cls._plain_list(raw)
            if not archive:
                continue
            out: List[Dict[str, Any]] = []
            for item in archive:
                if hasattr(item, "tolist"):
                    item = item.tolist()
                if not isinstance(item, dict):
                    continue
                try:
                    chunk = int(item.get("chunk", item.get("chunk_idx", -1)))
                except (TypeError, ValueError):
                    continue
                if 0 <= chunk < int(snapshot_chunk):
                    clean = dict(item)
                    clean.setdefault("chunk", chunk)
                    clean.setdefault("chunk_idx", chunk)
                    if "text" not in clean and "think" in clean:
                        clean["text"] = clean.get("think", "")
                    if clean.get("text"):
                        out.append(clean)
            if out:
                out.sort(key=lambda x: int(x.get("chunk", x.get("chunk_idx", 0))))
                return out
        return []

    def _find_initial_student_state(
        self,
        extra: Dict[str, Any],
        planned_start: int,
    ) -> tuple[int, Optional[Dict[str, Any]], str]:
        if planned_start <= 0:
            return 0, None, "rollout_from_zero"
        for snapshots_key in (
            "initial_student_state_by_chunk",
            "student_state_by_chunk",
            "student_memory_snapshots",
            "student_snapshots",
            "memory_snapshots",
            "snapshots",
        ):
            snapshot_chunk, initial_state = self._snapshot_for_chunk_with_key(
                extra.get(snapshots_key), planned_start,
            )
            if initial_state is None or snapshot_chunk is None:
                continue
            if not self._snapshot_has_recall_archive(initial_state):
                archive = self._student_archive_until(extra, int(snapshot_chunk))
                if archive:
                    initial_state = dict(initial_state)
                    initial_state["think_archive"] = archive
                    initial_state["retrieval_archive"] = archive
            if (
                self.segment_require_recall_archive
                and snapshot_chunk > 0
                and not self._snapshot_has_recall_archive(initial_state)
            ):
                continue
            return max(0, int(snapshot_chunk)), initial_state, snapshots_key
        return 0, None, "missing_cache_rollout_from_zero"

    def _materialize_single_question_row(
        self,
        row_dict: dict,
        q_idx: int,
        *,
        source_row_i: Optional[int] = None,
    ) -> dict:
        row_dict = dict(row_dict)
        extra = row_dict.get("extra_info", {}) or {}
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        if not isinstance(extra, dict):
            extra = {}
        else:
            extra = dict(extra)

        questions = self._normalize_questions(extra.get("questions"))
        if not questions or q_idx < 0 or q_idx >= len(questions):
            return row_dict

        q = dict(questions[q_idx])
        n_chunks = int(row_dict.get("n_chunks") or extra.get("n_chunks") or 0)
        planned_start, segment_end = self._window_for_question(q, n_chunks=n_chunks)
        segment_start, initial_state, initial_state_source = self._find_initial_student_state(
            extra,
            planned_start,
        )
        raw_gold_action = extra.get("gold_action_per_chunk") or {}
        gold_action = raw_gold_action
        if hasattr(gold_action, "tolist"):
            gold_action = gold_action.tolist()
        if not isinstance(gold_action, dict):
            gold_action = {}
        q_ask_chunks = self._valid_chunks(
            self._safe_int_list(q.get("ask_chunks") or [q.get("ask_chunk")]),
            n_chunks=n_chunks,
        )
        q_answer_chunks = self._valid_chunks(
            self._safe_int_list(q.get("answer_chunks")),
            n_chunks=n_chunks,
        )
        q_live_marks = q_ask_chunks + q_answer_chunks
        q_live_start = min(q_ask_chunks or q_live_marks or [segment_start])
        q_live_end = max(q_answer_chunks or q_ask_chunks or q_live_marks or [segment_end])
        segment_gold_action: Dict[str, Any] = {}
        for k, v in gold_action.items():
            try:
                ck = int(k)
            except (TypeError, ValueError):
                continue
            if segment_start <= ck <= segment_end:
                segment_gold_action[str(ck)] = (
                    v if q_live_start <= ck <= q_live_end else "silent"
                )

        single_extra = dict(extra)
        if source_row_i is not None:
            source_video_row_index = int(source_row_i)
        elif str(row_dict.get("index", "")).isdigit():
            source_video_row_index = int(row_dict.get("index", 0) or 0)
        else:
            source_video_row_index = 0
        single_extra.update({
            "episode_mode": "single_question",
            "source_video_row_index": source_video_row_index,
            "question_idx": int(q_idx),
            "question_index": int(q_idx),
            "questions": [q],
            "offline_compress_chunks": _merge_offline_compress_chunks(
                extra.get("offline_compress_chunks"),
                raw_gold_action,
                start_chunk=segment_start,
                end_chunk=segment_end,
            ),
            "gold_action_per_chunk": segment_gold_action,
            "all_ask_chunks": self._safe_int_list(q.get("ask_chunks") or [q.get("ask_chunk")]),
            "segment_planned_start_chunk": int(planned_start),
            "segment_start_chunk": int(segment_start),
            "segment_end_chunk": int(segment_end),
            "segment_prefix_source": initial_state_source,
            "video_id": str(row_dict.get("video_id", extra.get("video_id", ""))),
            "video_path": str(row_dict.get("video_path", extra.get("video_path", ""))),
            "n_chunks": n_chunks,
            "question": str(q.get("question", "")),
            "gold_answer": str(q.get("gold_answer") or q.get("correct_answer_text") or ""),
            "answer_form": str(q.get("answer_form", "")),
            "ask_chunks": q_ask_chunks,
            "answer_chunks": q_answer_chunks,
            "question_live_start_chunk": int(q_live_start),
            "question_live_end_chunk": int(q_live_end),
        })
        if initial_state is not None:
            single_extra["initial_student_state"] = initial_state
            single_extra["initial_student_state_source"] = initial_state_source
            cache_meta = extra.get("student_cache_meta")
            if hasattr(cache_meta, "tolist"):
                cache_meta = cache_meta.tolist()
            if isinstance(cache_meta, dict):
                single_extra["initial_student_state_meta"] = dict(cache_meta)
                for key in ("checkpoint", "global_step", "epoch"):
                    if cache_meta.get(key) not in (None, ""):
                        single_extra[f"initial_student_state_{key}"] = cache_meta.get(key)
                if cache_meta.get("source") not in (None, ""):
                    single_extra["initial_student_cache_source"] = cache_meta.get("source")
        elif planned_start > 0:
            single_extra["initial_student_state_missing"] = True
        row_dict["extra_info"] = single_extra

        gt = {
            "questions": [q],
            "offline_compress_chunks": single_extra.get("offline_compress_chunks", []),
            "gold_action_per_chunk": segment_gold_action,
            "episode_mode": "single_question",
            "source_video_row_index": source_video_row_index,
            "question_idx": int(q_idx),
            "question_index": int(q_idx),
            "segment_planned_start_chunk": int(planned_start),
            "segment_start_chunk": int(segment_start),
            "segment_end_chunk": int(segment_end),
        }
        reward_model = row_dict.get("reward_model") or {}
        if hasattr(reward_model, "tolist"):
            reward_model = reward_model.tolist()
        if not isinstance(reward_model, dict):
            reward_model = {}
        reward_model = dict(reward_model)
        reward_model["ground_truth"] = json.dumps(gt, ensure_ascii=False)
        row_dict["reward_model"] = reward_model
        row_dict["data_source"] = "thinkstream_v12_streaming_multi_q"
        return row_dict

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
        if (
            getattr(self, "thinkstream_episode_mode", "full") == "single_question"
            and getattr(self, "_question_episode_index", None)
        ):
            row_i, q_i = self._question_episode_index[item]
            row_dict: dict = self._materialize_single_question_row(
                self.dataframe[row_i], q_i, source_row_i=row_i,
            )
        else:
            row_dict = dict(self.dataframe[item])

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
            normalized_qs = self._normalize_questions(questions_raw)
            extra["questions"] = normalized_qs

            gap = extra.get("gold_action_per_chunk")
            extra["offline_compress_chunks"] = _merge_offline_compress_chunks(
                extra.get("offline_compress_chunks"),
                gap,
            )
            extra["gold_action_per_chunk"] = _strip_offline_compress_actions(gap)

            extra.update({
                "video_id": str(row_dict.get("video_id", "")),
                "video_path": str(row_dict.get("video_path", "")),
                "n_chunks": int(row_dict.get("n_chunks") or 0),
            })
        else:
            row_gap = row_dict.get("gold_action_per_chunk")
            extra.update({
                "video_id": str(row_dict.get("video_id", "")),
                "video_path": str(row_dict.get("video_path", "")),
                "question": str(row_dict.get("question", "")),
                "gold_answer": str(row_dict.get("gold_answer", "")),
                "answer_form": str(row_dict.get("answer_form", "")),
                "ask_chunks": list(row_dict.get("ask_chunks") or []),
                "offline_compress_chunks": _merge_offline_compress_chunks(
                    extra.get("offline_compress_chunks"),
                    row_gap,
                ),
                "gold_action_per_chunk": _strip_offline_compress_actions(row_gap),
                "n_chunks": int(row_dict.get("n_chunks") or 0),
            })

        row_dict["extra_info"] = extra
        row_dict["index"] = extra.get("index", str(row_dict.get("video_id", "")))
        row_dict["uid"] = str(
            row_dict.get("uid")
            or extra.get("uid")
            or row_dict.get("video_id")
            or row_dict["index"]
        )
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

    In normal ThinkStream rollouts, ``extra["ts_chunk_asst_texts"]`` carries
    raw decoded assistant turns with agent special tokens preserved and this
    helper is only a fallback. Each canonical v12 assistant turn opens with
    ``<think>``; split on that marker when it is available.
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


def _extract_answer_text_current(text: str, *, allow_bare_answer: bool = False) -> Optional[str]:
    """Extract answer text for outcome scoring using the current protocol only."""
    try:
        from thinkstream.data.agent_protocol import parse_agent_output

        parsed = parse_agent_output(
            text,
            allow_bare_answer=allow_bare_answer,
        )
        if parsed.get("kind") == "answer":
            return (parsed.get("answer_text") or "").strip()
    except Exception:  # noqa: BLE001
        pass

    m = re.search(r"</Response>\s*(.*?)\s*$", text or "", re.DOTALL)
    if m:
        return (m.group(1) or "").strip()
    return None


def _extract_final_answer(solution_str: str) -> Optional[str]:
    """Return the last current-protocol response."""
    chunks = _split_assistant_chunks(solution_str)
    for chunk in reversed(chunks or [solution_str]):
        ans = _extract_answer_text_current(chunk, allow_bare_answer=True)
        if ans:
            return ans.strip()
    matches = [
        (m.group(1) or "").strip()
        for m in re.finditer(r"</Response>\s*(.*?)\s*$", solution_str, re.DOTALL)
    ]
    if not matches:
        return None
    return matches[-1].strip()


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
        ans = _extract_answer_text_current(text or "") or ""
        return "silent" if not ans else "response"
    if kind == "recall":
        return "recall"
    if kind == "compress":
        return "compress"
    return "unknown"


def _per_chunk_action_avg(
    extra: Dict[str, Any],
    gold_action_per_chunk: Dict[str, str],
    audit_out: Optional[Dict[str, Any]] = None,
    *,
    score_compress: Optional[bool] = None,
) -> Optional[float]:
    """Small action-alignment diagnostic aligned to turn-local metadata.

    Recall is **monitor-only** (P7): a `recall_audit` dict is written into
    ``audit_out`` for wandb telemetry, but recall alignment is NOT pushed into
    the returned per-chunk action score — the correctness reward of the answer
    that consumes recall results does that shaping. ``recall_align_rate`` is
    gold-policy timing telemetry; ``recall_runtime_ok_rate`` only checks
    whether emitted recall tool calls are legal and executable.

    Compression is also monitor-only by default. Set
    ``THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD=1`` only for an explicit
    ablation; the initial RL objective should not directly reward tool
    compliance or pass2's exact compression policy.
    """
    if score_compress is None:
        score_compress = _env_bool("THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD", False)
    chunk_kinds = _safe_list(extra.get("ts_chunk_kinds"))
    if not chunk_kinds:
        return None

    chunk_texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    chunk_vidx = _safe_list(extra.get("ts_chunk_video_indices"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    scores: List[float] = []
    recall_seen_for_chunk: set[int] = set()
    recall_audit: Dict[str, int] = {"seen": 0, "matched": 0}
    recall_runtime_audit: Dict[str, int] = {"seen": 0, "ok": 0}
    compress_audit: Dict[str, int] = {"seen": 0, "matched": 0}

    try:
        from thinkstream.data.agent_protocol import parse_agent_output
    except Exception:
        parse_agent_output = None

    for turn_i, kind_raw in enumerate(chunk_kinds):
        kind = str(kind_raw or "unknown")
        turn_kind = (
            str(turn_kinds[turn_i] or "")
            if turn_i < len(turn_kinds)
            else ""
        )
        if turn_kind in {"recall_response", "post_recall"}:
            # The recall-response assistant turn is conditioned on the prior
            # tool result; answer correctness/timing scores it. The action
            # decision to train here is the preceding recall tool_call.
            continue

        text = chunk_texts[turn_i] if turn_i < len(chunk_texts) else ""
        model_action = _model_action_from_turn(kind, str(text or ""))
        if model_action == "recall":
            recall_runtime_audit["seen"] += 1
            if parse_agent_output is not None:
                parsed = parse_agent_output(str(text or ""))
                args = ((parsed.get("tool_call") or {}).get("arguments") or {})
                if _tool_time_range_runtime_ok(
                    "recall",
                    args,
                    current_chunk=_turn_current_chunk(extra, turn_i),
                ):
                    recall_runtime_audit["ok"] += 1

        if turn_kind == "compress":
            # Compression is system-triggered. Track whether the model
            # complied, but do not reward it in the initial objective.
            compress_audit["seen"] += 1
            if model_action == "compress":
                compress_audit["matched"] += 1
            if score_compress:
                scores.append(0.1 if model_action == "compress" else -0.05)
            continue

        try:
            video_chunk_idx = int(chunk_vidx[turn_i]) if turn_i < len(chunk_vidx) else turn_i
        except (TypeError, ValueError):
            video_chunk_idx = turn_i
        if video_chunk_idx < 0:
            continue

        gold_action = str(
            (gold_action_per_chunk or {}).get(str(video_chunk_idx), "")
        )
        if not gold_action:
            continue

        if gold_action == "compress":
            # Offline compress labels mark where pass2/eval happened to
            # compact memory. In RL the trigger is derived from the live
            # memory state, so these labels are not active policy targets.
            # Kept for legacy parquet only; new parquet strips them from
            # gold_action_per_chunk and stores offline_compress_chunks.
            continue

        if gold_action in {"recall", "recall_silent"}:
            # Per user directive (P7): recall is a MONITOR-only signal,
            # not a reward source. We tally alignment for telemetry
            # (``recall_align_*`` keys returned through ``audit_out``) but
            # do NOT push anything into ``scores`` here — recall behavior
            # is shaped end-to-end by the correctness reward of the
            # answer that consumes the recall result, not by per-chunk
            # action-match.
            recall_audit["seen"] += 1
            matched = (
                (model_action == "recall")
                or (gold_action == "recall_silent"
                    and model_action == "silent"
                    and video_chunk_idx in recall_seen_for_chunk)
            )
            if model_action == "recall":
                recall_seen_for_chunk.add(video_chunk_idx)
            if matched:
                recall_audit["matched"] += 1
            continue

        scores.append(0.1 if model_action == gold_action else -0.05)

    # Surface recall monitor metrics for wandb (P7: monitor only, not reward).
    if audit_out is not None:
        seen = recall_audit["seen"]
        matched = recall_audit["matched"]
        audit_out["recall_seen"] = seen
        audit_out["recall_matched"] = matched
        audit_out["recall_align_rate"] = (
            float(matched) / float(seen) if seen > 0 else 0.0
        )
        r_seen = recall_runtime_audit["seen"]
        r_ok = recall_runtime_audit["ok"]
        audit_out["recall_runtime_seen"] = r_seen
        audit_out["recall_runtime_ok"] = r_ok
        audit_out["recall_runtime_ok_rate"] = (
            float(r_ok) / float(r_seen) if r_seen > 0 else 0.0
        )
        c_seen = compress_audit["seen"]
        c_matched = compress_audit["matched"]
        audit_out["compress_seen"] = c_seen
        audit_out["compress_matched"] = c_matched
        audit_out["compress_align_rate"] = (
            float(c_matched) / float(c_seen) if c_seen > 0 else 0.0
        )

    if not scores:
        return None
    return sum(scores) / len(scores)


def _coerce_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_tool_time_range(kind: str, args: Dict[str, Any]) -> Optional[tuple[float, float]]:
    if kind == "recall":
        start = _coerce_float((args or {}).get("start_time"))
        end = _coerce_float((args or {}).get("end_time"))
    else:
        tr = (args or {}).get("time_range")
        if not isinstance(tr, (list, tuple)) or len(tr) != 2:
            return None
        start = _coerce_float(tr[0])
        end = _coerce_float(tr[1])
    if start is None or end is None:
        return None
    return start, end


def _turn_current_chunk(extra: Dict[str, Any], turn_i: int) -> Optional[int]:
    video_indices = _safe_list(extra.get("ts_chunk_video_indices"))
    event_indices = _safe_list(extra.get("ts_chunk_event_indices"))
    for values in (video_indices, event_indices):
        if turn_i >= len(values):
            continue
        try:
            chunk = int(values[turn_i])
        except (TypeError, ValueError):
            continue
        if chunk >= 0:
            return chunk
    try:
        n_chunks = int(extra.get("n_chunks") or -1)
    except (TypeError, ValueError):
        n_chunks = -1
    return max(0, n_chunks - 1) if n_chunks > 0 else None


def _tool_time_range_runtime_ok(
    kind: str,
    args: Dict[str, Any],
    *,
    current_chunk: Optional[int],
    chunk_sec: float = 1.0,
) -> bool:
    """Check only runtime-safe range constraints, not teacher/gold agreement.

    This keeps RL exploration open: non-gold ranges are allowed. Recall may be
    broad as long as it can touch observed memory. Compress is stricter because
    its range is written back into memory as summary provenance: it must be
    fully within already observed past time.
    """
    tr = _parse_tool_time_range(kind, args)
    if tr is None:
        return False
    start, end = tr
    if start < 0:
        return False
    if kind == "recall":
        if end < start:
            return False
    elif end <= start:
        return False
    if current_chunk is None:
        return True
    if kind == "compress":
        # Compression fires before processing current_chunk. Only chunks
        # strictly before current_chunk are in recent_thinks and safe to cover.
        compressible_end = max(0.0, float(current_chunk) * chunk_sec)
        return start >= 0.0 and end <= compressible_end
    # Recall may query any already-observed historical span, but not future
    # or the still-open current visual interval. The interval is closed, so
    # end_time must be strictly earlier than the current chunk timestamp.
    recallable_end = max(0.0, (float(current_chunk) * chunk_sec) - chunk_sec)
    return start >= 0.0 and end <= recallable_end


def _framework_format_score(extra: Dict[str, Any], solution_str: str) -> float:
    """CASIA-like RL format reward for framework executability.

    Score is the proportion of non-compress turns that can be parsed and
    executed. Compression has its own branch-local validity and quality reward,
    so compress turns are excluded here to avoid double-counting. This is a
    format-quality signal in [0, 1], not a hard trajectory gate.
    """
    from thinkstream.data.agent_protocol import parse_agent_output

    chunks = [
        str(x or "")
        for x in _safe_list(extra.get("ts_chunk_asst_texts"))
    ]
    if not chunks:
        chunks = _split_assistant_chunks(solution_str)
    if not chunks:
        return 0.0
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    scores: List[float] = []
    for turn_i, text in enumerate(chunks):
        turn_kind = str(turn_kinds[turn_i] or "") if turn_i < len(turn_kinds) else ""
        # Compression has its own branch-local validity and quality reward.
        # Keeping it out of the trajectory format branch avoids double-counting
        # the same malformed compact-memory turn in GDPO/HDPO.
        if turn_kind == "compress":
            continue
        parsed = parse_agent_output(
            text,
            allow_bare_answer=turn_kind in {"recall_response", "post_recall"},
            allow_bare_memory=turn_kind in {"", "compress"},
        )
        if str(parsed.get("kind") or "") == "compress":
            continue
        action_error = (
            str(action_errors[turn_i] or "").strip()
            if turn_i < len(action_errors)
            else ""
        )
        if parsed.get("format_error") or action_error:
            scores.append(0.0)
            continue
        kind = str(parsed.get("kind") or "")
        if kind == "recall":
            args = (parsed.get("tool_call") or {}).get("arguments") or {}
            ok = _tool_time_range_runtime_ok(
                kind,
                args,
                current_chunk=_turn_current_chunk(extra, turn_i),
            )
            scores.append(1.0 if ok else 0.0)
        else:
            scores.append(1.0)
    if not scores:
        # All turns were compress turns; leave trajectory-format neutral because
        # compression validity is scored by compress_quality.
        return 1.0
    return float(sum(scores) / len(scores))


def _int_chunks_from_value(value: Any) -> List[int]:
    value = _jsonable(value)
    if isinstance(value, (list, tuple, set)):
        out: List[int] = []
        for item in value:
            try:
                out.append(int(float(item)))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))
    try:
        return [int(float(value))]
    except (TypeError, ValueError):
        return []


def _chunks_from_memory_text(memory_text: str) -> set[int]:
    chunks: set[int] = set()
    for m in re.finditer(
        r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>',
        str(memory_text or ""),
        re.DOTALL | re.IGNORECASE,
    ):
        start = int(m.group(1))
        end = int(m.group(2) or m.group(1))
        if end < start:
            start, end = end, start
        chunks.update(range(start, end + 1))
    return chunks


_MEMORY_ENTRY_RE = re.compile(
    r'<m\s+t="(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?"\s*>(.*?)</m>',
    re.DOTALL | re.IGNORECASE,
)
_CAPTION_ENTRY_RE = re.compile(
    r'<c\s+t="(-?\d+(?:\.\d+)?)"\s*>(.*?)</c>',
    re.DOTALL | re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def _range_to_interval(start_raw: Any, end_raw: Any = None) -> Optional[tuple[float, float]]:
    try:
        start = float(start_raw)
        end = float(start_raw if end_raw in (None, "") else end_raw)
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return start, end + 1.0


def _clean_xml_text(text: Any) -> str:
    body = html.unescape(str(text or ""))
    body = re.sub(r"<[^>]+>", " ", body)
    return " ".join(body.split())


def _memory_entries_from_text(memory_text: str) -> List[tuple[float, float, str]]:
    entries: List[tuple[float, float, str]] = []
    for match in _MEMORY_ENTRY_RE.finditer(str(memory_text or "")):
        interval = _range_to_interval(match.group(1), match.group(2))
        body = _clean_xml_text(match.group(3))
        if interval is not None and body:
            entries.append((interval[0], interval[1], body))
    return entries


def _compact_memory_output_is_clean(output_text: str) -> bool:
    text = str(output_text or "")
    think_matches = list(re.finditer(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE))
    if len(think_matches) > 1:
        return False
    text = re.sub(r"<think>.*?</think>", " ", text, count=1, flags=re.DOTALL | re.IGNORECASE)
    text = _MEMORY_ENTRY_RE.sub(" ", text)
    text = re.sub(r"</?MEM>", " ", text, flags=re.IGNORECASE)
    return not text.strip()


def _source_entries_from_text(
    source_text: str,
    expected_chunks: Optional[set[int]] = None,
) -> List[tuple[float, float, str]]:
    entries = _memory_entries_from_text(source_text)
    for match in _CAPTION_ENTRY_RE.finditer(str(source_text or "")):
        interval = _range_to_interval(match.group(1), match.group(1))
        body = _clean_xml_text(match.group(2))
        if interval is not None and body:
            entries.append((interval[0], interval[1], body))
    if entries:
        return entries
    return [
        (float(chunk), float(chunk) + 1.0, "")
        for chunk in sorted(expected_chunks or [])
    ]


def _interval_len(interval: tuple[float, float]) -> float:
    return max(0.0, float(interval[1]) - float(interval[0]))


def _interval_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _interval_iou(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    inter = _interval_overlap(left, right)
    if inter <= 0.0:
        return 0.0
    union = _interval_len(left) + _interval_len(right) - inter
    return inter / union if union > 0.0 else 0.0


def _content_words(text: Any) -> set[str]:
    return {
        word.lower()
        for word in _WORD_RE.findall(_clean_xml_text(text))
        if len(word) > 1 or word.isdigit()
    }


def _merge_contiguous_intervals(
    intervals: List[tuple[float, float]],
    *,
    max_gap: float = 1e-6,
) -> List[tuple[float, float]]:
    merged: List[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + max_gap:
            merged.append((float(start), float(end)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], float(end)))
    return merged


def _caption_intervals_from_text(
    source_text: str,
    expected_chunks: Optional[set[int]] = None,
) -> List[tuple[float, float]]:
    intervals: List[tuple[float, float]] = []
    for match in _CAPTION_ENTRY_RE.finditer(str(source_text or "")):
        interval = _range_to_interval(match.group(1), match.group(1))
        if interval is not None:
            intervals.append((interval[0], interval[1]))
    if intervals:
        return sorted(intervals)
    return [
        (float(chunk), float(chunk) + 1.0)
        for chunk in sorted(expected_chunks or [])
    ]


def _merge_oldest_adjacent_units(
    units: List[tuple[float, float]],
    target_count: int,
) -> List[tuple[float, float]]:
    out = [(float(start), float(end)) for start, end in sorted(units)]
    target_count = max(1, int(target_count))
    while len(out) > target_count:
        first = out.pop(0)
        second = out.pop(0)
        out.insert(0, (first[0], max(first[1], second[1])))
    return out


def _partition_units_near_uniform(
    units: List[tuple[float, float]],
    target_count: int,
) -> List[tuple[float, float]]:
    units = [(float(start), float(end)) for start, end in sorted(units) if end > start]
    if not units:
        return []
    target_count = max(1, min(int(target_count), len(units)))
    if target_count >= len(units):
        return units
    total = sum(_interval_len(unit) for unit in units)
    if total <= 0.0:
        return units

    groups: List[List[tuple[float, float]]] = []
    current: List[tuple[float, float]] = []
    current_len = 0.0
    remaining_groups = target_count
    target_width = total / float(target_count)
    for idx, unit in enumerate(units):
        remaining_units = len(units) - idx
        if current and remaining_groups > 1:
            unit_len = _interval_len(unit)
            before = abs(current_len - target_width)
            after = abs((current_len + unit_len) - target_width)
            must_leave_units = remaining_units <= remaining_groups - 1
            if not must_leave_units and before <= after:
                groups.append(current)
                current = []
                current_len = 0.0
                remaining_groups -= 1
        current.append(unit)
        current_len += _interval_len(unit)
    if current:
        groups.append(current)

    while len(groups) > target_count:
        first = groups.pop(0)
        groups[0] = first + groups[0]
    while len(groups) < target_count and any(len(group) > 1 for group in groups):
        for group_i, group in enumerate(groups):
            if len(group) > 1:
                groups[group_i] = group[:-1]
                groups.insert(group_i + 1, [group[-1]])
                break

    return [(group[0][0], group[-1][1]) for group in groups if group]


def _boundary_aware_target_partitions(
    source_text: str,
    source_entries: List[tuple[float, float, str]],
    expected_chunks: Optional[set[int]],
) -> List[List[tuple[float, float]]]:
    max_items = max(1, _env_int("THINKSTREAM_COMPRESS_TARGET_MAX_ITEMS", 7))
    min_items = max(1, min(max_items, _env_int("THINKSTREAM_COMPRESS_TARGET_MIN_ITEMS", 4)))
    old_units = [(start, end) for start, end, _ in _memory_entries_from_text(source_text)]
    caption_units = _caption_intervals_from_text(source_text, expected_chunks)

    candidates: List[List[tuple[float, float]]] = []
    if old_units:
        # OLD_MEMORY boundaries are already recall indices. Keep them intact,
        # add contiguous NEW_CAPTIONS as whole units, and only merge adjacent
        # complete units when the memory budget is exceeded.
        base_units = _merge_contiguous_intervals(old_units, max_gap=-1.0)
        base_units.extend(_merge_contiguous_intervals(caption_units))
        base_units = _merge_contiguous_intervals(base_units, max_gap=-1.0)
        if not base_units:
            return []
        high = min(max_items, len(base_units))
        low = min(min_items, high)
        for target_count in range(low, high + 1):
            candidates.append(_merge_oldest_adjacent_units(base_units, target_count))
        if len(base_units) <= max_items:
            candidates.append(base_units)
    else:
        raw_units = caption_units or [(start, end) for start, end, _ in source_entries]
        raw_units = _merge_contiguous_intervals(raw_units, max_gap=-1.0)
        if not raw_units:
            return []
        high = min(max_items, len(raw_units))
        low = min(min_items, high)
        for target_count in range(low, high + 1):
            candidates.append(_partition_units_near_uniform(raw_units, target_count))

    deduped: List[List[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for candidate in candidates:
        key = tuple((round(start, 6), round(end, 6)) for start, end in candidate)
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _non_overlap_score(intervals: List[tuple[float, float]]) -> float:
    total = sum(_interval_len(interval) for interval in intervals)
    if total <= 0.0:
        return 0.0
    union = sum(_interval_len(interval) for interval in _merge_contiguous_intervals(intervals))
    return max(0.0, min(1.0, union / total))


def _partition_match_score(
    target: List[tuple[float, float]],
    emitted: List[tuple[float, float]],
) -> float:
    if not target or not emitted:
        return 0.0
    target_recall = sum(
        max(_interval_iou(tgt, out) for out in emitted)
        for tgt in target
    ) / float(len(target))
    emitted_precision = sum(
        max(_interval_iou(out, tgt) for tgt in target)
        for out in emitted
    ) / float(len(emitted))
    if target_recall <= 0.0 or emitted_precision <= 0.0:
        return 0.0
    return (target_recall * emitted_precision) ** 0.5


def _boundary_alignment_score(
    target: List[tuple[float, float]],
    emitted: List[tuple[float, float]],
) -> float:
    if not target or not emitted:
        return 0.0
    boundaries = [target[0][0]]
    for start, end in target:
        boundaries.append(start)
        boundaries.append(end)
    tolerance = _env_float("THINKSTREAM_COMPRESS_BOUNDARY_TOLERANCE", 1.01)
    hits = 0
    total = 0
    for start, end in emitted:
        for value in (start, end):
            total += 1
            if any(abs(float(value) - float(boundary)) <= tolerance for boundary in boundaries):
                hits += 1
    return float(hits / total) if total else 0.0


def _compress_time_score_with_details(
    output_entries: List[tuple[float, float, str]],
    source_entries: List[tuple[float, float, str]],
    *,
    source_text: str = "",
    expected_chunks: Optional[set[int]] = None,
) -> tuple[float, Dict[str, float]]:
    if not output_entries or not source_entries:
        return 0.0, {"count_score": 0.0, "target_item_count": 0.0, "non_overlap": 0.0}
    emitted = [(entry[0], entry[1]) for entry in output_entries]
    candidates = _boundary_aware_target_partitions(source_text, source_entries, expected_chunks)
    if not candidates:
        return 0.0, {"count_score": 0.0, "target_item_count": 0.0, "non_overlap": 0.0}

    best = 0.0
    best_count = 0
    best_boundary = 0.0
    for target in candidates:
        boundary_score = _boundary_alignment_score(target, emitted)
        score = _partition_match_score(target, emitted) * (0.5 + 0.5 * boundary_score)
        if score > best:
            best = score
            best_count = len(target)
            best_boundary = boundary_score

    max_items = max(1, _env_int("THINKSTREAM_COMPRESS_TARGET_MAX_ITEMS", 7))
    count = len(output_entries)
    count_score = 1.0 if count <= max_items else (float(max_items) / float(count)) ** 2
    non_overlap = _non_overlap_score(emitted)
    score = max(0.0, min(1.0, best * count_score * non_overlap))
    return score, {
        "count_score": float(count_score),
        "target_item_count": float(best_count),
        "non_overlap": float(non_overlap),
        "boundary_score": float(best_boundary),
    }


def _compress_time_score(
    output_entries: List[tuple[float, float, str]],
    source_entries: List[tuple[float, float, str]],
) -> float:
    score, _ = _compress_time_score_with_details(output_entries, source_entries)
    return score


def _compress_valid_time_score(
    output_entries: List[tuple[float, float, str]],
    source_entries: List[tuple[float, float, str]],
) -> float:
    if not output_entries or not source_entries:
        return 0.0
    horizon = (
        min(entry[0] for entry in source_entries),
        max(entry[1] for entry in source_entries),
    )
    emitted_len = sum(_interval_len((entry[0], entry[1])) for entry in output_entries)
    if emitted_len <= 0.0:
        return 0.0
    in_horizon = sum(
        _interval_overlap((entry[0], entry[1]), horizon)
        for entry in output_entries
    )
    return max(0.0, min(1.0, in_horizon / emitted_len))


def _compress_source_precision(
    output_entries: List[tuple[float, float, str]],
    source_entries: List[tuple[float, float, str]],
) -> float:
    if not output_entries or not source_entries:
        return 0.0
    scores: List[float] = []
    all_source_text = " ".join(entry[2] for entry in source_entries)
    all_source_words = _content_words(all_source_text)
    for out_start, out_end, out_text in output_entries:
        output_words = _content_words(out_text)
        if not output_words:
            scores.append(0.0)
            continue
        ref_text = " ".join(
            src_text
            for src_start, src_end, src_text in source_entries
            if _interval_overlap((out_start, out_end), (src_start, src_end)) > 0.0
        )
        ref_words = _content_words(ref_text) or all_source_words
        if not ref_words:
            scores.append(0.0)
            continue
        scores.append(len(output_words & ref_words) / float(len(output_words)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _compress_source_grounding_gate(source_precision: float) -> float:
    return max(0.0, min(1.0, float(source_precision) / 0.5))


def _score_compress_memory_update(
    output_text: str,
    source_text: str,
    expected_chunks: Optional[set[int]] = None,
    *,
    action_error: str = "",
    hit_max_tokens: bool = False,
) -> tuple[float, str, Dict[str, float]]:
    try:
        from thinkstream.data.agent_protocol import parse_agent_output
    except Exception:  # noqa: BLE001
        return 0.0, "parse_unavailable", {
            "valid_format": 0.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": 0.0,
        }

    if str(action_error or "").strip():
        return 0.0, "action_error", {
            "valid_format": 0.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": 0.0,
        }
    parsed = parse_agent_output(str(output_text or ""), allow_bare_memory=True)
    if parsed.get("format_error") or parsed.get("kind") != "compress":
        return 0.0, "invalid", {
            "valid_format": 0.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": 0.0,
        }
    output_entries = _memory_entries_from_text(str(parsed.get("memory_text") or ""))
    if not output_entries:
        return 0.0, "missing", {
            "valid_format": 0.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": 0.0,
        }
    if not _compact_memory_output_is_clean(output_text):
        return 0.0, "invalid", {
            "valid_format": 0.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": float(len(output_entries)),
        }
    source_entries = _source_entries_from_text(source_text, expected_chunks)
    if not source_entries:
        return 0.0, "no_source", {
            "valid_format": 1.0,
            "valid_time": 0.0,
            "time_score": 0.0,
            "source_precision": 0.0,
            "source_grounding": 0.0,
            "item_count": float(len(output_entries)),
        }

    valid_time = _compress_valid_time_score(output_entries, source_entries)
    time_score, time_details = _compress_time_score_with_details(
        output_entries,
        source_entries,
        source_text=source_text,
        expected_chunks=expected_chunks,
    )
    source_precision = _compress_source_precision(output_entries, source_entries)
    source_grounding = _compress_source_grounding_gate(source_precision)
    score = valid_time * source_grounding * (0.85 * time_score + 0.15 * source_precision)
    reason = "ok"
    if hit_max_tokens:
        score = 0.0
        reason = "hit_max"
    details = {
        "valid_format": 1.0,
        "valid_time": float(valid_time),
        "time_score": float(time_score),
        "source_precision": float(source_precision),
        "source_grounding": float(source_grounding),
        "item_count": float(len(output_entries)),
        **time_details,
    }
    return float(max(0.0, min(1.0, score))), reason, details


_COMPRESS_QUALITY_REASON_KEYS = (
    "parse_unavailable",
    "action_error",
    "invalid",
    "missing",
    "no_source",
    "ok",
    "hit_max",
)


def _compute_compress_quality(extra: Dict[str, Any]) -> Dict[str, float]:
    """Low-complexity compression branch signal.

    Scores whether compact-memory output is parseable, stays inside the source
    history, forms a selectable time partition, and uses words grounded in the
    corresponding OLD_MEMORY/NEW_CAPTIONS input. This is monitor-only in the
    scalar reward; recurrent GDPO applies the same formula per compress row.
    """
    texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    expected_by_turn = _safe_list(extra.get("ts_compress_expected_chunks"))
    hit_max_tokens = _safe_list(extra.get("ts_chunk_hit_max_tokens"))
    source_texts = _safe_list(extra.get("ts_compress_source_texts"))

    scores: List[float] = []
    valid_formats: List[float] = []
    valid_times: List[float] = []
    time_scores: List[float] = []
    source_precisions: List[float] = []
    source_groundings: List[float] = []
    item_counts: List[float] = []
    count_scores: List[float] = []
    boundary_scores: List[float] = []
    non_overlaps: List[float] = []
    target_item_counts: List[float] = []
    reasons: Dict[str, int] = {}
    n = max(len(texts), len(turn_kinds), len(expected_by_turn))
    for i in range(n):
        turn_kind = str(turn_kinds[i] or "") if i < len(turn_kinds) else ""
        if turn_kind != "compress":
            continue
        expected = set(
            _int_chunks_from_value(expected_by_turn[i])
            if i < len(expected_by_turn) else []
        )
        if not expected:
            continue
        action_error = (
            str(action_errors[i] or "").strip()
            if i < len(action_errors) else ""
        )
        text = str(texts[i] or "") if i < len(texts) else ""
        source_text = str(source_texts[i] or "") if i < len(source_texts) else ""
        score, reason, details = _score_compress_memory_update(
            text,
            source_text,
            expected,
            action_error=action_error,
            hit_max_tokens=bool(hit_max_tokens[i]) if i < len(hit_max_tokens) else False,
        )
        scores.append(score)
        valid_formats.append(float(details.get("valid_format", 0.0)))
        valid_times.append(float(details.get("valid_time", 0.0)))
        time_scores.append(float(details.get("time_score", 0.0)))
        source_precisions.append(float(details.get("source_precision", 0.0)))
        source_groundings.append(float(details.get("source_grounding", 0.0)))
        item_counts.append(float(details.get("item_count", 0.0)))
        count_scores.append(float(details.get("count_score", 0.0)))
        boundary_scores.append(float(details.get("boundary_score", 0.0)))
        non_overlaps.append(float(details.get("non_overlap", 0.0)))
        target_item_counts.append(float(details.get("target_item_count", 0.0)))
        reasons[reason] = reasons.get(reason, 0) + 1

    count = float(len(scores))
    if count <= 0:
        return {
            "compress_quality": 0.0,
            "compress_quality_count": 0.0,
            "compress_quality_parse_ok": 0.0,
            "compress_quality_cover_ok": 0.0,
            "compress_quality_old_only": 0.0,
            "compress_quality_valid_time": 0.0,
            "compress_quality_time_score": 0.0,
            "compress_quality_source_precision": 0.0,
            "compress_quality_source_grounding": 0.0,
            "compress_quality_item_count": 0.0,
            "compress_quality_count_score": 0.0,
            "compress_quality_boundary_score": 0.0,
            "compress_quality_non_overlap": 0.0,
            "compress_quality_target_item_count": 0.0,
            **{
                f"compress_quality_reason_{reason}": 0.0
                for reason in _COMPRESS_QUALITY_REASON_KEYS
            },
        }
    out = {
        "compress_quality": float(sum(scores) / len(scores)),
        "compress_quality_count": count,
        "compress_quality_parse_ok": float(sum(valid_formats) / len(scores)),
        "compress_quality_cover_ok": float(sum(1.0 for x in time_scores if x > 0.5) / len(scores)),
        "compress_quality_old_only": float(reasons.get("no_source", 0) / len(scores)),
        "compress_quality_valid_time": float(sum(valid_times) / len(scores)),
        "compress_quality_time_score": float(sum(time_scores) / len(scores)),
        "compress_quality_source_precision": float(sum(source_precisions) / len(scores)),
        "compress_quality_source_grounding": float(sum(source_groundings) / len(scores)),
        "compress_quality_item_count": float(sum(item_counts) / len(scores)),
        "compress_quality_count_score": float(sum(count_scores) / len(scores)),
        "compress_quality_boundary_score": float(sum(boundary_scores) / len(scores)),
        "compress_quality_non_overlap": float(sum(non_overlaps) / len(scores)),
        "compress_quality_target_item_count": float(sum(target_item_counts) / len(scores)),
    }
    for reason in _COMPRESS_QUALITY_REASON_KEYS:
        out[f"compress_quality_reason_{reason}"] = float(
            reasons.get(reason, 0) / len(scores)
        )
    for reason, reason_count in reasons.items():
        if reason not in _COMPRESS_QUALITY_REASON_KEYS:
            out[f"compress_quality_reason_{reason}"] = float(reason_count / len(scores))
    return out


_RL_ROLLOUT_AUDIT_COUNT = 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, max_chars: int = 600) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated chars={len(text)}>"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summarize_questions_for_audit(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in questions[: _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX_QUESTIONS", 64)]:
        out.append({
            "card_id": q.get("card_id") or q.get("id") or q.get("qid"),
            "family": q.get("family") or q.get("question_family") or q.get("answer_form"),
            "question": _short_text(q.get("question") or q.get("query") or "", 500),
            "ask_chunks": _jsonable(_safe_list(q.get("ask_chunks"))),
            "answer_chunks": _jsonable(_safe_list(q.get("answer_chunks"))),
            "support_chunks": _jsonable(_question_support_chunks(q)),
            "answer_form": q.get("answer_form"),
            "correct_option": q.get("correct_option"),
            "gold_answer": _short_text(q.get("gold_answer") or q.get("answer") or "", 500),
            "per_emit_answers": _jsonable(_safe_list(q.get("per_emit_answers"))),
        })
    return out


def _summarize_turns_for_audit(extra: Dict[str, Any], solution_str: str) -> List[Dict[str, Any]]:
    try:
        from thinkstream.data.agent_protocol import parse_agent_output
    except Exception:  # noqa: BLE001
        parse_agent_output = None

    texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    if not texts:
        texts = _split_assistant_chunks(solution_str)
    kinds = _safe_list(extra.get("ts_chunk_kinds"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    video_indices = _safe_list(extra.get("ts_chunk_video_indices"))
    event_indices = _safe_list(extra.get("ts_chunk_event_indices"))
    prompt_lens = _safe_list(extra.get("ts_chunk_prompt_lens"))
    response_lens = _safe_list(extra.get("ts_chunk_response_lens"))
    max_tokens = _safe_list(extra.get("ts_chunk_max_tokens"))
    hit_max_tokens = _safe_list(extra.get("ts_chunk_hit_max_tokens"))
    stop_reasons = _safe_list(extra.get("ts_chunk_stop_reasons"))
    recall_time_ranges = _safe_list(extra.get("ts_recall_time_ranges"))
    recall_returned_chunks = _safe_list(extra.get("ts_recall_returned_chunks"))
    recall_result_sources = _safe_list(extra.get("ts_recall_result_sources"))
    compress_expected_chunks = _safe_list(extra.get("ts_compress_expected_chunks"))
    compress_emitted_ranges = _safe_list(extra.get("ts_compress_emitted_ranges"))
    compress_source_texts = _safe_list(extra.get("ts_compress_source_texts"))
    max_turns = _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX_TURNS", 240)
    max_source_chars = _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX_SOURCE_CHARS", 8000)
    max_assistant_chars = _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX_ASSISTANT_CHARS", 1000)

    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(texts[:max_turns]):
        text = str(raw or "")
        turn_kind = str(turn_kinds[i] or "") if i < len(turn_kinds) else ""
        parsed = (
            parse_agent_output(
                text,
                allow_bare_answer=turn_kind in {"recall_response", "post_recall"},
                allow_bare_memory=turn_kind == "compress",
            )
            if parse_agent_output
            else {}
        )
        kind = str(parsed.get("kind") or (kinds[i] if i < len(kinds) else "") or "")
        item: Dict[str, Any] = {
            "turn": i,
            "kind": kind,
            "rollout_kind": kinds[i] if i < len(kinds) else None,
            "turn_kind": turn_kind or None,
            "video_chunk": video_indices[i] if i < len(video_indices) else None,
            "event_chunk": event_indices[i] if i < len(event_indices) else None,
            "action_space_error": action_errors[i] if i < len(action_errors) else "",
            "format_error": parsed.get("format_error"),
            "json_parse_error": (
                bool(parsed.get("format_error"))
                and "json" in str(parsed.get("format_error") or "").lower()
            ),
            "prompt_len": prompt_lens[i] if i < len(prompt_lens) else None,
            "response_len": response_lens[i] if i < len(response_lens) else None,
            "max_tokens": max_tokens[i] if i < len(max_tokens) else None,
            "hit_max_tokens": bool(hit_max_tokens[i]) if i < len(hit_max_tokens) else False,
            "stop_reason": stop_reasons[i] if i < len(stop_reasons) else "",
            "think": _short_text(parsed.get("think") or "", 360),
            "assistant_text": _short_text(text, max_assistant_chars),
        }
        if kind == "answer":
            item["answer_text"] = _short_text(parsed.get("answer_text") or "", 500)
        elif kind in {"recall", "compress"}:
            tool_call = parsed.get("tool_call") or {}
            args = tool_call.get("arguments") or {}
            item["tool_name"] = tool_call.get("name")
            item["tool_args"] = _jsonable(args)
            item["time_range_runtime_ok"] = _tool_time_range_runtime_ok(
                kind,
                args,
                current_chunk=_turn_current_chunk(extra, i),
            )
            if kind == "recall":
                item["requested_time_range"] = (
                    recall_time_ranges[i] if i < len(recall_time_ranges)
                    else {
                        "start_time": args.get("start_time"),
                        "end_time": args.get("end_time"),
                    }
                )
                item["returned_chunks"] = _jsonable(
                    recall_returned_chunks[i]
                    if i < len(recall_returned_chunks) else []
                )
                item["recall_result_source"] = (
                    recall_result_sources[i]
                    if i < len(recall_result_sources) else ""
                )
            elif kind == "compress":
                expected_chunks_raw = (
                    compress_expected_chunks[i]
                    if i < len(compress_expected_chunks) else []
                )
                source_text = (
                    str(compress_source_texts[i] or "")
                    if i < len(compress_source_texts) else ""
                )
                expected_chunks = set(_int_chunks_from_value(expected_chunks_raw))
                score, reason, details = _score_compress_memory_update(
                    text,
                    source_text,
                    expected_chunks,
                    action_error=str(action_errors[i] or "") if i < len(action_errors) else "",
                    hit_max_tokens=bool(hit_max_tokens[i]) if i < len(hit_max_tokens) else False,
                )
                item["expected_compressed_chunks"] = _jsonable(expected_chunks_raw)
                item["emitted_time_range"] = _jsonable(
                    compress_emitted_ranges[i]
                    if i < len(compress_emitted_ranges)
                    else args.get("time_range")
                )
                item["compress_source_text"] = _short_text(source_text, max_source_chars)
                item["compress_source_text_chars"] = len(source_text)
                item["compress_quality_score"] = score
                item["compress_quality_reason"] = reason
                item["compress_quality_details"] = _jsonable(details)
        out.append(item)
    if len(texts) > max_turns:
        out.append({"truncated_turns": len(texts) - max_turns})
    return out


def _audit_reasons(result: Dict[str, float], extra: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    score = float(result.get("score", 0.0) or 0.0)
    outcome = float(result.get("outcome", 0.0) or 0.0)
    fmt = float(result.get("format", 0.0) or 0.0)
    action_space = float(result.get("action_space", 0.0) or 0.0)
    n_questions = int(float(result.get("n_questions", 0.0) or 0.0))
    n_answered = int(float(result.get("n_answered", 0.0) or 0.0))

    if fmt < 0:
        reasons.append("format_invalid")
    if action_space < 0:
        reasons.append("action_space_error")
    if n_questions and n_answered < n_questions:
        reasons.append("unanswered_questions")
    if score > _env_float("THINKSTREAM_RL_AUDIT_HIGH_SCORE", 0.35) and outcome < 0.2:
        reasons.append("high_score_low_outcome")
    if outcome >= 0.9 and score < 0:
        reasons.append("good_outcome_negative_total")

    action_errors = [
        str(x) for x in _safe_list(extra.get("ts_chunk_action_space_errors"))
        if str(x or "").strip()
    ]
    if action_errors and "action_space_error" not in reasons:
        reasons.append("action_space_error")

    per_q_answers = _safe_list(extra.get("ts_per_q_answers"))
    for per_q in per_q_answers:
        for ev in _safe_list(per_q):
            if not isinstance(ev, dict):
                continue
            timing = str(ev.get("timing") or "")
            if timing == "early":
                reasons.append("early_answer")
                break
            if ev.get("counts_for_completion") is False:
                reasons.append("non_counted_answer")
                break

    if _safe_list(extra.get("ts_budget_abort_events")):
        reasons.append("budget_abort")
    if any(bool(x) for x in _safe_list(extra.get("ts_chunk_hit_max_tokens"))):
        reasons.append("hit_max_tokens")

    for raw in _safe_list(extra.get("ts_chunk_asst_texts")):
        parsed = {}
        try:
            from thinkstream.data.agent_protocol import parse_agent_output
            parsed = parse_agent_output(str(raw or ""))
        except Exception:  # noqa: BLE001
            parsed = {}
        if "json" in str(parsed.get("format_error") or "").lower():
            reasons.append("json_parse_error")
            break

    # Keep stable order while removing duplicates.
    seen: set[str] = set()
    return [r for r in reasons if not (r in seen or seen.add(r))]


def _append_jsonl_locked(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001
            pass
        f.write(line + "\n")
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            pass


def _maybe_audit_rl_rollout(
    *,
    data_source: str,
    extra: Dict[str, Any],
    solution_str: str,
    ground_truth: Any,
    result: Dict[str, float],
    questions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Sample reward-time rollout records for manual reward-hack audits."""
    global _RL_ROLLOUT_AUDIT_COUNT
    audit_path = (
        os.environ.get("THINKSTREAM_RL_ROLLOUT_AUDIT_PATH")
        or os.environ.get("THINKSTREAM_RL_ROLLOUT_AUDIT")
    )
    if not audit_path:
        return
    if _env_bool("THINKSTREAM_RL_ROLLOUT_AUDIT_FINAL_ONLY", True):
        action_is_final = extra.get("ts_action_is_final")
        if action_is_final is not None:
            if not bool(action_is_final):
                return
        else:
            action_index = extra.get("ts_action_index")
            n_actions = extra.get("ts_n_actions_in_traj")
            if action_index is not None and n_actions is not None:
                try:
                    if int(action_index) != int(n_actions) - 1:
                        return
                except (TypeError, ValueError):
                    pass
    max_records = _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX", 2000)
    if max_records >= 0 and _RL_ROLLOUT_AUDIT_COUNT >= max_records:
        return

    reasons = _audit_reasons(result, extra)
    prob = max(0.0, min(1.0, _env_float("THINKSTREAM_RL_ROLLOUT_AUDIT_PROB", 0.01)))
    random_sample = random.random() < prob
    if not reasons and not random_sample:
        return

    _RL_ROLLOUT_AUDIT_COUNT += 1
    max_solution_chars = _env_int("THINKSTREAM_RL_ROLLOUT_AUDIT_MAX_CHARS", 16000)
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "experiment_name": extra.get("experiment_name") or os.environ.get("THINKSTREAM_EXPERIMENT_NAME"),
        "sample_reason": reasons or ["random"],
        "data_source": data_source,
        "video_id": extra.get("video_id") or extra.get("trajectory_id") or extra.get("index"),
        "index": extra.get("index"),
        "reward": _jsonable(result),
        "counts": {
            "n_questions": result.get("n_questions"),
            "n_answers": result.get("n_answers"),
            "n_answered": result.get("n_answered"),
            "n_recall": extra.get("ts_n_recall"),
            "n_compress": extra.get("ts_n_compress"),
            "chunks_used": extra.get("ts_chunks_used"),
            "turns_used": extra.get("ts_turns_used"),
            "action_rows_used": extra.get("ts_action_rows_used"),
            "action_units_used": extra.get("ts_action_units_used"),
            "chunks_with_frames": extra.get("ts_chunks_with_frames"),
            "chunks_text_only": extra.get("ts_chunks_text_only"),
            "chunks_compress_inter": extra.get("ts_chunks_compress_inter"),
            "budget_aborts": len(_safe_list(extra.get("ts_budget_abort_events"))),
            "hit_max_token_turns": sum(
                1 for x in _safe_list(extra.get("ts_chunk_hit_max_tokens"))
                if bool(x)
            ),
            "action_index": extra.get("ts_action_index"),
            "n_actions_in_traj": extra.get("ts_n_actions_in_traj"),
            "action_is_final": extra.get("ts_action_is_final"),
            "action_unit_index": extra.get("ts_action_unit_index"),
            "n_action_units_in_traj": extra.get("ts_n_action_units_in_traj"),
            "action_subturn_index": extra.get("ts_action_subturn_index"),
            "action_unit_is_final": extra.get("ts_action_unit_is_final"),
        },
        "questions": _summarize_questions_for_audit(questions or []),
        "per_q_answers": _jsonable(_safe_list(extra.get("ts_per_q_answers"))),
        "turns": _summarize_turns_for_audit(extra, solution_str),
        "budget_abort_events": _jsonable(
            _safe_list(extra.get("ts_budget_abort_events"))[:20]
        ),
        "solution": _short_text(solution_str, max_solution_chars),
        "ground_truth": _jsonable(_coerce_ground_truth(ground_truth)),
    }
    try:
        _append_jsonl_locked(Path(audit_path), record)
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to write RL rollout audit sample: %s", e)


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


def _active_reward_keys() -> set[str]:
    """Reward keys that are allowed to affect the scalar RL objective.

    The default is the low-hack-risk profile for streaming QA: answer
    correctness, answer-decision timing, and framework format. Raw
    ``timing``/``silent_quality`` are telemetry; step/action/tool signals stay
    telemetry unless an explicit ablation opts in.
    """
    profile = (
        os.environ.get("THINKSTREAM_RL_REWARD_PROFILE")
        or os.environ.get("THINKSTREAM_REWARD_PROFILE")
        or "initial_outcome_time_format_decision"
    ).strip().lower()
    if profile in {"legacy", "full", "v12_full", "all"}:
        return {"outcome", "timing", "format", "silent_quality"}
    if profile in {"answer_only", "outcome_only"}:
        return {"outcome"}
    if profile in {"initial_outcome_time_format", "casia"}:
        return {"outcome", "timing", "format"}
    if profile in {
        "initial_outcome_time_format_decision_recall",
        "initial_outcome_decision_format_recall",
        "answer_decision_recall",
        "decision_recall",
    }:
        return {"outcome", "answer_decision", "format", "recall_answer", "compress_quality"}
    if profile in {
        "initial_outcome_time_format_decision",
        "initial_outcome_decision_format",
        "answer_decision",
        "decision",
    }:
        return {"outcome", "answer_decision", "format", "recall_answer", "compress_quality"}
    # Default / aliases: answer correctness + answer/no-answer timing decision.
    return {"outcome", "answer_decision", "format", "recall_answer", "compress_quality"}


def _step_action_reward_enabled() -> bool:
    """Whether per-chunk gold action alignment may change the scalar score."""
    return _env_bool("THINKSTREAM_ENABLE_STEP_ACTION_REWARD", False)


def _trajectory_solution_text(extra: Dict[str, Any], solution_str: str) -> str:
    """Use agent-loop recorded full turns when recurrent mode supplies only
    the current action's decoded ``solution_str`` to the reward function."""
    if not _env_bool("THINKSTREAM_SCORE_FULL_TRAJECTORY_TEXT", True):
        return solution_str
    texts = [
        str(x)
        for x in _safe_list(extra.get("ts_chunk_asst_texts"))
        if str(x or "").strip()
    ]
    if texts:
        return "\n".join(texts)
    return solution_str


def _combine_reward_parts(
    weights: Dict[str, float],
    parts: Dict[str, float],
) -> tuple[float, float]:
    """Combine reward components with correctness-scaled positive auxiliaries.

    Negative penalties always apply. Positive non-outcome rewards are scaled
    by the outcome gate so partial correctness receives partial auxiliary
    credit while wrong answers receive none.
    """
    active_keys = _active_reward_keys()
    gate = _outcome_gate(parts)
    outcome_total = (
        float(weights.get("outcome", 0.0) * parts.get("outcome", 0.0))
        if "outcome" in active_keys else 0.0
    )
    aux_total = 0.0
    for key, value in parts.items():
        if key == "outcome" or key not in active_keys:
            continue
        weighted = float(weights.get(key, 0.0) * value)
        if weighted > 0:
            aux_total += gate * weighted
        else:
            aux_total += weighted
    return outcome_total + aux_total, gate


def _reward_weights_with_recall(defaults: Dict[str, float]) -> Dict[str, float]:
    weights = dict(defaults)
    weights.setdefault(
        "recall_answer",
        _env_float("THINKSTREAM_RECALL_ANSWER_WEIGHT", 0.5),
    )
    weights.setdefault(
        "compress_quality",
        _env_float("THINKSTREAM_COMPRESS_QUALITY_WEIGHT", 0.3),
    )
    return _parse_hdpo_weight_overrides(weights)


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    clean_weights: List[float] = []
    for i in range(len(values)):
        try:
            w = float(weights[i])
        except (IndexError, TypeError, ValueError):
            w = 1.0
        clean_weights.append(w if w > 0.0 else 1.0)
    denom = sum(clean_weights)
    if denom <= 0.0:
        return sum(float(v) for v in values) / len(values)
    return sum(float(v) * w for v, w in zip(values, clean_weights)) / denom


def _answer_weight_for_question(q: Dict[str, Any]) -> float:
    """Number of expected answer slots represented by a question."""
    per_emit = _safe_list(q.get("per_emit_answers"))
    emit_chunks: List[int] = []
    for item in per_emit:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, dict) or item.get("chunk") is None:
            continue
        try:
            emit_chunks.append(int(item["chunk"]))
        except (TypeError, ValueError):
            continue
    if emit_chunks:
        return float(max(1, len(set(emit_chunks))))

    answer_chunks = _safe_list(q.get("answer_chunks"))
    answer_chunks_int: List[int] = []
    for x in answer_chunks:
        try:
            answer_chunks_int.append(int(x))
        except (TypeError, ValueError):
            continue
    if answer_chunks_int:
        return float(max(1, len(set(answer_chunks_int))))
    return 1.0


def _question_answer_opportunity_chunks(q: Dict[str, Any]) -> List[int]:
    """Return chunks where a question can first be fairly scored."""
    chunks: List[int] = []
    for item in _safe_list(q.get("per_emit_answers")):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, dict) or item.get("chunk") is None:
            continue
        try:
            chunks.append(int(item["chunk"]))
        except (TypeError, ValueError):
            continue
    for raw in _safe_list(q.get("answer_chunks")):
        try:
            chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(c for c in chunks if c >= 0))


def _gold_action_map_for_extra(extra: Dict[str, Any]) -> Dict[str, str]:
    gold_action = extra.get("gold_action_per_chunk") or {}
    if not gold_action and extra.get("video_id"):
        traj = _load_traj_index().get(str(extra["video_id"]))
        if traj:
            gold_action = traj.get("gold_action_per_chunk", {}) or {}
    return _strip_offline_compress_actions(gold_action)


def _recall_label_chunks_for_question(
    q: Dict[str, Any],
    gold_action_per_chunk: Dict[str, str],
) -> List[int]:
    recall_actions = {"recall", "recall_silent"}
    ask_chunks = _coerce_int_list(q.get("ask_chunks") or [q.get("ask_chunk")])
    direct = [
        chunk for chunk in ask_chunks
        if str((gold_action_per_chunk or {}).get(str(chunk), "")).strip() in recall_actions
    ]
    if direct:
        return sorted(set(direct))

    bounds = ask_chunks + _question_answer_opportunity_chunks(q)
    if not bounds:
        return []
    lo = min(bounds)
    hi = max(bounds)
    out: List[int] = []
    for key, action in (gold_action_per_chunk or {}).items():
        if str(action).strip() not in recall_actions:
            continue
        try:
            chunk = int(key)
        except (TypeError, ValueError):
            continue
        if lo <= chunk <= hi:
            out.append(chunk)
    return sorted(set(out))


def _answer_event_chunks_for_question(answer_events: Any, fallback_chunk: Any) -> List[int]:
    chunks: List[int] = []
    for ev in _safe_list(answer_events):
        if hasattr(ev, "tolist"):
            ev = ev.tolist()
        if not isinstance(ev, dict):
            continue
        try:
            chunk = int(ev.get("chunk", -1))
        except (TypeError, ValueError):
            chunk = -1
        if chunk >= 0:
            chunks.append(chunk)
    try:
        chunk = int(fallback_chunk)
    except (TypeError, ValueError):
        chunk = -1
    if chunk >= 0:
        chunks.append(chunk)
    return sorted(set(chunks))


def _used_recall_chunks(extra: Dict[str, Any]) -> set[int]:
    chunk_kinds = _safe_list(extra.get("ts_chunk_kinds"))
    if not chunk_kinds:
        return set()
    chunk_texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    out: set[int] = set()

    try:
        from thinkstream.data.agent_protocol import parse_agent_output
    except Exception:
        parse_agent_output = None

    require_valid = _env_bool("THINKSTREAM_RECALL_ANSWER_REQUIRE_VALID_RECALL", True)
    for turn_i, kind_raw in enumerate(chunk_kinds):
        turn_kind = (
            str(turn_kinds[turn_i] or "")
            if turn_i < len(turn_kinds)
            else ""
        )
        if turn_kind in {"recall_response", "post_recall", "compress"}:
            continue
        text = str(chunk_texts[turn_i] or "") if turn_i < len(chunk_texts) else ""
        if _model_action_from_turn(str(kind_raw or "unknown"), text) != "recall":
            continue
        action_error = (
            str(action_errors[turn_i] or "").strip()
            if turn_i < len(action_errors)
            else ""
        )
        if require_valid and action_error:
            continue
        if require_valid and parse_agent_output is not None:
            parsed = parse_agent_output(text)
            args = ((parsed.get("tool_call") or {}).get("arguments") or {})
            if not _tool_time_range_runtime_ok(
                "recall",
                args,
                current_chunk=_turn_current_chunk(extra, turn_i),
            ):
                continue
        chunk = _turn_event_chunk(extra, turn_i)
        if chunk >= 0:
            out.add(int(chunk))
    return out


def _question_used_recall(
    q: Dict[str, Any],
    *,
    label_chunks: List[int],
    used_recall_chunks: set[int],
    answer_event_chunks: List[int],
) -> bool:
    if not label_chunks or not used_recall_chunks:
        return False
    start = min(label_chunks)
    end_candidates = answer_event_chunks + _question_answer_opportunity_chunks(q) + label_chunks
    end = max(end_candidates) if end_candidates else max(label_chunks)
    return any(start <= chunk <= end for chunk in used_recall_chunks)


def _observed_horizon_chunk(extra: Dict[str, Any]) -> Optional[int]:
    """Highest video chunk actually rolled out for this trajectory."""
    observed: List[int] = []
    for raw in _safe_list(extra.get("ts_chunk_video_indices")):
        try:
            ci = int(raw)
        except (TypeError, ValueError):
            continue
        if ci >= 0:
            observed.append(ci)
    if observed:
        return max(observed)
    try:
        chunks_used = int(float(extra.get("ts_chunks_used", 0) or 0))
    except (TypeError, ValueError):
        chunks_used = 0
    if chunks_used > 0:
        return chunks_used - 1
    return None


def _combine_multi_q_reward_parts(
    weights: Dict[str, float],
    per_question_parts: List[Dict[str, float]],
    trajectory_parts: Dict[str, float],
    question_weights: Optional[List[float]] = None,
) -> tuple[float, float, List[float]]:
    """Combine multi-question rewards at expected-answer granularity.

    ``per_question_parts`` contains outcome/timing plus monitor-only fields
    such as silent_quality. Each question gates its own positive auxiliary
    rewards, then question scores are averaged with the number of expected
    answer slots as weight. Trajectory-level format is applied once. Tool/step
    fields remain in diagnostics unless the reward profile explicitly enables
    them.
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

    q_weights = question_weights or [1.0] * len(per_question_parts)
    total = _weighted_mean(per_question_scores, q_weights)
    gate = _weighted_mean(per_question_gates, q_weights)
    active_keys = _active_reward_keys()

    for key, value in trajectory_parts.items():
        if key not in active_keys:
            continue
        weighted = float(weights.get(key, 0.0) * value)
        if weighted > 0:
            total += gate * weighted
        else:
            total += weighted
    return total, gate, per_question_scores


def _parse_hdpo_weight_overrides(defaults: Dict[str, float]) -> Dict[str, float]:
    raw = os.environ.get("THINKSTREAM_HDPO_WEIGHTS", "").strip()
    if not raw:
        return dict(defaults)
    weights = dict(defaults)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key in weights:
                    weights[key] = float(value)
            return weights
    except Exception:
        pass
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            continue
        try:
            weights[key] = float(value)
        except ValueError:
            continue
    return weights


def _question_segment_answer_point(q: Dict[str, Any]) -> int:
    opportunity_chunks = _question_answer_opportunity_chunks(q)
    if opportunity_chunks:
        return int(max(opportunity_chunks))

    chunks: List[int] = []
    for raw in _safe_list(q.get("ask_chunks")):
        try:
            chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    try:
        ask_chunk = int(q.get("ask_chunk", -1))
    except (TypeError, ValueError):
        ask_chunk = -1
    if ask_chunk >= 0:
        chunks.append(ask_chunk)
    return int(max(chunks)) if chunks else 0


def _segment_index_for_chunk(
    chunk: int,
    segment_starts: List[int],
    segment_ends: List[int],
) -> Optional[int]:
    if chunk < 0:
        return None
    for seg_i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        if int(start) <= chunk <= int(end):
            return seg_i
    return None


def _turn_event_chunk(extra: Dict[str, Any], turn_i: int) -> int:
    event_indices = _safe_list(extra.get("ts_chunk_event_indices"))
    video_indices = _safe_list(extra.get("ts_chunk_video_indices"))
    for values in (event_indices, video_indices):
        if turn_i >= len(values):
            continue
        try:
            chunk = int(values[turn_i])
        except (TypeError, ValueError):
            continue
        if chunk >= 0:
            return chunk
    return -1


def _compress_turn_scores(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    expected_by_turn = _safe_list(extra.get("ts_compress_expected_chunks"))
    hit_max_tokens = _safe_list(extra.get("ts_chunk_hit_max_tokens"))
    source_texts = _safe_list(extra.get("ts_compress_source_texts"))

    out: List[Dict[str, Any]] = []
    n = max(len(texts), len(turn_kinds), len(expected_by_turn))
    for i in range(n):
        turn_kind = str(turn_kinds[i] or "") if i < len(turn_kinds) else ""
        if turn_kind != "compress":
            continue
        event_chunk = _turn_event_chunk(extra, i)
        expected = set(
            _int_chunks_from_value(expected_by_turn[i])
            if i < len(expected_by_turn) else []
        )
        if not expected:
            continue
        action_error = (
            str(action_errors[i] or "").strip()
            if i < len(action_errors) else ""
        )
        text = str(texts[i] or "") if i < len(texts) else ""
        source_text = str(source_texts[i] or "") if i < len(source_texts) else ""
        score, reason, _ = _score_compress_memory_update(
            text,
            source_text,
            expected,
            action_error=action_error,
            hit_max_tokens=bool(hit_max_tokens[i]) if i < len(hit_max_tokens) else False,
        )
        out.append({
            "turn": i,
            "event_chunk": int(event_chunk),
            "score": float(score),
            "reason": reason,
        })
    return out


def _compute_compress_quality_by_segment(
    extra: Dict[str, Any],
    segment_starts: List[int],
    segment_ends: List[int],
) -> tuple[List[float], List[float]]:
    per_segment_scores: List[List[float]] = [[] for _ in segment_starts]
    for item in _compress_turn_scores(extra):
        seg_i = _segment_index_for_chunk(
            int(item.get("event_chunk", -1)),
            segment_starts,
            segment_ends,
        )
        if seg_i is None:
            continue
        per_segment_scores[seg_i].append(float(item.get("score", 0.0)))

    means: List[float] = []
    counts: List[float] = []
    for scores in per_segment_scores:
        counts.append(float(len(scores)))
        means.append(float(sum(scores) / len(scores)) if scores else 0.0)
    return means, counts


def _combine_segment_reward_parts(
    weights: Dict[str, float],
    parts: Dict[str, float],
    *,
    has_questions: bool = True,
) -> tuple[float, float]:
    gate = _outcome_gate(parts)
    total = float(weights.get("outcome", 0.0) * parts.get("outcome", 0.0))
    for key in ("answer_decision", "format", "compress_quality"):
        weighted = float(weights.get(key, 0.0) * parts.get(key, 0.0))
        if weighted > 0.0:
            # Compression quality is local process quality. It should remain
            # visible even when the segment's later answer is wrong; answer
            # service pressure is applied by the trainer through the global and
            # future-answer advantage mix.
            local_gate = 1.0 if key == "compress_quality" else gate
            total += local_gate * weighted
        else:
            total += weighted
    return total, gate


def _framework_format_scores_by_segment(
    extra: Dict[str, Any],
    segment_starts: List[int],
    segment_ends: List[int],
    *,
    fallback: float,
) -> List[float]:
    try:
        from thinkstream.data.agent_protocol import parse_agent_output
    except Exception:
        return [float(fallback)] * len(segment_starts)

    texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    if not texts:
        return [float(fallback)] * len(segment_starts)
    action_errors = _safe_list(extra.get("ts_chunk_action_space_errors"))
    turn_kinds = _safe_list(extra.get("ts_chunk_turn_kinds"))
    per_segment_scores: List[List[float]] = [[] for _ in segment_starts]
    for turn_i, text in enumerate(texts):
        turn_kind = str(turn_kinds[turn_i] or "") if turn_i < len(turn_kinds) else ""
        if turn_kind == "compress":
            continue
        seg_i = _segment_index_for_chunk(
            _turn_event_chunk(extra, turn_i),
            segment_starts,
            segment_ends,
        )
        if seg_i is None:
            continue
        parsed = parse_agent_output(
            str(text or ""),
            allow_bare_answer=turn_kind in {"recall_response", "post_recall"},
            allow_bare_memory=turn_kind == "",
        )
        if str(parsed.get("kind") or "") == "compress":
            continue
        action_error = (
            str(action_errors[turn_i] or "").strip()
            if turn_i < len(action_errors)
            else ""
        )
        if parsed.get("format_error") or action_error:
            per_segment_scores[seg_i].append(0.0)
            continue
        kind = str(parsed.get("kind") or "")
        if kind == "recall":
            args = (parsed.get("tool_call") or {}).get("arguments") or {}
            ok = _tool_time_range_runtime_ok(
                kind,
                args,
                current_chunk=_turn_current_chunk(extra, turn_i),
            )
            per_segment_scores[seg_i].append(1.0 if ok else 0.0)
        else:
            per_segment_scores[seg_i].append(1.0)

    scores: List[float] = []
    for seg_scores in per_segment_scores:
        if not seg_scores:
            scores.append(1.0)
        else:
            scores.append(float(sum(seg_scores) / len(seg_scores)))
    return scores


def _json_compact(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _segment_meta_key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _segment_score_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((float(v) - mean) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


def _question_indices_in_segment(
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
    start: int,
    end: int,
) -> List[int]:
    out: List[int] = []
    for q_idx in scored_question_indices:
        if q_idx >= len(questions):
            continue
        answer_point = _question_segment_answer_point(questions[q_idx])
        if int(start) <= answer_point <= int(end):
            out.append(int(q_idx))
    return out


def _build_segment_reward_metadata_for_bounds(
    *,
    prefix: str,
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
    per_q_parts: List[Dict[str, float]],
    per_q_weights: List[float],
    fmt: float,
    extra: Dict[str, Any],
    segment_starts: List[int],
    segment_ends: List[int],
    segment_question_indices: Optional[List[List[int]]] = None,
    segment_compress_scores: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    q_pos_by_idx = {
        int(q_idx): pos
        for pos, q_idx in enumerate(scored_question_indices)
        if pos < len(per_q_parts)
    }
    if segment_question_indices is None:
        segment_question_indices = [
            _question_indices_in_segment(
                questions,
                scored_question_indices,
                int(start),
                int(end),
            )
            for start, end in zip(segment_starts, segment_ends)
        ]

    if segment_compress_scores is None:
        segment_compress, segment_compress_counts = _compute_compress_quality_by_segment(
            extra,
            segment_starts,
            segment_ends,
        )
    else:
        segment_compress = [
            float(sum(scores) / len(scores)) if scores else 0.0
            for scores in segment_compress_scores
        ]
        segment_compress_counts = [float(len(scores)) for scores in segment_compress_scores]

    segment_formats = _framework_format_scores_by_segment(
        extra,
        segment_starts,
        segment_ends,
        fallback=fmt,
    )
    gdpo_weights = _parse_hdpo_weight_overrides({
        "outcome": 1.0,
        "answer_decision": 0.5,
        "format": 0.1,
        "compress_quality": 0.3,
    })

    segment_scores: List[float] = []
    segment_gates: List[float] = []
    segment_outcomes: List[float] = []
    segment_decisions: List[float] = []
    segment_timings: List[float] = []
    segment_answer_scores: List[float] = []
    clean_question_indices: List[List[int]] = []
    for seg_i, q_indices in enumerate(segment_question_indices):
        positions = [
            q_pos_by_idx[int(q_idx)]
            for q_idx in q_indices
            if int(q_idx) in q_pos_by_idx
        ]
        q_weights = [
            float(per_q_weights[pos]) if pos < len(per_q_weights) else 1.0
            for pos in positions
        ]
        q_outcomes = [
            float(per_q_parts[pos].get("outcome", 0.0))
            for pos in positions
        ]
        q_decisions = [
            float(per_q_parts[pos].get("answer_decision", 0.0))
            for pos in positions
        ]
        q_timings = [
            float(per_q_parts[pos].get("timing", 0.0))
            for pos in positions
        ]
        seg_parts = {
            "outcome": _weighted_mean(q_outcomes, q_weights),
            "answer_decision": _weighted_mean(q_decisions, q_weights),
            "timing": _weighted_mean(q_timings, q_weights),
            "format": float(segment_formats[seg_i]) if seg_i < len(segment_formats) else float(fmt),
            "compress_quality": float(segment_compress[seg_i]) if seg_i < len(segment_compress) else 0.0,
        }
        seg_score, seg_gate = _combine_segment_reward_parts(
            gdpo_weights,
            seg_parts,
            has_questions=bool(positions),
        )
        answer_only_parts = dict(seg_parts)
        answer_only_parts["compress_quality"] = 0.0
        answer_score, _ = _combine_segment_reward_parts(
            gdpo_weights,
            answer_only_parts,
            has_questions=bool(positions),
        )
        segment_scores.append(float(seg_score))
        segment_gates.append(float(seg_gate))
        segment_outcomes.append(float(seg_parts["outcome"]))
        segment_decisions.append(float(seg_parts["answer_decision"]))
        segment_timings.append(float(seg_parts["timing"]))
        segment_answer_scores.append(float(answer_score))
        clean_question_indices.append([int(q_idx) for q_idx in q_indices])

    segment_count = float(len(segment_scores))
    compress_count = float(sum(segment_compress_counts))
    score_range = (
        float(max(segment_scores) - min(segment_scores))
        if segment_scores else 0.0
    )
    return {
        _segment_meta_key(prefix, "segment_count"): segment_count,
        _segment_meta_key(prefix, "segment_score_mean"): _mean_or_zero(segment_scores),
        _segment_meta_key(prefix, "segment_score_std"): _segment_score_std(segment_scores),
        _segment_meta_key(prefix, "segment_score_range"): score_range,
        _segment_meta_key(prefix, "segment_answer_score_mean"): _mean_or_zero(segment_answer_scores),
        _segment_meta_key(prefix, "segment_answer_score_std"): _segment_score_std(segment_answer_scores),
        _segment_meta_key(prefix, "segment_answer_score_range"): (
            float(max(segment_answer_scores) - min(segment_answer_scores))
            if segment_answer_scores else 0.0
        ),
        _segment_meta_key(prefix, "segment_outcome_mean"): _mean_or_zero(segment_outcomes),
        _segment_meta_key(prefix, "segment_answer_decision_mean"): _mean_or_zero(segment_decisions),
        _segment_meta_key(prefix, "segment_format_mean"): _mean_or_zero(segment_formats),
        _segment_meta_key(prefix, "segment_compress_quality_mean"): _mean_or_zero(segment_compress),
        _segment_meta_key(prefix, "segment_compress_count"): compress_count,
        _segment_meta_key(prefix, "segment_starts_json"): _json_compact(segment_starts),
        _segment_meta_key(prefix, "segment_ends_json"): _json_compact(segment_ends),
        _segment_meta_key(prefix, "segment_scores_json"): _json_compact(segment_scores),
        _segment_meta_key(prefix, "segment_answer_scores_json"): _json_compact(segment_answer_scores),
        _segment_meta_key(prefix, "segment_gates_json"): _json_compact(segment_gates),
        _segment_meta_key(prefix, "segment_outcomes_json"): _json_compact(segment_outcomes),
        _segment_meta_key(prefix, "segment_answer_decisions_json"): _json_compact(segment_decisions),
        _segment_meta_key(prefix, "segment_timings_json"): _json_compact(segment_timings),
        _segment_meta_key(prefix, "segment_formats_json"): _json_compact(segment_formats[:len(segment_scores)]),
        _segment_meta_key(prefix, "segment_compress_quality_json"): _json_compact(segment_compress),
        _segment_meta_key(prefix, "segment_compress_counts_json"): _json_compact(segment_compress_counts),
        _segment_meta_key(prefix, "segment_question_indices_json"): _json_compact(clean_question_indices),
    }


def _credit_horizon_chunk(
    extra: Dict[str, Any],
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
) -> int:
    candidates: List[int] = []
    horizon = _observed_horizon_chunk(extra)
    if horizon is not None:
        candidates.append(int(horizon))
    for q_idx in scored_question_indices:
        if q_idx < len(questions):
            candidates.append(_question_segment_answer_point(questions[q_idx]))
    for item in _compress_turn_scores(extra):
        chunk = int(item.get("event_chunk", -1))
        if chunk >= 0:
            candidates.append(chunk)
    return max(candidates) if candidates else 0


def _build_question_segment_reward_metadata(
    *,
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
    per_q_parts: List[Dict[str, float]],
    per_q_weights: List[float],
    fmt: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Build answer-point segments for recurrent credit assignment.

    Each segment covers the stream since the previous answer point and carries
    a local score: outcome + answer-decision + framework format + any compress
    quality that happened inside the same segment.
    """
    entries: List[tuple[int, int, int]] = []
    for pos, q_idx in enumerate(scored_question_indices):
        if pos >= len(per_q_parts) or q_idx >= len(questions):
            continue
        entries.append((
            _question_segment_answer_point(questions[q_idx]),
            pos,
            q_idx,
        ))
    entries.sort(key=lambda x: (int(x[0]), int(x[2])))

    grouped: List[tuple[int, List[int], List[int]]] = []
    for answer_point, pos, q_idx in entries:
        if grouped and grouped[-1][0] == answer_point:
            grouped[-1][1].append(pos)
            grouped[-1][2].append(q_idx)
        else:
            grouped.append((int(answer_point), [pos], [q_idx]))

    segment_starts: List[int] = []
    segment_ends: List[int] = []
    prev_end = -1
    for answer_point, _, _ in grouped:
        start = max(0, prev_end + 1)
        end = max(start, int(answer_point))
        segment_starts.append(start)
        segment_ends.append(end)
        prev_end = end

    segment_question_indices: List[List[int]] = []
    for _, _, q_indices in grouped:
        segment_question_indices.append([int(q_idx) for q_idx in q_indices])

    return _build_segment_reward_metadata_for_bounds(
        prefix="",
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
        segment_starts=segment_starts,
        segment_ends=segment_ends,
        segment_question_indices=segment_question_indices,
    )


def _compress_boundary_bounds(
    extra: Dict[str, Any],
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
) -> tuple[List[int], List[int]]:
    horizon = _credit_horizon_chunk(extra, questions, scored_question_indices)
    compress_chunks = sorted({
        int(item.get("event_chunk", -1))
        for item in _compress_turn_scores(extra)
        if int(item.get("event_chunk", -1)) >= 0
    })
    if not compress_chunks:
        return [0], [max(0, horizon)]

    starts: List[int] = []
    ends: List[int] = []
    first = compress_chunks[0]
    if first > 0:
        starts.append(0)
        ends.append(first - 1)
    for i, chunk in enumerate(compress_chunks):
        next_chunk = compress_chunks[i + 1] if i + 1 < len(compress_chunks) else None
        starts.append(int(chunk))
        end = int(next_chunk - 1) if next_chunk is not None else int(max(horizon, chunk))
        ends.append(max(int(chunk), end))
    return starts, ends


def _build_compress_boundary_segment_reward_metadata(
    *,
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
    per_q_parts: List[Dict[str, float]],
    per_q_weights: List[float],
    fmt: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    starts, ends = _compress_boundary_bounds(extra, questions, scored_question_indices)
    return _build_segment_reward_metadata_for_bounds(
        prefix="compress",
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
        segment_starts=starts,
        segment_ends=ends,
    )


def _build_compress_future_segment_reward_metadata(
    *,
    questions: List[Dict[str, Any]],
    scored_question_indices: List[int],
    per_q_parts: List[Dict[str, float]],
    per_q_weights: List[float],
    fmt: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    horizon = _credit_horizon_chunk(extra, questions, scored_question_indices)
    scores_by_chunk: Dict[int, List[float]] = {}
    for item in _compress_turn_scores(extra):
        chunk = int(item.get("event_chunk", -1))
        if chunk < 0:
            continue
        scores_by_chunk.setdefault(chunk, []).append(float(item.get("score", 0.0)))
    compress_chunks = sorted(scores_by_chunk)
    if not compress_chunks:
        return _build_segment_reward_metadata_for_bounds(
            prefix="compress_future",
            questions=questions,
            scored_question_indices=scored_question_indices,
            per_q_parts=per_q_parts,
            per_q_weights=per_q_weights,
            fmt=fmt,
            extra=extra,
            segment_starts=[],
            segment_ends=[],
            segment_question_indices=[],
            segment_compress_scores=[],
        )

    starts: List[int] = []
    ends: List[int] = []
    question_indices: List[List[int]] = []
    compress_scores: List[List[float]] = []
    for i, chunk in enumerate(compress_chunks):
        next_chunk = compress_chunks[i + 1] if i + 1 < len(compress_chunks) else None
        end = int(next_chunk - 1) if next_chunk is not None else int(max(horizon, chunk))
        starts.append(int(chunk))
        ends.append(max(int(chunk), end))
        question_indices.append(_question_indices_in_segment(
            questions,
            scored_question_indices,
            int(chunk),
            max(int(chunk), end),
        ))
        compress_scores.append(scores_by_chunk.get(int(chunk), []))

    return _build_segment_reward_metadata_for_bounds(
        prefix="compress_future",
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
        segment_starts=starts,
        segment_ends=ends,
        segment_question_indices=question_indices,
        segment_compress_scores=compress_scores,
    )


def _answer_decision_reward(
    rewards: Dict[str, Any],
    answer_chunk: Optional[int],
    visible_start: Optional[int],
    visible_end: Optional[int],
    *,
    late_window_chunks: int = 2,
    has_answer: Optional[bool] = None,
) -> float:
    fn = rewards.get("answer_decision")
    if callable(fn):
        try:
            return float(fn(
                answer_chunk,
                visible_start,
                visible_end,
                late_window_chunks=late_window_chunks,
                has_answer=has_answer,
            ))
        except TypeError:
            return float(fn(answer_chunk, visible_start, visible_end))
    if has_answer is None:
        has_answer = answer_chunk is not None and int(answer_chunk) >= 0
    if visible_start is None:
        return -1.0 if has_answer else 0.0
    if not has_answer or answer_chunk is None or int(answer_chunk) < 0:
        return -1.0
    return float(rewards["timing"](
        answer_chunk,
        visible_start,
        visible_end,
        late_window_chunks=late_window_chunks,
    ))


def _score_one_question(
    rewards: Dict[str, Any],
    *,
    q: Dict[str, Any],
    model_answer: str,
    answered_chunk: int,
) -> Dict[str, float]:
    """Score a single question's outcome + answer-decision timing.

    Returns dict with keys: outcome, answer_decision, timing,
    silent_quality, answered.
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
    # Timing is anchored on answer_chunks. The query may be active earlier
    # for forward/wait cards, but an answer before the expected answer chunk
    # is early, not on-time.
    window_marks = answer_chunks_int or ask_chunks_int
    visible_start = min(window_marks) if window_marks else None
    visible_end = max(window_marks) if window_marks else None

    answered = 1.0 if (model_answer or "").strip() else 0.0

    # Outcome — form-aware liberal matching. Dispatches to the right
    # matcher for each of pass3a's 5 answer_form values (multiple_choice
    # / binary / number / short_exact / descriptive). Falls back to
    # v12_rewards.compute_outcome_reward only if our dispatcher
    # raises (defensive — should be a no-op in practice).
    if not answered:
        outcome = 0.0
    elif visible_start is not None and answered_chunk < visible_start:
        # Correct text before the answer evidence/time is still an early
        # response; do not give outcome credit for it.
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

    # Timing — bucket the answered_chunk against expected answer chunks.
    timing = float(rewards["timing"](
        answered_chunk if answered_chunk >= 0 else None,
        visible_start, visible_end,
        late_window_chunks=2,
    ))
    answer_decision = _answer_decision_reward(
        rewards,
        answered_chunk if answered_chunk >= 0 else None,
        visible_start,
        visible_end,
        late_window_chunks=2,
        has_answer=bool(answered),
    )

    # Silent quality — audit P1.6: per-Q silent decision.
    # compute_silent_quality takes (final_answer, gold_action, gold_answer)
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
        "answer_decision": answer_decision,
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
        events.append({
            "chunk": chunk,
            "text": text,
            "timing": str(e.get("timing") or ""),
            "counts_for_completion": e.get("counts_for_completion"),
            "expected_chunk": e.get("expected_chunk"),
        })
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
    ask_chunks = _safe_list(q.get("ask_chunks"))
    ask_chunks_int: List[int] = []
    for x in ask_chunks:
        try:
            ask_chunks_int.append(int(x))
        except (TypeError, ValueError):
            continue
    per_emit = _safe_list(q.get("per_emit_answers"))
    is_multi = len(answer_chunks_int) > 1 or len(per_emit) > 1
    gold_default = q.get("gold_answer", "") or ""
    if not is_multi:
        expected_chunk = (
            answer_chunks_int[-1]
            if answer_chunks_int else
            (max(ask_chunks_int) if ask_chunks_int else -1)
        )
        early_events = [
            ev for ev in events
            if expected_chunk >= 0 and int(ev.get("chunk", -1)) < expected_chunk
        ]
        first = next(
            (ev for ev in events if int(ev.get("chunk", -1)) >= expected_chunk),
            events[0],
        )
        extra_events = [
            ev for ev in events
            if ev is not first and ev not in early_events
        ]
        sub = _score_one_question(
            rewards,
            q=q,
            model_answer=str(first.get("text", "")),
            answered_chunk=int(first.get("chunk", -1)),
        )
        if early_events and int(first.get("chunk", -1)) >= expected_chunk:
            early = early_events[0]
            early_timing = float(rewards["timing"](
                int(early.get("chunk", -1)),
                expected_chunk,
                expected_chunk,
                late_window_chunks=2,
            ))
            early_decision = _answer_decision_reward(
                rewards,
                int(early.get("chunk", -1)),
                expected_chunk,
                expected_chunk,
                late_window_chunks=2,
                has_answer=True,
            )
            sub["timing"] = min(float(sub["timing"]), early_timing)
            sub["answer_decision"] = min(
                float(sub.get("answer_decision", sub["timing"])),
                early_decision,
            )
            try:
                early_silent = float(rewards["silent_quality"](
                    str(early.get("text", "")),
                    "silent",
                    gold_default,
                ))
                sub["silent_quality"] = min(float(sub["silent_quality"]), early_silent)
            except Exception:
                pass
        if extra_events:
            sub["timing"] = min(float(sub["timing"]), -1.0)
            sub["answer_decision"] = min(
                float(sub.get("answer_decision", sub["timing"])),
                -1.0,
            )
            try:
                over_silent = min(
                    float(rewards["silent_quality"](
                        str(ev.get("text", "")), "silent", gold_default,
                    ))
                    for ev in extra_events
                )
                sub["silent_quality"] = min(float(sub["silent_quality"]), over_silent)
            except Exception:
                pass
        return sub

    options = _safe_list(q.get("options"))
    correct_option = q.get("correct_option", "")
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
    decision_scores: List[float] = []
    silent_scores: List[float] = []
    fp_timing_scores: List[float] = []
    fp_decision_scores: List[float] = []
    fp_silent_scores: List[float] = []
    for i, emit_chunk in enumerate(target_chunks):
        lo = emit_chunk
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
            decision_scores.append(_answer_decision_reward(
                rewards,
                None,
                emit_chunk,
                hi,
                late_window_chunks=slack,
                has_answer=False,
            ))
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
        timing_scores.append(float(rewards["timing"](
            ev_chunk, emit_chunk, emit_chunk, late_window_chunks=slack,
        )))
        decision_scores.append(_answer_decision_reward(
            rewards,
            ev_chunk,
            emit_chunk,
            emit_chunk,
            late_window_chunks=slack,
            has_answer=True,
        ))
        silent_scores.append(float(rewards["silent_quality"](
            model_answer, "response", gold_for_emit,
        )))

    for ei, ev in enumerate(events):
        if ei in used_event_idx:
            continue
        ev_chunk = int(ev.get("chunk", -1))
        early_by_meta = str(ev.get("timing") or "") == "early"
        later_targets = [t for t in target_chunks if ev_chunk < int(t)]
        early_by_chunk = bool(later_targets) and ev.get("counts_for_completion") is not True
        if early_by_meta or early_by_chunk:
            expected_chunk = min(later_targets) if later_targets else target_chunks[0]
            fp_timing_scores.append(float(rewards["timing"](
                ev_chunk, expected_chunk, expected_chunk, late_window_chunks=slack,
            )))
            fp_decision_scores.append(_answer_decision_reward(
                rewards,
                ev_chunk,
                expected_chunk,
                expected_chunk,
                late_window_chunks=slack,
                has_answer=True,
            ))
        else:
            fp_timing_scores.append(-1.0)
            fp_decision_scores.append(-1.0)
        fp_silent_scores.append(float(rewards["silent_quality"](
            str(ev.get("text", "")), "silent", gold_default,
        )))

    timing = sum(timing_scores) / len(timing_scores)
    answer_decision = sum(decision_scores) / len(decision_scores)
    silent_quality = sum(silent_scores) / len(silent_scores)
    if fp_timing_scores:
        timing = min(timing, min(fp_timing_scores))
    if fp_decision_scores:
        answer_decision = min(answer_decision, min(fp_decision_scores))
    if fp_silent_scores:
        silent_quality = min(silent_quality, min(fp_silent_scores))

    return {
        "outcome": sum(outcome_scores) / len(outcome_scores),
        "answer_decision": answer_decision,
        "timing": timing,
        "silent_quality": silent_quality,
        "answered": 1.0 if used_event_idx else 0.0,
    }


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_zero(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _metric_text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _question_support_chunks(q: Dict[str, Any]) -> List[int]:
    chunks: List[int] = []
    for key in (
        "support_chunks",
        "evidence_chunks",
        "grounding_chunks",
        "grounding_frames",
    ):
        chunks.extend(_int_chunks_from_value(q.get(key)))
    key_chunks = q.get("key_chunks")
    if isinstance(key_chunks, dict):
        for key in ("support", "supports", "evidence", "grounding"):
            chunks.extend(_int_chunks_from_value(key_chunks.get(key)))
    return sorted(set(c for c in chunks if c >= 0))


def _support_hits_in_time_range(
    raw_range: Any,
    support_chunks: set[int],
    *,
    chunk_sec: float = 1.0,
) -> set[int]:
    if not isinstance(raw_range, dict):
        return set()
    start = _safe_float_or_none(raw_range.get("start_time"))
    end = _safe_float_or_none(raw_range.get("end_time"))
    if start is None or end is None or end < start:
        return set()
    width = max(float(chunk_sec), 1e-6)
    hits: set[int] = set()
    for chunk in support_chunks:
        chunk_start = float(chunk) * width
        chunk_end = chunk_start + width
        if chunk_start <= end and chunk_end > start:
            hits.add(int(chunk))
    return hits


def _recall_range_and_post_answer_stats(
    extra: Dict[str, Any],
    questions: List[Dict[str, Any]],
    per_q_answers: List[Any],
) -> Dict[str, float]:
    """Monitor recall ranges and the current post-recall answer.

    The post-recall outcome intentionally scores only the answer emitted on
    the immediate post_recall turn. It does not scan forward to the next later
    answer, because that masks failed/empty post-recall behavior.
    """
    chunk_kinds = [str(x or "") for x in _safe_list(extra.get("ts_chunk_kinds"))]
    chunk_texts = _safe_list(extra.get("ts_chunk_asst_texts"))
    turn_kinds = [str(x or "") for x in _safe_list(extra.get("ts_chunk_turn_kinds"))]
    video_indices = _safe_list(extra.get("ts_chunk_video_indices"))
    recall_ranges = _safe_list(extra.get("ts_recall_time_ranges"))
    returned_chunks_all = _safe_list(extra.get("ts_recall_returned_chunks"))
    gold_action_per_chunk = _gold_action_map_for_extra(extra)

    request_spans: List[float] = []
    returned_spans: List[float] = []
    returned_counts: List[float] = []
    back_gaps: List[float] = []
    recall_chunks: List[int] = []

    support_targets: List[Dict[str, Any]] = []
    for q_idx, q in enumerate(questions):
        support = set(_question_support_chunks(q))
        if not support:
            continue
        labels = _recall_label_chunks_for_question(q, gold_action_per_chunk)
        if not labels:
            continue
        answer_bounds = _question_answer_opportunity_chunks(q)
        support_targets.append({
            "q_idx": int(q_idx),
            "labels": set(int(x) for x in labels),
            "start": int(min(labels)),
            "end": int(max(answer_bounds + labels)),
            "support": support,
        })

    recall_support_seen = 0
    recall_support_request_hits = 0
    recall_support_returned_hits = 0
    recall_support_request_cover: List[float] = []
    recall_support_returned_cover: List[float] = []

    for i, kind in enumerate(chunk_kinds):
        if kind != "recall":
            continue
        current_chunk = _turn_event_chunk(extra, i)
        if current_chunk < 0:
            try:
                current_chunk = int(video_indices[i]) if i < len(video_indices) else i
            except (TypeError, ValueError):
                current_chunk = i
        recall_chunks.append(current_chunk)

        raw_range = recall_ranges[i] if i < len(recall_ranges) else None
        if isinstance(raw_range, dict):
            start = _safe_float_or_none(raw_range.get("start_time"))
            end = _safe_float_or_none(raw_range.get("end_time"))
            if start is not None and end is not None:
                request_spans.append(max(0.0, end - start))

        raw_returned = returned_chunks_all[i] if i < len(returned_chunks_all) else []
        returned: List[int] = []
        for raw in _safe_list(raw_returned):
            try:
                returned.append(int(raw))
            except (TypeError, ValueError):
                continue
        if returned:
            returned = sorted(set(returned))
            returned_counts.append(float(len(returned)))
            returned_spans.append(float(max(returned) - min(returned) + 1))
            back_gaps.append(float(current_chunk - max(returned)))
        else:
            returned_counts.append(0.0)

        exact_support: set[int] = set()
        window_support: set[int] = set()
        for target in support_targets:
            support = set(target.get("support") or set())
            if current_chunk in set(target.get("labels") or set()):
                exact_support.update(support)
            elif int(target.get("start", -1)) <= current_chunk <= int(target.get("end", -1)):
                window_support.update(support)
        target_support = exact_support or window_support
        if target_support:
            recall_support_seen += 1
            request_hits = _support_hits_in_time_range(raw_range, target_support)
            returned_hits = set(returned) & target_support
            if request_hits:
                recall_support_request_hits += 1
            if returned_hits:
                recall_support_returned_hits += 1
            recall_support_request_cover.append(
                float(len(request_hits)) / float(len(target_support))
            )
            recall_support_returned_cover.append(
                float(len(returned_hits)) / float(len(target_support))
            )

    answer_events: List[Dict[str, Any]] = []
    for q_idx, raw_events in enumerate(per_q_answers[:len(questions)]):
        events = _safe_list(raw_events)
        q = questions[q_idx]
        per_emit = _safe_list(q.get("per_emit_answers"))
        chunk_gold = {
            int(e["chunk"]): str(e.get("value", q.get("gold_answer", "") or ""))
            for e in per_emit
            if isinstance(e, dict) and e.get("chunk") is not None
        }
        for ev in events:
            if not isinstance(ev, dict):
                continue
            text = str(ev.get("text", "")).strip()
            if not text:
                continue
            try:
                chunk = int(ev.get("chunk", -1))
            except (TypeError, ValueError):
                chunk = -1
            try:
                expected_chunk = int(ev.get("expected_chunk"))
            except (TypeError, ValueError):
                expected_chunk = chunk
            gold = chunk_gold.get(expected_chunk, str(q.get("gold_answer", "") or ""))
            outcome = float(_score_outcome_by_form(
                text,
                options=_safe_list(q.get("options")),
                correct_option=q.get("correct_option", ""),
                gold_answer=gold,
                answer_form=q.get("answer_form", "") or "",
            ))
            answer_events.append({
                "chunk": chunk,
                "q_idx": q_idx,
                "text": text,
                "text_key": _metric_text_key(text),
                "turn_kind": str(ev.get("turn_kind", "") or ""),
                "outcome": outcome,
            })
    answer_events.sort(key=lambda x: (int(x.get("chunk", -1)), int(x.get("q_idx", -1))))

    post_recall_turn_count = 0
    post_outcomes: List[float] = []
    used_answer_event_indices: set[int] = set()
    for turn_i, turn_kind in enumerate(turn_kinds):
        if turn_kind not in {"post_recall", "recall_response"}:
            continue
        post_recall_turn_count += 1
        text = str(chunk_texts[turn_i] or "") if turn_i < len(chunk_texts) else ""
        answer_text = _extract_answer_text_current(text, allow_bare_answer=True)
        if not answer_text:
            continue
        chunk = _turn_event_chunk(extra, turn_i)
        answer_key = _metric_text_key(answer_text)
        matched_idx: Optional[int] = None
        for idx, ev in enumerate(answer_events):
            if idx in used_answer_event_indices:
                continue
            if int(ev.get("chunk", -1)) == chunk and str(ev.get("text_key", "")) == answer_key:
                matched_idx = idx
                break
        if matched_idx is None:
            same_chunk = [
                idx for idx, ev in enumerate(answer_events)
                if idx not in used_answer_event_indices
                and int(ev.get("chunk", -1)) == chunk
            ]
            if len(same_chunk) == 1:
                matched_idx = same_chunk[0]
        if matched_idx is None:
            post_outcomes.append(0.0)
            continue
        used_answer_event_indices.add(matched_idx)
        post_outcomes.append(float(answer_events[matched_idx].get("outcome", 0.0)))

    if post_recall_turn_count == 0:
        for idx, ev in enumerate(answer_events):
            if str(ev.get("turn_kind", "") or "") not in {"post_recall", "recall_response"}:
                continue
            if idx in used_answer_event_indices:
                continue
            post_recall_turn_count += 1
            used_answer_event_indices.add(idx)
            post_outcomes.append(float(ev.get("outcome", 0.0)))

    recall_count = float(len(recall_chunks))
    post_count = float(len(post_outcomes))
    post_turn_count = float(post_recall_turn_count)
    support_seen = float(recall_support_seen)
    return {
        "recall_call_count": recall_count,
        "recall_request_span_mean": _mean_or_zero(request_spans),
        "recall_returned_span_mean": _mean_or_zero(returned_spans),
        "recall_returned_count_mean": _mean_or_zero(returned_counts),
        "recall_back_gap_mean": _mean_or_zero(back_gaps),
        "recall_support_seen": support_seen,
        "recall_support_request_hit": float(recall_support_request_hits),
        "recall_support_returned_hit": float(recall_support_returned_hits),
        "recall_support_request_hit_rate": (
            float(recall_support_request_hits) / support_seen
            if support_seen > 0.0 else 0.0
        ),
        "recall_support_returned_hit_rate": (
            float(recall_support_returned_hits) / support_seen
            if support_seen > 0.0 else 0.0
        ),
        "recall_support_request_cover_mean": _mean_or_zero(recall_support_request_cover),
        "recall_support_returned_cover_mean": _mean_or_zero(recall_support_returned_cover),
        "post_recall_turn_count": post_turn_count,
        "post_recall_answer_count": post_count,
        "post_recall_answer_rate": (
            post_count / post_turn_count if post_turn_count > 0.0 else 0.0
        ),
        "post_recall_answer_per_recall_rate": (
            post_count / recall_count if recall_count > 0.0 else 0.0
        ),
        "post_recall_outcome_mean": _mean_or_zero(post_outcomes),
        "post_recall_current_answer_count": post_count,
        "post_recall_current_answer_rate": (
            post_count / post_turn_count if post_turn_count > 0.0 else 0.0
        ),
        "post_recall_current_outcome_mean": _mean_or_zero(post_outcomes),
    }


def _compute_score_multi_q(
    rewards: Dict[str, Any],
    weights: Dict[str, float],
    questions: List[Dict[str, Any]],
    extra: Dict[str, Any],
    solution_str: str,
) -> Dict[str, float]:
    """Score a multi-Q trajectory. Aggregate by expected answer slot."""
    weights = _reward_weights_with_recall(weights)
    trajectory_solution = _trajectory_solution_text(extra, solution_str)
    n_q = len(questions)
    if n_q == 0:
        parts = {
            "outcome": 0.0,
            "timing": 0.0,
            "answer_decision": 0.0,
            "format": 0.0,
            "silent_quality": 0.0,
            **_compute_compress_quality(extra),
        }
        return {
            "score": 0.0,
            **parts,
            "n_questions": 0.0,
            "n_answers": 0.0,
            "n_answered": 0.0,
        }

    # Per-Q answer attribution from the agent loop's extra_fields.
    per_q_chunk_raw = _safe_list(extra.get("ts_per_q_answer_chunk"))
    per_q_text_raw = _safe_list(extra.get("ts_per_q_answer_text"))
    per_q_answers_raw = _safe_list(extra.get("ts_per_q_answers"))
    per_q_chunk = list(per_q_chunk_raw) + [-1] * (n_q - len(per_q_chunk_raw))
    per_q_text = list(per_q_text_raw) + [""] * (n_q - len(per_q_text_raw))
    per_q_answers = list(per_q_answers_raw) + [[]] * (n_q - len(per_q_answers_raw))
    gold_action_per_chunk = _gold_action_map_for_extra(extra)
    recall_chunks_used = _used_recall_chunks(extra)
    horizon_chunk = _observed_horizon_chunk(extra)
    scored_question_indices: List[int] = []
    excluded_future = 0
    for q_idx, q in enumerate(questions):
        opportunity_chunks = _question_answer_opportunity_chunks(q)
        if horizon_chunk is not None and opportunity_chunks and min(opportunity_chunks) > horizon_chunk:
            excluded_future += 1
            continue
        scored_question_indices.append(q_idx)

    # Per-Q scoring.
    per_q_outcome: List[float] = []
    per_q_timing: List[float] = []
    per_q_decision: List[float] = []
    per_q_silent: List[float] = []
    per_q_recall_answer: List[float] = []
    per_q_parts: List[Dict[str, float]] = []
    per_q_weights: List[float] = []
    recall_answer_labeled = 0
    recall_answer_used = 0
    recall_answer_success = 0
    n_answered = 0
    n_answered_total = 0
    for answer_events_raw in per_q_answers[:n_q]:
        if hasattr(answer_events_raw, "tolist"):
            answer_events_raw = answer_events_raw.tolist()
        if isinstance(answer_events_raw, (list, tuple)) and answer_events_raw:
            n_answered_total += 1
    for q_idx in scored_question_indices:
        q = questions[q_idx]
        per_q_weights.append(_answer_weight_for_question(q))
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
        label_chunks = _recall_label_chunks_for_question(q, gold_action_per_chunk)
        answer_event_chunks = _answer_event_chunks_for_question(
            answer_events,
            per_q_chunk[q_idx] if q_idx < len(per_q_chunk) else -1,
        )
        recall_labeled = bool(label_chunks)
        recall_used = _question_used_recall(
            q,
            label_chunks=label_chunks,
            used_recall_chunks=recall_chunks_used,
            answer_event_chunks=answer_event_chunks,
        )
        recall_answer_action = 1.0 if recall_labeled and recall_used else 0.0
        recall_answer_success_value = recall_answer_action * float(sub["outcome"])
        if recall_labeled:
            recall_answer_labeled += 1
            if recall_used:
                recall_answer_used += 1
            if recall_answer_success_value > 0.0:
                recall_answer_success += 1
        per_q_outcome.append(sub["outcome"])
        per_q_timing.append(sub["timing"])
        per_q_decision.append(sub["answer_decision"])
        per_q_silent.append(sub["silent_quality"])
        per_q_recall_answer.append(recall_answer_success_value)
        per_q_parts.append({
            "outcome": float(sub["outcome"]),
            "answer_decision": float(sub["answer_decision"]),
            "timing": float(sub["timing"]),
            "silent_quality": float(sub["silent_quality"]),
            # The scalar reward gate in _combine_reward_parts multiplies this
            # action-use bit by the same question's answer correctness.
            "recall_answer": float(recall_answer_action),
        })
        if sub["answered"] > 0:
            n_answered += 1

    # Trajectory-level aggregates.
    avg_outcome = _weighted_mean(per_q_outcome, per_q_weights)
    avg_timing = _weighted_mean(per_q_timing, per_q_weights)
    avg_decision = _weighted_mean(per_q_decision, per_q_weights)
    avg_silent = _weighted_mean(per_q_silent, per_q_weights)
    avg_recall_answer = _weighted_mean(per_q_recall_answer, per_q_weights)

    # Format is trajectory-level (not per-Q). It is a CASIA-like proportion of
    # parse/action-space/runtime-valid non-compress turns, without gold
    # range/query matching.
    fmt = float(_framework_format_score(extra, trajectory_solution))

    parts = {
        "outcome": avg_outcome,
        "answer_decision": avg_decision,
        "timing": avg_timing,
        "format": fmt,
        "silent_quality": avg_silent,
        "recall_answer": avg_recall_answer,
        "recall_answer_labeled": float(recall_answer_labeled),
        "recall_answer_used": float(recall_answer_used),
        "recall_answer_success": float(recall_answer_success),
        "recall_answer_used_rate": (
            float(recall_answer_used) / float(recall_answer_labeled)
            if recall_answer_labeled else 0.0
        ),
        "recall_answer_success_rate": (
            float(recall_answer_success) / float(recall_answer_labeled)
            if recall_answer_labeled else 0.0
        ),
    }
    parts.update(_compute_compress_quality(extra))
    total, gate, per_q_scores = _combine_multi_q_reward_parts(
        weights,
        per_q_parts,
        {
            "format": fmt,
            "compress_quality": float(parts.get("compress_quality", 0.0)),
        },
        question_weights=per_q_weights,
    )
    segment_meta = _build_question_segment_reward_metadata(
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
    )
    segment_meta.update(_build_compress_boundary_segment_reward_metadata(
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
    ))
    segment_meta.update(_build_compress_future_segment_reward_metadata(
        questions=questions,
        scored_question_indices=scored_question_indices,
        per_q_parts=per_q_parts,
        per_q_weights=per_q_weights,
        fmt=fmt,
        extra=extra,
    ))
    recall_audit: Dict[str, Any] = {}
    action_avg = _per_chunk_action_avg(
        extra, gold_action_per_chunk, audit_out=recall_audit,
    )
    if action_avg is not None:
        parts["per_chunk_action_avg"] = float(action_avg)
        if _step_action_reward_enabled():
            alpha = float(extra.get("gdpo_alpha", 0.7))
            gated_state = action_avg if action_avg <= 0 else gate * action_avg
            total = alpha * total + (1.0 - alpha) * gated_state
    # Recall monitor — wandb-only, not reward (P7).
    for k, v in recall_audit.items():
        parts[k] = float(v)
    parts.update(_recall_range_and_post_answer_stats(
        extra,
        questions,
        per_q_answers,
    ))

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
        **segment_meta,
        "outcome_gate": float(gate),
        "n_questions": float(len(scored_question_indices)),
        "n_questions_total": float(n_q),
        "n_questions_excluded_future": float(excluded_future),
        "horizon_chunk": float(horizon_chunk if horizon_chunk is not None else -1),
        "n_answers": float(sum(per_q_weights)),
        "n_answered": float(n_answered),
        "n_answered_total": float(n_answered_total),
        "per_q_outcome_min": float(min(per_q_outcome)) if per_q_outcome else 0.0,
        "per_q_outcome_max": float(max(per_q_outcome)) if per_q_outcome else 0.0,
        "trajectory_all_correct": float(min(per_q_outcome)) if per_q_outcome else 0.0,
        "trajectory_mean_correct": float(avg_outcome),
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
        question independently, aggregate by expected answer slot. Aligns
        with OVOBench eval form.
      - thinkstream_v12_streaming (legacy): single (video, question)
        flatten; score one question with v12 5-component reward.

    Returns a dict so verl logs per-component rewards to wandb:
        {"score": <total>, "outcome": ..., "answer_decision": ...,
         "timing": ..., "format": ..., "silent_quality": ...,
         "compress_quality": ...}
    """
    extra = extra_info or {}
    rewards, weights = _load_thinkstream_rewards()
    if not rewards:
        return {
            "score": 0.0,
            "outcome": 0.0,
            "timing": 0.0,
            "answer_decision": 0.0,
            "format": 0.0,
            "silent_quality": 0.0,
            **_compute_compress_quality(extra),
        }

    weights = _reward_weights_with_recall(weights)

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
        result = _compute_score_multi_q(
            rewards, weights, norm, extra, solution_str,
        )
        _maybe_audit_rl_rollout(
            data_source=data_source,
            extra=extra,
            solution_str=_trajectory_solution_text(extra, solution_str),
            ground_truth=ground_truth,
            result=result,
            questions=norm,
        )
        return result

    # Fall back to the trajectory index if extra_info doesn't carry the bundle
    # (e.g., when verl strips dict columns down to scalars at parquet load).
    if not extra.get("gold_action_per_chunk") and extra.get("video_id"):
        idx = _load_traj_index()
        traj = idx.get(str(extra["video_id"]))
        if traj:
            extra.setdefault(
                "gold_action_per_chunk",
                _strip_offline_compress_actions(
                    traj.get("gold_action_per_chunk", {})
                ),
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
    gold_action_per_chunk = _strip_offline_compress_actions(gold_action_per_chunk)

    trajectory_solution = _trajectory_solution_text(extra, solution_str)
    chunks = _split_assistant_chunks(trajectory_solution)
    final_answer_inferred = _extract_final_answer(trajectory_solution)
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
                if re.search(r"</Response>\s*(.+?)\s*$", chunk, re.DOTALL):
                    answer_chunk = idx
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
        parts["answer_decision"] = _answer_decision_reward(
            rewards,
            answer_chunk,
            visible_start,
            visible_end,
            late_window_chunks=1,
            has_answer=final_answer is not None and bool(str(final_answer).strip()),
        )
        parts["format"] = _framework_format_score(extra, trajectory_solution)
        gold_action = ""
        if answer_chunk is not None:
            gold_action = (gold_action_per_chunk or {}).get(str(answer_chunk), "")
        parts["silent_quality"] = rewards["silent_quality"](
            final_answer, gold_action, gold_answer
        )
        recall_q = {
            "ask_chunk": min(ask_chunks) if ask_chunks else -1,
            "ask_chunks": ask_chunks,
            "answer_chunks": [
                int(x) for x in (visible_start, visible_end)
                if x is not None
            ],
        }
        recall_label_chunks = _recall_label_chunks_for_question(
            recall_q,
            gold_action_per_chunk,
        )
        recall_used = _question_used_recall(
            recall_q,
            label_chunks=recall_label_chunks,
            used_recall_chunks=_used_recall_chunks(extra),
            answer_event_chunks=(
                [int(answer_chunk)] if answer_chunk is not None else []
            ),
        )
        parts["recall_answer"] = (
            1.0 if recall_label_chunks and recall_used and parts["outcome"] > 0.0
            else 0.0
        )
        parts["recall_answer_labeled"] = float(bool(recall_label_chunks))
        parts["recall_answer_used"] = float(bool(recall_label_chunks and recall_used))
        parts["recall_answer_success"] = float(parts["recall_answer"] > 0.0)
        parts.update(_compute_compress_quality(extra))
    except Exception as e:
        logger.warning("v12 reward component failed: %s", e)
        return {
            "score": 0.0,
            "outcome": 0.0,
            "timing": 0.0,
            "answer_decision": 0.0,
            "format": 0.0,
            "silent_quality": 0.0,
            **_compute_compress_quality(extra),
        }

    total, gate = _combine_reward_parts(weights, parts)

    recall_audit: Dict[str, Any] = {}
    action_avg = _per_chunk_action_avg(
        extra, gold_action_per_chunk, audit_out=recall_audit,
    )
    if action_avg is not None:
        parts["per_chunk_action_avg"] = float(action_avg)
        if _step_action_reward_enabled():
            alpha = float(extra.get("gdpo_alpha", 0.7))
            gated_state = action_avg if action_avg <= 0 else gate * action_avg
            total = alpha * total + (1.0 - alpha) * gated_state
    # Recall monitor — wandb-only, not reward (P7).
    for k, v in recall_audit.items():
        parts[k] = float(v)

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
    result = {
        "score": total,
        **{k: float(v) for k, v in parts.items()},
        "outcome_gate": float(gate),
    }
    single_question = {
        "question": extra.get("question", ""),
        "ask_chunks": ask_chunks,
        "answer_chunks": extra.get("answer_chunks") or [],
        "answer_form": answer_form,
        "correct_option": correct_option,
        "gold_answer": gold_answer,
    }
    _maybe_audit_rl_rollout(
        data_source=data_source,
        extra=extra,
        solution_str=trajectory_solution,
        ground_truth=ground_truth,
        result=result,
        questions=[single_question],
    )
    return result


if __name__ == "__main__":
    # Smoke path: raw agent tags are preserved in solution_str.
    sample = (
        "<think>chunk 0 silent</think>"
        "<think>chunk 1 final</think></Response> yes"
    )
    gt = json.dumps({"gold_answer": "yes", "answer_form": "binary",
                     "ask_chunks": [1], "gold_action_per_chunk": {"1": "response"}})
    print("score:", compute_score("thinkstream_v12_streaming", sample, gt,
                                  {"num_turns": 2}))
