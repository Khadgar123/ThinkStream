"""Flatten ThinkStream pass4 trajectory JSONL into a verl RLHFDataset parquet.

verl's `RLHFDataset` (and our `recipe/thinkstream/CustomRLHFDataset` subclass)
reads parquet — one row per RL training sample. Our pass4 emits one
trajectory per line with N nested questions:

    {"video_id":..., "video_path":...,
     "questions":[{"question":..., "gold_answer":..., "answer_form":...,
                   "ask_chunks":[..]}, ...],
     "gold_action_per_chunk":{...}, "stats":{...}}

This script flattens (video × question) → one parquet row with the columns
the recipe expects:

    prompt              List[Dict]           # list of {role,content} dicts
    video_id            str
    video_path          str
    question            str
    gold_answer         str
    answer_form         str
    ask_chunks          List[int]
    gold_action_per_chunk  Dict[str,str]     # no offline "compress" targets
    offline_compress_chunks List[int]        # diagnostics only
    n_chunks            int
    extra_info          Dict                 # passthrough metadata
    reward_model        Dict                 # verl convention: {"ground_truth": str, "style": str}

Usage:
    python -m scripts.agent_data.build_verl_parquet \\
        --jsonl data/agent_v5/final/train_rl_trajectories.jsonl \\
        --out   data/agent_v5/final/train_rl.parquet
"""
from __future__ import annotations

import argparse
import json
import gzip
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# verl/recipe expects pyarrow for parquet round-trip; pandas is a thin shim.
import pandas as pd

# Resolve repo root so we can import the v12 system prompt without
# relying on PYTHONPATH being preset.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinkstream.data.agent_protocol import (  # noqa: E402
    canonical_answer_instruction,
    normalize_frame_protocol,
    normalize_render_layout,
    system_prompt_for_frame_protocol,
)


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extend_chunk_candidates_from_list(out: List[int], values: Any) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        iv = _safe_int(value)
        if iv is not None and iv >= 0:
            out.append(iv)


def _canonical_instruction(q: Dict[str, Any]) -> str:
    return canonical_answer_instruction(q) or str(q.get("answer_instruction") or "")


def _canonical_answer_style(q: Dict[str, Any]) -> str:
    if str(q.get("answer_form") or "").strip() == "multiple_choice":
        return "letter_plus_text"
    return str(q.get("answer_style") or "")


_QUESTION_OPTIONAL_KEYS = (
    "open_until",
    "ovo_task",
    "ovo_category",
    "ovo_sample_id",
    "ovo_probe_index",
    "ovo_probe_type",
    "ovo_realtime",
    "ovo_ask_time",
    "ovo_clue_time",
    "ovo_support_intervals",
    "ovo_score_mode",
)


def _question_optional_payload(q: Dict[str, Any]) -> Dict[str, Any]:
    return {key: q[key] for key in _QUESTION_OPTIONAL_KEYS if key in q}


def _offline_compress_chunks_from_samples(traj: Dict[str, Any]) -> List[int]:
    """Extract compact-memory trigger chunks from canonical sample rows.

    A single video chunk can have both an inter-chunk compression event and the
    ordinary streaming action for that chunk. ``gold_action_per_chunk`` can only
    store one action, so response/silent targets may legitimately occupy the
    same key. The samples list preserves the text-only compress rows and is the
    canonical source for RL trigger timing in that collision case.
    """
    chunks: List[int] = []
    for sample in traj.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_type = str(sample.get("sample_type") or "").strip().lower()
        action = str(sample.get("action") or "").strip().lower()
        meta_action = str(
            (sample.get("metadata") or {}).get("gold_action") or ""
        ).strip().lower()
        text = str(sample.get("output") or sample.get("gold_caption") or "")
        is_compact_mem = bool(re.search(r'<m\s+t="[^"]+"\s*>.*?</m>', text, re.S | re.I))
        if (
            sample_type != "compress"
            and action != "compress"
            and meta_action != "compress"
            and not (
                bool(sample.get("inter_chunk") or sample.get("v12_inter_chunk"))
                and is_compact_mem
            )
        ):
            continue
        iv = _safe_int(sample.get("chunk_idx", sample.get("chunk")))
        if iv is not None and iv >= 0:
            chunks.append(iv)
    return chunks


def _rl_gold_actions_and_offline_compress(
    gold_action: Dict[str, Any],
    traj: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, str], List[int]]:
    """Return RL action targets plus offline compress diagnostics.

    Compression is a system event, not an action-shaping target. RL rollout may
    reuse these offline pass2 boundaries for trigger timing, but reward code must
    not train the policy from ``gold_action_per_chunk["compress"]`` labels.
    """
    sanitized: Dict[str, str] = {}
    offline_compress: List[int] = []
    for key, value in (gold_action or {}).items():
        action = str(value or "")
        if action == "compress":
            iv = _safe_int(key)
            if iv is not None and iv >= 0:
                offline_compress.append(iv)
            continue
        sanitized[str(key)] = action
    if traj is not None:
        offline_compress.extend(_offline_compress_chunks_from_samples(traj))
    return sanitized, sorted(set(offline_compress))


def _infer_n_chunks(traj: Dict[str, Any]) -> int:
    """Infer trajectory length for RL rollout.

    Older bank exports may omit ``stats.n_chunks_covered`` while still
    carrying ``chunk_idx_max`` and full per-chunk gold/sample metadata. A zero
    length makes verl fall back to the launcher MAX_TURNS cap, silently
    truncating long videos, so recover from all available structured fields.
    """
    stats = traj.get("stats") or {}
    direct = _safe_int(stats.get("n_chunks_covered"))
    if direct and direct > 0:
        return direct

    candidates: List[int] = []
    for key in ("chunk_idx_max", "max_chunk", "last_chunk"):
        iv = _safe_int(stats.get(key))
        if iv is not None and iv >= 0:
            candidates.append(iv)

    for key in (traj.get("gold_action_per_chunk") or {}).keys():
        iv = _safe_int(key)
        if iv is not None and iv >= 0:
            candidates.append(iv)

    for sample in traj.get("samples") or []:
        if isinstance(sample, dict):
            iv = _safe_int(sample.get("chunk_idx", sample.get("chunk")))
            if iv is not None and iv >= 0:
                candidates.append(iv)

    for q in traj.get("questions") or []:
        if not isinstance(q, dict):
            continue
        for key in (
            "ask_chunks",
            "answer_chunks",
            "expected_answer_chunks",
            "support_chunks",
            "gold_compress_chunks",
            "missing_answer_chunks",
        ):
            _extend_chunk_candidates_from_list(candidates, q.get(key))
        for key in ("ask_chunk", "answer_chunk"):
            iv = _safe_int(q.get(key))
            if iv is not None and iv >= 0:
                candidates.append(iv)

    return (max(candidates) + 1) if candidates else 0


def _jsonl_data_root(jsonl_path: Path) -> Path:
    """Return the batch root for final/*.jsonl inputs."""
    return jsonl_path.parent.parent if jsonl_path.parent.name == "final" else jsonl_path.parent


def _load_student_rollout(jsonl_path: Path, video_id: str) -> Dict[str, Any]:
    if not video_id:
        return {}
    path = _jsonl_data_root(jsonl_path) / "rollout" / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            rollout = json.load(f)
    except Exception:
        return {}
    if not isinstance(rollout, dict):
        return {}
    rollout["_source_path"] = str(path)
    return rollout


def _normalise_student_thinks(rollout: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in rollout.get("thinks") or []:
        if not isinstance(item, dict):
            continue
        chunk = _safe_int(item.get("chunk_idx", item.get("chunk")))
        if chunk is None or chunk < 0:
            continue
        text = item.get("think", item.get("text", "")) or ""
        if not text:
            continue
        out.append({
            "chunk": chunk,
            "chunk_idx": chunk,
            "time": item.get("time", ""),
            "text": text,
            "source": item.get("source", "student_rollout"),
        })
    out.sort(key=lambda x: int(x.get("chunk", 0)))
    return out


def _student_cache_payload(
    traj: Dict[str, Any],
    *,
    jsonl_path: Path,
    video_id: str,
) -> Dict[str, Any]:
    """Preserve student prefix state for segment RL.

    The trajectory JSONL is the canonical supervision file, but the student
    prefix memory usually lives in the sibling pass2 rollout cache. Keep
    snapshots and a compact per-video think archive separately, so parquet
    size stays O(n_chunks) rather than O(n_chunks^2). CustomRLHFDataset
    combines the one selected snapshot with archive entries strictly before
    that snapshot chunk at materialization time.
    """
    payload: Dict[str, Any] = {}
    for key in (
        "initial_student_state_by_chunk",
        "student_state_by_chunk",
        "student_memory_snapshots",
        "student_snapshots",
        "memory_snapshots",
        "snapshots",
        "student_think_archive",
        "pass2_thinks",
        "student_cache_meta",
    ):
        if key in traj:
            payload[key] = traj[key]

    rollout = _load_student_rollout(jsonl_path, video_id)
    if rollout:
        if not any(
            key in payload
            for key in (
                "initial_student_state_by_chunk",
                "student_state_by_chunk",
                "student_memory_snapshots",
                "student_snapshots",
                "memory_snapshots",
                "snapshots",
            )
        ):
            snapshots = rollout.get("snapshots")
            if isinstance(snapshots, dict):
                payload["student_state_by_chunk"] = snapshots
        if "student_think_archive" not in payload:
            payload["student_think_archive"] = _normalise_student_thinks(rollout)

        meta_raw = payload.get("student_cache_meta") or {}
        meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
        meta.update({
            "source": meta.get("source") or "pass2_student_rollout",
            "source_path": meta.get("source_path") or rollout.get("_source_path", ""),
            "schema": meta.get("schema") or "pass2_rollout_snapshots_v1",
            "checkpoint": (
                os.environ.get("THINKSTREAM_STUDENT_CACHE_CHECKPOINT")
                or meta.get("checkpoint")
                or rollout.get("checkpoint")
                or rollout.get("model_path")
                or ""
            ),
            "global_step": (
                os.environ.get("THINKSTREAM_STUDENT_CACHE_GLOBAL_STEP")
                or meta.get("global_step")
                or rollout.get("global_step")
                or ""
            ),
            "epoch": (
                os.environ.get("THINKSTREAM_STUDENT_CACHE_EPOCH")
                or meta.get("epoch")
                or ""
            ),
            "refresh_policy": (
                "refresh each epoch or when policy drift makes prefix "
                "memory distribution stale"
            ),
        })
        payload["student_cache_meta"] = meta
    return payload


def _iter_rows(
    jsonl_path: Path,
    max_questions_per_traj: int,
    *,
    frame_protocol: str,
    render_layout: str,
    include_student_cache: bool,
) -> Iterator[Dict[str, Any]]:
    system_prompt = system_prompt_for_frame_protocol(
        frame_protocol,
        render_layout=render_layout,
    )
    with _open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = traj.get("video_id") or traj.get("trajectory_id") or ""
            video_path = traj.get("video_path", "")
            raw_gold_action = traj.get("gold_action_per_chunk", {}) or {}
            gold_action, offline_compress_chunks = (
                _rl_gold_actions_and_offline_compress(raw_gold_action, traj)
            )
            n_chunks = _infer_n_chunks(traj)
            questions = (traj.get("questions") or [])[:max_questions_per_traj]
            segment_start_chunk = _safe_int(traj.get("segment_start_chunk"))
            segment_end_chunk = _safe_int(traj.get("segment_end_chunk"))
            student_cache = (
                _student_cache_payload(traj, jsonl_path=jsonl_path, video_id=str(video_id))
                if include_student_cache
                else {}
            )

            if not questions:
                continue

            for q_idx, q in enumerate(questions):
                question = q.get("question", "")
                gold_answer = q.get("gold_answer", "")
                answer_form = q.get("answer_form", "")
                options = list(q.get("options") or [])
                correct_option = q.get("correct_option", "")
                ask_chunks = list(q.get("ask_chunks") or [])
                answer_chunks = list(q.get("answer_chunks") or [])

                # ── Per-question gold_action_per_chunk (P1.9 fix).
                # The trajectory-level gold_action carries actions for
                # ALL questions' chunks. When this row is one (video,
                # question) pair, only the chunks within THIS question's
                # answerable range should be scored — otherwise question A's
                # rollout gets penalised for not emitting question B's
                # response at chunk where question B was supposed to fire.
                # Strategy: keep gold_action ONLY for chunks within this
                # question's live window: from ask_chunk through its last
                # expected answer chunk. For forward/wait cards, answer_chunks
                # can be much later than ask_chunks; clipping only to ask_chunks
                # would erase the true response action and train the model to
                # stay silent at the answer time.
                # Offline compress labels have already been removed from
                # gold_action. They are carried separately as trigger
                # boundaries/diagnostics, not per-question action targets.
                q_gold_action: Dict[str, str] = {}
                window_marks = ask_chunks + answer_chunks
                if window_marks:
                    q_lo = min(ask_chunks or window_marks)
                    q_hi = max(answer_chunks or ask_chunks or window_marks)
                    for ck, gold in (gold_action or {}).items():
                        try:
                            ck_int = int(ck)
                        except (TypeError, ValueError):
                            continue
                        if q_lo <= ck_int <= q_hi:
                            # In range — keep the original action.
                            q_gold_action[ck] = gold
                        else:
                            # Out of range — silent is the correct
                            # action for this question at this chunk.
                            q_gold_action[ck] = "silent"
                else:
                    # No ask_chunks → treat as no actionable supervision.
                    q_gold_action = {ck: "silent" for ck in (gold_action or {}).keys()}

                # The streaming agent loop injects the live question at
                # ask_chunk. Keep the seed prompt system-only so RL does not
                # see a duplicate static user turn that SFT/eval never see.
                prompt = [{"role": "system", "content": system_prompt}]

                yield {
                    "prompt": prompt,
                    "video_id": video_id,
                    "video_path": video_path,
                    "question": question,
                    "options": options,
                    "correct_option": correct_option,
                    "correct_answer_text": q.get("correct_answer_text", ""),
                    "accepted_answers": list(q.get("accepted_answers") or []),
                    "answer_style": _canonical_answer_style(q),
                    "answer_instruction": _canonical_instruction(q),
                    "gold_answer": gold_answer,
                    "answer_form": answer_form,
                    "answer_chunks": answer_chunks,
                    "per_emit_answers": list(q.get("per_emit_answers") or []),
                    "ask_chunks": ask_chunks,
                    **_question_optional_payload(q),
                    "gold_action_per_chunk": q_gold_action,
                    "n_chunks": n_chunks,
                    "extra_info": {
                        "index": f"{video_id}#{q_idx}",
                        "video_id": video_id,
                        "question_idx": q_idx,
                        "card_id": q.get("card_id", ""),
                        "family": q.get("family", ""),
                        "family_name": q.get("family_name", ""),
                        "category": q.get("category", ""),
                        "skill": q.get("skill", ""),
                        "ours_unique": bool(q.get("ours_unique", False)),
                        "options": options,
                        "correct_option": correct_option,
                        "answer_instruction": _canonical_instruction(q),
                        "support_chunks": list(q.get("support_chunks") or []),
                        "answer_chunks": answer_chunks,
                        "per_emit_answers": list(q.get("per_emit_answers") or []),
                        "offline_compress_chunks": offline_compress_chunks,
                        "compress_trigger_source": "offline_pass2_boundaries",
                        "render_layout": render_layout,
                        **(
                            {"segment_start_chunk": segment_start_chunk}
                            if segment_start_chunk is not None else {}
                        ),
                        **(
                            {"segment_end_chunk": segment_end_chunk}
                            if segment_end_chunk is not None else {}
                        ),
                        **student_cache,
                    },
                    # verl convention: reward_model.ground_truth is what the
                    # reward function receives as `ground_truth`. Use a dict
                    # so we can pass the full bundle, not just a string.
                    "reward_model": {
                        "ground_truth": json.dumps({
                            "gold_answer": gold_answer,
                            "answer_form": answer_form,
                            "options": options,
                            "correct_option": correct_option,
                            "ask_chunks": ask_chunks,
                            "answer_chunks": answer_chunks,
                            "per_emit_answers": list(q.get("per_emit_answers") or []),
                            **_question_optional_payload(q),
                            "visible_start_chunk": (
                                min(ask_chunks) if ask_chunks else
                                (min(answer_chunks) if answer_chunks else None)
                            ),
                            "visible_end_chunk": (
                                max(answer_chunks) if answer_chunks else
                                (max(ask_chunks) if ask_chunks else None)
                            ),
                            "gold_action_per_chunk": q_gold_action,
                            "offline_compress_chunks": offline_compress_chunks,
                            "compress_trigger_source": "offline_pass2_boundaries",
                            **(
                                {"segment_start_chunk": segment_start_chunk}
                                if segment_start_chunk is not None else {}
                            ),
                            **(
                                {"segment_end_chunk": segment_end_chunk}
                                if segment_end_chunk is not None else {}
                            ),
                        }, ensure_ascii=False),
                        "style": "thinkstream_v12",
                    },
                    "data_source": "thinkstream_v12_streaming",
                    "frame_protocol": frame_protocol,
                    "render_layout": render_layout,
                }


def _iter_rows_multi_q(
    jsonl_path: Path,
    max_questions_per_traj: int,
    *,
    frame_protocol: str,
    render_layout: str,
    include_student_cache: bool,
) -> Iterator[Dict[str, Any]]:
    """Multi-Q trajectory rows: 1 video → 1 row containing ALL questions.

    This shape matches OVOBench's eval form (one video, many MCQ time-points)
    and the actual semantics of streaming-video agents — memory state +
    compress / recall decisions are SHARED across questions in one video,
    so flattening to (video, question) duplicates the visual rollout N
    times and discards the joint-supervision signal that compress / recall
    need to learn from.

    Schema per row:
      questions: List[Dict] — full pass4 question list (card_id, family,
                              ask_chunk, options, correct_option,
                              gold_answer, answer_form, per_emit_answers, ...)
      gold_action_per_chunk: Dict[str, str] — full per-chunk action-shaping map
                                              with offline compress labels removed
      offline_compress_chunks: List[int] — pass2/pass3 compress trigger positions
      reward_model.ground_truth: JSON-encoded list of per-question targets
                                 + the sanitized action map.

    The streaming agent loop reads `extra_info.questions` and injects each
    question's text into <user_input> at its `ask_chunk`; compute_score
    then evaluates each Q independently and aggregates (mean by default).
    """
    system_prompt = system_prompt_for_frame_protocol(
        frame_protocol,
        render_layout=render_layout,
    )
    with _open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = traj.get("video_id") or traj.get("trajectory_id") or ""
            video_path = traj.get("video_path", "")
            raw_gold_action = traj.get("gold_action_per_chunk", {}) or {}
            gold_action, offline_compress_chunks = (
                _rl_gold_actions_and_offline_compress(raw_gold_action, traj)
            )
            n_chunks = _infer_n_chunks(traj)
            questions = (traj.get("questions") or [])[:max_questions_per_traj]
            segment_start_chunk = _safe_int(traj.get("segment_start_chunk"))
            segment_end_chunk = _safe_int(traj.get("segment_end_chunk"))
            student_cache = (
                _student_cache_payload(traj, jsonl_path=jsonl_path, video_id=str(video_id))
                if include_student_cache
                else {}
            )

            if not questions:
                continue

            # Per-question target bundle — what compute_score scores against.
            q_targets: List[Dict[str, Any]] = []
            all_ask_chunks: List[int] = []
            for q in questions:
                ask_chunks = list(q.get("ask_chunks") or [])
                if not ask_chunks and q.get("ask_chunk", -1) >= 0:
                    ask_chunks = [int(q["ask_chunk"])]
                all_ask_chunks.extend(ask_chunks)
                q_targets.append({
                    "card_id": q.get("card_id", ""),
                    "family": q.get("family", ""),
                    "question": q.get("question", ""),
                    "options": list(q.get("options") or []),
                    "correct_option": q.get("correct_option", ""),
                    "correct_answer_text": q.get("correct_answer_text", ""),
                    "accepted_answers": list(q.get("accepted_answers") or []),
                    "answer_style": _canonical_answer_style(q),
                    "answer_instruction": _canonical_instruction(q),
                    "gold_answer": q.get("gold_answer", ""),
                    "answer_form": q.get("answer_form", ""),
                    "ask_chunk": int(q.get("ask_chunk", -1)),
                    "ask_chunks": ask_chunks,
                    "answer_chunks": list(q.get("answer_chunks") or []),
                    "per_emit_answers": list(q.get("per_emit_answers") or []),
                    "support_chunks": list(q.get("support_chunks") or []),
                    "family_name": q.get("family_name", ""),
                    "category": q.get("category", ""),
                    "skill": q.get("skill", ""),
                    "ours_unique": bool(q.get("ours_unique", False)),
                    **_question_optional_payload(q),
                })

            # System-level seed prompt only. Actual question text/options are
            # injected by the agent loop at each ask_chunk and then rendered as
            # <active_query> after the visual window, matching pass5/SFT/eval.
            prompt = [{"role": "system", "content": system_prompt}]

            yield {
                "prompt": prompt,
                "video_id": video_id,
                "video_path": video_path,
                "n_chunks": n_chunks,
                "n_questions": len(questions),
                "extra_info": {
                    "index": video_id,
                    "video_id": video_id,
                    "questions": q_targets,
                    "gold_action_per_chunk": gold_action,
                    "offline_compress_chunks": offline_compress_chunks,
                    "compress_trigger_source": "offline_pass2_boundaries",
                    "all_ask_chunks": sorted(set(all_ask_chunks)),
                    "render_layout": render_layout,
                    **(
                        {"segment_start_chunk": segment_start_chunk}
                        if segment_start_chunk is not None else {}
                    ),
                    **(
                        {"segment_end_chunk": segment_end_chunk}
                        if segment_end_chunk is not None else {}
                    ),
                    **(
                        {"source_video_path": traj.get("source_video_path")}
                        if traj.get("source_video_path") else {}
                    ),
                    **(
                        {"ovo_split_meta": traj.get("ovo_split_meta")}
                        if traj.get("ovo_split_meta") else {}
                    ),
                    **student_cache,
                },
                "reward_model": {
                    "ground_truth": json.dumps({
                        "questions": q_targets,
                        "gold_action_per_chunk": gold_action,
                        "offline_compress_chunks": offline_compress_chunks,
                        "compress_trigger_source": "offline_pass2_boundaries",
                        **(
                            {"segment_start_chunk": segment_start_chunk}
                            if segment_start_chunk is not None else {}
                        ),
                        **(
                            {"segment_end_chunk": segment_end_chunk}
                            if segment_end_chunk is not None else {}
                        ),
                    }, ensure_ascii=False),
                    "style": "thinkstream_v12_multi_q",
                },
                "data_source": "thinkstream_v12_streaming_multi_q",
                "frame_protocol": frame_protocol,
                "render_layout": render_layout,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="pass4 trajectory JSONL[.gz]")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument(
        "--max_questions_per_traj",
        type=int,
        default=16,
        help="cap per video. Default 16 covers OVOBench worst case (max 16 Q/video).",
    )
    ap.add_argument(
        "--multi_q",
        action="store_true",
        default=True,
        help=(
            "Multi-Q trajectory mode (default ON): 1 video = 1 parquet row "
            "containing all questions. Aligns RL training with OVOBench eval "
            "form (one video, many MCQ time-points) and matches the user's "
            "directive: RL keeps the whole video, only SFT slices on compress. "
            "Kept only for old launch scripts; single-question parquet is retired."
        ),
    )
    ap.add_argument(
        "--frame-protocol",
        default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"),
        choices=["video_meta"],
        help=(
            "Visual protocol used by the RL rollout loop. The supported "
            "project entry uses video_meta plus the selected render layout."
        ),
    )
    ap.add_argument(
        "--render-layout",
        default="standard_query_last",
        choices=["standard_query_last"],
        help="Prompt layout used by SFT, RL, and eval.",
    )
    ap.add_argument(
        "--include-student-cache",
        action="store_true",
        help=(
            "Attach pass2 student snapshots and student think archive for "
            "segment RL. Keep this off for full-video RL to avoid large "
            "unused Ray/parquet payloads."
        ),
    )
    args = ap.parse_args()
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = normalize_render_layout(args.render_layout)

    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = _iter_rows_multi_q(
        in_path,
        max_questions_per_traj=args.max_questions_per_traj,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
        include_student_cache=args.include_student_cache,
    )
    rows: List[Dict[str, Any]] = list(iterator)
    if not rows:
        print(f"[build_verl_parquet] no rows produced from {in_path}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(
        f"[build_verl_parquet] {in_path.name}: {len(rows)} video rows "
        f"→ {out_path} ({out_path.stat().st_size/1024:.1f} KiB, "
        f"frame_protocol={frame_protocol}, render_layout={render_layout}, "
        f"student_cache={args.include_student_cache})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
