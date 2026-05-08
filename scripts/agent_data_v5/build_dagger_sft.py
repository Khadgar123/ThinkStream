#!/usr/bin/env python
"""Build on-policy / DAgger SFT messages from existing trajectories.

The existing pass5 SFT rows are teacher-forced snapshots: every prompt uses
gold memory.  This script rolls a policy checkpoint through the trajectory
chunk-by-chunk, lets the policy write the memory, then swaps in the gold
assistant target for that same chunk.  The resulting rows train the model to
recover from its own closed-loop memory state.

Typical usage:

  CUDA_VISIBLE_DEVICES=0 python -m scripts.agent_data_v5.build_dagger_sft \
    --ckpt output/agent-sft/checkpoint-250 \
    --trajectories data/agent_v5/batch1/final/train_sft_trajectories.jsonl \
    --frames-root data/agent_v5/batch1/frames \
    --out data/agent_v5/batch1/rendered/video_meta/train_sft_dagger_messages.jsonl \
    --frame-protocol video_meta --correction-only --max-trajectories 20

For production-scale construction, prefer build_dagger_sft_vllm.py with
--correction-only and shard/batch its rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from scripts.agent_data_v5.config import AGENT_CHUNK_SEC
from scripts.agent_data_v5.pass5_messages import (
    build_messages,
    _emit_row,
    _with_sft_turn_policy,
)
from scripts.eval.ovo.eval_full import detect_model_class, reset_visual_index
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    diagnose_compress_output_v12,
    has_compress_trigger,
    normalize_frame_protocol,
    normalize_render_layout,
    parse_agent_output_v12,
    system_prompt_for_frame_protocol,
)
from thinkstream.model.agent_loop import (
    StreamingAgentLoop,
    make_generate_fn,
    recall_time_range_margin_chunks,
)
from thinkstream.model.retrieval import make_retriever
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


def _default_batch_root(path: Optional[str]) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p if p.name != "final" else p.parent
    env = os.environ.get("THINKSTREAM_DATA_ROOT") or os.environ.get("AGENT_DATA_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p if p.name != "final" else p.parent
    return ROOT / "data" / "agent_v5"


def _resolve_path(raw: str, *, base: Path = ROOT) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base / p


def _iter_trajectory_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _propagate_sample_fields(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    video_id = traj.get("video_id", "")
    video_path = traj.get("video_path", "")
    traj_id = traj.get("trajectory_id", "")
    samples = []
    for s in traj.get("samples") or []:
        item = dict(s)
        item.setdefault("video_id", video_id)
        item.setdefault("video_path", video_path)
        item.setdefault("trajectory_id", traj_id)
        samples.append(item)
    samples.sort(key=lambda x: int(x.get("chunk_idx", 0)))
    return samples


def _group_by_chunk(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_chunk: Dict[int, List[Dict[str, Any]]] = {}
    for s in samples:
        try:
            chunk = int(s.get("chunk_idx", 0))
        except (TypeError, ValueError):
            continue
        by_chunk.setdefault(chunk, []).append(s)
    return dict(sorted(by_chunk.items()))


def _resolve_video_path(video_path: str, video_root: Optional[str]) -> Optional[str]:
    if not video_path:
        return None
    p = Path(video_path)
    if p.is_absolute():
        return str(p)
    if video_root:
        return str(Path(video_root) / video_path)
    return str(ROOT / video_path)


def _new_question(sample: Dict[str, Any]) -> Optional[str]:
    inp = sample.get("input") or {}
    ui = inp.get("user_input")
    if not isinstance(ui, str):
        return None
    ui = ui.strip()
    if not ui or has_compress_trigger(ui):
        return None
    return ui


def _question_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = sample.get("metadata") or {}
    answer_chunks = (
        sample.get("answer_chunks")
        or sample.get("expected_answer_chunks")
        or meta.get("answer_chunks")
        or meta.get("expected_answer_chunks")
        or []
    )
    per_emit_answers = sample.get("per_emit_answers") or meta.get("per_emit_answers") or []
    open_until = sample.get("open_until") or meta.get("open_until")
    if open_until is None and answer_chunks:
        try:
            open_until = max(int(x) for x in answer_chunks) * AGENT_CHUNK_SEC
        except (TypeError, ValueError):
            open_until = None
    return {
        "options": sample.get("options") or meta.get("options") or [],
        "answer_form": sample.get("answer_form") or meta.get("answer_form") or "",
        "answer_style": sample.get("answer_style") or meta.get("answer_style") or "",
        "answer_instruction": (
            sample.get("answer_instruction")
            or meta.get("answer_instruction")
            or ""
        ),
        "answer_chunks": list(answer_chunks),
        "per_emit_answers": list(per_emit_answers),
        "open_until": open_until,
    }


def _choose_control_sample(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    for s in samples:
        if _new_question(s):
            return s
    return samples[0]


def _content_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "\n".join(parts)


_USER_INPUT_RE = re.compile(r"<user_input>(.*?)</user_input>", re.DOTALL)


def _prompt_has_compress_trigger(messages: List[Dict[str, Any]]) -> bool:
    """True only when the actual user input carries a compress trigger.

    Compression may also appear in turn-local system/tool text. DAgger needs
    the runtime event, which is rendered under the user turn's
    ``<user_input>...</user_input>`` block.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            for match in _USER_INPUT_RE.finditer(text):
                if has_compress_trigger(match.group(1)):
                    return True
    return False


def _target_allowed(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    *,
    sample_types: set[str],
    include_failed_targets: bool,
) -> tuple[bool, str]:
    sample_type = str(sample.get("sample_type", ""))
    if sample_type not in sample_types:
        return False, "sample_type"
    if not include_failed_targets:
        verification = sample.get("verification") or {}
        if not bool(verification.get("passed", True)):
            return False, "verification_failed"

    prompt_has_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    if sample_type == "compress" and not prompt_has_compress:
        return False, "compress_target_without_trigger"
    if sample_type != "compress" and prompt_has_compress:
        return False, "visual_target_on_compress_prompt"
    return True, ""


DEFAULT_DAGGER_CORRECTION_REASONS = {
    "format_error",
    "repeated_or_stale_think",
    "missed_compress",
    "bad_compress_json",
    "bad_compress_range_schema",
    "bad_compress_range",
    "empty_compress_summary",
    "missed_recall",
    "bad_recall_query",
    "recall_retrieval_empty",
    "recall_retrieval_miss",
    "recall_time_range_miss",
    "over_recall",
    "missed_response",
    "wrong_response",
    "early_answer",
    "late_response",
}


def _parse_reason_set(raw: str) -> set[str]:
    raw = str(raw or "").strip()
    if not raw or raw.lower() in {"default", "defaults"}:
        return set(DEFAULT_DAGGER_CORRECTION_REASONS)
    if raw.lower() == "all":
        return set(DEFAULT_DAGGER_CORRECTION_REASONS) | {
            "recall_answer_visible_in_policy_prompt",
        }
    return {x.strip() for x in raw.split(",") if x.strip()}


def _extract_answer_text(output_text: str) -> str:
    parsed = parse_agent_output_v12(output_text or "")
    if parsed.get("kind") == "answer":
        return str(parsed.get("answer_text") or "").strip()
    return ""


def _gold_output_text(sample: Dict[str, Any]) -> str:
    if sample.get("sample_type") == "recall" and sample.get("v12_assistant_turn_2"):
        return str(sample.get("v12_assistant_turn_2") or "")
    return str(sample.get("output") or sample.get("v12_assistant_turn_1") or "")


def _gold_think(sample: Dict[str, Any]) -> str:
    output = _gold_output_text(sample)
    parsed = parse_agent_output_v12(output)
    return str(parsed.get("think") or "").strip()


def _gold_answer(sample: Dict[str, Any]) -> str:
    output = _gold_output_text(sample)
    answer = _extract_answer_text(output)
    if answer:
        return answer
    meta = sample.get("metadata") or {}
    return str(
        sample.get("gold_answer")
        or sample.get("canonical_answer")
        or meta.get("gold_answer")
        or meta.get("canonical_answer")
        or ""
    ).strip()


def _gold_compress_range(sample: Dict[str, Any]) -> Optional[List[int]]:
    output = str(sample.get("output") or "")
    parsed = parse_agent_output_v12(output)
    tc = parsed.get("tool_call") or {}
    if tc.get("name") != "compress":
        return None
    tr = (tc.get("arguments") or {}).get("time_range")
    if not isinstance(tr, list) or len(tr) != 2:
        return None
    try:
        return [int(tr[0]), int(tr[1])]
    except (TypeError, ValueError):
        return None


def _gold_compress_summary(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    output = str(sample.get("output") or sample.get("v12_assistant_turn_1") or "")
    parsed = parse_agent_output_v12(output)
    tc = parsed.get("tool_call") or {}
    if tc.get("name") != "compress":
        return None
    args = tc.get("arguments") or {}
    tr = args.get("time_range")
    text = args.get("text")
    if not isinstance(tr, list) or len(tr) != 2:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        tr_norm = [int(tr[0]), int(tr[1])]
    except (TypeError, ValueError):
        return None
    if tr_norm[1] <= tr_norm[0]:
        return None
    return {"time_range": tr_norm, "text": text.strip()}


def _apply_oracle_compress_recovery(
    memory: Any,
    compress_samples: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> bool:
    """Apply a gold compress target after policy missed system compression."""
    for sample in compress_samples:
        summary = _gold_compress_summary(sample)
        if summary is None:
            continue
        memory.compress(summary)
        stats["oracle_compress_recoveries"] = stats.get("oracle_compress_recoveries", 0) + 1
        return True
    stats["skipped"]["oracle_compress_recovery_missing_gold"] = (
        stats["skipped"].get("oracle_compress_recovery_missing_gold", 0)
        + max(1, len(compress_samples))
    )
    return False


def _compress_diag_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("compress_prefix_diagnostic")
    if isinstance(diag, dict) and diag:
        return diag
    raw = (
        result.get("raw_output")
        or result.get("raw")
        or result.get("output")
        or result.get("raw_text")
        or ""
    )
    diag = diagnose_compress_output_v12(str(raw))
    result["compress_prefix_diagnostic"] = diag
    return diag


def _record_compress_diag_stats(
    stats: Dict[str, Any],
    result: Dict[str, Any],
    *,
    prefix: str,
) -> None:
    diag = _compress_diag_from_result(result)
    bucket = stats.setdefault(prefix, {
        "total": 0,
        "prefix_level_sum": 0,
        "tool_call_open": 0,
        "front_prefix_ok": 0,
        "text_started": 0,
        "text_closed": 0,
        "json_complete": 0,
        "tool_call_closed": 0,
        "likely_truncated": 0,
        "missing_tool_close_after_complete_json": 0,
        "by_label": {},
    })
    bucket["total"] += 1
    bucket["prefix_level_sum"] += int(diag.get("prefix_level", 0) or 0)
    for key in [
        "tool_call_open",
        "front_prefix_ok",
        "text_started",
        "text_closed",
        "json_complete",
        "tool_call_closed",
        "likely_truncated",
        "missing_tool_close_after_complete_json",
    ]:
        if diag.get(key):
            bucket[key] += 1
    label = str(diag.get("label") or "unknown")
    bucket["by_label"][label] = bucket["by_label"].get(label, 0) + 1
    total = max(int(bucket["total"]), 1)
    bucket["prefix_level_mean"] = bucket["prefix_level_sum"] / total
    for key in [
        "tool_call_open",
        "front_prefix_ok",
        "text_started",
        "text_closed",
        "json_complete",
        "tool_call_closed",
        "likely_truncated",
        "missing_tool_close_after_complete_json",
    ]:
        bucket[f"{key}_rate"] = bucket[key] / total


def _normalise_range(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return [start, end]


def _word_tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(t) >= 3
    ]


def _token_overlap(a: str, b: str) -> float:
    aa = set(_word_tokens(a))
    if not aa:
        return 0.0
    return len(aa & set(_word_tokens(b))) / max(len(aa), 1)


def _answer_matches(gold_answer: str, policy_answer: str) -> bool:
    gold_answer = str(gold_answer or "").strip()
    policy_answer = str(policy_answer or "").strip()
    if not gold_answer or not policy_answer:
        return False
    return (
        policy_answer.lower() == gold_answer.lower()
        or _token_overlap(gold_answer, policy_answer) >= 0.65
    )


def _answer_visible_in_text(answer: str, text: str) -> bool:
    answer = str(answer or "").strip()
    if not answer:
        return False
    if answer.lower() in text.lower() and len(answer) >= 3:
        return True
    return _token_overlap(answer, text) >= 0.65


def _tagged_text_from_prompt(messages: List[Dict[str, Any]], tag: str) -> str:
    text = _content_text(messages)
    blocks = re.findall(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return "\n".join(blocks)


def _query_diagnostics_from_prompt(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = _content_text(messages)
    active_blocks = re.findall(
        r"<active_query>(.*?)</active_query>",
        text,
        flags=re.DOTALL,
    )
    response_blocks = re.findall(
        r"<response_history>(.*?)</response_history>",
        text,
        flags=re.DOTALL,
    )
    answer_count = 0
    for block in response_blocks:
        answer_count += len(re.findall(r"(?m)^\s*(?:\[[^\]]+\]\s*)?A:\s*\S", block))
    out: Dict[str, Any] = {
        "active_query_in_prompt": bool(active_blocks),
        "active_query_block_count": len(active_blocks),
        "response_history_answer_count": answer_count,
    }
    if active_blocks:
        m = re.search(r"(?m)^\s*(?:\[[^\]]+\]\s*)?Q:\s*(.+?)\s*$", active_blocks[-1])
        if m:
            out["active_query_text"] = m.group(1)[:240]
    return out


def _evidence_text_from_prompt(messages: List[Dict[str, Any]]) -> str:
    """Return text evidence only, excluding active_query/options/user_input.

    For MC questions the correct option text is intentionally present in
    <active_query>. That is not evidence that the student can answer without
    recall, so DAgger's missed-recall filter must look only at memory and
    returned recall evidence.
    """
    return "\n".join(
        part for part in (
            _tagged_text_from_prompt(messages, "memory"),
            _tagged_text_from_prompt(messages, "recall_result"),
            _tagged_text_from_prompt(messages, "recalled_frames"),
        )
        if part
    )


def _json_blocks_from_prompt(messages: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in re.findall(fr"<{tag}>(.*?)</{tag}>", _content_text(messages), flags=re.DOTALL):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _answer_visible_in_prompt_window(
    sample: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> bool:
    """Conservative check for whether answer frames are in visible evidence."""
    answer_ranges = [
        (chunk * AGENT_CHUNK_SEC, (chunk + 1) * AGENT_CHUNK_SEC)
        for chunk in _answer_chunks(sample)
    ]
    if not answer_ranges:
        return False

    visible_ranges: List[Tuple[float, float]] = []
    for block in _json_blocks_from_prompt(messages, "visual_window"):
        try:
            visible_ranges.append((float(block["start"]), float(block["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    for block in _json_blocks_from_prompt(messages, "recalled_frames"):
        raw_range = block.get("time_range")
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            continue
        try:
            visible_ranges.append((float(raw_range[0]), float(raw_range[1])))
        except (TypeError, ValueError):
            continue

    return any(
        _intervals_overlap(answer_range, visible_range)
        for answer_range in answer_ranges
        for visible_range in visible_ranges
    )


def _answer_visible_in_prompt(
    answer: str,
    messages: List[Dict[str, Any]],
    sample: Optional[Dict[str, Any]] = None,
) -> bool:
    if _answer_visible_in_text(answer, _evidence_text_from_prompt(messages)):
        return True
    if sample is not None and _answer_visible_in_prompt_window(sample, messages):
        return True
    return False


def _memory_text_from_prompt(messages: List[Dict[str, Any]]) -> str:
    return _tagged_text_from_prompt(messages, "memory")


def _has_ngram_repetition(tokens: List[str], n: int = 4, threshold: float = 0.22) -> bool:
    if len(tokens) < n * 3:
        return False
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return False
    return 1.0 - (len(set(grams)) / len(grams)) >= threshold


def _int_chunk_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    out: List[int] = []
    try:
        iterator = list(raw)
    except TypeError:
        return []
    for x in iterator:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _sample_chunk_field(sample: Dict[str, Any], *keys: str) -> List[int]:
    meta = sample.get("metadata") or {}
    for key in keys:
        vals = _int_chunk_list(sample.get(key))
        if vals:
            return vals
    for key in keys:
        vals = _int_chunk_list(meta.get(key))
        if vals:
            return vals
    return []


def _answer_chunks(sample: Dict[str, Any]) -> List[int]:
    out = _sample_chunk_field(
        sample,
        "answer_chunks",
        "expected_answer_chunks",
    )
    if out:
        return out
    meta = sample.get("metadata") or {}
    per_emit = sample.get("per_emit_answers") or meta.get("per_emit_answers") or []
    for item in per_emit:
        if isinstance(item, dict) and item.get("chunk") is not None:
            try:
                out.append(int(item["chunk"]))
            except (TypeError, ValueError):
                pass
    return sorted(set(out))


def _support_chunks(sample: Dict[str, Any]) -> List[int]:
    return _sample_chunk_field(
        sample,
        "support_chunks",
        "grounding_frames",
        "evidence_chunks",
    )


def _recall_target_chunks(sample: Dict[str, Any]) -> List[int]:
    """Chunks that a recall result should cover for this sample.

    Recall retrieves historical evidence, so support/grounding chunks are the
    primary target. Answer chunks are only a fallback for legacy rows that do
    not carry explicit support metadata.
    """
    return _support_chunks(sample) or _answer_chunks(sample)


def _policy_recall_query(result: Dict[str, Any]) -> Dict[str, Any]:
    query = ((result.get("payload") or {}).get("query") or {})
    return query if isinstance(query, dict) else {}


def _parse_recall_time_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*", value)
        if not m:
            return None
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
    return None


def _chunks_for_recall_time_range(value: Any) -> List[int]:
    parsed = _parse_recall_time_range(value)
    if parsed is None:
        return []
    t0, t1 = parsed
    if t0 > t1:
        t0, t1 = t1, t0
    if t1 <= t0:
        return []
    first = int(t0 // AGENT_CHUNK_SEC)
    last = int((t1 - 1e-9) // AGENT_CHUNK_SEC)
    if last < first:
        last = first
    return list(range(max(0, first), max(0, last) + 1))


def _min_chunk_distance(xs: Iterable[int], ys: Iterable[int]) -> Optional[int]:
    x_list = list(xs)
    y_list = list(ys)
    if not x_list or not y_list:
        return None
    return min(abs(int(x) - int(y)) for x in x_list for y in y_list)


def _question_text(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata") or {}
    return str(sample.get("question") or meta.get("question") or "").strip()


def _annotate_recall_diagnostics(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    detail: Dict[str, Any],
) -> Tuple[set[int], set[int], set[int], Optional[int]]:
    query = _policy_recall_query(result)
    query_text = str(query.get("query") or "").strip()
    query_range = query.get("time_range")
    if query:
        detail["recall_query_text"] = query_text
        detail["recall_query_time_range"] = query_range
        parsed_range = _parse_recall_time_range(query_range)
        detail["recall_query_time_range_valid"] = parsed_range is not None
        question = _question_text(sample)
        if query_text and question:
            detail["recall_query_question_overlap"] = round(
                _token_overlap(query_text, question),
                3,
            )

    support_chunks = set(_support_chunks(sample))
    answer_chunks = set(_answer_chunks(sample))
    target_chunks = set(_recall_target_chunks(sample))
    returned_chunks = set(_recall_returned_chunks(result))
    query_chunks = set(_chunks_for_recall_time_range(query_range))
    distance = _min_chunk_distance(query_chunks, target_chunks)

    if support_chunks:
        detail["recall_support_chunks"] = sorted(support_chunks)
    if answer_chunks:
        detail["recall_answer_chunks"] = sorted(answer_chunks)
    if target_chunks:
        detail["recall_target_chunks"] = sorted(target_chunks)
    if result.get("recall_result") is not None:
        detail["recall_returned_chunks"] = sorted(returned_chunks)
    if query_chunks:
        detail["recall_query_chunks"] = sorted(query_chunks)
        if target_chunks:
            detail["recall_query_hits_target"] = bool(query_chunks & target_chunks)
            if distance is not None:
                detail["recall_query_distance_to_target_chunks"] = int(distance)
                detail["recall_query_within_retrieval_margin"] = (
                    int(distance) <= recall_time_range_margin_chunks()
                )
    return target_chunks, returned_chunks, query_chunks, distance


def _classify_dagger_corrections(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Classify why this on-policy state deserves a gold correction row."""
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    sample_type = str(sample.get("sample_type", ""))
    policy_action = str(result.get("final_action") or result.get("action") or "")
    first_action = str(result.get("action") or "")
    prompt_has_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    format_ok = bool(result.get("format_ok", True)) and not result.get("action_space_error")
    if not format_ok or first_action in {"unknown", "invalid"}:
        reasons.append("format_error")

    policy_think = str(result.get("think") or "").strip()
    gold_think = _gold_think(sample)
    if policy_think:
        toks = _word_tokens(policy_think)
        unique_ratio = len(set(toks)) / max(len(toks), 1)
        memory_overlap = _token_overlap(policy_think, _memory_text_from_prompt(onpolicy_prompt))
        gold_overlap = _token_overlap(policy_think, gold_think)
        too_long_vs_gold = bool(gold_think and len(toks) > max(120, 3 * len(_word_tokens(gold_think))))
        if (
            unique_ratio < 0.42
            or _has_ngram_repetition(toks)
            or too_long_vs_gold
            or (memory_overlap >= 0.72 and gold_overlap < 0.35)
            or (gold_think and gold_overlap < 0.12 and len(toks) >= 18)
        ):
            reasons.append("repeated_or_stale_think")
            detail["think_unique_ratio"] = round(unique_ratio, 3)
            detail["think_memory_overlap"] = round(memory_overlap, 3)
            detail["think_gold_overlap"] = round(gold_overlap, 3)
            detail["policy_think_tokens"] = len(toks)

    if prompt_has_compress or sample_type == "compress":
        compress_diag = _compress_diag_from_result(result)
        detail["compress_prefix_diagnostic"] = compress_diag
        runtime_error = str(result.get("compress_runtime_error") or "").strip()
        if runtime_error:
            detail["compress_runtime_error"] = runtime_error
        if first_action != "compress":
            reasons.append("missed_compress")
            if compress_diag.get("front_prefix_ok") and compress_diag.get("likely_truncated"):
                reasons.append("compress_good_prefix_truncated")
            elif compress_diag.get("front_prefix_ok"):
                reasons.append("compress_good_prefix_unparsed")
            elif compress_diag.get("tool_call_open"):
                reasons.append("compress_bad_prefix_after_tool_open")
        else:
            pred_range = _normalise_range(
                ((result.get("payload") or {}).get("summary") or {}).get("time_range")
            )
            summary_text = str(
                (((result.get("payload") or {}).get("summary") or {}).get("text")) or ""
            ).strip()
            gold_range = _gold_compress_range(sample)
            if pred_range is None:
                reasons.append("bad_compress_range_schema")
                detail["policy_compress_range"] = (
                    ((result.get("payload") or {}).get("summary") or {}).get("time_range")
                )
            elif not summary_text:
                reasons.append("empty_compress_summary")
                detail["policy_compress_range"] = pred_range
            elif gold_range and (pred_range[1] <= gold_range[0] or pred_range[0] >= gold_range[1]):
                reasons.append("bad_compress_range")
                detail["gold_compress_range"] = gold_range
                detail["policy_compress_range"] = pred_range
            elif runtime_error and runtime_error not in {"ok", "applied"}:
                if runtime_error in {"range_no_recent_chunks", "range_too_small"}:
                    reasons.append("bad_compress_range")
                elif runtime_error in {"bad_range_schema", "empty_summary_text"}:
                    reasons.append("bad_compress_range_schema")

    gold_answer = _gold_answer(sample)
    policy_answer = str(((result.get("final_payload") or result.get("payload") or {}).get("response")) or "").strip()
    if sample_type == "recall":
        if first_action != "recall":
            if _answer_visible_in_prompt(gold_answer, onpolicy_prompt, sample):
                reasons.append("recall_answer_visible_in_policy_prompt")
            else:
                reasons.append("missed_recall")
        else:
            query = _policy_recall_query(result)
            query_text = str(query.get("query") or "").strip()
            if not query_text or result.get("recall_prepare_error_reason"):
                reasons.append("bad_recall_query")
                detail["recall_prepare_error_reason"] = result.get("recall_prepare_error_reason", "")
            target_chunks, returned_chunks, query_chunks, _ = _annotate_recall_diagnostics(
                sample,
                result,
                detail,
            )
            if target_chunks and query_chunks and not (target_chunks & query_chunks):
                reasons.append("recall_time_range_miss")
            if target_chunks and result.get("recall_result") is not None:
                if not returned_chunks:
                    reasons.append("recall_retrieval_empty")
                elif not (target_chunks & returned_chunks):
                    reasons.append("recall_retrieval_miss")
            if result.get("recall_step2_blocked"):
                reasons.append("format_error")
            elif policy_action == "silent" and gold_answer:
                reasons.append("missed_response")
            elif policy_action not in {"response", "silent"} and gold_answer:
                reasons.append("missed_response")
        if policy_action == "response" and gold_answer and policy_answer:
            if not _answer_matches(gold_answer, policy_answer):
                reasons.append("wrong_response")

    if sample_type in {"silent", "response"} and first_action == "recall":
        reasons.append("over_recall")
        target_chunks, _, query_chunks, _ = _annotate_recall_diagnostics(
            sample,
            result,
            detail,
        )
        if target_chunks and query_chunks and not (target_chunks & query_chunks):
            reasons.append("recall_time_range_miss")
        if result.get("recall_prepare_error_reason"):
            detail["recall_prepare_error_reason"] = result.get("recall_prepare_error_reason", "")

    if sample_type == "response":
        if policy_action == "silent":
            reasons.append("missed_response")
        elif policy_action == "response" and gold_answer and policy_answer:
            if not _answer_matches(gold_answer, policy_answer):
                reasons.append("wrong_response")

    current_chunk = int(sample.get("chunk_idx", 0) or 0)
    chunks = _answer_chunks(sample)
    if sample_type == "silent" and chunks and current_chunk < min(chunks):
        if policy_action == "response":
            reasons.append("early_answer")
            detail["answer_chunks"] = chunks
    if sample_type == "silent" and chunks and current_chunk > max(chunks):
        if policy_action == "response":
            reasons.append("late_response")
            detail["answer_chunks"] = chunks
            detail["late_by_chunks"] = current_chunk - max(chunks)
            detail["late_by_sec"] = (current_chunk - max(chunks)) * AGENT_CHUNK_SEC

    if "missed_response" in reasons:
        detail.update(_query_diagnostics_from_prompt(onpolicy_prompt))

    # This is a data-construction warning, not a useful DAgger correction:
    # under the student's memory state recall is no longer minimal.
    if "recall_answer_visible_in_policy_prompt" in reasons and "missed_recall" not in reasons:
        detail["recall_prompt_leak"] = True
    return sorted(set(reasons)), detail


def _build_dagger_messages(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    *,
    base_path: Path,
    data_dir: Path,
    frame_protocol: str,
    render_layout: str,
) -> List[Dict[str, Any]]:
    """Use model-memory prompt + gold assistant tail."""
    gold_messages = build_messages(
        sample,
        base_path,
        data_dir=data_dir,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
    )
    if len(gold_messages) < 3:
        raise ValueError("gold messages missing assistant target")
    if len(onpolicy_prompt) != 2:
        raise ValueError(f"expected single-step on-policy prompt, got {len(onpolicy_prompt)}")
    return deepcopy(onpolicy_prompt) + deepcopy(gold_messages[2:])


def _build_dagger_recall_query_messages(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    *,
    base_path: Path,
    data_dir: Path,
    frame_protocol: str,
    render_layout: str,
) -> List[Dict[str, Any]]:
    gold_messages = build_messages(
        sample,
        base_path,
        data_dir=data_dir,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
    )
    if len(gold_messages) < 3:
        raise ValueError("gold recall messages missing first assistant target")
    if len(onpolicy_prompt) != 2:
        raise ValueError(f"expected single-step on-policy prompt, got {len(onpolicy_prompt)}")
    return deepcopy(onpolicy_prompt) + [deepcopy(gold_messages[2])]


def _recall_returned_chunks(result: Dict[str, Any]) -> List[int]:
    rr = result.get("recall_result") or {}
    raw = rr.get("returned_chunks") or rr.get("chunks") or []
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _policy_recall_supports_gold(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    recall_messages: List[Dict[str, Any]],
) -> bool:
    gold_answer = _gold_answer(sample)
    if not gold_answer:
        return True
    answer_chunks = set(_recall_target_chunks(sample))
    returned_chunks = set(_recall_returned_chunks(result))
    if answer_chunks and returned_chunks:
        return bool(answer_chunks & returned_chunks)
    return _answer_visible_in_prompt(gold_answer, recall_messages, sample)


def _bump(stats: Dict[str, Any], key: str, amount: int = 1) -> None:
    stats[key] = stats.get(key, 0) + amount


def _bump_bucket(stats: Dict[str, Any], bucket_key: str, item_key: str, amount: int = 1) -> None:
    bucket = stats.setdefault(bucket_key, {})
    bucket[item_key] = bucket.get(item_key, 0) + amount


def _bump_nested_bucket(
    stats: Dict[str, Any],
    bucket_key: str,
    item_key: str,
    subkey: str,
    amount: int = 1,
) -> None:
    bucket = stats.setdefault(bucket_key, {})
    inner = bucket.setdefault(str(item_key or "unknown"), {})
    inner[str(subkey or "unknown")] = inner.get(str(subkey or "unknown"), 0) + amount


def _gold_first_action(sample: Dict[str, Any]) -> str:
    sample_type = str(sample.get("sample_type") or "")
    if sample_type in {"recall", "compress", "response", "silent"}:
        return sample_type
    parsed = parse_agent_output_v12(_gold_output_text(sample))
    kind = str(parsed.get("kind") or "")
    if kind == "answer":
        return "response" if str(parsed.get("answer_text") or "").strip() else "silent"
    return kind or sample_type or "unknown"


def _gold_final_action(sample: Dict[str, Any]) -> str:
    sample_type = str(sample.get("sample_type") or "")
    if sample_type == "compress":
        return "compress"
    parsed = parse_agent_output_v12(_gold_output_text(sample))
    kind = str(parsed.get("kind") or "")
    if kind == "answer":
        return "response" if str(parsed.get("answer_text") or "").strip() else "silent"
    if kind in {"recall", "compress"}:
        return kind
    return _gold_first_action(sample)


def _policy_answer(result: Dict[str, Any]) -> str:
    payload = result.get("final_payload") or result.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("response") or "").strip()


def _record_action_eval_stats(
    stats: Dict[str, Any],
    *,
    sample_type: str,
    gold_first: str,
    gold_final: str,
    first_action: str,
    final_action: str,
    format_ok: bool,
    parse_failed: bool,
    reasons: set[str],
    answer_eval: str,
) -> str:
    _bump(stats, "dagger_targets_seen")
    _bump_bucket(stats, "dagger_targets_by_sample_type", sample_type or "unknown")
    _bump_bucket(stats, "dagger_gold_first_action_targets", gold_first)
    _bump_bucket(stats, "dagger_gold_final_action_targets", gold_final)
    _bump_nested_bucket(stats, "dagger_policy_first_action_by_gold", gold_first, first_action)
    _bump_nested_bucket(stats, "dagger_policy_final_action_by_gold", gold_final, final_action)
    if first_action == gold_first:
        _bump(stats, "dagger_first_action_correct")
    else:
        _bump(stats, "dagger_first_action_incorrect")
    if final_action == gold_final:
        _bump(stats, "dagger_final_action_correct")
    else:
        _bump(stats, "dagger_final_action_incorrect")
    if format_ok:
        _bump(stats, "dagger_format_ok_targets")
    else:
        _bump(stats, "dagger_format_bad_targets")

    flags: List[str] = []
    if parse_failed:
        flags.append("parse_error")
    if not format_ok and not parse_failed:
        flags.append("format_error")
    if "wrong_response" in reasons or answer_eval == "wrong":
        flags.append("answer_error")
    if reasons & {
        "missed_response",
        "missed_recall",
        "missed_compress",
        "over_recall",
        "early_answer",
        "late_response",
    }:
        flags.append("action_error")
    if reasons & {
        "bad_recall_query",
        "recall_retrieval_empty",
        "recall_retrieval_miss",
        "recall_time_range_miss",
    }:
        flags.append("retrieval_error")
    if reasons & {
        "bad_compress_json",
        "bad_compress_range_schema",
        "bad_compress_range",
        "empty_compress_summary",
    }:
        flags.append("compress_error")
    if "repeated_or_stale_think" in reasons:
        flags.append("think_error")
    if not flags and reasons:
        flags.append("other_error")
    if not flags:
        flags.append("correct")

    for flag in sorted(set(flags)):
        _bump_bucket(stats, "dagger_error_flags", flag)
        _bump_nested_bucket(stats, "dagger_error_flags_by_sample_type", sample_type, flag)

    priority = [
        "parse_error",
        "format_error",
        "answer_error",
        "action_error",
        "retrieval_error",
        "compress_error",
        "think_error",
        "other_error",
        "correct",
    ]
    primary = next((flag for flag in priority if flag in flags), "other_error")
    _bump_bucket(stats, "dagger_primary_error_type", primary)
    _bump_nested_bucket(stats, "dagger_primary_error_type_by_sample_type", sample_type, primary)
    return primary


def _record_dagger_target_stats(
    stats: Dict[str, Any],
    sample: Dict[str, Any],
    result: Dict[str, Any],
    correction_detail: Dict[str, Any],
    reasons: Optional[List[str]] = None,
) -> None:
    sample_type = str(sample.get("sample_type") or "")
    first_action = str(result.get("action") or "unknown")
    final_action = str(result.get("final_action") or result.get("action") or "unknown")
    gold_first = _gold_first_action(sample)
    gold_final = _gold_final_action(sample)
    reason_set = set(reasons or [])
    format_ok = bool(result.get("format_ok", True)) and not bool(result.get("action_space_error"))
    parse_failed = bool(result.get("format_error")) or (
        first_action == "unknown" and not bool(result.get("action_space_error"))
    )
    gold_answer = _gold_answer(sample)
    policy_answer = _policy_answer(result)
    answer_eval = "not_applicable"
    if gold_answer:
        _bump(stats, "dagger_answer_targets_seen")
        _bump_bucket(stats, "dagger_answer_targets_by_sample_type", sample_type or "unknown")
        if final_action != "response":
            answer_eval = "missed"
            _bump(stats, "dagger_answer_missed")
        elif not policy_answer:
            answer_eval = "empty"
            _bump(stats, "dagger_answer_empty")
        elif _answer_matches(gold_answer, policy_answer):
            answer_eval = "correct"
            _bump(stats, "dagger_answer_correct")
        else:
            answer_eval = "wrong"
            _bump(stats, "dagger_answer_wrong")
        _bump_bucket(stats, "dagger_answer_eval", answer_eval)
        _bump_nested_bucket(stats, "dagger_answer_eval_by_sample_type", sample_type, answer_eval)

    primary_error = _record_action_eval_stats(
        stats,
        sample_type=sample_type or "unknown",
        gold_first=gold_first,
        gold_final=gold_final,
        first_action=first_action,
        final_action=final_action,
        format_ok=format_ok,
        parse_failed=parse_failed,
        reasons=reason_set,
        answer_eval=answer_eval,
    )
    correction_detail["target_eval"] = {
        "gold_first_action": gold_first,
        "gold_final_action": gold_final,
        "policy_first_action": first_action,
        "policy_final_action": final_action,
        "first_action_match": first_action == gold_first,
        "final_action_match": final_action == gold_final,
        "format_ok": format_ok,
        "parse_failed": parse_failed,
        "answer_eval": answer_eval,
        "primary_error_type": primary_error,
    }
    _bump_bucket(stats, "targets_by_type_seen", sample_type or "unknown")

    if sample_type == "recall":
        _bump(stats, "gold_recall_targets_seen")
        _bump_bucket(stats, "gold_recall_policy_first_action", first_action)
        if first_action == "recall":
            _bump(stats, "gold_recall_policy_recall")
            target_chunks = set(correction_detail.get("recall_target_chunks") or [])
            returned_chunks = set(correction_detail.get("recall_returned_chunks") or [])
            if result.get("recall_result") is None:
                retrieval_status = "not_run"
            elif not returned_chunks:
                retrieval_status = "empty"
            elif target_chunks and (target_chunks & returned_chunks):
                retrieval_status = "hit"
            else:
                retrieval_status = "miss"
            _bump_bucket(stats, "gold_recall_retrieval_status", retrieval_status)
            if "recall_query_hits_target" in correction_detail:
                _bump_bucket(
                    stats,
                    "gold_recall_query_time_range_status",
                    "hit" if correction_detail.get("recall_query_hits_target") else "miss",
                )
            _bump_bucket(stats, "gold_recall_post_action", final_action)
            if result.get("recall_step2_blocked"):
                _bump(stats, "gold_recall_post_blocked")
        else:
            _bump(stats, "gold_recall_policy_no_recall")

    if sample_type in {"silent", "response"} and first_action == "recall":
        _bump(stats, "policy_recall_on_non_recall_targets")
        _bump_bucket(stats, "policy_recall_on_non_recall_by_gold_type", sample_type)


def _safe_rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        den = float(denominator)
        if den <= 0:
            return None
        return round(float(numerator) / den, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def finalize_dagger_stats(stats: Dict[str, Any]) -> None:
    """Attach derived DAgger rates and drop private live-tracking fields."""
    metrics = stats.setdefault("metrics", {})
    targets = int(stats.get("dagger_targets_seen", 0) or 0)
    answers = int(stats.get("dagger_answer_targets_seen", 0) or 0)
    if targets:
        metrics["dagger_first_action_accuracy"] = _safe_rate(
            stats.get("dagger_first_action_correct", 0), targets,
        )
        metrics["dagger_final_action_accuracy"] = _safe_rate(
            stats.get("dagger_final_action_correct", 0), targets,
        )
        metrics["dagger_format_ok_rate"] = _safe_rate(
            stats.get("dagger_format_ok_targets", 0), targets,
        )
    if answers:
        metrics["dagger_answer_accuracy"] = _safe_rate(
            stats.get("dagger_answer_correct", 0), answers,
        )
        metrics["dagger_answer_emit_rate"] = _safe_rate(
            stats.get("dagger_answer_correct", 0) + stats.get("dagger_answer_wrong", 0),
            answers,
        )
    missed = int(stats.get("missed_response_timing_targets", 0) or 0)
    if missed:
        metrics["missed_response_late_attempt_rate"] = _safe_rate(
            stats.get("missed_response_late_attempts", 0), missed,
        )
        metrics["missed_response_late_correct_rate"] = _safe_rate(
            stats.get("missed_response_late_correct", 0), missed,
        )
        metrics["missed_response_late_wrong_rate"] = _safe_rate(
            stats.get("missed_response_late_wrong", 0), missed,
        )
        metrics["missed_response_unresolved_rate"] = _safe_rate(
            stats.get("missed_response_unresolved", 0), missed,
        )
    response_gold = int(stats.get("response_event_gold_total", 0) or 0)
    if response_gold:
        response_tp = (
            int(stats.get("response_event_matched_correct_on_time", 0) or 0)
            + int(stats.get("response_event_matched_correct_late", 0) or 0)
        )
        response_wrong = (
            int(stats.get("response_event_matched_wrong_on_time", 0) or 0)
            + int(stats.get("response_event_matched_wrong_late", 0) or 0)
        )
        metrics["response_event_outcome"] = _safe_rate(response_tp, response_gold)
        metrics["response_event_wrong_rate"] = _safe_rate(response_wrong, response_gold)
        metrics["response_event_initial_miss_rate"] = _safe_rate(
            stats.get("response_event_missed_initial", 0), response_gold,
        )
        metrics["response_event_unresolved_miss_rate"] = _safe_rate(
            stats.get("response_event_missed_unresolved", 0), response_gold,
        )
        metrics["response_event_late_tp_rate"] = _safe_rate(
            stats.get("response_event_matched_correct_late", 0), response_gold,
        )
    response_fp = (
        int(stats.get("response_event_false_positive_early", 0) or 0)
        + int(stats.get("response_event_false_positive_over", 0) or 0)
    )
    if response_gold or response_fp:
        denominator = response_gold + response_fp
        metrics["response_event_false_positive_rate"] = _safe_rate(response_fp, denominator)
        metrics["response_event_early_fp_rate"] = _safe_rate(
            stats.get("response_event_false_positive_early", 0), denominator,
        )
    for key in list(stats.keys()):
        if str(key).startswith("_"):
            stats.pop(key, None)


def _build_dagger_recall_response_messages(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    *,
    base_path: Path,
    data_dir: Path,
    frame_protocol: str,
    render_layout: str,
) -> Optional[List[Dict[str, Any]]]:
    recall_messages = result.get("recall_messages")
    if not isinstance(recall_messages, list) or len(recall_messages) < 4:
        return None
    if not _policy_recall_supports_gold(sample, result, recall_messages):
        return None
    gold_messages = build_messages(
        sample,
        base_path,
        data_dir=data_dir,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
    )
    if len(gold_messages) < 3:
        raise ValueError("gold messages missing final assistant target")
    out = deepcopy(recall_messages)
    if out and out[0].get("role") == "system":
        out[0] = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt_for_frame_protocol(
                    frame_protocol,
                    prompt_kind="post_recall",
                    render_layout=render_layout,
                ),
            }],
        }
    # For gold recall samples this is the teacher post-recall answer turn.
    # For over-recall corrections on silent/response samples it is the gold
    # direct answer/silent turn, reused as the no-tools post-recall decision.
    return out + [deepcopy(gold_messages[-1])]


def _attach_dagger_metadata(
    row: Dict[str, Any],
    *,
    ckpt: str,
    result: Dict[str, Any],
    prompt_is_compress: bool,
    correction_only: bool,
    reasons: List[str],
    selected_reasons: List[str],
    correction_detail: Dict[str, Any],
    dagger_subtype: str,
) -> None:
    row["dagger"] = {
        "policy_ckpt": ckpt,
        "rollout_action": result.get("action", ""),
        "rollout_final_action": result.get("final_action", ""),
        "rollout_format_ok": bool(result.get("format_ok", True)),
        "rollout_action_space_error": result.get("action_space_error", ""),
        "rollout_invalid_action": result.get("invalid_action", ""),
        "rollout_think": str(result.get("think", ""))[:1000],
        "rollout_payload": result.get("payload", {}),
        "rollout_raw_output": str(result.get("raw_output") or result.get("raw") or "")[:2000],
        "rollout_recall_step2_raw_output": str(result.get("recall_step2_raw_text", ""))[:2000],
        "rollout_compress_prefix_diagnostic": result.get("compress_prefix_diagnostic", {}),
        "rollout_inter_chunk_compress_prompt": bool(prompt_is_compress),
        "rollout_forced_compress_trigger": bool(result.get("forced_compress_trigger", False)),
        "rollout_compress_trigger_source": result.get("compress_trigger_source", ""),
        "pre_compress_memory_token_count": result.get("pre_compress_memory_token_count"),
        "pre_compress_recent_thinks": result.get("pre_compress_recent_thinks"),
        "post_compress_memory_token_count": result.get("post_compress_memory_token_count"),
        "compress_memory_token_delta": result.get("compress_memory_token_delta"),
        "compress_runtime_error": result.get("compress_runtime_error", ""),
        "compress_applied": bool(result.get("compress_applied", False)),
        "compress_applied_chunks": result.get("compress_applied_chunks", []),
        "compress_recovery": result.get("compress_recovery", {}),
        "recall_prepare_error_reason": result.get("recall_prepare_error_reason", ""),
        "recall_returned_chunks": result.get("recall_returned_chunks", []),
        "recall_step2_blocked": result.get("recall_step2_blocked", {}),
        "memory_token_count": result.get("memory_token_count"),
        "prompt_text_token_count": result.get("prompt_text_token_count"),
        "correction_only": bool(correction_only),
        "correction_reasons": reasons,
        "selected_correction_reasons": selected_reasons,
        "correction_detail": correction_detail,
        "dagger_subtype": dagger_subtype,
    }


def _write_dagger_row(
    row: Dict[str, Any],
    *,
    fout,
    stats: Dict[str, Any],
    selected_reasons: List[str],
) -> None:
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] += 1
    st = row.get("sample_type", "")
    stats["by_type"][st] = stats["by_type"].get(st, 0) + 1
    for r in selected_reasons:
        bucket = stats.setdefault("by_selected_correction_reason", {})
        bucket[r] = bucket.get(r, 0) + 1


def _emit_dagger_row(
    *,
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    result: Dict[str, Any],
    fout,
    stats: Dict[str, Any],
    ckpt: str,
    data_dir: Path,
    frame_protocol: str,
    render_layout: str,
    include_failed_targets: bool,
    sample_types: set[str],
    correction_only: bool,
    correction_reasons: set[str],
) -> bool:
    ok, reason = _target_allowed(
        sample,
        onpolicy_prompt,
        sample_types=sample_types,
        include_failed_targets=include_failed_targets,
    )
    if not ok:
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
        return False

    prompt_is_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    if prompt_is_compress or sample.get("sample_type") == "compress":
        _record_compress_diag_stats(
            stats, result, prefix="compress_prefix_by_target",
        )

    reasons, correction_detail = _classify_dagger_corrections(
        sample,
        onpolicy_prompt,
        result,
    )
    _record_dagger_target_stats(stats, sample, result, correction_detail, reasons)
    for r in reasons:
        bucket = stats.setdefault("by_correction_reason", {})
        bucket[r] = bucket.get(r, 0) + 1
    selected_reasons = sorted(set(reasons) & set(correction_reasons))
    if correction_only and not selected_reasons:
        stats["skipped"]["no_selected_correction"] = (
            stats["skipped"].get("no_selected_correction", 0) + 1
        )
        if reasons:
            key = "only_unselected_correction"
            stats["skipped"][key] = stats["skipped"].get(key, 0) + 1
        return False

    rows_written = 0

    try:
        if sample.get("sample_type") == "recall":
            first_action = str(result.get("action") or "")
            first_reasons = {
                "missed_recall",
                "repeated_or_stale_think",
                "bad_recall_query",
                "recall_retrieval_empty",
                "recall_retrieval_miss",
                "recall_time_range_miss",
            }
            first_needs = (
                not correction_only
                or bool(set(selected_reasons) & first_reasons)
                or first_action in {"unknown", "invalid"}
                or bool(result.get("action_space_error"))
            )
            second_needs = (
                (not correction_only or bool(set(selected_reasons) & {"missed_response", "wrong_response"}))
                and first_action == "recall"
            ) or bool(result.get("recall_step2_blocked"))

            if first_needs:
                messages = _build_dagger_recall_query_messages(
                    sample,
                    onpolicy_prompt,
                    base_path=ROOT,
                    data_dir=data_dir,
                    frame_protocol=frame_protocol,
                    render_layout=render_layout,
                )
                row = _emit_row(sample, messages, frame_protocol=frame_protocol)
                _with_sft_turn_policy(
                    row,
                    tool_schema_mode="streaming",
                    loss_assistant_turns="all",
                    sft_subtype="dagger_recall_query",
                    sample_id_suffix="dagger_recall_query",
                )
                _attach_dagger_metadata(
                    row,
                    ckpt=ckpt,
                    result=result,
                    prompt_is_compress=prompt_is_compress,
                    correction_only=correction_only,
                    reasons=reasons,
                    selected_reasons=selected_reasons,
                    correction_detail=correction_detail,
                    dagger_subtype="recall_query",
                )
                _write_dagger_row(
                    row, fout=fout, stats=stats, selected_reasons=selected_reasons,
                )
                rows_written += 1

            if second_needs:
                messages = _build_dagger_recall_response_messages(
                    sample,
                    result,
                    base_path=ROOT,
                    data_dir=data_dir,
                    frame_protocol=frame_protocol,
                    render_layout=render_layout,
                )
                if messages is None:
                    key = "recall_response_correction_unsupported_retrieval"
                    stats["skipped"][key] = (
                        stats["skipped"].get(key, 0) + 1
                    )
                else:
                    row = _emit_row(sample, messages, frame_protocol=frame_protocol)
                    _with_sft_turn_policy(
                        row,
                        tool_schema_mode="post_recall",
                        loss_assistant_turns="last",
                        sft_subtype="dagger_post_recall",
                        sample_id_suffix="dagger_post_recall",
                    )
                    _attach_dagger_metadata(
                        row,
                        ckpt=ckpt,
                        result=result,
                        prompt_is_compress=prompt_is_compress,
                        correction_only=correction_only,
                        reasons=reasons,
                        selected_reasons=selected_reasons,
                        correction_detail=correction_detail,
                        dagger_subtype="post_recall",
                    )
                    _write_dagger_row(
                        row, fout=fout, stats=stats, selected_reasons=selected_reasons,
                    )
                    rows_written += 1

            if rows_written:
                return True
            if correction_only:
                stats["skipped"]["no_emittable_recall_correction"] = (
                    stats["skipped"].get("no_emittable_recall_correction", 0) + 1
                )
                return False

        if (
            sample.get("sample_type") in {"silent", "response"}
            and str(result.get("action") or "") == "recall"
            and (
                not correction_only
                or "over_recall" in selected_reasons
                or bool(result.get("recall_step2_blocked"))
            )
        ):
            messages = _build_dagger_recall_response_messages(
                sample,
                result,
                base_path=ROOT,
                data_dir=data_dir,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
            )
            if messages is None:
                key = "post_recall_correction_unsupported_retrieval"
                stats["skipped"][key] = (
                    stats["skipped"].get(key, 0) + 1
                )
            else:
                row = _emit_row(sample, messages, frame_protocol=frame_protocol)
                _with_sft_turn_policy(
                    row,
                    tool_schema_mode="post_recall",
                    loss_assistant_turns="last",
                    sft_subtype="dagger_post_recall",
                    sample_id_suffix="dagger_post_recall",
                )
                _attach_dagger_metadata(
                    row,
                    ckpt=ckpt,
                    result=result,
                    prompt_is_compress=prompt_is_compress,
                    correction_only=correction_only,
                    reasons=reasons,
                    selected_reasons=selected_reasons,
                    correction_detail=correction_detail,
                    dagger_subtype="post_recall",
                )
                _write_dagger_row(
                    row, fout=fout, stats=stats, selected_reasons=selected_reasons,
                )

        messages = _build_dagger_messages(
            sample,
            onpolicy_prompt,
            base_path=ROOT,
            data_dir=data_dir,
            frame_protocol=frame_protocol,
            render_layout=render_layout,
        )
    except Exception as exc:
        reason = f"render_error:{type(exc).__name__}"
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
        return False

    row = _emit_row(sample, messages, frame_protocol=frame_protocol)
    _with_sft_turn_policy(
        row,
        tool_schema_mode="compress" if sample.get("v12_inter_chunk") else "streaming",
        loss_assistant_turns="all",
        sft_subtype=str(sample.get("sample_type") or ""),
    )
    _attach_dagger_metadata(
        row,
        ckpt=ckpt,
        result=result,
        prompt_is_compress=prompt_is_compress,
        correction_only=correction_only,
        reasons=reasons,
        selected_reasons=selected_reasons,
        correction_detail=correction_detail,
        dagger_subtype=str(sample.get("sample_type") or ""),
    )
    _write_dagger_row(row, fout=fout, stats=stats, selected_reasons=selected_reasons)
    return True


def build_dagger(
    *,
    ckpt: str,
    trajectories: Path,
    out: Path,
    data_dir: Path,
    frames_root: str,
    video_root: Optional[str],
    frame_protocol: str,
    render_layout: str,
    retriever_kind: str,
    max_results: int,
    alpha: float,
    max_new_tokens: int,
    profile: str,
    sample_types: set[str],
    include_failed_targets: bool,
    correction_only: bool,
    correction_reasons: set[str],
    max_trajectories: int,
    max_rows: int,
    num_shards: int,
    shard_index: int,
    no_bf16: bool,
    max_compress_turns_per_chunk: int,
    oracle_compress_recovery: bool,
    log_every_steps: int,
) -> Dict[str, Any]:
    from scripts.eval.eval_profiles import apply_profile, describe_profile

    profile_cfg = apply_profile(profile)
    print(describe_profile(profile))

    cls, model_type = detect_model_class(ckpt)
    print(f"Loading {cls.__name__} from {ckpt}")
    model = cls.from_pretrained(
        ckpt,
        dtype=torch.bfloat16 if not no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = load_processor_for_checkpoint(ckpt)
    data_args = DataArguments()
    data_args.min_pixels = int(os.environ.get("IMAGE_MIN_PIXELS", os.environ.get("MIN_PIXELS", "130000")))
    data_args.max_pixels = int(os.environ.get("IMAGE_MAX_PIXELS", os.environ.get("MAX_PIXELS", "220000")))
    processor = update_processor_pixels(processor, data_args)
    if hasattr(processor, "video_processor") and hasattr(
        processor.video_processor, "do_sample_frames"
    ):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt,
        model_max_length=profile_cfg["model_max_length"],
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_tokens(
        [
            t for t in processor.tokenizer.get_added_vocab().keys()
            if t not in tokenizer.get_vocab()
        ],
        special_tokens=True,
    )

    print(f"Building retriever: kind={retriever_kind}, alpha={alpha}")
    retriever = make_retriever(
        kind=retriever_kind,
        alpha=alpha,
        max_results=max_results,
        device="cuda",
    )
    loop = StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
        max_new_tokens=max_new_tokens,
        retriever=retriever,
        compress_mode="system",
        frames_root=frames_root,
        video_root=video_root,
        frame_protocol=frame_protocol,
    )

    stats: Dict[str, Any] = {
        "trajectories_seen": 0,
        "trajectories_used": 0,
        "steps": 0,
        "rows": 0,
        "skipped": {},
        "by_type": {},
        "by_correction_reason": {},
        "by_selected_correction_reason": {},
        "step_errors": 0,
        "policy_compress_turns": 0,
        "visual_retries_after_compress": 0,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fout:
        for traj_i, traj in enumerate(_iter_trajectory_rows(trajectories)):
            if max_trajectories and stats["trajectories_used"] >= max_trajectories:
                break
            stats["trajectories_seen"] += 1
            if num_shards > 1 and (traj_i % num_shards) != shard_index:
                continue

            samples = _propagate_sample_fields(traj)
            if not samples:
                continue
            video_path = _resolve_video_path(samples[0].get("video_path", ""), video_root)
            if not video_path or not Path(video_path).exists():
                stats["skipped"]["missing_video"] = stats["skipped"].get("missing_video", 0) + len(samples)
                continue

            loop.reset()
            reset_visual_index(loop.retriever)
            stats["trajectories_used"] += 1

            for chunk_idx, chunk_samples in _group_by_chunk(samples).items():
                compress_samples = [
                    s for s in chunk_samples
                    if str(s.get("sample_type", "")) == "compress"
                ]
                visual_samples = [
                    s for s in chunk_samples
                    if str(s.get("sample_type", "")) != "compress"
                ]
                control = _choose_control_sample(visual_samples or chunk_samples)
                q = _new_question(control)
                q_meta = _question_meta(control) if q else None

                compress_turns = 0
                while True:
                    try:
                        # Gold compress rows are inter-chunk memory-management
                        # events, so do not inject the visual question on a
                        # compress-only chunk. For visual chunks, keep the
                        # normal question routing.
                        result = loop.step(
                            chunk_idx=chunk_idx,
                            video_path=video_path,
                            user_question=q if visual_samples else None,
                            user_question_meta=q_meta if visual_samples else None,
                        )
                        onpolicy_prompt = deepcopy(loop._last_step_messages)
                        if not onpolicy_prompt:
                            raise RuntimeError("StreamingAgentLoop did not capture step prompt")
                    except Exception as exc:
                        stats["step_errors"] += 1
                        stats["skipped"]["step_error"] = stats["skipped"].get("step_error", 0) + len(chunk_samples)
                        if stats["step_errors"] <= 5:
                            print(
                                f"[warn] step failed traj={traj_i} chunk={chunk_idx}: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                        break

                    stats["steps"] += 1
                    prompt_is_compress = _prompt_has_compress_trigger(onpolicy_prompt)

                    if prompt_is_compress:
                        stats["policy_compress_turns"] += 1
                        _record_compress_diag_stats(
                            stats, result, prefix="compress_prefix_by_turn",
                        )
                        for sample in compress_samples:
                            _emit_dagger_row(
                                sample=sample,
                                onpolicy_prompt=onpolicy_prompt,
                                result=result,
                                fout=fout,
                                stats=stats,
                                ckpt=ckpt,
                                data_dir=data_dir,
                                frame_protocol=frame_protocol,
                                render_layout=render_layout,
                                include_failed_targets=include_failed_targets,
                                sample_types=sample_types,
                                correction_only=correction_only,
                                correction_reasons=correction_reasons,
                            )
                            if max_rows and stats["rows"] >= max_rows:
                                break
                        if max_rows and stats["rows"] >= max_rows:
                            break

                        # Critical DAgger alignment with verl/eval rollout:
                        # a system compress turn is between video chunks. If
                        # the policy actually compressed memory, retry the
                        # same chunk and train the visual target on the
                        # post-compress prompt. Do not train visual targets on
                        # the compress prompt.
                        if visual_samples:
                            if result.get("action") == "compress":
                                compress_turns += 1
                                if compress_turns <= max_compress_turns_per_chunk:
                                    stats["visual_retries_after_compress"] += 1
                                    continue
                                if (
                                    oracle_compress_recovery
                                    and compress_samples
                                    and _apply_oracle_compress_recovery(
                                        loop.memory, compress_samples, stats,
                                    )
                                ):
                                    stats["visual_retries_after_oracle_compress"] = (
                                        stats.get("visual_retries_after_oracle_compress", 0) + 1
                                    )
                                    continue
                                stats["skipped"]["too_many_policy_compress_turns"] = (
                                    stats["skipped"].get("too_many_policy_compress_turns", 0)
                                    + len(visual_samples)
                                )
                            else:
                                fail_action = str(result.get("action") or "unknown")
                                stats["skipped"]["policy_failed_compress_before_visual"] = (
                                    stats["skipped"].get("policy_failed_compress_before_visual", 0)
                                    + len(visual_samples)
                                )
                                key = f"policy_failed_compress_before_visual:{fail_action}"
                                stats["skipped"][key] = (
                                    stats["skipped"].get(key, 0) + len(visual_samples)
                                )
                                if oracle_compress_recovery and compress_samples:
                                    compress_turns += 1
                                    if (
                                        compress_turns <= max_compress_turns_per_chunk
                                        and _apply_oracle_compress_recovery(
                                            loop.memory, compress_samples, stats,
                                        )
                                    ):
                                        stats["visual_retries_after_oracle_compress"] = (
                                            stats.get("visual_retries_after_oracle_compress", 0) + 1
                                        )
                                        continue
                        break

                    for sample in visual_samples:
                        _emit_dagger_row(
                            sample=sample,
                            onpolicy_prompt=onpolicy_prompt,
                            result=result,
                            fout=fout,
                            stats=stats,
                            ckpt=ckpt,
                            data_dir=data_dir,
                            frame_protocol=frame_protocol,
                            render_layout=render_layout,
                            include_failed_targets=include_failed_targets,
                            sample_types=sample_types,
                            correction_only=correction_only,
                            correction_reasons=correction_reasons,
                        )
                        if max_rows and stats["rows"] >= max_rows:
                            break
                    if compress_samples:
                        stats["skipped"]["compress_target_without_trigger"] = (
                            stats["skipped"].get("compress_target_without_trigger", 0)
                            + len(compress_samples)
                        )
                    break

                if max_rows and stats["rows"] >= max_rows:
                    break

                if log_every_steps and stats["steps"] % log_every_steps == 0:
                    rate = stats["steps"] / max(time.time() - t0, 1e-6)
                    print(
                        f"[steps={stats['steps']}] rows={stats['rows']} "
                        f"traj_used={stats['trajectories_used']} "
                        f"compress_turns={stats['policy_compress_turns']} "
                        f"rate={rate:.3f} step/s skipped={stats['skipped']}",
                        flush=True,
                    )
                    fout.flush()

            if stats["trajectories_used"] % 5 == 0:
                rate = stats["steps"] / max(time.time() - t0, 1e-6)
                print(
                    f"[{stats['trajectories_used']} traj] rows={stats['rows']} "
                    f"steps={stats['steps']} rate={rate:.3f} step/s",
                    flush=True,
                )
            if max_rows and stats["rows"] >= max_rows:
                break

    finalize_dagger_stats(stats)
    stats["out"] = str(out)
    stats["elapsed_sec"] = round(time.time() - t0, 3)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    batch_root = _default_batch_root(None)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--trajectories",
        default=str(batch_root / "final" / "train_sft_trajectories.jsonl"),
    )
    p.add_argument(
        "--out",
        default=str(batch_root / "rendered" / "video_meta" / "train_sft_dagger_messages.jsonl"),
    )
    p.add_argument("--data-dir", default=str(batch_root))
    p.add_argument("--frames-root", default=str(batch_root / "frames"))
    p.add_argument("--video-root", default=None)
    p.add_argument("--frame-protocol", default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"))
    p.add_argument("--render-layout", default=os.environ.get("THINKSTREAM_RENDER_LAYOUT", "standard"))
    p.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    p.add_argument("--max-results", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--profile", default="16k", choices=["16k", "32k"])
    p.add_argument(
        "--sample-types",
        default="silent,response,recall,compress",
        help="Comma-separated target sample_type list. Compress rows are only emitted when the on-policy prompt has <compress_trigger/>.",
    )
    p.add_argument("--include-failed-targets", action="store_true")
    p.add_argument(
        "--correction-only",
        action="store_true",
        help=(
            "Emit only on-policy states whose rollout matches selected "
            "correction reasons. This is the recommended stage-2 SFT mode."
        ),
    )
    p.add_argument(
        "--correction-reasons",
        default=",".join(sorted(DEFAULT_DAGGER_CORRECTION_REASONS)),
        help=(
            "Comma-separated correction reasons kept under --correction-only. "
            "Use 'default' for the production set or 'all' to include "
            "diagnostic warnings such as recall prompt leaks."
        ),
    )
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument(
        "--max-compress-turns-per-chunk",
        type=int,
        default=2,
        help="Retry the same visual chunk after at most this many policy compress turns.",
    )
    p.add_argument(
        "--no-oracle-compress-recovery",
        action="store_true",
        help=(
            "Disable DAgger recovery that applies the gold compress target "
            "after a missed system-compress turn before retrying the same visual chunk."
        ),
    )
    p.add_argument(
        "--log-every-steps",
        type=int,
        default=20,
        help="Print DAgger rollout progress every N policy steps (0 disables).",
    )
    args = p.parse_args()

    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = normalize_render_layout(args.render_layout)
    os.environ["THINKSTREAM_RENDER_LAYOUT"] = render_layout
    if args.out == str(batch_root / "rendered" / "video_meta" / "train_sft_dagger_messages.jsonl"):
        out_dir = (
            batch_root / "rendered" / frame_protocol
            if render_layout == "standard"
            else batch_root / "rendered" / f"{frame_protocol}_{render_layout}"
        )
        args.out = str(out_dir / "train_sft_dagger_messages.jsonl")
    sample_types = {x.strip() for x in args.sample_types.split(",") if x.strip()}
    correction_reasons = _parse_reason_set(args.correction_reasons)
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    stats = build_dagger(
        ckpt=args.ckpt,
        trajectories=_resolve_path(args.trajectories),
        out=_resolve_path(args.out),
        data_dir=_resolve_path(args.data_dir),
        frames_root=str(_resolve_path(args.frames_root)),
        video_root=str(_resolve_path(args.video_root)) if args.video_root else None,
        frame_protocol=frame_protocol,
        render_layout=render_layout,
        retriever_kind=args.retriever,
        max_results=args.max_results,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        profile=args.profile,
        sample_types=sample_types,
        include_failed_targets=args.include_failed_targets,
        correction_only=args.correction_only,
        correction_reasons=correction_reasons,
        max_trajectories=args.max_trajectories,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        no_bf16=args.no_bf16,
        max_compress_turns_per_chunk=args.max_compress_turns_per_chunk,
        oracle_compress_recovery=not args.no_oracle_compress_recovery,
        log_every_steps=args.log_every_steps,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
