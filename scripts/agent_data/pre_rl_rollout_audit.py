#!/usr/bin/env python
"""Fast pre-RL student rollout audit.

Runs the current SFT/student checkpoint on full-video trajectory rows with the
same batch-vLLM, video_meta, query-last rollout path used by DAgger/RL. It does
not create training samples or run PPO update; it only reports whether the SFT
policy is good enough to start RL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkstream.data.agent_protocol import (  # noqa: E402
    AGENT_CHUNK_SEC,
    parse_agent_output,
    query_expected_answer_chunks,
)
from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS  # noqa: E402
from thinkstream.trainer.rewards import compute_answer_decision_reward  # noqa: E402
from thinkstream.trainer.outcome_match import score_outcome_by_form  # noqa: E402


def _read_jsonl(path: Path, limit: int = 0) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _plain(value: Any) -> Any:
    """Convert pandas/pyarrow numpy wrappers to plain Python containers."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _load_parquet_trajectory_rows(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    """Load the same multi-Q parquet rows used by RL training.

    ``build_verl_parquet --multi_q`` stores the training annotation in
    ``extra_info.questions`` plus ``reward_model.ground_truth``. The rollout
    audit runs on trajectory-shaped records, so normalize the parquet row back
    to that shared shape instead of re-reading final/*.jsonl.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("pandas is required to read parquet audit input") from exc

    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    rows: List[Dict[str, Any]] = []
    for raw in df.to_dict("records"):
        row = _plain(raw)
        extra = row.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            reward_model = {}
        gt_raw = reward_model.get("ground_truth")
        gt: Dict[str, Any] = {}
        if isinstance(gt_raw, str) and gt_raw.strip():
            try:
                loaded = json.loads(gt_raw)
                if isinstance(loaded, dict):
                    gt = _plain(loaded)
            except json.JSONDecodeError:
                gt = {}
        elif isinstance(gt_raw, dict):
            gt = _plain(gt_raw)

        questions = extra.get("questions") or gt.get("questions") or []
        if not isinstance(questions, list):
            questions = []
        gold_action = (
            extra.get("gold_action_per_chunk")
            or gt.get("gold_action_per_chunk")
            or row.get("gold_action_per_chunk")
            or {}
        )
        if not isinstance(gold_action, dict):
            gold_action = {}
        offline_compress = (
            extra.get("offline_compress_chunks")
            or gt.get("offline_compress_chunks")
            or row.get("offline_compress_chunks")
            or []
        )
        if not isinstance(offline_compress, list):
            offline_compress = []

        try:
            n_chunks = int(row.get("n_chunks") or extra.get("n_chunks") or 0)
        except (TypeError, ValueError):
            n_chunks = 0
        rows.append({
            "video_id": str(row.get("video_id") or extra.get("video_id") or ""),
            "trajectory_id": str(row.get("video_id") or extra.get("index") or ""),
            "video_path": str(row.get("video_path") or extra.get("video_path") or ""),
            "questions": questions,
            "gold_action_per_chunk": {
                str(k): str(v) for k, v in gold_action.items()
            },
            "offline_compress_chunks": [
                int(x) for x in offline_compress
                if isinstance(x, (int, float)) or str(x).lstrip("-").isdigit()
            ],
            "stats": {
                "n_chunks_covered": n_chunks,
                "chunk_idx_max": max(0, n_chunks - 1),
            },
            "extra_info": extra,
            "reward_model": reward_model,
            "_source_parquet": str(path),
        })
    return rows


def _load_source_rows(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return _load_parquet_trajectory_rows(path, limit=limit)
    return list(_read_jsonl(path, limit))


def _infer_scheme_root(source: Path) -> Path:
    if source.suffix.lower() == ".parquet":
        # data/.../rendered/<layout>/train_rl_multi_q.parquet -> data/...
        if source.parent.parent.name == "rendered":
            return source.parent.parent.parent
        if source.parent.name == "final":
            return source.parent.parent
        return source.parent
    return source.parent.parent if source.parent.name == "final" else source.parent


def _trajectory_max_chunk(row: Dict[str, Any]) -> int:
    """Best-effort final chunk index for jsonl and parquet trajectory rows."""
    candidates: List[int] = []
    stats = row.get("stats") or {}
    if isinstance(stats, dict):
        for key in ("chunk_idx_max", "max_chunk_idx"):
            val = stats.get(key)
            if str(val).lstrip("-").isdigit():
                candidates.append(int(val))
        n_chunks = stats.get("n_chunks_covered") or stats.get("n_chunks")
        if str(n_chunks).lstrip("-").isdigit():
            candidates.append(max(0, int(n_chunks) - 1))
    n_chunks_row = row.get("n_chunks")
    if str(n_chunks_row).lstrip("-").isdigit():
        candidates.append(max(0, int(n_chunks_row) - 1))
    for s in row.get("samples") or []:
        if isinstance(s, dict) and str(s.get("chunk_idx")).lstrip("-").isdigit():
            candidates.append(int(s["chunk_idx"]))
    for q in row.get("questions") or []:
        if not isinstance(q, dict):
            continue
        for key in ("ask_chunks", "answer_chunks", "support_chunks"):
            for ck in q.get(key) or []:
                if str(ck).lstrip("-").isdigit():
                    candidates.append(int(ck))
    for ck in row.get("offline_compress_chunks") or []:
        if str(ck).lstrip("-").isdigit():
            candidates.append(int(ck))
    return max(candidates, default=0)


def _select_shard(
    rows: List[Dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def _trajectory_for_rollout(traj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(traj)
    vid = str(out.get("video_id") or out.get("trajectory_id") or "")
    out["_original_video_path"] = out.get("video_path", "")
    out["video_path"] = f"{vid}.mp4" if vid else str(out.get("video_path", ""))
    out["data_path"] = ""
    return out


def _parse_agent_output(text: str) -> Dict[str, Any]:
    parsed = parse_agent_output(text or "")
    kind = parsed.get("kind") or "unknown"
    common = {
        "think": parsed.get("think", ""),
        "format_error": parsed.get("format_error"),
    }
    payload: Dict[str, Any] = {}
    if kind == "answer":
        response = str(parsed.get("answer") or "")
        return {
            "action": "response" if response.strip() else "silent",
            "payload": {"response": response},
            **common,
        }
    if kind in {"recall", "compress"}:
        tool_call = parsed.get("tool_call") or {}
        args = tool_call.get("arguments") or {}
        if kind == "recall":
            payload = {"query": args}
        else:
            payload = {"summary": args}
        return {"action": kind, "payload": payload, **common}
    return {"action": kind, "payload": {}, **common}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_bad(
    buckets: Dict[str, List[Dict[str, Any]]],
    kind: str,
    case: Dict[str, Any],
    *,
    per_kind_limit: int,
) -> None:
    if len(buckets[kind]) < per_kind_limit:
        buckets[kind].append({**case, "badcase_kind": kind})


def _strip_option_label(text: str) -> str:
    import re

    m = re.match(r"^\s*([A-Z])(\)|\.|:)\s*(.*)$", str(text or "").strip())
    return (m.group(3) if m else str(text or "")).strip()


def _mc_correct_text(q: Dict[str, Any]) -> str:
    opts = list(q.get("options") or [])
    correct = str(q.get("correct_option") or "").strip().upper()
    if len(correct) == 1 and "A" <= correct <= "Z":
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(opts):
            return _strip_option_label(opts[idx])
    return str(
        q.get("correct_answer_text")
        or q.get("canonical_answer")
        or q.get("gold_answer")
        or ""
    ).strip()


def _gold_for_expected_chunk(q: Dict[str, Any], expected_chunk: Optional[int]) -> str:
    if expected_chunk is not None:
        for item in q.get("per_emit_answers") or []:
            if not isinstance(item, dict):
                continue
            try:
                if int(item.get("chunk")) == int(expected_chunk):
                    value = str(item.get("value") or "").strip()
                    if value:
                        return value
            except (TypeError, ValueError):
                continue
    if str(q.get("answer_form") or "") == "multiple_choice":
        return _mc_correct_text(q)
    return str(
        q.get("gold_answer")
        or q.get("canonical_answer")
        or q.get("correct_answer_text")
        or ""
    ).strip()


def _event_chunk(answer_event: Dict[str, Any]) -> Optional[int]:
    try:
        chunk = int(answer_event.get("chunk", -1))
    except (TypeError, ValueError):
        return None
    return chunk if chunk >= 0 else None


def _decision_bucket(
    *,
    decision: float,
    timing: str,
    has_answer: bool,
    expected_chunk: Optional[int],
) -> str:
    if has_answer:
        if timing in {"early", "late", "on_time"}:
            return timing
        if expected_chunk is None:
            return "false_positive_extra"
        return "answered"
    if expected_chunk is None:
        return "normal_silent"
    if decision <= -1.0:
        return "missed"
    return "silent"


def _record_answer_decision(
    stats: Counter,
    nested: Dict[str, Counter],
    *,
    decision: float,
    bucket: str,
    correct: bool,
) -> None:
    stats["answer_decision_slots"] += 1
    stats["answer_decision_sum"] += float(decision)
    gated = float(decision) if decision <= 0.0 or correct else 0.0
    stats["answer_decision_gated_sum"] += gated
    nested["answer_decision"][bucket] += 1


def _question_key(q: Dict[str, Any]) -> Tuple[str, int]:
    question = str(q.get("question") or "").strip()
    ask_chunks = q.get("ask_chunks") or []
    try:
        ask = min(int(x) for x in ask_chunks) if ask_chunks else int(q.get("ask_chunk", -1))
    except (TypeError, ValueError):
        ask = -1
    return question, ask


def _question_deadline(q: Dict[str, Any]) -> int:
    chunks = []
    for x in q.get("answer_chunks") or []:
        try:
            chunks.append(int(x))
        except (TypeError, ValueError):
            continue
    if chunks:
        return max(chunks)
    _question, ask = _question_key(q)
    return ask


def _question_answer_chunks(q: Dict[str, Any]) -> List[int]:
    chunks = []
    for x in q.get("answer_chunks") or []:
        try:
            chunks.append(int(x))
        except (TypeError, ValueError):
            continue
    if chunks:
        return sorted(set(chunks))
    _question, ask = _question_key(q)
    return [ask] if ask >= 0 else []


def _question_support_chunks(q: Dict[str, Any]) -> List[int]:
    chunks = []
    for x in q.get("support_chunks") or []:
        try:
            chunks.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(x for x in chunks if x >= 0))


def _question_wait_bucket(q: Dict[str, Any]) -> str:
    if str(q.get("question_type") or "") == "multi_emit":
        return "multi_emit"
    _question, ask = _question_key(q)
    answers = _question_answer_chunks(q)
    if ask < 0 or not answers:
        return "unknown"
    wait = min(answers) - ask
    if wait <= 0:
        return "immediate"
    if wait <= 5:
        return "wait_1_5"
    if wait <= 30:
        return "wait_6_30"
    return "wait_gt_30"


def _question_evidence_bucket(q: Dict[str, Any]) -> str:
    _question, ask = _question_key(q)
    support = _question_support_chunks(q)
    if ask < 0 or not support:
        return "unknown"
    if max(support) < ask:
        return "history_only"
    if min(support) < ask <= max(support):
        return "history_to_current"
    return "current_or_future"


def _record_question_mix(
    questions: List[Dict[str, Any]],
    nested: Dict[str, Counter],
    *,
    scale: int,
) -> None:
    for q in questions:
        answer_form = str(q.get("answer_form") or "unknown")
        family = str(q.get("family") or "unknown")
        qtype = str(q.get("question_type") or "unknown")
        availability = str(q.get("availability") or "unknown")
        options = list(q.get("options") or [])
        nested["question_family"][family] += scale
        nested["question_answer_form"][answer_form] += scale
        nested["question_type"][qtype] += scale
        nested["question_availability"][availability] += scale
        nested["question_wait_bucket"][_question_wait_bucket(q)] += scale
        nested["question_evidence_bucket"][_question_evidence_bucket(q)] += scale
        if answer_form == "multiple_choice":
            nested["question_mc_option_count"][str(len(options))] += scale
            nested["question_mc_correct_option"][
                str(q.get("correct_option") or "unknown")
            ] += scale


def _recall_relation_to_questions(chunk: int, questions: List[Dict[str, Any]]) -> str:
    asks = []
    for q in questions:
        _question, ask = _question_key(q)
        if ask >= 0:
            asks.append(ask)
    if not asks:
        return "no_question"
    if chunk < min(asks):
        return "before_first_question"
    for q in questions:
        _question, ask = _question_key(q)
        if ask >= 0 and ask <= chunk <= _question_deadline(q):
            return "pending_after_question"
    if any(a > chunk for a in asks):
        return "between_questions"
    return "after_all_questions"


def _match_query_state(source_q: Dict[str, Any], query_states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    question, ask = _question_key(source_q)
    candidates = [
        q for q in query_states
        if str(q.get("question") or "").strip() == question
    ]
    if not candidates:
        return None
    if ask < 0:
        return candidates[-1]
    best = None
    best_delta = 10**9
    for q in candidates:
        try:
            q_ask = int(round(float(q.get("ask_time", -1)) / max(float(AGENT_CHUNK_SEC), 1.0)))
        except (TypeError, ValueError):
            q_ask = -1
        delta = abs(q_ask - ask) if q_ask >= 0 else 10**8
        if delta <= best_delta:
            best = q
            best_delta = delta
    return best


def _last_queries_for_gen(chunk_results: List[Dict[str, Any]], gen_idx: int) -> List[Dict[str, Any]]:
    for cr in reversed(chunk_results or []):
        q_after = cr.get("queries_after") or []
        if gen_idx < len(q_after) and q_after[gen_idx]:
            return q_after[gen_idx]
    return []


def _extract_answer(raw_output: str) -> str:
    parsed = _parse_agent_output(raw_output or "")
    if parsed.get("action") != "response":
        return ""
    return str((parsed.get("payload") or {}).get("response") or "").strip()


def _support_hit(returned_chunks: List[int], support_chunks: List[int]) -> bool:
    if not returned_chunks or not support_chunks:
        return False
    return bool(set(int(x) for x in returned_chunks) & set(int(x) for x in support_chunks))


def _count_stable_think(thinks: List[Tuple[int, str]], threshold: float) -> Tuple[int, List[Dict[str, Any]]]:
    bad: List[Dict[str, Any]] = []
    count = 0
    prev_chunk = None
    prev = ""
    for chunk, think in thinks:
        text = " ".join(str(think or "").split())
        if not text:
            continue
        if prev:
            ratio = SequenceMatcher(None, prev.lower(), text.lower()).ratio()
            if ratio >= threshold:
                count += 1
                if len(bad) < 3:
                    bad.append({
                        "prev_chunk": prev_chunk,
                        "chunk": chunk,
                        "similarity": ratio,
                        "prev_think": prev[:500],
                        "think": text[:500],
                    })
        prev_chunk = chunk
        prev = text
    return count, bad


def _summarize_rollout(
    *,
    source_traj: Dict[str, Any],
    rollout_result: Dict[str, Any],
    stats: Counter,
    nested: Dict[str, Counter],
    badcases: Dict[str, List[Dict[str, Any]]],
    memory_merge_events: List[Dict[str, Any]],
    per_kind_limit: int,
    stable_threshold: float,
) -> None:
    video_id = str(source_traj.get("video_id") or source_traj.get("trajectory_id") or "")
    questions = [q for q in source_traj.get("questions") or [] if isinstance(q, dict)]
    q_by_text = {str(q.get("question") or "").strip(): q for q in questions}
    gold_action = {
        int(k): str(v)
        for k, v in (source_traj.get("gold_action_per_chunk") or {}).items()
        if str(k).lstrip("-").isdigit()
    }
    chunk_results = rollout_result.get("chunk_results") or []
    group_size = max((len(cr.get("actions") or []) for cr in chunk_results), default=0)

    stats["videos"] += 1
    stats["questions"] += len(questions) * max(group_size, 1)
    stats["rollout_groups"] += max(group_size, 1)
    _record_question_mix(questions, nested, scale=max(group_size, 1))
    stats["offline_gold_compress_chunks"] += (
        (
            len(source_traj.get("offline_compress_chunks") or [])
            or sum(1 for v in gold_action.values() if str(v) == "compress")
        )
        * max(group_size, 1)
    )

    for gen_idx in range(max(group_size, 1)):
        final_queries = _last_queries_for_gen(chunk_results, gen_idx)
        thinks: List[Tuple[int, str]] = []
        recall_by_question: Counter = Counter()

        for cr in chunk_results:
            actions = cr.get("actions") or []
            first_actions = cr.get("first_actions") or []
            raw_outputs = cr.get("raw_outputs") or []
            returned = cr.get("recall_returned_chunks") or []
            turn_kinds = cr.get("turn_kinds") or []
            trigger_diags = cr.get("compress_trigger_diagnostics") or []
            prefix_diags = cr.get("compress_prefix_diagnostics") or []
            action_errors = cr.get("action_space_errors") or []
            chunk_indices = cr.get("chunk_indices") or []
            merge_events = cr.get("memory_merge_events") or []

            if gen_idx >= len(actions):
                continue
            action = str(actions[gen_idx] or "")
            first_action = str(first_actions[gen_idx] or action)
            raw = str(raw_outputs[gen_idx] or "")
            chunk = int(chunk_indices[gen_idx]) if gen_idx < len(chunk_indices) else int(cr.get("chunk_idx", -1))
            if chunk < 0:
                continue

            stats["steps"] += 1
            nested["actions"][action] += 1
            nested["first_actions"][first_action] += 1
            merge_event = (
                merge_events[gen_idx]
                if gen_idx < len(merge_events) and isinstance(merge_events[gen_idx], dict)
                else None
            )
            if merge_event:
                event_row = {
                    "video_id": video_id,
                    "gen_idx": gen_idx,
                    "chunk": chunk,
                    "action": action,
                    **merge_event,
                }
                memory_merge_events.append(event_row)
                stats["memory_merge_events"] += 1
                nested["memory_merge_reason"][str(merge_event.get("reason") or "unknown")] += 1
                duration = int(merge_event.get("merged_duration_sec_after") or 0)
                count_after = int(merge_event.get("merged_chunk_count_after") or 0)
                stats["memory_merge_duration_sec_sum"] += duration
                stats["memory_merge_chunk_count_after_sum"] += count_after
                stats["memory_merge_max_duration_sec"] = max(
                    int(stats["memory_merge_max_duration_sec"]),
                    duration,
                )
                nested["memory_merge_duration_sec_after"][str(duration)] += 1
                _append_bad(
                    badcases,
                    "memory_merge",
                    event_row,
                    per_kind_limit=per_kind_limit,
                )
            if gen_idx < len(action_errors) and action_errors[gen_idx]:
                stats["action_space_errors"] += 1
                _append_bad(
                    badcases,
                    "invalid_action",
                    {
                        "video_id": video_id,
                        "gen_idx": gen_idx,
                        "chunk": chunk,
                        "action": action,
                        "error": action_errors[gen_idx],
                        "raw_output": raw[:1000],
                    },
                    per_kind_limit=per_kind_limit,
                )

            parsed = _parse_agent_output(raw)
            if parsed.get("format_error"):
                stats["format_errors"] += 1
                if "json" in str(parsed.get("format_error") or "").lower():
                    stats["json_parse_errors"] += 1
            think = str(parsed.get("think") or "")
            if think:
                thinks.append((chunk, think))

            turn_kind = str(turn_kinds[gen_idx] or "")
            trigger_diag = (
                trigger_diags[gen_idx]
                if gen_idx < len(trigger_diags) and isinstance(trigger_diags[gen_idx], dict)
                else {}
            )
            if turn_kind == "compress" or trigger_diag.get("triggered"):
                stats["system_compress_required"] += 1
                if action == "compress":
                    stats["system_compress_success"] += 1
                else:
                    stats["system_compress_failure"] += 1
                    prefix = (
                        prefix_diags[gen_idx]
                        if gen_idx < len(prefix_diags) and isinstance(prefix_diags[gen_idx], dict)
                        else {}
                    )
                    nested["compress_failure_reason"][
                        str(prefix.get("label") or action or "unknown")
                    ] += 1
                    _append_bad(
                        badcases,
                        "compress_failure",
                        {
                            "video_id": video_id,
                            "gen_idx": gen_idx,
                            "chunk": chunk,
                            "action": action,
                            "trigger": trigger_diag,
                            "prefix_diag": prefix,
                            "raw_output": raw[:1000],
                        },
                        per_kind_limit=per_kind_limit,
                    )
                continue

            expected_action = gold_action.get(chunk, "")
            if expected_action == "compress":
                # Offline pass2/pass3 compress labels are diagnostics only.
                # Runtime compression should already have been counted above
                # through the system trigger diagnostic; do not include these
                # labels in action-gold accuracy.
                stats["offline_gold_compress_action_rows_skipped"] += 1
            elif expected_action:
                actual_for_gold = "recall" if first_action == "recall" else action
                stats["action_gold_total"] += 1
                nested["action_gold_total_by_type"][expected_action] += 1
                if actual_for_gold == expected_action:
                    stats["action_gold_correct"] += 1
                    nested["action_gold_correct_by_type"][expected_action] += 1
                else:
                    nested["action_gold_mismatch"][f"{expected_action}->{actual_for_gold}"] += 1
                    _append_bad(
                        badcases,
                        "action_mismatch",
                        {
                            "video_id": video_id,
                            "gen_idx": gen_idx,
                            "chunk": chunk,
                            "gold_action": expected_action,
                            "student_action": actual_for_gold,
                            "raw_output": raw[:1000],
                        },
                        per_kind_limit=per_kind_limit,
                    )

            if first_action == "recall":
                stats["recall_events"] += 1
                nested["recall_relation"][
                    _recall_relation_to_questions(chunk, questions)
                ] += 1
                q_before = cr.get("queries_before") or []
                active_question = ""
                if gen_idx < len(q_before) and q_before[gen_idx]:
                    active = q_before[gen_idx][-1]
                    active_question = str(active.get("question") or "")
                source_q = q_by_text.get(active_question)
                ret = returned[gen_idx] if gen_idx < len(returned) else []
                if ret:
                    stats["recall_returned_nonempty"] += 1
                if source_q:
                    recall_by_question[str(source_q.get("card_id") or active_question)] += 1
                    support = [int(x) for x in source_q.get("support_chunks") or [] if str(x).lstrip("-").isdigit()]
                    if _support_hit(ret, support):
                        stats["recall_support_hits"] += 1
                    else:
                        _append_bad(
                            badcases,
                            "recall_support_miss",
                            {
                                "video_id": video_id,
                                "gen_idx": gen_idx,
                                "chunk": chunk,
                                "card_id": source_q.get("card_id", ""),
                                "question": active_question,
                                "returned_chunks": ret,
                                "support_chunks": support,
                                "first_pass": (cr.get("recall_first_pass_text") or [""])[gen_idx]
                                if gen_idx < len(cr.get("recall_first_pass_text") or [])
                                else raw[:1000],
                            },
                            per_kind_limit=per_kind_limit,
                        )
                elif active_question:
                    _append_bad(
                        badcases,
                        "recall_unknown_query",
                        {
                            "video_id": video_id,
                            "gen_idx": gen_idx,
                            "chunk": chunk,
                            "question": active_question,
                            "returned_chunks": ret,
                        },
                        per_kind_limit=per_kind_limit,
                    )

        stable_count, stable_examples = _count_stable_think(thinks, stable_threshold)
        stats["stable_think_pairs"] += stable_count
        stats["think_pairs"] += max(0, len(thinks) - 1)
        for item in stable_examples:
            _append_bad(
                badcases,
                "stable_think",
                {"video_id": video_id, "gen_idx": gen_idx, **item},
                per_kind_limit=per_kind_limit,
            )

        for source_q in questions:
            nested_key = str(source_q.get("family") or source_q.get("question_type") or "unknown")
            form_key = str(source_q.get("answer_form") or "unknown")
            q_state = _match_query_state(source_q, final_queries)
            expected_chunks = query_expected_answer_chunks(source_q)
            if not expected_chunks:
                expected_chunks = [None]  # type: ignore[list-item]
            answers = q_state.get("answers", []) if q_state else []
            if not q_state:
                stats["question_missing_state"] += 1

            for expected_chunk in expected_chunks:
                stats["answer_slots"] += 1
                nested["answer_slots_by_family"][nested_key] += 1
                nested["answer_slots_by_form"][form_key] += 1
                gold = _gold_for_expected_chunk(source_q, expected_chunk)
                matching = [
                    a for a in answers
                    if isinstance(a, dict)
                    and (
                        expected_chunk is None
                        or str(a.get("expected_chunk")) == str(expected_chunk)
                    )
                ]
                completion = [
                    a for a in matching
                    if a.get("counts_for_completion") is not False
                ]
                early = [
                    a for a in matching
                    if str(a.get("timing") or "") == "early"
                ]
                picked = completion[0] if completion else None
                early_correct = any(
                    score_outcome_by_form(
                        str(a.get("text") or ""),
                        options=list(source_q.get("options") or []),
                        correct_option=source_q.get("correct_option", ""),
                        gold_answer=gold,
                        answer_form=str(source_q.get("answer_form") or ""),
                    ) >= 1.0
                    for a in early
                )
                if early:
                    stats["early_answer_slots"] += 1
                    if early_correct:
                        stats["early_correct_slots"] += 1

                if not picked:
                    early_chunk = _event_chunk(early[0]) if early else None
                    exp_chunk = int(expected_chunk) if expected_chunk is not None else None
                    if early_chunk is not None:
                        decision = compute_answer_decision_reward(
                            early_chunk,
                            exp_chunk,
                            exp_chunk,
                            late_window_chunks=2,
                            has_answer=True,
                        )
                        bucket = "early"
                    else:
                        decision = compute_answer_decision_reward(
                            None,
                            exp_chunk,
                            exp_chunk,
                            late_window_chunks=2,
                            has_answer=False,
                        )
                        bucket = _decision_bucket(
                            decision=decision,
                            timing="",
                            has_answer=False,
                            expected_chunk=expected_chunk,
                        )
                    _record_answer_decision(
                        stats,
                        nested,
                        decision=decision,
                        bucket=bucket,
                        correct=False,
                    )
                    stats["missing_answer_slots"] += 1
                    _append_bad(
                        badcases,
                        "missing_answer",
                        {
                            "video_id": video_id,
                            "gen_idx": gen_idx,
                            "card_id": source_q.get("card_id", ""),
                            "question": source_q.get("question", ""),
                            "expected_chunk": expected_chunk,
                            "gold": gold,
                            "answers": answers,
                        },
                        per_kind_limit=per_kind_limit,
                    )
                    continue

                timing = str(picked.get("timing") or "unknown")
                nested["timing"][timing] += 1
                if timing == "early":
                    stats["completion_early_slots"] += 1
                elif timing == "late":
                    stats["late_answer_slots"] += 1
                elif timing == "on_time":
                    stats["on_time_answer_slots"] += 1

                pred = str(picked.get("text") or "")
                score = score_outcome_by_form(
                    pred,
                    options=list(source_q.get("options") or []),
                    correct_option=source_q.get("correct_option", ""),
                    gold_answer=gold,
                    answer_form=str(source_q.get("answer_form") or ""),
                )
                exp_chunk = int(expected_chunk) if expected_chunk is not None else None
                picked_chunk = _event_chunk(picked)
                decision = compute_answer_decision_reward(
                    picked_chunk,
                    exp_chunk,
                    exp_chunk,
                    late_window_chunks=2,
                    has_answer=True,
                )
                _record_answer_decision(
                    stats,
                    nested,
                    decision=decision,
                    bucket=_decision_bucket(
                        decision=decision,
                        timing=timing,
                        has_answer=True,
                        expected_chunk=expected_chunk,
                    ),
                    correct=score >= 1.0,
                )
                if score >= 1.0:
                    stats["answer_correct_slots"] += 1
                    nested["answer_correct_by_family"][nested_key] += 1
                    nested["answer_correct_by_form"][form_key] += 1
                    if timing == "on_time":
                        stats["on_time_correct_slots"] += 1
                else:
                    stats["answer_wrong_slots"] += 1
                    _append_bad(
                        badcases,
                        "wrong_answer",
                        {
                            "video_id": video_id,
                            "gen_idx": gen_idx,
                            "card_id": source_q.get("card_id", ""),
                            "family": source_q.get("family", ""),
                            "answer_form": source_q.get("answer_form", ""),
                            "question": source_q.get("question", ""),
                            "expected_chunk": expected_chunk,
                            "timing": timing,
                            "prediction": pred,
                            "gold": gold,
                            "all_answers": answers,
                        },
                        per_kind_limit=per_kind_limit,
                    )

        for qid, n_recall in recall_by_question.items():
            nested["recall_per_question"][str(n_recall)] += 1


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _counter_dict(c: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in c.most_common()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        required=True,
        help=(
            "RL source trajectories jsonl or rendered train/val/test "
            "multi-Q parquet used by training"
        ),
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="summary json output path")
    p.add_argument("--badcase-out", default="")
    p.add_argument("--memory-merge-events-out", default="")
    p.add_argument("--limit-videos", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--rollout-batch-size", type=int, default=16)
    p.add_argument("--rollouts-per-video", type=int, default=1)
    p.add_argument("--max-chunks", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--compress-max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--tensor-parallel-size", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--vllm-max-images-per-prompt", type=int, default=64)
    p.add_argument("--vllm-max-videos-per-prompt", type=int, default=2)
    p.add_argument("--vllm-mm-processor-cache-gb", type=int, default=256)
    p.add_argument("--min-pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    p.add_argument("--max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    p.add_argument("--frames-root", default="")
    p.add_argument("--disable-recall", action="store_true")
    p.add_argument("--badcases-per-kind", type=int, default=50)
    p.add_argument("--stable-think-threshold", type=float, default=0.92)
    p.add_argument(
        "--memory-merge-similar-thinks",
        action="store_true",
        help=(
            "Rollout ablation: if a new think is very similar to the last "
            "visible memory_think, merge it into that record's time_range "
            "while preserving raw per-chunk archive entries for recall."
        ),
    )
    p.add_argument("--memory-merge-threshold", type=float, default=0.96)
    args = p.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    badcase_out = Path(args.badcase_out) if args.badcase_out else out.with_suffix(".badcases.jsonl")
    memory_merge_events_out = (
        Path(args.memory_merge_events_out)
        if args.memory_merge_events_out
        else out.with_suffix(".memory_merge_events.jsonl")
    )
    scheme_root = _infer_scheme_root(source)
    frames_root = Path(args.frames_root) if args.frames_root else scheme_root / "frames"

    all_rows = _load_source_rows(source, args.limit_videos)
    trajectories = _select_shard(all_rows, num_shards=args.num_shards, shard_index=args.shard_index)
    if not trajectories:
        raise SystemExit(f"no trajectories loaded from {source}")

    print(
        f"pre-RL rollout audit: videos={len(trajectories)}/{len(all_rows)} "
        f"batch={args.rollout_batch_size} groups={args.rollouts_per_video} "
        f"frames_root={frames_root}"
    )
    from scripts.eval.processor_loader import load_processor_for_checkpoint
    from thinkstream.eval.streaming_vllm import streaming_vllm_rollout
    from thinkstream.eval.vllm_engine import init_vllm_engine
    from thinkstream.sft.args import DataArguments
    from thinkstream.sft.data_processor import update_processor_pixels

    processor = load_processor_for_checkpoint(args.ckpt, trust_remote_code=True)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    llm = init_vllm_engine(
        args.ckpt,
        tensor_parallel_size=(args.tensor_parallel_size or None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_images_per_prompt=args.vllm_max_images_per_prompt,
        max_videos_per_prompt=args.vllm_max_videos_per_prompt,
        mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
        enable_prefix_caching=True,
    )

    stats: Counter = Counter()
    nested: Dict[str, Counter] = defaultdict(Counter)
    badcases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    memory_merge_events: List[Dict[str, Any]] = []
    t0 = time.time()

    for start in range(0, len(trajectories), args.rollout_batch_size):
        batch = trajectories[start:start + args.rollout_batch_size]
        max_batch_chunk = max((_trajectory_max_chunk(t) for t in batch), default=0)
        rollout_max_chunks = min(args.max_chunks, max_batch_chunk + 1)
        print(
            f"[{start}:{start + len(batch)}] B={len(batch)} "
            f"S={args.rollouts_per_video} max_chunks={rollout_max_chunks}"
        )
        rollout_results = streaming_vllm_rollout(
            [_trajectory_for_rollout(t) for t in batch],
            llm,
            processor,
            processor.tokenizer,
            group_size=args.rollouts_per_video,
            max_new_tokens=args.max_new_tokens,
            compress_max_new_tokens=args.compress_max_new_tokens,
            rollout_max_chunks=rollout_max_chunks,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            frames_root=str(frames_root),
            video_root=None,
            enable_recall=not args.disable_recall,
            frame_protocol="video_meta",
            render_layout="standard_query_last",
            memory_merge_similar_thinks=args.memory_merge_similar_thinks,
            memory_merge_similarity_threshold=args.memory_merge_threshold,
        )
        for source_traj, result in zip(batch, rollout_results):
            _summarize_rollout(
                source_traj=source_traj,
                rollout_result=result,
                stats=stats,
                nested=nested,
                badcases=badcases,
                memory_merge_events=memory_merge_events,
                per_kind_limit=args.badcases_per_kind,
                stable_threshold=args.stable_think_threshold,
            )

    elapsed = time.time() - t0
    summary = {
        "config": {
            "source": str(source),
            "ckpt": args.ckpt,
            "frames_root": str(frames_root),
            "rollout_batch_size": args.rollout_batch_size,
            "rollouts_per_video": args.rollouts_per_video,
            "frame_protocol": "video_meta",
            "render_layout": "standard_query_last",
            "mm_processor_cache_gb": args.vllm_mm_processor_cache_gb,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "recall_enabled": not args.disable_recall,
            "memory_merge_similar_thinks": bool(args.memory_merge_similar_thinks),
            "memory_merge_threshold": args.memory_merge_threshold,
        },
        "speed": {
            "elapsed_sec": elapsed,
            "videos_per_sec": float(stats["videos"]) / elapsed if elapsed > 0 else 0.0,
            "steps_per_sec": float(stats["steps"]) / elapsed if elapsed > 0 else 0.0,
            "steps": int(stats["steps"]),
        },
        "answer": {
            "slots": int(stats["answer_slots"]),
            "acc_completion": _rate(stats["answer_correct_slots"], stats["answer_slots"]),
            "wrong_rate": _rate(stats["answer_wrong_slots"], stats["answer_slots"]),
            "missing_rate": _rate(stats["missing_answer_slots"], stats["answer_slots"]),
            "on_time_rate": _rate(stats["on_time_answer_slots"], stats["answer_slots"]),
            "late_rate": _rate(stats["late_answer_slots"], stats["answer_slots"]),
            "early_slot_rate": _rate(stats["early_answer_slots"], stats["answer_slots"]),
            "early_correct_rate": _rate(stats["early_correct_slots"], stats["answer_slots"]),
            "on_time_correct_rate": _rate(stats["on_time_correct_slots"], stats["answer_slots"]),
            "timing": _counter_dict(nested["timing"]),
            "answer_decision_avg": _rate(
                stats["answer_decision_sum"],
                stats["answer_decision_slots"],
            ),
            "answer_decision_gated_avg": _rate(
                stats["answer_decision_gated_sum"],
                stats["answer_decision_slots"],
            ),
            "answer_decision": _counter_dict(nested["answer_decision"]),
        },
        "action": {
            "gold_total": int(stats["action_gold_total"]),
            "gold_acc": _rate(stats["action_gold_correct"], stats["action_gold_total"]),
            "actions": _counter_dict(nested["actions"]),
            "first_actions": _counter_dict(nested["first_actions"]),
            "by_gold_total": _counter_dict(nested["action_gold_total_by_type"]),
            "by_gold_correct": _counter_dict(nested["action_gold_correct_by_type"]),
            "mismatches": _counter_dict(nested["action_gold_mismatch"]),
            "action_space_errors": int(stats["action_space_errors"]),
        },
        "recall": {
            "events": int(stats["recall_events"]),
            "events_per_question": _rate(stats["recall_events"], stats["questions"]),
            "return_nonempty_rate": _rate(
                stats["recall_returned_nonempty"], stats["recall_events"]
            ),
            "support_hit_rate": _rate(stats["recall_support_hits"], stats["recall_events"]),
            "relation": _counter_dict(nested["recall_relation"]),
            "per_question_hist": _counter_dict(nested["recall_per_question"]),
        },
        "question_mix": {
            "families": _counter_dict(nested["question_family"]),
            "answer_forms": _counter_dict(nested["question_answer_form"]),
            "question_types": _counter_dict(nested["question_type"]),
            "availability": _counter_dict(nested["question_availability"]),
            "wait_buckets": _counter_dict(nested["question_wait_bucket"]),
            "evidence_buckets": _counter_dict(nested["question_evidence_bucket"]),
            "mc_option_counts": _counter_dict(nested["question_mc_option_count"]),
            "mc_correct_options": _counter_dict(nested["question_mc_correct_option"]),
        },
        "format_runtime": {
            "format_errors": int(stats["format_errors"]),
            "json_parse_errors": int(stats["json_parse_errors"]),
            "action_space_errors": int(stats["action_space_errors"]),
        },
        "compression": {
            "system_required": int(stats["system_compress_required"]),
            "success_rate": _rate(stats["system_compress_success"], stats["system_compress_required"]),
            "failure_rate": _rate(stats["system_compress_failure"], stats["system_compress_required"]),
            "failure_reasons": _counter_dict(nested["compress_failure_reason"]),
            "offline_gold_compress_chunks": int(stats["offline_gold_compress_chunks"]),
            "offline_gold_action_rows_skipped": int(
                stats["offline_gold_compress_action_rows_skipped"]
            ),
            "trigger_source": "offline_pass2_boundaries",
        },
        "memory_merge": {
            "events": int(stats["memory_merge_events"]),
            "event_rate_per_step": _rate(stats["memory_merge_events"], stats["steps"]),
            "threshold": args.memory_merge_threshold,
            "enabled": bool(args.memory_merge_similar_thinks),
            "avg_duration_sec_after": _rate(
                stats["memory_merge_duration_sec_sum"],
                stats["memory_merge_events"],
            ),
            "avg_chunk_count_after": _rate(
                stats["memory_merge_chunk_count_after_sum"],
                stats["memory_merge_events"],
            ),
            "max_duration_sec_after": int(stats["memory_merge_max_duration_sec"]),
            "reasons": _counter_dict(nested["memory_merge_reason"]),
            "duration_sec_after": _counter_dict(nested["memory_merge_duration_sec_after"]),
            "events_path": str(memory_merge_events_out),
        },
        "stable_think": {
            "pairs": int(stats["think_pairs"]),
            "stable_pairs": int(stats["stable_think_pairs"]),
            "stable_pair_rate": _rate(stats["stable_think_pairs"], stats["think_pairs"]),
            "threshold": args.stable_think_threshold,
        },
        "by_family": {
            fam: {
                "slots": int(nested["answer_slots_by_family"][fam]),
                "acc": _rate(
                    nested["answer_correct_by_family"][fam],
                    nested["answer_slots_by_family"][fam],
                ),
            }
            for fam in sorted(nested["answer_slots_by_family"])
        },
        "by_answer_form": {
            form: {
                "slots": int(nested["answer_slots_by_form"][form]),
                "acc": _rate(
                    nested["answer_correct_by_form"][form],
                    nested["answer_slots_by_form"][form],
                ),
            }
            for form in sorted(nested["answer_slots_by_form"])
        },
    }
    _write_json(out, summary)
    badcase_out.parent.mkdir(parents=True, exist_ok=True)
    with badcase_out.open("w", encoding="utf-8") as f:
        for kind, rows in sorted(badcases.items()):
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    memory_merge_events_out.parent.mkdir(parents=True, exist_ok=True)
    with memory_merge_events_out.open("w", encoding="utf-8") as f:
        for row in memory_merge_events:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out),
        "badcase_out": str(badcase_out),
        "memory_merge_events_out": str(memory_merge_events_out),
        "answer_acc": summary["answer"]["acc_completion"],
        "action_acc": summary["action"]["gold_acc"],
        "recall_support_hit": summary["recall"]["support_hit_rate"],
        "compress_success": summary["compression"]["success_rate"],
        "memory_merge_events": summary["memory_merge"]["events"],
        "stable_pair_rate": summary["stable_think"]["stable_pair_rate"],
        "steps_per_sec": summary["speed"]["steps_per_sec"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
