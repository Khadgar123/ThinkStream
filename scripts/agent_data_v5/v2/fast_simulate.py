"""Fast pass3 distribution simulator.

This intentionally stops at card / placement / recall scheduling. It does not
render pass3c messages, so it is much faster than ``v2.simulate`` and is meant
for threshold tuning on large batches.

Usage:
  python -m scripts.agent_data_v5.v2.fast_simulate \
    --root data/agent_v5 --batches 1-8 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..config import AGENT_CHUNK_SEC, COMPRESS_HYSTERESIS_THRESHOLD
from ..stable_hash import stable_seed
from .cards import generate_cards
from .design import (
    PATROL_KEEP_RATE_EMPTY,
    PATROL_KEEP_RATE_RICH,
    adaptive_q_count,
    assign_recall_noise,
    place_card,
    placement_timing_verdict,
    refine_placements_with_evidence,
    render_video_samples,
    select_trajectory,
)
from .llm_prompts import family_taxonomy
from .simulate import load_evidence, num_chunks_from


SFT_SILENT_TO_ACTIVE_RATIO = 0.90
SFT_PENDING_SILENT_FRACTION = 0.55
SFT_POST_ANSWER_SILENT_FRACTION = 0.25
SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO = 0.35
MULTI_EMIT_FAMILIES = {"F5", "F7", "CRR1", "PN1"}


def _parse_batches(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            out.extend(f"batch{i}" for i in range(lo, hi + 1))
        elif part.startswith("batch"):
            out.append(part)
        else:
            out.append(f"batch{int(part)}")
    return out


def _stats(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    values = sorted(float(x) for x in xs)

    def pct(p: float) -> float:
        k = (len(values) - 1) * p / 100.0
        lo = int(k)
        hi = min(lo + 1, len(values) - 1)
        frac = k - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "p50": round(pct(50), 2),
        "p75": round(pct(75), 2),
        "p90": round(pct(90), 2),
        "p95": round(pct(95), 2),
        "max": round(max(values), 2),
    }


def _json_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common()}


def _interval_floor_stats(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    total = len(xs)
    lt3 = sum(1 for x in xs if x < 3)
    lt4 = sum(1 for x in xs if x < 4)
    lt5 = sum(1 for x in xs if x < 5)
    return {
        "n": total,
        "lt3": lt3,
        "lt4": lt4,
        "lt5": lt5,
        "lt3_pct": round(lt3 / total * 100.0, 2),
        "lt4_pct": round(lt4 / total * 100.0, 2),
        "lt5_pct": round(lt5 / total * 100.0, 2),
    }


def _pct(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    return {
        str(k): round(v / max(total, 1) * 100.0, 2)
        for k, v in counter.most_common()
    }


def _sft_balance_estimate(
    *,
    recall_rows: int,
    compress_rows: int,
    ordinary_response: int,
    multi_emit_response: int,
    silent_roles: Counter,
) -> Dict[str, Any]:
    """Mirror pass5 SFT balancing at count level for fast simulation."""
    multi_keep = min(
        int(multi_emit_response),
        max(1, int(ordinary_response * SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO)),
    ) if multi_emit_response else 0
    active = int(recall_rows) + int(compress_rows) + int(ordinary_response) + multi_keep
    pending = int(silent_roles.get("pending_question", 0))
    post = int(silent_roles.get("post_answer", 0))
    base = int(silent_roles.get("no_question", 0))
    silent_total = pending + post + base
    if not active:
        return {
            "before_total": active + silent_total,
            "after_total": active + silent_total,
            "after_action": {
                "silent": silent_total,
                "response": ordinary_response + multi_emit_response,
                "recall": recall_rows,
                "compress": compress_rows,
            },
        }

    target_silent = min(silent_total, max(1, int(active * SFT_SILENT_TO_ACTIVE_RATIO)))
    keep_pending = min(pending, int(target_silent * SFT_PENDING_SILENT_FRACTION))
    keep_post = min(post, int(target_silent * SFT_POST_ANSWER_SILENT_FRACTION))
    remaining = target_silent - keep_pending - keep_post
    keep_base = min(base, max(0, remaining))
    remaining -= keep_base
    if remaining > 0:
        extra_pending = min(pending - keep_pending, remaining)
        keep_pending += extra_pending
        remaining -= extra_pending
    if remaining > 0:
        extra_post = min(post - keep_post, remaining)
        keep_post += extra_post
        remaining -= extra_post
    if remaining > 0:
        keep_base += min(base - keep_base, remaining)

    after_silent = keep_pending + keep_post + keep_base
    after_action = Counter({
        "silent": after_silent,
        "response": ordinary_response + multi_keep,
        "recall": recall_rows,
        "compress": compress_rows,
    })
    before_action = Counter({
        "silent": silent_total,
        "response": ordinary_response + multi_emit_response,
        "recall": recall_rows,
        "compress": compress_rows,
    })
    return {
        "before_total": int(sum(before_action.values())),
        "after_total": int(sum(after_action.values())),
        "before_action": _json_counter(before_action),
        "before_action_pct": _pct(before_action),
        "after_action": _json_counter(after_action),
        "after_action_pct": _pct(after_action),
        "active_kept": int(active),
        "ordinary_response_kept": int(ordinary_response),
        "multi_emit_response_before": int(multi_emit_response),
        "multi_emit_response_kept": int(multi_keep),
        "recall_kept": int(recall_rows),
        "compress_kept": int(compress_rows),
        "silent_before": int(silent_total),
        "silent_kept": int(after_silent),
        "pending_silent_before": int(pending),
        "post_answer_silent_before": int(post),
        "base_silent_before": int(base),
        "pending_silent_kept": int(keep_pending),
        "post_answer_silent_kept": int(keep_post),
        "base_silent_kept": int(keep_base),
    }


def _is_rich_by_chunk(evidence: List[Dict]) -> Dict[int, bool]:
    rich: Dict[int, bool] = {}
    seen_entities: set[str] = set()
    for cap in evidence:
        c = int(cap.get("chunk_idx", 0))
        is_rich = bool(cap.get("state_changes"))
        if not is_rich:
            for ent in cap.get("visible_entities", []) or []:
                eid = ent.get("id") or str(ent.get("desc", ""))[:30]
                if eid and eid not in seen_entities:
                    seen_entities.add(eid)
                    is_rich = True
        rich[c] = is_rich
    return rich


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_json_error": str(path)}


def _rollout_path_for_evidence(path: Path) -> Path:
    return path.parent.parent / "rollout" / path.name


def _int_list(values: object) -> List[int]:
    out: List[int] = []
    for value in values or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _time_range_chunks(time_range: object) -> List[int]:
    if not (isinstance(time_range, list) and len(time_range) == 2):
        return []
    try:
        start_s, end_s = float(time_range[0]), float(time_range[1])
    except (TypeError, ValueError):
        return []
    if end_s <= start_s:
        return []
    start_chunk = int(start_s / float(AGENT_CHUNK_SEC))
    end_chunk = int((end_s - 1e-6) / float(AGENT_CHUNK_SEC))
    return list(range(start_chunk, end_chunk + 1))


def _compression_audit(rollout: Dict[str, Any], num_chunks: int) -> Dict[str, Any]:
    events = list(rollout.get("compression_events") or [])
    violations = Counter()
    range_sizes: List[int] = []
    duration_chunks: List[int] = []
    trigger_lag_chunks: List[int] = []
    post_tokens: List[int] = []
    summary_words: List[int] = []
    parse_success = 0
    parse_fail = 0
    hysteresis_violations = 0
    hysteresis_missing = 0
    recompressed_chunks = 0
    seen_chunks: set[int] = set()
    previous_ranges: List[set[int]] = []

    for event in events:
        summary = event.get("summary") or {}
        trigger = event.get("trigger_chunk")
        try:
            trigger_i = int(trigger)
        except (TypeError, ValueError):
            trigger_i = -1
            violations["bad_trigger_chunk"] += 1
        if trigger_i < 0 or trigger_i >= max(num_chunks, 1):
            violations["trigger_out_of_bounds"] += 1

        chunks = _int_list(event.get("compressed_thinks_chunks"))
        if not chunks:
            chunks = _int_list(event.get("compressed_raw_think_chunks"))
        tr_chunks = _time_range_chunks(summary.get("time_range"))
        if not chunks:
            chunks = tr_chunks
            if chunks:
                violations["missing_explicit_source_chunks"] += 1
        if not chunks:
            violations["empty_compressed_range"] += 1

        source_set = set(chunks)
        if source_set:
            range_sizes.append(len(source_set))
            duration_chunks.append(max(source_set) - min(source_set) + 1)
            if trigger_i >= 0:
                trigger_lag_chunks.append(trigger_i - max(source_set))
            if min(source_set) < 0 or max(source_set) >= max(num_chunks, 1):
                violations["source_chunk_out_of_bounds"] += 1
            if trigger_i >= 0 and max(source_set) >= trigger_i:
                violations["source_reaches_or_exceeds_trigger"] += 1
            if tr_chunks:
                missing_from_tr = source_set - set(tr_chunks)
                if missing_from_tr:
                    violations["source_not_covered_by_time_range"] += 1
            for previous in previous_ranges:
                if previous & source_set:
                    violations["overlapping_compress_ranges"] += 1
                    break
            recompressed_chunks += len(seen_chunks & source_set)
            seen_chunks.update(source_set)
            previous_ranges.append(source_set)

        selected = _int_list(event.get("selected_indices"))
        if not selected:
            violations["empty_selected_indices"] += 1

        tr = summary.get("time_range")
        if not (isinstance(tr, list) and len(tr) == 2):
            violations["bad_summary_time_range"] += 1
        else:
            try:
                tr_start = float(tr[0])
                tr_end = float(tr[1])
            except (TypeError, ValueError):
                violations["bad_summary_time_range"] += 1
            else:
                if tr_end <= tr_start:
                    violations["non_positive_summary_time_range"] += 1
                if trigger_i >= 0 and tr_end > trigger_i * AGENT_CHUNK_SEC:
                    violations["summary_time_range_reaches_future"] += 1
                if tr_start < 0:
                    violations["negative_summary_time_range"] += 1

        text = str(summary.get("text") or "").strip()
        if not text:
            violations["empty_summary_text"] += 1
        else:
            summary_words.append(len(text.split()))
        if summary.get("parse_success", False):
            parse_success += 1
        else:
            parse_fail += 1

        try:
            pt = int(event.get("post_compress_tokens"))
        except (TypeError, ValueError):
            pt = -1
        if pt >= 0:
            post_tokens.append(pt)
            if pt > COMPRESS_HYSTERESIS_THRESHOLD:
                hysteresis_violations += 1
                violations["hysteresis_token_violation"] += 1
        else:
            hysteresis_missing += 1
        if event.get("hysteresis_ok") is False:
            hysteresis_violations += int(pt < 0)
            violations["hysteresis_flag_false"] += 1

    return {
        "compression_events": len(events),
        "compression_videos_with_events": int(bool(events)),
        "compression_range_sizes": range_sizes,
        "compression_duration_chunks": duration_chunks,
        "compression_trigger_lag_chunks": trigger_lag_chunks,
        "compression_post_tokens": post_tokens,
        "compression_summary_words": summary_words,
        "compression_parse_success": parse_success,
        "compression_parse_fail": parse_fail,
        "compression_hysteresis_violations": hysteresis_violations,
        "compression_hysteresis_missing": hysteresis_missing,
        "compression_recompressed_chunks": recompressed_chunks,
        "compression_violations": dict(violations),
    }


def _archive_before(rollout: Dict[str, Any], current_chunk: int) -> List[Dict[str, Any]]:
    archive: List[Dict[str, Any]] = []
    for think in rollout.get("thinks") or []:
        try:
            ci = int(think.get("chunk_idx", think.get("chunk", -1)))
        except (AttributeError, TypeError, ValueError):
            continue
        if ci < 0 or ci >= int(current_chunk):
            continue
        text = str(think.get("think", think.get("text", "")) or "").strip()
        if not text:
            continue
        archive.append({
            "chunk": ci,
            "time": f"{int(ci * AGENT_CHUNK_SEC)}-"
                    f"{int((ci + 1) * AGENT_CHUNK_SEC)}",
            "text": text,
        })
    return archive


def _recall_audit(
    selected: Sequence[Any],
    cards_by_id: Dict[str, Any],
    rollout: Dict[str, Any],
) -> Dict[str, Any]:
    # Import the pass3c production helpers here so the simulator audits the
    # exact deterministic query/recall path without paying this import cost
    # when rollout auditing is disabled.
    from ..pass3a_cards import _card_to_dict
    from ..pass3c_samples import (
        RECALL_RETURN_CHUNKS,
        _current_context_text_for_chunk,
        _memory_text_for_chunk,
        _needs_recall_hardening,
        _recall_query_available,
        _recall_query_for,
        _recall_query_leaks_answer,
        _recall_result_for,
        _recall_wait_query_for,
        _repair_recall_query_for_response,
        _support_chunks_before,
        _valid_recall_query,
        bm25_retrieve,
        select_recall_chunks,
    )

    counters = Counter()
    by_kind = Counter()
    by_mechanism = Counter()
    by_family = Counter()
    reasons = Counter()
    returned_counts: List[int] = []
    bm25_raw_counts: List[int] = []
    support_hit = 0
    support_miss = 0
    future_leak = 0
    invalid_query = 0
    result_fail = 0
    result_empty = 0

    for placement in selected:
        if not placement.recall_at:
            continue
        card_obj = cards_by_id.get(placement.card_id)
        if not card_obj:
            continue
        card = _card_to_dict(card_obj)
        for c_raw, noise_kind in placement.recall_at.items():
            try:
                c = int(c_raw)
            except (TypeError, ValueError):
                counters["bad_recall_chunk"] += 1
                continue
            counters["recall_slots"] += 1
            by_kind[str(noise_kind)] += 1
            by_mechanism[str(placement.mechanism)] += 1
            by_family[str(card.get("family", ""))] += 1
            reasons[str(placement.recall_reason_at.get(c, ""))] += 1

            if placement.mechanism == "recall_demo":
                counters["pass3c_hardening_scope_slots"] += 1
                memory_text = _memory_text_for_chunk(rollout, c)
                context_text = _current_context_text_for_chunk(
                    rollout, c, memory_text=memory_text,
                )
                if _needs_recall_hardening(card, context_text):
                    counters["pass3c_would_harden_or_downgrade"] += 1
                else:
                    counters["pass3c_already_hard"] += 1

            if str(noise_kind) == "not_yet":
                rq = _recall_wait_query_for(card, c)
                legal_query = _valid_recall_query(rq)
            else:
                rq = _recall_query_for(card, c)
                rq = _repair_recall_query_for_response(card, rq, c)
                legal_query = _recall_query_available(rq, c)
            if not legal_query:
                invalid_query += 1
                continue
            if _recall_query_leaks_answer(card, rq):
                counters["recall_query_answer_leak"] += 1

            archive = _archive_before(rollout, c)
            retrieved = bm25_retrieve(rq, archive, max_results=RECALL_RETURN_CHUNKS)
            raw_chunks = select_recall_chunks(retrieved.get("returned_chunks") or [])
            bm25_raw_counts.append(len(raw_chunks))
            if raw_chunks:
                counters["bm25_raw_nonempty"] += 1
            else:
                counters["bm25_raw_empty"] += 1

            rr = _recall_result_for(
                card,
                rollout,
                str(noise_kind),
                current_chunk=c,
                recall_query=rq,
            )
            chunks = _int_list(rr.get("returned_chunks"))
            returned_counts.append(len(chunks))
            if rr.get("source") == "failure":
                result_fail += 1
            if not chunks:
                result_empty += 1
            else:
                counters["history_frame_result_nonempty"] += 1
            if chunks and max(chunks) >= c:
                future_leak += 1
            if raw_chunks and chunks != raw_chunks:
                counters["bm25_adjusted_by_support_or_noise"] += 1
            elif not raw_chunks and chunks:
                counters["fallback_filled_history_frames"] += 1

            if str(noise_kind) != "not_yet":
                support = set(_support_chunks_before(card, c))
                if support:
                    if support & set(chunks):
                        support_hit += 1
                    else:
                        support_miss += 1

    return {
        "recall_audit": dict(counters),
        "recall_audit_by_kind": dict(by_kind),
        "recall_audit_by_mechanism": dict(by_mechanism),
        "recall_audit_by_family": dict(by_family),
        "recall_audit_reason": dict(reasons),
        "recall_returned_chunk_counts": returned_counts,
        "recall_bm25_raw_chunk_counts": bm25_raw_counts,
        "recall_support_hit": support_hit,
        "recall_support_miss": support_miss,
        "recall_future_leak": future_leak,
        "recall_invalid_query": invalid_query,
        "recall_result_fail": result_fail,
        "recall_result_empty": result_empty,
    }


def _simulate_one(path_s: str, seed: int, audit_rollout: bool = True) -> Dict[str, Any]:
    path = Path(path_s)
    video_id = path.stem
    evidence = load_evidence(path)
    num_chunks = num_chunks_from(evidence)
    rng = random.Random(stable_seed(seed, video_id, modulo=1_000_000))

    cards = generate_cards(evidence, video_id, seed=seed)
    placements_by_card = {}
    timing = Counter()
    for card in cards:
        plcs = refine_placements_with_evidence(
            card, place_card(card, num_chunks, rng), evidence
        )
        good = []
        for p in plcs:
            ok, reason = placement_timing_verdict(card, p)
            if ok:
                good.append(p)
            else:
                timing[reason] += 1
        placements_by_card[card.card_id] = good

    selected = select_trajectory(
        cards,
        placements_by_card,
        num_chunks,
        rng,
        max_q=adaptive_q_count(num_chunks),
    )
    assign_recall_noise(selected, rng, cards_by_id={c.card_id: c for c in cards})
    rollout: Dict[str, Any] = {}
    recall_refine_stats: Dict[str, int] = {}
    if audit_rollout:
        rollout = _load_json(_rollout_path_for_evidence(path))
        if rollout and not rollout.get("_json_error"):
            from ..pass3a_cards import _card_to_dict
            from ..pass3b_placement import _refine_selected_recall_with_rollout

            recall_refine_stats = _refine_selected_recall_with_rollout(
                selected,
                {c.card_id: _card_to_dict(c) for c in cards},
                rollout,
                video_id=video_id,
            )

    cards_by_id = {c.card_id: c for c in cards}
    sample = Counter()
    raw_action = Counter()
    silent_role = Counter()
    wait_phase = Counter()
    wait_phase_by_mechanism = Counter()
    sample_kind_by_mechanism = Counter()
    sft_after_action = Counter()
    sft_after_silent_role = Counter()
    mechanism = Counter()
    family = Counter()
    answer_form = Counter()
    question_type = Counter()
    category = Counter()
    family_mechanism = Counter()
    recall_kind = Counter()
    recall_mechanism = Counter()
    recall_family = Counter()
    recall_reason = Counter()
    ask_answer = Counter()
    used_chunks: set[int] = set()
    overlap = 0
    first_waits: List[int] = []
    wait_not_yet_per_forward: List[int] = []
    f5_recall_per_q: List[int] = []
    status_recall_per_q: List[int] = []
    q_with_recall = 0
    multi_recall_q = 0
    forward_recall_q = 0

    for p in selected:
        card = cards_by_id[p.card_id]
        mechanism[p.mechanism] += 1
        family[card.family] += 1
        answer_form[card.answer_form] += 1
        question_type[card.question_type] += 1
        category[family_taxonomy(card.family).get("category", "Unknown")] += 1
        family_mechanism[f"{card.family}|{p.mechanism}"] += 1
        if p.recall_at:
            q_with_recall += 1
            multi_recall_q += int(p.mechanism == "multi_emit")
            forward_recall_q += int(p.mechanism == "silent_then_response")
        if p.mechanism == "silent_then_response":
            wait_not_yet_per_forward.append(
                sum(1 for v in p.recall_at.values() if v == "not_yet")
            )
        if card.family == "F5":
            f5_recall_per_q.append(len(p.recall_at))
        if card.family in {"F7", "CRR1"}:
            status_recall_per_q.append(len(p.recall_at))

        response_chunks: List[int] = []
        for c_raw, (kind, _value) in p.chunk_actions.items():
            c = int(c_raw)
            if c in used_chunks:
                overlap += 1
            used_chunks.add(c)
            if kind == "response":
                response_chunks.append(c)
        if response_chunks:
            first_response = min(response_chunks)
            first_waits.append(first_response - int(p.ask_chunk))
            if p.mechanism in {"direct", "memory_direct", "recall_demo"}:
                if first_response != int(p.ask_chunk):
                    ask_answer[f"{p.mechanism}_answer_not_at_ask"] += 1
            elif p.mechanism == "silent_then_response":
                if not (int(p.ask_chunk) < first_response):
                    ask_answer["forward_answer_not_after_ask"] += 1
            elif p.mechanism == "multi_emit":
                if int(p.ask_chunk) > first_response:
                    ask_answer["multi_emit_ask_after_first_answer"] += 1

        if card.question_type == "single_emit":
            try:
                emit = int(card.gold_emits[0].chunk)
            except (IndexError, TypeError, ValueError):
                emit = -1
            expected_response = max(int(p.ask_chunk), emit)
            if response_chunks != [expected_response]:
                ask_answer["single_emit_response_chunk_changed"] += 1
            support = [int(x) for x in (card.grounding_frames or [emit])]
            if support and emit != max(support):
                ask_answer["single_emit_emit_not_latest_support"] += 1
        else:
            emit_map = {int(e.chunk): str(e.value) for e in card.gold_emits}
            for c in response_chunks:
                if c not in emit_map:
                    ask_answer["multi_emit_response_not_gold_emit"] += 1
                elif str(p.chunk_actions[c][1]) != str(emit_map[c]):
                    ask_answer["multi_emit_response_value_changed"] += 1

        for c, kind in p.recall_at.items():
            action_kind = (p.chunk_actions.get(int(c)) or ("", ""))[0]
            if kind == "not_yet" and action_kind != "silent":
                ask_answer["recall_silent_not_on_silent_chunk"] += 1
            if kind != "not_yet" and action_kind != "response":
                ask_answer["recall_response_not_on_response_chunk"] += 1
            recall_kind[kind] += 1
            recall_mechanism[p.mechanism] += 1
            recall_family[card.family] += 1
            recall_reason[p.recall_reason_at.get(c, "")] += 1

    selected_by_card: Dict[str, List[Any]] = {}
    for p in selected:
        selected_by_card.setdefault(p.card_id, []).append(p)
    compression_event_chunks = []
    if rollout and not rollout.get("_json_error"):
        compression_event_chunks = [
            int(e.get("trigger_chunk", -1))
            for e in rollout.get("compression_events", [])
            if e.get("trigger_chunk", -1) >= 0
        ]
    design_samples = render_video_samples(
        cards,
        selected_by_card,
        num_chunks,
        evidence=evidence,
        rng=rng,
        compression_event_chunks=compression_event_chunks,
    )
    sample.update(ds.sample_kind for ds in design_samples)

    placements_sorted = sorted(selected, key=lambda p: (int(p.ask_chunk), p.card_id))
    open_until_by_card: Dict[str, int] = {}
    for p in placements_sorted:
        response_chunks = [
            int(c) for c, (kind, _value) in p.chunk_actions.items()
            if kind == "response"
        ]
        if response_chunks:
            open_until_by_card[p.card_id] = max(response_chunks)
    query_status: Dict[str, str] = {}
    query_answers: Dict[str, int] = {}
    asked_cards: set[str] = set()
    ordinary_response_rows = 0
    multi_emit_response_rows = 0
    recall_rows_for_sft = 0
    compress_rows_for_sft = 0

    def add_open_queries(chunk: int) -> None:
        for p in placements_sorted:
            if p.card_id in asked_cards or int(p.ask_chunk) > int(chunk):
                continue
            asked_cards.add(p.card_id)
            query_status[p.card_id] = "open"
            query_answers[p.card_id] = 0

    def phase_for_wait(ds) -> str:
        related_ids: List[str]
        if ds.card_id:
            related_ids = [ds.card_id]
        else:
            related_ids = list(query_status.keys())
        open_ids = [
            cid for cid in related_ids
            if query_status.get(cid) in {"open", "pending", "active"}
        ]
        if open_ids:
            if any(query_answers.get(cid, 0) > 0 for cid in open_ids):
                return "between_answers_pending"
            return "pending_question"
        if any(query_answers.get(cid, 0) > 0 for cid in related_ids):
            return "post_answer"
        return "no_question"

    for ds in sorted(design_samples, key=lambda s: int(s.chunk_idx)):
        c = int(ds.chunk_idx)
        add_open_queries(c)
        kind = str(ds.sample_kind)
        card = cards_by_id.get(ds.card_id) if ds.card_id else None
        sample_kind_by_mechanism[f"{ds.mechanism}|{kind}"] += 1
        if kind in {"silent", "patrol"}:
            raw_action["silent"] += 1
            phase = phase_for_wait(ds)
            silent_role[
                "pending_question"
                if phase in {"pending_question", "between_answers_pending"}
                else phase
            ] += 1
            wait_phase[phase] += 1
            wait_phase_by_mechanism[f"{ds.mechanism}|{phase}"] += 1
        elif kind == "compress_silent":
            raw_action["compress"] += 1
            compress_rows_for_sft += 1
        elif kind == "response":
            raw_action["response"] += 1
            is_multi = (
                ds.mechanism == "multi_emit"
                or (card and card.question_type == "multi_emit")
                or (card and card.family in MULTI_EMIT_FAMILIES)
            )
            if is_multi:
                multi_emit_response_rows += 1
            else:
                ordinary_response_rows += 1
        elif kind == "recall+response":
            raw_action["recall_response"] += 1
            recall_rows_for_sft += 1
        elif kind == "recall+silent":
            raw_action["recall_silent"] += 1
            recall_rows_for_sft += 1
            phase = phase_for_wait(ds)
            wait_phase[f"recall_silent:{phase}"] += 1
            wait_phase_by_mechanism[f"{ds.mechanism}|recall_silent:{phase}"] += 1

        if kind in {"response", "recall+response"} and ds.card_id:
            query_answers[ds.card_id] = query_answers.get(ds.card_id, 0) + 1
            if c >= open_until_by_card.get(ds.card_id, c):
                query_status[ds.card_id] = "answered"
            else:
                query_status[ds.card_id] = "open"

    sft_estimate = _sft_balance_estimate(
        recall_rows=recall_rows_for_sft,
        compress_rows=compress_rows_for_sft,
        ordinary_response=ordinary_response_rows,
        multi_emit_response=multi_emit_response_rows,
        silent_roles=silent_role,
    )
    sft_after_action.update(sft_estimate.get("after_action") or {})
    sft_after_silent_role.update({
        "pending_question": int(sft_estimate.get("pending_silent_kept", 0)),
        "post_answer": int(sft_estimate.get("post_answer_silent_kept", 0)),
        "no_question": int(sft_estimate.get("base_silent_kept", 0)),
    })

    ask_chunks = sorted(int(p.ask_chunk) for p in selected)
    intervals = [
        ask_chunks[i + 1] - ask_chunks[i] for i in range(len(ask_chunks) - 1)
    ]
    out = {
        "videos": 1,
        "q_counts": [len(selected)],
        "q_intervals": intervals,
        "first_waits": first_waits,
        "wait_not_yet_per_forward": wait_not_yet_per_forward,
        "f5_recall_per_q": f5_recall_per_q,
        "status_recall_per_q": status_recall_per_q,
        "placements_total": len(selected),
        "questions_with_recall": q_with_recall,
        "multi_recall_questions": multi_recall_q,
        "forward_recall_questions": forward_recall_q,
        "sample": dict(sample),
        "raw_action": dict(raw_action),
        "silent_role": dict(silent_role),
        "wait_phase": dict(wait_phase),
        "wait_phase_by_mechanism": dict(wait_phase_by_mechanism),
        "sample_kind_by_mechanism": dict(sample_kind_by_mechanism),
        "sft_after_action": dict(sft_after_action),
        "sft_after_silent_role": dict(sft_after_silent_role),
        "sft_estimate": sft_estimate,
        "mechanism": dict(mechanism),
        "family": dict(family),
        "answer_form": dict(answer_form),
        "question_type": dict(question_type),
        "category": dict(category),
        "family_mechanism": dict(family_mechanism),
        "recall_kind": dict(recall_kind),
        "recall_mechanism": dict(recall_mechanism),
        "recall_family": dict(recall_family),
        "recall_reason": dict(recall_reason),
        "timing": dict(timing),
        "ask_answer": dict(ask_answer),
        "overlap": overlap,
        "recall_refine": recall_refine_stats,
    }
    if audit_rollout:
        out["rollout_present"] = int(bool(rollout) and not rollout.get("_json_error"))
        if rollout and not rollout.get("_json_error"):
            out.update(_compression_audit(rollout, num_chunks))
            out.update(_recall_audit(selected, cards_by_id, rollout))
        else:
            out["rollout_missing"] = 1
    return out


def _merge(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    sample = Counter()
    raw_action = Counter()
    silent_role = Counter()
    wait_phase = Counter()
    wait_phase_by_mechanism = Counter()
    sample_kind_by_mechanism = Counter()
    sft_after_action = Counter()
    sft_after_silent_role = Counter()
    mechanism = Counter()
    family = Counter()
    answer_form = Counter()
    question_type = Counter()
    category = Counter()
    family_mechanism = Counter()
    recall_kind = Counter()
    recall_mechanism = Counter()
    recall_family = Counter()
    recall_reason = Counter()
    recall_audit = Counter()
    recall_audit_by_kind = Counter()
    recall_audit_by_mechanism = Counter()
    recall_audit_by_family = Counter()
    recall_audit_reason = Counter()
    recall_refine = Counter()
    timing = Counter()
    ask_answer = Counter()
    compression_violations = Counter()
    q_counts: List[float] = []
    q_intervals: List[float] = []
    first_waits: List[float] = []
    wait_not_yet: List[float] = []
    f5_recall: List[float] = []
    status_recall: List[float] = []
    comp_range_sizes: List[float] = []
    comp_duration_chunks: List[float] = []
    comp_trigger_lag_chunks: List[float] = []
    comp_post_tokens: List[float] = []
    comp_summary_words: List[float] = []
    recall_returned_counts: List[float] = []
    recall_bm25_raw_counts: List[float] = []
    videos = placements_total = q_with_recall = 0
    multi_recall_q = forward_recall_q = overlap = 0
    rollout_present = rollout_missing = 0
    compression_events = compression_videos_with_events = 0
    compression_parse_success = compression_parse_fail = 0
    compression_hysteresis_violations = compression_hysteresis_missing = 0
    compression_recompressed_chunks = 0
    recall_support_hit = recall_support_miss = recall_future_leak = 0
    recall_invalid_query = recall_result_fail = recall_result_empty = 0

    for r in results:
        videos += int(r.get("videos", 0))
        placements_total += int(r.get("placements_total", 0))
        q_with_recall += int(r.get("questions_with_recall", 0))
        multi_recall_q += int(r.get("multi_recall_questions", 0))
        forward_recall_q += int(r.get("forward_recall_questions", 0))
        overlap += int(r.get("overlap", 0))
        q_counts.extend(r.get("q_counts") or [])
        q_intervals.extend(r.get("q_intervals") or [])
        first_waits.extend(r.get("first_waits") or [])
        wait_not_yet.extend(r.get("wait_not_yet_per_forward") or [])
        f5_recall.extend(r.get("f5_recall_per_q") or [])
        status_recall.extend(r.get("status_recall_per_q") or [])
        comp_range_sizes.extend(r.get("compression_range_sizes") or [])
        comp_duration_chunks.extend(r.get("compression_duration_chunks") or [])
        comp_trigger_lag_chunks.extend(r.get("compression_trigger_lag_chunks") or [])
        comp_post_tokens.extend(r.get("compression_post_tokens") or [])
        comp_summary_words.extend(r.get("compression_summary_words") or [])
        recall_returned_counts.extend(r.get("recall_returned_chunk_counts") or [])
        recall_bm25_raw_counts.extend(r.get("recall_bm25_raw_chunk_counts") or [])
        sample.update(r.get("sample") or {})
        raw_action.update(r.get("raw_action") or {})
        silent_role.update(r.get("silent_role") or {})
        wait_phase.update(r.get("wait_phase") or {})
        wait_phase_by_mechanism.update(r.get("wait_phase_by_mechanism") or {})
        sample_kind_by_mechanism.update(r.get("sample_kind_by_mechanism") or {})
        sft_after_action.update(r.get("sft_after_action") or {})
        sft_after_silent_role.update(r.get("sft_after_silent_role") or {})
        mechanism.update(r.get("mechanism") or {})
        family.update(r.get("family") or {})
        answer_form.update(r.get("answer_form") or {})
        question_type.update(r.get("question_type") or {})
        category.update(r.get("category") or {})
        family_mechanism.update(r.get("family_mechanism") or {})
        recall_kind.update(r.get("recall_kind") or {})
        recall_mechanism.update(r.get("recall_mechanism") or {})
        recall_family.update(r.get("recall_family") or {})
        recall_reason.update(r.get("recall_reason") or {})
        recall_audit.update(r.get("recall_audit") or {})
        recall_audit_by_kind.update(r.get("recall_audit_by_kind") or {})
        recall_audit_by_mechanism.update(r.get("recall_audit_by_mechanism") or {})
        recall_audit_by_family.update(r.get("recall_audit_by_family") or {})
        recall_audit_reason.update(r.get("recall_audit_reason") or {})
        recall_refine.update(r.get("recall_refine") or {})
        timing.update(r.get("timing") or {})
        ask_answer.update(r.get("ask_answer") or {})
        compression_violations.update(r.get("compression_violations") or {})
        rollout_present += int(r.get("rollout_present", 0))
        rollout_missing += int(r.get("rollout_missing", 0))
        compression_events += int(r.get("compression_events", 0))
        compression_videos_with_events += int(r.get("compression_videos_with_events", 0))
        compression_parse_success += int(r.get("compression_parse_success", 0))
        compression_parse_fail += int(r.get("compression_parse_fail", 0))
        compression_hysteresis_violations += int(r.get("compression_hysteresis_violations", 0))
        compression_hysteresis_missing += int(r.get("compression_hysteresis_missing", 0))
        compression_recompressed_chunks += int(r.get("compression_recompressed_chunks", 0))
        recall_support_hit += int(r.get("recall_support_hit", 0))
        recall_support_miss += int(r.get("recall_support_miss", 0))
        recall_future_leak += int(r.get("recall_future_leak", 0))
        recall_invalid_query += int(r.get("recall_invalid_query", 0))
        recall_result_fail += int(r.get("recall_result_fail", 0))
        recall_result_empty += int(r.get("recall_result_empty", 0))

    rows = sum(sample.values())
    responses = sample.get("response", 0) + sample.get("recall+response", 0)
    recall_rows = sum(recall_kind.values())
    return {
        "videos": videos,
        "questions_per_trajectory": _stats(q_counts),
        "q_interval_chunks": _stats(q_intervals),
        "q_interval_floor": _interval_floor_stats(q_intervals),
        "first_wait_chunks": _stats(first_waits),
        "mechanism_pct": {
            k: round(v / max(placements_total, 1) * 100, 2)
            for k, v in mechanism.items()
        },
        "selected_family": _json_counter(family),
        "selected_family_pct": _pct(family),
        "selected_answer_form": _json_counter(answer_form),
        "selected_answer_form_pct": _pct(answer_form),
        "selected_question_type": _json_counter(question_type),
        "selected_question_type_pct": _pct(question_type),
        "selected_category": _json_counter(category),
        "selected_category_pct": _pct(category),
        "selected_family_mechanism_top": _json_counter(family_mechanism),
        "sample_kind": _json_counter(sample),
        "sample_kind_pct": _pct(sample),
        "raw_single_step_action": _json_counter(raw_action),
        "raw_single_step_action_pct": _pct(raw_action),
        "raw_silent_role": _json_counter(silent_role),
        "raw_silent_role_pct": _pct(silent_role),
        "raw_wait_phase": _json_counter(wait_phase),
        "raw_wait_phase_pct": _pct(wait_phase),
        "raw_wait_phase_by_mechanism": _json_counter(wait_phase_by_mechanism),
        "sample_kind_by_mechanism": _json_counter(sample_kind_by_mechanism),
        "sft_balanced_action_estimate": _json_counter(sft_after_action),
        "sft_balanced_action_estimate_pct": _pct(sft_after_action),
        "sft_balanced_silent_role_estimate": _json_counter(sft_after_silent_role),
        "sft_balanced_silent_role_estimate_pct": _pct(sft_after_silent_role),
        "response_pct": round(responses / max(rows, 1) * 100, 2),
        "silent_pct": round((rows - responses) / max(rows, 1) * 100, 2),
        "recall_rows_total": int(recall_rows),
        "recall_row_pct_of_all_samples": round(recall_rows / max(rows, 1) * 100, 2),
        "recall_by_kind": _json_counter(recall_kind),
        "recall_by_mechanism": _json_counter(recall_mechanism),
        "recall_by_family": _json_counter(recall_family),
        "recall_reason": _json_counter(recall_reason),
        "questions_with_any_recall": {
            "n": int(q_with_recall),
            "pct": round(q_with_recall / max(placements_total, 1) * 100, 2),
        },
        "multi_emit_questions_with_recall": int(multi_recall_q),
        "forward_wait_questions_with_recall": int(forward_recall_q),
        "wait_not_yet_per_forward": _stats(wait_not_yet),
        "f5_recall_per_selected_f5": _stats(f5_recall),
        "status_recall_per_selected_status": _stats(status_recall),
        "overlap_violations": int(overlap),
        "timing_violations": _json_counter(timing),
        "ask_answer_violations": _json_counter(ask_answer),
        "rollout_audit": {
            "rollout_present": int(rollout_present),
            "rollout_missing": int(rollout_missing),
        },
        "pass2_compression_audit": {
            "events": int(compression_events),
            "videos_with_events": int(compression_videos_with_events),
            "events_per_video": round(compression_events / max(rollout_present, 1), 3),
            "range_size_chunks": _stats(comp_range_sizes),
            "duration_chunks": _stats(comp_duration_chunks),
            "trigger_lag_chunks": _stats(comp_trigger_lag_chunks),
            "post_compress_tokens": _stats(comp_post_tokens),
            "summary_words": _stats(comp_summary_words),
            "parse_success": int(compression_parse_success),
            "parse_fail": int(compression_parse_fail),
            "parse_success_rate": round(
                compression_parse_success
                / max(compression_parse_success + compression_parse_fail, 1),
                4,
            ),
            "hysteresis_threshold": int(COMPRESS_HYSTERESIS_THRESHOLD),
            "hysteresis_violations": int(compression_hysteresis_violations),
            "hysteresis_violation_rate": round(
                compression_hysteresis_violations / max(compression_events, 1),
                4,
            ),
            "hysteresis_missing": int(compression_hysteresis_missing),
            "recompressed_chunks": int(compression_recompressed_chunks),
            "violations": _json_counter(compression_violations),
        },
        "pass3_recall_audit": {
            "pass3b_rollout_refine": _json_counter(recall_refine),
            "slots": int(recall_audit.get("recall_slots", 0)),
            "by_kind": _json_counter(recall_audit_by_kind),
            "by_mechanism": _json_counter(recall_audit_by_mechanism),
            "by_family": _json_counter(recall_audit_by_family),
            "reason": _json_counter(recall_audit_reason),
            "pass3c_hardening_scope_slots": int(
                recall_audit.get("pass3c_hardening_scope_slots", 0)
            ),
            "pass3c_would_harden_or_downgrade": int(
                recall_audit.get("pass3c_would_harden_or_downgrade", 0)
            ),
            "pass3c_would_harden_or_downgrade_rate": round(
                int(recall_audit.get("pass3c_would_harden_or_downgrade", 0))
                / max(int(recall_audit.get("pass3c_hardening_scope_slots", 0)), 1),
                4,
            ),
            "pass3c_already_hard": int(recall_audit.get("pass3c_already_hard", 0)),
            "bm25_raw_nonempty": int(recall_audit.get("bm25_raw_nonempty", 0)),
            "bm25_raw_nonempty_rate": round(
                int(recall_audit.get("bm25_raw_nonempty", 0))
                / max(int(recall_audit.get("recall_slots", 0)), 1),
                4,
            ),
            "bm25_raw_empty": int(recall_audit.get("bm25_raw_empty", 0)),
            "recall_query_answer_leak": int(
                recall_audit.get("recall_query_answer_leak", 0)
            ),
            "bm25_adjusted_by_support_or_noise": int(
                recall_audit.get("bm25_adjusted_by_support_or_noise", 0)
            ),
            "fallback_filled_history_frames": int(
                recall_audit.get("fallback_filled_history_frames", 0)
            ),
            "history_frame_result_nonempty": int(
                recall_audit.get("history_frame_result_nonempty", 0)
            ),
            "returned_chunk_count": _stats(recall_returned_counts),
            "bm25_raw_chunk_count": _stats(recall_bm25_raw_counts),
            "support_hit": int(recall_support_hit),
            "support_miss": int(recall_support_miss),
            "support_hit_rate": round(
                recall_support_hit / max(recall_support_hit + recall_support_miss, 1),
                4,
            ),
            "future_leak": int(recall_future_leak),
            "invalid_query": int(recall_invalid_query),
            "result_fail": int(recall_result_fail),
            "result_empty": int(recall_result_empty),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/agent_v5"))
    ap.add_argument("--batches", default="1-8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--skip-rollout-audit",
        action="store_true",
        help="Only simulate pass3 card/placement ratios; skip pass2/recall rollout checks.",
    )
    args = ap.parse_args()

    paths: List[Path] = []
    for batch in _parse_batches(args.batches):
        paths.extend(sorted((args.root / batch / "evidence_1b").glob("*.json")))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no evidence files found")

    results: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _simulate_one,
                str(p),
                int(args.seed),
                not bool(args.skip_rollout_audit),
            )
            for p in paths
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"simulated {i}/{len(paths)}", flush=True)

    summary = _merge(results)
    summary["thresholds"] = {
        "recall_memory_gap_overlap_max": float(
            os.environ.get("THINKSTREAM_RECALL_MEMORY_GAP_OVERLAP_MAX", "0.95")
        ),
        "recall_memory_gap_min_age": int(
            os.environ.get("THINKSTREAM_RECALL_MEMORY_GAP_MIN_AGE", "60")
        ),
        "sft_silent_to_active_ratio": SFT_SILENT_TO_ACTIVE_RATIO,
        "sft_pending_silent_fraction": SFT_PENDING_SILENT_FRACTION,
        "sft_post_answer_silent_fraction": SFT_POST_ANSWER_SILENT_FRACTION,
        "sft_multi_emit_to_other_response_ratio": SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
