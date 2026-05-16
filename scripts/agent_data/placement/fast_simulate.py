"""Fast pass3 distribution simulator.

This intentionally stops at card / placement / recall scheduling. It does not
render pass3c messages, so it is much faster than ``v2.simulate`` and is meant
for threshold tuning on large batches.

Usage:
  python -m scripts.agent_data.placement.fast_simulate \
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
    _card_evidence_type,
    _card_question_way,
    _card_task_subtype,
    _chunk_position_bin,
    _placement_answer_mode,
    _placement_message_cost,
    _placement_response_bin,
    _placement_response_source,
    _placement_timing_bucket,
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


def _length_bucket(num_chunks: int) -> str:
    if int(num_chunks) < 64:
        return "short"
    if int(num_chunks) < 180:
        return "mid"
    return "long"


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
    if end_s < start_s:
        return []
    start_chunk = int(start_s / float(AGENT_CHUNK_SEC))
    end_chunk = int(end_s / float(AGENT_CHUNK_SEC))
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
    # exact deterministic time-range recall path without paying this import cost
    # when rollout auditing is disabled.
    from ..pass3a_cards import _card_to_dict
    from ..pass3c_samples import (
        _current_context_text_for_chunk,
        _memory_text_for_chunk,
        _needs_recall_hardening,
        _recall_chunks_for_request,
        _recall_query_available,
        _recall_query_for,
        _recall_result_for,
        _recall_wait_query_for,
        _repair_recall_query_for_response,
        _support_chunks_before,
        _valid_recall_query,
        select_recall_chunks_uniform,
    )

    counters = Counter()
    by_kind = Counter()
    by_mechanism = Counter()
    by_family = Counter()
    reasons = Counter()
    returned_counts: List[int] = []
    time_range_raw_counts: List[int] = []
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
            raw_chunks = select_recall_chunks_uniform(_recall_chunks_for_request(rollout, rq, c))
            time_range_raw_counts.append(len(raw_chunks))
            if raw_chunks:
                counters["time_range_raw_nonempty"] += 1
            else:
                counters["time_range_raw_empty"] += 1

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
                counters["time_range_adjusted_by_noise"] += 1
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
        "recall_time_range_raw_chunk_counts": time_range_raw_counts,
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
    mechanism = Counter()
    timing_bucket = Counter()
    question_style = Counter()
    question_way = Counter()
    evidence_type = Counter()
    task_subtype = Counter()
    family_source = Counter()
    task_subtype_source = Counter()
    question_way_source = Counter()
    answer_mode = Counter()
    task_mode = Counter()
    ask_bin = Counter()
    response_bin = Counter()
    response_rows_by_source = Counter()
    response_rows_by_task_source = Counter()
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
    multi_response_counts: List[int] = []
    multi_response_counts_by_family: Dict[str, List[int]] = {}
    multi_answer_gaps: List[int] = []
    multi_answer_spans: List[int] = []
    multi_ask_to_first: List[int] = []
    multi_answer_gaps_by_family: Dict[str, List[int]] = {}
    future_ask_to_answer: List[int] = []
    future_active_spans: List[int] = []
    q_with_recall = 0
    multi_recall_q = 0
    forward_recall_q = 0

    for p in selected:
        card = cards_by_id[p.card_id]
        mechanism[p.mechanism] += 1
        source = _placement_response_source(p, card)
        cost = _placement_message_cost(p, card)
        subtype = _card_task_subtype(card)
        way = _card_question_way(card)
        mode = _placement_answer_mode(p, card)
        timing_bucket[_placement_timing_bucket(p, card)] += 1
        question_style[str(getattr(card, "question_style", "") or "benchmark_core")] += 1
        question_way[way] += 1
        evidence_type[_card_evidence_type(card)] += 1
        task_subtype[subtype] += 1
        answer_mode[mode] += 1
        task_mode[f"{subtype}|{mode}"] += 1
        ask_bin[str(_chunk_position_bin(int(p.ask_chunk), num_chunks))] += 1
        rb = _placement_response_bin(p, num_chunks)
        if rb is not None:
            response_bin[str(rb)] += 1
        family[card.family] += 1
        family_source[f"{card.family}|{source}"] += 1
        task_subtype_source[f"{subtype}|{source}"] += 1
        question_way_source[f"{way}|{source}"] += 1
        response_rows_by_source["direct"] += cost.direct_response_rows
        response_rows_by_source["recall"] += cost.recall_response_rows
        response_rows_by_source["hld_recall"] += cost.hld_recall_response_rows
        response_rows_by_source["future"] += cost.future_response_rows
        response_rows_by_source["multi"] += cost.multi_response_rows
        if cost.direct_response_rows:
            response_rows_by_task_source[f"{subtype}|direct"] += cost.direct_response_rows
        if cost.recall_response_rows:
            response_rows_by_task_source[f"{subtype}|recall"] += cost.recall_response_rows
        if cost.hld_recall_response_rows:
            response_rows_by_task_source[f"{subtype}|hld_recall"] += cost.hld_recall_response_rows
        if cost.future_response_rows:
            response_rows_by_task_source[f"{subtype}|future"] += cost.future_response_rows
        if cost.multi_response_rows:
            response_rows_by_task_source[f"{subtype}|multi"] += cost.multi_response_rows
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
        if p.mechanism == "multi_emit":
            multi_response_counts.append(len(response_chunks))
            multi_response_counts_by_family.setdefault(card.family, []).append(len(response_chunks))
            if response_chunks:
                gaps = [
                    response_chunks[i + 1] - response_chunks[i]
                    for i in range(len(response_chunks) - 1)
                ]
                multi_answer_gaps.extend(gaps)
                multi_answer_gaps_by_family.setdefault(card.family, []).extend(gaps)
                multi_answer_spans.append(max(response_chunks) - min(response_chunks) + 1)
                multi_ask_to_first.append(min(response_chunks) - int(p.ask_chunk))
        if source == "future" and response_chunks:
            future_ask_to_answer.append(min(response_chunks) - int(p.ask_chunk))
            future_active_spans.append(max(response_chunks) - int(p.ask_chunk) + 1)

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
        elif kind == "response":
            raw_action["response"] += 1
        elif kind == "recall+response":
            raw_action["recall_response"] += 1
        elif kind == "recall+silent":
            raw_action["recall_silent"] += 1
            phase = phase_for_wait(ds)
            wait_phase[f"recall_silent:{phase}"] += 1
            wait_phase_by_mechanism[f"{ds.mechanism}|recall_silent:{phase}"] += 1

        if kind in {"response", "recall+response"} and ds.card_id:
            query_answers[ds.card_id] = query_answers.get(ds.card_id, 0) + 1
            if c >= open_until_by_card.get(ds.card_id, c):
                query_status[ds.card_id] = "answered"
            else:
                query_status[ds.card_id] = "open"

    ask_chunks = sorted(int(p.ask_chunk) for p in selected)
    intervals = [
        ask_chunks[i + 1] - ask_chunks[i] for i in range(len(ask_chunks) - 1)
    ]
    total_rows = sum(sample.values())
    total_response_rows = sample.get("response", 0) + sample.get("recall+response", 0)
    out = {
        "videos": 1,
        "num_chunks": int(num_chunks),
        "row_count": int(total_rows),
        "response_row_count": int(total_response_rows),
        "length_bucket": _length_bucket(num_chunks),
        "q_counts": [len(selected)],
        "q_intervals": intervals,
        "first_waits": first_waits,
        "wait_not_yet_per_forward": wait_not_yet_per_forward,
        "f5_recall_per_q": f5_recall_per_q,
        "status_recall_per_q": status_recall_per_q,
        "multi_response_counts": multi_response_counts,
        "multi_response_counts_by_family": multi_response_counts_by_family,
        "multi_answer_gaps": multi_answer_gaps,
        "multi_answer_spans": multi_answer_spans,
        "multi_ask_to_first": multi_ask_to_first,
        "multi_answer_gaps_by_family": multi_answer_gaps_by_family,
        "future_ask_to_answer": future_ask_to_answer,
        "future_active_spans": future_active_spans,
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
        "mechanism": dict(mechanism),
        "timing_bucket": dict(timing_bucket),
        "question_style": dict(question_style),
        "question_way": dict(question_way),
        "evidence_type": dict(evidence_type),
        "task_subtype": dict(task_subtype),
        "family_source": dict(family_source),
        "task_subtype_source": dict(task_subtype_source),
        "question_way_source": dict(question_way_source),
        "answer_mode": dict(answer_mode),
        "task_mode": dict(task_mode),
        "ask_bin": dict(ask_bin),
        "response_bin": dict(response_bin),
        "response_rows_by_source": dict(response_rows_by_source),
        "response_rows_by_task_source": dict(response_rows_by_task_source),
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
    mechanism = Counter()
    timing_bucket = Counter()
    question_style = Counter()
    question_way = Counter()
    evidence_type = Counter()
    task_subtype = Counter()
    family_source = Counter()
    task_subtype_source = Counter()
    question_way_source = Counter()
    answer_mode = Counter()
    task_mode = Counter()
    ask_bin = Counter()
    response_bin = Counter()
    response_rows_by_source = Counter()
    response_rows_by_task_source = Counter()
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
    length_bucket_videos = Counter()
    length_bucket_rows = Counter()
    length_bucket_response_rows = Counter()
    q_counts: List[float] = []
    q_intervals: List[float] = []
    first_waits: List[float] = []
    wait_not_yet: List[float] = []
    f5_recall: List[float] = []
    status_recall: List[float] = []
    multi_response_counts: List[float] = []
    multi_response_counts_by_family: Dict[str, List[float]] = {}
    multi_answer_gaps: List[float] = []
    multi_answer_spans: List[float] = []
    multi_ask_to_first: List[float] = []
    multi_answer_gaps_by_family: Dict[str, List[float]] = {}
    future_ask_to_answer: List[float] = []
    future_active_spans: List[float] = []
    comp_range_sizes: List[float] = []
    comp_duration_chunks: List[float] = []
    comp_trigger_lag_chunks: List[float] = []
    comp_post_tokens: List[float] = []
    comp_summary_words: List[float] = []
    recall_returned_counts: List[float] = []
    recall_time_range_raw_counts: List[float] = []
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
        bucket = str(r.get("length_bucket") or _length_bucket(int(r.get("num_chunks", 0) or 0)))
        length_bucket_videos[bucket] += int(r.get("videos", 0))
        length_bucket_rows[bucket] += int(r.get("row_count", 0))
        length_bucket_response_rows[bucket] += int(r.get("response_row_count", 0))
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
        multi_response_counts.extend(r.get("multi_response_counts") or [])
        for family_name, counts in (r.get("multi_response_counts_by_family") or {}).items():
            multi_response_counts_by_family.setdefault(str(family_name), []).extend(counts or [])
        multi_answer_gaps.extend(r.get("multi_answer_gaps") or [])
        multi_answer_spans.extend(r.get("multi_answer_spans") or [])
        multi_ask_to_first.extend(r.get("multi_ask_to_first") or [])
        for family_name, gaps in (r.get("multi_answer_gaps_by_family") or {}).items():
            multi_answer_gaps_by_family.setdefault(str(family_name), []).extend(gaps or [])
        future_ask_to_answer.extend(r.get("future_ask_to_answer") or [])
        future_active_spans.extend(r.get("future_active_spans") or [])
        comp_range_sizes.extend(r.get("compression_range_sizes") or [])
        comp_duration_chunks.extend(r.get("compression_duration_chunks") or [])
        comp_trigger_lag_chunks.extend(r.get("compression_trigger_lag_chunks") or [])
        comp_post_tokens.extend(r.get("compression_post_tokens") or [])
        comp_summary_words.extend(r.get("compression_summary_words") or [])
        recall_returned_counts.extend(r.get("recall_returned_chunk_counts") or [])
        recall_time_range_raw_counts.extend(r.get("recall_time_range_raw_chunk_counts") or [])
        sample.update(r.get("sample") or {})
        raw_action.update(r.get("raw_action") or {})
        silent_role.update(r.get("silent_role") or {})
        wait_phase.update(r.get("wait_phase") or {})
        wait_phase_by_mechanism.update(r.get("wait_phase_by_mechanism") or {})
        sample_kind_by_mechanism.update(r.get("sample_kind_by_mechanism") or {})
        mechanism.update(r.get("mechanism") or {})
        timing_bucket.update(r.get("timing_bucket") or {})
        question_style.update(r.get("question_style") or {})
        question_way.update(r.get("question_way") or {})
        evidence_type.update(r.get("evidence_type") or {})
        task_subtype.update(r.get("task_subtype") or {})
        family_source.update(r.get("family_source") or {})
        task_subtype_source.update(r.get("task_subtype_source") or {})
        question_way_source.update(r.get("question_way_source") or {})
        answer_mode.update(r.get("answer_mode") or {})
        task_mode.update(r.get("task_mode") or {})
        ask_bin.update(r.get("ask_bin") or {})
        response_bin.update(r.get("response_bin") or {})
        response_rows_by_source.update(r.get("response_rows_by_source") or {})
        response_rows_by_task_source.update(r.get("response_rows_by_task_source") or {})
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
    raw_direct_answer_rows = int(sample.get("response", 0))
    raw_recall_answer_rows = int(sample.get("recall+response", 0))
    raw_answer_rows = raw_direct_answer_rows + raw_recall_answer_rows
    normalized_response_source = Counter(response_rows_by_source)
    if normalized_response_source.get("hld_recall"):
        normalized_response_source["recall"] += normalized_response_source.pop("hld_recall")
    response_by_length = {
        bucket: {
            "videos": int(length_bucket_videos.get(bucket, 0)),
            "rows": int(length_bucket_rows.get(bucket, 0)),
            "response_rows": int(length_bucket_response_rows.get(bucket, 0)),
            "response_pct": round(
                length_bucket_response_rows.get(bucket, 0)
                / max(length_bucket_rows.get(bucket, 0), 1)
                * 100.0,
                2,
            ),
        }
        for bucket in ("short", "mid", "long")
        if length_bucket_videos.get(bucket, 0)
    }
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
        "timing_bucket": _json_counter(timing_bucket),
        "timing_bucket_pct": _pct(timing_bucket),
        "question_style": _json_counter(question_style),
        "question_style_pct": _pct(question_style),
        "question_way": _json_counter(question_way),
        "question_way_pct": _pct(question_way),
        "evidence_type": _json_counter(evidence_type),
        "evidence_type_pct": _pct(evidence_type),
        "task_subtype": _json_counter(task_subtype),
        "task_subtype_pct": _pct(task_subtype),
        "question_source_by_family": _json_counter(family_source),
        "question_source_by_task_subtype": _json_counter(task_subtype_source),
        "question_source_by_question_way": _json_counter(question_way_source),
        "answer_mode": _json_counter(answer_mode),
        "answer_mode_pct": _pct(answer_mode),
        "task_answer_mode": _json_counter(task_mode),
        "task_answer_mode_pct": _pct(task_mode),
        "ask_position_bin": _json_counter(ask_bin),
        "ask_position_bin_pct": _pct(ask_bin),
        "response_position_bin": _json_counter(response_bin),
        "response_position_bin_pct": _pct(response_bin),
        "response_rows_by_source": _json_counter(response_rows_by_source),
        "response_rows_by_source_pct": _pct(response_rows_by_source),
        "response_rows_by_source_normalized": _json_counter(normalized_response_source),
        "response_rows_by_source_normalized_pct": _pct(normalized_response_source),
        "response_rows_by_task_source": _json_counter(response_rows_by_task_source),
        "response_rows_by_task_source_pct": _pct(response_rows_by_task_source),
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
        "raw_sample_action": _json_counter(raw_action),
        "raw_sample_action_pct": _pct(raw_action),
        "raw_silent_role": _json_counter(silent_role),
        "raw_silent_role_pct": _pct(silent_role),
        "raw_wait_phase": _json_counter(wait_phase),
        "raw_wait_phase_pct": _pct(wait_phase),
        "raw_wait_phase_by_mechanism": _json_counter(wait_phase_by_mechanism),
        "sample_kind_by_mechanism": _json_counter(sample_kind_by_mechanism),
        "answer_rows_direct_vs_recall_raw": {
            "direct_answer_rows": raw_direct_answer_rows,
            "recall_after_answer_rows": raw_recall_answer_rows,
            "total_answer_rows": raw_answer_rows,
            "direct_answer_pct": round(
                raw_direct_answer_rows / max(raw_answer_rows, 1) * 100, 2
            ),
            "recall_after_answer_pct": round(
                raw_recall_answer_rows / max(raw_answer_rows, 1) * 100, 2
            ),
        },
        "questions_direct_vs_recall": {
            "direct_or_no_recall_questions": int(placements_total - q_with_recall),
            "recall_after_questions": int(q_with_recall),
            "total_questions": int(placements_total),
            "direct_or_no_recall_pct": round(
                (placements_total - q_with_recall) / max(placements_total, 1) * 100, 2
            ),
            "recall_after_pct": round(
                q_with_recall / max(placements_total, 1) * 100, 2
            ),
        },
        "response_pct": round(responses / max(rows, 1) * 100, 2),
        "response_pct_by_length_bucket": response_by_length,
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
        "multi_emit_response_count": _stats(multi_response_counts),
        "multi_emit_response_count_hist": _json_counter(
            Counter(str(int(x)) for x in multi_response_counts)
        ),
        "multi_emit_response_count_by_family": {
            family_name: _stats(counts)
            for family_name, counts in sorted(multi_response_counts_by_family.items())
        },
        "multi_answer_gap_chunks": _stats(multi_answer_gaps),
        "multi_answer_span_chunks": _stats(multi_answer_spans),
        "multi_ask_to_first_answer_chunks": _stats(multi_ask_to_first),
        "multi_answer_gap_chunks_by_family": {
            family_name: _stats(gaps)
            for family_name, gaps in sorted(multi_answer_gaps_by_family.items())
        },
        "future_ask_to_answer_chunks": _stats(future_ask_to_answer),
        "future_active_span_chunks": _stats(future_active_spans),
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
            "time_range_raw_nonempty": int(recall_audit.get("time_range_raw_nonempty", 0)),
            "time_range_raw_nonempty_rate": round(
                int(recall_audit.get("time_range_raw_nonempty", 0))
                / max(int(recall_audit.get("recall_slots", 0)), 1),
                4,
            ),
            "time_range_raw_empty": int(recall_audit.get("time_range_raw_empty", 0)),
            "time_range_adjusted_by_noise": int(
                recall_audit.get("time_range_adjusted_by_noise", 0)
            ),
            "fallback_filled_history_frames": int(
                recall_audit.get("fallback_filled_history_frames", 0)
            ),
            "history_frame_result_nonempty": int(
                recall_audit.get("history_frame_result_nonempty", 0)
            ),
            "returned_chunk_count": _stats(recall_returned_counts),
            "time_range_raw_chunk_count": _stats(recall_time_range_raw_counts),
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
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
