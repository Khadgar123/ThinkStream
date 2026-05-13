"""Pass 3-B — Placement + Trajectory planning (v2 model-agnostic).

Card family now represents the question/reasoning type, not a fixed
availability profile. pass3b generates several availability candidates per
single-emit card (current/direct, memory_direct, recall, and selected
future/wait variants), applies a cheap evidence-based recall necessity
refinement, then greedily selects one non-overlapping multi-question
trajectory per video.

Pipeline contract preserved:
  compute_all_placements(cards, rollout, evidence, client, video_id) -> List[Dict]
  plan_trajectories(placements, cards_map, num_chunks, evidence, seed) -> List[Dict]
  save_placements / load_placements
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PLACEMENTS_DIR, VISUAL_WINDOW_CHUNKS
from .pass3a_cards import dict_to_card
from .stable_hash import stable_seed
from .placement.design import (
    Placement,
    adaptive_q_count,
    assign_recall_noise,
    place_card,
    placement_timing_verdict,
    refine_placements_with_evidence,
    select_trajectory,
)

logger = logging.getLogger(__name__)


RECALL_MEMORY_GAP_MIN_AGE = int(
    os.environ.get("THINKSTREAM_RECALL_MEMORY_GAP_MIN_AGE", str(VISUAL_WINDOW_CHUNKS + 1))
)


def _refine_selected_recall_with_rollout(
    placements: List[Placement],
    cards_map: Dict[str, Dict],
    rollout: Optional[Dict],
    *,
    video_id: str = "",
) -> Dict[str, int]:
    """Keep only recall slots with historical evidence outside the KV window.

    Pass3B already receives pass2 rollout, so it can cheaply reject low-value
    recall before pass3c rendering. Compact text memory is lossy state, not a
    substitute for old visual KV, so recall necessity is determined by evidence
    age/retrievability rather than answer-word overlap with memory text.
    """
    stats = {
        "slots_seen": 0,
        "kept_memory_gap": 0,
        "kept_teacher_hardening": 0,
        "kept_wait_history": 0,
        "dropped_invalid_query": 0,
        "dropped_empty_history": 0,
        "dropped_future_leak": 0,
        "dropped_no_memory_gap": 0,
    }
    if not rollout:
        return stats

    # Local import avoids making pass3b import pass3c at module load time.
    from .pass3c_samples import (
        RECALL_RETURN_CHUNKS,
        _is_unanswerable_card,
        _recall_query_available,
        _recall_query_for,
        _recall_result_for,
        _recall_wait_query_for,
        _repair_recall_query_for_response,
        _support_chunks,
        _support_chunks_before,
        _valid_recall_query,
        bm25_retrieve,
        select_recall_chunks,
    )

    def archive_before(current_chunk: int) -> List[Dict]:
        archive: List[Dict] = []
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
                "time": f"{ci}-{ci + 1}",
                "text": text,
            })
        return archive

    def has_kv_age_gap(chunks: List[int], current_chunk: int) -> bool:
        """True when any chunk is strictly older than the visual KV window."""
        for ch in chunks:
            try:
                ci = int(ch)
            except (TypeError, ValueError):
                continue
            if int(current_chunk) - ci >= int(RECALL_MEMORY_GAP_MIN_AGE):
                return True
        return False

    def raw_bm25_chunks(query: Dict, current_chunk: int) -> List[int]:
        retrieved = bm25_retrieve(
            query,
            archive_before(current_chunk),
            max_results=RECALL_RETURN_CHUNKS,
        )
        out: List[int] = []
        for c in select_recall_chunks(retrieved.get("returned_chunks") or []):
            try:
                out.append(int(c))
            except (TypeError, ValueError):
                continue
        return out

    def drop_slot(p: Placement, c: int, reason: str) -> None:
        p.recall_at.pop(c, None)
        p.recall_reason_at[c] = f"dropped:{reason}"
        stats[f"dropped_{reason}"] = stats.get(f"dropped_{reason}", 0) + 1

    for p in placements:
        if not p.recall_at:
            continue
        card = cards_map.get(p.card_id) or {}
        for c in sorted(list(p.recall_at.keys())):
            stats["slots_seen"] += 1
            kind = str(p.recall_at.get(c) or "")
            if kind == "not_yet":
                support_all = set(_support_chunks(card))
                if support_all and max(support_all) < int(c):
                    stats["dropped_empty_history"] += 1
                    drop_slot(p, int(c), "answer_support_already_past")
                    continue
                rq = _recall_wait_query_for(card, int(c))
                if not _valid_recall_query(rq):
                    stats["dropped_invalid_query"] += 1
                    drop_slot(p, int(c), "invalid_wait_query")
                    continue
                chunks = raw_bm25_chunks(rq, int(c))
                if not chunks:
                    stats["dropped_empty_history"] += 1
                    drop_slot(p, int(c), "empty_wait_history")
                    continue
                if max(chunks) >= int(c):
                    stats["dropped_future_leak"] += 1
                    drop_slot(p, int(c), "wait_future_leak")
                    continue
                if not has_kv_age_gap(chunks, int(c)):
                    stats["dropped_no_memory_gap"] += 1
                    drop_slot(p, int(c), "wait_history_still_visual")
                    continue
                p.recall_reason_at[int(c)] = (
                    p.recall_reason_at.get(int(c), "") or "elapsed_history_check"
                )
                stats["kept_wait_history"] += 1
                continue

            rq = _recall_query_for(card, int(c))
            rq = _repair_recall_query_for_response(card, rq, int(c))
            if not _recall_query_available(rq, int(c)):
                stats["dropped_invalid_query"] += 1
                drop_slot(p, int(c), "invalid_response_query")
                continue
            absence_check = _is_unanswerable_card(card)
            chunks = raw_bm25_chunks(rq, int(c))
            if not chunks:
                if absence_check:
                    rr = _recall_result_for(
                        card,
                        rollout,
                        kind,
                        current_chunk=int(c),
                        recall_query=rq,
                    )
                    chunks = select_recall_chunks(rr.get("returned_chunks") or [])
                if not chunks:
                    stats["dropped_empty_history"] += 1
                    drop_slot(p, int(c), "empty_response_history")
                    continue
            if max(chunks) >= int(c):
                stats["dropped_future_leak"] += 1
                drop_slot(p, int(c), "response_future_leak")
                continue
            if absence_check:
                p.recall_reason_at[int(c)] = (
                    p.recall_reason_at.get(int(c), "") or "hld_absence_evidence_check"
                )
                stats["kept_hld_absence"] = stats.get("kept_hld_absence", 0) + 1
                continue

            recall_reason = str(p.recall_reason_at.get(int(c), "") or "")
            multi_event_history_probe = (
                p.mechanism in {"multi_emit", "silent_then_response"}
                and recall_reason in {
                    "cumulative_history",
                    "status_history",
                    "future_answer_historical_anchor",
                }
            )
            if multi_event_history_probe:
                support = set(_support_chunks_before(card, int(c)))
                if not has_kv_age_gap(list(support or set(chunks)), int(c)):
                    stats["dropped_no_memory_gap"] += 1
                    drop_slot(p, int(c), "multi_event_history_still_visual")
                    continue
                if support and not (support & set(chunks)):
                    rr = _recall_result_for(
                        card,
                        rollout,
                        kind,
                        current_chunk=int(c),
                        recall_query=rq,
                    )
                    try:
                        rr_chunks = {int(x) for x in rr.get("returned_chunks") or []}
                    except (TypeError, ValueError):
                        rr_chunks = set()
                    if not rr_chunks or not (support & rr_chunks):
                        stats["dropped_empty_history"] += 1
                        drop_slot(p, int(c), "support_not_retrievable")
                        continue
                p.recall_reason_at[int(c)] = recall_reason or "multi_event_history"
                stats["kept_multi_event_history"] = (
                    stats.get("kept_multi_event_history", 0) + 1
                )
                continue

            visual_verification_probe = (
                p.mechanism == "memory_direct"
                and str(getattr(p, "recall_need", "") or "")
                == "memory_direct_visual_verification"
            )
            support_before = _support_chunks_before(card, int(c))
            if not has_kv_age_gap(support_before or chunks, int(c)):
                stats["dropped_no_memory_gap"] += 1
                drop_slot(p, int(c), "response_history_still_visual")
                continue

            support = set(_support_chunks_before(card, int(c)))
            if support and not (support & set(chunks)):
                rr = _recall_result_for(
                    card,
                    rollout,
                    kind,
                    current_chunk=int(c),
                    recall_query=rq,
                )
                try:
                    rr_chunks = {int(x) for x in rr.get("returned_chunks") or []}
                except (TypeError, ValueError):
                    rr_chunks = set()
                if not rr_chunks or not (support & rr_chunks):
                    stats["dropped_empty_history"] += 1
                    drop_slot(p, int(c), "support_not_retrievable")
                    continue

            p.recall_reason_at[int(c)] = (
                p.recall_reason_at.get(int(c), "") or "memory_gap_historical_evidence"
            )
            if visual_verification_probe:
                p.mechanism = "recall_demo"
                p.difficulty_mode = "visual_verification_recall"
                stats["kept_visual_verification"] = (
                    stats.get("kept_visual_verification", 0) + 1
                )
                stats["promoted_visual_verification_recall"] = (
                    stats.get("promoted_visual_verification_recall", 0) + 1
                )
            else:
                stats["kept_memory_gap"] += 1

    if any(v for k, v in stats.items() if k != "slots_seen"):
        logger.info("[%s] 3b recall rollout refinement: %s", video_id, stats)
    return stats


# ---------------------------------------------------------------------------
# Placement ↔ dict serialization
# ---------------------------------------------------------------------------


def _placement_to_dict(p: Placement) -> Dict:
    return {
        "card_id": p.card_id,
        "ask_chunk": int(p.ask_chunk),
        "mechanism": p.mechanism,
        "difficulty_mode": getattr(p, "difficulty_mode", ""),
        "recall_need": getattr(p, "recall_need", ""),
        "chunk_actions": {str(k): list(v) for k, v in p.chunk_actions.items()},
        "recall_at": {str(k): v for k, v in p.recall_at.items()},
        "recall_reason_at": {str(k): v for k, v in p.recall_reason_at.items()},
    }


def _dict_to_placement(d: Dict) -> Placement:
    return Placement(
        card_id=d["card_id"],
        ask_chunk=int(d["ask_chunk"]),
        mechanism=d["mechanism"],
        difficulty_mode=d.get("difficulty_mode", ""),
        recall_need=d.get("recall_need", ""),
        chunk_actions={int(k): tuple(v) for k, v in d.get("chunk_actions", {}).items()},
        recall_at={int(k): v for k, v in d.get("recall_at", {}).items()},
        recall_reason_at={int(k): v for k, v in d.get("recall_reason_at", {}).items()},
    )


def _placement_crosses_compress_boundary(
    placement: Placement,
    compression_boundaries: List[int],
) -> bool:
    """True if a compact-memory boundary would split a question episode."""
    if not compression_boundaries:
        return False
    response_chunks = [
        int(c) for c, action in placement.chunk_actions.items()
        if action and str(action[0]) == "response"
    ]
    if not response_chunks:
        return False
    ask = int(placement.ask_chunk)
    last_answer = max(response_chunks)
    return any(ask < int(boundary) <= last_answer for boundary in compression_boundaries)


# ---------------------------------------------------------------------------
# Public API (matches old pass3b interface)
# ---------------------------------------------------------------------------


async def compute_all_placements(
    cards: List[Dict],
    rollout: Dict,
    evidence: List[Dict],
    client=None,                     # kept for API compat (no LLM call now)
    video_id: str = "",
    seed: int = 42,
) -> List[Dict]:
    """Generate ALL candidate placements (multiple tiers per card).

    No LLM calls. Profile-driven, model-agnostic — depends only on each
    card's (gold_emits, grounding_frames) and the video length.
    """
    rng = random.Random(stable_seed(seed, video_id, modulo=1_000_000))
    num_chunks = int(rollout.get("num_chunks", 0))
    cards_obj = [dict_to_card(c) for c in cards]

    all_placements: List[Placement] = []
    rejected: Dict[str, int] = {}
    for card in cards_obj:
        plcs = refine_placements_with_evidence(
            card, place_card(card, num_chunks, rng), evidence
        )
        for p in plcs:
            ok, reason = placement_timing_verdict(card, p)
            if ok:
                all_placements.append(p)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
    if rejected:
        logger.info(
            f"[{video_id}] 3b placement timing rejected: "
            f"{dict(sorted(rejected.items(), key=lambda x: -x[1]))}"
        )
    return [_placement_to_dict(p) for p in all_placements]


def plan_trajectories(
    placements: List[Dict],
    cards_map: Dict[str, Dict] = None,
    num_chunks: int = 60,
    evidence: List[Dict] = None,     # unused in v2; kept for API compat
    rollout: Dict = None,
    video_id: str = "",
    seed: int = 42,
    **_,
) -> List[Dict]:
    """Select MAX_QUESTIONS_PER_TRAJECTORY placements for ONE trajectory.

    v2: 1 trajectory/video × adaptive 6-14 questions (q-interval ~14s).
    Diversity-weighted greedy selection across family / mechanism /
    answer_form / ask_chunk spread.
    """
    if not placements:
        return []
    cards_map = cards_map or {}
    rng = random.Random(seed)

    placement_objs = [_dict_to_placement(p) for p in placements]
    placements_by_card: Dict[str, List[Placement]] = {}
    for p in placement_objs:
        placements_by_card.setdefault(p.card_id, []).append(p)

    cards_obj = [dict_to_card(c) for c in cards_map.values()]
    cards_by_id = {c.card_id: c for c in cards_obj}
    filtered_by_card: Dict[str, List[Placement]] = {}
    rejected: Dict[str, int] = {}
    compression_boundaries = [
        int(e.get("trigger_chunk"))
        for e in (rollout or {}).get("compression_events", [])
        if e.get("trigger_chunk") is not None
    ]
    for cid, plcs in placements_by_card.items():
        card = cards_by_id.get(cid)
        if not card:
            continue
        for p in plcs:
            if _placement_crosses_compress_boundary(p, compression_boundaries):
                rejected["crosses_compress_boundary"] = (
                    rejected.get("crosses_compress_boundary", 0) + 1
                )
                continue
            ok, reason = placement_timing_verdict(card, p)
            if ok:
                filtered_by_card.setdefault(cid, []).append(p)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
    if rejected:
        logger.info(
            "3b selected-placement input timing rejected: "
            f"{dict(sorted(rejected.items(), key=lambda x: -x[1]))}"
        )
    target_q = adaptive_q_count(num_chunks)
    selected = select_trajectory(
        cards_obj, filtered_by_card, num_chunks, rng, max_q=target_q,
    )
    assign_recall_noise(selected, rng, cards_by_id={c.card_id: c for c in cards_obj})
    _refine_selected_recall_with_rollout(
        selected,
        cards_map,
        rollout,
        video_id=video_id,
    )

    if not selected:
        return []
    # 1 trajectory per video — return single-element list to match
    # pipeline expectation that plan_trajectories yields a list.
    return [{
        "trajectory_id": f"traj_0",
        "placements": [_placement_to_dict(p) for p in selected],
        "n_questions": len(selected),
        "ask_chunks": sorted([p.ask_chunk for p in selected]),
    }]


def save_placements(video_id: str, data: Dict,
                    output_dir: Path = PLACEMENTS_DIR) -> None:
    """Save {placements: [...], trajectories: [...]} for a video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )


def load_placements(video_id: str,
                    placements_dir: Path = PLACEMENTS_DIR) -> Optional[Dict]:
    from .cache_version import stage_version_ok
    allow_stale = os.environ.get("THINKSTREAM_ALLOW_STALE_PASS3_CACHE", "").lower() in {"1", "true", "yes", "on"}
    if not allow_stale and not stage_version_ok("3b"):
        return None
    p = placements_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Back-compat helpers (used by pass3e_verify)
# ---------------------------------------------------------------------------

import re as _re


def _keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in text. Used by pass3e_verify."""
    if not keywords:
        return 0.0
    text_words = set(_re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
    found = sum(1 for kw in keywords if kw in text_words)
    return found / len(keywords)
