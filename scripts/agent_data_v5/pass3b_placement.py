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

from .config import PLACEMENTS_DIR
from .pass3a_cards import dict_to_card
from .stable_hash import stable_seed
from .v2.design import (
    Placement,
    RECENT_THINKS_HORIZON,
    adaptive_q_count,
    assign_recall_noise,
    place_card,
    placement_timing_verdict,
    refine_placements_with_evidence,
    select_trajectory,
)

logger = logging.getLogger(__name__)


RECALL_MEMORY_GAP_OVERLAP_MAX = float(
    os.environ.get("THINKSTREAM_RECALL_MEMORY_GAP_OVERLAP_MAX", "0.95")
)
RECALL_MEMORY_GAP_MIN_AGE = int(
    os.environ.get("THINKSTREAM_RECALL_MEMORY_GAP_MIN_AGE", str(RECENT_THINKS_HORIZON))
)


def _refine_selected_recall_with_rollout(
    placements: List[Placement],
    cards_map: Dict[str, Dict],
    rollout: Optional[Dict],
    *,
    video_id: str = "",
) -> Dict[str, int]:
    """Keep only recall slots that have a real memory gap or can be hardened.

    Pass3B already receives pass2 rollout, so it can cheaply reject low-value
    recall before pass3c rendering. This is intentionally conservative:
    - response recall is kept when current memory/context does not expose the
      answer and BM25 can return historical chunks;
    - if the current memory already exposes the answer, keep the slot only as a
      teacher-hardening candidate so pass3c can rewrite the card without
      changing timing;
    - wait-state recall is kept only when elapsed-history BM25 returns a real
      past chunk whose detailed text is not already in the current memory.
      Otherwise a plain silent row is better training signal.
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
        _current_context_text_for_chunk,
        _memory_text_for_chunk,
        _memory_overlap_score,
        _needs_recall_hardening,
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

    def archive_text_by_chunk(current_chunk: int) -> Dict[int, str]:
        return {
            int(item["chunk"]): str(item.get("text", ""))
            for item in archive_before(current_chunk)
        }

    def has_real_memory_gap(
        chunks: List[int],
        current_chunk: int,
        memory_text: str,
    ) -> bool:
        """True when a retrieved historical chunk is outside current memory.

        Recent pass2 thinks are already present in the model prompt; recalling
        them would teach unnecessary tool use. A recall slot should require
        either older history or compressed-away visual detail.
        """
        by_chunk = archive_text_by_chunk(current_chunk)
        for ch in chunks:
            try:
                ci = int(ch)
            except (TypeError, ValueError):
                continue
            if int(current_chunk) - ci <= int(RECALL_MEMORY_GAP_MIN_AGE):
                continue
            text = by_chunk.get(ci, "")
            if not text:
                continue
            if _memory_overlap_score(text, memory_text) <= RECALL_MEMORY_GAP_OVERLAP_MAX:
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
                memory_text = _memory_text_for_chunk(rollout, int(c))
                context_text = _current_context_text_for_chunk(
                    rollout,
                    int(c),
                    memory_text=memory_text,
                )
                if _needs_recall_hardening(card, context_text):
                    stats["dropped_empty_history"] += 1
                    drop_slot(p, int(c), "answer_visible_before_wait_response")
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
                if not has_real_memory_gap(chunks, int(c), memory_text):
                    stats["dropped_no_memory_gap"] += 1
                    drop_slot(p, int(c), "wait_history_already_in_memory")
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

            memory_text = _memory_text_for_chunk(rollout, int(c))
            context_text = _current_context_text_for_chunk(
                rollout,
                int(c),
                memory_text=memory_text,
            )
            visual_verification_probe = (
                p.mechanism == "memory_direct"
                and str(getattr(p, "recall_need", "") or "")
                == "memory_direct_visual_verification"
            )
            if p.mechanism == "recall_demo" and _needs_recall_hardening(card, context_text):
                p.recall_reason_at[int(c)] = "teacher_hardening_memory_gap"
                p.recall_need = "teacher_hardening_answer_visible_in_memory"
                stats["kept_teacher_hardening"] += 1
                continue
            if _needs_recall_hardening(card, context_text) and not visual_verification_probe:
                stats["dropped_no_memory_gap"] += 1
                drop_slot(p, int(c), "answer_visible_in_memory")
                continue
            if not has_real_memory_gap(chunks, int(c), memory_text) and not visual_verification_probe:
                stats["dropped_no_memory_gap"] += 1
                drop_slot(p, int(c), "response_history_already_in_memory")
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
                stats["kept_visual_verification"] = (
                    stats.get("kept_visual_verification", 0) + 1
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
    for cid, plcs in placements_by_card.items():
        card = cards_by_id.get(cid)
        if not card:
            continue
        for p in plcs:
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
    if not stage_version_ok("3b"):
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
