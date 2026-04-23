"""
Pass 3-B: Placement + Behavior Sequence Planning

For each task card, determines:
1. WHERE to ask (ask_chunk)
2. WHAT behavior sequence to expect (immediate_response / recall / event_watch / ...)
3. KEY CHUNKS in the sequence (where to generate training samples)

Then combines placements into multi-question trajectories.

Pure program — zero 397B calls.

Output: placements/{video_id}.json
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    PLACEMENTS_DIR,
    VISUAL_WINDOW_CHUNKS,
)
from .pass3a_cards import RETENTION_CLASS, extract_keywords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retention Bitmap (precompute per card)
# ---------------------------------------------------------------------------


def _keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in text."""
    if not keywords:
        return 0.0
    text_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))
    found = sum(1 for kw in keywords if kw in text_words)
    return found / len(keywords)


def precompute_retention(card: Dict, rollout: Dict) -> Dict:
    """Check if student thinks/summaries retained the answer.

    Only checks support_chunks (not all chunks).
    Returns: {"thinks_retained": {chunk: bool}, "summary_retained": {event_idx: bool}}
    """
    answer_kw = extract_keywords(card.get("canonical_answer", ""))
    retention_class = RETENTION_CLASS.get(card.get("family", ""), "medium")
    threshold = {"low": 0.5, "medium": 0.35, "high": 0.2}[retention_class]

    observations = rollout.get("thinks", [])
    thinks_retained = {}
    for chunk_idx in card.get("support_chunks", []):
        if chunk_idx < len(observations):
            think_text = observations[chunk_idx].get("think", "")
            thinks_retained[chunk_idx] = _keyword_overlap(think_text, answer_kw) > threshold
        else:
            thinks_retained[chunk_idx] = False

    summary_retained = {}
    support_set = set(card.get("support_chunks", []))
    for idx, event in enumerate(rollout.get("compression_events", [])):
        compressed = set(event.get("compressed_thinks_chunks", []))
        if support_set & compressed:
            summary_text = event.get("summary", {}).get("text", "")
            summary_retained[idx] = _keyword_overlap(summary_text, answer_kw) > 0.3

    return {"thinks_retained": thinks_retained, "summary_retained": summary_retained}


# ---------------------------------------------------------------------------
# Availability Classification
# ---------------------------------------------------------------------------


def classify_availability(
    card: Dict, ask_chunk: int, rollout: Dict, bitmap: Dict,
) -> str:
    """Determine answer availability at ask_chunk using structural lookups."""
    support_chunks = set(card.get("support_chunks", []))
    support_start = min(support_chunks) if support_chunks else 0
    support_end = max(support_chunks) if support_chunks else 0
    snapshot = rollout["snapshots"].get(ask_chunk, rollout["snapshots"].get(str(ask_chunk)))
    if snapshot is None:
        return "unavailable"

    # in_future
    if support_start > ask_chunk:
        return "in_future"

    # in_visual
    window_start = snapshot["visual_window_start"]
    window_end = snapshot["chunk_idx"]
    if any(window_start <= c <= window_end for c in support_chunks):
        return "in_visual"

    # in_recent_thinks
    recent_chunks = {item["chunk"] for item in snapshot.get("recent_thinks", [])}
    retained_present = support_chunks & recent_chunks
    if any(bitmap["thinks_retained"].get(c, False) for c in retained_present):
        return "in_recent_thinks"

    # in_compressed
    for idx, event in enumerate(rollout.get("compression_events", [])):
        if event["trigger_chunk"] > ask_chunk:
            break
        compressed = set(event.get("compressed_thinks_chunks", []))
        if support_chunks & compressed:
            if bitmap["summary_retained"].get(idx, False):
                return "in_compressed"

    # in_history_only
    if support_end < ask_chunk:
        return "in_history_only"

    return "unavailable"


# ---------------------------------------------------------------------------
# Behavior Sequence Types
# ---------------------------------------------------------------------------


def determine_sequence_type(card: Dict, availability: str) -> str:
    """Map availability + family → behavior sequence type."""
    if card.get("family") == "M1":
        return "multi_response"
    if availability == "in_future":
        return "event_watch"
    if availability in ("in_visual", "in_recent_thinks", "in_compressed"):
        return "immediate_response"
    if availability == "in_history_only":
        return "recall_success"
    return "immediate_response"


def _find_next_evidence_chunk(card: Dict, after_chunk: int, evidence: List[Dict]) -> Optional[int]:
    """Find next chunk after after_chunk where answer evidence appears."""
    answer_kw = extract_keywords(card.get("canonical_answer", ""))
    if not answer_kw:
        return None
    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        if idx <= after_chunk:
            continue
        for fact in cap.get("atomic_facts", []):
            if _keyword_overlap(fact.get("fact", ""), answer_kw) > 0.3:
                return idx
    return None


def compute_placement(
    card: Dict, ask_chunk: int, sequence_type: str,
    rollout: Dict, evidence: List[Dict],
) -> Optional[Dict]:
    """Compute behavior sequence blueprint (key_chunks)."""
    num_chunks = rollout["num_chunks"]
    key_chunks = {"ask": ask_chunk}

    if sequence_type == "immediate_response":
        key_chunks["post_silent"] = min(ask_chunk + 1, num_chunks - 1)

    elif sequence_type == "recall_success":
        key_chunks["post_recall"] = ask_chunk
        key_chunks["post_silent"] = min(ask_chunk + 1, num_chunks - 1)

    elif sequence_type == "recall_fail_then_found":
        key_chunks["post_recall"] = ask_chunk
        found = _find_next_evidence_chunk(card, ask_chunk, evidence)
        if found and found < num_chunks:
            key_chunks["wait_silent"] = [min(ask_chunk + 1, num_chunks - 1)]
            key_chunks["found_response"] = found
            key_chunks["post_silent"] = min(found + 1, num_chunks - 1)
        else:
            return None

    elif sequence_type == "event_watch":
        trigger = min(card.get("support_chunks", [ask_chunk]))
        if trigger <= ask_chunk:
            return None
        gap = trigger - ask_chunk
        wait_samples = list(range(ask_chunk + 2, trigger, max(1, gap // 3)))
        key_chunks["wait_silent"] = wait_samples[:2]
        key_chunks["trigger"] = trigger
        key_chunks["post_silent"] = min(trigger + 1, num_chunks - 1)

    elif sequence_type == "multi_response":
        ev_by_idx = {cap.get("chunk_idx", i): cap for i, cap in enumerate(evidence)}
        followup_r = []
        followup_s = []
        for c in range(ask_chunk + 1, min(num_chunks, ask_chunk + 30)):
            cap = ev_by_idx.get(c, {})
            if cap.get("state_changes"):
                followup_r.append(c)
            elif len(followup_s) < 2:
                followup_s.append(c)
        key_chunks["no_change_silent"] = followup_s[:2]
        key_chunks["followup_response"] = followup_r[:5]
        if followup_r:
            key_chunks["post_silent"] = min(followup_r[-1] + 1, num_chunks - 1)

    return {
        "card_id": card["card_id"],
        "ask_chunk": ask_chunk,
        "sequence_type": sequence_type,
        "key_chunks": key_chunks,
    }


# ---------------------------------------------------------------------------
# Compute all placements for a video
# ---------------------------------------------------------------------------


def compute_all_placements(
    cards: List[Dict], rollout: Dict, evidence: List[Dict],
) -> List[Dict]:
    """Compute all valid placements for all cards."""
    num_chunks = rollout["num_chunks"]
    all_placements = []

    for card in cards:
        vis_type = card.get("visibility_type", "transient")
        support_chunks = card.get("support_chunks", [])
        if not support_chunks:
            continue
        support_end = max(support_chunks)

        bitmap = precompute_retention(card, rollout) if vis_type == "transient" else {}

        # Candidate ask_chunks by visibility_type
        if vis_type == "persistent":
            candidates = [num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4]
            for ask in candidates:
                if 0 <= ask < num_chunks:
                    p = compute_placement(card, ask, "immediate_response", rollout, evidence)
                    if p:
                        all_placements.append(p)

        else:  # transient
            # in_visual
            visual_mid = min(support_end + VISUAL_WINDOW_CHUNKS // 2, num_chunks - 1)
            if visual_mid >= support_end:
                avail = classify_availability(card, visual_mid, rollout, bitmap)
                if avail == "in_visual":
                    seq = determine_sequence_type(card, avail)
                    p = compute_placement(card, visual_mid, seq, rollout, evidence)
                    if p:
                        all_placements.append(p)

            # in_history_only (recall)
            history_chunk = min(support_end + VISUAL_WINDOW_CHUNKS + 5, num_chunks - 1)
            if history_chunk < num_chunks:
                avail = classify_availability(card, history_chunk, rollout, bitmap)
                seq = determine_sequence_type(card, avail)
                p = compute_placement(card, history_chunk, seq, rollout, evidence)
                if p:
                    all_placements.append(p)

            # recall_fail variant (30% of recall cards)
            if random.random() < 0.3 and history_chunk < num_chunks:
                p = compute_placement(card, history_chunk, "recall_fail_then_found",
                                       rollout, evidence)
                if p:
                    all_placements.append(p)

        # E2 event_watch
        if card.get("family") == "E2":
            support_start = min(support_chunks)
            if support_start >= 5:
                p = compute_placement(card, max(0, support_start - 8),
                                       "event_watch", rollout, evidence)
                if p:
                    all_placements.append(p)

        # M1 multi_response
        if card.get("family") == "M1":
            ask = min(5, num_chunks - 1)
            p = compute_placement(card, ask, "multi_response", rollout, evidence)
            if p:
                all_placements.append(p)

    logger.info(f"  3-B: {len(all_placements)} placements from {len(cards)} cards")
    return all_placements


# ---------------------------------------------------------------------------
# Trajectory Planning
# ---------------------------------------------------------------------------


def plan_trajectories(
    placements: List[Dict],
    target: int = 30,
    seed: int = 42,
) -> List[Dict]:
    """Combine placements into multi-question trajectories.

    Each trajectory = 1-3 placements (different questions, >=5 chunks apart).
    """
    rng = random.Random(seed)
    trajectories = []

    # Group by sequence_type for diversity
    by_type = {}
    for p in placements:
        by_type.setdefault(p["sequence_type"], []).append(p)

    # Phase 1: each sequence_type gets representative trajectories
    for seq_type, ps in by_type.items():
        rng.shuffle(ps)
        for p in ps[:3]:
            trajectories.append({
                "trajectory_id": f"traj_{len(trajectories)}",
                "placements": [p],
            })

    # Phase 2: combine remaining into multi-question trajectories
    used_ids = {id(p) for t in trajectories for p in t["placements"]}
    remaining = [p for p in placements if id(p) not in used_ids]
    remaining.sort(key=lambda p: p["ask_chunk"])

    group = []
    for p in remaining:
        if len(trajectories) >= target:
            break
        if not group or p["ask_chunk"] - group[-1]["ask_chunk"] >= 5:
            group.append(p)
            if len(group) >= 3:
                trajectories.append({
                    "trajectory_id": f"traj_{len(trajectories)}",
                    "placements": group,
                })
                group = []
    if group and len(trajectories) < target:
        trajectories.append({
            "trajectory_id": f"traj_{len(trajectories)}",
            "placements": group,
        })

    logger.info(f"  3-B: {len(trajectories)} trajectories "
                f"({sum(len(t['placements']) for t in trajectories)} total placements)")
    return trajectories


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_placements(video_id: str, data: Dict):
    PLACEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLACEMENTS_DIR / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_placements(video_id: str) -> Optional[Dict]:
    path = PLACEMENTS_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
