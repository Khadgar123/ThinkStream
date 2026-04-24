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
from .pass3a_cards import RETENTION_CLASS, extract_keywords, extract_card_keywords

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
    """Check if student thinks/summaries retained the answer-relevant info.

    Uses extract_card_keywords which handles binary/MC/number answers
    by extracting keywords from the question text instead of the
    uninformative canonical_answer ("Yes", "A", "3").

    Only checks support_chunks (not all chunks).
    Returns: {"thinks_retained": {chunk: bool}, "summary_retained": {event_idx: bool}}
    """
    card_kw = extract_card_keywords(card)
    if not card_kw:
        # No keywords extractable — cannot determine retention
        return {"thinks_retained": {}, "summary_retained": {}}

    retention_class = RETENTION_CLASS.get(card.get("family", ""), "medium")
    threshold = {"low": 0.5, "medium": 0.35, "high": 0.2}[retention_class]

    observations = rollout.get("thinks", [])
    thinks_retained = {}
    for chunk_idx in card.get("support_chunks", []):
        if chunk_idx < len(observations):
            think_text = observations[chunk_idx].get("think", "")
            thinks_retained[chunk_idx] = _keyword_overlap(think_text, card_kw) > threshold
        else:
            thinks_retained[chunk_idx] = False

    summary_retained = {}
    support_set = set(card.get("support_chunks", []))
    for idx, event in enumerate(rollout.get("compression_events", [])):
        compressed = set(event.get("compressed_thinks_chunks", []))
        if support_set & compressed:
            summary_text = event.get("summary", {}).get("text", "")
            summary_retained[idx] = _keyword_overlap(summary_text, card_kw) > 0.3

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
    snapshots = rollout["snapshots"]
    snapshot = snapshots.get(ask_chunk) or snapshots.get(str(ask_chunk))
    if snapshot is None:
        logger.debug(f"  classify_availability: no snapshot for chunk {ask_chunk}")
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
    card_kw = extract_card_keywords(card)
    if not card_kw:
        return None
    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        if idx <= after_chunk:
            continue
        # Check atomic_facts
        for fact in cap.get("atomic_facts", []):
            if _keyword_overlap(fact.get("fact", ""), card_kw) > 0.3:
                return idx
        # Also check entity descs + actions (helps binary/MC about entities)
        for ent in cap.get("visible_entities", []):
            ent_text = f"{ent.get('desc', '')} {ent.get('action', '')}"
            if _keyword_overlap(ent_text, card_kw) > 0.4:
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
    seed: int = 42,
) -> List[Dict]:
    """Compute all valid placements for all cards."""
    num_chunks = rollout["num_chunks"]
    all_placements = []
    rng = random.Random(seed)

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
            if rng.random() < 0.3 and history_chunk < num_chunks:
                p = compute_placement(card, history_chunk, "recall_fail_then_found",
                                       rollout, evidence)
                if p:
                    all_placements.append(p)

        # E2 event_watch
        if card.get("family") == "E2":
            support_start = min(support_chunks)
            if support_start >= 5:
                # Ensure ask_chunk >= 2 so snapshot exists (rollout needs warmup)
                ask_ew = max(2, support_start - 8)
                p = compute_placement(card, ask_ew,
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


def _score_placement(
    p: Dict,
    cards_map: Dict[str, Dict],
    used_families: Set[str],
    used_seq_types: Set[str],
    used_ask_chunks: List[int],
) -> float:
    """Score a placement for greedy selection. Higher = better."""
    card = cards_map.get(p["card_id"], {})
    score = 0.0

    # Quality: penalize inferred support_chunks
    if card.get("_support_inferred"):
        score -= 2.0

    # Quality: prefer auto-verifiable answer_forms (RL reward works)
    auto_verify = {"binary", "multiple_choice", "number", "short_exact"}
    if card.get("answer_form") in auto_verify:
        score += 1.0

    # Diversity: unseen family bonus
    family = card.get("family", "")
    if family not in used_families:
        score += 2.0

    # Diversity: unseen sequence_type bonus
    if p["sequence_type"] not in used_seq_types:
        score += 2.0

    # Spread: distance from nearest used ask_chunk
    if used_ask_chunks:
        min_dist = min(abs(p["ask_chunk"] - c) for c in used_ask_chunks)
        score += min(min_dist / 10.0, 1.5)  # cap at 1.5
    else:
        score += 1.5  # first placement gets full spread bonus

    return score


def plan_trajectories(
    placements: List[Dict],
    cards_map: Dict[str, Dict] = None,
    target: int = 5,
    max_placements_per_traj: int = 5,
    min_chunk_gap: int = 8,
    seed: int = 42,
) -> List[Dict]:
    """Select placements and build trajectories via greedy diversity scoring.

    Design principles:
    1. 4-6 questions per trajectory (keeps silent/response ratio ~60/30)
    2. Questions spread across the video timeline (min gap = 8 chunks = 16s)
    3. Maximize diversity: different families, sequence_types
    4. Fewer trajectories (5 per video) — each is a dense, complete episode

    Args:
        placements: all valid placements from compute_all_placements
        cards_map: {card_id: card_dict} for quality scoring
        target: target number of trajectories per video
        max_placements_per_traj: max questions per trajectory (4-6)
        min_chunk_gap: minimum gap between questions in same trajectory
        seed: RNG seed for reproducibility
    """
    if cards_map is None:
        cards_map = {}
    rng = random.Random(seed)

    # --- Phase 1: Greedy selection of best placements ---
    used_families: Set[str] = set()
    used_seq_types: Set[str] = set()
    used_ask_chunks: List[int] = []
    used_card_ids: Set[str] = set()
    selected: List[Dict] = []

    candidates = list(placements)
    # Select up to target * max_placements_per_traj placements
    budget = target * max_placements_per_traj

    while candidates and len(selected) < budget:
        # Score all remaining candidates
        scored = []
        for p in candidates:
            # Skip if same card already selected (different ask_chunk is ok,
            # but same card_id means same question — prefer variety)
            if p["card_id"] in used_card_ids:
                continue
            s = _score_placement(p, cards_map, used_families,
                                 used_seq_types, used_ask_chunks)
            scored.append((s, p))

        if not scored:
            break

        # Pick top candidate (with randomized tie-breaking)
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        ties = [sp for sp in scored if sp[0] >= top_score - 0.1]
        _, best = ties[rng.randint(0, len(ties) - 1)]

        selected.append(best)
        card = cards_map.get(best["card_id"], {})
        used_families.add(card.get("family", ""))
        used_seq_types.add(best["sequence_type"])
        used_ask_chunks.append(best["ask_chunk"])
        used_card_ids.add(best["card_id"])
        candidates.remove(best)

    # --- Phase 2: Build trajectories from selected placements ---
    # Sort selected by ask_chunk for temporal ordering
    selected.sort(key=lambda p: p["ask_chunk"])

    trajectories = []

    if max_placements_per_traj == 1:
        # Simple: each placement is its own trajectory
        for p in selected[:target]:
            trajectories.append({
                "trajectory_id": f"traj_{len(trajectories)}",
                "placements": [p],
            })
    else:
        # Group placements that are ≥min_chunk_gap apart into multi-question trajectories
        paired: Set[int] = set()
        for i in range(len(selected)):
            if i in paired or len(trajectories) >= target:
                break
            group = [selected[i]]
            paired.add(i)
            for j in range(i + 1, len(selected)):
                if j in paired:
                    continue
                if selected[j]["ask_chunk"] - group[-1]["ask_chunk"] >= min_chunk_gap:
                    group.append(selected[j])
                    paired.add(j)
                    if len(group) >= max_placements_per_traj:
                        break
            trajectories.append({
                "trajectory_id": f"traj_{len(trajectories)}",
                "placements": group,
            })

        # Remaining unpaired placements as single-question trajectories
        for i in range(len(selected)):
            if i not in paired and len(trajectories) < target:
                trajectories.append({
                    "trajectory_id": f"traj_{len(trajectories)}",
                    "placements": [selected[i]],
                })

    # --- Stats ---
    family_dist = {}
    seq_dist = {}
    for t in trajectories:
        for p in t["placements"]:
            card = cards_map.get(p["card_id"], {})
            f = card.get("family", "?")
            family_dist[f] = family_dist.get(f, 0) + 1
            s = p["sequence_type"]
            seq_dist[s] = seq_dist.get(s, 0) + 1

    total_p = sum(len(t["placements"]) for t in trajectories)
    fam_str = " ".join(f"{f}:{n}" for f, n in sorted(family_dist.items()))
    seq_str = " ".join(f"{s}:{n}" for s, n in sorted(seq_dist.items()))
    logger.info(f"  3-B: {len(trajectories)} trajectories, {total_p} placements")
    logger.info(f"  3-B families: {fam_str}")
    logger.info(f"  3-B sequences: {seq_str}")

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
