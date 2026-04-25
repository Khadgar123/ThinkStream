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

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    MAX_ACTIVE_QUERIES,
    MAX_QUESTIONS_PER_TRAJECTORY,
    MAX_TRAJECTORIES_PER_VIDEO,
    PASS_CONFIG,
    PLACEMENTS_DIR,
    VISUAL_WINDOW_CHUNKS,
)
from .pass3a_cards import FAMILY_TARGETS, RETENTION_CLASS, extract_keywords, extract_card_keywords

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
# LLM Visibility Check (397B, independent per (card, chunk))
# ---------------------------------------------------------------------------

VISIBILITY_CHECK_PROMPT = """A student is watching a streaming video. At the current moment, the student has the following memory.

Recent observations:
{recent_thinks}

Compressed memory:
{compressed_segments}

Question: "{question}"
Expected answer: "{canonical_answer}"

Based ONLY on the student's observations and memory above, does the student have enough information to answer this question correctly?

Output JSON only: {{"answerable": true/false}}"""


def _format_snapshot_text(snapshot: Dict) -> Tuple[str, str]:
    """Format snapshot into recent_thinks and compressed_segments text."""
    recent_parts = []
    for item in snapshot.get("recent_thinks", []):
        text = item.get("text", item.get("think", ""))
        recent_parts.append(f"[{item.get('time', '?')}] {text}")
    recent_text = "\n".join(recent_parts[-10:]) if recent_parts else "(empty)"

    compressed_parts = []
    for seg in snapshot.get("compressed_segments", []):
        tr = seg.get("time_range", "?")
        compressed_parts.append(f"[{tr}] {seg.get('text', '')}")
    compressed_text = "\n".join(compressed_parts) if compressed_parts else "(empty)"

    return recent_text, compressed_text


async def _check_visibility_one(
    card: Dict, ask_chunk: int, snapshot: Dict, client, video_id: str,
) -> bool:
    """Check if student can answer at ask_chunk. Independent 397B call.

    Returns True if the student's current memory contains enough info.
    """
    recent_text, compressed_text = _format_snapshot_text(snapshot)

    prompt = VISIBILITY_CHECK_PROMPT.format(
        recent_thinks=recent_text,
        compressed_segments=compressed_text,
        question=card.get("question", ""),
        canonical_answer=card.get("canonical_answer", ""),
    )

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3b_visibility", {}).get("max_tokens", 512),
        temperature=0.1,
        request_id=f"{video_id}_vis_{card.get('card_id', '')}_{ask_chunk}",
    )

    if not raw:
        return False

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    try:
        result = json.loads(raw)
        return bool(result.get("answerable", False))
    except (json.JSONDecodeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            return bool(result.get("answerable", False))
        except (json.JSONDecodeError, ValueError):
            pass
    return False


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


def _get_snapshot(rollout: Dict, chunk_idx: int) -> Optional[Dict]:
    """Get snapshot for a chunk, handling int/str key mismatch."""
    snapshots = rollout["snapshots"]
    return snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx))


async def compute_all_placements(
    cards: List[Dict], rollout: Dict, evidence: List[Dict],
    client=None, video_id: str = "",
    seed: int = 42,
) -> List[Dict]:
    """Compute all valid placements for all cards.

    When client is provided, uses 397B to judge student visibility at
    history chunks (each call is independent → high concurrency).
    When client is None, falls back to keyword-based retention check.
    """
    num_chunks = rollout["num_chunks"]
    immediate_placements = []
    candidates_needing_llm = []  # (card, ask_chunk) pairs for batch LLM check
    rng = random.Random(seed)

    for card in cards:
        vis_type = card.get("visibility_type", "transient")
        support_chunks = card.get("support_chunks", [])
        if not support_chunks:
            continue
        support_end = max(support_chunks)
        support_start = min(support_chunks)

        if vis_type == "persistent":
            # Pure math: persistent → always immediate_response
            candidates = [num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4]
            for ask in candidates:
                if 0 <= ask < num_chunks:
                    p = compute_placement(card, ask, "immediate_response", rollout, evidence)
                    if p:
                        immediate_placements.append(p)

        else:  # transient
            # in_visual: pure math — check if support is within visual window
            visual_mid = min(support_end + VISUAL_WINDOW_CHUNKS // 2, num_chunks - 1)
            if visual_mid >= support_end:
                snapshot = _get_snapshot(rollout, visual_mid)
                if snapshot:
                    window_start = snapshot["visual_window_start"]
                    window_end = snapshot["chunk_idx"]
                    if any(window_start <= c <= window_end for c in support_chunks):
                        seq = determine_sequence_type(card, "in_visual")
                        p = compute_placement(card, visual_mid, seq, rollout, evidence)
                        if p:
                            immediate_placements.append(p)

            # history_chunk: needs retention check (LLM or keyword fallback)
            history_chunk = min(support_end + VISUAL_WINDOW_CHUNKS + 5, num_chunks - 1)
            if history_chunk < num_chunks and support_end < history_chunk:
                candidates_needing_llm.append((card, history_chunk))

            # recall_fail variant (30% of recall cards, always recall_fail_then_found)
            if rng.random() < 0.3 and history_chunk < num_chunks:
                p = compute_placement(card, history_chunk, "recall_fail_then_found",
                                       rollout, evidence)
                if p:
                    immediate_placements.append(p)

        # E2 event_watch: pure math (ask before support starts)
        if card.get("family") == "E2":
            if support_start >= 5:
                ask_ew = max(2, support_start - 8)
                p = compute_placement(card, ask_ew,
                                       "event_watch", rollout, evidence)
                if p:
                    immediate_placements.append(p)

        # M1 multi_response
        if card.get("family") == "M1":
            ask = min(5, num_chunks - 1)
            p = compute_placement(card, ask, "multi_response", rollout, evidence)
            if p:
                immediate_placements.append(p)

    # --- Batch LLM visibility checks (all independent, high concurrency) ---
    llm_placements = []
    if client and candidates_needing_llm:
        async def _check_one(card, ask_chunk):
            snapshot = _get_snapshot(rollout, ask_chunk)
            if not snapshot:
                return card, ask_chunk, False
            answerable = await _check_visibility_one(
                card, ask_chunk, snapshot, client, video_id)
            return card, ask_chunk, answerable

        tasks = [_check_one(c, a) for c, a in candidates_needing_llm]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        n_answerable = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  [{video_id}] visibility check failed: {result}")
                continue
            card, ask_chunk, answerable = result
            if answerable:
                seq = "immediate_response"
                n_answerable += 1
            else:
                seq = "recall_success"
            p = compute_placement(card, ask_chunk, seq, rollout, evidence)
            if p:
                llm_placements.append(p)

        logger.info(
            f"  [{video_id}] 3-B visibility: {len(candidates_needing_llm)} checked, "
            f"{n_answerable} answerable"
        )
    elif candidates_needing_llm:
        # Fallback: keyword-based retention (no client)
        for card, ask_chunk in candidates_needing_llm:
            bitmap = precompute_retention(card, rollout)
            avail = classify_availability(card, ask_chunk, rollout, bitmap)
            seq = determine_sequence_type(card, avail)
            p = compute_placement(card, ask_chunk, seq, rollout, evidence)
            if p:
                llm_placements.append(p)

    all_placements = immediate_placements + llm_placements
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
    used_answers: Set[str],
    evidence: List[Dict] = None,
) -> float:
    """Score a placement for greedy selection (8 dimensions). Higher = better."""
    card = cards_map.get(p["card_id"], {})
    score = 0.0

    # 1. Quality: penalize inferred support_chunks
    if card.get("_support_inferred"):
        score -= 2.0

    # 2. Verifiability: prefer auto-verifiable answer_forms (RL reward works)
    auto_verify = {"binary", "multiple_choice", "number", "short_exact"}
    if card.get("answer_form") in auto_verify:
        score += 1.0

    # 3. Evidence quality: prefer high-confidence support facts
    if evidence:
        ev_by_idx = {c.get("chunk_idx", i): c for i, c in enumerate(evidence)}
        for sc in card.get("support_chunks", []):
            cap = ev_by_idx.get(sc, {})
            facts = cap.get("atomic_facts", [])
            avg_conf = sum(f.get("confidence", 0) for f in facts) / max(len(facts), 1)
            if avg_conf >= 0.85:
                score += 0.5  # high-confidence evidence

    # 4. Diversity: unseen family bonus
    family = card.get("family", "")
    if family not in used_families:
        score += 2.0

    # 5. Diversity: unseen sequence_type bonus
    if p["sequence_type"] not in used_seq_types:
        score += 2.0

    # 6. Spread: distance from nearest used ask_chunk
    if used_ask_chunks:
        min_dist = min(abs(p["ask_chunk"] - c) for c in used_ask_chunks)
        score += min(min_dist / 10.0, 1.5)  # cap at 1.5
    else:
        score += 1.5  # first placement gets full spread bonus

    # 7. Dedup: penalize if same canonical_answer already selected
    canonical = card.get("canonical_answer", "").strip().lower()
    if canonical and canonical in used_answers:
        score -= 1.5  # same answer → likely redundant question

    return score


# ---------------------------------------------------------------------------
# Pending Lifetime Helpers
# ---------------------------------------------------------------------------


def _resolution_chunk(p: Dict) -> int:
    """Chunk at which this placement's question gets answered.

    Used to track how many questions are simultaneously pending
    (asked but not yet answered) within a trajectory.
    """
    kc = p["key_chunks"]
    seq = p["sequence_type"]
    if seq == "recall_fail_then_found":
        return kc.get("found_response", kc["ask"])
    if seq == "event_watch":
        return kc.get("trigger", kc["ask"])
    # immediate_response, recall_success, multi_response: resolved at ask
    return kc["ask"]


def _count_pending_at(group: List[Dict], chunk: int) -> int:
    """Count how many questions in group are still pending at chunk.

    A question is pending if it has been asked (ask_chunk <= chunk)
    but not yet resolved (resolution_chunk > chunk).
    """
    pending = 0
    for p in group:
        if p["ask_chunk"] <= chunk < _resolution_chunk(p):
            pending += 1
    return pending


# ---------------------------------------------------------------------------
# Family Coverage
# ---------------------------------------------------------------------------

# Minimum families that should appear across all trajectories for a video.
# Not all families are required — some (P1, R1, C1) depend on video content.
MIN_FAMILIES_PER_VIDEO = 4


def plan_trajectories(
    placements: List[Dict],
    cards_map: Dict[str, Dict] = None,
    num_chunks: int = 60,
    max_placements_per_traj: int = 5,
    min_chunk_gap: int = 8,
    seed: int = 42,
    evidence: List[Dict] = None,
) -> List[Dict]:
    """Select placements and build trajectories via greedy diversity scoring.

    Design principles:
    1. 4-6 questions per trajectory (keeps silent/response ratio ~60/30)
    2. Questions spread across the video timeline (min gap = 8 chunks = 16s)
    3. Maximize diversity: families, sequence_types, answers
    4. Trajectory count scales with video length (1 per ~20 chunks)
    5. Penalize duplicate answers and low-confidence evidence
    6. MAX_ACTIVE_QUERIES enforced per trajectory (pending lifetime)
    7. Post-check family coverage, backfill if below MIN_FAMILIES_PER_VIDEO

    Args:
        placements: all valid placements from compute_all_placements
        cards_map: {card_id: card_dict} for quality scoring
        num_chunks: video length in chunks (for dynamic target)
        max_placements_per_traj: max questions per trajectory (4-6)
        min_chunk_gap: minimum gap between questions in same trajectory
        seed: RNG seed for reproducibility
        evidence: optional evidence list for confidence scoring
    """
    if cards_map is None:
        cards_map = {}
    rng = random.Random(seed)

    # Dynamic target: ~1 trajectory per 20 chunks, clamped to [2, MAX]
    target = min(max(2, num_chunks // 20), MAX_TRAJECTORIES_PER_VIDEO)
    max_placements_per_traj = min(max_placements_per_traj, MAX_QUESTIONS_PER_TRAJECTORY)

    # --- Phase 1: Greedy selection of best placements ---
    used_families: Set[str] = set()
    used_seq_types: Set[str] = set()
    used_ask_chunks: List[int] = []
    used_card_ids: Set[str] = set()
    used_answers: Set[str] = set()
    selected: List[Dict] = []

    candidates = list(placements)
    budget = target * max_placements_per_traj

    while candidates and len(selected) < budget:
        scored = []
        for p in candidates:
            if p["card_id"] in used_card_ids:
                continue
            s = _score_placement(p, cards_map, used_families,
                                 used_seq_types, used_ask_chunks,
                                 used_answers, evidence)
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
        canonical = card.get("canonical_answer", "").strip().lower()
        if canonical:
            used_answers.add(canonical)
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
        # Also enforce MAX_ACTIVE_QUERIES: at any point in the trajectory,
        # no more than MAX_ACTIVE_QUERIES questions can be pending (asked but unresolved).
        paired: Set[int] = set()
        for i in range(len(selected)):
            if i in paired or len(trajectories) >= target:
                break
            group = [selected[i]]
            paired.add(i)
            for j in range(i + 1, len(selected)):
                if j in paired:
                    continue
                if selected[j]["ask_chunk"] - group[-1]["ask_chunk"] < min_chunk_gap:
                    continue
                # Pending check: would adding this placement exceed limit?
                pending = _count_pending_at(group, selected[j]["ask_chunk"])
                if pending >= MAX_ACTIVE_QUERIES:
                    continue
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

    # --- Phase 3: Family coverage backfill ---
    covered_families = set()
    used_card_ids_final = set()
    for t in trajectories:
        for p in t["placements"]:
            card = cards_map.get(p["card_id"], {})
            covered_families.add(card.get("family", ""))
            used_card_ids_final.add(p["card_id"])

    if len(covered_families) < MIN_FAMILIES_PER_VIDEO:
        # Try to add single-placement trajectories from unused placements
        # covering missing families
        missing = set(FAMILY_TARGETS.keys()) - covered_families
        remaining = [p for p in placements
                     if p["card_id"] not in used_card_ids_final]
        for p in remaining:
            if len(trajectories) >= target + 2:  # allow up to 2 extra for coverage
                break
            card = cards_map.get(p["card_id"], {})
            fam = card.get("family", "")
            if fam in missing:
                trajectories.append({
                    "trajectory_id": f"traj_{len(trajectories)}",
                    "placements": [p],
                })
                missing.discard(fam)
                covered_families.add(fam)
                if len(covered_families) >= MIN_FAMILIES_PER_VIDEO:
                    break

    if len(covered_families) < MIN_FAMILIES_PER_VIDEO:
        logger.warning(
            f"  3-B: only {len(covered_families)} families covered "
            f"(min {MIN_FAMILIES_PER_VIDEO}): {sorted(covered_families)}"
        )

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
