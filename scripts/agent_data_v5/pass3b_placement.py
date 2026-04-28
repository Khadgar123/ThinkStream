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
    """Map availability + family → behavior sequence type.

    v9.5 — multi_response dispatch widened beyond M1:
      * F5 (REC repetition counting): OVO REC asks the same activity-count
        question at multiple realtime probes; the answer increments as
        more occurrences happen. Pure immediate_response would give the
        student only the FINAL count.
      * E2 (event_watch state-change Yes/No): OVO SSR asks
        "Has step X happened yet?" at multiple probes — type=0 (No)
        before, type=1 (Yes) after. Same question, answer flips. Pure
        immediate_response gives only one snapshot.
    Other families remain on the (immediate_response / event_watch /
    recall_success) trio, which already covers RT/BT/single-probe FT.
    """
    family = card.get("family", "")
    if family == "M1":
        return "multi_response"
    # NEW: SSR-style and REC-style families get multi-probe dispatch.
    # availability still gates: a card whose evidence is purely in_future
    # falls back to event_watch (it has no answer yet at ask_chunk).
    if family in ("F5", "E2") and availability != "in_future":
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

    # Persist support_chunks + family + a derived availability hint on the
    # placement so downstream audits and Pass 4 don't have to re-join cards.
    # v9.1 audit (Pass 3-B) couldn't verify legality without joining cards
    # because these fields were absent.
    sup = card.get("support_chunks", []) or []
    if not sup:
        availability = "unknown"
    elif min(sup) > ask_chunk:
        availability = "in_future"
    elif max(sup) >= ask_chunk - VISUAL_WINDOW_CHUNKS:
        availability = "in_visual"
    else:
        availability = "in_history_only"

    return {
        "card_id": card["card_id"],
        "ask_chunk": ask_chunk,
        "sequence_type": sequence_type,
        "key_chunks": key_chunks,
        # audit fields — denormalized from the card for traceability
        "family": card.get("family", ""),
        "support_chunks": list(sup),
        "availability": availability,
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

    v9.3 — DIFFICULTY TIERS for transient cards:
      Tier 1 (easy_in_visual):   ask ∈ [support_end+1, support_end+11]
                                 — answer in recent_thinks, model copies from window
      Tier 2 (medium_in_compressed): ask ∈ [support_end+14, support_end+28]
                                 — answer just past visual window, in compressed memory
      Tier 3 (hard_history_only): ask near end of video (support_end+30..num-2)
                                 — long-distance, requires recall (50% recall_fail_then_found)

    For each transient card with sufficient runway, we sample one ask per tier
    (when feasible). This forces a 1:1:1 difficulty mix instead of batch1's
    skew toward in_visual, which let the model "look up" answers in the window
    without ever exercising compression / recall.

    When client is provided, uses 397B to judge student visibility at
    history chunks. Tier 2 always goes through LLM check; tier 3 splits 50/50
    between recall_fail_then_found and LLM-judged recall_success.
    """
    num_chunks = rollout["num_chunks"]
    immediate_placements = []
    # v9.3: tuples are (card, ask_chunk, tier) — tier carries through to
    # placement metadata so audits / pass3d can balance the difficulty mix.
    candidates_needing_llm: List[Tuple[Dict, int, str]] = []
    rng = random.Random(seed)

    for card in cards:
        vis_type = card.get("visibility_type", "transient")
        support_chunks = card.get("support_chunks", [])
        if not support_chunks:
            continue
        support_end = max(support_chunks)
        support_start = min(support_chunks)
        family = card.get("family", "")

        # ---- v9 new families ---------------------------------------------
        # F5 (REC): ask AFTER last repetition so count is complete.
        if family == "F5":
            ask_f5 = min(support_end + 2, num_chunks - 1)
            if ask_f5 > support_end:
                p = compute_placement(card, ask_f5, "immediate_response",
                                       rollout, evidence)
                if p:
                    p["difficulty_tier"] = "easy_in_visual"
                    immediate_placements.append(p)
            continue

        # F6 (FPD): ask AT support_end (state observed, predict next).
        # Skip if too close to end — need future evidence for verify.
        if family == "F6":
            if support_end < num_chunks - 2:
                p = compute_placement(card, support_end, "immediate_response",
                                       rollout, evidence)
                if p:
                    p["difficulty_tier"] = "easy_in_visual"
                    immediate_placements.append(p)
            continue

        # N1 (HLD, MC v9.3): the correct option appears in support_chunks; the
        # 3 distractors don't appear anywhere. Asking AT support is easy
        # (option visible); asking far past support exercises memory of which
        # of the 4 options actually appeared. Spread 3 asks: one at support
        # (easy), one mid-history (medium), one near end (hard).
        if family == "N1":
            tier_asks = []
            ask_easy = min(support_end + 2, num_chunks - 1)
            if ask_easy > support_end:
                tier_asks.append((ask_easy, "easy_in_visual"))
            ask_med = min(support_end + VISUAL_WINDOW_CHUNKS + 4, num_chunks - 2)
            if ask_med > ask_easy:
                tier_asks.append((ask_med, "medium_in_compressed"))
            ask_hard = num_chunks - 2
            if ask_hard > ask_med + 6:
                tier_asks.append((ask_hard, "hard_history_only"))
            for ask_n1, tier in tier_asks:
                p = compute_placement(card, ask_n1, "immediate_response",
                                       rollout, evidence)
                if p:
                    p["difficulty_tier"] = tier
                    immediate_placements.append(p)
            continue

        # ─── v9.4 reasoning families (CR2/CR4 force tier 2/3 only) ───
        # CR2 (temporal ordering): 3 events spread across support_chunks.
        # If we placed an ask in the visual window, the LATEST event would
        # be visible and the model could "look up" 2/3 of the answer instead
        # of recalling the order. Force ask past visual window so all 3
        # events are out of visual (in summaries / compressed segments).
        if family == "CR2":
            comp_lo = support_end + VISUAL_WINDOW_CHUNKS + 2  # +14
            comp_hi = min(support_end + VISUAL_WINDOW_CHUNKS + 16, num_chunks - 2)
            if comp_hi > comp_lo:
                ask_med = comp_lo + (comp_hi - comp_lo) // 2
                candidates_needing_llm.append((card, ask_med, "medium_in_compressed"))
            hist_lo = support_end + VISUAL_WINDOW_CHUNKS + 18  # +30
            if num_chunks - 2 >= hist_lo and num_chunks - 2 >= num_chunks * 2 // 3:
                ask_hard = num_chunks - 2
                if rng.random() < 0.5:
                    p = compute_placement(card, ask_hard, "recall_fail_then_found",
                                           rollout, evidence)
                    if p:
                        p["difficulty_tier"] = "hard_history_only"
                        immediate_placements.append(p)
                else:
                    candidates_needing_llm.append((card, ask_hard, "hard_history_only"))
            continue

        # CR4 (compositional): 2+ observations whose conjunction is the answer.
        # Same logic as CR2: support chunks may be only 4-6 apart, so a tier-1
        # ask sees both → trivial. Force tier 2/3.
        if family == "CR4":
            comp_lo = support_end + VISUAL_WINDOW_CHUNKS + 2
            comp_hi = min(support_end + VISUAL_WINDOW_CHUNKS + 16, num_chunks - 2)
            if comp_hi > comp_lo:
                ask_med = comp_lo + (comp_hi - comp_lo) // 2
                candidates_needing_llm.append((card, ask_med, "medium_in_compressed"))
            hist_lo = support_end + VISUAL_WINDOW_CHUNKS + 18
            if num_chunks - 2 >= hist_lo and num_chunks - 2 >= num_chunks * 2 // 3:
                ask_hard = num_chunks - 2
                if rng.random() < 0.5:
                    p = compute_placement(card, ask_hard, "recall_fail_then_found",
                                           rollout, evidence)
                    if p:
                        p["difficulty_tier"] = "hard_history_only"
                        immediate_placements.append(p)
                else:
                    candidates_needing_llm.append((card, ask_hard, "hard_history_only"))
            continue

        # CR1 (causal why) and CR3 (intent) — fall through to default
        # transient/persistent path. CR1 is transient (cause/effect both in
        # specific chunks); CR3 should be tagged persistent by the teacher
        # (goal holds once revealed). Default 3-tier / 3-spread path is correct.
        # ---- end v9 new families -----------------------------------------

        if vis_type == "persistent":
            # Persistent: answer always visible → always immediate_response.
            # Sample 3 evenly-spaced ask_chunks. Label tier as "persistent_spread"
            # for stats; difficulty is uniformly easy regardless of ask position.
            candidates = [num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4]
            for ask in candidates:
                if 0 <= ask < num_chunks:
                    p = compute_placement(card, ask, "immediate_response",
                                           rollout, evidence)
                    if p:
                        p["difficulty_tier"] = "persistent_spread"
                        immediate_placements.append(p)

        else:  # transient — 3-tier sampling
            # ─── Tier 1: easy_in_visual (answer in window, ~24s after support) ───
            visual_lo = support_end + 1
            visual_hi = min(support_end + VISUAL_WINDOW_CHUNKS - 1, num_chunks - 1)
            if visual_hi > visual_lo:
                span = visual_hi - visual_lo
                # Sample 2 within visual window for SoftAsk diversity.
                visual_asks = sorted(set([
                    visual_lo + max(1, span // 3),
                    visual_lo + max(1, (2 * span) // 3),
                ]))
                for ask in visual_asks:
                    if not (visual_lo <= ask <= visual_hi):
                        continue
                    snapshot = _get_snapshot(rollout, ask)
                    if not snapshot:
                        continue
                    window_start = snapshot["visual_window_start"]
                    window_end = snapshot["chunk_idx"]
                    if any(window_start <= c <= window_end for c in support_chunks):
                        seq = determine_sequence_type(card, "in_visual")
                        p = compute_placement(card, ask, seq, rollout, evidence)
                        if p:
                            p["difficulty_tier"] = "easy_in_visual"
                            immediate_placements.append(p)

            # ─── Tier 2: medium_in_compressed (just past visual window) ───
            # 14..28 chunks past support_end ≈ 28..56 s. By this distance
            # the supporting chunks have rolled out of the visual window
            # and (with COMPRESS_TOKEN_THRESHOLD~480) typically been
            # compressed into a summary segment. Model must read summary,
            # not visual frames.
            comp_lo = support_end + VISUAL_WINDOW_CHUNKS + 2     # +14
            comp_hi = min(support_end + VISUAL_WINDOW_CHUNKS + 16,
                          num_chunks - 2)                         # +28 cap
            if comp_hi > comp_lo:
                ask_med = comp_lo + (comp_hi - comp_lo) // 2
                # Always LLM-checked: outcome is immediate_response (if
                # answerable from compressed) or recall_success.
                candidates_needing_llm.append((card, ask_med, "medium_in_compressed"))

            # ─── Tier 3: hard_history_only (very far past, requires recall) ───
            # Place near end of video AND at least 30 chunks past support.
            # 50% become recall_fail_then_found (mixed mechanism), 50% go
            # through LLM check (recall_success or in_compressed).
            hist_min_gap = VISUAL_WINDOW_CHUNKS + 18   # 30 chunks = 60s
            hist_lo = support_end + hist_min_gap
            hist_hi = num_chunks - 2
            # Only emit tier-3 if there's actually room (video long enough,
            # support not already near end).
            if hist_hi >= hist_lo and hist_hi >= num_chunks * 2 // 3:
                ask_hard = hist_hi
                if rng.random() < 0.5:
                    p = compute_placement(card, ask_hard, "recall_fail_then_found",
                                           rollout, evidence)
                    if p:
                        p["difficulty_tier"] = "hard_history_only"
                        immediate_placements.append(p)
                else:
                    candidates_needing_llm.append((card, ask_hard, "hard_history_only"))

        # E2 event_watch: pure math (ask before support starts)
        if card.get("family") == "E2":
            if support_start >= 5:
                ask_ew = max(2, support_start - 8)
                p = compute_placement(card, ask_ew,
                                       "event_watch", rollout, evidence)
                if p:
                    p["difficulty_tier"] = "event_watch"
                    immediate_placements.append(p)

        # M1 multi_response
        if card.get("family") == "M1":
            ask = min(5, num_chunks - 1)
            p = compute_placement(card, ask, "multi_response", rollout, evidence)
            if p:
                p["difficulty_tier"] = "multi_response"
                immediate_placements.append(p)

    # --- Batch LLM visibility checks (all independent, high concurrency) ---
    llm_placements = []
    if client and candidates_needing_llm:
        async def _check_one(card, ask_chunk, tier):
            snapshot = _get_snapshot(rollout, ask_chunk)
            if not snapshot:
                return card, ask_chunk, tier, False
            answerable = await _check_visibility_one(
                card, ask_chunk, snapshot, client, video_id)
            return card, ask_chunk, tier, answerable

        tasks = [_check_one(c, a, t) for c, a, t in candidates_needing_llm]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        n_answerable = 0
        tier_counts = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  [{video_id}] visibility check failed: {result}")
                continue
            card, ask_chunk, tier, answerable = result
            if answerable:
                seq = "immediate_response"
                n_answerable += 1
            else:
                seq = "recall_success"
            p = compute_placement(card, ask_chunk, seq, rollout, evidence)
            if p:
                p["difficulty_tier"] = tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                llm_placements.append(p)

        tier_str = " ".join(f"{t}:{n}" for t, n in sorted(tier_counts.items()))
        logger.info(
            f"  [{video_id}] 3-B visibility: {len(candidates_needing_llm)} checked, "
            f"{n_answerable} answerable | tiers: {tier_str}"
        )
    elif candidates_needing_llm:
        # Fallback: keyword-based retention (no client)
        for card, ask_chunk, tier in candidates_needing_llm:
            bitmap = precompute_retention(card, rollout)
            avail = classify_availability(card, ask_chunk, rollout, bitmap)
            seq = determine_sequence_type(card, avail)
            p = compute_placement(card, ask_chunk, seq, rollout, evidence)
            if p:
                p["difficulty_tier"] = tier
                llm_placements.append(p)

    all_placements = immediate_placements + llm_placements

    # v9.3: tier distribution log so we can audit difficulty mix per video.
    tier_dist = {}
    for p in all_placements:
        t = p.get("difficulty_tier", "untagged")
        tier_dist[t] = tier_dist.get(t, 0) + 1
    tier_str = " ".join(f"{t}:{n}" for t, n in sorted(tier_dist.items()))
    logger.info(f"  3-B: {len(all_placements)} placements from {len(cards)} cards | tiers: {tier_str}")
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

    # 5b. OVOBench-relevant rare-sequence boost.
    # v9.1 audit found event_watch=3.2% across the dataset — too low for AAR/EPM
    # tasks. Add a per-placement bonus (independent of "unseen") so multiple
    # event_watch placements survive tie-breaking against immediate_response.
    RARE_SEQ_BONUS = {
        "event_watch": 1.5,             # AAR/EPM
        "recall_fail_then_found": 0.6,  # mixed mechanism, harder
        "multi_response": 0.5,          # M1 family
    }
    score += RARE_SEQ_BONUS.get(p["sequence_type"], 0.0)

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

    # Dynamic target: ~1 trajectory per 12 chunks (was 20).
    # v9.1 audit: previous 20-chunk divisor gave only 30% capacity utilization
    # (avg 3.5/video vs MAX=10). Tightening the divisor + bumping per-traj cap
    # roughly doubles trajectories per video, increasing fork-sample volume.
    target = min(max(3, num_chunks // 12), MAX_TRAJECTORIES_PER_VIDEO)
    max_placements_per_traj = min(
        max(max_placements_per_traj, 6),  # was effectively 5
        MAX_QUESTIONS_PER_TRAJECTORY,
    )

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
        #
        # v9: 50% of trajectories use ENTITY BRIDGING — pick the next placement
        # from temporally-valid candidates with highest card_keyword overlap to
        # the trajectory so far (≥30% threshold). The other 50% stay independent.
        paired: Set[int] = set()
        for i in range(len(selected)):
            if i in paired or len(trajectories) >= target:
                break
            group = [selected[i]]
            paired.add(i)
            entity_bridge = rng.random() < 0.5  # 50% bridged trajectories

            def _valid(j: int) -> bool:
                if j in paired:
                    return False
                if selected[j]["ask_chunk"] - group[-1]["ask_chunk"] < min_chunk_gap:
                    return False
                if _count_pending_at(group, selected[j]["ask_chunk"]) >= MAX_ACTIVE_QUERIES:
                    return False
                return True

            def _bridging_kw(card: Dict) -> Set[str]:
                """For bridging we want ENTITY overlap, so combine question text +
                canonical_answer keywords across all answer_forms (richer than
                extract_card_keywords which is tuned for retention scoring)."""
                q = card.get("question", "")
                a = card.get("canonical_answer", "")
                return set(extract_keywords(q)) | set(extract_keywords(a))

            def _bridging_score(j: int) -> float:
                """Max keyword overlap between j's card and any card in current group."""
                card_j = cards_map.get(selected[j]["card_id"], {})
                kw_j = _bridging_kw(card_j)
                if not kw_j:
                    return 0.0
                best = 0.0
                for p_in in group:
                    card_in = cards_map.get(p_in["card_id"], {})
                    kw_in = _bridging_kw(card_in)
                    if not kw_in:
                        continue
                    overlap = len(kw_j & kw_in) / max(min(len(kw_j), len(kw_in)), 1)
                    best = max(best, overlap)
                return best

            j_indices = list(range(i + 1, len(selected)))
            while j_indices:
                valid = [j for j in j_indices if _valid(j)]
                if not valid:
                    break
                if entity_bridge:
                    # Prefer highest bridging score; require ≥0.3 to count as bridged.
                    scored = [(j, _bridging_score(j)) for j in valid]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    best_j, best_score = scored[0]
                    if best_score >= 0.3:
                        chosen_j = best_j
                    else:
                        # No bridging candidate — fall back to temporal order
                        chosen_j = valid[0]
                else:
                    chosen_j = valid[0]
                group.append(selected[chosen_j])
                paired.add(chosen_j)
                j_indices = [j for j in j_indices if j != chosen_j]
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
    from .cache_version import stage_version_ok
    if not stage_version_ok("3b"):
        return None
    path = PLACEMENTS_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
