"""
Pass 3: Task Planning

Per-type task mining from evidence graph + rollout.
Each task type has independent mining logic.

Key: Action Minimality — gold_action is determined by what the student
can ACTUALLY see at ask_time (not what the teacher knows).

Output: Candidate tasks with gold_answer, gold_action, visibility_matrix.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    CONFIDENCE_THRESHOLD,
    LEAKAGE_OVERLAP_THRESHOLD,
    MAX_CANDIDATES_PER_VIDEO,
    PASS_CONFIG,
    RECALL_QUERY_PROMPT,
    TASK_QUESTION_PROMPT,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)


def is_usable_fact(fact: Dict) -> bool:
    """Check if a teacher fact is suitable for task mining.

    Requires: high confidence, direct observation in current chunk,
    visible at target resolution. Rejects unknown/repaired facts.
    """
    if fact.get("confidence", 0) < CONFIDENCE_THRESHOLD:
        return False
    if fact.get("parse_repaired"):
        return False
    # Only allow facts directly observed in current chunk
    support = fact.get("support_level", "unknown")
    if support not in ("direct_current_chunk",):
        return False
    # Reject facts not visible at target resolution
    if fact.get("target_resolution_visible") is False:
        return False
    return True


# ---------------------------------------------------------------------------
# Action Minimality
# ---------------------------------------------------------------------------


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text for matching.

    Filters out very short words (<=2 chars) and common stop words.
    Returns deduplicated list preserving order for stable matching.
    """
    stop = {"the", "a", "an", "is", "was", "were", "are", "in", "on", "at",
            "to", "of", "and", "or", "it", "its", "this", "that", "with",
            "for", "from", "has", "had", "have", "been", "not", "but", "can",
            "will", "would", "could", "should", "may", "about", "into"}
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in stop and len(w) > 2 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


# Common short adjectives that often cause false positives in keyword matching
_AMBIGUOUS_SHORT_WORDS = {
    "red", "big", "old", "new", "hot", "low", "top", "set", "cut", "run",
    "put", "get", "got", "let", "saw", "use", "add", "end", "try", "way",
}


def keyword_overlap(text: str, keywords: List[str]) -> float:
    """Compute fraction of keywords found in text (word-boundary match).

    Short ambiguous words (<=3 chars) are down-weighted to 0.5 match
    to reduce false positives (e.g., "red" matching in unrelated context).
    Requires at least 2 keyword matches for overlap > 0 when total keywords >= 3.
    """
    if not keywords:
        return 0.0
    text_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))
    weighted_total = 0.0
    weighted_found = 0.0
    raw_found = 0
    for kw in keywords:
        weight = 0.5 if (len(kw) <= 3 and kw in _AMBIGUOUS_SHORT_WORDS) else 1.0
        weighted_total += weight
        if kw in text_words:
            weighted_found += weight
            raw_found += 1
    if weighted_total == 0:
        return 0.0
    # Require at least 2 raw matches when we have 3+ keywords
    # to avoid single-word coincidental matches driving gold_action
    if len(keywords) >= 3 and raw_found < 2:
        return 0.0
    return weighted_found / weighted_total


# ---------------------------------------------------------------------------
# Semantic similarity (embedding-based, with keyword fallback)
# ---------------------------------------------------------------------------

_embedding_model = None
_embedding_available = None  # None = not checked, True/False = checked


def _get_embedding_model():
    """Lazy-load sentence embedding model. Returns None if unavailable."""
    global _embedding_model, _embedding_available
    if _embedding_available is False:
        return None
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cpu",
        )
        _embedding_available = True
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        return _embedding_model
    except (ImportError, Exception) as e:
        logger.warning(f"Embedding model unavailable ({e}), using keyword fallback")
        _embedding_available = False
        return None


def semantic_overlap(text: str, keywords: List[str],
                     threshold_boost: float = 0.15) -> float:
    """Compute semantic similarity between text and keywords.

    Uses sentence embedding cosine similarity as primary signal,
    falls back to keyword_overlap if embeddings are unavailable.

    Args:
        text: The text to check (e.g., a compressed summary or think).
        keywords: Keywords representing the answer/fact.
        threshold_boost: Bonus added to keyword_overlap when embedding
                         confirms semantic match (>0.5 cosine similarity).

    Returns:
        float in [0, 1]: combined overlap score.
    """
    # Always compute keyword overlap as baseline
    kw_score = keyword_overlap(text, keywords)

    model = _get_embedding_model()
    if model is None:
        return kw_score

    # Compute embedding similarity
    query = " ".join(keywords)
    try:
        embeddings = model.encode([query, text], convert_to_tensor=True)
        import torch
        cosine = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ).item()
    except Exception:
        return kw_score

    # Combine: if embedding says >0.5 similar, boost keyword score
    if cosine > 0.5:
        return min(1.0, kw_score + threshold_boost)
    # If embedding says <0.3, demote keyword score (possible false positive)
    elif cosine < 0.3 and kw_score > 0:
        return kw_score * 0.5
    return kw_score


def determine_gold_action(
    answer_keywords: List[str],
    snapshot: Dict,
    evidence_chunks: List[int],
    observations: List[Dict],
) -> Tuple[str, str]:
    """Determine the correct action using minimality principle.

    Returns: (gold_action, reason)

    Priority:
    1. answer in visual window frames → response(from_frames)
    2. answer in recent_thinks → response(from_memory)
    3. answer in compressed_summaries → response(from_compressed)
    4. answer in historical obs/frames → recall
    5. answer nowhere → response (uncertain)
    """
    # 1. Check visual window (by checking if evidence chunks are in window)
    window_start = snapshot["visual_window_start"]
    window_end = snapshot["chunk_idx"]
    evidence_in_window = any(window_start <= c <= window_end for c in evidence_chunks)
    if evidence_in_window:
        return "response", "answer_in_visual_window"

    # 2. Check recent_thinks (semantic + keyword)
    for obs_item in snapshot.get("recent_thinks", []):
        if semantic_overlap(obs_item.get("text", obs_item.get("obs", "")), answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
            return "response", "answer_in_recent_thinks"

    # 3. Check compressed summaries (semantic + keyword)
    for seg in snapshot["compressed_segments"]:
        if semantic_overlap(seg["text"], answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
            return "response", "answer_in_compressed_summary"

    # 4. Check if answer exists in historical observations (outside current visibility)
    for eidx in evidence_chunks:
        if eidx < len(observations):
            obs_text = observations[eidx].get("think", observations[eidx].get("observation", ""))
            if semantic_overlap(obs_text, answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
                return "recall", "answer_in_historical_observation"

    # 5. Answer not found anywhere accessible → still a response action (uncertain)
    return "response", "unanswerable_not_in_any_accessible_source"


def build_visibility_matrix(
    ask_chunk: int,
    snapshot: Dict,
    answer_keywords: List[str],
    evidence_chunks: List[int],
) -> Dict:
    """Build visibility matrix for a task."""
    window_start = snapshot["visual_window_start"]
    window_end = snapshot["chunk_idx"]

    return {
        "at_chunk": ask_chunk,
        "at_time": ask_chunk * AGENT_CHUNK_SEC,
        "visual_window": [window_start * AGENT_CHUNK_SEC, (window_end + 1) * AGENT_CHUNK_SEC],
        "n_compressed_segments": len(snapshot["compressed_segments"]),
        "n_recent_thinks": len(snapshot.get("recent_thinks", [])),
        "evidence_chunks": evidence_chunks,
        "evidence_in_window": any(window_start <= c <= window_end for c in evidence_chunks),
        "answer_in_recent_obs": any(
            keyword_overlap(item.get("text", item.get("obs", "")), answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD
            for item in snapshot.get("recent_thinks", [])
        ),
        "answer_in_compressed": any(
            keyword_overlap(seg["text"], answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD
            for seg in snapshot["compressed_segments"]
        ),
    }


# ---------------------------------------------------------------------------
# Task Mining: Per-type
# ---------------------------------------------------------------------------


def mine_response_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine A1-type tasks: answer visible in current frames.

    Finds chunks with high-confidence facts and generates questions.
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]

    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue

        for fact in caption.get("atomic_facts", []):
            if not is_usable_fact(fact):
                continue

            # Ask at the same chunk or 1-2 chunks later (evidence still in window)
            for ask_offset in range(0, 3):
                ask_chunk = chunk_idx + ask_offset
                if ask_chunk >= len(observations):
                    continue
                if str(ask_chunk) not in snapshots and ask_chunk not in snapshots:
                    continue

                snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                if snapshot is None:
                    continue

                answer_keywords = extract_keywords(fact["fact"])
                gold_action, reason = determine_gold_action(
                    answer_keywords, snapshot, [chunk_idx], observations
                )

                if gold_action == "response" and reason == "answer_in_visual_window":
                    candidates.append({
                        "task_type": "response_from_frames",
                        "ask_chunk": ask_chunk,
                        "evidence_chunks": [chunk_idx],
                        "fact": fact["fact"],
                        "confidence": fact["confidence"],
                        "entities": [e["id"] for e in caption.get("visible_entities", [])],
                        "gold_action": gold_action,
                        "action_reason": reason,
                        "visibility": build_visibility_matrix(
                            ask_chunk, snapshot, answer_keywords, [chunk_idx]
                        ),
                    })
                    break  # One task per fact

    return candidates


def mine_response_from_memory_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine tasks where the answer is in recent_thinks text memory (not in visual window).

    The evidence chunk has already left the visual window, but the student's
    think text still contains the answer keywords in recent_thinks.
    Gold action = response (from_memory), NOT recall.
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]

    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue

        for fact in caption.get("atomic_facts", []):
            if not is_usable_fact(fact):
                continue

            answer_keywords = extract_keywords(fact["fact"])
            if not answer_keywords:
                continue

            # Ask after the evidence has left the visual window
            # but while the think is still in recent_thinks (not yet compressed)
            for ask_chunk in range(chunk_idx + VISUAL_WINDOW_CHUNKS + 1,
                                   min(num_chunks, chunk_idx + VISUAL_WINDOW_CHUNKS + 10)):
                snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                if snapshot is None:
                    continue

                gold_action, reason = determine_gold_action(
                    answer_keywords, snapshot, [chunk_idx], observations
                )

                if gold_action == "response" and "recent_think" in reason:
                    candidates.append({
                        "task_type": "response_from_memory",
                        "ask_chunk": ask_chunk,
                        "evidence_chunks": [chunk_idx],
                        "fact": fact["fact"],
                        "confidence": fact["confidence"],
                        "entities": [e["id"] for e in caption.get("visible_entities", [])],
                        "gold_action": "response",
                        "action_reason": reason,
                        "visibility": build_visibility_matrix(
                            ask_chunk, snapshot, answer_keywords, [chunk_idx]
                        ),
                    })
                    break  # One ask_time per fact

    return candidates


def mine_recall_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine recall tasks: answer NOT in any visible source.

    Finds facts that were observed earlier but are now outside the window
    and NOT in recent_thinks or compressed_summaries.
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]

    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue

        for fact in caption.get("atomic_facts", []):
            if not is_usable_fact(fact):
                continue

            answer_keywords = extract_keywords(fact["fact"])

            # Try asking much later (evidence must have left window)
            for ask_chunk in range(chunk_idx + VISUAL_WINDOW_CHUNKS + 3,
                                   min(num_chunks, chunk_idx + VISUAL_WINDOW_CHUNKS + 20)):
                snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                if snapshot is None:
                    continue

                gold_action, reason = determine_gold_action(
                    answer_keywords, snapshot, [chunk_idx], observations
                )

                if gold_action == "recall":
                    # Determine recall sub-type based on content
                    sub_type = classify_recall_subtype(caption, fact)

                    candidates.append({
                        "task_type": f"recall_{sub_type}",
                        "ask_chunk": ask_chunk,
                        "evidence_chunks": [chunk_idx],
                        "fact": fact["fact"],
                        "confidence": fact["confidence"],
                        "entities": [e["id"] for e in caption.get("visible_entities", [])],
                        "gold_action": "recall",
                        "action_reason": reason,
                        "visibility": build_visibility_matrix(
                            ask_chunk, snapshot, answer_keywords, [chunk_idx]
                        ),
                    })
                    break  # One ask_time per fact

    return candidates


def mine_compress_recall_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine B-type tasks: evidence was compressed, summary LOST the detail.

    The key: answer is NOT in compressed_summary (otherwise it's response, not recall).
    Student must recall from historical frames.
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]
    compression_events = rollout["compression_events"]

    for comp_event in compression_events:
        compressed_chunks = comp_event.get("compressed_thinks_chunks", comp_event.get("compressed_obs_chunks", []))
        summary_text = comp_event["summary"].get("text", "")

        # Find facts from compressed chunks that are NOT in the summary
        for chunk_idx in compressed_chunks:
            if chunk_idx >= len(evidence):
                continue
            caption = evidence[chunk_idx]
            if not caption.get("parse_success", False):
                continue

            for fact in caption.get("atomic_facts", []):
                if fact.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                    continue

                answer_keywords = extract_keywords(fact["fact"])

                # Check: is this fact MISSING from the summary?
                if keyword_overlap(summary_text, answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
                    continue  # Fact IS in summary → would be response, not recall

                # Find a valid ask_time after compression
                trigger_chunk = comp_event["trigger_chunk"]
                for ask_chunk in range(trigger_chunk + VISUAL_WINDOW_CHUNKS + 1,
                                       min(num_chunks, trigger_chunk + VISUAL_WINDOW_CHUNKS + 15)):
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot is None:
                        continue

                    gold_action, reason = determine_gold_action(
                        answer_keywords, snapshot, [chunk_idx], observations
                    )

                    if gold_action == "recall":
                        sub_type = classify_recall_subtype(caption, fact)
                        candidates.append({
                            "task_type": f"compress_recall_{sub_type}",
                            "ask_chunk": ask_chunk,
                            "evidence_chunks": [chunk_idx],
                            "compression_event_chunk": trigger_chunk,
                            "fact": fact["fact"],
                            "fact_in_summary": False,
                            "confidence": fact["confidence"],
                            "gold_action": "recall",
                            "action_reason": reason,
                            "visibility": build_visibility_matrix(
                                ask_chunk, snapshot, answer_keywords, [chunk_idx]
                            ),
                        })
                        break

    return candidates


def mine_compress_response_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine B8-type tasks: evidence in compressed summary → response (not recall).

    Tests whether the model correctly reads from compressed memory instead of recalling.
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]
    compression_events = rollout["compression_events"]

    for comp_event in compression_events:
        compressed_chunks = comp_event.get("compressed_thinks_chunks", comp_event.get("compressed_obs_chunks", []))
        summary_text = comp_event["summary"].get("text", "")

        for chunk_idx in compressed_chunks:
            if chunk_idx >= len(evidence):
                continue
            caption = evidence[chunk_idx]
            if not caption.get("parse_success", False):
                continue

            for fact in caption.get("atomic_facts", []):
                if fact.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                    continue

                answer_keywords = extract_keywords(fact["fact"])

                # Check: fact IS in summary → should be response
                if keyword_overlap(summary_text, answer_keywords) <= LEAKAGE_OVERLAP_THRESHOLD:
                    continue

                trigger_chunk = comp_event["trigger_chunk"]
                for ask_chunk in range(trigger_chunk + VISUAL_WINDOW_CHUNKS + 1,
                                       min(num_chunks, trigger_chunk + VISUAL_WINDOW_CHUNKS + 10)):
                    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                    if snapshot is None:
                        continue

                    gold_action, reason = determine_gold_action(
                        answer_keywords, snapshot, [chunk_idx], observations
                    )

                    if gold_action == "response" and "compressed" in reason:
                        candidates.append({
                            "task_type": "compress_response",
                            "ask_chunk": ask_chunk,
                            "evidence_chunks": [chunk_idx],
                            "compression_event_chunk": trigger_chunk,
                            "fact": fact["fact"],
                            "fact_in_summary": True,
                            "gold_action": "response",
                            "action_reason": reason,
                            "visibility": build_visibility_matrix(
                                ask_chunk, snapshot, answer_keywords, [chunk_idx]
                            ),
                        })
                        break

    return candidates


def mine_unanswerable_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine unanswerable tasks: questions about things NOT in the video.

    Three categories:
    1. Generic absent topics (audio, price, brand, speech)
    2. Entity-specific absent attributes (entity exists but attribute unseen)
    3. Counterfactual / temporal impossibility
    """
    import random as _rng
    candidates = []
    num_chunks = rollout["num_chunks"]
    snapshots = rollout["snapshots"]

    # Collect all entities and their observed attributes
    all_entities = {}  # entity_id -> set of observed attribute words
    all_evidence_text = ""
    for cap in evidence:
        all_evidence_text += json.dumps(cap).lower() + " "
        for e in cap.get("visible_entities", []):
            eid = e["id"]
            if eid not in all_entities:
                all_entities[eid] = set()
            for attr in e.get("attributes", []):
                all_entities[eid].update(attr.lower().split())

    # --- Category 1: Generic absent topics ---
    generic_absent = [
        ("What brand is the knife?", "brand", "factoid"),
        ("How much does this cost?", "price", "factoid"),
        ("What did the person say?", "speech_content", "factoid"),
        ("What music is playing in the background?", "audio", "factoid"),
        ("What is the temperature in the room?", "temperature", "factoid"),
        ("What language are they speaking?", "language", "factoid"),
        ("How many calories does this dish have?", "calories", "factoid"),
        ("What is the recipe name?", "recipe_name", "factoid"),
        ("Who is watching this?", "audience", "factoid"),
        ("What time of day is it?", "time_of_day", "factoid"),
    ]

    for question, topic, answer_type in generic_absent:
        if topic.lower() in all_evidence_text:
            continue
        # Vary ask positions across the video
        ask_positions = [num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4]
        for ask_chunk in ask_positions:
            snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
            if snapshot:
                candidates.append({
                    "task_type": "unanswerable",
                    "ask_chunk": ask_chunk,
                    "evidence_chunks": [],
                    "question": question,
                    "question_preset": question,
                    "gold_action": "response",
                    "action_reason": "unanswerable_no_evidence",
                    "gold_answer": "I cannot determine that from the available visual information.",
                    "answer_type": "uncertain",
                    "visibility": build_visibility_matrix(ask_chunk, snapshot, [], []),
                })
                break  # One position per question

    # --- Category 2: Entity-specific absent attributes ---
    absent_attr_templates = [
        ("What material is {entity} made of?", "material"),
        ("How heavy is {entity}?", "weight"),
        ("What brand is {entity}?", "brand"),
        ("How old is {entity}?", "age"),
        ("What is the exact size of {entity}?", "exact_size"),
    ]
    entity_list = list(all_entities.keys())
    for eid in entity_list[:5]:  # Limit to 5 entities
        for template, attr_topic in absent_attr_templates:
            if attr_topic in all_entities.get(eid, set()):
                continue
            if attr_topic in all_evidence_text:
                continue
            question = template.format(entity=eid.replace("_", " "))
            ask_chunk = min(num_chunks - 1, num_chunks // 2 + _rng.randint(0, 5))
            snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
            if snapshot:
                candidates.append({
                    "task_type": "unanswerable",
                    "ask_chunk": ask_chunk,
                    "evidence_chunks": [],
                    "question": question,
                    "question_preset": question,
                    "gold_action": "response",
                    "action_reason": "unanswerable_attribute_not_visible",
                    "gold_answer": f"I cannot determine the {attr_topic} of {eid.replace('_', ' ')} from the visual information.",
                    "answer_type": "uncertain",
                    "visibility": build_visibility_matrix(ask_chunk, snapshot, [], []),
                })
                break  # One absent-attr question per entity

    # --- Category 3: Counterfactual (entity never appeared) ---
    counterfactual_entities = ["dog", "cat", "child", "phone", "laptop", "car"]
    for cf_entity in counterfactual_entities:
        if cf_entity in all_evidence_text:
            continue
        question = f"What is the {cf_entity} doing?"
        ask_chunk = num_chunks // 2
        snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
        if snapshot:
            candidates.append({
                "task_type": "unanswerable",
                "ask_chunk": ask_chunk,
                "evidence_chunks": [],
                "question": question,
                "question_preset": question,
                "gold_action": "response",
                "action_reason": "unanswerable_entity_absent",
                "gold_answer": f"I don't see a {cf_entity} in the video.",
                "answer_type": "uncertain",
                "visibility": build_visibility_matrix(ask_chunk, snapshot, [], []),
            })

    return candidates


# ---------------------------------------------------------------------------
# Recall Sub-type Classification
# ---------------------------------------------------------------------------


def classify_recall_subtype(caption: Dict, fact: Dict) -> str:
    """Classify recall into sub-types based on content."""
    fact_text = fact.get("fact", "").lower()
    entities = caption.get("visible_entities", [])

    # RC2: OCR/numbers
    if caption.get("ocr") or any(c.isdigit() for c in fact_text):
        return "ocr_number"

    # RC1: visual attributes (color, shape, material)
    attr_words = {"red", "blue", "green", "white", "black", "silver", "wooden",
                  "round", "square", "small", "large", "tall", "short"}
    if any(w in fact_text for w in attr_words):
        return "visual_detail"

    # RC3: procedural (actions, steps)
    proc_words = {"slice", "cut", "add", "pour", "stir", "place", "move", "pick", "open"}
    if any(w in fact_text for w in proc_words):
        return "procedural"

    # RC4: state change
    if caption.get("state_changes"):
        return "state_change"

    return "general"


def mine_pending_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine S3/S4-type tasks: user asks about a future event (event-watch).

    Pattern:
    - User asks "Tell me when X happens" at ask_chunk
    - X actually happens at trigger_chunk (later)
    - Between ask_chunk and trigger_chunk: model should be silent (pending)
    - At trigger_chunk: model should respond

    Produces 2 samples per task:
    - Silent sample with pending question (before trigger)
    - Response sample when event is observed (at trigger)
    """
    candidates = []
    observations = rollout.get("thinks", rollout.get("observations", []))
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]

    # Natural event-watch question templates
    _event_watch_templates = [
        "Let me know when {event_short}.",
        "Notify me when {event_short}.",
        "Watch for when {event_short} and tell me.",
        "Alert me the moment {event_short}.",
        "Keep an eye out — tell me when {event_short}.",
    ]
    import random as _rng

    # Look for state_changes that can be turned into event-watch questions
    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue
        if not caption.get("state_changes"):
            continue

        for change in caption["state_changes"]:
            if not change or len(change) < 5:
                continue

            # Generate a natural question from the event
            event_short = change[0].lower() + change[1:] if change else change
            event_short = event_short.rstrip(".")
            question = _rng.choice(_event_watch_templates).format(event_short=event_short)

            # Ask 5-15 chunks before the event happens
            for offset in range(5, min(16, chunk_idx)):
                ask_chunk = chunk_idx - offset
                if ask_chunk < 0:
                    continue

                snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
                if snapshot is None:
                    continue

                # Pick an intermediate chunk for the silent-with-pending sample
                mid_chunk = (ask_chunk + chunk_idx) // 2
                mid_snapshot = snapshots.get(mid_chunk, snapshots.get(str(mid_chunk)))

                candidates.append({
                    "task_type": "pending_event_watch",
                    "ask_chunk": ask_chunk,
                    "trigger_chunk": chunk_idx,
                    "mid_chunk": mid_chunk,
                    "event": change,
                    "fact": change,
                    "question": question,
                    "gold_answer": change,
                    "answer_type": "event_watch",
                    "evidence_chunks": [chunk_idx],
                    "gold_action": "response",
                    "action_reason": "pending_trigger_satisfied",
                    "visibility": build_visibility_matrix(
                        chunk_idx, snapshots.get(chunk_idx, snapshots.get(str(chunk_idx), {})),
                        extract_keywords(change), [chunk_idx],
                    ) if snapshots.get(chunk_idx, snapshots.get(str(chunk_idx))) else {},
                })
                break  # One ask_time per event

    return candidates


# ---------------------------------------------------------------------------
# Main Task Mining
# ---------------------------------------------------------------------------


async def run_pass3(
    video_id: str,
    evidence: List[Dict],
    rollout: Dict,
    client,
    frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Run task mining for a single video.

    Mines all task types independently, then generates questions/answers via 397B.
    """
    all_candidates = {}

    # 1. Response tasks (answer in frames)
    response_tasks = mine_response_tasks(evidence, rollout)
    all_candidates["response_from_frames"] = response_tasks
    logger.info(f"  [{video_id}] Response(frames) candidates: {len(response_tasks)}")

    # 1b. Response tasks (answer in text memory, not in visual window)
    memory_tasks = mine_response_from_memory_tasks(evidence, rollout)
    all_candidates["response_from_memory"] = memory_tasks
    logger.info(f"  [{video_id}] Response(memory) candidates: {len(memory_tasks)}")

    # 2. Recall tasks (answer outside all visible sources)
    recall_tasks = mine_recall_tasks(evidence, rollout)
    all_candidates["recall"] = recall_tasks
    logger.info(f"  [{video_id}] Recall candidates: {len(recall_tasks)}")

    # 3. Compress-recall tasks (answer lost in compression)
    compress_recall = mine_compress_recall_tasks(evidence, rollout)
    all_candidates["compress_recall"] = compress_recall
    logger.info(f"  [{video_id}] Compress+Recall candidates: {len(compress_recall)}")

    # 4. Compress-response tasks (answer preserved in summary)
    compress_response = mine_compress_response_tasks(evidence, rollout)
    all_candidates["compress_response"] = compress_response
    logger.info(f"  [{video_id}] Compress+Response candidates: {len(compress_response)}")

    # 5. Unanswerable tasks
    unanswerable = mine_unanswerable_tasks(evidence, rollout)
    all_candidates["unanswerable"] = unanswerable
    logger.info(f"  [{video_id}] Unanswerable candidates: {len(unanswerable)}")

    # 6. Compress tasks (directly from rollout compression events)
    compress_tasks = []
    for event in rollout["compression_events"]:
        compress_tasks.append({
            "task_type": "compress",
            "trigger_chunk": event["trigger_chunk"],
            "summary": event["summary"],
            "gold_action": "compress",
        })
    all_candidates["compress"] = compress_tasks

    # 7. Pending/event-watch tasks (user asks about a future event)
    pending_tasks = mine_pending_tasks(evidence, rollout)
    all_candidates["pending"] = pending_tasks
    logger.info(f"  [{video_id}] Pending/event-watch candidates: {len(pending_tasks)}")

    # --- Apply per-type candidate limits ---
    # Reduces API cost: instead of generating questions for ALL candidates,
    # keep only top-N per type (diverse by ask_chunk spread).
    import random as _limit_rng
    _limit_rng.seed(hash(video_id))  # deterministic per video
    before_total = sum(len(t) for k, t in all_candidates.items()
                       if not k.startswith("_") and isinstance(t, list))
    for task_type, limit in MAX_CANDIDATES_PER_VIDEO.items():
        if limit <= 0 or task_type not in all_candidates:
            continue
        candidates = all_candidates[task_type]
        if len(candidates) > limit:
            # Sample to keep diversity across ask_chunks
            _limit_rng.shuffle(candidates)
            all_candidates[task_type] = candidates[:limit]
    after_total = sum(len(t) for k, t in all_candidates.items()
                      if not k.startswith("_") and isinstance(t, list))
    if before_total != after_total:
        logger.info(f"  [{video_id}] Candidate limiting: {before_total} → {after_total}")

    # --- Generate questions and answers via 397B ---
    # Deduplicate by fact: same fact across different task types shares one question.
    # This reduces 397B calls from O(candidates) to O(unique_facts).
    skip_types = {"compress", "pending"}
    fact_to_tasks: Dict[str, List[Dict]] = {}
    for task_type, tasks in all_candidates.items():
        if task_type in skip_types:
            continue
        for task in tasks:
            if not task.get("question") and "question_preset" not in task:
                # Key by fact + evidence_chunk: same fact at different times
                # may refer to different events (e.g. "chef adds salt" twice)
                evidence_chunk = task["evidence_chunks"][0] if task.get("evidence_chunks") else 0
                fact_key = f"{evidence_chunk}:{task.get('fact', '')}"
                if task.get("fact"):
                    fact_to_tasks.setdefault(fact_key, []).append(task)

    # Only generate question once per unique fact
    unique_tasks = []
    for fact_key, task_group in fact_to_tasks.items():
        unique_tasks.append(task_group[0])  # representative task

    if unique_tasks:
        await generate_questions_batch(unique_tasks, evidence, client, video_id,
                                       frame_paths=frame_paths)
        # Propagate question/answer to all tasks sharing the same fact
        for fact_key, task_group in fact_to_tasks.items():
            source = task_group[0]
            for task in task_group[1:]:
                task["question"] = source.get("question", "")
                task["gold_answer"] = source.get("gold_answer", "")
                task["answer_type"] = source.get("answer_type", "factoid")
                task["support_fact"] = source.get("support_fact", "")

    logger.info(
        f"  [{video_id}] Question generation: {len(unique_tasks)} unique facts "
        f"(from {sum(len(g) for g in fact_to_tasks.values())} candidates)"
    )

    return all_candidates


async def generate_questions_batch(
    tasks: List[Dict],
    evidence: List[Dict],
    client,
    video_id: str,
    frame_paths: Optional[List[str]] = None,
):
    """Generate questions and answers for candidate tasks via 397B.

    When frame_paths is provided, includes the evidence chunk's frames
    in the prompt so the 397B can generate more grounded questions.
    """
    from .pass1_evidence import build_vision_content, get_chunk_frame_paths

    for task in tasks:
        evidence_chunk = task["evidence_chunks"][0] if task["evidence_chunks"] else 0
        caption = evidence[evidence_chunk] if evidence_chunk < len(evidence) else {}

        entities_str = ", ".join(
            f'{e["id"]}({", ".join(e.get("attributes", []))})'
            for e in caption.get("visible_entities", [])
        )

        prompt = TASK_QUESTION_PROMPT.format(
            entity=entities_str,
            attributes=entities_str,
            fact=task.get("fact", ""),
            time=task["evidence_chunks"][0] * AGENT_CHUNK_SEC if task["evidence_chunks"] else 0,
            answer=task.get("fact", ""),
        )

        # Build message content with frames if available
        if frame_paths:
            chunk_frames = get_chunk_frame_paths(frame_paths, evidence_chunk)
            content = build_vision_content(prompt, chunk_frames)
        else:
            content = prompt

        raw = await client._call_one(
            messages=[{"role": "user", "content": content}],
            max_tokens=PASS_CONFIG["pass3_tasks"]["max_tokens"],
            temperature=PASS_CONFIG["pass3_tasks"]["temperature"],
            request_id=f"{video_id}_q_{task.get('ask_chunk', 0)}",
        )

        if raw:
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            try:
                parsed = json.loads(raw)
                task["question"] = parsed.get("question", "")
                task["answer_type"] = parsed.get("answer_type", "factoid")
                # Use concise_answer from 397B if available, else fall back to fact
                concise = parsed.get("concise_answer", "")
                task["gold_answer"] = concise if concise else task.get("fact", "")
                task["support_fact"] = task.get("fact", "")

                # Validate: question should not contain the answer
                if task["question"] and task["gold_answer"]:
                    q_words = set(re.findall(r'\b\w+\b', task["question"].lower()))
                    a_words = set(
                        w for w in re.findall(r'\b\w+\b', task["gold_answer"].lower())
                        if len(w) > 2
                    )
                    if a_words and a_words.issubset(q_words):
                        task["question"] = ""  # Question leaks answer, discard
            except (json.JSONDecodeError, ValueError):
                task["question"] = ""
                task["answer_type"] = "factoid"
                task["gold_answer"] = task.get("fact", "")
        else:
            task["question"] = ""
            task["gold_answer"] = task.get("fact", "")


# ---------------------------------------------------------------------------
# Coverage / leakage audit
# ---------------------------------------------------------------------------

CORE_TASK_TYPES = [
    "response_from_frames",
    "response_from_memory",
    "recall",
    "compress",
    "compress_response",
    "compress_recall",
    "unanswerable",
    "pending",
]


def _answer_leaks_in_question(task: Dict) -> bool:
    """Heuristic check: question should not reveal its concise answer."""
    question = task.get("question", "") or task.get("question_preset", "")
    answer = task.get("gold_answer", "")
    if not question or not answer:
        return False
    q_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', question.lower()))
    a_words = {
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', answer.lower())
        if len(w) > 2 and w not in {"the", "and", "with", "from", "into"}
    }
    if not a_words:
        return False
    return a_words.issubset(q_words)


def expected_task_types_for_rollout(rollout: Dict) -> List[str]:
    """Compute which task types should be possible for this video's length/state."""
    num_chunks = int(rollout.get("num_chunks", 0))
    duration = num_chunks * AGENT_CHUNK_SEC
    n_compressions = len(rollout.get("compression_events", []))

    expected = []
    if duration >= 24:
        expected.append("response_from_frames")
    if duration >= 30:
        expected.extend(["response_from_memory", "recall", "unanswerable"])
    if n_compressions > 0:
        expected.append("compress")
    if duration >= 50 and n_compressions > 0:
        expected.extend(["compress_response", "compress_recall"])
    if duration >= 60:
        expected.append("pending")
    return expected


def audit_task_coverage(video_id: str, all_candidates: Dict, rollout: Dict) -> Dict:
    """Audit whether task mining produced the expected task families.

    This is the explicit guard for: "某一任务没造出来" and "提前看到答案".
    It does not fail the pipeline by itself; downstream orchestration can decide
    whether to resample videos, lower filters, or block export.
    """
    counts = {
        task_type: len(all_candidates.get(task_type, []))
        for task_type in CORE_TASK_TYPES
    }
    expected = expected_task_types_for_rollout(rollout)
    missing = [t for t in expected if counts.get(t, 0) == 0]

    leakage_tasks = []
    action_minimality_risks = []
    for task_type, tasks in all_candidates.items():
        if task_type.startswith("_") or not isinstance(tasks, list):
            continue
        for idx, task in enumerate(tasks):
            if _answer_leaks_in_question(task):
                leakage_tasks.append({
                    "task_type": task_type,
                    "index": idx,
                    "reason": "question_contains_gold_answer",
                    "question": task.get("question", ""),
                    "gold_answer": task.get("gold_answer", ""),
                })
            visibility = task.get("visibility", {})
            if task.get("gold_action") == "recall" and (
                visibility.get("evidence_in_window")
                or visibility.get("answer_in_recent_obs")
                or visibility.get("answer_in_compressed")
            ):
                action_minimality_risks.append({
                    "task_type": task_type,
                    "index": idx,
                    "reason": "recall_selected_while_answer_visible",
                    "visibility": visibility,
                })

    return {
        "video_id": video_id,
        "num_chunks": rollout.get("num_chunks", 0),
        "duration_sec": int(rollout.get("num_chunks", 0)) * AGENT_CHUNK_SEC,
        "num_compression_events": len(rollout.get("compression_events", [])),
        "counts": counts,
        "expected_task_types": expected,
        "missing_expected_task_types": missing,
        "question_answer_leakage": leakage_tasks,
        "action_minimality_risks": action_minimality_risks,
        "passed": not missing and not leakage_tasks and not action_minimality_risks,
    }
