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
    PASS_CONFIG,
    RECALL_QUERY_PROMPT,
    TASK_QUESTION_PROMPT,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action Minimality
# ---------------------------------------------------------------------------


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text for matching."""
    stop = {"the", "a", "an", "is", "was", "were", "are", "in", "on", "at",
            "to", "of", "and", "or", "it", "its", "this", "that", "with"}
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def keyword_overlap(text: str, keywords: List[str]) -> float:
    """Compute fraction of keywords found in text (word-boundary match)."""
    if not keywords:
        return 0.0
    # Use word-boundary matching to avoid substring false positives
    # e.g., "red" should not match "prepared"
    text_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))
    found = sum(1 for kw in keywords if kw in text_words)
    return found / len(keywords)


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
    2. answer in recent_observations → response(from_memory)
    3. answer in compressed_summaries → response(from_compressed)
    4. answer in historical obs/frames → recall
    5. answer nowhere → unanswerable
    """
    # 1. Check visual window (by checking if evidence chunks are in window)
    window_start = snapshot["visual_window_start"]
    window_end = snapshot["chunk_idx"]
    evidence_in_window = any(window_start <= c <= window_end for c in evidence_chunks)
    if evidence_in_window:
        return "response", "answer_in_visual_window"

    # 2. Check recent_observations
    for obs_item in snapshot["recent_observations"]:
        if keyword_overlap(obs_item["obs"], answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
            return "response", "answer_in_recent_observations"

    # 3. Check compressed summaries
    for seg in snapshot["compressed_segments"]:
        if keyword_overlap(seg["text"], answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
            return "response", "answer_in_compressed_summary"

    # 4. Check if answer exists in historical observations (outside current visibility)
    for eidx in evidence_chunks:
        if eidx < len(observations):
            obs_text = observations[eidx]["observation"]
            if keyword_overlap(obs_text, answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD:
                return "recall", "answer_in_historical_observation"

    # 5. Answer not found anywhere accessible
    return "unanswerable", "answer_not_in_any_accessible_source"


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
        "n_recent_observations": len(snapshot["recent_observations"]),
        "evidence_chunks": evidence_chunks,
        "evidence_in_window": any(window_start <= c <= window_end for c in evidence_chunks),
        "answer_in_recent_obs": any(
            keyword_overlap(item["obs"], answer_keywords) > LEAKAGE_OVERLAP_THRESHOLD
            for item in snapshot["recent_observations"]
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
    observations = rollout["observations"]
    snapshots = rollout["snapshots"]

    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue

        for fact in caption.get("atomic_facts", []):
            if fact.get("confidence", 0) < CONFIDENCE_THRESHOLD:
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


def mine_recall_tasks(
    evidence: List[Dict],
    rollout: Dict,
) -> List[Dict]:
    """Mine recall tasks: answer NOT in any visible source.

    Finds facts that were observed earlier but are now outside the window
    and NOT in recent_observations or compressed_summaries.
    """
    candidates = []
    observations = rollout["observations"]
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]

    for caption in evidence:
        chunk_idx = caption.get("chunk_idx", 0)
        if not caption.get("parse_success", False):
            continue

        for fact in caption.get("atomic_facts", []):
            if fact.get("confidence", 0) < CONFIDENCE_THRESHOLD:
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
    observations = rollout["observations"]
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]
    compression_events = rollout["compression_events"]

    for comp_event in compression_events:
        compressed_chunks = comp_event.get("compressed_obs_chunks", [])
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
    observations = rollout["observations"]
    snapshots = rollout["snapshots"]
    num_chunks = rollout["num_chunks"]
    compression_events = rollout["compression_events"]

    for comp_event in compression_events:
        compressed_chunks = comp_event.get("compressed_obs_chunks", [])
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
    """Mine unanswerable tasks: questions about things NOT in the video."""
    candidates = []
    num_chunks = rollout["num_chunks"]
    snapshots = rollout["snapshots"]

    # Collect all entities ever seen
    all_entities = set()
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            all_entities.add(e["id"])

    # Generate questions about things NOT present
    absent_topics = [
        ("What brand is the knife?", "brand", "factoid"),
        ("How much does this cost?", "price", "factoid"),
        ("What did the person say?", "speech_content", "factoid"),
        ("What music is playing?", "audio", "factoid"),
    ]

    for question, topic, answer_type in absent_topics:
        # Check that the topic is genuinely absent from all evidence
        topic_in_evidence = any(
            topic.lower() in json.dumps(cap).lower()
            for cap in evidence
        )
        if topic_in_evidence:
            continue

        ask_chunk = num_chunks // 2  # Ask midway
        snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
        if snapshot:
            candidates.append({
                "task_type": "unanswerable",
                "ask_chunk": ask_chunk,
                "evidence_chunks": [],
                "question_preset": question,
                "gold_action": "response",
                "action_reason": "unanswerable_no_evidence",
                "gold_answer": "I cannot determine that from the available visual information.",
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


# ---------------------------------------------------------------------------
# Main Task Mining
# ---------------------------------------------------------------------------


async def run_pass3(
    video_id: str,
    evidence: List[Dict],
    rollout: Dict,
    client,
) -> Dict:
    """Run task mining for a single video.

    Mines all task types independently, then generates questions/answers via 397B.
    """
    all_candidates = {}

    # 1. Response tasks (answer in frames)
    response_tasks = mine_response_tasks(evidence, rollout)
    all_candidates["response_from_frames"] = response_tasks
    logger.info(f"  [{video_id}] Response candidates: {len(response_tasks)}")

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

    # --- Generate questions and answers via 397B ---
    tasks_needing_questions = []
    for task_type, tasks in all_candidates.items():
        for task in tasks:
            if task_type != "compress" and "question_preset" not in task:
                tasks_needing_questions.append(task)

    if tasks_needing_questions:
        await generate_questions_batch(tasks_needing_questions, evidence, client, video_id)

    return all_candidates


async def generate_questions_batch(
    tasks: List[Dict],
    evidence: List[Dict],
    client,
    video_id: str,
):
    """Generate questions and answers for candidate tasks via 397B."""
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

        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
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
                task["gold_answer"] = task.get("fact", "")
            except (json.JSONDecodeError, ValueError):
                task["question"] = ""
                task["answer_type"] = "factoid"
                task["gold_answer"] = task.get("fact", "")
        else:
            task["question"] = ""
            task["gold_answer"] = task.get("fact", "")
