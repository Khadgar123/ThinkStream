"""
Pass 4: Question-aware Forks

From cached memory snapshots, fork at ask_time and generate final training samples.

Key constraints:
- Past memory is NOT rewritten (question-blind principle)
- Standard answer acts as anchor constraining 397B generation
- recall_result uses only student-accessible content
"""

import json
import logging
import random
import re
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    PASS_CONFIG,
    RECALL_QUERY_PROMPT,
    RECALL_RETURN_FRAMES,
    RESPONSE_PROMPT,
    SYSTEM_PROMPT,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample Construction
# ---------------------------------------------------------------------------


def build_sample_input(snapshot: Dict, user_input: str, visual_window_meta: Dict) -> Dict:
    """Construct the training sample input (what student model sees).

    Matches §2.2 of the design doc exactly.
    """
    # Memory block
    memory = {
        "compressed": snapshot["compressed_segments"],
        "recent_observations": [
            f'[{item["time"]}] {item["obs"]}'
            for item in snapshot["recent_observations"]
        ],
    }

    # Pending questions
    if snapshot.get("pending_questions"):
        memory["pending"] = snapshot["pending_questions"]

    return {
        "system": SYSTEM_PROMPT,
        "memory": memory,
        "visual_window": visual_window_meta,
        "user_input": user_input,
    }


def build_silent_sample(
    chunk_idx: int,
    observation: str,
    snapshot: Dict,
    frame_paths: List[str],
) -> Dict:
    """Build a training sample for a silent timestep."""
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    visual_meta = {
        "video_start": window_start * AGENT_CHUNK_SEC,
        "video_end": (chunk_idx + 1) * AGENT_CHUNK_SEC,
        "frames": (chunk_idx - window_start + 1) * 2,
    }

    sample_input = build_sample_input(snapshot, user_input="", visual_window_meta=visual_meta)
    output = f"<observation>{observation}</observation><action>silent</action>"

    return {
        "sample_type": "silent",
        "chunk_idx": chunk_idx,
        "input": sample_input,
        "output": output,
    }


def build_compress_sample(
    chunk_idx: int,
    observation: str,
    snapshot: Dict,
    compression_event: Dict,
) -> Dict:
    """Build a training sample for a compression timestep."""
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    visual_meta = {
        "video_start": window_start * AGENT_CHUNK_SEC,
        "video_end": (chunk_idx + 1) * AGENT_CHUNK_SEC,
        "frames": (chunk_idx - window_start + 1) * 2,
    }

    sample_input = build_sample_input(
        snapshot, user_input="<compress_trigger/>", visual_window_meta=visual_meta
    )
    summary = compression_event["summary"]
    output = (
        f"<observation>{observation}</observation>"
        f"<action>compress</action>"
        f'<summary>{json.dumps(summary, ensure_ascii=False)}</summary>'
    )

    return {
        "sample_type": "compress",
        "chunk_idx": chunk_idx,
        "input": sample_input,
        "output": output,
        "metadata": {
            "gold_action": "compress",
            "compressed_range": summary.get("time_range"),
        },
    }


async def build_response_sample(
    task: Dict,
    snapshot: Dict,
    observations: List[Dict],
    client,
    video_id: str,
) -> Optional[Dict]:
    """Build a training sample for a response action.

    Uses 397B to generate the natural response text, constrained by gold_answer.
    """
    ask_chunk = task["ask_chunk"]
    window_start = max(0, ask_chunk - VISUAL_WINDOW_CHUNKS + 1)
    visual_meta = {
        "video_start": window_start * AGENT_CHUNK_SEC,
        "video_end": (ask_chunk + 1) * AGENT_CHUNK_SEC,
        "frames": (ask_chunk - window_start + 1) * 2,
    }

    # Determine evidence available to student
    evidence_text = ""
    reason = task.get("action_reason", "")
    if "visual_window" in reason:
        evidence_text = "Visible in current frames."
    elif "recent_observations" in reason:
        for item in snapshot["recent_observations"]:
            if task.get("fact", "").lower()[:20] in item["obs"].lower():
                evidence_text = item["obs"]
                break
    elif "compressed" in reason:
        for seg in snapshot["compressed_segments"]:
            keywords = task.get("fact", "").split()[:3]
            if any(kw.lower() in seg["text"].lower() for kw in keywords):
                evidence_text = seg["text"]
                break

    if not evidence_text:
        evidence_text = "Based on available observations."

    # Determine response length
    answer_type = task.get("answer_type", "factoid")
    length_map = {"factoid": "5-40 tokens", "procedural": "40-120 tokens",
                  "summary": "80-200 tokens", "uncertain": "20-60 tokens"}
    length_guide = length_map.get(answer_type, "20-80 tokens")

    # Generate response via 397B
    prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=evidence_text,
        answer_type=answer_type,
        gold_answer=task.get("gold_answer", ""),
        length_guide=length_guide,
    )

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG["pass4_forks"]["max_tokens"],
        temperature=PASS_CONFIG["pass4_forks"]["temperature"],
        request_id=f"{video_id}_resp_{ask_chunk}",
    )

    if not raw:
        return None

    response_text = raw.strip().strip('"')

    # Build current observation (at ask time)
    current_obs = observations[ask_chunk]["observation"] if ask_chunk < len(observations) else ""

    sample_input = build_sample_input(
        snapshot, user_input=task.get("question", ""), visual_window_meta=visual_meta
    )
    output = (
        f"<observation>{current_obs}</observation>"
        f"<action>response</action>"
        f"<response>{response_text}</response>"
    )

    return {
        "sample_type": "response",
        "chunk_idx": ask_chunk,
        "input": sample_input,
        "output": output,
        "metadata": {
            "task_type": task["task_type"],
            "gold_action": task["gold_action"],
            "gold_answer": task.get("gold_answer", ""),
            "action_reason": task.get("action_reason", ""),
            "visibility": task.get("visibility", {}),
            "question": task.get("question", ""),
        },
    }


async def build_recall_sample(
    task: Dict,
    snapshot: Dict,
    observations: List[Dict],
    client,
    video_id: str,
) -> Optional[List[Dict]]:
    """Build training samples for a recall action (2 samples: query + response).

    Recall produces TWO samples:
    1. The recall query sample (model outputs recall action)
    2. The post-recall response sample (model sees recall_result, outputs response)

    recall_result uses ONLY student-accessible content (observations/summaries/frames).
    """
    ask_chunk = task["ask_chunk"]
    window_start = max(0, ask_chunk - VISUAL_WINDOW_CHUNKS + 1)
    visual_meta = {
        "video_start": window_start * AGENT_CHUNK_SEC,
        "video_end": (ask_chunk + 1) * AGENT_CHUNK_SEC,
        "frames": (ask_chunk - window_start + 1) * 2,
    }

    # --- Generate recall query ---
    evidence_time = task["evidence_chunks"][0] * AGENT_CHUNK_SEC if task["evidence_chunks"] else 0
    answer_topic = " ".join(task.get("fact", "").split()[:5])  # First 5 words as topic

    query_prompt = RECALL_QUERY_PROMPT.format(
        question=task.get("question", ""),
        answer_topic=answer_topic,
        evidence_time=int(evidence_time),
        time_range=f"{int(max(0, evidence_time-4))}-{int(evidence_time+4)}",
    )

    query_raw = await client._call_one(
        messages=[{"role": "user", "content": query_prompt}],
        max_tokens=128,
        temperature=0.3,
        request_id=f"{video_id}_query_{ask_chunk}",
    )

    if not query_raw:
        return None

    query_raw = re.sub(r'<think>.*?</think>', '', query_raw, flags=re.DOTALL).strip()
    try:
        query_json = json.loads(query_raw)
    except (json.JSONDecodeError, ValueError):
        # Try extract
        start = query_raw.find("{")
        end = query_raw.rfind("}")
        if start >= 0 and end > start:
            try:
                query_json = json.loads(query_raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    # --- Leakage check: query must not contain answer ---
    answer_keywords = set(task.get("gold_answer", "").lower().split())
    query_keywords = set(query_json.get("query", "").lower().split())
    if answer_keywords & query_keywords - {"the", "a", "is", "was", "in", "on", "at"}:
        # Potential leakage — try to fix by removing answer words
        clean_query = " ".join(w for w in query_json["query"].split()
                               if w.lower() not in answer_keywords)
        query_json["query"] = clean_query

    # --- Simulate recall result (student-accessible content only) ---
    recall_result = simulate_recall_result(task, snapshot, observations)

    # --- Generate post-recall response ---
    resp_prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=recall_result.get("text_content", ""),
        answer_type=task.get("answer_type", "factoid"),
        gold_answer=task.get("gold_answer", ""),
        length_guide="5-40 tokens" if task.get("answer_type") == "factoid" else "20-80 tokens",
    )

    resp_raw = await client._call_one(
        messages=[{"role": "user", "content": resp_prompt}],
        max_tokens=256,
        temperature=0.3,
        request_id=f"{video_id}_postresp_{ask_chunk}",
    )

    if not resp_raw:
        return None

    response_text = resp_raw.strip().strip('"')
    current_obs = observations[ask_chunk]["observation"] if ask_chunk < len(observations) else ""

    # --- Sample 1: Recall query ---
    sample1_input = build_sample_input(
        snapshot, user_input=task.get("question", ""), visual_window_meta=visual_meta
    )
    sample1_output = (
        f"<observation>{current_obs}</observation>"
        f"<action>recall</action>"
        f'<query>{json.dumps(query_json, ensure_ascii=False)}</query>'
    )

    sample1 = {
        "sample_type": "recall_query",
        "chunk_idx": ask_chunk,
        "input": sample1_input,
        "output": sample1_output,
        "metadata": {
            "task_type": task["task_type"],
            "gold_action": "recall",
            "gold_answer": task.get("gold_answer", ""),
            "visibility": task.get("visibility", {}),
            "question": task.get("question", ""),
            "leakage_checks": {
                "query_contains_answer": bool(answer_keywords & query_keywords),
            },
        },
    }

    # --- Sample 2: Post-recall response ---
    sample2_input = build_sample_input(
        snapshot,
        user_input="Continue following the protocol to respond.",
        visual_window_meta={
            **visual_meta,
            "recalled_frames": {
                "time": recall_result.get("time"),
                "n_frames": RECALL_RETURN_FRAMES,
                "source": recall_result.get("source", "historical_frames"),
            },
        },
    )
    # Add recall_result to input
    sample2_input["recall_result"] = recall_result

    sample2_output = (
        f"<observation>Retrieved evidence from t={recall_result.get('time', '?')}.</observation>"
        f"<action>response</action>"
        f"<response>{response_text}</response>"
    )

    sample2 = {
        "sample_type": "recall_response",
        "chunk_idx": ask_chunk,
        "input": sample2_input,
        "output": sample2_output,
        "metadata": {
            "task_type": task["task_type"],
            "gold_action": "response",
            "gold_answer": task.get("gold_answer", ""),
            "recall_source": recall_result.get("source"),
        },
    }

    return [sample1, sample2]


# ---------------------------------------------------------------------------
# Recall Result Simulation
# ---------------------------------------------------------------------------


def simulate_recall_result(
    task: Dict,
    snapshot: Dict,
    observations: List[Dict],
) -> Dict:
    """Simulate what the retrieval system would return.

    ONLY uses student-accessible content:
    - Student's own observations (past)
    - Student's compressed summaries
    - Historical video frames (referenced by time)

    Never uses teacher_caption.
    """
    noise = random.random()
    evidence_chunks = task.get("evidence_chunks", [])

    if noise < 0.70:
        # Oracle: correct evidence
        source, content = _get_correct_result(evidence_chunks, snapshot, observations)
    elif noise < 0.90:
        # Noisy: correct in top-4 but not rank 1
        source, content = _get_correct_result(evidence_chunks, snapshot, observations)
        # Prepend a distractor
        distractor = _get_distractor(evidence_chunks, snapshot, observations)
        content = f"{distractor}\n---\n{content}"
    elif noise < 0.95:
        # All distractors
        source = "distractor"
        content = _get_distractor(evidence_chunks, snapshot, observations)
    else:
        # Failure
        source = "failure"
        content = "No matching results found."

    time_range = ""
    if evidence_chunks:
        t_start = evidence_chunks[0] * AGENT_CHUNK_SEC
        t_end = (evidence_chunks[-1] + 1) * AGENT_CHUNK_SEC
        time_range = f"{int(t_start)}-{int(t_end)}"

    return {
        "source": source,
        "time": time_range,
        "text_content": content,
        "noise_level": "oracle" if noise < 0.70 else "noisy" if noise < 0.90 else "distractor" if noise < 0.95 else "failure",
    }


def _get_correct_result(
    evidence_chunks: List[int],
    snapshot: Dict,
    observations: List[Dict],
) -> tuple:
    """Get the correct recall result from student-accessible sources.

    For recall tasks, the evidence is NOT in current visibility.
    Priority: historical frames > student past observation > compressed segment containing evidence.
    """
    if not evidence_chunks:
        return "failure", "No matching results found."

    evidence_time_start = evidence_chunks[0] * AGENT_CHUNK_SEC
    evidence_time_end = (evidence_chunks[-1] + 1) * AGENT_CHUNK_SEC

    # Priority 1: Return historical frames (most useful for visual detail recall)
    # This is a reference — actual frames get added to visual window during inference
    frame_text = f"Retrieved frames from t={int(evidence_time_start)}-{int(evidence_time_end)}s."
    # Also include the student's original observation for that time as text context
    obs_context = ""
    for ec in evidence_chunks:
        if ec < len(observations):
            obs_context += f" [{ec*AGENT_CHUNK_SEC}-{(ec+1)*AGENT_CHUNK_SEC}] {observations[ec]['observation']}"

    if obs_context:
        return "historical_frames", f"{frame_text}\nText memory:{obs_context.strip()}"

    # Priority 2: Find relevant compressed segment (matching time range)
    for seg in snapshot["compressed_segments"]:
        seg_start, seg_end = seg["time_range"]
        if seg_start <= evidence_time_start and seg_end >= evidence_time_end:
            return "compressed_summary", seg["text"]

    # Fallback: frame reference only
    return "historical_frames", frame_text


def _get_distractor(
    evidence_chunks: List[int],
    snapshot: Dict,
    observations: List[Dict],
) -> str:
    """Get a distractor (similar time, different content)."""
    # Pick a random observation NOT from evidence chunks
    available = [
        obs for obs in observations
        if obs["chunk_idx"] not in evidence_chunks
    ]
    if available:
        pick = random.choice(available)
        t = f"{pick['chunk_idx'] * AGENT_CHUNK_SEC}-{(pick['chunk_idx']+1) * AGENT_CHUNK_SEC}"
        return f"[{t}] {pick['observation']}"

    return "Unrelated observation from a different time."
