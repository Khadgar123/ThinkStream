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
# Frame Path Helpers
# ---------------------------------------------------------------------------


def get_visual_frame_info(
    chunk_idx: int,
    frame_paths: List[str],
    window_start: Optional[int] = None,
) -> Dict:
    """Compute visual window metadata WITH real frame paths.

    At 1fps with 2 frames per chunk, frame indices = chunk_idx * 2, chunk_idx * 2 + 1.
    """
    if window_start is None:
        window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)

    chunk_indices = list(range(window_start, chunk_idx + 1))
    frame_indices = []
    paths = []
    for c in chunk_indices:
        for f_offset in range(2):  # 2 frames per chunk
            idx = c * 2 + f_offset
            frame_indices.append(idx)
            if frame_paths and idx < len(frame_paths):
                paths.append(frame_paths[idx])

    meta = {
        "video_start": window_start * AGENT_CHUNK_SEC,
        "video_end": (chunk_idx + 1) * AGENT_CHUNK_SEC,
        "frames": len(frame_indices),
        "chunk_indices": chunk_indices,
        "frame_indices": frame_indices,
    }
    if paths:
        meta["frame_paths"] = paths
    return meta


def get_recalled_frame_info(
    evidence_chunks: List[int],
    frame_paths: List[str],
    source: str = "historical_frames",
) -> Dict:
    """Compute recalled frame metadata with real frame paths."""
    frame_indices = []
    paths = []
    for c in evidence_chunks:
        for f_offset in range(2):
            idx = c * 2 + f_offset
            frame_indices.append(idx)
            if frame_paths and idx < len(frame_paths):
                paths.append(frame_paths[idx])

    t_start = evidence_chunks[0] * AGENT_CHUNK_SEC if evidence_chunks else 0
    t_end = (evidence_chunks[-1] + 1) * AGENT_CHUNK_SEC if evidence_chunks else 0

    info = {
        "time_range": [int(t_start), int(t_end)],
        "n_frames": len(frame_indices),
        "source": source,
        "frame_indices": frame_indices,
    }
    if paths:
        info["frame_paths"] = paths
    return info


# ---------------------------------------------------------------------------
# Sample Construction
# ---------------------------------------------------------------------------


def build_per_timestep_messages(
    snapshot: Dict,
    chunk_idx: int,
    video_path: str,
    assistant_output: str,
    user_text_suffix: str = "",
    recalled_frames: Optional[Dict] = None,
) -> Dict:
    """Build a single-turn messages sample matching inference_step() exactly.

    This is the canonical output format for SFT training.
    Each sample = one inference step snapshot.

    Args:
        snapshot: Pre-action memory state (compressed_segments + recent_thinks + pending)
        chunk_idx: Current chunk index
        video_path: Absolute path to video file
        assistant_output: The gold output string (think + action + payload)
        user_text_suffix: Additional text after memory block (question, compress_trigger, recall_result)
        recalled_frames: Optional recalled frame info for recall_response samples
    """
    # ── Memory text block (matches inference_step §0.2) ──
    text_parts = []

    # Compressed segments
    for seg in snapshot.get("compressed_segments", []):
        seg_json = json.dumps(
            {"time_range": seg["time_range"], "text": seg["text"]},
            ensure_ascii=False,
        )
        text_parts.append(f"<compressed>{seg_json}</compressed>")

    # Recent thinks
    for item in snapshot.get("recent_thinks", snapshot.get("recent_observations", [])):
        time_str = item.get("time", f"{item.get('chunk', 0)*2}-{item.get('chunk', 0)*2+2}")
        text = item.get("text", item.get("obs", ""))
        text_parts.append(f"[{time_str}] {text}")

    # Pending questions
    for pq in snapshot.get("pending_questions", []):
        since = pq.get("since_chunk", 0) * AGENT_CHUNK_SEC
        text_parts.append(f'<pending since="{int(since)}">{pq["question"]}</pending>')

    # User text suffix (question / compress_trigger / recall_result)
    if user_text_suffix:
        text_parts.append(user_text_suffix)

    memory_text = "\n".join(text_parts)

    # ── Visual window ──
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC

    user_content = [
        {
            "type": "video",
            "video_start": video_start,
            "video_end": video_end,
            "nframes": VISUAL_WINDOW_CHUNKS * 2,  # 24 frames
        },
    ]

    # Recalled frames (recall_response only)
    if recalled_frames:
        user_content.append({
            "type": "video",
            "video_start": recalled_frames["time_range"][0],
            "video_end": recalled_frames["time_range"][1],
            "nframes": recalled_frames.get("n_frames", 4),
        })

    user_content.append({"type": "text", "text": memory_text})

    # ── Assemble messages ──
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_output},
    ]

    return {
        "video_path": video_path,
        "protocol_version": "4action",
        "messages": messages,
    }


def build_sample_input(snapshot: Dict, user_input: str, visual_window_meta: Dict) -> Dict:
    """Construct the training sample input (what student model sees).

    Matches §2.2 of the design doc exactly.
    """
    # Memory block
    recent = snapshot.get("recent_thinks", snapshot.get("recent_observations", []))
    memory = {
        "compressed": snapshot["compressed_segments"],
        "recent_thinks": [
            f'[{item["time"]}] {item.get("text", item.get("obs", ""))}'
            for item in recent
        ],
    }

    # Pending questions (includes recall-awaiting questions)
    if snapshot.get("pending_questions"):
        pending_list = []
        for pq in snapshot["pending_questions"]:
            item = {
                "question": pq["question"],
                "since": pq.get("since_chunk", 0) * AGENT_CHUNK_SEC,
                "type": "awaiting_recall_response"
                if pq.get("last_action") == "recall"
                else "event_watch",
            }
            if pq.get("query"):
                item["query"] = pq["query"]
            pending_list.append(item)
        memory["pending"] = [
            item for item in pending_list
        ]

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
    visual_meta = get_visual_frame_info(chunk_idx, frame_paths)

    sample_input = build_sample_input(snapshot, user_input="", visual_window_meta=visual_meta)
    output = f"<think>{observation}</think><action>silent</action>"

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
    frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Build a C1 training sample for compression.

    C1: System triggers + specifies the gold range (teacher-chosen optimal).
    The model learns the compress behavior: given a trigger with range,
    produce a faithful summary.

    C2 (future): No trigger; model decides when and what to compress.
    """
    visual_meta = get_visual_frame_info(chunk_idx, frame_paths or [])
    summary = compression_event["summary"]
    time_range = summary.get("time_range", [0, 0])

    # C1: system trigger includes the specified range
    # IMPORTANT: use pre-action snapshot (without current think).
    # Compress range only covers thinks in the INPUT, not the current output think.
    trigger = f'<compress_trigger range="{time_range[0]}-{time_range[1]}"/>'

    sample_input = build_sample_input(
        snapshot, user_input=trigger, visual_window_meta=visual_meta
    )
    output = (
        f"<think>{observation}</think>"
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
            "compressed_range": time_range,
            "compressed_chunks": compression_event.get("compressed_thinks_chunks", []),
            "teacher_policy": compression_event.get("teacher_policy", {}),
            "phase": "C1",
        },
    }


def build_compress_sample_c2(
    chunk_idx: int,
    observation: str,
    snapshot: Dict,
    compression_event: Dict,
    frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Build a C2 training sample for compression.

    C2: System triggers (no range specified), model self-selects the range.
    Gold range comes from teacher policy (same as C1) but the model must learn
    to choose it autonomously.
    """
    visual_meta = get_visual_frame_info(chunk_idx, frame_paths or [])
    summary = compression_event["summary"]
    time_range = summary.get("time_range", [0, 0])

    # C2: trigger WITHOUT range — model decides what to compress
    trigger = "<compress_trigger/>"

    sample_input = build_sample_input(
        snapshot, user_input=trigger, visual_window_meta=visual_meta
    )
    output = (
        f"<think>{observation}</think>"
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
            "compressed_range": time_range,
            "compressed_chunks": compression_event.get("compressed_thinks_chunks", []),
            "teacher_policy": compression_event.get("teacher_policy", {}),
            "phase": "C2",
        },
    }


def build_merge_compress_sample(
    snapshot: Dict,
    seg_a: Dict,
    seg_b: Dict,
    merged_text: str,
    chunk_idx: int,
    observation: str,
    frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Build a training sample for second-level compression (summary of summaries).

    When compressed_segments exceed MAX_COMPRESSED_SEGMENTS, the system merges
    the two oldest segments. This sample teaches the model to do that merge.
    """
    visual_meta = get_visual_frame_info(chunk_idx, frame_paths or [])

    tr_a = seg_a["time_range"]
    tr_b = seg_b["time_range"]
    trigger = f'<merge_compress_trigger segments="{tr_a[0]}-{tr_a[1]},{tr_b[0]}-{tr_b[1]}"/>'

    sample_input = build_sample_input(
        snapshot, user_input=trigger, visual_window_meta=visual_meta
    )

    merged_summary = {
        "time_range": [tr_a[0], tr_b[1]],
        "text": merged_text,
        "source_segments": [tr_a, tr_b],
    }

    output = (
        f"<think>{observation}</think>"
        f"<action>compress</action>"
        f'<summary>{json.dumps(merged_summary, ensure_ascii=False)}</summary>'
    )

    return {
        "sample_type": "compress",
        "chunk_idx": chunk_idx,
        "input": sample_input,
        "output": output,
        "metadata": {
            "gold_action": "compress",
            "compressed_range": [tr_a[0], tr_b[1]],
            "phase": "C1",
            "task_type": "merge_compress",
        },
    }


async def build_response_sample(
    task: Dict,
    snapshot: Dict,
    observations: List[Dict],
    client,
    video_id: str,
    frame_paths: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Build a training sample for a response action.

    Uses 397B to generate the natural response text, constrained by gold_answer.
    """
    ask_chunk = task["ask_chunk"]
    visual_meta = get_visual_frame_info(ask_chunk, frame_paths or [])
    window_start = max(0, ask_chunk - VISUAL_WINDOW_CHUNKS + 1)
    # Determine evidence available to student
    evidence_text = ""
    reason = task.get("action_reason", "")
    if "visual_window" in reason:
        evidence_text = "Visible in current frames."
    elif "recent_think" in reason:
        for item in snapshot.get("recent_thinks", snapshot.get("recent_observations", [])):
            text = item.get("text", item.get("obs", ""))
            if task.get("fact", "").lower()[:20] in text.lower():
                evidence_text = text
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

    response_text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')

    # Build current observation (at ask time)
    current_obs = observations[ask_chunk]["think"] if ask_chunk < len(observations) else ""

    sample_input = build_sample_input(
        snapshot, user_input=task.get("question", ""), visual_window_meta=visual_meta
    )
    output = (
        f"<think>{current_obs}</think>"
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
    frame_paths: Optional[List[str]] = None,
) -> Optional[List[Dict]]:
    """Build training samples for a recall action (2 samples: query + response).

    Recall produces TWO samples:
    1. The recall query sample (model outputs recall action)
    2. The post-recall response sample (model sees recall_result, outputs response)

    recall_result uses ONLY student-accessible content (observations/summaries/frames).
    """
    ask_chunk = task["ask_chunk"]
    _fp = frame_paths or []
    visual_meta = get_visual_frame_info(ask_chunk, _fp)
    # --- Generate recall query ---
    # Build visible context from snapshot (what the student can actually see)
    visible_parts = []
    for seg in snapshot.get("compressed_segments", []):
        tr = seg["time_range"]
        visible_parts.append(f"[{tr[0]}-{tr[1]}] {seg['text'][:80]}")
    for obs_item in snapshot.get("recent_thinks", snapshot.get("recent_observations", [])):
        visible_parts.append(f"[{obs_item['time']}] {obs_item.get('text', obs_item.get('obs', ''))}")
    visible_context = "\n".join(visible_parts[-10:]) if visible_parts else "(minimal context)"

    # Derive time_range from visible memory timestamps (not from teacher evidence)
    all_times = []
    for seg in snapshot.get("compressed_segments", []):
        all_times.extend(seg["time_range"])
    for obs_item in snapshot.get("recent_thinks", snapshot.get("recent_observations", [])):
        parts = obs_item["time"].split("-")
        all_times.extend(int(p) for p in parts)
    time_range_str = f"0-{max(all_times)}" if all_times else "0-60"

    query_prompt = RECALL_QUERY_PROMPT.format(
        question=task.get("question", ""),
        visible_context=visible_context,
        time_range=time_range_str,
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

    # --- Leakage check: query must not contain answer or near-synonyms ---
    stop_words = {"the", "a", "an", "is", "was", "in", "on", "at", "to", "of", "and"}
    answer_keywords = set(
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', task.get("gold_answer", "").lower())
        if w not in stop_words and len(w) > 2
    )
    query_text = query_json.get("query", "")
    query_keywords = set(
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', query_text.lower())
        if w not in stop_words and len(w) > 2
    )
    leaked = answer_keywords & query_keywords
    if leaked:
        # Remove leaked words from query
        clean_query = " ".join(
            w for w in query_json["query"].split()
            if w.lower() not in leaked
        )
        if not clean_query.strip():
            # All query words were answer words — cannot salvage, discard sample
            return None
        query_json["query"] = clean_query
        # Re-check after cleanup
        remaining_leaked = answer_keywords & set(
            re.findall(r'\b[a-zA-Z0-9]+\b', clean_query.lower())
        )
        if remaining_leaked:
            return None  # Still leaking after cleanup

    # --- Simulate recall result (student-accessible content only) ---
    recall_result = simulate_recall_result(task, snapshot, observations, ask_chunk, query_json)

    # --- Handle recall failure/distractor: no gold_answer for response ---
    is_failed_recall = recall_result.get("noise_level") in ("distractor", "failure")

    if is_failed_recall:
        # Recall failed — generate uncertain response, NOT gold answer
        effective_answer = "I could not find enough evidence in the recalled results to answer confidently."
        effective_answer_type = "uncertain"
        effective_length = "20-60 tokens"
    else:
        effective_answer = task.get("gold_answer", "")
        effective_answer_type = task.get("answer_type", "factoid")
        effective_length = "5-40 tokens" if effective_answer_type == "factoid" else "20-80 tokens"

    # --- Generate post-recall response ---
    resp_prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=recall_result.get("text_content", ""),
        answer_type=effective_answer_type,
        gold_answer=effective_answer,
        length_guide=effective_length,
    )

    resp_raw = await client._call_one(
        messages=[{"role": "user", "content": resp_prompt}],
        max_tokens=256,
        temperature=0.3,
        request_id=f"{video_id}_postresp_{ask_chunk}",
    )

    if not resp_raw:
        return None

    response_text = re.sub(r'<think>.*?</think>', '', resp_raw, flags=re.DOTALL).strip().strip('"')
    current_obs = observations[ask_chunk]["think"] if ask_chunk < len(observations) else ""

    # --- Sample 1: Recall query ---
    sample1_input = build_sample_input(
        snapshot, user_input=task.get("question", ""), visual_window_meta=visual_meta
    )
    sample1_output = (
        f"<think>{current_obs}</think>"
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
                "query_contains_answer": bool(
                    set(re.findall(r'\b[a-zA-Z0-9]+\b', query_json.get("query", "").lower()))
                    & answer_keywords
                ),
            },
        },
    }

    # --- Sample 2: Post-recall response ---
    # Inject original question as pending so the model knows what to answer
    snapshot_with_pending = dict(snapshot)
    snapshot_with_pending["pending_questions"] = [{
        "question": task.get("question", ""),
        "since_chunk": ask_chunk,
        "last_action": "recall",
        "query": query_json,
    }]

    # Use returned_chunks (not evidence_chunks) — failure/distractor get different frames
    returned_chunks = recall_result.get("returned_chunks", [])
    if returned_chunks and recall_result.get("source") == "historical_frames":
        recalled_info = get_recalled_frame_info(
            returned_chunks, _fp, source="historical_frames"
        )
    else:
        recalled_info = None  # No frames for text-only recall or failure

    recall_visual = {**visual_meta}
    if recalled_info:
        recall_visual["recalled_frames"] = recalled_info
    sample2_input = build_sample_input(
        snapshot_with_pending,
        user_input="Continue following the protocol to respond.",
        visual_window_meta=recall_visual,
    )
    # Add recall_result to input
    sample2_input["recall_result"] = recall_result

    # Use actual visual observation for current chunk, not meta-description
    # Post-recall turn: no new observation (observation was already emitted in sample1
    # for this chunk_idx; re-emitting would duplicate it in memory at runtime).
    sample2_output = (
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
# Pending / Event-watch Samples
# ---------------------------------------------------------------------------


async def build_pending_samples(
    task: Dict,
    snapshots: Dict,
    observations: List[Dict],
    client,
    video_id: str,
    frame_paths: Optional[List[str]] = None,
) -> Optional[List[Dict]]:
    """Build training samples for a pending/event-watch task (3 samples).

    Sample 0: User requests event-watch → model outputs silent (pending starts).
    Sample 1: Silent with pending question (mid-point, event not yet happened).
    Sample 2: Response when the event is observed (at trigger_chunk).
    """
    ask_chunk = task["ask_chunk"]
    trigger_chunk = task["trigger_chunk"]
    mid_chunk = task.get("mid_chunk", (ask_chunk + trigger_chunk) // 2)
    _fp = frame_paths or []
    samples = []

    # --- Sample 0: Pending start (user asks, model stays silent) ---
    ask_snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
    if ask_snapshot is not None and ask_chunk < len(observations):
        start_visual = get_visual_frame_info(ask_chunk, _fp)
        ask_obs = observations[ask_chunk]["think"]
        # User input is the event-watch question; model should NOT answer yet
        start_input = build_sample_input(
            ask_snapshot, user_input=task.get("question", ""),
            visual_window_meta=start_visual,
        )
        start_output = f"<think>{ask_obs}</think><action>silent</action>"
        samples.append({
            "sample_type": "silent",
            "chunk_idx": ask_chunk,
            "input": start_input,
            "output": start_output,
            "metadata": {
                "task_type": "pending_start",
                "has_pending": False,  # pending is created AFTER this output
                "pending_question": task.get("question", ""),
            },
        })

    # --- Sample 1: Silent with pending (mid-point) ---
    mid_snapshot = snapshots.get(mid_chunk, snapshots.get(str(mid_chunk)))
    if mid_snapshot is None:
        return samples or None

    mid_snapshot_with_pending = dict(mid_snapshot)
    mid_snapshot_with_pending["pending_questions"] = [{
        "question": task.get("question", ""),
        "since_chunk": ask_chunk,
    }]

    mid_visual = get_visual_frame_info(mid_chunk, _fp)

    mid_obs = observations[mid_chunk]["think"] if mid_chunk < len(observations) else ""
    sample1_input = build_sample_input(mid_snapshot_with_pending, user_input="", visual_window_meta=mid_visual)
    sample1_output = f"<think>{mid_obs}</think><action>silent</action>"

    samples.append({
        "sample_type": "silent",
        "chunk_idx": mid_chunk,
        "input": sample1_input,
        "output": sample1_output,
        "metadata": {
            "task_type": "pending_silent",
            "has_pending": True,
            "pending_question": task.get("question", ""),
        },
    })

    # --- Sample 2: Response at trigger ---
    trigger_snapshot = snapshots.get(trigger_chunk, snapshots.get(str(trigger_chunk)))
    if trigger_snapshot is None:
        return samples

    trigger_snapshot_with_pending = dict(trigger_snapshot)
    trigger_snapshot_with_pending["pending_questions"] = [{
        "question": task.get("question", ""),
        "since_chunk": ask_chunk,
    }]

    trigger_visual = get_visual_frame_info(trigger_chunk, _fp)

    trigger_obs = observations[trigger_chunk]["think"] if trigger_chunk < len(observations) else ""

    # Generate response via 397B
    resp_prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=f"Current observation: {trigger_obs}. Event: {task.get('event', '')}",
        answer_type="factoid",
        gold_answer=task.get("event", ""),
        length_guide="20-60 tokens",
    )
    resp_raw = await client._call_one(
        messages=[{"role": "user", "content": resp_prompt}],
        max_tokens=256,
        temperature=0.3,
        request_id=f"{video_id}_pending_resp_{trigger_chunk}",
    )
    response_text = (
        re.sub(r'<think>.*?</think>', '', resp_raw, flags=re.DOTALL).strip().strip('"')
        if resp_raw else task.get("event", "Event observed.")
    )

    sample2_input = build_sample_input(trigger_snapshot_with_pending, user_input="", visual_window_meta=trigger_visual)
    sample2_output = (
        f"<think>{trigger_obs}</think>"
        f"<action>response</action>"
        f"<response>{response_text}</response>"
    )

    samples.append({
        "sample_type": "response",
        "chunk_idx": trigger_chunk,
        "input": sample2_input,
        "output": sample2_output,
        "metadata": {
            "task_type": "pending_response",
            "gold_action": "response",
            "action_reason": "pending_trigger_satisfied",
            "pending_question": task.get("question", ""),
            "event": task.get("event", ""),
        },
    })

    return samples


# ---------------------------------------------------------------------------
# Recall Result Simulation
# ---------------------------------------------------------------------------


def simulate_recall_result(
    task: Dict,
    snapshot: Dict,
    observations: List[Dict],
    ask_chunk: int = 0,
    query_json: Optional[Dict] = None,
) -> Dict:
    """Simulate what the retrieval system would return.

    ONLY uses student-accessible content from BEFORE ask_chunk.
    Returns returned_chunks so caller knows which frames to attach.
    """
    noise = random.random()
    evidence_chunks = task.get("evidence_chunks", [])
    noise_level = (
        "oracle" if noise < 0.70 else
        "noisy" if noise < 0.90 else
        "distractor" if noise < 0.95 else
        "failure"
    )

    returned_chunks = []  # Actual chunks the retrieval "found"

    if noise_level == "oracle":
        source, content = _get_correct_result(evidence_chunks, snapshot, observations, ask_chunk)
        returned_chunks = [c for c in evidence_chunks if c < ask_chunk]
    elif noise_level == "noisy":
        source, content = _get_correct_result(evidence_chunks, snapshot, observations, ask_chunk)
        distractor = _get_distractor(evidence_chunks, observations, ask_chunk)
        content = f"{distractor}\n---\n{content}"
        returned_chunks = [c for c in evidence_chunks if c < ask_chunk]
    elif noise_level == "distractor":
        source = "distractor"
        content, dist_chunk = _get_distractor_with_chunk(evidence_chunks, observations, ask_chunk)
        if dist_chunk is not None:
            returned_chunks = [dist_chunk]
    else:  # failure
        source = "failure"
        content = "No matching results found."
        returned_chunks = []

    time_range = ""
    if returned_chunks:
        t_start = returned_chunks[0] * AGENT_CHUNK_SEC
        t_end = (returned_chunks[-1] + 1) * AGENT_CHUNK_SEC
        time_range = f"{int(t_start)}-{int(t_end)}"

    return {
        "source": source,
        "time": time_range,
        "text_content": content,
        "noise_level": noise_level,
        "returned_chunks": returned_chunks,
    }


def _get_correct_result(
    evidence_chunks: List[int],
    snapshot: Dict,
    observations: List[Dict],
    ask_chunk: int = 0,
) -> tuple:
    """Get the correct recall result from student-accessible sources.

    Priority (most reliable → least reliable):
    1. Original observations from _retrieval_archive (highest fidelity)
    2. Low merge_level compressed segments (single compression)
    3. High merge_level compressed segments (multiple compressions, lossy)
    4. Historical frames only (no text, just frame pointers)

    Only uses observations from BEFORE ask_chunk (no future leakage).
    """
    if not evidence_chunks:
        return "failure", "No matching results found."

    evidence_time_start = evidence_chunks[0] * AGENT_CHUNK_SEC
    evidence_time_end = (evidence_chunks[-1] + 1) * AGENT_CHUNK_SEC

    frame_text = f"Retrieved frames from t={int(evidence_time_start)}-{int(evidence_time_end)}s."

    # Priority 1: Original observations (highest fidelity)
    obs_context = ""
    for ec in evidence_chunks:
        if ec < ask_chunk and ec < len(observations):
            obs_context += f" [{ec*AGENT_CHUNK_SEC}-{(ec+1)*AGENT_CHUNK_SEC}] {observations[ec]['think']}"

    if obs_context:
        return "historical_frames", f"{frame_text}\nText memory:{obs_context.strip()}"

    # Priority 2-3: Compressed segments (prefer low merge_level)
    matching_segs = []
    for seg in snapshot["compressed_segments"]:
        seg_start, seg_end = seg["time_range"]
        if seg_start <= evidence_time_start and seg_end >= evidence_time_end:
            matching_segs.append(seg)

    if matching_segs:
        # Sort by merge_level ascending (prefer least-merged)
        matching_segs.sort(key=lambda s: s.get("merge_level", 1))
        best_seg = matching_segs[0]
        source = "compressed_summary"
        if best_seg.get("merge_level", 1) > 2:
            source = "compressed_summary_high_merge"  # Flag for downstream quality awareness
        return source, best_seg["text"]

    # Priority 4: Frame pointers only
    return "historical_frames", frame_text


def _get_distractor(
    evidence_chunks: List[int],
    observations: List[Dict],
    ask_chunk: int = 0,
) -> str:
    """Get a distractor text (similar time, different content, past only)."""
    text, _ = _get_distractor_with_chunk(evidence_chunks, observations, ask_chunk)
    return text


def _get_distractor_with_chunk(
    evidence_chunks: List[int],
    observations: List[Dict],
    ask_chunk: int = 0,
) -> tuple:
    """Get a distractor with its chunk_idx. Returns (text, chunk_idx)."""
    available = [
        obs for obs in observations
        if obs["chunk_idx"] not in evidence_chunks
        and obs["chunk_idx"] < ask_chunk
    ]
    if available:
        pick = random.choice(available)
        t = f"{pick['chunk_idx'] * AGENT_CHUNK_SEC}-{(pick['chunk_idx']+1) * AGENT_CHUNK_SEC}"
        return f"[{t}] {pick['think']}", pick["chunk_idx"]

    return "Unrelated observation from a different time.", None


# ---------------------------------------------------------------------------
# Multi-turn Conversation Builder (replaces per-timestep assembly)
# ---------------------------------------------------------------------------


async def build_video_conversation(
    video_id: str,
    video_path: str,
    rollout: Dict,
    tasks: Dict,
    client,
    frame_paths: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Build one multi-turn conversation for an entire video.

    This is the main output format for SFT training. Each video produces
    ONE conversation sample with alternating user/assistant turns.

    Format matches _build_agent_messages expectations:
    - user: [{type: video, video_start, video_end}, {type: text, text: question}]
    - assistant: "<think>...</think><action>X</action>..."

    KV cache residual: after compression, old thinks remain in the conversation
    history. This is CORRECT — it matches inference where KV cache retains
    old thinks. The model learns to use summary as primary, old thinks as fallback.
    """
    observations = rollout.get("thinks", rollout.get("observations", []))
    num_chunks = rollout["num_chunks"]
    snapshots = rollout["snapshots"]
    compression_events = rollout["compression_events"]

    if not observations or num_chunks == 0:
        return None

    # --- Build lookup tables: what happens at each chunk? ---
    compress_at = {}  # chunk_idx -> compression_event
    for event in compression_events:
        compress_at[event["trigger_chunk"]] = event

    # Collect all tasks with ask_chunk, generate response/query text via 397B
    task_at = {}  # chunk_idx -> task (with generated response/query)
    recall_result_at = {}  # chunk_idx -> recall result (for post-recall turn)

    for task_type, task_list in tasks.items():
        if task_type.startswith("_") or not isinstance(task_list, list):
            continue
        for task in task_list:
            ask_chunk = task.get("ask_chunk")
            if ask_chunk is None or not task.get("question"):
                continue
            if ask_chunk in task_at:
                continue  # First task wins at this chunk

            # Generate response/query text via 397B
            gold_action = task.get("gold_action", "response")

            if gold_action == "response":
                resp_text = await _generate_response_text(
                    task, snapshots, observations, client, video_id
                )
                if resp_text:
                    task["_generated_response"] = resp_text
                    task_at[ask_chunk] = task

            elif gold_action == "recall":
                query_json, resp_text, recall_result = await _generate_recall_texts(
                    task, snapshots, observations, client, video_id
                )
                if query_json:
                    task["_generated_query"] = query_json
                    task["_generated_response"] = resp_text
                    task["_recall_result"] = recall_result
                    task_at[ask_chunk] = task
                    recall_result_at[ask_chunk] = recall_result

    # --- Action priority: question > compress > silent ---
    interaction_chunks = set(task_at.keys())

    # Pending tasks: need special handling (ask_chunk=silent, trigger_chunk=response)
    pending_tasks = []
    for task in tasks.get("pending", []):
        if task.get("question"):
            pending_tasks.append(task)

    pending_active = {}  # chunk_idx -> pending question text (active from ask to trigger)
    for pt in pending_tasks:
        ask = pt["ask_chunk"]
        trigger = pt["trigger_chunk"]
        for c in range(ask, trigger + 1):
            pending_active[c] = pt

    # --- Assemble multi-turn conversation ---
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * AGENT_CHUNK_SEC
        chunk_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
        think_text = observations[chunk_idx]["think"] if chunk_idx < len(observations) else ""

        # ── User message ──
        user_content = [{
            "type": "video",
            "video_start": chunk_start,
            "video_end": chunk_end,
            "nframes": 2,
        }]

        # Add compression trigger (if not preempted by user question)
        has_compress = chunk_idx in compress_at and chunk_idx not in interaction_chunks
        if has_compress:
            event = compress_at[chunk_idx]
            tr = event["summary"]["time_range"]
            user_content.append({
                "type": "text",
                "text": f'<compress_trigger range="{tr[0]}-{tr[1]}"/>',
            })

        # Add user question
        has_question = chunk_idx in task_at
        if has_question:
            user_content.append({
                "type": "text",
                "text": task_at[chunk_idx]["question"],
            })

        # Add pending question start (user asks, model should be silent for now)
        if chunk_idx in pending_active and chunk_idx == pending_active[chunk_idx].get("ask_chunk"):
            user_content.append({
                "type": "text",
                "text": pending_active[chunk_idx]["question"],
            })

        messages.append({"role": "user", "content": user_content})

        # ── Assistant message ──
        if has_question:
            task = task_at[chunk_idx]
            if task["gold_action"] == "response":
                resp = task.get("_generated_response", "")
                assistant_text = (
                    f"<think>{think_text}</think>"
                    f"<action>response</action>"
                    f"<response>{resp}</response>"
                )
            elif task["gold_action"] == "recall":
                query_json = task.get("_generated_query", {})
                assistant_text = (
                    f"<think>{think_text}</think>"
                    f"<action>recall</action>"
                    f'<query>{json.dumps(query_json, ensure_ascii=False)}</query>'
                )
        elif has_compress:
            event = compress_at[chunk_idx]
            summary_json = json.dumps(event["summary"], ensure_ascii=False)
            assistant_text = (
                f"<think>{think_text}</think>"
                f"<action>compress</action>"
                f"<summary>{summary_json}</summary>"
            )
        elif chunk_idx in pending_active and chunk_idx == pending_active[chunk_idx].get("trigger_chunk"):
            # Pending trigger: event happened, respond
            pt = pending_active[chunk_idx]
            resp = pt.get("event", "Event observed.")
            assistant_text = (
                f"<think>{think_text}</think>"
                f"<action>response</action>"
                f"<response>{resp}</response>"
            )
        else:
            assistant_text = f"<think>{think_text}</think><action>silent</action>"

        messages.append({"role": "assistant", "content": assistant_text})

        # ── Post-recall turn: inject recall result + response ──
        if has_question and task_at[chunk_idx]["gold_action"] == "recall":
            recall_result = task_at[chunk_idx].get("_recall_result", {})
            recall_text = recall_result.get("text_content", "No results found.")

            messages.append({
                "role": "user",
                "content": f"<recall_result>{recall_text}</recall_result>",
            })

            resp = task_at[chunk_idx].get("_generated_response", "")
            is_failed = recall_result.get("noise_level") in ("distractor", "failure")
            if is_failed:
                resp = "I could not find enough evidence to answer confidently."

            # Post-recall response: NO think (avoid double memory write)
            messages.append({
                "role": "assistant",
                "content": f"<action>response</action><response>{resp}</response>",
            })

    return {
        "video_id": video_id,
        "video_path": video_path,
        "protocol_version": "3action",  # Routes to preprocess_qwen_visual_agent in SFT
        "messages": messages,
        "num_chunks": num_chunks,
        "num_compression_events": len(compression_events),
        "num_tasks": len(task_at),
        "metadata": {
            "compression_events": [
                {"trigger_chunk": e["trigger_chunk"], "time_range": e["summary"]["time_range"]}
                for e in compression_events
            ],
            "task_chunks": list(task_at.keys()),
        },
    }


async def _generate_response_text(task, snapshots, observations, client, video_id):
    """Generate response text via 397B for a response task."""
    ask_chunk = task["ask_chunk"]
    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
    if not snapshot:
        return None

    evidence_text = "Based on available observations."
    answer_type = task.get("answer_type", "factoid")
    length_map = {"factoid": "5-40 tokens", "procedural": "40-120 tokens",
                  "summary": "80-200 tokens", "uncertain": "20-60 tokens"}

    prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=evidence_text,
        answer_type=answer_type,
        gold_answer=task.get("gold_answer", ""),
        length_guide=length_map.get(answer_type, "20-80 tokens"),
    )
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG["pass4_forks"]["max_tokens"],
        temperature=PASS_CONFIG["pass4_forks"]["temperature"],
        request_id=f"{video_id}_resp_{ask_chunk}",
    )
    if not raw:
        return None
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')


async def _generate_recall_texts(task, snapshots, observations, client, video_id):
    """Generate recall query + response texts via 397B."""
    ask_chunk = task["ask_chunk"]
    snapshot = snapshots.get(ask_chunk, snapshots.get(str(ask_chunk)))
    if not snapshot:
        return None, None, None

    # Build visible context
    visible_parts = []
    for seg in snapshot.get("compressed_segments", []):
        tr = seg["time_range"]
        visible_parts.append(f"[{tr[0]}-{tr[1]}] {seg['text'][:80]}")
    for obs_item in snapshot.get("recent_thinks", []):
        visible_parts.append(f"[{obs_item['time']}] {obs_item.get('text', '')}")
    visible_context = "\n".join(visible_parts[-10:]) or "(minimal)"

    all_times = []
    for seg in snapshot.get("compressed_segments", []):
        all_times.extend(seg["time_range"])
    for obs_item in snapshot.get("recent_thinks", []):
        parts = obs_item["time"].split("-")
        all_times.extend(int(p) for p in parts)
    time_range_str = f"0-{max(all_times)}" if all_times else "0-60"

    # Generate query
    query_prompt = RECALL_QUERY_PROMPT.format(
        question=task.get("question", ""),
        visible_context=visible_context,
        time_range=time_range_str,
    )
    query_raw = await client._call_one(
        messages=[{"role": "user", "content": query_prompt}],
        max_tokens=128, temperature=0.3,
        request_id=f"{video_id}_query_{ask_chunk}",
    )
    if not query_raw:
        return None, None, None

    query_raw = re.sub(r'<think>.*?</think>', '', query_raw, flags=re.DOTALL).strip()
    try:
        query_json = json.loads(query_raw)
    except (json.JSONDecodeError, ValueError):
        start = query_raw.find("{")
        end = query_raw.rfind("}")
        if start >= 0 and end > start:
            try:
                query_json = json.loads(query_raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                return None, None, None
        else:
            return None, None, None

    # Leakage cleanup
    stop = {"the", "a", "an", "is", "was", "in", "on", "at", "to", "of", "and"}
    answer_kw = set(
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', task.get("gold_answer", "").lower())
        if w not in stop and len(w) > 2
    )
    query_text = query_json.get("query", "")
    leaked = answer_kw & set(re.findall(r'\b[a-zA-Z0-9]+\b', query_text.lower()))
    if leaked:
        clean = " ".join(w for w in query_text.split() if w.lower() not in leaked)
        if not clean.strip():
            return None, None, None
        query_json["query"] = clean

    # Simulate recall result
    recall_result = simulate_recall_result(task, snapshot, observations, ask_chunk, query_json)

    # Generate response
    is_failed = recall_result.get("noise_level") in ("distractor", "failure")
    eff_answer = "I could not find enough evidence." if is_failed else task.get("gold_answer", "")
    eff_type = "uncertain" if is_failed else task.get("answer_type", "factoid")

    resp_prompt = RESPONSE_PROMPT.format(
        question=task.get("question", ""),
        evidence=recall_result.get("text_content", ""),
        answer_type=eff_type,
        gold_answer=eff_answer,
        length_guide="20-60 tokens" if is_failed else "5-40 tokens",
    )
    resp_raw = await client._call_one(
        messages=[{"role": "user", "content": resp_prompt}],
        max_tokens=256, temperature=0.3,
        request_id=f"{video_id}_postresp_{ask_chunk}",
    )
    resp_text = ""
    if resp_raw:
        resp_text = re.sub(r'<think>.*?</think>', '', resp_raw, flags=re.DOTALL).strip().strip('"')

    return query_json, resp_text, recall_result
