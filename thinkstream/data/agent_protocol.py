"""Shared agent protocol: input construction and output parsing.

This module is the single source of truth for the agent's input/output format.
Used by:
- Data construction (scripts/agent_data_v5/pass4_forks.py)
- SFT training (thinkstream/sft/data_processor.py)
- Inference (thinkstream/model/agent_loop.py)

Any change to the protocol format MUST be made here to guarantee
train/inference format identity.
"""

import json
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants (canonical values, importable by all consumers)
# ---------------------------------------------------------------------------

AGENT_CHUNK_SEC = 2
VISUAL_WINDOW_CHUNKS = 12
FRAMES_PER_CHUNK = 2

SYSTEM_PROMPT = (
    "You are a streaming video agent. You observe 2-second video chunks and maintain memory.\n\n"
    "Each turn you receive: visual frames (recent 24s window) + text memory "
    "(compressed summaries + recent thinks + pending questions). "
    "Output exactly ONE action:\n\n"
    "1) <think>obs</think><action>silent</action>\n"
    "   Use when: no question to answer, or pending question whose answer is not yet visible.\n\n"
    "2) <think>obs</think><action>response</action><response>answer</response>\n"
    "   Use when: the answer is currently visible — in frames, recent thinks, "
    "compressed summaries, or a recall_result just returned by the system.\n\n"
    "3) <think>obs</think><action>recall</action>"
    '<query>{"query":"keywords","time_range":"start-end"}</query>\n'
    "   Use when: the answer is NOT in any visible source but you believe it was "
    "observed earlier. The system will search past observations and return a "
    "<recall_result> in your next input. Write 3-5 discriminative keywords "
    "(entity names + attributes), no answer values.\n\n"
    "4) <think>obs</think><action>compress</action>"
    '<summary>{"time_range":[s,e],"text":"..."}</summary>\n'
    "   Use when: system sends <compress_trigger>. Summarize the specified thinks. "
    "Retain ALL entity names, visual attributes, OCR, and state changes. "
    "If no range is specified, select the oldest thinks to compress.\n\n"
    "Think rules:\n"
    "- 40-60 tokens, describe ONLY what is newly visible in the current chunk.\n"
    "- No meta-reasoning, no sound/smell/emotion, no speculation.\n"
    "- Maintain consistent entity names from memory (e.g., chef_1, pot_1)."
)


# ---------------------------------------------------------------------------
# Memory Formatting
# ---------------------------------------------------------------------------


def format_memory_block(memory: Dict) -> str:
    """Format memory state as text with tags.

    Input can be either:
    - A snapshot dict with "compressed_segments", "recent_thinks", "pending_questions"
    - A pre-structured dict with "compressed", "recent_thinks", "pending"
      (as used in per-timestep pipeline samples)

    Both paths produce identical output text.
    """
    parts = []

    # Compressed segments
    compressed = memory.get("compressed_segments", memory.get("compressed", []))
    for seg in compressed:
        seg_json = json.dumps(
            {"time_range": seg["time_range"], "text": seg["text"]},
            ensure_ascii=False,
        )
        parts.append(f"<compressed>{seg_json}</compressed>")

    # Recent thinks
    recent = memory.get("recent_thinks", memory.get("recent_observations", []))
    for item in recent:
        if isinstance(item, str):
            # Already formatted "[time] text" string (from pipeline samples)
            parts.append(item)
        elif isinstance(item, dict):
            time_str = item.get(
                "time",
                f'{item.get("chunk", 0) * AGENT_CHUNK_SEC}-'
                f'{item.get("chunk", 0) * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC}',
            )
            text = item.get("text", item.get("obs", ""))
            parts.append(f"[{time_str}] {text}")

    # Pending questions
    pending = memory.get("pending_questions", memory.get("pending", []))
    for pq in pending:
        since = pq.get("since_chunk", pq.get("since", 0))
        if isinstance(since, int) and since > 100:
            # Already in seconds (from pipeline "since" field)
            since_sec = since
        else:
            # Chunk index → seconds
            since_sec = int(since * AGENT_CHUNK_SEC) if isinstance(since, int) else int(since)
        pq_json = json.dumps(
            {"since": since_sec, "question": pq["question"]},
            ensure_ascii=False,
        )
        parts.append(f"<pending>{pq_json}</pending>")

    return "\n".join(parts)


def format_queries_block(queries: List[Dict]) -> str:
    """Format the queries zone: only show answered Q&A pairs.

    Only includes questions that have at least one answer.
    Unanswered questions (answers=[]) are omitted — the model infers
    what's pending from user_input history and its own memory.

    This keeps the queries zone compact and avoids training the model
    to depend on an explicit "pending" list.
    """
    if not queries:
        return ""
    lines = []
    for q in queries:
        answers = q.get("answers", [])
        if not answers:
            continue  # skip unanswered — model infers from context
        q_json = json.dumps({
            "question": q.get("question", ""),
            "answers": answers,
        }, ensure_ascii=False)
        lines.append(q_json)
    if not lines:
        return ""
    return "<queries>\n" + "\n".join(lines) + "\n</queries>"


def build_user_content(
    memory_text: str,
    chunk_idx: int,
    video_path: str,
    *,
    user_input: str = "",
    queries: Optional[List[Dict]] = None,
    recalled_frames: Optional[Dict] = None,
    recall_result: Optional[Dict] = None,
    min_pixels: int = 100352,
    max_pixels: int = 150528,
    frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Build the user content list for a single-step message.

    Ordering (sft_engineering.md v3.0 §2.1, must not violate):
    <visual_window> + frames → <recalled_frames> + frames → <memory>
    → <queries> → <recall_result> → <user_input>

    Args:
        memory_text: Pre-formatted memory block from format_memory_block().
        chunk_idx: Current chunk index.
        video_path: Path to video file.
        user_input: Question, compress_trigger, "Continue...", or empty.
        recalled_frames: Optional recalled frame info for recall_response.
        recall_result: Optional recall result for recall_response.
        min_pixels, max_pixels: Resolution limits.
        frame_paths: Optional explicit frame paths (training). If None, uses
                     video_path with time range (inference).
    """
    chunk_sec = AGENT_CHUNK_SEC
    user_content = []

    # ── Zone B: Visual window + video frames (固定大小, position 稳定) ──
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * chunk_sec
    video_end = (chunk_idx + 1) * chunk_sec
    current_start = chunk_idx * chunk_sec
    current_end = current_start + chunk_sec
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    vw_header = json.dumps({
        "start": video_start,
        "end": video_end,
        "frames": n_frames,
        "current_time": [current_start, current_end],
    })
    user_content.append({
        "type": "text",
        "text": f"<visual_window>{vw_header}</visual_window>",
    })

    if frame_paths:
        user_content.append({"type": "video", "video": frame_paths})
    else:
        user_content.append({
            "type": "video",
            "video": video_path,
            "video_start": video_start,
            "video_end": video_end,
            "nframes": n_frames,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })

    # ── Zone B continued: Recalled frames (recall_response only) ──
    if recalled_frames:
        rf_header = json.dumps({
            "time_range": recalled_frames["time_range"],
            "source": recalled_frames.get("source", "historical_frames"),
            "n_frames": recalled_frames.get("n_frames", 4),
        })
        user_content.append({
            "type": "text",
            "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
        })
        if recalled_frames.get("frame_paths"):
            user_content.append({"type": "video", "video": recalled_frames["frame_paths"]})
        elif video_path:
            user_content.append({
                "type": "video",
                "video": video_path,
                "video_start": recalled_frames["time_range"][0],
                "video_end": recalled_frames["time_range"][1],
                "nframes": recalled_frames.get("n_frames", 4),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            })

    # ── Zone C: Memory block (可变大小, 追加式增长) ──
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>",
    })

    # ── Zone Q: Queries (past Q&A, independent of memory, not compressed) ──
    if queries:
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({
                "type": "text",
                "text": f"\n{queries_text}",
            })

    # ── Zone C continued: Recall result (recall_response only) ──
    if recall_result:
        rr_json = json.dumps({
            "source": recall_result.get("source", ""),
            "time": recall_result.get("time", ""),
            "text": recall_result.get("text_content", recall_result.get("text", "")),
        }, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })

    # ── Zone D: User input (每步变化) ──
    if user_input:
        user_content.append({
            "type": "text",
            "text": f"\n<user_input>{user_input}</user_input>",
        })

    return user_content


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------


def parse_agent_output(output_text: str) -> Dict:
    """Parse model output into structured components.

    All action types follow: <think>...</think><action>X</action>[payload]
    """
    result = {"raw": output_text, "think": "", "action": "", "payload": {}}

    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    action_match = re.search(r'<action>(.*?)</action>', output_text, re.DOTALL)
    if action_match:
        result["action"] = action_match.group(1).strip()

    if result["action"] == "response":
        resp_match = re.search(r'<response>(.*?)</response>', output_text, re.DOTALL)
        if resp_match:
            result["payload"]["response"] = resp_match.group(1).strip()

    elif result["action"] == "recall":
        query_match = re.search(r'<query>(.*?)</query>', output_text, re.DOTALL)
        if query_match:
            try:
                result["payload"]["query"] = json.loads(query_match.group(1))
            except (json.JSONDecodeError, ValueError):
                result["payload"]["query_raw"] = query_match.group(1).strip()

    elif result["action"] == "compress":
        summary_match = re.search(r'<summary>(.*?)</summary>', output_text, re.DOTALL)
        if summary_match:
            try:
                result["payload"]["summary"] = json.loads(summary_match.group(1))
            except (json.JSONDecodeError, ValueError):
                result["payload"]["summary_raw"] = summary_match.group(1).strip()

    return result
