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
    - A snapshot dict with "compressed_segments", "recent_thinks"
    - A pre-structured dict with "compressed", "recent_thinks"
      (as used in per-timestep pipeline samples)

    Both paths produce identical output text.

    Pending status is NOT rendered here — it lives in
    `format_queries_block` as a query entry with empty answers list.
    All 12,405 v9.2 SFT samples were rendered with `pending_questions`
    empty (it was unused in production), so the model has never seen
    a `<pending>` tag. We assert the legacy field is empty here so
    any future caller accidentally populating it fails loudly instead
    of injecting an OOD tag the model can't interpret.
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

    # Defensive: SFT data has no <pending> tags; runtime no longer
    # populates pending_questions. If anyone smuggles in a non-empty
    # field, refuse to render rather than silently emit an OOD tag.
    legacy_pending = memory.get("pending_questions") or memory.get("pending")
    if legacy_pending:
        raise ValueError(
            f"format_memory_block received populated pending_questions "
            f"({len(legacy_pending)} entries). v11.1 represents pending "
            f"questions via the queries log (empty answers list), not "
            f"via a memory field. Caller must migrate."
        )

    return "\n".join(parts)


# Eval-side caps. Aligned to SFT distribution upper bounds:
#   - QUERIES_HISTORY_CAP=8 ≥ MAX_QUESTIONS_PER_TRAJECTORY=6 (no trim in dist)
#   - RECALL_TEXT_MAX_CHARS=1600 ≈ 4 × THINK_TOKENS.max(100 tok × ~4 char)
# Both are upper-bound guards; SFT samples never hit them.
# The "32k" eval profile (scripts/eval/eval_profiles.py) loosens further.
QUERIES_HISTORY_CAP = 8
RECALL_TEXT_MAX_CHARS = 1600


def format_queries_block(queries: List[Dict]) -> str:
    """Format the queries zone as a chronological event stream.

    Q and A events interleave on a timeline. All questions are shown
    (including unanswered/pending ones) so the model knows what it's
    tracking. Unanswered questions appear as Q without a following A.

    v9.4.2 (context-overflow fix): cap to the most recent
    QUERIES_HISTORY_CAP entries (default 8 = matches SFT max trajectory
    queries with slack). Eval-side override: see eval_profiles.py.
    We keep PENDING queries (unanswered) regardless of age — those are
    the only ones the model must still attend to — and trim ANSWERED
    queries to keep the most recent ones up to the cap.

    Example output:
      <queries>
      [10s] Q: Tell me when plating starts
      [20s] Q: What color is the apron?
      [20s] A: Red
      [50s] Q: How many tomatoes?
      [52s] A: 3
      </queries>
    """
    if not queries:
        return ""

    # Split: pending queries (always kept) vs answered (cap to recent)
    pending = [q for q in queries if not q.get("answers")]
    answered = [q for q in queries if q.get("answers")]
    # Keep all pending + last (cap - len(pending)) answered. If pending
    # alone exceeds cap, that's a signal of agent malfunction; keep them
    # all anyway — answered subset trims to 0.
    keep_n_answered = max(0, QUERIES_HISTORY_CAP - len(pending))
    queries = answered[-keep_n_answered:] + pending if keep_n_answered \
        else pending

    # Build chronological event list: (time, "Q"/"A", text)
    events = []
    for q in queries:
        answers = q.get("answers", [])
        question = q.get("question", "")
        ask_t = q.get("ask_time", "")

        # Question event — always shown (even if unanswered/pending)
        events.append((ask_t, "Q", question))
        # Answer event(s) — each carries its own timestamp
        for ans in answers:
            if isinstance(ans, dict):
                events.append((ans.get("time", ask_t), "A", ans.get("text", "")))
            else:
                events.append((q.get("response_time", ask_t), "A", str(ans)))

    if not events:
        return ""

    # Sort by time (stable sort preserves Q-before-A at same timestamp)
    events.sort(key=lambda e: (float(e[0]) if e[0] != "" else 0, 0 if e[1] == "Q" else 1))

    lines = []
    for t, kind, text in events:
        prefix = f"[{int(t)}s]" if t != "" else ""
        lines.append(f"{prefix} {kind}: {text}")

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
    # v9.4.2: cap recall text_content at RECALL_TEXT_MAX_CHARS. Default
    # 800 (~200 tok) matches the 16k profile; 32k profile bumps to 3000
    # via eval_profiles.apply_profile(). Top-4 retrieved thinks naturally
    # stack to 200-480 tokens; the cap catches pathological retrievals
    # where individual thinks were unusually long.
    if recall_result:
        rr_text = recall_result.get("text_content",
                                    recall_result.get("text", "")) or ""
        if len(rr_text) > RECALL_TEXT_MAX_CHARS:
            rr_text = rr_text[:RECALL_TEXT_MAX_CHARS] + "…"
        rr_json = json.dumps({
            "source": recall_result.get("source", ""),
            "time": recall_result.get("time", ""),
            "text": rr_text,
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
