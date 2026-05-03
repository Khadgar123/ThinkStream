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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants (canonical values, importable by all consumers)
# ---------------------------------------------------------------------------

# v12.5: canonical values now live in scripts/agent_data_v5/config.py.
# Kept here as fallbacks when that import isn't available (deployed inference
# environments without the data-construction package).
try:
    from scripts.agent_data_v5.config import (
        AGENT_CHUNK_SEC,
        VISUAL_WINDOW_CHUNKS,
        FRAMES_PER_CHUNK,
    )
except ImportError:
    AGENT_CHUNK_SEC = 1
    VISUAL_WINDOW_CHUNKS = 16
    FRAMES_PER_CHUNK = 2


def infer_video_metadata(
    frames: Sequence[Any],
    *,
    fps: Optional[float] = None,
    start_frame_index: int = 0,
    total_num_frames: Optional[int] = None,
) -> Dict:
    """Infer Qwen3-VL video metadata for pre-sampled frame lists.

    Qwen3-VL uses ``fps`` + ``frames_indices`` to render text-layer timestamp
    anchors for video frames. Project frame files are normally named
    ``frame_000001.jpg`` (1-based), so this helper converts those names back to
    zero-based indices. Non-path frame objects fall back to a contiguous range
    starting at ``start_frame_index``.
    """
    eff_fps = float(fps or (FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)))
    frame_seq = list(frames) if frames is not None else []
    indices: List[int] = []

    for offset, frame in enumerate(frame_seq):
        idx: Optional[int] = None
        if isinstance(frame, (str, Path)):
            stem = Path(str(frame)).stem
            raw = stem[6:] if stem.startswith("frame_") else stem
            if raw.isdigit():
                parsed = int(raw)
                # Project frame_*.jpg files are 1-based. Plain numeric names
                # are left as-is because some eval frame dumps use 0-based
                # second offsets such as 000000.jpg, 000001.jpg.
                if stem.startswith("frame_"):
                    parsed -= 1
                idx = max(0, parsed)
        if idx is None:
            idx = int(start_frame_index) + offset
        indices.append(int(idx))

    inferred_total = max(indices) + 1 if indices else len(frame_seq)
    return {
        "fps": eff_fps,
        "frames_indices": indices,
        "total_num_frames": int(total_num_frames or inferred_total),
        "do_sample_frames": False,
    }

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

    # Build chronological event list: (time, "Q"/"A"/"O", text)
    # "O" = Options (rendered for pending MC queries; v12.13 P0-3 fix).
    events = []
    for q in queries:
        answers = q.get("answers", [])
        question = q.get("question", "")
        ask_t = q.get("ask_time", "")

        # Question event — always shown (even if unanswered/pending)
        events.append((ask_t, "Q", question))

        # v12.13 fix (P0-3): for pending MC queries, render the options
        # right after the Q line so the model sees A-D choices when the
        # response chunk fires LATER than the ask (forward / silent_then
        # _response). Without this, pending MC queries reduce to "pick a
        # letter without seeing options".
        if (not answers
                and q.get("answer_form") == "multiple_choice"
                and q.get("options")):
            opts = " ".join(q["options"])    # e.g., "A) red B) blue C) ..."
            events.append((ask_t, "O", opts))

        # Answer event(s) — each carries its own timestamp
        for ans in answers:
            if isinstance(ans, dict):
                events.append((ans.get("time", ask_t), "A", ans.get("text", "")))
            else:
                events.append((q.get("response_time", ask_t), "A", str(ans)))

    if not events:
        return ""

    # Sort by time (stable sort preserves Q-before-O-before-A at same timestamp)
    _kind_order = {"Q": 0, "O": 1, "A": 2}
    events.sort(key=lambda e: (float(e[0]) if e[0] != "" else 0,
                                _kind_order.get(e[1], 3)))

    lines = []
    for t, kind, text in events:
        prefix = f"[{int(t)}s]" if t != "" else ""
        if kind == "O":
            lines.append(f"{prefix} Options: {text}")
        else:
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
    # v12.12 (2026-05-02): defaults aligned to RUNTIME_MM_PROCESSOR_KWARGS in
    # scripts.agent_data_v5.config — Qwen3-VL smart_resize bounds for the
    # student/runtime profile. SFT/RL/Eval/deploy ALL use these values; pass1a
    # uses HIRES (set explicitly via mm_processor_kwargs at request level).
    min_pixels: int = 130_000,
    max_pixels: int = 220_000,
    frame_paths: Optional[List[str]] = None,
    inter_chunk: bool = False,
) -> List[Dict]:
    """Build the user content list for a single-step message.

    Ordering (v12.12, 2026-05-01 — supersedes v3.0 zone order):
    <memory> → <queries> → <visual_window> + frames →
    <recalled_frames> + frames → <recall_result> → <user_input>

    Why this order: memory and queries are monotonically appended across
    chunks of one trajectory (modulo periodic compression rewrites), so
    placing them FIRST makes the largest stable token prefix. With
    `--enable-prefix-caching` on the vLLM serving side, the prefix
    [system + memory_at_t-1 + queries_at_t-1] is byte-identical to the
    head of [system + memory_at_t + queries_at_t] up through chunk t-1's
    last appended token → cache hits ~25-40% of the per-chunk prefill
    instead of the ~5% (system only) that the old vision-first layout
    achieved. Visual window changes every chunk (sliding) so it's the
    cache-miss boundary; placing it AFTER the stable text means the miss
    starts later in the sequence, not at the front.

    MROPE temporal alignment for vision tokens is metadata-driven (via
    `frames_indices` in video_metadata, see processing_qwen3_vl.py),
    not position-in-sequence-driven, so reordering does not break the
    `<X.X seconds>` per-frame temporal anchors.

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
        inter_chunk: v12.6 — when True (compress system trigger fires
                     between two visual chunks), DROP <visual_window> and
                     the video frame block. Matches pass5 inter_chunk
                     shape C (compress sample has memory + trigger only,
                     no visual context). Compression is a system event
                     between visual timesteps; treating it as a visual
                     decision creates train/infer divergence.
    """
    chunk_sec = AGENT_CHUNK_SEC
    user_content = []

    # ── Memory block (FIRST — stable monotonic prefix, v12.12) ──
    # No leading "\n": memory is the first block in user content.
    user_content.append({
        "type": "text",
        "text": f"<memory>\n{memory_text}\n</memory>",
    })

    # ── Queries (past Q&A, also monotonic; second-stable prefix) ──
    if queries:
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({
                "type": "text",
                "text": f"\n{queries_text}",
            })

    # ── Visual window + video frames (cache-miss boundary) ──
    # v12.6: inter_chunk compress turns SKIP this block entirely (matches
    # pass5 shape C). Compression is a system event between visual chunks
    # and consumes no new frames; including a visual_window here would
    # diverge from the SFT distribution.
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * chunk_sec
    video_end = (chunk_idx + 1) * chunk_sec
    current_start = chunk_idx * chunk_sec
    current_end = current_start + chunk_sec
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    if not inter_chunk:
        vw_header = json.dumps({
            "start": video_start,
            "end": video_end,
            "frames": n_frames,
            "current_time": [current_start, current_end],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<visual_window>{vw_header}</visual_window>",
        })

        if frame_paths:
            # v12.6: attach video_metadata so Qwen3-VL processor renders
            # per-frame timestamp tokens (`<X.X seconds>`) with REAL video
            # time. Without metadata, processor defaults to fps=24 and
            # frame_indices=0..N → wrong timestamps that always start at 0.
            # Real video time = (chunk_idx × FRAMES_PER_CHUNK + i) / FPS.
            # See processing_qwen3_vl.py:217-224 for how metadata drives
            # the `<X.X seconds>` text-layer temporal anchor.
            #
            # v12.12: also include min_pixels/max_pixels at video item level.
            # qwen-vl-utils.process_vision_info reads these and emits them as
            # mm_processor_kwargs, which vLLM forwards to the Qwen3-VL
            # processor's smart_resize. This is the HF-direct + offline-vLLM
            # parity path; the OpenAI HTTP path uses request-top-level
            # mm_processor_kwargs (set in vllm_client._call_one).
            user_content.append({
                "type": "video",
                "video": frame_paths,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video_metadata": {
                    "fps": float(FRAMES_PER_CHUNK / chunk_sec),
                    "frames_indices": [
                        window_start * FRAMES_PER_CHUNK + i
                        for i in range(len(frame_paths))
                    ],
                    "total_num_frames": (chunk_idx + 1) * FRAMES_PER_CHUNK,
                    "do_sample_frames": False,
                },
            })
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

    # ── Recalled frames (recall_response only) ──
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
            # v12.6: timestamps for recall frames must reference their
            # ORIGINAL video time (not start at 0). Reconstruct
            # frames_indices from time_range to get the real `<T seconds>`
            # text token anchored to historical chunk position.
            tr_start, tr_end = recalled_frames["time_range"]
            n_rf = len(recalled_frames["frame_paths"])
            rf_chunks = max(1, (tr_end - tr_start))  # in chunks (chunk_sec=1)
            user_content.append({
                "type": "video",
                "video": recalled_frames["frame_paths"],
                "min_pixels": min_pixels,    # v12.12: runtime profile bounds
                "max_pixels": max_pixels,
                "video_metadata": {
                    "fps": float(FRAMES_PER_CHUNK / chunk_sec),
                    "frames_indices": [
                        int(tr_start * FRAMES_PER_CHUNK) + i for i in range(n_rf)
                    ],
                    "total_num_frames": int(tr_end * FRAMES_PER_CHUNK),
                    "do_sample_frames": False,
                },
            })
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

    # ── Recall result (recall_response only) ──
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

    # ── User input (LAST — every step varies) ──
    if user_input:
        user_content.append({
            "type": "text",
            "text": f"\n<user_input>{user_input}</user_input>",
        })

    return user_content


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------
# Architecture:
#   answer (terminal)   = <answer>text</answer> or <answer></answer> (silent)
#   tool (recall)       = <tool_call>{"name":"recall","arguments":{...}}</tool_call>
#   system event (compress) = system injects <compress_trigger/> into user role
#                             (boolean signal only — NO range, v12.12); the
#                             assistant emits a compress tool_call carrying its
#                             OWN derived time_range + summary text
# Tools registered via system <tools> block (auto-rendered by chat_template
# when tools=tools is passed to apply_chat_template).

SYSTEM_PROMPT_V12 = (
    "You are a streaming video agent. You observe 1-second video chunks and maintain memory.\n\n"
    "Each turn you receive: visual frames (recent 16s window) + memory state. "
    "You may either (a) call a tool, (b) emit a final answer, or (c) emit an empty "
    "answer if no response is warranted.\n\n"
    "Tools:\n"
    "- recall: search past observations by keywords + time range. Use when the answer is "
    "NOT in any visible source but you believe it was observed earlier. "
    "You may call recall AT MOST ONCE per question. After receiving "
    "<recall_result>, emit <answer> directly — do not call recall again.\n"
    "- compress: summarize a chunk range. Called ONLY when the system injects "
    "<compress_trigger/> into your input as a memory-pressure signal. "
    "You must derive the time range to compress yourself from <memory> "
    "contents (oldest contiguous chunks that can be safely condensed). "
    "Retain entity names, visual attributes, OCR, state changes.\n\n"
    "Output format (every turn must follow this exactly):\n"
    "  <think>40-60 tokens describing only what is newly visible</think>\n"
    "  Then ONE of:\n"
    "    <tool_call>{\"name\":\"recall\",\"arguments\":{...}}</tool_call>\n"
    "    <tool_call>{\"name\":\"compress\",\"arguments\":{...}}</tool_call>\n"
    "    <answer>response text</answer>\n"
    "    <answer></answer>   (silent — no question to answer right now)\n\n"
    "Think rules: describe ONLY what is newly visible in the current chunk. "
    "No meta-reasoning, no sound/smell/emotion, no speculation. "
    "Maintain consistent entity names from memory."
)


# Tool JSON schemas — passed as `tools=TOOLS_SCHEMA` to apply_chat_template.
# Format follows OpenAI function-calling spec, recognized by Qwen2.5-VL's
# chat template which auto-renders <tools>...</tools> in the system prompt.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search past video observations by keywords and time range. "
                "Returns matched historical thinks. Use when the answer is "
                "not in any visible source but was observed earlier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "3-5 discriminative keywords (entity names + attributes). "
                            "No answer values. Example: 'red apron chef pot'."
                        ),
                    },
                    "time_range": {
                        "type": "string",
                        "description": (
                            "Time range in seconds, format 'start-end'. "
                            "Example: '20-60'. Constrains search to this window."
                        ),
                    },
                },
                "required": ["query", "time_range"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress",
            "description": (
                "Summarize a chunk range when the system signals memory pressure "
                "via <compress_trigger/>. You decide which range from <memory> to "
                "compress. Output a concise summary retaining all entities, "
                "attributes, and state changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": (
                            "[start_sec, end_sec] of the range to summarize. "
                            "You select this range from <memory> contents "
                            "(oldest contiguous chunks under memory pressure)."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "The summary text. Retain entity names, visual "
                            "attributes, OCR text, and state changes."
                        ),
                    },
                },
                "required": ["time_range", "text"],
            },
        },
    },
]


def build_assistant_content_v12(
    *,
    think: str,
    kind: str,                    # "answer" | "recall" | "compress"
    answer_text: str = "",        # for kind="answer" (empty → silent)
    recall_query: Optional[Dict] = None,
    compress_summary: Optional[Dict] = None,
) -> str:
    """Build assistant message content in v12.0 format.

    Returns a single string with <think>...</think> followed by exactly one
    of: <tool_call>{...}</tool_call> | <answer>...</answer>.

    Args:
        think: think content (40-60 tokens recommended).
        kind: which terminal to emit.
        answer_text: text inside <answer>...</answer> (empty for silent).
        recall_query: dict with "query" + "time_range" keys.
        compress_summary: dict with "time_range" (list) + "text" keys.
    """
    parts = [f"<think>{think}</think>"]

    if kind == "answer":
        # Empty string → <answer></answer> = silent.
        parts.append(f"<answer>{answer_text}</answer>")
    elif kind == "recall":
        if not recall_query:
            raise ValueError("kind='recall' requires recall_query dict")
        tool_call = {
            "name": "recall",
            "arguments": {
                "query": recall_query.get("query", ""),
                "time_range": recall_query.get("time_range", ""),
            },
        }
        parts.append(
            f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>'
        )
    elif kind == "compress":
        if not compress_summary:
            raise ValueError("kind='compress' requires compress_summary dict")
        tool_call = {
            "name": "compress",
            "arguments": {
                "time_range": compress_summary.get("time_range", []),
                "text": compress_summary.get("text", ""),
            },
        }
        parts.append(
            f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>'
        )
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Expected answer|recall|compress.")

    return "".join(parts)


def parse_agent_output_v12(output_text: str) -> Dict:
    """Parse v12.0 agent output (think + tool_call|answer).

    Returns:
        {
            "raw": str,
            "think": str,
            "kind": "answer" | "recall" | "compress" | "unknown",
            "answer_text": str | None,         # set when kind=answer
            "tool_call": dict | None,          # parsed JSON when kind=recall|compress
            "format_error": str | None,        # set when parsing fails
        }
    """
    result = {
        "raw": output_text,
        "think": "",
        "kind": "unknown",
        "answer_text": None,
        "tool_call": None,
        "format_error": None,
    }

    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    tool_match = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)

    # Both present → format error (must be one or the other, not both)
    if answer_match and tool_match:
        result["format_error"] = "both <answer> and <tool_call> present"
        return result

    if answer_match:
        result["kind"] = "answer"
        result["answer_text"] = answer_match.group(1).strip()
        return result

    if tool_match:
        try:
            tool_obj = json.loads(tool_match.group(1).strip())
        except (json.JSONDecodeError, ValueError) as e:
            result["format_error"] = f"tool_call JSON parse error: {e}"
            return result

        name = tool_obj.get("name", "")
        if name == "recall":
            result["kind"] = "recall"
        elif name == "compress":
            result["kind"] = "compress"
        else:
            result["format_error"] = f"unknown tool name: {name!r}"
            return result
        result["tool_call"] = tool_obj
        return result

    result["format_error"] = "neither <answer> nor <tool_call> emitted"
    return result


def has_compress_trigger(user_text: str) -> bool:
    """Check if a user message contains a system-injected <compress_trigger/>.

    Used by training/eval to verify trigger→tool_call binding, and by the
    rollout controller to know whether the assistant must emit compress.
    """
    return bool(re.search(r'<compress_trigger\b', user_text or ""))


def extract_compress_trigger_range(user_text: str) -> Optional[List[int]]:
    """Extract a legacy trigger range if present.

    Current v12 data uses boolean ``<compress_trigger/>`` and puts the gold
    range only in the assistant compress tool_call. This parser is retained
    for archived v11/v12.0 samples and eval fixtures that still carry a
    range attribute.
    """
    m = re.search(
        r"<compress_trigger\s+range\s*=\s*['\"]?(\d+)\s*-\s*(\d+)['\"]?\s*/?>",
        user_text or "",
    )
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2))]
