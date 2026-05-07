"""Shared agent protocol: input construction and output parsing.

This module is the single source of truth for the agent's input/output format.
Used by:
- Data construction (scripts/agent_data_v5/pass2_rollout.py / pass5_messages.py)
- SFT training (thinkstream/sft/data_processor.py)
- RL rollout (verl/recipe_thinkstream/streaming_agent_loop.py)
- Inference (thinkstream/model/agent_loop.py)

Any change to the protocol format MUST be made here to guarantee
train/inference format identity.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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
        RECALL_RETURN_CHUNKS,
        compute_visual_window_start,
    )
except ImportError:
    AGENT_CHUNK_SEC = 1
    VISUAL_WINDOW_CHUNKS = 16
    FRAMES_PER_CHUNK = 2
    RECALL_RETURN_CHUNKS = 4

    def compute_visual_window_start(
        chunk_idx: int,
        visual_window_chunks: int = VISUAL_WINDOW_CHUNKS,
        mode: Optional[str] = None,
    ) -> int:
        return max(0, int(chunk_idx) - int(visual_window_chunks) + 1)


COMPRESS_TRIGGER_TAG = "<compress_trigger/>"
_COMPRESS_TRIGGER_RE = re.compile(r"<compress_trigger\b")


def _contains_compress_trigger(user_text: str) -> bool:
    return bool(_COMPRESS_TRIGGER_RE.search(user_text or ""))


def build_compress_trigger_user_input() -> str:
    """Canonical system-injected user input for inter-chunk compression.

    The compression instructions live in the compression system prompt. The
    user-side payload stays as a minimal legacy event marker so SFT/RL/eval
    can detect compression turns without mixing policy rules into user input.
    """
    return COMPRESS_TRIGGER_TAG


def normalize_user_input_for_turn(user_input: str, *, inter_chunk: bool = False) -> str:
    """Render legacy bare compress triggers as explicit compaction events."""
    text = str(user_input or "")
    if inter_chunk and _contains_compress_trigger(text):
        return build_compress_trigger_user_input()
    return text


def format_user_input_block(user_input: str, *, inter_chunk: bool = False) -> str:
    """Canonical tagged user_input block used by SFT/RL/eval/runtime."""
    text = normalize_user_input_for_turn(user_input, inter_chunk=inter_chunk)
    if not text:
        return ""
    return f"\n<user_input>{text}</user_input>"


# User-input placement. Keep the current external event/question first so it
# cannot be buried behind a long memory block.
USER_INPUT_POSITION = "front"


def normalize_user_input_position(position: Optional[str] = None) -> str:
    value = (
        position
        or os.environ.get("THINKSTREAM_USER_INPUT_POSITION")
        or USER_INPUT_POSITION
    )
    value = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "first": "front",
        "prepend": "front",
        "head": "front",
        "last": "tail",
        "append": "tail",
        "end": "tail",
        "compress_first": "compress_front",
        "inter_chunk_front": "compress_front",
    }
    value = aliases.get(value, value)
    if value not in {"tail", "front", "compress_front"}:
        return USER_INPUT_POSITION
    return value


def user_input_should_prepend(
    *,
    inter_chunk: bool = False,
    user_input_position: Optional[str] = None,
) -> bool:
    position = normalize_user_input_position(user_input_position)
    return position == "front" or (position == "compress_front" and inter_chunk)


def select_recall_chunks(
    chunks: Optional[Sequence[Any]],
    max_chunks: Optional[int] = None,
) -> List[int]:
    """Canonical recall chunk post-processing.

    Retrieval ranks candidate chunks first; every SFT/RL/eval caller then
    de-duplicates, caps to top-K, and sorts the selected chunk ids before
    rendering frames. This prevents any path from expanding a recall time range
    into an unbounded number of visual frames.
    """
    limit = RECALL_RETURN_CHUNKS if max_chunks is None else int(max_chunks)
    if limit <= 0:
        return []
    selected: List[int] = []
    seen = set()
    for raw in chunks or []:
        try:
            chunk = int(raw)
        except (TypeError, ValueError):
            continue
        if chunk < 0 or chunk in seen:
            continue
        selected.append(chunk)
        seen.add(chunk)
        if len(selected) >= limit:
            break
    return sorted(selected)


def recall_time_range_for_chunks(
    chunks: Optional[Sequence[Any]],
    *,
    chunk_sec: float = AGENT_CHUNK_SEC,
) -> Optional[List[int]]:
    """Return the exclusive-end video time range covered by selected chunks."""
    selected = select_recall_chunks(chunks)
    if not selected:
        return None
    start = min(selected) * float(chunk_sec)
    end = (max(selected) + 1) * float(chunk_sec)
    return [int(start), int(end)]


def recall_time_string_for_chunks(
    chunks: Optional[Sequence[Any]],
    *,
    chunk_sec: float = AGENT_CHUNK_SEC,
) -> str:
    tr = recall_time_range_for_chunks(chunks, chunk_sec=chunk_sec)
    if not tr:
        return ""
    return f"{tr[0]}-{tr[1]}"


def build_recalled_frames_metadata(
    chunks: Optional[Sequence[Any]],
    frame_paths: Optional[Sequence[Any]] = None,
    *,
    source: str = "historical_frames",
    chunk_sec: float = AGENT_CHUNK_SEC,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> Optional[Dict[str, Any]]:
    """Build the canonical <recalled_frames> metadata block.

    When frame_paths are supplied, callers should build them from the same
    selected chunks returned by select_recall_chunks().
    """
    selected = select_recall_chunks(chunks)
    if not selected:
        return None
    tr = recall_time_range_for_chunks(selected, chunk_sec=chunk_sec)
    if not tr:
        return None
    paths = list(frame_paths or [])
    out: Dict[str, Any] = {
        "time_range": tr,
        "n_frames": len(paths) if paths else len(selected) * int(frames_per_chunk),
        "source": source,
    }
    if paths:
        out["frame_paths"] = paths
    return out


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


FRAME_PROTOCOL_TS_IMAGE = "ts_image"
FRAME_PROTOCOL_VIDEO_META = "video_meta"
FRAME_PROTOCOL_ENV = "THINKSTREAM_FRAME_PROTOCOL"
VALID_FRAME_PROTOCOLS = {FRAME_PROTOCOL_TS_IMAGE, FRAME_PROTOCOL_VIDEO_META}


def normalize_frame_protocol(frame_protocol: Optional[str] = None) -> str:
    """Return the active student/eval frame protocol.

    The protocol switch is deliberately late-bound. Teacher pass caches store
    frame paths, timestamps, memory, questions, and answers; SFT/RL/eval decide
    only at render time whether those same frames are carried as timestamped
    images or as a pre-sampled Qwen video block with metadata.
    """
    value = (frame_protocol or os.environ.get(FRAME_PROTOCOL_ENV)
             or FRAME_PROTOCOL_TS_IMAGE)
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "timestamp_image": FRAME_PROTOCOL_TS_IMAGE,
        "timestamped_image": FRAME_PROTOCOL_TS_IMAGE,
        "timestamped_images": FRAME_PROTOCOL_TS_IMAGE,
        "image": FRAME_PROTOCOL_TS_IMAGE,
        "images": FRAME_PROTOCOL_TS_IMAGE,
        "video": FRAME_PROTOCOL_VIDEO_META,
        "video_metadata": FRAME_PROTOCOL_VIDEO_META,
        "video_meta_frames": FRAME_PROTOCOL_VIDEO_META,
    }
    value = aliases.get(value, value)
    if value not in VALID_FRAME_PROTOCOLS:
        raise ValueError(
            f"Unsupported frame protocol {frame_protocol!r}; expected one of "
            f"{sorted(VALID_FRAME_PROTOCOLS)}"
        )
    return value


def append_timestamped_image_list(
    content: List[Dict],
    frames: Sequence[Any],
    *,
    fps: Optional[float] = None,
    start_frame_index: int = 0,
    total_num_frames: Optional[int] = None,
    latest_start_frame_index: Optional[int] = None,
    context_label: str = "older context",
    timestamp_labels: Optional[Sequence[str]] = None,
    image_key: str = "image",
    image_url_encoder: Optional[Callable[[Any], str]] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Append canonical timestamped pre-extracted frames to chat content.

    This is the robust timestamped-image side of the AB test. The same
    ordered pre-extracted frames can alternatively be rendered by
    append_video_metadata_frame_list() as a native Qwen video block with
    explicit metadata. The timestamped-image protocol is:

        {"type": "text", "text": "<frame ts=\"12.5\" role=\"latest chunk\" />"}
        {"type": "image", "image": "/abs/frame_000026.jpg", ...}

    OpenAI-compatible rollout uses the same frame-tag text with
    ``image_url`` items by setting ``image_key="image_url"`` and passing an
    encoder. Local SFT/RL/eval use ``image`` items so qwen-vl-utils and vLLM
    process them as images while the frame tag supplies video time.
    """
    frame_seq = list(frames) if frames is not None else []
    if not frame_seq:
        return

    eff_fps = float(fps or (FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)))
    metadata = infer_video_metadata(
        frame_seq,
        fps=eff_fps,
        start_frame_index=start_frame_index,
        total_num_frames=total_num_frames,
    )
    indices = list(metadata.get("frames_indices") or [])

    if timestamp_labels is not None and len(timestamp_labels) != len(frame_seq):
        raise ValueError(
            "timestamp_labels length must match frames length "
            f"({len(timestamp_labels)} != {len(frame_seq)})"
        )
    if image_key not in {"image", "image_url"}:
        raise ValueError(f"Unsupported image_key={image_key!r}")

    for offset, frame in enumerate(frame_seq):
        frame_idx = int(indices[offset]) if offset < len(indices) else (
            int(start_frame_index) + offset
        )
        if timestamp_labels is not None:
            label = str(timestamp_labels[offset])
        elif latest_start_frame_index is not None and frame_idx >= int(latest_start_frame_index):
            label = "latest chunk"
        else:
            label = context_label

        content.append({
            "type": "text",
            "text": f'<frame ts="{frame_idx / eff_fps:.1f}" role="{label}" />',
        })

        if image_key == "image_url":
            if image_url_encoder is None:
                raise ValueError("image_url_encoder is required for image_url items")
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url_encoder(frame)},
            })
        else:
            item = {"type": "image", "image": frame}
            if min_pixels is not None:
                item["min_pixels"] = min_pixels
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)


def append_video_metadata_frame_list(
    content: List[Dict],
    frames: Sequence[Any],
    *,
    fps: Optional[float] = None,
    start_frame_index: int = 0,
    total_num_frames: Optional[int] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Append pre-extracted frames as one Qwen video block with metadata.

    This is the native-video side of the AB test. The frames are still the
    exact project-extracted JPEGs, so no server-side video decoding or
    re-sampling is introduced. Qwen receives the frame list plus
    ``video_metadata`` and ``do_sample_frames=False`` so timestamps are derived
    from ``fps`` and ``frames_indices``.
    """
    frame_seq = list(frames) if frames is not None else []
    if not frame_seq:
        return

    metadata = infer_video_metadata(
        frame_seq,
        fps=fps,
        start_frame_index=start_frame_index,
        total_num_frames=total_num_frames,
    )
    item: Dict[str, Any] = {
        "type": "video",
        "video": frame_seq,
        "video_metadata": metadata,
    }
    if min_pixels is not None:
        item["min_pixels"] = min_pixels
    if max_pixels is not None:
        item["max_pixels"] = max_pixels
    content.append(item)


def append_visual_frames(
    content: List[Dict],
    frames: Sequence[Any],
    *,
    frame_protocol: Optional[str] = None,
    fps: Optional[float] = None,
    start_frame_index: int = 0,
    total_num_frames: Optional[int] = None,
    latest_start_frame_index: Optional[int] = None,
    context_label: str = "older context",
    timestamp_labels: Optional[Sequence[str]] = None,
    image_key: str = "image",
    image_url_encoder: Optional[Callable[[Any], str]] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Append frames using the selected student/eval visual protocol.

    Both protocols consume the same ordered ``frames`` list and the same
    ``<visual_window>`` text block. The only difference is the media carrier:
    timestamped individual images versus one native video block with explicit
    Qwen video metadata.
    """
    protocol = normalize_frame_protocol(frame_protocol)
    if protocol == FRAME_PROTOCOL_TS_IMAGE:
        append_timestamped_image_list(
            content,
            frames,
            fps=fps,
            start_frame_index=start_frame_index,
            total_num_frames=total_num_frames,
            latest_start_frame_index=latest_start_frame_index,
            context_label=context_label,
            timestamp_labels=timestamp_labels,
            image_key=image_key,
            image_url_encoder=image_url_encoder,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return

    if image_key != "image":
        raise ValueError(
            "video_meta protocol is only supported for local frame paths "
            "(image_key='image'), not image_url API payloads"
        )
    append_video_metadata_frame_list(
        content,
        frames,
        fps=fps,
        start_frame_index=start_frame_index,
        total_num_frames=total_num_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

# ---------------------------------------------------------------------------
# Memory Formatting
# ---------------------------------------------------------------------------

_RECENT_THINK_LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.DOTALL)


def _coerce_memory_think(item: Any) -> Dict[str, str]:
    """Normalize a recent-think memory item for tagged rendering."""
    if isinstance(item, str):
        m = _RECENT_THINK_LINE_RE.match(item)
        if m:
            return {"time": m.group(1).strip(), "text": m.group(2).strip()}
        return {"time": "", "text": item.strip()}
    if isinstance(item, dict):
        time_str = item.get(
            "time",
            f'{item.get("chunk", 0) * AGENT_CHUNK_SEC}-'
            f'{item.get("chunk", 0) * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC}',
        )
        return {
            "time": str(time_str),
            "text": str(item.get("text", item.get("obs", ""))).strip(),
        }
    return {"time": "", "text": str(item).strip()}


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

    # Recent thinks. Render as tagged JSON records rather than prose lines so
    # the model treats them as archival memory, not a continuation template.
    recent = memory.get("recent_thinks", memory.get("recent_observations", []))
    for item in recent:
        rec = _coerce_memory_think(item)
        if rec.get("text"):
            rec_json = json.dumps(rec, ensure_ascii=False)
            parts.append(f"<memory_think>{rec_json}</memory_think>")

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


# Eval-side caps. Aligned to the current independent-question distribution:
#   - QUERY_HISTORY_POLICY now selects only live query records. Answered/closed
#     questions are intentionally not rendered in later turns; the model only
#     sees the current active question plus answer history for that same query.
#   - QUERIES_HISTORY_CAP is a defensive bound for unexpected concurrent open
#     queries. Production pass3 enforces one active question at a time.
#   - RECALL_TEXT_MAX_CHARS=1600 ≈ 4 × THINK_TOKENS.max(100 tok × ~4 char)
# These are upper-bound guards; SFT samples normally have a single active query.
# The "32k" eval profile (scripts/eval/eval_profiles.py) loosens further.
QUERY_HISTORY_POLICY = "recent_k"
QUERIES_HISTORY_CAP = 8
RECALL_TEXT_MAX_CHARS = 1600


def answer_format_instruction(
    answer_form: str,
    *,
    answer_style: str = "",
    options: Optional[Sequence[str]] = None,
) -> str:
    """Render the model-visible answer-format instruction for a question.

    This is part of the data/eval contract. Gold labels store structured
    fields (answer_form, options, correct_option, accepted_answers), while the
    prompt must still tell the model which surface form to emit.
    """
    form = str(answer_form or "").strip().lower()
    style = str(answer_style or "").strip().lower()

    if form == "multiple_choice":
        if style == "letter_plus_text":
            return "Answer format: letter plus option text, e.g. A) option text."
        if style == "text_only":
            return "Answer format: answer text only, no option letter."
        # Default and OvO-compatible style.
        return "Answer format: one letter only (A, B, C, or D)."
    if form == "binary":
        return "Answer format: a concise binary answer such as Yes or No."
    if form == "number":
        return "Answer format: a number only, no explanation."
    if form == "short_exact":
        return "Answer format: a concise exact phrase, no explanation."
    if form == "descriptive":
        return "Answer format: a short natural-language answer."
    return ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _normalize_query_history_policy(policy: Optional[str] = None) -> str:
    value = (
        policy
        or os.environ.get("THINKSTREAM_QUERY_HISTORY_POLICY")
        or QUERY_HISTORY_POLICY
    )
    value = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "all_pending": "multi_pending",
        "pending": "multi_pending",
        "current": "single_active",
        "active": "single_active",
        "latest": "single_active",
        "replace": "replace_on_new",
        "last_k": "recent_k",
        "recent": "recent_k",
    }
    value = aliases.get(value, value)
    if value not in {"multi_pending", "recent_k", "single_active", "replace_on_new"}:
        return QUERY_HISTORY_POLICY
    return value


def _query_time_key(q: Dict) -> float:
    value = q.get("ask_time", q.get("time", q.get("response_time", "")))
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return 0.0
    return 0.0


_OPEN_QUERY_STATUSES = {"open", "pending", "active"}
_CLOSED_QUERY_STATUSES = {"answered", "closed", "done", "replaced"}


def _query_expected_answer_count(q: Dict) -> int:
    """Expected non-empty answers for the query lifecycle."""
    answer_chunks = q.get("answer_chunks") or []
    if hasattr(answer_chunks, "tolist"):
        answer_chunks = answer_chunks.tolist()
    per_emit = q.get("per_emit_answers") or []
    if hasattr(per_emit, "tolist"):
        per_emit = per_emit.tolist()
    try:
        n_chunks = len(answer_chunks)
    except TypeError:
        n_chunks = 0
    try:
        n_emit = len(per_emit)
    except TypeError:
        n_emit = 0
    return max(1, n_chunks, n_emit)


def _query_is_open(q: Dict) -> bool:
    """Return whether a query should be rendered as the active question."""
    status = str(q.get("status", "")).strip().lower()
    if status in _OPEN_QUERY_STATUSES:
        return True
    if status in _CLOSED_QUERY_STATUSES:
        return False
    answers = q.get("answers") or []
    if not answers:
        return True
    return len(answers) < _query_expected_answer_count(q)


def _format_query_time_prefix(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not m:
            return ""
        try:
            num = float(m.group(0))
        except ValueError:
            return ""
    if num.is_integer():
        return f"[{int(num)}s]"
    return f"[{num:g}s]"


def _select_queries_for_prompt(
    queries: List[Dict],
    *,
    policy: Optional[str] = None,
    cap: Optional[int] = None,
) -> List[Dict]:
    """Select live query records visible in the prompt."""
    if not queries:
        return []

    mode = _normalize_query_history_policy(policy)
    limit = int(cap if cap is not None else _env_int(
        "THINKSTREAM_QUERIES_HISTORY_CAP",
        QUERIES_HISTORY_CAP,
    ))
    limit = max(1, limit)
    indexed = list(enumerate(queries))
    open_items = [(i, q) for i, q in indexed if _query_is_open(q)]
    if not open_items:
        return []

    if mode in {"single_active", "replace_on_new"}:
        # If several open questions somehow exist, the newest question is the
        # active task. Closed questions are not rendered after their answer.
        latest = max(open_items, key=lambda x: (_query_time_key(x[1]), x[0]))
        return [latest[1]]

    if mode == "recent_k":
        selected = sorted(
            open_items,
            key=lambda x: (_query_time_key(x[1]), x[0]),
        )[-limit:]
        selected.sort(key=lambda x: x[0])
        return [q for _, q in selected]

    # Backward-compatible multi-pending mode, but still no answered history.
    selected = sorted(open_items, key=lambda x: (_query_time_key(x[1]), x[0]))
    return [q for _, q in selected[-limit:]]


def format_queries_block(
    queries: List[Dict],
    *,
    policy: Optional[str] = None,
    cap: Optional[int] = None,
) -> str:
    """Format the active question and its answer history.

    The prompt has two separate query zones:
    - <active_query>: the currently live question, including options and answer
      format derived from question type.
    - <response_history>: non-empty answers already emitted for that same active
      query. Answers from closed/older questions are not shown.

    If no query is open, this returns an empty string so the next timestep after
    a final answer cannot see stale historical Q&A.

    Example output:
      <active_query>
      [20s] Q: What color is the apron?
      [20s] Answer format: a concise exact phrase, no explanation.
      </active_query>
      <response_history>
      </response_history>
    """
    if not queries:
        return ""

    queries = _select_queries_for_prompt(queries, policy=policy, cap=cap)
    if not queries:
        return ""

    # Production pass3 enforces one active question. If a legacy/eval path passes
    # multiple open queries, render the newest one so answer/silent has one target.
    q = max(
        enumerate(queries),
        key=lambda x: (_query_time_key(x[1]), x[0]),
    )[1]
    ask_t = q.get("ask_time", "")
    prefix = _format_query_time_prefix(ask_t)
    question = str(q.get("question", ""))

    active_lines = [f"{prefix} Q: {question}" if prefix else f"Q: {question}"]
    if q.get("answer_form") == "multiple_choice" and q.get("options"):
        opts = " ".join(str(opt) for opt in q.get("options") or [])
        active_lines.append(
            f"{prefix} Options: {opts}" if prefix else f"Options: {opts}"
        )
    instruction = (q.get("answer_instruction") or "").strip()
    if not instruction:
        instruction = answer_format_instruction(
            q.get("answer_form", ""),
            answer_style=q.get("answer_style", ""),
            options=q.get("options") or [],
        )
    if instruction:
        active_lines.append(f"{prefix} {instruction}" if prefix else instruction)

    answers = []
    for ans in q.get("answers", []) or []:
        if isinstance(ans, dict):
            answers.append((ans.get("time", ask_t), str(ans.get("text", ""))))
        else:
            answers.append((q.get("response_time", ask_t), str(ans)))
    answers.sort(key=lambda x: _query_time_key({"time": x[0]}))
    response_lines = []
    for t, text in answers:
        aprefix = _format_query_time_prefix(t)
        response_lines.append(f"{aprefix} A: {text}" if aprefix else f"A: {text}")

    active_block = "<active_query>\n" + "\n".join(active_lines) + "\n</active_query>"
    response_body = "\n".join(response_lines)
    response_block = f"<response_history>\n{response_body}\n</response_history>"
    return active_block + "\n" + response_block


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
    frame_protocol: Optional[str] = None,
    inter_chunk: bool = False,
) -> List[Dict]:
    """Build the user content list for a single-step message.

    Ordering:
    <user_input> → <memory> → <active_query>/<response_history> (visual turns only) →
    <visual_window> + frames → <recalled_frames> + frames → <recall_result>

    Why this order: memory and the active-query state stay before the visual
    window, while old closed Q&A is intentionally omitted. The fresh user event
    is placed before memory so questions and memory-compaction triggers are not
    buried behind long historical text. Visual window changes every chunk, so it
    remains after the stable text zones.

    Pre-extracted frames are rendered by the late-bound frame protocol:
    ``ts_image`` (frame-tag text + image items) or ``video_meta`` (one Qwen
    video block with explicit metadata). All other text state is identical
    across protocols.

    Args:
        memory_text: Pre-formatted memory block from format_memory_block().
        chunk_idx: Current chunk index.
        video_path: Path to video file.
        user_input: Question, bare compress_trigger marker, "Continue...", or
                    empty. Compression rules come from the compression system
                    prompt, not from this field.
        recalled_frames: Optional recalled frame info for recall_response.
        recall_result: Optional recall result for recall_response.
        min_pixels, max_pixels: Resolution limits.
        frame_paths: Optional explicit frame paths. This is the canonical path
                     for pass/SFT/RL/eval. If None, uses video_path with time
                     range as a legacy fallback.
        frame_protocol: "ts_image" or "video_meta"; defaults to
                        THINKSTREAM_FRAME_PROTOCOL or "ts_image".
        inter_chunk: Memory-compaction turn. Queries, recalled frames, and the
                     current visual sliding window are suppressed; compression
                     is a text-memory action between visual timesteps.
    """
    chunk_sec = AGENT_CHUNK_SEC
    user_content = []
    user_input_block = format_user_input_block(
        user_input,
        inter_chunk=inter_chunk,
    ) if user_input else ""
    prepend_user_input = bool(user_input_block) and user_input_should_prepend(
        inter_chunk=inter_chunk,
    )

    if prepend_user_input:
        user_content.append({
            "type": "text",
            "text": user_input_block.lstrip("\n"),
        })

    # ── Memory block ──
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>" if user_content
        else f"<memory>\n{memory_text}\n</memory>",
    })

    # ── Active query + response history for that same query ──
    # Inter-chunk compression is a system memory-pressure event, so omit
    # queries to prevent the model from answering instead of compacting memory.
    if queries and not inter_chunk:
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({
                "type": "text",
                "text": f"\n{queries_text}",
            })

    # ── Visual window + protocol-selected frame carrier ──
    # Compression is between visual timesteps and should not condition on the
    # current frame window. Ordinary streaming / recall-response turns keep it.
    if not inter_chunk:
        window_start = compute_visual_window_start(chunk_idx, VISUAL_WINDOW_CHUNKS)
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
            "text": f"\n<visual_window>{vw_header}</visual_window>",
        })

        if frame_paths:
            append_visual_frames(
                user_content,
                frame_paths,
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / chunk_sec),
                start_frame_index=window_start * FRAMES_PER_CHUNK,
                total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
                latest_start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
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
    if recalled_frames and not inter_chunk:
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
            # Timestamp recalled images at their ORIGINAL video time.
            tr_start, tr_end = recalled_frames["time_range"]
            append_visual_frames(
                user_content,
                recalled_frames["frame_paths"],
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / chunk_sec),
                start_frame_index=int(tr_start * FRAMES_PER_CHUNK),
                total_num_frames=int(tr_end * FRAMES_PER_CHUNK),
                context_label="recalled frame",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
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
    if recall_result and not inter_chunk:
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

    if user_input_block and not prepend_user_input:
        user_content.append({
            "type": "text",
            "text": user_input_block,
        })

    return user_content


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------
# Architecture:
#   answer (terminal)   = <answer>text</answer> or <answer></answer> (silent)
#   tool (recall)       = <tool_call>{"name":"recall","arguments":{...}}</tool_call>
#   compress turn       = compression-only system prompt + optional legacy
#                         <compress_trigger/> marker (boolean signal only; NO
#                         range). The assistant emits a compress tool_call
#                         carrying its OWN derived time_range + summary text.
    # Tools registered via system <tools> block (auto-rendered by chat_template
# when tools=tools is passed to apply_chat_template).

_FRAME_TAG_LINE_RE = re.compile(
    r'\s*<frame\s+ts="[^"]+"\s+role="[^"]+"\s*/>\s*',
    re.IGNORECASE,
)
_FRAME_TAG_INLINE_RE = re.compile(
    r'<frame\s+ts="[^"]+"\s+role="[^"]+"\s*/>',
    re.IGNORECASE,
)


def strip_frame_metadata_tags(text: str) -> str:
    """Remove copied frame metadata tags from model-visible output text."""
    if not text:
        return text
    kept_lines = []
    for line in str(text).splitlines():
        if _FRAME_TAG_LINE_RE.fullmatch(line):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = _FRAME_TAG_INLINE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

_FRAME_CARRIER_TS_PROMPT = (
    "Each turn you receive: frame-tagged visual frames (recent 16s window) + "
    "tagged memory state. Every image is preceded by a structural tag like "
    "<frame ts=\"12.5\" role=\"latest chunk\" />; use these frame tags together "
    "with <visual_window>.current_time to identify the current chunk. Frame "
    "tags are routing metadata only: never copy or paraphrase any <frame .../> "
    "tag, timestamp marker, role marker, or metadata line in your output. "
)

_FRAME_CARRIER_VIDEO_META_PROMPT = (
    "Each turn you receive: a pre-sampled video block (recent 16s window) + "
    "tagged memory state. The video block uses Qwen video_metadata (fps, "
    "frames_indices, total_num_frames) to carry frame timestamps; use those "
    "timestamps together with <visual_window>.current_time to identify the "
    "current chunk. Temporal metadata is routing metadata only: never copy or "
    "paraphrase timestamp markers, frame indices, role markers, or metadata "
    "lines in your output. "
)

SYSTEM_PROMPT_V12_STREAMING = (
    "You are a streaming video agent. You observe 1-second video chunks and "
    "maintain memory. Use this prompt for ordinary streaming-video turns; "
    "memory compaction uses a separate turn-local system prompt.\n\n"
    f"{_FRAME_CARRIER_TS_PROMPT}"
    "The user payload may also include <user_input> for a new question/event, "
    "<active_query> for the one currently live question with its answer-format "
    "instruction, <response_history> for answers already emitted for that same "
    "active_query only, <recalled_frames> for historical visual evidence returned "
    "by recall, and <recall_result> for historical text evidence returned by "
    "recall. Closed questions and their old answers are not shown.\n\n"
    "Ordinary streaming turns have exactly three terminal forms:\n"
    "1. Recall tool: a recall <tool_call>. This is a tool request, not an "
    "answer. After a <recall_result> is returned for a question, do not call "
    "recall again for that same question.\n"
    "2. Answer response: a non-empty <answer>response text</answer>. If "
    "<active_query> is present, the response text belongs to that active_query "
    "and must follow its answer-format instruction.\n"
    "3. Silent answer: an empty <answer></answer>. It carries no response text "
    "and does not add anything to <response_history>.\n\n"
    "Do not emit the compress tool when this ordinary streaming prompt is the "
    "active turn policy. Compression uses a separate system prompt.\n\n"
    "Required output grammar for ordinary streaming turns:\n"
    "- Every assistant message must be exactly one <think> block followed by "
    "exactly one terminal block. Do not write any text outside these tags.\n"
    "- Think block format:\n"
    "  <think>40-80 tokens describing only observable facts in the current "
    "chunk and the selected terminal form: recall tool vs answer response vs "
    "silent answer</think>\n"
    "- Recall tool format:\n"
    "  <tool_call>{\"name\":\"recall\",\"arguments\":{\"query\":\"3-5 keywords\",\"time_range\":\"start-end\"}}</tool_call>\n"
    "  The JSON must use double quotes. query must contain search keywords, "
    "not the answer value. time_range is seconds as a string like \"20-60\".\n"
    "- Answer response format:\n"
    "  <answer>response text</answer>\n"
    "  If <active_query> includes an 'Answer format:' line, response text must "
    "follow it exactly. For MC letter-only questions, output only A, B, "
    "C, or D with no explanation.\n"
    "- Silent answer format:\n"
    "  <answer></answer>\n"
    "  The silent answer must be empty; do not put words, spaces, or rationale "
    "inside it.\n"
    "- Compression is not an allowed terminal block under this prompt. Never "
    "emit <tool_call>{\"name\":\"compress\",...}</tool_call> here.\n\n"
    "Think rules: describe ONLY observable visual facts in the current chunk "
    "and the minimal action decision. Evidence priority: (1) current "
    "frame-tagged images determine the current think; (2) recalled frames and "
    "recall_result are historical evidence only; (3) tagged memory records "
    "are history and entity naming only; (4) if current frames conflict with "
    "memory, ignore memory for the current visual description. Do not use "
    "memory as evidence that a past object/action is still visible. Use "
    "continuation phrases such as 'continues', 'remains', or 'unchanged' only "
    "when the current frames visibly show the same object/action; otherwise "
    "name the new object/action directly. No meta-reasoning, no sound/smell/"
    "emotion, no speculation."
)

SYSTEM_PROMPT_V12_COMPRESS = (
    "You are the memory-compaction controller for a streaming video agent. "
    "Directly compress memory now. This is not an ordinary QA, recall, answer, "
    "or silent turn.\n\n"
    "The user payload may include <user_input><compress_trigger/></user_input> "
    "as a legacy event marker. The marker is not an instruction source; this "
    "system prompt is the instruction. Compression turns include text memory "
    "only; active queries, recalled frames, and the current visual window are "
    "not part of this action.\n\n"
    "Required compression behavior:\n"
    "- Do not answer any user question.\n"
    "- Do not emit a silent answer.\n"
    "- Do not call recall.\n"
    "- Emit exactly one compress tool_call after a short compression think.\n"
    "- Choose an older contiguous time range from <memory> and summarize it so "
    "the summary can replace those text memory records.\n"
    "- The compression target must come from <memory>, not from fresh visual "
    "frames or active QA context.\n"
    "- Retain entity names, visual attributes, OCR text, spatial relations, "
    "and state changes. Do not invent facts or drop details needed for future "
    "questions.\n\n"
    "Required output grammar for compression turns:\n"
    "- Every assistant message must be exactly one <think> block followed by "
    "exactly one compress <tool_call>. Do not write any text outside these "
    "tags.\n"
    "- Think block format:\n"
    "  <think>20-60 tokens stating that memory is over budget and which older "
    "contiguous time range should be compressed</think>\n"
    "- Compress tool format:\n"
    "  <tool_call>{\"name\":\"compress\",\"arguments\":{\"time_range\":[start_sec,end_sec],\"text\":\"summary text\"}}</tool_call>\n"
    "  The JSON must use double quotes. time_range must be a two-integer array "
    "[start_sec,end_sec] with seconds from the selected older <memory> range. "
    "text must be the replacement summary for that range.\n"
    "- Do not emit <answer>...</answer>, <answer></answer>, or a recall "
    "tool_call on compression turns."
)

# Backward-compat: old imports refer to the ordinary streaming prompt.
SYSTEM_PROMPT_V12 = SYSTEM_PROMPT_V12_STREAMING

SYSTEM_PROMPT_V12_VIDEO_META = (
    SYSTEM_PROMPT_V12_STREAMING
    .replace(_FRAME_CARRIER_TS_PROMPT, _FRAME_CARRIER_VIDEO_META_PROMPT)
    .replace(
        "Evidence priority: (1) current frame-tagged images determine the current think; ",
        "Evidence priority: (1) current visual frames determine the current think; ",
    )
)

SYSTEM_PROMPT_V12_COMPRESS_VIDEO_META = (
    SYSTEM_PROMPT_V12_COMPRESS
    .replace(_FRAME_CARRIER_TS_PROMPT, _FRAME_CARRIER_VIDEO_META_PROMPT)
)


def normalize_system_prompt_kind(
    prompt_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
) -> str:
    """Return ``streaming`` or ``compress`` for the current turn."""
    if inter_chunk:
        return "compress"
    value = str(prompt_kind or "streaming").strip().lower().replace("-", "_")
    aliases = {
        "normal": "streaming",
        "ordinary": "streaming",
        "visual": "streaming",
        "video": "streaming",
        "memory_compaction": "compress",
        "compaction": "compress",
        "compression": "compress",
        "compress_prompt": "compress",
        "system_prompt_compress": "compress",
        "inter_chunk": "compress",
    }
    value = aliases.get(value, value)
    if value not in {"streaming", "compress"}:
        return "streaming"
    return value


def system_prompt_for_frame_protocol(
    frame_protocol: Optional[str] = None,
    *,
    prompt_kind: Optional[str] = None,
    inter_chunk: bool = False,
) -> str:
    """Return the protocol-aligned system prompt.

    Prompt semantics stay aligned across AB variants. Ordinary streaming turns
    use the prompt that allows recall / answer response / silent answer.
    Memory-compaction turns use the compression-only prompt. Within either
    prompt kind, only the sentence describing the visual carrier differs,
    because one protocol exposes text frame tags and the other relies on Qwen
    video metadata.
    """
    protocol = normalize_frame_protocol(frame_protocol)
    kind = normalize_system_prompt_kind(prompt_kind, inter_chunk=inter_chunk)
    if protocol == FRAME_PROTOCOL_VIDEO_META:
        if kind == "compress":
            return SYSTEM_PROMPT_V12_COMPRESS_VIDEO_META
        return SYSTEM_PROMPT_V12_VIDEO_META
    if kind == "compress":
        return SYSTEM_PROMPT_V12_COMPRESS
    return SYSTEM_PROMPT_V12_STREAMING


# Tool JSON schemas — passed as `tools=...` to apply_chat_template.
# Format follows OpenAI function-calling spec, recognized by Qwen2.5/3-VL's
# chat template which auto-renders <tools>...</tools> in the system prompt.
#
# Keep the per-tool dictionaries separate so callers can expose the action
# space that is valid for the current turn instead of always showing both tools.
RECALL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Ordinary streaming-turn tool. Search past video observations "
            "by keywords and time range. Returns matched historical "
            "thinks. Use when the answer is not in any visible source but "
            "was observed earlier, or once for a pending future question "
            "to verify that the answer has not appeared in the past yet. "
            "This is not a final answer."
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
}

COMPRESS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compress",
        "description": (
            "Compression-turn tool. Use only under the compression system "
            "prompt or its legacy <compress_trigger/> event marker. Do "
            "not answer questions, emit a silent answer, or call recall on "
            "that turn. Decide which older contiguous range from <memory> "
            "to compress and output a concise summary retaining all "
            "entities, attributes, OCR, and state changes."
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
}

# Back-compat name for old call sites. New code should use tools_for_turn().
TOOLS_SCHEMA = [RECALL_TOOL_SCHEMA, COMPRESS_TOOL_SCHEMA]
STREAMING_TOOLS_SCHEMA = [RECALL_TOOL_SCHEMA]
COMPRESS_TOOLS_SCHEMA = [COMPRESS_TOOL_SCHEMA]


def normalize_tool_turn_kind(
    turn_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
    recall_response: bool = False,
) -> str:
    """Return streaming|compress|recall_response for tool/action gating."""
    if recall_response:
        return "recall_response"
    if inter_chunk:
        return "compress"
    value = str(turn_kind or "streaming").strip().lower().replace("-", "_")
    aliases = {
        "ordinary": "streaming",
        "normal": "streaming",
        "visual": "streaming",
        "video": "streaming",
        "memory_compaction": "compress",
        "compaction": "compress",
        "compression": "compress",
        "inter_chunk": "compress",
        "recall_result": "recall_response",
        "recall_answer": "recall_response",
        "tool_result": "recall_response",
        "answer_after_recall": "recall_response",
        "none": "recall_response",
        "no_tools": "recall_response",
    }
    value = aliases.get(value, value)
    if value not in {"streaming", "compress", "recall_response"}:
        return "streaming"
    return value


def tools_for_turn(
    turn_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
    recall_response: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Return the Qwen tool schema valid for one generation turn.

    - streaming turns expose recall only;
    - compression turns expose compress only;
    - recall-result answer turns expose no tools.
    """
    kind = normalize_tool_turn_kind(
        turn_kind,
        inter_chunk=inter_chunk,
        recall_response=recall_response,
    )
    if kind == "streaming":
        return STREAMING_TOOLS_SCHEMA
    if kind == "compress":
        return COMPRESS_TOOLS_SCHEMA
    return None


def allowed_actions_for_turn(
    turn_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
    recall_response: bool = False,
) -> set[str]:
    """Return canonical parsed actions allowed for this generation turn."""
    kind = normalize_tool_turn_kind(
        turn_kind,
        inter_chunk=inter_chunk,
        recall_response=recall_response,
    )
    if kind == "streaming":
        return {"silent", "response", "recall", "answer"}
    if kind == "compress":
        return {"compress"}
    return {"silent", "response", "answer"}


def is_action_allowed_for_turn(
    action: str,
    turn_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
    recall_response: bool = False,
) -> bool:
    """Check one parsed action/kind against the turn-local action space."""
    return str(action or "") in allowed_actions_for_turn(
        turn_kind,
        inter_chunk=inter_chunk,
        recall_response=recall_response,
    )


def action_space_error_for_turn(
    action: str,
    turn_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
    recall_response: bool = False,
) -> str:
    """Return an error string when an action is illegal for the turn."""
    kind = normalize_tool_turn_kind(
        turn_kind,
        inter_chunk=inter_chunk,
        recall_response=recall_response,
    )
    if is_action_allowed_for_turn(action, kind):
        return ""
    allowed = ",".join(sorted(allowed_actions_for_turn(kind)))
    return f"action_not_allowed:{action or 'unknown'}@{kind};allowed={allowed}"


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
        think: think content (40-80 tokens recommended).
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


_RECALL_TOOL_TIME_RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$"
)


def _validate_recall_tool_args(args: Any) -> Optional[str]:
    if not isinstance(args, dict):
        return "recall arguments must be an object"
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return "recall query must be a non-empty string"
    time_range = args.get("time_range")
    if not isinstance(time_range, str):
        return "recall time_range must be a start-end string"
    m = _RECALL_TOOL_TIME_RANGE_RE.fullmatch(time_range)
    if not m:
        return "recall time_range must match start-end"
    if float(m.group(2)) <= float(m.group(1)):
        return "recall time_range end must be greater than start"
    return None


def _validate_compress_tool_args(args: Any) -> Optional[str]:
    if not isinstance(args, dict):
        return "compress arguments must be an object"
    time_range = args.get("time_range")
    if (
        not isinstance(time_range, list)
        or len(time_range) != 2
        or not all(isinstance(v, (int, float)) for v in time_range)
    ):
        return "compress time_range must be a two-number array"
    if float(time_range[1]) <= float(time_range[0]):
        return "compress time_range end must be greater than start"
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return "compress text must be a non-empty string"
    return None


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
    output_text = strip_frame_metadata_tags(output_text or "")
    result = {
        "raw": output_text,
        "think": "",
        "kind": "unknown",
        "answer_text": None,
        "tool_call": None,
        "format_error": None,
    }

    think_matches = list(re.finditer(r'<think>(.*?)</think>', output_text, re.DOTALL))
    if len(think_matches) == 1:
        result["think"] = think_matches[0].group(1).strip()
        if not result["think"]:
            result["format_error"] = "empty <think> block"
    else:
        result["format_error"] = (
            "missing <think> block" if not think_matches
            else "multiple <think> blocks"
        )

    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', output_text, re.DOTALL))
    tool_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL))
    answer_match = (
        answer_matches[0]
        if answer_matches else None
    )
    tool_match = (
        tool_matches[0]
        if tool_matches else None
    )

    # Both present → format error (must be one or the other, not both)
    if answer_match and tool_match:
        result["format_error"] = "both <answer> and <tool_call> present"
        return result
    if len(answer_matches) > 1:
        result["format_error"] = "multiple <answer> blocks"
        return result
    if len(tool_matches) > 1:
        result["format_error"] = "multiple <tool_call> blocks"
        return result

    if (
        result["format_error"] is None
        and len(think_matches) == 1
        and ((answer_match is None) ^ (tool_match is None))
    ):
        terminal_match = answer_match or tool_match
        assert terminal_match is not None
        think_match = think_matches[0]
        if think_match.start() > terminal_match.start():
            result["format_error"] = "terminal block appears before <think>"
        else:
            outside = (
                output_text[:think_match.start()]
                + output_text[think_match.end():terminal_match.start()]
                + output_text[terminal_match.end():]
            )
            if outside.strip():
                result["format_error"] = (
                    "text outside required <think> plus terminal blocks"
                )

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

        if not isinstance(tool_obj, dict):
            result["format_error"] = "tool_call JSON must be an object"
            return result

        result["tool_call"] = tool_obj
        name = tool_obj.get("name", "")
        args = tool_obj.get("arguments")
        if name == "recall":
            schema_error = _validate_recall_tool_args(args)
            if schema_error:
                result["format_error"] = schema_error
                return result
            result["kind"] = "recall"
        elif name == "compress":
            schema_error = _validate_compress_tool_args(args)
            if schema_error:
                result["format_error"] = schema_error
                return result
            result["kind"] = "compress"
        else:
            result["format_error"] = f"unknown tool name: {name!r}"
            return result
        return result

    result["format_error"] = "neither <answer> nor <tool_call> emitted"
    return result


def has_compress_trigger(user_text: str) -> bool:
    """Check if a user message contains a system-injected <compress_trigger/>.

    Used by training/eval to verify trigger→tool_call binding, and by the
    rollout controller to know whether the assistant must emit compress.
    """
    return _contains_compress_trigger(user_text)


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
