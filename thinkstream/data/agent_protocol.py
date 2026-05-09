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

RENDER_LAYOUT_STANDARD = "standard"
RENDER_LAYOUT_TIMELINE_VIDEO = "timeline_video"
RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD = "timeline_video_imagepad"
RENDER_LAYOUT_ENV = "THINKSTREAM_RENDER_LAYOUT"
VALID_RENDER_LAYOUTS = {
    RENDER_LAYOUT_STANDARD,
    RENDER_LAYOUT_TIMELINE_VIDEO,
    RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
}


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


def normalize_render_layout(render_layout: Optional[str] = None) -> str:
    """Return the active prompt layout for SFT/RL/eval rendering."""
    value = (
        render_layout
        or os.environ.get(RENDER_LAYOUT_ENV)
        or RENDER_LAYOUT_STANDARD
    )
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "block": RENDER_LAYOUT_STANDARD,
        "legacy": RENDER_LAYOUT_STANDARD,
        "timeline": RENDER_LAYOUT_TIMELINE_VIDEO,
        "interleaved": RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
        "timeline_imagepad": RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
        "video_imagepad": RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
    }
    value = aliases.get(value, value)
    if value not in VALID_RENDER_LAYOUTS:
        raise ValueError(
            f"Unsupported render layout {render_layout!r}; expected one of "
            f"{sorted(VALID_RENDER_LAYOUTS)}"
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


def append_imagepad_frame_list(
    content: List[Dict],
    frames: Sequence[Any],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Append pre-extracted frames as independent image-pad items.

    Pass5 may store these rows as ``type=video`` for dataset routing, but
    runtime processors and vLLM consume the carrier as ordinary image items.
    The surrounding timeline tags carry the temporal grouping.
    """
    for frame in frames or []:
        item: Dict[str, Any] = {"type": "image", "image": frame}
        if min_pixels is not None:
            item["min_pixels"] = min_pixels
        if max_pixels is not None:
            item["max_pixels"] = max_pixels
        content.append(item)


def _group_frames_by_chunk(
    frame_paths: Optional[Sequence[Any]],
    *,
    window_start: int,
) -> Dict[int, List[Any]]:
    grouped: Dict[int, List[Any]] = {}
    for offset, path in enumerate(frame_paths or []):
        chunk = int(window_start) + int(offset // FRAMES_PER_CHUNK)
        grouped.setdefault(chunk, []).append(path)
    return grouped


def _coerce_timeline_think(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("observation") or item.get("obs") or "").strip()
        if not text:
            return None
        try:
            chunk = int(item.get("chunk"))
        except (TypeError, ValueError):
            chunk = None
        time_text = str(item.get("time") or "").strip()
        if chunk is None and time_text:
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", time_text)
            if m:
                chunk = int(float(m.group(1)) // float(AGENT_CHUNK_SEC))
        if chunk is None:
            return None
        return {"chunk": chunk, "time": time_text, "text": text}
    raw = str(item or "").strip()
    if not raw:
        return None
    m = re.match(r"^\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)$", raw, re.DOTALL)
    if m:
        start = float(m.group(1))
        end = float(m.group(2))
        return {
            "chunk": int(start // float(AGENT_CHUNK_SEC)),
            "time": (
                f"{int(start) if start.is_integer() else start}-"
                f"{int(end) if end.is_integer() else end}"
            ),
            "text": m.group(3).strip(),
        }
    return {"chunk": 0, "time": "", "text": raw}


def _segment_chunks(seg: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for raw in seg.get("source_chunks") or seg.get("chunks") or []:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    if out:
        return sorted(set(out))
    tr = seg.get("time_range") or []
    if isinstance(tr, list) and len(tr) == 2:
        try:
            start = int(float(tr[0]) // float(AGENT_CHUNK_SEC))
            end = int((float(tr[1]) - 1e-9) // float(AGENT_CHUNK_SEC))
            return list(range(max(0, start), max(0, end) + 1))
        except (TypeError, ValueError):
            return []
    return []


def _format_timeline_summary(seg: Dict[str, Any]) -> str:
    chunks = _segment_chunks(seg)
    if chunks:
        start = min(chunks) * AGENT_CHUNK_SEC
        end = (max(chunks) + 1) * AGENT_CHUNK_SEC
        time_range = f"{int(start)}-{int(end)}"
    else:
        tr = seg.get("time_range") or ["?", "?"]
        time_range = f"{tr[0]}-{tr[1]}"
    text = str(seg.get("text") or "").strip()
    return "\n".join([
        f"<SUMMARY time_range={json.dumps(time_range, ensure_ascii=False)}>",
        text,
        "</SUMMARY>",
    ])


def _format_timeline_memory_think(rec: Dict[str, Any]) -> str:
    chunk = int(rec.get("chunk", 0) or 0)
    time_text = str(rec.get("time") or "").strip()
    if not time_text:
        time_text = f"{int(chunk * AGENT_CHUNK_SEC)}"
    elif "-" in time_text:
        time_text = time_text.split("-", 1)[0].strip()
    text = str(rec.get("text") or "").strip()
    return f"<MEMORY_THINK time={json.dumps(time_text)}>{text}</MEMORY_THINK>"


def append_timeline_visual_chunk(
    content: List[Dict],
    *,
    frames: Sequence[Any],
    chunk: int,
    frame_protocol: Optional[str],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    imagepad: bool,
) -> None:
    start = int(chunk * AGENT_CHUNK_SEC)
    content.append({"type": "text", "text": f"\n<VISUAL_CHUNK time=\"{start}\">"})
    if imagepad:
        append_imagepad_frame_list(
            content,
            frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        append_visual_frames(
            content,
            frames,
            frame_protocol=frame_protocol,
            fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
            start_frame_index=chunk * FRAMES_PER_CHUNK,
            total_num_frames=(chunk + 1) * FRAMES_PER_CHUNK,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    content.append({"type": "text", "text": "</VISUAL_CHUNK>"})


def append_timeline_recalled_chunks(
    content: List[Dict],
    *,
    recalled_frames: Dict[str, Any],
    frame_protocol: Optional[str],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    imagepad: bool,
) -> None:
    frame_paths = recalled_frames.get("frame_paths") or []
    if not frame_paths:
        return
    tr0, tr1 = recalled_frames.get("time_range", [0, 0])
    start_chunk = int(float(tr0) // float(AGENT_CHUNK_SEC))
    grouped = _group_frames_by_chunk(frame_paths, window_start=start_chunk)
    for chunk in sorted(grouped):
        start = int(chunk * AGENT_CHUNK_SEC)
        content.append({
            "type": "text",
            "text": f"\n<RECALLED_CHUNK time=\"{start}\">",
        })
        if imagepad:
            append_imagepad_frame_list(
                content,
                grouped[chunk],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            append_visual_frames(
                content,
                grouped[chunk],
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
                start_frame_index=chunk * FRAMES_PER_CHUNK,
                total_num_frames=max(
                    int(float(tr1) * FRAMES_PER_CHUNK),
                    (chunk + 1) * FRAMES_PER_CHUNK,
                ),
                context_label="recalled frame",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        content.append({"type": "text", "text": "</RECALLED_CHUNK>"})

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


def build_recall_result_user_content(
    recalled_frames: Optional[Dict] = None,
    recall_result: Optional[Dict] = None,
    *,
    frame_protocol: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    render_layout: Optional[str] = None,
) -> List[Dict]:
    """Build the second user payload after a recall tool call."""
    layout = normalize_render_layout(render_layout)
    imagepad = layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD
    user_content: List[Dict] = []
    if recalled_frames:
        rf_header = json.dumps({
            "time_range": recalled_frames["time_range"],
            "source": recalled_frames.get("source", "historical_frames"),
            "n_frames": recalled_frames.get("n_frames", 4),
        })
        user_content.append({
            "type": "text",
            "text": f"<recalled_frames>{rf_header}</recalled_frames>",
        })
        if recalled_frames.get("frame_paths"):
            if layout in {
                RENDER_LAYOUT_TIMELINE_VIDEO,
                RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
            }:
                append_timeline_recalled_chunks(
                    user_content,
                    recalled_frames=recalled_frames,
                    frame_protocol=frame_protocol,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    imagepad=imagepad,
                )
            else:
                tr_start, tr_end = recalled_frames["time_range"]
                append_visual_frames(
                    user_content,
                    recalled_frames["frame_paths"],
                    frame_protocol=frame_protocol,
                    fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
                    start_frame_index=int(float(tr_start)) * FRAMES_PER_CHUNK,
                    total_num_frames=int(float(tr_end)) * FRAMES_PER_CHUNK,
                    context_label="recalled frame",
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
    if recall_result:
        rr_text = recall_result.get("text_content",
                                    recall_result.get("text", "")) or ""
        if len(rr_text) > RECALL_TEXT_MAX_CHARS:
            rr_text = rr_text[:RECALL_TEXT_MAX_CHARS] + "..."
        rr_json = json.dumps({
            "source": recall_result.get("source", ""),
            "time": recall_result.get("time", ""),
            "text": rr_text,
        }, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"<recall_result>{rr_json}</recall_result>",
        })
    return user_content


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
    return max(1, len(query_expected_answer_chunks(q)))


def query_expected_answer_chunks(q: Dict) -> List[int]:
    """Return canonical expected answer chunks for a live query."""
    answer_chunks = q.get("answer_chunks") or []
    if hasattr(answer_chunks, "tolist"):
        answer_chunks = answer_chunks.tolist()
    out: List[int] = []
    if isinstance(answer_chunks, (list, tuple)):
        for x in answer_chunks:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
    if out:
        return sorted(set(out))

    per_emit = q.get("per_emit_answers") or []
    if hasattr(per_emit, "tolist"):
        per_emit = per_emit.tolist()
    if isinstance(per_emit, (list, tuple)):
        for item in per_emit:
            if not isinstance(item, dict):
                continue
            try:
                out.append(int(item.get("chunk")))
            except (TypeError, ValueError):
                continue
    return sorted(set(out))


def query_completed_answer_chunks(q: Dict) -> List[int]:
    """Expected chunks already satisfied by on-time or late answers."""
    expected = set(query_expected_answer_chunks(q))
    if not expected:
        return []
    done: set[int] = set()
    for ans in q.get("answers", []) or []:
        if not isinstance(ans, dict):
            continue
        if ans.get("counts_for_completion") is False:
            continue
        try:
            expected_chunk = int(ans.get("expected_chunk"))
        except (TypeError, ValueError):
            continue
        if expected_chunk in expected:
            done.add(expected_chunk)
    return sorted(done)


def query_completed_answer_count(q: Dict) -> int:
    expected = query_expected_answer_chunks(q)
    if expected:
        return len(query_completed_answer_chunks(q))
    return len([a for a in (q.get("answers") or []) if str(
        a.get("text", "") if isinstance(a, dict) else a
    ).strip()])


def query_is_complete(q: Dict) -> bool:
    return query_completed_answer_count(q) >= _query_expected_answer_count(q)


def classify_query_answer_timing(q: Dict, response_chunk: int) -> Dict[str, Any]:
    """Classify a response against this query's remaining expected emits.

    Early responses are recorded in response_history for diagnostics, but they
    do not count toward completion. Exact expected chunks count first, even if
    an earlier emit was missed, so a multi-emit query cannot consume a future
    answer slot by answering too early.
    """
    expected = query_expected_answer_chunks(q)
    if not expected:
        return {
            "timing": "unknown",
            "expected_chunk": None,
            "counts_for_completion": True,
            "lead_chunks": None,
            "delay_chunks": None,
        }

    completed = set(query_completed_answer_chunks(q))
    unmatched = [c for c in expected if c not in completed]
    if not unmatched:
        return {
            "timing": "over_emit",
            "expected_chunk": None,
            "counts_for_completion": False,
            "lead_chunks": None,
            "delay_chunks": None,
        }

    if response_chunk in unmatched:
        return {
            "timing": "on_time",
            "expected_chunk": response_chunk,
            "counts_for_completion": True,
            "lead_chunks": 0,
            "delay_chunks": 0,
        }

    first_unmatched = min(unmatched)
    if response_chunk < first_unmatched:
        return {
            "timing": "early",
            "expected_chunk": first_unmatched,
            "counts_for_completion": False,
            "lead_chunks": first_unmatched - response_chunk,
            "delay_chunks": None,
        }

    late_candidates = [c for c in unmatched if c <= response_chunk]
    expected_chunk = min(late_candidates) if late_candidates else first_unmatched
    return {
        "timing": "late",
        "expected_chunk": expected_chunk,
        "counts_for_completion": True,
        "lead_chunks": None,
        "delay_chunks": response_chunk - expected_chunk,
    }


def append_query_answer_with_timing(
    q: Dict,
    answer: str,
    response_time: float,
    *,
    chunk_sec: int = AGENT_CHUNK_SEC,
) -> Dict[str, Any]:
    """Append an answer record and return its timing metadata."""
    try:
        response_chunk = int(round(float(response_time) / max(float(chunk_sec), 1.0)))
    except (TypeError, ValueError):
        response_chunk = -1
    timing = classify_query_answer_timing(q, response_chunk)
    record = {
        "text": answer,
        "time": response_time,
        "chunk": response_chunk,
        "timing": timing.get("timing"),
        "expected_chunk": timing.get("expected_chunk"),
        "counts_for_completion": bool(timing.get("counts_for_completion")),
    }
    if timing.get("lead_chunks") is not None:
        record["lead_chunks"] = timing.get("lead_chunks")
    if timing.get("delay_chunks") is not None:
        record["delay_chunks"] = timing.get("delay_chunks")
    q.setdefault("answers", []).append(record)
    return timing


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
    return query_completed_answer_count(q) < _query_expected_answer_count(q)


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
            if ans.get("counts_for_completion") is False:
                continue
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


def _build_timeline_user_content(
    memory_snapshot: Dict[str, Any],
    *,
    memory_text: str,
    chunk_idx: int,
    user_input: str,
    queries: Optional[List[Dict]],
    recalled_frames: Optional[Dict],
    recall_result: Optional[Dict],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    frame_paths: Optional[List[str]],
    frame_protocol: Optional[str],
    inter_chunk: bool,
    render_layout: str,
) -> List[Dict]:
    content: List[Dict] = []
    user_input_block = format_user_input_block(
        user_input,
        inter_chunk=inter_chunk,
    ) if user_input else ""
    prepend_user_input = bool(user_input_block) and user_input_should_prepend(
        inter_chunk=inter_chunk,
    )
    if prepend_user_input:
        content.append({"type": "text", "text": user_input_block.lstrip("\n")})

    compressed = list(
        memory_snapshot.get(
            "compressed_segments",
            memory_snapshot.get("compressed", []),
        ) or []
    )
    recent = [
        rec for rec in (
            _coerce_timeline_think(item)
            for item in (
                memory_snapshot.get(
                    "recent_thinks",
                    memory_snapshot.get("recent_observations", []),
                ) or []
            )
        )
        if rec is not None
    ]
    recent_by_chunk: Dict[int, List[Dict[str, Any]]] = {}
    for rec in recent:
        recent_by_chunk.setdefault(int(rec["chunk"]), []).append(rec)

    window_start = compute_visual_window_start(chunk_idx, VISUAL_WINDOW_CHUNKS)
    visual_by_chunk: Dict[int, List[Any]] = {}
    if frame_paths and not inter_chunk:
        visual_by_chunk = _group_frames_by_chunk(
            frame_paths,
            window_start=window_start,
        )

    summary_by_chunk: Dict[int, List[str]] = {}
    for seg in compressed:
        if not isinstance(seg, dict):
            continue
        chunks = _segment_chunks(seg)
        start_chunk = min(chunks) if chunks else 0
        summary_by_chunk.setdefault(start_chunk, []).append(
            _format_timeline_summary(seg)
        )

    timeline_chunks = sorted(
        set(summary_by_chunk) | set(recent_by_chunk) | set(visual_by_chunk)
    )
    in_memory = False

    def open_memory() -> None:
        nonlocal in_memory
        if not in_memory:
            content.append({"type": "text", "text": "\n<memory>" if content else "<memory>"})
            in_memory = True

    def close_memory() -> None:
        nonlocal in_memory
        if in_memory:
            content.append({"type": "text", "text": "\n</memory>"})
            in_memory = False

    if not timeline_chunks and memory_text.strip():
        open_memory()
        content.append({"type": "text", "text": f"\n{memory_text.strip()}"})

    imagepad = render_layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD
    for chunk in timeline_chunks:
        summaries = summary_by_chunk.get(chunk, [])
        if summaries:
            open_memory()
            for capsule in summaries:
                content.append({"type": "text", "text": f"\n{capsule}"})
        if chunk in visual_by_chunk:
            close_memory()
            append_timeline_visual_chunk(
                content,
                frames=visual_by_chunk[chunk],
                chunk=chunk,
                frame_protocol=frame_protocol,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                imagepad=imagepad,
            )
            if chunk < chunk_idx:
                recs = recent_by_chunk.get(chunk, [])
                if recs:
                    open_memory()
                    for rec in recs:
                        content.append({
                            "type": "text",
                            "text": f"\n{_format_timeline_memory_think(rec)}",
                        })
        else:
            recs = recent_by_chunk.get(chunk, [])
            if recs:
                open_memory()
                for rec in recs:
                    content.append({
                        "type": "text",
                        "text": f"\n{_format_timeline_memory_think(rec)}",
                    })
    close_memory()

    if queries and not inter_chunk:
        queries_text = format_queries_block(queries)
        if queries_text:
            content.append({"type": "text", "text": f"\n{queries_text}"})

    if (recalled_frames or recall_result) and not inter_chunk:
        recall_items = build_recall_result_user_content(
            recalled_frames,
            recall_result,
            frame_protocol=frame_protocol,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            render_layout=render_layout,
        )
        for i, item in enumerate(recall_items):
            if i == 0 and content and item.get("type") == "text":
                item = dict(item)
                item["text"] = "\n" + str(item.get("text", ""))
            content.append(item)

    if user_input_block and not prepend_user_input:
        content.append({"type": "text", "text": user_input_block})

    return content


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
    memory_snapshot: Optional[Dict[str, Any]] = None,
    render_layout: Optional[str] = None,
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
    layout = normalize_render_layout(render_layout)
    if (
        layout in {
            RENDER_LAYOUT_TIMELINE_VIDEO,
            RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
        }
        and (frame_paths or inter_chunk)
        and memory_snapshot is not None
    ):
        return _build_timeline_user_content(
            memory_snapshot,
            memory_text=memory_text,
            chunk_idx=chunk_idx,
            user_input=user_input,
            queries=queries,
            recalled_frames=recalled_frames,
            recall_result=recall_result,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            frame_paths=frame_paths,
            frame_protocol=frame_protocol,
            inter_chunk=inter_chunk,
            render_layout=layout,
        )

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


_CHAT_TEMPLATE_BOUNDARY_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
)


def strip_chat_template_boundary_tokens(text: str) -> str:
    """Remove Qwen chat-template boundary tokens from decoded assistant text.

    Teacher-forced eval decodes the predicted assistant span including the
    supervised ``<|im_end|>`` token. Runtime generation can also surface it in
    raw text depending on backend settings. The protocol parser should judge
    the assistant payload, not fail because of a chat-template boundary marker
    adjacent to an otherwise valid ``<think>`` + terminal block.
    """
    text = strip_frame_metadata_tags(text or "").strip()
    changed = True
    while changed and text:
        changed = False
        for marker in _CHAT_TEMPLATE_BOUNDARY_TOKENS:
            if text.startswith(marker):
                text = text[len(marker):].strip()
                changed = True
            if text.endswith(marker):
                text = text[: -len(marker)].strip()
                changed = True
    return text


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

_FRAME_CARRIER_TIMELINE_IMAGEPAD_PROMPT = (
    "Each turn you receive: a time-ordered timeline with tagged memory "
    "capsules and time-marked visual chunks. Each visual chunk contains "
    "the pre-sampled image-pad frames for one second. Summary capsules "
    "are historical memory at their covered time range; recalled evidence "
    "is newly retrieved for the current decision but cites older time ranges. "
    "Use the last visual chunk as the primary source for current observation. "
    "Single-step visual and memory tags use time points, while multi-step "
    "summaries and recalled evidence use time ranges. Timeline tags never "
    "expose frame or chunk ids. Temporal metadata is routing metadata only: "
    "never copy or paraphrase timestamp markers or metadata lines in your "
    "output. "
)

_FRAME_CARRIER_TIMELINE_VIDEO_PROMPT = (
    "Each turn you receive: a time-ordered timeline with tagged memory "
    "capsules and one or more pre-sampled video blocks. Each video block uses "
    "Qwen video_metadata (fps, frames_indices, total_num_frames) to carry "
    "absolute frame timestamps. Summary capsules are historical memory at "
    "their covered time range; recalled evidence is newly retrieved for the "
    "current decision but cites older time ranges. Use the last visual chunk "
    "as the primary source for current observation. Single-step visual and "
    "memory tags use time points, while multi-step summaries and recalled "
    "evidence use time ranges. Timeline tags never expose frame or chunk ids. "
    "Temporal metadata is routing metadata only: never copy or paraphrase "
    "timestamp markers or metadata lines in your output. "
)

SYSTEM_PROMPT_V12_STREAMING = (
    "[STREAMING_QA / RECALL-ELIGIBLE TURN]\n"
    "You are a streaming video agent. You observe 1-second video chunks and "
    "maintain memory. This is an ordinary streaming-video turn: decide whether "
    "to answer the active query, stay silent, or call recall for historical "
    "evidence. Memory compaction is a separate memory-maintenance mode with a "
    "different system prompt and action space.\n\n"
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
    "Recall decision policy:\n"
    "- First decide the evidence source for <active_query>: answer only when "
    "the current visual window or an already returned recall result contains "
    "the required evidence; stay silent when the answer is a future event; "
    "recall when the required evidence is likely historical and not visible "
    "now.\n"
    "- Use the recall tool when <active_query> cannot be answered from the "
    "current <visual_window> or the already returned <recall_result>, but the "
    "needed evidence may be in earlier video history.\n"
    "- Use recall when the current frames are unclear, cropped, too late, or "
    "do not visibly contain the requested historical object, action, OCR, "
    "count, state, or attribute. Do not guess and do not answer from memory "
    "alone in those cases.\n"
    "- A single useful recall is preferable to guessing from stale memory when "
    "historical evidence could materially improve the answer.\n"
    "- Do not call recall when the answer is already visible in the current "
    "visual window. Do not call recall repeatedly for the same active query "
    "after a recall result has already been returned.\n\n"
    "Do not emit the compress tool when this ordinary streaming prompt is the "
    "active turn policy. Compression uses a separate system prompt.\n\n"
    "Required output grammar for ordinary streaming turns:\n"
    "- Every assistant message must be exactly one <think> block followed by "
    "exactly one terminal block. Do not write any text outside these tags.\n"
    "- Think block format:\n"
    "  <think>40-100 tokens describing only observable facts in the current "
    "chunk and the selected terminal form: recall tool vs answer response vs "
    "silent answer. When selecting recall, explicitly state that the current "
    "visible evidence is insufficient and that a historical window is being "
    "recalled.</think>\n"
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
    "name the new object/action directly. Do not dump raw <memory> or raw "
    "<recall_result> text inside the think block; state only the minimal "
    "evidence/action decision. No meta-reasoning, no sound/smell/emotion, no "
    "speculation."
)

SYSTEM_PROMPT_V12_COMPRESS = (
    "[MEMORY_MAINTENANCE / COMPRESS TURN]\n"
    "You are the memory-compaction controller for a streaming video agent. "
    "Directly compress memory now. This turn MUST emit a compress tool_call; "
    "it is not an ordinary QA, recall, answer, silent, or continue-watching "
    "turn.\n\n"
    "The user payload may include <user_input><compress_trigger/></user_input> "
    "as a legacy event marker. The marker is not an instruction source; this "
    "system prompt is the instruction. Compression turns include text memory "
    "only; active queries, recalled frames, and the current visual window are "
    "not part of this action.\n\n"
    "Required compression behavior:\n"
    "- Compression is mandatory on this turn; the only legal terminal action "
    "is one compress tool_call.\n"
    "- Do not answer any user question.\n"
    "- Do not emit a silent answer.\n"
    "- Do not call recall.\n"
    "- Do not continue watching or describe the current visual chunk.\n"
    "- Emit exactly one compress tool_call after a short compression think.\n"
    "- Choose an older contiguous time range from <memory> and summarize it so "
    "the summary can replace those text memory records.\n"
    "- Choose the contiguous range whose replacement by a summary is least "
    "likely to hurt later reasoning. Prefer memory that is repetitive, stable, "
    "or already easy to summarize; preserve rare entities, OCR, state changes, "
    "and details tied to unresolved questions in the summary if they are inside "
    "the selected range.\n"
    "- The compression target must come from <memory>, not from fresh visual "
    "frames or active QA context.\n"
    "- Keep the replacement summary concise: target 120-280 tokens, and do "
    "not exceed 280 tokens unless essential OCR/entity details require it.\n"
    "- Retain entity names, visual attributes, OCR text, spatial relations, "
    "and state changes. Do not invent facts or drop details needed for future "
    "questions.\n\n"
    "Required output grammar for compression turns:\n"
    "- Every assistant message must be exactly one <think> block followed by "
    "exactly one compress <tool_call>. This is required, not optional. Do "
    "not write any text outside these tags.\n"
    "- Think block format:\n"
    "  <think>20-60 tokens stating that memory is over budget and which older "
    "contiguous time range should be compressed</think>\n"
    "- Compress tool format:\n"
    "  <tool_call>{\"name\":\"compress\",\"arguments\":{\"time_range\":[start_sec,end_sec],\"text\":\"summary text\"}}</tool_call>\n"
    "  The JSON must use double quotes. time_range must be a two-integer array "
    "[start_sec,end_sec] with seconds from the selected older <memory> range. "
    "text must be the replacement summary for that range, normally within "
    "120-280 tokens.\n"
    "- Do not emit <answer>...</answer>, <answer></answer>, or a recall "
    "tool_call on compression turns."
)


SYSTEM_PROMPT_V12_RECALL_RESPONSE = (
    "[POST_RECALL DECISION TURN]\n"
    "You are the post-recall decision controller for a streaming video agent. "
    "This turn happens immediately after the model requested recall for the "
    "same video chunk. No tools are available on this turn.\n\n"
    f"{_FRAME_CARRIER_TS_PROMPT}"
    "The conversation contains the original chunk user message, the previous "
    "assistant recall tool_call, and a new user payload with <recall_result> "
    "and optionally <recalled_frames>. The current chunk's text memory was "
    "already emitted in the previous recall tool turn; do not create another "
    "current-frame observation.\n\n"
    "Required post-recall decision behavior:\n"
    "- Do not emit any <tool_call>. Do not call recall or compress.\n"
    "- Use <recall_result> and <recalled_frames> as historical evidence for "
    "the active query. Use the original current visual window only as context "
    "for the same chunk, not as proof of a historical detail.\n"
    "- If the recalled evidence is sufficient, output the answer in the "
    "active query's required answer format. For MC letter-only questions, "
    "output only A, B, C, or D.\n"
    "- If the recalled evidence is insufficient or shows the answer has not "
    "appeared yet, output an empty <answer></answer>.\n\n"
    "Required output grammar for post-recall decision turns:\n"
    "- Every assistant message must be exactly one <think> block followed by "
    "exactly one <answer> block. Do not write text outside these tags.\n"
    "- Think block format:\n"
    "  <think>20-60 tokens comparing the recalled evidence to the active "
    "query and deciding answer vs empty answer. Do not dump raw memory or raw "
    "recall_result text.</think>\n"
    "- Non-empty answer format:\n"
    "  <answer>response text</answer>\n"
    "- Silent answer format:\n"
    "  <answer></answer>\n"
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

SYSTEM_PROMPT_V12_RECALL_RESPONSE_VIDEO_META = (
    SYSTEM_PROMPT_V12_RECALL_RESPONSE
    .replace(_FRAME_CARRIER_TS_PROMPT, _FRAME_CARRIER_VIDEO_META_PROMPT)
    .replace(
        "Use the original current visual window only as context ",
        "Use the original current visual frames only as context ",
    )
)


def normalize_system_prompt_kind(
    prompt_kind: Optional[str] = None,
    *,
    inter_chunk: bool = False,
) -> str:
    """Return streaming|compress|recall_response for the current turn.

    ``recall_response`` is retained as the canonical internal name for
    compatibility with existing data and runtime call sites. Newly rendered
    SFT rows may use the less ambiguous metadata alias ``post_recall``.
    """
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
        "recall_result": "recall_response",
        "recall_answer": "recall_response",
        "answer_after_recall": "recall_response",
        "post_recall": "recall_response",
        "recall_followup": "recall_response",
        "post_recall_decision": "recall_response",
        "tool_result": "recall_response",
        "no_tools": "recall_response",
    }
    value = aliases.get(value, value)
    if value not in {"streaming", "compress", "recall_response"}:
        return "streaming"
    return value


def system_prompt_for_frame_protocol(
    frame_protocol: Optional[str] = None,
    *,
    prompt_kind: Optional[str] = None,
    inter_chunk: bool = False,
    render_layout: Optional[str] = None,
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
    layout = normalize_render_layout(render_layout)
    kind = normalize_system_prompt_kind(prompt_kind, inter_chunk=inter_chunk)
    if kind == "recall_response":
        if protocol == FRAME_PROTOCOL_VIDEO_META:
            prompt = SYSTEM_PROMPT_V12_RECALL_RESPONSE_VIDEO_META
        else:
            prompt = SYSTEM_PROMPT_V12_RECALL_RESPONSE
        return _apply_render_layout_to_system_prompt(prompt, layout)
    if protocol == FRAME_PROTOCOL_VIDEO_META:
        if kind == "compress":
            prompt = SYSTEM_PROMPT_V12_COMPRESS_VIDEO_META
        else:
            prompt = SYSTEM_PROMPT_V12_VIDEO_META
        return _apply_render_layout_to_system_prompt(prompt, layout)
    if kind == "compress":
        prompt = SYSTEM_PROMPT_V12_COMPRESS
    else:
        prompt = SYSTEM_PROMPT_V12_STREAMING
    return _apply_render_layout_to_system_prompt(prompt, layout)


def _apply_render_layout_to_system_prompt(prompt: str, layout: str) -> str:
    if layout == RENDER_LAYOUT_STANDARD:
        return prompt
    carrier = (
        _FRAME_CARRIER_TIMELINE_IMAGEPAD_PROMPT
        if layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD
        else _FRAME_CARRIER_TIMELINE_VIDEO_PROMPT
    )
    prompt = prompt.replace(_FRAME_CARRIER_TS_PROMPT, carrier)
    prompt = prompt.replace(_FRAME_CARRIER_VIDEO_META_PROMPT, carrier)
    return (
        prompt.replace("current <visual_window>", "current visual chunks")
        .replace("the current <visual_window>", "the current visual chunks")
        .replace("current visual window", "current visual chunks")
        .replace("current visual window", "current visual chunks")
        .replace("current visual frames", "current visual frames")
        .replace("original current visual window", "original current visual chunks")
        .replace("<visual_window>.current_time", "the visual chunk timestamps")
    )


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
            "thinks. Recall is user-question driven: use it when the active "
            "query asks for an earlier "
            "object/action/OCR/count/state/attribute that is not visible in "
            "the current window, or when current frames are unclear/cropped "
            "and historical evidence may resolve the answer. Do not use it "
            "when the answer is already visible now. This is not a final "
            "answer."
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
            "Memory-maintenance compression-turn tool. Use only under the "
            "compression system prompt or its legacy <compress_trigger/> "
            "event marker. Compression is system-memory-pressure driven, not "
            "user-question driven. Do not answer questions, emit a silent "
            "answer, continue watching, or call recall on that turn. Decide "
            "which older contiguous range from <memory> to compress and "
            "output a concise summary retaining all entities, attributes, "
            "OCR, and state changes."
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
                        "You select this range from contiguous <memory> "
                        "contents under memory pressure."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "The summary text. Retain entity names, visual "
                        "attributes, OCR text, and state changes. Target "
                        "120-280 tokens."
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
    """Return streaming|compress|recall_response for tool/action gating.

    ``post_recall`` is accepted as the preferred metadata spelling for the
    no-tools turn immediately after a recall tool result.
    """
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
        "post_recall": "recall_response",
        "recall_followup": "recall_response",
        "post_recall_decision": "recall_response",
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


def _find_json_string_value_state(text: str, key: str = "text") -> Dict[str, Any]:
    """Locate a JSON string value and report whether it is closed.

    This intentionally works on partial JSON prefixes, so it can diagnose
    max-token truncation after ``"text": "`` without requiring a complete
    ``</tool_call>`` block.
    """
    marker = f'"{key}"'
    search_from = 0
    while True:
        key_pos = text.find(marker, search_from)
        if key_pos < 0:
            return {
                "key_found": False,
                "started": False,
                "closed": False,
                "start": -1,
                "end": -1,
            }
        pos = key_pos + len(marker)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ":":
            search_from = key_pos + 1
            continue
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return {
                "key_found": True,
                "started": False,
                "closed": False,
                "start": -1,
                "end": -1,
            }
        if text[pos] != '"':
            search_from = key_pos + 1
            continue

        start = pos + 1
        pos = start
        escaped = False
        while pos < len(text):
            ch = text[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return {
                    "key_found": True,
                    "started": True,
                    "closed": True,
                    "start": start,
                    "end": pos,
                }
            pos += 1
        return {
            "key_found": True,
            "started": True,
            "closed": False,
            "start": start,
            "end": len(text),
        }


def _scan_json_prefix_state(text: str) -> Dict[str, Any]:
    """Scan a possibly truncated JSON object prefix."""
    first = text.find("{")
    if first < 0:
        return {
            "started": False,
            "balanced_so_far": False,
            "complete": False,
            "state": "no_json_object",
            "depth": 0,
        }

    stack: List[str] = []
    in_string = False
    escaped = False
    complete_at: Optional[int] = None
    mismatch = False
    for i, ch in enumerate(text[first:], start=first):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                mismatch = True
                break
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                mismatch = True
                break
            if not stack:
                complete_at = i
                break

    if mismatch:
        return {
            "started": True,
            "balanced_so_far": False,
            "complete": False,
            "state": "mismatched_bracket",
            "depth": len(stack),
        }
    if complete_at is not None:
        return {
            "started": True,
            "balanced_so_far": True,
            "complete": True,
            "state": "complete",
            "depth": 0,
            "complete_at": complete_at,
        }
    if in_string:
        state = "open_string"
    elif stack:
        state = "open_object" if stack[-1] == "{" else "open_array"
    else:
        state = "incomplete"
    return {
        "started": True,
        "balanced_so_far": True,
        "complete": False,
        "state": state,
        "depth": len(stack),
    }


def diagnose_compress_output_v12(output_text: str) -> Dict[str, Any]:
    """Diagnose partial compress tool-call structure.

    Strict parsing deliberately fails when generation is truncated before
    ``</tool_call>``. This helper is a fallback metric: it checks whether the
    *front* of the compress tool call is correct, so eval/DAgger can distinguish
    "never entered compress format" from "correct prefix but likely ran out of
    decode budget while writing summary text".
    """
    text = strip_chat_template_boundary_tokens(output_text or "")
    think_closed = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    tool_open_pos = text.find("<tool_call>")
    tool_close_pos = text.find("</tool_call>")
    tool_call_open = tool_open_pos >= 0
    tool_call_closed = tool_close_pos > tool_open_pos >= 0
    if tool_call_open:
        body_start = tool_open_pos + len("<tool_call>")
        body = text[body_start:tool_close_pos if tool_call_closed else None]
    else:
        body = ""

    # Chat end tokens are useful for strict parsing but noisy inside partial
    # JSON diagnostics when the model ended before the closing tool tag.
    body_for_prefix = body
    for marker in ("<|im_end|>", "<|endoftext|>"):
        marker_pos = body_for_prefix.find(marker)
        if marker_pos >= 0:
            body_for_prefix = body_for_prefix[:marker_pos]
    body_for_prefix = body_for_prefix.strip()

    json_scan = _scan_json_prefix_state(body_for_prefix)
    name_compress = bool(re.search(r'"name"\s*:\s*"compress"', body_for_prefix))
    arguments_open = bool(re.search(r'"arguments"\s*:\s*\{', body_for_prefix))
    time_range_open = bool(re.search(r'"time_range"\s*:\s*\[', body_for_prefix))
    time_range_pair = bool(re.search(
        r'"time_range"\s*:\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]',
        body_for_prefix,
    ))
    text_state = _find_json_string_value_state(body_for_prefix, "text")
    text_key_open = bool(re.search(r'"text"\s*:\s*"', body_for_prefix))
    text_started = bool(text_state.get("started"))
    text_closed = bool(text_state.get("closed"))
    json_complete = bool(json_scan.get("complete"))
    front_prefix_ok = bool(
        tool_call_open
        and name_compress
        and arguments_open
        and time_range_pair
        and text_key_open
    )
    likely_truncated = bool(
        front_prefix_ok
        and not tool_call_closed
        and (
            (text_started and not text_closed)
            or json_scan.get("state") in {"open_string", "open_object", "open_array"}
        )
    )
    missing_tool_close_after_complete_json = bool(
        front_prefix_ok and json_complete and not tool_call_closed
    )

    stages = [
        think_closed,
        tool_call_open,
        bool(json_scan.get("started")),
        name_compress,
        arguments_open,
        time_range_open,
        time_range_pair,
        text_key_open,
        text_started,
        text_closed,
        json_complete,
        tool_call_closed,
    ]
    prefix_level = 0
    for ok in stages:
        if not ok:
            break
        prefix_level += 1

    if not tool_call_open:
        label = "no_tool_call"
    elif not bool(json_scan.get("started")):
        label = "tool_open_no_json"
    elif not name_compress:
        label = "json_no_compress_name"
    elif not arguments_open:
        label = "compress_name_no_arguments"
    elif not time_range_pair:
        label = "arguments_no_time_range_pair"
    elif not text_key_open:
        label = "time_range_no_text_key"
    elif likely_truncated:
        label = "good_prefix_likely_truncated"
    elif missing_tool_close_after_complete_json:
        label = "good_json_missing_tool_close"
    elif not text_closed:
        label = "text_not_closed"
    elif not json_complete:
        label = f"json_{json_scan.get('state', 'incomplete')}"
    elif not tool_call_closed:
        label = "missing_tool_close"
    else:
        label = "complete"

    return {
        "think_closed": think_closed,
        "tool_call_open": tool_call_open,
        "json_object_open": bool(json_scan.get("started")),
        "name_compress": name_compress,
        "arguments_open": arguments_open,
        "time_range_open": time_range_open,
        "time_range_pair": time_range_pair,
        "text_key_open": text_key_open,
        "text_started": text_started,
        "text_closed": text_closed,
        "json_complete": json_complete,
        "tool_call_closed": tool_call_closed,
        "front_prefix_ok": front_prefix_ok,
        "likely_truncated": likely_truncated,
        "missing_tool_close_after_complete_json": missing_tool_close_after_complete_json,
        "json_prefix_state": json_scan.get("state", ""),
        "json_prefix_depth": int(json_scan.get("depth", 0) or 0),
        "prefix_level": prefix_level,
        "label": label,
    }


def parse_agent_output_v12(
    output_text: str,
    *,
    allow_bare_answer: bool = False,
    allow_malformed_tool_call: bool = False,
) -> Dict:
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
    output_text = strip_chat_template_boundary_tokens(output_text or "")
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
            tool_obj = _loads_tool_call_json_lenient(tool_match.group(1).strip())
        except (json.JSONDecodeError, ValueError) as e:
            result["format_error"] = f"tool_call JSON parse error: {e}"
            return result

        return _finish_tool_call_parse(result, tool_obj)

    if allow_malformed_tool_call and len(think_matches) == 1:
        tool_body = _extract_malformed_tool_call_body(output_text, think_matches[0])
        if tool_body:
            try:
                tool_obj = _loads_tool_call_json_lenient(tool_body)
            except (json.JSONDecodeError, ValueError) as e:
                result["format_error"] = f"tool_call JSON parse error: {e}"
                return result
            return _finish_tool_call_parse(result, tool_obj)

    if allow_bare_answer and len(think_matches) == 1:
        think_match = think_matches[0]
        prefix = output_text[:think_match.start()].strip()
        bare = output_text[think_match.end():].strip()
        if (
            prefix == ""
            and bare
            and "<tool_call" not in bare
            and "<answer" not in bare
        ):
            result["kind"] = "answer"
            result["answer_text"] = _normalize_bare_answer_text(bare)
            result["format_error"] = None
            return result

    result["format_error"] = "neither <answer> nor <tool_call> emitted"
    return result


def _finish_tool_call_parse(result: Dict[str, Any], tool_obj: Any) -> Dict[str, Any]:
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


def _extract_malformed_tool_call_body(output_text: str, think_match: re.Match) -> Optional[str]:
    suffix = str(output_text or "")[think_match.end():].strip()
    if not suffix.startswith("<tool"):
        return None
    brace = suffix.find("{")
    if brace < 0:
        return None
    body = suffix[brace:].strip()
    for marker in ("</tool_call>", "</tool>", "<|im_end|>", "<|endoftext|>"):
        marker_pos = body.find(marker)
        if marker_pos >= 0:
            body = body[:marker_pos].strip()
            break
    scan = _scan_json_prefix_state(body)
    complete_at = scan.get("complete_at")
    if scan.get("complete") and complete_at is not None:
        try:
            body = body[: int(complete_at) + 1]
        except (TypeError, ValueError):
            pass
    return body or None


def _loads_tool_call_json_lenient(raw: str) -> Dict:
    """Load tool-call JSON with narrow model-output repairs.

    Some rollouts emit Python-style escaped apostrophes inside JSON strings
    (``\'``). That sequence is invalid JSON because apostrophes do not need
    escaping, but the intended value is unambiguous.

    Compression summaries also occasionally contain raw OCR quotes/newlines, or
    a duplicated ``<tool_call>`` prefix inside the summary text. For that case,
    recover only the known ``compress`` object shape and leave other malformed
    JSON strict.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as original_error:
        repaired = raw.replace("\\'", "'")
        if repaired != raw:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        fallback = _parse_tool_call_json_fallback(repaired)
        if fallback is not None:
            return fallback
        raise original_error


def _normalize_bare_answer_text(text: str) -> str:
    """Clean post-recall bare answers without relaxing the global grammar."""
    value = str(text or "").strip()
    simple_tag = re.fullmatch(r"<([A-Za-z][A-Za-z0-9_-]*)>(.*?)</\1>", value, re.DOTALL)
    if simple_tag:
        return simple_tag.group(2).strip()
    return value


def _json_unescape_best_effort(value: str) -> str:
    value = str(value or "")
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value.replace(r"\'", "'")
            .replace(r'\"', '"')
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
            .replace(r"\\", "\\")
        )


def _extract_json_string_value_lenient(raw: str, key: str, *, last: bool = True) -> Optional[str]:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*"', re.DOTALL)
    matches = list(pattern.finditer(raw or ""))
    if not matches:
        return None
    match = matches[-1] if last else matches[0]
    start = match.end()

    if key == "text":
        tail = raw[start:]
        close_match = re.search(r'"\s*\}\s*\}[\s\}\]]*$', tail, re.DOTALL)
        if close_match:
            return _json_unescape_best_effort(tail[:close_match.start()])

    pos = start
    escaped = False
    while pos < len(raw):
        ch = raw[pos]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return _json_unescape_best_effort(raw[start:pos])
        pos += 1
    return None


def _parse_tool_call_json_fallback(raw: str) -> Optional[Dict[str, Any]]:
    """Recover known tool-call shapes from malformed model JSON."""
    text = str(raw or "").strip()
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    if not names:
        return None
    name = names[-1]

    if name == "compress":
        ranges = list(re.finditer(
            r'"time_range"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            text,
            re.DOTALL,
        ))
        if not ranges:
            return None
        start_raw, end_raw = ranges[-1].group(1), ranges[-1].group(2)
        summary = _extract_json_string_value_lenient(text, "text", last=True)
        if summary is None or not summary.strip():
            return None
        start = float(start_raw) if "." in start_raw else int(start_raw)
        end = float(end_raw) if "." in end_raw else int(end_raw)
        return {
            "name": "compress",
            "arguments": {"time_range": [start, end], "text": summary.strip()},
        }

    if name == "recall":
        query = _extract_json_string_value_lenient(text, "query", last=True)
        time_range = _extract_json_string_value_lenient(text, "time_range", last=True)
        if query is None or time_range is None:
            return None
        return {
            "name": "recall",
            "arguments": {"query": query.strip(), "time_range": time_range.strip()},
        }

    return None


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
