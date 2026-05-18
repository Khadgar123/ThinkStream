"""Shared agent protocol: input construction and output parsing.

This module is the single source of truth for the agent's input/output format.
Used by:
- Data construction (scripts/agent_data/pass2_rollout.py / pass5_messages.py)
- SFT training (thinkstream/sft/data_processor.py)
- RL rollout (thinkstream/rl/streaming_agent_loop.py)
- Inference (thinkstream/model/agent_loop.py)

Any change to the protocol format MUST be made here to guarantee
train/inference format identity.
"""

import json
import os
import re
import html
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS

# ---------------------------------------------------------------------------
# Constants (canonical values, importable by all consumers)
# ---------------------------------------------------------------------------

# v12.5: canonical values now live in scripts/agent_data/config.py.
# Kept here as fallbacks when that import isn't available (deployed inference
# environments without the data-construction package).
try:
    from scripts.agent_data.config import (
        AGENT_CHUNK_SEC,
        VISUAL_WINDOW_CHUNKS,
        FRAMES_PER_CHUNK,
        RECALL_RETURN_CHUNKS,
    )
except ImportError:
    AGENT_CHUNK_SEC = 1
    VISUAL_WINDOW_CHUNKS = 8
    FRAMES_PER_CHUNK = 2
    RECALL_RETURN_CHUNKS = 4


COMPRESS_TRIGGER_TAG = "<compress_trigger/>"
_COMPRESS_TRIGGER_RE = re.compile(r"<compress_trigger\b")


def chunk_frame_indices(
    chunk_idx: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> List[int]:
    """Zero-based source frame indices for a zero-based chunk.

    Chunk/time semantics are 0-based everywhere: chunk 0 covers t=[0, 1)
    and source frame indices [0, 1] at 2 fps. Project JPEG files produced by
    ffmpeg's ``frame_%06d.jpg`` are 1-based names, so file-name conversion is
    handled separately below.
    """
    start = int(chunk_idx) * int(frames_per_chunk)
    return [start + i for i in range(int(frames_per_chunk))]


def project_frame_filename(frame_index: int, *, width: int = 6) -> str:
    """Return the project JPEG name for a zero-based source frame index."""
    return f"frame_{int(frame_index) + 1:0{int(width)}d}.jpg"


def chunk_frame_filenames(
    chunk_idx: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    *,
    width: int = 6,
) -> List[str]:
    return [
        project_frame_filename(idx, width=width)
        for idx in chunk_frame_indices(chunk_idx, frames_per_chunk)
    ]


def resolve_chunk_frame_paths(
    frame_dir: Path,
    chunk_idx: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    *,
    allow_legacy_zero_based: bool = True,
) -> List[str]:
    """Resolve the two project JPEGs for one zero-based chunk.

    Primary layout is pass1/ffmpeg's 1-based ``frame_000001.jpg`` naming.
    The optional fallback keeps older synthetic/eval dumps with 0-based file
    names readable without changing the canonical time semantics.
    """
    root = Path(frame_dir)
    zero_indices = chunk_frame_indices(chunk_idx, frames_per_chunk)

    def _first_existing(candidates: Sequence[Path]) -> Optional[Path]:
        return next((p for p in candidates if p.exists()), None)

    # Try the canonical project layout as a complete set first. Mixing
    # canonical and legacy probes frame-by-frame can duplicate frame_000001 in
    # old 0-based synthetic dumps, so fallback is all-or-nothing.
    canonical: List[str] = []
    for zero_idx in zero_indices:
        chosen = _first_existing([
            root / project_frame_filename(zero_idx, width=6),
            root / project_frame_filename(zero_idx, width=5),
            root / project_frame_filename(zero_idx, width=4),
        ])
        if chosen is None:
            canonical = []
            break
        canonical.append(str(chosen))
    if len(canonical) == len(zero_indices):
        return canonical

    if not allow_legacy_zero_based:
        return []

    legacy: List[str] = []
    for zero_idx in zero_indices:
        chosen = _first_existing([
            root / f"frame_{zero_idx:06d}.jpg",
            root / f"frame_{zero_idx:05d}.jpg",
            root / f"{zero_idx:06d}.jpg",
            root / f"{zero_idx:05d}.jpg",
        ])
        if chosen is None:
            return []
        legacy.append(str(chosen))
    return legacy


def _contains_compress_trigger(user_text: str) -> bool:
    return bool(_COMPRESS_TRIGGER_RE.search(user_text or ""))


def build_compress_trigger_user_input() -> str:
    """Legacy marker for archived compression-trigger samples.

    New runtime/SFT paths keep compression triggers in controller metadata and
    must not render this marker into model-visible user text.
    """
    return COMPRESS_TRIGGER_TAG


def normalize_user_input_for_turn(user_input: str, *, inter_chunk: bool = False) -> str:
    """Normalize user-side event markers for the current turn kind."""
    text = str(user_input or "")
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


def select_recall_chunks_uniform(
    chunks: Optional[Sequence[Any]],
    max_chunks: Optional[int] = None,
) -> List[int]:
    """De-duplicate and uniformly sample recall chunks over their time span.

    Time-range recall has no query ranking, so "top K" is not meaningful.
    Select evenly spaced chunks instead, preserving the first and last chunks
    whenever the range is larger than the recall visual budget.
    """
    limit = RECALL_RETURN_CHUNKS if max_chunks is None else int(max_chunks)
    if limit <= 0:
        return []
    clean: List[int] = []
    seen = set()
    for raw in chunks or []:
        try:
            chunk = int(raw)
        except (TypeError, ValueError):
            continue
        if chunk < 0 or chunk in seen:
            continue
        seen.add(chunk)
        clean.append(chunk)
    clean = sorted(clean)
    if len(clean) <= limit:
        return clean
    if limit == 1:
        return [clean[0]]
    n = len(clean)
    idxs = [
        int(round(i * (n - 1) / float(limit - 1)))
        for i in range(limit)
    ]
    out: List[int] = []
    used = set()
    for idx in idxs:
        idx = min(max(0, idx), n - 1)
        val = clean[idx]
        if val not in used:
            used.add(val)
            out.append(val)
    # Rounding can collide for small ranges; fill deterministically.
    if len(out) < limit:
        for val in clean:
            if val in used:
                continue
            used.add(val)
            out.append(val)
            if len(out) >= limit:
                break
    return sorted(out)


def recall_time_range_for_chunks(
    chunks: Optional[Sequence[Any]],
    *,
    chunk_sec: float = AGENT_CHUNK_SEC,
) -> Optional[List[int]]:
    """Return the closed timestamp span reported for selected recall chunks."""
    selected = select_recall_chunks_uniform(chunks)
    if not selected:
        return None
    start = min(selected) * float(chunk_sec)
    end = max(selected) * float(chunk_sec)
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


PREFER_PATH_FRAME_INDEX_ENV = "THINKSTREAM_PREFER_PATH_FRAME_INDEX"


def prefer_path_frame_index() -> bool:
    value = os.environ.get(PREFER_PATH_FRAME_INDEX_ENV)
    if value is None:
        return True
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


RECALL_VISUAL_LAYOUT_ENV = "THINKSTREAM_RECALL_VISUAL_LAYOUT"
RECALL_VISUAL_LAYOUT_CHUNKED = "chunked"
RECALL_VISUAL_LAYOUT_PACKED = "packed"

AGENT_SPECIAL_TOKENS = (
    "<think>",
    "</think>",
    "</Response>",
    "</Silence>",
    "<tool_call>",
    "</tool_call>",
)

WRONG_RESPONSE_SPECIAL_TOKENS = (
    "<response>",
    "</response>",
    "<answer>",
    "</answer>",
)


def missing_agent_special_tokens(tokenizer) -> List[str]:
    """Return canonical agent tags that are not registered as special tokens."""
    vocab = set(getattr(tokenizer, "get_vocab", lambda: {})().keys())
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    return [
        tok for tok in AGENT_SPECIAL_TOKENS
        if tok not in vocab or tok not in special
    ]


def ensure_agent_special_tokens(tokenizer, model: Optional[Any] = None) -> int:
    """Register the current Streamo-style agent tags on a tokenizer.

    This intentionally registers ``</Response>`` and ``</Silence>`` as the
    action tokens. It never registers old paired tags such as
    ``<response>...</response>`` or ``<answer>...</answer>``.
    """
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    filtered_existing: List[str] = []
    seen = set()
    for tok in existing:
        if tok in WRONG_RESPONSE_SPECIAL_TOKENS or tok in seen:
            continue
        filtered_existing.append(tok)
        seen.add(tok)
    merged = list(filtered_existing)
    for tok in AGENT_SPECIAL_TOKENS:
        if tok not in seen:
            merged.append(tok)
            seen.add(tok)

    missing = missing_agent_special_tokens(tokenizer)
    has_wrong_registered = any(tok in WRONG_RESPONSE_SPECIAL_TOKENS for tok in existing)
    if not missing and not has_wrong_registered and merged == existing:
        return 0
    old_size = len(tokenizer)
    added = int(tokenizer.add_special_tokens({
        "additional_special_tokens": merged,
    }, replace_additional_special_tokens=True))
    if model is not None:
        try:
            emb = model.get_input_embeddings()
            model_vocab_size = int(getattr(getattr(emb, "weight", None), "shape", [0])[0])
        except Exception:
            model_vocab_size = 0
        if model_vocab_size and len(tokenizer) > model_vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    return added


def validate_agent_special_tokens(tokenizer) -> None:
    """Fail fast when the active action tags are not single special tokens."""
    for tok in ("</Response>", "</Silence>", "</think>"):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"{tok!r} must be one tokenizer token; got token ids={ids}. "
                "Register AGENT_SPECIAL_TOKENS before training/eval/rollout."
            )
    wrong = sorted(
        set(getattr(tokenizer, "additional_special_tokens", []) or [])
        & set(WRONG_RESPONSE_SPECIAL_TOKENS)
    )
    if wrong:
        raise RuntimeError(
            "Old paired response/action tags are registered as special tokens: "
            f"{wrong}. Use AGENT_SPECIAL_TOKENS with </Response>/</Silence> only."
        )


def normalize_recall_visual_layout(value: Optional[str] = None) -> str:
    raw = value if value is not None else os.environ.get(RECALL_VISUAL_LAYOUT_ENV)
    layout = str(raw or RECALL_VISUAL_LAYOUT_CHUNKED).strip().lower().replace("-", "_")
    if layout in {"chunk", "chunks", "per_chunk", "chunked"}:
        return RECALL_VISUAL_LAYOUT_CHUNKED
    if layout in {"pack", "packed", "single", "single_video"}:
        return RECALL_VISUAL_LAYOUT_PACKED
    raise ValueError(
        f"Unsupported recall visual layout {raw!r}; expected "
        f"{RECALL_VISUAL_LAYOUT_CHUNKED!r} or {RECALL_VISUAL_LAYOUT_PACKED!r}."
    )


def build_recalled_frames_metadata(
    chunks: Optional[Sequence[Any]],
    frame_paths: Optional[Sequence[Any]] = None,
    *,
    source: str = "historical_frames",
    chunk_sec: float = AGENT_CHUNK_SEC,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> Optional[Dict[str, Any]]:
    """Build canonical internal recalled-frame metadata.

    When frame_paths are supplied, callers should build them from the same
    selected chunks returned by select_recall_chunks_uniform().
    """
    selected = select_recall_chunks_uniform(chunks)
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
        "returned_chunks": selected,
    }
    if paths:
        out["frame_paths"] = paths
    return out


def build_recall_result_metadata(
    recall_result: Optional[Dict[str, Any]] = None,
    recalled_frames: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return compact recall metadata for diagnostics/legacy callers.

    The retriever may keep text_content/text internally for debugging and hit
    attribution, but the agent prompt must not expose retrieved textual
    summaries as evidence. Current prompts carry recall metadata inside the
    tool-response visual frames. This function is internal metadata only; the
    prompt renderer does not expose a JSON/XML recall header.
    """
    rr = dict(recall_result or {})
    returned_chunks = select_recall_chunks_uniform(rr.get("returned_chunks") or [])
    out: Dict[str, Any] = {
        "source": rr.get("source", ""),
        "time": rr.get("time", ""),
        "returned_chunks": returned_chunks,
        "status": rr.get("status", "ok" if returned_chunks else "empty"),
    }
    if recalled_frames:
        out["time_range"] = prompt_time_range(recalled_frames.get("time_range"))
        out["n_frames"] = recalled_frames.get("n_frames", 0)
    else:
        tr = recall_time_range_for_chunks(returned_chunks)
        if tr:
            out["time_range"] = prompt_time_range(tr)
    return out


def infer_video_metadata(
    frames: Sequence[Any],
    *,
    fps: Optional[float] = None,
    start_frame_index: int = 0,
    total_num_frames: Optional[int] = None,
    prefer_path_indices: Optional[bool] = None,
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
    use_path_indices = (
        prefer_path_frame_index()
        if prefer_path_indices is None
        else bool(prefer_path_indices)
    )

    for offset, frame in enumerate(frame_seq):
        idx: Optional[int] = None
        if use_path_indices and isinstance(frame, (str, Path)):
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


def prompt_time_value(value: Any) -> Any:
    """Return a compact JSON-safe time value for prompt metadata.

    Chunk-level project times are integer seconds. Data files sometimes carry
    them as floats (for example 12.0); rendering them as JSON integers avoids
    teaching the model that chunk boundaries live at fractional times. True
    sub-second frame timestamps are still carried only by frame metadata.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if num.is_integer():
        return int(num)
    return round(num, 3)


def prompt_time_range(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [prompt_time_value(value[0]), prompt_time_value(value[1])]
    return value


FRAME_PROTOCOL_TS_IMAGE = "ts_image"
FRAME_PROTOCOL_VIDEO_META = "video_meta"
FRAME_PROTOCOL_ENV = "THINKSTREAM_FRAME_PROTOCOL"
ALLOW_TS_IMAGE_PROTOCOL = (
    os.environ.get("THINKSTREAM_ALLOW_TS_IMAGE_PROTOCOL", "").strip().lower()
    in {"1", "true", "yes", "on"}
)
VALID_FRAME_PROTOCOLS = (
    {FRAME_PROTOCOL_VIDEO_META, FRAME_PROTOCOL_TS_IMAGE}
    if ALLOW_TS_IMAGE_PROTOCOL
    else {FRAME_PROTOCOL_VIDEO_META}
)

RENDER_LAYOUT_STANDARD_QUERY_LAST = "standard_query_last"
RENDER_LAYOUT_ENV = "THINKSTREAM_RENDER_LAYOUT"
VALID_RENDER_LAYOUTS = {RENDER_LAYOUT_STANDARD_QUERY_LAST}

MEMORY_POSITION_ENV = "THINKSTREAM_MEMORY_POSITION"
MEMORY_POSITION_BEFORE_VISUAL = "before_visual"
VALID_MEMORY_POSITIONS = {MEMORY_POSITION_BEFORE_VISUAL}


def normalize_frame_protocol(frame_protocol: Optional[str] = None) -> str:
    """Return the active student/eval frame protocol.

    The protocol switch is deliberately late-bound. Teacher pass caches store
    frame paths, timestamps, memory, questions, and answers; SFT/RL/eval decide
    only at render time whether those same frames are carried as timestamped
    images or as a pre-sampled Qwen video block with metadata.
    """
    value = (frame_protocol or os.environ.get(FRAME_PROTOCOL_ENV)
             or FRAME_PROTOCOL_VIDEO_META)
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
        extra = (
            " Set THINKSTREAM_ALLOW_TS_IMAGE_PROTOCOL=1 for legacy/debug "
            "timestamp-image rendering; production KV eviction tracks video "
            "tokens only."
            if value == FRAME_PROTOCOL_TS_IMAGE and not ALLOW_TS_IMAGE_PROTOCOL
            else ""
        )
        raise ValueError(
            f"Unsupported frame protocol {frame_protocol!r}; expected one of "
            f"{sorted(VALID_FRAME_PROTOCOLS)}.{extra}"
        )
    return value


def normalize_render_layout(render_layout: Optional[str] = None) -> str:
    """Return the active prompt layout for SFT/RL/eval rendering."""
    value = (
        render_layout
        or os.environ.get(RENDER_LAYOUT_ENV)
        or RENDER_LAYOUT_STANDARD_QUERY_LAST
    )
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "query_last": RENDER_LAYOUT_STANDARD_QUERY_LAST,
        "standard_querylast": RENDER_LAYOUT_STANDARD_QUERY_LAST,
    }
    value = aliases.get(value, value)
    if value not in VALID_RENDER_LAYOUTS:
        raise ValueError(
            f"Unsupported render layout {render_layout!r}; expected one of "
            f"{sorted(VALID_RENDER_LAYOUTS)}"
        )
    return value


def normalize_memory_position(memory_position: Optional[str] = None) -> str:
    """Return where text memory is rendered relative to the visual window.

    This is an eval-time ablation knob. The default preserves the training
    prompt contract; production SFT/RL launchers do not set this env var.
    """
    value = (
        memory_position
        or os.environ.get(MEMORY_POSITION_ENV)
        or MEMORY_POSITION_BEFORE_VISUAL
    )
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "before": MEMORY_POSITION_BEFORE_VISUAL,
        "top": MEMORY_POSITION_BEFORE_VISUAL,
        "memory_first": MEMORY_POSITION_BEFORE_VISUAL,
    }
    value = aliases.get(value, value)
    if value not in VALID_MEMORY_POSITIONS:
        raise ValueError(
            f"Unsupported memory position {memory_position!r}; expected one of "
            f"{sorted(VALID_MEMORY_POSITIONS)}"
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
    kv_scope: Optional[str] = None,
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
    if kv_scope:
        item["kv_scope"] = str(kv_scope)
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
    kv_scope: Optional[str] = None,
) -> None:
    """Append frames using the selected student/eval visual protocol.

    Production uses one native video block with explicit Qwen video metadata.
    Timestamped individual images are retained only for legacy/debug probes
    and require ``THINKSTREAM_ALLOW_TS_IMAGE_PROTOCOL=1``.
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
        kv_scope=kv_scope,
    )


# ---------------------------------------------------------------------------
# Memory Formatting
# ---------------------------------------------------------------------------

_RECENT_THINK_LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.DOTALL)


def _memory_time_point(value: Any) -> Any:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if "-" in text:
        text = text.split("-", 1)[0].strip()
    try:
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return text


def _coerce_memory_think(item: Any) -> Dict[str, Any]:
    """Normalize an archived chunk observation for tagged rendering."""
    if isinstance(item, str):
        m = _RECENT_THINK_LINE_RE.match(item)
        if m:
            return {
                "time": _memory_time_point(m.group(1).strip()),
                "text": m.group(2).strip(),
            }
        return {"time": "", "text": item.strip()}
    if isinstance(item, dict):
        text = str(item.get("text", item.get("obs", ""))).strip()
        tr = item.get("time_range")
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            tr = prompt_time_range(tr)
            return {
                "time": f"{tr[0]}-{tr[1]}",
                "text": text,
            }
        chunks = item.get("chunks") or []
        use_range = bool(item.get("range_merged")) or (
            isinstance(chunks, list) and len(chunks) > 1
        )
        if use_range and isinstance(tr, (list, tuple)) and len(tr) >= 2:
            tr = prompt_time_range(tr)
            return {
                "time": f"{tr[0]}-{tr[1]}",
                "text": text,
            }
        time_str = item.get(
            "time",
            item.get("chunk", 0) * AGENT_CHUNK_SEC,
        )
        return {
            "time": _memory_time_point(time_str),
            "text": text,
        }
    return {"time": "", "text": str(item).strip()}


def _memory_line_sort_key(time_value: Any) -> tuple:
    text = "" if time_value is None else str(time_value).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return (float("inf"), float("inf"), text)
    start = float(nums[0])
    end = float(nums[-1]) if len(nums) > 1 else start
    return (start, end, text)


def _memory_line(time_value: Any, text: Any) -> str:
    ts = "" if time_value is None else str(time_value).strip()
    body = " ".join(str(text or "").strip().split())
    if not ts or not body:
        return ""
    return f'<m t="{ts}">{body}</m>'


def format_memory_block(memory: Dict) -> str:
    """Format memory state as bare compact-memory lines.

    Input can be either:
    - A snapshot dict with "compressed_segments", "recent_thinks"
    - A pre-structured dict with "compressed", "recent_thinks"
      (as used in per-timestep pipeline samples)

    Both paths produce identical output text:
      <m t="start-end">summary text</m>
      <m t="N-N+1">archived observation text</m>

    Pending status is NOT rendered here — it lives in
    `format_queries_block` as a query entry with empty answers list.
    All 12,405 v9.2 SFT samples were rendered with `pending_questions`
    empty (it was unused in production), so the model has never seen
    a `<pending>` tag. We assert the legacy field is empty here so
    any future caller accidentally populating it fails loudly instead
    of injecting an OOD tag the model can't interpret.
    """
    entries = []

    # Compressed segments
    compressed = memory.get("compressed_segments", memory.get("compressed", []))
    for seg in compressed:
        tr = prompt_time_range(seg.get("time_range"))
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            t_value = f"{tr[0]}-{tr[1]}"
        else:
            t_value = str(tr or "").strip()
        line = _memory_line(t_value, seg.get("text", ""))
        if line:
            entries.append((_memory_line_sort_key(t_value), line))

    # Archived chunk observations. Render them with the same compact <m> form
    # as compression output so streaming turns never expose old JSON tags.
    recent = memory.get("recent_thinks", memory.get("recent_observations", []))
    for item in recent:
        rec = _coerce_memory_think(item)
        if rec.get("text"):
            line = _memory_line(rec.get("time", ""), rec.get("text", ""))
            if line:
                entries.append((_memory_line_sort_key(rec.get("time", "")), line))

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

    entries.sort(key=lambda item: item[0])
    return "\n".join(line for _, line in entries)


def _xml_prompt_text(text: Any) -> str:
    """Escape text embedded inside XML-ish memory-update prompt tags."""
    return html.escape(" ".join(str(text or "").strip().split()), quote=False)


def format_compact_memory_update_input(
    memory: Dict,
    *,
    covered_range: Optional[Sequence[Any]] = None,
) -> str:
    """Format the text-only compact-memory update user payload.

    Streaming turns render recent observations as historical ``<m>`` memory.
    Compression turns are different: SFT trains them as
    ``OLD_MEMORY`` + ``NEW_CAPTIONS`` where raw recent observations are ``<c>``
    lines. Keeping that distinction prevents the model from treating new
    captions as already-preserved old memory.
    """
    old_lines: List[str] = []
    compressed = memory.get("compressed_segments", memory.get("compressed", []))
    for seg in compressed or []:
        if not isinstance(seg, dict):
            continue
        tr = prompt_time_range(seg.get("time_range", seg.get("t", "")))
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            t_value = f"{tr[0]}-{tr[1]}"
        else:
            t_value = str(tr or "").strip()
        body = _xml_prompt_text(seg.get("text", ""))
        if t_value and body:
            old_lines.append(f'  <m t="{t_value}">{body}</m>')
    old_block = "<MEM>\n" + "\n".join(old_lines) + "\n</MEM>"

    caption_lines: List[str] = []
    caption_times: List[float] = []
    recent = memory.get("recent_thinks", memory.get("recent_observations", []))
    for item in recent or []:
        rec = _coerce_memory_think(item)
        body = _xml_prompt_text(rec.get("text", ""))
        if not body:
            continue
        t_value = _memory_time_point(rec.get("time", ""))
        if t_value == "":
            continue
        caption_lines.append(f'  <c t="{t_value}">{body}</c>')
        try:
            caption_times.append(float(t_value))
        except (TypeError, ValueError):
            pass
    captions_block = "<NEW_CAPTIONS>\n" + "\n".join(caption_lines) + "\n</NEW_CAPTIONS>"

    start: Any = ""
    end: Any = ""
    if covered_range is not None and len(covered_range) >= 2:
        start = prompt_time_value(covered_range[0])
        end = prompt_time_value(covered_range[1])
    elif caption_times:
        start_f = min(caption_times)
        end_f = max(caption_times)
        start = int(start_f) if start_f.is_integer() else start_f
        end = int(end_f) if end_f.is_integer() else end_f
    else:
        start = end = 0

    return (
        "OLD_MEMORY:\n"
        f"{old_block}\n\n"
        "NEW_CAPTIONS:\n"
        f"{captions_block}\n\n"
        f"Covered latest span: t={start}-{end}\n"
        "Coverage check: preserve useful OLD_MEMORY and cover the listed "
        "NEW_CAPTIONS using their real timestamps.\n"
        "Boundary rule: keep OLD_MEMORY <m> ranges as whole units; do not "
        "repartition all history into new uniform slices. Treat contiguous "
        "NEW_CAPTIONS as the latest new unit when OLD_MEMORY exists. If the "
        "memory list is too long, merge adjacent complete units starting from "
        "the oldest; never split an old <m> range or merge only half of one "
        "range with half of another.\n"
        "Return only compact-memory XML lines:\n"
        '<m t="start-end">one concise event or state.</m>\n'
        "Do not output NEW_MEMORY:, markdown, prose, analysis, or any text "
        "outside the <m> lines."
    )


def build_recall_result_user_content(
    recalled_frames: Optional[Dict] = None,
    recall_result: Optional[Dict] = None,
    *,
    frame_protocol: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    render_layout: Optional[str] = None,
    recall_visual_layout: Optional[str] = None,
) -> List[Dict]:
    """Build the visual payload returned by the recall tool.

    Keep the model-visible result LongVT/Qwen-like: a short plain status line
    followed by recalled video blocks. Structured recall metadata stays in the
    sample dict for audits and masking; it is not exposed as XML/JSON prompt
    text.
    """
    normalize_render_layout(render_layout)
    visual_layout = normalize_recall_visual_layout(recall_visual_layout)
    user_content: List[Dict] = []
    effective_frames = recalled_frames or {}
    if effective_frames:
        returned_chunks = select_recall_chunks_uniform(
            effective_frames.get("returned_chunks")
            or (recall_result or {}).get("returned_chunks")
            or []
        )
        tr = prompt_time_range(effective_frames["time_range"])
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            range_text = f"t={tr[0]}-{tr[1]}"
        else:
            range_text = "the requested historical range"
        status = (recall_result or {}).get(
            "status", "ok" if returned_chunks else "empty"
        )
        if status == "ok" and effective_frames.get("frame_paths"):
            result_text = (
                "The recall tool returned historical video frames for "
                f"{range_text}."
            )
        else:
            result_text = (
                "The recall tool returned no historical video frames for "
                f"{range_text}."
            )
        user_content.append({
            "type": "text",
            "text": result_text,
            "kv_scope": "recall",
        })
        if effective_frames.get("frame_paths"):
            paths = list(effective_frames["frame_paths"])
            frames_per_chunk = int(FRAMES_PER_CHUNK)
            split_by_chunk = (
                visual_layout == RECALL_VISUAL_LAYOUT_CHUNKED
                and returned_chunks
                and len(paths) == len(returned_chunks) * frames_per_chunk
            )
            if split_by_chunk:
                for i, chunk in enumerate(returned_chunks):
                    chunk_paths = paths[
                        i * frames_per_chunk:(i + 1) * frames_per_chunk
                    ]
                    append_visual_frames(
                        user_content,
                        chunk_paths,
                        frame_protocol=frame_protocol,
                        fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
                        start_frame_index=int(chunk) * frames_per_chunk,
                        total_num_frames=(int(chunk) + 1) * frames_per_chunk,
                        context_label="recalled frame",
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        kv_scope="recall",
                    )
            else:
                tr_start, tr_end = effective_frames["time_range"]
                fps = float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC))
                append_visual_frames(
                    user_content,
                    paths,
                    frame_protocol=frame_protocol,
                    fps=fps,
                    start_frame_index=int(float(tr_start) * fps),
                    total_num_frames=int((float(tr_end) + AGENT_CHUNK_SEC) * fps),
                    context_label="recalled frame",
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    kv_scope="recall",
                )
    elif recall_result:
        returned_chunks = select_recall_chunks_uniform(
            (recall_result or {}).get("returned_chunks") or []
        )
        tr = recall_time_range_for_chunks(returned_chunks)
        tr = prompt_time_range(tr) if tr else []
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            range_text = f"t={tr[0]}-{tr[1]}"
        else:
            range_text = "the requested historical range"
        user_content.append({
            "type": "text",
            "text": (
                "The recall tool returned no historical video frames for "
                f"{range_text}."
            ),
            "kv_scope": "recall",
        })
    return user_content


# Eval-side caps. Aligned to the current independent-question distribution:
#   - QUERY_HISTORY_POLICY now selects only live query records. Answered/closed
#     questions are intentionally not rendered in later turns; the model only
#     sees the current active question plus answer history for that same query.
#   - QUERIES_HISTORY_CAP is a defensive bound for unexpected concurrent open
#     queries. Production pass3 enforces one active question at a time.
#   - RECALL_TEXT_MAX_CHARS is legacy/no-op for prompts; recall_result is
#     internal metadata and recalled_frames carry visual evidence.
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
        # Project style: keep the option letter attached to the option
        # text so the supervised target carries semantic content.
        return "Answer format: letter plus option text, e.g. A) option text."
    if form == "binary":
        return "Answer format: Yes or No only."
    if form == "number":
        return "Answer format: a number only, no explanation."
    if form in {"short_exact", "literal"}:
        return "Answer format: a concise exact phrase, no explanation."
    if form == "descriptive":
        return "Answer format: one concise sentence, no extra explanation."
    return ""


def canonical_answer_instruction(question: Dict[str, Any]) -> str:
    """Return the canonical model-visible answer-format instruction.

    Older generated rows may carry stale/free-form instructions such as
    "letter only", "Integer count.", or a custom binary phrase. For known
    answer_form values, structured metadata is the source of truth and the
    returned line always uses the canonical "Answer format:" surface. Unknown
    legacy forms fall back to the stored instruction.
    """
    if not isinstance(question, dict):
        return ""
    answer_form = str(question.get("answer_form") or "").strip()
    if not answer_form and question.get("options"):
        answer_form = "multiple_choice"
    answer_style = str(question.get("answer_style") or "").strip()
    provided = str(question.get("answer_instruction") or "").strip()

    if answer_form.lower() == "multiple_choice":
        return answer_format_instruction(
            "multiple_choice",
            # Keep MCQ surface protocol uniform across generated data, SFT, RL,
            # and eval. The semantic matcher still accepts the bare letter for
            # robustness, but prompts should train on letter + option text.
            answer_style="letter_plus_text",
            options=question.get("options") or [],
        )

    instruction = answer_format_instruction(
        answer_form,
        answer_style=answer_style,
        options=question.get("options") or [],
    )
    if instruction:
        return instruction
    # Unknown legacy forms may still carry a usable explicit instruction.
    return provided


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

QUERY_RESPONSE_HISTORY_INCLUDE = "include"
QUERY_RESPONSE_HISTORY_OMIT = "omit"


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


def query_response_history_policy(q: Dict) -> str:
    """Return whether prior answers should be rendered for this open query.

    Cumulative/counting questions need their previous emitted counts in
    ``<response_history>``. Independent probe questions, such as current-status
    or evidence-sufficiency checks at repeated timestamps, should see the
    active question again but not prior answers because each probe is local to
    the current timestep.
    """
    explicit = str(
        q.get("response_history_policy")
        or q.get("query_response_history_policy")
        or ""
    ).strip().lower().replace("-", "_")
    if explicit in {"include", "show", "history", "with_history", "cumulative"}:
        return QUERY_RESPONSE_HISTORY_INCLUDE
    if explicit in {"omit", "hide", "none", "no_history", "independent_probe"}:
        return QUERY_RESPONSE_HISTORY_OMIT

    answer_form = str(q.get("answer_form") or "").strip().lower()
    question_type = str(q.get("question_type") or "").strip().lower()
    question_way = str(q.get("question_way") or "").strip().lower()
    evidence_type = str(q.get("evidence_type") or "").strip().lower()
    fields = " ".join(
        str(q.get(k) or "")
        for k in (
            "family",
            "task",
            "source_task",
            "ovo_task",
            "mechanism",
            "question_way",
            "evidence_type",
        )
    ).upper()
    tokens = {tok for tok in re.split(r"[^A-Z0-9]+", fields) if tok}

    if (
        answer_form == "number"
        or question_way == "repeated_count"
        or {"F5", "REC"} & tokens
    ):
        return QUERY_RESPONSE_HISTORY_INCLUDE

    if (
        question_way in {"current_status_probe", "evidence_sufficiency_probe"}
        or evidence_type == "status_probe_stream"
        or {"F7", "CRR1", "CRR", "SSR"} & tokens
    ):
        return QUERY_RESPONSE_HISTORY_OMIT

    if question_type == "multi_emit" and answer_form in {"binary", "yes_no", "yes/no"}:
        return QUERY_RESPONSE_HISTORY_OMIT

    return QUERY_RESPONSE_HISTORY_INCLUDE


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


_HISTORY_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _response_history_answer_text(q: Dict, answer_text: str) -> str:
    """Render prior answers in the active-query history.

    The history is a model-visible control signal, not the raw audit log. For
    cumulative numeric queries, keep it in the same surface form the active
    query asks for: one number only. This prevents malformed outputs such as
    ``1/1`` from becoming the next turn's exemplar.
    """
    text = str(answer_text or "").strip()
    if str(q.get("answer_form") or "").strip().lower() != "number":
        return text
    match = _HISTORY_NUM_RE.search(text)
    if not match:
        return text
    value = match.group(0)
    if value.endswith(".0"):
        value = value[:-2]
    return value


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
    - <response_history>: prior answers for that same active query when the
      query is cumulative. Independent probe queries keep the block empty.

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
    answer_form = str(q.get("answer_form") or "").strip()
    if (answer_form == "multiple_choice" or (not answer_form and q.get("options"))) and q.get("options"):
        opts = " ".join(str(opt) for opt in q.get("options") or [])
        active_lines.append(
            f"{prefix} Options: {opts}" if prefix else f"Options: {opts}"
        )
    instruction = canonical_answer_instruction(q)
    if instruction:
        active_lines.append(f"{prefix} {instruction}" if prefix else instruction)

    answers = []
    if query_response_history_policy(q) != QUERY_RESPONSE_HISTORY_OMIT:
        for ans in q.get("answers", []) or []:
            if isinstance(ans, dict):
                if ans.get("counts_for_completion") is False:
                    continue
                answers.append((
                    ans.get("time", ask_t),
                    _response_history_answer_text(q, str(ans.get("text", ""))),
                ))
            else:
                answers.append((
                    q.get("response_time", ask_t),
                    _response_history_answer_text(q, str(ans)),
                ))
        answers.sort(key=lambda x: _query_time_key({"time": x[0]}))
    response_lines = []
    for t, text in answers:
        aprefix = _format_query_time_prefix(t)
        response_lines.append(f"{aprefix} A: {text}" if aprefix else f"A: {text}")

    active_block = "<active_query>\n" + "\n".join(active_lines) + "\n</active_query>"
    response_body = "\n".join(response_lines)
    response_block = f"<response_history>\n{response_body}\n</response_history>"
    return active_block + "\n" + response_block


def user_input_is_active_query_duplicate(
    user_input: str,
    queries: Optional[List[Dict]],
    *,
    inter_chunk: bool = False,
) -> bool:
    """Return whether user_input only repeats the rendered active question."""
    if inter_chunk or not queries:
        return False
    text = normalize_user_input_for_turn(user_input, inter_chunk=False).strip()
    if not text:
        return False
    selected = _select_queries_for_prompt(list(queries))
    if not selected:
        return False
    q = max(
        enumerate(selected),
        key=lambda x: (_query_time_key(x[1]), x[0]),
    )[1]
    question = str(q.get("question", "")).strip()
    if not question:
        return False

    def norm(value: str) -> str:
        value = re.sub(r"^\s*(?:\[[^\]\n]+s\]\s*)?Q:\s*", "", value)
        return " ".join(value.split()).strip().lower()

    return norm(text) == norm(question)


def build_user_content(
    memory_text: str,
    chunk_idx: int,
    video_path: str,
    *,
    user_input: str = "",
    queries: Optional[List[Dict]] = None,
    recalled_frames: Optional[Dict] = None,
    recall_result: Optional[Dict] = None,
    # Streaming-runtime profile. Same values as
    # ``schema.DEFAULT_VIDEO_{MIN,MAX}_PIXELS`` and ``sft.args.video_*_pixels``.
    # SFT/RL/Eval/deploy unified; pass1a uses HIRES (set explicitly via
    # mm_processor_kwargs at request level).
    min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    frame_paths: Optional[List[str]] = None,
    frame_protocol: Optional[str] = None,
    inter_chunk: bool = False,
    memory_snapshot: Optional[Dict[str, Any]] = None,
    render_layout: Optional[str] = None,
) -> List[Dict]:
    """Build the user content list for a single-step message.

    Ordinary streaming ordering:
    <user_input> -> bare <m t="..."> memory lines -> current <t=N> +
    video_meta chunk -> <active_query>/<response_history>.

    The fresh user event stays at the front, historical text memory appears
    before vision, and the active query/answer format appears after the visual
    timestamp so the model sees the latest evidence before the final task.

    Pre-extracted frames are rendered by the active frame protocol. The
    supported production setting is ``video_meta``: one Qwen video block with
    explicit frame metadata. All text state is shared across SFT/RL/eval.

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
        frame_protocol: "video_meta" for the supported entrypoints; the old
                        timestamped-image carrier remains only as an internal
                        legacy parser fallback.
        inter_chunk: Memory-compaction turn. Queries, recalled frames, and the
                     current visual sliding window are suppressed; compression
                     is a text-memory action between visual timesteps.

    Note:
        Post-recall turns should call build_recall_result_user_content() instead
        of this helper. They contain historical recalled frames only and no
        current visual window.
    """
    layout = normalize_render_layout(render_layout)
    chunk_sec = AGENT_CHUNK_SEC
    user_content = []

    effective_user_input = (
        ""
        if user_input_is_active_query_duplicate(
            user_input,
            queries,
            inter_chunk=inter_chunk,
        )
        else user_input
    )
    user_input_block = format_user_input_block(
        effective_user_input,
        inter_chunk=inter_chunk,
    )
    prepend_user_input = bool(user_input_block) and user_input_should_prepend(
        inter_chunk=inter_chunk,
    )

    if prepend_user_input:
        user_content.append({
            "type": "text",
            "text": user_input_block.lstrip("\n"),
        })

    memory_position = normalize_memory_position()
    query_last = layout == RENDER_LAYOUT_STANDARD_QUERY_LAST

    def append_memory_block() -> None:
        text = str(memory_text or "").strip()
        if not text:
            return
        user_content.append({
            "type": "text",
            "text": f"\n{text}" if user_content else text,
        })

    def append_queries_block() -> None:
        # Inter-chunk compression is a system memory-pressure event, so omit
        # queries to prevent the model from answering instead of compacting memory.
        if queries and not inter_chunk:
            queries_text = format_queries_block(queries)
            if queries_text:
                user_content.append({
                    "type": "text",
                    "text": f"\n{queries_text}",
                })

    if memory_position == MEMORY_POSITION_BEFORE_VISUAL:
        append_memory_block()

    if not query_last:
        append_queries_block()

    # ── Current visual chunk + protocol-selected frame carrier ──
    # The recurrent KV engine owns the 8-chunk visual window. Each ordinary
    # streaming prompt supplies only the current 1s chunk (2 frames).
    if not inter_chunk:
        video_start = chunk_idx * chunk_sec
        video_end = video_start + chunk_sec
        current_start = chunk_idx * chunk_sec
        t_marker = f"<t={prompt_time_value(current_start)}>"
        user_content.append({
            "type": "text",
            "text": f"\n{t_marker}" if user_content else t_marker,
        })

        if frame_paths:
            current_frame_paths = list(frame_paths)[-FRAMES_PER_CHUNK:]
            append_visual_frames(
                user_content,
                current_frame_paths,
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / chunk_sec),
                start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
                total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
                latest_start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                kv_scope="ordinary",
            )
        elif video_path:
            user_content.append({
                "type": "video",
                "video": video_path,
                "video_start": prompt_time_value(video_start),
                "video_end": prompt_time_value(video_end),
                "nframes": FRAMES_PER_CHUNK,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "kv_scope": "ordinary",
            })

    if query_last:
        append_queries_block()
    # ── Recall result metadata / frames (legacy single-turn path) ──
    # The normal post-recall path calls build_recall_result_user_content()
    # directly. Keep this fallback byte-aligned with that renderer.
    if (recalled_frames or recall_result) and not inter_chunk:
        recall_payload = build_recall_result_user_content(
            recalled_frames,
            recall_result,
            frame_protocol=frame_protocol,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            render_layout=render_layout,
        )
        if recall_payload and user_content and recall_payload[0].get("type") == "text":
            recall_payload = [dict(recall_payload[0]), *recall_payload[1:]]
            recall_payload[0]["text"] = "\n" + str(recall_payload[0].get("text", ""))
        user_content.extend(recall_payload)

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
#   response terminal   = </Response> text or </Silence>
#   tool (recall)       = <tool_call>{"name":"recall","arguments":{...}}</tool_call>
#   compress turn       = stage-marked memory update. The assistant emits
#                         bare compact <m> entries only.
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
    if re.search(r"<MEM\b", cleaned, re.IGNORECASE):
        return cleaned.strip()
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


def decode_agent_output_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode generated assistant tokens while preserving agent tags."""
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return strip_chat_template_boundary_tokens(text)


# ---------------------------------------------------------------------------
# System prompt + sample-field helpers
# ---------------------------------------------------------------------------
#
# The canonical streaming system prompt lives in ``thinkstream.data.schema``.
# Stage transitions (streaming / compress / post-recall) are signalled via
# user-side ``<stage:...>`` markers in the message content, not by selecting
# a different prompt.

from thinkstream.data.schema import (  # noqa: E402
    COMPACT_MEMORY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)


# Canonical sample-dict field for "inter-chunk compress turn" — read sites
# should use :func:`is_inter_chunk` to also accept the legacy name from older
# data files. New write sites use the canonical key directly.
INTER_CHUNK_FIELD = "inter_chunk"
LEGACY_INTER_CHUNK_FIELD = "v12_inter_chunk"


def is_inter_chunk(sample: Dict) -> bool:
    """Return ``True`` if a sample dict is an inter-chunk compress turn.

    Checks the canonical ``inter_chunk`` field first, then falls back to the
    legacy ``v12_inter_chunk`` for compatibility with trajectory JSONL files
    generated before the field rename.
    """
    if not isinstance(sample, dict):
        return False
    value = sample.get(INTER_CHUNK_FIELD)
    if value is None:
        value = sample.get(LEGACY_INTER_CHUNK_FIELD)
    return bool(value)


def get_canonical_system_prompt() -> str:
    """Return the canonical streaming system prompt."""
    return SYSTEM_PROMPT


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
    """Return the turn-local system prompt."""
    layout = normalize_render_layout(render_layout)
    kind = normalize_system_prompt_kind(prompt_kind, inter_chunk=inter_chunk)
    if kind == "compress":
        return COMPACT_MEMORY_SYSTEM_PROMPT
    return _apply_render_layout_to_system_prompt(SYSTEM_PROMPT, layout)


def _apply_render_layout_to_system_prompt(prompt: str, layout: str) -> str:
    if layout != RENDER_LAYOUT_STANDARD_QUERY_LAST:
        raise ValueError(f"Unsupported render layout {layout!r}")
    return prompt


# Tool JSON schemas — passed as ``tools=...`` to apply_chat_template.
# Single source of truth is :func:`thinkstream.data.schema.build_tools_schema`;
# we re-export the per-tool dicts here so legacy call sites that want only
# the recall tool (or only the compress tool) can still pick one out of the
# pair. Both SFT data rendering and RL rollout system prompts see the same
# tool descriptions this way — no drift.
from thinkstream.data.schema import build_tools_schema as _build_tools_schema  # noqa: E402

_RECALL_TOOLS = _build_tools_schema(include_recall=True, include_compress=False)
RECALL_TOOL_SCHEMA = _RECALL_TOOLS[0]
COMPRESS_TOOL_SCHEMA = None
del _RECALL_TOOLS

# Back-compat name for old call sites. New code should use tools_for_turn().
TOOLS_SCHEMA = [RECALL_TOOL_SCHEMA]
STREAMING_TOOLS_SCHEMA = [RECALL_TOOL_SCHEMA]
COMPRESS_TOOLS_SCHEMA = []


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
    - compression turns expose no tools; compact memory is raw <m> content;
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
        # Compact-memory update is plain assistant content (<m> lines),
        # not a function call. Keep legacy compress tool schema defined above
        # for old cached data, but do not expose it in new turns.
        return None
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


def build_assistant_content(
    *,
    think: str,
    kind: str,                    # "answer" | "recall" | "compress"
    answer_text: str = "",        # for kind="answer" (empty → silent)
    recall_query: Optional[Dict] = None,
    compress_summary: Optional[Dict] = None,
) -> str:
    """Build assistant message content in canonical v12 format.

    Returns a single string with <think>...</think> followed by exactly one
    of: recall <tool_call>{...}</tool_call> | </Response> answer | </Silence>.
    For ``kind="compress"``, returns bare compact-memory ``<m>`` lines only.

    Args:
        think: think content (40-80 tokens recommended).
        kind: which terminal to emit.
        answer_text: text after </Response> (empty for silent).
        recall_query: dict with "start_time" and "end_time" keys. Recall tool
            calls intentionally omit text queries; the retriever samples from
            the requested historical video interval.
        compress_summary: dict with "time_range" (list) + "text" keys.
    """
    if kind == "answer":
        parts = [f"<think>{think}</think>"]
        if str(answer_text or "").strip():
            parts.append(f"</Response> {str(answer_text).strip()}")
        else:
            parts.append("</Silence>")
    elif kind == "recall":
        parts = [f"<think>{think}</think>"]
        if not recall_query:
            raise ValueError("kind='recall' requires recall_query dict")
        start_time = recall_query.get("start_time")
        end_time = recall_query.get("end_time")
        if start_time is None or end_time is None:
            raise ValueError("kind='recall' requires start_time and end_time")
        tool_call = {
            "name": "recall",
            "arguments": {
                "start_time": prompt_time_value(start_time),
                "end_time": prompt_time_value(end_time),
            },
        }
        parts.append(
            f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>'
        )
    elif kind == "compress":
        if not compress_summary:
            raise ValueError("kind='compress' requires compress_summary dict")
        memory_text = str(compress_summary.get("memory_text") or "").strip()
        if memory_text:
            return memory_text
        time_range = compress_summary.get("time_range", [])
        if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
            start, end = time_range[0], time_range[1]
        else:
            start, end = 0, 0
        try:
            start_i = int(float(start))
            end_i = int(float(end))
        except (TypeError, ValueError):
            start_i, end_i = 0, 0
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        text = html.escape(str(compress_summary.get("text", "")).strip(), quote=False)
        return f'<m t="{start_i}-{end_i}">{text}</m>'
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Expected answer|recall|compress.")

    return "".join(parts)


def _validate_recall_tool_args(args: Any) -> Optional[str]:
    """Validate recall tool_call arguments.

    Canonical form: ``start_time`` and ``end_time`` absolute seconds. The
    interval is closed: [start_time, end_time].
    """
    if not isinstance(args, dict):
        return "recall arguments must be an object"
    allowed = {"start_time", "end_time"}
    extra = sorted(str(k) for k in args.keys() if k not in allowed)
    if extra:
        return f"recall arguments only support start_time and end_time; got extra keys: {extra}"
    start_time = args.get("start_time")
    end_time = args.get("end_time")
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        return "recall start_time and end_time must be numbers"
    if float(start_time) < 0:
        return "recall start_time must be non-negative"
    if float(end_time) < float(start_time):
        return "recall end_time must be greater than or equal to start_time"
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


def diagnose_compress_output(output_text: str) -> Dict[str, Any]:
    """Diagnose partial compact-memory output structure."""
    text = strip_chat_template_boundary_tokens(output_text or "")
    think_closed = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    mem_body = re.sub(r"</?MEM>", "", text, flags=re.IGNORECASE).strip()
    mem_lines = re.findall(r'<m\s+t="[^"]+"\s*>.*?</m>', mem_body, flags=re.DOTALL | re.IGNORECASE)
    mem_entry_count_ok = 4 <= len(mem_lines) <= 6
    if mem_lines:
        first_line = re.search(r'<m\s+t="[^"]+"\s*>', mem_body, flags=re.IGNORECASE)
        raw_mem_prefix = bool(first_line and not mem_body[:first_line.start()].strip())
        front_prefix_ok = bool(len(mem_lines) >= 1 and (think_closed or raw_mem_prefix))
        open_m = len(re.findall(r"<m\b", mem_body, flags=re.IGNORECASE))
        close_m = len(re.findall(r"</m>", mem_body, flags=re.IGNORECASE))
        mem_closed = open_m == close_m and open_m > 0
        likely_truncated = bool(front_prefix_ok and not mem_closed)
        label = (
            "complete"
            if mem_closed and mem_entry_count_ok
            else "mem_bad_entry_count"
            if mem_closed
            else "mem_likely_truncated"
        )
        return {
            "think_closed": think_closed,
            "tool_call_open": False,
            "json_object_open": False,
            "name_compress": False,
            "arguments_open": False,
            "time_range_open": False,
            "time_range_pair": False,
            "text_key_open": False,
            "text_started": bool(mem_lines),
            "text_closed": mem_closed,
            "json_complete": False,
            "tool_call_closed": False,
            "mem_open": True,
            "mem_closed": mem_closed,
            "mem_entry_count": len(mem_lines),
            "mem_entry_count_ok": mem_entry_count_ok,
            "front_prefix_ok": front_prefix_ok,
            "likely_truncated": likely_truncated,
            "missing_tool_close_after_complete_json": False,
            "json_prefix_state": "",
            "json_prefix_depth": 0,
            "prefix_level": 13 if mem_closed and mem_entry_count_ok else (4 if front_prefix_ok else 0),
            "label": label,
        }

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


def parse_agent_output(
    output_text: str,
    *,
    allow_bare_answer: bool = False,
    allow_bare_memory: bool = True,
    allow_malformed_tool_call: bool = False,
    allow_unclosed_response: bool = False,
) -> Dict:
    """Parse agent output (think + response/silent/tool_call/compact memory).

    Returns:
        {
            "raw": str,
            "think": str,
            "kind": "answer" | "recall" | "compress" | "unknown",
            "answer_text": str | None,         # set when kind=answer
            "tool_call": dict | None,          # parsed JSON when kind=recall
            "memory_text": str | None,         # parsed <m> lines when kind=compress
            "format_error": str | None,        # set when parsing fails
            "lenient_unclosed_response": bool, # kept false; old paired response tags are invalid
        }
    """
    output_text = strip_chat_template_boundary_tokens(output_text or "")
    result = {
        "raw": output_text,
        "think": "",
        "kind": "unknown",
        "answer_text": None,
        "tool_call": None,
        "memory_text": None,
        "format_error": None,
        "lenient_unclosed_response": False,
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

    response_matches = list(re.finditer(r'</Response>\s*(.*?)\s*$', output_text, re.DOTALL))
    silent_matches = list(re.finditer(r'</Silence>\s*', output_text, re.DOTALL))
    tool_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL))
    mem_matches = list(re.finditer(r'<MEM>\s*(.*?)\s*</MEM>', output_text, re.DOTALL | re.IGNORECASE))
    bare_m_matches = list(re.finditer(
        r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>',
        output_text,
        re.DOTALL | re.IGNORECASE,
    ))
    response_match = response_matches[0] if response_matches else None
    silent_match = silent_matches[0] if silent_matches else None
    tool_match = tool_matches[0] if tool_matches else None
    mem_match = mem_matches[0] if mem_matches else None

    n_terminals = (
        int(response_match is not None)
        + int(silent_match is not None)
        + int(tool_match is not None)
        + int(mem_match is not None)
    )
    if n_terminals > 1:
        result["format_error"] = "multiple terminal blocks present"
        return result
    if len(response_matches) > 1:
        result["format_error"] = "multiple response terminal blocks"
        return result
    if len(silent_matches) > 1:
        result["format_error"] = "multiple silent terminal blocks"
        return result
    if len(tool_matches) > 1:
        result["format_error"] = "multiple <tool_call> blocks"
        return result
    if len(mem_matches) > 1:
        result["format_error"] = "multiple legacy <MEM> blocks"
        return result

    if (
        result["format_error"] is None
        and len(think_matches) == 1
        and n_terminals == 1
    ):
        terminal_match = response_match or silent_match or tool_match or mem_match
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

    if response_match:
        result["kind"] = "answer"
        result["answer_text"] = response_match.group(1).strip()
        return result

    if silent_match:
        result["kind"] = "answer"
        result["answer_text"] = ""
        return result

    if mem_match:
        body = mem_match.group(1).strip()
        line_matches = list(re.finditer(
            r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>',
            body,
            re.DOTALL | re.IGNORECASE,
        ))
        if not line_matches:
            result["format_error"] = "memory update must contain at least one <m> line"
            return result
        if any(not (m.group(3) or "").strip() for m in line_matches):
            result["format_error"] = "memory update contains empty <m> line"
            return result
        memory_text = "\n".join(m.group(0).strip() for m in line_matches)
        result["kind"] = "compress"
        result["memory_text"] = memory_text
        result["format_error"] = None
        result["tool_call"] = {
            "name": "memory_update",
            "arguments": {"memory_text": memory_text},
        }
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

    if allow_bare_memory and bare_m_matches:
        if not bare_m_matches:
            result["format_error"] = "memory update must contain at least one <m> line"
            return result
        if any(not (m.group(3) or "").strip() for m in bare_m_matches):
            result["format_error"] = "memory update contains empty <m> line"
            return result
        memory_text = "\n".join(m.group(0).strip() for m in bare_m_matches)
        result["kind"] = "compress"
        result["memory_text"] = memory_text
        result["format_error"] = None
        result["tool_call"] = {
            "name": "memory_update",
            "arguments": {"memory_text": memory_text},
        }
        return result

    if allow_bare_answer and len(think_matches) == 1:
        think_match = think_matches[0]
        prefix = output_text[:think_match.start()].strip()
        bare = output_text[think_match.end():].strip()
        if (
            prefix == ""
            and bare
            and "<tool_call" not in bare
            and "<answer" not in bare
            and "<response" not in bare
            and "<silent" not in bare
        ):
            result["kind"] = "answer"
            result["answer_text"] = _normalize_bare_answer_text(bare)
            result["format_error"] = None
            return result

    result["format_error"] = "neither </Response>/</Silence> nor recall <tool_call> nor compact-memory <m> lines emitted"
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
        result["format_error"] = "compress tool_call is not allowed; emit bare compact-memory <m> lines"
        return result
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

    Recall tool calls may be truncated or lightly malformed in rollout probes;
    recover only the known recall object shape and leave other malformed JSON
    strict. Compress no longer has a tool-call representation.
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

    if name == "recall":
        starts = list(re.finditer(
            r'"start_time"\s*:\s*(-?\d+(?:\.\d+)?)',
            text,
            re.DOTALL,
        ))
        ends = list(re.finditer(
            r'"end_time"\s*:\s*(-?\d+(?:\.\d+)?)',
            text,
            re.DOTALL,
        ))
        if not starts or not ends:
            return None
        start_raw, end_raw = starts[-1].group(1), ends[-1].group(1)
        start = float(start_raw) if "." in start_raw else int(start_raw)
        end = float(end_raw) if "." in end_raw else int(end_raw)
        return {
            "name": "recall",
            "arguments": {"start_time": start, "end_time": end},
        }

    return None


def has_compress_trigger(user_text: str) -> bool:
    """Check if a user message contains a legacy <compress_trigger/>.

    Retained for archived data/parquet compatibility. New runtime/eval uses
    explicit turn_kind/inter_chunk metadata instead of model-visible markers.
    """
    return _contains_compress_trigger(user_text)


def extract_compress_trigger_range(user_text: str) -> Optional[List[int]]:
    """Extract a legacy trigger range if present.

    This parser is retained for archived v11/v12.0 samples and eval fixtures
    that still carry a range attribute.
    """
    m = re.search(
        r"<compress_trigger\s+range\s*=\s*['\"]?(\d+)\s*-\s*(\d+)['\"]?\s*/?>",
        user_text or "",
    )
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2))]
