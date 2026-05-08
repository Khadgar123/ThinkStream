"""PASS 5 — Convert single-step samples to LLaMA-Factory ShareGPT messages format.

Reads pass4 outputs and emits one row per sample in the multi-turn messages
format used by LLaMA-Factory / DeepEyesV2 / VST. Each row is a stand-alone
training sample matching fresh-KV-per-chunk inference: every sample's
user.content carries the full state for ordinary visual turns (user_input +
memory + queries + visual_window + recalled_frames) so the model trains under
the exact same input distribution it sees at inference.

Three sample shapes preserved (canonical pass/SFT/RL/eval timestamped-image protocol):
  A. Single-turn       (silent / response / lonely recall / inter-chunk compress)
  B. Multi-turn recall (recall_query → tool turn → final answer, within one chunk)
  C. Inter-chunk compress (system inserts <compress_trigger> before memory,
     with no visual_window/images/videos because compression is between chunks)

Self-contained: imports only stdlib + thinkstream.data.agent_protocol (which
itself is stdlib-only). No transformers required.

INPUT
  data/agent_v5/final/{train_sft_full,val,test}.jsonl   (flat single-step rows)
  data/agent_v5/final/{train_sft,...}_trajectories.jsonl (samples nested)

OUTPUT
  data/agent_v5/final/{train_sft,val,test}_messages.jsonl
  data/agent_v5/final/dataset_info.json   (LLaMA-Factory entry stub)

Usage:
  python -m scripts.agent_data_v5.pass5_messages
  python -m scripts.agent_data_v5.pass5_messages --input flat
  python -m scripts.agent_data_v5.pass5_messages --input traj
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    append_video_metadata_frame_list,
    format_memory_block,
    format_queries_block,
    format_user_input_block,
    append_visual_frames,
    build_recalled_frames_metadata,
    infer_video_metadata,
    normalize_frame_protocol,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
)

logger = logging.getLogger(__name__)

RENDER_LAYOUT_STANDARD = "standard"
RENDER_LAYOUT_TIMELINE_VIDEO = "timeline_video"
RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD = "timeline_video_imagepad"

# Project layout:
#   <batch_root>/                         (DEFAULT_DATA_DIR)
#   <batch_root>/final/*.jsonl            (FINAL_DIR)
#   <batch_root>/frames/<vid>/...jpg      (frame paths in samples)
#
# Frame paths inside samples may be absolute or relative to PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from scripts.agent_data_v5.config import DATA_ROOT as DEFAULT_DATA_DIR
except Exception:
    DEFAULT_DATA_DIR = Path(
        os.environ.get("THINKSTREAM_DATA_ROOT")
        or os.environ.get("AGENT_DATA_DIR")
        or str(PROJECT_ROOT / "data" / "agent_v5")
    )
    if not DEFAULT_DATA_DIR.is_absolute():
        DEFAULT_DATA_DIR = PROJECT_ROOT / DEFAULT_DATA_DIR
    if DEFAULT_DATA_DIR.name == "final":
        DEFAULT_DATA_DIR = DEFAULT_DATA_DIR.parent
FINAL_DIR = DEFAULT_DATA_DIR / "final"

SPLITS = [
    ("train_sft_full", "train_sft_trajectories", "train_sft_messages"),
    ("val", "val_trajectories", "val_messages"),
    ("test", "test_trajectories", "test_messages"),
]

# SFT rows are independent fresh-KV snapshots, unlike RL/eval trajectories
# which must keep a complete replay timeline. Keep all high-information
# actions, then downsample low-information patrol silence so SFT still learns
# silence without drowning recall/compress/answer actions.
SFT_SILENT_TO_ACTIVE_RATIO = 1.25
SFT_ACTIVE_SILENT_FRACTION = 0.70
SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO = 0.35


# ---------------------------------------------------------------------------
# Messages construction (self-contained mirror of data_processor logic)
# ---------------------------------------------------------------------------

def _resolve_cli_path(raw: str, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base / p


def _frame_rel_prefix(data_dir: Path) -> str:
    try:
        return str((data_dir / "frames").relative_to(PROJECT_ROOT))
    except ValueError:
        return str(data_dir / "frames")


def _resolve_paths(paths: List[str], base_path: Path, data_dir: Path) -> List[str]:
    """Resolve frame paths after moving a batch directory between machines."""
    roots = [base_path]
    if data_dir not in roots:
        roots.append(data_dir)
    if DEFAULT_DATA_DIR not in roots:
        roots.append(DEFAULT_DATA_DIR)
    out: List[str] = []
    for raw in paths:
        p = Path(str(raw))
        direct = p if p.is_absolute() else base_path / p
        if direct.exists():
            out.append(str(direct))
            continue

        parts = p.parts
        if "frames" in parts:
            idx = parts.index("frames")
            for root in roots:
                candidate = root / "frames" / Path(*parts[idx + 1:])
                if candidate.exists():
                    out.append(str(candidate))
                    break
            else:
                out.append(str(direct))
            continue
        out.append(str(direct))
    return out


def _chunks_from_recalled_time_range(rf: Dict, chunk_sec: float) -> List[int]:
    tr = rf.get("time_range") or []
    if len(tr) < 2:
        return []
    try:
        start = int(float(tr[0]) / float(chunk_sec))
        end = int((float(tr[1]) - float(chunk_sec)) / float(chunk_sec))
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    if end < start:
        return []
    return list(range(max(0, start), end + 1))


def _normalise_recalled_frames(inp: Dict, chunk_sec: float) -> Optional[Dict]:
    """Cap recalled frames with the same helper used by RL/eval/runtime."""
    rf = inp.get("recalled_frames") or {}
    if not rf:
        return None
    rr = inp.get("recall_result") or {}
    original_chunks: List[int] = []
    for raw in rr.get("returned_chunks") or []:
        try:
            original_chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not original_chunks:
        original_chunks = _chunks_from_recalled_time_range(rf, chunk_sec)
    selected_chunks = select_recall_chunks(original_chunks)
    if not selected_chunks:
        return None

    original_paths = list(rf.get("frame_paths") or [])
    selected_paths: List[str] = []
    if original_paths and original_chunks:
        by_chunk: Dict[int, List[str]] = {}
        cursor = 0
        for chunk in original_chunks:
            chunk_paths = original_paths[cursor:cursor + FRAMES_PER_CHUNK]
            cursor += FRAMES_PER_CHUNK
            if chunk_paths:
                by_chunk[int(chunk)] = chunk_paths
        for chunk in selected_chunks:
            selected_paths.extend(by_chunk.get(int(chunk), []))
    if original_paths and not selected_paths:
        selected_paths = original_paths[:len(selected_chunks) * FRAMES_PER_CHUNK]

    return build_recalled_frames_metadata(
        selected_chunks,
        selected_paths,
        chunk_sec=chunk_sec,
        frames_per_chunk=FRAMES_PER_CHUNK,
    )


def _compress_management_think_from_output(output: str) -> str:
    think = (
        "Memory is over budget, so I should compress older observations "
        "into a concise summary."
    )
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output or "", re.DOTALL)
    if not m:
        return think
    try:
        tool_call = json.loads(m.group(1))
        tr = (tool_call.get("arguments") or {}).get("time_range") or []
        if isinstance(tr, list) and len(tr) == 2:
            return (
                f"Memory is over budget, so I should compress older observations "
                f"from t={int(tr[0])}-{int(tr[1])} into a concise summary."
            )
    except Exception:
        return think
    return think


def _normalise_assistant_output(sample: Dict) -> str:
    output = str(sample.get("output", ""))
    if sample.get("sample_type") != "compress":
        return output
    think = _compress_management_think_from_output(output)
    replacement = f"<think>{think}</think>"
    if re.search(r"<think>.*?</think>", output, flags=re.DOTALL):
        return re.sub(
            r"<think>.*?</think>",
            replacement,
            output,
            count=1,
            flags=re.DOTALL,
        )
    return replacement + output


def _runtime_mm_kwargs() -> Dict[str, int]:
    try:
        from scripts.agent_data_v5.config import RUNTIME_MM_PROCESSOR_KWARGS as _RTKW
    except ImportError:
        return {"min_pixels": 130_000, "max_pixels": 220_000}
    return {
        "min_pixels": int(_RTKW.get("min_pixels", 130_000)),
        "max_pixels": int(_RTKW.get("max_pixels", 220_000)),
    }


def _visual_window_start(chunk_idx: int) -> int:
    try:
        from scripts.agent_data_v5.config import (
            VISUAL_WINDOW_CHUNKS as _VWC,
            compute_visual_window_start as _cvws,
        )
    except ImportError:
        return max(0, int(chunk_idx) - 15)
    return int(_cvws(int(chunk_idx), _VWC))


def _infer_visual_frame_paths(
    sample: Dict[str, Any],
    data_dir: Path,
    *,
    frame_rel_prefix: str,
) -> List[str]:
    inp = sample.get("input") or {}
    vw = inp.get("visual_window") or {}
    if "frame_paths" in vw:
        return list(vw.get("frame_paths") or [])
    if "frames" not in vw:
        return []
    vid = sample.get("video_id", "")
    if not vid:
        return []
    chunk_idx = int(sample.get("chunk_idx", 0) or 0)
    paths: List[str] = []
    for ci in range(_visual_window_start(chunk_idx), chunk_idx + 1):
        for fi in range(FRAMES_PER_CHUNK):
            fnum = ci * FRAMES_PER_CHUNK + fi + 1
            paths.append(f"{frame_rel_prefix}/{vid}/frame_{fnum:06d}.jpg")
    return paths


def _group_frames_by_chunk(
    frame_paths: List[str],
    *,
    window_start: int,
) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for offset, path in enumerate(frame_paths):
        chunk = int(window_start) + int(offset // FRAMES_PER_CHUNK)
        grouped.setdefault(chunk, []).append(path)
    return grouped


def _coerce_timeline_think(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("observation") or "").strip()
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
        text = m.group(3).strip()
        return {
            "chunk": int(start // float(AGENT_CHUNK_SEC)),
            "time": (
                f"{int(start) if start.is_integer() else start}-"
                f"{int(end) if end.is_integer() else end}"
            ),
            "text": text,
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


def _format_summary_capsule(seg: Dict[str, Any], idx: int) -> str:
    chunks = _segment_chunks(seg)
    if chunks:
        start = min(chunks) * AGENT_CHUNK_SEC
        end = (max(chunks) + 1) * AGENT_CHUNK_SEC
        time_range = f"{int(start)}-{int(end)}"
    else:
        tr = seg.get("time_range") or ["?", "?"]
        time_range = f"{tr[0]}-{tr[1]}"
    text = str(seg.get("text") or "").strip()
    lines = [
        f"<SUMMARY time_range={json.dumps(time_range, ensure_ascii=False)}>",
        text,
    ]
    lines.append("</SUMMARY>")
    return "\n".join(lines)


def _format_memory_think_capsule(rec: Dict[str, Any]) -> str:
    chunk = int(rec.get("chunk", 0) or 0)
    time_text = str(rec.get("time") or "").strip()
    if not time_text:
        start = chunk * AGENT_CHUNK_SEC
        time_text = f"{int(start)}"
    elif "-" in time_text:
        time_text = time_text.split("-", 1)[0].strip()
    text = str(rec.get("text") or "").strip()
    return f"<MEMORY_THINK time={json.dumps(time_text)}>{text}</MEMORY_THINK>"


def _timeline_system_prompt(
    *,
    frame_protocol: str,
    prompt_kind: Optional[str],
    inter_chunk: bool,
    render_layout: str = RENDER_LAYOUT_TIMELINE_VIDEO,
) -> str:
    prompt = system_prompt_for_frame_protocol(
        frame_protocol,
        prompt_kind=prompt_kind,
        inter_chunk=inter_chunk,
    )
    if render_layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD:
        carrier = (
            "Each turn you receive: a time-ordered timeline with tagged memory "
            "capsules and time-marked visual chunks. Each visual chunk contains "
            "the pre-sampled image-pad frames for one second. Summary capsules "
            "are historical memory at their covered time range; recalled "
            "evidence is newly retrieved for the current decision but cites "
            "older time ranges. Use the last visual chunk as the primary source "
            "for current observation. Single-step visual and memory tags use "
            "time points, while multi-step summaries and recalled evidence use "
            "time ranges. Timeline tags never expose frame or chunk ids. "
            "Temporal metadata is routing metadata only: never copy or "
            "paraphrase timestamp markers or metadata lines in your output. "
        )
    else:
        carrier = (
            "Each turn you receive: a time-ordered timeline with tagged memory "
            "capsules and one or more pre-sampled video blocks. Each video block uses "
            "Qwen video_metadata (fps, frames_indices, total_num_frames) to carry "
            "absolute frame timestamps. Summary capsules are historical memory at "
            "their covered time range; recalled evidence is newly retrieved for the "
            "current decision but cites older time ranges. Use the last "
            "visual chunk as the primary source for current observation. Single-step "
            "visual and memory tags use time points, while multi-step summaries "
            "and recalled evidence use time ranges. Timeline tags never expose "
            "frame or chunk ids. Temporal metadata is routing metadata only: "
            "never copy or paraphrase timestamp markers or metadata lines in "
            "your output. "
        )
    prompt = prompt.replace(
        "Each turn you receive: a pre-sampled video block (recent 16s window) + "
        "tagged memory state. The video block uses Qwen video_metadata (fps, "
        "frames_indices, total_num_frames) to carry frame timestamps; use those "
        "timestamps together with <visual_window>.current_time to identify the "
        "current chunk. Temporal metadata is routing metadata only: never copy or "
        "paraphrase timestamp markers, frame indices, role markers, or metadata "
        "lines in your output. ",
        carrier,
    )
    return (
        prompt.replace("current <visual_window>", "current visual chunks")
        .replace("the current <visual_window>", "the current visual chunks")
        .replace("current visual window", "current visual chunks")
        .replace("original current visual window", "original current visual chunks")
    )


def _append_video_typed_imagepad_frame_list(
    content: List[Dict[str, Any]],
    frames: List[str],
    *,
    min_pixels: int,
    max_pixels: int,
) -> None:
    for frame in frames:
        item: Dict[str, Any] = {
            "type": "video",
            "image": frame,
            "visual_carrier": "image_pad",
        }
        if min_pixels is not None:
            item["min_pixels"] = min_pixels
        if max_pixels is not None:
            item["max_pixels"] = max_pixels
        content.append(item)


def _append_chunk_video_block(
    content: List[Dict[str, Any]],
    *,
    frames: List[str],
    chunk: int,
    current_chunk: int,
    role: str,
    min_pixels: int,
    max_pixels: int,
    imagepad_video_type: bool = False,
) -> None:
    start = chunk * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    content.append({
        "type": "text",
        "text": f"\n<VISUAL_CHUNK time=\"{int(start)}\">",
    })
    fps = float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC))
    if imagepad_video_type:
        _append_video_typed_imagepad_frame_list(
            content,
            frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        append_video_metadata_frame_list(
            content,
            frames,
            fps=fps,
            start_frame_index=chunk * FRAMES_PER_CHUNK,
            total_num_frames=(current_chunk + 1) * FRAMES_PER_CHUNK,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    content.append({"type": "text", "text": "</VISUAL_CHUNK>"})


def _append_recalled_video_blocks(
    content: List[Dict[str, Any]],
    *,
    rf: Dict[str, Any],
    base_path: Path,
    data_dir: Path,
    min_pixels: int,
    max_pixels: int,
    imagepad_video_type: bool = False,
) -> None:
    if "frame_paths" not in rf:
        return
    frame_paths = _resolve_paths(rf["frame_paths"], base_path, data_dir)
    tr0, tr1 = rf["time_range"]
    start_chunk = int(float(tr0) // float(AGENT_CHUNK_SEC))
    grouped = _group_frames_by_chunk(frame_paths, window_start=start_chunk)
    for chunk in sorted(grouped):
        start = chunk * AGENT_CHUNK_SEC
        end = start + AGENT_CHUNK_SEC
        content.append({
            "type": "text",
            "text": (
                f"\n<RECALLED_CHUNK time=\"{int(start)}\">"
            ),
        })
        fps = float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC))
        total_num_frames = max(
            int(float(tr1) * FRAMES_PER_CHUNK),
            (chunk + 1) * FRAMES_PER_CHUNK,
        )
        if imagepad_video_type:
            _append_video_typed_imagepad_frame_list(
                content,
                grouped[chunk],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            append_video_metadata_frame_list(
                content,
                grouped[chunk],
                fps=fps,
                start_frame_index=chunk * FRAMES_PER_CHUNK,
                total_num_frames=total_num_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        content.append({"type": "text", "text": "</RECALLED_CHUNK>"})


def _append_recall_result_text(content: List[Dict[str, Any]], rr: Dict[str, Any]) -> None:
    rr_json = json.dumps({
        "source": rr.get("source", ""),
        "time": rr.get("time", ""),
        "text": rr.get("text_content", rr.get("text", "")),
    }, ensure_ascii=False)
    content.append({
        "type": "text",
        "text": f"\n<recall_result>{rr_json}</recall_result>",
    })


def _build_timeline_user_content(
    sample: Dict[str, Any],
    base_path: Path,
    data_dir: Path,
    *,
    frame_rel_prefix: str,
    include_legacy_recall: bool,
    render_layout: str = RENDER_LAYOUT_TIMELINE_VIDEO,
    imagepad_video_type: bool = False,
) -> List[Dict[str, Any]]:
    inp = sample["input"]
    chunk_idx = int(sample.get("chunk_idx", 0) or 0)
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    mm = _runtime_mm_kwargs()
    content: List[Dict[str, Any]] = []

    if inp.get("user_input"):
        user_input_block = format_user_input_block(
            inp["user_input"],
            inter_chunk=inter_chunk,
        )
        if user_input_block:
            content.append({"type": "text", "text": user_input_block.lstrip("\n")})

    memory = inp.get("memory", {}) or {}
    compressed = list(memory.get("compressed_segments", memory.get("compressed", [])) or [])
    recent = [
        rec for rec in (
            _coerce_timeline_think(item)
            for item in (memory.get("recent_thinks", memory.get("recent_observations", [])) or [])
        )
        if rec is not None
    ]
    recent_by_chunk: Dict[int, List[Dict[str, Any]]] = {}
    for rec in recent:
        recent_by_chunk.setdefault(int(rec["chunk"]), []).append(rec)

    visual_by_chunk: Dict[int, List[str]] = {}
    window_start = _visual_window_start(chunk_idx)
    if not inter_chunk:
        frame_paths = _infer_visual_frame_paths(
            sample,
            data_dir,
            frame_rel_prefix=frame_rel_prefix,
        )
        frame_paths = _resolve_paths(frame_paths, base_path, data_dir)
        visual_by_chunk = _group_frames_by_chunk(frame_paths, window_start=window_start)

    summary_by_chunk: Dict[int, List[str]] = {}
    for idx, seg in enumerate(compressed, start=1):
        chunks = _segment_chunks(seg)
        start_chunk = min(chunks) if chunks else 0
        summary_by_chunk.setdefault(start_chunk, []).append(_format_summary_capsule(seg, idx))

    timeline_chunks = sorted(set(summary_by_chunk) | set(recent_by_chunk) | set(visual_by_chunk))
    in_memory_timeline = False

    def _open_memory_timeline() -> None:
        nonlocal in_memory_timeline
        if not in_memory_timeline:
            content.append({"type": "text", "text": "\n<memory>" if content else "<memory>"})
            in_memory_timeline = True

    def _close_memory_timeline() -> None:
        nonlocal in_memory_timeline
        if in_memory_timeline:
            content.append({"type": "text", "text": "\n</memory>"})
            in_memory_timeline = False

    for chunk in timeline_chunks:
        summary_capsules = summary_by_chunk.get(chunk, [])
        if summary_capsules:
            _open_memory_timeline()
            for capsule in summary_capsules:
                content.append({"type": "text", "text": f"\n{capsule}"})
        if chunk in visual_by_chunk:
            _close_memory_timeline()
            role = "current" if chunk == chunk_idx else "older_context"
            _append_chunk_video_block(
                content,
                frames=visual_by_chunk[chunk],
                chunk=chunk,
                current_chunk=chunk_idx,
                role=role,
                min_pixels=mm["min_pixels"],
                max_pixels=mm["max_pixels"],
                imagepad_video_type=imagepad_video_type,
            )
            if chunk < chunk_idx:
                recent_recs = recent_by_chunk.get(chunk, [])
                if recent_recs:
                    _open_memory_timeline()
                    for rec in recent_recs:
                        content.append({"type": "text", "text": f"\n{_format_memory_think_capsule(rec)}"})
        else:
            recent_recs = recent_by_chunk.get(chunk, [])
            if recent_recs:
                _open_memory_timeline()
                for rec in recent_recs:
                    content.append({"type": "text", "text": f"\n{_format_memory_think_capsule(rec)}"})
    _close_memory_timeline()

    queries = inp.get("queries", [])
    if queries and not inter_chunk:
        qt = format_queries_block(queries)
        if qt:
            content.append({"type": "text", "text": f"\n{qt}"})

    if include_legacy_recall and not inter_chunk:
        rf = _normalise_recalled_frames(inp, float(AGENT_CHUNK_SEC))
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
                "current_step_chunk": chunk_idx,
            })
            content.append({
                "type": "text",
                "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
            })
            _append_recalled_video_blocks(
                content,
                rf=rf,
                base_path=base_path,
                data_dir=data_dir,
                min_pixels=mm["min_pixels"],
                max_pixels=mm["max_pixels"],
                imagepad_video_type=imagepad_video_type,
            )
        if inp.get("recall_result"):
            _append_recall_result_text(content, inp["recall_result"])

    return content


def _build_timeline_video_messages(
    sample: Dict[str, Any],
    base_path: Path,
    *,
    data_dir: Path,
    render_layout: str = RENDER_LAYOUT_TIMELINE_VIDEO,
) -> List[Dict[str, Any]]:
    frame_rel_prefix = _frame_rel_prefix(data_dir)
    inp = sample["input"]
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )
    prompt_kind = (
        "post_recall"
        if (
            sample.get("sample_type") == "recall_response"
            or sample.get("sample_type") == "post_recall"
            or (inp.get("recall_result") and not is_recall_multiturn)
        )
        else None
    )
    messages: List[Dict[str, Any]] = [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": _timeline_system_prompt(
                frame_protocol="video_meta",
                prompt_kind=prompt_kind,
                inter_chunk=inter_chunk,
                render_layout=render_layout,
            ),
        }],
    }]
    messages.append({
        "role": "user",
        "content": _build_timeline_user_content(
            sample,
            base_path,
            data_dir,
            frame_rel_prefix=frame_rel_prefix,
            include_legacy_recall=not is_recall_multiturn,
            render_layout=render_layout,
            imagepad_video_type=(
                render_layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD
            ),
        ),
    })

    if is_recall_multiturn:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })
        rr = sample.get("recall_result") or inp.get("recall_result") or {}
        tool_payload: List[Dict[str, Any]] = []
        rf = _normalise_recalled_frames(inp, float(AGENT_CHUNK_SEC))
        mm = _runtime_mm_kwargs()
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
                "current_step_chunk": sample.get("chunk_idx"),
            })
            tool_payload.append({
                "type": "text",
                "text": f"<recalled_frames>{rf_header}</recalled_frames>",
            })
            _append_recalled_video_blocks(
                tool_payload,
                rf=rf,
                base_path=base_path,
                data_dir=data_dir,
                min_pixels=mm["min_pixels"],
                max_pixels=mm["max_pixels"],
                imagepad_video_type=(
                    render_layout == RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD
                ),
            )
        _append_recall_result_text(tool_payload, rr)
        messages.append({"role": "user", "content": tool_payload})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_2"]}],
        })
    else:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": _normalise_assistant_output(sample)}],
        })
    return messages


def build_messages(
    sample: Dict,
    base_path: Path,
    *,
    data_dir: Optional[Path] = None,
    frame_protocol: str = "ts_image",
    render_layout: str = RENDER_LAYOUT_STANDARD,
) -> List[Dict]:
    """Produce v12 ShareGPT messages for one sample. Stdlib-only.

    This is the canonical offline renderer. It must stay aligned with
    thinkstream.data.agent_protocol.build_user_content and the verl RL
    prompt builder: user_input, memory, queries, visual_window,
    protocol-selected visual frames, recalled frames, then recall_result.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    if render_layout in {
        RENDER_LAYOUT_TIMELINE_VIDEO,
        RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
    }:
        protocol = normalize_frame_protocol(frame_protocol)
        if protocol != "video_meta":
            raise ValueError(
                f"{render_layout} render_layout requires frame_protocol=video_meta"
            )
        return _build_timeline_video_messages(
            sample,
            base_path,
            data_dir=data_dir,
            render_layout=render_layout,
        )
    if render_layout != RENDER_LAYOUT_STANDARD:
        raise ValueError(f"Unsupported render_layout={render_layout!r}")
    frame_rel_prefix = _frame_rel_prefix(data_dir)

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = float(AGENT_CHUNK_SEC)
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )

    messages: List[Dict] = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt_for_frame_protocol(
                    frame_protocol,
                    prompt_kind=(
                        "post_recall"
                        if (
                            sample.get("sample_type") == "recall_response"
                            or sample.get("sample_type") == "post_recall"
                            or (
                                inp.get("recall_result")
                                and not is_recall_multiturn
                            )
                        )
                        else None
                    ),
                    inter_chunk=inter_chunk,
                ),
            }],
        }
    ]

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)

    user_content: List[Dict] = []

    # ── User input first ───────────────────────────────────────────────
    user_input_block = ""
    if inp.get("user_input"):
        user_input_block = format_user_input_block(
            inp["user_input"],
            inter_chunk=inter_chunk,
        )
        if user_input_block:
            user_content.append({
                "type": "text",
                "text": user_input_block.lstrip("\n"),
            })

    # ── Memory block ───────────────────────────────────────────────────
    memory_text = format_memory_block(inp.get("memory", {}))
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>" if user_content
        else f"<memory>\n{memory_text}\n</memory>",
    })

    # ── Active query + response history for that same query ─────────────
    queries = inp.get("queries", [])
    if queries and not inter_chunk:
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": f"\n{qt}"})

    # ── Visual window + frames ──────────────────────────────────────────
    # Inter-chunk compression is a text-memory action and does not consume a
    # visual timestep, so omit visual_window/images/videos on compress turns.
    if not inter_chunk:
        vw = inp["visual_window"]
        current_start = chunk_idx * chunk_sec
        current_end = current_start + chunk_sec
        vw_header = json.dumps({
            "start": vw["video_start"],
            "end": vw["video_end"],
            "frames": vw["frames"],
            "current_time": [current_start, current_end],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<visual_window>{vw_header}</visual_window>",
        })

        # Pass4 flat files may omit frame_paths — infer from video_id +
        # chunk_idx offset (NOT just frame_000001..n which would bind every
        # late chunk to video-start frames). Mirrors pass1a get_chunk_frame_paths
        # (chunk_idx × FRAMES_PER_CHUNK) so frame numbers track real video time.
        if "frame_paths" not in vw and "frames" in vw:
            vid = sample.get("video_id", "")
            if vid:
                from thinkstream.data.agent_protocol import (
                    FRAMES_PER_CHUNK as _FPC,
                    VISUAL_WINDOW_CHUNKS as _VWC,
                )
                from scripts.agent_data_v5.config import (
                    compute_visual_window_start as _cvws,
                )
                window_start = _cvws(chunk_idx, _VWC)
                paths: List[str] = []
                for ci in range(window_start, chunk_idx + 1):
                    for fi in range(_FPC):
                        fnum = ci * _FPC + fi + 1
                        paths.append(
                            f"{frame_rel_prefix}/{vid}/frame_{fnum:06d}.jpg"
                        )
                vw["frame_paths"] = paths

        if "frame_paths" in vw:
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
                VISUAL_WINDOW_CHUNKS as _VWC,
            )
            from scripts.agent_data_v5.config import (
                compute_visual_window_start as _cvws,
            )
            window_start = _cvws(chunk_idx, _VWC)
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            append_visual_frames(
                user_content,
                _resolve_paths(vw["frame_paths"], base_path, data_dir),
                frame_protocol=frame_protocol,
                fps=float(_FPC / chunk_sec),
                start_frame_index=window_start * _FPC,
                total_num_frames=(chunk_idx + 1) * _FPC,
                latest_start_frame_index=chunk_idx * _FPC,
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
            )
        elif "frame_indices" in vw and video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": vw["video_start"], "video_end": vw["video_end"],
            })
        else:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: visual_window has neither "
                f"frame_paths nor frame_indices."
            )

    # ── Recalled frames (legacy single-turn recall) ────────────────────
    if (
        "recalled_frames" in inp
        and inp["recalled_frames"]
        and not is_recall_multiturn
        and not inter_chunk
    ):
        rf = _normalise_recalled_frames(inp, chunk_sec) or inp["recalled_frames"]
        rf_header = json.dumps({
            "time_range": rf["time_range"],
            "source": rf.get("source", "historical_frames"),
            "n_frames": rf["n_frames"],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
        })
        if "frame_paths" in rf:
            tr0, tr1 = rf["time_range"]
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            append_visual_frames(
                user_content,
                _resolve_paths(rf["frame_paths"], base_path, data_dir),
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / chunk_sec),
                start_frame_index=int(tr0 * FRAMES_PER_CHUNK),
                total_num_frames=int(tr1 * FRAMES_PER_CHUNK),
                context_label="recalled frame",
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
            )
        elif video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })

    # ── Legacy single-turn recall_result (text only, no tool turn) ──────
    if inp.get("recall_result") and not is_recall_multiturn and not inter_chunk:
        rr = inp["recall_result"]
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })

    messages.append({"role": "user", "content": user_content})

    # ── Assistant turn(s) ──────────────────────────────────────────────
    if is_recall_multiturn:
        # Shape B: 2 assistant turns sandwiching a tool turn.
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })

        # Tool turn — recall_result + optional historical frames.
        # v12.11 audit-5 P0 #1 fix (2026-05-01): order MUST mirror runtime
        # (agent_loop.py:942-988): <recalled_frames> + video THEN
        # <recall_result>{...}</recall_result>. Previous order put the
        # raw recall_result JSON FIRST, then frames — train/infer drift
        # for shape-B recall second-turn answer training.
        rr = sample.get("recall_result") or inp.get("recall_result") or {}
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        tool_payload: List[Dict] = []

        rf = _normalise_recalled_frames(inp, chunk_sec)
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
            })
            tool_payload.append({
                "type": "text",
                "text": f"<recalled_frames>{rf_header}</recalled_frames>",
            })
            if "frame_paths" in rf:
                from scripts.agent_data_v5.config import (
                    AGENT_CHUNK_SEC as _CHUNK_SEC,
                )
                tr_start, tr_end = rf["time_range"]
                try:
                    from scripts.agent_data_v5.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
                tr_start_chunk = int(tr_start / float(_CHUNK_SEC))
                append_visual_frames(
                    tool_payload,
                    _resolve_paths(rf["frame_paths"], base_path, data_dir),
                    frame_protocol=frame_protocol,
                    fps=float(FRAMES_PER_CHUNK / float(_CHUNK_SEC)),
                    start_frame_index=tr_start_chunk * FRAMES_PER_CHUNK,
                    total_num_frames=int(tr_end / float(_CHUNK_SEC)) * FRAMES_PER_CHUNK,
                    context_label="recalled frame",
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )
            elif video_path:
                tool_payload.append({
                    "type": "video", "video": video_path,
                    "video_start": rf["time_range"][0],
                    "video_end": rf["time_range"][1],
                })

        # v12.11 audit-5 P0 #1: append <recall_result> AFTER frames so the
        # token stream matches runtime: [<recalled_frames>{...}, video,
        # <recall_result>{...}</recall_result>].
        tool_payload.append({
            "type": "text",
            "text": f"<recall_result>{rr_json}</recall_result>",
        })

        # DeepEyesV2-aligned ShareGPT has no `tool` role — inject as user content.
        # Qwen3-VL chat_template would otherwise nest <tool_response> under
        # <|im_start|>user, so the on-the-wire token stream is identical.
        messages.append({"role": "user", "content": tool_payload})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_2"]}],
        })
    else:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": _normalise_assistant_output(sample)}],
        })

    return messages


# ---------------------------------------------------------------------------
# IO + driver
# ---------------------------------------------------------------------------

def _iter_flat(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_trajectories(path: Path) -> Iterable[Dict]:
    """Yield each sample inside trajectory rows, propagating top-level fields."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            video_id = traj.get("video_id", "")
            video_path = traj.get("video_path", "")
            traj_id = traj.get("trajectory_id", "")
            for s in traj.get("samples", []):
                s.setdefault("video_id", video_id)
                s.setdefault("video_path", video_path)
                s.setdefault("trajectory_id", traj_id)
                yield s


def _emit_row(sample: Dict, messages: List[Dict], *, frame_protocol: str) -> Dict:
    # v12.12 fix (P0-5): propagate verification verdict + metadata so SFT
    # data_processor can filter / downweight failed samples. pipeline.py
    # tags every sample via pass3e with verification.passed/.fail_reasons
    # but keeps all samples in the trajectory; the consumer (SFT loader)
    # is responsible for the actual drop policy.
    verification = sample.get("verification") or {}
    tool_schema_mode = "compress" if sample.get("v12_inter_chunk") else "streaming"
    return {
        "trajectory_id": sample.get("trajectory_id", ""),
        "video_id": sample.get("video_id", ""),
        "chunk_idx": sample.get("chunk_idx", -1),
        "sample_type": sample.get("sample_type", ""),
        "sample_id": sample.get("sample_id", ""),
        "frame_protocol": frame_protocol,
        "v12_inter_chunk": bool(sample.get("v12_inter_chunk", False)),
        "tool_schema_mode": tool_schema_mode,
        "messages": messages,
        "videos": None,
        "verification": {
            "passed": bool(verification.get("passed", True)),
            "fail_reasons": list(verification.get("fail_reasons") or []),
        },
        # Forward metadata so consumers can compute reward / question lookup
        # without re-running render_samples. (pass4 already reads this; SFT
        # filter consults verification.passed.)
        "metadata": sample.get("metadata") or {},
    }


def _with_sft_turn_policy(
    row: Dict[str, Any],
    *,
    tool_schema_mode: str,
    loss_assistant_turns: str = "all",
    sft_subtype: str = "",
    loss_class: str = "",
    sample_id_suffix: str = "",
) -> Dict[str, Any]:
    """Attach row-local tool schema + label-mask policy metadata."""
    row["tool_schema_mode"] = tool_schema_mode
    row["loss_assistant_turns"] = loss_assistant_turns
    if not loss_class:
        subtype = str(sft_subtype or row.get("sft_subtype") or "").strip().lower()
        if "recall_query" in subtype:
            loss_class = "recall"
        elif "post_recall" in subtype or subtype in {"recall_answer", "recall_response"}:
            loss_class = "post_recall"
        else:
            loss_class = str(row.get("sample_type") or "")
    if loss_class:
        row["loss_class"] = loss_class
    if sft_subtype:
        row["sft_subtype"] = sft_subtype
        meta = dict(row.get("metadata") or {})
        meta["sft_subtype"] = sft_subtype
        meta["loss_class"] = loss_class
        meta["loss_assistant_turns"] = loss_assistant_turns
        row["metadata"] = meta
    if sample_id_suffix:
        sid = str(row.get("sample_id") or "")
        if sid:
            row["sample_id"] = f"{sid}:{sample_id_suffix}"
    return row


def _recall_action_think_for_sample(sample: Dict, visual_think: str = "") -> str:
    """Action-aware first-turn think for recall_query SFT rows."""
    base = str(visual_think or "").strip()
    low = base.lower()
    if "visible evidence" in low and "recall" in low:
        return base
    if sample.get("action") == "silent" or sample.get("base_role") == "recall_silent":
        decision = (
            "Current visible evidence is insufficient to "
            "answer the active query. The answer may not have appeared yet, "
            "so I will recall elapsed history once and stay silent if still "
            "unsupported."
        )
    else:
        decision = (
            "Current visible evidence is insufficient to answer the active "
            "query because the needed evidence is historical, so I will "
            "recall the earlier window rather than guess."
        )
    return f"{base} {decision}".strip()


def _extract_first_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(
        r"<think>(.*?)</think>",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _replace_first_think(text: str, think: str) -> str:
    if not isinstance(text, str):
        return text
    replaced, n = re.subn(
        r"<think>.*?</think>",
        f"<think>{think}</think>",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return replaced if n else text


def _rewrite_recall_query_turn1_think(sample: Dict, messages: List[Dict]) -> None:
    """Rewrite the first recall assistant turn in-place for old bank samples.

    Existing trajectory banks may carry pass2's question-blind visual think in
    v12_assistant_turn_1. Rewriting at pass5 render time lets us regenerate
    better SFT/DAgger messages without rerunning pass3.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or "text" not in item:
                continue
            text = item.get("text") or ""
            if '"name": "recall"' in text or '"name":"recall"' in text:
                think = _recall_action_think_for_sample(
                    sample,
                    _extract_first_think(text),
                )
                item["text"] = _replace_first_think(text, think)
                return


def _set_system_prompt_kind(
    messages: List[Dict],
    *,
    frame_protocol: str,
    prompt_kind: str,
    render_layout: str = RENDER_LAYOUT_STANDARD,
) -> None:
    """Replace the first system prompt in-place for turn-local prompt kinds."""
    if not messages or messages[0].get("role") != "system":
        return
    if render_layout in {
        RENDER_LAYOUT_TIMELINE_VIDEO,
        RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
    }:
        prompt = _timeline_system_prompt(
            frame_protocol=frame_protocol,
            prompt_kind=prompt_kind,
            inter_chunk=False,
            render_layout=render_layout,
        )
    else:
        prompt = system_prompt_for_frame_protocol(
            frame_protocol,
            prompt_kind=prompt_kind,
        )
    content = messages[0].get("content")
    if isinstance(content, list) and content:
        if isinstance(content[0], dict):
            content[0]["text"] = prompt
            return
    messages[0]["content"] = [{"type": "text", "text": prompt}]


def _is_recall_multiturn_messages(sample: Dict, messages: List[Dict]) -> bool:
    return (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
        and "v12_assistant_turn_2" in sample
        and len(messages) >= 5
        and messages[2].get("role") == "assistant"
        and messages[3].get("role") == "user"
        and messages[4].get("role") == "assistant"
    )


def _is_post_recall_single_turn(sample: Dict) -> bool:
    """Rows that already represent the no-tools answer turn after recall."""
    stype = str(sample.get("sample_type") or "").strip().lower()
    if stype in {"post_recall", "recall_response", "recall_answer"}:
        return True
    subtype = str(sample.get("sft_subtype") or "").strip().lower()
    if "post_recall" in subtype or subtype in {"recall_response", "recall_answer"}:
        return True
    inp = sample.get("input") or {}
    return bool(inp.get("recall_result")) and stype != "recall"


def build_sft_rows(
    sample: Dict,
    messages: List[Dict],
    *,
    frame_protocol: str,
    render_layout: str = RENDER_LAYOUT_STANDARD,
) -> List[Dict]:
    """Return one or more runtime-aligned SFT rows for a rendered sample.

    Qwen's chat template renders tools at the conversation level for a single
    apply_chat_template call. A multi-turn recall row therefore cannot train
    both the recall tool call and the post-recall answer in one sample without
    leaking recall tools into the second assistant turn. Split it:
      - recall_query: first assistant turn only, recall schema available;
      - post_recall: full prefix including recall_result, no tool schema,
        loss only on the final assistant decision turn.
    """
    if _is_recall_multiturn_messages(sample, messages):
        aligned_messages = deepcopy(messages)
        _rewrite_recall_query_turn1_think(sample, aligned_messages)

        first_messages = deepcopy(aligned_messages[:3])
        first_row = _emit_row(sample, first_messages, frame_protocol=frame_protocol)
        _with_sft_turn_policy(
            first_row,
            tool_schema_mode="streaming",
            loss_assistant_turns="all",
            sft_subtype="recall_query",
            loss_class="recall",
            sample_id_suffix="recall_query",
        )

        second_messages = deepcopy(aligned_messages)
        _set_system_prompt_kind(
            second_messages,
            frame_protocol=frame_protocol,
            prompt_kind="post_recall",
            render_layout=render_layout,
        )
        second_row = _emit_row(sample, second_messages, frame_protocol=frame_protocol)
        _with_sft_turn_policy(
            second_row,
            tool_schema_mode="post_recall",
            loss_assistant_turns="last",
            sft_subtype="post_recall",
            loss_class="post_recall",
            sample_id_suffix="post_recall",
        )
        return [first_row, second_row]

    row = _emit_row(sample, deepcopy(messages), frame_protocol=frame_protocol)
    if _is_post_recall_single_turn(sample):
        tool_schema_mode = "post_recall"
        sft_subtype = "post_recall"
    elif sample.get("v12_inter_chunk"):
        tool_schema_mode = "compress"
        sft_subtype = str(sample.get("sample_type") or "")
    else:
        tool_schema_mode = "streaming"
        sft_subtype = str(sample.get("sample_type") or "")
    _with_sft_turn_policy(
        row,
        tool_schema_mode=tool_schema_mode,
        loss_assistant_turns="all",
        sft_subtype=sft_subtype,
    )
    return [row]


def _sample_rank(sample: Dict, idx: int) -> str:
    key = "|".join([
        str(sample.get("video_id", "")),
        str(sample.get("trajectory_id", "")),
        str(sample.get("sample_id", "")),
        str(sample.get("chunk_idx", "")),
        str(idx),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _is_active_silent(sample: Dict) -> bool:
    if sample.get("sample_type") != "silent" or sample.get("action") != "silent":
        return False
    meta = sample.get("metadata") or {}
    if sample.get("card_id") or meta.get("question"):
        return True
    return sample.get("sequence_type") in {
        "event_watch",
        "multi_response",
        "recall_success",
        "immediate_response",
        "memory_response",
    }


def _choose_ranked(items: List[tuple[int, Dict]], n: int) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    return sorted(items, key=lambda x: _sample_rank(x[1], x[0]))[:n]


def _is_multi_emit_response(sample: Dict) -> bool:
    if sample.get("sample_type") != "response":
        return False
    meta = sample.get("metadata") or {}
    return (
        meta.get("question_type") == "multi_emit"
        or meta.get("family") in {"F5", "F7", "PN1"}
    )


def _choose_multi_emit_response(
    items: List[tuple[int, Dict]],
    n: int,
) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    by_family: Dict[str, List[tuple[int, Dict]]] = {}
    for item in items:
        meta = item[1].get("metadata") or {}
        fam = meta.get("family") or "unknown"
        by_family.setdefault(fam, []).append(item)
    for fam in by_family:
        by_family[fam] = _choose_ranked(by_family[fam], len(by_family[fam]))

    selected: List[tuple[int, Dict]] = []
    cursors = {fam: 0 for fam in by_family}
    families = sorted(by_family)
    while len(selected) < n:
        progressed = False
        for fam in families:
            cur = cursors[fam]
            bucket = by_family[fam]
            if cur >= len(bucket):
                continue
            selected.append(bucket[cur])
            cursors[fam] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def balance_sft_samples(samples: List[Dict]) -> tuple[List[Dict], Dict[str, int]]:
    """Balance only the SFT messages split.

    Policy:
      - keep every recall / compress row;
      - keep ordinary response rows;
      - cap multi-emit response rows so F5/PN1 do not dominate SFT;
      - keep enough silent rows to make silent roughly 55-60% of SFT;
      - prefer active query-bearing silent rows over patrol/background rows.
    """
    indexed = list(enumerate(samples))
    recall_compress = [
        (i, s) for i, s in indexed
        if s.get("sample_type") in {"recall", "compress"}
    ]
    multi_emit_response = [
        (i, s) for i, s in indexed if _is_multi_emit_response(s)
    ]
    ordinary_response = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "response" and not _is_multi_emit_response(s)
    ]
    other_active = [
        (i, s) for i, s in indexed
        if s.get("sample_type") not in {"silent", "response", "recall", "compress"}
    ]
    multi_limit = min(
        len(multi_emit_response),
        max(1, int(len(ordinary_response) * SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO)),
    )
    active = (
        recall_compress
        + ordinary_response
        + other_active
        + _choose_multi_emit_response(multi_emit_response, multi_limit)
    )
    active_silent = [(i, s) for i, s in indexed if _is_active_silent(s)]
    base_silent = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "silent" and not _is_active_silent(s)
    ]
    if not active:
        return samples, {"before": len(samples), "after": len(samples)}

    target_silent = min(
        len(active_silent) + len(base_silent),
        max(1, int(len(active) * SFT_SILENT_TO_ACTIVE_RATIO)),
    )
    target_active_silent = min(
        len(active_silent),
        int(target_silent * SFT_ACTIVE_SILENT_FRACTION),
    )
    kept_silent = _choose_ranked(active_silent, target_active_silent)
    remaining = target_silent - len(kept_silent)
    if remaining > 0:
        kept_silent.extend(_choose_ranked(base_silent, remaining))
    if len(kept_silent) < target_silent:
        used = {i for i, _s in kept_silent}
        rest = [(i, s) for i, s in active_silent if i not in used]
        kept_silent.extend(_choose_ranked(rest, target_silent - len(kept_silent)))

    selected = active + kept_silent
    selected.sort(key=lambda x: x[0])
    out = [s for _i, s in selected]
    return out, {
        "before": len(samples),
        "after": len(out),
        "active_kept": len(active),
        "ordinary_response_kept": len(ordinary_response),
        "multi_emit_response_before": len(multi_emit_response),
        "multi_emit_response_kept": min(len(multi_emit_response), multi_limit),
        "recall_compress_kept": len(recall_compress),
        "silent_before": len(active_silent) + len(base_silent),
        "silent_kept": len(kept_silent),
        "active_silent_kept": sum(1 for _i, s in kept_silent if _is_active_silent(s)),
        "base_silent_kept": sum(1 for _i, s in kept_silent if not _is_active_silent(s)),
    }


def convert(
    src: Path,
    dst: Path,
    *,
    is_trajectory: bool,
    base_path: Path,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    balance_sft: bool = False,
    frame_protocol: str = "ts_image",
    render_layout: str = RENDER_LAYOUT_STANDARD,
) -> Dict[str, int]:
    data_dir = data_dir or DEFAULT_DATA_DIR
    iter_fn = _iter_trajectories if is_trajectory else _iter_flat
    counts = {"ok": 0, "failed": 0}
    by_type: Dict[str, int] = {}
    balance_stats: Dict[str, int] = {}
    sample_iter: Iterable[Dict]
    if balance_sft:
        materialized = list(iter_fn(src))
        materialized, balance_stats = balance_sft_samples(materialized)
        sample_iter = materialized
    else:
        sample_iter = iter_fn(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as out:
        for i, sample in enumerate(sample_iter):
            if limit and counts["ok"] >= limit:
                break
            try:
                messages = build_messages(
                    sample,
                    base_path,
                    data_dir=data_dir,
                    frame_protocol=frame_protocol,
                    render_layout=render_layout,
                )
            except (KeyError, ValueError) as exc:
                counts["failed"] += 1
                if counts["failed"] <= 5:
                    sid = sample.get("sample_id") or sample.get("trajectory_id") or i
                    logger.warning(f"[{src.name}] sample {sid} skipped: {exc}")
                continue

            for row in build_sft_rows(
                sample,
                messages,
                frame_protocol=frame_protocol,
                render_layout=render_layout,
            ):
                if limit and counts["ok"] >= limit:
                    break
                row["render_layout"] = render_layout
                meta = dict(row.get("metadata") or {})
                meta["render_layout"] = render_layout
                row["metadata"] = meta
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["ok"] += 1
                by_type[row["sample_type"]] = by_type.get(row["sample_type"], 0) + 1

    counts["by_type"] = by_type
    if balance_stats:
        counts["balance"] = balance_stats
    return counts


def write_dataset_info(out_dir: Path, splits_done: List[str]) -> None:
    """LLaMA-Factory entry stub. Aligns with DeepEyesV2 (system/user/assistant only)."""
    entries = {}
    for stem in splits_done:
        entries[f"thinkstream_{stem}"] = {
            "file_name": f"{stem}_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", choices=["flat", "traj", "auto"], default="traj",
        help=(
            "'flat' = *_full.jsonl single-step rows, "
            "'traj' = *_trajectories.jsonl (default — matches main pipeline), "
            "'auto' = prefer flat (legacy; can produce data inconsistent with "
            "the main `pipeline.py` invocation which forces --input traj). "
            "v12.11 audit-5 P1 #4 fix (2026-05-01): default flipped from auto "
            "→ traj to match pipeline.py:1266."
        ),
    )
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory for rendered *_messages.jsonl outputs. Defaults to "
            "--final-dir. Use this to render AB variants from one canonical "
            "trajectory set, e.g. final/rendered/ts_image and "
            "final/rendered/video_meta."
        ),
    )
    parser.add_argument(
        "--frame-protocol",
        default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "ts_image"),
        choices=["ts_image", "video_meta"],
        help=(
            "Student/eval visual carrier. ts_image = timestamp text + image "
            "items; video_meta = pre-extracted frame list as Qwen video block "
            "with video_metadata. Teacher pass caches are unchanged."
        ),
    )
    parser.add_argument(
        "--render-layout",
        default=os.environ.get("THINKSTREAM_RENDER_LAYOUT", RENDER_LAYOUT_STANDARD),
        choices=[
            RENDER_LAYOUT_STANDARD,
            RENDER_LAYOUT_TIMELINE_VIDEO,
            RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
        ],
        help=(
            "User payload layout. standard keeps the existing block order; "
            "timeline_video keeps frame_protocol=video_meta but splits extracted "
            "frames into time-ordered type=video chunk blocks and places summary/"
            "memory/recall tags around that timeline. timeline_video_imagepad "
            "uses type=video items with image-pad frame carriers plus timestamp "
            "text instead of Qwen video_metadata blocks."
        ),
    )
    parser.add_argument("--base-path", default=str(PROJECT_ROOT),
                        help="Project root for resolving relative video/frame paths. "
                        "Generated samples store frame paths relative to the repo "
                        "or absolute paths under the batch root. The batch root "
                        "for frame lookup is inferred from --final-dir.")
    parser.add_argument("--limit", type=int, default=0, help="Per-split sample cap (0 = unlimited).")
    parser.add_argument("--no-balance-sft", action="store_true",
                        help="Disable train_sft_messages silent downsampling.")
    args = parser.parse_args()

    final_dir = _resolve_cli_path(args.final_dir)
    output_dir = _resolve_cli_path(args.output_dir) if args.output_dir else final_dir
    base_path = _resolve_cli_path(args.base_path)
    data_dir = final_dir.parent if final_dir.name == "final" else DEFAULT_DATA_DIR
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = str(args.render_layout or RENDER_LAYOUT_STANDARD)
    if (
        render_layout in {
            RENDER_LAYOUT_TIMELINE_VIDEO,
            RENDER_LAYOUT_TIMELINE_VIDEO_IMAGEPAD,
        }
        and frame_protocol != "video_meta"
    ):
        raise SystemExit(
            f"--render-layout {render_layout} requires --frame-protocol video_meta"
        )
    if not final_dir.exists():
        raise SystemExit(f"final dir not found: {final_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_done: List[str] = []
    for flat_stem, traj_stem, out_stem in SPLITS:
        flat_path = final_dir / f"{flat_stem}.jsonl"
        traj_path = final_dir / f"{traj_stem}.jsonl"

        if args.input == "flat":
            src, is_traj = flat_path, False
        elif args.input == "traj":
            src, is_traj = traj_path, True
        else:
            if flat_path.exists():
                src, is_traj = flat_path, False
            elif traj_path.exists():
                src, is_traj = traj_path, True
            else:
                logger.warning(f"No input found for split {flat_stem}/{traj_stem}, skipping.")
                continue

        if not src.exists():
            logger.warning(f"Input missing: {src}, skipping.")
            continue

        dst = output_dir / f"{out_stem}.jsonl"
        logger.info(
            f"Converting {src.name} → {dst} "
            f"(is_trajectory={is_traj}, frame_protocol={frame_protocol}, "
            f"render_layout={render_layout})"
        )
        balance = out_stem == "train_sft_messages" and not args.no_balance_sft
        counts = convert(src, dst, is_trajectory=is_traj, base_path=base_path,
                         data_dir=data_dir,
                         limit=args.limit or None, balance_sft=balance,
                         frame_protocol=frame_protocol,
                         render_layout=render_layout)
        logger.info(
            f"  ok={counts['ok']} failed={counts['failed']} by_type={counts['by_type']}"
        )
        if counts.get("balance"):
            logger.info(f"  SFT balance: {counts['balance']}")
        splits_done.append(out_stem.replace("_messages", ""))

    if splits_done:
        write_dataset_info(output_dir, splits_done)
        (output_dir / "render_manifest.json").write_text(json.dumps({
            "generated_by": "pass5_messages.py",
            "source_final_dir": str(final_dir),
            "output_dir": str(output_dir),
            "frame_protocol": frame_protocol,
            "render_layout": render_layout,
            "splits": splits_done,
        }, ensure_ascii=False, indent=2))
        logger.info(f"Wrote dataset_info.json → {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
