"""PASS 5 — Convert single-step samples to LLaMA-Factory ShareGPT messages format.

Reads pass4 outputs and emits one row per sample in the multi-turn messages
format used by LLaMA-Factory / DeepEyesV2 / VST. Each row is a stand-alone
training sample matching fresh-KV-per-chunk inference: every sample's
user.content carries the full state for ordinary visual turns (user_input +
memory + visual_window + active_query; recalled frames are only in the recall
tool-response turn) so the model trains under
the exact same input distribution it sees at inference.

Three sample shapes preserved (canonical pass/SFT/RL/eval video_meta protocol):
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
    RENDER_LAYOUT_STANDARD_QUERY_LAST,
    format_memory_block,
    format_queries_block,
    format_user_input_block,
    append_visual_frames,
    build_recalled_frames_metadata,
    build_recall_result_metadata,
    canonical_answer_instruction,
    normalize_frame_protocol,
    prompt_time_range,
    prompt_time_value,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
    user_input_is_active_query_duplicate,
)

logger = logging.getLogger(__name__)

RENDER_LAYOUT_QUERY_LAST = RENDER_LAYOUT_STANDARD_QUERY_LAST
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
ACTIVE_QUERY_BLOCK_RE = re.compile(
    r"<active_query>\s*(.*?)\s*</active_query>",
    flags=re.DOTALL,
)
RESPONSE_HISTORY_BLOCK_RE = re.compile(
    r"<response_history>\s*(.*?)\s*</response_history>",
    flags=re.DOTALL,
)
USER_INPUT_BLOCK_RE = re.compile(
    r"<user_input>\s*(.*?)\s*</user_input>",
    flags=re.DOTALL,
)
QUESTION_LINE_RE = re.compile(r"^\s*(?:\[[^\]\n]+s\]\s+)?Q:\s*(.*?)\s*$", re.MULTILINE)
OPTIONS_LINE_RE = re.compile(r"^\s*(?:\[[^\]\n]+s\]\s+)?Options:\s*(.*?)\s*$", re.MULTILINE)
ANSWER_FORMAT_LINE_RE = re.compile(
    r"^\s*(?:\[[^\]\n]+s\]\s+)?Answer format:\s*(.*?)\s*$",
    re.MULTILINE,
)
OPTION_LABEL_RE = re.compile(
    r"^\s*(?:\(([A-Z])\)|([A-Z])[\).:])\s*(.*)$",
    flags=re.IGNORECASE,
)
OPTION_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class QueryRenderContractError(ValueError):
    """Raised when query/options/answer-format rendering is missing or duplicated."""

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
SFT_SILENT_TO_ACTIVE_RATIO = 0.90
SFT_PENDING_SILENT_FRACTION = 0.55
SFT_POST_ANSWER_SILENT_FRACTION = 0.25
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


def _strip_option_label(text: str) -> str:
    match = OPTION_LABEL_RE.match(str(text or ""))
    return (match.group(3) if match else str(text or "")).strip()


def _mc_letter_text(question: Dict[str, Any]) -> tuple[str, str]:
    options = list(question.get("options") or [])
    correct = str(question.get("correct_option") or "").strip().upper()
    if correct in OPTION_LETTERS and len(options) >= OPTION_LETTERS.index(correct) + 1:
        return correct, _strip_option_label(options[OPTION_LETTERS.index(correct)])
    text = (
        question.get("correct_answer_text")
        or question.get("canonical_answer")
        or question.get("gold_answer")
        or ""
    )
    return correct, _strip_option_label(str(text))


def _accepted_answers_for_question(question: Dict[str, Any]) -> List[str]:
    if question.get("answer_form") != "multiple_choice":
        gold = str(question.get("canonical_answer") or question.get("gold_answer") or "").strip()
        return [gold] if gold else []
    letter, text = _mc_letter_text(question)
    values = []
    if letter:
        values.append(letter)
    if letter and text:
        values.append(f"{letter}) {text}")
    if text:
        values.append(text)
    seen = set()
    return [v for v in values if v and not (v.lower() in seen or seen.add(v.lower()))]


def _per_emit_target(question: Dict[str, Any], chunk_idx: Any) -> str:
    try:
        current = int(chunk_idx)
    except Exception:
        current = None
    for emit in question.get("per_emit_answers") or []:
        if not isinstance(emit, dict):
            continue
        try:
            emit_chunk = int(emit.get("chunk"))
        except Exception:
            emit_chunk = None
        if current is None or emit_chunk == current:
            value = str(emit.get("value") or "").strip()
            if value:
                return value
    return ""


def _mc_target_for_question(question: Dict[str, Any], chunk_idx: Any) -> str:
    letter, text = _mc_letter_text(question)
    return letter or text


def _canonical_answer_target(sample: Dict[str, Any]) -> str:
    question = sample.get("_trajectory_question") or sample.get("metadata") or {}
    if question.get("answer_form") == "multiple_choice":
        return _mc_target_for_question(question, sample.get("chunk_idx"))
    return _per_emit_target(question, sample.get("chunk_idx"))


def _replace_answer_target(text: str, target: str) -> str:
    if not target:
        return text
    match = ANSWER_BLOCK_RE.search(text)
    if not match or not match.group(1).strip():
        return text
    return ANSWER_BLOCK_RE.sub(f"<answer>{target}</answer>", text, count=1)


def _normalise_assistant_output(sample: Dict, output: Optional[str] = None) -> str:
    output = str(sample.get("output", "") if output is None else output)
    if sample.get("sample_type") != "compress":
        return _replace_answer_target(output, _canonical_answer_target(sample))
    think = _compress_management_think_from_output(output)
    replacement = f"<think>{think}</think>"
    if re.search(r"<think>.*?</think>", output, flags=re.DOTALL):
        output = re.sub(
            r"<think>.*?</think>",
            replacement,
            output,
            count=1,
            flags=re.DOTALL,
        )
    else:
        output = replacement + output
    return _replace_answer_target(output, _canonical_answer_target(sample))


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


def build_messages(
    sample: Dict,
    base_path: Path,
    *,
    data_dir: Optional[Path] = None,
    frame_protocol: str = "video_meta",
    render_layout: str = RENDER_LAYOUT_QUERY_LAST,
) -> List[Dict]:
    """Produce v12 ShareGPT messages for one sample. Stdlib-only.

    This is the canonical offline renderer. It must stay aligned with
    thinkstream.data.agent_protocol.build_user_content and the verl RL
    prompt builder. Rows render user_input, memory, visual_window/video_meta,
    then active_query/response_history.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    if render_layout != RENDER_LAYOUT_QUERY_LAST:
        raise ValueError(f"Unsupported render_layout={render_layout!r}")
    frame_rel_prefix = _frame_rel_prefix(data_dir)

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = AGENT_CHUNK_SEC
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )
    sample_type = str(sample.get("sample_type") or "").strip().lower()
    explicit_post_recall = sample_type in {
        "post_recall",
        "recall_response",
        "recall_answer",
    }
    legacy_post_recall = (
        explicit_post_recall
        or (bool(inp.get("recall_result")) and not is_recall_multiturn and sample_type != "recall")
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
                            legacy_post_recall
                        )
                        else None
                    ),
                    inter_chunk=inter_chunk,
                    render_layout=render_layout,
                ),
            }],
        }
    ]

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)

    user_content: List[Dict] = []

    # ── Legacy standalone post-recall row ──────────────────────────────
    # Canonical recall samples are shape B:
    #   user(current chunk + active_query) → assistant(recall) →
    #   user(recalled frames + metadata) → assistant(answer)
    # Older pass3 rows can already be the final post-recall answer without
    # the first two turns. In that case render only the active query plus
    # recall evidence; never add a fresh current visual window under the
    # post-recall system prompt.
    if legacy_post_recall and not is_recall_multiturn and not inter_chunk:
        queries = inp.get("queries", [])
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": qt})

        rf = _normalise_recalled_frames(inp, chunk_sec) or inp.get("recalled_frames")
        if rf:
            rf_header = json.dumps({
                "time_range": prompt_time_range(rf["time_range"]),
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
                    "video_start": prompt_time_value(rf["time_range"][0]),
                    "video_end": prompt_time_value(rf["time_range"][1]),
                })

        rr_json = json.dumps(
            build_recall_result_metadata(inp.get("recall_result") or {}, rf),
            ensure_ascii=False,
        )
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": _normalise_assistant_output(sample)}],
        })
        return messages

    # ── User input first ───────────────────────────────────────────────
    raw_user_input = inp.get("user_input", "")
    if user_input_is_active_query_duplicate(
        raw_user_input,
        inp.get("queries", []),
        inter_chunk=inter_chunk,
    ):
        raw_user_input = ""
    user_input_block = format_user_input_block(
        raw_user_input,
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

    query_last = render_layout == RENDER_LAYOUT_QUERY_LAST

    # ── Active query + response history for that same query ─────────────
    queries = inp.get("queries", [])
    if queries and not inter_chunk and not query_last:
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
            "start": prompt_time_value(vw["video_start"]),
            "end": prompt_time_value(vw["video_end"]),
            "frames": vw["frames"],
            "current_time": prompt_time_value(current_start),
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
                "video_start": prompt_time_value(vw["video_start"]),
                "video_end": prompt_time_value(vw["video_end"]),
            })
        else:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: visual_window has neither "
                f"frame_paths nor frame_indices."
            )

    if queries and not inter_chunk and query_last:
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": f"\n{qt}"})

    # ── Recalled frames (legacy single-turn recall) ────────────────────
    if (
        "recalled_frames" in inp
        and inp["recalled_frames"]
        and not is_recall_multiturn
        and not inter_chunk
    ):
        rf = _normalise_recalled_frames(inp, chunk_sec) or inp["recalled_frames"]
        rf_header = json.dumps({
            "time_range": prompt_time_range(rf["time_range"]),
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
                "video_start": prompt_time_value(rf["time_range"][0]),
                "video_end": prompt_time_value(rf["time_range"][1]),
            })

    # ── Legacy single-turn recall_result metadata (no text evidence) ────
    if inp.get("recall_result") and not is_recall_multiturn and not inter_chunk:
        rr = inp["recall_result"]
        rr_json = json.dumps(
            build_recall_result_metadata(
                rr,
                _normalise_recalled_frames(inp, chunk_sec),
            ),
            ensure_ascii=False,
        )
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
        tool_payload: List[Dict] = []

        rf = _normalise_recalled_frames(inp, chunk_sec)
        if rf:
            rf_header = json.dumps({
                "time_range": prompt_time_range(rf["time_range"]),
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
                    "video_start": prompt_time_value(rf["time_range"][0]),
                    "video_end": prompt_time_value(rf["time_range"][1]),
                })

        # Append metadata-only <recall_result> AFTER frames. Retrieved text is
        # intentionally hidden from the model; the visual frames are evidence.
        rr_json = json.dumps(
            build_recall_result_metadata(rr, rf),
            ensure_ascii=False,
        )
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
            "content": [{
                "type": "text",
                "text": _normalise_assistant_output(
                    sample, sample["v12_assistant_turn_2"]
                ),
            }],
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
            questions_by_card = {
                q.get("card_id"): q
                for q in traj.get("questions", [])
                if isinstance(q, dict) and q.get("card_id")
            }
            for s in traj.get("samples", []):
                s.setdefault("video_id", video_id)
                s.setdefault("video_path", video_path)
                s.setdefault("trajectory_id", traj_id)
                card_id = s.get("card_id") or (s.get("metadata") or {}).get("card_id")
                q = questions_by_card.get(card_id)
                if q:
                    s["_trajectory_question"] = q
                    meta = dict(s.get("metadata") or {})
                    for key in (
                        "card_id",
                        "question",
                        "options",
                        "correct_option",
                        "answer_form",
                        "answer_style",
                        "answer_instruction",
                        "question_type",
                        "family",
                        "family_name",
                        "category",
                        "skill",
                        "ours_unique",
                        "availability",
                        "support_chunks",
                        "gold_compress_chunks",
                        "ask_chunk",
                        "per_emit_answers",
                    ):
                        if key in q:
                            value = q.get(key)
                            meta[key] = list(value) if isinstance(value, list) else value
                    if q.get("answer_form") == "multiple_choice":
                        meta["answer_style"] = "letter_only"
                    instruction = canonical_answer_instruction(meta)
                    if instruction:
                        meta["answer_instruction"] = instruction
                    if q.get("answer_form") == "multiple_choice":
                        _letter, correct_text = _mc_letter_text(q)
                        meta["correct_answer_text"] = correct_text
                        meta["canonical_answer"] = correct_text or q.get("canonical_answer", "")
                        meta["gold_answer"] = correct_text or q.get("gold_answer", "")
                        meta["accepted_answers"] = _accepted_answers_for_question(q)
                        target = _mc_target_for_question(q, s.get("chunk_idx"))
                        if target:
                            meta["sft_answer"] = target
                    s["metadata"] = meta
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
        "card_id": sample.get("card_id", ""),
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
    better SFT messages without rerunning pass3.
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
    render_layout: str = RENDER_LAYOUT_QUERY_LAST,
) -> None:
    """Replace the first system prompt in-place for turn-local prompt kinds."""
    if not messages or messages[0].get("role") != "system":
        return
    prompt = system_prompt_for_frame_protocol(
        frame_protocol,
        prompt_kind=prompt_kind,
        render_layout=render_layout,
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


def _message_text(messages: List[Dict], *, roles: Optional[set[str]] = None) -> str:
    parts: List[str] = []
    for msg in messages:
        if roles is not None and str(msg.get("role") or "") not in roles:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


def _query_sort_key(q: Dict[str, Any], idx: int) -> tuple[float, int]:
    for key in ("ask_time", "time", "timestamp"):
        try:
            return float(q.get(key, 0)), idx
        except (TypeError, ValueError):
            continue
    return 0.0, idx


def _selected_open_query(queries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates: List[tuple[int, Dict[str, Any]]] = []
    for i, q in enumerate(queries or []):
        if not isinstance(q, dict):
            continue
        status = str(q.get("status", "") or "").strip().lower()
        if status in {"open", "pending", "active"} or (
            not status and not q.get("answers")
        ):
            candidates.append((i, q))
    if not candidates:
        return None
    return max(candidates, key=lambda item: _query_sort_key(item[1], item[0]))[1]


def _normalise_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _answer_format_body(instruction: str) -> str:
    text = str(instruction or "").strip()
    if text.lower().startswith("answer format:"):
        text = text.split(":", 1)[1].strip()
    return text


def validate_query_render_contract(sample: Dict, messages: List[Dict]) -> None:
    """Hard-check active_query rendering in final ShareGPT messages.

    This catches the two failure modes that are otherwise easy to miss:
    silently losing the options/answer-format line, or rendering them more than
    once after pass4/pass5/rebalance transformations.
    """
    inp = sample.get("input") or {}
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    queries = list(inp.get("queries") or [])
    expected_query = None if inter_chunk else _selected_open_query(queries)
    user_text = _message_text(messages, roles={"user"})
    active_blocks = ACTIVE_QUERY_BLOCK_RE.findall(user_text)
    history_blocks = RESPONSE_HISTORY_BLOCK_RE.findall(user_text)
    user_input_blocks = USER_INPUT_BLOCK_RE.findall(user_text)
    sample_id = sample.get("sample_id") or sample.get("trajectory_id") or "?"

    if expected_query is None:
        if active_blocks or history_blocks:
            raise QueryRenderContractError(
                f"sample={sample_id}: inactive/compress turn rendered "
                f"active_query={len(active_blocks)} response_history={len(history_blocks)}"
            )
        return

    if len(active_blocks) != 1 or len(history_blocks) != 1:
        raise QueryRenderContractError(
            f"sample={sample_id}: expected exactly one active_query and one "
            f"response_history, got active_query={len(active_blocks)} "
            f"response_history={len(history_blocks)}"
        )

    active = active_blocks[0]
    q_lines = QUESTION_LINE_RE.findall(active)
    if len(q_lines) != 1 or not q_lines[0].strip():
        raise QueryRenderContractError(
            f"sample={sample_id}: active_query must contain one non-empty Q line"
        )
    expected_question = str(expected_query.get("question") or "").strip()
    if not expected_question:
        raise QueryRenderContractError(
            f"sample={sample_id}: structured active query question is empty"
        )
    if _normalise_ws(q_lines[0]) != _normalise_ws(expected_question):
        raise QueryRenderContractError(
            f"sample={sample_id}: rendered query text mismatch: "
            f"{q_lines[0]!r} != {expected_question!r}"
        )
    for user_input in user_input_blocks:
        if user_input_is_active_query_duplicate(
            user_input,
            queries,
            inter_chunk=inter_chunk,
        ):
            raise QueryRenderContractError(
                f"sample={sample_id}: active query duplicated in <user_input>"
            )

    option_lines = OPTIONS_LINE_RE.findall(active)
    answer_format_lines = ANSWER_FORMAT_LINE_RE.findall(active)
    answer_form = str(expected_query.get("answer_form") or "").strip()
    if answer_form == "multiple_choice":
        options = [str(x) for x in expected_query.get("options") or [] if str(x).strip()]
        if not options:
            raise QueryRenderContractError(
                f"sample={sample_id}: MC active query has empty structured options"
            )
        if len(option_lines) != 1 or not option_lines[0].strip():
            raise QueryRenderContractError(
                f"sample={sample_id}: MC active_query must render exactly one "
                "non-empty Options line"
            )
        rendered_options = _normalise_ws(option_lines[0])
        expected_options = _normalise_ws(" ".join(options))
        if rendered_options != expected_options:
            raise QueryRenderContractError(
                f"sample={sample_id}: rendered Options mismatch: "
                f"{rendered_options!r} != {expected_options!r}"
            )
        if len(answer_format_lines) != 1 or not answer_format_lines[0].strip():
            raise QueryRenderContractError(
                f"sample={sample_id}: MC active_query must render exactly one "
                "non-empty Answer format line"
            )
        expected_instruction = _answer_format_body(
            canonical_answer_instruction(expected_query)
        )
        if (
            expected_instruction
            and _normalise_ws(answer_format_lines[0])
            != _normalise_ws(expected_instruction)
        ):
            raise QueryRenderContractError(
                f"sample={sample_id}: rendered Answer format mismatch: "
                f"{answer_format_lines[0]!r} != {expected_instruction!r}"
            )
    else:
        if option_lines:
            raise QueryRenderContractError(
                f"sample={sample_id}: non-MC active_query rendered Options line"
            )
        if answer_form and (len(answer_format_lines) != 1 or not answer_format_lines[0].strip()):
            raise QueryRenderContractError(
                f"sample={sample_id}: active_query answer_form={answer_form!r} "
                "requires one non-empty Answer format line"
            )


def _assistant_answer_blocks(messages: List[Dict]) -> List[str]:
    out: List[str] = []
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = str(content or "")
        out.extend(m.group(1).strip() for m in ANSWER_BLOCK_RE.finditer(text))
    return out


def validate_answer_render_contract(sample: Dict, messages: List[Dict]) -> None:
    """Hard-check response targets against the structured question metadata.

    Query rendering can be correct while the supervised response is still wrong
    because a pass moved options/correct_option/gold_answer out of sync. This
    check fails pass5 instead of emitting an SFT row whose answer would receive
    zero reward under the shared RL/eval matcher.
    """
    answers = _assistant_answer_blocks(messages)
    nonempty = [a for a in answers if a.strip()]
    sample_type = str(sample.get("sample_type") or "")
    action = str(sample.get("action") or sample_type or "")
    sample_id = sample.get("sample_id") or sample.get("trajectory_id") or "?"

    if sample_type == "silent" or action == "silent":
        if nonempty:
            raise QueryRenderContractError(
                f"sample={sample_id}: silent row rendered non-empty answer {nonempty[:1]!r}"
            )
        return

    if not nonempty:
        if sample_type == "response" or action == "response":
            raise QueryRenderContractError(
                f"sample={sample_id}: response row rendered empty/no <answer>"
            )
        return

    if len(nonempty) > 1:
        raise QueryRenderContractError(
            f"sample={sample_id}: rendered multiple non-empty answers {nonempty!r}"
        )

    q = sample.get("_trajectory_question") or sample.get("metadata") or {}
    if not q:
        # Background rows should not answer. If they did, fail loudly.
        raise QueryRenderContractError(
            f"sample={sample_id}: non-empty answer without question metadata"
        )

    answer = nonempty[0]
    answer_form = str(q.get("answer_form") or "").strip()
    options = [str(x) for x in q.get("options") or [] if str(x).strip()]
    correct_option = q.get("correct_option", "")

    if answer_form == "multiple_choice":
        letter, correct_text = _mc_letter_text(q)
        if not options:
            raise QueryRenderContractError(
                f"sample={sample_id}: MC response has no options in metadata"
            )
        if letter:
            idx = OPTION_LETTERS.index(letter)
            if idx >= len(options):
                raise QueryRenderContractError(
                    f"sample={sample_id}: correct_option={letter!r} outside "
                    f"options len={len(options)}"
                )
            option_text = _strip_option_label(options[idx])
            if correct_text and _normalise_ws(correct_text) != _normalise_ws(option_text):
                raise QueryRenderContractError(
                    f"sample={sample_id}: correct answer text mismatch: "
                    f"{correct_text!r} != option[{letter}] {option_text!r}"
                )
        target = _mc_target_for_question(q, sample.get("chunk_idx"))
        if target and _normalise_ws(answer) != _normalise_ws(target):
            raise QueryRenderContractError(
                f"sample={sample_id}: MC SFT answer mismatch: {answer!r} != {target!r}"
            )

    gold = (
        _per_emit_target(q, sample.get("chunk_idx"))
        or str(q.get("sft_answer") or "").strip()
        or str(q.get("gold_answer") or "").strip()
        or str(q.get("canonical_answer") or "").strip()
        or str(q.get("correct_answer_text") or "").strip()
    )
    if not gold:
        raise QueryRenderContractError(
            f"sample={sample_id}: non-empty answer {answer!r} has empty gold target"
        )

    try:
        from thinkstream.trainer.outcome_match import score_outcome_by_form
    except Exception as exc:
        raise QueryRenderContractError(
            f"sample={sample_id}: cannot import shared outcome matcher: {exc}"
        ) from exc

    score = score_outcome_by_form(
        answer,
        options=options,
        correct_option=correct_option,
        gold_answer=gold,
        answer_form=answer_form,
    )
    if score < 1.0:
        raise QueryRenderContractError(
            f"sample={sample_id}: rendered answer would score 0 under shared "
            f"matcher: answer={answer!r}, gold={gold!r}, "
            f"answer_form={answer_form!r}, correct_option={correct_option!r}"
        )


def build_sft_rows(
    sample: Dict,
    messages: List[Dict],
    *,
    frame_protocol: str,
    render_layout: str = RENDER_LAYOUT_QUERY_LAST,
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


def _silent_role(sample: Dict) -> str:
    """Classify silent rows by training value.

    pending_question:
      The model has an open query and must intentionally wait.
    post_answer:
      The model has just answered or the query is already closed; this teaches
      not to repeat answers.
    no_question:
      Background/patrol silence; useful but low information in bulk.
    """
    if sample.get("sample_type") != "silent" or sample.get("action") != "silent":
        return ""
    queries = list(sample.get("queries") or [])
    card_id = str(sample.get("card_id") or "")
    if queries:
        related = [
            q for q in queries
            if not card_id or str(q.get("card_id") or "") == card_id
        ]
        if not related:
            related = queries
        has_open = any(
            str(q.get("status", "")).lower() in {"open", "pending", "active"}
            for q in related
        )
        if has_open:
            return "pending_question"
        has_answer = any(q.get("answers") for q in related)
        if has_answer:
            return "post_answer"
    meta = sample.get("metadata") or {}
    if card_id or meta.get("question"):
        return "pending_question"
    return "no_question"


def _choose_ranked(items: List[tuple[int, Dict]], n: int) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    return sorted(items, key=lambda x: _sample_rank(x[1], x[0]))[:n]


def _silent_diversity_key(sample: Dict) -> str:
    role = _silent_role(sample)
    meta = sample.get("metadata") or {}
    family = str(meta.get("family") or "none")
    answer_form = str(meta.get("answer_form") or "none")
    availability = str(
        meta.get("availability")
        or sample.get("sequence_type")
        or "none"
    )
    question_type = str(meta.get("question_type") or "single_emit")
    base_role = str(sample.get("base_role") or "")

    if role == "pending_question":
        if base_role == "recall_wait_no_history":
            subtype = "recall_wait_no_history"
        elif availability == "event_watch":
            subtype = "future_event_wait"
        elif availability == "multi_response" or question_type == "multi_emit":
            subtype = "multi_emit_wait"
        elif availability == "recall_success":
            subtype = "recall_answer_pending"
        elif availability == "memory_response":
            subtype = "memory_answer_pending"
        elif availability == "immediate_response":
            subtype = "immediate_boundary_wait"
        else:
            subtype = availability or "pending"
        return f"{role}|{subtype}|{family}|{answer_form}|{question_type}"

    if role == "post_answer":
        return f"{role}|{family}|{answer_form}|{question_type}"
    return f"{role}|{base_role or 'patrol'}"


def _choose_diverse_silent(
    items: List[tuple[int, Dict]],
    n: int,
) -> List[tuple[int, Dict]]:
    """Deterministically sample silent rows while preserving boundary variety."""
    if n <= 0 or not items:
        return []
    by_key: Dict[str, List[tuple[int, Dict]]] = {}
    for item in items:
        by_key.setdefault(_silent_diversity_key(item[1]), []).append(item)
    for key in by_key:
        by_key[key] = _choose_ranked(by_key[key], len(by_key[key]))

    selected: List[tuple[int, Dict]] = []
    cursors = {key: 0 for key in by_key}
    keys = sorted(by_key, key=lambda k: (-len(by_key[k]), k))
    while len(selected) < n:
        progressed = False
        for key in keys:
            cur = cursors[key]
            bucket = by_key[key]
            if cur >= len(bucket):
                continue
            selected.append(bucket[cur])
            cursors[key] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def _is_multi_emit_response(sample: Dict) -> bool:
    if sample.get("sample_type") != "response":
        return False
    meta = sample.get("metadata") or {}
    return (
        meta.get("question_type") == "multi_emit"
        or meta.get("family") in {"F5", "F7", "CRR1", "PN1"}
    )


def _choose_multi_emit_response(
    items: List[tuple[int, Dict]],
    n: int,
) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    by_key: Dict[str, List[tuple[int, Dict]]] = {}
    for item in items:
        meta = item[1].get("metadata") or {}
        key = "|".join([
            str(meta.get("family") or "unknown"),
            str(meta.get("answer_form") or "unknown"),
            str(meta.get("availability") or item[1].get("sequence_type") or "unknown"),
        ])
        by_key.setdefault(key, []).append(item)
    for key in by_key:
        by_key[key] = _choose_ranked(by_key[key], len(by_key[key]))

    selected: List[tuple[int, Dict]] = []
    cursors = {key: 0 for key in by_key}
    keys = sorted(by_key)
    while len(selected) < n:
        progressed = False
        for key in keys:
            cur = cursors[key]
            bucket = by_key[key]
            if cur >= len(bucket):
                continue
            selected.append(bucket[cur])
            cursors[key] += 1
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
      - keep enough silent rows to make silent roughly 40-45% of SFT;
      - prefer pending-query and post-answer silent rows over patrol/background
        rows so SFT learns answer timing boundaries instead of just idle chunks.
    """
    indexed = list(enumerate(samples))
    recall_rows = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "recall"
    ]
    compress_rows = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "compress"
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
    response_rows = (
        ordinary_response
        + other_active
        + _choose_multi_emit_response(multi_emit_response, multi_limit)
    )
    active = (
        recall_rows
        + compress_rows
        + response_rows
    )
    pending_silent = [
        (i, s) for i, s in indexed if _silent_role(s) == "pending_question"
    ]
    post_answer_silent = [
        (i, s) for i, s in indexed if _silent_role(s) == "post_answer"
    ]
    base_silent = [(i, s) for i, s in indexed if _silent_role(s) == "no_question"]
    if not active:
        return samples, {"before": len(samples), "after": len(samples)}

    target_silent = min(
        len(pending_silent) + len(post_answer_silent) + len(base_silent),
        max(1, int(len(active) * SFT_SILENT_TO_ACTIVE_RATIO)),
    )
    target_pending = min(
        len(pending_silent),
        int(target_silent * SFT_PENDING_SILENT_FRACTION),
    )
    target_post_answer = min(
        len(post_answer_silent),
        int(target_silent * SFT_POST_ANSWER_SILENT_FRACTION),
    )
    kept_silent = (
        _choose_diverse_silent(pending_silent, target_pending)
        + _choose_diverse_silent(post_answer_silent, target_post_answer)
    )
    remaining = target_silent - len(kept_silent)
    if remaining > 0:
        kept_silent.extend(_choose_diverse_silent(base_silent, remaining))
    if len(kept_silent) < target_silent:
        used = {i for i, _s in kept_silent}
        rest = [
            (i, s) for bucket in (pending_silent, post_answer_silent, base_silent)
            for i, s in bucket
            if i not in used
        ]
        kept_silent.extend(_choose_diverse_silent(rest, target_silent - len(kept_silent)))

    selected = active + kept_silent
    selected.sort(key=lambda x: x[0])
    out = [s for _i, s in selected]
    if sum(1 for s in out if s.get("sample_type") == "recall") != len(recall_rows):
        raise RuntimeError("SFT balancing must not drop recall samples")
    if sum(1 for s in out if s.get("sample_type") == "compress") != len(compress_rows):
        raise RuntimeError("SFT balancing must not drop compress samples")
    return out, {
        "before": len(samples),
        "after": len(out),
        "active_kept": len(active),
        "ordinary_response_kept": len(ordinary_response),
        "multi_emit_response_before": len(multi_emit_response),
        "multi_emit_response_kept": min(len(multi_emit_response), multi_limit),
        "recall_kept": len(recall_rows),
        "compress_before": len(compress_rows),
        "compress_kept": len(compress_rows),
        "recall_compress_kept": len(recall_rows) + len(compress_rows),
        "silent_before": len(pending_silent) + len(post_answer_silent) + len(base_silent),
        "silent_kept": len(kept_silent),
        "pending_silent_before": len(pending_silent),
        "post_answer_silent_before": len(post_answer_silent),
        "base_silent_before": len(base_silent),
        "pending_silent_kept": sum(
            1 for _i, s in kept_silent if _silent_role(s) == "pending_question"
        ),
        "post_answer_silent_kept": sum(
            1 for _i, s in kept_silent if _silent_role(s) == "post_answer"
        ),
        "base_silent_kept": sum(
            1 for _i, s in kept_silent if _silent_role(s) == "no_question"
        ),
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
    frame_protocol: str = "video_meta",
    render_layout: str = RENDER_LAYOUT_QUERY_LAST,
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
                validate_query_render_contract(sample, messages)
                validate_answer_render_contract(sample, messages)
            except QueryRenderContractError:
                raise
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
            "--final-dir. New training/eval runs should use "
            "rendered/video_meta_standard_query_last."
        ),
    )
    parser.add_argument(
        "--frame-protocol",
        default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"),
        choices=["video_meta"],
        help=(
            "Student/eval visual carrier. The supported project entry uses "
            "video_meta plus the selected render layout."
        ),
    )
    parser.add_argument(
        "--render-layout",
        default=RENDER_LAYOUT_QUERY_LAST,
        choices=[RENDER_LAYOUT_QUERY_LAST],
        help=(
            "Prompt layout. standard_query_last keeps memory before visual and "
            "places active_query after the current visual window."
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
    render_layout = str(args.render_layout or RENDER_LAYOUT_QUERY_LAST)
    if frame_protocol != "video_meta":
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
