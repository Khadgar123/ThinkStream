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
    format_memory_block,
    format_queries_block,
    format_user_input_block,
    append_visual_frames,
    build_recalled_frames_metadata,
    normalize_frame_protocol,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
)

logger = logging.getLogger(__name__)

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


def build_messages(
    sample: Dict,
    base_path: Path,
    *,
    data_dir: Optional[Path] = None,
    frame_protocol: str = "ts_image",
) -> List[Dict]:
    """Produce v12 ShareGPT messages for one sample. Stdlib-only.

    This is the canonical offline renderer. It must stay aligned with
    thinkstream.data.agent_protocol.build_user_content and the verl RL
    prompt builder: user_input, memory, queries, visual_window,
    protocol-selected visual frames, recalled frames, then recall_result.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
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
    sample_id_suffix: str = "",
) -> Dict[str, Any]:
    """Attach row-local tool schema + label-mask policy metadata."""
    row["tool_schema_mode"] = tool_schema_mode
    row["loss_assistant_turns"] = loss_assistant_turns
    if sft_subtype:
        row["sft_subtype"] = sft_subtype
        meta = dict(row.get("metadata") or {})
        meta["sft_subtype"] = sft_subtype
        meta["loss_assistant_turns"] = loss_assistant_turns
        row["metadata"] = meta
    if sample_id_suffix:
        sid = str(row.get("sample_id") or "")
        if sid:
            row["sample_id"] = f"{sid}:{sample_id_suffix}"
    return row


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


def build_sft_rows(
    sample: Dict,
    messages: List[Dict],
    *,
    frame_protocol: str,
) -> List[Dict]:
    """Return one or more runtime-aligned SFT rows for a rendered sample.

    Qwen's chat template renders tools at the conversation level for a single
    apply_chat_template call. A multi-turn recall row therefore cannot train
    both the recall tool call and the post-recall answer in one sample without
    leaking recall tools into the second assistant turn. Split it:
      - recall_query: first assistant turn only, recall schema available;
      - recall_answer: full prefix including recall_result, no tool schema,
        loss only on the final assistant turn.
    """
    if _is_recall_multiturn_messages(sample, messages):
        first_messages = deepcopy(messages[:3])
        first_row = _emit_row(sample, first_messages, frame_protocol=frame_protocol)
        _with_sft_turn_policy(
            first_row,
            tool_schema_mode="streaming",
            loss_assistant_turns="all",
            sft_subtype="recall_query",
            sample_id_suffix="recall_query",
        )

        second_messages = deepcopy(messages)
        second_row = _emit_row(sample, second_messages, frame_protocol=frame_protocol)
        _with_sft_turn_policy(
            second_row,
            tool_schema_mode="recall_response",
            loss_assistant_turns="last",
            sft_subtype="recall_answer",
            sample_id_suffix="recall_answer",
        )
        return [first_row, second_row]

    row = _emit_row(sample, deepcopy(messages), frame_protocol=frame_protocol)
    _with_sft_turn_policy(
        row,
        tool_schema_mode="compress" if sample.get("v12_inter_chunk") else "streaming",
        loss_assistant_turns="all",
        sft_subtype=str(sample.get("sample_type") or ""),
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
                )
            except (KeyError, ValueError) as exc:
                counts["failed"] += 1
                if counts["failed"] <= 5:
                    sid = sample.get("sample_id") or sample.get("trajectory_id") or i
                    logger.warning(f"[{src.name}] sample {sid} skipped: {exc}")
                continue

            for row in build_sft_rows(sample, messages, frame_protocol=frame_protocol):
                if limit and counts["ok"] >= limit:
                    break
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
            f"(is_trajectory={is_traj}, frame_protocol={frame_protocol})"
        )
        balance = out_stem == "train_sft_messages" and not args.no_balance_sft
        counts = convert(src, dst, is_trajectory=is_traj, base_path=base_path,
                         data_dir=data_dir,
                         limit=args.limit or None, balance_sft=balance,
                         frame_protocol=frame_protocol)
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
            "splits": splits_done,
        }, ensure_ascii=False, indent=2))
        logger.info(f"Wrote dataset_info.json → {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
