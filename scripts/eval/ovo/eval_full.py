"""Unified OVO-Bench eval on the ORIGINAL ovo_bench_new.json.

Why this script vs eval_ovo.py:
  eval_ovo.py reads ovo-bench-formatted.jsonl, where each FT test_info
  point was pre-expanded into an independent (ask, answer at video_end)
  record. That destroys the ask_time / clue_time / test_info delay
  structure that CRR was specifically designed to test, and makes
  REC's "running count" semantics indistinguishable from a single-shot
  QA at the last probe time.

  This script reads the original 1,640-record ovo_bench_new.json and
  dispatches per task family:

    BT / RT (EPM ASI HLD OCR ACR ATR STU FPD OJR)
       - Single realtime, MCQ A/B/C/D.
       - ask_chunk = realtime / 2; one probe; score first letter vs gt.

    FT-REC
       - test_info has multiple {realtime, count} probes (cumulative count).
       - Question active from chunk 0; probe at each test_info.realtime
         and regex-extract integer.

    FT-SSR
       - Each test_info entry asks about a specific step at a specific
         time (different probes can ask about different steps). Treated
         as independent single-shot Yes/No queries (no shared ask_time).

    FT-CRR
       - Has ask_time AND clue_time. Question is asked once at ask_time;
         the agent must wait for evidence (clue_time) before saying Yes.
         test_info type=0 = before clue (expect No / silent),
         type=1 = after clue (expect Yes). This is the only OVO task
         with genuine ask-vs-answer delay structure.

  All tasks share the same agent loop with the same compress_mode and
  retriever — so a single eval run gives directly comparable per-task
  numbers.

Reports:
  Per task: content accuracy plus timing-aware variants
            (no early / no late / exactly on probe)
  Per category (RT / BT / FT): mean of task accuracies
  Overall: mean of category averages (matches OVO paper Table 2)
  CRR specifically: also reports type=0 / type=1 / fp_rate breakdown
                    (the delay-sensitive metric)

Usage:
    python scripts/eval/ovo/eval_full.py \\
        --ckpt output/agent-sft \\
        --benchmark_json /path/to/ovo_bench_new.json \\
        --video_root /path/to/videos \\
        --compress_mode system \\
        --retriever time_range \\
        [--tasks CRR,SSR,REC] [--n 30]
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from thinkstream.data.agent_protocol import (
    WRONG_RESPONSE_SPECIAL_TOKENS,
    ensure_agent_special_tokens,
    validate_agent_special_tokens,
)

from thinkstream.models.agent_loop import (
    COMPRESS_RANGE_MIN,
    COMPRESS_RANGE_MAX,
    COMPRESS_TOKEN_THRESHOLD,
    RECENT_THINKS_TOKEN_BUDGET,
    MemoryState,
    StreamingAgentLoop,
    AGENT_CHUNK_SEC,
    _parse_agent_output,
    build_single_step_messages,
    make_generate_fn,
    recall_query_available_for_chunk,
    select_compress_range_by_tokens,
)
from thinkstream.models.retrieval import make_retriever
from thinkstream.data.agent_protocol import (
    FRAMES_PER_CHUNK,
    VISUAL_WINDOW_CHUNKS,
    action_space_error_for_turn,
    build_recalled_frames_metadata,
    build_recall_result_metadata,
    build_recall_result_user_content,
    canonical_answer_instruction,
    normalize_frame_protocol,
    normalize_memory_position,
    normalize_render_layout,
    select_recall_chunks_uniform,
    system_prompt_for_frame_protocol,
    tools_for_turn,
)
from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS
from thinkstream.eval.prompt_contract import (
    build_plain_mcq_prompt,
    build_streaming_query_meta,
)
from thinkstream.sft.args import DataArguments
from thinkstream.sft.data_processor import (
    update_processor_pixels,
)


class FrameCacheMissError(RuntimeError):
    """Raised when eval forbids raw-video decode fallback."""


# ─── Task taxonomy (mirrors official constant.py) ────────────────────────────

RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
BT_TASKS = {"EPM", "ASI", "HLD"}
FT_TASKS = {"REC", "SSR", "CRR"}
ALL_TASKS = RT_TASKS | BT_TASKS | FT_TASKS

# Pre-extracted OVO frames are mixed: most streams are 2 frames/chunk, while
# Perception Test REC videos are 1 frame/chunk. 0 means infer per video.
SOURCE_FRAMES_PER_CHUNK = 0


# ─── Detect ckpt model class ─────────────────────────────────────────────────


def detect_model_class(ckpt: str):
    config_text = ""
    try:
        cfg = AutoConfig.from_pretrained(ckpt, local_files_only=True)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        architectures = [str(x).lower() for x in getattr(cfg, "architectures", [])]
        config_text = " ".join([model_type, *architectures])
    except Exception:
        config_text = Path(ckpt.rstrip("/")).name.lower()

    if "qwen3.5" in config_text or "qwen_3.5" in config_text or "qwen3_5" in config_text:
        from transformers import Qwen3_5ForConditionalGeneration as Cls
        return Cls, "qwen3_5"
    if "qwen3_vl_moe" in config_text or "qwen3vlmoe" in config_text:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen3" in config_text or "qwen3_vl" in config_text:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen2.5" in config_text or "qwen-2.5" in config_text or "qwen2_5" in config_text:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls, "qwen2.5vl"
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls, "qwen3vl"


def resolve_video_path(video_field, video_root):
    p = Path(video_field)
    return str(p) if p.is_absolute() else str(Path(video_root) / video_field)


# ─── Per-task prompt builders (match OVO official prompt families) ───────────

def build_mcq_question(sample):
    """BT/RT MCQ prompt: question + options + 'Answer with a single letter.'"""
    return build_plain_mcq_prompt(
        sample["question"],
        sample.get("options", []),
        option_style="dot",
        instruction="Answer with a single letter.",
    )


def build_mcq_agent_question(sample):
    """Bare MCQ question for StreamingAgentLoop active-query rendering."""
    return str(sample.get("question", ""))


def build_mcq_query_meta(sample):
    """Structured MCQ metadata; active_query renders options exactly once."""
    return build_streaming_query_meta(
        sample,
        answer_form="multiple_choice",
        answer_style="letter_plus_text",
    )


def build_rec_question(sample):
    """REC prompt asks for cumulative integer count."""
    activity = sample.get("activity", "perform the action")
    return (
        f"You're watching a video where people may perform a certain action "
        f"repetitively. The performer is referred to as 'they'.\n"
        f"How many times have they {activity} so far?\n"
        f"Your response type should be INT, for example, 0/1/2/3."
    )


def build_ssr_question(step_text):
    """SSR prompt asks Yes/No about a specific step."""
    return (
        f"You're watching a tutorial video which contains a sequence of steps. "
        f"The following is one step from the procedure:\n\n{step_text}\n\n"
        f"Your task is to decide: Is the person in the video currently "
        f"carrying out this step?\n"
        f"Return \"Yes\" if they are; return \"No\" if not."
    )


def build_crr_question(sample):
    """CRR prompt asks Yes/No about whether a described action has occurred."""
    return (
        f"{sample['question']}\n"
        f"Return \"Yes\" if the action described has happened in the visible "
        f"video so far; otherwise return \"No\"."
    )


# ─── Answer extraction & scoring ─────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-Za-z])\b")
_INT_RE = re.compile(r"\d+")


def extract_letter(text):
    if not text:
        return None
    t = text.strip()
    # First option letter (word boundary or first char).
    if t and t[0].isalpha() and (len(t) == 1 or not t[1].isalpha()):
        return t[0].upper()
    m = _LETTER_RE.search(t)
    return m.group(1).upper() if m else None


def extract_int(text):
    if not text:
        return None
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def is_yes(text):
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("yes") or t.startswith("y ") or t == "y"


def is_no(text):
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("no") or t.startswith("n ") or t == "n"


# ─── Agent runner: shared streaming loop ─────────────────────────────────────


def run_agent(loop, video_path, ask_chunks, max_chunk, telemetry=None,
              ask_meta=None):
    """Run agent through chunks 0..max_chunk, injecting questions per ask_chunks.

    ask_chunks: dict {chunk_idx: question_text} — question(s) to inject at
                specific chunks.
    telemetry:  optional dict; if provided, populated with per-step stats:
                  compress_events: list of {chunk, thinks_count, n_compressed}
                  recall_events:   list of {chunk, returned_chunks}
                These feed the compact `summary.health` abnormal-behavior
                report in addition to per-probe accuracy.

    Returns: dict {chunk_idx: (action, response_text)}
    """
    per_chunk = {}
    ask_meta = ask_meta or {}
    chunk_idx = 0
    while chunk_idx <= max_chunk:
        q = ask_chunks.get(chunk_idx)
        try:
            result = loop.step(
                chunk_idx=chunk_idx,
                video_path=video_path,
                user_question=q,
                user_question_meta=ask_meta.get(chunk_idx),
            )
        except Exception as e:
            per_chunk[chunk_idx] = ("error", str(e))
            if telemetry is not None:
                telemetry["n_step_errors"] = telemetry.get("n_step_errors", 0) + 1
                telemetry.setdefault("step_errors", []).append({
                    "chunk": chunk_idx,
                    "error": f"{type(e).__name__}: {e}",
                })
                telemetry["total_steps"] = telemetry.get("total_steps", 0) + 1
            chunk_idx += 1
            continue

        # v9.4.2 telemetry — record compress/recall events when they fire,
        # plus per-step prompt/think/format/compress-success metrics.
        if telemetry is not None:
            ct = result.get("compress_telemetry")
            if ct:
                from thinkstream.models.agent_loop import COMPRESS_RANGE_MIN
                compressed_chunks = ct.get("compressed_chunks") or []
                telemetry.setdefault("compress_events", []).append({
                    "chunk": chunk_idx,
                    "thinks_count": ct["thinks_count_at_trigger"],
                    "thinks_token_count": ct.get("thinks_token_count"),
                    "compress_threshold": ct.get("compress_threshold", COMPRESS_TOKEN_THRESHOLD),
                    "compress_range_min": ct.get("compress_range_min", COMPRESS_RANGE_MIN),
                    "compress_range_max": ct.get("compress_range_max"),
                    "n_compressed": len(compressed_chunks),
                    "compressed_chunks": list(compressed_chunks),
                    "system_trigger_rule_ok": ct.get("system_trigger_rule_ok"),
                    "system_range_rule_ok": ct.get("system_range_rule_ok"),
                    "succeeded": bool(result.get("compress_succeeded")),
                    "partial": (0 < len(compressed_chunks) < COMPRESS_RANGE_MIN),
                })
                # Track per-chunk compression count (revisit detection)
                rev = telemetry.setdefault("compress_chunk_count", {})
                for c in compressed_chunks:
                    rev[int(c)] = rev.get(int(c), 0) + 1
            if result.get("action") == "recall":
                payload = result.get("payload") or {}
                recall_args = payload.get("recall_args") or {}
                schema = "with_start_end" if (
                    isinstance(recall_args, dict)
                    and recall_args.get("start_time") is not None
                    and recall_args.get("end_time") is not None
                ) else "missing_start_end"
                recall_result = result.get("recall_result") or {}
                recall_metadata_chars = len(json.dumps(recall_result, ensure_ascii=False))
                telemetry.setdefault("recall_events", []).append({
                    "chunk": chunk_idx,
                    "returned_chunks": list(result.get("recall_returned_chunks", [])),
                    "schema": schema,
                    "requested_time_range": {
                        "start_time": recall_args.get("start_time"),
                        "end_time": recall_args.get("end_time"),
                    } if isinstance(recall_args, dict) else {},
                    "source": recall_result.get("source", ""),
                    "result_time": recall_result.get("time", ""),
                    "result_metadata_chars": recall_metadata_chars,
                    "result_text_chars": 0,
                })
            # Per-step extras (always recorded if available)
            if result.get("prompt_text_token_count") is not None:
                telemetry.setdefault("prompt_tokens_per_step", []).append(
                    result["prompt_text_token_count"])
            if result.get("think_token_count") is not None:
                telemetry.setdefault("think_tokens_per_step", []).append(
                    result["think_token_count"])
            if not result.get("format_ok", True):
                telemetry["n_format_violations"] = (
                    telemetry.get("n_format_violations", 0) + 1)
            if result.get("action_space_error") or result.get("invalid_action"):
                telemetry["n_action_space_errors"] = (
                    telemetry.get("n_action_space_errors", 0) + 1)
            if result.get("recall_step2_blocked"):
                telemetry["n_recall_step2_blocked"] = (
                    telemetry.get("n_recall_step2_blocked", 0) + 1)
            telemetry["total_steps"] = telemetry.get("total_steps", 0) + 1

        action = result.get("action", "?")
        payload = result.get("payload") or {}
        if action == "recall":
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload") or {}
            per_chunk[chunk_idx] = (final_action, final_payload.get("response", ""))
        elif action == "response":
            per_chunk[chunk_idx] = ("response", payload.get("response", ""))
        elif action == "silent":
            per_chunk[chunk_idx] = ("silent", "")
        elif action == "compress":
            per_chunk[chunk_idx] = ("compress", "")
        else:
            per_chunk[chunk_idx] = (action, "")
        if telemetry is not None:
            final_action, final_response = per_chunk.get(chunk_idx, ("missing", ""))
            if action == "recall" and telemetry.get("recall_events"):
                telemetry["recall_events"][-1]["final_action"] = final_action
                telemetry["recall_events"][-1]["final_response"] = final_response
            telemetry.setdefault("step_records", []).append({
                "chunk": chunk_idx,
                "action": action,
                "final_action": final_action,
                "response": final_response,
                "think": result.get("think", ""),
                "think_tokens": result.get("think_token_count"),
                "prompt_tokens": result.get("prompt_text_token_count"),
                "memory_tokens": result.get("memory_token_count"),
                "format_ok": bool(result.get("format_ok", True)),
                "action_space_error": result.get("action_space_error", ""),
                "invalid_action": result.get("invalid_action", ""),
                "recall_step2_blocked": bool(result.get("recall_step2_blocked")),
                "compress_succeeded": result.get("compress_succeeded"),
            })
        if action == "compress" and result.get("compress_telemetry") and result.get("compress_succeeded"):
            # Inter-chunk memory management: retry the same video chunk after
            # successful compression so ask_chunk questions are not skipped.
            continue
        chunk_idx += 1
    return per_chunk


def latest_response_at_or_before(per_chunk, upto_chunk, since_chunk=0):
    """Return the latest response in [since_chunk, upto_chunk]."""
    for c in range(int(upto_chunk), int(since_chunk) - 1, -1):
        action, resp = per_chunk.get(c, ("missing", ""))
        if action == "response" and resp:
            return c, resp
    return None, ""


def first_yes_response_between(per_chunk, start_chunk, end_chunk):
    """Return the first Yes response in [start_chunk, end_chunk], if any."""
    for c in range(int(start_chunk), int(end_chunk) + 1):
        action, resp = per_chunk.get(c, ("missing", ""))
        if action == "response" and is_yes(resp):
            return c, resp
    return None, ""


def make_loop(model, processor, tokenizer, model_type, retriever,
              compress_mode, max_new_tokens, frames_root=None, video_root=None,
              frame_protocol="video_meta", memory_mode="full",
              min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
              max_pixels=DEFAULT_VIDEO_MAX_PIXELS):
    # Runtime profile aligned with schema/pass2/SFT/RL defaults.
    return StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        retriever=retriever,
        compress_mode=compress_mode,
        memory_mode=memory_mode,
        frames_root=frames_root,
        video_root=video_root,
        frame_protocol=frame_protocol,
    )


def reset_visual_index(retriever):
    if hasattr(retriever, "chunk_embeddings"):
        retriever.chunk_embeddings.clear()


class NullRetriever:
    """Recall ablation: keep the recall action path, but return no evidence."""

    def index_chunk(self, chunk_idx, video_path, think_text):
        return None

    def __call__(self, query, archive):
        return {
            "source": "failure",
            "time": "",
            "returned_chunks": [],
        }


def _ordinary_prompt_snapshot(memory_mode: str, snapshot: Dict) -> Dict:
    if memory_mode not in {"no_prompt", "none"}:
        return snapshot
    out = dict(snapshot)
    out["compressed_segments"] = []
    out["compressed"] = []
    out["recent_thinks"] = []
    return out


def _resolve_frame_dir(
    video_path: str,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Optional[Path]:
    if not frames_root:
        return None
    vp = Path(video_path)
    if video_root:
        try:
            frame_dir = Path(frames_root) / vp.relative_to(Path(video_root)).with_suffix("")
        except ValueError:
            frame_dir = Path(frames_root) / vp.with_suffix("")
    else:
        frame_dir = Path(frames_root) / vp.with_suffix("")
    if not frame_dir.exists():
        flat_dir = Path(frames_root) / vp.stem
        frame_dir = flat_dir if flat_dir.exists() else frame_dir
    return frame_dir if frame_dir.exists() else None


def _selected_source_frame_offsets(source_fpc: int) -> List[int]:
    source_fpc = max(1, int(source_fpc))
    target_fpc = max(1, int(FRAMES_PER_CHUNK))
    if target_fpc == 1:
        return [source_fpc // 2]
    return [
        min(source_fpc - 1, int(round(i * (source_fpc - 1) / max(1, target_fpc - 1))))
        for i in range(target_fpc)
    ]


@lru_cache(maxsize=4096)
def _frame_cache_max_number(
    video_path: str,
    frames_root: str,
    video_root: str,
) -> int:
    frame_dir = _resolve_frame_dir(video_path, frames_root, video_root)
    if frame_dir is None:
        return 0
    max_no = 0
    for fp in frame_dir.glob("frame_*.jpg"):
        raw = fp.stem[6:] if fp.stem.startswith("frame_") else fp.stem
        if raw.isdigit():
            max_no = max(max_no, int(raw))
    return max_no


@lru_cache(maxsize=4096)
def _video_duration_seconds(video_path: str) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nframes = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0 and nframes > 0:
            return nframes / fps
    except Exception:
        return 0.0
    return 0.0


def _effective_source_frames_per_chunk(
    video_path: str,
    frames_root: Optional[str],
    video_root: Optional[str],
    *,
    max_chunk_hint: Optional[int] = None,
) -> int:
    configured = int(SOURCE_FRAMES_PER_CHUNK)
    if configured > 0:
        return configured
    max_frame_no = _frame_cache_max_number(
        str(video_path),
        str(frames_root or ""),
        str(video_root or ""),
    )
    duration = _video_duration_seconds(str(video_path))
    if duration > 0 and max_frame_no > 0:
        return max(1, min(8, int(round(max_frame_no / duration))))
    if max_chunk_hint is not None and max_frame_no > 0:
        ratio = max_frame_no / max(1, int(max_chunk_hint) + 1)
        return max(1, min(8, int(round(ratio))))
    return max(1, int(FRAMES_PER_CHUNK))


def _resolve_window_frame_paths(
    video_path: str,
    chunk_idx: int,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Optional[List[str]]:
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK
    frame_dir = _resolve_frame_dir(video_path, frames_root, video_root)
    if frame_dir is None:
        return None
    out = []
    source_fpc = _effective_source_frames_per_chunk(
        video_path, frames_root, video_root, max_chunk_hint=chunk_idx,
    )
    offsets = _selected_source_frame_offsets(source_fpc)
    for ci in range(window_start, chunk_idx + 1):
        for fi in offsets:
            fp = frame_dir / f"frame_{ci * source_fpc + fi + 1:06d}.jpg"
            if fp.exists():
                out.append(str(fp))
    if len(out) < max(1, n_frames // 2):
        return None
    return out


def _resolve_chunk_frame_paths(
    video_path: str,
    chunk_idx: int,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> List[str]:
    frame_dir = _resolve_frame_dir(video_path, frames_root, video_root)
    if frame_dir is None:
        return []
    out = []
    source_fpc = _effective_source_frames_per_chunk(
        video_path, frames_root, video_root, max_chunk_hint=chunk_idx,
    )
    offsets = _selected_source_frame_offsets(source_fpc)
    for fi in offsets:
        fp = frame_dir / f"frame_{int(chunk_idx) * source_fpc + fi + 1:06d}.jpg"
        if fp.exists():
            out.append(str(fp))
    return out


def _clone_retriever(template):
    if hasattr(template, "clone_empty"):
        return template.clone_empty()
    if isinstance(template, NullRetriever):
        return NullRetriever()
    return template


def _build_retriever_template(args):
    if args.retriever == "none" or args.memory_mode in {"no_recall", "none"}:
        return NullRetriever()
    return make_retriever(
        kind=args.retriever,
        max_results=args.max_results,
        frames_root=args.frames_root,
        video_root=args.video_root,
    )


def _format_result_telemetry(result: Dict, tokenizer, messages: List[Dict]) -> None:
    result["memory_token_count"] = result.get("memory_token_count", 0)
    result["compress_threshold"] = COMPRESS_TOKEN_THRESHOLD
    result["compress_budget"] = RECENT_THINKS_TOKEN_BUDGET
    prompt_text_tokens = 0
    if tokenizer is not None:
        try:
            text_acc = []
            for msg in messages or []:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_acc.append(item.get("text", ""))
                elif isinstance(content, str):
                    text_acc.append(content)
            if text_acc:
                prompt_text_tokens = len(tokenizer.encode(
                    "\n".join(text_acc), add_special_tokens=False))
        except Exception:
            prompt_text_tokens = 0
    result["prompt_text_token_count"] = prompt_text_tokens
    think_tokens = 0
    if result.get("think") and tokenizer is not None:
        try:
            think_tokens = len(tokenizer.encode(
                result["think"], add_special_tokens=False))
        except Exception:
            think_tokens = 0
    result["think_token_count"] = think_tokens
    valid_actions = {"silent", "response", "recall", "compress"}
    action = result.get("action") or ""
    format_ok = (
        bool(result.get("think"))
        and action in valid_actions
        and not result.get("action_space_error")
        and not result.get("format_error")
    )
    if format_ok:
        payload = result.get("payload") or {}
        if action == "response":
            format_ok = "response" in payload and bool(payload["response"])
        elif action == "recall":
            format_ok = "recall_args" in payload
        elif action == "compress":
            summary = payload.get("summary")
            format_ok = bool(summary) and "time_range" in (summary or {})
    result["format_ok"] = format_ok
    if result.get("compress_telemetry") is not None:
        payload = result.get("payload") or {}
        summary = payload.get("summary")
        result["compress_succeeded"] = (
            action == "compress" and bool(summary) and "time_range" in summary
        )
    else:
        result["compress_succeeded"] = None


@dataclass
class _VllmAgentRunner:
    idx: int
    task: str
    sample_id: Any
    video_path: str
    ask_chunks: Dict[int, str]
    ask_meta: Dict[int, Dict]
    max_chunk: int
    tokenizer: Any
    retriever: Any
    frames_root: Optional[str]
    video_root: Optional[str]
    frame_protocol: str
    render_layout: str
    compress_mode: str
    memory_mode: str
    min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    require_frame_cache: bool = False
    current_chunk: int = 0
    done: bool = False
    per_chunk: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    telemetry: Dict = field(default_factory=dict)
    memory: MemoryState = field(init=False)
    last_messages: Optional[List[Dict]] = None
    last_turn_kind: str = "streaming"
    last_compress_telemetry: Optional[Dict] = None
    last_result: Optional[Dict] = None
    last_output_text: str = ""

    def __post_init__(self):
        self.memory = MemoryState(tokenizer=self.tokenizer)

    def _record_answer(self, answer_text: str, chunk_idx: int) -> None:
        if not answer_text:
            return
        response_time = chunk_idx * AGENT_CHUNK_SEC
        for q in reversed(self.memory.queries):
            status = str(q.get("status", "")).strip().lower()
            if status in {"open", "pending", "active"} or (
                not status and not q.get("answers")
            ):
                self.memory.answer_query(q["question"], answer_text, response_time)
                return

    def prepare(self) -> Tuple[List[Dict], str]:
        chunk_idx = self.current_chunk
        full_snapshot = self.memory.snapshot(chunk_idx)
        user_question = self.ask_chunks.get(chunk_idx)
        if user_question:
            ask_time = chunk_idx * AGENT_CHUNK_SEC
            already = any(
                q["question"] == user_question and q.get("ask_time") == ask_time
                for q in self.memory.queries
            )
            if not already:
                meta = self.ask_meta.get(chunk_idx) or {}
                instruction = canonical_answer_instruction(meta) or meta.get(
                    "answer_instruction"
                )
                self.memory.add_query(
                    user_question, ask_time,
                    options=meta.get("options"),
                    answer_form=meta.get("answer_form"),
                    answer_style=meta.get("answer_style"),
                    answer_instruction=instruction,
                    answer_chunks=meta.get("answer_chunks"),
                    per_emit_answers=meta.get("per_emit_answers"),
                    open_until=meta.get("open_until"),
                )

        compress_trigger = ""
        compress_telemetry = None
        if (
            self.compress_mode == "system"
            and self.memory_mode != "none"
            and self.memory.should_compress()
        ):
            n = select_compress_range_by_tokens(
                self.memory.recent_thinks,
                token_count_fn=self.memory._token_count,
            )
            oldest = self.memory.recent_thinks[:n] if n > 0 else []
            if oldest:
                chunks = [t["chunk"] for t in oldest]
                compress_trigger = "<compress_trigger/>"
                compress_telemetry = {
                    "thinks_count_at_trigger": len(self.memory.recent_thinks),
                    "thinks_token_count": self.memory.count_recent_tokens(),
                    "compressed_chunks": chunks,
                    "trigger_chunk": chunk_idx,
                    "compress_threshold": COMPRESS_TOKEN_THRESHOLD,
                    "compress_range_min": COMPRESS_RANGE_MIN,
                    "compress_range_max": COMPRESS_RANGE_MAX,
                    "system_trigger_rule_ok": (
                        self.memory.count_recent_tokens() >= COMPRESS_TOKEN_THRESHOLD
                        and len(self.memory.recent_thinks) >= COMPRESS_RANGE_MIN
                    ),
                    "system_range_rule_ok": (
                        COMPRESS_RANGE_MIN <= len(chunks) <= COMPRESS_RANGE_MAX
                    ),
                }

        user_input = compress_trigger or user_question or ""
        is_inter_chunk = bool(compress_trigger)
        snapshot = (
            full_snapshot if is_inter_chunk
            else _ordinary_prompt_snapshot(self.memory_mode, full_snapshot)
        )
        frame_paths = _resolve_window_frame_paths(
            self.video_path, chunk_idx, self.frames_root, self.video_root,
        )
        if not is_inter_chunk and self.frames_root and not frame_paths:
            miss = {
                "chunk": int(chunk_idx),
                "video_path": self.video_path,
                "window_chunks": int(VISUAL_WINDOW_CHUNKS),
                "frames_per_chunk": int(FRAMES_PER_CHUNK),
            }
            self.telemetry["n_frame_cache_misses"] = (
                self.telemetry.get("n_frame_cache_misses", 0) + 1
            )
            self.telemetry.setdefault("frame_cache_misses", []).append(miss)
            if self.require_frame_cache:
                raise FrameCacheMissError(
                    "frame cache miss at "
                    f"chunk={chunk_idx} video={self.video_path}"
                )
        messages = build_single_step_messages(
            snapshot,
            chunk_idx,
            self.video_path,
            user_input=user_input,
            queries=self.memory.queries,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            frame_paths=frame_paths,
            frame_protocol=self.frame_protocol,
            inter_chunk=is_inter_chunk,
            render_layout=self.render_layout,
        )
        self.last_messages = messages
        self.last_turn_kind = "compress" if is_inter_chunk else "streaming"
        self.last_compress_telemetry = compress_telemetry
        return messages, self.last_turn_kind

    def apply_first_output(self, output_text: str) -> Dict:
        self.last_output_text = output_text
        parsed = _parse_agent_output(output_text)
        action = parsed.get("action") or ""
        action_error = action_space_error_for_turn(action, self.last_turn_kind)
        if action_error:
            parsed["action_space_error"] = action_error
            parsed["invalid_action"] = action
            parsed["action"] = "invalid"
            action = "invalid"

        if (
            parsed.get("think")
            and action != "compress"
            and self.last_turn_kind != "compress"
            and self.memory_mode != "none"
        ):
            self.memory.add_think(self.current_chunk, parsed["think"])
            try:
                self.retriever.index_chunk(
                    self.current_chunk, self.video_path, parsed["think"])
            except Exception:
                pass

        if action == "compress":
            summary = parsed.get("payload", {}).get("summary", {})
            if summary and "time_range" in summary:
                tr = summary["time_range"]
                compressed_chunks = []
                for t in self.memory.recent_thinks:
                    cs = t["chunk"] * AGENT_CHUNK_SEC
                    ce = cs + AGENT_CHUNK_SEC
                    if cs >= tr[0] and ce <= tr[1]:
                        compressed_chunks.append(t["chunk"])
                self.memory.compress(summary, compressed_chunks=compressed_chunks)
        elif action == "response":
            self._record_answer(
                parsed.get("payload", {}).get("response", ""),
                self.current_chunk,
            )

        parsed["compress_telemetry"] = self.last_compress_telemetry
        parsed["memory_token_count"] = self.memory.count_recent_tokens()
        parsed["recall_returned_chunks"] = []
        _format_result_telemetry(parsed, self.tokenizer, self.last_messages or [])
        self.last_result = parsed
        return parsed

    def build_recall_messages(self) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        result = self.last_result or {}
        recall_args = (result.get("payload") or {}).get("recall_args") or {}
        if not recall_args:
            return None, None
        archive = (
            []
            if self.memory_mode in {"no_recall", "none"}
            else self.memory.retrieval_archive
        )
        if recall_query_available_for_chunk(
            recall_args,
            self.current_chunk,
            AGENT_CHUNK_SEC,
        ):
            recall_archive = []
            for item in archive:
                try:
                    item_chunk = int(item.get("chunk", -1))
                except (AttributeError, TypeError, ValueError):
                    continue
                if item_chunk < int(self.current_chunk):
                    recall_archive.append(item)
            raw_recall_result = self.retriever(recall_args, recall_archive)
        else:
            raw_recall_result = {
                "source": "failure",
                "time": "",
                "text_content": "No valid historical recall range provided.",
                "returned_chunks": [],
            }
        returned = select_recall_chunks_uniform(raw_recall_result.get("returned_chunks", []))
        raw_recall_result["returned_chunks"] = returned
        result["recall_returned_chunks"] = returned

        recalled_frames = None
        if returned and raw_recall_result.get("source") == "historical_frames":
            rf_paths: List[str] = []
            frame_chunks: List[int] = []
            for rc in returned:
                paths = _resolve_chunk_frame_paths(
                    self.video_path, rc, self.frames_root, self.video_root,
                )
                if paths:
                    frame_chunks.append(rc)
                    rf_paths.extend(paths)
            recalled_frames = build_recalled_frames_metadata(
                frame_chunks if rf_paths else returned,
                rf_paths,
                chunk_sec=AGENT_CHUNK_SEC,
                frames_per_chunk=FRAMES_PER_CHUNK,
            )
        recall_result = build_recall_result_metadata(
            raw_recall_result,
            recalled_frames,
        )
        result["recall_result"] = recall_result

        recall_messages = deepcopy(self.last_messages or [])
        if recall_messages and recall_messages[0].get("role") == "system":
            recall_messages[0] = {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt_for_frame_protocol(
                        self.frame_protocol,
                        prompt_kind="post_recall",
                        render_layout=self.render_layout,
                    ),
                }],
            }
        recall_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": self.last_output_text}],
        })
        recall_messages.append({
            "role": "user",
            "content": build_recall_result_user_content(
                recalled_frames,
                recall_result,
                frame_protocol=self.frame_protocol,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                render_layout=self.render_layout,
            ),
        })
        return recall_messages, recall_result

    def apply_recall_output(self, output_text: str, recall_messages: List[Dict]) -> None:
        result = self.last_result or {}
        rc = _parse_agent_output(output_text)
        err = action_space_error_for_turn(rc.get("action", ""), "post_recall")
        if err or rc.get("action") in ("recall", "compress"):
            result["recall_step2_blocked"] = {
                "action": rc.get("action"),
                "action_space_error": err,
                "raw_output": output_text,
            }
            rc = {"action": "silent", "payload": {}, "raw_output": ""}
        result["recall_step2"] = rc
        result["recall_step2_raw_text"] = output_text
        if rc.get("action") in ("response", "silent"):
            result["final_action"] = rc["action"]
            result["final_payload"] = rc.get("payload", {})
        if rc.get("action") == "response":
            self._record_answer(
                rc.get("payload", {}).get("response", ""),
                self.current_chunk,
            )
        _format_result_telemetry(result, self.tokenizer, recall_messages)

    def finish_step(self) -> None:
        result = self.last_result or {}
        action = result.get("action", "?")
        payload = result.get("payload") or {}
        if action == "recall":
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload") or {}
            self.per_chunk[self.current_chunk] = (
                final_action,
                final_payload.get("response", ""),
            )
        elif action == "response":
            self.per_chunk[self.current_chunk] = (
                "response",
                payload.get("response", ""),
            )
        elif action == "silent":
            self.per_chunk[self.current_chunk] = ("silent", "")
        elif action == "compress":
            self.per_chunk[self.current_chunk] = ("compress", "")
        else:
            self.per_chunk[self.current_chunk] = (action, "")

        _record_vllm_step_telemetry(self.telemetry, self.current_chunk, result, self.per_chunk)
        if (
            action == "compress"
            and result.get("compress_telemetry")
            and result.get("compress_succeeded")
        ):
            return
        self.current_chunk += 1
        if self.current_chunk > self.max_chunk:
            self.done = True

    def record_error(self, exc: Exception) -> None:
        self.per_chunk[self.current_chunk] = ("error", str(exc))
        self.telemetry["n_step_errors"] = self.telemetry.get("n_step_errors", 0) + 1
        self.telemetry.setdefault("step_errors", []).append({
            "chunk": self.current_chunk,
            "error": f"{type(exc).__name__}: {exc}",
        })
        self.telemetry["total_steps"] = self.telemetry.get("total_steps", 0) + 1
        self.current_chunk += 1
        if self.current_chunk > self.max_chunk:
            self.done = True


def _record_vllm_step_telemetry(
    telemetry: Dict,
    chunk_idx: int,
    result: Dict,
    per_chunk: Dict[int, Tuple[str, str]],
) -> None:
    ct = result.get("compress_telemetry")
    if ct:
        compressed_chunks = ct.get("compressed_chunks") or []
        telemetry.setdefault("compress_events", []).append({
            "chunk": chunk_idx,
            "thinks_count": ct["thinks_count_at_trigger"],
            "thinks_token_count": ct.get("thinks_token_count"),
            "compress_threshold": ct.get("compress_threshold", COMPRESS_TOKEN_THRESHOLD),
            "compress_range_min": ct.get("compress_range_min", COMPRESS_RANGE_MIN),
            "compress_range_max": ct.get("compress_range_max"),
            "n_compressed": len(compressed_chunks),
            "compressed_chunks": list(compressed_chunks),
            "system_trigger_rule_ok": ct.get("system_trigger_rule_ok"),
            "system_range_rule_ok": ct.get("system_range_rule_ok"),
            "succeeded": bool(result.get("compress_succeeded")),
            "partial": (0 < len(compressed_chunks) < COMPRESS_RANGE_MIN),
        })
        rev = telemetry.setdefault("compress_chunk_count", {})
        for c in compressed_chunks:
            rev[int(c)] = rev.get(int(c), 0) + 1
    if result.get("action") == "recall":
        payload = result.get("payload") or {}
        recall_args = payload.get("recall_args") or {}
        schema = (
            "with_start_end"
            if (
                isinstance(recall_args, dict)
                and recall_args.get("start_time") is not None
                and recall_args.get("end_time") is not None
            )
            else "missing_start_end"
        )
        recall_result = result.get("recall_result") or {}
        recall_metadata_chars = len(json.dumps(recall_result, ensure_ascii=False))
        telemetry.setdefault("recall_events", []).append({
            "chunk": chunk_idx,
            "returned_chunks": list(result.get("recall_returned_chunks", [])),
            "schema": schema,
            "requested_time_range": {
                "start_time": recall_args.get("start_time"),
                "end_time": recall_args.get("end_time"),
            } if isinstance(recall_args, dict) else {},
            "source": recall_result.get("source", ""),
            "result_time": recall_result.get("time", ""),
            "result_metadata_chars": recall_metadata_chars,
            "result_text_chars": 0,
        })
    if result.get("prompt_text_token_count") is not None:
        telemetry.setdefault("prompt_tokens_per_step", []).append(
            result["prompt_text_token_count"])
    if result.get("think_token_count") is not None:
        telemetry.setdefault("think_tokens_per_step", []).append(
            result["think_token_count"])
    if not result.get("format_ok", True):
        telemetry["n_format_violations"] = (
            telemetry.get("n_format_violations", 0) + 1)
    if result.get("action_space_error") or result.get("invalid_action"):
        telemetry["n_action_space_errors"] = (
            telemetry.get("n_action_space_errors", 0) + 1)
    if result.get("recall_step2_blocked"):
        telemetry["n_recall_step2_blocked"] = (
            telemetry.get("n_recall_step2_blocked", 0) + 1)
    telemetry["total_steps"] = telemetry.get("total_steps", 0) + 1
    final_action, final_response = per_chunk.get(chunk_idx, ("missing", ""))
    if result.get("action") == "recall" and telemetry.get("recall_events"):
        telemetry["recall_events"][-1]["final_action"] = final_action
        telemetry["recall_events"][-1]["final_response"] = final_response
    telemetry.setdefault("step_records", []).append({
        "chunk": chunk_idx,
        "action": result.get("action", "?"),
        "final_action": final_action,
        "response": final_response,
        "think": result.get("think", ""),
        "think_tokens": result.get("think_token_count"),
        "prompt_tokens": result.get("prompt_text_token_count"),
        "memory_tokens": result.get("memory_token_count"),
        "format_ok": bool(result.get("format_ok", True)),
        "action_space_error": result.get("action_space_error", ""),
        "invalid_action": result.get("invalid_action", ""),
        "recall_step2_blocked": bool(result.get("recall_step2_blocked")),
        "compress_succeeded": result.get("compress_succeeded"),
    })


_STABLE_PHRASE_RE = re.compile(
    r"\b(still|continues?|remain(?:s|ing)?|same|unchanged|again|"
    r"keeps?|ongoing)\b",
    re.IGNORECASE,
)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _pct(num, den):
    return float(num) / float(den) if den else 0.0


def _support_hit(chunks, intervals: Optional[List[Tuple[int, int]]]):
    if not chunks or not intervals:
        return False
    for raw_c in chunks:
        try:
            c = int(raw_c)
        except (TypeError, ValueError):
            continue
        for a, b in intervals:
            if int(a) <= c <= int(b):
                return True
    return False


def _query_range_chunks(time_window) -> List[int]:
    if not time_window:
        return []
    start = end = None
    if isinstance(time_window, dict):
        try:
            start = float(time_window.get("start_time"))
            end = float(time_window.get("end_time"))
        except (TypeError, ValueError):
            start = end = None
    if start is None or end is None or end < start:
        return []
    lo = max(0, int(start // AGENT_CHUNK_SEC))
    hi = max(lo, int(end // AGENT_CHUNK_SEC))
    return list(range(lo, hi + 1))


def _normalise_think(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or "").lower())
    text = re.sub(r"\d+(?:\.\d+)?", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_think_metrics(step_records: List[Dict]) -> Dict:
    """Heuristic drift detector for stale/stable thinks.

    A stable-think warning is not a correctness label. It flags repeated
    consecutive thoughts and continuation wording that often indicate the
    model is copying text memory instead of describing the current frames.
    """
    thinks = [
        (int(r.get("chunk", -1)), str(r.get("think") or ""))
        for r in step_records
        if str(r.get("think") or "").strip()
    ]
    if not thinks:
        return {
            "n_thinks": 0,
            "n_stable_pairs": 0,
            "n_stable_runs": 0,
            "max_stable_run": 0,
            "stable_pair_rate": 0.0,
            "continuation_phrase_rate": 0.0,
        }

    stable_pairs = []
    current_run = 1
    max_run = 1
    n_runs = 0
    prev_norm = ""
    for i, (chunk, text) in enumerate(thinks):
        norm = _normalise_think(text)
        if i > 0 and norm and prev_norm:
            sim = SequenceMatcher(None, prev_norm, norm).ratio()
            long_enough = len(norm.split()) >= 12
            if long_enough and sim >= 0.92:
                stable_pairs.append({
                    "prev_chunk": thinks[i - 1][0],
                    "chunk": chunk,
                    "similarity": round(sim, 4),
                })
                current_run += 1
            else:
                if current_run >= 3:
                    n_runs += 1
                current_run = 1
            max_run = max(max_run, current_run)
        prev_norm = norm
    if current_run >= 3:
        n_runs += 1

    continuation = sum(1 for _, text in thinks if _STABLE_PHRASE_RE.search(text))
    return {
        "n_thinks": len(thinks),
        "n_stable_pairs": len(stable_pairs),
        "n_stable_runs": n_runs,
        "max_stable_run": max_run,
        "stable_pair_rate": _pct(len(stable_pairs), max(1, len(thinks) - 1)),
        "continuation_phrase_rate": _pct(continuation, len(thinks)),
        "examples": stable_pairs[:5],
    }


def telemetry_summary(
    telemetry: Dict,
    *,
    support_intervals: Optional[List[Tuple[int, int]]] = None,
    save_step_trace: bool = False,
) -> Dict:
    steps = telemetry.get("step_records", [])
    recalls = telemetry.get("recall_events", [])
    compresses = telemetry.get("compress_events", [])
    compress_trigger_checked = [
        e for e in compresses if e.get("system_trigger_rule_ok") is not None
    ]
    compress_trigger_ok = sum(
        1 for e in compress_trigger_checked if e.get("system_trigger_rule_ok")
    )
    compress_range_checked = [
        e for e in compresses
        if e.get("system_range_rule_ok") is not None or e.get("n_compressed") is not None
    ]
    compress_range_ok = 0
    for e in compress_range_checked:
        if e.get("system_range_rule_ok") is not None:
            compress_range_ok += int(bool(e.get("system_range_rule_ok")))
            continue
        n_compressed = int(e.get("n_compressed") or 0)
        compress_range_ok += int(COMPRESS_RANGE_MIN <= n_compressed <= COMPRESS_RANGE_MAX)
    compress_calc_checked = [
        e for e in compresses
        if e.get("system_trigger_rule_ok") is not None
        and (e.get("system_range_rule_ok") is not None or e.get("n_compressed") is not None)
    ]
    compress_calc_ok = 0
    for e in compress_calc_checked:
        n_compressed = int(e.get("n_compressed") or 0)
        range_ok = (
            bool(e.get("system_range_rule_ok"))
            if e.get("system_range_rule_ok") is not None
            else COMPRESS_RANGE_MIN <= n_compressed <= COMPRESS_RANGE_MAX
        )
        compress_calc_ok += int(bool(e.get("system_trigger_rule_ok")) and range_ok)
    recall_hits = sum(
        1 for e in recalls
        if _support_hit(e.get("returned_chunks", []), support_intervals)
    )
    recall_query_range_hits = sum(
        1 for e in recalls
        if _support_hit(_query_range_chunks(e.get("requested_time_range")), support_intervals)
    )
    recall_range_lens = [
        len(_query_range_chunks(e.get("requested_time_range"))) for e in recalls
    ]
    recall_return_lens = [
        len(e.get("returned_chunks") or []) for e in recalls
    ]
    action_hist = defaultdict(int)
    for r in steps:
        action_hist[str(r.get("final_action") or r.get("action") or "unknown")] += 1

    out = {
        "total_steps": telemetry.get("total_steps", len(steps)),
        "n_step_errors": telemetry.get("n_step_errors", 0),
        "step_errors": telemetry.get("step_errors", [])[:10],
        "n_frame_cache_misses": telemetry.get("n_frame_cache_misses", 0),
        "frame_cache_misses": telemetry.get("frame_cache_misses", [])[:10],
        "n_format_violations": telemetry.get("n_format_violations", 0),
        "n_action_space_errors": telemetry.get("n_action_space_errors", 0),
        "n_recall_step2_blocked": telemetry.get("n_recall_step2_blocked", 0),
        "action_histogram": dict(action_hist),
        "n_recall_events": len(recalls),
        "n_recall_returned_nonempty": sum(
            1 for e in recalls if e.get("returned_chunks")
        ),
        "n_recall_support_hits": recall_hits,
        "recall_support_hit_rate": _pct(recall_hits, len(recalls)),
        "n_recall_query_range_support_hits": recall_query_range_hits,
        "recall_query_range_support_hit_rate": _pct(
            recall_query_range_hits, len(recalls)
        ),
        "recall_query_range_len_mean": _mean(recall_range_lens),
        "recall_returned_len_mean": _mean(recall_return_lens),
        "recall_events": recalls[:50],
        "n_compress_events": len(compresses),
        "n_compress_succeeded": sum(
            1 for e in compresses if e.get("succeeded")
        ),
        "compress_success_rate": _pct(
            sum(1 for e in compresses if e.get("succeeded")), len(compresses)
        ),
        "n_compress_system_trigger_rule_checked": len(compress_trigger_checked),
        "n_compress_system_trigger_rule_ok": compress_trigger_ok,
        "compress_system_trigger_rule_rate": _pct(
            compress_trigger_ok, len(compress_trigger_checked)
        ),
        "n_compress_system_range_rule_checked": len(compress_range_checked),
        "n_compress_system_range_rule_ok": compress_range_ok,
        "compress_system_range_rule_rate": _pct(
            compress_range_ok, len(compress_range_checked)
        ),
        "n_compress_system_calc_checked": len(compress_calc_checked),
        "n_compress_system_calc_ok": compress_calc_ok,
        "compress_system_calc_ok_rate": _pct(
            compress_calc_ok, len(compress_calc_checked)
        ),
        "compress_thinks_at_trigger": [
            e.get("thinks_count") for e in compresses
        ],
        "compress_chunks_per_event": [
            e.get("n_compressed") for e in compresses
        ],
        "prompt_tokens_mean": _mean(telemetry.get("prompt_tokens_per_step", [])),
        "prompt_tokens_max": max(telemetry.get("prompt_tokens_per_step", []) or [0]),
        "think_tokens_mean": _mean(telemetry.get("think_tokens_per_step", [])),
        "think_tokens_max": max(telemetry.get("think_tokens_per_step", []) or [0]),
        "stable_think": stable_think_metrics(steps),
    }
    if save_step_trace:
        trace = []
        for r in steps:
            rr = dict(r)
            if "think" in rr:
                rr["think"] = rr["think"][:500]
            if "response" in rr:
                rr["response"] = rr["response"][:500]
            trace.append(rr)
        out["step_trace"] = trace
    return out


def merge_telemetry_summaries(items: List[Dict]) -> Dict:
    """Merge per-probe telemetry summaries for expanded SSR samples."""
    if not items:
        return {}
    keys = [
        "total_steps",
        "n_step_errors",
        "n_frame_cache_misses",
        "n_format_violations",
        "n_action_space_errors",
        "n_recall_step2_blocked",
        "n_recall_events",
        "n_recall_returned_nonempty",
        "n_recall_support_hits",
        "n_recall_query_range_support_hits",
        "n_compress_events",
        "n_compress_succeeded",
        "n_compress_system_trigger_rule_checked",
        "n_compress_system_trigger_rule_ok",
        "n_compress_system_range_rule_checked",
        "n_compress_system_range_rule_ok",
        "n_compress_system_calc_checked",
        "n_compress_system_calc_ok",
    ]
    out = {key: sum(int(item.get(key, 0) or 0) for item in items) for key in keys}
    out["step_errors"] = [
        e for item in items for e in item.get("step_errors", [])
    ][:10]
    out["frame_cache_misses"] = [
        e for item in items for e in item.get("frame_cache_misses", [])
    ][:10]
    hist = Counter()
    for item in items:
        hist.update(item.get("action_histogram") or {})
    out["action_histogram"] = dict(hist)
    out["recall_support_hit_rate"] = _pct(
        out["n_recall_support_hits"], out["n_recall_events"]
    )
    out["recall_query_range_support_hit_rate"] = _pct(
        out["n_recall_query_range_support_hits"], out["n_recall_events"]
    )
    out["compress_success_rate"] = _pct(
        out["n_compress_succeeded"], out["n_compress_events"]
    )
    out["compress_system_trigger_rule_rate"] = _pct(
        out["n_compress_system_trigger_rule_ok"],
        out["n_compress_system_trigger_rule_checked"],
    )
    out["compress_system_range_rule_rate"] = _pct(
        out["n_compress_system_range_rule_ok"],
        out["n_compress_system_range_rule_checked"],
    )
    out["compress_system_calc_ok_rate"] = _pct(
        out["n_compress_system_calc_ok"],
        out["n_compress_system_calc_checked"],
    )
    out["prompt_tokens_max"] = max((item.get("prompt_tokens_max", 0) or 0) for item in items)
    out["think_tokens_max"] = max((item.get("think_tokens_max", 0) or 0) for item in items)
    out["stable_think"] = {
        "n_stable_pairs": sum(
            int((item.get("stable_think") or {}).get("n_stable_pairs", 0) or 0)
            for item in items
        )
    }
    return out


def recall_events_between(telemetry: Dict, since_chunk: int, until_chunk: int) -> List[Dict]:
    return [
        e for e in telemetry.get("recall_events", [])
        if int(since_chunk) <= int(e.get("chunk", -1)) <= int(until_chunk)
    ]


# ─── Per-task evaluators ─────────────────────────────────────────────────────


LENIENT_MAX_EXTRA_CHUNKS = 60   # 120s past ask_chunk; covers most BT/RT videos


def _time_to_chunk(t) -> int:
    return int(float(t) / AGENT_CHUNK_SEC)


def _interval_from_times(start, end=None) -> Tuple[int, int]:
    a = _time_to_chunk(start)
    b = _time_to_chunk(start if end is None else end)
    if b < a:
        a, b = b, a
    return a, b


def support_intervals_for_sample(sample, task=None) -> List[Tuple[int, int]]:
    """Best-effort OVO support intervals for recall diagnostics.

    OVO does not expose dense support labels for every task. These intervals
    are used only to classify recall ranges as plausibly useful, not for
    accuracy scoring.
    """
    task = task or sample.get("task")
    if task == "REC":
        return [
            _interval_from_times(s, e)
            for s, e in zip(sample.get("start_times", []), sample.get("end_times", []))
        ]
    if task == "SSR":
        return [
            _interval_from_times(s, e)
            for s, e in zip(sample.get("start_time", []), sample.get("end_time", []))
        ]
    if task == "CRR" and sample.get("clue_time") is not None:
        c = _time_to_chunk(sample["clue_time"])
        return [(c, c)]
    if sample.get("realtime") is not None:
        c = _time_to_chunk(sample["realtime"])
        return [(c, c)]
    return []


def _probe_chunk(probe: Dict) -> int:
    return _time_to_chunk(probe.get("realtime", 0))


def _rec_query_meta(sample: Dict) -> Dict:
    """REC is one question with many expected running-count answers."""
    per_emit = []
    chunks = []
    for probe in sample.get("test_info", []) or []:
        c = _probe_chunk(probe)
        chunks.append(c)
        per_emit.append({"chunk": c, "value": str(int(probe.get("count", 0)))})
    chunks = sorted(set(chunks))
    meta = {
        "answer_form": "number",
        "answer_chunks": chunks,
        "per_emit_answers": per_emit,
    }
    if chunks:
        meta["open_until"] = max(chunks) * AGENT_CHUNK_SEC
    return meta


def _crr_query_meta(sample: Dict) -> Dict:
    """CRR is a persistent Yes/No probe over all test_info checkpoints."""
    per_emit = []
    chunks = []
    for probe in sample.get("test_info", []) or []:
        c = _probe_chunk(probe)
        chunks.append(c)
        per_emit.append({
            "chunk": c,
            "value": "Yes" if int(probe.get("type", 0) or 0) == 1 else "No",
        })
    chunks = sorted(set(chunks))
    meta = {
        "answer_form": "binary",
        "answer_chunks": chunks,
        "per_emit_answers": per_emit,
    }
    if chunks:
        meta["open_until"] = max(chunks) * AGENT_CHUNK_SEC
    return meta


def _ssr_query_meta(chunk: int, probe: Dict) -> Dict:
    gt = "Yes" if int(probe.get("type", 0) or 0) == 1 else "No"
    return {
        "answer_form": "binary",
        "answer_chunks": [int(chunk)],
        "per_emit_answers": [{"chunk": int(chunk), "value": gt}],
        "open_until": int(chunk) * AGENT_CHUNK_SEC,
    }


def attach_probe_recall_fields(
    probe: Dict,
    telemetry: Dict,
    *,
    since_chunk: int,
    until_chunk: int,
    support_intervals: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    events = recall_events_between(telemetry, since_chunk, until_chunk)
    probe["used_recall_before_response"] = bool(events)
    probe["n_recall_before_response"] = len(events)
    probe["recall_returned_chunks_before_response"] = [
        e.get("returned_chunks", []) for e in events
    ]
    probe["recall_support_hit_before_response"] = any(
        _support_hit(e.get("returned_chunks", []), support_intervals)
        for e in events
    )
    probe["recall_query_range_hit_before_response"] = any(
        _support_hit(_query_range_chunks(e.get("requested_time_range")), support_intervals)
        for e in events
    )
    probe["recall_returned_hit_before_response"] = probe[
        "recall_support_hit_before_response"
    ]
    return probe


def eval_mcq(sample, loop, retriever, video_root, scoring="strict",
             save_step_trace=False):
    """BT / RT: single-realtime MCQ. Inject question at ask_chunk, score the
    first response.

    scoring="strict":  walk to ask_chunk + 2; only count responses inside
                       that window. Original OVO timing-strict behaviour.
    scoring="lenient": walk to ask_chunk + 60 (= 120s); count any response
                       at any later chunk. "If the model ever answered,
                       count it" — measures format/correctness independent
                       of response-timing. Note: this gives the model up to
                       60 extra chunks of free observation, so it's an
                       upper-bound metric, not directly comparable to OVO
                       paper numbers.
    """
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    realtime = float(sample["realtime"])
    ask_chunk = int(realtime / AGENT_CHUNK_SEC)
    extra = LENIENT_MAX_EXTRA_CHUNKS if scoring == "lenient" else 2
    max_chunk = ask_chunk + extra

    question = build_mcq_agent_question(sample)
    q_meta = {ask_chunk: build_mcq_query_meta(sample)}
    loop.reset()
    reset_visual_index(retriever)
    telemetry: Dict = {}
    per_chunk = run_agent(loop, video_path, {ask_chunk: question}, max_chunk,
                          telemetry=telemetry, ask_meta=q_meta)

    # Find first response at or after ask_chunk. Premature responses
    # (chunk < ask_chunk) are flagged in telemetry but not used as the answer.
    pred_letter = None
    response_chunk = None
    n_premature = sum(1 for c in range(0, ask_chunk)
                      if per_chunk.get(c, ("?", ""))[0] == "response")
    for c in range(ask_chunk, max_chunk + 1):
        action, resp = per_chunk.get(c, ("missing", ""))
        if action == "response" and resp:
            pred_letter = extract_letter(resp)
            response_chunk = c
            break

    gt_idx = sample["gt"]
    gt_letter = chr(65 + gt_idx)
    correct = pred_letter == gt_letter

    support_intervals = support_intervals_for_sample(sample, sample["task"])
    probe_out = {
        "realtime": realtime,
        "ask_chunk": ask_chunk,
        "probe_chunk": ask_chunk,
        "response_chunk": response_chunk,
        "response_offset_chunks": (response_chunk - ask_chunk
                                   if response_chunk is not None else None),
        "response_to_probe_offset_chunks": (
            response_chunk - ask_chunk if response_chunk is not None else None
        ),
        "gt": gt_letter,
        "pred": pred_letter,
        "correct": correct,
        "targeted_correct": correct,
        "strict_correct": correct and response_chunk == ask_chunk,
    }
    attach_probe_recall_fields(
        probe_out,
        telemetry,
        since_chunk=ask_chunk,
        until_chunk=response_chunk if response_chunk is not None else max_chunk,
        support_intervals=support_intervals,
    )

    return {
        "task": sample["task"],
        "id": sample.get("id"),
        "probes": [probe_out],
        "telemetry": {
            **telemetry_summary(
                telemetry,
                support_intervals=support_intervals,
                save_step_trace=save_step_trace,
            ),
            "n_premature_responses": n_premature,
        },
    }


def eval_rec(sample, loop, retriever, video_root, save_step_trace=False):
    """FT-REC: cumulative integer count. Inject question at chunk 0,
    score at each test_info probe."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    test_info = sample["test_info"]
    last_probe = max(float(t["realtime"]) for t in test_info)
    max_chunk = int(last_probe / AGENT_CHUNK_SEC)

    question = build_rec_question(sample)
    loop.reset()
    reset_visual_index(retriever)
    telemetry: Dict = {}
    per_chunk = run_agent(
        loop,
        video_path,
        {0: question},
        max_chunk,
        telemetry=telemetry,
        ask_meta={0: _rec_query_meta(sample)},
    )

    probes = []
    support_intervals = support_intervals_for_sample(sample, "REC")
    for probe in test_info:
        t = float(probe["realtime"])
        gt_count = int(probe["count"])
        c = int(t / AGENT_CHUNK_SEC)
        response_chunk, resp = latest_response_at_or_before(per_chunk, c, 0)
        pred_count = extract_int(resp)
        probe_out = {
            "realtime": t,
            "chunk_idx": c,
            "ask_chunk": 0,
            "probe_chunk": c,
            "response_chunk": response_chunk,
            "response_offset_chunks": (
                response_chunk if response_chunk is not None else None
            ),
            "response_to_probe_offset_chunks": (
                response_chunk - c if response_chunk is not None else None
            ),
            "gt": gt_count,
            "pred": pred_count,
            "correct": pred_count == gt_count,
            "targeted_correct": pred_count == gt_count,
            "strict_correct": (
                pred_count == gt_count and response_chunk == c
            ),
            "count_abs_error": abs(pred_count - gt_count) if pred_count is not None else None,
        }
        attach_probe_recall_fields(
            probe_out,
            telemetry,
            since_chunk=0,
            until_chunk=response_chunk if response_chunk is not None else c,
            support_intervals=support_intervals,
        )
        probes.append(probe_out)
    return {
        "task": "REC",
        "id": sample.get("id"),
        "probes": probes,
        "telemetry": telemetry_summary(
            telemetry,
            support_intervals=support_intervals,
            save_step_trace=save_step_trace,
        ),
    }


def eval_ssr(sample, loop, retriever, video_root, save_step_trace=False):
    """FT-SSR: each test_info entry asks about a specific step at a specific
    time. Treat each as independent (inject fresh question per probe)."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    test_info = sample["test_info"]
    probes = []
    telemetry_parts = []
    support_intervals = support_intervals_for_sample(sample, "SSR")
    for probe_i, probe in enumerate(test_info):
        t = float(probe["realtime"])
        c = int(t / AGENT_CHUNK_SEC)
        ptype = probe["type"]
        loop.reset()
        reset_visual_index(retriever)
        telemetry: Dict = {}
        per_chunk = run_agent(
            loop,
            video_path,
            {c: build_ssr_question(probe.get("step", ""))},
            c,
            telemetry=telemetry,
            ask_meta={c: _ssr_query_meta(c, probe)},
        )
        action, resp = per_chunk.get(c, ("missing", ""))
        said_yes = is_yes(resp) if action == "response" else False
        said_no = is_no(resp) if action == "response" else False
        was_silent = action == "silent"
        if ptype == 0:
            strict = said_no
            lenient = was_silent or said_no
            fp = said_yes
        else:
            strict = said_yes
            lenient = said_yes
            fp = False
        probe_out = {
            "realtime": t,
            "chunk_idx": c,
            "ask_chunk": c,
            "type": ptype,
            "probe_index": probe_i,
            "action": action,
            "response": resp,
            "response_chunk": c if action == "response" and resp else None,
            "response_offset_chunks": 0 if action == "response" and resp else None,
            "response_to_probe_offset_chunks": 0 if action == "response" and resp else None,
            "strict_correct": strict,
            "lenient_correct": lenient,
            "targeted_correct": lenient,
            "false_positive": fp,
        }
        attach_probe_recall_fields(
            probe_out,
            telemetry,
            since_chunk=c,
            until_chunk=c,
            support_intervals=support_intervals,
        )
        probes.append(probe_out)
        telemetry_parts.append(telemetry_summary(
            telemetry,
            support_intervals=support_intervals,
            save_step_trace=save_step_trace,
        ))
    return {
        "task": "SSR",
        "id": sample.get("id"),
        "probes": probes,
        "telemetry": merge_telemetry_summaries(telemetry_parts),
    }


def eval_crr(sample, loop, retriever, video_root, save_step_trace=False):
    """FT-CRR: ask once at ask_time; probe at each test_info time. Delay-sensitive."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    ask_time = float(sample["ask_time"])
    test_info = sample["test_info"]
    last_probe = max(float(t["realtime"]) for t in test_info)
    ask_chunk = int(ask_time / AGENT_CHUNK_SEC)
    max_chunk = int(last_probe / AGENT_CHUNK_SEC)

    question = build_crr_question(sample)
    loop.reset()
    reset_visual_index(retriever)
    telemetry: Dict = {}
    per_chunk = run_agent(
        loop,
        video_path,
        {ask_chunk: question},
        max_chunk,
        telemetry=telemetry,
        ask_meta={ask_chunk: _crr_query_meta(sample)},
    )

    probes = []
    support_intervals = support_intervals_for_sample(sample, "CRR")
    for probe in test_info:
        t = float(probe["realtime"])
        ptype = probe["type"]
        c = int(t / AGENT_CHUNK_SEC)
        exact_action, _ = per_chunk.get(c, ("missing", ""))
        response_chunk, resp = latest_response_at_or_before(
            per_chunk, c, since_chunk=ask_chunk,
        )
        action = "response" if response_chunk is not None else exact_action
        said_yes = is_yes(resp) if response_chunk is not None else False
        said_no = is_no(resp) if response_chunk is not None else False
        was_silent = exact_action == "silent" and response_chunk is None
        fp_chunk, fp_resp = first_yes_response_between(per_chunk, ask_chunk, c)
        had_early_yes = fp_chunk is not None
        if ptype == 0:
            strict = said_no and not had_early_yes
            lenient = (was_silent or said_no) and not had_early_yes
            fp = had_early_yes
        else:
            strict = said_yes
            lenient = said_yes
            fp = False
        probe_out = {
            "realtime": t,
            "chunk_idx": c,
            "ask_chunk": ask_chunk,
            "type": ptype,
            "action": action,
            "exact_action": exact_action,
            "response": resp,
            "response_chunk": response_chunk,
            "response_offset_chunks": (
                response_chunk - ask_chunk if response_chunk is not None else None
            ),
            "response_to_probe_offset_chunks": (
                response_chunk - c if response_chunk is not None else None
            ),
            "strict_correct": strict,
            "lenient_correct": lenient,
            "targeted_correct": lenient,
            "false_positive": fp,
            "false_positive_chunk": fp_chunk,
            "false_positive_response": fp_resp,
        }
        attach_probe_recall_fields(
            probe_out,
            telemetry,
            since_chunk=ask_chunk,
            until_chunk=response_chunk if response_chunk is not None else c,
            support_intervals=support_intervals,
        )
        probes.append(probe_out)
    return {
        "task": "CRR", "id": sample.get("id"),
        "ask_time": ask_time, "clue_time": sample.get("clue_time"),
        "probes": probes,
        "telemetry": telemetry_summary(
            telemetry,
            support_intervals=support_intervals,
            save_step_trace=save_step_trace,
        ),
    }


def dispatch_eval(sample, loop, retriever, video_root, scoring="strict",
                  save_step_trace=False):
    """Dispatch to per-task evaluator. Lenient `scoring` only changes MCQ
    timing — REC/SSR/CRR have intrinsic timing semantics (cumulative count
    / per-step state / clue-reveal delay) that don't have a meaningful
    'time-agnostic' interpretation, so they ignore the flag and always run
    in their natural mode."""
    task = sample.get("task")
    if task in BT_TASKS or task in RT_TASKS:
        return eval_mcq(
            sample, loop, retriever, video_root,
            scoring=scoring, save_step_trace=save_step_trace,
        )
    if task == "REC":
        return eval_rec(sample, loop, retriever, video_root,
                        save_step_trace=save_step_trace)
    if task == "SSR":
        return eval_ssr(sample, loop, retriever, video_root,
                        save_step_trace=save_step_trace)
    if task == "CRR":
        return eval_crr(sample, loop, retriever, video_root,
                        save_step_trace=save_step_trace)
    return None


def build_agent_job(sample, video_root, scoring="strict"):
    task = sample.get("task")
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    if task in BT_TASKS or task in RT_TASKS:
        realtime = float(sample["realtime"])
        ask_chunk = int(realtime / AGENT_CHUNK_SEC)
        extra = LENIENT_MAX_EXTRA_CHUNKS if scoring == "lenient" else 2
        return {
            "task": task,
            "id": sample.get("id"),
            "sample": sample,
            "video_path": video_path,
            "ask_chunks": {ask_chunk: build_mcq_agent_question(sample)},
            "ask_meta": {ask_chunk: build_mcq_query_meta(sample)},
            "max_chunk": ask_chunk + extra,
            "kind": "mcq",
        }

    if task == "REC":
        last_probe = max(float(t["realtime"]) for t in sample["test_info"])
        return {
            "task": "REC",
            "id": sample.get("id"),
            "sample": sample,
            "video_path": video_path,
            "ask_chunks": {0: build_rec_question(sample)},
            "ask_meta": {0: _rec_query_meta(sample)},
            "max_chunk": int(last_probe / AGENT_CHUNK_SEC),
            "kind": "rec",
        }

    if task == "SSR":
        ask_chunks = {}
        ask_meta = {}
        last_probe = max(float(t["realtime"]) for t in sample["test_info"])
        for probe in sample["test_info"]:
            c = int(float(probe["realtime"]) / AGENT_CHUNK_SEC)
            ask_chunks[c] = build_ssr_question(probe.get("step", ""))
            ask_meta[c] = _ssr_query_meta(c, probe)
        return {
            "task": "SSR",
            "id": sample.get("id"),
            "sample": sample,
            "video_path": video_path,
            "ask_chunks": ask_chunks,
            "ask_meta": ask_meta,
            "max_chunk": int(last_probe / AGENT_CHUNK_SEC) + 1,
            "kind": "ssr",
        }

    if task == "CRR":
        ask_time = float(sample["ask_time"])
        ask_chunk = int(ask_time / AGENT_CHUNK_SEC)
        last_probe = max(float(t["realtime"]) for t in sample["test_info"])
        return {
            "task": "CRR",
            "id": sample.get("id"),
            "sample": sample,
            "video_path": video_path,
            "ask_chunks": {ask_chunk: build_crr_question(sample)},
            "ask_meta": {ask_chunk: _crr_query_meta(sample)},
            "max_chunk": int(last_probe / AGENT_CHUNK_SEC),
            "kind": "crr",
            "ask_time": ask_time,
        }

    return None


def build_agent_jobs_for_sample(sample, video_root, scoring="strict") -> List[Dict]:
    """Build one or more vLLM trajectory jobs for an OVO sample.

    SSR follows the Streamo benchmark script's expansion strategy: each
    step/probe is its own full-prefix trajectory, because several steps can
    share the same realtime chunk and a single active query cannot represent
    multiple distinct step questions at once.
    """
    if sample.get("task") != "SSR":
        job = build_agent_job(sample, video_root, scoring=scoring)
        return [job] if job is not None else []

    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return []
    jobs = []
    for probe_i, probe in enumerate(sample.get("test_info", []) or []):
        c = int(float(probe["realtime"]) / AGENT_CHUNK_SEC)
        jobs.append({
            "task": "SSR",
            "id": f"{sample.get('id')}:{probe_i}",
            "sample": sample,
            "video_path": video_path,
            "ask_chunks": {c: build_ssr_question(probe.get("step", ""))},
            "ask_meta": {c: _ssr_query_meta(c, probe)},
            "max_chunk": c,
            "kind": "ssr_probe",
            "probe_index": probe_i,
            "probe": probe,
        })
    return jobs


def job_frame_cache_complete(
    job: Dict,
    frames_root: Optional[str],
    video_root: Optional[str],
) -> Tuple[bool, Optional[Dict]]:
    """Return whether all streaming chunks can use pre-extracted frames."""
    if not frames_root:
        return True, None
    max_chunk = int(job.get("max_chunk", -1))
    source_fpc = _effective_source_frames_per_chunk(
        str(job["video_path"]),
        frames_root,
        video_root,
        max_chunk_hint=max_chunk,
    )
    offsets = _selected_source_frame_offsets(source_fpc)
    max_offset = max(offsets) if offsets else 0
    needed_frame_no = max_chunk * source_fpc + max_offset + 1
    max_frame_no = _frame_cache_max_number(
        str(job["video_path"]),
        str(frames_root or ""),
        str(video_root or ""),
    )
    if max_frame_no < needed_frame_no:
        first_missing_chunk = max(0, int((max_frame_no - max_offset) // source_fpc))
        return False, {
            "task": job.get("task"),
            "id": job.get("id"),
            "video_path": job.get("video_path"),
            "chunk": int(first_missing_chunk),
            "max_chunk": max_chunk,
            "max_frame_no": int(max_frame_no),
            "needed_frame_no": int(needed_frame_no),
        }
    return True, None


def score_agent_job(job, per_chunk, telemetry, save_step_trace=False):
    sample = job["sample"]
    task = job["task"]

    if job["kind"] == "mcq":
        realtime = float(sample["realtime"])
        ask_chunk = int(realtime / AGENT_CHUNK_SEC)
        max_chunk = job["max_chunk"]
        pred_letter = None
        response_chunk = None
        n_premature = sum(
            1 for c in range(0, ask_chunk)
            if per_chunk.get(c, ("?", ""))[0] == "response"
        )
        for c in range(ask_chunk, max_chunk + 1):
            action, resp = per_chunk.get(c, ("missing", ""))
            if action == "response" and resp:
                pred_letter = extract_letter(resp)
                response_chunk = c
                break
        gt_letter = chr(65 + int(sample["gt"]))
        support_intervals = support_intervals_for_sample(sample, task)
        probe_out = {
            "realtime": realtime,
            "ask_chunk": ask_chunk,
            "probe_chunk": ask_chunk,
            "response_chunk": response_chunk,
            "response_offset_chunks": (
                response_chunk - ask_chunk if response_chunk is not None else None
            ),
            "response_to_probe_offset_chunks": (
                response_chunk - ask_chunk if response_chunk is not None else None
            ),
            "gt": gt_letter,
            "pred": pred_letter,
            "correct": pred_letter == gt_letter,
            "targeted_correct": pred_letter == gt_letter,
            "strict_correct": (
                pred_letter == gt_letter and response_chunk == ask_chunk
            ),
        }
        attach_probe_recall_fields(
            probe_out,
            telemetry,
            since_chunk=ask_chunk,
            until_chunk=response_chunk if response_chunk is not None else max_chunk,
            support_intervals=support_intervals,
        )
        return {
            "task": task,
            "id": sample.get("id"),
            "probes": [probe_out],
            "telemetry": {
                **telemetry_summary(
                    telemetry,
                    support_intervals=support_intervals,
                    save_step_trace=save_step_trace,
                ),
                "n_premature_responses": n_premature,
            },
        }

    if job["kind"] == "rec":
        probes = []
        support_intervals = support_intervals_for_sample(sample, "REC")
        for probe in sample["test_info"]:
            t = float(probe["realtime"])
            c = int(t / AGENT_CHUNK_SEC)
            response_chunk, resp = latest_response_at_or_before(per_chunk, c, 0)
            pred_count = extract_int(resp)
            probe_out = {
                "realtime": t,
                "chunk_idx": c,
                "ask_chunk": 0,
                "probe_chunk": c,
                "response_chunk": response_chunk,
                "response_offset_chunks": (
                    response_chunk if response_chunk is not None else None
                ),
                "response_to_probe_offset_chunks": (
                    response_chunk - c if response_chunk is not None else None
                ),
                "gt": int(probe["count"]),
                "pred": pred_count,
                "correct": pred_count == int(probe["count"]),
                "targeted_correct": pred_count == int(probe["count"]),
                "strict_correct": (
                    pred_count == int(probe["count"]) and response_chunk == c
                ),
                "count_abs_error": (
                    abs(pred_count - int(probe["count"]))
                    if pred_count is not None else None
                ),
            }
            attach_probe_recall_fields(
                probe_out,
                telemetry,
                since_chunk=0,
                until_chunk=response_chunk if response_chunk is not None else c,
                support_intervals=support_intervals,
            )
            probes.append(probe_out)
        return {
            "task": "REC",
            "id": sample.get("id"),
            "probes": probes,
            "telemetry": telemetry_summary(
                telemetry,
                support_intervals=support_intervals,
                save_step_trace=save_step_trace,
            ),
        }

    if job["kind"] == "ssr_probe":
        probe = job["probe"]
        t = float(probe["realtime"])
        c = int(t / AGENT_CHUNK_SEC)
        ptype = probe["type"]
        support_intervals = support_intervals_for_sample(sample, "SSR")
        action, resp = per_chunk.get(c, ("missing", ""))
        said_yes = is_yes(resp) if action == "response" else False
        said_no = is_no(resp) if action == "response" else False
        was_silent = action == "silent"
        if ptype == 0:
            strict = said_no
            lenient = was_silent or said_no
            fp = said_yes
        else:
            strict = said_yes
            lenient = said_yes
            fp = False
        probe_out = {
            "realtime": t,
            "chunk_idx": c,
            "ask_chunk": c,
            "type": ptype,
            "probe_index": job.get("probe_index"),
            "action": action,
            "response": resp,
            "response_chunk": c if action == "response" and resp else None,
            "response_offset_chunks": 0 if action == "response" and resp else None,
            "response_to_probe_offset_chunks": 0 if action == "response" and resp else None,
            "strict_correct": strict,
            "lenient_correct": lenient,
            "targeted_correct": lenient,
            "false_positive": fp,
        }
        attach_probe_recall_fields(
            probe_out,
            telemetry,
            since_chunk=c,
            until_chunk=c,
            support_intervals=support_intervals,
        )
        return {
            "task": "SSR",
            "id": job.get("id"),
            "source_id": sample.get("id"),
            "probes": [probe_out],
            "telemetry": telemetry_summary(
                telemetry,
                support_intervals=support_intervals,
                save_step_trace=save_step_trace,
            ),
        }

    if job["kind"] == "ssr":
        probes = []
        support_intervals = support_intervals_for_sample(sample, "SSR")
        for probe in sample["test_info"]:
            t = float(probe["realtime"])
            c = int(t / AGENT_CHUNK_SEC)
            ptype = probe["type"]
            action, resp = per_chunk.get(c, ("missing", ""))
            said_yes = is_yes(resp) if action == "response" else False
            said_no = is_no(resp) if action == "response" else False
            was_silent = action == "silent"
            if ptype == 0:
                strict = said_no
                lenient = was_silent or said_no
                fp = said_yes
            else:
                strict = said_yes
                lenient = said_yes
                fp = False
            probe_out = {
                "realtime": t,
                "chunk_idx": c,
                "ask_chunk": c,
                "type": ptype,
                "action": action,
                "response": resp,
                "response_chunk": c if action == "response" and resp else None,
                "response_offset_chunks": 0 if action == "response" and resp else None,
                "response_to_probe_offset_chunks": 0 if action == "response" and resp else None,
                "strict_correct": strict,
                "lenient_correct": lenient,
                "targeted_correct": lenient,
                "false_positive": fp,
            }
            attach_probe_recall_fields(
                probe_out,
                telemetry,
                since_chunk=c,
                until_chunk=c,
                support_intervals=support_intervals,
            )
            probes.append(probe_out)
        return {
            "task": "SSR",
            "id": sample.get("id"),
            "probes": probes,
            "telemetry": telemetry_summary(
                telemetry,
                support_intervals=support_intervals,
                save_step_trace=save_step_trace,
            ),
        }

    if job["kind"] == "crr":
        probes = []
        ask_chunk = int(float(sample["ask_time"]) / AGENT_CHUNK_SEC)
        support_intervals = support_intervals_for_sample(sample, "CRR")
        for probe in sample["test_info"]:
            t = float(probe["realtime"])
            ptype = probe["type"]
            c = int(t / AGENT_CHUNK_SEC)
            exact_action, _ = per_chunk.get(c, ("missing", ""))
            response_chunk, resp = latest_response_at_or_before(
                per_chunk, c, since_chunk=ask_chunk,
            )
            action = "response" if response_chunk is not None else exact_action
            said_yes = is_yes(resp) if response_chunk is not None else False
            said_no = is_no(resp) if response_chunk is not None else False
            was_silent = exact_action == "silent" and response_chunk is None
            fp_chunk, fp_resp = first_yes_response_between(per_chunk, ask_chunk, c)
            had_early_yes = fp_chunk is not None
            if ptype == 0:
                strict = said_no and not had_early_yes
                lenient = (was_silent or said_no) and not had_early_yes
                fp = had_early_yes
            else:
                strict = said_yes
                lenient = said_yes
                fp = False
            probe_out = {
                "realtime": t,
                "chunk_idx": c,
                "ask_chunk": ask_chunk,
                "type": ptype,
                "action": action,
                "exact_action": exact_action,
                "response": resp,
                "response_chunk": response_chunk,
                "response_offset_chunks": (
                    response_chunk - ask_chunk if response_chunk is not None else None
                ),
                "response_to_probe_offset_chunks": (
                    response_chunk - c if response_chunk is not None else None
                ),
                "strict_correct": strict,
                "lenient_correct": lenient,
                "targeted_correct": lenient,
                "false_positive": fp,
                "false_positive_chunk": fp_chunk,
                "false_positive_response": fp_resp,
            }
            attach_probe_recall_fields(
                probe_out,
                telemetry,
                since_chunk=ask_chunk,
                until_chunk=response_chunk if response_chunk is not None else c,
                support_intervals=support_intervals,
            )
            probes.append(probe_out)
        return {
            "task": "CRR",
            "id": sample.get("id"),
            "ask_time": float(sample["ask_time"]),
            "clue_time": sample.get("clue_time"),
            "probes": probes,
            "telemetry": telemetry_summary(
                telemetry,
                support_intervals=support_intervals,
                save_step_trace=save_step_trace,
            ),
        }

    return None


def run_agent_jobs_vllm(
    jobs: List[Dict],
    *,
    llm,
    processor,
    tokenizer,
    args,
) -> List[Dict]:
    from thinkstream.eval.vllm_engine import (
        generate_with_turn_sampling,
        make_sampling_params,
        prepare_vllm_input,
    )

    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = normalize_render_layout(args.render_layout)
    retriever_template = _build_retriever_template(args)
    runners = [
        _VllmAgentRunner(
            idx=i,
            task=job["task"],
            sample_id=job.get("id"),
            video_path=job["video_path"],
            ask_chunks=dict(job["ask_chunks"]),
            ask_meta=dict(job["ask_meta"]),
            max_chunk=int(job["max_chunk"]),
            tokenizer=tokenizer,
            retriever=_clone_retriever(retriever_template),
            frames_root=args.frames_root,
            video_root=args.video_root,
            frame_protocol=frame_protocol,
            render_layout=render_layout,
            compress_mode="off" if args.compress_mode == "none" else args.compress_mode,
            memory_mode=args.memory_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            require_frame_cache=bool(args.require_frame_cache),
        )
        for i, job in enumerate(jobs)
    ]

    sampling = make_sampling_params(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=1,
        repetition_penalty=args.vllm_repetition_penalty,
    )
    compress_sampling = make_sampling_params(
        max_new_tokens=args.compress_max_new_tokens,
        temperature=0.0,
        top_k=1,
        repetition_penalty=args.vllm_repetition_penalty,
    )

    batch_size = max(1, int(args.rollout_batch_size))
    t0 = time.time()
    step_round = 0
    while True:
        live = [r for r in runners if not r.done]
        if not live:
            break
        active = live[:batch_size]
        prepared = []
        for r in active:
            try:
                messages, turn_kind = r.prepare()
                prepared.append((r, messages, turn_kind))
            except FrameCacheMissError as exc:
                r.record_error(exc)
                r.done = True
            except Exception as exc:
                r.record_error(exc)
        if not prepared:
            continue
        try:
            inputs = [
                prepare_vllm_input(m, processor, tools=tools_for_turn(kind))
                for _, m, kind in prepared
            ]
            outs = generate_with_turn_sampling(
                llm,
                inputs,
                [kind for _, _, kind in prepared],
                sampling,
                {"compress": compress_sampling},
            )
        except Exception as exc:
            for r, _, _ in prepared:
                r.record_error(exc)
            continue

        recall_items = []
        for (r, messages, _), out in zip(prepared, outs):
            try:
                text = out.outputs[0].text
                result = r.apply_first_output(text)
                if result.get("action") == "recall":
                    recall_messages, _ = r.build_recall_messages()
                    if recall_messages is not None:
                        recall_items.append((r, recall_messages))
                    else:
                        r.finish_step()
                else:
                    r.finish_step()
            except Exception as exc:
                r.record_error(exc)

        if recall_items:
            try:
                rc_inputs = [
                    prepare_vllm_input(
                        m,
                        processor,
                        tools=tools_for_turn("post_recall"),
                    )
                    for _, m in recall_items
                ]
                rc_outs = llm.generate(rc_inputs, sampling_params=sampling)
            except Exception as exc:
                for r, _ in recall_items:
                    r.record_error(exc)
            else:
                for (r, recall_messages), rc_out in zip(recall_items, rc_outs):
                    try:
                        r.apply_recall_output(rc_out.outputs[0].text, recall_messages)
                        r.finish_step()
                    except Exception as exc:
                        r.record_error(exc)

        step_round += 1
        if args.progress_every and step_round % int(args.progress_every) == 0:
            done = sum(1 for r in runners if r.done)
            active_chunks = sum(r.current_chunk for r in runners)
            rate = active_chunks / max(1e-6, time.time() - t0)
            print(
                f"[vllm] done={done}/{len(runners)} "
                f"runner_chunks={active_chunks} rate={rate:.1f} chunks/s",
                flush=True,
            )

    results = []
    for job, runner in zip(jobs, runners):
        scored = score_agent_job(
            job,
            runner.per_chunk,
            runner.telemetry,
            save_step_trace=args.save_step_trace,
        )
        if scored is not None:
            results.append(scored)
    return results


# ─── Reporting ───────────────────────────────────────────────────────────────


def aggregate(results):
    """Build per-task / per-category / overall accuracy. Matches OVO Table 2."""
    by_task = defaultdict(lambda: {"n": 0, "correct": 0,
                                    "strict_metric_correct": 0,
                                    "targeted_correct": 0,
                                    "fp_n": 0, "fp": 0,
                                    "type0_n": 0, "type0_strict": 0, "type0_lenient": 0,
                                    "type1_n": 0, "type1_strict": 0, "type1_lenient": 0,
                                    "with_recall_n": 0, "with_recall_correct": 0,
                                    "without_recall_n": 0, "without_recall_correct": 0,
                                    "with_query_range_hit_recall_n": 0,
                                    "with_query_range_hit_recall_correct": 0,
                                    "with_returned_hit_recall_n": 0,
                                    "with_returned_hit_recall_correct": 0,
                                    "no_early_correct": 0,
                                    "no_late_correct": 0,
                                    "on_time_correct": 0,
                                    "response_offsets": [],
                                    "response_to_probe_offsets": [],
                                    "count_abs_errors": [],
                                    "response_missing_n": 0,
                                    "response_early_n": 0,
                                    "response_late_n": 0,
                                    "n_recall_events": 0, "n_recall_support_hits": 0,
                                    "n_recall_query_range_support_hits": 0,
                                    "n_recall_returned_nonempty": 0,
                                    "n_recall_before_response": 0,
                                    "n_recall_support_hit_before_response": 0,
                                    "n_recall_query_range_hit_before_response": 0,
                                    "n_compress_events": 0, "n_compress_succeeded": 0,
                                    "n_compress_system_trigger_rule_checked": 0,
                                    "n_compress_system_trigger_rule_ok": 0,
                                    "n_compress_system_range_rule_checked": 0,
                                    "n_compress_system_range_rule_ok": 0,
                                    "n_compress_system_calc_checked": 0,
                                    "n_compress_system_calc_ok": 0,
                                    "n_step_errors": 0, "n_format_violations": 0,
                                    "n_frame_cache_misses": 0,
                                    "n_action_space_errors": 0,
                                    "n_recall_step2_blocked": 0,
                                    "total_steps": 0,
                                    "action_histogram": Counter(),
                                    "stable_think_samples": 0,
                                    "stable_think_pairs": 0,
                                    "prompt_tokens_max": 0,
                                    "think_tokens_max": 0})
    for r in results:
        task = r["task"]
        telemetry = r.get("telemetry") or {}
        by_task[task]["n_recall_events"] += int(telemetry.get("n_recall_events", 0) or 0)
        by_task[task]["n_recall_support_hits"] += int(
            telemetry.get("n_recall_support_hits", 0) or 0
        )
        by_task[task]["n_recall_query_range_support_hits"] += int(
            telemetry.get("n_recall_query_range_support_hits", 0) or 0
        )
        by_task[task]["n_recall_returned_nonempty"] += int(
            telemetry.get("n_recall_returned_nonempty", 0) or 0
        )
        by_task[task]["n_compress_events"] += int(telemetry.get("n_compress_events", 0) or 0)
        by_task[task]["n_compress_succeeded"] += int(
            telemetry.get("n_compress_succeeded", 0) or 0
        )
        by_task[task]["n_compress_system_trigger_rule_checked"] += int(
            telemetry.get("n_compress_system_trigger_rule_checked", 0) or 0
        )
        by_task[task]["n_compress_system_trigger_rule_ok"] += int(
            telemetry.get("n_compress_system_trigger_rule_ok", 0) or 0
        )
        by_task[task]["n_compress_system_range_rule_checked"] += int(
            telemetry.get("n_compress_system_range_rule_checked", 0) or 0
        )
        by_task[task]["n_compress_system_range_rule_ok"] += int(
            telemetry.get("n_compress_system_range_rule_ok", 0) or 0
        )
        by_task[task]["n_compress_system_calc_checked"] += int(
            telemetry.get("n_compress_system_calc_checked", 0) or 0
        )
        by_task[task]["n_compress_system_calc_ok"] += int(
            telemetry.get("n_compress_system_calc_ok", 0) or 0
        )
        by_task[task]["n_step_errors"] += int(telemetry.get("n_step_errors", 0) or 0)
        by_task[task]["n_frame_cache_misses"] += int(
            telemetry.get("n_frame_cache_misses", 0) or 0
        )
        by_task[task]["n_format_violations"] += int(
            telemetry.get("n_format_violations", 0) or 0
        )
        by_task[task]["n_action_space_errors"] += int(
            telemetry.get("n_action_space_errors", 0) or 0
        )
        by_task[task]["n_recall_step2_blocked"] += int(
            telemetry.get("n_recall_step2_blocked", 0) or 0
        )
        by_task[task]["total_steps"] += int(telemetry.get("total_steps", 0) or 0)
        by_task[task]["action_histogram"].update(telemetry.get("action_histogram") or {})
        stable = telemetry.get("stable_think") or {}
        if int(stable.get("n_stable_pairs", 0) or 0) > 0:
            by_task[task]["stable_think_samples"] += 1
        by_task[task]["stable_think_pairs"] += int(stable.get("n_stable_pairs", 0) or 0)
        by_task[task]["prompt_tokens_max"] = max(
            by_task[task]["prompt_tokens_max"],
            int(telemetry.get("prompt_tokens_max", 0) or 0),
        )
        by_task[task]["think_tokens_max"] = max(
            by_task[task]["think_tokens_max"],
            int(telemetry.get("think_tokens_max", 0) or 0),
        )
        for p in r["probes"]:
            by_task[task]["n"] += 1
            # MCQ / REC use 'correct'; FT-Y/N uses strict_correct
            if "correct" in p:
                correct = int(p["correct"])
                by_task[task]["correct"] += correct
            else:
                correct = int(p["strict_correct"])
                by_task[task]["correct"] += correct
                # delay-specific buckets
                if p.get("type") == 0:
                    by_task[task]["type0_n"] += 1
                    by_task[task]["type0_strict"] += int(p["strict_correct"])
                    by_task[task]["type0_lenient"] += int(p["lenient_correct"])
                    by_task[task]["fp"] += int(p.get("false_positive", False))
                    by_task[task]["fp_n"] += 1
                elif p.get("type") == 1:
                    by_task[task]["type1_n"] += 1
                    by_task[task]["type1_strict"] += int(p["strict_correct"])
                    by_task[task]["type1_lenient"] += int(p["lenient_correct"])
            strict_metric = int(bool(p.get("strict_correct", bool(correct))))
            targeted_metric = int(bool(
                p.get("targeted_correct", p.get("lenient_correct", bool(correct)))
            ))
            by_task[task]["strict_metric_correct"] += strict_metric
            by_task[task]["targeted_correct"] += targeted_metric
            if p.get("count_abs_error") is not None:
                by_task[task]["count_abs_errors"].append(float(p["count_abs_error"]))
            if p.get("used_recall_before_response"):
                by_task[task]["with_recall_n"] += 1
                by_task[task]["with_recall_correct"] += correct
                by_task[task]["n_recall_before_response"] += int(
                    p.get("n_recall_before_response", 0) or 0
                )
                if p.get("recall_support_hit_before_response"):
                    by_task[task]["n_recall_support_hit_before_response"] += 1
                    by_task[task]["with_returned_hit_recall_n"] += 1
                    by_task[task]["with_returned_hit_recall_correct"] += correct
                if p.get("recall_query_range_hit_before_response"):
                    by_task[task]["n_recall_query_range_hit_before_response"] += 1
                    by_task[task]["with_query_range_hit_recall_n"] += 1
                    by_task[task]["with_query_range_hit_recall_correct"] += correct
            else:
                by_task[task]["without_recall_n"] += 1
                by_task[task]["without_recall_correct"] += correct
            if p.get("response_offset_chunks") is not None:
                by_task[task]["response_offsets"].append(p["response_offset_chunks"])
            probe_offset = p.get("response_to_probe_offset_chunks")
            if probe_offset is None:
                by_task[task]["response_missing_n"] += 1
            else:
                by_task[task]["response_to_probe_offsets"].append(probe_offset)
                if correct and probe_offset >= 0:
                    by_task[task]["no_early_correct"] += 1
                if correct and probe_offset <= 0:
                    by_task[task]["no_late_correct"] += 1
                if correct and probe_offset == 0:
                    by_task[task]["on_time_correct"] += 1
                if probe_offset < 0:
                    by_task[task]["response_early_n"] += 1
                elif probe_offset > 0:
                    by_task[task]["response_late_n"] += 1

    # Category averages: mean of task accuracies (matches OVO paper)
    def cat_avg(tasks, correct_key="correct"):
        accs = []
        for t in tasks:
            v = by_task.get(t)
            if v and v["n"] > 0:
                accs.append(v[correct_key] / v["n"])
        return sum(accs) / len(accs) if accs else 0.0, len(accs)

    rt_avg, rt_n = cat_avg(RT_TASKS)
    bt_avg, bt_n = cat_avg(BT_TASKS)
    ft_avg, ft_n = cat_avg(FT_TASKS)
    overall = (rt_avg + bt_avg + ft_avg) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))
    rt_strict, _ = cat_avg(RT_TASKS, "strict_metric_correct")
    bt_strict, _ = cat_avg(BT_TASKS, "strict_metric_correct")
    ft_strict, _ = cat_avg(FT_TASKS, "strict_metric_correct")
    overall_strict = (
        rt_strict + bt_strict + ft_strict
    ) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))
    rt_targeted, _ = cat_avg(RT_TASKS, "targeted_correct")
    bt_targeted, _ = cat_avg(BT_TASKS, "targeted_correct")
    ft_targeted, _ = cat_avg(FT_TASKS, "targeted_correct")
    overall_targeted = (
        rt_targeted + bt_targeted + ft_targeted
    ) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))
    rt_no_early, _ = cat_avg(RT_TASKS, "no_early_correct")
    bt_no_early, _ = cat_avg(BT_TASKS, "no_early_correct")
    ft_no_early, _ = cat_avg(FT_TASKS, "no_early_correct")
    overall_no_early = (
        rt_no_early + bt_no_early + ft_no_early
    ) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))
    rt_no_late, _ = cat_avg(RT_TASKS, "no_late_correct")
    bt_no_late, _ = cat_avg(BT_TASKS, "no_late_correct")
    ft_no_late, _ = cat_avg(FT_TASKS, "no_late_correct")
    overall_no_late = (
        rt_no_late + bt_no_late + ft_no_late
    ) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))
    rt_on_time, _ = cat_avg(RT_TASKS, "on_time_correct")
    bt_on_time, _ = cat_avg(BT_TASKS, "on_time_correct")
    ft_on_time, _ = cat_avg(FT_TASKS, "on_time_correct")
    overall_on_time = (
        rt_on_time + bt_on_time + ft_on_time
    ) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))

    diagnostics = {}
    for task, v in by_task.items():
        offsets = v.get("response_offsets", [])
        probe_offsets = v.get("response_to_probe_offsets", [])
        diagnostics[task] = {
            "total_steps": v["total_steps"],
            "recall_events": v["n_recall_events"],
            "recall_events_per_step": _pct(v["n_recall_events"], v["total_steps"]),
            "recall_events_per_probe": _pct(v["n_recall_events"], v["n"]),
            "recall_return_nonempty_rate": _pct(
                v["n_recall_returned_nonempty"], v["n_recall_events"]
            ),
            "recall_support_hit_rate": _pct(v["n_recall_support_hits"], v["n_recall_events"]),
            "recall_query_range_support_hit_rate": _pct(
                v["n_recall_query_range_support_hits"], v["n_recall_events"]
            ),
            "recall_before_response_rate": _pct(v["with_recall_n"], v["n"]),
            "recall_before_response_support_hit_rate": _pct(
                v["n_recall_support_hit_before_response"], v["with_recall_n"]
            ),
            "recall_before_response_query_range_hit_rate": _pct(
                v["n_recall_query_range_hit_before_response"], v["with_recall_n"]
            ),
            "n_recall_before_response": v["n_recall_before_response"],
            "acc_with_recall": _pct(v["with_recall_correct"], v["with_recall_n"]),
            "n_with_recall": v["with_recall_n"],
            "acc_with_query_range_hit_recall": _pct(
                v["with_query_range_hit_recall_correct"],
                v["with_query_range_hit_recall_n"],
            ),
            "n_with_query_range_hit_recall": v["with_query_range_hit_recall_n"],
            "acc_with_returned_hit_recall": _pct(
                v["with_returned_hit_recall_correct"],
                v["with_returned_hit_recall_n"],
            ),
            "n_with_returned_hit_recall": v["with_returned_hit_recall_n"],
            "acc_without_recall": _pct(v["without_recall_correct"], v["without_recall_n"]),
            "n_without_recall": v["without_recall_n"],
            "acc_content": _pct(v["correct"], v["n"]),
            "acc_strict": _pct(v["strict_metric_correct"], v["n"]),
            "acc_targeted": _pct(v["targeted_correct"], v["n"]),
            "count_mae": _mean(v.get("count_abs_errors", [])),
            "acc_no_early": _pct(v["no_early_correct"], v["n"]),
            "acc_no_late": _pct(v["no_late_correct"], v["n"]),
            "acc_on_time": _pct(v["on_time_correct"], v["n"]),
            "compress_events": v["n_compress_events"],
            "compress_events_per_step": _pct(v["n_compress_events"], v["total_steps"]),
            "compress_events_per_probe": _pct(v["n_compress_events"], v["n"]),
            "compress_success_rate": _pct(v["n_compress_succeeded"], v["n_compress_events"]),
            "compress_system_trigger_rule_rate": _pct(
                v["n_compress_system_trigger_rule_ok"],
                v["n_compress_system_trigger_rule_checked"],
            ),
            "n_compress_system_trigger_rule_checked": v[
                "n_compress_system_trigger_rule_checked"
            ],
            "compress_system_range_rule_rate": _pct(
                v["n_compress_system_range_rule_ok"],
                v["n_compress_system_range_rule_checked"],
            ),
            "n_compress_system_range_rule_checked": v[
                "n_compress_system_range_rule_checked"
            ],
            "compress_system_calc_ok_rate": _pct(
                v["n_compress_system_calc_ok"],
                v["n_compress_system_calc_checked"],
            ),
            "n_compress_system_calc_checked": v["n_compress_system_calc_checked"],
            "step_errors": v["n_step_errors"],
            "frame_cache_misses": v["n_frame_cache_misses"],
            "format_violations": v["n_format_violations"],
            "action_space_errors": v["n_action_space_errors"],
            "recall_step2_blocked": v["n_recall_step2_blocked"],
            "step_error_rate": _pct(v["n_step_errors"], v["total_steps"]),
            "format_violation_rate": _pct(v["n_format_violations"], v["total_steps"]),
            "action_space_error_rate": _pct(v["n_action_space_errors"], v["total_steps"]),
            "recall_step2_blocked_rate": _pct(v["n_recall_step2_blocked"], v["n_recall_events"]),
            "action_histogram": dict(v["action_histogram"]),
            "stable_think_samples": v["stable_think_samples"],
            "stable_think_pairs": v["stable_think_pairs"],
            "stable_think_sample_rate": _pct(v["stable_think_samples"], v["n"]),
            "stable_think_pair_rate": _pct(v["stable_think_pairs"], v["total_steps"]),
            "prompt_tokens_max": v["prompt_tokens_max"],
            "think_tokens_max": v["think_tokens_max"],
            "response_offset_mean": _mean(offsets),
            "response_offset_max": max(offsets) if offsets else None,
            "response_to_probe_offset_mean": _mean(probe_offsets),
            "response_to_probe_offset_min": min(probe_offsets) if probe_offsets else None,
            "response_to_probe_offset_max": max(probe_offsets) if probe_offsets else None,
            "response_missing_rate": _pct(v["response_missing_n"], v["n"]),
            "response_early_rate": _pct(v["response_early_n"], v["n"]),
            "response_late_rate": _pct(v["response_late_n"], v["n"]),
        }

    total = defaultdict(int)
    action_hist = Counter()
    all_probe_offsets = []
    for v in by_task.values():
        for key in (
            "n", "correct", "strict_metric_correct", "targeted_correct",
            "no_early_correct", "no_late_correct",
            "on_time_correct", "response_missing_n", "response_early_n",
            "response_late_n", "with_recall_n", "with_recall_correct",
            "without_recall_n", "without_recall_correct",
            "with_query_range_hit_recall_n",
            "with_query_range_hit_recall_correct",
            "with_returned_hit_recall_n",
            "with_returned_hit_recall_correct",
            "n_recall_events", "n_recall_returned_nonempty",
            "n_recall_support_hits", "n_recall_query_range_support_hits",
            "n_recall_before_response",
            "n_recall_support_hit_before_response",
            "n_recall_query_range_hit_before_response", "n_compress_events",
            "n_compress_succeeded", "n_compress_system_trigger_rule_checked",
            "n_compress_system_trigger_rule_ok",
            "n_compress_system_range_rule_checked",
            "n_compress_system_range_rule_ok",
            "n_compress_system_calc_checked",
            "n_compress_system_calc_ok", "n_step_errors",
            "n_frame_cache_misses", "n_format_violations", "n_action_space_errors",
            "n_recall_step2_blocked", "total_steps",
            "stable_think_samples", "stable_think_pairs",
        ):
            total[key] += int(v.get(key, 0) or 0)
        action_hist.update(v.get("action_histogram") or {})
        all_probe_offsets.extend(v.get("response_to_probe_offsets") or [])
        total.setdefault("count_abs_errors", [])
        total["count_abs_errors"].extend(v.get("count_abs_errors") or [])
        total["prompt_tokens_max"] = max(
            total["prompt_tokens_max"], int(v.get("prompt_tokens_max", 0) or 0)
        )
        total["think_tokens_max"] = max(
            total["think_tokens_max"], int(v.get("think_tokens_max", 0) or 0)
        )

    health = {
        "answer": {
            "probes": total["n"],
            "content_acc": _pct(total["correct"], total["n"]),
            "strict_acc": _pct(total["strict_metric_correct"], total["n"]),
            "targeted_acc": _pct(total["targeted_correct"], total["n"]),
            "no_early_acc": _pct(total["no_early_correct"], total["n"]),
            "no_late_acc": _pct(total["no_late_correct"], total["n"]),
            "on_time_acc": _pct(total["on_time_correct"], total["n"]),
            "count_mae": _mean(total.get("count_abs_errors", [])),
            "missing_rate": _pct(total["response_missing_n"], total["n"]),
            "early_rate": _pct(total["response_early_n"], total["n"]),
            "late_rate": _pct(total["response_late_n"], total["n"]),
            "response_to_probe_offset_mean": _mean(all_probe_offsets),
            "response_to_probe_offset_min": min(all_probe_offsets) if all_probe_offsets else None,
            "response_to_probe_offset_max": max(all_probe_offsets) if all_probe_offsets else None,
        },
        "recall": {
            "events": total["n_recall_events"],
            "events_per_step": _pct(total["n_recall_events"], total["total_steps"]),
            "events_per_probe": _pct(total["n_recall_events"], total["n"]),
            "return_nonempty_rate": _pct(
                total["n_recall_returned_nonempty"], total["n_recall_events"]
            ),
            "support_hit_rate": _pct(total["n_recall_support_hits"], total["n_recall_events"]),
            "query_range_support_hit_rate": _pct(
                total["n_recall_query_range_support_hits"], total["n_recall_events"]
            ),
            "before_response_rate": _pct(total["with_recall_n"], total["n"]),
            "before_response_support_hit_rate": _pct(
                total["n_recall_support_hit_before_response"], total["with_recall_n"]
            ),
            "before_response_query_range_hit_rate": _pct(
                total["n_recall_query_range_hit_before_response"],
                total["with_recall_n"],
            ),
            "acc_with_recall": _pct(total["with_recall_correct"], total["with_recall_n"]),
            "acc_with_query_range_hit_recall": _pct(
                total["with_query_range_hit_recall_correct"],
                total["with_query_range_hit_recall_n"],
            ),
            "acc_with_returned_hit_recall": _pct(
                total["with_returned_hit_recall_correct"],
                total["with_returned_hit_recall_n"],
            ),
            "acc_without_recall": _pct(
                total["without_recall_correct"], total["without_recall_n"]
            ),
            "step2_blocked_rate": _pct(
                total["n_recall_step2_blocked"], total["n_recall_events"]
            ),
        },
        "compression": {
            "events": total["n_compress_events"],
            "events_per_step": _pct(total["n_compress_events"], total["total_steps"]),
            "events_per_probe": _pct(total["n_compress_events"], total["n"]),
            "success_rate": _pct(total["n_compress_succeeded"], total["n_compress_events"]),
            "system_trigger_rule_rate": _pct(
                total["n_compress_system_trigger_rule_ok"],
                total["n_compress_system_trigger_rule_checked"],
            ),
            "system_range_rule_rate": _pct(
                total["n_compress_system_range_rule_ok"],
                total["n_compress_system_range_rule_checked"],
            ),
            "system_calc_ok_rate": _pct(
                total["n_compress_system_calc_ok"],
                total["n_compress_system_calc_checked"],
            ),
        },
        "format_runtime": {
            "steps": total["total_steps"],
            "step_error_rate": _pct(total["n_step_errors"], total["total_steps"]),
            "frame_cache_miss_rate": _pct(
                total["n_frame_cache_misses"], total["total_steps"]
            ),
            "frame_cache_misses": total["n_frame_cache_misses"],
            "format_violation_rate": _pct(
                total["n_format_violations"], total["total_steps"]
            ),
            "action_space_error_rate": _pct(
                total["n_action_space_errors"], total["total_steps"]
            ),
            "stable_think_sample_rate": _pct(total["stable_think_samples"], total["n"]),
            "stable_think_pair_rate": _pct(
                total["stable_think_pairs"], total["total_steps"]
            ),
            "prompt_tokens_max": total["prompt_tokens_max"],
            "think_tokens_max": total["think_tokens_max"],
            "action_histogram": dict(action_hist),
        },
    }

    return {
        "by_task": dict(by_task),
        "category": {
            "RT": {
                "avg": rt_avg, "n_tasks": rt_n,
                "strict_acc": rt_strict,
                "targeted_acc": rt_targeted,
                "acc_no_early": rt_no_early,
                "acc_no_late": rt_no_late,
                "acc_on_time": rt_on_time,
            },
            "BT": {
                "avg": bt_avg, "n_tasks": bt_n,
                "strict_acc": bt_strict,
                "targeted_acc": bt_targeted,
                "acc_no_early": bt_no_early,
                "acc_no_late": bt_no_late,
                "acc_on_time": bt_on_time,
            },
            "FT": {
                "avg": ft_avg, "n_tasks": ft_n,
                "strict_acc": ft_strict,
                "targeted_acc": ft_targeted,
                "acc_no_early": ft_no_early,
                "acc_no_late": ft_no_late,
                "acc_on_time": ft_on_time,
            },
        },
        "overall": overall,
        "overall_strict": overall_strict,
        "overall_targeted": overall_targeted,
        "overall_no_early": overall_no_early,
        "overall_no_late": overall_no_late,
        "overall_on_time": overall_on_time,
        "health": health,
        "diagnostics": diagnostics,
    }


def print_report(agg):
    print()
    print(f"{'task':<8}  {'n':>6}  {'acc':>7}  {'strict':>7}  {'target':>7}  {'noE':>7}  {'noL':>7}  {'onT':>7}  {'recall':>7}  {'comp':>6}  {'early':>6}  {'late':>6}  notes")
    print("-" * 144)
    for task in sorted(agg["by_task"]):
        v = agg["by_task"][task]
        if v["n"] == 0:
            continue
        acc = v["correct"] / v["n"]
        diag = (agg.get("diagnostics") or {}).get(task, {})
        notes = ""
        if task in ("CRR", "SSR"):
            t0 = v["type0_n"]
            t1 = v["type1_n"]
            t0a = (v["type0_strict"] / t0) if t0 else 0.0
            t1a = (v["type1_strict"] / t1) if t1 else 0.0
            fp = (v["fp"] / v["fp_n"]) if v["fp_n"] > 0 else 0.0
            notes = f"t0={t0a:.3f}({t0}) t1={t1a:.3f}({t1}) fp={fp:.3f}"
        print(
            f"{task:<8}  {v['n']:>6}  {acc:>7.3f}  "
            f"{diag.get('acc_strict', 0.0):>7.3f}  "
            f"{diag.get('acc_targeted', 0.0):>7.3f}  "
            f"{diag.get('acc_no_early', 0.0):>7.3f}  "
            f"{diag.get('acc_no_late', 0.0):>7.3f}  "
            f"{diag.get('acc_on_time', 0.0):>7.3f}  "
            f"{diag.get('recall_events', 0):>7}  "
            f"{diag.get('compress_events', 0):>6}  "
            f"{diag.get('stable_think_pairs', 0):>6}  "
            f"{diag.get('response_early_rate', 0.0):>6.3f}  "
            f"{diag.get('response_late_rate', 0.0):>6.3f}  {notes}"
        )

    print()
    print(f"Real-Time Visual Perception (RT): {agg['category']['RT']['avg']:.3f} "
          f"strict={agg['category']['RT'].get('strict_acc', 0.0):.3f} "
          f"target={agg['category']['RT'].get('targeted_acc', 0.0):.3f} "
          f"({agg['category']['RT']['n_tasks']} tasks)")
    print(f"Backward Tracing (BT):            {agg['category']['BT']['avg']:.3f} "
          f"strict={agg['category']['BT'].get('strict_acc', 0.0):.3f} "
          f"target={agg['category']['BT'].get('targeted_acc', 0.0):.3f} "
          f"({agg['category']['BT']['n_tasks']} tasks)")
    print(f"Forward Active Responding (FT):   {agg['category']['FT']['avg']:.3f} "
          f"strict={agg['category']['FT'].get('strict_acc', 0.0):.3f} "
          f"target={agg['category']['FT'].get('targeted_acc', 0.0):.3f} "
          f"({agg['category']['FT']['n_tasks']} tasks)")
    print(f"OVERALL (mean of categories):     {agg['overall']:.3f}")
    print(f"OVERALL strict/targeted:          "
          f"{agg.get('overall_strict', 0.0):.3f} / "
          f"{agg.get('overall_targeted', 0.0):.3f}")
    print(f"OVERALL no-early/no-late/on-time: "
          f"{agg.get('overall_no_early', 0.0):.3f} / "
          f"{agg.get('overall_no_late', 0.0):.3f} / "
          f"{agg.get('overall_on_time', 0.0):.3f}")
    health = agg.get("health") or {}
    recall = health.get("recall") or {}
    compression = health.get("compression") or {}
    runtime = health.get("format_runtime") or {}
    print(f"Health recall: events={recall.get('events', 0)} "
          f"per_step={recall.get('events_per_step', 0.0):.3f} "
          f"query_range_hit={recall.get('query_range_support_hit_rate', 0.0):.3f} "
          f"returned_hit={recall.get('support_hit_rate', 0.0):.3f} "
          f"acc_with/without={recall.get('acc_with_recall', 0.0):.3f}/"
          f"{recall.get('acc_without_recall', 0.0):.3f} "
          f"acc_range_hit={recall.get('acc_with_query_range_hit_recall', 0.0):.3f} "
          f"acc_returned_hit={recall.get('acc_with_returned_hit_recall', 0.0):.3f}")
    print(f"Health compression: events={compression.get('events', 0)} "
          f"success={compression.get('success_rate', 0.0):.3f} "
          f"system_calc_ok={compression.get('system_calc_ok_rate', 0.0):.3f}")
    print(f"Health runtime: steps={runtime.get('steps', 0)} "
          f"format_bad={runtime.get('format_violation_rate', 0.0):.3f} "
          f"action_bad={runtime.get('action_space_error_rate', 0.0):.3f} "
          f"stable_pair={runtime.get('stable_think_pair_rate', 0.0):.3f}")
    print()
    print("Diagnostics: recall = event count; comp = compression trigger count; "
          "acc = main content/official-style score; strict = targeted answer "
          "at the exact probe chunk; target = task-specific scorer that allows "
          "valid non-speaking states such as No-by-silence; noE/noL/onT split "
          "answer timing after content correctness.")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmark_json", required=True,
                   help="Path to ORIGINAL ovo_bench_new.json")
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default=None,
                   help="Root dir with pre-extracted 1fps frames (skip online decode)")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated subset (e.g., CRR,SSR,REC). Default: all 12.")
    p.add_argument("--n_per_task", type=int, default=None,
                   help="Cap samples per task (for quick smoke-tests)")
    p.add_argument("--sample_ids", default=None,
                   help="Comma-separated OVO sample ids to keep before task and "
                        "length caps. Useful for targeted long-trajectory "
                        "protocol stress tests.")
    p.add_argument("--max_job_chunk", type=int, default=None,
                   help="Protocol-search filter: keep only samples whose "
                        "streaming trajectory jobs end at or before this "
                        "chunk. This preserves correctness while avoiding "
                        "very long full-video tails in fast design sweeps.")
    p.add_argument("--prefer_short_jobs", action="store_true",
                   help="Sort samples by trajectory length before applying "
                        "--n_per_task. Useful for quick, fixed protocol sweeps.")
    p.add_argument("--prefer_long_jobs", action="store_true",
                   help="Sort samples by descending trajectory length before "
                        "applying --n_per_task. Intended for memory-pressure "
                        "stress tests.")
    p.add_argument("--max_agent_jobs", type=int, default=None,
                   help="Hard cap on built vLLM trajectory jobs after all "
                        "sample filters. Intended only for protocol smoke tests.")
    p.add_argument("--retriever", default="time_range", choices=["none", "time_range"])
    p.add_argument("--compress_mode", default="system", choices=["system", "self", "off", "none"])
    p.add_argument("--memory_mode", default=os.environ.get("THINKSTREAM_EVAL_MEMORY_MODE", "full"),
                   choices=["full", "no_prompt", "no_recall", "none"],
                   help="Text-memory ablation for the streaming agent. full: "
                        "normal memory. no_prompt: hide text memory from "
                        "ordinary turns but keep recall archive. no_recall: "
                        "keep memory prompt but recall retrieves nothing. "
                        "none: no text memory prompt/archive/compression.")
    p.add_argument("--max_results", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--compress_max_new_tokens", type=int, default=512)
    p.add_argument("--profile", default="16k", choices=["16k", "32k"],
                   help="Eval context profile (see scripts/eval/eval_profiles.py "
                        "for full token-budget breakdown). 16k = SFT-aligned "
                        "(default). 32k = extended for Qwen3-VL native context, "
                        "loosens queries cap (8→24), recall cap (800→3000 char), "
                        "max_new_tokens (128→256).")
    p.add_argument("--scoring", default="strict", choices=["strict", "lenient"],
                   help="strict (default): MCQ response must come within 2 "
                        "chunks of ask_chunk (matches OVO paper). lenient: "
                        "walk up to 60 chunks past ask_chunk; any response "
                        "counts. REC/SSR/CRR ignore this flag (their timing "
                        "is the test).")
    p.add_argument(
        "--frame-protocol",
        default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"),
        choices=["video_meta", "ts_image"],
        help="Visual carrier for pre-extracted frames. Production SFT/RL "
             "wrappers still lock this to video_meta.",
    )
    p.add_argument(
        "--render-layout",
        dest="render_layout",
        default="standard_query_last",
        choices=["standard_query_last"],
        help="Prompt layout. Production SFT/RL wrappers default to "
             "standard_query_last.",
    )
    p.add_argument("--memory-position",
                   default=os.environ.get("THINKSTREAM_MEMORY_POSITION", "before_visual"),
                   choices=["before_visual"],
                   help="Render text memory before the visual window.")
    p.add_argument("--min_pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    p.add_argument("--visual_window_chunks", type=int, default=None,
                   help="Eval-only override for the sliding visual window size.")
    p.add_argument("--frames_per_chunk", type=int, default=None,
                   help="Eval-only selected frames per 1-second chunk.")
    p.add_argument("--source_frames_per_chunk", type=int, default=0,
                   help="Frames per chunk in the pre-extracted frame cache. "
                        "0 means infer per video; OVO mixes 1fps and 2fps "
                        "frame dumps.")
    p.add_argument("--require_frame_cache", action="store_true",
                   help="Forbid fallback to raw video_path decoding when a "
                        "pre-extracted frame window is missing. Use for fair "
                        "visual-protocol speed/quality sweeps.")
    p.add_argument("--drop_incomplete_frame_cache", action="store_true",
                   help="Before applying --n_per_task, skip samples whose "
                        "trajectory would need raw-video fallback because the "
                        "pre-extracted frame cache ends before max_chunk.")
    p.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                   help="Inference backend. vllm batches multiple live video "
                        "trajectories and advances their memory states locally.")
    p.add_argument("--rollout_batch_size", type=int, default=8,
                   help="vLLM engine only: number of active trajectories to "
                        "generate per scheduler round.")
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="vLLM engine only. Defaults to visible GPU count.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--vllm_max_model_len", type=int, default=None)
    p.add_argument("--vllm_max_images_per_prompt", type=int, default=96)
    p.add_argument("--vllm_max_videos_per_prompt", type=int, default=2)
    p.add_argument("--vllm_mm_processor_cache_gb", type=int, default=None)
    p.add_argument("--disable_vllm_mm_preprocessor_cache", action="store_true")
    p.add_argument("--vllm_repetition_penalty", type=float, default=1.0)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--query-policy", default=os.environ.get(
        "THINKSTREAM_QUERY_HISTORY_POLICY", "single_active"),
                   choices=["recent_k", "single_active", "replace_on_new", "multi_pending"],
                   help="Which live query records may be rendered in active_query. "
                        "OVO tasks are independent, so default keeps only the active query.")
    p.add_argument("--queries-history-cap", type=int, default=None,
                   help="Override query history cap after applying profile.")
    p.add_argument("--recall_text_max_chars", type=int, default=None,
                   help="Legacy no-op for model prompts; recall_result is metadata-only.")
    p.add_argument("--recent_thinks_token_budget", type=int, default=None,
                   help="Override inference memory recent_thinks token budget; "
                        "the compression trigger remains 80%% of this value.")
    p.add_argument("--summary_tokens_max", type=int, default=None,
                   help="Override inference compressed-summary token cap.")
    p.add_argument("--out", default=None)
    p.add_argument("--save_step_trace", action="store_true",
                   help="Store per-step truncated think/response trace in JSON. "
                        "Useful for stable-think audits; large on full OVO.")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()
    global VISUAL_WINDOW_CHUNKS, FRAMES_PER_CHUNK, SOURCE_FRAMES_PER_CHUNK
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    args.render_layout = normalize_render_layout(args.render_layout)
    args.memory_position = normalize_memory_position(args.memory_position)
    os.environ["THINKSTREAM_RENDER_LAYOUT"] = args.render_layout
    os.environ["THINKSTREAM_MEMORY_POSITION"] = args.memory_position
    os.environ["THINKSTREAM_PREFER_PATH_FRAME_INDEX"] = "0"

    if args.visual_window_chunks is not None:
        VISUAL_WINDOW_CHUNKS = max(1, int(args.visual_window_chunks))
    if args.frames_per_chunk is not None:
        FRAMES_PER_CHUNK = max(1, int(args.frames_per_chunk))
    SOURCE_FRAMES_PER_CHUNK = max(0, int(args.source_frames_per_chunk))
    from thinkstream.data import agent_protocol as _agent_protocol
    _agent_protocol.VISUAL_WINDOW_CHUNKS = VISUAL_WINDOW_CHUNKS
    _agent_protocol.FRAMES_PER_CHUNK = FRAMES_PER_CHUNK

    # Apply eval profile (mutates agent_protocol token-budget globals).
    from scripts.eval.eval_profiles import apply_profile, describe_profile
    profile_cfg = apply_profile(args.profile)
    from thinkstream.data import agent_protocol
    agent_protocol.VISUAL_WINDOW_CHUNKS = VISUAL_WINDOW_CHUNKS
    agent_protocol.FRAMES_PER_CHUNK = FRAMES_PER_CHUNK
    agent_protocol.QUERY_HISTORY_POLICY = args.query_policy
    if args.queries_history_cap is not None:
        agent_protocol.QUERIES_HISTORY_CAP = int(args.queries_history_cap)
    elif args.query_policy in {"single_active", "replace_on_new"}:
        agent_protocol.QUERIES_HISTORY_CAP = 1
    if args.recall_text_max_chars is not None:
        agent_protocol.RECALL_TEXT_MAX_CHARS = max(1, int(args.recall_text_max_chars))
        profile_cfg = dict(profile_cfg)
        profile_cfg["recall_text_max_chars"] = agent_protocol.RECALL_TEXT_MAX_CHARS
    if args.recent_thinks_token_budget is not None:
        import thinkstream.models.agent_loop as _agent_loop
        global RECENT_THINKS_TOKEN_BUDGET, COMPRESS_TOKEN_THRESHOLD
        RECENT_THINKS_TOKEN_BUDGET = max(1, int(args.recent_thinks_token_budget))
        _agent_loop.RECENT_THINKS_TOKEN_BUDGET = RECENT_THINKS_TOKEN_BUDGET
        _agent_loop.COMPRESS_TOKEN_THRESHOLD = int(
            RECENT_THINKS_TOKEN_BUDGET * _agent_loop.COMPRESS_TRIGGER_RATIO
        )
        COMPRESS_TOKEN_THRESHOLD = _agent_loop.COMPRESS_TOKEN_THRESHOLD
    if args.summary_tokens_max is not None:
        import thinkstream.models.agent_loop as _agent_loop
        _agent_loop.SUMMARY_TOKENS_MAX = max(1, int(args.summary_tokens_max))
    print(describe_profile(args.profile))
    print(f"query_history: policy={agent_protocol.QUERY_HISTORY_POLICY}, "
          f"cap={agent_protocol.QUERIES_HISTORY_CAP}")
    if (
        args.recent_thinks_token_budget is not None
        or args.summary_tokens_max is not None
        or args.recall_text_max_chars is not None
    ):
        import thinkstream.models.agent_loop as _agent_loop
        print(
            "memory_budget_override: "
            f"recent_tokens={_agent_loop.RECENT_THINKS_TOKEN_BUDGET}, "
            f"compress_threshold={_agent_loop.COMPRESS_TOKEN_THRESHOLD}, "
            f"summary_tokens={_agent_loop.SUMMARY_TOKENS_MAX}, "
            "recall_metadata_only=True",
            flush=True,
        )
    print(f"agent_ablation: compress_mode={args.compress_mode}, "
          f"memory_mode={args.memory_mode}, retriever={args.retriever}, "
          f"engine={args.engine}")
    print(f"visual_ablation: frame_protocol={frame_protocol}, "
          f"render_layout={args.render_layout}, "
          f"memory_position={args.memory_position}, "
          f"window_chunks={VISUAL_WINDOW_CHUNKS}, "
          f"frames_per_chunk={FRAMES_PER_CHUNK}, "
          f"source_frames_per_chunk={SOURCE_FRAMES_PER_CHUNK or 'auto'}, "
          f"pixels={args.min_pixels}-{args.max_pixels}")
    if args.max_new_tokens == 128 and args.profile == "32k":
        args.max_new_tokens = profile_cfg["max_new_tokens_default"]

    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=profile_cfg["model_max_length"],
        padding_side="right", use_fast=False,
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()
         and t not in WRONG_RESPONSE_SPECIAL_TOKENS],
        special_tokens=True,
    )
    ensure_agent_special_tokens(processor.tokenizer)
    ensure_agent_special_tokens(tokenizer)
    validate_agent_special_tokens(processor.tokenizer)
    validate_agent_special_tokens(tokenizer)

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)

    task_filter = set(t.strip() for t in args.tasks.split(",")) if args.tasks else ALL_TASKS

    # Group by task to apply per-task caps and report progress
    by_task = defaultdict(list)
    sample_id_filter = None
    if args.sample_ids:
        sample_id_filter = {
            raw.strip() for raw in str(args.sample_ids).split(",") if raw.strip()
        }

    for s in all_samples:
        if sample_id_filter is not None and str(s.get("id")) not in sample_id_filter:
            continue
        if s.get("task") in task_filter:
            by_task[s["task"]].append(s)

    if args.drop_incomplete_frame_cache:
        skipped = 0
        examples = []
        for task in list(by_task.keys()):
            kept = []
            for sample in by_task[task]:
                jobs = build_agent_jobs_for_sample(
                    sample, args.video_root, scoring=args.scoring,
                )
                if not jobs:
                    skipped += 1
                    continue
                complete = True
                first_miss = None
                for job in jobs:
                    ok, miss = job_frame_cache_complete(
                        job, args.frames_root, args.video_root,
                    )
                    if not ok:
                        complete = False
                        first_miss = miss
                        break
                if complete:
                    kept.append(sample)
                else:
                    skipped += 1
                    if first_miss and len(examples) < 5:
                        examples.append(first_miss)
            by_task[task] = kept
        print(
            "frame_cache_filter: "
            f"skipped={skipped}, examples={examples}",
            flush=True,
        )

    if args.prefer_short_jobs and args.prefer_long_jobs:
        raise ValueError("--prefer_short_jobs and --prefer_long_jobs are mutually exclusive")

    if args.max_job_chunk is not None or args.prefer_short_jobs or args.prefer_long_jobs:
        max_job_chunk = (
            max(0, int(args.max_job_chunk))
            if args.max_job_chunk is not None else None
        )
        skipped = 0
        for task in list(by_task.keys()):
            ranked = []
            for sample in by_task[task]:
                jobs = build_agent_jobs_for_sample(
                    sample, args.video_root, scoring=args.scoring,
                )
                if not jobs:
                    skipped += 1
                    continue
                sample_max_chunk = max(int(job.get("max_chunk", -1)) for job in jobs)
                if max_job_chunk is not None and sample_max_chunk > max_job_chunk:
                    skipped += 1
                    continue
                ranked.append((sample_max_chunk, sample))
            if args.prefer_short_jobs:
                ranked.sort(key=lambda x: x[0])
            elif args.prefer_long_jobs:
                ranked.sort(key=lambda x: x[0], reverse=True)
            by_task[task] = [sample for _, sample in ranked]
        print(
            "job_length_filter: "
            f"max_job_chunk={max_job_chunk}, "
            f"prefer_short={bool(args.prefer_short_jobs)}, "
            f"prefer_long={bool(args.prefer_long_jobs)}, "
            f"skipped={skipped}",
            flush=True,
        )

    if args.n_per_task:
        for t in by_task:
            by_task[t] = by_task[t][: args.n_per_task]

    total_samples = sum(len(v) for v in by_task.values())
    print(f"Running on {total_samples} samples across {len(by_task)} tasks: "
          f"{sorted(by_task.keys())}")

    t0 = time.time()
    if args.engine == "vllm":
        from thinkstream.eval.vllm_engine import init_vllm_engine
        max_model_len = int(args.vllm_max_model_len or profile_cfg["model_max_length"])
        print(
            f"Loading vLLM from {args.ckpt} "
            f"(tp={args.tensor_parallel_size or 'visible'}, "
            f"batch={args.rollout_batch_size}, max_len={max_model_len}) ..."
        )
        llm = init_vllm_engine(
            args.ckpt,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
            max_images_per_prompt=args.vllm_max_images_per_prompt,
            max_videos_per_prompt=args.vllm_max_videos_per_prompt,
            mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
            disable_mm_preprocessor_cache=args.disable_vllm_mm_preprocessor_cache,
            enable_prefix_caching=True,
        )
        jobs = []
        for task in sorted(by_task.keys()):
            for sample in by_task[task]:
                try:
                    jobs.extend(build_agent_jobs_for_sample(
                        sample, args.video_root, scoring=args.scoring,
                    ))
                except Exception as e:
                    print(
                        f"[{task} id={sample.get('id')}] job failed: "
                        f"{type(e).__name__}: {e}"
                    )
        if args.max_agent_jobs is not None:
            jobs = jobs[: max(1, int(args.max_agent_jobs))]
        print(f"Built {len(jobs)} vLLM trajectory jobs")
        results = run_agent_jobs_vllm(
            jobs,
            llm=llm,
            processor=processor,
            tokenizer=tokenizer,
            args=args,
        )
    else:
        Cls, model_type = detect_model_class(args.ckpt)
        print(f"Loading {Cls.__name__} from {args.ckpt} ...")
        model = Cls.from_pretrained(
            args.ckpt,
            dtype=torch.bfloat16 if not args.no_bf16 else None,
            attn_implementation="flash_attention_2",
        )
        ensure_agent_special_tokens(tokenizer, model=model)
        validate_agent_special_tokens(tokenizer)
        model = model.cuda()
        model.eval()
        print(f"Building retriever: kind={args.retriever}")
        if args.retriever == "none" or args.memory_mode in {"no_recall", "none"}:
            retriever = NullRetriever()
        else:
            retriever = make_retriever(
                kind=args.retriever,
                max_results=args.max_results,
                frames_root=args.frames_root,
                video_root=args.video_root,
            )

        loop = make_loop(model, processor, tokenizer, model_type, retriever,
                         args.compress_mode, args.max_new_tokens,
                         frames_root=args.frames_root, video_root=args.video_root,
                         frame_protocol=frame_protocol, memory_mode=args.memory_mode,
                         min_pixels=args.min_pixels, max_pixels=args.max_pixels)

        results = []
        done = 0
        for task in sorted(by_task.keys()):
            for sample in by_task[task]:
                try:
                    r = dispatch_eval(sample, loop, retriever, args.video_root,
                                      scoring=args.scoring,
                                      save_step_trace=args.save_step_trace)
                    if r is not None:
                        results.append(r)
                except Exception as e:
                    print(f"[{task} id={sample.get('id')}] failed: {type(e).__name__}: {e}")
                done += 1
                if done % 10 == 0:
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"[{done}/{total_samples}] {rate*60:.1f} samples/min")

    if not results:
        print("No successful samples.")
        return

    agg = aggregate(results)
    print_report(agg)

    out_path = args.out or (
        f"{args.ckpt}/eval/ovo_full/"
        f"full_{args.compress_mode}_{args.memory_mode}_{args.retriever}_"
        f"{args.scoring}_{args.profile}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "compress_mode": args.compress_mode,
            "memory_mode": args.memory_mode,
            "retriever": {"kind": args.retriever},
            "scoring": args.scoring,
            "profile": args.profile,
            "frame_protocol": frame_protocol,
            "render_layout": args.render_layout,
            "memory_position": args.memory_position,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "visual_window_chunks": VISUAL_WINDOW_CHUNKS,
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "source_frames_per_chunk": (
                SOURCE_FRAMES_PER_CHUNK if SOURCE_FRAMES_PER_CHUNK > 0 else "auto"
            ),
            "prefer_path_frame_index": False,
            "require_frame_cache": bool(args.require_frame_cache),
            "drop_incomplete_frame_cache": bool(args.drop_incomplete_frame_cache),
            "engine": args.engine,
            "rollout_batch_size": args.rollout_batch_size if args.engine == "vllm" else None,
            "vllm_mm_processor_cache_gb": args.vllm_mm_processor_cache_gb,
            "disable_vllm_mm_preprocessor_cache": args.disable_vllm_mm_preprocessor_cache,
            "profile_cfg": profile_cfg,
            "max_job_chunk": args.max_job_chunk,
            "prefer_short_jobs": bool(args.prefer_short_jobs),
            "prefer_long_jobs": bool(args.prefer_long_jobs),
            "sample_ids": args.sample_ids,
            "max_agent_jobs": args.max_agent_jobs,
            "recall_text_max_chars": agent_protocol.RECALL_TEXT_MAX_CHARS,
            "recent_thinks_token_budget": RECENT_THINKS_TOKEN_BUDGET,
            "compress_token_threshold": COMPRESS_TOKEN_THRESHOLD,
            "summary_tokens_max": (
                __import__("thinkstream.models.agent_loop", fromlist=["SUMMARY_TOKENS_MAX"])
                .SUMMARY_TOKENS_MAX
            ),
            "tasks_evaluated": sorted(by_task.keys()),
            "n_samples": len(results),
            "summary": agg,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
