"""Legacy flat-row PASS 5 helpers.

The runnable flat/single-step renderer has been retired. Current SFT/RL/eval
data must be rendered through ``scripts.agent_data.pass5`` into
``rendered/trajectory/*_trajectory.jsonl`` so each row is a recurrent
multi-turn video trajectory. This module keeps importable helper functions for
older audits/tests that compare prompt construction, but its CLI now exits with
a clear error instead of writing ``*_messages.jsonl``.
"""
from __future__ import annotations

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
    format_compact_memory_update_input,
    format_memory_block,
    format_queries_block,
    format_user_input_block,
    append_visual_frames,
    build_recalled_frames_metadata,
    build_recall_result_user_content,
    canonical_answer_instruction,
    chunk_frame_filenames,
    is_inter_chunk,
    normalize_frame_protocol,
    parse_agent_output,
    prompt_time_value,
    select_recall_chunks_uniform,
    system_prompt_for_frame_protocol,
    user_input_is_active_query_duplicate,
)

logger = logging.getLogger(__name__)

RENDER_LAYOUT_QUERY_LAST = RENDER_LAYOUT_STANDARD_QUERY_LAST
RESPONSE_BLOCK_RE = re.compile(
    r"</Response>\s*(.*?)\s*$|<response>(.*?)</response>",
    flags=re.DOTALL,
)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
SILENT_BLOCK_RE = re.compile(
    r"</Silence>\s*|<silent>\s*(?:</silent>)?",
    flags=re.DOTALL,
)
STREAMING_TERMINAL_RE = re.compile(
    r"</Response>\s*.*$|</Silence>\s*|<response>.*?</response>|"
    r"<answer>.*?</answer>|<silent>\s*(?:</silent>)?",
    flags=re.DOTALL,
)
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
    from scripts.agent_data.config import DATA_ROOT as DEFAULT_DATA_DIR
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
        end = int(float(tr[1]) / float(chunk_sec))
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    if end < start:
        return []
    return list(range(max(0, start), end + 1))


def _normalise_recalled_frames(
    inp: Dict,
    chunk_sec: float,
    recall_result: Optional[Dict] = None,
) -> Optional[Dict]:
    """Cap recalled frames with the same helper used by RL/eval/runtime."""
    rf = inp.get("recalled_frames") or {}
    if not rf:
        return None
    rr = recall_result or inp.get("recall_result") or {}
    original_chunks: List[int] = []
    for raw in rr.get("returned_chunks") or []:
        try:
            original_chunks.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not original_chunks:
        original_chunks = _chunks_from_recalled_time_range(rf, chunk_sec)
    selected_chunks = select_recall_chunks_uniform(original_chunks)
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
    if re.search(r'<m\s+t="[^"]+"\s*>.*?</m>', output or "", re.DOTALL | re.IGNORECASE):
        return (
            "Memory is near budget, so I should update the compact memory "
            "from old memory and recent observations."
        )
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
    if letter and text:
        return f"{letter}) {text}"
    return letter or text


def _canonical_answer_target(sample: Dict[str, Any]) -> str:
    question = sample.get("_trajectory_question") or sample.get("metadata") or {}
    if question.get("answer_form") == "multiple_choice":
        return _mc_target_for_question(question, sample.get("chunk_idx"))
    return _per_emit_target(question, sample.get("chunk_idx"))


def _normalise_mc_query_for_prompt(query: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(query, dict):
        return query
    if query.get("answer_form") != "multiple_choice":
        return query
    out = dict(query)
    out["answer_style"] = "letter_plus_text"
    instruction = canonical_answer_instruction(out)
    if instruction:
        out["answer_instruction"] = instruction
    target = _mc_target_for_question(out, None)
    if target:
        emits = []
        for emit in out.get("per_emit_answers") or []:
            if isinstance(emit, dict):
                e = dict(emit)
                if str(e.get("value") or "").strip():
                    e["value"] = target
                emits.append(e)
            else:
                emits.append(emit)
        if emits:
            out["per_emit_answers"] = emits

        answers = []
        for ans in out.get("answers") or []:
            if isinstance(ans, dict):
                a = dict(ans)
                if str(a.get("text") or "").strip():
                    a["text"] = target
                answers.append(a)
            elif str(ans or "").strip():
                answers.append(target)
            else:
                answers.append(ans)
        if answers:
            out["answers"] = answers
        out["sft_answer"] = target
    letter, correct_text = _mc_letter_text(out)
    if correct_text:
        out["correct_answer_text"] = correct_text
        out["canonical_answer"] = correct_text
        out["gold_answer"] = correct_text
    if letter:
        out["correct_option"] = letter
    out["accepted_answers"] = _accepted_answers_for_question(out)
    return out


def _normalise_queries_for_prompt(queries: Any) -> List[Dict[str, Any]]:
    return [
        _normalise_mc_query_for_prompt(q)
        for q in (queries or [])
        if isinstance(q, dict)
    ]


def _canonicalize_response_tags(text: str, target: str = "") -> str:
    target = str(target or "").strip()

    def _terminal(value: str) -> str:
        value = str(value or "").strip()
        return f"</Response> {value}" if value else "</Silence>"

    def legacy_repl(match: re.Match) -> str:
        value = target if target and match.group(1).strip() else match.group(1).strip()
        return _terminal(value)

    if target:
        if STREAMING_TERMINAL_RE.search(text):
            return STREAMING_TERMINAL_RE.sub(_terminal(target), text, count=1)
    text = ANSWER_BLOCK_RE.sub(legacy_repl, text)
    text = RESPONSE_BLOCK_RE.sub(
        lambda m: _terminal((m.group(1) if m.group(1) is not None else m.group(2)).strip()),
        text,
    )
    text = SILENT_BLOCK_RE.sub(_terminal(""), text)
    return text


def _normalise_assistant_output(sample: Dict, output: Optional[str] = None) -> str:
    output = str(sample.get("output", "") if output is None else output)
    if sample.get("sample_type") != "compress":
        return _canonicalize_response_tags(output, _canonical_answer_target(sample))
    m_lines = list(re.finditer(
        r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>.*?</m>',
        output or "",
        re.DOTALL | re.IGNORECASE,
    ))
    if m_lines:
        return "\n".join(m.group(0).strip() for m in m_lines)
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
    return _canonicalize_response_tags(output, _canonical_answer_target(sample))


def _infer_visual_frame_paths(
    sample: Dict[str, Any],
    data_dir: Path,
    *,
    frame_rel_prefix: str,
) -> List[str]:
    inp = sample.get("input") or {}
    vw = inp.get("visual_window") or {}
    if "frame_paths" in vw:
        return list(vw.get("frame_paths") or [])[-FRAMES_PER_CHUNK:]
    if "frames" not in vw:
        return []
    vid = sample.get("video_id", "")
    if not vid:
        return []
    chunk_idx = int(sample.get("chunk_idx", 0) or 0)
    return [
        f"{frame_rel_prefix}/{vid}/{name}"
        for name in chunk_frame_filenames(chunk_idx, FRAMES_PER_CHUNK)
    ]


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
    prompt builder. Rows render user_input, memory, current chunk video_meta,
    then active_query/response_history.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    if render_layout != RENDER_LAYOUT_QUERY_LAST:
        raise ValueError(f"Unsupported render_layout={render_layout!r}")
    frame_rel_prefix = _frame_rel_prefix(data_dir)

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = AGENT_CHUNK_SEC
    inter_chunk = is_inter_chunk(sample)
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
    queries_for_prompt = _normalise_queries_for_prompt(inp.get("queries", []))

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

    # Compact memory update is no longer a streaming visual turn. Render it as
    # one text-only system turn: user(old memory + recent captions) ->
    # assistant(bare <m> lines).
    if inter_chunk:
        update_input = (
            sample.get("memory_update_input")
            or (sample.get("metadata") or {}).get("memory_update_input")
            or inp.get("memory_update_input")
        )
        if not str(update_input or "").strip():
            update_input = format_compact_memory_update_input(inp.get("memory", {}))
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": str(update_input).strip()}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": _normalise_assistant_output(sample)}],
        })
        return messages

    # ── Legacy standalone post-recall row ──────────────────────────────
    # Canonical recall samples are shape B:
    #   user(current chunk + active_query) → assistant(recall) →
    #   user(recalled frames + metadata) → assistant(answer)
    # Older pass3 rows can already be the final post-recall answer without
    # the first two turns. In that case render only the active query plus
    # recall evidence; never add a fresh current visual window under the
    # post-recall system prompt.
    if legacy_post_recall and not is_recall_multiturn and not inter_chunk:
        queries = queries_for_prompt
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": qt})

        rf = _normalise_recalled_frames(
            inp,
            chunk_sec,
            inp.get("recall_result") or sample.get("recall_result"),
        ) or inp.get("recalled_frames")
        if rf:
            if "frame_paths" in rf:
                try:
                    from scripts.agent_data.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 200_704, "max_pixels": 401_408}
                rf = dict(rf)
                rf["frame_paths"] = _resolve_paths(rf["frame_paths"], base_path, data_dir)
            elif video_path:
                user_content.append({
                    "type": "video", "video": video_path,
                    "video_start": prompt_time_value(rf["time_range"][0]),
                    "video_end": prompt_time_value(rf["time_range"][1]),
                    "kv_scope": "recall",
                })
        try:
            from scripts.agent_data.config import (
                RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
            )
        except ImportError:
            _RTKW = {"min_pixels": 200_704, "max_pixels": 401_408}
        recall_payload = build_recall_result_user_content(
            rf,
            inp.get("recall_result") or {},
            frame_protocol=frame_protocol,
            min_pixels=_RTKW["min_pixels"],
            max_pixels=_RTKW["max_pixels"],
            render_layout=render_layout,
        )
        if recall_payload and user_content and recall_payload[0].get("type") == "text":
            recall_payload = [dict(recall_payload[0]), *recall_payload[1:]]
            recall_payload[0]["text"] = "\n" + str(recall_payload[0].get("text", ""))
        user_content.extend(recall_payload)
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
        queries_for_prompt,
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
    if str(memory_text or "").strip():
        user_content.append({
            "type": "text",
            "text": f"\n{memory_text.strip()}" if user_content else memory_text.strip(),
        })

    query_last = render_layout == RENDER_LAYOUT_QUERY_LAST

    # ── Active query + response history for that same query ─────────────
    queries = queries_for_prompt
    if queries and not inter_chunk and not query_last:
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": f"\n{qt}"})

    # ── Current visual chunk + frames ───────────────────────────────────
    # Inter-chunk compression is a text-memory action and does not consume a
    # visual timestep. Ordinary streaming rows carry only the current 1s
    # chunk (2 frames); the 8-chunk visual horizon is represented by recurrent
    # video KV, not by replaying old frames in every prompt.
    if not inter_chunk:
        vw = inp["visual_window"]
        current_start = chunk_idx * chunk_sec
        current_end = current_start + chunk_sec
        t_marker = f"<t={prompt_time_value(current_start)}>"
        user_content.append({
            "type": "text",
            "text": f"\n{t_marker}" if user_content else t_marker,
        })

        # Pass4 flat files may omit frame_paths — infer the current chunk by
        # video_id + chunk_idx. Older cached rows may carry a full visual
        # window in frame_paths; trim to the current chunk below.
        if "frame_paths" not in vw and "frames" in vw:
            vid = sample.get("video_id", "")
            if vid:
                from thinkstream.data.agent_protocol import (
                    FRAMES_PER_CHUNK as _FPC,
                )
                paths = [
                    f"{frame_rel_prefix}/{vid}/{name}"
                    for name in chunk_frame_filenames(chunk_idx, _FPC)
                ]
                vw["frame_paths"] = paths

        if "frame_paths" in vw:
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
            )
            try:
                from scripts.agent_data.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 200_704, "max_pixels": 401_408}
            append_visual_frames(
                user_content,
                _resolve_paths(vw["frame_paths"], base_path, data_dir)[-_FPC:],
                frame_protocol=frame_protocol,
                fps=float(_FPC / chunk_sec),
                start_frame_index=chunk_idx * _FPC,
                total_num_frames=(chunk_idx + 1) * _FPC,
                latest_start_frame_index=chunk_idx * _FPC,
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
                kv_scope="ordinary",
            )
        elif "frame_indices" in vw and video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": prompt_time_value(current_start),
                "video_end": prompt_time_value(current_end),
                "kv_scope": "ordinary",
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

    # ── Legacy single-turn recall payload ──────────────────────────────
    if not is_recall_multiturn and not inter_chunk and (
        inp.get("recalled_frames") or inp.get("recall_result")
    ):
        rf = _normalise_recalled_frames(
            inp,
            chunk_sec,
            inp.get("recall_result") or sample.get("recall_result"),
        ) or inp.get("recalled_frames")
        if rf and rf.get("frame_paths"):
            rf = dict(rf)
            rf["frame_paths"] = _resolve_paths(rf["frame_paths"], base_path, data_dir)
        try:
            from scripts.agent_data.config import (
                RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
            )
        except ImportError:
            _RTKW = {"min_pixels": 200_704, "max_pixels": 401_408}
        recall_payload = build_recall_result_user_content(
            rf,
            inp.get("recall_result") or {},
            frame_protocol=frame_protocol,
            min_pixels=_RTKW["min_pixels"],
            max_pixels=_RTKW["max_pixels"],
            render_layout=render_layout,
        )
        if recall_payload and user_content and recall_payload[0].get("type") == "text":
            recall_payload = [dict(recall_payload[0]), *recall_payload[1:]]
            recall_payload[0]["text"] = "\n" + str(recall_payload[0].get("text", ""))
        user_content.extend(recall_payload)

    messages.append({"role": "user", "content": user_content})

    # ── Assistant turn(s) ──────────────────────────────────────────────
    if is_recall_multiturn:
        # Shape B: 2 assistant turns sandwiching a tool turn.
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })

        # Tool turn — short recall status + optional historical frames.
        # Keep this renderer byte-aligned with runtime/eval.
        rr = sample.get("recall_result") or inp.get("recall_result") or {}
        rf = _normalise_recalled_frames(inp, chunk_sec, rr)
        if rf:
            rf = dict(rf)
            if rf.get("frame_paths"):
                rf["frame_paths"] = _resolve_paths(rf["frame_paths"], base_path, data_dir)
        try:
            from scripts.agent_data.config import (
                RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
            )
        except ImportError:
            _RTKW = {"min_pixels": 200_704, "max_pixels": 401_408}
        tool_payload = build_recall_result_user_content(
            rf,
            rr,
            frame_protocol=frame_protocol,
            min_pixels=_RTKW["min_pixels"],
            max_pixels=_RTKW["max_pixels"],
            render_layout=render_layout,
        )

        messages.append({
            "role": "tool",
            "tool_call_id": "recall",
            "content": tool_payload,
        })
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
                        meta["answer_style"] = "letter_plus_text"
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
    tool_schema_mode = "compress" if is_inter_chunk(sample) else "streaming"
    return {
        "trajectory_id": sample.get("trajectory_id", ""),
        "video_id": sample.get("video_id", ""),
        "card_id": sample.get("card_id", ""),
        "chunk_idx": sample.get("chunk_idx", -1),
        "sample_type": sample.get("sample_type", ""),
        "sample_id": sample.get("sample_id", ""),
        "frame_protocol": frame_protocol,
        "inter_chunk": is_inter_chunk(sample),
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


def _mark_render_contract_failure(sample: Dict[str, Any], reason: str) -> None:
    """Tag a sample with a non-destructive render-contract failure."""
    verification = dict(sample.get("verification") or {})
    reasons = list(verification.get("fail_reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    verification["passed"] = False
    verification["fail_reasons"] = reasons
    sample["verification"] = verification


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
    """First-turn think for recall_query SFT rows.

    Keep the current visual observation only. The tool_call itself teaches the
    recall action; appending a generic recall rationale caused SFT models to
    overfit fixed action text.
    """
    return _strip_recall_action_boilerplate(visual_think)


_RECALL_ACTION_BOILERPLATE_RE = re.compile(
    r"\s*(?:"
    r"Current visible evidence is insufficient|"
    r"The active query depends on elapsed context|"
    r"A related moment may have occurred earlier|"
    r"Before answering the pending query|"
    r"The query has stayed open long enough|"
    r"The current moment is relevant, but the answer also depends|"
    r"The status question depends on an event"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_recall_action_boilerplate(text: str) -> str:
    """Remove old recall-action rationale templates from first-turn thinks."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _RECALL_ACTION_BOILERPLATE_RE.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


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
    inter_chunk = is_inter_chunk(sample)
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
        parsed = parse_agent_output(text)
        if parsed.get("kind") == "answer":
            out.append(str(parsed.get("answer_text") or "").strip())
            continue
        for m in RESPONSE_BLOCK_RE.finditer(text):
            out.append((m.group(1) if m.group(1) is not None else m.group(2)).strip())
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
                f"sample={sample_id}: response row rendered empty/no </Response>"
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
    elif is_inter_chunk(sample):
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
    raise RuntimeError(
        "pass5_messages.convert is retired with flat/single-step rendering. "
        "Use scripts.agent_data.pass5.convert_file/convert_dir for trajectory SFT."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(
        "pass5_messages.py flat/single-step rendering is retired. "
        "Use `python -m scripts.agent_data.pass5` to render "
        "rendered/trajectory/*_trajectory.jsonl."
    )


if __name__ == "__main__":
    main()
