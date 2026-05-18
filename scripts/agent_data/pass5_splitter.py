"""Pass5 v2 — trajectory splitter + sample-to-turn translator.

Bridges pass3/pass4 schema → ``thinkstream.data.schema`` schema.

Input: a pass4 ``trajectory_record`` dict (one per video × trajectory_id),
       containing ``samples`` (chunk-sorted) + ``questions`` metadata.

Output: list of ``TrajectoryRow``-shaped dicts, one per (sub-)trajectory
        produced by splitting on compress events. Ready to be JSON-serialised
        and written by pass5.

Design choices (anchored to the v2 design doc, simplified per user feedback):
- Each compress event becomes a trajectory boundary (hard split, no merging).
- First (sub-)trajectory is ``from_start``; subsequent are ``from_compress``
  with a text-only raw <m> re-prefill user message followed by
  assistant("Memory loaded.") before the next visual turn.
- Compact-memory samples use standalone bare <m> assistant lines and do not
  expose a compress tool.
- Recall samples (multi-turn within one chunk) render as a single
  ``TurnSpec`` carrying the tool-2-turn pattern.
- This module is pure-function and stateless: no I/O, no global config.
"""
from __future__ import annotations

import os
import json
import re
import html
from typing import Callable, Dict, List, Optional, Tuple

from thinkstream.data.agent_protocol import (
    build_recall_result_user_content,
    canonical_answer_instruction,
    query_response_history_policy,
)
from thinkstream.data.agent_protocol import is_inter_chunk as sample_is_inter_chunk
from thinkstream.data.schema import (
    ACTION_COMPRESS_SELECT,
    ACTION_MEMORY_UPDATE,
    ACTION_RECALL,
    ACTION_RESPONSE,
    ACTION_SILENT,
    AssistantSpec,
    ChunkUserSpec,
    COMPACT_MEMORY_SYSTEM_PROMPT,
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_MIN_PIXELS,
    MEMORY_LOAD_ACK,
    MemoryEntry,
    QuerySpec,
    TOOL_NAME_COMPRESS,
    TOOL_NAME_RECALL,
    TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
    TRAJ_TYPE_FROM_COMPRESS,
    TRAJ_TYPE_FROM_START,
    TrajectorySpec,
    TurnSpec,
    build_tool_response_message,
    render_trajectory_messages,
)

# Runtime visual KV window size. Trajectory headers never bulk-load old
# visual chunks; they only prefill text memory.
SLIDING_WINDOW_CHUNKS = 8
QUERY_POLICY_CURRENT_ASK_BOUNDARY = "current_ask_boundary"
QUERY_POLICY_ASK_ANSWER = "ask_answer"
QUERY_POLICY_ASK_WINDOW4_ANSWER = "ask_window4_answer"
QUERY_POLICY_ASK_WINDOW8_ANSWER = "ask_window8_answer"
QUERY_POLICY_PERIODIC8_ANSWER = "periodic8_answer"
QUERY_POLICY_SUPPORT_ANSWER = "support_answer"
QUERY_POLICY_EVERY_ACTIVE = "every_active"
QUERY_POLICY_ADAPTIVE = "adaptive"
VALID_QUERY_INJECTION_POLICIES = {
    QUERY_POLICY_CURRENT_ASK_BOUNDARY,
    QUERY_POLICY_ASK_ANSWER,
    QUERY_POLICY_ASK_WINDOW4_ANSWER,
    QUERY_POLICY_ASK_WINDOW8_ANSWER,
    QUERY_POLICY_PERIODIC8_ANSWER,
    QUERY_POLICY_SUPPORT_ANSWER,
    QUERY_POLICY_EVERY_ACTIVE,
    QUERY_POLICY_ADAPTIVE,
}
DEFAULT_QUERY_INJECTION_POLICY = QUERY_POLICY_ADAPTIVE
ADAPTIVE_QUERY_REFRESH_PERIOD = int(os.environ.get("THINKSTREAM_QUERY_REFRESH_PERIOD", "8"))


# ---------------------------------------------------------------------------
# Pass3 output text parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"</Response>\s*(.*?)\s*$", re.DOTALL)
_LEGACY_ANSWER_RE = re.compile(
    r"<(?:answer|response)>\s*(.*?)\s*</(?:answer|response)>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_SILENCE_RE = re.compile(r"</Silence>\s*$", re.DOTALL)
_LEGACY_SILENCE_RE = re.compile(r"<silent>\s*$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_MEM_RE = re.compile(r"<MEM>\s*(.*?)\s*</MEM>", re.DOTALL | re.IGNORECASE)
_OPTION_LABEL_RE = re.compile(r"^\s*(?:\(([A-E])\)|([A-E])[\).:])\s*(.*)\s*$", re.DOTALL)
_M_LINE_RE = re.compile(
    r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>',
    re.DOTALL | re.IGNORECASE,
)
_INT_RE = re.compile(r"\d+")


def _normalise_time_range_arg(value):
    """Normalize teacher-emitted ranges to the tool schema's integer pair."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return value
    if isinstance(value, str):
        nums = _INT_RE.findall(value)
        if len(nums) >= 2:
            return [int(nums[0]), int(nums[1])]
        if len(nums) == 1:
            n = int(nums[0])
            return [n, n]
    return value


def _normalise_recall_time_arg(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    return int(num) if num.is_integer() else num


def _normalise_recall_args(args: Dict) -> Dict:
    start = _normalise_recall_time_arg(args.get("start_time"))
    end = _normalise_recall_time_arg(args.get("end_time"))
    if start is None or end is None:
        legacy_range = _normalise_time_range_arg(
            args.get("time_range") or args.get("time")
        )
        if isinstance(legacy_range, (list, tuple)) and len(legacy_range) >= 2:
            start = _normalise_recall_time_arg(legacy_range[0])
            end = _normalise_recall_time_arg(legacy_range[1])
    if start is None or end is None:
        raise ValueError(
            "recall tool_call requires start_time and end_time; "
            "legacy time_range/time arguments are obsolete"
        )
    return {"start_time": start, "end_time": end}


def parse_assistant_output_to_spec(text: str, chunk_idx: int) -> AssistantSpec:
    """Parse a pass3-format assistant output string into a v2 AssistantSpec.

    Pass3 emits ``<think>...</think>`` plus ONE of:
    - ``</Response> X``            — non-empty: response action with answer X
    - ``</Silence>``               — empty: silent action
    - ``<tool_call>{...}</tool_call>`` — recall tool_call
    - bare ``<m t="...">...</m>`` lines — compact-memory update

    Tool call ids are minted deterministically from ``chunk_idx`` so the
    same sample always produces the same id (useful for diffing).
    """
    think_match = _THINK_RE.search(text)
    think = think_match.group(1).strip() if think_match else ""

    tool_match = _TOOL_CALL_RE.search(text)
    if tool_match:
        try:
            tool_obj = json.loads(tool_match.group(1))
        except json.JSONDecodeError:
            tool_obj = {}
        name = (tool_obj.get("name") or "").strip()
        args = tool_obj.get("arguments") or {}
        if name == TOOL_NAME_COMPRESS:
            raise ValueError("compress tool_call is obsolete; emit bare <m> compact-memory lines")
        if name == TOOL_NAME_RECALL:
            recall_args = _normalise_recall_args(args)
            return AssistantSpec(
                think=think,
                action_type=ACTION_RECALL,
                tool_call_id=f"rec_{chunk_idx}",
                tool_arguments=recall_args,
            )
        # Unknown tool name — fall through to silent so we don't crash; the
        # caller should log this case in their audit.
        return AssistantSpec(think=think, action_type=ACTION_SILENT)

    mem_match = _MEM_RE.search(text)
    if mem_match or _M_LINE_RE.search(text or ""):
        mem_text = _normalise_mem_block(text)
        return AssistantSpec(
            think=think,
            action_type=ACTION_MEMORY_UPDATE,
            tool_arguments={"memory_text": mem_text},
        )

    ans_match = _ANSWER_RE.search(text)
    if ans_match and ans_match.group(1).strip():
        return AssistantSpec(
            think=think,
            action_type=ACTION_RESPONSE,
            response_text=ans_match.group(1).strip(),
        )
    legacy_ans_match = _LEGACY_ANSWER_RE.search(text)
    if legacy_ans_match and legacy_ans_match.group(1).strip():
        return AssistantSpec(
            think=think,
            action_type=ACTION_RESPONSE,
            response_text=legacy_ans_match.group(1).strip(),
        )
    if _SILENCE_RE.search(text) or _LEGACY_SILENCE_RE.search(text):
        return AssistantSpec(think=think, action_type=ACTION_SILENT)
    return AssistantSpec(think=think, action_type=ACTION_SILENT)


# ---------------------------------------------------------------------------
# Question metadata → QuerySpec map
# ---------------------------------------------------------------------------

def _query_answer_instruction(question: Dict) -> Optional[str]:
    """Return the model-visible answer-format line for a structured question."""
    instruction = canonical_answer_instruction(question)
    if instruction:
        return instruction
    fallback = str(question.get("answer_instruction") or "").strip()
    return fallback or None


def build_question_metadata(questions_list: List[Dict]) -> List[Dict]:
    """Per-question metadata for cross-segment inheritance tracking.

    Returns a list of dicts ``{ask_chunk, answer_chunks, max_answer_chunk,
    spec}`` for each well-formed question. ``answer_chunks`` defaults to
    ``[ask_chunk]`` when the upstream record doesn't carry it (single-emit
    cards). Used by ``render_trajectory_record_to_rows`` to fill
    ``inherited_queries`` / ``inherited_responses`` on from_compress
    segments — questions whose ``ask_chunk`` is in a prior segment but
    whose answers extend into (or past) the new segment must be carried
    across the compress boundary, otherwise the model in the from_compress
    segment sees a gold ``</Response> X`` token at some chunk
    with no upstream query as context.
    """
    out: List[Dict] = []
    for q in questions_list:
        ask = q.get("ask_chunk", -1)
        if not isinstance(ask, int) or ask < 0:
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        raw_ans = q.get("answer_chunks") or []
        answer_chunks = sorted(
            int(c) for c in raw_ans if isinstance(c, int) and int(c) >= 0
        )
        if not answer_chunks:
            answer_chunks = [int(ask)]
        support_chunks = sorted(
            int(c) for c in (q.get("support_chunks") or [])
            if isinstance(c, int) and int(c) >= 0
        )
        options = q.get("options") or None
        instruction = _query_answer_instruction(q)
        history_policy = query_response_history_policy(q)
        out.append({
            "ask_chunk": int(ask),
            "answer_chunks": answer_chunks,
            "max_answer_chunk": max(answer_chunks),
            "support_chunks": support_chunks,
            "response_history_policy": history_policy,
            "answer_form": q.get("answer_form", ""),
            "question_type": q.get("question_type", ""),
            "question_way": q.get("question_way", ""),
            "evidence_type": q.get("evidence_type", ""),
            "family": q.get("family", q.get("task", "")),
            "task": q.get("task", ""),
            "spec": QuerySpec(
                text=text,
                options=list(options) if options else None,
                answer_format=instruction,
                answer_form=q.get("answer_form", ""),
                answer_style=q.get("answer_style", ""),
                response_history_policy=history_policy,
            ),
        })
    return out


def normalize_query_injection_policy(policy: Optional[str] = None) -> str:
    value = (
        policy
        or os.environ.get("THINKSTREAM_QUERY_INJECTION_POLICY")
        or DEFAULT_QUERY_INJECTION_POLICY
    )
    value = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_QUERY_INJECTION_POLICY,
        "adaptive": QUERY_POLICY_ADAPTIVE,
        "auto": QUERY_POLICY_ADAPTIVE,
        "current": QUERY_POLICY_CURRENT_ASK_BOUNDARY,
        "ask_only": QUERY_POLICY_CURRENT_ASK_BOUNDARY,
        "ask_once": QUERY_POLICY_CURRENT_ASK_BOUNDARY,
        "ask_boundary": QUERY_POLICY_CURRENT_ASK_BOUNDARY,
        "answer": QUERY_POLICY_ASK_ANSWER,
        "force_answer": QUERY_POLICY_ASK_ANSWER,
        "window4": QUERY_POLICY_ASK_WINDOW4_ANSWER,
        "direct4": QUERY_POLICY_ASK_WINDOW4_ANSWER,
        "window8": QUERY_POLICY_ASK_WINDOW8_ANSWER,
        "direct8": QUERY_POLICY_ASK_WINDOW8_ANSWER,
        "periodic8": QUERY_POLICY_PERIODIC8_ANSWER,
        "support": QUERY_POLICY_SUPPORT_ANSWER,
        "every": QUERY_POLICY_EVERY_ACTIVE,
    }
    value = aliases.get(value, value)
    if value not in VALID_QUERY_INJECTION_POLICIES:
        raise ValueError(
            f"unknown query_injection_policy={policy!r}; valid="
            f"{sorted(VALID_QUERY_INJECTION_POLICIES)}"
        )
    return value


def _query_active_at(q_meta: Dict, chunk_idx: int) -> bool:
    return int(q_meta["ask_chunk"]) <= int(chunk_idx) <= int(q_meta["max_answer_chunk"])


def _query_uses_response_history(q_meta: Dict) -> bool:
    return str(q_meta.get("response_history_policy") or "include").lower() != "omit"


def _is_cumulative_count_query(q_meta: Dict) -> bool:
    if _query_uses_response_history(q_meta) is False:
        return False
    answer_form = str(q_meta.get("answer_form") or "").strip().lower()
    question_way = str(q_meta.get("question_way") or "").strip().lower()
    fields = " ".join(
        str(q_meta.get(k) or "")
        for k in ("family", "task", "question_way", "evidence_type")
    ).upper()
    tokens = {tok for tok in re.split(r"[^A-Z0-9]+", fields) if tok}
    return (
        answer_form == "number"
        or question_way == "repeated_count"
        or bool({"F5", "REC"} & tokens)
    )


def _should_force_query_adaptive(q_meta: Dict, chunk_idx: int) -> bool:
    ask = int(q_meta["ask_chunk"])
    final = int(q_meta["max_answer_chunk"])
    chunk = int(chunk_idx)
    if not (ask < chunk <= final):
        return False
    answer_chunks = {int(c) for c in (q_meta.get("answer_chunks") or [])}
    if chunk in answer_chunks:
        return True
    if _is_cumulative_count_query(q_meta) and (chunk - 1) in answer_chunks:
        # After a count response, immediately refresh the same question plus
        # response_history so the next timestep carries the updated running
        # state instead of relying on a distant KV trace.
        return True
    period = max(0, int(ADAPTIVE_QUERY_REFRESH_PERIOD))
    if period and chunk > ask and (chunk - ask) % period == 0:
        return True
    return False


def _should_force_query_at_chunk(q_meta: Dict, chunk_idx: int, policy: str) -> bool:
    ask = int(q_meta["ask_chunk"])
    final = int(q_meta["max_answer_chunk"])
    chunk = int(chunk_idx)
    if not (ask <= chunk <= final):
        return False
    answer_chunks = set(int(c) for c in (q_meta.get("answer_chunks") or []))
    if policy == QUERY_POLICY_ADAPTIVE:
        return _should_force_query_adaptive(q_meta, chunk_idx)
    if policy == QUERY_POLICY_CURRENT_ASK_BOUNDARY:
        return False
    if policy == QUERY_POLICY_ASK_ANSWER:
        return chunk in answer_chunks
    if policy == QUERY_POLICY_ASK_WINDOW4_ANSWER:
        return chunk in answer_chunks or ask < chunk <= min(final, ask + 3)
    if policy == QUERY_POLICY_ASK_WINDOW8_ANSWER:
        return chunk in answer_chunks or ask < chunk <= min(final, ask + 7)
    if policy == QUERY_POLICY_PERIODIC8_ANSWER:
        return chunk in answer_chunks or (chunk > ask and (chunk - ask) % 8 == 0)
    if policy == QUERY_POLICY_SUPPORT_ANSWER:
        support_chunks = set(int(c) for c in (q_meta.get("support_chunks") or []))
        return chunk in answer_chunks or chunk in support_chunks
    if policy == QUERY_POLICY_EVERY_ACTIVE:
        return chunk > ask
    return False


def _prior_responses_for_question(
    q_meta: Dict,
    chunk_idx: int,
    responses_at_chunk: Dict[int, str],
) -> List[Tuple[int, str]]:
    if not _query_uses_response_history(q_meta):
        return []
    inherited_r: List[Tuple[int, str]] = []
    for ans_chunk in q_meta.get("answer_chunks") or []:
        ans_chunk = int(ans_chunk)
        if ans_chunk < int(chunk_idx) and ans_chunk in responses_at_chunk:
            inherited_r.append((ans_chunk, responses_at_chunk[ans_chunk]))
    return sorted(inherited_r)


def _append_inherited_query_once(
    user: ChunkUserSpec,
    q_meta: Dict,
    responses: List[Tuple[int, str]],
) -> None:
    ask_chunk = int(q_meta["ask_chunk"])
    spec = q_meta["spec"]
    existing = list(user.inherited_queries or [])
    if not any(
        int(old_ask) == ask_chunk
        and getattr(old_spec, "text", "") == getattr(spec, "text", "")
        for old_ask, old_spec in existing
    ):
        existing.append((ask_chunk, spec))
        user.inherited_queries = sorted(existing, key=lambda item: int(item[0]))
    if responses:
        current = list(user.inherited_responses or [])
        seen = {(int(t), str(v)) for t, v in current}
        for item in responses:
            key = (int(item[0]), str(item[1]))
            if key not in seen:
                current.append(item)
                seen.add(key)
        user.inherited_responses = sorted(current)


def build_questions_by_chunk(questions_list: List[Dict]) -> Dict[int, List[QuerySpec]]:
    """Build ``{ask_chunk: [QuerySpec, ...]}`` from a pass4 questions array.

    Only the canonical ``ask_chunk`` is used; multi-emit cards inject their
    extra emissions via answer_chunks (handled separately by the loss
    routing in pass4). For QuerySpec rendering we just need the user-visible
    question text + options + answer instruction at the asking chunk.

    When multiple questions share the same ``ask_chunk`` the value is a list
    (insertion order preserved) — earlier code overwrote, losing all but the
    last. The caller may flatten / pick / iterate as needed; the trajectory
    renderer's ``ChunkUserSpec.active_query`` takes the first one for
    backward compatibility.
    """
    out: Dict[int, List[QuerySpec]] = {}
    for q in questions_list:
        ask = q.get("ask_chunk", -1)
        if not isinstance(ask, int) or ask < 0:
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        options = q.get("options") or None
        instruction = _query_answer_instruction(q)
        spec = QuerySpec(
            text=text,
            options=list(options) if options else None,
            answer_format=instruction,
            answer_form=q.get("answer_form", ""),
            answer_style=q.get("answer_style", ""),
            response_history_policy=query_response_history_policy(q),
        )
        out.setdefault(ask, []).append(spec)
    return out


def _strip_option_label(option: str) -> str:
    m = _OPTION_LABEL_RE.match(str(option or ""))
    return (m.group(3) if m else str(option or "")).strip()


def _mc_correct_text(question: Dict) -> str:
    options = list(question.get("options") or [])
    correct = str(question.get("correct_option") or "").strip().upper()
    if not options or len(correct) != 1:
        return ""
    idx = ord(correct) - ord("A")
    if idx < 0 or idx >= len(options):
        return ""
    return _strip_option_label(str(options[idx]))


def _mc_accepted_answers(options: List[str], correct: str) -> List[str]:
    correct = str(correct or "").strip().upper()
    text = ""
    if len(correct) == 1:
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(options):
            text = _strip_option_label(str(options[idx]))
    out = []
    if correct:
        out.append(correct)
    if correct and text:
        out.append(f"{correct}) {text}")
    if text:
        out.append(text)
    seen = set()
    return [x for x in out if x and not (x.lower() in seen or seen.add(x.lower()))]


def _normalise_mc_question_for_render(question: Dict) -> Dict:
    """Keep rendered MC prompts, metadata, and SFT targets on letter+text."""
    if str(question.get("answer_form") or "").strip() != "multiple_choice":
        return question
    out = dict(question)
    options = list(out.get("options") or [])
    correct = str(out.get("correct_option") or "").strip().upper()
    if len(correct) != 1 or not options:
        return out
    if not (0 <= ord(correct) - ord("A") < len(options)):
        return out
    correct_text = _mc_correct_text(out)
    target = f"{correct}) {correct_text}" if correct_text else correct
    out["correct_option"] = correct
    out["answer_style"] = "letter_plus_text"
    out["answer_instruction"] = canonical_answer_instruction(out)
    if correct_text:
        out["gold_answer"] = correct_text
        out["canonical_answer"] = correct_text
        out["correct_answer_text"] = correct_text
        out["accepted_answers"] = _mc_accepted_answers(options, correct)
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
    out["sft_answer"] = target
    return out


def _build_mc_response_targets_by_chunk(questions_list: List[Dict]) -> Dict[int, str]:
    """Map MC answer chunks to the canonical letter+text SFT surface form."""
    targets: Dict[int, str] = {}
    for q in questions_list:
        if str(q.get("answer_form") or "").strip() != "multiple_choice":
            continue
        correct = str(q.get("correct_option") or "").strip().upper()
        options = list(q.get("options") or [])
        if len(correct) != 1 or not (0 <= ord(correct) - ord("A") < len(options)):
            continue
        correct_text = _mc_correct_text(q)
        target = f"{correct}) {correct_text}" if correct_text else correct
        chunks = []
        for emit in q.get("per_emit_answers") or []:
            if isinstance(emit, dict) and "chunk" in emit:
                try:
                    chunks.append(int(emit["chunk"]))
                except (TypeError, ValueError):
                    pass
        if not chunks:
            chunks = [int(c) for c in (q.get("answer_chunks") or []) if isinstance(c, int)]
        if not chunks and isinstance(q.get("ask_chunk"), int):
            chunks = [int(q["ask_chunk"])]
        for chunk in chunks:
            targets.setdefault(chunk, target)

    return targets


def _apply_response_override(spec: AssistantSpec, response_override: Optional[str]) -> AssistantSpec:
    if (
        response_override
        and spec.action_type == ACTION_RESPONSE
        and spec.response_text
    ):
        spec.response_text = response_override
    return spec


# ---------------------------------------------------------------------------
# Recall multi-turn pack → TurnSpec
# ---------------------------------------------------------------------------

def _format_recall_result_for_tool_response(
    recall_result: Optional[Dict],
) -> str:
    """Format a pass3 recall_result dict into a tool_response content string.

    The original pass3 schema may contain retrieved memory text. Keep it out
    of the model-visible tool response; the answer should be grounded in
    returned frames when they exist.
    """
    if not recall_result:
        return "(no recall hits)"
    time_range = recall_result.get("time_range") or recall_result.get("time")
    if time_range:
        return f"The recall tool returned no historical video frames for t={time_range}."
    return "The recall tool returned no historical video frames."


def _build_recall_tool_response_content(sample: Dict):
    """Build the Shape-B recall tool payload.

    The tool response must carry recalled visual frames when available. Any
    retriever text stays out of the prompt so answer supervision is grounded
    in the returned frames rather than retrieved text.
    """
    recall_result = (
        sample.get("recall_result")
        or (sample.get("input") or {}).get("recall_result")
        or {}
    )
    recalled_frames = (
        sample.get("recalled_frames")
        or (sample.get("input") or {}).get("recalled_frames")
        or None
    )
    if recalled_frames or recall_result:
        items = build_recall_result_user_content(
            recalled_frames,
            recall_result,
            min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
            max_pixels=DEFAULT_VIDEO_MAX_PIXELS,
        )
        if items:
            return items
    return _format_recall_result_for_tool_response(recall_result)


def translate_recall_sample(
    sample: Dict,
    user: ChunkUserSpec,
    response_override: Optional[str] = None,
) -> TurnSpec:
    """Recall sample → one TurnSpec with the tool-2-turn pattern."""
    chunk_idx = sample["chunk_idx"]
    turn1_text = sample.get("v12_assistant_turn_1", "") or ""
    turn2_text = sample.get("v12_assistant_turn_2", "") or ""

    a1 = parse_assistant_output_to_spec(turn1_text, chunk_idx)
    # Force the first assistant turn to be a tool_call; if the upstream
    # parsing somehow yielded a silent/response, repair it conservatively.
    if a1.action_type != ACTION_RECALL:
        # Treat as silent fallback; tool_response/followup will be skipped.
        return TurnSpec(user=user, assistant=a1)

    tool_content = _build_recall_tool_response_content(sample)
    if isinstance(tool_content, list):
        tool_resp = build_tool_response_message(
            tool_call_id=a1.tool_call_id,
            content_items=tool_content,
        )
    else:
        tool_resp = build_tool_response_message(
            tool_call_id=a1.tool_call_id,
            content_text=tool_content,
        )
    a2 = _apply_response_override(
        parse_assistant_output_to_spec(turn2_text, chunk_idx),
        response_override,
    )
    return TurnSpec(
        user=user,
        assistant=a1,
        tool_response=tool_resp,
        followup_assistant=a2,
    )


# ---------------------------------------------------------------------------
# Pass3 sample → v2 TurnSpec
# ---------------------------------------------------------------------------

FrameResolver = Callable[[int], List[str]]


def translate_sample_to_turn(
    sample: Dict,
    questions_by_chunk: Dict[int, List[QuerySpec]],
    frame_resolver: FrameResolver,
    response_override: Optional[str] = None,
) -> TurnSpec:
    """Convert one pass3/pass4 sample into a v2 TurnSpec.

    ``questions_by_chunk`` maps chunk_idx → list of QuerySpecs (multiple
    questions may share the same ask_chunk after the P3.12 fix). The
    renderer picks the FIRST query for ``ChunkUserSpec.active_query`` —
    multi-question-per-chunk fanout is left to the caller / future work.
    """
    chunk_idx = int(sample.get("chunk_idx", 0))
    sample_type = sample.get("sample_type", ACTION_SILENT)
    if is_compress_sample(sample):
        raise ValueError(
            "compact-memory/compress samples must render as standalone "
            "compact_memory_update rows, not streaming trajectory turns"
        )

    queries_here = questions_by_chunk.get(chunk_idx) or []
    active = queries_here[0] if queries_here else None
    user = ChunkUserSpec(
        chunk_idx=chunk_idx,
        frame_paths=list(frame_resolver(chunk_idx)),
        active_query=active,
    )

    if sample_type == "recall":
        return translate_recall_sample(sample, user, response_override)

    # silent / response / compress (single-turn output)
    output_text = sample.get("output", "") or ""
    assistant = _apply_response_override(
        parse_assistant_output_to_spec(output_text, chunk_idx),
        response_override,
    )
    if assistant.action_type == ACTION_COMPRESS_SELECT:
        raise ValueError(
            "compress tool calls are not valid streaming actions; rerun pass3 "
            "so compact memory is emitted as standalone compact_memory_update"
        )
    return TurnSpec(user=user, assistant=assistant)


# ---------------------------------------------------------------------------
# Trajectory splitter
# ---------------------------------------------------------------------------

def is_compress_sample(sample: Dict) -> bool:
    return (
        sample.get("sample_type") == "compress"
        or sample_is_inter_chunk(sample)
    )


def split_samples_by_compress(samples: List[Dict]) -> List[List[Dict]]:
    """Split a chunk-sorted samples list at compress boundaries.

    Each segment INCLUDES its terminating compress sample. The streaming part
    is rendered without that sample, then the compact update is emitted as a
    standalone text-only row. The compact update is an inter-chunk event before
    its ``chunk_idx``; the next segment starts at that same trigger chunk.

    A trailing segment without a compress (video ended before compress
    triggered) is still emitted.
    """
    segments: List[List[Dict]] = []
    current: List[Dict] = []
    for s in samples:
        current.append(s)
        if is_compress_sample(s):
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _sample_sort_key(sample: Dict) -> Tuple[int, int]:
    # Pass2 emits trigger chunk N before appending the current chunk's think,
    # so the compact-memory update summarizes chunks <= N-1 and must be
    # rendered before the normal streaming turn for chunk N.
    order = {
        "compress": -1,
        "recall_query": 0,
        "recall_response": 1,
        "recall": 1,
        "response": 2,
        "silent": 4,
        "recall_silent": 5,
    }
    sample_type = str(sample.get("sample_type") or "")
    return int(sample.get("chunk_idx", 0)), (
        -1 if is_compress_sample(sample) else order.get(sample_type, 6)
    )


def parse_mem_block_to_entries(mem_text: str) -> List[MemoryEntry]:
    """Parse compact-memory <m> lines into inherited memory entries."""
    m = _MEM_RE.search(mem_text or "")
    body = m.group(1) if m else str(mem_text or "")
    entries: List[MemoryEntry] = []
    for line in _M_LINE_RE.finditer(body):
        start = line.group(1)
        end = line.group(2)
        time_str = f"{int(start)}-{int(end)}" if end is not None else str(int(start))
        text = html.unescape(re.sub(r"\s+", " ", line.group(3)).strip())
        if text:
            entries.append(MemoryEntry(time_str=time_str, text=text))
    return sorted(entries, key=lambda e: (e.start_sec, e.end_sec))


def entries_to_mem_block(entries: List[MemoryEntry]) -> str:
    lines = [e.to_text() for e in sorted(entries, key=lambda x: (x.start_sec, x.end_sec))]
    return "\n".join(lines)


def extract_compress_summary(compress_sample: Dict) -> Tuple[str, List[int]]:
    """Pull the gold compression summary text and source chunks from a
    compress sample. Used to build the next trajectory's prefix_memory.

    Priority:
      1. ``gold_caption`` (pass3c stamps this from the pass2 oracle)
      2. parse the tool_call arguments in ``output`` for ``summary`` /
         ``text`` field
      3. empty string fallback
    """
    def _wrap_legacy_summary(raw_text: str, raw_chunks: List[int]) -> str:
        body = str(raw_text or "").strip()
        if not body:
            return body
        if _M_LINE_RE.search(body):
            return _normalise_mem_block(body)
        if raw_chunks:
            start, end = min(raw_chunks), max(raw_chunks)
        else:
            start = end = int(compress_sample.get("chunk_idx", 0))
        safe = html.escape(body, quote=False)
        return f'  <m t="{start}-{end}">{safe}</m>'

    text = (compress_sample.get("gold_caption") or "").strip()
    chunks = list(compress_sample.get("gold_compress_chunks") or [])
    if text:
        if _M_LINE_RE.search(text) and not chunks:
            for entry in parse_mem_block_to_entries(text):
                chunks.extend(range(entry.start_sec, entry.end_sec + 1))
        return _wrap_legacy_summary(text, sorted(set(int(c) for c in chunks))), chunks

    output = compress_sample.get("output", "") or ""
    mem_match = _MEM_RE.search(output)
    if mem_match:
        text = _normalise_mem_block(output)
        if not chunks:
            for entry in parse_mem_block_to_entries(text):
                chunks.extend(range(entry.start_sec, entry.end_sec + 1))
        return text, sorted(set(int(c) for c in chunks))

    tool_match = _TOOL_CALL_RE.search(output)
    if tool_match:
        try:
            tool_obj = json.loads(tool_match.group(1))
            args = tool_obj.get("arguments") or {}
            text = (args.get("text") or args.get("summary") or "").strip()
            tr = args.get("time_range") or []
            if isinstance(tr, list) and len(tr) == 2 and not chunks:
                chunks = list(range(int(tr[0]), int(tr[1])))
        except json.JSONDecodeError:
            pass
    return _wrap_legacy_summary(text, sorted(set(int(c) for c in chunks))), chunks


def _normalise_mem_block(mem_text: str) -> str:
    """Return compact-memory text, preferring normalized <m> lines."""
    raw = str(mem_text or "").strip()
    matches = list(re.finditer(
        r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>.*?</m>',
        raw,
        re.DOTALL | re.IGNORECASE,
    ))
    if not matches:
        return raw
    return "\n".join(m.group(0).strip() for m in matches)


def _memory_update_input_from_sample(compress_sample: Dict) -> str:
    """User payload for the standalone compact-memory update row.

    New pass2/retrofit stamps the exact teacher input as memory_update_input.
    Older rows did not, so keep a conservative fallback that still has the
    right task shape without inventing visual evidence.
    """
    direct = (
        compress_sample.get("memory_update_input")
        or (compress_sample.get("metadata") or {}).get("memory_update_input")
        or (compress_sample.get("input") or {}).get("memory_update_input")
    )
    if str(direct or "").strip():
        return str(direct).strip()

    mem = (compress_sample.get("input") or {}).get("memory") or {}
    old_lines: List[str] = []
    for item in mem.get("compressed_segments") or []:
        if isinstance(item, dict):
            t = item.get("time_range") or item.get("t") or ""
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                t = f"{int(t[0])}-{int(t[1])}"
            text = str(item.get("text") or "").strip()
            if text:
                old_lines.append(f'  <m t="{t}">{text}</m>')
    think_lines: List[str] = []
    for item in mem.get("recent_thinks") or []:
        if isinstance(item, dict):
            t = item.get("chunk_idx", item.get("time", ""))
            if isinstance(t, (list, tuple)) and len(t) >= 1:
                t = int(float(t[0]))
            text = str(item.get("text") or item.get("think") or "").strip()
            if text:
                think_lines.append(f'  <c t="{t}">{text}</c>')
    old_block = "\n".join(old_lines) if old_lines else "(empty)"
    new_block = "\n".join(think_lines) if think_lines else "(no recent captions available)"
    return (
        f"OLD_MEMORY:\n{old_block}\n\n"
        f"NEW_CAPTIONS:\n{new_block}\n\n"
        "Return only compact-memory XML lines:\n"
        "<m t=\"start-end\">one concise summary for that exact source range.</m>\n\n"
        "Use real input timestamps. If OLD_MEMORY is not empty, keep useful old "
        "memory in at least one line. If NEW_CAPTIONS is not empty, cover the "
        "latest new captions. If the source provides one contiguous summary range, "
        "keep it as one line; do not invent finer timestamp segments. Do not "
        "output NEW_MEMORY, markdown, prose, "
        "analysis, or any text outside the <m> lines."
    )


# ---------------------------------------------------------------------------
# Top-level entry: render one pass4 trajectory record → list of output rows
# ---------------------------------------------------------------------------

def render_trajectory_record_to_rows(
    traj_record: Dict,
    frame_resolver: FrameResolver,
    query_injection_policy: Optional[str] = None,
) -> List[Dict]:
    """Render one pass4 trajectory_record into a list of v2 trajectory rows.

    Each compress event terminates a sub-trajectory. In pass2/retrofit data the
    compact row's ``chunk_idx`` is the first visual chunk after the compressed
    raw range (``max(gold_compress_chunks) == chunk_idx - 1``). The boundaries are:
      - sub-trajectory 0: from_start, visual chunks before K1.
        Then a standalone compact-memory row:
        system(compact prompt) -> user(old memory + captions) -> asst(<m> lines).
      - sub-trajectory 1: from_compress, chunks K1..K2-1.
        Prefix: user(<m> lines) -> assistant("Memory loaded."), then the first real
        visual turn for chunk K1.
      - sub-trajectory 2..N: same as #1.

    Memory state is replaced by parsed <m> lines at each boundary; the exact
    raw <m> text is reused for re-prefill.
    """
    raw_samples = traj_record.get("samples") or []
    if not raw_samples:
        return []
    query_policy = normalize_query_injection_policy(query_injection_policy)
    samples = sorted(raw_samples, key=_sample_sort_key)
    segments = split_samples_by_compress(samples)
    questions_raw = [
        _normalise_mc_question_for_render(q)
        for q in (traj_record.get("questions") or [])
        if isinstance(q, dict)
    ]
    questions_by_chunk = build_questions_by_chunk(questions_raw)
    mc_response_targets = _build_mc_response_targets_by_chunk(questions_raw)
    # Per-question metadata + per-chunk emitted response, indexed once for
    # the whole record so we can answer "is this question still open at
    # segment boundary N?" in O(1) per check.
    question_metadata = build_question_metadata(questions_raw)
    responses_at_chunk: Dict[int, str] = {}
    for s in samples:
        if is_compress_sample(s):
            continue
        c = int(s.get("chunk_idx", 0))
        output_text = s.get("output", "") or ""
        spec = parse_assistant_output_to_spec(output_text, c)
        if spec.action_type == ACTION_RESPONSE and spec.response_text:
            responses_at_chunk[c] = mc_response_targets.get(c, spec.response_text)

    video_id = traj_record.get("video_id", "")
    parent_traj_id = traj_record.get("trajectory_id", "")

    # Accumulated memory state across all sub-trajectories of this video.
    # ``memory_prefill_text`` is the exact compact-memory text injected into the next
    # trajectory. ``memory_state`` is only parsed diagnostics / fallback state;
    # rendering does not parse and rewrite that text.
    memory_state: List[MemoryEntry] = []
    memory_prefill_text: Optional[str] = None

    rows: List[Dict] = []
    for seg_idx, seg_samples in enumerate(segments):
        traj_type = (
            TRAJ_TYPE_FROM_START if seg_idx == 0 else TRAJ_TYPE_FROM_COMPRESS
        )
        terminal_compress = (
            seg_samples[-1] if is_compress_sample(seg_samples[-1]) else None
        )
        stream_samples = seg_samples[:-1] if terminal_compress else seg_samples
        leaked_compress = [
            s.get("sample_id") or s.get("chunk_idx")
            for s in stream_samples
            if is_compress_sample(s)
        ]
        if leaked_compress:
            raise ValueError(
                "compact-memory/compress samples leaked into streaming segment: "
                + ", ".join(str(x) for x in leaked_compress[:5])
            )

        # Snapshot memory state BEFORE consuming this segment. The exact text
        # snapshot is what the model inherits at the trajectory boundary.
        inherited_snapshot = list(memory_state)
        inherited_mem_text = memory_prefill_text

        # Build per-sample streaming TurnSpecs. The terminal compact-memory
        # update, if present, is rendered later as its own text-only system row.
        turns: List[TurnSpec] = []
        for s in stream_samples:
            chunk_idx = int(s.get("chunk_idx", 0))
            turn = translate_sample_to_turn(
                s,
                questions_by_chunk,
                frame_resolver,
                response_override=mc_response_targets.get(chunk_idx),
            )
            turns.append(turn)
            think_text = (turn.assistant.think or "").strip()
            if think_text:
                memory_state.append(MemoryEntry(
                    time_str=str(chunk_idx),
                    text=think_text,
                ))

        # Attach open queries / prior responses to the FIRST real turn of a
        # from_compress segment. Compact memory itself is inserted later as a
        # separate text-only user prefill followed by a short assistant ack. Do
        # not also render it as wrapped memory on the visual turn, and do not bulk
        # load prior visual chunks here; the runtime KV starts from text
        # compact state and then receives the next current chunk.
        # Uses the pre-loop snapshot for memory so the model sees only what
        # existed when its KV cache was reset. Open queries are detected
        # by ``ask_chunk < seg_start AND max(answer_chunks) >= seg_start``
        # (asked before the boundary, at least one expected answer chunk
        # is still ahead).
        if traj_type == TRAJ_TYPE_FROM_COMPRESS and turns and stream_samples:
            head_chunk = int(stream_samples[0].get("chunk_idx", 0))
            open_qs = [
                q for q in question_metadata
                if int(q["ask_chunk"]) < head_chunk
                and int(q["max_answer_chunk"]) >= head_chunk
            ]
            if open_qs:
                turns[0].user.inherited_queries = [
                    (q["ask_chunk"], q["spec"]) for q in open_qs
                ]
                inherited_r: List[Tuple[int, str]] = []
                for q in open_qs:
                    if not _query_uses_response_history(q):
                        continue
                    for ans_chunk in q["answer_chunks"]:
                        if (ans_chunk < head_chunk
                                and ans_chunk in responses_at_chunk):
                            inherited_r.append(
                                (ans_chunk, responses_at_chunk[ans_chunk])
                            )
                if inherited_r:
                    turns[0].user.inherited_responses = sorted(inherited_r)

        # Adaptive query refresh. The first ask chunk is already carried by
        # ``active_query``. Later chunks re-render the open question at expected
        # answer/probe chunks, periodically during long waits, and right after
        # cumulative count answers so response_history is near the next step.
        if query_policy != QUERY_POLICY_CURRENT_ASK_BOUNDARY:
            for turn in turns:
                chunk_idx = int(turn.user.chunk_idx)
                if turn.user.active_query is not None:
                    continue
                active_qs = [
                    q for q in question_metadata
                    if _query_active_at(q, chunk_idx)
                    and _should_force_query_at_chunk(q, chunk_idx, query_policy)
                ]
                if not active_qs:
                    continue
                # Pass3 enforces one active question. If a legacy row overlaps,
                # keep the newest open query, matching format_queries_block().
                q = sorted(active_qs, key=lambda item: int(item["ask_chunk"]))[-1]
                _append_inherited_query_once(
                    turn.user,
                    q,
                    _prior_responses_for_question(q, chunk_idx, responses_at_chunk),
                )

        if turns and stream_samples:
            chunk_start = int(stream_samples[0].get("chunk_idx", 0))
            chunk_end = int(stream_samples[-1].get("chunk_idx", 0))
            loss_class = traj_type
            memory_prefill = (
                inherited_mem_text
                if traj_type == TRAJ_TYPE_FROM_COMPRESS and inherited_mem_text
                else None
            )
            spec = TrajectorySpec(
                trajectory_type=traj_type,
                turns=turns,
                available_tools=(TOOL_NAME_RECALL,),
                memory_prefill_text=memory_prefill,
                trajectory_idx=len(rows),
                video_id=video_id,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            messages, tools = render_trajectory_messages(spec)
            loss_assistant_indices = None
            if memory_prefill:
                assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
                loss_assistant_indices = list(range(1, assistant_count))

            questions_here = [
                q
                for q in questions_raw
                if chunk_start <= int(q.get("ask_chunk", -1)) <= chunk_end
            ]

            rows.append({
                "trajectory_idx": len(rows),
                "trajectory_type": traj_type,
                "video_id": video_id,
                "parent_trajectory_id": parent_traj_id,
                "sample_type": "streaming_trajectory",
                "action": "streaming",
                "loss_class": loss_class,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "n_chunks": chunk_end - chunk_start + 1,
                "messages": messages,
                "tools": tools,
                **({"loss_assistant_indices": loss_assistant_indices} if loss_assistant_indices is not None else {}),
                "questions_in_segment": questions_here,
                "compress_event": None,
                "metadata": {
                    "loss_class": loss_class,
                    "trajectory_type": traj_type,
                    "memory_prefill": memory_prefill,
                    "memory_prefill_ack": MEMORY_LOAD_ACK if memory_prefill else "",
                    "memory_state_after_segment": [
                        {"t": e.time_str, "text": e.text} for e in memory_state
                    ],
                },
            })

        if terminal_compress is not None:
            mem_text, chunks = extract_compress_summary(terminal_compress)
            mem_text = _normalise_mem_block(mem_text)
            entries = parse_mem_block_to_entries(mem_text)
            tr = [
                min((e.start_sec for e in entries), default=0),
                max((e.end_sec for e in entries), default=0),
            ]
            compress_chunk = int(terminal_compress.get("chunk_idx", 0))
            update_input = _memory_update_input_from_sample(terminal_compress)
            compress_event = {
                "chunk_idx": compress_chunk,
                "summary_text": mem_text,
                "time_range": tr,
                "memory_update_mode": "compact_mem",
                "source_chunks": list(chunks),
                "entries": [
                    {"t": e.time_str, "text": e.text}
                    for e in entries
                ],
            }
            rows.append({
                "trajectory_idx": len(rows),
                "trajectory_type": TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
                "video_id": video_id,
                "parent_trajectory_id": parent_traj_id,
                "sample_type": "compress",
                "action": "compress",
                "loss_class": "compress",
                "chunk_start": compress_chunk,
                "chunk_end": compress_chunk,
                "n_chunks": 0,
                "messages": [
                    {"role": "system", "content": COMPACT_MEMORY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": update_input}],
                    },
                    {"role": "assistant", "content": mem_text},
                ],
                "tools": [],
                "questions_in_segment": [],
                "compress_event": compress_event,
                "metadata": {
                    "task_type": "compact_memory_update",
                    "memory_update_mode": "compact_mem",
                    "loss_class": "compress",
                    "memory_update_input": update_input,
                    "memory_state_after_segment": [
                        {"t": e.time_str, "text": e.text} for e in entries
                    ],
                },
            })
            if mem_text.strip():
                memory_prefill_text = mem_text
            if entries:
                memory_state = entries

    return rows
