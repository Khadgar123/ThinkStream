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
  with ``prefix_memory_text`` populated from the previous compress summary.
- Compress sample's ``inter_chunk=True`` drops the visual block from its
  user turn and injects a ``<stage:compress>`` marker (matches v12 SFT
  intent without requiring a re-run of pass2/3).
- Recall samples (multi-turn within one chunk) render as a single
  ``TurnSpec`` carrying the tool-2-turn pattern.
- This module is pure-function and stateless: no I/O, no global config.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from thinkstream.data.agent_protocol import is_inter_chunk as sample_is_inter_chunk
from thinkstream.data.schema import (
    ACTION_COMPRESS_SELECT,
    ACTION_COMPRESS_SUMMARY,
    ACTION_RECALL,
    ACTION_RESPONSE,
    ACTION_SILENT,
    AssistantSpec,
    ChunkUserSpec,
    MemoryEntry,
    QuerySpec,
    STAGE_COMPRESS_MARKER,
    TOOL_NAME_COMPRESS,
    TOOL_NAME_RECALL,
    TRAJ_TYPE_FROM_COMPRESS,
    TRAJ_TYPE_FROM_START,
    TrajectorySpec,
    TurnSpec,
    apply_compress_save,
    build_tool_response_message,
    render_trajectory_messages,
)

# Sliding-window size for the bulk-load at trajectory headers. Must match
# the runtime engine's video_flex_window_size.
SLIDING_WINDOW_CHUNKS = 16


# ---------------------------------------------------------------------------
# Pass3 output text parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_assistant_output_to_spec(text: str, chunk_idx: int) -> AssistantSpec:
    """Parse a pass3-format assistant output string into a v2 AssistantSpec.

    Pass3 emits ``<think>...</think>`` plus ONE of:
    - ``<answer>X</answer>``       — non-empty: response action with answer X
    - ``<answer></answer>``        — empty: silent action
    - ``<tool_call>{...}</tool_call>`` — compress or recall tool_call

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
            # Pass3-format compress carries (time_range, text) in one call.
            # Map to the canonical two-step form: this AssistantSpec
            # represents the Step-1 select tool_call (with time_range
            # only). The Step-2 summary <m> emit is produced separately
            # by ``build_compress_two_step_turns`` using the text in args.
            tr = args.get("time_range")
            return AssistantSpec(
                think=think,
                action_type=ACTION_COMPRESS_SELECT,
                tool_call_id=f"comp_{chunk_idx}",
                tool_arguments={
                    "time_range": tr,
                    # Stash the gold summary text on the spec so the
                    # caller can build the Step-2 emit; it is dropped at
                    # render time for the Step-1 tool_call payload.
                    "_gold_summary_text": (args.get("text") or args.get("summary") or "").strip(),
                },
            )
        if name == TOOL_NAME_RECALL:
            return AssistantSpec(
                think=think,
                action_type=ACTION_RECALL,
                tool_call_id=f"rec_{chunk_idx}",
                tool_arguments=args,
            )
        # Unknown tool name — fall through to silent so we don't crash; the
        # caller should log this case in their audit.
        return AssistantSpec(think=think, action_type=ACTION_SILENT)

    ans_match = _ANSWER_RE.search(text)
    ans_text = ans_match.group(1).strip() if ans_match else ""
    if ans_text:
        return AssistantSpec(
            think=think, action_type=ACTION_RESPONSE, response_text=ans_text,
        )
    return AssistantSpec(think=think, action_type=ACTION_SILENT)


# ---------------------------------------------------------------------------
# Question metadata → QuerySpec map
# ---------------------------------------------------------------------------

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
    segment sees a gold ``<response>X</response>`` token at some chunk
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
        options = q.get("options") or None
        instruction = (q.get("answer_instruction") or "").strip() or None
        out.append({
            "ask_chunk": int(ask),
            "answer_chunks": answer_chunks,
            "max_answer_chunk": max(answer_chunks),
            "spec": QuerySpec(
                text=text,
                options=list(options) if options else None,
                answer_format=instruction,
            ),
        })
    return out


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
        instruction = (q.get("answer_instruction") or "").strip() or None
        spec = QuerySpec(
            text=text,
            options=list(options) if options else None,
            answer_format=instruction,
        )
        out.setdefault(ask, []).append(spec)
    return out


# ---------------------------------------------------------------------------
# Recall multi-turn pack → TurnSpec
# ---------------------------------------------------------------------------

def _format_recall_result_for_tool_response(
    recall_result: Optional[Dict],
) -> str:
    """Format a pass3 recall_result dict into a tool_response content string.

    The original pass3 schema stuffs frame metadata + retrieved memory text
    into recall_result. For SFT we render it as a short JSON-flavoured text
    body so the tool turn always renders deterministically.
    """
    if not recall_result:
        return "(no recall hits)"
    keep_keys = ("time_range", "source", "n_frames", "text", "preview")
    body = {k: recall_result[k] for k in keep_keys if k in recall_result}
    if not body:
        return json.dumps(recall_result, ensure_ascii=False)[:400]
    return json.dumps(body, ensure_ascii=False)


def translate_recall_sample(
    sample: Dict,
    user: ChunkUserSpec,
) -> TurnSpec:
    """Recall sample → one TurnSpec with the tool-2-turn pattern."""
    chunk_idx = sample["chunk_idx"]
    turn1_text = sample.get("v12_assistant_turn_1", "") or ""
    turn2_text = sample.get("v12_assistant_turn_2", "") or ""

    a1 = parse_assistant_output_to_spec(turn1_text, chunk_idx)
    # Force the first assistant turn to be a tool_call; if the upstream
    # parsing somehow yielded a silent/response, repair it conservatively.
    if a1.action_type not in (ACTION_COMPRESS_SELECT, ACTION_RECALL):
        # Treat as silent fallback; tool_response/followup will be skipped.
        return TurnSpec(user=user, assistant=a1)

    tool_resp = build_tool_response_message(
        tool_call_id=a1.tool_call_id,
        content_text=_format_recall_result_for_tool_response(
            sample.get("recall_result"),
        ),
    )
    a2 = parse_assistant_output_to_spec(turn2_text, chunk_idx)
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
) -> TurnSpec:
    """Convert one pass3/pass4 sample into a v2 TurnSpec.

    ``questions_by_chunk`` maps chunk_idx → list of QuerySpecs (multiple
    questions may share the same ask_chunk after the P3.12 fix). The
    renderer picks the FIRST query for ``ChunkUserSpec.active_query`` —
    multi-question-per-chunk fanout is left to the caller / future work.
    """
    chunk_idx = int(sample.get("chunk_idx", 0))
    sample_type = sample.get("sample_type", ACTION_SILENT)
    is_inter_chunk = sample_is_inter_chunk(sample)

    if is_inter_chunk:
        # Compress trigger — no video, stage marker injected.
        user = ChunkUserSpec(
            chunk_idx=chunk_idx,
            frame_paths=[],
            stage_marker=STAGE_COMPRESS_MARKER,
            stage_text="Memory near budget. Pick a compressible past range.",
        )
    else:
        queries_here = questions_by_chunk.get(chunk_idx) or []
        active = queries_here[0] if queries_here else None
        user = ChunkUserSpec(
            chunk_idx=chunk_idx,
            frame_paths=list(frame_resolver(chunk_idx)),
            active_query=active,
        )

    if sample_type == "recall":
        return translate_recall_sample(sample, user)

    # silent / response / compress (single-turn output)
    output_text = sample.get("output", "") or ""
    assistant = parse_assistant_output_to_spec(output_text, chunk_idx)
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

    Each segment INCLUDES its terminating compress sample (so the segment's
    last assistant turn IS the compress tool_call). The next segment starts
    with the chunk immediately after that compress.

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


def extract_compress_summary(compress_sample: Dict) -> Tuple[str, List[int]]:
    """Pull the gold compression summary text and source chunks from a
    compress sample. Used to build the next trajectory's prefix_memory.

    Priority:
      1. ``gold_caption`` (pass3c stamps this from the pass2 oracle)
      2. parse the tool_call arguments in ``output`` for ``summary`` /
         ``text`` field
      3. empty string fallback
    """
    text = (compress_sample.get("gold_caption") or "").strip()
    chunks = list(compress_sample.get("gold_compress_chunks") or [])
    if text:
        return text, chunks

    output = compress_sample.get("output", "") or ""
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
    return text, chunks


# ---------------------------------------------------------------------------
# Top-level entry: render one pass4 trajectory record → list of output rows
# ---------------------------------------------------------------------------

def _build_compress_step2_summary_turn(
    last_compress_spec: AssistantSpec,
    raw_memory_at_event: List[MemoryEntry],
) -> TurnSpec:
    """Build the Step-2 compress-summary assistant turn.

    The Step-1 spec carries the gold summary text in
    ``_gold_summary_text``. This helper:
      1. Renders a synthetic ``role: tool`` response containing the raw
         <m> entries in the selected range (mirrors what the runtime tool
         would return).
      2. Emits the Step-2 assistant content with the gold ``<m>`` block.

    Returns a TurnSpec with ``tool_response`` filled and
    ``followup_assistant`` carrying the Step-2 ACTION_COMPRESS_SUMMARY.
    """
    tool_args = last_compress_spec.tool_arguments or {}
    time_range = tool_args.get("time_range") or [0, 0]
    summary_text = tool_args.get("_gold_summary_text", "")

    # Tool-response body: raw <m> entries inside the selected range.
    start, end = int(time_range[0]), int(time_range[1])
    raw_entries_in_range = [
        e for e in raw_memory_at_event
        if start <= e.end_sec and e.start_sec <= end
    ]
    if raw_entries_in_range:
        raw_text = "\n".join(e.to_text() for e in raw_entries_in_range)
    else:
        raw_text = '  <m t="">(no entries in selected range)</m>'

    tool_response = build_tool_response_message(
        tool_call_id=last_compress_spec.tool_call_id,
        content_text=f"<memory>\n{raw_text}\n</memory>",
    )

    followup_assistant = AssistantSpec(
        think="Reviewed raw entries; writing the merged summary.",
        action_type=ACTION_COMPRESS_SUMMARY,
        tool_arguments={
            "time_range": [start, end],
            "text": summary_text,
        },
    )
    # The TurnSpec uses .user as the COMPRESS-trigger user already built
    # by the caller; we just attach tool_response + followup_assistant.
    return TurnSpec(
        user=None,  # filled by caller
        assistant=last_compress_spec,
        tool_response=tool_response,
        followup_assistant=followup_assistant,
    )


def _build_window_prior_chunks(
    history_samples: List[Dict],
    current_chunk_idx: int,
    frame_resolver: FrameResolver,
    window_size: int = SLIDING_WINDOW_CHUNKS,
) -> List[Tuple[int, List[str]]]:
    """Build the bulk-load list for a from_compress trajectory header.

    Selects up to ``window_size - 1`` most-recent chunks BEFORE
    ``current_chunk_idx`` from ``history_samples`` (samples already
    consumed by prior segments). Returns ``[(chunk_idx, frame_paths), ...]``
    in ascending chunk order.

    Compress-trigger chunks (no video) are skipped. Chunks without
    resolvable frames are skipped.
    """
    candidates = sorted(
        [s for s in history_samples if not is_compress_sample(s)],
        key=lambda s: int(s.get("chunk_idx", 0)),
    )
    target_count = max(0, window_size - 1)
    # Keep only the most-recent ``target_count`` chunks BEFORE current.
    eligible = [
        s for s in candidates if int(s.get("chunk_idx", -1)) < current_chunk_idx
    ]
    selected = eligible[-target_count:] if target_count else []
    out: List[Tuple[int, List[str]]] = []
    for s in selected:
        c = int(s.get("chunk_idx", -1))
        frames = list(frame_resolver(c) or [])
        if frames:
            out.append((c, frames))
    return out


def render_trajectory_record_to_rows(
    traj_record: Dict,
    frame_resolver: FrameResolver,
) -> List[Dict]:
    """Render one pass4 trajectory_record into a list of v2 trajectory rows.

    Each compress event terminates a sub-trajectory. The boundaries are:
      - sub-trajectory 0: from_start, chunks 0..K1 (K1 = first compress)
        Final turn(s): user(<stage:compress>) → asst(compress_select) →
                       tool(raw entries) → asst(<m>summary</m>).
      - sub-trajectory 1: from_compress, chunks K1+1..K2.
        First turn: user with inherited <memory> + 15 chunks bulk-load +
                    current chunk K1+1's video.
      - sub-trajectory 2..N: same as #1.

    Memory state evolves via ``apply_compress_save`` at each boundary.
    """
    raw_samples = traj_record.get("samples") or []
    if not raw_samples:
        return []
    samples = sorted(raw_samples, key=lambda s: int(s.get("chunk_idx", 0)))
    segments = split_samples_by_compress(samples)
    questions_raw = traj_record.get("questions") or []
    questions_by_chunk = build_questions_by_chunk(questions_raw)
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
        m = _ANSWER_RE.search(output_text)
        if m:
            text = m.group(1).strip()
            if text:
                responses_at_chunk[c] = text

    video_id = traj_record.get("video_id", "")
    parent_traj_id = traj_record.get("trajectory_id", "")

    # Accumulated memory state across all sub-trajectories of this video.
    # Each segment's last compress event ``apply_compress_save``s its
    # summary into this list; the next segment's first user turn reads it.
    memory_state: List[MemoryEntry] = []
    history_samples: List[Dict] = []  # samples consumed by prior segments

    rows: List[Dict] = []
    for seg_idx, seg_samples in enumerate(segments):
        traj_type = (
            TRAJ_TYPE_FROM_START if seg_idx == 0 else TRAJ_TYPE_FROM_COMPRESS
        )

        # Snapshot memory state BEFORE consuming this segment — this is
        # what the model inherits at the trajectory boundary. Mutating
        # memory_state during the loop below changes it to its
        # END-of-segment value, which becomes the next segment's inheritance.
        inherited_snapshot = list(memory_state)

        # Build per-sample TurnSpecs. Compress samples become Step-1 specs
        # (carrying _gold_summary_text); we expand them into the full
        # 2-step pattern after iteration.
        # Each non-compress sample also writes a raw `<m t="N">` entry into
        # memory_state: the model's "observation log" mirrors what runtime
        # would have written into the streaming memory store. Any future
        # compress event replaces a range of these raw entries with one
        # summary <m t="X-Y"> via apply_compress_save.
        turns: List[TurnSpec] = []
        for s in seg_samples:
            turn = translate_sample_to_turn(
                s, questions_by_chunk, frame_resolver,
            )
            turns.append(turn)
            if not is_compress_sample(s):
                think_text = (turn.assistant.think or "").strip()
                if think_text:
                    memory_state.append(MemoryEntry(
                        time_str=str(int(s.get("chunk_idx", 0))),
                        text=think_text,
                    ))

        # Attach inherited memory + window bulk-load + open queries /
        # prior responses to the FIRST turn of a from_compress segment.
        # Uses the pre-loop snapshot for memory so the model sees only what
        # existed when its KV cache was reset. Open queries are detected
        # by ``ask_chunk < seg_start AND max(answer_chunks) >= seg_start``
        # (asked before the boundary, at least one expected answer chunk
        # is still ahead).
        if traj_type == TRAJ_TYPE_FROM_COMPRESS and turns:
            head_chunk = int(seg_samples[0].get("chunk_idx", 0))
            if inherited_snapshot:
                turns[0].user.inherited_memory = list(inherited_snapshot)
            turns[0].user.window_prior_chunks = _build_window_prior_chunks(
                history_samples, head_chunk, frame_resolver,
            )
            open_qs = [
                q for q in question_metadata
                if q["ask_chunk"] < head_chunk
                and q["max_answer_chunk"] >= head_chunk
            ]
            if open_qs:
                turns[0].user.inherited_queries = [
                    (q["ask_chunk"], q["spec"]) for q in open_qs
                ]
                inherited_r: List[Tuple[int, str]] = []
                for q in open_qs:
                    for ans_chunk in q["answer_chunks"]:
                        if (ans_chunk < head_chunk
                                and ans_chunk in responses_at_chunk):
                            inherited_r.append(
                                (ans_chunk, responses_at_chunk[ans_chunk])
                            )
                if inherited_r:
                    turns[0].user.inherited_responses = sorted(inherited_r)

        # Expand the terminal compress turn (if any) into the 2-step
        # pattern: replace last TurnSpec with one that has tool_response
        # + followup_assistant emitting the <m> summary.
        last_sample = seg_samples[-1]
        compress_event: Optional[Dict] = None
        if is_compress_sample(last_sample) and turns:
            last_turn = turns[-1]
            select_spec = last_turn.assistant
            # Sanity: should be a compress_select spec produced by the
            # updated parse_assistant_output_to_spec.
            if select_spec.action_type == ACTION_COMPRESS_SELECT:
                step2_turn = _build_compress_step2_summary_turn(
                    select_spec, memory_state,
                )
                # Attach tool_response + followup to the same TurnSpec
                # (sharing the .user from the compress-trigger turn).
                turns[-1] = TurnSpec(
                    user=last_turn.user,
                    assistant=select_spec,
                    tool_response=step2_turn.tool_response,
                    followup_assistant=step2_turn.followup_assistant,
                )
                tool_args = select_spec.tool_arguments or {}
                tr = tool_args.get("time_range") or [0, 0]
                summary_text = tool_args.get("_gold_summary_text", "")
                compress_event = {
                    "chunk_idx": int(last_sample.get("chunk_idx", 0)),
                    "summary_text": summary_text,
                    "time_range": [int(tr[0]), int(tr[1])],
                }
                # Update accumulated memory for the NEXT segment.
                if summary_text:
                    memory_state = apply_compress_save(
                        memory_state, (int(tr[0]), int(tr[1])), summary_text,
                    )

        chunk_start = int(seg_samples[0].get("chunk_idx", 0))
        chunk_end = int(seg_samples[-1].get("chunk_idx", 0))

        spec = TrajectorySpec(
            trajectory_type=traj_type,
            turns=turns,
            trajectory_idx=seg_idx,
            video_id=video_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        messages, tools = render_trajectory_messages(spec)

        # Questions whose ask_chunk falls inside this segment.
        questions_here = [
            q for q in (traj_record.get("questions") or [])
            if chunk_start <= int(q.get("ask_chunk", -1)) <= chunk_end
        ]

        rows.append({
            "trajectory_idx": seg_idx,
            "trajectory_type": traj_type,
            "video_id": video_id,
            "parent_trajectory_id": parent_traj_id,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "n_chunks": chunk_end - chunk_start + 1,
            "messages": messages,
            "tools": tools,
            "questions_in_segment": questions_here,
            "compress_event": compress_event,
            "metadata": {
                "memory_state_after_segment": [
                    {"t": e.time_str, "text": e.text} for e in memory_state
                ],
            },
        })

        # Roll this segment's samples into history for the next segment's
        # window bulk-load.
        history_samples.extend(seg_samples)

    return rows
