"""ThinkStream message schema — multi-turn dialogue + Qwen tool_call.

Pure-function building blocks for rendering one ``TrajectorySegment`` as a
Qwen-native ``messages + tools`` payload. Used by pass5 and the SFT data
loader. Independent of pass1/2/3 schema; consumes already-decided per-chunk
samples and emits final on-disk JSONL rows.

Key conventions:
- One trajectory = one output JSONL row
- Trajectory boundary = a compress event in the rollout
- Two trajectory types:
    - ``from_start``           — chunk 0 is the first sample, no prefix memory
    - ``from_compress``        — prefix memory carried from previous trajectory
- User content uses **single-point** timestamps ``<t=N>`` (not a range),
  followed by a video block. Queries (with options + answer format) only
  injected at the chunks where they actually arrive.
  - Tools are passed as a Qwen-compatible ``tools=[...]`` parameter (template
  auto-renders them into the system block). New streaming turns expose only the
  recall tool. Compact-memory updates are separate text-only system turns.

Reference: Qwen-Agent ``qwen_agent/llm/schema.py`` (message dataclass pattern)
and Qwen3-VL official chat template (``tools=`` parameter rendering).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Single-point timestamp format, kept short to save tokens.
TIMESTAMP_FORMAT = "<t={chunk_idx}>"

# Stage markers (text items inside user content). Compact memory no longer uses
# a streaming stage marker; STAGE_COMPRESS_MARKER is legacy-only.
STAGE_COMPRESS_MARKER = "<stage:compress>"
STAGE_FORCE_ANSWER_MARKER = "<stage:force_answer>"

# Legacy wrapper constants kept for old loaders/tests only. New prompts render
# compact memory as bare <m t="...">...</m> lines.
MEMORY_OPEN = "<memory>"
MEMORY_CLOSE = "</memory>"

# Trajectory types (string enum to keep JSON-serializable).
TRAJ_TYPE_FROM_START = "from_start"
TRAJ_TYPE_FROM_COMPRESS = "from_compress"
TRAJ_TYPE_COMPACT_MEMORY_UPDATE = "compact_memory_update"
MEMORY_LOAD_ACK = "Memory loaded."

# Per-chunk video resolution defaults (Qwen3-VL smart_resize bounds).
# Streaming-runtime profile selected for the 8B local HF + KV-window path:
#   min = 256·28·28 = 200,704
#   max = 512·28·28 = 401,408
# These constants are the SINGLE source of truth — mirrored by:
#   - sft.args.video_{min,max}_pixels (processor.video_processor config)
#   - scripts.agent_data.config.RUNTIME_MM_PROCESSOR_KWARGS
# Schema-rendered video_meta messages also write min/max into each video item
# because qwen-vl-utils consumes inline video-item bounds before tensorization.
DEFAULT_VIDEO_MIN_PIXELS = 256 * 28 * 28   # 200,704
DEFAULT_VIDEO_MAX_PIXELS = 512 * 28 * 28   # 401,408

# Streaming chunking config (must match the data pipeline + runtime):
#   AGENT_CHUNK_SEC × FRAMES_PER_CHUNK = frame interval in the original video.
# Used to derive each chunk's ``frames_indices`` in the source video so
# Qwen3-VL's temporal RoPE anchors at the correct second.
AGENT_CHUNK_SEC = 1.0
FRAMES_PER_CHUNK = 2
VIDEO_FPS = FRAMES_PER_CHUNK / AGENT_CHUNK_SEC   # 2.0


def apply_compress_save(
    memory_entries: "List[MemoryEntry]",
    time_range: Tuple[int, int],
    text: str,
) -> "List[MemoryEntry]":
    """Apply a compress_save event to a memory list.

    Replaces every ``<m>`` entry whose time range INTERSECTS ``time_range``
    with a single new entry covering the union of removed ranges. The new
    entry's ``time_str`` is ``"min_start-max_end"`` (snapped to the actual
    boundaries of the entries that got removed — so a model-emitted range
    that overshoots actual memory gets clamped automatically).

    Pure function; does not mutate the input list.
    """
    start, end = int(time_range[0]), int(time_range[1])
    if start > end:
        start, end = end, start

    kept: List["MemoryEntry"] = []
    removed_starts: List[int] = []
    removed_ends: List[int] = []
    for entry in memory_entries:
        e_start = entry.start_sec
        e_end = entry.end_sec
        # Two ranges intersect iff start <= e_end AND end >= e_start.
        intersects = (start <= e_end) and (end >= e_start)
        if intersects:
            removed_starts.append(e_start)
            removed_ends.append(e_end)
        else:
            kept.append(entry)

    if not removed_starts:
        # Nothing intersected; just append the new entry at its declared range.
        new_entry = MemoryEntry(time_str=f"{start}-{end}", text=text.strip())
    else:
        clamped_start = min(min(removed_starts), start)
        clamped_end = max(max(removed_ends), end)
        new_entry = MemoryEntry(
            time_str=f"{clamped_start}-{clamped_end}", text=text.strip()
        )
    kept.append(new_entry)
    kept.sort(key=lambda e: e.start_sec)
    return kept


def infer_video_metadata(
    frame_paths: List[str],
    chunk_idx: int,
    *,
    fps: Optional[float] = None,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> Dict[str, Any]:
    """Build the Qwen3-VL ``video_metadata`` block for a chunk's frames.

    For chunk ``N`` with ``frames_per_chunk`` frames sampled at ``fps`` from
    the source video, the frames' positions in the source are
    ``[N * frames_per_chunk, N * frames_per_chunk + 1, ...]`` (zero-based).
    Qwen3-VL's processor consumes ``fps`` + ``frames_indices`` to anchor
    text-layer timestamp tokens at the correct second; ``do_sample_frames``
    must be ``False`` so the processor does not re-sample our pre-extracted
    frames at its own internal cadence.
    """
    eff_fps = float(fps if fps is not None else VIDEO_FPS)
    base = int(chunk_idx) * int(frames_per_chunk)
    indices = [base + i for i in range(len(frame_paths))]
    return {
        "fps": eff_fps,
        "frames_indices": indices,
        "total_num_frames": (indices[-1] + 1) if indices else 0,
        "do_sample_frames": False,
    }

# Action types — must stay aligned with pass3c sample_type enum.
ACTION_SILENT = "silent"
ACTION_RESPONSE = "response"
ACTION_RECALL = "recall"
# Legacy one-step compression action. New compact-memory data uses
# ACTION_MEMORY_UPDATE and emits bare <m t="...">...</m> lines directly,
# without a compress tool call or a wrapper tag.
ACTION_COMPRESS_SELECT = "compress_select"
ACTION_MEMORY_UPDATE = "memory_update"         # Single-step compact-memory replacement
# Legacy alias for old code paths that still say "compress".
ACTION_COMPRESS = ACTION_COMPRESS_SELECT

# Tool name strings. New active turns expose recall only; TOOL_NAME_COMPRESS is
# retained only for legacy parser/backfill compatibility.
TOOL_NAME_COMPRESS = "compress"
TOOL_NAME_RECALL = "recall"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a streaming video assistant. Observe the source video chunk-by-chunk; ONE action per turn.

Format: open with <think>...</think>, then end the turn with exactly one action. No text outside <think> and the chosen action.

Common terminal actions:
- </Silence>: use when there is no active query, or the query is not ready to answer yet.
- </Response> answer: use when the active query is answerable now. Match the requested answer format exactly.

Exceptional tool action:
- recall tool_call: use only when the active query needs older visual evidence that is strictly earlier than the current chunk timestamp, and that evidence is not already available from current frames, compact memory, response_history, or a previous recall result.

Anchors in user content:
- <t=N>: current integer second; this turn's frames cover [N, N+1).
- <m t="...">...</m>: historical compact memory loaded before the next current chunk. It is history only, not current visual.
- <active_query>...</active_query>: the currently open question. Lines use [Ns] timestamps showing when the question opened.
- <response_history>...</response_history>: prior valid answers for cumulative open questions, especially running counts. It can be empty for independent current-status probes.

Startup modes:
- If the conversation starts with <t=N>, there is no compact history yet.
- If compact memory is loaded before streaming resumes, it appears as a separate user turn containing only <m t="...">...</m> lines, followed by assistant text "Memory loaded."; the next user turn starts with <t=N> and the current frames.

Decision priority:
- If there is no active query, output </Silence>.
- If the active query is answerable from current frames, loaded compact memory, response_history, or recalled frames from a previous tool result, output </Response> followed by the final answer.
- For a future/proactive trigger, output </Silence> until the requested cue or event completion is visible. Do not give partial answers or guesses before the trigger is observed.
- For running counts, use response_history to continue the count at the next required update. For independent status probes, answer from the current timestep's evidence without treating prior answers as the current state.
- If the answer choices include an unknown/abstain option, treat it like an ordinary answer choice: choose it only when the evidence available at the required answer time supports that choice, not as a waiting action.
- Use recall only if older missing visual evidence is required. Recall is not a waiting action. Do not call recall for current-frame questions, future/proactive triggers, already-answerable questions, or ordinary uncertainty.
- Match the active query's answer format exactly. Multiple-choice formats normally require the option letter plus its text; binary/number formats get only that value; short/descriptive formats get one concise phrase or sentence, without extra explanation unless requested.

Recall arguments: use absolute seconds from the source video with start_time and end_time. The interval is closed [start_time, end_time]. end_time may equal start_time for a single-second recall and must be strictly earlier than the current chunk timestamp. Recall cannot fetch future evidence.

After a recall tool result: use the recalled frames with the current context, then output </Response> if sufficient or </Silence> if the query is still not settled. Do not call recall again unless a different older interval is truly required.
"""


COMPACT_MEMORY_SYSTEM_PROMPT = """You are doing a strict compact-memory update. This is not a captioning task and not a question-answering task.

Return only compact-memory XML lines and nothing else:
<m t="start-end">one concise English summary for that exact source range.</m>

Use only OLD_MEMORY and NEW_CAPTIONS timestamps and facts.
Keep source ranges honest. If the source provides one contiguous summary range, keep it as one <m t="start-end">...</m> line; do not invent finer timestamp segments.
If OLD_MEMORY has any <m> lines, preserve useful old memory when it is still relevant.
If NEW_CAPTIONS has any <c> lines, cover the latest new captions.
Preserve important objects, actions, OCR/text, names, numbers, counts, colors, and state changes. Merge repeats.
No JSON, Markdown, bullets, prose, analysis, answers, or tool calls."""


# ---------------------------------------------------------------------------
# Tools schema (Qwen tools= parameter format)
# ---------------------------------------------------------------------------

def build_tools_schema(include_recall: bool = True, include_compress: bool = True) -> List[Dict]:
    """Return the OpenAI-compatible tools list. Pass directly to
    ``tokenizer.apply_chat_template(messages, tools=...)``.
    """
    tools: List[Dict] = []
    if include_recall:
        tools.append({
            "type": "function",
            "function": {
                "name": TOOL_NAME_RECALL,
                "description": (
                    "Exceptional action only: retrieve older visual frames "
                    "between start_time and end_time when the active query "
                    "cannot be answered from current frames, compact memory, "
                    "response_history, or a previous recall result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "number",
                            "description": (
                                "Absolute source-video start second. Must be "
                                "non-negative and <= end_time."
                            ),
                        },
                        "end_time": {
                            "type": "number",
                            "description": (
                                "Absolute source-video end second. The closed "
                                "interval [start_time, end_time] must be "
                                "strictly earlier than the current chunk."
                            ),
                        },
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        })
    # include_compress is accepted for backward-compatible call signatures.
    # Compact-memory update is not a callable tool in the active protocol.
    return tools


# ---------------------------------------------------------------------------
# User content building blocks
# ---------------------------------------------------------------------------

@dataclass
class QuerySpec:
    """A user query arriving at a specific chunk. Renders to one text item."""
    text: str                          # the question itself
    options: Optional[List[str]] = None  # ["A. ...", "B. ..."]
    answer_format: Optional[str] = None  # legacy explicit fallback text
    answer_form: Optional[str] = None
    answer_style: Optional[str] = None
    response_history_policy: Optional[str] = None  # include | omit

    def to_text(self) -> str:
        parts = [self.text.strip()]
        if self.options:
            parts.append("\n".join(opt.strip() for opt in self.options))
        answer_format = _query_answer_format_text(self, list(self.options or []))
        if answer_format:
            parts.append(answer_format)
        return "\n".join(parts).strip()


def _query_answer_format_text(query: QuerySpec, options: Optional[List[str]] = None) -> str:
    """Return the canonical model-visible answer-format line for QuerySpec."""
    options = list(options or getattr(query, "options", None) or [])
    explicit = str(getattr(query, "answer_format", "") or "").strip()
    answer_form = str(getattr(query, "answer_form", "") or "").strip()
    answer_style = str(getattr(query, "answer_style", "") or "").strip()
    if explicit and not answer_form and not options:
        return explicit
    inferred_form = answer_form or ("multiple_choice" if options else "")
    if inferred_form:
        try:
            from thinkstream.data.agent_protocol import canonical_answer_instruction

            instruction = canonical_answer_instruction({
                "answer_form": inferred_form,
                "answer_style": answer_style,
                "answer_instruction": explicit,
                "options": options,
            })
            if instruction:
                return instruction
        except ImportError:
            pass
    return explicit


@dataclass
class MemoryEntry:
    """One compact-memory entry rendered as ``<m t="...">...</m>``.

    ``time_str`` is either a single integer second (``"33"``) for a raw
    chunk think, or a range (``"0-32"``) for a compressed summary. The
    distinction is implicit in the time syntax — no ``level`` attribute
    is exposed to the model.
    """
    time_str: str
    text: str

    @property
    def start_sec(self) -> int:
        """First-second for sorting (handles both ``N`` and ``X-Y``)."""
        return int(self.time_str.split("-")[0])

    @property
    def end_sec(self) -> int:
        """End-second for overlap math."""
        parts = self.time_str.split("-")
        return int(parts[-1])

    def to_text(self) -> str:
        return f'  <m t="{self.time_str}">{self.text}</m>'


@dataclass
class ChunkUserSpec:
    """All the data needed to render one user turn at a single chunk.

    Trajectory-first-turn fields (only set on the FIRST chunk of a
    trajectory; carried over from the prior trajectory's terminal state):

    - ``inherited_memory``: time-sorted ``MemoryEntry`` list. Mix of
      compressed summaries (``<m t="X-Y">``) and raw thinks
      (``<m t="N">``). ``build_user_content`` can render it directly, but
      full ``from_compress`` trajectories move it into a separate prefill turn:
      user(<m> lines) -> assistant("Memory loaded.") -> user(<t=N> + frames).
    - ``inherited_queries``: ``[(ask_chunk, QuerySpec), ...]`` — open
      queries inherited from prior trajectories. Resolved queries don't
      cross trajectory boundaries.
    - ``inherited_responses``: ``[(response_chunk, response_text), ...]``
      — prior responses to those still-open queries.

    Always-fields:
    - ``chunk_idx``: integer second anchor for ``<t=N>``.
    - ``frame_paths``: 2 frames of THIS chunk only (sliding window is a
      KV-side mechanism, not a message-side one — never replay 16 chunks
      in user content).
    - ``active_query``: a NEW query arriving at this chunk. Rendered as
      a ``<active_query>`` block (or merged with inherited_queries on the
      first turn).
    - ``stage_marker``: optional legacy stage marker.

    Video resolution is carried on the video item as Qwen3VL metadata:
    each current-chunk video block gets the canonical min/max pixels.
    """
    chunk_idx: int
    frame_paths: List[str] = field(default_factory=list)
    active_query: Optional[QuerySpec] = None
    stage_marker: Optional[str] = None            # e.g. STAGE_COMPRESS_MARKER
    stage_text: Optional[str] = None              # context text for the stage
    # First-turn-only state-inheritance fields (None / empty on subsequent turns):
    inherited_memory: Optional[List[MemoryEntry]] = None
    inherited_queries: Optional[List[Tuple[int, "QuerySpec"]]] = None
    inherited_responses: Optional[List[Tuple[int, str]]] = None


def _format_query_lines(chunk_idx: int, query: QuerySpec) -> List[str]:
    """Render one active query with the same compact timestamp style used online."""
    prefix = f"[{int(chunk_idx)}s]"
    if hasattr(query, "text"):
        text = str(query.text or "").strip()
        options = list(query.options or [])
        answer_format = _query_answer_format_text(query, options)
    else:
        text = str(query or "").strip()
        options = []
        answer_format = ""
    lines: List[str] = []
    if text:
        lines.append(f"{prefix} Q: {text}")
    cleaned_options = [str(opt).strip() for opt in options if str(opt).strip()]
    if cleaned_options:
        lines.append(f"{prefix} Options: " + " ".join(cleaned_options))
    if answer_format:
        lines.append(f"{prefix} {answer_format}")
    return lines


def _response_history_enabled(spec: ChunkUserSpec) -> bool:
    """Whether the currently rendered query should show prior answers."""
    active = spec.active_query
    if active is None and spec.inherited_queries:
        # Match the visible active-query ordering: inherited queries are sorted
        # by ask time, and the newest open query is the effective target when
        # legacy data overlaps.
        active = sorted(spec.inherited_queries, key=lambda item: int(item[0]))[-1][1]
    policy = str(getattr(active, "response_history_policy", "") or "").strip().lower()
    return policy not in {"omit", "hide", "none", "no_history", "independent_probe"}


def build_user_content(spec: ChunkUserSpec) -> List[Dict]:
    """Render a single user turn's ``content`` list.

    Block order (fixed):
      1. bare ``<m t="...">`` memory lines — inherited summaries + raw thinks
      2. ``<t=chunk_idx>`` + the current 1s video chunk (2 frames).
         A legacy stage marker inserts short text right before the timestamp.
      3. ``<active_query>`` — inherited open queries + new query arriving this chunk
      4. ``<response_history>`` — prior responses for inherited open queries

    Rationale for the text→video→query→response order: the model sees the
    historical text state first (memory), then the visual evidence in time
    order, THEN the question to answer. Mirrors human "look, then think,
    then answer" flow and keeps the answer-format instructions adjacent to
    the freshest perception.
    """
    content: List[Dict] = []

    # (1) Bare <m> memory lines — only the first turn of a trajectory carries this.
    if spec.inherited_memory:
        # Time-sorted; summaries (<m t="X-Y">) and raw thinks (<m t="N">)
        # mixed in chronological order.
        entries_sorted = sorted(spec.inherited_memory, key=lambda e: e.start_sec)
        lines = [e.to_text() for e in entries_sorted]
        content.append({"type": "text", "text": "\n".join(lines)})

    # (2a) Stage marker (compress / force_answer / ...) — sits right before
    # the current chunk's timestamp since it qualifies the current turn.
    if spec.stage_marker:
        marker_text = spec.stage_marker
        if spec.stage_text:
            marker_text = f"{spec.stage_marker}\n{spec.stage_text.strip()}"
        content.append({"type": "text", "text": marker_text + "\n"})

    # (2b) Current chunk's <t=N> + video.
    content.append({
        "type": "text",
        "text": TIMESTAMP_FORMAT.format(chunk_idx=spec.chunk_idx),
    })
    if spec.stage_marker != STAGE_COMPRESS_MARKER and spec.frame_paths:
        frame_list = list(spec.frame_paths)[-FRAMES_PER_CHUNK:]
        content.append({
            "type": "video",
            "video": frame_list,
            "video_metadata": infer_video_metadata(frame_list, spec.chunk_idx),
            "min_pixels": DEFAULT_VIDEO_MIN_PIXELS,
            "max_pixels": DEFAULT_VIDEO_MAX_PIXELS,
            "kv_scope": "ordinary",
        })

    # (3) <active_query> — inherited open queries + new query at this chunk,
    # in ask-chunk ascending order. Comes AFTER the visual evidence so the
    # model sees the chunk first, then is asked.
    query_lines: List[str] = []
    for ask_chunk, q in (spec.inherited_queries or []):
        query_lines.extend(_format_query_lines(int(ask_chunk), q))
    if spec.active_query is not None:
        query_lines.extend(_format_query_lines(int(spec.chunk_idx), spec.active_query))
    if query_lines:
        content.append({
            "type": "text",
            "text": (
                "<active_query>\n"
                + "\n".join(query_lines)
                + "\n</active_query>"
            ),
        })

    # (4) <response_history> — prior responses for inherited open queries.
    # Render an empty block whenever a query is active so the absence of prior
    # answers is explicit to the small policy model.
    resp_lines: List[str] = []
    if _response_history_enabled(spec):
        resp_lines = [
            f"[{int(rt_chunk)}s] A: {str(rt_text).strip()}"
            for rt_chunk, rt_text in sorted(spec.inherited_responses or [])
            if str(rt_text).strip()
        ]
    if query_lines:
        content.append({
            "type": "text",
            "text": (
                "<response_history>\n"
                + "\n".join(resp_lines)
                + "\n</response_history>"
            ),
        })

    return content


# ---------------------------------------------------------------------------
# Assistant content building blocks
# ---------------------------------------------------------------------------

@dataclass
class AssistantSpec:
    """One assistant turn's output."""
    think: str                         # always present (may be short)
    action_type: str                   # ACTION_SILENT / ACTION_RESPONSE / ACTION_MEMORY_UPDATE / ACTION_RECALL
    response_text: Optional[str] = None    # only when action_type == ACTION_RESPONSE
    tool_call_id: Optional[str] = None     # only for tool actions
    tool_arguments: Optional[Dict[str, Any]] = None  # only for tool actions


def build_assistant_message(spec: AssistantSpec) -> Dict:
    """Render one assistant turn as an OpenAI-format message dict.

    For silent / response: returns ``{"role": "assistant", "content": "..."}``
    with the action text embedded (e.g. ``<think>...</think></Silence>``).

    For tool actions: returns
        {"role": "assistant",
         "content": "<think>...</think>",
         "tool_calls": [{"id": ..., "type": "function",
                         "function": {"name": ..., "arguments": <dict>}}]}

    On the ``id`` field — this is OpenAI-tool-API bookkeeping carried into
    the message dict so a follow-up ``role: tool`` message can reference
    ``tool_call_id`` to match a specific call. Qwen's chat template renders
    only ``<tool_call>\\n{"name": ..., "arguments": ...}\\n</tool_call>`` —
    the id never reaches the token stream. We keep it because:
      (a) ``role: tool`` messages need it for unambiguous pairing;
      (b) the dict round-trips through standard OpenAI clients unchanged.

    ``arguments`` is a Python ``dict`` (NOT a JSON-encoded string). Qwen3-VL's
    chat template applies ``| tojson`` to this field — passing an
    already-stringified value would cause double encoding. Some OpenAI
    clients expect arguments-as-string; serialise at the wire boundary if
    you need that form, never at message-build time.
    """
    think_block = f"<think>{spec.think.strip()}</think>"

    if spec.action_type == ACTION_SILENT:
        return {
            "role": "assistant",
            "content": f"{think_block}</Silence>",
        }

    if spec.action_type == ACTION_RESPONSE:
        if spec.response_text is None:
            raise ValueError("response action requires response_text")
        return {
            "role": "assistant",
            "content": f"{think_block}</Response> {spec.response_text.strip()}",
        }

    if spec.action_type == ACTION_MEMORY_UPDATE:
        if spec.tool_arguments is None:
            raise ValueError("memory_update requires tool_arguments {memory_text}")
        mem_text = (spec.tool_arguments.get("memory_text") or "").strip()
        if "<m" not in mem_text or "</m>" not in mem_text:
            raise ValueError("memory_update memory_text must contain <m> lines")
        return {
            "role": "assistant",
            "content": mem_text,
        }

    _TOOL_ACTIONS = {
        ACTION_RECALL: TOOL_NAME_RECALL,
        ACTION_COMPRESS_SELECT: TOOL_NAME_COMPRESS,
        ACTION_COMPRESS: TOOL_NAME_COMPRESS,  # legacy alias
    }
    if spec.action_type in _TOOL_ACTIONS:
        if spec.tool_arguments is None:
            raise ValueError(f"{spec.action_type} requires tool_arguments")
        if spec.tool_call_id is None:
            raise ValueError(f"{spec.action_type} requires tool_call_id")
        tool_name = _TOOL_ACTIONS[spec.action_type]
        # Drop any fields starting with "_" (internal-use only — e.g. the
        # splitter stashes ``_gold_summary_text`` on compress_select specs
        # so the Step-2 emit builder can pick it up later). Public tool
        # arguments must be JSON-clean.
        public_args = {
            k: v for k, v in spec.tool_arguments.items() if not str(k).startswith("_")
        }
        if spec.action_type == ACTION_RECALL:
            allowed = {"start_time", "end_time"}
            extra = sorted(str(k) for k in public_args.keys() if k not in allowed)
            if extra:
                raise ValueError(
                    f"recall arguments only support start_time and end_time; got extra keys: {extra}"
                )
            if public_args.get("start_time") is None or public_args.get("end_time") is None:
                raise ValueError("recall arguments require start_time and end_time")
            if not all(isinstance(public_args.get(k), (int, float)) for k in ("start_time", "end_time")):
                raise ValueError("recall start_time and end_time must be numbers")
            if float(public_args["end_time"]) < float(public_args["start_time"]):
                raise ValueError("recall end_time must be greater than or equal to start_time")
        # NOTE: pass the dict directly. Qwen3-VL's chat template renders
        # tool_calls[*].function.arguments via ``{{- tool_call.arguments |
        # tojson }}`` — it expects a Python dict and will JSON-encode it
        # exactly once. If we json.dumps here, the template's ``| tojson``
        # produces a doubly-encoded escaped JSON string ("\\"key\\": ..."),
        # which makes downstream parsers (json.loads + validators that
        # require dict) reject the tool_call and demote the action to
        # invalid. This is the on-disk message dict; serialisation to
        # JSONL still works because dict + json.dumps at write time is
        # equivalent to one-shot encoding.
        return {
            "role": "assistant",
            "content": think_block,
            "tool_calls": [{
                "id": spec.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": public_args,
                },
            }],
        }

    raise ValueError(f"unknown action_type: {spec.action_type}")


# ---------------------------------------------------------------------------
# Tool response building block
# ---------------------------------------------------------------------------

def build_tool_response_message(
    tool_call_id: str,
    content_text: Optional[str] = None,
    content_items: Optional[List[Dict]] = None,
) -> Dict:
    """Render one tool response message.

    Qwen chat template renders ``role: tool`` with its built-in tool response
    wrapper in the final token stream.

    Use ``content_text`` for plain text returns (compress ack). Use
    ``content_items`` for structured returns including recalled frames
    (a list of {"type": "text"} and {"type": "image"} items).
    """
    if content_items is not None:
        body: Any = content_items
    elif content_text is not None:
        body = content_text
    else:
        raise ValueError("either content_text or content_items required")
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": body,
    }


# ---------------------------------------------------------------------------
# Top-level trajectory renderer
# ---------------------------------------------------------------------------

@dataclass
class TurnSpec:
    """One full turn within a trajectory: a (user, assistant) pair plus an
    optional follow-up tool response + assistant message (the Qwen tool 2-turn
    pattern). This module keeps things strictly data; rendering happens via
    ``render_trajectory_messages`` below.
    """
    user: ChunkUserSpec
    assistant: AssistantSpec
    # If the assistant called a tool, the tool returns here and the followup
    # assistant message gives the post-tool observation + action.
    tool_response: Optional[Dict] = None       # built by build_tool_response_message
    followup_assistant: Optional[AssistantSpec] = None


@dataclass
class TrajectorySpec:
    """All inputs needed to render one trajectory."""
    trajectory_type: str                       # TRAJ_TYPE_FROM_START / TRAJ_TYPE_FROM_COMPRESS
    turns: List[TurnSpec]
    available_tools: Tuple[str, ...] = (TOOL_NAME_RECALL,)
    custom_system_prompt: Optional[str] = None  # override default system prompt
    # For ``from_compress`` rows, compact memory is loaded in its own text-only
    # prefill turn before the first visual turn. If omitted, the renderer falls
    # back to the first turn's ``user.inherited_memory`` entries and removes
    # them from that visual turn.
    memory_prefill_text: Optional[str] = None
    # Diagnostics / metadata (preserved into output row; not consumed by the
    # tokenizer):
    trajectory_idx: Optional[int] = None
    video_id: Optional[str] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None


def _memory_entries_to_prefill_text(entries: Sequence[MemoryEntry]) -> str:
    ordered = sorted(entries, key=lambda e: (e.start_sec, e.end_sec))
    return "\n".join(f'<m t="{e.time_str}">{e.text}</m>' for e in ordered)


def render_trajectory_messages(spec: TrajectorySpec) -> Tuple[List[Dict], List[Dict]]:
    """Render a trajectory into ``(messages, tools)``.

    Returns:
        messages: list of OpenAI-format role messages, ready for
            ``tokenizer.apply_chat_template(messages, tools=tools, ...)``.
        tools: list of tool function specs (subset of recall/compress).

    Note: schema only constructs the dict structures. Tokenization, label
    masking, FlexAttention block-mask construction, and frame loading are
    downstream concerns handled by the SFT data loader.
    """
    system_prompt = spec.custom_system_prompt or SYSTEM_PROMPT
    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
    ]

    turns = list(spec.turns)
    memory_prefill_text = str(spec.memory_prefill_text or "").strip()
    if spec.trajectory_type == TRAJ_TYPE_FROM_COMPRESS and turns:
        first_user = turns[0].user
        if first_user.inherited_memory:
            if not memory_prefill_text:
                memory_prefill_text = _memory_entries_to_prefill_text(
                    first_user.inherited_memory
                )
            first_user = replace(first_user, inherited_memory=None)
            turns[0] = replace(turns[0], user=first_user)
        if memory_prefill_text:
            messages.extend([
                {
                    "role": "user",
                    "content": [{"type": "text", "text": memory_prefill_text}],
                },
                {"role": "assistant", "content": MEMORY_LOAD_ACK},
            ])

    for turn in turns:
        user_content = build_user_content(turn.user)
        messages.append({"role": "user", "content": user_content})
        messages.append(build_assistant_message(turn.assistant))

        if turn.tool_response is not None:
            messages.append(turn.tool_response)
            if turn.followup_assistant is not None:
                messages.append(build_assistant_message(turn.followup_assistant))

    tools = build_tools_schema(
        include_recall=TOOL_NAME_RECALL in spec.available_tools,
        include_compress=TOOL_NAME_COMPRESS in spec.available_tools,
    )

    return messages, tools


# ---------------------------------------------------------------------------
# JSONL output schema for one trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRow:
    """The on-disk schema for one trajectory's training row.

    This is what pass5 writes per line. The SFT data loader reads it
    and calls ``apply_chat_template(messages, tools=tools, ...)`` directly.
    """
    trajectory_idx: int
    trajectory_type: str
    video_id: str
    chunk_start: int
    chunk_end: int
    messages: List[Dict]
    tools: List[Dict]
    # Token-level metadata for loss masking and class-balance bookkeeping:
    chunk_boundaries: List[Tuple[int, int]] = field(default_factory=list)
    questions_in_trajectory: List[Dict] = field(default_factory=list)
    compress_event: Optional[Dict] = None
    # Diagnostics:
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "trajectory_idx": self.trajectory_idx,
            "trajectory_type": self.trajectory_type,
            "video_id": self.video_id,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "messages": self.messages,
            "tools": self.tools,
            "chunk_boundaries": [list(t) for t in self.chunk_boundaries],
            "questions_in_trajectory": self.questions_in_trajectory,
            "compress_event": self.compress_event,
            "metadata": self.metadata,
        }
