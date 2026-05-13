"""Unit tests for thinkstream.data.schema.

Run:
  python tests/test_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from thinkstream.data.schema import (  # noqa: E402
    ACTION_COMPRESS,
    ACTION_RECALL,
    ACTION_RESPONSE,
    ACTION_SILENT,
    AssistantSpec,
    ChunkUserSpec,
    MEMORY_CLOSE,
    MEMORY_OPEN,
    QuerySpec,
    STAGE_COMPRESS_MARKER,
    SYSTEM_PROMPT,
    TIMESTAMP_FORMAT,
    TOOL_NAME_COMPRESS,
    TOOL_NAME_RECALL,
    TRAJ_TYPE_FROM_COMPRESS,
    TRAJ_TYPE_FROM_START,
    TrajectorySpec,
    TurnSpec,
    build_assistant_message,
    build_tool_response_message,
    build_tools_schema,
    build_user_content,
    render_trajectory_messages,
)


# ---------- Tools schema ----------

def test_tools_schema_default_recall_only():
    tools = build_tools_schema()
    names = {t["function"]["name"] for t in tools}
    assert names == {TOOL_NAME_RECALL}
    print("[OK] tools_schema_default_recall_only")


def test_tools_schema_filter():
    tools = build_tools_schema(include_compress=False)
    names = {t["function"]["name"] for t in tools}
    assert names == {TOOL_NAME_RECALL}
    print("[OK] tools_schema_filter")


# ---------- User content ----------

def test_user_content_silent_chunk():
    spec = ChunkUserSpec(chunk_idx=5, frame_paths=["f5_0.jpg", "f5_1.jpg"])
    content = build_user_content(spec)
    # Expect: [<t=5> text, video item]
    assert len(content) == 2, f"expected 2 items, got {len(content)}: {content}"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "<t=5>"
    assert content[1]["type"] == "video"
    assert content[1]["video"] == ["f5_0.jpg", "f5_1.jpg"]
    print("[OK] user_content_silent_chunk")


def test_user_content_with_query():
    query = QuerySpec(
        text="What color is the jacket?",
        options=["A. Red", "B. Blue", "C. Green"],
        answer_format="Answer with the letter only.",
    )
    spec = ChunkUserSpec(chunk_idx=12, frame_paths=["f.jpg"], active_query=query)
    content = build_user_content(spec)
    # Block order: <t=N> → video → <active_query> → <response_history>.
    assert len(content) == 4, f"expected 4 items: {content}"
    assert content[0]["text"] == "<t=12>"
    assert content[1]["type"] == "video"
    query_text = content[2]["text"]
    assert "<active_query>" in query_text
    assert "What color is the jacket?" in query_text
    assert "A. Red" in query_text
    assert "Answer with the letter only." in query_text
    assert "<response_history>" in content[3]["text"]
    print("[OK] user_content_with_query")


def test_user_content_compress_stage_drops_video():
    spec = ChunkUserSpec(
        chunk_idx=32,
        frame_paths=["should_not_be_used.jpg"],  # should be dropped due to stage
        stage_marker=STAGE_COMPRESS_MARKER,
        stage_text="Compress range [0, 32].",
    )
    content = build_user_content(spec)
    # Expect: [stage marker text, <t=32>] — no video
    types = [c["type"] for c in content]
    assert "video" not in types, f"video should be dropped on compress stage: {types}"
    assert any(STAGE_COMPRESS_MARKER in c["text"] for c in content if c["type"] == "text")
    print("[OK] user_content_compress_stage_drops_video")


def test_user_content_from_compress_prefix_memory():
    from thinkstream.data.schema import MemoryEntry
    spec = ChunkUserSpec(
        chunk_idx=33,
        frame_paths=["f33.jpg"],
        inherited_memory=[
            MemoryEntry(time_str="0-30", text="Range 0-30 summary: light remained red."),
            MemoryEntry(time_str="31", text="person walking"),
            MemoryEntry(time_str="32", text="still red"),
        ],
    )
    content = build_user_content(spec)
    # First item must be memory block; then timestamp; then video.
    assert content[0]["type"] == "text"
    assert MEMORY_OPEN in content[0]["text"] and MEMORY_CLOSE in content[0]["text"]
    assert "light remained red" in content[0]["text"]
    assert '<m t="31">' in content[0]["text"]
    assert content[1]["text"] == "<t=33>"
    assert content[2]["type"] == "video"
    print("[OK] user_content_from_compress_prefix_memory")


# ---------- Assistant content ----------

def test_assistant_silent():
    msg = build_assistant_message(AssistantSpec(
        think="Person walking, light still red.",
        action_type=ACTION_SILENT,
    ))
    assert msg["role"] == "assistant"
    assert "tool_calls" not in msg
    assert msg["content"] == "<think>Person walking, light still red.</think><silent>"
    print("[OK] assistant_silent")


def test_assistant_response():
    msg = build_assistant_message(AssistantSpec(
        think="Light just turned green.",
        action_type=ACTION_RESPONSE,
        response_text="The light is now green!",
    ))
    assert "<think>" in msg["content"]
    assert "<response>The light is now green!</response>" in msg["content"]
    assert "tool_calls" not in msg
    print("[OK] assistant_response")


def test_assistant_compress_tool_call():
    msg = build_assistant_message(AssistantSpec(
        think="Compressing range [0, 32], stable observations.",
        action_type=ACTION_COMPRESS,
        tool_call_id="comp_1",
        tool_arguments={"time_range": [0, 32], "text": "Light red throughout."},
    ))
    assert msg["role"] == "assistant"
    assert msg["content"] == "<think>Compressing range [0, 32], stable observations.</think>"
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == TOOL_NAME_COMPRESS
    # arguments must be a dict (Qwen3-VL template applies | tojson exactly once)
    args = tc["function"]["arguments"]
    assert isinstance(args, dict), f"arguments must be dict, got {type(args).__name__}"
    assert args["time_range"] == [0, 32]
    assert args["text"] == "Light red throughout."
    print("[OK] assistant_compress_tool_call")


def test_assistant_recall_tool_call():
    msg = build_assistant_message(AssistantSpec(
        think="Need historical context.",
        action_type=ACTION_RECALL,
        tool_call_id="rec_1",
        tool_arguments={"time_range": [0, 10], "query": "first red light"},
    ))
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == TOOL_NAME_RECALL
    args = tc["function"]["arguments"]
    assert isinstance(args, dict)
    assert args["query"] == "first red light"
    assert args["time_range"] == [0, 10]
    print("[OK] assistant_recall_tool_call")


def test_assistant_response_missing_text_raises():
    try:
        build_assistant_message(AssistantSpec(think="x", action_type=ACTION_RESPONSE))
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("[OK] assistant_response_missing_text_raises")


def test_assistant_tool_call_missing_args_raises():
    try:
        build_assistant_message(AssistantSpec(
            think="x", action_type=ACTION_COMPRESS, tool_call_id="c1"))
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("[OK] assistant_tool_call_missing_args_raises")


# ---------- Tool response ----------

def test_tool_response_text():
    msg = build_tool_response_message(tool_call_id="comp_1", content_text="Memory saved.")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "comp_1"
    assert msg["content"] == "Memory saved."
    print("[OK] tool_response_text")


def test_tool_response_items():
    items = [
        {"type": "text", "text": "Recall: 2 frames"},
        {"type": "image", "image": "recalled_2.jpg"},
    ]
    msg = build_tool_response_message(tool_call_id="rec_1", content_items=items)
    assert msg["content"] == items
    print("[OK] tool_response_items")


# ---------- Trajectory rendering end-to-end ----------

def test_render_trajectory_from_start_silent_chain():
    turns = [
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=0, frame_paths=["f0.jpg"]),
            assistant=AssistantSpec(think="Just starting.", action_type=ACTION_SILENT),
        ),
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=1, frame_paths=["f1.jpg"]),
            assistant=AssistantSpec(think="No change.", action_type=ACTION_SILENT),
        ),
    ]
    spec = TrajectorySpec(
        trajectory_type=TRAJ_TYPE_FROM_START,
        turns=turns,
        trajectory_idx=0,
        video_id="vid_a",
    )
    messages, tools = render_trajectory_messages(spec)
    # Expect: system + (user + assistant) x 2 = 5 messages
    assert len(messages) == 5, f"expected 5 messages, got {len(messages)}"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[4]["role"] == "assistant"
    # Active streaming rows expose recall only; compact memory is a separate row.
    assert {t["function"]["name"] for t in tools} == {TOOL_NAME_RECALL}
    print("[OK] render_trajectory_from_start_silent_chain")


def test_render_trajectory_with_tool_call_two_turn_pattern():
    """A recall tool_call followed by tool response + followup assistant."""
    tool_response = build_tool_response_message(
        tool_call_id="rec_1",
        content_text="Light first red at t=2s.",
    )
    turns = [
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=10, frame_paths=["f10.jpg"]),
            assistant=AssistantSpec(
                think="Need historical info.",
                action_type=ACTION_RECALL,
                tool_call_id="rec_1",
                tool_arguments={"time_range": [0, 5], "query": "first red light"},
            ),
            tool_response=tool_response,
            followup_assistant=AssistantSpec(
                think="Got it.",
                action_type=ACTION_RESPONSE,
                response_text="Light first red at 2s.",
            ),
        ),
    ]
    spec = TrajectorySpec(trajectory_type=TRAJ_TYPE_FROM_START, turns=turns)
    messages, _ = render_trajectory_messages(spec)
    # Expect: system + user + assistant(tool_call) + tool + assistant(followup)
    assert len(messages) == 5, f"expected 5 messages, got {len(messages)}"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "tool_calls" in messages[2]
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "rec_1"
    assert messages[4]["role"] == "assistant"
    assert "<response>" in messages[4]["content"]
    print("[OK] render_trajectory_with_tool_call_two_turn_pattern")


def test_render_trajectory_from_compress_first_turn_has_memory():
    from thinkstream.data.schema import MemoryEntry
    turns = [
        TurnSpec(
            user=ChunkUserSpec(
                chunk_idx=33,
                frame_paths=["f33.jpg"],
                inherited_memory=[
                    MemoryEntry(time_str="0-32", text="0-32 summary"),
                ],
            ),
            assistant=AssistantSpec(think="Continuing from compress.", action_type=ACTION_SILENT),
        ),
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=34, frame_paths=["f34.jpg"]),  # no memory
            assistant=AssistantSpec(think="Same.", action_type=ACTION_SILENT),
        ),
    ]
    spec = TrajectorySpec(trajectory_type=TRAJ_TYPE_FROM_COMPRESS, turns=turns)
    messages, _ = render_trajectory_messages(spec)
    # First user message must contain memory block
    first_user = messages[1]
    assert first_user["role"] == "user"
    first_text = first_user["content"][0]["text"]
    assert MEMORY_OPEN in first_text
    assert "0-32 summary" in first_text
    # Second user message must NOT contain memory block (it's a continuation)
    second_user = messages[3]
    assert second_user["role"] == "user"
    second_text = second_user["content"][0]["text"]
    assert MEMORY_OPEN not in second_text
    print("[OK] render_trajectory_from_compress_first_turn_has_memory")


def test_render_trajectory_compress_event_ends_with_tool_call():
    """A trajectory that ends in a compress event: last assistant turn calls
    compress, followed by tool response."""
    tool_response = build_tool_response_message(
        tool_call_id="comp_1",
        content_text="Memory consolidated. Range [0, 32].",
    )
    turns = [
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=0, frame_paths=["f0.jpg"]),
            assistant=AssistantSpec(think="x", action_type=ACTION_SILENT),
        ),
        TurnSpec(
            user=ChunkUserSpec(
                chunk_idx=32,
                stage_marker=STAGE_COMPRESS_MARKER,
                stage_text="Compress chunks [0, 32].",
            ),
            assistant=AssistantSpec(
                think="Consolidating 0-32.",
                action_type=ACTION_COMPRESS,
                tool_call_id="comp_1",
                tool_arguments={"time_range": [0, 32], "summary": "Light red throughout."},
            ),
            tool_response=tool_response,
            followup_assistant=AssistantSpec(
                think="Memory saved.",
                action_type=ACTION_SILENT,
            ),
        ),
    ]
    spec = TrajectorySpec(trajectory_type=TRAJ_TYPE_FROM_START, turns=turns)
    messages, _ = render_trajectory_messages(spec)
    # system + (user+assistant) + (user+assistant_with_compress+tool+assistant)
    # = 1 + 2 + 4 = 7 messages
    assert len(messages) == 7, f"expected 7 messages, got {len(messages)}"
    # Last but one: tool_call assistant
    compress_msg = messages[4]
    assert "tool_calls" in compress_msg
    assert compress_msg["tool_calls"][0]["function"]["name"] == TOOL_NAME_COMPRESS
    print("[OK] render_trajectory_compress_event_ends_with_tool_call")


# ---------- Integration: actual Qwen chat_template applies cleanly ----------

def test_messages_apply_chat_template_smoke():
    """If a Qwen tokenizer is available, render a trajectory and ensure the
    chat template doesn't raise. Skipped silently if no tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[SKIP] apply_chat_template smoke (transformers unavailable)")
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True,
        )
    except Exception as e:
        print(f"[SKIP] apply_chat_template smoke (tokenizer load failed: {e})")
        return

    turns = [
        TurnSpec(
            user=ChunkUserSpec(chunk_idx=0, frame_paths=["dummy.jpg"]),
            assistant=AssistantSpec(think="hi", action_type=ACTION_SILENT),
        ),
        TurnSpec(
            user=ChunkUserSpec(
                chunk_idx=1,
                frame_paths=["dummy.jpg"],
                active_query=QuerySpec(text="What?"),
            ),
            assistant=AssistantSpec(
                think="answering",
                action_type=ACTION_RESPONSE,
                response_text="A.",
            ),
        ),
    ]
    spec = TrajectorySpec(trajectory_type=TRAJ_TYPE_FROM_START, turns=turns)
    messages, tools = render_trajectory_messages(spec)

    # Replace "video" items with simple text so the template doesn't blow up
    # on missing video features (we're only testing the chat-template skeleton).
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            m["content"] = [
                c if c["type"] == "text" else {"type": "text", "text": "[video]"}
                for c in m["content"]
            ]

    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False,
    )
    assert "<|im_start|>system" in rendered
    assert "<tools>" in rendered or "tool_call" in rendered.lower()
    assert "<silent>" in rendered
    assert "<response>" in rendered
    print(f"[OK] apply_chat_template_smoke ({len(rendered)} chars)")


def test_system_prompt_v2_contains_key_rules():
    """The unified v2 prompt absorbs key rules from all three V12 stage
    variants. Spot-check that the headline rules are present."""
    # Output grammar
    assert "<silent>" in SYSTEM_PROMPT
    assert "<response>" in SYSTEM_PROMPT
    assert "<think>" in SYSTEM_PROMPT
    # Recall rules (from V12_STREAMING)
    assert "recall" in SYSTEM_PROMPT.lower()
    assert "time_range" in SYSTEM_PROMPT
    # Memory orientation rules
    assert "<memory>" in SYSTEM_PROMPT or "<MEM>" in SYSTEM_PROMPT
    print("[OK] system_prompt_v2_contains_key_rules")


def test_agent_protocol_re_exports_canonical():
    """agent_protocol.py re-exports SYSTEM_PROMPT and get_canonical_system_prompt."""
    from thinkstream.data.agent_protocol import (
        SYSTEM_PROMPT as ap_prompt,
        get_canonical_system_prompt,
    )
    assert ap_prompt == SYSTEM_PROMPT
    assert get_canonical_system_prompt() == SYSTEM_PROMPT
    print("[OK] agent_protocol_re_exports_canonical")


def test_get_canonical_system_prompt_always_returns_unified():
    """get_canonical_system_prompt() always returns the unified prompt; env
    var no longer affects the result (stage transitions now use user-side
    ``<stage:...>`` markers, not different prompts)."""
    import os
    from thinkstream.data.agent_protocol import get_canonical_system_prompt

    saved = os.environ.pop("THINKSTREAM_SYSTEM_PROMPT", None)
    try:
        assert get_canonical_system_prompt() == SYSTEM_PROMPT
        for value in ["v2", "unified", "1", "true", "yes", "legacy", "anything"]:
            os.environ["THINKSTREAM_SYSTEM_PROMPT"] = value
            assert get_canonical_system_prompt() == SYSTEM_PROMPT
    finally:
        if saved is None:
            os.environ.pop("THINKSTREAM_SYSTEM_PROMPT", None)
        else:
            os.environ["THINKSTREAM_SYSTEM_PROMPT"] = saved
    print("[OK] get_canonical_system_prompt_always_returns_unified")


def test_system_prompt_for_frame_protocol_selects_compact_prompt():
    """Streaming/post-recall use the canonical prompt; compress/inter-chunk
    uses the compact-memory prompt for the standalone memory-update row."""
    from thinkstream.data.agent_protocol import (
        COMPACT_MEMORY_SYSTEM_PROMPT,
        system_prompt_for_frame_protocol,
        SYSTEM_PROMPT as ap_prompt,
    )
    p1 = system_prompt_for_frame_protocol(prompt_kind="streaming")
    p2 = system_prompt_for_frame_protocol(prompt_kind="compress")
    p3 = system_prompt_for_frame_protocol(inter_chunk=True)
    p4 = system_prompt_for_frame_protocol(prompt_kind="recall_response")
    p5 = system_prompt_for_frame_protocol()
    assert p1 == p4 == p5 == ap_prompt
    assert p2 == p3 == COMPACT_MEMORY_SYSTEM_PROMPT
    print("[OK] system_prompt_for_frame_protocol_selects_compact_prompt")


if __name__ == "__main__":
    test_tools_schema_default_recall_only()
    test_tools_schema_filter()
    test_user_content_silent_chunk()
    test_user_content_with_query()
    test_user_content_compress_stage_drops_video()
    test_user_content_from_compress_prefix_memory()
    test_assistant_silent()
    test_assistant_response()
    test_assistant_compress_tool_call()
    test_assistant_recall_tool_call()
    test_assistant_response_missing_text_raises()
    test_assistant_tool_call_missing_args_raises()
    test_tool_response_text()
    test_tool_response_items()
    test_render_trajectory_from_start_silent_chain()
    test_render_trajectory_with_tool_call_two_turn_pattern()
    test_render_trajectory_from_compress_first_turn_has_memory()
    test_render_trajectory_compress_event_ends_with_tool_call()
    test_messages_apply_chat_template_smoke()
    test_system_prompt_v2_contains_key_rules()
    test_agent_protocol_re_exports_canonical()
    test_get_canonical_system_prompt_always_returns_unified()
    test_system_prompt_for_frame_protocol_selects_compact_prompt()
    print("\n✅ all schema tests passed")
