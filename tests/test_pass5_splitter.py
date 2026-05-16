"""Tests for pass5 v2 splitter + translator.

Run:
  python tests/test_pass5_splitter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data.pass5_splitter import (  # noqa: E402
    build_question_metadata,
    build_questions_by_chunk,
    extract_compress_summary,
    is_compress_sample,
    parse_assistant_output_to_spec,
    render_trajectory_record_to_rows,
    split_samples_by_compress,
    translate_recall_sample,
    translate_sample_to_turn,
)
from thinkstream.data.schema import (  # noqa: E402
    ACTION_COMPRESS,
    ACTION_COMPRESS_SELECT,
    ACTION_RECALL,
    ACTION_RESPONSE,
    ACTION_SILENT,
    ChunkUserSpec,
    MEMORY_LOAD_ACK,
    MEMORY_OPEN,
    STAGE_COMPRESS_MARKER,
    TOOL_NAME_COMPRESS,
    TOOL_NAME_RECALL,
    TRAJ_TYPE_FROM_COMPRESS,
    TRAJ_TYPE_FROM_START,
)


def _dummy_frame_resolver(chunk_idx: int):
    return [f"frame_{chunk_idx}_0.jpg", f"frame_{chunk_idx}_1.jpg"]


def _row_user_text_for_chunk(row: dict, chunk_idx: int) -> str:
    needle = f"<t={int(chunk_idx)}>"
    for message in row.get("messages", []):
        if message.get("role") != "user":
            continue
        text = "\n".join(
            c.get("text", "")
            for c in message.get("content", [])
            if c.get("type") == "text"
        )
        if needle in text:
            return text
    return ""


# ============== parse_assistant_output_to_spec ==============

def test_parse_silent_empty_answer():
    spec = parse_assistant_output_to_spec(
        "<think>nothing yet</think></Silence>", chunk_idx=0,
    )
    assert spec.action_type == ACTION_SILENT
    assert spec.think == "nothing yet"
    print("[OK] parse_silent_empty_answer")


def test_parse_response_with_answer():
    spec = parse_assistant_output_to_spec(
        "<think>light is green</think></Response> The light turned green!",
        chunk_idx=5,
    )
    assert spec.action_type == ACTION_RESPONSE
    assert spec.response_text == "The light turned green!"
    print("[OK] parse_response_with_answer")


def test_parse_legacy_answer_tag_to_response():
    spec = parse_assistant_output_to_spec(
        "<think>count changed</think><answer>3</answer>",
        chunk_idx=8,
    )
    assert spec.action_type == ACTION_RESPONSE
    assert spec.response_text == "3"
    print("[OK] parse_legacy_answer_tag_to_response")


def test_parse_compress_tool_call_is_rejected():
    text = (
        "<think>compressing 0-32</think>"
        '<tool_call>{"name": "compress", "arguments": '
        '{"time_range": [0, 32], "text": "Light red throughout."}}</tool_call>'
    )
    try:
        parse_assistant_output_to_spec(text, chunk_idx=32)
    except ValueError as exc:
        assert "compress tool_call is obsolete" in str(exc)
    else:
        raise AssertionError("compress tool_call should be rejected")
    print("[OK] parse_compress_tool_call_is_rejected")


def test_parse_recall_tool_call():
    text = (
        "<think>need history</think>"
        '<tool_call>{"name": "recall", "arguments": '
        '{"start_time": 0, "end_time": 11}}</tool_call>'
    )
    spec = parse_assistant_output_to_spec(text, chunk_idx=15)
    assert spec.action_type == ACTION_RECALL
    assert spec.tool_call_id == "rec_15"
    assert spec.tool_arguments == {"start_time": 0, "end_time": 11}
    print("[OK] parse_recall_tool_call")


def test_legacy_v12_inter_chunk_is_compress_boundary():
    sample = {
        "chunk_idx": 7,
        "sample_type": "silent",
        "v12_inter_chunk": True,
        "output": (
            "<think>compress</think>"
            '<tool_call>{"name":"compress","arguments":'
            '{"time_range":[0,6],"text":"summary"}}</tool_call>'
        ),
    }
    assert is_compress_sample(sample)
    try:
        translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    except ValueError as exc:
        assert "standalone compact_memory_update rows" in str(exc)
    else:
        raise AssertionError("compress sample rendered as a streaming turn")
    print("[OK] legacy_v12_inter_chunk_is_compress_boundary")


def test_parse_malformed_tool_call_falls_back_to_silent():
    text = "<think>x</think><tool_call>not json</tool_call>"
    spec = parse_assistant_output_to_spec(text, chunk_idx=0)
    assert spec.action_type == ACTION_SILENT
    print("[OK] parse_malformed_tool_call_falls_back_to_silent")


def test_parse_no_think_section():
    spec = parse_assistant_output_to_spec("</Response> hello", chunk_idx=0)
    assert spec.action_type == ACTION_RESPONSE
    assert spec.think == ""
    print("[OK] parse_no_think_section")


# ============== questions_by_chunk ==============

def test_build_questions_by_chunk_filters_missing_ask():
    questions = [
        {
            "ask_chunk": 5,
            "question": "Q1?",
            "options": ["A", "B"],
            "answer_instruction": "letter only",
        },
        {
            "ask_chunk": -1,  # invalid
            "question": "ignored",
        },
        {
            "ask_chunk": 10,
            "question": "",  # empty question, should be skipped
        },
        {
            "ask_chunk": 20,
            "question": "Q2?",
        },
    ]
    by_chunk = build_questions_by_chunk(questions)
    assert set(by_chunk.keys()) == {5, 20}
    # values are now list[QuerySpec] (P3.12 fix: multi-question per chunk).
    assert len(by_chunk[5]) == 1
    assert by_chunk[5][0].text == "Q1?"
    assert by_chunk[5][0].options == ["A", "B"]
    assert by_chunk[5][0].answer_format == (
        "Answer format: letter plus option text, e.g. A) option text."
    )
    assert by_chunk[20][0].options is None
    print("[OK] build_questions_by_chunk_filters_missing_ask")


def test_build_questions_by_chunk_preserves_multi_question_per_chunk():
    """Multiple questions with the same ask_chunk must NOT overwrite (P3.12)."""
    questions = [
        {"ask_chunk": 7, "question": "Q1?"},
        {"ask_chunk": 7, "question": "Q2?"},
        {"ask_chunk": 7, "question": "Q3?"},
    ]
    by_chunk = build_questions_by_chunk(questions)
    assert set(by_chunk.keys()) == {7}
    assert len(by_chunk[7]) == 3
    assert [q.text for q in by_chunk[7]] == ["Q1?", "Q2?", "Q3?"]
    print("[OK] build_questions_by_chunk_preserves_multi_question_per_chunk")


def test_build_questions_by_chunk_adds_canonical_answer_format():
    questions = [{
        "ask_chunk": 9,
        "question": "Is the door currently open?",
        "answer_form": "binary",
    }]
    by_chunk = build_questions_by_chunk(questions)
    spec = by_chunk[9][0]
    assert spec.answer_form == "binary"
    assert spec.answer_format == "Answer format: Yes or No only."

    meta = build_question_metadata(questions)[0]
    assert meta["spec"].answer_format == "Answer format: Yes or No only."


# ============== compress detection + splitting ==============

def test_is_compress_sample_both_flags():
    assert is_compress_sample({"sample_type": "compress"}) is True
    assert is_compress_sample({"inter_chunk": True}) is True
    assert is_compress_sample({"sample_type": "silent"}) is False
    assert is_compress_sample({}) is False
    print("[OK] is_compress_sample_both_flags")


def test_split_samples_by_compress_single_segment():
    samples = [
        {"chunk_idx": 0, "sample_type": "silent"},
        {"chunk_idx": 1, "sample_type": "silent"},
        {"chunk_idx": 2, "sample_type": "response"},
    ]
    segs = split_samples_by_compress(samples)
    assert len(segs) == 1
    assert len(segs[0]) == 3
    print("[OK] split_samples_by_compress_single_segment")


def test_split_samples_by_compress_multi_segment():
    samples = [
        {"chunk_idx": 0, "sample_type": "silent"},
        {"chunk_idx": 1, "sample_type": "silent"},
        {"chunk_idx": 2, "sample_type": "compress", "inter_chunk": True},
        {"chunk_idx": 3, "sample_type": "silent"},
        {"chunk_idx": 4, "sample_type": "response"},
        {"chunk_idx": 5, "sample_type": "compress", "inter_chunk": True},
        {"chunk_idx": 6, "sample_type": "silent"},
    ]
    segs = split_samples_by_compress(samples)
    assert len(segs) == 3
    assert [s["chunk_idx"] for s in segs[0]] == [0, 1, 2]   # ends WITH compress
    assert [s["chunk_idx"] for s in segs[1]] == [3, 4, 5]
    assert [s["chunk_idx"] for s in segs[2]] == [6]
    print("[OK] split_samples_by_compress_multi_segment")


def test_split_samples_compress_at_end_no_trailing_empty():
    samples = [
        {"chunk_idx": 0, "sample_type": "silent"},
        {"chunk_idx": 1, "sample_type": "compress", "inter_chunk": True},
    ]
    segs = split_samples_by_compress(samples)
    assert len(segs) == 1  # no trailing empty segment
    print("[OK] split_samples_compress_at_end_no_trailing_empty")


# ============== extract_compress_summary ==============

def test_extract_compress_summary_from_gold_caption_preserves_gold_chunk_range():
    sample = {
        "gold_caption": (
            "Light red throughout 0-8. A person walks across the room. "
            "The camera shows the table. The person picks up the cup."
        ),
        "gold_compress_chunks": [0, 1, 2, 30, 31, 32],
        "output": "ignored",
    }
    text, chunks = extract_compress_summary(sample)
    assert text.count("<m ") == 1
    assert text == (
        '  <m t="0-32">Light red throughout 0-8. A person walks across the '
        'room. The camera shows the table. The person picks up the cup.</m>'
    )
    assert chunks == [0, 1, 2, 30, 31, 32]
    print("[OK] extract_compress_summary_from_gold_caption_preserves_gold_chunk_range")


def test_extract_compress_summary_from_tool_call_fallback():
    sample = {
        "output": (
            "<think>x</think>"
            '<tool_call>{"name": "compress", "arguments": '
            '{"time_range": [0, 16], "text": "fallback summary"}}</tool_call>'
        ),
    }
    text, chunks = extract_compress_summary(sample)
    assert text == '  <m t="0-15">fallback summary</m>'
    assert chunks == list(range(0, 16))
    print("[OK] extract_compress_summary_from_tool_call_fallback")


# ============== translate_sample_to_turn ==============

def test_translate_silent_sample_renders_video_and_no_query():
    sample = {
        "chunk_idx": 7,
        "sample_type": "silent",
        "output": "<think>walking</think></Silence>",
    }
    turn = translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    assert turn.user.chunk_idx == 7
    assert turn.user.frame_paths == ["frame_7_0.jpg", "frame_7_1.jpg"]
    assert turn.user.active_query is None
    assert turn.user.stage_marker is None
    assert turn.assistant.action_type == ACTION_SILENT
    print("[OK] translate_silent_sample")


def test_translate_response_sample_with_query():
    from thinkstream.data.schema import QuerySpec
    sample = {
        "chunk_idx": 12,
        "sample_type": "response",
        "output": "<think>green!</think></Response> A",
    }
    questions = {12: [QuerySpec(text="What color?", options=["A", "B"])]}
    turn = translate_sample_to_turn(sample, questions, _dummy_frame_resolver)
    assert turn.assistant.action_type == ACTION_RESPONSE
    assert turn.assistant.response_text == "A"
    assert turn.user.active_query is not None
    assert turn.user.active_query.text == "What color?"
    print("[OK] translate_response_sample_with_query")


def test_render_trajectory_normalizes_mc_answer_to_letter_plus_text():
    record = {
        "video_id": "vid_mc",
        "trajectory_id": "vid_mc_t0",
        "questions": [{
            "ask_chunk": 4,
            "answer_chunks": [4],
            "question": "Which color?",
            "answer_form": "multiple_choice",
            "answer_style": "letter_only",
            "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
            "options": ["A) red", "B) blue", "C) green", "D) yellow"],
            "correct_option": "B",
            "per_emit_answers": [{"chunk": 4, "value": "B"}],
        }],
        "samples": [{
            "chunk_idx": 4,
            "sample_type": "response",
            "output": "<think>blue</think></Response> B",
        }],
    }

    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    row = rows[0]
    user_text = _row_user_text_for_chunk(row, 4)
    assistant_text = "\n".join(
        str(m.get("content") or "")
        for m in row["messages"]
        if m.get("role") == "assistant"
    )

    assert "Answer format: letter plus option text" in user_text
    assert "one letter only" not in user_text
    assert "</Response> B) blue" in assistant_text
    assert row["questions_in_segment"][0]["sft_answer"] == "B) blue"
    print("[OK] render_trajectory_normalizes_mc_answer_to_letter_plus_text")


def test_translate_compress_sample_rejected_from_streaming_turn():
    sample = {
        "chunk_idx": 32,
        "sample_type": "compress",
        "inter_chunk": True,
        "output": (
            "<think>compressing</think>"
            '<tool_call>{"name": "compress", "arguments": '
            '{"time_range": [0, 32], "text": "summary"}}</tool_call>'
        ),
    }
    try:
        translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    except ValueError as exc:
        assert "compact_memory_update" in str(exc)
    else:
        raise AssertionError("compress samples must not render as streaming turns")
    print("[OK] translate_compress_sample_rejected_from_streaming_turn")


def test_translate_recall_sample_builds_two_turn_pattern():
    sample = {
        "chunk_idx": 15,
        "sample_type": "recall",
        "v12_assistant_turn_1": (
            "<think>need history</think>"
            '<tool_call>{"name": "recall", "arguments": '
            '{"start_time": 0, "end_time": 1}}</tool_call>'
        ),
        "v12_assistant_turn_2": "<think>got it</think></Response> 2s",
        "recall_result": {
            "source": "historical_frames",
            "time": "0-1",
            "returned_chunks": [0, 1],
            "status": "ok",
            "time_range": [0, 1],
            "n_frames": 2,
            "text": "Light first red at t=2s.",
        },
        "recalled_frames": {
            "time_range": [0, 1],
            "source": "historical_frames",
            "n_frames": 2,
            "frame_paths": ["recall_0.jpg", "recall_1.jpg"],
        },
    }
    turn = translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    assert turn.assistant.action_type == ACTION_RECALL
    assert turn.assistant.tool_arguments == {"start_time": 0, "end_time": 1}
    assert turn.tool_response is not None
    assert turn.tool_response["role"] == "tool"
    body = turn.tool_response["content"]
    assert isinstance(body, list)
    assert body[0]["type"] == "text"
    assert body[0]["text"] == "The recall tool returned historical video frames for t=0-1."
    assert "<recalled_frames>" not in body[0]["text"]
    assert body[1]["type"] == "video"
    assert body[1]["video"] == ["recall_0.jpg", "recall_1.jpg"]
    assert body[1]["min_pixels"] == 200704
    assert body[1]["max_pixels"] == 401408
    assert len([x for x in body if x.get("type") == "text"]) == 1
    assert "<recall_result>" not in json.dumps(body, ensure_ascii=False)
    assert "Light first red" not in json.dumps(body, ensure_ascii=False)
    assert turn.followup_assistant is not None
    assert turn.followup_assistant.action_type == ACTION_RESPONSE
    assert turn.followup_assistant.response_text == "2s"
    print("[OK] translate_recall_sample_builds_two_turn_pattern")


# ============== End-to-end trajectory record rendering ==============

def _make_pass4_record_with_compress():
    return {
        "video_id": "vid_a",
        "trajectory_id": "vid_a_traj_0",
        "questions": [
            {
                "ask_chunk": 4,
                "question": "What color?",
                "options": ["A. Red", "B. Blue"],
                "answer_instruction": "letter only",
            },
            {
                "ask_chunk": 33,
                "question": "When did it change?",
                "options": None,
                "answer_instruction": None,
            },
        ],
        "samples": [
            {
                "chunk_idx": 0,
                "sample_type": "silent",
                "output": "<think>start</think></Silence>",
            },
            {
                "chunk_idx": 1,
                "sample_type": "silent",
                "output": "<think>quiet</think></Silence>",
            },
            {
                "chunk_idx": 4,
                "sample_type": "response",
                "output": "<think>I see red</think></Response> A",
            },
            {
                "chunk_idx": 5,
                "sample_type": "silent",
                "output": "<think>moving on</think></Silence>",
            },
            {
                "chunk_idx": 32,
                "sample_type": "compress",
                "inter_chunk": True,
                "gold_caption": "Light red 0-5, person walked.",
                "gold_compress_chunks": [0, 1, 2, 3, 4, 5],
                "output": (
                    "<think>compressing 0-5</think>"
                    '<tool_call>{"name": "compress", "arguments": '
                    '{"time_range": [0, 6], "text": "Light red 0-5, person walked."}}</tool_call>'
                ),
            },
            {
                "chunk_idx": 33,
                "sample_type": "response",
                "output": "<think>green now</think></Response> 33",
            },
            {
                "chunk_idx": 34,
                "sample_type": "silent",
                "output": "<think>x</think></Silence>",
            },
        ],
    }


def test_render_trajectory_record_produces_compact_boundary_row():
    record = _make_pass4_record_with_compress()
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)

    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"

    # Row 0: from_start, visual chunks before the compact update.
    seg0 = rows[0]
    assert seg0["trajectory_type"] == TRAJ_TYPE_FROM_START
    assert seg0["chunk_start"] == 0
    assert seg0["chunk_end"] == 5
    assert seg0["compress_event"] is None

    # Row 1: standalone text-only compact-memory update.
    compact = rows[1]
    assert compact["trajectory_type"] == "compact_memory_update"
    assert compact["chunk_start"] == 32
    assert compact["chunk_end"] == 32
    assert compact["compress_event"]["chunk_idx"] == 32
    assert "Light red 0-5, person walked." in compact["compress_event"]["summary_text"]

    # Row 2: from_compress, chunks 33..34, prefix memory present.
    seg1 = rows[2]
    assert seg1["trajectory_type"] == TRAJ_TYPE_FROM_COMPRESS
    assert seg1["chunk_start"] == 33
    assert seg1["chunk_end"] == 34
    assert seg1["compress_event"] is None

    # Segment 1 must first load compact-memory text in a standalone prefill turn,
    # then acknowledge it before the first visual turn.
    assert seg1["messages"][1]["role"] == "user"
    prefill_text = "\n".join(
        c["text"] for c in seg1["messages"][1]["content"] if c.get("type") == "text"
    )
    assert "<m " in prefill_text
    assert seg1["messages"][2] == {"role": "assistant", "content": MEMORY_LOAD_ACK}
    first_visual = seg1["messages"][3]
    first_visual_text = "\n".join(
        c["text"] for c in first_visual["content"] if c.get("type") == "text"
    )
    assert "<m " not in first_visual_text
    assert "<t=33>" in first_visual_text
    assert any(c.get("type") == "video" for c in first_visual["content"])
    print("[OK] render_trajectory_record_produces_compact_boundary_row")


def test_same_chunk_compress_precedes_trigger_chunk_response():
    record = _make_pass4_record_with_compress()
    record["samples"] = [
        {"chunk_idx": 31, "sample_type": "silent",
         "output": "<think>before</think></Silence>"},
        {"chunk_idx": 32, "sample_type": "response",
         "output": "<think>answer at boundary</think></Response> done"},
        {
            "chunk_idx": 32,
            "sample_type": "compress",
            "inter_chunk": True,
            "gold_caption": (
                '  <m t="0-31">summary before trigger chunk</m>'
            ),
            "gold_compress_chunks": list(range(32)),
            "output": (
                '  <m t="0-31">summary before trigger chunk</m>'
            ),
        },
        {"chunk_idx": 33, "sample_type": "silent",
         "output": "<think>after</think></Silence>"},
    ]

    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    assert rows[0]["trajectory_type"] == TRAJ_TYPE_FROM_START
    assert rows[0]["chunk_end"] == 31
    assert rows[1]["trajectory_type"] == "compact_memory_update"
    assert rows[1]["chunk_start"] == 32
    assert rows[2]["trajectory_type"] == TRAJ_TYPE_FROM_COMPRESS
    assert rows[2]["chunk_start"] == 32
    assert any(
        m.get("role") == "assistant" and "answer at boundary" in str(m.get("content"))
        for m in rows[2]["messages"]
    )
    assert "after" in rows[2]["messages"][-1]["content"]
    print("[OK] same_chunk_compress_precedes_trigger_chunk_response")


def test_render_questions_partitioned_by_segment():
    record = _make_pass4_record_with_compress()
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)

    seg0_qs = rows[0]["questions_in_segment"]
    seg1_qs = rows[2]["questions_in_segment"]

    seg0_ask_chunks = {q["ask_chunk"] for q in seg0_qs}
    seg1_ask_chunks = {q["ask_chunk"] for q in seg1_qs}

    assert seg0_ask_chunks == {4}, f"seg0 questions: {seg0_ask_chunks}"
    assert seg1_ask_chunks == {33}, f"seg1 questions: {seg1_ask_chunks}"
    print("[OK] render_questions_partitioned_by_segment")


def test_render_handles_video_with_no_compress():
    record = {
        "video_id": "vid_no_compress",
        "trajectory_id": "vid_no_compress_traj_0",
        "questions": [],
        "samples": [
            {"chunk_idx": 0, "sample_type": "silent",
             "output": "<think>x</think></Silence>"},
            {"chunk_idx": 1, "sample_type": "silent",
             "output": "<think>x</think></Silence>"},
        ],
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    assert len(rows) == 1
    assert rows[0]["trajectory_type"] == TRAJ_TYPE_FROM_START
    assert rows[0]["compress_event"] is None
    print("[OK] render_handles_video_with_no_compress")


def test_render_messages_are_json_serializable():
    record = _make_pass4_record_with_compress()
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    # Round-trip through JSON; trips on any non-serializable artifact.
    blob = json.dumps(rows, ensure_ascii=False)
    restored = json.loads(blob)
    assert len(restored) == len(rows)
    assert restored[0]["trajectory_type"] == TRAJ_TYPE_FROM_START
    print(f"[OK] render_messages_are_json_serializable ({len(blob)} bytes)")


def test_inherited_queries_carry_across_compress_boundary():
    """Open queries asked before a compress event but answered after must be
    re-injected into the from_compress segment's first user turn.

    Without this, the SFT row emits a gold </Response> X at some
    chunk with no upstream query context — the model has no input signal
    correlating its answer to any question, so the training token loses
    its anchor.
    """
    record = {
        "video_id": "vid_open_q",
        "trajectory_id": "vid_open_q_t0",
        "questions": [
            {
                "ask_chunk": 30,
                "answer_chunks": [35],
                "question": "Did the person re-enter?",
                "options": ["A. Yes", "B. No"],
                "answer_instruction": "Letter only.",
            },
        ],
        "samples": (
            [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": f"<think>obs{c}</think></Silence>"}
                for c in range(32)
            ]
            + [{
                "chunk_idx": 32,
                "sample_type": "compress",
                "inter_chunk": True,
                "gold_caption": "0-30 setup",
                "output": (
                    "<think>compress</think>"
                    '<tool_call>{"name":"compress","arguments":'
                    '{"time_range":[0,30],"text":"0-30 setup"}}</tool_call>'
                ),
            }]
            + [
                {"chunk_idx": 33, "sample_type": "silent",
                 "output": "<think>scanning</think></Silence>"},
                {"chunk_idx": 34, "sample_type": "silent",
                 "output": "<think>scanning</think></Silence>"},
                {"chunk_idx": 35, "sample_type": "response",
                 "output": "<think>re-entered</think></Response> A"},
            ]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    assert len(rows) == 3
    seg1 = rows[2]
    assert seg1["trajectory_type"] == TRAJ_TYPE_FROM_COMPRESS
    first_user = seg1["messages"][3]
    texts = [c["text"] for c in first_user["content"] if c.get("type") == "text"]
    joined = "\n".join(texts)
    assert "<active_query>" in joined, "open query should appear in first visual turn"
    assert "Did the person re-enter?" in joined
    assert "[30s] Q:" in joined, "should preserve original ask_chunk"
    # No prior response: chunk 35 is the only answer slot and it's in seg 1.
    assert "<response_history>" in joined
    assert "A:" not in joined
    answer_user_text = _row_user_text_for_chunk(seg1, 35)
    assert "<active_query>" in answer_user_text
    assert "Did the person re-enter?" in answer_user_text
    assert "<response_history>" in answer_user_text
    assert "A:" not in answer_user_text
    print("[OK] inherited_queries_carry_across_compress_boundary")


def test_inherited_responses_carry_for_multi_event_query():
    """For a multi-event query already answered before the boundary AND with
    more answers expected after, both the query and the prior responses
    must be inherited."""
    record = {
        "video_id": "vid_multi_evt",
        "trajectory_id": "vid_multi_evt_t0",
        "questions": [
            {
                "ask_chunk": 5,
                "answer_chunks": [10, 35],
                "question": "How many people walked by?",
                "options": None,
                "answer_instruction": "Integer count.",
            },
        ],
        "samples": (
            [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": f"<think>q{c}</think></Silence>"}
                for c in range(10)
            ]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>one person</think></Response> 1"}]
            + [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": f"<think>obs{c}</think></Silence>"}
                for c in range(11, 32)
            ]
            + [{
                "chunk_idx": 32, "sample_type": "compress", "inter_chunk": True,
                "gold_caption": "0-30 one person walked",
                "output": (
                    "<think>compress</think>"
                    '<tool_call>{"name":"compress","arguments":'
                    '{"time_range":[0,30],"text":"0-30 one person walked"}}</tool_call>'
                ),
            }]
            + [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": f"<think>scan{c}</think></Silence>"}
                for c in range(33, 35)
            ]
            + [{"chunk_idx": 35, "sample_type": "response",
                "output": "<think>second person</think></Response> 2"}]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    seg1 = rows[2]
    first_user = seg1["messages"][3]
    joined = "\n".join(
        c["text"] for c in first_user["content"] if c.get("type") == "text"
    )
    assert "<active_query>" in joined and 'How many people' in joined
    assert "<response_history>" in joined, "prior chunk-10 response must be inherited"
    assert "[10s] A: 1" in joined
    answer_user_text = _row_user_text_for_chunk(seg1, 35)
    assert "<active_query>" in answer_user_text
    assert "How many people walked by?" in answer_user_text
    assert "<response_history>" in answer_user_text
    assert "[10s] A: 1" in answer_user_text
    print("[OK] inherited_responses_carry_for_multi_event_query")


def test_adaptive_count_query_refreshes_after_answer():
    record = {
        "video_id": "vid_count_refresh",
        "trajectory_id": "vid_count_refresh_t0",
        "questions": [
            {
                "ask_chunk": 5,
                "answer_chunks": [10, 15],
                "question": "How many cups have appeared so far?",
                "answer_form": "number",
                "question_type": "multi_emit",
                "question_way": "repeated_count",
                "answer_instruction": "Integer count.",
            },
        ],
        "samples": (
            [{"chunk_idx": c, "sample_type": "silent",
              "output": f"<think>obs{c}</think></Silence>"}
             for c in range(5)]
            + [{"chunk_idx": 5, "sample_type": "silent",
                "output": "<think>question opened</think></Silence>"}]
            + [{"chunk_idx": c, "sample_type": "silent",
                "output": f"<think>wait{c}</think></Silence>"}
               for c in range(6, 10)]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>one cup</think></Response> 1"}]
            + [{"chunk_idx": c, "sample_type": "silent",
                "output": f"<think>track{c}</think></Silence>"}
               for c in range(11, 15)]
            + [{"chunk_idx": 15, "sample_type": "response",
                "output": "<think>two cups</think></Response> 2"}]
        ),
    }
    row = render_trajectory_record_to_rows(record, _dummy_frame_resolver)[0]
    post_answer_text = _row_user_text_for_chunk(row, 11)
    assert "<active_query>" in post_answer_text
    assert "How many cups have appeared so far?" in post_answer_text
    assert "<response_history>" in post_answer_text
    assert "[10s] A: 1" in post_answer_text


def test_adaptive_status_probe_refresh_omits_prior_history():
    record = {
        "video_id": "vid_status_probe",
        "trajectory_id": "vid_status_probe_t0",
        "questions": [
            {
                "ask_chunk": 5,
                "answer_chunks": [5, 10],
                "question": "Is the door currently open?",
                "answer_form": "binary",
                "question_type": "multi_emit",
                "question_way": "current_status_probe",
                "evidence_type": "status_probe_stream",
                "answer_instruction": "Answer Yes or No.",
            },
        ],
        "samples": (
            [{"chunk_idx": c, "sample_type": "silent",
              "output": f"<think>obs{c}</think></Silence>"}
             for c in range(5)]
            + [{"chunk_idx": 5, "sample_type": "response",
                "output": "<think>open now</think></Response> Yes"}]
            + [{"chunk_idx": c, "sample_type": "silent",
                "output": f"<think>wait{c}</think></Silence>"}
               for c in range(6, 10)]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>closed now</think></Response> No"}]
        ),
    }
    row = render_trajectory_record_to_rows(record, _dummy_frame_resolver)[0]
    probe_text = _row_user_text_for_chunk(row, 10)
    assert "<active_query>" in probe_text
    assert "Is the door currently open?" in probe_text
    assert "<response_history>" in probe_text
    assert "[5s] A: Yes" not in probe_text


def test_resolved_query_not_inherited():
    """A query whose ALL answer_chunks are before the segment boundary must
    NOT be inherited — it's already fully resolved."""
    record = {
        "video_id": "vid_resolved",
        "trajectory_id": "vid_resolved_t0",
        "questions": [
            {
                "ask_chunk": 5,
                "answer_chunks": [10],
                "question": "Closed before boundary",
                "options": None,
                "answer_instruction": None,
            },
        ],
        "samples": (
            [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": "<think>x</think></Silence>"}
                for c in range(10)
            ]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>done</think></Response> done"}]
            + [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": "<think>x</think></Silence>"}
                for c in range(11, 32)
            ]
            + [{"chunk_idx": 32, "sample_type": "compress", "inter_chunk": True,
                "gold_caption": "0-30", "output": (
                    "<think>compress</think>"
                    '<tool_call>{"name":"compress","arguments":'
                    '{"time_range":[0,30],"text":"0-30"}}</tool_call>')}]
            + [{"chunk_idx": 33, "sample_type": "silent",
                "output": "<think>x</think></Silence>"}]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    seg1 = rows[2]
    first_user = seg1["messages"][3]
    joined = "\n".join(
        c["text"] for c in first_user["content"] if c.get("type") == "text"
    )
    assert "Closed before boundary" not in joined
    assert "<" + "query>" not in joined
    print("[OK] resolved_query_not_inherited")


def test_render_empty_record_returns_empty():
    rows = render_trajectory_record_to_rows(
        {"video_id": "x", "trajectory_id": "y", "samples": []},
        _dummy_frame_resolver,
    )
    assert rows == []
    print("[OK] render_empty_record_returns_empty")


if __name__ == "__main__":
    test_parse_silent_empty_answer()
    test_parse_response_with_answer()
    test_parse_legacy_answer_tag_to_response()
    test_parse_compress_tool_call()
    test_parse_recall_tool_call()
    test_parse_malformed_tool_call_falls_back_to_silent()
    test_parse_no_think_section()
    test_build_questions_by_chunk_filters_missing_ask()
    test_build_questions_by_chunk_preserves_multi_question_per_chunk()
    test_build_questions_by_chunk_adds_canonical_answer_format()
    test_is_compress_sample_both_flags()
    test_split_samples_by_compress_single_segment()
    test_split_samples_by_compress_multi_segment()
    test_split_samples_compress_at_end_no_trailing_empty()
    test_extract_compress_summary_from_gold_caption()
    test_extract_compress_summary_from_tool_call_fallback()
    test_translate_silent_sample_renders_video_and_no_query()
    test_translate_response_sample_with_query()
    test_translate_compress_sample_rejected_from_streaming_turn()
    test_translate_recall_sample_builds_two_turn_pattern()
    test_render_trajectory_record_produces_compact_boundary_row()
    test_same_chunk_compress_precedes_trigger_chunk_response()
    test_render_questions_partitioned_by_segment()
    test_render_handles_video_with_no_compress()
    test_render_messages_are_json_serializable()
    test_inherited_queries_carry_across_compress_boundary()
    test_inherited_responses_carry_for_multi_event_query()
    test_resolved_query_not_inherited()
    test_render_empty_record_returns_empty()
    print("\n✅ all pass5_splitter tests passed")
