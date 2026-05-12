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
    MEMORY_OPEN,
    STAGE_COMPRESS_MARKER,
    TOOL_NAME_COMPRESS,
    TOOL_NAME_RECALL,
    TRAJ_TYPE_FROM_COMPRESS,
    TRAJ_TYPE_FROM_START,
)


def _dummy_frame_resolver(chunk_idx: int):
    return [f"frame_{chunk_idx}_0.jpg", f"frame_{chunk_idx}_1.jpg"]


# ============== parse_assistant_output_to_spec ==============

def test_parse_silent_empty_answer():
    spec = parse_assistant_output_to_spec(
        "<think>nothing yet</think><answer></answer>", chunk_idx=0,
    )
    assert spec.action_type == ACTION_SILENT
    assert spec.think == "nothing yet"
    print("[OK] parse_silent_empty_answer")


def test_parse_response_with_answer():
    spec = parse_assistant_output_to_spec(
        "<think>light is green</think><answer>The light turned green!</answer>",
        chunk_idx=5,
    )
    assert spec.action_type == ACTION_RESPONSE
    assert spec.response_text == "The light turned green!"
    print("[OK] parse_response_with_answer")


def test_parse_compress_tool_call():
    text = (
        "<think>compressing 0-32</think>"
        '<tool_call>{"name": "compress", "arguments": '
        '{"time_range": [0, 32], "text": "Light red throughout."}}</tool_call>'
    )
    spec = parse_assistant_output_to_spec(text, chunk_idx=32)
    # Pass3-format compress is mapped to the canonical Step-1 ACTION_COMPRESS_SELECT.
    # The gold summary text rides on a private ``_gold_summary_text`` field
    # so the splitter can emit the Step-2 <m> block.
    assert spec.action_type == ACTION_COMPRESS_SELECT
    assert spec.tool_call_id == "comp_32"
    assert spec.tool_arguments["time_range"] == [0, 32]
    assert spec.tool_arguments["_gold_summary_text"] == "Light red throughout."
    print("[OK] parse_compress_tool_call")


def test_parse_recall_tool_call():
    text = (
        "<think>need history</think>"
        '<tool_call>{"name": "recall", "arguments": '
        '{"time_range": [0, 10], "query": "first red light"}}</tool_call>'
    )
    spec = parse_assistant_output_to_spec(text, chunk_idx=15)
    assert spec.action_type == ACTION_RECALL
    assert spec.tool_call_id == "rec_15"
    assert spec.tool_arguments["query"] == "first red light"
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
    turn = translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    assert turn.user.stage_marker == STAGE_COMPRESS_MARKER
    assert turn.user.frame_paths == []
    print("[OK] legacy_v12_inter_chunk_is_compress_boundary")


def test_parse_malformed_tool_call_falls_back_to_silent():
    text = "<think>x</think><tool_call>not json</tool_call>"
    spec = parse_assistant_output_to_spec(text, chunk_idx=0)
    assert spec.action_type == ACTION_SILENT
    print("[OK] parse_malformed_tool_call_falls_back_to_silent")


def test_parse_no_think_section():
    spec = parse_assistant_output_to_spec("<answer>hello</answer>", chunk_idx=0)
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
    assert by_chunk[5][0].answer_format == "letter only"
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

def test_extract_compress_summary_from_gold_caption():
    sample = {
        "gold_caption": "Light red throughout 0-32.",
        "gold_compress_chunks": [0, 1, 2, 30, 31, 32],
        "output": "ignored",
    }
    text, chunks = extract_compress_summary(sample)
    assert text == "Light red throughout 0-32."
    assert chunks == [0, 1, 2, 30, 31, 32]
    print("[OK] extract_compress_summary_from_gold_caption")


def test_extract_compress_summary_from_tool_call_fallback():
    sample = {
        "output": (
            "<think>x</think>"
            '<tool_call>{"name": "compress", "arguments": '
            '{"time_range": [0, 16], "text": "fallback summary"}}</tool_call>'
        ),
    }
    text, chunks = extract_compress_summary(sample)
    assert text == "fallback summary"
    assert chunks == list(range(0, 16))
    print("[OK] extract_compress_summary_from_tool_call_fallback")


# ============== translate_sample_to_turn ==============

def test_translate_silent_sample_renders_video_and_no_query():
    sample = {
        "chunk_idx": 7,
        "sample_type": "silent",
        "output": "<think>walking</think><answer></answer>",
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
        "output": "<think>green!</think><answer>A</answer>",
    }
    questions = {12: [QuerySpec(text="What color?", options=["A", "B"])]}
    turn = translate_sample_to_turn(sample, questions, _dummy_frame_resolver)
    assert turn.assistant.action_type == ACTION_RESPONSE
    assert turn.assistant.response_text == "A"
    assert turn.user.active_query is not None
    assert turn.user.active_query.text == "What color?"
    print("[OK] translate_response_sample_with_query")


def test_translate_compress_sample_drops_video_and_injects_stage():
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
    turn = translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    assert turn.user.stage_marker == STAGE_COMPRESS_MARKER
    assert turn.user.frame_paths == []  # no video on compress
    # The assistant for a compress sample is the Step-1 select tool_call;
    # the Step-2 <m>-summary turn is built later by
    # render_trajectory_record_to_rows when it has access to memory state.
    assert turn.assistant.action_type == ACTION_COMPRESS_SELECT
    assert turn.assistant.tool_arguments["time_range"] == [0, 32]
    print("[OK] translate_compress_sample_drops_video_and_injects_stage")


def test_translate_recall_sample_builds_two_turn_pattern():
    sample = {
        "chunk_idx": 15,
        "sample_type": "recall",
        "v12_assistant_turn_1": (
            "<think>need history</think>"
            '<tool_call>{"name": "recall", "arguments": '
            '{"time_range": [0, 5], "query": "first red"}}</tool_call>'
        ),
        "v12_assistant_turn_2": "<think>got it</think><answer>2s</answer>",
        "recall_result": {
            "time_range": [0, 5],
            "n_frames": 2,
            "text": "Light first red at t=2s.",
        },
    }
    turn = translate_sample_to_turn(sample, {}, _dummy_frame_resolver)
    assert turn.assistant.action_type == ACTION_RECALL
    assert turn.tool_response is not None
    assert turn.tool_response["role"] == "tool"
    # tool_response content carries the recall preview text
    body = turn.tool_response["content"]
    assert "Light first red" in body
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
                "output": "<think>start</think><answer></answer>",
            },
            {
                "chunk_idx": 1,
                "sample_type": "silent",
                "output": "<think>quiet</think><answer></answer>",
            },
            {
                "chunk_idx": 4,
                "sample_type": "response",
                "output": "<think>I see red</think><answer>A</answer>",
            },
            {
                "chunk_idx": 5,
                "sample_type": "silent",
                "output": "<think>moving on</think><answer></answer>",
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
                "output": "<think>green now</think><answer>33</answer>",
            },
            {
                "chunk_idx": 34,
                "sample_type": "silent",
                "output": "<think>x</think><answer></answer>",
            },
        ],
    }


def test_render_trajectory_record_produces_two_segments():
    record = _make_pass4_record_with_compress()
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)

    assert len(rows) == 2, f"expected 2 segments, got {len(rows)}"

    # Segment 0: from_start, chunks 0..32 (ending in compress)
    seg0 = rows[0]
    assert seg0["trajectory_type"] == TRAJ_TYPE_FROM_START
    assert seg0["chunk_start"] == 0
    assert seg0["chunk_end"] == 32
    assert seg0["compress_event"] is not None
    assert seg0["compress_event"]["chunk_idx"] == 32
    assert seg0["compress_event"]["summary_text"] == "Light red 0-5, person walked."

    # Segment 1: from_compress, chunks 33..34, prefix memory present
    seg1 = rows[1]
    assert seg1["trajectory_type"] == TRAJ_TYPE_FROM_COMPRESS
    assert seg1["chunk_start"] == 33
    assert seg1["chunk_end"] == 34
    assert seg1["compress_event"] is None

    # Segment 1's first user message must carry the memory block
    first_user_msg = next(m for m in seg1["messages"] if m["role"] == "user")
    first_user_text_blocks = [
        c for c in first_user_msg["content"] if c.get("type") == "text"
    ]
    assert any(MEMORY_OPEN in c["text"] for c in first_user_text_blocks), (
        "from_compress segment's first user message should contain a memory block"
    )
    print("[OK] render_trajectory_record_produces_two_segments")


def test_render_questions_partitioned_by_segment():
    record = _make_pass4_record_with_compress()
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)

    seg0_qs = rows[0]["questions_in_segment"]
    seg1_qs = rows[1]["questions_in_segment"]

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
             "output": "<think>x</think><answer></answer>"},
            {"chunk_idx": 1, "sample_type": "silent",
             "output": "<think>x</think><answer></answer>"},
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

    Without this, the SFT row emits a gold <response>X</response> at some
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
                 "output": f"<think>obs{c}</think><answer></answer>"}
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
                 "output": "<think>scanning</think><answer></answer>"},
                {"chunk_idx": 34, "sample_type": "silent",
                 "output": "<think>scanning</think><answer></answer>"},
                {"chunk_idx": 35, "sample_type": "response",
                 "output": "<think>re-entered</think><answer>A</answer>"},
            ]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    assert len(rows) == 2
    seg1 = rows[1]
    assert seg1["trajectory_type"] == TRAJ_TYPE_FROM_COMPRESS
    first_user = next(m for m in seg1["messages"] if m["role"] == "user")
    texts = [c["text"] for c in first_user["content"] if c.get("type") == "text"]
    joined = "\n".join(texts)
    assert "<query>" in joined, "open query should appear in from_compress header"
    assert "Did the person re-enter?" in joined
    assert '<q t="30">' in joined, "should preserve original ask_chunk"
    # No prior response: chunk 35 is the only answer slot and it's in seg 1.
    assert "<response>" not in joined
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
                 "output": f"<think>q{c}</think><answer></answer>"}
                for c in range(10)
            ]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>one person</think><answer>1</answer>"}]
            + [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": f"<think>obs{c}</think><answer></answer>"}
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
                 "output": f"<think>scan{c}</think><answer></answer>"}
                for c in range(33, 35)
            ]
            + [{"chunk_idx": 35, "sample_type": "response",
                "output": "<think>second person</think><answer>2</answer>"}]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    seg1 = rows[1]
    first_user = next(m for m in seg1["messages"] if m["role"] == "user")
    joined = "\n".join(
        c["text"] for c in first_user["content"] if c.get("type") == "text"
    )
    assert "<query>" in joined and 'How many people' in joined
    assert "<response>" in joined, "prior chunk-10 response must be inherited"
    assert '<r t="10">1</r>' in joined
    print("[OK] inherited_responses_carry_for_multi_event_query")


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
                 "output": "<think>x</think><answer></answer>"}
                for c in range(10)
            ]
            + [{"chunk_idx": 10, "sample_type": "response",
                "output": "<think>done</think><answer>done</answer>"}]
            + [
                {"chunk_idx": c, "sample_type": "silent",
                 "output": "<think>x</think><answer></answer>"}
                for c in range(11, 32)
            ]
            + [{"chunk_idx": 32, "sample_type": "compress", "inter_chunk": True,
                "gold_caption": "0-30", "output": (
                    "<think>compress</think>"
                    '<tool_call>{"name":"compress","arguments":'
                    '{"time_range":[0,30],"text":"0-30"}}</tool_call>')}]
            + [{"chunk_idx": 33, "sample_type": "silent",
                "output": "<think>x</think><answer></answer>"}]
        ),
    }
    rows = render_trajectory_record_to_rows(record, _dummy_frame_resolver)
    seg1 = rows[1]
    first_user = next(m for m in seg1["messages"] if m["role"] == "user")
    joined = "\n".join(
        c["text"] for c in first_user["content"] if c.get("type") == "text"
    )
    assert "Closed before boundary" not in joined
    assert "<query>" not in joined
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
    test_parse_compress_tool_call()
    test_parse_recall_tool_call()
    test_parse_malformed_tool_call_falls_back_to_silent()
    test_parse_no_think_section()
    test_build_questions_by_chunk_filters_missing_ask()
    test_build_questions_by_chunk_preserves_multi_question_per_chunk()
    test_is_compress_sample_both_flags()
    test_split_samples_by_compress_single_segment()
    test_split_samples_by_compress_multi_segment()
    test_split_samples_compress_at_end_no_trailing_empty()
    test_extract_compress_summary_from_gold_caption()
    test_extract_compress_summary_from_tool_call_fallback()
    test_translate_silent_sample_renders_video_and_no_query()
    test_translate_response_sample_with_query()
    test_translate_compress_sample_drops_video_and_injects_stage()
    test_translate_recall_sample_builds_two_turn_pattern()
    test_render_trajectory_record_produces_two_segments()
    test_render_questions_partitioned_by_segment()
    test_render_handles_video_with_no_compress()
    test_render_messages_are_json_serializable()
    test_inherited_queries_carry_across_compress_boundary()
    test_inherited_responses_carry_for_multi_event_query()
    test_resolved_query_not_inherited()
    test_render_empty_record_returns_empty()
    print("\n✅ all pass5_splitter tests passed")
