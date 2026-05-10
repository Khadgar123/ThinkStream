import json
from pathlib import Path

from scripts.agent_data_v5.pass5_messages import build_messages
from thinkstream.data.agent_protocol import build_user_content, format_queries_block
from thinkstream.model.agent_loop import MemoryState, build_single_step_messages


def _join_user_text(content):
    return "\n".join(
        item.get("text", "")
        for item in content
        if item.get("type") == "text"
    )


def _visual_window_payload(text: str) -> dict:
    start = text.index("<visual_window>") + len("<visual_window>")
    end = text.index("</visual_window>")
    return json.loads(text[start:end])


def test_shared_renderer_query_last_and_integer_times():
    query = {
        "question": "Which object is on the table?",
        "options": ["A) cup", "B) book", "C) phone", "D) key"],
        "answer_form": "multiple_choice",
        "answer_style": "letter_only",
        "ask_time": 3.0,
        "answer_chunks": [5],
        "answers": [],
    }
    content = build_user_content(
        "<memory_think>{\"time\":2,\"text\":\"setup\"}</memory_think>",
        chunk_idx=3,
        video_path="/unused.mp4",
        user_input=query["question"],
        queries=[query],
        frame_paths=[f"frame_{i:06d}.jpg" for i in range(1, 9)],
        frame_protocol="video_meta",
        render_layout="standard_query_last",
    )
    text = _join_user_text(content)
    assert "<user_input>" not in text
    assert text.index("<memory>") < text.index("<visual_window>")
    assert text.index("<visual_window>") < text.index("<active_query>")
    assert text.count("<active_query>") == 1
    assert text.count("Options:") == 1
    assert text.count("Answer format:") == 1
    vw = _visual_window_payload(text)
    assert isinstance(vw["start"], int)
    assert isinstance(vw["end"], int)
    assert isinstance(vw["current_time"], int)
    assert vw["current_time"] == 3


def test_pass5_and_runtime_builders_match_query_last_contract(tmp_path: Path):
    frame_dir = tmp_path / "data" / "agent_v5" / "frames" / "vid0"
    frame_dir.mkdir(parents=True)
    for i in range(8):
        (frame_dir / f"frame_{i + 1:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    query = {
        "question": "Is the cup visible?",
        "answer_form": "binary",
        "answer_instruction": "Answer format: a concise binary answer such as Yes or No.",
        "ask_time": 3.0,
        "answer_chunks": [5],
        "answers": [],
    }
    sample = {
        "sample_id": "s0",
        "video_id": "vid0",
        "video_path": "videos/vid0.mp4",
        "sample_type": "silent",
        "chunk_idx": 3,
        "input": {
            "user_input": "Is the cup visible?",
            "memory": {"compressed_segments": [], "recent_thinks": []},
            "queries": [query],
            "visual_window": {
                "video_start": 0.0,
                "video_end": 4.0,
                "frames": 8,
                "frame_paths": [
                    f"data/agent_v5/frames/vid0/frame_{i + 1:06d}.jpg"
                    for i in range(8)
                ],
            },
        },
        "output": "<think>x</think><answer></answer>",
    }
    pass5_messages = build_messages(
        sample,
        tmp_path,
        data_dir=tmp_path / "data" / "agent_v5",
        frame_protocol="video_meta",
        render_layout="standard_query_last",
    )
    runtime_messages = build_single_step_messages(
        {"compressed_segments": [], "recent_thinks": []},
        3,
        str(tmp_path / "videos" / "vid0.mp4"),
        user_input="Is the cup visible?",
        queries=[query],
        frame_paths=[
            str(frame_dir / f"frame_{i + 1:06d}.jpg") for i in range(8)
        ],
        frame_protocol="video_meta",
        render_layout="standard_query_last",
    )
    for messages in (pass5_messages, runtime_messages):
        user = next(m for m in messages if m["role"] == "user")
        text = _join_user_text(user["content"])
        assert "<user_input>" not in text
        assert text.index("<memory>") < text.index("<visual_window>")
        assert text.index("<visual_window>") < text.index("<active_query>")
        assert _visual_window_payload(text)["current_time"] == 3
        assert text.count("Answer format:") == 1


def test_query_lifecycle_records_early_on_time_and_late_answers():
    memory = MemoryState()
    memory.add_query(
        "Count objects",
        ask_time=3,
        answer_form="number",
        answer_chunks=[5, 8],
        per_emit_answers=[
            {"chunk": 5, "value": "1"},
            {"chunk": 8, "value": "2"},
        ],
        open_until=8,
    )
    memory.answer_query("Count objects", "0", response_time=4)
    memory.answer_query("Count objects", "1", response_time=5)
    memory.answer_query("Count objects", "2", response_time=9)
    q = memory.queries[-1]
    assert [a["timing"] for a in q["answers"]] == ["early", "on_time", "late"]
    assert [a["counts_for_completion"] for a in q["answers"]] == [False, True, True]
    assert [a["expected_chunk"] for a in q["answers"]] == [5, 5, 8]
    assert q["status"] == "answered"

    rendered = format_queries_block([q])
    assert rendered == ""
