import json
from pathlib import Path

from scripts.agent_data.pass5_messages import build_messages
from thinkstream.data.agent_protocol import (
    build_user_content,
    chunk_frame_filenames,
    chunk_frame_indices,
    format_queries_block,
    infer_video_metadata,
    resolve_chunk_frame_paths,
)
from thinkstream.models.agent_loop import MemoryState, build_single_step_messages


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


def test_zero_based_chunks_map_to_ffmpeg_one_based_filenames(tmp_path: Path):
    assert chunk_frame_indices(0, 2) == [0, 1]
    assert chunk_frame_filenames(0, 2) == [
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]
    assert chunk_frame_filenames(1, 2) == [
        "frame_000003.jpg",
        "frame_000004.jpg",
    ]

    meta = infer_video_metadata(
        ["frame_000001.jpg", "frame_000002.jpg"],
        prefer_path_indices=True,
    )
    assert meta["frames_indices"] == [0, 1]

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for name in ("frame_000001.jpg", "frame_000002.jpg"):
        (canonical / name).write_bytes(b"\xff\xd8\xff\xd9")
    assert [
        Path(p).name for p in resolve_chunk_frame_paths(canonical, 0, 2)
    ] == ["frame_000001.jpg", "frame_000002.jpg"]

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("frame_000000.jpg", "frame_000001.jpg"):
        (legacy / name).write_bytes(b"\xff\xd8\xff\xd9")
    assert [
        Path(p).name for p in resolve_chunk_frame_paths(legacy, 0, 2)
    ] == ["frame_000000.jpg", "frame_000001.jpg"]


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
    videos = [item for item in content if item.get("type") == "video"]
    assert len(videos) == 1
    assert [Path(p).name for p in videos[0]["video"]] == [
        "frame_000007.jpg",
        "frame_000008.jpg",
    ]
    assert videos[0]["video_metadata"]["frames_indices"] == [6, 7]


def test_recall_result_is_metadata_only_in_shared_renderer():
    content = build_user_content(
        "",
        chunk_idx=12,
        video_path="/unused.mp4",
        queries=[{
            "question": "What text was on the sign?",
            "answer_form": "open",
            "answer_instruction": "Answer format: a concise exact phrase.",
            "ask_time": 12,
            "answers": [],
        }],
        recalled_frames={
            "time_range": [4, 6],
            "source": "historical_frames",
            "n_frames": 4,
            "frame_paths": [f"frame_{i:06d}.jpg" for i in range(9, 13)],
        },
        recall_result={
            "source": "historical_frames",
            "time": "4-6",
            "text_content": "leaking textual answer",
            "text": "another leaking textual answer",
            "returned_chunks": [4, 5],
        },
        frame_paths=[f"frame_{i:06d}.jpg" for i in range(17, 25)],
        frame_protocol="video_meta",
        render_layout="standard_query_last",
    )
    text = _join_user_text(content)
    assert "<recalled_frames>" in text
    assert "<recall_result>" in text
    assert "returned_chunks" in text
    assert "leaking textual answer" not in text
    assert '"text"' not in text


def test_runtime_post_recall_has_no_current_visual_window():
    messages = build_single_step_messages(
        {"compressed_segments": [], "recent_thinks": []},
        12,
        "/unused.mp4",
        recalled_frames={
            "time_range": [4, 6],
            "source": "historical_frames",
            "n_frames": 4,
            "frame_paths": [f"frame_{i:06d}.jpg" for i in range(9, 13)],
        },
        recall_result={
            "source": "historical_frames",
            "time": "4-6",
            "text_content": "leaking textual answer",
            "returned_chunks": [4, 5],
        },
        frame_protocol="video_meta",
        render_layout="standard_query_last",
    )
    system_text = messages[0]["content"][0]["text"]
    user_text = _join_user_text(messages[1]["content"])
    assert "streaming video assistant" in system_text.lower()
    assert "<visual_window>" not in user_text
    assert "<active_query>" not in user_text
    assert "<recalled_frames>" in user_text
    assert "<recall_result>" in user_text
    assert "leaking textual answer" not in user_text


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


def test_pass5_answer_contract_rejects_wrong_mc_response():
    from scripts.agent_data.pass5_messages import (
        QueryRenderContractError,
        validate_answer_render_contract,
    )

    sample = {
        "sample_id": "bad-mcq",
        "sample_type": "response",
        "action": "response",
        "chunk_idx": 5,
        "metadata": {
            "question": "Which color?",
            "answer_form": "multiple_choice",
            "answer_style": "letter_only",
            "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
            "options": ["A) red", "B) blue", "C) green", "D) yellow"],
            "correct_option": "B",
            "gold_answer": "blue",
            "canonical_answer": "blue",
        },
    }
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "<think>x</think><answer>A</answer>"}]},
    ]
    try:
        validate_answer_render_contract(sample, messages)
    except QueryRenderContractError:
        pass
    else:
        raise AssertionError("wrong MC response should fail pass5 answer contract")


def test_pass5_query_contract_uses_e_option_in_answer_format():
    from scripts.agent_data.pass5_messages import validate_query_render_contract
    from thinkstream.data.agent_protocol import format_queries_block

    q = {
        "question": "Which object appears?",
        "ask_time": 3,
        "status": "open",
        "options": ["A) brush", "B) spoon", "C) cup", "D) book", "E) plate"],
        "answer_form": "multiple_choice",
        "answer_style": "letter_only",
        "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
        "answers": [],
    }
    text = format_queries_block([q])
    assert "A, B, C, D, or E" in text
    validate_query_render_contract(
        {"sample_id": "mcq-e", "input": {"queries": [q]}},
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
