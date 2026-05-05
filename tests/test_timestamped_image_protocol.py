from pathlib import Path

from scripts.agent_data_v5.pass1a_evidence import build_evidence_request, parse_evidence_result
from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    build_observation_request,
)
from scripts.agent_data_v5.pass5_messages import build_messages
from thinkstream.data.agent_protocol import (
    SYSTEM_PROMPT_V12,
    build_user_content,
    format_memory_block,
    parse_agent_output_v12,
)


def _jpeg_frames(tmp_path: Path, n: int):
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i + 1:06d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")
        paths.append(str(p))
    return paths


def _frame_timestamp_texts(content):
    return [
        item.get("text", "")
        for item in content
        if item.get("type") == "text"
        and item.get("text", "").startswith("<frame ")
    ]


def test_runtime_build_user_content_uses_timestamped_images():
    content = build_user_content(
        memory_text="",
        chunk_idx=1,
        video_path="/unused.mp4",
        frame_paths=[
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
            "frame_000004.jpg",
        ],
    )

    types = [item["type"] for item in content]
    assert "video" not in types
    assert types.count("image") == 4
    assert _frame_timestamp_texts(content) == [
        '<frame ts="0.0" role="older context" />',
        '<frame ts="0.5" role="older context" />',
        '<frame ts="1.0" role="latest chunk" />',
        '<frame ts="1.5" role="latest chunk" />',
    ]


def test_pass5_sft_messages_use_same_timestamped_image_protocol():
    sample = {
        "sample_id": "s0",
        "video_id": "vid0",
        "video_path": "videos/vid0.mp4",
        "sample_type": "silent",
        "chunk_idx": 1,
        "input": {
            "memory": {"compressed_segments": [], "recent_thinks": []},
            "visual_window": {
                "video_start": 0,
                "video_end": 2,
                "frames": 4,
                "frame_paths": [
                    "data/agent_v5/frames/vid0/frame_000001.jpg",
                    "data/agent_v5/frames/vid0/frame_000002.jpg",
                    "data/agent_v5/frames/vid0/frame_000003.jpg",
                    "data/agent_v5/frames/vid0/frame_000004.jpg",
                ],
            },
        },
        "output": "<think>x</think><answer></answer>",
    }
    messages = build_messages(sample, Path("/repo"))
    content = messages[1]["content"]

    types = [item["type"] for item in content]
    assert "video" not in types
    assert types.count("image") == 4
    assert _frame_timestamp_texts(content)[-1] == (
        '<frame ts="1.5" role="latest chunk" />'
    )


def test_pass5_relocates_moved_absolute_frame_paths(tmp_path):
    frame_dir = tmp_path / "frames" / "vid0"
    frame_dir.mkdir(parents=True)
    for idx in range(4):
        (frame_dir / f"frame_{idx + 1:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    old_paths = [
        f"/old/cluster/root/frames/vid0/frame_{idx + 1:06d}.jpg"
        for idx in range(4)
    ]
    sample = {
        "sample_id": "s0",
        "video_id": "vid0",
        "video_path": "videos/vid0.mp4",
        "sample_type": "silent",
        "chunk_idx": 1,
        "input": {
            "memory": {"compressed_segments": [], "recent_thinks": []},
            "visual_window": {
                "video_start": 0,
                "video_end": 2,
                "frames": 4,
                "frame_paths": old_paths,
            },
        },
        "output": "<think>x</think><answer></answer>",
    }
    messages = build_messages(sample, tmp_path)
    image_paths = [
        item["image"]
        for item in messages[1]["content"]
        if item.get("type") == "image"
    ]
    assert all(Path(p).exists() for p in image_paths)
    assert str(frame_dir / "frame_000004.jpg") in image_paths


def test_teacher_passes_use_timestamped_image_url_protocol(tmp_path):
    frames = _jpeg_frames(tmp_path, 6)

    p1 = build_evidence_request(chunk_idx=1, frame_paths=frames, video_id="vid0")
    p1_content = p1["messages"][0]["content"]
    assert "video_url" not in [item["type"] for item in p1_content]
    assert [item["type"] for item in p1_content].count("image_url") == 2
    assert _frame_timestamp_texts(p1_content) == [
        '<frame ts="1.0" role="current chunk" />',
        '<frame ts="1.5" role="current chunk" />',
    ]

    p2 = build_observation_request(
        chunk_idx=2,
        frame_paths=frames,
        memory=MemoryState(),
        video_id="vid0",
    )
    p2_content = p2["messages"][0]["content"]
    assert "video_url" not in [item["type"] for item in p2_content]
    assert [item["type"] for item in p2_content].count("image_url") == 6
    assert _frame_timestamp_texts(p2_content)[-2:] == [
        '<frame ts="2.0" role="latest chunk" />',
        '<frame ts="2.5" role="latest chunk" />',
    ]
    assert "media_io_kwargs" not in p2


def test_runtime_prompt_and_parser_use_frame_tags():
    assert '<frame ts="12.5" role="latest chunk" />' in SYSTEM_PROMPT_V12
    assert "never copy" in SYSTEM_PROMPT_V12.lower()

    parsed = parse_agent_output_v12(
        '<think><frame ts="2.0" role="latest chunk" /> new brush appears</think>'
        '<answer>A</answer>'
    )
    assert parsed["think"] == "new brush appears"
    assert parsed["answer_text"] == "A"


def test_pass1a_parser_requires_observation_note_think_field():
    meta = {"time": [0, 1]}
    parsed = parse_evidence_result(
        '{"time":[0,1],"visible_entities":[{"desc":"red bowl","action":"static"}],'
        '"atomic_facts":["a red bowl is on the counter"],"ocr":[],"spatial":"",'
        '"think":"The current frames show a red bowl resting on the counter."}',
        meta,
    )
    assert parsed["parse_success"] is True
    assert parsed["think"].startswith("The current frames show")

    missing = parse_evidence_result(
        '{"time":[0,1],"visible_entities":[{"desc":"red bowl","action":"static"}],'
        '"atomic_facts":["a red bowl is on the counter"],"ocr":[],"spatial":""}',
        meta,
    )
    assert missing["parse_success"] is False
    assert missing["_missing_think"] is True


def test_memory_recent_thinks_are_tagged_json_records():
    text = format_memory_block({
        "compressed": [{"time_range": [0, 8], "text": "Earlier setup."}],
        "recent_thinks": ["[8-9] A red bowl appears on the counter."],
    })
    assert "<compressed>{" in text
    assert "<memory_think>{" in text
    assert '"time": "8-9"' in text
    assert '"text": "A red bowl appears on the counter."' in text


def test_inter_chunk_compress_omits_queries_like_sft_messages():
    content = build_user_content(
        memory_text='<memory_think>{"time":"0-1","text":"setup"}</memory_think>',
        chunk_idx=5,
        video_path="/unused.mp4",
        user_input="<compress_trigger/>",
        queries=[{
            "question": "What color is the bowl?",
            "ask_time": 4,
            "answer_form": "multiple_choice",
            "options": ["A) red", "B) blue", "C) green", "D) black"],
            "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
            "answers": [],
        }],
        frame_paths=["frame_000001.jpg", "frame_000002.jpg"],
        inter_chunk=True,
    )
    joined = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    assert "<queries>" not in joined
    assert "<visual_window>" not in joined
    assert "<compress_trigger/>" in joined


def test_query_renderer_keeps_mc_answer_instruction():
    from thinkstream.data.agent_protocol import format_queries_block
    text = format_queries_block([{
        "question": "Which object appears?",
        "ask_time": 3,
        "options": ["A) brush", "B) spoon", "C) cup", "D) book"],
        "answer_form": "multiple_choice",
        "answer_style": "letter_only",
        "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
        "answers": [],
    }])
    assert "Options: A) brush B) spoon C) cup D) book" in text
    assert "Answer format: one letter only" in text
