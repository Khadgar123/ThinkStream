from pathlib import Path

from scripts.agent_data_v5.pass1a_evidence import build_evidence_request, parse_evidence_result
from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    build_observation_request,
)
from scripts.agent_data_v5.pass5_messages import build_messages
from thinkstream.data.agent_protocol import (
    SYSTEM_PROMPT_V12,
    answer_format_instruction,
    build_user_content,
    format_memory_block,
    format_queries_block,
    parse_agent_output_v12,
    system_prompt_for_frame_protocol,
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


def test_runtime_build_user_content_defaults_to_video_metadata_protocol():
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
    assert types.count("video") == 1
    assert "image" not in types
    video = next(item for item in content if item["type"] == "video")
    assert video["video_metadata"]["frames_indices"] == [0, 1, 2, 3]


def test_runtime_build_user_content_can_use_timestamped_images_explicitly():
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
        frame_protocol="ts_image",
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


def test_runtime_build_user_content_can_use_video_metadata_protocol():
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
        frame_protocol="video_meta",
    )

    types = [item["type"] for item in content]
    assert types.count("video") == 1
    assert "image" not in types
    video = next(item for item in content if item["type"] == "video")
    assert video["video"] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
        "frame_000004.jpg",
    ]
    assert video["video_metadata"]["fps"] == 2.0
    assert video["video_metadata"]["frames_indices"] == [0, 1, 2, 3]
    assert video["video_metadata"]["total_num_frames"] == 4
    assert video["video_metadata"]["do_sample_frames"] is False


def test_pass5_sft_messages_use_video_metadata_protocol_by_default():
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
    assert types.count("video") == 1
    assert "image" not in types
    video = next(item for item in content if item.get("type") == "video")
    assert video["video_metadata"]["frames_indices"] == [0, 1, 2, 3]


def test_pass5_sft_messages_can_render_video_meta_protocol():
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
    messages = build_messages(sample, Path("/repo"), frame_protocol="video_meta")
    content = messages[1]["content"]

    assert "Qwen video block" in messages[0]["content"][0]["text"]
    assert [item["type"] for item in content].count("video") == 1
    video = next(item for item in content if item.get("type") == "video")
    assert video["video_metadata"]["frames_indices"] == [0, 1, 2, 3]
    assert video["video_metadata"]["do_sample_frames"] is False


def test_pass5_compress_messages_use_compress_system_prompt():
    sample = {
        "sample_id": "c0",
        "video_id": "vid0",
        "video_path": "videos/vid0.mp4",
        "sample_type": "compress",
        "v12_inter_chunk": True,
        "chunk_idx": 3,
        "input": {
            "user_input": "<compress_trigger/>",
            "memory": {"compressed_segments": [], "recent_thinks": []},
            "queries": [{
                "question": "What color is it?",
                "ask_time": 2,
                "answers": [],
            }],
            "visual_window": {
                "video_start": 0,
                "video_end": 4,
                "frames": 2,
                "frame_paths": [
                    "data/agent_v5/frames/vid0/frame_000007.jpg",
                    "data/agent_v5/frames/vid0/frame_000008.jpg",
                ],
            },
        },
        "output": (
            "<think>compress old memory</think>"
            "<tool_call>{\"name\":\"compress\",\"arguments\":{\"time_range\":[0,2],\"text\":\"setup\"}}</tool_call>"
        ),
    }
    messages = build_messages(sample, Path("/repo"))
    system_text = messages[0]["content"][0]["text"]
    user_text = "\n".join(
        item.get("text", "") for item in messages[1]["content"]
        if item.get("type") == "text"
    )
    assert "[MEMORY_MAINTENANCE / FORCED_COMPRESS]" in system_text
    assert "<queries>" not in user_text
    assert "<active_query>" not in user_text
    assert "<visual_window>" not in user_text
    assert not any(item.get("type") in {"image", "video"} for item in messages[1]["content"])
    assert "<user_input><compress_trigger/></user_input>" in user_text
    assert "<memory_compaction>" not in user_text

    no_visual = dict(sample)
    no_visual["input"] = dict(sample["input"])
    no_visual["input"].pop("visual_window")
    messages = build_messages(no_visual, Path("/repo"))
    user_text = "\n".join(
        item.get("text", "") for item in messages[1]["content"]
        if item.get("type") == "text"
    )
    assert "<visual_window>" not in user_text


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
    video = next(item for item in messages[1]["content"] if item.get("type") == "video")
    assert all(Path(p).exists() for p in video["video"])
    assert str(frame_dir / "frame_000004.jpg") in video["video"]


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
    assert "Qwen video block" in SYSTEM_PROMPT_V12
    assert "video_metadata carries frame timestamps" in SYSTEM_PROMPT_V12
    assert '<frame ts="12.5" role="latest chunk" />' not in SYSTEM_PROMPT_V12

    parsed = parse_agent_output_v12(
        '<think><frame ts="2.0" role="latest chunk" /> new brush appears</think>'
        '<answer>A</answer>'
    )
    assert parsed["think"] == "new brush appears"
    assert parsed["answer_text"] == "A"


def test_protocol_prompts_and_query_answer_format_are_explicit():
    ts_prompt = system_prompt_for_frame_protocol("ts_image")
    vm_prompt = system_prompt_for_frame_protocol("video_meta")
    compress_prompt = system_prompt_for_frame_protocol("ts_image", inter_chunk=True)
    assert "Each frame has a timestamp tag" in ts_prompt
    assert "Qwen video block" in vm_prompt
    assert "answer format" in ts_prompt
    assert "answer format" in vm_prompt
    assert "Recall arguments:" in ts_prompt
    assert "Silent: <answer></answer>" in ts_prompt
    assert "<active_query>" in ts_prompt
    assert "<response_history>" in ts_prompt
    assert "Output grammar:" in ts_prompt
    assert "{\"name\":\"recall\",\"arguments\":{\"query\"" in ts_prompt
    assert "<answer>response text</answer>" in ts_prompt
    assert "This is an ordinary streaming video QA turn" in ts_prompt
    assert "[MEMORY_MAINTENANCE / FORCED_COMPRESS]" in compress_prompt
    assert "No recall" in compress_prompt
    assert "Required output:" in compress_prompt
    assert "<tool_call>{\"name\":\"compress\"" in compress_prompt
    assert "\"time_range\":[start_sec,end_sec]" in compress_prompt
    assert "No answer. No silent answer." in compress_prompt

    assert answer_format_instruction("number") == (
        "Answer format: a number only, no explanation."
    )
    queries = [{
        "question": "How many cups are visible?",
        "ask_time": 3,
        "answer_form": "number",
        "answers": [],
    }]
    rendered = format_queries_block(queries)
    assert "<active_query>" in rendered
    assert "[3s] Answer format: a number only, no explanation." in rendered


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
    assert '"time": 8' in text
    assert '"text": "A red bowl appears on the counter."' in text


def test_inter_chunk_compress_omits_visual_and_queries_like_sft_messages():
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
    assert "<active_query>" not in joined
    assert "<visual_window>" not in joined
    assert not any(item.get("type") in {"image", "video"} for item in content)
    assert "<compress_trigger/>" in joined
    assert "<memory_compaction>" not in joined


def test_query_renderer_canonicalizes_mc_answer_instruction_from_options():
    from thinkstream.data.agent_protocol import format_queries_block
    text = format_queries_block([{
        "question": "Which object appears?",
        "ask_time": 3,
        "options": ["A) brush", "B) spoon", "C) cup", "D) book", "E) plate"],
        "answer_form": "multiple_choice",
        "answer_style": "letter_only",
        # Legacy stale text from an earlier four-option render pass. The
        # renderer must use structured options as the source of truth.
        "answer_instruction": "Answer format: one letter only (A, B, C, or D).",
        "answers": [],
    }])
    assert "<active_query>" in text
    assert "<response_history>" in text
    assert "Options: A) brush B) spoon C) cup D) book E) plate" in text
    assert "Answer format: one letter only (A, B, C, D, or E)." in text
    assert "Answer format: one letter only (A, B, C, or D)." not in text


def test_query_renderer_hides_closed_history_after_answer():
    from thinkstream.data.agent_protocol import format_queries_block
    text = format_queries_block([{
        "question": "Which object appears?",
        "ask_time": 3,
        "status": "answered",
        "answers": [{"time": 4, "text": "A"}],
    }])
    assert text == ""
