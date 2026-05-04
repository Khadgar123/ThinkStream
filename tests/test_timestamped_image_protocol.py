from pathlib import Path

from scripts.agent_data_v5.pass1a_evidence import build_evidence_request
from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    build_observation_request,
)
from scripts.agent_data_v5.pass5_messages import build_messages
from thinkstream.data.agent_protocol import build_user_content


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
        and item.get("text", "").startswith("Frame timestamp")
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
        "Frame timestamp t=0.0s (older context).",
        "Frame timestamp t=0.5s (older context).",
        "Frame timestamp t=1.0s (latest chunk).",
        "Frame timestamp t=1.5s (latest chunk).",
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
        "Frame timestamp t=1.5s (latest chunk)."
    )


def test_teacher_passes_use_timestamped_image_url_protocol(tmp_path):
    frames = _jpeg_frames(tmp_path, 6)

    p1 = build_evidence_request(chunk_idx=1, frame_paths=frames, video_id="vid0")
    p1_content = p1["messages"][0]["content"]
    assert "video_url" not in [item["type"] for item in p1_content]
    assert [item["type"] for item in p1_content].count("image_url") == 2
    assert _frame_timestamp_texts(p1_content) == [
        "Frame timestamp t=1.0s (current chunk).",
        "Frame timestamp t=1.5s (current chunk).",
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
        "Frame timestamp t=2.0s (latest chunk).",
        "Frame timestamp t=2.5s (latest chunk).",
    ]
    assert "media_io_kwargs" not in p2
