from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    _safe_max_tokens_for_pass2,
    build_observation_request,
)
from scripts.agent_data_v5.config import FRAMES_PER_CHUNK, VISUAL_TOKENS_PER_FRAME_RUNTIME


def test_pass2_observation_uses_vllm_video_url_with_media_metadata(tmp_path):
    frame_paths = []
    for idx in range(FRAMES_PER_CHUNK * 3):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        frame_paths.append(str(path))

    req = build_observation_request(
        chunk_idx=2,
        frame_paths=frame_paths,
        memory=MemoryState(),
        video_id="vid_test",
    )

    content = req["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "video_url"]
    assert content[1]["video_url"]["url"].startswith("data:video/jpeg;base64,")
    assert '"type": "video"' not in str(content)

    video_meta = req["media_io_kwargs"]["video"]
    assert video_meta["frames_indices"] == list(range(FRAMES_PER_CHUNK * 3))
    assert video_meta["fps"] > 0
    assert video_meta["total_num_frames"] == FRAMES_PER_CHUNK * 3
    assert video_meta["do_sample_frames"] is False


def test_pass2_safe_token_estimate_counts_video_url_frames(tmp_path):
    frame_paths = []
    for idx in range(FRAMES_PER_CHUNK * 2):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        frame_paths.append(str(path))

    req = build_observation_request(
        chunk_idx=1,
        frame_paths=frame_paths,
        memory=MemoryState(),
        video_id="vid_test",
    )
    safe = _safe_max_tokens_for_pass2(req, configured_max=65536, floor=1)
    expected_vision = FRAMES_PER_CHUNK * 2 * VISUAL_TOKENS_PER_FRAME_RUNTIME
    assert safe <= 65536 - expected_vision
