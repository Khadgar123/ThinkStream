from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    _safe_max_tokens_for_pass2,
    build_observation_request,
    build_observation_repair_request,
    should_repair_observation,
)
from scripts.agent_data_v5.config import FRAMES_PER_CHUNK, VISUAL_TOKENS_PER_FRAME_RUNTIME
from scripts.agent_data_v5.audit_pass2_stale import audit_rollouts
from scripts.agent_data_v5.cache_version import STAGE_VERSIONS


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


def test_pass2_repair_request_uses_only_current_chunk_frames(tmp_path):
    frame_paths = []
    for idx in range(FRAMES_PER_CHUNK * 5):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        frame_paths.append(str(path))

    memory = MemoryState()
    for c in range(4):
        memory.add_think(c, "The hand continues holding four paint tubes.")

    req = build_observation_repair_request(
        chunk_idx=4,
        frame_paths=frame_paths,
        memory=memory,
        video_id="vid_test",
        stale_text="The hand continues holding four paint tubes.",
    )

    content = req["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "video_url"]
    meta = req["media_io_kwargs"]["video"]
    assert meta["frames_indices"] == [FRAMES_PER_CHUNK * 4, FRAMES_PER_CHUNK * 4 + 1]
    assert meta["total_num_frames"] == FRAMES_PER_CHUNK * 5
    assert meta["do_sample_frames"] is False


def test_should_repair_observation_uses_evidence_drift():
    memory = MemoryState()
    stale = "The hand continues holding four paint tubes over the canvas."
    for c in range(4):
        memory.add_think(c, stale)

    evidence = [
        {
            "visible_entities": [{"desc": "hand with four paint tubes", "action": "holding tubes"}],
            "atomic_facts": ["A hand holds four paint tubes."],
        },
        {
            "visible_entities": [{"desc": "hand with four paint tubes", "action": "holding tubes"}],
            "atomic_facts": ["A hand holds four paint tubes."],
        },
        {
            "visible_entities": [{"desc": "hand with four paint tubes", "action": "holding tubes"}],
            "atomic_facts": ["A hand holds four paint tubes."],
        },
        {
            "visible_entities": [{"desc": "hand with four paint tubes", "action": "holding tubes"}],
            "atomic_facts": ["A hand holds four paint tubes."],
        },
        {
            "visible_entities": [{"desc": "hand with mineral spirits bottle", "action": "tilting bottle"}],
            "atomic_facts": ["A hand tilts a bottle of Mineral Spirits over the canvas."],
        },
    ]

    should_repair, meta = should_repair_observation(
        stale,
        memory.recent_thinks,
        chunk_idx=4,
        evidence=evidence,
    )
    assert should_repair
    assert meta["reason"] == "exact_repeat_with_evidence_drift"


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


def test_pass2_cache_bump_invalidates_old_video_http_rollouts():
    assert STAGE_VERSIONS["2"] == "v12.16"
    # Downstream stages consume pass2 rollout text, so they must not reuse
    # v12.14/v12.15 placements/samples/final messages after pass2 changes.
    assert STAGE_VERSIONS["3b"] == "v12.18"
    assert STAGE_VERSIONS["3c"] == "v12.18"
    assert STAGE_VERSIONS["4"] == "v12.18"
    assert STAGE_VERSIONS["5"] == "v12.18"


def test_pass2_stale_audit_flags_repeated_thinks_when_evidence_changes():
    rollout = {
        "thinks": [
            {"think": "The hand continues holding four paint tubes over the canvas."}
            for _ in range(24)
        ],
        "compression_events": [],
    }
    evidence = [
        {
            "visible_entities": [
                {"desc": "hand holding paint tubes", "action": "holding tubes"}
            ],
            "atomic_facts": ["A hand holds paint tubes."],
        }
    ]
    for i in range(1, 24):
        evidence.append({
            "visible_entities": [
                {"desc": f"different tool {i}", "action": f"new action {i}"}
            ],
            "atomic_facts": [f"A different tool is used at second {i}."],
        })

    report = audit_rollouts({"vid": rollout}, {"vid": evidence}, min_run_chunks=24)
    assert report["totals"]["hard_stale_videos"] == 1
    assert report["top_stale_runs"][0]["range"] == [0, 23]
