from scripts.agent_data_v5.pass2_rollout import (
    MemoryState,
    _is_repair_better,
    _safe_max_tokens_for_pass2,
    build_observation_request,
    build_observation_repair_request,
    should_repair_observation,
)
from scripts.agent_data_v5.config import FRAMES_PER_CHUNK, VISUAL_TOKENS_PER_FRAME_RUNTIME
from scripts.agent_data_v5.audit_pass2_stale import audit_rollouts
from scripts.agent_data_v5.cache_version import STAGE_VERSIONS


def test_pass2_observation_uses_timestamped_image_window(tmp_path):
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
    assert [item["type"] for item in content] == [
        "text",
        *sum((["text", "image_url"] for _ in range(FRAMES_PER_CHUNK * 3)), []),
    ]
    prompt = content[0]["text"]
    assert "CURRENT TASK FIRST" in prompt
    assert "timestamped image list" in prompt
    assert f"({FRAMES_PER_CHUNK} frames)" in prompt
    assert "untrusted history for entity naming only" in prompt
    assert "only evidence for the current think" in prompt
    assert "older context to the latest chunk" in prompt
    assert content[1]["text"] == "Frame timestamp t=0.0s (older context)."
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[-2]["text"] == "Frame timestamp t=2.5s (latest chunk)."
    assert "media_io_kwargs" not in req


def test_pass2_repair_request_uses_only_current_chunk_timestamped_images(tmp_path):
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
    assert [item["type"] for item in content] == ["text", "text", "image_url", "text", "image_url"]
    assert content[1]["text"] == "Frame timestamp t=4.0s (latest chunk)."
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[3]["text"] == "Frame timestamp t=4.5s (latest chunk)."
    assert "media_io_kwargs" not in req


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


def test_should_repair_observation_flags_near_repeat_without_stale_words():
    memory = MemoryState()
    stale = (
        "The close-up view shows a person wearing blue jeans pressing a brown "
        "leather shoe against the rapidly spinning brush and roller."
    )
    for c in range(6):
        memory.add_think(c, stale)

    candidate = (
        "The close-up view captures a person wearing blue jeans pressing a brown "
        "leather shoe against the rapidly spinning brush and roller."
    )
    evidence = []
    for _ in range(6):
        evidence.append({
            "visible_entities": [{"desc": "brown leather shoe and brush", "action": "polishing"}],
            "atomic_facts": ["A shoe is pressed against a spinning brush."],
        })
    evidence.append({
        "visible_entities": [{"desc": "Heute Maschinenfabrik branding card", "action": "static"}],
        "atomic_facts": ["A Quality made in GERMANY badge and logo are displayed."],
    })

    should_repair, meta = should_repair_observation(
        candidate,
        memory.recent_thinks,
        chunk_idx=6,
        evidence=evidence,
    )
    assert should_repair
    assert meta["reason"] == "near_repeat_with_evidence_drift"


def test_repair_acceptance_rejects_still_repeated_text():
    memory = MemoryState()
    stale = (
        "The close-up view shows a person wearing blue jeans pressing a brown "
        "leather shoe against the rapidly spinning brush and roller."
    )
    for c in range(3):
        memory.add_think(c, stale)

    repeated = (
        "The close-up view captures a person wearing blue jeans pressing a brown "
        "leather shoe against the rapidly spinning brush and roller."
    )
    corrected = (
        "The latest frames show a static Heute Maschinenfabrik branding card "
        "with a Quality made in GERMANY badge, three footprint-and-gear icons, "
        "and a magenta vertical bar."
    )

    assert not _is_repair_better(repeated, stale, memory.recent_thinks)
    assert _is_repair_better(corrected, stale, memory.recent_thinks)


def test_pass2_safe_token_estimate_counts_timestamped_image_frames(tmp_path):
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
    assert STAGE_VERSIONS["1a"] == "v12.22"
    assert STAGE_VERSIONS["2"] == "v12.22"
    # Downstream stages must not reuse cached data after the project-wide
    # timestamped image-list protocol change.
    assert STAGE_VERSIONS["3b"] == "v12.22"
    assert STAGE_VERSIONS["3c"] == "v12.22"
    assert STAGE_VERSIONS["4"] == "v12.22"
    assert STAGE_VERSIONS["5"] == "v12.22"


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
