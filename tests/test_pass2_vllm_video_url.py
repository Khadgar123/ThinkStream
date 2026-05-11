from scripts.agent_data.pass2_rollout import (
    MemoryState,
    _is_repair_better,
    _safe_max_tokens_for_pass2,
    build_observation_request,
    build_observation_repair_request,
    parse_observation_result,
    run_pass2_single_video,
    should_repair_observation,
)
from scripts.agent_data.config import FRAMES_PER_CHUNK, VISUAL_TOKENS_PER_FRAME_RUNTIME
from scripts.agent_data.audit_pass2_stale import audit_rollouts
from scripts.agent_data.cache_version import STAGE_VERSIONS


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
    assert "timestamp-tagged image list" in prompt
    assert f"({FRAMES_PER_CHUNK} frames)" in prompt
    assert "History ledger below is archival memory for naming only" in prompt
    assert "only evidence for the current think" in prompt
    assert "Never copy or paraphrase any frame tag" in prompt
    assert "older context to the latest chunk" in prompt
    assert "<history_ledger>" in prompt
    assert content[1]["text"] == '<frame ts="0.0" role="older context" />'
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[-2]["text"] == '<frame ts="2.5" role="latest chunk" />'
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
    assert content[1]["text"] == '<frame ts="4.0" role="latest chunk" />'
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[3]["text"] == '<frame ts="4.5" role="latest chunk" />'
    assert "media_io_kwargs" not in req


def test_parse_observation_result_strips_frame_tags():
    raw = (
        '<frame ts="12.0" role="latest chunk" />\n'
        '<frame ts="12.5" role="latest chunk" />\n'
        "A woman in a red apron stirs food in a silver pan."
    )
    assert parse_observation_result(raw) == "A woman in a red apron stirs food in a silver pan."


def test_parse_observation_result_strips_inline_frame_tags():
    raw = (
        '<frame ts="12.0" role="latest chunk" /> '
        '<frame ts="12.5" role="latest chunk" /> '
        "A white bowl rests on a wooden counter."
    )
    assert parse_observation_result(raw) == "A white bowl rests on a wooden counter."


def test_memory_observation_prompt_uses_structured_history_ledger():
    memory = MemoryState()
    memory.add_think(0, "A woman in a red apron stands near a stove.")
    memory.add_think(1, "The red-apron woman lifts a silver pan.")
    memory.compress(
        {"time_range": [0, 1], "text": "A red-apron cook appears by the stove and lifts a silver pan."},
        selected_indices=[0, 1],
    )
    memory.add_think(2, "A white bowl sits on the wooden counter.")

    text = memory.format_for_observation_prompt()
    assert "<compressed>{" in text
    assert "<memory_think>{" in text
    assert '"time_range": [0, 1]' in text
    assert '"time": "2-3"' in text
    assert '[2-3]' not in text


def test_memory_repair_prompt_uses_structured_recent_history():
    memory = MemoryState()
    for c in range(3):
        memory.add_think(c, f"Think {c}")
    text = memory.format_recent_for_repair_prompt(limit=2)
    assert '<memory_think>{"time": "1-2", "text": "Think 1"}</memory_think>' in text
    assert '<memory_think>{"time": "2-3", "text": "Think 2"}</memory_think>' in text
    assert '"time": "0-1"' not in text


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


def test_should_repair_observation_skips_static_visual_delta():
    memory = MemoryState()
    stale = "The title card remains unchanged with the same centered text."
    for c in range(4):
        memory.add_think(c, stale)

    should_repair, meta = should_repair_observation(
        stale,
        memory.recent_thinks,
        chunk_idx=4,
        evidence=[
            {"visible_entities": [{"desc": "title card", "action": "static"}], "atomic_facts": ["A title card is visible."]}
            for _ in range(5)
        ],
        visual_delta_mse=0.0,
    )
    assert not should_repair
    assert meta["reason"] == "static_visual_delta_skip"


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
    assert STAGE_VERSIONS["1a"] == "v12.25"
    assert STAGE_VERSIONS["1b"] == "v12.25"
    assert STAGE_VERSIONS["2"] == "v12.25"
    # Downstream stages must not reuse cached samples after pass3 changed
    # recall_silent into a non-terminal not_yet wait state.
    assert STAGE_VERSIONS["3b"] == "v12.28"
    assert STAGE_VERSIONS["3c"] == "v12.28"
    assert STAGE_VERSIONS["4"] == "v12.28"
    assert STAGE_VERSIONS["5"] == "v12.28"


def test_pass2_uses_pass1_observation_note_without_observation_call():
    class NoCallClient:
        async def _call_one(self, **_kwargs):
            raise AssertionError("pass2 observation should not call teacher")

    evidence = [
        {
            "chunk_idx": 0,
            "think": "The current frames show a red bowl on the counter.",
            "visible_entities": [{"desc": "red bowl", "action": "static"}],
            "atomic_facts": [{"fact": "a red bowl is on the counter"}],
        },
        {
            "chunk_idx": 1,
            "think": "The current frames show a spoon inside the red bowl.",
            "visible_entities": [{"desc": "spoon", "action": "static"}],
            "atomic_facts": [{"fact": "a spoon is inside the bowl"}],
        },
    ]

    import asyncio

    rollout = asyncio.run(run_pass2_single_video(
        video_id="vid_pass1_think",
        frame_paths=[],
        num_chunks=2,
        client=NoCallClient(),
        evidence=evidence,
    ))

    assert [t["source"] for t in rollout["thinks"]] == [
        "pass1_observation_note",
        "pass1_observation_note",
    ]
    assert rollout["thinks"][1]["think"] == evidence[1]["think"]
    assert rollout["snapshots"][1]["recent_thinks"][0]["text"] == evidence[0]["think"]


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
