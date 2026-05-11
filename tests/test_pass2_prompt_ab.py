from scripts.agent_data.pass2_prompt_ab import (
    VariantSpec,
    _serialize_memory,
    build_variant_repair_request,
    build_variant_observation_request,
    select_repair_heavy_videos,
)
from scripts.agent_data.pass2_rollout import MemoryState


def _seed_memory() -> MemoryState:
    memory = MemoryState()
    memory.add_think(0, "A woman in a red apron stands near a stove.")
    memory.add_think(1, "The red-apron woman lifts a silver pan.")
    memory.compress(
        {
            "time_range": [0, 1],
            "text": "A red-apron cook appears by the stove and lifts a silver pan.",
        },
        selected_indices=[0, 1],
    )
    memory.add_think(2, "A white bowl sits on the wooden counter.")
    return memory


def test_serialize_memory_minfields_strips_history_flags():
    text = _serialize_memory(_seed_memory(), "structured_minfields")
    assert '"kind": "summary"' in text
    assert '"kind": "think"' in text
    assert '"history_only"' not in text
    assert '"use"' not in text


def test_serialize_memory_student_tags_uses_only_time_tags():
    text = _serialize_memory(
        _seed_memory(),
        "student_tags",
        recent_limit=2,
        include_summaries=True,
    )
    assert '<memory t="0-1">' in text
    assert '<memory t="1-2">' in text
    assert '<memory t="2-3">' in text
    assert '"kind"' not in text
    assert '"chunk"' not in text
    assert '"history_only"' not in text


def test_serialize_memory_student_tags_full_timeline_keeps_all_memory():
    text = _serialize_memory(
        _seed_memory(),
        "student_tags",
        recent_limit=None,
        include_summaries=True,
    )
    assert '<memory t="0-1">' in text
    assert '<memory t="2-3">' in text
    assert text.count("<memory ") == 2


def test_serialize_memory_student_tags_summaries_first_reorders_memory():
    text = _serialize_memory(
        _seed_memory(),
        "student_tags",
        recent_limit=None,
        include_summaries=True,
        memory_order="summaries_first",
    )
    lines = text.splitlines()
    assert lines[0].startswith('<memory t="0-1">')
    assert lines[1].startswith('<memory t="2-3">')


def test_variant_request_can_put_images_first_and_reverse(tmp_path):
    frame_paths = []
    for idx in range(4):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"jpg")
        frame_paths.append(str(path))

    spec = VariantSpec(
        name="structured_images_first_reverse",
        memory_mode="structured",
        block_order="images_first",
        frame_order="reverse",
    )
    req = build_variant_observation_request(spec, 1, frame_paths, MemoryState(), "vid")
    content = req["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == '<frame ts="1.5" role="latest chunk" />'
    assert content[1]["type"] == "image_url"
    assert content[-1]["type"] == "text"
    assert "CURRENT TASK FIRST" in content[-1]["text"]


def test_variant_request_can_prepend_latest_chunk_before_window(tmp_path):
    frame_paths = []
    for idx in range(4):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"jpg")
        frame_paths.append(str(path))

    spec = VariantSpec(
        name="fulltag_strong_timeline_dup_latest_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        prepend_latest_chunk_first=True,
    )
    req = build_variant_observation_request(spec, 1, frame_paths, MemoryState(), "vid")
    content = req["messages"][0]["content"]
    assert content[1]["text"] == '<frame ts="1.0" role="latest chunk" />'
    assert content[3]["text"] == '<frame ts="1.5" role="latest chunk" />'
    assert content[5]["text"] == '<frame ts="0.0" role="older context" />'


def test_variant_repair_request_can_be_current_visual_only(tmp_path):
    frame_paths = []
    for idx in range(4):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"jpg")
        frame_paths.append(str(path))

    memory = MemoryState()
    memory.add_think(0, "Old think 0")
    memory.add_think(1, "Old think 1")
    spec = VariantSpec(
        name="fulltag_strong_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        repair_mode="current_only_visual",
    )
    req = build_variant_repair_request(
        spec,
        1,
        frame_paths,
        memory,
        "vid",
        stale_text="Old stale text",
    )
    prompt = req["messages"][0]["content"][0]["text"]
    assert "Do not use any history or prior draft" in prompt
    assert "<memory" not in prompt


def test_variant_request_can_use_max_strict_prompt(tmp_path):
    frame_paths = []
    for idx in range(4):
        path = tmp_path / f"frame_{idx + 1:06d}.jpg"
        path.write_bytes(b"jpg")
        frame_paths.append(str(path))

    spec = VariantSpec(
        name="fulltag_max_strict_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        instruction_mode="max_strict",
    )
    req = build_variant_observation_request(spec, 1, frame_paths, MemoryState(), "vid")
    prompt = req["messages"][0]["content"][0]["text"]
    assert "Past text memory is never continuation context." in prompt
    assert "First isolate the two frames tagged" in prompt
    assert 'ONLY t=1-2s' in prompt


def test_select_repair_heavy_videos_prefers_more_rejections(tmp_path):
    p = tmp_path / "pass2_chunks.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"video_id":"a","repair_attempted":true,"repair_rejected":true}',
                '{"video_id":"a","repair_attempted":true,"repair_rejected":true}',
                '{"video_id":"b","repair_attempted":true,"repair_rejected":true}',
                '{"video_id":"b","repair_attempted":true,"repair_rejected":false}',
            ]
        )
        + "\n"
    )
    assert select_repair_heavy_videos(p, top_k=2) == ["a", "b"]
