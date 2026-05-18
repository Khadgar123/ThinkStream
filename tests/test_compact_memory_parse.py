from scripts.agent_data.pass2_rollout import (
    build_compress_request,
    parse_compact_memory_entries,
    parse_compress_result,
)


def _compact_meta():
    return {
        "task_type": "compact_memory_update",
        "time_range": [0, 7],
        "chunks": list(range(8)),
        "merge_level": 1,
    }


def test_compact_memory_parser_keeps_bare_m_lines():
    raw = """
  <m t="0-1">A red title card appears.</m>
  <m t="2-3">Players enter the cricket field.</m>
  <m t="4-5">The bowler starts a delivery.</m>
  <m t="6-7">The batsman swings at the ball.</m>
"""

    entries = parse_compact_memory_entries(raw)
    assert len(entries) == 4
    assert entries[0]["time_range"] == [0, 1]

    parsed = parse_compress_result(raw, _compact_meta())
    assert parsed["parse_success"] is True
    assert parsed["format_repaired"] is True
    assert parsed["text"].lstrip().startswith("<m ")
    assert "<MEM>" not in parsed["text"]


def test_compact_memory_parser_closes_line_separated_m_tags():
    raw = """<MEM>
  <m t="0-1">A red title card appears.
  <m t="2-3">Players enter the cricket field.
  <m t="4-5">The bowler starts a delivery.
  <m t="6-7">The batsman swings at the ball.
</MEM>"""

    entries = parse_compact_memory_entries(raw)
    assert len(entries) == 4
    assert [e["time_range"] for e in entries] == [[0, 1], [2, 3], [4, 5], [6, 7]]

    parsed = parse_compress_result(raw, _compact_meta())
    assert parsed["parse_success"] is True
    assert parsed["format_repaired"] is True
    assert parsed["text"].count("<m ") == 4
    assert parsed["text"].count("</m>") == 4


def test_compact_memory_targets_do_not_split_old_summary_units():
    timeline = [
        {
            "type": "summary",
            "time_range": [start, end],
            "source_chunks": list(range(start, end + 1)),
            "text": f"Old summary {start}-{end}.",
            "merge_level": 1,
            "compact_memory": True,
        }
        for start, end in ([0, 14], [15, 29], [30, 44], [45, 59], [60, 74], [75, 89])
    ]
    timeline.extend(
        {"type": "think", "chunk": chunk, "text": f"New caption {chunk}."}
        for chunk in range(90, 120)
    )

    req = build_compress_request(timeline, None, "vid", 120)
    assert req is not None
    meta = req["_meta"]
    target_ranges = meta["target_ranges"]
    old_ranges = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 74), (75, 89)]

    for target_start, target_end in target_ranges:
        for old_start, old_end in old_ranges:
            if target_end < old_start or target_start > old_end:
                continue
            assert target_start <= old_start
            assert target_end >= old_end

    assert [0, 19] not in target_ranges
    assert [20, 39] not in target_ranges
    assert any(
        item["requires_rewrite"]
        and sum(1 for source in item["sources"] if source["kind"] == "old_memory") >= 2
        for item in meta["target_source_units"]
    )
    prompt = req["messages"][1]["content"]
    assert "TARGET_SOURCE_UNITS" in prompt
    assert 'action="rewrite-summary"' in prompt
    assert "OLD_MEMORY:0-14" in prompt


def test_compact_memory_overlapping_old_summaries_are_rewritten_whole():
    timeline = [
        {
            "type": "summary",
            "time_range": [0, 14],
            "source_chunks": list(range(0, 15)),
            "text": "First old summary.",
            "merge_level": 1,
            "compact_memory": True,
        },
        {
            "type": "summary",
            "time_range": [14, 28],
            "source_chunks": list(range(14, 29)),
            "text": "Second old summary.",
            "merge_level": 1,
            "compact_memory": True,
        },
    ]
    timeline.extend(
        {"type": "think", "chunk": chunk, "text": f"New caption {chunk}."}
        for chunk in range(29, 59)
    )

    req = build_compress_request(timeline, None, "vid", 59)
    assert req is not None
    meta = req["_meta"]

    assert [0, 19] not in meta["target_ranges"]
    assert [20, 39] not in meta["target_ranges"]
    merged_old_targets = [
        item for item in meta["target_source_units"]
        if any(source["start"] == 0 and source["end"] == 14 for source in item["sources"])
        and any(source["start"] == 14 and source["end"] == 28 for source in item["sources"])
    ]
    assert merged_old_targets
    assert merged_old_targets[0]["requires_rewrite"] is True
