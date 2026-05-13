from scripts.agent_data.pass2_rollout import (
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


def test_compact_memory_parser_wraps_bare_m_lines():
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
    assert parsed["text"].startswith("<MEM>")
    assert parsed["text"].endswith("</MEM>")


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
