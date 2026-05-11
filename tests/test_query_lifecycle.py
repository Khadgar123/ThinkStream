from thinkstream.data.agent_protocol import format_queries_block
from thinkstream.models.agent_loop import MemoryState


def test_new_query_replaces_previous_open_query():
    memory = MemoryState()
    memory.add_query("first question", 0.0, answer_chunks=[5])
    memory.add_query("second question", 3.0, answer_chunks=[7])

    assert memory.queries[0]["status"] == "replaced"
    assert memory.queries[0]["close_reason"] == "new_query"
    assert memory.queries[1]["status"] == "open"

    rendered = format_queries_block(memory.queries)
    assert "second question" in rendered
    assert "first question" not in rendered
