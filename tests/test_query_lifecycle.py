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


def test_count_query_renders_response_history():
    rendered = format_queries_block([{
        "question": "How many cups have appeared so far?",
        "ask_time": 5,
        "status": "open",
        "answer_form": "number",
        "question_type": "multi_emit",
        "question_way": "repeated_count",
        "answer_chunks": [10, 15],
        "answers": [{"time": 10, "text": "1", "expected_chunk": 10}],
    }])

    assert "<response_history>" in rendered
    assert "[10s] A: 1" in rendered


def test_count_query_response_history_canonicalizes_number_surface():
    rendered = format_queries_block([{
        "question": "How many paper pieces have appeared so far?",
        "ask_time": 0,
        "status": "open",
        "answer_form": "number",
        "question_type": "multi_emit",
        "question_way": "repeated_count",
        "answer_chunks": [5, 10],
        "answers": [{"time": 5, "text": "1/1", "expected_chunk": 5}],
    }])

    assert "[5s] A: 1" in rendered
    assert "1/1" not in rendered


def test_status_probe_omits_response_history_answers():
    rendered = format_queries_block([{
        "question": "Is the door currently open?",
        "ask_time": 5,
        "status": "open",
        "answer_form": "binary",
        "question_type": "multi_emit",
        "question_way": "current_status_probe",
        "evidence_type": "status_probe_stream",
        "answer_chunks": [5, 10],
        "answers": [{"time": 5, "text": "Yes", "expected_chunk": 5}],
    }])

    assert "<active_query>" in rendered
    assert "<response_history>" in rendered
    assert "[5s] A: Yes" not in rendered
