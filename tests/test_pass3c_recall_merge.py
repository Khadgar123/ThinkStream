"""Regression tests for the current pass3c v12 multi-turn recall shape.

The old v12.5 path built separate recall_query / recall_response rows and then
merged them. Current pass3c emits one sample_type='recall' row directly, with
v12_assistant_turn_1 for the tool call and v12_assistant_turn_2 for the
post-recall answer or silence.
"""

import ast
from pathlib import Path


def test_recall_response_sample_is_direct_multiturn():
    from scripts.agent_data.pass3c_samples import _recall_response_sample

    sample = _recall_response_sample(
        5,
        "current view lacks the old detail",
        "red",
        queries=[{"question": "What color was the apron?", "answers": []}],
        recall_query={"start_time": 2, "end_time": 3},
        recall_result={"source": "historical_frames", "returned_chunks": [2, 3], "time": "2-3"},
        trajectory_id="t0",
        card_id="c1",
        sequence_type="recall_success",
        card={"answer_form": "short_exact", "canonical_answer": "red"},
    )

    assert sample["sample_type"] == "recall"
    assert sample["output"] == sample["v12_assistant_turn_2"]
    assert "<tool_call>" in sample["v12_assistant_turn_1"]
    assert '"name":"recall"' in sample["v12_assistant_turn_1"].replace(" ", "")
    assert "needed evidence is historical" not in sample["v12_assistant_turn_1"]
    assert "I will recall" not in sample["v12_assistant_turn_1"]
    assert "</Response> red" in sample["v12_assistant_turn_2"]
    assert sample["recall_result"]["returned_chunks"] == [2, 3]


def test_recall_silent_sample_is_direct_multiturn():
    from scripts.agent_data.pass3c_samples import _recall_silent_multiturn_sample

    sample = _recall_silent_multiturn_sample(
        5,
        "current view lacks the future detail",
        queries=[{"question": "What happens next?", "answers": []}],
        recall_query={"start_time": 0, "end_time": 1},
        recall_result={"source": "historical_frames", "returned_chunks": [0, 1], "time": "0-1"},
        trajectory_id="t0",
        card_id="c1",
        sequence_type="event_watch",
    )

    assert sample["sample_type"] == "recall"
    assert sample["action"] == "silent"
    assert sample["output"] == sample["v12_assistant_turn_2"]
    assert "<tool_call>" in sample["v12_assistant_turn_1"]
    assert "I will recall" not in sample["v12_assistant_turn_1"]
    assert "</Silence>" in sample["v12_assistant_turn_2"]
    assert sample["base_role"] == "recall_silent"


def test_legacy_merge_path_is_removed():
    src = Path(__file__).resolve().parents[1] / "scripts" / "agent_data" / "pass3c_samples.py"
    tree = ast.parse(src.read_text())

    function_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called_names = {
        sub.func.id for sub in ast.walk(tree)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }

    assert "_merge_recall_pairs" not in function_names
    assert "_merge_recall_pairs" not in called_names
