"""Regression tests for pass5 recall split rows.

Run:
  python tests/test_pass5_recall_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data_v5.pass5_messages import build_sft_rows  # noqa: E402
from thinkstream.sft.data_processor import _select_loss_assistant_spans  # noqa: E402


def _recall_sample():
    return {
        "trajectory_id": "t",
        "video_id": "v",
        "chunk_idx": 12,
        "sample_id": "r0",
        "sample_type": "recall",
        "v12_assistant_turn_1": "<think>need history</think><tool_call>{}</tool_call>",
        "v12_assistant_turn_2": "<think>result has it</think><answer>red apron</answer>",
        "metadata": {"gold_action": "recall"},
    }


def test_pass5_recall_rows_split_tool_schema_and_loss_policy():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "recall call"}]},
        {"role": "user", "content": [{"type": "text", "text": "recall result"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    rows = build_sft_rows(_recall_sample(), messages, frame_protocol="video_meta")
    assert len(rows) == 2

    first, second = rows
    assert first["sample_id"] == "r0:recall_query"
    assert first["sft_subtype"] == "recall_query"
    assert first["tool_schema_mode"] == "streaming"
    assert first["loss_assistant_turns"] == "all"
    assert len(first["messages"]) == 3

    assert second["sample_id"] == "r0:recall_answer"
    assert second["sft_subtype"] == "recall_answer"
    assert second["tool_schema_mode"] == "recall_response"
    assert second["loss_assistant_turns"] == "last"
    assert len(second["messages"]) == 5


def test_select_loss_assistant_spans_supports_last_only():
    spans = [(10, 20), (40, 50)]
    selected, indices = _select_loss_assistant_spans(spans, "last")
    assert selected == [(40, 50)]
    assert indices == [1]

    selected, indices = _select_loss_assistant_spans(spans, "all")
    assert selected == spans
    assert indices == [0, 1]


def main() -> None:
    test_pass5_recall_rows_split_tool_schema_and_loss_policy()
    test_select_loss_assistant_spans_supports_last_only()
    print("PASS test_pass5_recall_split")


if __name__ == "__main__":
    main()
