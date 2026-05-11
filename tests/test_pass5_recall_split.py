"""Regression tests for pass5 recall split rows.

Run:
  python tests/test_pass5_recall_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data.pass5_messages import build_sft_rows  # noqa: E402
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

    assert second["sample_id"] == "r0:post_recall"
    assert second["sft_subtype"] == "post_recall"
    assert second["tool_schema_mode"] == "post_recall"
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


def test_rl_compress_turn_is_text_only_in_source():
    root = Path(__file__).resolve().parents[1]
    src = (root / "thinkstream/rl/streaming_agent_loop.py").read_text()
    build_start = src.index("def _build_chunk_user_content(")
    visual_start = src.index("# ── Visual window header", build_start)
    inter_chunk_return = src.index("if inter_chunk:\n                return content", build_start)
    assert inter_chunk_return < visual_start


def test_sft_action_eval_uses_turn_local_tools():
    root = Path(__file__).resolve().parents[1]
    action_src = (root / "scripts/eval/sft_action_acc.py").read_text()
    assert "tools_for_turn(_tool_mode_for_prompt(s, msgs))" in action_src


def test_grpo_loss_replay_uses_turn_local_tool_kind():
    root = Path(__file__).resolve().parents[1]
    processor_src = (root / "thinkstream/data/stream_data_processor.py").read_text()
    grpo_src = (root / "thinkstream/trainer/grpo.py").read_text()
    assert "tool_turn_kind: Optional[str] = None" in processor_src
    assert "if tool_turn_kind is not None:" in processor_src
    assert "_infer_tool_turn_kind_for_loss_messages(messages)" in grpo_src
    assert "return \"recall_response\"" in grpo_src


def main() -> None:
    test_pass5_recall_rows_split_tool_schema_and_loss_policy()
    test_select_loss_assistant_spans_supports_last_only()
    test_rl_compress_turn_is_text_only_in_source()
    test_sft_action_eval_uses_turn_local_tools()
    test_grpo_loss_replay_uses_turn_local_tool_kind()
    print("PASS test_pass5_recall_split")


if __name__ == "__main__":
    main()
