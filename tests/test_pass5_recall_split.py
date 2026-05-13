"""Regression tests for pass5 recall split rows.

Run:
  python tests/test_pass5_recall_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data.pass5_messages import build_sft_rows  # noqa: E402
from thinkstream.data.agent_protocol import build_user_content  # noqa: E402
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
    content = build_user_content(
        "",
        3,
        "/unused.mp4",
        queries=[{"question": "q?", "answers": []}],
        frame_paths=["frame_000001.jpg", "frame_000002.jpg"],
        frame_protocol="video_meta",
        inter_chunk=True,
    )
    joined_text = "\n".join(
        str(item.get("text", "")) for item in content if isinstance(item, dict)
    )
    assert all(item.get("type") != "video" for item in content)
    assert "<visual_window>" not in joined_text
    assert "<active_query>" not in joined_text


def test_sft_preprocess_uses_row_local_tools():
    root = Path(__file__).resolve().parents[1]
    processor_src = (root / "thinkstream/sft/data_processor.py").read_text()
    assert "_normalise_tools_for_sft_row(sample, sample.get(\"tools\"))" in processor_src
    assert "template_kwargs[\"tools\"] = tools" in processor_src


def test_replay_paths_use_turn_local_tool_kind():
    root = Path(__file__).resolve().parents[1]
    processor_src = (root / "thinkstream/data/stream_data_processor.py").read_text()
    agent_src = (root / "thinkstream/models/agent_loop.py").read_text()
    assert "tool_turn_kind: Optional[str] = None" in processor_src
    assert "if tool_turn_kind is not None:" in processor_src
    assert "tool_turn_kind = \"compress\" if is_inter_chunk else \"streaming\"" in agent_src
    assert "recall_gen_kwargs[\"tool_turn_kind\"] = \"post_recall\"" in agent_src


def test_rl_uses_offline_compress_boundaries_as_trigger_source():
    root = Path(__file__).resolve().parents[1]
    loop_src = (root / "thinkstream/rl/streaming_agent_loop.py").read_text()
    parquet_src = (root / "scripts/agent_data/build_verl_parquet.py").read_text()
    assert '"offline_pass2_boundaries"' in loop_src
    assert "offline_compress_chunks" in loop_src
    assert "chunk_idx in offline_compress_pending" in loop_src
    assert '"compress_trigger_source": "offline_pass2_boundaries"' in parquet_src


def test_rl_offline_compress_boundaries_include_sample_collisions():
    from scripts.agent_data.build_verl_parquet import (
        _rl_gold_actions_and_offline_compress,
    )

    gold_action = {
        "32": "compress",
        "63": "response",
    }
    traj = {
        "samples": [
            {"chunk_idx": 32, "sample_type": "compress", "action": "compress"},
            {"chunk_idx": 63, "sample_type": "compress", "action": "compress"},
            {"chunk_idx": 63, "sample_type": "response", "action": "response"},
        ],
    }

    sanitized, offline = _rl_gold_actions_and_offline_compress(gold_action, traj)
    assert sanitized == {"63": "response"}
    assert offline == [32, 63]


def main() -> None:
    test_pass5_recall_rows_split_tool_schema_and_loss_policy()
    test_select_loss_assistant_spans_supports_last_only()
    test_rl_compress_turn_is_text_only_in_source()
    test_sft_preprocess_uses_row_local_tools()
    test_replay_paths_use_turn_local_tool_kind()
    test_rl_uses_offline_compress_boundaries_as_trigger_source()
    test_rl_offline_compress_boundaries_include_sample_collisions()
    print("PASS test_pass5_recall_split")


if __name__ == "__main__":
    main()
