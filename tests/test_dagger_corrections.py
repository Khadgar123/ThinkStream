"""Smoke tests for correction-only DAgger row selection.

Run:
  python tests/test_dagger_corrections.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data_v5 import build_dagger_sft as dagger_mod  # noqa: E402
from scripts.agent_data_v5.build_dagger_sft import _classify_dagger_corrections  # noqa: E402
from thinkstream.data.agent_protocol import build_assistant_content_v12  # noqa: E402


def _prompt(user_input: str = "", memory: str = ""):
    return [
        {"role": "system", "content": "agent"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"<memory>{memory}</memory>\n"
                        f"<user_input>{user_input}</user_input>"
                    ),
                }
            ],
        },
    ]


def _prompt_with_query(user_input: str = "", memory: str = "", query: str = ""):
    text = f"<memory>{memory}</memory>\n"
    if query:
        text += f"<active_query>\n{query}\n</active_query>\n"
    text += f"<user_input>{user_input}</user_input>"
    return [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _prompt_with_visual_window(memory: str = "", *, start: int = 9, end: int = 11):
    return [
        {"role": "system", "content": "agent"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"<memory>{memory}</memory>\n"
                        f"<visual_window>{{\"start\":{start},\"end\":{end}}}</visual_window>"
                    ),
                }
            ],
        },
    ]


def _answer_sample(sample_type: str, answer: str = "red apron", *, chunk_idx: int = 5):
    return {
        "sample_type": sample_type,
        "chunk_idx": chunk_idx,
        "output": build_assistant_content_v12(
            think="current chunk shows a cook at the counter",
            kind="answer",
            answer_text=answer if sample_type == "response" else "",
        ),
        "v12_assistant_turn_2": build_assistant_content_v12(
            think="recall result confirms the earlier red apron",
            kind="answer",
            answer_text=answer,
        ),
        "answer_chunks": [10],
    }


def test_missed_compress_and_bad_range():
    sample = {
        "sample_type": "compress",
        "chunk_idx": 20,
        "output": build_assistant_content_v12(
            think="memory is over budget; summarize older cooking steps",
            kind="compress",
            compress_summary={"time_range": [8, 16], "text": "cook prepares sauce"},
        ),
    }
    reasons, _ = _classify_dagger_corrections(
        sample,
        _prompt("<compress_trigger/>"),
        {"action": "silent", "payload": {}, "think": "I will wait"},
    )
    assert "missed_compress" in reasons

    reasons, detail = _classify_dagger_corrections(
        sample,
        _prompt("<compress_trigger/>"),
        {
            "action": "compress",
            "payload": {"summary": {"time_range": [40, 50], "text": "wrong"}},
            "think": "summarize a later section",
        },
    )
    assert "bad_compress_range" in reasons
    assert detail["gold_compress_range"] == [8, 16]


def test_missed_recall_checks_student_prompt_answer_visibility():
    sample = _answer_sample("recall", answer="red apron")
    reasons, _ = _classify_dagger_corrections(
        sample,
        _prompt(memory="the earlier scene only shows a kitchen counter"),
        {"action": "silent", "payload": {}, "think": "not enough information"},
    )
    assert "missed_recall" in reasons

    reasons, detail = _classify_dagger_corrections(
        sample,
        _prompt(memory="the cook wore a red apron before stirring the pot"),
        {"action": "silent", "payload": {}, "think": "not enough information"},
    )
    assert "missed_recall" not in reasons
    assert "recall_answer_visible_in_policy_prompt" in reasons
    assert detail["recall_prompt_leak"] is True

    reasons, _ = _classify_dagger_corrections(
        sample,
        _prompt_with_query(
            memory="the earlier scene only shows a kitchen counter",
            query="Question: What was the cook wearing?\nOptions: red apron; blue coat",
        ),
        {"action": "silent", "payload": {}, "think": "not enough information"},
    )
    assert "missed_recall" in reasons
    assert "recall_answer_visible_in_policy_prompt" not in reasons

    reasons, detail = _classify_dagger_corrections(
        sample,
        _prompt_with_visual_window(memory="the earlier scene only shows a kitchen counter"),
        {"action": "silent", "payload": {}, "think": "not enough information"},
    )
    assert "missed_recall" not in reasons
    assert "recall_answer_visible_in_policy_prompt" in reasons
    assert detail["recall_prompt_leak"] is True


def test_early_answer_and_repeated_think():
    early_sample = _answer_sample("silent", answer="red apron", chunk_idx=5)
    reasons, detail = _classify_dagger_corrections(
        early_sample,
        _prompt(memory="the question is still open and no answer has appeared"),
        {
            "action": "response",
            "payload": {"response": "red apron"},
            "think": "I can answer now",
        },
    )
    assert "early_answer" in reasons
    assert detail["answer_chunks"] == [10]

    repeated = "same scene same scene same scene same scene " * 12
    reasons, detail = _classify_dagger_corrections(
        _answer_sample("response", answer="red apron", chunk_idx=10),
        _prompt(memory=repeated),
        {
            "action": "response",
            "payload": {"response": "red apron"},
            "think": repeated,
        },
    )
    assert "repeated_or_stale_think" in reasons
    assert detail["policy_think_tokens"] >= 18


def test_emit_row_correction_only_selects_targeted_errors():
    old_build = dagger_mod._build_dagger_messages
    old_emit = dagger_mod._emit_row
    try:
        dagger_mod._build_dagger_messages = lambda *a, **k: [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "gold"},
        ]
        dagger_mod._emit_row = lambda sample, messages, frame_protocol: {
            "sample_type": sample["sample_type"],
            "video_id": "v",
            "messages": messages,
        }
        stats = {"rows": 0, "skipped": {}, "by_type": {}}
        out = StringIO()
        wrote = dagger_mod._emit_dagger_row(
            sample=_answer_sample("response", answer="red apron", chunk_idx=10),
            onpolicy_prompt=_prompt(memory="current frame shows the cook"),
            result={"action": "silent", "payload": {}, "think": "I will wait"},
            fout=out,
            stats=stats,
            ckpt="ckpt",
            data_dir=Path("."),
            frame_protocol="video_meta",
            include_failed_targets=False,
            sample_types={"response"},
            correction_only=True,
            correction_reasons={"missed_response"},
        )
        assert wrote is True
        row = json.loads(out.getvalue())
        assert row["dagger"]["selected_correction_reasons"] == ["missed_response"]
        assert stats["by_selected_correction_reason"]["missed_response"] == 1

        skipped = dagger_mod._emit_dagger_row(
            sample=_answer_sample("response", answer="red apron", chunk_idx=10),
            onpolicy_prompt=_prompt(memory="current frame shows the cook"),
            result={
                "action": "response",
                "payload": {"response": "red apron"},
                "think": "current chunk shows a cook at the counter",
            },
            fout=out,
            stats=stats,
            ckpt="ckpt",
            data_dir=Path("."),
            frame_protocol="video_meta",
            include_failed_targets=False,
            sample_types={"response"},
            correction_only=True,
            correction_reasons={"missed_response"},
        )
        assert skipped is False
        assert stats["skipped"]["no_selected_correction"] == 1
    finally:
        dagger_mod._build_dagger_messages = old_build
        dagger_mod._emit_row = old_emit


def test_dagger_missed_recall_emits_first_turn_only():
    old_build = dagger_mod._build_dagger_recall_query_messages
    old_emit = dagger_mod._emit_row
    try:
        dagger_mod._build_dagger_recall_query_messages = lambda *a, **k: [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "gold recall"},
        ]
        dagger_mod._emit_row = lambda sample, messages, frame_protocol: {
            "sample_type": sample["sample_type"],
            "sample_id": "sid",
            "metadata": {},
            "messages": messages,
        }
        stats = {"rows": 0, "skipped": {}, "by_type": {}}
        out = StringIO()
        wrote = dagger_mod._emit_dagger_row(
            sample=_answer_sample("recall", answer="red apron", chunk_idx=10),
            onpolicy_prompt=_prompt(memory="only a vague kitchen scene"),
            result={"action": "silent", "payload": {}, "think": "not enough information"},
            fout=out,
            stats=stats,
            ckpt="ckpt",
            data_dir=Path("."),
            frame_protocol="video_meta",
            include_failed_targets=False,
            sample_types={"recall"},
            correction_only=True,
            correction_reasons={"missed_recall"},
        )
        assert wrote is True
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        assert len(rows) == 1
        assert rows[0]["sft_subtype"] == "dagger_recall_query"
        assert rows[0]["tool_schema_mode"] == "streaming"
        assert rows[0]["loss_assistant_turns"] == "all"
    finally:
        dagger_mod._build_dagger_recall_query_messages = old_build
        dagger_mod._emit_row = old_emit


def test_dagger_recall_answer_emits_no_tools_last_span_row():
    old_build = dagger_mod._build_dagger_recall_response_messages
    old_emit = dagger_mod._emit_row
    try:
        dagger_mod._build_dagger_recall_response_messages = lambda *a, **k: [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "student recall"},
            {"role": "user", "content": "recall result"},
            {"role": "assistant", "content": "gold answer"},
        ]
        dagger_mod._emit_row = lambda sample, messages, frame_protocol: {
            "sample_type": sample["sample_type"],
            "sample_id": "sid",
            "metadata": {},
            "messages": messages,
        }
        stats = {"rows": 0, "skipped": {}, "by_type": {}}
        out = StringIO()
        wrote = dagger_mod._emit_dagger_row(
            sample=_answer_sample("recall", answer="red apron", chunk_idx=10),
            onpolicy_prompt=_prompt(memory="only a vague kitchen scene"),
            result={
                "action": "recall",
                "final_action": "silent",
                "payload": {"query": {"query": "apron", "time_range": "0-8"}},
                "final_payload": {},
                "think": "I need the earlier visual detail",
                "recall_messages": [{"role": "system", "content": "agent"}],
                "recall_result": {"returned_chunks": [10]},
            },
            fout=out,
            stats=stats,
            ckpt="ckpt",
            data_dir=Path("."),
            frame_protocol="video_meta",
            include_failed_targets=False,
            sample_types={"recall"},
            correction_only=True,
            correction_reasons={"missed_response"},
        )
        assert wrote is True
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        assert len(rows) == 1
        assert rows[0]["sft_subtype"] == "dagger_recall_answer"
        assert rows[0]["tool_schema_mode"] == "recall_response"
        assert rows[0]["loss_assistant_turns"] == "last"
    finally:
        dagger_mod._build_dagger_recall_response_messages = old_build
        dagger_mod._emit_row = old_emit


def main() -> None:
    test_missed_compress_and_bad_range()
    test_missed_recall_checks_student_prompt_answer_visibility()
    test_early_answer_and_repeated_think()
    test_emit_row_correction_only_selects_targeted_errors()
    test_dagger_missed_recall_emits_first_turn_only()
    test_dagger_recall_answer_emits_no_tools_last_span_row()
    print("PASS test_dagger_corrections")


if __name__ == "__main__":
    main()
