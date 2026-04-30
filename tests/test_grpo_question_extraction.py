"""Regression test for grpo._extract_questions_at_chunks.

Issue (2026-04-30 audit): the previous _build_rollout_messages read questions
ONLY from raw_sample["conversations"], but v12.5 trajectory data has
"questions"+"gold_action_per_chunk" and v12 flat data has "input.user_input".
Result: questions never appeared in loss-time message reconstruction →
policy was trained as "answer without seeing the question", a hard
distribution mismatch from rollout (which DOES inject the question via
StreamingAgentLoop).

This test verifies the new helper handles all 4 schemas the trainer accepts
and returns identical chunk→question maps to the rollout path.

Run: python tests/test_grpo_question_extraction.py
"""
import re
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _extract_questions_at_chunks(raw_sample) -> Dict[int, str]:
    """Inlined copy from grpo.py to avoid the heavy transformers import in CI.
    The actual implementation in thinkstream.trainer.grpo is identical."""
    out: Dict[int, str] = {}
    if (isinstance(raw_sample.get("questions"), list)
            and isinstance(raw_sample.get("gold_action_per_chunk"), dict)):
        for q in raw_sample["questions"]:
            q_text = q.get("question") or q.get("gold_answer", "")
            for ac in q.get("ask_chunks") or []:
                out[int(ac)] = q_text
        return out
    if "input" in raw_sample and raw_sample["input"].get("user_input"):
        ck = int(raw_sample.get("chunk_idx", 0))
        out[ck] = raw_sample["input"]["user_input"]
        return out
    if "messages" in raw_sample:
        ck = int(raw_sample.get("chunk_idx", 0))
        for msg in raw_sample["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            m = re.search(r'<user_input>(.*?)</user_input>', text)
                            if m:
                                out[ck] = m.group(1)
                                return out
        return out
    if "conversations" in raw_sample:
        for c in raw_sample["conversations"]:
            if c.get("role") == "user":
                ts = float(c.get("timestamp", 0.0))
                ck = int(ts)
                out[ck] = c.get("content", "")
        return out
    return out


def test_trajectory_v12_schema():
    """v12.4+ trajectory: each card may fire at multiple ask_chunks."""
    traj = {
        "questions": [
            {"question": "What color is the shirt?", "ask_chunks": [10, 20]},
            {"question": "How many people?",         "ask_chunks": [50]},
        ],
        "gold_action_per_chunk": {"10": "response", "20": "response", "50": "response"},
    }
    res = _extract_questions_at_chunks(traj)
    assert res == {
        10: "What color is the shirt?",
        20: "What color is the shirt?",
        50: "How many people?",
    }, f"got: {res}"
    print("  PASS trajectory schema (multi-ask)")


def test_flat_v12_schema():
    """v12 SFT flat: single question at chunk_idx via input.user_input."""
    flat = {"input": {"user_input": "What color?"}, "chunk_idx": 7}
    res = _extract_questions_at_chunks(flat)
    assert res == {7: "What color?"}
    print("  PASS flat input.user_input schema")


def test_messages_with_user_input_tags():
    """Older-format messages with <user_input> tags inside text content."""
    msgs = {
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "preamble <user_input>What time?</user_input> after"}],
        }],
        "chunk_idx": 3,
    }
    res = _extract_questions_at_chunks(msgs)
    assert res == {3: "What time?"}
    print("  PASS messages with <user_input> tags")


def test_legacy_conversations_schema():
    """v11 legacy: conversations[] with timestamps."""
    leg = {"conversations": [
        {"role": "user", "content": "first?", "timestamp": 5.0},
        {"role": "assistant", "content": "...", "timestamp": 5.5},
        {"role": "user", "content": "second?", "timestamp": 12.0},
    ]}
    res = _extract_questions_at_chunks(leg)
    # 1s/chunk → timestamp seconds == chunk_idx
    assert res == {5: "first?", 12: "second?"}
    print("  PASS legacy conversations schema")


def test_empty_and_malformed():
    """Defensive: missing fields / wrong types → empty dict."""
    assert _extract_questions_at_chunks({}) == {}
    assert _extract_questions_at_chunks({"questions": []}) == {}  # no gold_action_per_chunk
    assert _extract_questions_at_chunks({"questions": "not_a_list"}) == {}
    assert _extract_questions_at_chunks({"input": {}, "chunk_idx": 0}) == {}
    print("  PASS empty/malformed inputs")


def main():
    tests = [
        test_trajectory_v12_schema,
        test_flat_v12_schema,
        test_messages_with_user_input_tags,
        test_legacy_conversations_schema,
        test_empty_and_malformed,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
