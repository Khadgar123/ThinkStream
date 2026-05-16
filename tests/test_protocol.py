"""v12.0 protocol smoke tests.

Tests the parts of v12.0 that don't need a real tokenizer / model:
- Protocol generation + parsing roundtrip
- pass3c v12 sample format
- compress_trigger injection logic
- Gate classification logic

Run: python -m pytest tests/test_v12_protocol.py -v
   or: python tests/test_v12_protocol.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_v12_assistant_content_roundtrip():
    from thinkstream.data.agent_protocol import (
        build_assistant_content,
        parse_agent_output,
    )

    # silent
    s = build_assistant_content(think="x", kind="answer", answer_text="")
    p = parse_agent_output(s)
    assert p["kind"] == "answer" and p["answer_text"] == "" and p["format_error"] is None

    # response
    s = build_assistant_content(think="x", kind="answer", answer_text="hello")
    p = parse_agent_output(s)
    assert p["kind"] == "answer" and p["answer_text"] == "hello"

    # recall tool_call
    s = build_assistant_content(
        think="x", kind="recall",
        recall_query={"start_time": 10, "end_time": 31},
    )
    p = parse_agent_output(s)
    assert p["kind"] == "recall"
    assert p["tool_call"]["name"] == "recall"
    assert "query" not in p["tool_call"]["arguments"]
    assert p["tool_call"]["arguments"] == {"start_time": 10, "end_time": 31}

    # compact-memory update
    s = build_assistant_content(
        think="x", kind="compress",
        compress_summary={"time_range": [4, 12], "text": "summary"},
    )
    p = parse_agent_output(s)
    assert p["kind"] == "compress"
    assert p["tool_call"]["name"] == "memory_update"
    assert p["memory_text"] == '<m t="4-12">summary</m>'

    # compact-memory turns may be emitted as bare chronological <m> lines.
    bare_mem = "\n".join([
        '<m t="0-1">A red title card appears.</m>',
        '<m t="2-3">Players enter the cricket field.</m>',
        '<m t="4-5">The bowler starts a delivery.</m>',
        '<m t="6-7">The batsman swings at the ball.</m>',
    ])
    p = parse_agent_output(bare_mem)
    assert p["kind"] == "compress"
    assert p["format_error"] is None
    assert p["memory_text"].lstrip().startswith("<m ")
    assert "<MEM>" not in p["memory_text"]

    # format error: no terminal
    p = parse_agent_output("<think>x</think>")
    assert p["format_error"] == "neither </Response>/</Silence> nor recall <tool_call> nor compact-memory <m> lines emitted"

    # format error: both terminals
    p = parse_agent_output(
        "<think>x</think></Response> a<tool_call>{}</tool_call>"
    )
    assert p["format_error"] == "multiple terminal blocks present"

    # format error: bad json
    p = parse_agent_output("<think>x</think><tool_call>not json</tool_call>")
    assert "JSON parse error" in p["format_error"]

    # narrow JSON repair: model sometimes escapes apostrophes as \'
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"compress","arguments":'
        '{"time_range":[4,12],"text":"chef adds \\\'Honey\\\' to bowl"}}'
        '</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "compress tool_call is not allowed" in p["format_error"]
    assert p["tool_call"]["arguments"]["text"] == "chef adds 'Honey' to bowl"

    # malformed legacy compress tool_call is rejected instead of repaired into
    # the active compact-memory protocol.
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"compress","arguments":'
        '{"time_range":[0,53],"text":"Text reads "\'Healthy Weight Loss Recipe,\' '
        'then line one\nline two"}}}</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "JSON parse error" in p["format_error"]

    # duplicated legacy compress prefix is also rejected.
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"compress","arguments":'
        '{"time_range":[6,29],"text":"bad prefix</think><tool_call>\n'
        '{"name":"compress","arguments":{"time_range":[6,29],"text":"usable summary"}}}'
        '</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "JSON parse error" in p["format_error"]

    # post-recall runtime may accept a bare answer, but the default parser stays strict.
    bare = "<think>use recalled frames</think>B"
    assert parse_agent_output(bare)["format_error"] == "neither </Response>/</Silence> nor recall <tool_call> nor compact-memory <m> lines emitted"
    p = parse_agent_output(bare, allow_bare_answer=True)
    assert p["kind"] == "answer"
    assert p["answer_text"] == "B"

    # compress-turn runtime may recover a malformed tool tag, but default stays strict.
    bad_tag = (
        '<think>compress memory</think><tool {"name":"compress","arguments":'
        '{"time_range":[111,135],"text":"usable compression summary"}}</tool>'
    )
    assert parse_agent_output(bad_tag)["format_error"] == "neither </Response>/</Silence> nor recall <tool_call> nor compact-memory <m> lines emitted"
    p = parse_agent_output(bad_tag, allow_malformed_tool_call=True)
    assert p["kind"] == "unknown"
    assert "compress tool_call is not allowed" in p["format_error"]
    assert p["tool_call"]["arguments"]["time_range"] == [111, 135]

    # format error: unknown tool
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"foo","arguments":{}}</tool_call>'
    )
    assert "unknown tool" in p["format_error"]

    # format error: recall schema
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"recall","arguments":{"start_time":1}}</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "start_time" in p["format_error"]
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"recall","arguments":{"time_range":[1,5]}}</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "start_time" in p["format_error"]

    # format error: compress schema
    p = parse_agent_output(
        '<think>x</think><tool_call>{"name":"compress","arguments":{"time_range":"4-12","text":"summary"}}</tool_call>'
    )
    assert p["kind"] == "unknown"
    assert "compress tool_call is not allowed" in p["format_error"]

    # format error: extra text outside required skeleton
    p = parse_agent_output("<think>x</think></Silence>\nextra")
    assert "text outside" in p["format_error"]

    print("✓ v12_assistant_content_roundtrip")


def test_compress_trigger():
    from thinkstream.data.agent_protocol import (
        build_compress_trigger_user_input,
        has_compress_trigger,
        extract_compress_trigger_range,
    )

    assert build_compress_trigger_user_input() == "<compress_trigger/>"
    assert has_compress_trigger("<compress_trigger range='4-12'/>") is True
    assert has_compress_trigger("<compress_trigger/>") is True
    assert has_compress_trigger("nothing here") is False
    assert has_compress_trigger("") is False
    assert has_compress_trigger(None) is False

    assert extract_compress_trigger_range("<compress_trigger range='4-12'/>") == [4, 12]
    assert extract_compress_trigger_range("<compress_trigger range=\"100-200\"/>") == [100, 200]
    assert extract_compress_trigger_range("<compress_trigger/>") is None
    assert extract_compress_trigger_range("") is None

    print("✓ compress_trigger detection")


def test_tools_schema_shape():
    from thinkstream.data.agent_protocol import TOOLS_SCHEMA

    assert isinstance(TOOLS_SCHEMA, list)
    assert len(TOOLS_SCHEMA) == 1
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert names == {"recall"}, f"Got: {names}"

    for tool in TOOLS_SCHEMA:
        assert tool["type"] == "function"
        f = tool["function"]
        assert "name" in f and "description" in f and "parameters" in f
        params = f["parameters"]
        assert params["type"] == "object"
        assert "properties" in params and "required" in params

    print("✓ TOOLS_SCHEMA shape")


def test_agent_special_tokens_current_only():
    from thinkstream.data.agent_protocol import (
        AGENT_SPECIAL_TOKENS,
        WRONG_RESPONSE_SPECIAL_TOKENS,
        ensure_agent_special_tokens,
        validate_agent_special_tokens,
    )

    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"<|im_start|>": 0, "<response>": 1, "</response>": 2}
            self.additional_special_tokens = [
                "<|im_start|>",
                "<response>",
                "</response>",
            ]

        def __len__(self):
            return len(self.vocab)

        @property
        def all_special_tokens(self):
            return list(self.additional_special_tokens)

        def get_vocab(self):
            return dict(self.vocab)

        def add_special_tokens(self, special_tokens_dict, replace_additional_special_tokens=True):
            tokens = list(special_tokens_dict.get("additional_special_tokens") or [])
            added = 0
            if replace_additional_special_tokens:
                self.additional_special_tokens = []
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                    added += 1
                if tok not in self.additional_special_tokens:
                    self.additional_special_tokens.append(tok)
            return added

        def encode(self, text, add_special_tokens=False):
            if text in self.additional_special_tokens and text in self.vocab:
                return [self.vocab[text]]
            return [1000 + ord(ch) for ch in text]

    tok = DummyTokenizer()
    added = ensure_agent_special_tokens(tok)
    validate_agent_special_tokens(tok)

    assert added == len([t for t in AGENT_SPECIAL_TOKENS if t not in {"<response>", "</response>"}])
    assert "<|im_start|>" in tok.additional_special_tokens
    for old in WRONG_RESPONSE_SPECIAL_TOKENS:
        assert old not in tok.additional_special_tokens
    for current in AGENT_SPECIAL_TOKENS:
        assert current in tok.additional_special_tokens

    print("✓ agent special tokens current-only registration")


def test_turn_local_tools_and_action_space():
    from thinkstream.data.agent_protocol import (
        action_space_error_for_turn,
        allowed_actions_for_turn,
        tools_for_turn,
    )

    assert [t["function"]["name"] for t in tools_for_turn("streaming")] == ["recall"]
    assert tools_for_turn("compress") is None
    assert tools_for_turn("recall_response") is None

    assert allowed_actions_for_turn("streaming") == {
        "answer", "recall", "response", "silent",
    }
    assert allowed_actions_for_turn("compress") == {"compress"}
    assert allowed_actions_for_turn("recall_response") == {
        "answer", "response", "silent",
    }

    assert action_space_error_for_turn("compress", "streaming")
    assert action_space_error_for_turn("recall", "compress")
    assert action_space_error_for_turn("recall", "recall_response")
    assert action_space_error_for_turn("silent", "recall_response") == ""

    print("✓ turn-local tools/action space")


def test_pass3c_v12_emission():
    """Current pass3c builders emit v12 sample outputs."""
    from scripts.agent_data import pass3c_samples
    from thinkstream.data.agent_protocol import parse_agent_output

    s = pass3c_samples._silent_sample(
        5, "frame shows kitchen", [], "t1", card_id="c1",
        sequence_type="base",
    )
    assert s["sample_type"] == "silent"
    assert s["output"] == "<think>frame shows kitchen</think></Silence>"

    s = pass3c_samples._response_sample(
        5, "user asked color", "red", [], "t1", "c1",
        "immediate_response",
    )
    assert s["sample_type"] == "response"
    assert "<think>user asked color</think>" in s["output"]
    assert "</Response> red" in s["output"]

    s = pass3c_samples._recall_response_sample(
        5, "need history", "red", [],
        {"start_time": 10, "end_time": 31},
        {"source": "historical_frames", "time": "10-30", "text_content": "red"},
        "t1", "c1", "recall",
    )
    assert s["sample_type"] == "recall"
    assert "<tool_call>" in s["v12_assistant_turn_1"]
    parsed = json.loads(
        s["v12_assistant_turn_1"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
    )
    assert parsed["name"] == "recall"
    assert parsed["arguments"] == {"start_time": 10, "end_time": 31}
    assert "</Response> red" in s["v12_assistant_turn_2"]

    s = pass3c_samples._compress_sample(
        8, "memory full", [], "t1",
        {
            "summary": {"time_range": [4, 12], "text": "chef cooks"}
        },
    )
    assert s["sample_type"] == "compress"
    assert s["user_input"] == ""
    parsed = parse_agent_output(s["output"])
    assert parsed["kind"] == "compress"
    assert parsed["memory_text"] == '<m t="4-12">chef cooks</m>'

    print("✓ pass3c v12 emission")


def test_freegen_gate_classifier():
    """v12 gate's classify_emission categorizes outputs correctly."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    try:
        from v12_freegen_gate import classify_emission
    except ModuleNotFoundError:
        print("[SKIP] v12_freegen_gate helper is not present in this checkout")
        return

    # silent
    c = classify_emission("<think>x</think></Silence>")
    assert c["category"] == "answer_silent"

    # response
    c = classify_emission("<think>x</think></Response> red")
    assert c["category"] == "answer_response"

    # recall tool
    c = classify_emission(
        '<think>x</think><tool_call>{"name":"recall","arguments":{"start_time":1,"end_time":5}}</tool_call>'
    )
    assert c["category"] == "tool_recall"

    # compress tool
    c = classify_emission(
        '<think>x</think><tool_call>{"name":"compress","arguments":{"time_range":[4,12],"text":"s"}}</tool_call>'
    )
    assert c["category"] == "tool_compress"

    # format errors
    assert classify_emission("<think>x</think>")["category"] == "format_error"
    assert classify_emission("<think>x</think><tool_call>not json</tool_call>")["category"] == "format_error"
    assert classify_emission(
        '<think>x</think><tool_call>{"name":"foo"}</tool_call>'
    )["category"] == "format_error"

    print("✓ freegen gate classifier")


def test_v12_recall_multiturn_merge():
    """Current pass3c emits recall as a single shape-B multi-turn sample."""
    from scripts.agent_data import pass3c_samples as pass3c

    recall = pass3c._recall_response_sample(
        5, "need history", "red", [],
        {"start_time": 1, "end_time": 6},
        {"source": "historical_frames", "time": "1-5", "text_content": "red apron"},
        "t1", "c1", "recall",
    )
    assert "v12_assistant_turn_1" in recall
    assert "v12_assistant_turn_2" in recall
    assert "tool_call" in recall["v12_assistant_turn_1"]
    assert "</Response> red" in recall["v12_assistant_turn_2"]
    assert recall["recall_result"]["text_content"] == "red apron"
    assert recall["output"] == recall["v12_assistant_turn_2"]

    print("✓ v12 recall multi-turn merge")


def test_v12_recall_silent_merge():
    """recall_silent uses the same shape-B recall sample with empty answer."""
    from scripts.agent_data import pass3c_samples as pass3c

    r = pass3c._recall_silent_multiturn_sample(
        5, "x", [],
        {"start_time": 1, "end_time": 6},
        {"source": "memory", "text_content": "no relevant past observation"},
        "t1", "c1", "recall",
    )
    assert r["sample_type"] == "recall"
    assert r["action"] == "silent"
    assert "</Silence>" in r["v12_assistant_turn_2"]

    print("✓ v12 recall_silent merge")


def test_v12_compress_inter_chunk_flag():
    """v12 compress samples carry inter_chunk=True flag."""
    from scripts.agent_data import pass3c_samples as pass3c

    s = pass3c._compress_sample(
        8, "full", [], "t1",
        {
            "summary": {"time_range": [4, 12], "text": "summary"}
        },
    )
    assert s.get("inter_chunk") is True, (
        "compress sample should be flagged as inter-chunk in v12"
    )

    s2 = pass3c._silent_sample(
        5, "x", [], "t1", card_id="c1",
        sequence_type="base",
    )
    assert s2.get("inter_chunk") is None or s2.get("inter_chunk") is False

    print("✓ v12 compress inter_chunk flag")


def test_pass4_v12_format_acceptance():
    """pass4 verify_format must ACCEPT well-formed v12 samples (silent /
    response / multi-turn recall / inter-chunk compress) and REJECT
    legacy non-v12 samples that arrive marked as v12."""
    from scripts.agent_data.pass3e_verify import verify_format

    # silent — Streamo-style silence token
    silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>frame shows kitchen with chef</think></Silence>",
    }
    ok, reason = verify_format(silent)
    assert ok, f"silent rejected: {reason}"

    # response — Streamo-style response token plus answer text
    resp = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>user asked color of chef apron, it is red</think></Response> red",
    }
    ok, reason = verify_format(resp)
    assert ok, f"response rejected: {reason}"

    # multi-turn recall
    recall = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": '<think>need history about color of apron worn earlier</think><tool_call>\n{"name":"recall","arguments":{"start_time":10,"end_time":31}}\n</tool_call>',
        "v12_assistant_turn_2": "<think>found red apron worn by chef earlier</think></Response> red",
    }
    ok, reason = verify_format(recall)
    assert ok, f"recall multi-turn rejected: {reason}"

    # inter-chunk compress
    compress = {
        "sample_type": "compress",
        "protocol_version": "v12",
        "inter_chunk": True,
        "output": "<think>update compact memory</think>\n" + "\n".join([
            '<m t="0-1">The chef prepares ingredients at the counter.</m>',
            '<m t="2-3">The chef adds oil to the pan.</m>',
            '<m t="4-5">The chef adds salt and pepper.</m>',
            '<m t="6-7">The chef stirs the food in the pan.</m>',
        ]),
    }
    ok, reason = verify_format(compress)
    assert ok, f"compress rejected: {reason}"

    # bad: silent with non-empty answer
    bad_silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>nothing new in scene yet</think></Response> red",
    }
    ok, reason = verify_format(bad_silent)
    assert not ok and "v12_silent_answer_must_be_empty" in reason

    # bad: compress without inter-chunk flag
    bad_compress = {
        "sample_type": "compress",
        "protocol_version": "v12",
        # no inter_chunk flag
        "output": "<think>x is happening here</think>\n" + "\n".join([
            '<m t="0-1">The chef prepares ingredients at the counter.</m>',
            '<m t="2-3">The chef adds oil to the pan.</m>',
            '<m t="4-5">The chef adds salt and pepper.</m>',
            '<m t="6-7">The chef stirs the food in the pan.</m>',
        ]),
    }
    ok, reason = verify_format(bad_compress)
    assert not ok and "v12_compress_missing_inter_chunk_flag" in reason

    # bad: tool_call invalid JSON
    bad_json = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history</think><tool_call>not json</tool_call>",
        "v12_assistant_turn_2": "<think>x</think></Response> red",
    }
    ok, reason = verify_format(bad_json)
    assert not ok and "JSON parse error" in reason

    print("✓ pass4 verify_format v12 acceptance/rejection")


def test_pass4_v12_information_flow():
    """v12 information_flow validates yes/no/MC/number response strict format."""
    from scripts.agent_data.pass3e_verify import verify_information_flow

    # binary form — must be exactly Yes/No
    good_binary = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>asks if van is in scene, I see white van clearly</think></Response> Yes",
        "metadata": {"answer_form": "binary"},
    }
    ok, reason = verify_information_flow(good_binary)
    assert ok, f"good binary rejected: {reason}"

    bad_binary = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>asks if van is in scene currently visible</think></Response> yes definitely",
        "metadata": {"answer_form": "binary"},
    }
    ok, reason = verify_information_flow(bad_binary)
    assert not ok and "binary_response_not_yes_no" in reason

    # number form
    bad_number = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>counted three apples carefully now</think></Response> three",
        "metadata": {"answer_form": "number"},
    }
    ok, reason = verify_information_flow(bad_number)
    assert not ok and "number_response_not_digits" in reason

    # silent samples should not trip empty-response check
    silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>nothing new visible in current chunk</think></Silence>",
        "metadata": {},
    }
    ok, reason = verify_information_flow(silent)
    assert ok, f"silent rejected: {reason}"

    print("✓ pass4 verify_information_flow v12")


def test_pass4_v12_grounding_multiturn():
    """verify_grounding is currently a non-destructive policy skip for v12."""
    from scripts.agent_data.pass3e_verify import verify_grounding

    # Multi-turn recall sample — output popped, turns in v12_assistant_turn_*
    multi_recall = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history about chef apron color</think><tool_call>{}</tool_call>",
        "v12_assistant_turn_2": "<think>found red apron in earlier scene</think></Response> red",
    }
    ok, reason = verify_grounding(multi_recall)
    assert ok, f"v12 multi-turn recall grounding rejected: {reason}"

    # Non-visual phrase audits are policy-skipped rather than hard-failed.
    bad_multi = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history about chef apron color</think><tool_call>{}</tool_call>",
        "v12_assistant_turn_2": "<think>chef tastes the dish smells aromatic</think></Response> red",
    }
    ok, reason = verify_grounding(bad_multi)
    assert ok and reason == "skipped_think_grounding_policy"

    print("✓ pass4 verify_grounding v12 policy skip")


def test_pass4_v12_recall_evidence_reachable():
    """verify_recall_evidence_reachable must trigger on v12 sample_type='recall'."""
    from scripts.agent_data.pass3e_verify import verify_recall_evidence_reachable

    # v12 recall with evidence in future → should fail
    bad = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "card_id": "c1",
        "chunk_idx": 5,
        "metadata": {"support_chunks": [10]},  # future evidence
        "recall_result": {
            "source": "historical_frames",
            "time": "4-10",
            "returned_chunks": [10],
        },
    }
    ok, reason = verify_recall_evidence_reachable(bad)
    assert not ok and "future" in reason

    # v12 recall with valid past evidence → pass
    good = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "card_id": "c1",
        "chunk_idx": 5,
        "metadata": {"support_chunks": [2, 3]},
        "recall_result": {
            "source": "historical_frames",
            "time": "2-3",
            "returned_chunks": [2, 3],
        },
    }
    ok, reason = verify_recall_evidence_reachable(good)
    assert ok, reason

    print("✓ pass4 verify_recall_evidence_reachable v12 sample_type=recall")


def test_pass4_v12_metadata_complete():
    """verify_metadata_complete must trigger on v12 sample_type='recall'."""
    from scripts.agent_data.pass3e_verify import verify_metadata_complete

    # v12 recall without gold_answer → should fail
    bad = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "card_id": "c1",
        "metadata": {"gold_answer": ""},
    }
    ok, reason = verify_metadata_complete(bad)
    assert not ok and "gold_answer empty" in reason

    # v12 recall with gold_answer → pass
    good = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "card_id": "c1",
        "metadata": {"gold_answer": "red"},
    }
    ok, reason = verify_metadata_complete(good)
    assert ok, reason

    print("✓ pass4 verify_metadata_complete v12 sample_type=recall")


def test_pass4_v12_action_minimality():
    """verify_action_minimality soft-allows v12 recall sequence variants."""
    from scripts.agent_data.pass3e_verify import verify_action_minimality

    # v12 recall in an immediate sequence is now allowed; recall can verify
    # historical details beyond the old recall_success label.
    immediate = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "immediate_response",  # NOT a recall seq
        "metadata": {},
    }
    ok, reason = verify_action_minimality(immediate)
    assert ok, reason

    # v12 recall in valid sequence → pass
    good = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "recall_success",
        "metadata": {"visibility": {}},
    }
    ok, reason = verify_action_minimality(good)
    assert ok, reason

    # Legacy visibility flags are audit signals and no longer hard-fail rows.
    visible = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "recall_success",
        "metadata": {"visibility": {"answer_in_recent_obs": True}},
    }
    ok, reason = verify_action_minimality(visible)
    assert ok, reason

    print("✓ pass4 verify_action_minimality v12 soft allow")


def test_pass4_v11_backward_compat():
    """v11 backward-compat path was removed when the codebase consolidated
    on v12. Kept as a skipped placeholder for traceability."""
    import pytest
    pytest.skip("v11 protocol removed; pass3e_verify now exercises v12 only.")


def test_freegen_gate_aggregation():
    """End-to-end gate: synthetic samples + classifications → metrics + verdict."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    try:
        from v12_freegen_gate import aggregate_gate_metrics, evaluate_gates, DEFAULT_GATES
    except ModuleNotFoundError:
        print("[SKIP] v12_freegen_gate helper is not present in this checkout")
        return

    samples = [
        {"sample_type": "silent", "user_input": ""},
        {"sample_type": "silent", "user_input": ""},
        {"sample_type": "response", "user_input": "what color"},
        {"sample_type": "recall_query", "user_input": ""},
        {"sample_type": "compress", "user_input": "<compress_trigger range='4-12'/>"},
        {"sample_type": "compress", "user_input": "<compress_trigger range='4-12'/>"},
    ]
    classifications = [
        {"category": "answer_silent"},
        {"category": "answer_silent"},
        {"category": "answer_response"},
        {"category": "tool_recall"},
        {"category": "tool_compress"},
        {"category": "tool_compress"},
    ]
    m = aggregate_gate_metrics(samples, classifications)
    assert m["n_total"] == 6
    assert m["n_trigger_samples"] == 2
    assert m["metrics"]["compress_emit_rate"] == 1.0
    assert m["metrics"]["recall_emit_rate"] == 1 / 6
    assert m["metrics"]["format_compliance"] == 1.0
    assert m["metrics"]["answer_emit_rate"] == 3 / 4  # 3 answer / 4 non-trigger

    g = evaluate_gates(m, DEFAULT_GATES)
    # recall_emit_rate 1/6 ≈ 0.167 >= 0.025 ✓
    # compress 1.0 >= 0.95 ✓
    # format 1.0 >= 0.9 ✓
    # answer 0.75 < 0.95 ✗
    assert g["A_recall_emit"]["pass"] is True
    assert g["B_compress_emit"]["pass"] is True
    assert g["C_format_compliance"]["pass"] is True
    assert g["D_answer_emit"]["pass"] is False
    assert g["overall_pass"] is False

    print("✓ freegen gate aggregation")


if __name__ == "__main__":
    test_v12_assistant_content_roundtrip()
    test_compress_trigger()
    test_tools_schema_shape()
    test_agent_special_tokens_current_only()
    test_turn_local_tools_and_action_space()
    test_pass3c_v12_emission()
    test_freegen_gate_classifier()
    test_v12_recall_multiturn_merge()
    test_v12_recall_silent_merge()
    test_v12_compress_inter_chunk_flag()
    test_pass4_v12_format_acceptance()
    test_pass4_v12_information_flow()
    test_pass4_v12_grounding_multiturn()
    test_pass4_v12_recall_evidence_reachable()
    test_pass4_v12_metadata_complete()
    test_pass4_v12_action_minimality()
    test_freegen_gate_aggregation()
    print("\n✅ all v12.0 smoke tests passed")
