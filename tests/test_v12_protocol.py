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
        build_assistant_content_v12,
        parse_agent_output_v12,
    )

    # silent
    s = build_assistant_content_v12(think="x", kind="answer", answer_text="")
    p = parse_agent_output_v12(s)
    assert p["kind"] == "answer" and p["answer_text"] == "" and p["format_error"] is None

    # response
    s = build_assistant_content_v12(think="x", kind="answer", answer_text="hello")
    p = parse_agent_output_v12(s)
    assert p["kind"] == "answer" and p["answer_text"] == "hello"

    # recall tool_call
    s = build_assistant_content_v12(
        think="x", kind="recall",
        recall_query={"query": "red apron", "time_range": "10-30"},
    )
    p = parse_agent_output_v12(s)
    assert p["kind"] == "recall"
    assert p["tool_call"]["name"] == "recall"
    assert p["tool_call"]["arguments"]["query"] == "red apron"
    assert p["tool_call"]["arguments"]["time_range"] == "10-30"

    # compress tool_call
    s = build_assistant_content_v12(
        think="x", kind="compress",
        compress_summary={"time_range": [4, 12], "text": "summary"},
    )
    p = parse_agent_output_v12(s)
    assert p["kind"] == "compress"
    assert p["tool_call"]["arguments"]["time_range"] == [4, 12]
    assert p["tool_call"]["arguments"]["text"] == "summary"

    # format error: no terminal
    p = parse_agent_output_v12("<think>x</think>")
    assert p["format_error"] == "neither <answer> nor <tool_call> emitted"

    # format error: both terminals
    p = parse_agent_output_v12(
        "<think>x</think><answer>a</answer><tool_call>{}</tool_call>"
    )
    assert p["format_error"] == "both <answer> and <tool_call> present"

    # format error: bad json
    p = parse_agent_output_v12("<think>x</think><tool_call>not json</tool_call>")
    assert "JSON parse error" in p["format_error"]

    # format error: unknown tool
    p = parse_agent_output_v12(
        '<think>x</think><tool_call>{"name":"foo","arguments":{}}</tool_call>'
    )
    assert "unknown tool" in p["format_error"]

    print("✓ v12_assistant_content_roundtrip")


def test_compress_trigger():
    from thinkstream.data.agent_protocol import (
        has_compress_trigger,
        extract_compress_trigger_range,
    )

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
    assert len(TOOLS_SCHEMA) == 2
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert names == {"recall", "compress"}, f"Got: {names}"

    for tool in TOOLS_SCHEMA:
        assert tool["type"] == "function"
        f = tool["function"]
        assert "name" in f and "description" in f and "parameters" in f
        params = f["parameters"]
        assert params["type"] == "object"
        assert "properties" in params and "required" in params

    print("✓ TOOLS_SCHEMA shape")


def test_pass3c_v12_emission():
    """Current pass3c builders emit v12 sample outputs."""
    from scripts.agent_data_v5 import pass3c_samples

    s = pass3c_samples._silent_sample(
        5, "frame shows kitchen", [], "t1", card_id="c1",
        sequence_type="base",
    )
    assert s["sample_type"] == "silent"
    assert s["output"] == "<think>frame shows kitchen</think><answer></answer>"

    s = pass3c_samples._response_sample(
        5, "user asked color", "red", [], "t1", "c1",
        "immediate_response",
    )
    assert s["sample_type"] == "response"
    assert "<think>user asked color</think>" in s["output"]
    assert "<answer>red</answer>" in s["output"]

    s = pass3c_samples._recall_response_sample(
        5, "need history", "red", [],
        {"query": "red chef apron", "time_range": "10-30"},
        {"source": "historical_frames", "time": "10-30", "text_content": "red"},
        "t1", "c1", "recall",
    )
    assert s["sample_type"] == "recall"
    assert "<tool_call>" in s["v12_assistant_turn_1"]
    parsed = json.loads(
        s["v12_assistant_turn_1"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
    )
    assert parsed["name"] == "recall"
    assert parsed["arguments"]["query"] == "red chef apron"
    assert "<answer>red</answer>" in s["v12_assistant_turn_2"]

    s = pass3c_samples._compress_sample(
        8, "memory full", [], "t1",
        {
            "summary": {"time_range": [4, 12], "text": "chef cooks"}
        },
    )
    assert s["sample_type"] == "compress"
    assert s["user_input"] == "<compress_trigger/>"
    parsed = json.loads(
        s["output"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
    )
    assert parsed["name"] == "compress"
    assert parsed["arguments"]["time_range"] == [4, 12]

    print("✓ pass3c v12 emission")


def test_freegen_gate_classifier():
    """v12 gate's classify_emission categorizes outputs correctly."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    from v12_freegen_gate import classify_emission

    # silent
    c = classify_emission("<think>x</think><answer></answer>")
    assert c["category"] == "answer_silent"

    # response
    c = classify_emission("<think>x</think><answer>red</answer>")
    assert c["category"] == "answer_response"

    # recall tool
    c = classify_emission(
        '<think>x</think><tool_call>{"name":"recall","arguments":{"query":"q","time_range":"1-5"}}</tool_call>'
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
    from scripts.agent_data_v5 import pass3c_samples as pass3c

    recall = pass3c._recall_response_sample(
        5, "need history", "red", [],
        {"query": "q", "time_range": "1-5"},
        {"source": "historical_frames", "time": "1-5", "text_content": "red apron"},
        "t1", "c1", "recall",
    )
    assert "v12_assistant_turn_1" in recall
    assert "v12_assistant_turn_2" in recall
    assert "tool_call" in recall["v12_assistant_turn_1"]
    assert "<answer>red</answer>" in recall["v12_assistant_turn_2"]
    assert recall["recall_result"]["text_content"] == "red apron"
    assert recall["output"] == recall["v12_assistant_turn_2"]

    print("✓ v12 recall multi-turn merge")


def test_v12_recall_silent_merge():
    """recall_silent uses the same shape-B recall sample with empty answer."""
    from scripts.agent_data_v5 import pass3c_samples as pass3c

    r = pass3c._recall_silent_multiturn_sample(
        5, "x", [],
        {"query": "q", "time_range": "1-5"},
        {"source": "failure", "text_content": "no results"},
        "t1", "c1", "recall",
    )
    assert r["sample_type"] == "recall"
    assert r["_recall_failure"] is True
    assert r["action"] == "silent"
    assert "<answer></answer>" in r["v12_assistant_turn_2"]

    print("✓ v12 recall_silent merge")


def test_v12_compress_inter_chunk_flag():
    """v12 compress samples carry v12_inter_chunk=True flag."""
    from scripts.agent_data_v5 import pass3c_samples as pass3c

    s = pass3c._compress_sample(
        8, "full", [], "t1",
        {
            "summary": {"time_range": [4, 12], "text": "summary"}
        },
    )
    assert s.get("v12_inter_chunk") is True, (
        "compress sample should be flagged as inter-chunk in v12"
    )

    s2 = pass3c._silent_sample(
        5, "x", [], "t1", card_id="c1",
        sequence_type="base",
    )
    assert s2.get("v12_inter_chunk") is None or s2.get("v12_inter_chunk") is False

    print("✓ v12 compress inter_chunk flag")


def test_pass4_v12_format_acceptance():
    """pass4 verify_format must ACCEPT well-formed v12 samples (silent /
    response / multi-turn recall / inter-chunk compress) and REJECT
    legacy v11 <action> samples that arrive marked as v12."""
    from scripts.agent_data_v5.pass3e_verify import verify_format

    # silent — empty <answer></answer>
    silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>frame shows kitchen with chef</think><answer></answer>",
    }
    ok, reason = verify_format(silent)
    assert ok, f"silent rejected: {reason}"

    # response — non-empty <answer>
    resp = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>user asked color of chef apron, it is red</think><answer>red</answer>",
    }
    ok, reason = verify_format(resp)
    assert ok, f"response rejected: {reason}"

    # multi-turn recall
    recall = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": '<think>need history about color of apron worn earlier</think><tool_call>\n{"name":"recall","arguments":{"query":"red apron","time_range":"10-30"}}\n</tool_call>',
        "v12_assistant_turn_2": "<think>found red apron worn by chef earlier</think><answer>red</answer>",
    }
    ok, reason = verify_format(recall)
    assert ok, f"recall multi-turn rejected: {reason}"

    # inter-chunk compress
    compress = {
        "sample_type": "compress",
        "protocol_version": "v12",
        "v12_inter_chunk": True,
        "output": '<think>memory full, summarize chunks 4 to 12 of cooking</think><tool_call>\n{"name":"compress","arguments":{"time_range":[4,12],"text":"chef adds salt and pepper to pan"}}\n</tool_call>',
    }
    ok, reason = verify_format(compress)
    assert ok, f"compress rejected: {reason}"

    # bad: silent with non-empty answer
    bad_silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>nothing new in scene yet</think><answer>red</answer>",
    }
    ok, reason = verify_format(bad_silent)
    assert not ok and "v12_silent_answer_must_be_empty" in reason

    # bad: compress without inter-chunk flag
    bad_compress = {
        "sample_type": "compress",
        "protocol_version": "v12",
        # no v12_inter_chunk flag
        "output": '<think>x is happening here</think><tool_call>\n{"name":"compress","arguments":{"time_range":[4,12],"text":"summary content"}}\n</tool_call>',
    }
    ok, reason = verify_format(bad_compress)
    assert not ok and "v12_compress_missing_inter_chunk_flag" in reason

    # bad: tool_call invalid JSON
    bad_json = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history</think><tool_call>not json</tool_call>",
        "v12_assistant_turn_2": "<think>x</think><answer>red</answer>",
    }
    ok, reason = verify_format(bad_json)
    assert not ok and "invalid_json" in reason

    print("✓ pass4 verify_format v12 acceptance/rejection")


def test_pass4_v12_information_flow():
    """v12 information_flow validates yes/no/MC/number response strict format."""
    from scripts.agent_data_v5.pass3e_verify import verify_information_flow

    # binary form — must be exactly Yes/No
    good_binary = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>asks if van is in scene, I see white van clearly</think><answer>Yes</answer>",
        "metadata": {"answer_form": "binary"},
    }
    ok, reason = verify_information_flow(good_binary)
    assert ok, f"good binary rejected: {reason}"

    bad_binary = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>asks if van is in scene currently visible</think><answer>yes definitely</answer>",
        "metadata": {"answer_form": "binary"},
    }
    ok, reason = verify_information_flow(bad_binary)
    assert not ok and "binary_response_not_yes_no" in reason

    # number form
    bad_number = {
        "sample_type": "response",
        "protocol_version": "v12",
        "output": "<think>counted three apples carefully now</think><answer>three</answer>",
        "metadata": {"answer_form": "number"},
    }
    ok, reason = verify_information_flow(bad_number)
    assert not ok and "number_response_not_digits" in reason

    # silent samples should not trip empty-response check
    silent = {
        "sample_type": "silent",
        "protocol_version": "v12",
        "output": "<think>nothing new visible in current chunk</think><answer></answer>",
        "metadata": {},
    }
    ok, reason = verify_information_flow(silent)
    assert ok, f"silent rejected: {reason}"

    print("✓ pass4 verify_information_flow v12")


def test_pass4_v12_grounding_multiturn():
    """verify_grounding must read both turns of v12 multi-turn recall samples."""
    from scripts.agent_data_v5.pass3e_verify import verify_grounding

    # Multi-turn recall sample — output popped, turns in v12_assistant_turn_*
    multi_recall = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history about chef apron color</think><tool_call>{}</tool_call>",
        "v12_assistant_turn_2": "<think>found red apron in earlier scene</think><answer>red</answer>",
    }
    ok, reason = verify_grounding(multi_recall)
    assert ok, f"v12 multi-turn recall grounding rejected: {reason}"

    # Multi-turn with non-visual phrase in turn 2 should be caught
    bad_multi = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "v12_assistant_turn_1": "<think>need history about chef apron color</think><tool_call>{}</tool_call>",
        "v12_assistant_turn_2": "<think>chef tastes the dish smells aromatic</think><answer>red</answer>",
    }
    ok, reason = verify_grounding(bad_multi)
    assert not ok and ("smell" in reason or "aroma" in reason)

    print("✓ pass4 verify_grounding v12 multi-turn")


def test_pass4_v12_recall_evidence_reachable():
    """verify_recall_evidence_reachable must trigger on v12 sample_type='recall'."""
    from scripts.agent_data_v5.pass3e_verify import verify_recall_evidence_reachable

    # v12 recall with evidence in future → should fail
    bad = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "card_id": "c1",
        "chunk_idx": 5,
        "metadata": {"support_chunks": [10]},  # future evidence
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
    }
    ok, reason = verify_recall_evidence_reachable(good)
    assert ok, reason

    print("✓ pass4 verify_recall_evidence_reachable v12 sample_type=recall")


def test_pass4_v12_metadata_complete():
    """verify_metadata_complete must trigger on v12 sample_type='recall'."""
    from scripts.agent_data_v5.pass3e_verify import verify_metadata_complete

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
    """verify_action_minimality must trigger on v12 sample_type='recall'."""
    from scripts.agent_data_v5.pass3e_verify import verify_action_minimality

    # v12 recall in non-recall sequence → should fail
    bad = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "immediate_response",  # NOT a recall seq
        "metadata": {},
    }
    ok, reason = verify_action_minimality(bad)
    assert not ok and "non_recall_sequence" in reason

    # v12 recall in valid sequence → pass
    good = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "recall_success",
        "metadata": {"visibility": {}},
    }
    ok, reason = verify_action_minimality(good)
    assert ok, reason

    # v12 recall with answer-already-visible visibility flag → fail
    bad_vis = {
        "sample_type": "recall",
        "protocol_version": "v12",
        "sequence_type": "recall_success",
        "metadata": {"visibility": {"answer_in_recent_obs": True}},
    }
    ok, reason = verify_action_minimality(bad_vis)
    assert not ok and "recall_unnecessary_answer_in_observations" in reason

    print("✓ pass4 verify_action_minimality v12 sample_type=recall")


def test_pass4_v11_backward_compat():
    """v11 backward-compat path was removed when the codebase consolidated
    on v12. Kept as a skipped placeholder for traceability."""
    import pytest
    pytest.skip("v11 protocol removed; pass3e_verify now exercises v12 only.")


def test_freegen_gate_aggregation():
    """End-to-end gate: synthetic samples + classifications → metrics + verdict."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    from v12_freegen_gate import aggregate_gate_metrics, evaluate_gates, DEFAULT_GATES

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
