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
    """End-to-end: pass3c v12 mode emits correct sample.output for each
    sample_type, with compress_trigger injected into user_input on
    compress samples."""
    os.environ["THINKSTREAM_PROTOCOL"] = "v12"
    # Re-import to pick up env var. Done at module level via PROTOCOL_VERSION.
    if "scripts.agent_data_v5.pass3c_samples" in sys.modules:
        del sys.modules["scripts.agent_data_v5.pass3c_samples"]
    from scripts.agent_data_v5 import pass3c_samples
    assert pass3c_samples.PROTOCOL_VERSION == "v12"

    _make_sample = pass3c_samples._make_sample

    # silent
    s = _make_sample(
        chunk_idx=5, prompt_type="ASK_PROMPT", action="silent",
        think="frame shows kitchen", queries=[], snapshot=None,
        trajectory_id="t1", card_id="c1", sequence_type="base",
    )
    assert s["sample_type"] == "silent"
    assert s["output"] == "<think>frame shows kitchen</think><answer></answer>"
    assert s["protocol_version"] == "v12"

    # response
    s = _make_sample(
        chunk_idx=5, prompt_type="ASK_PROMPT", action="response",
        think="user asked color", queries=[], response="red",
        trajectory_id="t1", card_id="c1", sequence_type="immediate_response",
    )
    assert s["sample_type"] == "response"
    assert "<think>user asked color</think>" in s["output"]
    assert "<answer>red</answer>" in s["output"]

    # recall tool_call
    s = _make_sample(
        chunk_idx=5, prompt_type="ASK_PROMPT", action="recall",
        think="need history", queries=[],
        query={"query": "red chef apron", "time_range": "10-30"},
        trajectory_id="t1", card_id="c1", sequence_type="recall",
    )
    assert s["sample_type"] == "recall_query"
    assert "<tool_call>" in s["output"]
    parsed = json.loads(
        s["output"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
    )
    assert parsed["name"] == "recall"
    assert parsed["arguments"]["query"] == "red chef apron"

    # compress tool_call + trigger injection
    s = _make_sample(
        chunk_idx=8, prompt_type="ASK_PROMPT", action="compress",
        think="memory full", queries=[],
        snapshot={"_compress_event": {
            "summary": {"time_range": [4, 12], "text": "chef cooks"}
        }},
        user_input="continue",
        trajectory_id="t1", card_id="c1", sequence_type="compress",
    )
    assert s["sample_type"] == "compress"
    assert "<compress_trigger range='4-12'/>" in s["user_input"]
    assert "continue" in s["user_input"]  # original user_input preserved
    parsed = json.loads(
        s["output"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
    )
    assert parsed["name"] == "compress"
    assert parsed["arguments"]["time_range"] == [4, 12]

    print("✓ pass3c v12 emission")


def test_pass3c_v11_unchanged():
    """v11 mode (default) must remain byte-identical to pre-v12.0 output
    for compress samples — same <action>compress</action><summary>{...}.
    """
    import importlib
    os.environ["THINKSTREAM_PROTOCOL"] = "v11"
    # Drop both the module and its parent package cache so the re-import
    # re-evaluates module-level PROTOCOL_VERSION = os.environ[...].
    for mod_name in list(sys.modules):
        if mod_name.startswith("scripts.agent_data_v5"):
            del sys.modules[mod_name]
    pass3c_samples = importlib.import_module("scripts.agent_data_v5.pass3c_samples")
    assert pass3c_samples.PROTOCOL_VERSION == "v11"

    s = pass3c_samples._make_sample(
        chunk_idx=8, prompt_type="ASK_PROMPT", action="compress",
        think="mem full", queries=[],
        snapshot={"_compress_event": {
            "summary": {"time_range": [4, 12], "text": "chef cooks"}
        }},
        trajectory_id="t1", card_id="c1", sequence_type="compress",
    )
    assert "<action>compress</action>" in s["output"]
    assert "<summary>" in s["output"]
    assert "<compress_trigger" not in (s["user_input"] or "")
    assert s["protocol_version"] == "v11"

    print("✓ pass3c v11 backward compat")


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
    """Merge function pairs (recall_query, recall_response) at same chunk into
    one multi-turn sample with v12_assistant_turn_1/2 fields."""
    import importlib
    os.environ["THINKSTREAM_PROTOCOL"] = "v12"
    for mod_name in list(sys.modules):
        if mod_name.startswith("scripts.agent_data_v5"):
            del sys.modules[mod_name]
    pass3c = importlib.import_module("scripts.agent_data_v5.pass3c_samples")
    assert pass3c.PROTOCOL_VERSION == "v12"

    # Mock samples — 1 unrelated silent + a (recall_query, recall_response) pair
    # at same chunk + a lonely compress.
    samples = [
        {
            "chunk_idx": 3, "sample_type": "silent", "trajectory_id": "t1",
            "card_id": "", "output": "<think>x</think><answer></answer>",
            "queries": [], "user_input": "", "recall_result": None,
            "protocol_version": "v12",
        },
        {
            "chunk_idx": 5, "sample_type": "recall_query", "trajectory_id": "t1",
            "card_id": "c1",
            "output": '<think>need history</think><tool_call>\n{"name":"recall","arguments":{"query":"q","time_range":"1-5"}}\n</tool_call>',
            "queries": [], "user_input": "what color", "recall_result": None,
            "protocol_version": "v12",
        },
        {
            "chunk_idx": 5, "sample_type": "recall_response", "trajectory_id": "t1",
            "card_id": "c1",
            "output": "<think>found</think><answer>red</answer>",
            "queries": [], "user_input": "",
            "recall_result": {"source": "historical_frames", "time": "1-5", "text_content": "red apron"},
            "protocol_version": "v12",
        },
        {
            "chunk_idx": 8, "sample_type": "compress", "trajectory_id": "t1",
            "card_id": "", "output": '<think>full</think><tool_call>\n{"name":"compress","arguments":{"time_range":[4,12],"text":"s"}}\n</tool_call>',
            "queries": [], "user_input": "<compress_trigger range='4-12'/>",
            "recall_result": None, "v12_inter_chunk": True,
            "protocol_version": "v12",
        },
    ]

    merged = pass3c._merge_recall_pairs_v12(samples)

    # Expected: 1 silent (chunk 3) + 1 merged recall (chunk 5) + 1 compress (chunk 8)
    assert len(merged) == 3, f"Got {len(merged)} samples: {[s.get('sample_type') for s in merged]}"
    types = sorted([s["sample_type"] for s in merged])
    assert types == ["compress", "recall", "silent"]

    recall = next(s for s in merged if s["sample_type"] == "recall")
    assert "v12_assistant_turn_1" in recall
    assert "v12_assistant_turn_2" in recall
    assert "tool_call" in recall["v12_assistant_turn_1"]
    assert "<answer>red</answer>" in recall["v12_assistant_turn_2"]
    assert recall["recall_result"]["text_content"] == "red apron"
    assert recall["v12_post_recall_was_silent"] is False
    # Original 'output' field should be removed to prevent ambiguity
    assert "output" not in recall

    print("✓ v12 recall multi-turn merge")


def test_v12_recall_silent_merge():
    """recall_silent merges into recall sample with empty answer."""
    import importlib
    os.environ["THINKSTREAM_PROTOCOL"] = "v12"
    for mod_name in list(sys.modules):
        if mod_name.startswith("scripts.agent_data_v5"):
            del sys.modules[mod_name]
    pass3c = importlib.import_module("scripts.agent_data_v5.pass3c_samples")

    samples = [
        {
            "chunk_idx": 5, "sample_type": "recall_query", "trajectory_id": "t1",
            "card_id": "c1", "output": "<think>x</think><tool_call>...</tool_call>",
            "queries": [], "user_input": "", "recall_result": None,
            "protocol_version": "v12",
        },
        {
            "chunk_idx": 5, "sample_type": "recall_silent", "trajectory_id": "t1",
            "card_id": "c1", "output": "<think>not found</think><answer></answer>",
            "queries": [], "user_input": "",
            "recall_result": {"source": "failure", "text_content": "no results"},
            "protocol_version": "v12",
        },
    ]
    merged = pass3c._merge_recall_pairs_v12(samples)
    assert len(merged) == 1
    r = merged[0]
    assert r["sample_type"] == "recall"
    assert r["v12_post_recall_was_silent"] is True
    assert "<answer></answer>" in r["v12_assistant_turn_2"]

    print("✓ v12 recall_silent merge")


def test_v12_compress_inter_chunk_flag():
    """v12 compress samples carry v12_inter_chunk=True flag."""
    import importlib
    os.environ["THINKSTREAM_PROTOCOL"] = "v12"
    for mod_name in list(sys.modules):
        if mod_name.startswith("scripts.agent_data_v5"):
            del sys.modules[mod_name]
    pass3c = importlib.import_module("scripts.agent_data_v5.pass3c_samples")

    s = pass3c._make_sample(
        chunk_idx=8, prompt_type="ASK_PROMPT", action="compress",
        think="full", queries=[],
        snapshot={"_compress_event": {
            "summary": {"time_range": [4, 12], "text": "summary"}
        }},
        trajectory_id="t1", card_id="c1", sequence_type="compress",
    )
    assert s.get("v12_inter_chunk") is True, (
        "compress sample should be flagged as inter-chunk in v12"
    )

    # Non-compress samples should NOT have the flag.
    s2 = pass3c._make_sample(
        chunk_idx=5, prompt_type="ASK_PROMPT", action="silent",
        think="x", queries=[], trajectory_id="t1", card_id="c1",
        sequence_type="base",
    )
    assert s2.get("v12_inter_chunk") is None or s2.get("v12_inter_chunk") is False

    print("✓ v12 compress inter_chunk flag")


def test_pass4_v12_format_acceptance():
    """pass4 verify_format must ACCEPT well-formed v12 samples (silent /
    response / multi-turn recall / inter-chunk compress) and REJECT
    legacy v11 <action> samples that arrive marked as v12."""
    from scripts.agent_data_v5.pass4 import verify_format

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
    from scripts.agent_data_v5.pass4 import verify_information_flow

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
    from scripts.agent_data_v5.pass4 import verify_grounding

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
    from scripts.agent_data_v5.pass4 import verify_recall_evidence_reachable

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
    from scripts.agent_data_v5.pass4 import verify_metadata_complete

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
    from scripts.agent_data_v5.pass4 import verify_action_minimality

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
    """v11 samples (no protocol_version field) still go through legacy validation."""
    from scripts.agent_data_v5.pass4 import verify_format

    v11_resp = {
        "sample_type": "response",
        # NO protocol_version field — legacy default
        "output": "<think>chef is adding salt slowly to the pot at the stove</think><action>response</action><response>red</response>",
    }
    ok, reason = verify_format(v11_resp)
    # v11 verify_format requires <action>, this passes
    assert ok, f"v11 response rejected: {reason}"

    print("✓ pass4 v11 backward compat")


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
    test_pass3c_v11_unchanged()
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
    test_pass4_v11_backward_compat()
    test_freegen_gate_aggregation()
    print("\n✅ all v12.0 smoke tests passed")
