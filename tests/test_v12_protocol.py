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
    test_freegen_gate_aggregation()
    print("\n✅ all v12.0 smoke tests passed")
