import asyncio

from scripts.agent_data import pass2_rollout
from scripts.agent_data.evidence_think import build_think_from_pass1_evidence
from scripts.agent_data.pass2b_rescue_from_pass1 import (
    run_pass2b_single_video,
)


def test_build_think_from_pass1_evidence_is_current_only():
    cap = {
        "chunk_idx": 12,
        "visible_entities": [
            {
                "desc": "red ceramic bowl on a wooden counter",
                "action": "static",
                "position": "center",
            },
            {
                "desc": "right hand holding a silver spoon",
                "action": "stirring batter",
                "position": "right foreground",
            },
        ],
        "atomic_facts": [
            {"fact": "the spoon is inside the red bowl"},
            "pale batter is visible against the spoon",
        ],
        "ocr": ["MIX"],
        "spatial": "The hand is right of the bowl. The spoon is inside the bowl.",
        "state_changes": ["spoon entered the bowl"],
    }

    think = build_think_from_pass1_evidence(cap)

    assert "red ceramic bowl" in think
    assert "stirring batter" in think
    assert "MIX" in think
    assert "<frame" not in think


def test_pass2b_rollout_schema_and_compression(monkeypatch):
    # Keep token counting deterministic and avoid loading a local model tokenizer
    # in unit tests.
    monkeypatch.setattr(pass2_rollout, "get_tokenizer", lambda: None)

    long_fact = " ".join(["unique current visual fact"] * 80)
    evidence = []
    for i in range(70):
        evidence.append({
            "chunk_idx": i,
            "visible_entities": [
                {
                    "desc": f"object_{i} with blue marker",
                    "action": "static",
                    "position": "center",
                }
            ],
            "atomic_facts": [f"{long_fact} {i}"],
            "ocr": [],
            "spatial": "",
            "state_changes": [],
        })

    rollout = asyncio.run(run_pass2b_single_video(
        video_id="unit_video",
        evidence=evidence,
        compress_mode="deterministic",
    ))

    assert rollout["video_id"] == "unit_video"
    assert rollout["rollout_source"] == "pass2b_rescue_from_pass1"
    assert rollout["num_chunks"] == 70
    assert len(rollout["thinks"]) == 70
    assert rollout["snapshots"][0]["recent_thinks"] == []
    assert rollout["compression_events"]
    assert rollout["compression_events"][0]["summary"]["parse_success"] is True

    final_think_chunks = [
        item["chunk"]
        for item in rollout["final_memory"]["timeline"]
        if item.get("type") == "think"
    ]
    assert len(final_think_chunks) == len(set(final_think_chunks))
    assert 69 in final_think_chunks
