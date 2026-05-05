import random
import json
import re
import asyncio

import pytest

from scripts.agent_data_v5.v2.design import (
    Card,
    GoldEmit,
    Placement,
    assign_recall_noise,
    render_video_samples,
    select_trajectory,
)
from thinkstream.data.agent_protocol import format_queries_block


def _card(card_id: str, family: str = "C1") -> Card:
    return Card(
        card_id=card_id,
        family=family,
        question=f"Question {card_id}?",
        answer_form="short_exact",
        question_type="single_emit",
        gold_emits=[GoldEmit(chunk=0, value="answer")],
        grounding_frames=[0],
    )


def _placement(card_id: str, ask: int, chunks: range) -> Placement:
    actions = {
        c: ("response", "answer") if c == max(chunks) - 1 else ("silent", "")
        for c in chunks
    }
    return Placement(
        card_id=card_id,
        ask_chunk=ask,
        mechanism="silent_then_response",
        chunk_actions=actions,
    )


def test_select_trajectory_prevents_overlapping_active_windows():
    cards = [_card("a", "C1"), _card("b", "N1"), _card("c", "M1")]
    placements = {
        "a": [_placement("a", 5, range(5, 11))],
        "b": [_placement("b", 8, range(8, 13))],
        "c": [_placement("c", 20, range(20, 22))],
    }

    selected = select_trajectory(
        cards, placements, num_chunks=30, rng=random.Random(0), max_q=3
    )

    used = set()
    for p in selected:
        chunks = set(p.chunk_actions)
        assert not chunks & used
        used.update(chunks)
    assert len(selected) == 2


def test_select_trajectory_applies_recall_floor():
    cards = [_card(f"r{i}", "N1") for i in range(4)]
    cards += [_card(f"d{i}", "C1") for i in range(4)]
    placements = {
        **{
            f"r{i}": [
                Placement(
                    card_id=f"r{i}",
                    ask_chunk=10 + i * 10,
                    mechanism="recall_demo",
                    chunk_actions={
                        10 + i * 10: ("response", "answer"),
                        11 + i * 10: ("silent", ""),
                    },
                )
            ]
            for i in range(4)
        },
        **{
            f"d{i}": [
                Placement(
                    card_id=f"d{i}",
                    ask_chunk=50 + i * 3,
                    mechanism="direct",
                    chunk_actions={
                        50 + i * 3: ("response", "answer"),
                        51 + i * 3: ("silent", ""),
                    },
                )
            ]
            for i in range(4)
        },
    }

    selected = select_trajectory(
        cards, placements, num_chunks=80, rng=random.Random(0), max_q=6
    )

    assert sum(p.mechanism == "recall_demo" for p in selected) >= 3


def test_render_video_samples_rejects_shared_answer_chunk():
    cards = [_card("a", "C1"), _card("b", "N1")]
    placements = {
        "a": [_placement("a", 4, range(4, 7))],
        "b": [_placement("b", 5, range(5, 7))],
    }

    with pytest.raises(ValueError, match="overlapping question placements"):
        render_video_samples(cards, placements, num_chunks=10)


def test_recall_noise_never_creates_failure_samples():
    placements = [
        Placement(
            card_id=f"r{i}",
            ask_chunk=i,
            mechanism="recall_demo",
            chunk_actions={i: ("response", "answer")},
        )
        for i in range(200)
    ]

    assign_recall_noise(placements, random.Random(0))

    kinds = {kind for p in placements for kind in p.recall_at.values()}
    assert kinds <= {"oracle", "noisy"}
    assert "failure" not in kinds


def test_forward_question_gets_nonterminal_recall_silent():
    placement = Placement(
        card_id="f0",
        ask_chunk=5,
        mechanism="silent_then_response",
        chunk_actions={
            5: ("silent", ""),
            6: ("silent", ""),
            7: ("response", "answer"),
            8: ("silent", ""),
        },
    )

    assign_recall_noise([placement], random.Random(0))
    samples = render_video_samples(
        [_card("f0", "E2")],
        {"f0": [placement]},
        num_chunks=10,
        rng=random.Random(1),
    )

    by_chunk = {s.chunk_idx: s for s in samples if s.card_id == "f0"}
    assert by_chunk[5].sample_kind == "recall+silent"
    assert by_chunk[5].recall_result_kind == "not_yet"
    assert by_chunk[7].sample_kind == "response"


def test_recall_silent_query_does_not_leak_future_grounding_range():
    from scripts.agent_data_v5.pass3c_samples import generate_trajectory_samples

    card = {
        "card_id": "f0",
        "family": "E2",
        "question": "What object appears next?",
        "answer_form": "short_exact",
        "question_type": "single_emit",
        "canonical_answer": "red cup",
        "gold_emits": [{"chunk": 7, "value": "red cup"}],
        "grounding_frames": [7],
    }
    trajectory = {
        "trajectory_id": "t0",
        "placements": [{
            "card_id": "f0",
            "ask_chunk": 5,
            "mechanism": "silent_then_response",
            "chunk_actions": {
                "5": ["silent", ""],
                "6": ["silent", ""],
                "7": ["response", "red cup"],
                "8": ["silent", ""],
            },
            "recall_at": {"5": "not_yet"},
        }],
    }
    rollout = {
        "num_chunks": 10,
        "thinks": [{"chunk_idx": i, "think": f"chunk {i}"} for i in range(10)],
    }

    samples = asyncio.run(generate_trajectory_samples(
        trajectory,
        {"f0": card},
        rollout,
        evidence=[],
        client=None,
        video_id="vid",
    ))

    wait_sample = next(
        s for s in samples
        if s.get("card_id") == "f0" and s.get("chunk_idx") == 5
    )
    m = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        wait_sample["v12_assistant_turn_1"],
        re.S,
    )
    payload = json.loads(m.group(1))
    args = payload["arguments"]
    assert args["time_range"] == "0-5"
    assert args["time_range"] != "7-8"
    assert wait_sample["recall_result"]["source"] == "memory"
    assert wait_sample["queries"][0]["status"] == "open"

    response_sample = next(
        s for s in samples
        if s.get("card_id") == "f0" and s.get("chunk_idx") == 7
    )
    assert response_sample["action"] == "response"
    assert response_sample["queries"][0]["status"] == "open"


def test_recall_failure_injection_is_rejected():
    card = _card("r0", "N1")
    placement = Placement(
        card_id="r0",
        ask_chunk=5,
        mechanism="recall_demo",
        chunk_actions={5: ("response", "answer")},
        recall_at={5: "failure"},
    )

    with pytest.raises(ValueError, match="recall failure is disabled"):
        render_video_samples([card], {"r0": [placement]}, num_chunks=10)


def test_card_with_canonical_but_no_gold_emit_is_rejected():
    from scripts.agent_data_v5.pass3a_cards import _verify_card_layers

    card = {
        "card_id": "c0",
        "family": "C1",
        "question": "What is visible at this moment?",
        "answer_form": "short_exact",
        "canonical_answer": "red bowl",
        "gold_emits": [],
        "grounding_frames": [0],
    }

    assert _verify_card_layers(card, {}) == "schema_no_gold_emits"


def test_open_multi_answer_query_stays_open_after_first_answer():
    text = format_queries_block([
        {
            "question": "Report each event.",
            "ask_time": 10,
            "status": "open",
            "answers": [{"time": 12, "text": "first event"}],
        }
    ])

    assert "Still open" in text
    assert "[12s] A: first event" in text
