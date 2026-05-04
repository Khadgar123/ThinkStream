import random

import pytest

from scripts.agent_data_v5.v2.design import (
    Card,
    GoldEmit,
    Placement,
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
