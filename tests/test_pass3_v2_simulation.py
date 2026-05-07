"""Synthetic pass3 v2 logic checks.

Run directly:
  python tests/test_pass3_v2_simulation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data_v5.pass3c_samples import generate_trajectory_samples
from scripts.agent_data_v5.v2.design import Card, GoldEmit, placement_timing_verdict
from scripts.agent_data_v5.v2.simulate import aggregate, assert_quality, simulate_one_video


def _evidence(num_chunks: int, facts_by_chunk: dict[int, str]) -> list[dict]:
    rows = []
    for c in range(num_chunks):
        fact = facts_by_chunk.get(c, f"neutral background observation {c}")
        rows.append({
            "chunk_idx": c,
            "think": fact,
            "visible_entities": [{"desc": fact, "action": "visible", "id": f"e{c}"}],
            "atomic_facts": [{"fact": fact, "confidence": 1.0}],
            "ocr": [],
            "spatial": "",
            "state_changes": [],
        })
    return rows


def _mc_card(
    family: str,
    idx: int,
    *,
    emit: int,
    answer: str,
    correct: str = "A",
    question_type: str = "single_emit",
) -> Card:
    options = [
        f"A) {answer}",
        f"B) distractor {idx}b",
        f"C) distractor {idx}c",
        f"D) distractor {idx}d",
    ]
    if correct != "A":
        options = [
            "A) distractor a",
            "B) distractor b",
            "C) distractor c",
            f"D) {answer}",
        ]
    return Card(
        card_id=f"synthetic_{family}_{idx}",
        family=family,
        question=f"What historical detail answers synthetic {family} question {idx}?",
        answer_form="multiple_choice",
        question_type=question_type,  # type: ignore[arg-type]
        gold_emits=[GoldEmit(chunk=emit, value=correct)],
        grounding_frames=[emit],
        options=options,
        correct_option=correct,
    )


def _card_dict(card: Card) -> dict:
    return {
        "card_id": card.card_id,
        "family": card.family,
        "question": card.question,
        "answer_form": card.answer_form,
        "question_type": card.question_type,
        "gold_emits": [
            {"chunk": int(e.chunk), "value": str(e.value)}
            for e in card.gold_emits
        ],
        "grounding_frames": list(card.grounding_frames),
        "support_chunks": list(card.grounding_frames),
        "options": list(card.options or []),
        "correct_option": card.correct_option,
        "canonical_answer": (
            "Unable to answer"
            if card.family == "HLD1"
            else str(card.options[0]).split(")", 1)[-1].strip()
        ),
    }


def test_synthetic_pass3_recall_quality():
    num_chunks = 180
    hard_families = ["CR1", "CR2", "CR4", "CR5", "M1", "C1", "STU1", "OJR1", "CR7"]
    facts: dict[int, str] = {}
    cards: list[Card] = []
    for i, fam in enumerate(hard_families):
        emit = 8 + i * 7
        answer = f"rare visual token {fam.lower()} {i}"
        facts[emit] = f"The historical frame contains {answer}, visible only at this moment."
        cards.append(_mc_card(fam, i, emit=emit, answer=answer))

    # Simple memory family: answer is explicitly in historical text, so recall
    # candidates should be downgraded to memory_direct instead of selected.
    facts[84] = "The easy memory fact says the green logo is visible."
    cards.append(_mc_card("N1", 100, emit=84, answer="green logo"))

    # HLD/unanswerable cards are useful negatives, but never successful recall.
    facts[96] = "This frame intentionally does not support any concrete answer."
    cards.append(_mc_card("HLD1", 101, emit=96, answer="Unable to answer", correct="D"))

    result = simulate_one_video(
        "synthetic_pass3",
        _evidence(num_chunks, facts),
        num_chunks,
        seed=7,
        cards_override=cards,
    )
    agg = aggregate([result])
    failures = assert_quality(
        agg,
        min_recall_question_pct=50.0,
        max_recall_question_pct=93.0,
    )
    assert not failures, failures

    cards_by_id = {c.card_id: c for c in result["cards"]}
    selected = result["placements"]
    assert selected, "synthetic simulation selected no placements"
    assert any(p.mechanism == "recall_demo" for p in selected), (
        "high-value historical cards should produce recall_demo placements"
    )

    used_chunks = set()
    for p in selected:
        card = cards_by_id[p.card_id]
        ok, reason = placement_timing_verdict(card, p)
        assert ok, f"{p.card_id} timing failed: {reason}"
        overlap = used_chunks & set(int(c) for c in p.chunk_actions)
        assert not overlap, f"overlap at chunks {sorted(overlap)}"
        used_chunks.update(int(c) for c in p.chunk_actions)
        if card.family in {"HLD1", "N1"}:
            assert p.mechanism != "recall_demo", (
                f"{card.family} should not be selected as high-value recall"
            )


def test_pass3c_downgrades_stale_bad_recall_slots():
    """pass3c should repair stale 3b recall slots before rendering samples."""
    hld = _mc_card("HLD1", 201, emit=12, answer="Unable to answer", correct="D")
    easy = _mc_card("N1", 202, emit=20, answer="green logo")
    cards = {_card.card_id: _card_dict(_card) for _card in [hld, easy]}
    trajectory = {
        "trajectory_id": "traj_stale",
        "placements": [
            {
                "card_id": hld.card_id,
                "ask_chunk": 40,
                "mechanism": "recall_demo",
                "difficulty_mode": "recall_mid",
                "recall_need": "stale_unanswerable",
                "chunk_actions": {"40": ["response", "D"], "41": ["silent", ""]},
                "recall_at": {"40": "oracle"},
            },
            {
                "card_id": easy.card_id,
                "ask_chunk": 60,
                "mechanism": "recall_demo",
                "difficulty_mode": "recall_mid",
                "recall_need": "stale_memory_answerable",
                "chunk_actions": {"60": ["response", "A"], "61": ["silent", ""]},
                "recall_at": {"60": "oracle"},
            },
        ],
    }
    rollout = {
        "num_chunks": 80,
        "thinks": [
            {"chunk_idx": c, "think": f"neutral observation {c}"}
            for c in range(80)
        ],
        "snapshots": {
            "40": {"recent_thinks": [{"time": "0-40", "text": "ordinary prior memory"}]},
            "60": {"recent_thinks": [{"time": "0-60", "text": "the green logo was visible earlier"}]},
        },
        "compression_events": [],
    }
    evidence = _evidence(80, {
        12: "No concrete answer is supported here.",
        20: "The easy memory fact says the green logo is visible.",
    })
    samples = asyncio.run(generate_trajectory_samples(
        trajectory,
        cards,
        rollout,
        evidence,
        client=None,
        video_id="synthetic_pass3c",
    ))
    by_card = {s.get("card_id"): s for s in samples if s.get("sample_type") == "response"}
    assert hld.card_id in by_card, "HLD stale recall should render as direct response negative"
    assert easy.card_id in by_card, "memory-answerable stale recall should render as response"
    assert not [
        s for s in samples
        if s.get("sample_type") == "recall" and s.get("card_id") in {hld.card_id, easy.card_id}
    ], "stale bad recall slots must be downgraded before pass3c emits samples"


if __name__ == "__main__":
    test_synthetic_pass3_recall_quality()
    test_pass3c_downgrades_stale_bad_recall_slots()
    print("PASS synthetic pass3 recall-quality simulation")
