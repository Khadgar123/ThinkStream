import random
import json
import re
import asyncio

import pytest

from scripts.agent_data.placement.design import (
    Card,
    GoldEmit,
    Placement,
    assign_recall_noise,
    render_video_samples,
    select_trajectory,
)
from thinkstream.data.agent_protocol import format_queries_block


def _card(card_id: str, family: str = "C1") -> Card:
    question = f"Question {card_id}?"
    if family == "E2":
        question = "When the target appears, output answer."
    return Card(
        card_id=card_id,
        family=family,
        question=question,
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
    cards = [_card("a", "E2"), _card("b", "E2"), _card("c", "E2")]
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
    from scripts.agent_data.placement import design

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

    recall_cap = min(6, max(0, int(6 * design.RECALL_MAX_FRACTION + 0.999)))
    expected_floor = min(recall_cap, design._recall_floor(6))
    assert sum(p.mechanism == "recall_demo" for p in selected) >= expected_floor


def test_render_video_samples_rejects_shared_answer_chunk():
    cards = [_card("a", "C1"), _card("b", "N1")]
    placements = {
        "a": [_placement("a", 4, range(4, 7))],
        "b": [_placement("b", 5, range(5, 7))],
    }

    with pytest.raises(ValueError, match="overlapping question placements"):
        render_video_samples(cards, placements, num_chunks=10)


def test_render_video_samples_emits_dense_silent_timeline():
    cards = [_card("a", "C1")]
    placements = {"a": [_placement("a", 4, range(4, 7))]}

    samples = render_video_samples(
        cards,
        placements,
        num_chunks=10,
        evidence=[],
        rng=random.Random(0),
    )

    assert [s.chunk_idx for s in samples] == list(range(10))
    by_chunk = {s.chunk_idx: s for s in samples}
    assert by_chunk[0].sample_kind == "silent"
    assert by_chunk[4].sample_kind == "silent"
    assert by_chunk[5].sample_kind == "response"


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


def test_forward_question_waits_without_nonterminal_recall():
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
    assert by_chunk[5].sample_kind == "silent"
    assert by_chunk[5].recall_result_kind is None
    assert by_chunk[7].sample_kind == "response"


def test_recall_silent_query_does_not_leak_future_grounding_range():
    from scripts.agent_data.pass3c_samples import generate_trajectory_samples

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
    if "v12_assistant_turn_1" not in wait_sample:
        assert wait_sample["action"] == "silent"
        assert wait_sample.get("recall_result") is None
        return
    m = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        wait_sample["v12_assistant_turn_1"],
        re.S,
    )
    payload = json.loads(m.group(1))
    args = payload["arguments"]
    assert args["start_time"] == 0
    assert args["end_time"] < wait_sample["chunk_idx"]
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


def test_recall_validity_ignores_compact_memory_answer_overlap():
    from scripts.agent_data.pass3c_samples import (
        _needs_recall_hardening,
        _validate_hardened_recall_card,
    )

    card = {
        "card_id": "r-memory",
        "family": "F1",
        "question": "What object was on the table earlier?",
        "answer_form": "short_exact",
        "canonical_answer": "red cup",
        "gold_emits": [{"chunk": 1, "value": "red cup"}],
        "grounding_frames": [1],
        "recall_query": {"start_time": 1, "end_time": 1},
    }
    evidence = {
        1: {
            "chunk_idx": 1,
            "atomic_facts": [{"fact": "A red cup is on the table."}],
        }
    }
    memory_text = '  <m t="0-10">The red cup was on the table.</m>'

    assert _needs_recall_hardening(card, memory_text) is False
    ok, reason = _validate_hardened_recall_card(
        card,
        current_chunk=20,
        memory_text=memory_text,
        evidence_by_chunk=evidence,
    )
    assert ok, reason


def test_recall_support_must_be_strictly_outside_8s_window():
    from scripts.agent_data.pass3c_samples import _validate_hardened_recall_card

    card = {
        "card_id": "r-window",
        "family": "F1",
        "question": "What object was shown earlier?",
        "answer_form": "short_exact",
        "canonical_answer": "blue bag",
        "gold_emits": [{"chunk": 12, "value": "blue bag"}],
        "grounding_frames": [12],
        "recall_query": {"start_time": 12, "end_time": 12},
    }
    evidence = {
        12: {
            "chunk_idx": 12,
            "atomic_facts": [{"fact": "A blue bag is shown."}],
        }
    }

    ok, reason = _validate_hardened_recall_card(
        card,
        current_chunk=20,
        memory_text="",
        evidence_by_chunk=evidence,
    )
    assert not ok
    assert reason.startswith("grounding_inside_visual_window")


def test_card_with_canonical_but_no_gold_emit_is_rejected():
    from scripts.agent_data.pass3a_cards import _verify_card_layers

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


def test_hld1_rejects_supported_concrete_option():
    from scripts.agent_data.pass3a_cards import _verify_card_layers

    card = {
        "card_id": "h0",
        "family": "HLD1",
        "question": "What color is the shirt worn by the person?",
        "answer_form": "multiple_choice",
        "question_type": "single_emit",
        "canonical_answer": "Unable to answer",
        "correct_option": "A",
        "options": ["A) Unable to answer", "B) Blue", "C) Black", "D) White"],
        "gold_emits": [{"chunk": 0, "value": "A"}],
        "grounding_frames": [0],
    }
    evidence = {
        0: {
            "chunk_idx": 0,
            "visible_entities": [{"desc": "person in a blue shirt"}],
            "atomic_facts": [{"fact": "The person is wearing a blue shirt."}],
        }
    }

    reason = _verify_card_layers(card, evidence)
    assert reason.startswith("hld1_concrete_option_supported")


def test_mc_card_rejects_empty_distractor_option():
    from scripts.agent_data.pass3a_cards import _verify_card_layers

    card = {
        "card_id": "p0",
        "family": "P1",
        "question": "What letters were printed on the shorts?",
        "answer_form": "multiple_choice",
        "question_type": "single_emit",
        "canonical_answer": "G.",
        "correct_option": "C",
        "options": ["A) U.S.C.", "B) N.C.", "C) G.", "D) "],
        "gold_emits": [{"chunk": 0, "value": "C"}],
        "grounding_frames": [0],
    }

    assert _verify_card_layers(card, {}) == "schema_mc_empty_option"


def test_pass3b_rejects_stale_empty_mc_option():
    from scripts.agent_data.pass3b_placement import _card_reject_reason

    bad = {
        "card_id": "p0",
        "family": "P1",
        "question": "What letters were printed on the shorts?",
        "answer_form": "multiple_choice",
        "question_type": "single_emit",
        "canonical_answer": "G.",
        "correct_option": "C",
        "options": ["A) U.S.C.", "B) N.C.", "C) G.", "D) "],
        "gold_emits": [{"chunk": 0, "value": "C"}],
        "grounding_frames": [0],
    }
    good = dict(bad, options=["A) U.S.C.", "B) N.C.", "C) G.", "D) E.G."])

    assert _card_reject_reason(bad) == "schema_mc_empty_option"
    assert _card_reject_reason(good) == ""


def test_pass3b_rejects_stale_option_rendering_leak():
    from scripts.agent_data.pass3b_placement import _card_reject_reason

    card = {
        "card_id": "p0",
        "family": "CR2",
        "question": (
            "After the vegetables were seen simmering in the pot, what was the "
            "chef doing in the next observed step among these options?"
        ),
        "answer_form": "multiple_choice",
        "question_type": "single_emit",
        "canonical_answer": "stirring the pot",
        "correct_option": "A",
        "options": [
            "A) stirring the pot",
            "B) chopping herbs",
            "C) washing a bowl",
            "D) opening a drawer",
        ],
        "gold_emits": [{"chunk": 0, "value": "A"}],
        "grounding_frames": [0],
    }

    assert _card_reject_reason(card) == "question_option_rendering_leak"


def test_hld1_prompt_asks_for_diverse_ovo_negatives():
    from scripts.agent_data.pass3a_cards import PASS3A_TARGETS_BY_FAMILY
    from scripts.agent_data.placement.llm_prompts import card_generation_prompt

    prompt = card_generation_prompt(
        "HLD1",
        [{
            "chunk_idx": 0,
            "visible_entities": [{"desc": "a person standing near a table"}],
            "atomic_facts": [{"fact": "The person is standing near a table."}],
        }],
        target_n=PASS3A_TARGETS_BY_FAMILY["HLD1"],
    )

    assert f"Produce {PASS3A_TARGETS_BY_FAMILY['HLD1']} card(s)" in prompt
    assert "explicit negative/unanswerable card" in prompt
    for subtype in [
        "location/where",
        "placement/object",
        "state yes/no",
        "count",
        "color/attribute",
        "before-memory",
    ]:
        assert subtype in prompt
    assert 'correct option text MUST be exactly "Unable to answer"' in prompt
    assert "output an empty JSON list" in prompt


def test_hld1_unable_option_slot_is_stably_normalized():
    from scripts.agent_data.pass3a_cards import _normalize_card_in_place

    positions = set()
    for i in range(16):
        card = {
            "card_id": f"hld-{i}",
            "family": "HLD1",
            "question": "Where did I put the object?",
            "answer_form": "multiple_choice",
            "question_type": "single_emit",
            "canonical_answer": "Unable to answer",
            "correct_option": "C",
            "options": ["A) On the table", "B) In the sink", "C) Unable to answer", "D) On the shelf"],
            "gold_emits": [{"chunk": 4, "value": "C"}],
            "grounding_frames": [4],
        }
        _normalize_card_in_place(card)
        co = card["correct_option"]
        positions.add(co)
        idx = ord(co) - ord("A")
        assert "Unable to answer" in card["options"][idx]
        assert card["canonical_answer"] == "Unable to answer"
        assert card["gold_emits"][0]["value"] == co

    assert len(positions) >= 3


def test_non_hld_mc_correct_option_slot_is_stably_normalized():
    from scripts.agent_data.pass3a_cards import _normalize_card_in_place

    positions = set()
    for i in range(24):
        card = {
            "card_id": f"mc-{i}",
            "family": "C1",
            "question": "What text is visible?",
            "answer_form": "multiple_choice",
            "question_type": "single_emit",
            "canonical_answer": "TARGET",
            "correct_option": "A",
            "options": ["A) TARGET", "B) DISTRACTOR 1", "C) DISTRACTOR 2", "D) DISTRACTOR 3"],
            "gold_emits": [{"chunk": 2, "value": "A"}],
            "grounding_frames": [2],
        }
        _normalize_card_in_place(card)
        first_options = list(card["options"])
        first_correct = card["correct_option"]
        _normalize_card_in_place(card)

        assert card["options"] == first_options
        assert card["correct_option"] == first_correct
        assert card["canonical_answer"] == "TARGET"
        assert card["gold_emits"][0]["value"] == first_correct
        assert "TARGET" in card["options"][ord(first_correct) - ord("A")]
        positions.add(first_correct)

    assert len(positions) == 4


def test_open_multi_answer_query_stays_open_after_first_answer():
    text = format_queries_block([
        {
            "question": "Report each event.",
            "ask_time": 10,
            "status": "open",
            "answers": [{"time": 12, "text": "first event"}],
        }
    ])

    assert "<active_query>" in text
    assert "<response_history>" in text
    assert "[12s] A: first event" in text
