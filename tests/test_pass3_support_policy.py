import random

from scripts.agent_data.placement.design import (
    Card,
    GoldEmit,
    Placement,
    SUPPORT_CURRENT_VISUAL,
    SUPPORT_FUTURE_CURRENT_CUE,
    SUPPORT_HISTORICAL_STATE_MEMORY,
    SUPPORT_HISTORICAL_VISUAL_RECALL,
    SUPPORT_PROBE_STATUS,
    assign_recall_noise,
    infer_card_policy_fields,
    place_card,
)
from scripts.agent_data.pass3b_placement import _drop_degraded_recall_placements


def _single_card(family: str, question: str, emit: int = 10) -> Card:
    return Card(
        card_id=f"{family}-single",
        family=family,
        question=question,
        answer_form="multiple_choice",
        question_type="single_emit",
        gold_emits=[GoldEmit(chunk=emit, value="A")],
        grounding_frames=[emit],
        options=["A) answer", "B) distractor"],
        correct_option="A",
    )


def test_historical_visual_policy_is_the_only_single_emit_recall_path():
    card = _single_card("C1", "What text was displayed earlier on the label?", emit=5)
    policy = infer_card_policy_fields(card)

    assert policy["support_policy"] == SUPPORT_HISTORICAL_VISUAL_RECALL
    assert SUPPORT_CURRENT_VISUAL in policy["allowed_support_policies"]
    assert SUPPORT_HISTORICAL_VISUAL_RECALL in policy["allowed_support_policies"]

    placements = place_card(card, num_chunks=120, rng=random.Random(7))
    assert any(p.mechanism == "direct" for p in placements)
    assert any(p.mechanism == "recall_demo" for p in placements)

    assign_recall_noise(placements, random.Random(7), {card.card_id: card})
    for placement in placements:
        if placement.mechanism == "recall_demo":
            assert placement.support_policy == SUPPORT_HISTORICAL_VISUAL_RECALL
            assert placement.recall_at
        else:
            assert not placement.recall_at


def test_f6_future_current_cue_does_not_create_recall_variants():
    card = _single_card("F6", "What is likely to happen next based on the current scene?", emit=20)
    policy = infer_card_policy_fields(card)

    assert policy["support_policy"] == SUPPORT_FUTURE_CURRENT_CUE
    assert not policy["recall_eligible"]

    placements = place_card(card, num_chunks=120, rng=random.Random(3))
    assert placements
    assert {p.mechanism for p in placements} == {"direct"}
    assign_recall_noise(placements, random.Random(3), {card.card_id: card})
    assert all(not p.recall_at for p in placements)


def test_cr5_hybrid_ours_card_keeps_direct_recall_and_wait_options():
    card = _single_card(
        "CR5",
        "What did the early clue resolve into after the later observation?",
        emit=30,
    )
    policy = infer_card_policy_fields(card)

    assert policy["support_policy"] == SUPPORT_HISTORICAL_VISUAL_RECALL
    assert SUPPORT_CURRENT_VISUAL in policy["allowed_support_policies"]
    assert SUPPORT_HISTORICAL_VISUAL_RECALL in policy["allowed_support_policies"]
    assert SUPPORT_FUTURE_CURRENT_CUE in policy["allowed_support_policies"]

    placements = place_card(card, num_chunks=120, rng=random.Random(17))
    mechanisms = {p.mechanism for p in placements}
    assert {"direct", "recall_demo", "silent_then_response"}.issubset(mechanisms)


def test_f5_cumulative_count_is_state_memory_not_visual_recall():
    card = Card(
        card_id="F5-count",
        family="F5",
        question="How many times has the person picked up a cup by now?",
        answer_form="number",
        question_type="multi_emit",
        gold_emits=[
            GoldEmit(chunk=10, value="1"),
            GoldEmit(chunk=80, value="2"),
            GoldEmit(chunk=160, value="3"),
        ],
        grounding_frames=[10, 80, 160],
    )
    policy = infer_card_policy_fields(card)

    assert policy["support_policy"] == SUPPORT_HISTORICAL_STATE_MEMORY
    assert policy["state_memory_required"]
    assert not policy["recall_eligible"]

    placements = place_card(card, num_chunks=200, rng=random.Random(11))
    assert placements and placements[0].mechanism == "multi_emit"
    assign_recall_noise(placements, random.Random(11), {card.card_id: card})
    assert all(not p.recall_at for p in placements)


def test_f7_current_step_status_is_probe_not_recall():
    card = Card(
        card_id="F7-step",
        family="F7",
        question="Is the person currently washing the bowl?",
        answer_form="binary",
        question_type="single_emit",
        gold_emits=[
            GoldEmit(chunk=12, value="Yes"),
        ],
        grounding_frames=[12],
    )
    policy = infer_card_policy_fields(card)

    assert policy["support_policy"] == SUPPORT_CURRENT_VISUAL
    assert not policy["recall_eligible"]

    placements = place_card(card, num_chunks=80, rng=random.Random(13))
    assert placements and placements[0].mechanism == "direct"
    assign_recall_noise(placements, random.Random(13), {card.card_id: card})
    assert all(not p.recall_at for p in placements)


def test_degraded_recall_demo_is_removed_after_rollout_refine():
    good = Placement(
        card_id="good",
        ask_chunk=20,
        mechanism="recall_demo",
        support_policy=SUPPORT_HISTORICAL_VISUAL_RECALL,
        chunk_actions={20: ("response", "A")},
        recall_at={20: "oracle"},
    )
    degraded = Placement(
        card_id="bad",
        ask_chunk=30,
        mechanism="recall_demo",
        support_policy=SUPPORT_HISTORICAL_VISUAL_RECALL,
        chunk_actions={30: ("response", "A")},
        recall_at={},
    )
    direct = Placement(
        card_id="direct",
        ask_chunk=40,
        mechanism="direct",
        support_policy=SUPPORT_CURRENT_VISUAL,
        chunk_actions={40: ("response", "A")},
        recall_at={},
    )

    kept, dropped = _drop_degraded_recall_placements([good, degraded, direct])

    assert dropped == 1
    assert kept == [good, direct]
