from scripts.agent_data.pass3_slot_planner import (
    build_pass3_slot_plan,
    pass3a_batch_source_row_targets,
    pass3a_source_row_targets,
    pass3a_source_slot_requirements,
)
from scripts.agent_data.pass3a_cards import _dedupe_cards_by_question_signature
from scripts.agent_data.placement.design import (
    EVIDENCE_MULTIMODAL_ALIGNMENT,
    EVIDENCE_SOURCE_DISCRIMINATION,
    QUESTION_STYLE_OURS_UNIQUE,
    QUESTION_WAY_MULTIMODAL_ALIGNMENT,
    QUESTION_WAY_SOURCE_DISCRIMINATION,
    QUESTION_WAY_SEQUENTIAL_REFERENCE,
    infer_card_semantic_fields,
)


def _evidence(num_chunks: int = 96) -> list[dict]:
    rows = []
    for c in range(num_chunks):
        rows.append({
            "chunk_idx": c,
            "think": (
                f"chunk {c}: a person handles a labeled red box near a table; "
                "the visible label and object state provide visual evidence."
            ),
            "visible_entities": [
                {"id": f"person_{c % 3}", "desc": "person in a blue shirt", "action": "holding"},
                {"id": f"box_{c % 4}", "desc": "red box with a label", "action": "visible"},
            ],
            "atomic_facts": [
                {"fact": "The red box is on the table before it is moved."},
                {"fact": "The label visually supports the answer."},
            ],
            "ocr": [{"text": "FRAGILE"}] if c % 9 == 0 else [],
            "spatial": "the box is left of the cup",
            "state_changes": ["the person moves the box"] if c % 13 == 0 else [],
        })
    return rows


def test_slot_plan_adds_missing_streamingbench_question_ways():
    slots = build_pass3_slot_plan(
        _evidence(160),
        "taxonomy_video",
        {"N1": 3, "CR4": 3, "R1": 3, "CR5": 1, "M1": 1, "PN1": 1},
        source_row_targets={"direct": 20, "recall": 20, "future": 4, "multi": 2},
    )

    by_subtype = {slot["task_subtype"]: slot for slot in slots}
    assert by_subtype["sequential_reference"]["question_way"] == QUESTION_WAY_SEQUENTIAL_REFERENCE
    assert by_subtype["source_discrimination"]["question_way"] == QUESTION_WAY_SOURCE_DISCRIMINATION
    assert by_subtype["source_discrimination"]["evidence_type"] == EVIDENCE_SOURCE_DISCRIMINATION
    assert by_subtype["multimodal_alignment"]["question_way"] == QUESTION_WAY_MULTIMODAL_ALIGNMENT
    assert by_subtype["multimodal_alignment"]["evidence_type"] == EVIDENCE_MULTIMODAL_ALIGNMENT

    cr5 = next(slot for slot in slots if slot["family"] == "CR5")
    assert cr5["question_style"] == QUESTION_STYLE_OURS_UNIQUE
    assert cr5["task_family"] != "ours_agentic"
    assert cr5["slot_group"] != "ours_agentic"


def test_semantic_inference_recognizes_new_question_forms():
    source = infer_card_semantic_fields({
        "family": "CR4",
        "question": "Which visual source supports that the red box was moved?",
    })
    assert source["question_way"] == QUESTION_WAY_SOURCE_DISCRIMINATION
    assert source["evidence_type"] == EVIDENCE_SOURCE_DISCRIMINATION

    alignment = infer_card_semantic_fields({
        "family": "R1",
        "question": "Does the visible label match the object being handled?",
    })
    assert alignment["question_way"] == QUESTION_WAY_MULTIMODAL_ALIGNMENT
    assert alignment["evidence_type"] == EVIDENCE_MULTIMODAL_ALIGNMENT


def test_pass3a_source_slot_requirements_scale_by_response_rows():
    short = pass3a_source_slot_requirements(48)
    mid = pass3a_source_slot_requirements(96)
    long = pass3a_source_slot_requirements(300)

    assert pass3a_source_row_targets(300) == {
        "direct": 22,
        "recall": 11,
        "future": 4,
        "multi": 4,
    }
    assert short == {"direct": 7, "recall": 3, "future": 0, "multi": 0}
    assert mid["direct"] > short["direct"]
    assert mid["recall"] > short["recall"]
    assert mid["future"] > short["future"]
    assert long["direct"] > mid["direct"]
    assert long["multi"] > mid["multi"]
    assert sum(long.values()) <= 65


def test_pass3a_batch_source_targets_preserve_video_totals_and_batch_mix():
    targets = pass3a_batch_source_row_targets({
        "short": 96,
        "mid": 180,
        "long": 300,
    })

    assert {k: sum(v.values()) for k, v in targets.items()} == {
        "short": 13,
        "mid": 25,
        "long": 41,
    }
    total = {source: sum(v[source] for v in targets.values()) for source in ("direct", "recall", "future", "multi")}
    assert 0.48 <= total["direct"] / 79 <= 0.57
    assert 0.23 <= total["recall"] / 79 <= 0.32
    assert 0.07 <= total["future"] / 79 <= 0.13
    assert 0.08 <= total["multi"] / 79 <= 0.16


def test_pass3a_dedupes_teacher_question_signatures():
    cards = [
        {"card_id": "a", "question": "What color is the box?"},
        {"card_id": "b", "question": "What color is the box"},
        {"card_id": "c", "question": "Where is the box?"},
    ]
    kept, rejected = _dedupe_cards_by_question_signature(cards)
    assert [c["card_id"] for c in kept] == ["a", "c"]
    assert rejected == {"question_signature_duplicate": 1}
