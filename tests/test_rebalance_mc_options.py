from scripts.agent_data.rebalance_mc_options import (
    _label_options,
    _strip_label,
    patch_row,
)


def _mapping():
    question = "What text is embossed on the handle?"
    return {
        ("video_1", "card_a", question): {
            "options": ["A) OXO", "B) IKEA", "C) CALPHALON", "D) CUISINART"],
            "correct_option": "A",
        },
        ("video_1", "card_b", question): {
            "options": ["A) XOX", "B) OXX", "C) XXO", "D) OXO"],
            "correct_option": "D",
        },
    }


def test_rebalance_uses_parent_card_id_for_row_metadata():
    question = "What text is embossed on the handle?"
    row = {
        "video_id": "video_1",
        "card_id": "card_b",
        "metadata": {
            "question": question,
            "answer_form": "multiple_choice",
            "answer_style": "letter_plus_text",
            "options": ["A) OXO", "B) IKEA", "C) CALPHALON", "D) CUISINART"],
            "correct_option": "A",
        },
        "output": "<think>x</think></Response> A",
    }

    changed = patch_row(row, _mapping(), {})

    assert changed > 0
    assert row["metadata"]["card_id"] == "card_b"
    assert row["metadata"]["options"] == ["A) XOX", "B) OXX", "C) XXO", "D) OXO"]
    assert row["metadata"]["correct_option"] == "D"
    assert row["metadata"]["answer_style"] == "letter_plus_text"
    assert row["output"] == "<think>x</think></Response> D) OXO"


def test_rebalance_does_not_question_fallback_when_ambiguous():
    question = "What text is embossed on the handle?"
    row = {
        "video_id": "video_1",
        "metadata": {
            "question": question,
            "answer_form": "multiple_choice",
            "answer_style": "letter_plus_text",
            "options": ["A) OXO", "B) IKEA", "C) CALPHALON", "D) CUISINART"],
            "correct_option": "A",
        },
    }

    changed = patch_row(row, _mapping(), {})

    assert changed == 0
    assert "card_id" not in row["metadata"]
    assert row["metadata"]["correct_option"] == "A"


def test_rebalance_preserves_letter_like_answer_text():
    assert _strip_label("C) G.") == "G."
    assert _strip_label("G.") == "G."
    assert _label_options(["U.S.C.", "N.C.", "", "G."]) == [
        "A) U.S.C.",
        "B) N.C.",
        "C) ",
        "D) G.",
    ]
