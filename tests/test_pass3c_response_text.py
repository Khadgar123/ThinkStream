"""Regression tests for pass3c answer rendering."""

import asyncio

from scripts.agent_data_v5.pass3c_samples import (
    _response_text_for,
    _response_text_via_llm,
)


def test_mc_response_uses_correct_option_text_not_gold_emit_fragment():
    card = {
        "answer_form": "multiple_choice",
        "question_type": "single_emit",
        "canonical_answer": "The artist points at the canvas, presents paint tubes, shakes a bottle of liquid, and then applies black paint.",
        "options": [
            "A) The artist applies black paint first.",
            "B) The artist shakes a bottle before showing paint tubes.",
            "C) The artist presents paint tubes after black paint.",
            "D) The artist points at the canvas, presents paint tubes, shakes a bottle of liquid, and then applies black paint.",
        ],
        "correct_option": "D",
    }

    text = _response_text_for(
        card,
        "Artist's hand enters frame and points at the blank white can",
    )

    assert text.startswith("The artist points at the canvas")
    assert "blank white can" not in text


def test_mc_response_can_use_letter_only_style():
    card = {
        "answer_form": "multiple_choice",
        "answer_style": "letter_only",
        "options": ["A) red", "B) blue", "C) green", "D) yellow"],
        "correct_option": "C",
    }

    assert _response_text_for(card, "ignored") == "C"


def test_mc_response_can_use_letter_plus_text_style():
    card = {
        "answer_form": "multiple_choice",
        "answer_style": "letter_plus_text",
        "options": ["A) red", "B) blue", "C) green", "D) yellow"],
        "correct_option": "B",
    }

    assert _response_text_for(card, "ignored") == "B) blue"


def test_descriptive_multi_emit_uses_current_emit_not_future_canonical():
    card = {
        "answer_form": "descriptive",
        "question_type": "multi_emit",
        "canonical_answer": "The video first shows butter being poured, then stirring, then kneading later.",
    }

    assert _response_text_for(card, "pouring_butter@1") == "pouring butter"


class _FailingClient:
    async def _call_one(self, **_kwargs):
        raise AssertionError("multi_emit should not call LLM")


def test_descriptive_multi_emit_via_llm_short_circuits():
    card = {
        "answer_form": "descriptive",
        "question_type": "multi_emit",
        "canonical_answer": "A future full sequence that should not be emitted.",
    }

    text = asyncio.run(
        _response_text_via_llm(
            card, "stirring_with_spoon@2", _FailingClient(), "vid", 12
        )
    )

    assert text == "stirring with spoon"
