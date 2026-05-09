"""Prompt/reward/eval contract tests for MCQ options and answer formats.

Run: python tests/test_prompt_eval_contract.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_active_query_renders_mcq_options_once():
    from thinkstream.data.agent_protocol import format_queries_block
    from thinkstream.eval.prompt_contract import label_mc_options

    block = format_queries_block([
        {
            "question": "What color is the cup?",
            "ask_time": 12,
            "options": label_mc_options(
                ["A. red", "B) blue", "C: green", "D. yellow"],
                style="paren",
            ),
            "answer_form": "multiple_choice",
            "answer_style": "letter_only",
        }
    ])

    assert block.count("Q: What color is the cup?") == 1
    assert block.count("Options:") == 1
    assert "A) red" in block
    assert "B) blue" in block
    assert "Answer format: one letter only" in block


def test_plain_baseline_prompt_labels_options_once():
    from thinkstream.eval.prompt_contract import build_plain_mcq_prompt

    prompt = build_plain_mcq_prompt(
        "What happened?",
        ["A. sat down", "B) stood up", "C: ran", "D. jumped"],
        option_style="dot",
        instruction="Answer with a single letter.",
        include_options_header=True,
    )

    assert prompt.count("Options:") == 1
    assert prompt.count("A. sat down") == 1
    assert "A. A." not in prompt
    assert "B. B)" not in prompt
    assert prompt.endswith("Answer with a single letter.")


def test_streaming_agent_contract_uses_bare_question_and_structured_meta():
    from thinkstream.data.agent_protocol import format_queries_block
    from thinkstream.eval.prompt_contract import build_streaming_query_meta

    sample = {
        "id": 1,
        "task": "EPM",
        "realtime": 3.0,
        "question": "What did the person pick up?",
        "options": ["cup", "book", "phone", "bag"],
        "gt": 2,
        "answer": "phone",
    }

    question = sample["question"]
    meta = build_streaming_query_meta(sample, answer_form="multiple_choice")
    block = format_queries_block([{
        "question": question,
        "ask_time": 3,
        **meta,
    }])

    assert question == sample["question"]
    assert "cup" not in question
    assert meta["answer_form"] == "multiple_choice"
    assert meta["correct_option"] == "C"
    assert meta["options"] == ["A) cup", "B) book", "C) phone", "D) bag"]
    assert "one letter only" in meta["answer_instruction"]
    assert block.count("Options:") == 1
    assert block.count("A) cup") == 1


def test_streaming_meta_strips_existing_option_labels():
    from thinkstream.eval.prompt_contract import build_streaming_query_meta

    item = {
        "question": "Which object appears?",
        "options": ["A. cup", "B. book", "C. phone", "D. bag"],
        "answer": "C",
    }
    meta = build_streaming_query_meta(item)

    assert meta["options"] == ["A) cup", "B) book", "C) phone", "D) bag"]
    assert meta["correct_option"] == "C"


def test_shared_reward_eval_matcher_handles_all_answer_forms():
    from thinkstream.trainer.outcome_match import score_outcome_by_form

    assert score_outcome_by_form(
        "C",
        options=["A) red", "B) blue", "C) green", "D) yellow"],
        correct_option="C",
        gold_answer="green",
        answer_form="multiple_choice",
    ) == 1.0
    assert score_outcome_by_form(
        "Yes.", gold_answer="yes", answer_form="binary",
    ) == 1.0
    assert score_outcome_by_form(
        "$12.00", gold_answer="12", answer_form="number",
    ) == 1.0
    assert score_outcome_by_form(
        "the red apron", gold_answer="red apron", answer_form="short_exact",
    ) == 1.0
    assert score_outcome_by_form(
        "the chef adds salt to the pot",
        gold_answer="chef adds salt",
        answer_form="descriptive",
    ) == 1.0


if __name__ == "__main__":
    test_active_query_renders_mcq_options_once()
    test_plain_baseline_prompt_labels_options_once()
    test_streaming_agent_contract_uses_bare_question_and_structured_meta()
    test_streaming_meta_strips_existing_option_labels()
    test_shared_reward_eval_matcher_handles_all_answer_forms()
    print("prompt/eval/reward contract tests passed")
