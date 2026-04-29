"""v12.0 eval-adapter smoke tests.

Tests adapter input building + scoring logic on synthetic data, no GPU
or vLLM required.

Run: python tests/test_v12_eval_adapters.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ovo_build_user_input():
    from scripts.eval.adapters import OVOBenchAdapter

    item = {
        "task": "EPM",
        "realtime": 14,  # → question_chunk = 7
        "question": "What did they do?",
        "options": ["sat down", "stood up", "ran away", "yelled"],
        "gt": 1,
    }
    # Pre-question chunks: empty
    assert OVOBenchAdapter.build_user_input(item, 0) == ""
    assert OVOBenchAdapter.build_user_input(item, 6) == ""

    # Question chunk: full prompt
    prompt = OVOBenchAdapter.build_user_input(item, 7)
    assert "What did they do?" in prompt
    assert "A. sat down" in prompt
    assert "B. stood up" in prompt
    assert "C. ran away" in prompt
    assert "D. yelled" in prompt
    assert "Answer with one letter (A/B/C/D)" in prompt

    # After-question chunks: empty (silent expected)
    assert OVOBenchAdapter.build_user_input(item, 8) == ""

    print("✓ OVO build_user_input")


def test_ovo_score_letter_match():
    from scripts.eval.adapters import OVOBenchAdapter

    item = {
        "task": "EPM",
        "realtime": 14,  # question_chunk=7
        "options": ["a", "b", "c", "d"],
        "gt": 2,  # → "C"
    }

    # Correct letter at question chunk
    outs = [
        {"chunk_idx": 5, "text": "<think>...</think><answer></answer>"},
        {"chunk_idx": 7, "text": "<think>x</think><answer>C</answer>"},
    ]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is True
    assert r["answer_chunk"] == 7
    assert r["delay_chunks"] == 0
    assert r["fmt"] == "letter"

    # Correct letter delayed by 2 chunks (within window)
    outs = [
        {"chunk_idx": 7, "text": "<think>...</think><answer></answer>"},
        {"chunk_idx": 8, "text": "<think>...</think><answer></answer>"},
        {"chunk_idx": 9, "text": "<think>x</think><answer>C</answer>"},
    ]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is True  # delay=2 ≤ late_window
    assert r["delay_chunks"] == 2

    # Late beyond window — fmt=letter but in_window=False
    outs = [{"chunk_idx": 12, "text": "<think>x</think><answer>C</answer>"}]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is False  # delay=5 > 2 → out of window
    assert r["in_window"] is False

    # Wrong letter
    outs = [{"chunk_idx": 7, "text": "<think>x</think><answer>A</answer>"}]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is False
    assert r["fmt"] == "letter"

    # Free-text answer matches option content (substring match)
    outs = [{"chunk_idx": 7, "text": "<think>x</think><answer>I think it is c</answer>"}]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is True
    assert r["fmt"] == "free_text_substring"

    # Never answered → no_answer
    outs = [
        {"chunk_idx": 7, "text": "<think>x</think><answer></answer>"},
        {"chunk_idx": 9, "text": "<think>x</think><answer></answer>"},
    ]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is False
    assert r["fmt"] == "no_answer"
    assert r["answer_chunk"] is None

    print("✓ OVO score (letter / late / wrong / free-text / no-answer)")


def test_ovo_hld_unable_to_answer():
    from scripts.eval.adapters import OVOBenchAdapter

    item = {
        "task": "HLD",
        "realtime": 6,  # question_chunk=3
        "options": ["empty bottles", "Unable to answer", "newspapers", "scraps"],
        "gt": 1,  # "Unable to answer" is the correct option
    }
    # Model emits the correct letter
    outs = [{"chunk_idx": 3, "text": "<think>x</think><answer>B</answer>"}]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is True

    # Model stays silent (avoiding the question) — wrong even though semantically
    # it's "I don't know", because OVO requires picking the option
    outs = [{"chunk_idx": 3, "text": "<think>x</think><answer></answer>"}]
    r = OVOBenchAdapter.score(item, outs)
    assert r["correct"] is False

    print("✓ OVO HLD 'Unable to answer' option handling")


def test_our_open_ended_adapter():
    from scripts.eval.adapters import OurOpenEndedAdapter

    # binary form — strict yes/no match
    item = {
        "ask_chunk": 5,
        "question": "Is the chef wearing red?",
        "gold_answer": "Yes",
        "answer_form": "binary",
    }
    assert OurOpenEndedAdapter.build_user_input(item, 4) == ""
    p = OurOpenEndedAdapter.build_user_input(item, 5)
    assert "Q: Is the chef wearing red?" in p
    assert "Options:" not in p

    outs = [{"chunk_idx": 5, "text": "<think>x</think><answer>Yes</answer>"}]
    r = OurOpenEndedAdapter.score(item, outs)
    assert r["correct"] is True

    outs = [{"chunk_idx": 5, "text": "<think>x</think><answer>yes definitely</answer>"}]
    r = OurOpenEndedAdapter.score(item, outs)
    # binary form requires exact "Yes" — "yes definitely" fails strict match
    assert r["correct"] is False

    # descriptive: fuzzy substring
    item2 = {
        "ask_chunk": 3,
        "question": "Describe what happens",
        "gold_answer": "chef adds salt",
        "answer_form": "descriptive",
    }
    outs = [{"chunk_idx": 3, "text": "<think>x</think><answer>The chef adds salt to the pot</answer>"}]
    r = OurOpenEndedAdapter.score(item2, outs)
    assert r["correct"] is True  # fuzzy: "chef adds salt" ⊂ answer

    print("✓ OurOpenEndedAdapter (binary strict + descriptive fuzzy)")


def test_our_adapter_with_options():
    """If our val item has options, adapter formats them MC-style."""
    from scripts.eval.adapters import OurOpenEndedAdapter

    item = {
        "ask_chunk": 3,
        "question": "What color?",
        "options": ["red", "blue", "green", "yellow"],
        "gold_answer": "C",
        "answer_form": "multiple_choice",
    }
    p = OurOpenEndedAdapter.build_user_input(item, 3)
    assert "Options:" in p
    assert "C. green" in p
    print("✓ OurOpenEndedAdapter with options")


def test_aggregate_function():
    from scripts.eval.v12_ovo_eval import aggregate

    results = [
        {"correct": True, "task": "EPM", "delay_chunks": 0, "fmt": "letter"},
        {"correct": True, "task": "EPM", "delay_chunks": 1, "fmt": "letter"},
        {"correct": False, "task": "HLD", "delay_chunks": 2, "fmt": "letter"},
        {"correct": False, "task": "HLD", "delay_chunks": None, "fmt": "no_answer"},
    ]
    s = aggregate(results)
    assert s["total"] == 4
    assert s["correct"] == 2
    assert s["accuracy"] == 0.5
    assert s["by_task"]["EPM"]["accuracy"] == 1.0
    assert s["by_task"]["HLD"]["accuracy"] == 0.0
    assert s["delay_chunks"]["p50"] == 1
    assert s["no_answer_rate"] == 0.25
    print("✓ aggregate function (per-task + delay stats)")


def test_score_offline_predictions():
    from scripts.eval.v12_ovo_eval import score_offline_predictions
    from scripts.eval.adapters import OVOBenchAdapter

    items = [
        {"id": 0, "task": "EPM", "realtime": 14, "options": ["a", "b", "c", "d"], "gt": 0},
        {"id": 1, "task": "HLD", "realtime": 4, "options": ["a", "b", "c", "d"], "gt": 1},
    ]
    predictions = [
        [{"chunk_idx": 7, "text": "<think>x</think><answer>A</answer>"}],
        [{"chunk_idx": 2, "text": "<think>x</think><answer>X</answer>"}],
    ]
    results = score_offline_predictions(
        items=items, predictions=predictions, adapter=OVOBenchAdapter,
    )
    assert results[0]["correct"] is True
    assert results[0]["item_id"] == 0
    assert results[1]["correct"] is False
    print("✓ score_offline_predictions e2e")


def test_real_ovo_data_smoke():
    """Sanity check: load a few real OVO-Bench items and build prompts."""
    p = Path("/Users/hzh/Downloads/ovo_bench_new.json")
    if not p.exists():
        print("⊘ skipping real OVO data test (file not found)")
        return
    import json
    from scripts.eval.adapters import OVOBenchAdapter
    with p.open() as f:
        data = json.load(f)
    sample = data[0]
    qc = OVOBenchAdapter.question_chunk(sample)
    prompt = OVOBenchAdapter.build_user_input(sample, qc)
    assert "Q:" in prompt
    assert "Options:" in prompt
    assert "Answer with one letter" in prompt
    # Verify each option is on its own line
    for letter, opt in zip("ABCD", sample["options"]):
        assert f"{letter}. {opt}" in prompt, f"{letter}. {opt} not in prompt"
    print(f"✓ Real OVO data smoke (item 0, task={sample['task']})")


if __name__ == "__main__":
    test_ovo_build_user_input()
    test_ovo_score_letter_match()
    test_ovo_hld_unable_to_answer()
    test_our_open_ended_adapter()
    test_our_adapter_with_options()
    test_aggregate_function()
    test_score_offline_predictions()
    test_real_ovo_data_smoke()
    print("\n✅ all v12.0 eval-adapter tests passed")
