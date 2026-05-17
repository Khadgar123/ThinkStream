"""Multi-Q compute_score smoke test.

Loads a multi-Q parquet (built with `build_verl_parquet --multi_q`) and
exercises the OVOBench-aligned multi-Q scoring path in
thinkstream/rl/thinkstream.py:compute_score.

Coverage:
  - All-correct rollout → outcome 1.0
  - First-Q-wrong rollout → answer-weighted partial outcome
  - All-unanswered rollout → outcome 0.0, n_answered 0.0
  - liberal MCQ matching: option letter, option text, gold_answer fallback
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_compute_score():
    spec = importlib.util.spec_from_file_location(
        "rt_thinkstream",
        PROJECT_ROOT / "thinkstream/rl" / "thinkstream.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_dict(q: Any) -> Dict[str, Any]:
    if hasattr(q, "tolist"):
        q = q.tolist()
    return dict(q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    rt = _load_compute_score()
    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df)} multi-Q rows from {args.parquet}")

    # Pick the row with the most questions to exercise the multi-Q path
    n_q_col = df["n_questions"].astype(int).tolist()
    target = max(range(len(n_q_col)), key=lambda i: n_q_col[i])
    r = df.iloc[target]
    extra = dict(r["extra_info"])
    gt_str = r["reward_model"]["ground_truth"]

    qs_raw = extra.get("questions")
    if qs_raw is None:
        qs_raw = []
    if hasattr(qs_raw, "tolist"):
        qs_raw = qs_raw.tolist()
    questions: List[Dict[str, Any]] = [_to_dict(q) for q in qs_raw]
    n_q = len(questions)
    print(f"  target video={r['video_id']} n_q={n_q}")

    # ── ALL CORRECT
    # Current multi-Q reward scores answer content at the expected answer
    # chunk, not merely at the query injection chunk. Delayed questions can
    # have ask_chunk << answer_chunks[0].
    per_q_chunk = []
    for q in questions:
        answer_chunks = q.get("answer_chunks")
        if answer_chunks is None:
            answer_chunks = []
        if hasattr(answer_chunks, "tolist"):
            answer_chunks = answer_chunks.tolist()
        if answer_chunks:
            per_q_chunk.append(int(answer_chunks[0]))
        else:
            per_q_chunk.append(int(q.get("ask_chunk", -1)))
    per_q_text = [
        str(q.get("correct_option") or q.get("gold_answer", ""))
        for q in questions
    ]
    extra_in = dict(extra)
    extra_in["ts_per_q_answer_chunk"] = per_q_chunk
    extra_in["ts_per_q_answer_text"] = per_q_text
    solution = "".join(
        f"<think>x</think><answer>{t}</answer>" for t in per_q_text
    )
    result = rt.compute_score(
        "thinkstream_v12_streaming_multi_q", solution, gt_str, extra_in,
    )
    assert result["outcome"] == 1.0, f"all-correct: {result}"
    assert result["n_questions"] == n_q
    print(
        f"  ✓ all-correct: outcome={result['outcome']:.3f} "
        f"decision={result['answer_decision']:.3f} timing={result['timing']:.3f}"
    )

    # ── FIRST WRONG (use a string that won't match any option text)
    extra_in["ts_per_q_answer_text"] = ["NOPE_NOT_AN_OPTION"] + per_q_text[1:]
    r2 = rt.compute_score(
        "thinkstream_v12_streaming_multi_q", solution, gt_str, extra_in,
    )
    answer_weights = [rt._answer_weight_for_question(q) for q in questions]
    expected = (
        (sum(answer_weights) - answer_weights[0]) / max(sum(answer_weights), 1.0)
        if answer_weights else 0.0
    )
    assert abs(r2["outcome"] - expected) < 1e-6, f"partial: got {r2['outcome']} expected {expected}"
    print(f"  ✓ first-wrong: outcome={r2['outcome']:.3f} (expected {expected:.3f})")

    # ── ALL UNANSWERED
    extra_in["ts_per_q_answer_chunk"] = [-1] * n_q
    extra_in["ts_per_q_answer_text"] = [""] * n_q
    r3 = rt.compute_score(
        "thinkstream_v12_streaming_multi_q", "", gt_str, extra_in,
    )
    assert r3["outcome"] == 0.0
    assert r3["n_answered"] == 0.0
    assert r3["timing"] < 0  # missed-bucket penalty
    assert r3["answer_decision"] <= r3["timing"]
    print(
        f"  ✓ unanswered: outcome={r3['outcome']:.3f} "
        f"decision={r3['answer_decision']:.3f} timing={r3['timing']:.3f} "
        f"n_answered={r3['n_answered']:.0f}"
    )

    # ── LIBERAL MCQ matching tests
    cases = [
        # (model_answer, options, correct_option, gold_answer, expected_match)
        ("C", ["a", "b", "c", "d"], "C", "", True),
        ("c.", ["a", "b", "c", "d"], "C", "", True),
        ("(C)", ["a", "b", "c", "d"], "C", "", True),
        ("blue shirt", ["red", "green", "blue shirt", "white"], "C", "", True),
        ("A", ["a", "b", "c", "d"], "C", "", False),
        ("eggplant", [], "", "eggplant", True),
        ("Eggplant.", [], "", "eggplant", True),
        # Single-letter ma against text-heavy options (the bug we fixed)
        ("B", ["on the table", "in the cabinet", "on the counter", "on the sink"], 0, "", False),
        # If an explicit option label is present, it must be the gold letter.
        ("B) Unable to answer", ["Unable to answer", "Unable to answer"], "A", "Unable to answer", False),
        ("I cannot infer it. B) Unable to answer", ["Unable to answer", "Unable to answer"], "A", "Unable to answer", False),
    ]
    for ma, opts, co, ga, exp in cases:
        got = rt._match_mcq_answer(ma, opts, co, ga)
        assert got == exp, f"_match_mcq_answer({ma!r}, opts={len(opts)}, correct={co!r}, gold={ga!r}) got={got} exp={exp}"
    print(f"  ✓ {len(cases)} liberal MCQ matching tests pass")

    # ── Form-aware liberal outcome tests
    form_cases = [
        # (answer_form, model_answer, gold_answer, options, correct_option, expected)
        # binary
        ("binary",      "Yes.",         "yes",   [], "", 1.0),
        ("binary",      "y",            "yes",   [], "", 1.0),
        ("binary",      "false",        "no",    [], "", 1.0),
        ("binary",      "no",           "yes",   [], "", 0.0),
        # number
        ("number",      "100.",         "100",   [], "", 1.0),
        ("number",      "$100 mph",     "100",   [], "", 1.0),
        ("number",      "99",           "100",   [], "", 0.0),
        ("number",      "not numeric",  "100",   [], "", 0.0),
        # short_exact
        ("short_exact", "the apple",    "apple", [], "", 1.0),
        ("short_exact", "Apple.",       "apple", [], "", 1.0),
        ("short_exact", "orange",       "apple", [], "", 0.0),
        # descriptive
        ("descriptive", "the man walks home", "the man walks", [], "", 1.0),
        ("descriptive", "the woman runs",     "the man walks", [], "", 0.0),
        # multiple_choice via dispatcher
        ("multiple_choice", "C", ["a","b","c","d"], ["a","b","c","d"], "C", 1.0),
        # unanswered
        ("binary", "", "yes", [], "", 0.0),
        ("number", None, "100", [], "", 0.0),
    ]
    for af, ma, ga, opts, co, exp in form_cases:
        if isinstance(ga, list):  # MCQ row order shifted
            opts, ga = ga, ""
        got = rt._score_outcome_by_form(
            ma, options=opts, correct_option=co, gold_answer=ga, answer_form=af,
        )
        assert got == exp, f"_score_outcome_by_form(form={af}, ma={ma!r}, ga={ga!r}) got={got} exp={exp}"
    print(f"  ✓ {len(form_cases)} form-aware outcome matcher tests pass (binary/number/short_exact/descriptive/multiple_choice)")

    print("✓ ALL MULTI-Q SCORING TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
