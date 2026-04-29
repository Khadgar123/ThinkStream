"""v12 streaming benchmark eval test — verifies the eval framework
operates correctly on real test_trajectories.jsonl.gz.

Run: python tests/test_v12_streaming_bench.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.v12_streaming_bench import (
    aggregate,
    compute_chunk_window,
    enrich_question_text,
    load_trajectories,
    make_dry_run_step_fn,
    run_streaming_eval,
    score_trajectory,
)


DATA_FINAL = Path(__file__).resolve().parents[1] / "data" / "agent_v5" / "final"


def test_load_trajectories_reads_gz():
    """Loader must handle .jsonl.gz transparently."""
    p = DATA_FINAL / "test_trajectories.jsonl.gz"
    if not p.exists():
        # Maybe only .jsonl is available
        p = DATA_FINAL / "test_trajectories.jsonl"
        if not p.exists():
            print("  SKIP no test trajectory file on disk")
            return
    rows = load_trajectories(p)
    assert len(rows) > 0
    assert "video_id" in rows[0]
    assert "questions" in rows[0]
    assert "gold_action_per_chunk" in rows[0]
    print(f"  PASS load_trajectories from {p.name}: {len(rows)} rows")


def test_enrich_question_text():
    """Questions get their text recovered from sample.input.user_input."""
    p = DATA_FINAL / "test_trajectories.jsonl.gz"
    if not p.exists():
        p = DATA_FINAL / "test_trajectories.jsonl"
    if not p.exists():
        print("  SKIP")
        return
    rows = load_trajectories(p)
    found_real = 0
    found_implicit = 0
    for row in rows[:20]:
        enriched = enrich_question_text(row)
        for q in enriched:
            text = q.get("question_text") or ""
            if text == "<implicit>":
                found_implicit += 1
            elif text:
                found_real += 1
    assert found_real > 0, "no question text recovered from samples"
    print(f"  PASS enrich: {found_real} real, {found_implicit} implicit (E2 etc)")


def test_compute_chunk_window():
    """Chunk window covers max ask_chunk + post_window."""
    traj = {
        "questions": [
            {"ask_chunks": [3]},
            {"ask_chunks": [10, 12]},
        ],
        "stats": {"chunk_idx_max": 30},
    }
    n = compute_chunk_window(traj, post_window=5)
    assert n == 17, f"expected 12+5=17, got {n}"
    print(f"  PASS compute_chunk_window={n}")


def test_run_streaming_eval_with_stub():
    """End-to-end: stub step_fn, verify chunk_outputs shape."""
    traj = {
        "video_id": "vid_x", "trajectory_id": "traj_0",
        "questions": [{"ask_chunks": [5], "question_text": "what color?"}],
        "gold_action_per_chunk": {str(i): "silent" for i in range(10)},
    }
    traj["gold_action_per_chunk"]["5"] = "response"
    step_fn = make_dry_run_step_fn()
    result = run_streaming_eval(traj, step_fn, post_window=3)
    assert result["video_id"] == "vid_x"
    assert result["n_chunks"] == 8  # ask=5 + post=3 = 8
    assert len(result["chunk_outputs"]) == 8
    for out in result["chunk_outputs"]:
        assert out["kind"] == "answer"
        assert out["answer_text"] is None  # stub silent
    print("  PASS run_streaming_eval with stub: 8 chunks emitted")


def test_score_trajectory_silent_run():
    """Stub silent run → outcome=0 (no answer), silent_quality positive
    on chunks where gold also says silent."""
    traj = {
        "video_id": "vid_x", "trajectory_id": "traj_0",
        "questions": [{
            "card_id": "c1", "gold_answer": "red",
            "answer_form": "literal", "ask_chunks": [5],
            "family": "F1",
        }],
        "gold_action_per_chunk": {str(i): "silent" for i in range(10)},
    }
    traj["gold_action_per_chunk"]["5"] = "response"
    step_fn = make_dry_run_step_fn()
    result = run_streaming_eval(traj, step_fn, post_window=3)
    score = score_trajectory(traj, result["chunk_outputs"])
    assert score["outcome"] == 0.0  # silent-when-should-respond
    assert score["n_questions"] == 1
    assert score["n_correct"] == 0
    assert score["n_missed"] == 1
    # silent_q: 7 silent-correct + 1 missed → mean = (7*0.3 - 0.6) / 8 = 0.1875
    assert abs(score["silent_quality"] - (7 * 0.3 - 0.6) / 8) < 1e-6
    print(f"  PASS score_trajectory: outcome=0, silent_q={score['silent_quality']:.3f}")


def test_aggregate_combines_trajectories():
    """Aggregate with multiple trajectory scores → correct totals."""
    scores = [
        {
            "video_id": "v1", "trajectory_id": "t1",
            "outcome": 1.0, "n_questions": 2, "n_answered": 2, "n_correct": 2,
            "silent_quality": 0.3, "n_correct_silent": 5,
            "n_hallucinate": 0, "n_missed": 0,
            "per_family": {"F1": [1.0, 1.0]},
        },
        {
            "video_id": "v2", "trajectory_id": "t2",
            "outcome": 0.0, "n_questions": 1, "n_answered": 0, "n_correct": 0,
            "silent_quality": 0.1, "n_correct_silent": 3,
            "n_hallucinate": 0, "n_missed": 1,
            "per_family": {"F1": [0.0]},
        },
    ]
    agg = aggregate(scores)
    assert agg["n_trajectories"] == 2
    assert agg["n_questions"] == 3
    assert agg["n_correct"] == 2
    assert abs(agg["outcome_acc"] - 2/3) < 1e-6  # weighted by n_questions
    assert agg["per_family"]["F1"]["mean"] == 2/3
    assert agg["per_family"]["F1"]["n"] == 3
    print(f"  PASS aggregate: outcome_acc={agg['outcome_acc']:.3f}, F1 acc=2/3")


def test_full_dryrun_on_5_real_trajectories():
    """Smoke test: run on 5 real trajectories from disk, verify no crash."""
    p = DATA_FINAL / "test_trajectories.jsonl.gz"
    if not p.exists():
        p = DATA_FINAL / "test_trajectories.jsonl"
    if not p.exists():
        print("  SKIP")
        return
    rows = load_trajectories(p)[:5]
    step_fn = make_dry_run_step_fn()
    per_traj = []
    for traj in rows:
        traj = {**traj, "questions": enrich_question_text(traj)}
        result = run_streaming_eval(traj, step_fn)
        score = score_trajectory(traj, result["chunk_outputs"])
        per_traj.append(score)
    agg = aggregate(per_traj)
    assert agg["n_trajectories"] == 5
    assert agg["n_questions"] >= 5    # at least 1 question per trajectory
    # Stub silent: every question should be missed → outcome=0
    assert agg["outcome_acc"] == 0.0
    # silent_quality should be positive (most chunks are gold-silent)
    assert agg["silent_quality_avg"] > 0
    print(f"  PASS 5-traj dry run: {agg['n_questions']} questions, "
          f"silent_q={agg['silent_quality_avg']:.3f}, missed={agg['n_missed_total']}")


def main():
    tests = [
        test_load_trajectories_reads_gz,
        test_enrich_question_text,
        test_compute_chunk_window,
        test_run_streaming_eval_with_stub,
        test_score_trajectory_silent_run,
        test_aggregate_combines_trajectories,
        test_full_dryrun_on_5_real_trajectories,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
