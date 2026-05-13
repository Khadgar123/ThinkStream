import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.ovo.build_rl_trajectories import build_trajectories  # noqa: E402
from scripts.audit.summarize_rl_recurrent_validation import summarize  # noqa: E402
from scripts.agent_data.build_verl_parquet import _iter_rows_multi_q  # noqa: E402
from thinkstream.rl.streaming_agent_loop import _resolve_frame_dir  # noqa: E402


def _mcq(sample_id, realtime, video="Ego4D/clips/v.mp4", task="EPM"):
    return {
        "id": sample_id,
        "task": task,
        "video": video,
        "realtime": realtime,
        "question": f"Question {sample_id}?",
        "options": ["red", "blue", "green", "white"],
        "gt": 1,
    }


def test_ovo_builder_packs_non_overlapping_questions_without_cutting_answers():
    rows, summary = build_trajectories(
        [_mcq(1, 20), _mcq(2, 80)],
        max_span_chunks=128,
        pre_context_chunks=8,
        post_context_chunks=2,
    )

    assert summary["trajectories"] == 1
    row = rows[0]
    assert len(row["questions"]) == 2
    for q in row["questions"]:
        for ck in q["ask_chunks"] + q["answer_chunks"]:
            assert row["segment_start_chunk"] <= ck <= row["segment_end_chunk"]


def test_ovo_builder_keeps_task_families_separate_by_default():
    samples = [
        _mcq(1, 20, task="EPM"),
        _mcq(2, 80, task="HLD"),
    ]
    rows, summary = build_trajectories(
        samples,
        max_span_chunks=128,
        pre_context_chunks=8,
        post_context_chunks=2,
    )
    packed_rows, packed_summary = build_trajectories(
        samples,
        max_span_chunks=128,
        pre_context_chunks=8,
        post_context_chunks=2,
        pack_across_tasks=True,
    )

    assert summary["pack_across_tasks"] is False
    assert len(rows) == 2
    assert all(len({q["ovo_task"] for q in row["questions"]}) == 1 for row in rows)
    assert packed_summary["pack_across_tasks"] is True
    assert len(packed_rows) == 1
    assert {q["ovo_task"] for q in packed_rows[0]["questions"]} == {"EPM", "HLD"}


def test_ovo_builder_splits_overlapping_active_queries():
    crr = {
        "id": 10,
        "task": "CRR",
        "video": "MovieNet/v.mp4",
        "question": "Did the actor open the door?",
        "ask_time": 20,
        "clue_time": 40,
        "test_info": [
            {"realtime": 20, "type": 0},
            {"realtime": 40, "type": 1},
        ],
    }
    rows, _summary = build_trajectories(
        [crr, _mcq(11, 30, video="MovieNet/v.mp4")],
        max_span_chunks=128,
        pre_context_chunks=4,
        post_context_chunks=1,
    )

    assert len(rows) == 2
    assert sorted(len(r["questions"]) for r in rows) == [1, 1]
    for row in rows:
        for q in row["questions"]:
            for ck in q["ask_chunks"] + q["answer_chunks"]:
                assert row["segment_start_chunk"] <= ck <= row["segment_end_chunk"]


def test_ovo_segment_fields_survive_verl_parquet_rows(tmp_path):
    rows, _summary = build_trajectories(
        [_mcq(1, 20)],
        max_span_chunks=64,
        pre_context_chunks=8,
        post_context_chunks=2,
    )
    jsonl = tmp_path / "ovo_trajectories.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    parquet_rows = list(
        _iter_rows_multi_q(
            jsonl,
            max_questions_per_traj=16,
            frame_protocol="video_meta",
            render_layout="standard_query_last",
            include_student_cache=False,
        )
    )

    extra = parquet_rows[0]["extra_info"]
    assert extra["segment_start_chunk"] == rows[0]["segment_start_chunk"]
    assert extra["segment_end_chunk"] == rows[0]["segment_end_chunk"]
    assert extra["questions"][0]["ovo_task"] == "EPM"


def test_rl_frame_resolver_supports_ovo_nested_frame_layout(tmp_path):
    frame_dir = tmp_path / "Ego4D" / "clips" / "abc"
    frame_dir.mkdir(parents=True)
    assert _resolve_frame_dir("Ego4D/clips/abc.mp4", str(tmp_path)) == frame_dir


def test_recurrent_validation_summary_groups_ovo_tasks(tmp_path):
    gt = {
        "questions": [{
            "ovo_task": "EPM",
            "ovo_category": "BT",
            "question": "Q?",
            "answer_chunks": [4],
            "gold_answer": "A",
        }],
    }
    generations = tmp_path / "0.jsonl"
    generations.write_text(
        json.dumps({
            "gts": json.dumps(gt),
            "score": 0.8,
            "outcome": 1.0,
            "answer_decision": 0.9,
            "format": 1.0,
            "trajectory_mean_correct": 1.0,
            "trajectory_all_correct": 1.0,
            "n_questions": 1,
            "n_answered": 1,
        }) + "\n",
        encoding="utf-8",
    )

    out = summarize(generations)

    assert out["overall"]["questions"] == 1
    assert out["by_task"]["EPM"]["trajectory_mean_correct_question_weighted"] == 1.0
    assert out["category_task_macro"]["BT"] == 1.0
    assert out["health"]["answer"]["answered_rate"] == 1.0
