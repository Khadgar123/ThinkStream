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


def test_ovo_builder_default_track_is_strict25_45_without_packing():
    rows, summary = build_trajectories(
        [_mcq(1, 20), _mcq(2, 80)],
        post_context_chunks=2,
    )

    assert summary["split_policy"] == "strict25_45"
    assert summary["benchmark_tracks"] == ["strict25_45", "strict25_45_stateful"]
    assert len(rows) == 2
    assert all(len(r["questions"]) == 1 for r in rows)
    for row in rows:
        span = row["segment_end_chunk"] - row["segment_start_chunk"] + 1
        ask = row["questions"][0]["ask_chunk"]
        assert 25 <= span <= 45
        assert row["segment_start_chunk"] <= ask <= row["segment_end_chunk"]
        assert ask != row["segment_start_chunk"]


def test_ovo_builder_rejects_retired_split_policies():
    for policy in ["query_span", "casia", "short20_40", "short20_40_stateful"]:
        try:
            build_trajectories([_mcq(1, 20)], split_policy=policy)
        except ValueError as exc:
            assert "strict25_45" in str(exc)
        else:
            raise AssertionError(f"retired split policy still accepted: {policy}")


def test_ovo_builder_strict25_45_policy_targets_short_rows_without_packing():
    rows, summary = build_trajectories(
        [_mcq(1, 20), _mcq(2, 80)],
        split_policy="strict25_45",
        post_context_chunks=2,
    )

    assert summary["split_policy"] == "strict25_45"
    assert summary["trajectories"] == 2
    assert all(len(r["questions"]) == 1 for r in rows)
    spans = [r["segment_end_chunk"] - r["segment_start_chunk"] + 1 for r in rows]
    assert all(25 <= span <= 45 for span in spans)
    assert sorted(spans) == [25, 45]


def test_ovo_builder_stateful_policy_splits_long_rec_into_short_scored_parts():
    rec = {
        "id": 99,
        "task": "REC",
        "video": "thumos/v.mp4",
        "activity": "jump",
        "start_times": [0],
        "end_times": [85],
        "test_info": [
            {"realtime": 5, "count": 1},
            {"realtime": 45, "count": 2},
            {"realtime": 85, "count": 3},
        ],
    }
    rows, summary = build_trajectories(
        [rec],
        split_policy="strict25_45_stateful",
        post_context_chunks=2,
    )

    spans = [r["segment_end_chunk"] - r["segment_start_chunk"] + 1 for r in rows]
    assert summary["source_question_units"] == 1
    assert summary["stateful_split_parent_rows"] == 1
    assert summary["stateful_context_rows"] == 0
    assert len(rows) == 2
    assert min(spans) >= 25
    assert summary["stateful_over_45_rows"] == 0
    assert [r["questions"][0]["answer_chunks"] for r in rows] == [[5], [45, 85]]
    assert rows[1]["questions"][0]["per_emit_answers"] == [
        {"chunk": 45, "value": "2"},
        {"chunk": 85, "value": "3"},
    ]
    assert all(
        r["segment_start_chunk"] <= r["questions"][0]["ask_chunk"] <= r["segment_end_chunk"]
        for r in rows
    )
    assert all(
        r["questions"][0]["ask_chunk"] != r["segment_start_chunk"]
        for r in rows
    )


def test_ovo_builder_coalesces_duplicate_rec_probe_chunks():
    rec = {
        "id": 101,
        "task": "REC",
        "video": "thumos/v.mp4",
        "activity": "jump",
        "start_times": [0],
        "end_times": [20],
        "test_info": [
            {"realtime": 5, "count": 1},
            {"realtime": 5, "count": 2},
            {"realtime": 20, "count": 3},
        ],
    }
    rows, _summary = build_trajectories(
        [rec],
        split_policy="strict25_45_stateful",
        post_context_chunks=2,
    )

    q = rows[0]["questions"][0]
    assert q["answer_chunks"] == [5, 20]
    assert q["per_emit_answers"] == [
        {"chunk": 5, "value": "2"},
        {"chunk": 20, "value": "3"},
    ]


def test_ovo_builder_stateful_policy_keeps_context_parts_for_long_ssr():
    ssr = {
        "id": 100,
        "task": "SSR",
        "video": "COIN/v.mp4",
        "start_time": [0],
        "end_time": [85],
        "test_info": [
            {"realtime": 85, "type": 1, "step": "tighten the screw"},
        ],
    }
    rows, summary = build_trajectories(
        [ssr],
        split_policy="strict25_45_stateful",
        post_context_chunks=2,
    )

    assert len(rows) == 2
    assert summary["stateful_context_rows"] == 1
    assert [len(r["questions"]) for r in rows] == [0, 1]
    assert rows[-1]["questions"][0]["answer_chunks"] == [85]
    spans = [r["segment_end_chunk"] - r["segment_start_chunk"] + 1 for r in rows]
    assert spans == [44, 44]


def test_ovo_builder_stateful_allows_reasonable_tail_merge_over_45():
    ssr = {
        "id": 102,
        "task": "SSR",
        "video": "COIN/v.mp4",
        "start_time": [0],
        "end_time": [46],
        "test_info": [
            {"realtime": 46, "type": 1, "step": "tighten the screw"},
        ],
    }
    rows, summary = build_trajectories(
        [ssr],
        split_policy="strict25_45_stateful",
        post_context_chunks=2,
    )

    spans = [r["segment_end_chunk"] - r["segment_start_chunk"] + 1 for r in rows]
    assert spans == [49]
    assert summary["stateful_over_45_rows"] == 1
    assert summary["stateful_max_part_span"] == 49


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
