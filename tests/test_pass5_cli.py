"""End-to-end smoke test for pass5 CLI driver.

Writes a synthetic pass4 trajectories.jsonl, runs convert_file, and asserts
the v2 output is well-formed.

Run:
  python tests/test_pass5_cli.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data.pass5 import (  # noqa: E402
    FrameResolverConfig,
    SFT_TRAJECTORY_SPLITS,
    convert_dir,
    convert_file,
    make_frame_resolver,
    make_sample_aware_resolver,
)


def _synthetic_pass4_record(video_id: str, with_compress: bool = True):
    samples = [
        {
            "chunk_idx": 0, "sample_type": "silent",
            "output": "<think>start</think><answer></answer>",
            "trajectory_id": f"{video_id}_traj_0",
        },
        {
            "chunk_idx": 1, "sample_type": "response",
            "output": "<think>I see it</think><answer>A</answer>",
            "trajectory_id": f"{video_id}_traj_0",
            "card_id": "c1",
        },
    ]
    if with_compress:
        samples.append({
            "chunk_idx": 2, "sample_type": "compress",
            "inter_chunk": True,
            "gold_caption": "compressed summary 0-2",
            "gold_compress_chunks": [0, 1, 2],
            "output": (
                "<think>compress 0-2</think>"
                '<tool_call>{"name": "compress", "arguments": '
                '{"time_range": [0, 3], "text": "compressed summary 0-2"}}</tool_call>'
            ),
            "trajectory_id": f"{video_id}_traj_0",
        })
        samples.append({
            "chunk_idx": 3, "sample_type": "silent",
            "output": "<think>x</think><answer></answer>",
            "trajectory_id": f"{video_id}_traj_0",
        })
    return {
        "video_id": video_id,
        "trajectory_id": f"{video_id}_traj_0",
        "questions": [
            {
                "ask_chunk": 1,
                "question": "What color?",
                "options": ["A. Red", "B. Blue"],
                "answer_instruction": "letter only",
            }
        ],
        "samples": samples,
    }


def test_frame_resolver_pattern_only():
    cfg = FrameResolverConfig(frames_root=None)
    fn = make_frame_resolver("vid_x", cfg)
    paths = fn(5)
    # 2 frames per zero-based chunk, chunk 5 covers source frame indices
    # 10 and 11. Project JPEG names are ffmpeg-style 1-based.
    assert paths == ["frame_000011.jpg", "frame_000012.jpg"], paths
    print("[OK] frame_resolver_pattern_only")


def test_frame_resolver_with_root_relative():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = FrameResolverConfig(frames_root=root, absolute=False)
        fn = make_frame_resolver("vid_y", cfg)
        paths = fn(3)
        assert paths[0].startswith("vid_y/"), paths
        assert paths[0].endswith("frame_000007.jpg"), paths
    print("[OK] frame_resolver_with_root_relative")


def test_sample_aware_resolver_prefers_explicit():
    cfg = FrameResolverConfig(frames_root=None)
    samples_by_chunk = {
        5: {"input": {"visual_window": {"frame_paths": ["custom/5a.jpg", "custom/5b.jpg"]}}},
    }
    fn = make_sample_aware_resolver("vid_z", cfg, samples_by_chunk)
    # Chunk 5 has explicit paths → use them
    assert fn(5) == ["custom/5a.jpg", "custom/5b.jpg"]
    # Chunk 6 has no explicit paths → fall back to pattern
    assert fn(6) == ["frame_000013.jpg", "frame_000014.jpg"]
    print("[OK] sample_aware_resolver_prefers_explicit")


def test_convert_file_smoke():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "train_trajectories.jsonl"
        dst = td_path / "train_trajectory.jsonl"

        with src.open("w", encoding="utf-8") as f:
            for v in ["vid_a", "vid_b"]:
                rec = _synthetic_pass4_record(v, with_compress=True)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # one record without compress to cover that path too
            rec = _synthetic_pass4_record("vid_c", with_compress=False)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stats = convert_file(src, dst)

        assert stats.n_records_in == 3
        # vid_a + vid_b: stream + compact-memory update + from-compress
        # segment each. vid_c: one stream segment.
        assert stats.n_rows_out == 3 + 3 + 1, f"rows_out={stats.n_rows_out}"
        assert stats.n_compress_events == 2
        assert stats.n_from_start == 3   # one per video
        assert stats.n_from_compress == 2  # one after each compress
        assert stats.n_compact_memory_update == 2
        assert stats.n_questions == 3     # 1 per record

        # Validate produced JSONL
        rows = []
        with dst.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        assert len(rows) == 7
        # Streaming rows have recall tools; compact-memory rows are text-only.
        for r in rows:
            assert len(r["messages"]) >= 3   # system + at least 1 user + 1 assistant
            if r["trajectory_type"] == "compact_memory_update":
                assert r["tools"] == []
            else:
                assert len(r["tools"]) == 1

        # vid_a should be stream -> compact-memory update -> from-compress.
        vid_a_rows = [r for r in rows if r["video_id"] == "vid_a"]
        assert vid_a_rows[0]["trajectory_type"] == "from_start"
        assert vid_a_rows[0]["chunk_end"] == 1
        assert vid_a_rows[0]["compress_event"] is None
        assert vid_a_rows[1]["trajectory_type"] == "compact_memory_update"
        assert vid_a_rows[1]["compress_event"]["summary_text"].startswith("<MEM>")
        assert vid_a_rows[2]["trajectory_type"] == "from_compress"
        assert vid_a_rows[2]["chunk_start"] == 3

    print(f"[OK] convert_file_smoke ({stats.n_rows_out} rows, "
          f"{stats.n_compress_events} compress events)")


def test_convert_file_with_limit():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "in.jsonl"
        dst = td_path / "out.jsonl"
        with src.open("w", encoding="utf-8") as f:
            for v in ["a", "b", "c", "d", "e"]:
                f.write(json.dumps(_synthetic_pass4_record(v)) + "\n")
        stats = convert_file(src, dst, limit=2)
        assert stats.n_records_in == 2
        print(f"[OK] convert_file_with_limit (records_in={stats.n_records_in})")


def test_convert_dir_split_whitelist_skips_rl_files():
    """``convert_dir`` should only render SFT/eval/test trajectory JSONL.
    pass4 emits ``train_rl_trajectories.jsonl`` for RL-side use, which RL
    then consumes via ``build_verl_parquet`` — there's no downstream
    reader for ``train_rl_trajectory.jsonl``, so emitting it just burns
    disk."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_dir = td_path / "in"; in_dir.mkdir()
        out_dir = td_path / "out"; out_dir.mkdir()
        # Sources mirror pass4's actual emission for one video each.
        for split in ("train_sft", "train_rl", "val", "test"):
            src = in_dir / f"{split}_trajectories.jsonl"
            with src.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_synthetic_pass4_record(f"vid_{split}")) + "\n")
        manifest = convert_dir(in_dir, out_dir)

        rendered = sorted(p.name for p in out_dir.glob("*_trajectory.jsonl"))
        assert "train_sft_trajectory.jsonl" in rendered
        assert "val_trajectory.jsonl" in rendered
        assert "test_trajectory.jsonl" in rendered
        assert "train_rl_trajectory.jsonl" not in rendered, (
            "train_rl split must NOT be rendered to per-trajectory JSONL — "
            "RL consumes pass4's *_trajectories.jsonl directly via parquet."
        )
        assert set(manifest.keys()) <= SFT_TRAJECTORY_SPLITS
        print(f"[OK] convert_dir_split_whitelist_skips_rl_files ({rendered})")


def test_data_list_trajectory_train_matches_pass5_output():
    """data_list[stream_agent_trajectory_train].annotation_path MUST equal
    the on-disk filename pass5.convert_dir writes for the train_sft split.
    Catches the historical typo (train_trajectory.jsonl vs the actual
    train_sft_trajectory.jsonl)."""
    try:
        from thinkstream.sft.data_list import DATASET_REGISTRY
    except ImportError as e:
        # Some local envs have a transformers <-> tokenizers version skew
        # that blocks importing the SFT data_list module. Skip cleanly.
        print(f"[SKIP] data_list_trajectory_train_matches_pass5_output ({e})")
        return
    entry = DATASET_REGISTRY["stream_agent_trajectory_train"]
    assert entry["annotation_path"].endswith("train_sft_trajectory.jsonl"), (
        f"data_list points at {entry['annotation_path']!r}, but pass5 writes "
        f"train_sft_trajectory.jsonl (from train_sft_trajectories.jsonl)."
    )
    print("[OK] data_list_trajectory_train_matches_pass5_output")


if __name__ == "__main__":
    test_frame_resolver_pattern_only()
    test_frame_resolver_with_root_relative()
    test_sample_aware_resolver_prefers_explicit()
    test_convert_file_smoke()
    test_convert_file_with_limit()
    test_convert_dir_split_whitelist_skips_rl_files()
    test_data_list_trajectory_train_matches_pass5_output()
    print("\nall pass5 CLI smoke tests passed")
