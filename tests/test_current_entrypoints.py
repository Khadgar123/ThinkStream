"""Current production entrypoint contract smoke tests.

Run directly:
    python tests/test_current_entrypoints.py
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_dataset_registry_only_current_entries():
    from thinkstream.sft.data_list import DATASET_REGISTRY

    expected = {
        "stream_agent_sft",
        "stream_agent_val",
        "stream_agent_test",
        "stream_agent_rl_traj",
        "stream_agent_val_traj",
        "stream_agent_test_traj",
    }
    assert set(DATASET_REGISTRY) == expected


def test_sft_rl_eval_use_canonical_prompt_contract():
    sft = _text("scripts/sft_per_timestep.sh")
    rl = _text("scripts/grpo_train_verl.sh")
    ovo_sft = _text("scripts/eval/ovo/run_sft_full.sh")
    ovo_rl = _text("scripts/eval/ovo/run_rl_full.sh")

    for text in (sft, rl, ovo_sft, ovo_rl):
        assert "video_meta" in text
        assert "standard_query_last" in text

    assert "canonical SFT uses FRAME_PROTOCOL=video_meta" in sft
    assert "canonical RL uses FRAME_PROTOCOL=video_meta" in rl
    assert "canonical OVO" in ovo_sft
    assert "canonical OVO" in ovo_rl


def test_rl_defaults_use_multi_trajectory_recurrent_update():
    rl = _text("scripts/grpo_train_verl.sh")
    run = _text("scripts/run_sft_rl.sh")
    recipe = _text("verl/recipe_thinkstream/run_thinkstream_grpo.sh")

    for text in (rl, recipe):
        assert 'THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"' in text
        assert "MAX_NEW_TOKEN=4096" in text or "MAX_RESP_LEN=4096" in text

    assert 'BATCH_SIZE=${BATCH_SIZE:-4}' in rl
    assert 'BATCH_SIZE="${RL_BATCH_SIZE:-${BATCH_SIZE:-4}}"' in run
    assert "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}" in rl
    assert "LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}" in rl
    assert "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-true}" in rl


def test_training_scheme_builds_full_and_segment_rl_inputs():
    scheme = _text("scripts/agent_data_v5/make_training_scheme.py")

    assert "_link_rollouts_for_selected" in scheme
    assert "train_rl_multi_q.parquet" in scheme
    assert "train_rl_multi_q_segment_cache.parquet" in scheme
    assert "include_student_cache=include_student_cache" in scheme
    assert "balanced keeps family/recall/compress ratios" in scheme


def main() -> None:
    tests = [
        test_dataset_registry_only_current_entries,
        test_sft_rl_eval_use_canonical_prompt_contract,
        test_rl_defaults_use_multi_trajectory_recurrent_update,
        test_training_scheme_builds_full_and_segment_rl_inputs,
    ]
    for test in tests:
        test()
    print("current entrypoint contract tests passed")


if __name__ == "__main__":
    main()
