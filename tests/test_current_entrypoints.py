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
        "stream_agent_rl_traj",
        "stream_agent_val_traj",
        "stream_agent_test_traj",
        "stream_agent_trajectory_train",
        "stream_agent_trajectory_val",
        "stream_agent_trajectory_test",
    }
    assert set(DATASET_REGISTRY) == expected


def test_sft_rl_eval_use_canonical_prompt_contract():
    sft = _text("scripts/sft_trajectory.sh")
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
    recipe = _text("thinkstream/rl/run_thinkstream_grpo.sh")

    for text in (rl, recipe):
        assert 'THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"' in text
        assert "MAX_NEW_TOKEN=4096" in text or "MAX_RESP_LEN=4096" in text

    assert 'BATCH_SIZE=${BATCH_SIZE:-4}' in rl
    assert 'BATCH_SIZE="${RL_BATCH_SIZE:-${BATCH_SIZE:-4}}"' in run
    assert "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}" in rl
    assert "LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}" in rl
    assert "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-true}" in rl
    assert "ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-streaming}" in rl
    assert "TP_SIZE=${TP_SIZE:-1}" in rl
    assert "GEN_TP=1" in recipe


def test_training_scheme_builds_full_and_segment_rl_inputs():
    scheme = _text("scripts/agent_data/make_training_scheme.py")

    assert "_link_rollouts_for_selected" in scheme
    assert "train_rl_multi_q.parquet" in scheme
    assert "train_rl_multi_q_segment_cache.parquet" in scheme
    assert "include_student_cache=include_student_cache" in scheme
    assert "balanced keeps family/recall/compress ratios" in scheme
    parquet = _text("scripts/agent_data/build_verl_parquet.py")
    assert 'prompt = [{"role": "system", "content": system_prompt}]' in parquet


def test_pre_rl_rollout_audit_is_legacy_sidecar_not_canonical_rollout():
    audit = _text("scripts/agent_data/pre_rl_rollout_audit.py")

    assert "Legacy fast pre-RL student rollout audit" in audit
    assert "prefer building the same parquet rows" in audit
    assert "streaming_vllm_rollout" in audit
    assert 'frame_protocol="video_meta"' in audit
    assert 'render_layout="standard_query_last"' in audit
    assert "rollout_batch_size" in audit
    assert "badcase_out" in audit
    assert "stable_think" in audit


def test_ovo_rl_eval_builds_canonical_rollout_inputs():
    builder = _text("scripts/eval/ovo/build_rl_trajectories.py")
    run_recurrent = _text("scripts/eval/ovo/run_rl_recurrent_eval.sh")
    readme = _text("scripts/eval/ovo/README.md")

    assert "Render OVO-Bench into ThinkStream RL multi-Q trajectory rows" in builder
    assert "through the verl recurrent rollout path" in builder
    assert "_iter_rows_multi_q" in builder
    assert "segment_start_chunk" in builder
    assert "segment_end_chunk" in builder
    assert "span_exceeded_soft_limit" in builder
    assert "ovo_rl_multi_q.parquet" in readme
    assert "same verl" in readme
    assert "recurrent AgentLoop validation/test path" in readme
    assert "VAL_ONLY=true" in run_recurrent
    assert "VALIDATION_DATA_DIR" in run_recurrent
    assert "bash scripts/grpo_train_verl.sh" in run_recurrent
    assert 'MAX_CHUNKS="${MAX_CHUNKS:-2048}"' in run_recurrent


def test_verl_launcher_supports_validation_only_rollout():
    launcher = _text("scripts/grpo_train_verl.sh")
    recipe = _text("thinkstream/rl/run_thinkstream_grpo.sh")

    assert "VAL_ONLY" in launcher
    assert "VAL_BEFORE_TRAIN" in launcher
    assert "VALIDATION_DATA_DIR" in launcher
    assert "trainer.val_before_train=${VAL_BEFORE_TRAIN}" in recipe
    assert "+trainer.val_only=${VAL_ONLY}" in recipe
    assert "+trainer.validation_data_dir" in recipe


def test_verl_launcher_exposes_multinode_ray_address():
    launcher = _text("scripts/grpo_train_verl.sh")
    recipe = _text("thinkstream/rl/run_thinkstream_grpo.sh")

    assert "NNODES" in launcher
    assert "RAY_ADDRESS" in launcher
    assert 'export NNODES="${NNODES:-1}"' in launcher
    assert 'export RAY_ADDRESS="${RAY_ADDRESS:-}"' in launcher
    assert "RAY_ADDRESS_ARGS" in recipe
    assert 'ray_kwargs.ray_init.address="${RAY_ADDRESS}"' in recipe
    assert "trainer.nnodes=${NNODES}" in recipe


def main() -> None:
    tests = [
        test_dataset_registry_only_current_entries,
        test_sft_rl_eval_use_canonical_prompt_contract,
        test_rl_defaults_use_multi_trajectory_recurrent_update,
        test_training_scheme_builds_full_and_segment_rl_inputs,
        test_pre_rl_rollout_audit_is_legacy_sidecar_not_canonical_rollout,
        test_ovo_rl_eval_builds_canonical_rollout_inputs,
        test_verl_launcher_supports_validation_only_rollout,
        test_verl_launcher_exposes_multinode_ray_address,
    ]
    for test in tests:
        test()
    print("current entrypoint contract tests passed")


if __name__ == "__main__":
    main()
