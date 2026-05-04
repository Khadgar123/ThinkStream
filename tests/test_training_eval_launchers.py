from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_sft_launcher_defaults_and_no_duplicate_save_flags():
    text = _read("scripts/sft_per_timestep.sh")
    assert "lr=${LR:-2e-5}; epochs=${EPOCHS:-2}" in text
    assert "BSZ=${BSZ:-8}" in text
    assert "GRAD_ACCUM=${GRAD_ACCUM:-1}" in text
    assert "--max_steps ${MAX_STEPS}" in text
    assert "--save_strategy epoch" not in text
    assert "--save_total_limit 3" not in text


def test_top_level_rl_launcher_delegates_to_verl_recipe():
    text = _read("scripts/grpo_train_verl.sh")
    assert "GROUP_SIZE=${GROUP_SIZE:-8}" in text
    assert "BATCH_SIZE=${BATCH_SIZE:-4}" in text
    assert "PPO_MINI_BS=${PPO_MINI_BS:-${BATCH_SIZE}}" in text
    assert "LR=${LR:-5e-7}" in text
    assert "MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-32768}" in text
    assert "MAX_ACTION_TOKENS=${MAX_ACTION_TOKENS:-4096}" in text
    assert "bash recipe_thinkstream/run_thinkstream_grpo.sh" in text
    assert "python3 -m verl.trainer.main_ppo" not in text


def test_verl_recipe_launcher_uses_prompt_unit_ppo_mini_batch():
    text = _read("verl/recipe_thinkstream/run_thinkstream_grpo.sh")
    assert "BATCH_SIZE=${BATCH_SIZE:-4}" in text
    assert "PPO_MINI_BS=${PPO_MINI_BS:-${BATCH_SIZE}}" in text
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS}" in text
    assert "actor_rollout_ref.rollout.response_length=${MAX_RESP_LEN}" in text
    assert "reward.custom_reward_function.path=" in text
    assert "custom_reward_function.path=" not in text.replace(
        "reward.custom_reward_function.path=", ""
    )


def test_rl_loop_has_separate_per_action_generation_cap():
    text = _read("verl/recipe_thinkstream/streaming_agent_loop.py")
    assert "self.max_tokens_per_action" in text
    assert '"max_tokens": max_tokens_this_turn' in text
    assert "len(chunk_prompt_ids) + self.response_length" not in text


def test_eval_defaults_are_full_and_deterministic():
    matrix = _read("scripts/eval/run_matrix.sh")
    assert 'N_TEST="${N_TEST:-0}"' in matrix
    assert 'N_PER_OVO_TASK="${N_PER_OVO_TASK:-0}"' in matrix

    loop = _read("thinkstream/model/agent_loop.py")
    assert 'temperature = float(kwargs.get("temperature", 0.0))' in loop
    assert '"do_sample": do_sample' in loop


def test_ovo_full_wrappers_accept_preextracted_frames():
    sft = _read("scripts/eval/ovo/run_sft_full.sh")
    rl = _read("scripts/eval/ovo/run_rl_full.sh")
    for text in (sft, rl):
        assert "FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}" in text
        assert "--frames_root" in text
        assert "--profile" in text
        assert "--scoring" in text
