#!/bin/bash
# 2-GPU verl GRPO debug — minimal smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Use the latest SFT checkpoint (or debug checkpoint fallback)
LLM="${LLM:-${PROJECT_DIR}/output/agent-sft/checkpoint-200}"
if [[ ! -d "${LLM}" ]]; then
    LLM="${PROJECT_DIR}/output/agent-sft-debug/checkpoint-5"
fi

OUTPUT_DIR="${PROJECT_DIR}/output/agent-verl-debug-$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

# Minimal synthetic data for fast turnaround
TRAIN_PARQUET="/tmp/test_rl.parquet"
VAL_PARQUET="/tmp/test_rl.parquet"

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export THINKSTREAM_TRAJ_INDEX_PATH="${PROJECT_DIR}/data/test_rl/synthetic_trajectories.jsonl"
export THINKSTREAM_FRAMES_ROOT=""

cd "${PROJECT_DIR}/verl"

echo "=== ThinkStream verl GRPO 2-GPU debug ==="
echo "Checkpoint: ${LLM}"
echo "Output:     ${OUTPUT_DIR}"
echo "========================================="

CUDA_VISIBLE_DEVICES=6,7 \
python3 -m verl.trainer.main_ppo \
    --config-path="${PROJECT_DIR}/verl/recipe_thinkstream/configs" \
    --config-name="thinkstream_grpo" \
    actor_rollout_ref.model.path="${LLM}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    reward.custom_reward_function.path="recipe_thinkstream/thinkstream.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.total_epochs=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.experiment_name="verl-debug" \
    trainer.project_name="thinkstream-debug" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=2

echo ""
echo "=== verl debug complete ==="
ls -la "${OUTPUT_DIR}"
