#!/bin/bash
# verl-based GRPO training launcher for ThinkStream — parallel to
# scripts/grpo_train.sh which uses slyme.
#
# Why two paths:
#   - slyme path (grpo_train.sh): proprietary framework; HF rollout backend,
#     no vLLM bridge yet. Slow (no prefix-caching, no TP).
#   - verl path (THIS SCRIPT):    open framework; vLLM rollout backend,
#     prefix-caching, TP, dynamic-bsz, FSDP+offload — 10-30× faster on
#     long-trajectory rollouts. Cross-validates slyme algorithmically.
#
# verl is vendored at ThinkStream/verl/ (a customized fork of
# verl-project/verl with our recipe at verl/recipe_thinkstream/). We do NOT
# `pip install verl`; instead we run it in-place via PYTHONPATH so that
# (a) recipe edits take effect without reinstall and (b) verl can import
# `thinkstream.*` from the parent ThinkStream checkout.
#
# verl assumes the SFT checkpoint is on disk (Hugging Face format). Run
# scripts/sft_per_timestep.sh first; this script picks up the resulting
# checkpoint via $LLM.
#
# Usage:
#   LLM=output/agent-sft bash scripts/grpo_train_verl.sh
#
# Required env:
#   LLM             — SFT checkpoint path (HF format).
#
# Optional env (defaults shown):
#   NPROC           — GPUs per node (8)
#   GROUP_SIZE      — GRPO group size G (8) — verl supports up to ~16 cleanly
#   MAXLEN          — max prompt length (16384) — vLLM context cap
#   MAX_NEW_TOKEN   — max response length per turn (2048) — covers compress JSON
#   MAX_CHUNKS      — max turns per video (360 = 6 min × 1s/chunk)
#   GPU_MEM_UTIL    — vLLM gpu_memory_utilization (0.55 — leave room for FSDP)
#   TP_SIZE         — tensor_parallel_size for vLLM rollout (2 on 8-GPU node)
#   BATCH_SIZE      — videos per training step (8)
#   PPO_MINI_BS     — ppo_mini_batch_size (32)
#   LR              — learning rate (1e-6)
#   EPOCHS          — total_epochs (1)
#   SAVE_FREQ       — save every N steps (50)
#   TEST_FREQ       — eval on val every N steps (25)
#   RUN_NAME        — wandb experiment name (grpo-v126-verl)
#   WANDB_PROJECT   — wandb project (thinkstream-v12)
#   PARAM_OFFLOAD   — FSDP offload params to CPU (false). Enable for tight HBM.
#   OPTIMIZER_OFFLOAD — FSDP offload optimizer state (false).
#   ROLLOUT_BACKEND — rollout backend: vllm | sglang | hf (vllm).
#   TRAIN_PARQUET / VAL_PARQUET — flattened (video, question) parquets.
#                  If unset, we auto-build from data/agent_v5/final/*.jsonl
#                  via scripts/agent_data_v5/build_verl_parquet.py.

set -euo pipefail

LLM=${LLM:?'LLM= required (path to SFT checkpoint, e.g. output/agent-sft)'}

NPROC=${NPROC:-8}
# Conservative first-run defaults. Tune up after smoke-testing.
#   GROUP_SIZE × MAX_CHUNKS × BATCH_SIZE = vLLM requests per training step.
#   8 × 60 × 4 = 1920 reqs/step ≈ 5-15 min on 8×H20 with prefix cache.
#   8 × 360 × 8 = 23040 reqs/step → 60-90 min/step. Only enable once
#   smoke confirmed.
GROUP_SIZE=${GROUP_SIZE:-4}
MAXLEN=${MAXLEN:-16384}
# P0.4 fix (post-review 2026-05-01): MAX_NEW_TOKEN sets verl's
# rollout.response_length, which is the TOTAL stitched length across
# all chunks' user_blocks + assistant turns. NOT a per-turn budget.
# 60 chunks × ~120 tok/chunk ≈ 7K so we need ≥ 8K. Old default 2048
# would force the loop to break after ~16 chunks.
MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-16384}
MAX_CHUNKS=${MAX_CHUNKS:-60}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
TP_SIZE=${TP_SIZE:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
PPO_MINI_BS=${PPO_MINI_BS:-16}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
RUN_NAME=${RUN_NAME:-grpo-v126-verl}
WANDB_PROJECT=${WANDB_PROJECT:-thinkstream-v12}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
DATASET=${DATASET:-stream_agent_rl_traj}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${PROJECT_DIR}/verl"
RECIPE_DIR="${VERL_DIR}/recipe_thinkstream/configs"
RECIPE_NAME="thinkstream_grpo"

OUTPUT_DIR="${THINKSTREAM_OUTPUT_DIR:-${PROJECT_DIR}/output/${RUN_NAME}}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/agent_v5/final/train_rl_trajectories.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_DIR}/data/agent_v5/final/val_trajectories.jsonl}"

# verl's RLHFDataset reads parquet; auto-build from JSONL if user didn't
# supply a parquet directly.
TRAIN_PARQUET="${TRAIN_PARQUET:-${PROJECT_DIR}/data/agent_v5/final/train_rl.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${PROJECT_DIR}/data/agent_v5/final/val_rl.parquet}"

# MULTI_Q=1 → 1 video = 1 row, all questions co-evaluated (OVOBench-aligned).
# Default still flatten (video, question) for backward compat with existing
# parquets; new runs should set MULTI_Q=1 to use the proper trajectory shape.
MULTI_Q_FLAG=""
if [[ "${MULTI_Q:-0}" == "1" ]]; then
    MULTI_Q_FLAG="--multi_q"
fi

if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "Building train parquet from ${TRAIN_JSONL}…  (multi_q=${MULTI_Q:-0})"
    python3 "${PROJECT_DIR}/scripts/agent_data_v5/build_verl_parquet.py" \
        --jsonl "${TRAIN_JSONL}" --out "${TRAIN_PARQUET}" ${MULTI_Q_FLAG}
fi
if [[ ! -f "${VAL_PARQUET}" ]]; then
    echo "Building val parquet from ${VAL_JSONL}…  (multi_q=${MULTI_Q:-0})"
    python3 "${PROJECT_DIR}/scripts/agent_data_v5/build_verl_parquet.py" \
        --jsonl "${VAL_JSONL}" --out "${VAL_PARQUET}" ${MULTI_Q_FLAG}
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== ThinkStream GRPO via verl ==="
echo "Checkpoint:        ${LLM}"
echo "Vendored verl:     ${VERL_DIR}"
echo "Recipe dir:        ${RECIPE_DIR}"
echo "Recipe name:       ${RECIPE_NAME}"
echo "Train parquet:     ${TRAIN_PARQUET}"
echo "Val parquet:       ${VAL_PARQUET}"
echo "Output:            ${OUTPUT_DIR}"
echo "GPUs:              ${NPROC}"
echo "Rollout backend:   ${ROLLOUT_BACKEND}"
echo "TP size:           ${TP_SIZE}"
echo "Group size G:      ${GROUP_SIZE}"
echo "Max chunks:        ${MAX_CHUNKS}"
echo "Max prompt len:    ${MAXLEN}"
echo "Max new tokens:    ${MAX_NEW_TOKEN}"
echo "GPU mem util:      ${GPU_MEM_UTIL}"
echo "LR:                ${LR}"
echo "Epochs:            ${EPOCHS}"
echo "PPO mini bs:       ${PPO_MINI_BS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "FSDP param offload: ${PARAM_OFFLOAD}"
echo "FSDP opt offload:   ${OPTIMIZER_OFFLOAD}"
echo "================================="

# verl uses Ray; let it handle multi-GPU orchestration.
# We pass per-flag overrides on top of verl/recipe_thinkstream/configs/thinkstream_grpo.yaml.
#
# PYTHONPATH ordering matters:
#   ${VERL_DIR}     — vendored verl python package (in-place, no pip install)
#   ${PROJECT_DIR}  — ThinkStream package, so reward fn can import
#                     thinkstream.trainer.v12_rewards
export PYTHONPATH="${VERL_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
# Lets the recipe's compute_score read trajectory metadata (gold_action_per_chunk,
# ask_chunks) when verl's parquet column flattening drops nested dicts.
export THINKSTREAM_TRAJ_INDEX_PATH="${TRAIN_JSONL}"
# Where pre-extracted JPEG frames live (one subdir per video stem). The
# streaming agent loop reads this to inject per-chunk visual frames every
# turn. Empty / unset → loop falls back to text-only RL.
FRAMES_ROOT="${FRAMES_ROOT:-${PROJECT_DIR}/data/agent_v5/frames}"
export THINKSTREAM_FRAMES_ROOT="${FRAMES_ROOT}"

cd "${VERL_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path="${RECIPE_DIR}" \
    --config-name="${RECIPE_NAME}" \
    actor_rollout_ref.model.path="${LLM}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND} \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.response_length=${MAX_NEW_TOKEN} \
    actor_rollout_ref.rollout.prompt_length=${MAXLEN} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_CHUNKS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_CHUNKS} \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAXLEN} \
    data.max_response_length=${MAX_NEW_TOKEN} \
    reward.custom_reward_function.path="${VERL_DIR}/recipe_thinkstream/thinkstream.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.total_epochs=${EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${NPROC} \
    trainer.nnodes=1
