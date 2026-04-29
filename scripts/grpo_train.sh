#!/bin/bash
# GDPO-style GRPO training launcher (single RL stage; runs after SFT).
#
# Reads from a SFT checkpoint produced by `scripts/sft_per_timestep.sh`
# (PHASE=mixed) and runs the slyme `grpo` builder defined in
# `thinkstream/trainer/builder.py:build_grpo_train`.
#
# Reward + advantage design: NVIDIA GDPO 2601.05242 style — per-reward
# group-norm + weighted sum + batch-whiten. See:
#   - thinkstream/trainer/grpo.py (calc_rewards, compute_gdpo_advantages)
#   - thinkstream/trainer/gdpo_advantage.py (pure-tensor algorithm)
#   - docs/data_construction_zh.md §4 (training pipeline)
#   - docs/data_batch1_plan.md §5.3 (reward weights and masks)
#
# Usage:
#   LLM=output/agent-sft bash scripts/grpo_train.sh
#
# Required:
#   LLM         - SFT checkpoint path. Use output/agent-sft if you ran
#                 PHASE=sft (recommended). output/agent-mixed for the
#                 legacy single-stage run.
#
# Optional environment variables (defaults shown):
#   NPROC               - GPUs per node (8)
#   GROUP_SIZE          - GRPO group size G (4)
#   MICRO_BATCH         - per-device micro batch (4)
#   ROLLOUT_MAX_CHUNKS  - max chunks per rollout (default 100 for
#                         v12.5 trajectory rollouts; was 30 for legacy
#                         single-question). Audit on RL trajectories:
#                           p10/p50/p90 max(ask_chunks) = 9/20/87, max=187
#                         At cap=30 → 41% truncation (later questions
#                         never get answered → spurious 0 outcome).
#                         Cap=100 covers ~95% of RL trajectories.
#                         Lower to 30 if compute-bound and using
#                         legacy stream_agent_rl single-question dataset.
#   ROLLOUT_MAX_NEW_TOK - max new tokens per chunk generation (128)
#   ROLLOUT_TEMP        - rollout temperature (1.0)
#   LR                  - learning rate (5e-7)
#   EPOCHS              - num train epochs (2)
#   BETA                - KL beta (1e-3)
#   DATASET             - dataset_use key from data_list.py.
#                         Default: stream_agent_rl (held-out 50 vids /
#                         ~2.5k samples — disjoint from SFT). Use
#                         stream_agent_p5 only for ablations where
#                         RL/SFT-disjoint distinction is intentionally
#                         dropped.
#   SAVE_STEPS          - save every N steps (200). Total steps on
#                         train_rl ≈ 624 → 3 ckpts at ~16GB each.
#                         slyme has no save_total_limit; bump SAVE_STEPS
#                         higher if disk-bound, lower for finer rollback.
#   AUDIT_DIR           - GDPO per-step audit dir (auto: $LLM/../grpo-audit)
#   RUN_NAME            - W&B run name (agent-grpo)

set -euo pipefail

LLM=${LLM:?'LLM= required (path to SFT checkpoint, e.g. output/agent-sft)'}

NPROC=${NPROC:-8}
GROUP_SIZE=${GROUP_SIZE:-4}
MICRO_BATCH=${MICRO_BATCH:-4}
ROLLOUT_MAX_CHUNKS=${ROLLOUT_MAX_CHUNKS:-100}
ROLLOUT_MAX_NEW_TOK=${ROLLOUT_MAX_NEW_TOK:-128}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
LR=${LR:-5e-7}
EPOCHS=${EPOCHS:-2}
BETA=${BETA:-1e-3}
# v12.5 default: stream_agent_rl_traj (300 trajectories / 15,294 samples).
# Detected by grpo.calc_rewards via raw_sample.questions field → routes to
# _calc_rewards_v12_trajectory (multi-question per-ask scoring + per-chunk
# silent_quality). Override with DATASET=stream_agent_rl for legacy
# single-question flat format (1,635 samples, post-MAX_SAMPLES_PER_VIDEO=15
# cap that v12.5 retired).
DATASET=${DATASET:-stream_agent_rl_traj}
# slyme/deepslyme has no save_total_limit Ref registered — older runs had
# unbounded ckpts. Default save_steps=200 gives ~3 ckpts over the 624-step
# run on train_rl.jsonl (≈ 48GB on disk). Lower SAVE_STEPS for finer-grained
# rollback at the cost of disk; manually delete old ckpts between runs.
SAVE_STEPS=${SAVE_STEPS:-200}
RUN_NAME=${RUN_NAME:-agent-grpo}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED="${SCRIPT_DIR}/zero3.json"
ENTRY="${PROJECT_DIR}/thinkstream/train.py"

OUTPUT_DIR="${PROJECT_DIR}/output/${RUN_NAME}"
AUDIT_DIR=${AUDIT_DIR:-${OUTPUT_DIR}/audit}

mkdir -p "${OUTPUT_DIR}" "${AUDIT_DIR}"

echo "=== ThinkStream GDPO RL (single stage) ==="
echo "Checkpoint:        ${LLM}"
echo "Dataset:           ${DATASET}"
echo "Output:            ${OUTPUT_DIR}"
echo "Audit:             ${AUDIT_DIR}"
echo "GPUs:              ${NPROC}"
echo "Group size G:      ${GROUP_SIZE}"
echo "Micro batch:       ${MICRO_BATCH}"
echo "Rollout max chunks:${ROLLOUT_MAX_CHUNKS}"
echo "Rollout temp:      ${ROLLOUT_TEMP}"
echo "LR:                ${LR}"
echo "Epochs:            ${EPOCHS}"
echo "KL beta:           ${BETA}"
echo "Save every:        ${SAVE_STEPS} steps"
echo "====================================="

# Audit dir is read by grpo.py via env var (see _grpo_audit_writers).
# Tail-friendly: `tail -f ${AUDIT_DIR}/grpo_step.jsonl | jq .` while training.
export THINKSTREAM_AUDIT_DIR="${AUDIT_DIR}"
export THINKSTREAM_OUTPUT_DIR="${OUTPUT_DIR}"

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC} \
    "${ENTRY}" grpo \
    --args.train.deepspeed "${DEEPSPEED}" \
    --args.model.name_or_path "${LLM}" \
    --args.model.model_type qwen3vl \
    --args.model.max_length 16384 \
    --args.data.dataset_use "${DATASET}" \
    --args.train.output_dir "${OUTPUT_DIR}" \
    --args.train.num_train_epochs ${EPOCHS} \
    --args.train.per_device_train_batch_size 1 \
    --args.train.gradient_accumulation_steps 1 \
    --args.train.learning_rate ${LR} \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.save_steps ${SAVE_STEPS} \
    --args.train.bf16 True \
    --args.train.group_size ${GROUP_SIZE} \
    --args.train.micro_batch_size ${MICRO_BATCH} \
    --args.train.beta ${BETA} \
    --args.train.rollout_max_new_tokens ${ROLLOUT_MAX_NEW_TOK} \
    --args.train.rollout_max_think_tokens 60 \
    --args.train.rollout_temperature ${ROLLOUT_TEMP} \
    --args.train.rollout_top_k 50 \
    --args.train.rollout_top_p 0.95 \
    --args.train.rollout_max_chunks ${ROLLOUT_MAX_CHUNKS} \
    --args.train.rollout_min_pixels 100352 \
    --args.train.rollout_max_pixels 150528 \
    --args.train.rollout_fpc 2.0 \
    --args.train.time_reward_window 5 \
    --args.train.time_reward_slack 3.0 \
    --args.train.dataloader.num_workers 4 \
    --args.train.dataloader.pin_memory True
