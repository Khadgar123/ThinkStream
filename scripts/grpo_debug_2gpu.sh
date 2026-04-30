#!/bin/bash
# 2-GPU RL debug launcher — verifies pre-extracted frames (no online decode).
#
# Uses:
#   - 2 GPUs (CUDA_VISIBLE_DEVICES=6,7)
#   - Backup data (agent_v5_current_backup/final)
#   - Pre-extracted frames under data/agent_v5/frames/
#   - Short run (MAX_STEPS=5) to verify rollout speed + loss
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LLM="${LLM:-${PROJECT_DIR}/output/agent-sft/checkpoint-200}"
OUTPUT_DIR="${PROJECT_DIR}/output/agent-grpo-debug-$(date +%Y%m%d_%H%M%S)"
AUDIT_DIR="${OUTPUT_DIR}/audit"
mkdir -p "${OUTPUT_DIR}" "${AUDIT_DIR}"

# Use backup trajectory data while agent_v5 is being re-generated
export AGENT_DATA_DIR="${PROJECT_DIR}/data/agent_v5_current_backup/final"
export THINKSTREAM_AUDIT_DIR="${AUDIT_DIR}"
export THINKSTREAM_OUTPUT_DIR="${OUTPUT_DIR}"

# Pre-extracted frames: eliminates 300s+ torchcodec online decode per rollout
FRAMES_ROOT="${PROJECT_DIR}/data/agent_v5/frames"
VIDEO_ROOT="/home/tione/notebook/gaozhenkun/hzh/data/datasets/VideoMind-Dataset/cosmo_cap/videos"

echo "=== ThinkStream RL Debug (2-GPU, pre-extracted frames) ==="
echo "Checkpoint:  ${LLM}"
echo "Data dir:    ${AGENT_DATA_DIR}"
echo "Frames:      ${FRAMES_ROOT}"
echo "Video root:  ${VIDEO_ROOT}"
echo "Output:      ${OUTPUT_DIR}"
echo "=========================================================="

ENV_DIR="/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream"
CUDA_VISIBLE_DEVICES=6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"${ENV_DIR}/bin/torchrun" --nproc_per_node=2 \
    "${PROJECT_DIR}/thinkstream/train.py" grpo \
    --args.train.deepspeed "${SCRIPT_DIR}/zero3.json" \
    --args.model.name_or_path "${LLM}" \
    --args.model.model_type qwen3vl \
    --args.data.dataset_use stream_agent_rl_traj \
    --args.train.output_dir "${OUTPUT_DIR}" \
    --args.train.num_train_epochs 1 \
    --args.train.per_device_train_batch_size 1 \
    --args.train.gradient_accumulation_steps 1 \
    --args.train.learning_rate 5e-7 \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.bf16 True \
    --args.train.group_size 4 \
    --args.train.micro_batch_size 2 \
    --args.train.beta 1e-3 \
    --args.train.rollout_max_new_tokens 128 \
    --args.train.rollout_max_think_tokens 60 \
    --args.train.rollout_temperature 1.0 \
    --args.train.rollout_top_k 50 \
    --args.train.rollout_top_p 0.95 \
    --args.train.rollout_max_chunks 20 \
    --args.train.rollout_min_pixels 100352 \
    --args.train.rollout_max_pixels 150528 \
    --args.train.rollout_fpc 2.0 \
    --args.train.time_reward_window 5 \
    --args.train.time_reward_slack 3.0 \
    --args.train.vllm_rollout_frames_root "${FRAMES_ROOT}" \
    --args.train.vllm_rollout_video_root "${VIDEO_ROOT}" \
    --args.train.dataloader.num_workers 2 \
    --args.train.dataloader.pin_memory True \
    --args.train.max_steps 5
