#!/bin/bash
# 2-GPU quick debug launcher for SFT + optional RL.
# Uses GPUs 6,7 while 0-5 are occupied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# Conda
set +u
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /home/tione/notebook/gaozhenkun/hzh/envs/thinkstream
set -u

export AGENT_DATA_DIR=${AGENT_DATA_DIR:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_current_backup/final}

DEEPSPEED="${SCRIPT_DIR}/zero3.json"
SFT_ENTRY="${PROJECT_DIR}/thinkstream/sft/train.py"

# ── SFT debug (2 GPUs, tiny budget for fast iteration) ───────────────
CUDA_VISIBLE_DEVICES=6,7 \
torchrun --nproc_per_node=2 \
    "${SFT_ENTRY}" \
    --deepspeed "${DEEPSPEED}" \
    --model_name_or_path /home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct \
    --dataset_use stream_agent_sft_full \
    --eval_dataset_use stream_agent_val \
    --eval_max_samples 50 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir "${PROJECT_DIR}/output/agent-sft-debug" \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --model_max_length 4096 \
    --max_sample_tokens 3000 \
    --torch_empty_cache_steps 1 \
    --dataloader_num_workers 4 \
    --video_min_pixels 100352 \
    --video_max_pixels 150528 \
    --video_fps 1.0 \
    --report_to wandb \
    --run_name agent-sft-debug \
    --protocol_version v12 \
    --class_balanced_sampler False \
    --focal_alpha_action False \
    --dataloader_num_workers 0

echo ""
echo "SFT debug complete. Checking checkpoint..."

# ── RL debug (1 epoch, tiny budget) ──────────────────────────────────
SFT_OUT="${PROJECT_DIR}/output/agent-sft-debug"
if [ ! -d "${SFT_OUT}" ]; then
    echo "WARNING: No SFT output dir ${SFT_OUT}, skipping RL."
    exit 0
fi
CKPT="${SFT_OUT}"
echo "Using SFT output: ${CKPT}"

RL_ENTRY="${PROJECT_DIR}/thinkstream/train.py"
CUDA_VISIBLE_DEVICES=6,7 \
torchrun --nproc_per_node=2 \
    "${RL_ENTRY}" grpo \
    --args.train.deepspeed "${DEEPSPEED}" \
    --args.model.name_or_path "${CKPT}" \
    --args.model.model_type qwen3vl \
    --args.data.dataset_use stream_agent_rl_traj \
    --args.train.output_dir "${PROJECT_DIR}/output/agent-grpo-debug" \
    --args.train.num_train_epochs 1 \
    --args.train.max_steps 30 \
    --args.train.per_device_train_batch_size 1 \
    --args.train.gradient_accumulation_steps 1 \
    --args.train.learning_rate 5e-7 \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.save_steps 30 \
    --args.train.bf16 True \
    --args.train.group_size 4 \
    --args.train.micro_batch_size 2 \
    --args.train.beta 1e-3 \
    --args.train.rollout_max_new_tokens 128 \
    --args.train.rollout_max_think_tokens 60 \
    --args.train.rollout_temperature 1.0 \
    --args.train.rollout_top_k 50 \
    --args.train.rollout_top_p 0.95 \
    --args.train.rollout_max_chunks 100 \
    --args.train.rollout_min_pixels 100352 \
    --args.train.rollout_max_pixels 150528 \
    --args.train.rollout_fpc 2.0 \
    --args.train.time_reward_window 5 \
    --args.train.time_reward_slack 3.0 \
    --args.train.dataloader.num_workers 4 \
    --args.train.dataloader.pin_memory True

echo "RL debug complete."
