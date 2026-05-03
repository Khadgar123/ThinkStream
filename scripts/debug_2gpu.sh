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
    --video_min_pixels 130000 \
    --video_max_pixels 220000 \
    --video_fps 2.0 \
    --report_to wandb \
    --run_name agent-sft-debug \
    --protocol_version v12 \
    --class_balanced_sampler False \
    --focal_alpha_action False \
    --dataloader_num_workers 0

echo ""
echo "SFT debug complete. Checking checkpoint..."

# ── RL debug (verl, tiny budget) ─────────────────────────────────────
SFT_OUT="${PROJECT_DIR}/output/agent-sft-debug"
if [ ! -d "${SFT_OUT}" ]; then
    echo "WARNING: No SFT output dir ${SFT_OUT}, skipping RL."
    exit 0
fi
CKPT="${SFT_OUT}"
echo "Using SFT output: ${CKPT}"

LLM="${CKPT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
NPROC=2 \
GROUP_SIZE="${GROUP_SIZE:-2}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
PPO_MINI_BS="${PPO_MINI_BS:-1}" \
MAX_CHUNKS="${MAX_CHUNKS:-8}" \
MAXLEN="${MAXLEN:-4096}" \
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-4096}" \
MULTI_Q="${MULTI_Q:-1}" \
RUN_NAME="${RUN_NAME:-agent-verl-debug}" \
THINKSTREAM_OUTPUT_DIR="${PROJECT_DIR}/output/agent-verl-debug" \
bash "${SCRIPT_DIR}/grpo_train_verl.sh"

echo "RL debug complete."
