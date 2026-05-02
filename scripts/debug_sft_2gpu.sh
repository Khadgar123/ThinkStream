#!/bin/bash
# 2-GPU SFT debug launcher — verifies input/output/loss/update/save pipeline.
#
# NOTE on resolution: this script targets data/agent_v5_current_backup/final
# (v12.5 SFT data). The pixel budgets below (min=100352, max=150528) match
# THAT backup. For v12.13 data (data/agent_v5/final/), use:
#   --video_min_pixels 130000 --video_max_pixels 220000
# matching scripts/agent_data_v5/config.py:RUNTIME_MM_PROCESSOR_KWARGS.
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

CUDA_VISIBLE_DEVICES=6,7 \
torchrun --nproc_per_node=2 \
    "${SFT_ENTRY}" \
    --deepspeed "${DEEPSPEED}" \
    --model_name_or_path /home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct \
    --dataset_use stream_agent_sft \
    --eval_dataset_use stream_agent_val \
    --eval_max_samples 50 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir "${PROJECT_DIR}/output/agent-sft-debug" \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 3 \
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
    --video_min_pixels 100352 \
    --video_max_pixels 150528 \
    --video_fps 1.0 \
    --report_to none \
    --dataloader_num_workers 0

echo ""
echo "=== SFT debug complete ==="
ls -la "${PROJECT_DIR}/output/agent-sft-debug/"
