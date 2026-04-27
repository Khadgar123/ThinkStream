#!/bin/bash
# Evaluate checkpoint-100 on test set.

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output/agent-sft/checkpoint-100}"
DATASET="${DATASET:-stream_agent_test}"
BSZ="${BSZ:-8}"
NPROC="${NPROC:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENTRY="${PROJECT_DIR}/thinkstream/sft/eval_test.py"

echo "=== Evaluating on Test Set ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset:    ${DATASET}"
echo "Batch:      ${BSZ} per device × ${NPROC} GPUs"
echo "================================"

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC} \
    ${ENTRY} \
    --checkpoint_dir "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --batch_size ${BSZ} \
    --model_name_or_path "${CHECKPOINT}" \
    --bf16 True \
    --model_max_length 16384 \
    --max_sample_tokens 12000 \
    --video_min_pixels 100352 \
    --video_max_pixels 150528 \
    --video_fps 1.0 \
    --dataloader_num_workers 4 \
    --output_dir "${PROJECT_DIR}/output/agent-sft-test-eval"
