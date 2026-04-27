#!/bin/bash
# Per-timestep agent SFT training script.
#
# v11 (2026-04-27): PHASE=mixed is the ONLY production path. Single-stage
# SFT on the full mixed dataset (stream_agent_p5). After this SFT,
# proceed to GDPO RL via the GRPO launcher (see docs/data_batch1_plan.md §8).
#
# Per-category training cases (1 | 2 | C1) are ablation-only knobs.
# C2 was removed in v11 — model-self-pick range moved to RL.
#
# Usage (production):
#   PHASE=mixed bash scripts/sft_per_timestep.sh
#
# Ablation only (DO NOT chain into a curriculum):
#   PHASE=1  bash scripts/sft_per_timestep.sh   # basic silent+response
#   PHASE=2  bash scripts/sft_per_timestep.sh   # recall samples
#   PHASE=C1 bash scripts/sft_per_timestep.sh   # compress samples
#
# Environment variables:
#   PHASE       - mixed (recommended) | 1 | 2 | C1
#   LLM         - Model path (default: Qwen/Qwen3-VL-8B)
#   NPROC       - GPUs per node (default: 8)
#   BSZ         - Per-device batch size (default: 8)
#   GRAD_ACCUM  - Gradient accumulation steps (default: 1)

set -euo pipefail

PHASE=${PHASE:-mixed}
NPROC=${NPROC:-8}
BSZ=${BSZ:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED="${SCRIPT_DIR}/zero3.json"
ENTRY="${PROJECT_DIR}/thinkstream/sft/train.py"

# extra_args is appended in the case-block when phase needs special flags
extra_args=""

case $PHASE in
    mixed)
        # v10 single-phase: full data + class-balanced sampler from base.
        llm=${LLM:-Qwen/Qwen3-VL-8B}
        datasets=stream_agent_p5            # already the merged mix
        lr=2e-5; epochs=3
        run_name="agent-mixed"
        # class_balanced_sampler defaults to True in DataArguments;
        # set False here if you want a uniform-sampling ablation.
        ;;
    # Ablation: train only on basic silent+response samples.
    1)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_p1
        lr=1e-5; epochs=3
        run_name="agent-ablate-p1"
        extra_args="--class_balanced_sampler False"
        ;;
    # Ablation: train only on recall samples.
    2)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_p2
        lr=5e-6; epochs=3
        run_name="agent-ablate-p2"
        extra_args="--class_balanced_sampler False"
        ;;
    # Ablation: train only on compress samples (system trigger + teacher gold range).
    C1)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_c1
        lr=3e-6; epochs=2
        run_name="agent-ablate-c1"
        extra_args="--class_balanced_sampler False"
        ;;
    *)
        echo "Unknown PHASE=$PHASE. Use: mixed (production) | 1 | 2 | C1 (ablation only)"
        exit 1
        ;;
esac

output_dir="${PROJECT_DIR}/output/${run_name}"
echo "=== Per-timestep Agent SFT ==="
echo "Phase:    ${PHASE}"
echo "Model:    ${llm}"
echo "Dataset:  ${datasets}"
echo "LR:       ${lr}"
echo "Epochs:   ${epochs}"
echo "Output:   ${output_dir}"
echo "GPUs:     ${NPROC}"
echo "Batch:    ${BSZ} × ${GRAD_ACCUM} accum"
echo "=============================="

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC} \
    ${ENTRY} \
    --deepspeed ${DEEPSPEED} \
    --model_name_or_path "${llm}" \
    --dataset_use "${datasets}" \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir "${output_dir}" \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --model_max_length 16384 \
    --max_sample_tokens 12000 \
    --torch_empty_cache_steps 1 \
    --dataloader_num_workers 4 \
    --video_min_pixels 100352 \
    --video_max_pixels 150528 \
    --video_fps 1.0 \
    --report_to wandb \
    --run_name "${run_name}" \
    ${extra_args}
