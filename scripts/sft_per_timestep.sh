#!/bin/bash
# Per-timestep agent SFT training script.
#
# v10 RECOMMENDED: PHASE=mixed — single-phase SFT on full mixed data with
# class-balanced sampler. Backed by NVIDIA Front-Loading + ICML'25 Data
# Mixing studies showing quality-weighted single-phase ≈ tuned curriculum,
# and 2025 streaming-VLM literature (StreamingVLM / LiveCC) reaching SOTA
# without phase-by-phase curriculum.
#
# Usage (recommended):
#   PHASE=mixed bash scripts/sft_per_timestep.sh
#
# Legacy 5-phase curriculum (kept for ablation only):
#   PHASE=1  bash scripts/sft_per_timestep.sh
#   PHASE=C1 LLM=output/agent-phase2 bash scripts/sft_per_timestep.sh
#
# Environment variables:
#   PHASE       - mixed (recommended) | 1 | 2 | C1 | C2 | 5
#   LLM         - Model path (default: Qwen/Qwen3-VL-8B for mixed/1)
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
    1)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_p1
        lr=1e-5; epochs=3
        run_name="agent-phase1"
        extra_args="--class_balanced_sampler False"
        ;;
    2)
        llm=${LLM:?'Phase 2 requires LLM= (Phase 1 checkpoint)'}
        datasets=stream_agent_p2
        lr=5e-6; epochs=3
        run_name="agent-phase2"
        extra_args="--class_balanced_sampler False"
        ;;
    C1)
        llm=${LLM:?'Phase C1 requires LLM= (Phase 2 checkpoint)'}
        datasets=stream_agent_c1
        lr=3e-6; epochs=2
        run_name="agent-c1"
        extra_args="--class_balanced_sampler False"
        ;;
    C2)
        llm=${LLM:?'Phase C2 requires LLM= (C1 checkpoint)'}
        datasets=stream_agent_c2
        lr=2e-6; epochs=2
        run_name="agent-c2"
        extra_args="--class_balanced_sampler False"
        ;;
    5)
        llm=${LLM:?'Phase 5 requires LLM= (C2 checkpoint)'}
        datasets=stream_agent_p5
        lr=1e-6; epochs=1
        run_name="agent-phase5"
        extra_args="--class_balanced_sampler False"
        ;;
    *)
        echo "Unknown PHASE=$PHASE. Use: mixed (recommended) | 1 | 2 | C1 | C2 | 5"
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
