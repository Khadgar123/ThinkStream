#!/bin/bash
# Per-timestep agent SFT training script.
#
# Usage:
#   PHASE=1 bash scripts/sft_per_timestep.sh                    # Phase 1
#   PHASE=C1 LLM=output/agent-phase2 bash scripts/sft_per_timestep.sh  # C1 from P2 ckpt
#
# Environment variables:
#   PHASE       - Training phase: 1, 2, C1, C2, 5 (default: 1)
#   LLM         - Model path (required for Phase 2+)
#   NPROC       - GPUs per node (default: 8)
#   BSZ         - Per-device batch size (default: 8)
#   GRAD_ACCUM  - Gradient accumulation steps (default: 1)
#   MODEL_TYPE  - qwen2.5vl | qwen3vl (default: auto-detect)

set -euo pipefail

PHASE=${PHASE:-1}
NPROC=${NPROC:-8}
BSZ=${BSZ:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED="${SCRIPT_DIR}/zero3.json"
ENTRY="${PROJECT_DIR}/thinkstream/sft/train.py"

case $PHASE in
    1)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_p1
        lr=1e-5; epochs=3
        run_name="agent-phase1"
        ;;
    2)
        llm=${LLM:?'Phase 2 requires LLM= (Phase 1 checkpoint)'}
        datasets=stream_agent_p2
        lr=5e-6; epochs=3
        run_name="agent-phase2"
        ;;
    C1)
        llm=${LLM:?'Phase C1 requires LLM= (Phase 2 checkpoint)'}
        datasets=stream_agent_c1
        lr=3e-6; epochs=2
        run_name="agent-c1"
        ;;
    C2)
        llm=${LLM:?'Phase C2 requires LLM= (C1 checkpoint)'}
        datasets=stream_agent_c2
        lr=2e-6; epochs=2
        run_name="agent-c2"
        ;;
    5)
        llm=${LLM:?'Phase 5 requires LLM= (C2 checkpoint)'}
        datasets=stream_agent_p5
        lr=1e-6; epochs=1
        run_name="agent-phase5"
        ;;
    *)
        echo "Unknown PHASE=$PHASE. Use: 1, 2, C1, C2, 5"
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
    --max_sample_tokens 8192 \
    --torch_empty_cache_steps 1 \
    --dataloader_num_workers 4 \
    --video_min_pixels 100352 \
    --video_max_pixels 150528 \
    --video_fps 1.0 \
    --report_to wandb \
    --run_name "${run_name}"
