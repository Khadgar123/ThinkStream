#!/bin/bash
# Per-timestep agent SFT training script.
#
# v11.1 (2026-04-27): PHASE=sft is the recommended production path —
# trains on the SFT-disjoint pool (train_sft.jsonl, ~9.9k samples / 199
# videos), leaving train_rl.jsonl held out for the GDPO stage so RL
# cannot reward-hack via memorization on SFT-seen prompts.
#
# Legacy PHASE=mixed trains on the full union (train.jsonl / phase5)
# — kept for backward compatibility and single-stage baselines.
#
# Per-category training cases (1 | 2 | C1) are ablation-only knobs.
# C2 was removed in v11 — model-self-pick range moved to RL.
#
# Usage (production):
#   PHASE=sft bash scripts/sft_per_timestep.sh
#
# Backward compat (single-stage on full data):
#   PHASE=mixed bash scripts/sft_per_timestep.sh
#
# Ablation only (DO NOT chain into a curriculum):
#   PHASE=1  bash scripts/sft_per_timestep.sh   # basic silent+response
#   PHASE=2  bash scripts/sft_per_timestep.sh   # recall samples
#   PHASE=C1 bash scripts/sft_per_timestep.sh   # compress samples
#
# Environment variables:
#   PHASE       - sft (recommended) | mixed | 1 | 2 | C1
#   LLM         - Model path (default: Qwen/Qwen3-VL-8B)
#   NPROC       - GPUs per node (default: 8)
#   BSZ         - Per-device batch size (default: 8)
#   GRAD_ACCUM  - Gradient accumulation steps (default: 1)
#   EVAL_STEPS  - Eval frequency in optimizer steps (PHASE=sft, default 50)
#   EVAL_N      - Subsample size for in-loop eval (PHASE=sft, default 300)
#   EVAL_BSZ    - Per-device eval batch size (PHASE=sft, default = BSZ)
#   SAVE_LIMIT  - Max retained checkpoints (PHASE=sft, default 5; ~30-50GB each
#                 for 8B + zero-3, plus the best ckpt is always preserved)
#
# Step budget reference (BSZ=8 × NPROC=8 × GRAD_ACCUM=1 → eff. batch 64):
#   PHASE=sft   : 9,900 / 64 = 154 steps/epoch × 4 epochs = 616 steps
#   PHASE=mixed : 12,405 / 64 = 193 steps/epoch × 3 epochs = 579 steps

set -euo pipefail

PHASE=${PHASE:-sft}
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
    sft)
        # v11.1 production: SFT-disjoint pool (train_sft.jsonl).
        # train_rl.jsonl (~2.5k samples / 50 videos) is held out for
        # the GDPO stage; see thinkstream/sft/data_list.py for the
        # split manifest. epochs=4 gives ~39.6k samples-seen, matching
        # the previous PHASE=mixed (12.4k × 3) corpus exposure.
        #
        # v11.2: in-loop eval on stream_agent_val (1,550-sample held-out
        # video-disjoint pool). Subsampled to EVAL_N (default 300) so
        # one eval pass takes ~2 min on 8×GPU instead of ~10 min.
        # v12.5 default: stream_agent_sft_full (18,229 samples, 11.2x
        # recovery vs legacy stream_agent_sft=1,635 post-cap). The flat
        # file is the unpacked form of train_sft_trajectories.jsonl
        # produced by `python -m scripts.agent_data_v5.pass4`.
        # Override with DATASETS=stream_agent_sft for the v11.1 baseline.
        llm=${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
        datasets=${DATASETS:-stream_agent_sft_full}
        eval_datasets=${EVAL_DATASETS:-stream_agent_val}
        # v12.5: corpus 11.2x larger; reduce default epochs proportionally
        # so total steps stay in the v11.1 ballpark (was 4×154=616; now
        # 2×285=570 with effective_bsz=64). Override with EPOCHS=N.
        lr=${LR:-2e-5}; epochs=${EPOCHS:-2}
        run_name="agent-sft-v12.5"
        # Save aligned to eval cadence so every eval has a corresponding
        # ckpt to roll back to. load_best_model_at_end keeps the lowest
        # eval_loss ckpt even if it falls outside the rolling window.
        # NB: this OVERRIDES the global --save_strategy epoch below.
        extra_args="--eval_dataset_use stream_agent_val \
            --eval_max_samples ${EVAL_N:-300} \
            --eval_strategy steps \
            --eval_steps ${EVAL_STEPS:-50} \
            --per_device_eval_batch_size ${EVAL_BSZ:-${BSZ}} \
            --save_strategy steps \
            --save_steps ${EVAL_STEPS:-50} \
            --save_total_limit ${SAVE_LIMIT:-5} \
            --load_best_model_at_end True \
            --metric_for_best_model eval_loss \
            --greater_is_better False"
        ;;
    mixed)
        # Backward-compat: full union (train.jsonl). Use this only if
        # you do NOT plan to run GDPO afterwards, or for single-stage
        # baselines where SFT-seen / RL-disjoint distinction is moot.
        llm=${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
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
        echo "Unknown PHASE=$PHASE. Use: sft (production) | mixed (legacy) | 1 | 2 | C1 (ablation only)"
        exit 1
        ;;
esac

output_dir="${PROJECT_DIR}/output/${run_name}"
echo "=== Per-timestep Agent SFT ==="
echo "Phase:    ${PHASE}"
echo "Model:    ${llm}"
echo "Dataset:  ${datasets}"
echo "Eval:     ${eval_datasets:-none}"
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
    --save_strategy epoch \
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
