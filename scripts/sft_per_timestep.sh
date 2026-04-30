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
        # v12.6 default: stream_agent_sft (LLaMA-Factory ShareGPT messages
        # format; pass5_messages.py converts trajectory→messages preserving
        # all 18,229 samples). This is the canonical entry — flat
        # train_sft_full.jsonl is kept as stream_agent_sft_full for
        # backward-compat with archived ablations only.
        # Override with DATASETS=stream_agent_sft_full to use legacy flat.
        llm=${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
        datasets=${DATASETS:-stream_agent_sft}
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
        # v12.6: --protocol_version flag removed from DataArguments (v12 is
        # now the only supported protocol — see thinkstream/sft/argument.py).
        if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
            extra_args="${extra_args} --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
        fi
        ;;
    mixed|1|2|C1)
        # v12.6: legacy PHASEs (mixed, 1, 2, C1) are gated. They pointed at
        # archived flat datasets (stream_agent_p1/p2/p5/c1) that don't have
        # the messages key required by preprocess_per_timestep. To re-enable,
        # convert those datasets through pass5_messages.py first OR use
        # PHASE=sft on the canonical messages dataset.
        echo "ERROR: PHASE=$PHASE is archived in v12.6."
        echo "  Legacy phases pointed at flat *_full.jsonl datasets (no"
        echo "  'messages' key); preprocess_per_timestep now requires"
        echo "  messages format. To run an ablation:"
        echo "    1) Convert the flat dataset:"
        echo "       python -m scripts.agent_data_v5.pass5_messages \\"
        echo "         --input flat --final-dir <dir>"
        echo "    2) Add a stream_agent_<name> entry in"
        echo "       thinkstream/sft/data_list.py pointing at the .messages.jsonl"
        echo "    3) Run with PHASE=sft DATASETS=stream_agent_<name>"
        exit 2
        ;;
    *)
        echo "Unknown PHASE=$PHASE. Use: sft (production)."
        echo "  Legacy phases (mixed | 1 | 2 | C1) archived in v12.6 — see error above."
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
