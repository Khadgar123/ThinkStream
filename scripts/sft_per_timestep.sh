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
#   BSZ         - Per-device batch size (default: 4)
#   GRAD_ACCUM  - Gradient accumulation steps (default: 2)
#   EVAL_STEPS  - Eval frequency in optimizer steps (PHASE=sft, default 50)
#   EVAL_N      - Subsample size for in-loop eval (PHASE=sft, default 300).
#                 Set 0 to evaluate the full eval dataset.
#   EVAL_BSZ    - Per-device eval batch size (PHASE=sft, default = BSZ)
#   MAX_STEPS   - Optional optimizer-step cap for very large batches
#   SAVE_LIMIT  - Max retained checkpoints (PHASE=sft, default 0 = no rolling
#                 deletion; 8B + zero-3 checkpoints can be ~30-50GB each)
#   THINKSTREAM_DATA_ROOT / AGENT_DATA_DIR
#               - Generated batch root. Default: data/agent_v5.
#   THINKSTREAM_FINAL_DIR
#               - Optional rendered messages dir. If unset and
#                 rendered/$FRAME_PROTOCOL exists, this script uses it.
#   FRAME_PROTOCOL / THINKSTREAM_FRAME_PROTOCOL
#               - ts_image | video_meta. Must match the rendered SFT
#                 messages and later RL/eval protocol.
#   INCLUDE_FAILED_VERIFICATION
#               - False drops verifier-failed samples for the main cold-start
#                 SFT path. Override to True only for robustness/continuity
#                 ablations where verifier failures are intentionally kept.
#   MAX_SAMPLE_TOKENS
#               - Overlong filter threshold. Default 16384 matches
#                 model_max_length, so batch3 keeps the full rendered SFT set.
#                 Set 0 to disable token filtering entirely.
#   CLASS_LOSS_TARGET_RATIOS
#               - Optional sample_type target ratios, e.g.
#                 silent=0.35,response=0.25,recall=0.25,compress=0.15.
#                 Uses weighted loss instead of physically duplicating rows.
#   CLASS_LOSS_ALPHA / CLASS_LOSS_MAX_WEIGHT
#               - Reweighting strength and clamp.
#   GROUP_BY_MODALITY
#               - 1 by default. Keeps text-only compress rows and visual rows
#                 in separate global batches to avoid ZeRO3 ranks taking
#                 different vision-module paths.
#   THINKSTREAM_ENV
#               - Conda/venv path for SFT. Defaults to the local
#                 envs/thinkstream env when present, so bare shell launches do
#                 not accidentally use /root/miniconda3.
#   TORCHRUN_BIN
#               - Explicit torchrun path override.
#   DRY_RUN     - Set to 1 to print resolved config without launching.
#
# Step budget:
#   effective_batch = BSZ × NPROC × GRAD_ACCUM.
#   PHASE=sft default keeps exposure moderate for the v12 messages corpus:
#   2 epochs at effective_batch=64. BSZ=4 avoids the video_meta CE-logits OOM
#   seen with 32-frame prompts on 8×96GB H20 while preserving the old global
#   batch through GRAD_ACCUM=2. Set MAX_STEPS to cap very large batches.

set -euo pipefail

PHASE=${PHASE:-sft}
NPROC=${NPROC:-8}
BSZ=${BSZ:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED="${SCRIPT_DIR}/zero3.json"
ENTRY="${PROJECT_DIR}/thinkstream/sft/train.py"
PARENT_DIR="$(dirname "${PROJECT_DIR}")"
if [[ -x "${PARENT_DIR}/envs/thinkstream/bin/torchrun" ]]; then
    DEFAULT_ENV="${PARENT_DIR}/envs/thinkstream"
else
    DEFAULT_ENV="${PROJECT_DIR}/envs/thinkstream"
fi
THINKSTREAM_ENV="${THINKSTREAM_ENV:-${DEFAULT_ENV}}"
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    if [[ -x "${THINKSTREAM_ENV}/bin/torchrun" ]]; then
        TORCHRUN_BIN="${THINKSTREAM_ENV}/bin/torchrun"
    else
        TORCHRUN_BIN="$(command -v torchrun)"
    fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${THINKSTREAM_ENV}/bin/python" ]]; then
        PYTHON_BIN="${THINKSTREAM_ENV}/bin/python"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
AGENT_DATA_ROOT="${THINKSTREAM_DATA_ROOT:-${AGENT_DATA_DIR:-${PROJECT_DIR}/data/agent_v5}}"
if [[ "${AGENT_DATA_ROOT}" == */final ]]; then
    AGENT_DATA_ROOT="$(dirname "${AGENT_DATA_ROOT}")"
fi
FRAME_PROTOCOL="${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-ts_image}}"
INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION:-False}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-16384}"
TORCH_EMPTY_CACHE_STEPS="${TORCH_EMPTY_CACHE_STEPS:-0}"
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS:-}"
CLASS_LOSS_ALPHA="${CLASS_LOSS_ALPHA:-1.0}"
CLASS_LOSS_MAX_WEIGHT="${CLASS_LOSS_MAX_WEIGHT:-8.0}"
GROUP_BY_MODALITY="${GROUP_BY_MODALITY:-1}"
export THINKSTREAM_GROUP_BY_MODALITY="${GROUP_BY_MODALITY}"
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
if [[ -z "${THINKSTREAM_FINAL_DIR:-}" ]]; then
    if [[ -d "${AGENT_DATA_ROOT}/rendered/${FRAME_PROTOCOL}" ]]; then
        export THINKSTREAM_FINAL_DIR="${AGENT_DATA_ROOT}/rendered/${FRAME_PROTOCOL}"
    else
        export THINKSTREAM_FINAL_DIR="${AGENT_DATA_ROOT}/final"
    fi
fi

# extra_args is appended in the case-block when phase needs special flags
extra_args=""

case $PHASE in
    sft)
        # Production: SFT-disjoint messages pool. train_rl trajectories
        # stay held out for verl GRPO so the policy does not optimize
        # rewards on prompts it memorized during SFT.
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
        eval_n=${EVAL_N:-300}
        # v12.x: keep the default exposure conservative; exact steps scale
        # with the current batch size. Override with EPOCHS=N or MAX_STEPS=N.
        lr=${LR:-2e-5}; epochs=${EPOCHS:-2}
        run_name="${RUN_NAME:-agent-sft-v12.26-${FRAME_PROTOCOL}}"
        # Save aligned to eval cadence so every eval has a corresponding
        # ckpt to roll back to. load_best_model_at_end keeps the lowest
        # eval_loss ckpt even if it falls outside the rolling window.
        # Save/eval cadence is step-based for the production phase.
        extra_args="--eval_dataset_use ${eval_datasets} \
            --eval_strategy steps \
            --eval_steps ${EVAL_STEPS:-50} \
            --per_device_eval_batch_size ${EVAL_BSZ:-${BSZ}} \
            --save_strategy steps \
            --save_steps ${EVAL_STEPS:-50} \
            --save_total_limit ${SAVE_LIMIT:-0} \
            --load_best_model_at_end True \
            --metric_for_best_model eval_loss \
            --greater_is_better False"
        if [[ "${eval_n}" != "0" ]]; then
            extra_args="${extra_args} --eval_max_samples ${eval_n}"
        fi
        # v12.6: --protocol_version flag removed from DataArguments (v12 is
        # now the only supported protocol — see thinkstream/sft/argument.py).
        if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
            extra_args="${extra_args} --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
        fi
        if [ -n "${MAX_STEPS:-}" ]; then
            extra_args="${extra_args} --max_steps ${MAX_STEPS}"
        fi
        if [ -n "${CLASS_LOSS_TARGET_RATIOS}" ]; then
            extra_args="${extra_args} --class_loss_target_ratios ${CLASS_LOSS_TARGET_RATIOS}"
        fi
        extra_args="${extra_args} --class_loss_alpha ${CLASS_LOSS_ALPHA}"
        extra_args="${extra_args} --class_loss_max_weight ${CLASS_LOSS_MAX_WEIGHT}"
        if [[ "${TORCH_EMPTY_CACHE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
            extra_args="${extra_args} --torch_empty_cache_steps ${TORCH_EMPTY_CACHE_STEPS}"
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
echo "Data:     ${AGENT_DATA_ROOT}"
echo "Final:    ${THINKSTREAM_FINAL_DIR}"
echo "Protocol: ${FRAME_PROTOCOL}"
echo "Include failed verification: ${INCLUDE_FAILED_VERIFICATION}"
echo "Max sample tokens: ${MAX_SAMPLE_TOKENS}"
echo "Torch empty cache steps: ${TORCH_EMPTY_CACHE_STEPS}"
echo "Class loss target ratios: ${CLASS_LOSS_TARGET_RATIOS:-none}"
echo "Class loss alpha: ${CLASS_LOSS_ALPHA}"
echo "Group by modality: ${GROUP_BY_MODALITY}"
echo "LR:       ${lr}"
echo "Epochs:   ${epochs}"
echo "Output:   ${output_dir}"
echo "GPUs:     ${NPROC}"
echo "Batch:    ${BSZ} × ${GRAD_ACCUM} accum"
echo "Python:   ${PYTHON_BIN}"
echo "Torchrun: ${TORCHRUN_BIN}"
echo "=============================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: resolved configuration only; not launching torchrun."
    exit 0
fi

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${TORCHRUN_BIN}" --nproc_per_node=${NPROC} \
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
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --model_max_length 16384 \
    --max_sample_tokens ${MAX_SAMPLE_TOKENS} \
    --include_failed_verification ${INCLUDE_FAILED_VERIFICATION} \
    --dataloader_num_workers 4 \
    --video_min_pixels 130000 \
    --video_max_pixels 220000 \
    --video_fps 2.0 \
    --report_to wandb \
    --run_name "${run_name}" \
    ${extra_args}
