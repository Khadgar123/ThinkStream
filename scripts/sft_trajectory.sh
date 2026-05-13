#!/bin/bash
# Trajectory-mixed agent SFT training script.
#
# Production SFT now trains three trajectory abilities together in one
# standard Qwen messages corpus:
#   1. from_start streaming trajectories
#   2. from_compress streaming trajectories with memory prefill
#   3. compact_memory_update standalone <MEM> compression trajectories
#
# Usage (production):
#   bash scripts/sft_trajectory.sh
#
# Environment variables:
#   LLM         - Model path (default: Qwen/Qwen3-VL-8B)
#   NPROC       - GPUs per node (default: 8)
#   BSZ         - Per-device batch size (default: 4)
#   GRAD_ACCUM  - Gradient accumulation steps (default: 2)
#   EVAL_STEPS  - Eval frequency in optimizer steps (default 50)
#   EVAL_N      - Subsample size for in-loop eval (default 300).
#                 Set 0 to evaluate the full eval dataset.
#   EVAL_BSZ    - Per-device eval batch size (default = BSZ)
#   EVAL_BALANCE_STRATEGY
#               - none (default) | loss_class | loss_class_silent_diverse.
#                 When EVAL_N > 0, chooses a deterministic balanced eval
#                 subset for checkpoint selection. The silent-diverse mode
#                 stratifies silent rows using the pass5 pending/post/no-query
#                 and ask/answer boundary buckets.
#   EVAL_BALANCE_TARGET_RATIOS / EVAL_BALANCE_SEED
#               - Optional eval class ratios and deterministic seed.
#   MAX_STEPS   - Optional optimizer-step cap for very large batches
#   SAVE_LIMIT  - Max retained checkpoints (default 0 = no rolling
#                 deletion; 8B + zero-3 checkpoints can be ~30-50GB each)
#   THINKSTREAM_DATA_ROOT / AGENT_DATA_DIR
#               - Generated batch root. Default: data/agent_v5.
#   THINKSTREAM_FINAL_DIR
#               - Optional rendered/trajectory dir.
#   FRAME_PROTOCOL / THINKSTREAM_FRAME_PROTOCOL
#               - video_meta. SFT/RL/eval intentionally share one canonical
#                 video_meta protocol.
#   THINKSTREAM_RENDER_LAYOUT
#               - standard_query_last.
#   INCLUDE_FAILED_VERIFICATION
#               - True keeps verifier-failed samples with their tags, matching
#                 pass3e/pass4 tag-only trajectory continuity. Override to
#                 False only for strict filtering ablations.
#   MAX_SAMPLE_TOKENS
#               - Overlong filter threshold. Defaults to MODEL_MAX_LENGTH.
#                 Set 0 to disable token filtering entirely.
#   MODEL_MAX_LENGTH
#               - Tokenizer/model training sequence budget. Defaults to 16384
#                 for backward compatibility; mixed multi-turn trajectory SFT
#                 should raise this together with MAX_SAMPLE_TOKENS when the
#                 rendered rows intentionally contain long multi-round context.
#   CLASS_LOSS_TARGET_RATIOS
#               - Optional trajectory loss-class target ratios, e.g.
#                 from_start=0.425,from_compress=0.425,compress=0.15.
#                 Uses weighted loss instead of physically duplicating rows.
#   ACTION_CLASS_LOSS_MODE
#               - none (default) | inverse_freq | focal. Token-level action
#                 marker balancing; keep none unless an ablation tests it.
#   CLASS_LOSS_ALPHA / CLASS_LOSS_MAX_WEIGHT
#               - Reweighting strength and clamp.
#   COMPRESS_TOKEN_WEIGHTING
#               - False by default. Compress rows use normal assistant-token
#                 CE so the <MEM> body is trained with the same weight as the
#                 surrounding format tokens. Enable only for ablations.
#   COMPRESS_STRUCTURE_TOKEN_WEIGHT / COMPRESS_BODY_TOKEN_WEIGHT
#   COMPRESS_CLOSE_TOKEN_WEIGHT / COMPRESS_CLOSE_TAIL_TOKENS
#               - Ablation-only fine-grained compress token weights. Defaults:
#                 structure=2.0, body=0.35, close=4.0, close_tail=24.
#   GROUP_BY_MODALITY
#               - 1 by default. Keeps text-only compress rows and visual rows
#                 in separate global batches to avoid ZeRO3 ranks taking
#                 different vision-module paths.
#   THINKSTREAM_ATTN_IMPLEMENTATION
#               - streaming_attention by default so SFT uses the same 8-chunk
#                 visual sliding window as runtime/RL. Override to
#                 flash_attention_2 only for speed/ablation runs that accept
#                 full-history visual attention.
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
#   Default keeps exposure moderate for the v12 trajectory corpus:
#   2 epochs at effective_batch=64. BSZ=4 avoids the video_meta CE-logits OOM
#   seen with larger visual prompts on 8×96GB H20 while preserving the old global
#   batch through GRAD_ACCUM=2. Set MAX_STEPS to cap very large batches.

set -euo pipefail

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
FRAME_PROTOCOL="${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-video_meta}}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard_query_last}"
case "${THINKSTREAM_RENDER_LAYOUT}" in
    standard_query_last) ;;
    *)
        echo "ERROR: unsupported THINKSTREAM_RENDER_LAYOUT=${THINKSTREAM_RENDER_LAYOUT}" >&2
        exit 2
        ;;
esac
if [[ "${FRAME_PROTOCOL}" != "video_meta" ]]; then
    echo "ERROR: canonical SFT uses FRAME_PROTOCOL=video_meta" >&2
    echo "       got FRAME_PROTOCOL=${FRAME_PROTOCOL} THINKSTREAM_RENDER_LAYOUT=${THINKSTREAM_RENDER_LAYOUT}" >&2
    exit 2
fi
INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION:-True}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-16384}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-${MODEL_MAX_LENGTH}}"
TORCH_EMPTY_CACHE_STEPS="${TORCH_EMPTY_CACHE_STEPS:-0}"
CLASS_LOSS_ALPHA="${CLASS_LOSS_ALPHA:-0.5}"
CLASS_LOSS_MAX_WEIGHT="${CLASS_LOSS_MAX_WEIGHT:-4.0}"
ACTION_CLASS_LOSS_MODE="${ACTION_CLASS_LOSS_MODE:-none}"
COMPRESS_TOKEN_WEIGHTING="${COMPRESS_TOKEN_WEIGHTING:-False}"
COMPRESS_STRUCTURE_TOKEN_WEIGHT="${COMPRESS_STRUCTURE_TOKEN_WEIGHT:-2.0}"
COMPRESS_BODY_TOKEN_WEIGHT="${COMPRESS_BODY_TOKEN_WEIGHT:-0.35}"
COMPRESS_CLOSE_TOKEN_WEIGHT="${COMPRESS_CLOSE_TOKEN_WEIGHT:-4.0}"
COMPRESS_CLOSE_TAIL_TOKENS="${COMPRESS_CLOSE_TAIL_TOKENS:-24}"
GROUP_BY_MODALITY="${GROUP_BY_MODALITY:-1}"
export THINKSTREAM_GROUP_BY_MODALITY="${GROUP_BY_MODALITY}"
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
export THINKSTREAM_RENDER_LAYOUT
export THINKSTREAM_MEMORY_POSITION="${THINKSTREAM_MEMORY_POSITION:-before_visual}"
export THINKSTREAM_ATTN_IMPLEMENTATION="${THINKSTREAM_ATTN_IMPLEMENTATION:-streaming_attention}"
export THINKSTREAM_VISION_ATTN_IMPLEMENTATION="${THINKSTREAM_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-${MIN_PIXELS:-}}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-${MAX_PIXELS:-}}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-200704}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
image_pixel_args=""
if [[ -n "${IMAGE_MIN_PIXELS}" ]]; then
    image_pixel_args="${image_pixel_args} --min_pixels ${IMAGE_MIN_PIXELS}"
fi
if [[ -n "${IMAGE_MAX_PIXELS}" ]]; then
    image_pixel_args="${image_pixel_args} --max_pixels ${IMAGE_MAX_PIXELS}"
fi
DEFAULT_SFT_FINAL_DIR="${AGENT_DATA_ROOT}/rendered/trajectory"
DEFAULT_TRAIN_DATASET="stream_agent_trajectory_train"
DEFAULT_EVAL_DATASET="stream_agent_trajectory_val"
DEFAULT_CLASS_LOSS_TARGET_RATIOS=""
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS:-${DEFAULT_CLASS_LOSS_TARGET_RATIOS}}"
if [[ -z "${THINKSTREAM_FINAL_DIR:-}" ]]; then
    export THINKSTREAM_FINAL_DIR="${DEFAULT_SFT_FINAL_DIR}"
fi
if [[ ! -d "${THINKSTREAM_FINAL_DIR}" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "ERROR: rendered SFT dir not found: ${THINKSTREAM_FINAL_DIR}" >&2
    echo "       build it with scripts/agent_data/make_training_scheme.py or pipeline pass45." >&2
    exit 2
fi

llm=${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
datasets=${DATASETS:-${DEFAULT_TRAIN_DATASET}}
eval_datasets=${EVAL_DATASETS:-${DEFAULT_EVAL_DATASET}}
eval_n=${EVAL_N:-300}
eval_balance_strategy=${EVAL_BALANCE_STRATEGY:-none}
eval_balance_target_ratios=${EVAL_BALANCE_TARGET_RATIOS:-}
eval_balance_seed=${EVAL_BALANCE_SEED:-0}
lr=${LR:-2e-5}; epochs=${EPOCHS:-2}
run_name="${RUN_NAME:-agent-trajectory-sft-v12.26-${FRAME_PROTOCOL}}"

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
    extra_args="${extra_args} --eval_balance_strategy ${eval_balance_strategy}"
    extra_args="${extra_args} --eval_balance_seed ${eval_balance_seed}"
    if [[ -n "${eval_balance_target_ratios}" ]]; then
        extra_args="${extra_args} --eval_balance_target_ratios ${eval_balance_target_ratios}"
    fi
fi
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
extra_args="${extra_args} --action_class_loss_mode ${ACTION_CLASS_LOSS_MODE}"
extra_args="${extra_args} --compress_token_weighting ${COMPRESS_TOKEN_WEIGHTING}"
extra_args="${extra_args} --compress_structure_token_weight ${COMPRESS_STRUCTURE_TOKEN_WEIGHT}"
extra_args="${extra_args} --compress_body_token_weight ${COMPRESS_BODY_TOKEN_WEIGHT}"
extra_args="${extra_args} --compress_close_token_weight ${COMPRESS_CLOSE_TOKEN_WEIGHT}"
extra_args="${extra_args} --compress_close_tail_tokens ${COMPRESS_CLOSE_TAIL_TOKENS}"
if [[ "${TORCH_EMPTY_CACHE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    extra_args="${extra_args} --torch_empty_cache_steps ${TORCH_EMPTY_CACHE_STEPS}"
fi

output_dir="${OUTPUT_DIR:-${PROJECT_DIR}/output/${run_name}}"
echo "=== Trajectory-Mixed Agent SFT ==="
echo "Model:    ${llm}"
echo "Dataset:  ${datasets}"
echo "Eval:     ${eval_datasets:-none}"
echo "Data:     ${AGENT_DATA_ROOT}"
echo "Final:    ${THINKSTREAM_FINAL_DIR}"
echo "Protocol: ${FRAME_PROTOCOL}"
echo "Include failed verification: ${INCLUDE_FAILED_VERIFICATION}"
echo "Max sample tokens: ${MAX_SAMPLE_TOKENS}"
echo "Model max length: ${MODEL_MAX_LENGTH}"
echo "Torch empty cache steps: ${TORCH_EMPTY_CACHE_STEPS}"
echo "Class loss target ratios: ${CLASS_LOSS_TARGET_RATIOS:-none}"
echo "Class loss alpha: ${CLASS_LOSS_ALPHA}"
echo "Action class loss mode: ${ACTION_CLASS_LOSS_MODE}"
echo "Eval balance strategy: ${EVAL_BALANCE_STRATEGY:-none}"
echo "Eval balance target ratios: ${EVAL_BALANCE_TARGET_RATIOS:-equal-present}"
echo "Compress token weighting: ${COMPRESS_TOKEN_WEIGHTING} (structure=${COMPRESS_STRUCTURE_TOKEN_WEIGHT} body=${COMPRESS_BODY_TOKEN_WEIGHT} close=${COMPRESS_CLOSE_TOKEN_WEIGHT} tail=${COMPRESS_CLOSE_TAIL_TOKENS}; ignored unless enabled)"
echo "Group by modality: ${GROUP_BY_MODALITY}"
echo "Attention: ${THINKSTREAM_ATTN_IMPLEMENTATION}"
echo "Vision attention: ${THINKSTREAM_VISION_ATTN_IMPLEMENTATION}"
echo "Image pixels: ${IMAGE_MIN_PIXELS:-default} .. ${IMAGE_MAX_PIXELS:-default}"
echo "Video pixels: ${VIDEO_MIN_PIXELS} .. ${VIDEO_MAX_PIXELS}"
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
    --model_max_length ${MODEL_MAX_LENGTH} \
    --max_sample_tokens ${MAX_SAMPLE_TOKENS} \
    --include_failed_verification ${INCLUDE_FAILED_VERIFICATION} \
    --dataloader_num_workers 4 \
    ${image_pixel_args} \
    --video_min_pixels ${VIDEO_MIN_PIXELS} \
    --video_max_pixels ${VIDEO_MAX_PIXELS} \
    --video_fps 2.0 \
    --report_to wandb \
    --run_name "${run_name}" \
    ${extra_args}
