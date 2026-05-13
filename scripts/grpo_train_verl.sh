#!/bin/bash
# verl-based GRPO training launcher for ThinkStream.
#
# This is the only supported RL backend. The old slyme/HF-rollout GRPO path
# has been retired to avoid train/eval drift and duplicate reward logic.
#
# verl is vendored at ThinkStream/verl/ (a customized fork of
# verl-project/verl with our recipe at thinkstream/rl/). We do NOT
# `pip install verl`; instead we run it in-place via PYTHONPATH so that
# (a) recipe edits take effect without reinstall and (b) verl can import
# `thinkstream.*` from the parent ThinkStream checkout.
#
# verl assumes the SFT checkpoint is on disk (Hugging Face format). Run
# scripts/sft_trajectory.sh first; this script picks up the resulting
# checkpoint via $LLM.
#
# Usage:
#   LLM=output/agent-sft bash scripts/grpo_train_verl.sh
#
# Required env:
#   LLM             — SFT checkpoint path (HF format).
#
# Optional env (defaults shown):
#   NPROC           — GPUs per node (8)
#   NNODES          — number of Ray nodes for training (1)
#   RAY_ADDRESS     — existing Ray cluster address, e.g. auto or
#                     ray://HEAD:10001 / HEAD:6379 depending on Ray launch.
#   GROUP_SIZE      — GRPO group size G (8) — enough variance for GRPO
#   MAXLEN          — max prompt length (16384) — true-KV context cap
#   MAX_NEW_TOKEN   — response buffer. Defaults to 4096 in recurrent mode
#                     and 32768 in stitched mode.
#   MAX_ACTION_TOKENS — per-action streaming/recall generation cap (256)
#   MAX_COMPRESS_ACTION_TOKENS — per-action compression generation cap (512)
#   MAX_CHUNKS      — max turns per video (120 by default; use recurrent for 240+)
#   GPU_MEM_UTIL    — kept for verl config compatibility (0.55).
#   MM_CACHE_GB     — kept for legacy config compatibility.
#   FRAME_PROTOCOL  — video_meta. Must match SFT/eval.
#   THINKSTREAM_RENDER_LAYOUT — standard_query_last.
#   IMAGE_MIN_PIXELS / IMAGE_MAX_PIXELS — optional runtime image resize bounds.
#   LIMIT_IMAGES    — legacy multimodal prompt cap for timestamped frames (64)
#   LIMIT_VIDEOS    — legacy multimodal prompt cap for video_meta blocks (2)
#   TP_SIZE         — streaming rollout tensor parallel size. Must be 1.
#   BATCH_SIZE      — videos per training step (4)
#   PPO_MINI_BS     — ppo_mini_batch_size in prompt units (default=BATCH_SIZE)
#   LR              — learning rate (5e-7)
#   EPOCHS          — total_epochs (1)
#   SAVE_FREQ       — save every N steps (50)
#   TEST_FREQ       — eval on val every N steps (25)
#   VAL_ONLY        — run val_before_train through the RL AgentLoop and exit.
#   VALIDATION_DATA_DIR — optional JSONL dump dir for validation generations.
#   RUN_NAME        — wandb experiment name (grpo-v12.26-verl-$FRAME_PROTOCOL)
#   WANDB_PROJECT   — wandb project (thinkstream-v12)
#   PARAM_OFFLOAD   — FSDP offload params to CPU (true).
#   OPTIMIZER_OFFLOAD — FSDP offload optimizer state (true).
#   FREEZE_VISION_TOWER — freeze Qwen VL vision tower during RL update (true).
#   PPO_MAX_TOKEN_LEN_PER_GPU / LOG_PROB_MAX_TOKEN_LEN_PER_GPU
#                  — dynamic micro-batch token budget (65536).
#   RUNTIME_ROOT    — Ray/HF/Triton cache root. Defaults to
#                    .runtime/$RUN_NAME to keep /tmp from filling.
#   THINKSTREAM_RECURRENT_MODE — recurrent | stitched. Recurrent is the
#                    production default: rollout emits one action row at a
#                    time, so B>1 multi-trajectory batches do not allocate
#                    one giant stitched response tensor per trajectory.
#   THINKSTREAM_RL_EPISODE_MODE — full | segment. full preserves one
#                    full-video trajectory per sample; segment uses student
#                    prefix state when available for faster training windows.
#   THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE — offline_pass2_boundaries by
#                    default. Uses annotated compact-memory trigger chunks
#                    from parquet instead of runtime token counting.
#   THINKSTREAM_RL_REWARD_PROFILE — initial_outcome_time_format_decision by
#                    default: answer correctness + answer_decision + format.
#                    Raw timing/silent_quality plus step/action/tool rewards
#                    stay monitor-only unless explicitly enabled for an
#                    ablation.
#   THINKSTREAM_ROLLOUT_ENGINE — streaming. Full-prompt/vLLM rollout is not
#                    supported because it breaks true-KV recall deletion.
#   ROLLOUT_BACKEND — must be streaming.
#   TRAIN_PARQUET / VAL_PARQUET — multi-Q trajectory verl parquets. If
#                  unset, we auto-build from data/agent_v5/final/*.jsonl via
#                  scripts/agent_data/build_verl_parquet.py.
#   THINKSTREAM_DATA_ROOT / AGENT_DATA_DIR — generated batch root
#                  (default: data/agent_v5). final/ and frames/ are
#                  resolved underneath this root.
#   MULTI_Q        — must be 1: one video row with all questions.

set -euo pipefail

LLM=${LLM:?'LLM= required (path to SFT checkpoint, e.g. output/agent-sft)'}

NPROC=${NPROC:-8}
# Production defaults. For smoke tests override MAX_CHUNKS=8 BATCH_SIZE=1
# GROUP_SIZE=2 MAX_STEPS=2.
GROUP_SIZE=${GROUP_SIZE:-8}
MAXLEN=${MAXLEN:-16384}
THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"
case "${THINKSTREAM_RECURRENT_MODE}" in
    recurrent|stitched) ;;
    *)
        echo "ERROR: THINKSTREAM_RECURRENT_MODE must be recurrent or stitched, got ${THINKSTREAM_RECURRENT_MODE}" >&2
        exit 2
        ;;
esac
# MAX_NEW_TOKEN sets verl's rollout.response_length. In stitched mode this is
# the total trajectory response buffer; in recurrent mode it is per action-row.
# Keep recurrent small so B>1 multi-trajectory rollout/update does not allocate
# dense 32K response tensors for every action.
if [[ -z "${MAX_NEW_TOKEN:-}" ]]; then
    if [[ "${THINKSTREAM_RECURRENT_MODE}" == "recurrent" ]]; then
        MAX_NEW_TOKEN=4096
    else
        MAX_NEW_TOKEN=32768
    fi
fi
MAX_ACTION_TOKENS=${MAX_ACTION_TOKENS:-256}
MAX_COMPRESS_ACTION_TOKENS=${MAX_COMPRESS_ACTION_TOKENS:-512}
MAX_CHUNKS=${MAX_CHUNKS:-120}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
auto_mm_cache_gb() {
    local avail_kb avail_gb
    avail_kb="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)"
    avail_gb=$((avail_kb / 1024 / 1024))
    if (( avail_gb >= 1536 )); then
        echo 512
    elif (( avail_gb >= 768 )); then
        echo 256
    elif (( avail_gb >= 384 )); then
        echo 128
    elif (( avail_gb >= 128 )); then
        echo 64
    else
        echo 16
    fi
}
MM_CACHE_GB="${MM_CACHE_GB:-${THINKSTREAM_MM_CACHE_GB:-${VLLM_MM_PROCESSOR_CACHE_GB:-}}}"
if [[ -z "${MM_CACHE_GB}" ]]; then
    MM_CACHE_GB="$(auto_mm_cache_gb)"
fi
LIMIT_IMAGES=${LIMIT_IMAGES:-64}
LIMIT_VIDEOS=${LIMIT_VIDEOS:-2}
TP_SIZE=${TP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
PPO_MINI_BS=${PPO_MINI_BS:-${BATCH_SIZE}}
LR=${LR:-5e-7}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
VAL_ONLY=${VAL_ONLY:-false}
if [[ "${VAL_ONLY}" == "1" ]]; then
    VAL_ONLY=true
fi
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-${VAL_ONLY}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
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
    echo "ERROR: canonical RL uses FRAME_PROTOCOL=video_meta" >&2
    echo "       got FRAME_PROTOCOL=${FRAME_PROTOCOL} THINKSTREAM_RENDER_LAYOUT=${THINKSTREAM_RENDER_LAYOUT}" >&2
    exit 2
fi
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-${MIN_PIXELS:-}}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-${MAX_PIXELS:-}}"
RUN_NAME=${RUN_NAME:-grpo-v12.26-verl-${FRAME_PROTOCOL}}
WANDB_PROJECT=${WANDB_PROJECT:-thinkstream-v12}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-true}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-streaming}
THINKSTREAM_ROLLOUT_ENGINE="${THINKSTREAM_ROLLOUT_ENGINE:-streaming}"
if [[ "${ROLLOUT_BACKEND}" != "streaming" ]]; then
    echo "ERROR: ThinkStream RL now requires ROLLOUT_BACKEND=streaming for true-KV rollout." >&2
    echo "       Full-prompt/vLLM rollout is disabled because recall KV deletion would be incorrect." >&2
    exit 2
fi
if [[ "${THINKSTREAM_ROLLOUT_ENGINE}" != "streaming" ]]; then
    echo "ERROR: THINKSTREAM_ROLLOUT_ENGINE must be streaming." >&2
    exit 2
fi
if [[ "${TP_SIZE}" != "1" ]]; then
    echo "ERROR: streaming rollout runs one local HF model per GPU; set TP_SIZE=1." >&2
    exit 2
fi
THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE="${THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE:-offline_pass2_boundaries}"
THINKSTREAM_RL_REWARD_PROFILE="${THINKSTREAM_RL_REWARD_PROFILE:-initial_outcome_time_format_decision}"
THINKSTREAM_ENABLE_STEP_ACTION_REWARD="${THINKSTREAM_ENABLE_STEP_ACTION_REWARD:-0}"
THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD="${THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD:-0}"
THINKSTREAM_RL_EPISODE_MODE="${THINKSTREAM_RL_EPISODE_MODE:-full}"
case "${THINKSTREAM_RL_EPISODE_MODE}" in
    full|full_video|trajectory)
        THINKSTREAM_RL_EPISODE_MODE="full"
        ;;
    segment|single_question|single_q|question|per_question)
        THINKSTREAM_RL_EPISODE_MODE="segment"
        ;;
    *)
        echo "ERROR: THINKSTREAM_RL_EPISODE_MODE must be full or segment, got ${THINKSTREAM_RL_EPISODE_MODE}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${PROJECT_DIR}/verl"
RECIPE_DIR="${VERL_DIR}/thinkstream/rl/configs"
RECIPE_NAME="thinkstream_grpo"
PARENT_DIR="$(dirname "${PROJECT_DIR}")"
if [[ -z "${THINKSTREAM_ENV:-}" ]]; then
    if [[ -x "${PARENT_DIR}/envs/thinkstream/bin/python" ]]; then
        THINKSTREAM_ENV="${PARENT_DIR}/envs/thinkstream"
    elif [[ -x "${PROJECT_DIR}/envs/thinkstream/bin/python" ]]; then
        THINKSTREAM_ENV="${PROJECT_DIR}/envs/thinkstream"
    else
        THINKSTREAM_ENV=""
    fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${THINKSTREAM_ENV}" && -x "${THINKSTREAM_ENV}/bin/python" ]]; then
        PYTHON_BIN="${THINKSTREAM_ENV}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
AGENT_DATA_ROOT="${THINKSTREAM_DATA_ROOT:-${AGENT_DATA_DIR:-${PROJECT_DIR}/data/agent_v5}}"
if [[ "${AGENT_DATA_ROOT}" == */final ]]; then
    AGENT_DATA_ROOT="$(dirname "${AGENT_DATA_ROOT}")"
fi

OUTPUT_DIR="${THINKSTREAM_OUTPUT_DIR:-${PROJECT_DIR}/output/${RUN_NAME}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_DIR}/.runtime/${RUN_NAME}}"
TRAIN_JSONL="${TRAIN_JSONL:-${AGENT_DATA_ROOT}/final/train_rl_trajectories.jsonl}"
VAL_JSONL="${VAL_JSONL:-${AGENT_DATA_ROOT}/final/val_trajectories.jsonl}"
MULTI_Q="${MULTI_Q:-1}"
PARQUET_DIR="${PARQUET_DIR:-${AGENT_DATA_ROOT}/rendered/${FRAME_PROTOCOL}_${THINKSTREAM_RENDER_LAYOUT}}"
if [[ "${MULTI_Q}" != "1" ]]; then
    echo "ERROR: RL training now requires multi-Q trajectory parquet (MULTI_Q=1)." >&2
    echo "       Legacy single-question parquet generation was retired from this launcher." >&2
    exit 2
fi

# verl's RLHFDataset reads parquet; auto-build from JSONL if user didn't
# supply a parquet directly.
if [[ "${THINKSTREAM_RL_EPISODE_MODE}" == "segment" ]]; then
    DEFAULT_TRAIN_PARQUET="${PARQUET_DIR}/train_rl_multi_q_segment_cache.parquet"
    DEFAULT_VAL_PARQUET="${PARQUET_DIR}/val_rl_multi_q_segment_cache.parquet"
else
    DEFAULT_TRAIN_PARQUET="${PARQUET_DIR}/train_rl_multi_q.parquet"
    DEFAULT_VAL_PARQUET="${PARQUET_DIR}/val_rl_multi_q.parquet"
fi
TRAIN_PARQUET="${TRAIN_PARQUET:-${DEFAULT_TRAIN_PARQUET}}"
VAL_PARQUET="${VAL_PARQUET:-${DEFAULT_VAL_PARQUET}}"

# MULTI_Q=1 → 1 video = 1 row, all questions co-evaluated (OVOBench-aligned).
MULTI_Q_FLAG="--multi_q"
STUDENT_CACHE_FLAG=()
if [[ "${THINKSTREAM_RL_EPISODE_MODE}" == "segment" ]]; then
    STUDENT_CACHE_FLAG=(--include-student-cache)
fi

if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "Building train parquet from ${TRAIN_JSONL}…  (multi_q=${MULTI_Q})"
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/agent_data/build_verl_parquet.py" \
        --jsonl "${TRAIN_JSONL}" --out "${TRAIN_PARQUET}" \
        --frame-protocol "${FRAME_PROTOCOL}" \
        --render-layout "${THINKSTREAM_RENDER_LAYOUT}" ${MULTI_Q_FLAG} "${STUDENT_CACHE_FLAG[@]}"
fi
if [[ ! -f "${VAL_PARQUET}" ]]; then
    echo "Building val parquet from ${VAL_JSONL}…  (multi_q=${MULTI_Q})"
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/agent_data/build_verl_parquet.py" \
        --jsonl "${VAL_JSONL}" --out "${VAL_PARQUET}" \
        --frame-protocol "${FRAME_PROTOCOL}" \
        --render-layout "${THINKSTREAM_RENDER_LAYOUT}" ${MULTI_Q_FLAG} "${STUDENT_CACHE_FLAG[@]}"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RUNTIME_ROOT}"/{tmp,ray,hf,torch,triton,xdg}

# Keep Ray spill files and Triton/HF caches on
# the project filesystem. Long recurrent rollouts can otherwise fill /tmp.
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export RUNTIME_ROOT
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_ROOT}/ray}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${RUNTIME_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg}"

echo "=== ThinkStream GRPO via verl ==="
echo "Checkpoint:        ${LLM}"
echo "Vendored verl:     ${VERL_DIR}"
echo "Recipe dir:        ${RECIPE_DIR}"
echo "Recipe name:       ${RECIPE_NAME}"
echo "Python:            ${PYTHON_BIN}"
echo "Data root:         ${AGENT_DATA_ROOT}"
echo "Frame protocol:    ${FRAME_PROTOCOL}"
echo "Render layout:     ${THINKSTREAM_RENDER_LAYOUT}"
echo "Image pixels:      ${IMAGE_MIN_PIXELS:-default} .. ${IMAGE_MAX_PIXELS:-default}"
echo "Train parquet:     ${TRAIN_PARQUET}"
echo "Val parquet:       ${VAL_PARQUET}"
echo "Multi-Q rows:      ${MULTI_Q}"
echo "RL episode mode:   ${THINKSTREAM_RL_EPISODE_MODE}"
echo "Recurrent mode:    ${THINKSTREAM_RECURRENT_MODE}"
echo "Compress trigger:  ${THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE}"
echo "Reward profile:    ${THINKSTREAM_RL_REWARD_PROFILE}"
echo "Output:            ${OUTPUT_DIR}"
echo "Runtime root:      ${RUNTIME_ROOT}"
echo "GPUs:              ${NPROC}"
echo "Nodes:             ${NNODES:-1}"
[ -n "${RAY_ADDRESS:-}" ] && echo "Ray address:       ${RAY_ADDRESS}"
echo "Rollout backend:   ${ROLLOUT_BACKEND}"
echo "Rollout engine:    ${THINKSTREAM_ROLLOUT_ENGINE}"
echo "TP size:           ${TP_SIZE}"
echo "Group size G:      ${GROUP_SIZE}"
echo "Max chunks:        ${MAX_CHUNKS}"
echo "Max prompt len:    ${MAXLEN}"
echo "Max new tokens:    ${MAX_NEW_TOKEN}"
echo "Max action tokens: ${MAX_ACTION_TOKENS}"
echo "Max compress toks: ${MAX_COMPRESS_ACTION_TOKENS}"
echo "GPU mem util:      ${GPU_MEM_UTIL}"
echo "MM cache GB:       ${MM_CACHE_GB}"
echo "Image limit:       ${LIMIT_IMAGES}"
echo "Video limit:       ${LIMIT_VIDEOS}"
echo "LR:                ${LR}"
echo "Epochs:            ${EPOCHS}"
echo "Max steps:         ${MAX_STEPS:-<epoch-based>}"
echo "Val only:          ${VAL_ONLY}"
[ -n "${VALIDATION_DATA_DIR}" ] && echo "Val dump dir:      ${VALIDATION_DATA_DIR}"
echo "PPO mini bs:       ${PPO_MINI_BS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "PPO max tok/GPU:   ${PPO_MAX_TOKEN_LEN_PER_GPU}"
echo "Logprob max tok/GPU: ${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
echo "Freeze vision:     ${FREEZE_VISION_TOWER}"
echo "FSDP param offload: ${PARAM_OFFLOAD}"
echo "FSDP opt offload:   ${OPTIMIZER_OFFLOAD}"
echo "================================="

# verl uses Ray; let it handle multi-GPU orchestration.
# We pass per-flag overrides on top of thinkstream/rl/configs/thinkstream_grpo.yaml.
#
# PYTHONPATH ordering matters:
#   ${VERL_DIR}     — vendored verl python package (in-place, no pip install)
#   ${PROJECT_DIR}  — ThinkStream package, so reward fn can import
#                     thinkstream.trainer.rewards
export PYTHONPATH="${VERL_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHON_BIN
export THINKSTREAM_ENV
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export MM_CACHE_GB
export THINKSTREAM_MM_CACHE_GB="${MM_CACHE_GB}"
# Lets the recipe's compute_score read trajectory metadata (gold_action_per_chunk,
# ask_chunks) when verl's parquet column flattening drops nested dicts.
export THINKSTREAM_TRAJ_INDEX_PATH="${TRAIN_JSONL}"
# Where pre-extracted JPEG frames live (one subdir per video stem). The
# streaming agent loop reads this to inject per-chunk visual frames every
# turn. Empty / unset → loop falls back to text-only RL.
FRAMES_ROOT="${FRAMES_ROOT:-${AGENT_DATA_ROOT}/frames}"
export THINKSTREAM_FRAMES_ROOT="${FRAMES_ROOT}"
export THINKSTREAM_MAX_TOKENS_PER_ACTION="${MAX_ACTION_TOKENS}"
export THINKSTREAM_COMPRESS_MAX_TOKENS_PER_ACTION="${MAX_COMPRESS_ACTION_TOKENS}"
export THINKSTREAM_RL_EPISODE_MODE
export THINKSTREAM_RECURRENT_MODE
export THINKSTREAM_ROLLOUT_ENGINE
export THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE
export THINKSTREAM_RL_REWARD_PROFILE
export THINKSTREAM_ENABLE_STEP_ACTION_REWARD
export THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD

export THINKSTREAM_HOME="${PROJECT_DIR}"
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
export THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}"
export THINKSTREAM_MEMORY_POSITION="${THINKSTREAM_MEMORY_POSITION:-before_visual}"
if [[ -n "${IMAGE_MIN_PIXELS}" ]]; then
    export IMAGE_MIN_PIXELS
fi
if [[ -n "${IMAGE_MAX_PIXELS}" ]]; then
    export IMAGE_MAX_PIXELS
fi
export HF_MODEL_PATH="${LLM}"
export TRAIN_PARQUET="${TRAIN_PARQUET}"
export VAL_PARQUET="${VAL_PARQUET}"
export N_GPUS_PER_NODE="${NPROC}"
export NNODES="${NNODES:-1}"
export RAY_ADDRESS="${RAY_ADDRESS:-}"
export GEN_TP="${TP_SIZE}"
export GROUP_SIZE="${GROUP_SIZE}"
export BATCH_SIZE="${BATCH_SIZE}"
export PPO_MINI_BS="${PPO_MINI_BS}"
export LR="${LR}"
export EPOCHS="${EPOCHS}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL}"
export LIMIT_IMAGES="${LIMIT_IMAGES}"
export LIMIT_VIDEOS="${LIMIT_VIDEOS}"
export MAX_PROMPT_LEN="${MAXLEN}"
export MAX_RESP_LEN="${MAX_NEW_TOKEN}"
export MAX_ACTION_TOKENS="${MAX_ACTION_TOKENS}"
export MAX_COMPRESS_ACTION_TOKENS="${MAX_COMPRESS_ACTION_TOKENS}"
export MAX_TURNS="${MAX_CHUNKS}"
export PROJECT_NAME="${WANDB_PROJECT}"
export EXPERIMENT_NAME="${RUN_NAME}"
export SAVE_DIR="${OUTPUT_DIR}"
export SAVE_FREQ="${SAVE_FREQ}"
export TEST_FREQ="${TEST_FREQ}"
export VAL_ONLY="${VAL_ONLY}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD}"
export FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}"
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND}"
if [[ -n "${MAX_STEPS}" ]]; then
    export MAX_STEPS
fi

cd "${VERL_DIR}"
bash thinkstream/rl/run_thinkstream_grpo.sh
