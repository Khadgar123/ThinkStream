#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.18.9:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}"

BATCH="batch1"
BATCH_ROOT="${PROJECT_ROOT}/data/agent_v5/${BATCH}"
VIDEOS_JSONL="${VIDEOS_JSONL:-${BATCH_ROOT}/selected_videos.jsonl}"
if [[ ! -f "${VIDEOS_JSONL}" ]]; then
  VIDEOS_JSONL="${BATCH_ROOT}/video_registry.jsonl"
fi
NUM_VIDEOS="$(wc -l < "${VIDEOS_JSONL}")"
LOG_DIR="${BATCH_ROOT}/logs"
LOG="${LOG_DIR}/pass45_rerender_4card_${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-40000}"
export THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO="${THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO:-0.95}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.18.9"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.18.9"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

log "START ${BATCH} pass45 rerender"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "videos_jsonl=${VIDEOS_JSONL}"
log "num_videos=${NUM_VIDEOS}"
log "data_root=${BATCH_ROOT}"
log "run_id=${RUN_ID}"

THINKSTREAM_DATA_ROOT="${BATCH_ROOT}" \
AGENT_DATA_DIR="${BATCH_ROOT}" \
THINKSTREAM_BATCH="${BATCH}" \
python -m scripts.agent_data_v5.pipeline run \
  --api_base "${API_BASE}" \
  --model "${MODEL}" \
  --videos_jsonl "${VIDEOS_JSONL}" \
  --num_videos "${NUM_VIDEOS}" \
  --skip_pass 1 2 \
  --force_rerun_from 4 \
  2>&1 | tee -a "${LOG}"
pipeline_rc=${PIPESTATUS[0]}
log "pipeline EXIT:${pipeline_rc}"
if [[ "${pipeline_rc}" -ne 0 ]]; then
  exit "${pipeline_rc}"
fi

log "export trajectory_bank"
THINKSTREAM_DATA_ROOT="${BATCH_ROOT}" \
AGENT_DATA_DIR="${BATCH_ROOT}" \
THINKSTREAM_BATCH="${BATCH}" \
python -m scripts.agent_data_v5.export_trajectory_bank \
  --data-dir "${BATCH_ROOT}" \
  2>&1 | tee -a "${LOG}"
bank_rc=${PIPESTATUS[0]}
log "trajectory_bank EXIT:${bank_rc}"
if [[ "${bank_rc}" -ne 0 ]]; then
  exit "${bank_rc}"
fi

log "final counts"
wc -l \
  "${BATCH_ROOT}/final/train.jsonl" \
  "${BATCH_ROOT}/final/train_sft.jsonl" \
  "${BATCH_ROOT}/final/train_rl.jsonl" \
  "${BATCH_ROOT}/final/val.jsonl" \
  "${BATCH_ROOT}/final/test.jsonl" \
  "${BATCH_ROOT}/final/train_sft_trajectories.jsonl" \
  "${BATCH_ROOT}/final/train_rl_trajectories.jsonl" \
  "${BATCH_ROOT}/final/val_trajectories.jsonl" \
  "${BATCH_ROOT}/final/test_trajectories.jsonl" \
  "${BATCH_ROOT}/rendered/video_meta_standard_query_last/train_sft_messages.jsonl" \
  "${BATCH_ROOT}/rendered/video_meta_standard_query_last/val_messages.jsonl" \
  "${BATCH_ROOT}/rendered/video_meta_standard_query_last/test_messages.jsonl" \
  "${BATCH_ROOT}/trajectory_bank/video_stats.jsonl" \
  "${BATCH_ROOT}/trajectory_bank/trajectory_stats.jsonl" \
  2>&1 | tee -a "${LOG}"

log "DONE ${BATCH} pass45 rerender"
