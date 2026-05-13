#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."

export NO_PROXY="${NO_PROXY:-10.16.12.175,localhost,127.0.0.1}"
export no_proxy="${no_proxy:-10.16.12.175,localhost,127.0.0.1}"
export THINKSTREAM_DATA_ROOT="${THINKSTREAM_DATA_ROOT:-data/agent_v5/batch9}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-65536}"
export THINKSTREAM_COMPACT_MEMORY_UPDATE_MODE="${THINKSTREAM_COMPACT_MEMORY_UPDATE_MODE:-1}"
export THINKSTREAM_COMPACT_MEMORY_BALANCE_SEGMENTS="${THINKSTREAM_COMPACT_MEMORY_BALANCE_SEGMENTS:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1024}"
VIDEO_CONCURRENT="${VIDEO_CONCURRENT:-500}"
BATCH_ROOT="${BATCH_ROOT:-data/agent_v5/batch9}"
VIDEOS_JSONL="${VIDEOS_JSONL:-data/agent_v5/batch9_videos.jsonl}"

mkdir -p "${BATCH_ROOT}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${BATCH_ROOT}/logs/pass2_mem_8card_${STAMP}.log}"

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "batch9 pass2 compact-memory rerun"
echo "api_base=${API_BASE}"
echo "model=${MODEL}"
echo "batch_root=${BATCH_ROOT}"
echo "videos_jsonl=${VIDEOS_JSONL}"
echo "max_concurrent=${MAX_CONCURRENT}"
echo "video_concurrent=${VIDEO_CONCURRENT}"
echo "log=${LOG_PATH}"

"${PYTHON_BIN}" scripts/audit/run_pass2_only.py \
  --batch-root "${BATCH_ROOT}" \
  --videos-jsonl "${VIDEOS_JSONL}" \
  --api-base "${API_BASE}" \
  --model "${MODEL}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --video-concurrent "${VIDEO_CONCURRENT}"
