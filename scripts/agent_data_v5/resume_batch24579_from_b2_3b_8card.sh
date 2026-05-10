#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-65536}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET="${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET:-64000000}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.12.175"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.12.175"

batch_root="${PROJECT_ROOT}/data/agent_v5/batch2"
batch_log="${batch_root}/logs/pass3b_resume_8card_${RUN_ID}.log"
mkdir -p "${batch_root}/logs"

echo "[$(date '+%F %T')] [batch2] RESUME from 3b on 8-card" | tee -a "${batch_log}"
echo "api_base=${API_BASE}" | tee -a "${batch_log}"
echo "model=${MODEL}" | tee -a "${batch_log}"
echo "run_id=${RUN_ID}" | tee -a "${batch_log}"
echo "skip_pass=1 2" | tee -a "${batch_log}"
echo "force_rerun_from=3b" | tee -a "${batch_log}"

set +e
THINKSTREAM_DATA_ROOT="${batch_root}" \
AGENT_DATA_DIR="${batch_root}" \
THINKSTREAM_BATCH="batch2" \
python -m scripts.agent_data_v5.pipeline run \
  --api_base "${API_BASE}" \
  --model "${MODEL}" \
  --videos_jsonl "${PROJECT_ROOT}/data/agent_v5/batch2_videos.jsonl" \
  --num_videos 500 \
  --skip_pass 1 2 \
  --force_rerun_from 3b \
  2>&1 | tee -a "${batch_log}"
rc=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] [batch2] EXIT:${rc}" | tee -a "${batch_log}"
if [[ "${rc}" -ne 0 ]]; then
  exit "${rc}"
fi

BATCHES="4 5 7 9" RUN_ID="${RUN_ID}" \
  bash scripts/agent_data_v5/rerun_batches24579_from_pass1b_8card.sh
