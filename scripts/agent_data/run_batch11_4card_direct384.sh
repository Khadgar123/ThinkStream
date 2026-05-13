#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/tione/notebook/gaozhenkun/hzh/ThinkStream"
BATCH_ROOT="${PROJECT_ROOT}/data/agent_v5/batch11"
VIDEOS_JSONL="${PROJECT_ROOT}/data/agent_v5/batch11_videos.jsonl"
API_BASE="http://10.16.18.9:8000/v1"
MODEL="/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="${BATCH_ROOT}/logs/rerun_4card_direct384_${STAMP}.log"

cd "${PROJECT_ROOT}"
mkdir -p "${BATCH_ROOT}/logs"

# Direct internal route only. The failed 20260512 run omitted 10.16.18.9 from
# no_proxy and sent the burst through the local proxy, which returned 503.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="10.16.18.9,127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

export PATH="/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin:${PATH}"
export THINKSTREAM_DATA_ROOT="${BATCH_ROOT}"
export AGENT_DATA_DIR="${BATCH_ROOT}"
export THINKSTREAM_BATCH="batch11"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="65536"

# 4-card direct probe on 2026-05-12:
# 192/256/384/512 concurrent 2-image requests all returned 200.
# 384 had better observed throughput than 512, so use it as the production cap.
export THINKSTREAM_VLLM_MAX_CONCURRENT="384"
export THINKSTREAM_PASS1A_CONCURRENT="384"
export THINKSTREAM_PASS1B_CONCURRENT="384"
export THINKSTREAM_PASS2_ROLLOUT_CONCURRENT="384"
export THINKSTREAM_PASS3A_CONCURRENT="384"
export THINKSTREAM_PASS3C_CONCURRENT="384"
export PYTHONUNBUFFERED="1"

{
  echo "[$(date '+%F %T')] [batch11] START full rerun from pass1a on 4-card 122B direct"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "api_base=${API_BASE}"
  echo "model=${MODEL}"
  echo "videos_jsonl=${VIDEOS_JSONL}"
  echo "num_videos=500"
  echo "data_root=${BATCH_ROOT}"
  echo "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
  echo "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
  echo "no_proxy=${no_proxy}"
  echo "force_rerun_from=1a"
} | tee "${LOG}"

set +e
python -m scripts.agent_data_v5.pipeline run \
  --api_base "${API_BASE}" \
  --model "${MODEL}" \
  --videos_jsonl "${VIDEOS_JSONL}" \
  --num_videos 500 \
  --force_rerun_from 1a \
  2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] [batch11] EXIT:${rc}" | tee -a "${LOG}"
exit "${rc}"
