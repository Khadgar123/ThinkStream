#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${PROJECT_ROOT}/data/agent_v5/audits"
JSONL="${LOG_DIR}/vllm_adaptive_keepalive_${STAMP}.jsonl"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="10.16.18.9,10.16.12.175,10.16.10.172,10.16.11.160,127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" scripts/audit/vllm_adaptive_keepalive.py \
  --metrics-interval 3 \
  --log-interval 15 \
  --jsonl "${JSONL}"
