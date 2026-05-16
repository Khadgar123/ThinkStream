#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-pass3_rebalanced_$(date +%Y%m%d_%H%M%S)}"
ACTION="${ACTION:-${1:-}}"

cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  ACTION=batch1_4_3bc bash scripts/agent_data/launch_pass3_rebalanced_jobs.sh
  ACTION=batch1_4_full bash scripts/agent_data/launch_pass3_rebalanced_jobs.sh
  ACTION=batch6_10_refresh bash scripts/agent_data/launch_pass3_rebalanced_jobs.sh

Actions:
  batch1_4_3bc      Continue completed pass3A cards through pass3B/C only.
  batch1_4_full     Continue completed pass3A cards through pass3B/C, pass4, pass5.
  batch6_10_refresh Regenerate pass3A for all selected videos, then pass3B/C.
EOF
}

launch_job() {
  local session="$1"
  local api_base="$2"
  local batches="$3"
  local run_refresh="$4"
  local run_pass45="$5"
  local run_id="$6"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    return 1
  fi

  local cmd
  printf -v cmd '%q ' \
    env \
    API_BASE="${api_base}" \
    MODEL="${MODEL}" \
    BATCHES="${batches}" \
    RUN_REFRESH="${run_refresh}" \
    RUN_PASS45="${run_pass45}" \
    REFRESH_ALL_SELECTED=1 \
    REFRESH_FAMILIES= \
    RUN_ID="${run_id}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/agent_data/run_current_pass3_refresh_3bc.sh

  tmux new-session -d -s "${session}" "cd ${PROJECT_ROOT@Q} && ${cmd}"
  echo "launched ${session}: ${batches} via ${api_base}, RUN_REFRESH=${run_refresh}, RUN_PASS45=${run_pass45}"
}

launch_batch1_4_3bc() {
  local run_id="${RUN_ID_PREFIX}_b1_4_3bc"
  launch_job "pass3bc_b1_2_${RUN_ID_PREFIX}" "http://10.16.12.175:8000/v1" "batch1 batch2" 0 0 "${run_id}_12"
  launch_job "pass3bc_b3_${RUN_ID_PREFIX}" "http://10.16.10.172:8000/v1" "batch3" 0 0 "${run_id}_10"
  launch_job "pass3bc_b4_${RUN_ID_PREFIX}" "http://10.16.11.160:8000/v1" "batch4" 0 0 "${run_id}_11"
}

launch_batch1_4_full() {
  local run_id="${RUN_ID_PREFIX}_b1_4_full"
  launch_job "pass3bc45_b1_2_${RUN_ID_PREFIX}" "http://10.16.12.175:8000/v1" "batch1 batch2" 0 1 "${run_id}_12"
  launch_job "pass3bc45_b3_${RUN_ID_PREFIX}" "http://10.16.10.172:8000/v1" "batch3" 0 1 "${run_id}_10"
  launch_job "pass3bc45_b4_${RUN_ID_PREFIX}" "http://10.16.11.160:8000/v1" "batch4" 0 1 "${run_id}_11"
}

launch_batch6_10_refresh() {
  local run_id="${RUN_ID_PREFIX}_b6_10_refresh"
  launch_job "pass3a3bc_b6_b9_${RUN_ID_PREFIX}" "http://10.16.12.175:8000/v1" "batch6 batch9" 1 0 "${run_id}_12"
  launch_job "pass3a3bc_b7_b10_${RUN_ID_PREFIX}" "http://10.16.10.172:8000/v1" "batch7 batch10" 1 0 "${run_id}_10"
  launch_job "pass3a3bc_b8_${RUN_ID_PREFIX}" "http://10.16.11.160:8000/v1" "batch8" 1 0 "${run_id}_11"
}

case "${ACTION}" in
  batch1_4_3bc)
    launch_batch1_4_3bc
    ;;
  batch1_4_full)
    launch_batch1_4_full
    ;;
  batch6_10_refresh)
    launch_batch6_10_refresh
    ;;
  *)
    usage
    exit 2
    ;;
esac
