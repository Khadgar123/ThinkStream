#!/usr/bin/env bash
# Rerun batch1 from the pass4 boundary and rebuild trajectory banks.
#
# This reuses existing pass1/pass2/pass3 caches. It validates the batch1
# pass3c cache first, aligns the pass3c version marker to the current code
# version, clears stale trajectory_bank output, then runs:
#   pipeline --skip_pass 1 2 3 --force_rerun_from 4
# followed by export_trajectory_bank.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
ROOT="${ROOT:-data/agent_v5/batch1}"
VIDEOS_JSONL="${VIDEOS_JSONL:-${ROOT}/selected_videos.jsonl}"
NUM_VIDEOS="${NUM_VIDEOS:-312}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/pipeline_pass45_bank_${RUN_ID}.log}"
BANK_LOG="${BANK_LOG:-${LOG_DIR}/trajectory_bank_pass45_${RUN_ID}.log}"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

line_count() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        wc -l < "${path}" | tr -d ' '
    else
        echo 0
    fi
}

count_json_files() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' '
    else
        echo 0
    fi
}

ensure_not_running() {
    local running
    running="$(ps -ef | grep -F 'python -m scripts.agent_data_v5.pipeline run' | grep -F "${VIDEOS_JSONL}" | grep -v grep || true)"
    if [[ -n "${running}" ]]; then
        printf '%s\n' "${running}"
        die "batch1 pipeline is already running for ${VIDEOS_JSONL}"
    fi
}

stamp_pass3c_marker() {
    THINKSTREAM_DATA_ROOT="${ROOT}" "${PYTHON}" -c \
        'from pathlib import Path; from scripts.agent_data_v5.cache_version import STAGE_VERSIONS; import sys; Path(sys.argv[1]).write_text(STAGE_VERSIONS["3c"])' \
        "${ROOT}/samples_3c/_version"
}

validate_preconditions() {
    [[ -d "${ROOT}" ]] || die "missing root: ${ROOT}"
    [[ -s "${VIDEOS_JSONL}" ]] || die "missing videos jsonl: ${VIDEOS_JSONL}"
    [[ "$(line_count "${VIDEOS_JSONL}")" == "${NUM_VIDEOS}" ]] || \
        die "videos_jsonl count $(line_count "${VIDEOS_JSONL}") != NUM_VIDEOS=${NUM_VIDEOS}"
    [[ "$(count_json_files "${ROOT}/samples_3c")" == "${NUM_VIDEOS}" ]] || \
        die "samples_3c cache count $(count_json_files "${ROOT}/samples_3c") != NUM_VIDEOS=${NUM_VIDEOS}"
    ensure_not_running
}

validate_outputs() {
    local final_version bank_count video_count split_total
    final_version="$(test -f "${ROOT}/final/_version" && head -n 1 "${ROOT}/final/_version" || echo MISSING)"
    bank_count="$(count_json_files "${ROOT}/trajectory_bank/trajectories")"
    video_count="$(line_count "${ROOT}/trajectory_bank/video_stats.jsonl")"
    split_total=$(( \
        $(line_count "${ROOT}/final/train_sft_trajectories.jsonl") + \
        $(line_count "${ROOT}/final/train_rl_trajectories.jsonl") + \
        $(line_count "${ROOT}/final/val_trajectories.jsonl") + \
        $(line_count "${ROOT}/final/test_trajectories.jsonl") \
    ))

    log "final_version=${final_version} split_total=${split_total} bank_trajectories=${bank_count} bank_videos=${video_count}"
    [[ "${split_total}" == "${NUM_VIDEOS}" ]] || die "split_total ${split_total} != NUM_VIDEOS=${NUM_VIDEOS}"
    [[ "${bank_count}" == "${NUM_VIDEOS}" ]] || die "bank trajectory count ${bank_count} != NUM_VIDEOS=${NUM_VIDEOS}"
    [[ "${video_count}" == "${NUM_VIDEOS}" ]] || die "bank video count ${video_count} != NUM_VIDEOS=${NUM_VIDEOS}"
}

main() {
    log "start batch1 pass4/pass5/bank rerun RUN_ID=${RUN_ID}"
    log "root=${ROOT}"
    log "videos_jsonl=${VIDEOS_JSONL}"
    log "num_videos=${NUM_VIDEOS}"
    log "pipeline_log=${PIPELINE_LOG}"
    log "bank_log=${BANK_LOG}"

    validate_preconditions

    log "align pass3c marker to current code version"
    stamp_pass3c_marker

    log "clear old trajectory_bank"
    rm -rf "${ROOT}/trajectory_bank"

    log "rerun pipeline from pass4"
    THINKSTREAM_DATA_ROOT="${ROOT}" AGENT_DATA_DIR="${ROOT}" \
        "${PYTHON}" -m scripts.agent_data_v5.pipeline run \
        --api_base "${API_BASE}" \
        --model "${MODEL}" \
        --videos_jsonl "${VIDEOS_JSONL}" \
        --num_videos "${NUM_VIDEOS}" \
        --skip_pass 1 2 3 \
        --force_rerun_from 4 \
        2>&1 | tee "${PIPELINE_LOG}"

    log "export trajectory_bank"
    THINKSTREAM_DATA_ROOT="${ROOT}" AGENT_DATA_DIR="${ROOT}" \
        "${PYTHON}" -m scripts.agent_data_v5.export_trajectory_bank \
        --data-dir "${ROOT}" \
        2>&1 | tee "${BANK_LOG}"

    validate_outputs
    log "done"
}

main "$@"
