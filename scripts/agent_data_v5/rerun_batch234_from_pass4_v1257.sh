#!/usr/bin/env bash
# Rerun batch2/3/4 from the pass4 boundary and rebuild trajectory banks.
#
# This intentionally reuses existing pass1/pass2/pass3 caches. It verifies the
# pass3c cache count first, refreshes the pass3c marker to the current code
# version, clears stale trajectory_bank directories, then runs pipeline
# --force_rerun_from 4 so verification/final/pass5/RL parquet are rebuilt.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
API_BASE="${API_BASE:-http://10.16.18.9:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8}"
NUM_VIDEOS="${NUM_VIDEOS:-500}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_DIR="${LAUNCH_DIR:-${PROJECT_ROOT}/data/agent_v5/audits/rerun_batch234_from4_v1257_${RUN_ID}}"

mkdir -p "${LAUNCH_DIR}"
exec > >(tee -a "${LAUNCH_DIR}/supervisor.log") 2>&1

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

count_json_files() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        echo 0
        return
    fi
    find "${dir}" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' '
}

ensure_not_running() {
    local batch="$1"
    local videos_jsonl="$2"
    local running
    running="$(ps -ef | grep -F 'python -m scripts.agent_data_v5.pipeline run' | grep -F "${videos_jsonl}" | grep -v grep || true)"
    if [[ -n "${running}" ]]; then
        printf '%s\n' "${running}"
        die "${batch}: pipeline is already running"
    fi
}

stamp_pass3c_marker() {
    local root="$1"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -c 'from pathlib import Path; from scripts.agent_data_v5.cache_version import STAGE_VERSIONS; import sys; Path(sys.argv[1]).write_text(STAGE_VERSIONS["3c"])' "${root}/samples_3c/_version"
}

run_batch() {
    local batch="$1"
    local root="data/agent_v5/${batch}"
    local videos_jsonl="data/agent_v5/${batch}_videos.jsonl"
    local log_dir="${root}/logs"
    local pipeline_log="${log_dir}/pipeline_from4_v1257_${RUN_ID}.log"
    local bank_log="${log_dir}/trajectory_bank_v1257_${RUN_ID}.log"

    log "===== ${batch}: validate pass3c cache ====="
    [[ -s "${videos_jsonl}" ]] || die "${batch}: missing ${videos_jsonl}"
    [[ "$(wc -l < "${videos_jsonl}" | tr -d ' ')" == "${NUM_VIDEOS}" ]] || die "${batch}: videos_jsonl count is not ${NUM_VIDEOS}"
    [[ "$(count_json_files "${root}/samples_3c")" == "${NUM_VIDEOS}" ]] || die "${batch}: samples_3c cache count is not ${NUM_VIDEOS}"
    ensure_not_running "${batch}" "${videos_jsonl}"
    mkdir -p "${log_dir}"

    log "${batch}: align pass3c marker to current code version for pass4-only rerun"
    stamp_pass3c_marker "${root}"

    log "${batch}: clear old trajectory_bank"
    rm -rf "${root}/trajectory_bank"

    log "${batch}: rerun pipeline from pass4"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.pipeline run \
        --api_base "${API_BASE}" \
        --model "${MODEL}" \
        --videos_jsonl "${videos_jsonl}" \
        --num_videos "${NUM_VIDEOS}" \
        --skip_pass 1 2 3 \
        --force_rerun_from 4 \
        2>&1 | tee "${pipeline_log}"

    log "${batch}: export trajectory_bank"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.export_trajectory_bank \
        --data-dir "${root}" \
        2>&1 | tee "${bank_log}"

    local final_version bank_count video_count
    final_version="$(test -f "${root}/final/_version" && head -n 1 "${root}/final/_version" || echo MISSING)"
    bank_count="$(count_json_files "${root}/trajectory_bank/trajectories")"
    video_count="$(test -f "${root}/trajectory_bank/video_stats.jsonl" && wc -l < "${root}/trajectory_bank/video_stats.jsonl" | tr -d ' ' || echo 0)"
    log "${batch}: final_version=${final_version} bank_trajectories=${bank_count} bank_videos=${video_count}"

    [[ "${final_version}" == "v12.57" ]] || die "${batch}: final version is ${final_version}, expected v12.57"
    [[ "${bank_count}" == "${NUM_VIDEOS}" ]] || die "${batch}: bank trajectory count is ${bank_count}, expected ${NUM_VIDEOS}"
    [[ "${video_count}" == "${NUM_VIDEOS}" ]] || die "${batch}: bank video count is ${video_count}, expected ${NUM_VIDEOS}"
}

log "start rerun_batch234_from_pass4_v1257 RUN_ID=${RUN_ID}"
for batch in batch2 batch3 batch4; do
    run_batch "${batch}"
done
log "all done"
