#!/usr/bin/env bash
# Re-render pass5 artifacts and rebuild trajectory banks for selected batches.
# Defaults to batch2/3/4; override with e.g.
#   BATCHES="batch1 batch2 batch3 batch4 batch5" bash scripts/agent_data_v5/rerender_batch234_pass5_bank.sh
#
# This is intentionally pass5-only: it preserves pass4 trajectory JSONL files
# and only regenerates *_messages.jsonl, rendered protocol variants, RL
# parquets, the pass5 version marker, and trajectory_bank.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-batch2 batch3 batch4}"
LAUNCH_DIR="${LAUNCH_DIR:-${PROJECT_ROOT}/data/agent_v5/audits/rerender_pass5_bank_${RUN_ID}}"

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

line_count() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        wc -l < "${path}" | tr -d ' '
    else
        echo 0
    fi
}

current_pass5_version() {
    "${PYTHON}" -c 'from scripts.agent_data_v5.cache_version import STAGE_VERSIONS; print(STAGE_VERSIONS["5"])'
}

stamp_pass5_marker() {
    local root="$1"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -c 'from scripts.agent_data_v5.cache_version import write_stage_version; write_stage_version("5")'
}

validate_trajectories() {
    local batch="$1"
    local root="$2"
    local train_sft train_rl val test
    train_sft="$(line_count "${root}/final/train_sft_trajectories.jsonl")"
    train_rl="$(line_count "${root}/final/train_rl_trajectories.jsonl")"
    val="$(line_count "${root}/final/val_trajectories.jsonl")"
    test="$(line_count "${root}/final/test_trajectories.jsonl")"
    local total selected
    total=$((train_sft + train_rl + val + test))
    selected="$(line_count "${root}/selected_videos.jsonl")"
    log "${batch}: pass4 trajectories train_sft=${train_sft} train_rl=${train_rl} val=${val} test=${test} total=${total} selected=${selected}"
    (( train_sft > 0 )) || die "${batch}: train_sft trajectories is empty/missing"
    (( train_rl > 0 )) || die "${batch}: train_rl trajectories is empty/missing"
    (( val > 0 )) || die "${batch}: val trajectories is empty/missing"
    (( test > 0 )) || die "${batch}: test trajectories is empty/missing"
    if (( selected > 0 )); then
        [[ "${total}" == "${selected}" ]] || die "${batch}: final split total ${total} != selected_videos ${selected}"
    fi
}

clean_pass5_outputs() {
    local root="$1"
    rm -f \
        "${root}/final/train_sft_messages.jsonl" \
        "${root}/final/val_messages.jsonl" \
        "${root}/final/test_messages.jsonl" \
        "${root}/final/dataset_info.json" \
        "${root}/final/_version"
    for protocol in ts_image video_meta; do
        local dir="${root}/rendered/${protocol}"
        mkdir -p "${dir}"
        rm -f \
            "${dir}/train_sft_messages.jsonl" \
            "${dir}/val_messages.jsonl" \
            "${dir}/test_messages.jsonl" \
            "${dir}/dataset_info.json" \
            "${dir}/train_rl_multi_q.parquet" \
            "${dir}/val_rl_multi_q.parquet"
    done
    rm -rf "${root}/trajectory_bank"
}

render_pass5() {
    local root="$1"
    local label="$2"
    local out_dir="$3"
    local protocol="$4"
    log "${label}: pass5 render protocol=${protocol} out=${out_dir}"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.pass5_messages \
        --input traj \
        --final-dir "${root}/final" \
        --output-dir "${out_dir}" \
        --frame-protocol "${protocol}"
}

build_parquets() {
    local root="$1"
    for protocol in ts_image video_meta; do
        local dir="${root}/rendered/${protocol}"
        log "${root}: build RL parquet protocol=${protocol}"
        THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.build_verl_parquet \
            --jsonl "${root}/final/train_rl_trajectories.jsonl" \
            --out "${dir}/train_rl_multi_q.parquet" \
            --multi_q \
            --frame-protocol "${protocol}" \
            --render-layout standard
        THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.build_verl_parquet \
            --jsonl "${root}/final/val_trajectories.jsonl" \
            --out "${dir}/val_rl_multi_q.parquet" \
            --multi_q \
            --frame-protocol "${protocol}" \
            --render-layout standard
    done
}

validate_outputs() {
    local batch="$1"
    local root="$2"
    local want_version="$3"
    local final_version
    final_version="$(test -f "${root}/final/_version" && head -n 1 "${root}/final/_version" || echo MISSING)"
    [[ "${final_version}" == "${want_version}" ]] || die "${batch}: final/_version=${final_version}, expected ${want_version}"

    local final_train final_sft_traj final_rl final_val final_test expected_total ts_train vm_train bank_count video_count
    final_train="$(line_count "${root}/final/train_sft_messages.jsonl")"
    final_sft_traj="$(line_count "${root}/final/train_sft_trajectories.jsonl")"
    final_rl="$(line_count "${root}/final/train_rl_trajectories.jsonl")"
    final_val="$(line_count "${root}/final/val_trajectories.jsonl")"
    final_test="$(line_count "${root}/final/test_trajectories.jsonl")"
    expected_total=$((final_sft_traj + final_rl + final_val + final_test))
    ts_train="$(line_count "${root}/rendered/ts_image/train_sft_messages.jsonl")"
    vm_train="$(line_count "${root}/rendered/video_meta/train_sft_messages.jsonl")"
    bank_count="$(count_json_files "${root}/trajectory_bank/trajectories")"
    video_count="$(line_count "${root}/trajectory_bank/video_stats.jsonl")"

    log "${batch}: messages final=${final_train} ts_image=${ts_train} video_meta=${vm_train}; bank=${bank_count}; videos=${video_count}; expected_total=${expected_total}; version=${final_version}"
    [[ "${final_train}" != "0" ]] || die "${batch}: final train_sft_messages is empty/missing"
    [[ "${ts_train}" == "${final_train}" ]] || die "${batch}: ts_image train_sft count ${ts_train} != final ${final_train}"
    [[ "${vm_train}" == "${final_train}" ]] || die "${batch}: video_meta train_sft count ${vm_train} != final ${final_train}"
    [[ "${bank_count}" == "${expected_total}" ]] || die "${batch}: bank trajectory count ${bank_count}, expected ${expected_total}"
    [[ "${video_count}" == "${expected_total}" ]] || die "${batch}: bank video count ${video_count}, expected ${expected_total}"
}

run_batch() {
    local batch="$1"
    local root="data/agent_v5/${batch}"
    local log_dir="${root}/logs"
    mkdir -p "${log_dir}"

    [[ -d "${root}" ]] || die "${batch}: root not found: ${root}"
    validate_trajectories "${batch}" "${root}"
    log "${batch}: clean pass5 outputs and old trajectory_bank"
    clean_pass5_outputs "${root}"

    render_pass5 "${root}" "${batch}/legacy-final" "${root}/final" "ts_image" \
        2>&1 | tee "${log_dir}/pass5_final_ts_image_${RUN_ID}.log"
    render_pass5 "${root}" "${batch}/rendered-ts_image" "${root}/rendered/ts_image" "ts_image" \
        2>&1 | tee "${log_dir}/pass5_rendered_ts_image_${RUN_ID}.log"
    render_pass5 "${root}" "${batch}/rendered-video_meta" "${root}/rendered/video_meta" "video_meta" \
        2>&1 | tee "${log_dir}/pass5_rendered_video_meta_${RUN_ID}.log"

    stamp_pass5_marker "${root}"
    build_parquets "${root}" 2>&1 | tee "${log_dir}/rl_parquet_${RUN_ID}.log"

    log "${batch}: export trajectory_bank"
    THINKSTREAM_DATA_ROOT="${root}" "${PYTHON}" -m scripts.agent_data_v5.export_trajectory_bank \
        --data-dir "${root}" \
        2>&1 | tee "${log_dir}/trajectory_bank_${RUN_ID}.log"
}

PASS5_VERSION="$(current_pass5_version)"
log "start rerender_pass5_bank RUN_ID=${RUN_ID} pass5_version=${PASS5_VERSION} batches=${BATCHES}"
for batch in ${BATCHES}; do
    run_batch "${batch}"
    validate_outputs "${batch}" "data/agent_v5/${batch}" "${PASS5_VERSION}"
done
log "all done"
