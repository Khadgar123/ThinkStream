#!/bin/bash
# Sequential overnight training for freshly regenerated batch2 + batch3.
#
# Default sequence:
#   1. wait/validate batch2 and batch3 rendered data
#   2. SFT on batch2 from BASE_MODEL
#   3. SFT on batch3 from the best/latest batch2 SFT checkpoint
#   4. optional RL on RL_BATCH (default: last batch, usually batch3)
#
# Run in tmux:
#   tmux new -s batch23_train
#   bash scripts/run_batch23_overnight_train.sh
#
# Useful overrides:
#   BATCHES="batch2 batch3" RUN_RL=0 bash scripts/run_batch23_overnight_train.sh
#   START_AT=batch3 BASE_MODEL=/path/to/ckpt bash scripts/run_batch23_overnight_train.sh
#   WAIT_FOR_DATA=1 WAIT_DATA_TIMEOUT_SEC=28800 bash scripts/run_batch23_overnight_train.sh
#   WAIT_FOR_FRESH_DATA=0 bash scripts/run_batch23_overnight_train.sh  # data already regenerated
#   DATA_READY_MARKER=/path/to/done bash scripts/run_batch23_overnight_train.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
SCRIPT_START_EPOCH="$(date +%s)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/batch23_overnight_${TS}}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

on_error() {
    local code=$?
    log "FAILED with exit code ${code}. See ${LOG_FILE}"
    exit "${code}"
}
trap on_error ERR

BATCHES="${BATCHES:-batch2 batch3}"
START_AT="${START_AT:-}"
RUN_SFT="${RUN_SFT:-1}"
RUN_RL="${RUN_RL:-1}"
WAIT_FOR_DATA="${WAIT_FOR_DATA:-1}"
WAIT_DATA_TIMEOUT_SEC="${WAIT_DATA_TIMEOUT_SEC:-21600}"
WAIT_FOR_FRESH_DATA="${WAIT_FOR_FRESH_DATA:-0}"
DATA_READY_MARKER="${DATA_READY_MARKER:-}"
REQUIRE_FRESH_MARKER="${REQUIRE_FRESH_MARKER:-1}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
BASE_MODEL="${BASE_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"
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

NPROC="${NPROC:-8}"
BSZ="${BSZ:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_N="${EVAL_N:-300}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-16384}"
INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION:-False}"
# Macro action balance: silent/response/recall/compress = 0.25 each.
# Recall is split into tool-call and post-recall no-tools answer rows in pass5,
# so the two recall subtypes share the 0.25 macro bucket.
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS:-silent=0.25,response=0.25,recall=0.125,post_recall=0.125,compress=0.25}"
CLASS_LOSS_ALPHA="${CLASS_LOSS_ALPHA:-1.0}"
CLASS_LOSS_MAX_WEIGHT="${CLASS_LOSS_MAX_WEIGHT:-8.0}"
GROUP_BY_MODALITY="${GROUP_BY_MODALITY:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

RL_BATCH="${RL_BATCH:-last}"
RL_MAX_STEPS="${RL_MAX_STEPS:-}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-4}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
RL_MAX_CHUNKS="${RL_MAX_CHUNKS:-120}"
RL_TP_SIZE="${RL_TP_SIZE:-2}"
RL_GPU_MEM_UTIL="${RL_GPU_MEM_UTIL:-0.55}"
RL_SAVE_FREQ="${RL_SAVE_FREQ:-50}"
RL_TEST_FREQ="${RL_TEST_FREQ:-25}"

LOCK_FILE="${LOCK_FILE:-${PROJECT_DIR}/logs/batch23_overnight.lock}"
if [[ -e "${LOCK_FILE}" ]]; then
    die "lock exists: ${LOCK_FILE}. Remove it only if no overnight training is running."
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

wait_for_path() {
    local path="$1"
    local label="$2"
    local start now
    start="$(date +%s)"
    while [[ ! -e "${path}" ]]; do
        if [[ "${WAIT_FOR_DATA}" != "1" ]]; then
            die "missing ${label}: ${path}"
        fi
        now="$(date +%s)"
        if (( now - start > WAIT_DATA_TIMEOUT_SEC )); then
            die "timeout waiting for ${label}: ${path}"
        fi
        log "waiting for ${label}: ${path}"
        sleep 60
    done
}

wait_for_fresh_path() {
    local path="$1"
    local label="$2"
    wait_for_path "${path}" "${label}"
    if [[ "${WAIT_FOR_FRESH_DATA}" != "1" || -n "${DATA_READY_MARKER}" ]]; then
        return
    fi

    local mtime
    while true; do
        mtime="$(stat -c %Y "${path}")"
        if (( mtime >= SCRIPT_START_EPOCH )); then
            return
        fi
        log "waiting for fresh ${label}: ${path} (mtime before script start)"
        sleep 60
    done
}

wait_for_marker() {
    if [[ -z "${DATA_READY_MARKER}" ]]; then
        return
    fi
    local marker="${DATA_READY_MARKER}"
    local start now mtime
    start="$(date +%s)"
    while true; do
        if [[ -f "${marker}" ]]; then
            if [[ "${REQUIRE_FRESH_MARKER}" != "1" ]]; then
                log "data ready marker accepted: ${marker}"
                return
            fi
            mtime="$(stat -c %Y "${marker}")"
            if (( mtime >= SCRIPT_START_EPOCH )); then
                log "fresh data ready marker accepted: ${marker}"
                return
            fi
            log "found stale data ready marker; waiting for fresh touch: ${marker}"
        elif [[ "${WAIT_FOR_DATA}" != "1" ]]; then
            die "missing DATA_READY_MARKER: ${marker}"
        else
            log "waiting for data ready marker: ${marker}"
        fi
        now="$(date +%s)"
        if (( now - start > WAIT_DATA_TIMEOUT_SEC )); then
            die "timeout waiting for DATA_READY_MARKER: ${marker}"
        fi
        sleep 60
    done
}

count_jsonl() {
    local path="$1"
    "${PYTHON_BIN}" - "$path" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0)
else:
    with p.open("rb") as f:
        print(sum(1 for line in f if line.strip()))
PY
}

validate_batch() {
    local batch="$1"
    local root="${PROJECT_DIR}/data/agent_v5/${batch}"
    local rendered="${root}/rendered/${FRAME_PROTOCOL}"
    local final="${root}/final"

    wait_for_path "${root}" "${batch} root"
    wait_for_fresh_path "${rendered}/train_sft_messages.jsonl" "${batch} rendered train_sft_messages"
    wait_for_fresh_path "${rendered}/val_messages.jsonl" "${batch} rendered val_messages"
    wait_for_fresh_path "${final}/train_rl_trajectories.jsonl" "${batch} train_rl_trajectories"
    wait_for_fresh_path "${final}/val_trajectories.jsonl" "${batch} val_trajectories"
    wait_for_path "${root}/frames" "${batch} frames"

    local n_train n_val n_rl n_rl_val
    n_train="$(count_jsonl "${rendered}/train_sft_messages.jsonl")"
    n_val="$(count_jsonl "${rendered}/val_messages.jsonl")"
    n_rl="$(count_jsonl "${final}/train_rl_trajectories.jsonl")"
    n_rl_val="$(count_jsonl "${final}/val_trajectories.jsonl")"
    log "${batch} counts: train_sft_messages=${n_train}, val_messages=${n_val}, train_rl_trajectories=${n_rl}, val_trajectories=${n_rl_val}"

    [[ "${n_train}" -gt 0 ]] || die "${batch}: empty train_sft_messages"
    [[ "${n_val}" -gt 0 ]] || die "${batch}: empty val_messages"
    [[ "${n_rl}" -gt 0 ]] || die "${batch}: empty train_rl_trajectories"
    [[ "${n_rl_val}" -gt 0 ]] || die "${batch}: empty val_trajectories"
}

best_or_latest_ckpt() {
    local output_dir="$1"
    "${PYTHON_BIN}" - "$output_dir" <<'PY'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
state = out / "trainer_state.json"
if state.exists():
    try:
        best = json.loads(state.read_text()).get("best_model_checkpoint")
        if best and Path(best).exists():
            print(best)
            raise SystemExit
    except Exception:
        pass
ckpts = []
for p in out.glob("checkpoint-*"):
    m = re.search(r"checkpoint-(\d+)$", p.name)
    if m:
        ckpts.append((int(m.group(1)), p))
if ckpts:
    print(str(sorted(ckpts)[-1][1]))
elif out.exists():
    print(str(out))
else:
    raise SystemExit(f"no checkpoint found in {out}")
PY
}

run_sft_batch() {
    local batch="$1"
    local model="$2"
    local run_name="${SFT_RUN_PREFIX:-agent-sft-batch23}-${batch}-${FRAME_PROTOCOL}-${TS}"
    local output_dir="${PROJECT_DIR}/output/${run_name}"

    log "SFT start: batch=${batch}, model=${model}, output=${output_dir}"
    env \
        WANDB_MODE="${WANDB_MODE}" \
        THINKSTREAM_DATA_ROOT="data/agent_v5/${batch}" \
        FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
        THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
        LLM="${model}" \
        RUN_NAME="${run_name}" \
        NPROC="${NPROC}" \
        BSZ="${BSZ}" \
        GRAD_ACCUM="${GRAD_ACCUM}" \
        EPOCHS="${EPOCHS}" \
        LR="${LR}" \
        EVAL_STEPS="${EVAL_STEPS}" \
        EVAL_N="${EVAL_N}" \
        SAVE_LIMIT="${SAVE_LIMIT}" \
        MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS}" \
        INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION}" \
        CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS}" \
        CLASS_LOSS_ALPHA="${CLASS_LOSS_ALPHA}" \
        CLASS_LOSS_MAX_WEIGHT="${CLASS_LOSS_MAX_WEIGHT}" \
        GROUP_BY_MODALITY="${GROUP_BY_MODALITY}" \
        bash scripts/sft_per_timestep.sh

    local ckpt
    ckpt="$(best_or_latest_ckpt "${output_dir}")"
    log "SFT done: batch=${batch}, checkpoint=${ckpt}"
    echo "${ckpt}" > "${LOG_ROOT}/last_sft_checkpoint.txt"
    LAST_SFT_CKPT="${ckpt}"
}

run_rl_batch() {
    local batch="$1"
    local model="$2"
    local run_name="${RL_RUN_PREFIX:-grpo-batch23}-${batch}-${FRAME_PROTOCOL}-${TS}"
    local output_dir="${PROJECT_DIR}/output/${run_name}"

    log "RL start: batch=${batch}, model=${model}, output=${output_dir}"
    local max_steps_env=()
    if [[ -n "${RL_MAX_STEPS}" ]]; then
        max_steps_env=(MAX_STEPS="${RL_MAX_STEPS}")
    fi
    env \
        WANDB_MODE="${WANDB_MODE}" \
        THINKSTREAM_DATA_ROOT="data/agent_v5/${batch}" \
        FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
        LLM="${model}" \
        RUN_NAME="${run_name}" \
        NPROC="${NPROC}" \
        BATCH_SIZE="${RL_BATCH_SIZE}" \
        GROUP_SIZE="${RL_GROUP_SIZE}" \
        MAX_CHUNKS="${RL_MAX_CHUNKS}" \
        TP_SIZE="${RL_TP_SIZE}" \
        GPU_MEM_UTIL="${RL_GPU_MEM_UTIL}" \
        SAVE_FREQ="${RL_SAVE_FREQ}" \
        TEST_FREQ="${RL_TEST_FREQ}" \
        THINKSTREAM_OUTPUT_DIR="${output_dir}" \
        THINKSTREAM_ENV="${THINKSTREAM_ENV}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        "${max_steps_env[@]}" \
        bash scripts/grpo_train_verl.sh
    log "RL done: batch=${batch}, output=${output_dir}"
}

log "overnight training log: ${LOG_FILE}"
log "batches: ${BATCHES}"
log "frame protocol: ${FRAME_PROTOCOL}"
log "base model: ${BASE_MODEL}"
log "processor model: ${PROCESSOR_MODEL}"
log "python: ${PYTHON_BIN}"
log "run SFT=${RUN_SFT}, run RL=${RUN_RL}, RL batch=${RL_BATCH}"
log "wait fresh data: ${WAIT_FOR_FRESH_DATA}, data ready marker: ${DATA_READY_MARKER:-<none>}"

read -r -a BATCH_ARRAY <<< "${BATCHES}"
[[ "${#BATCH_ARRAY[@]}" -gt 0 ]] || die "BATCHES is empty"

wait_for_marker

for batch in "${BATCH_ARRAY[@]}"; do
    validate_batch "${batch}"
done

current_model="${BASE_MODEL}"
started=0
last_batch="${BATCH_ARRAY[$((${#BATCH_ARRAY[@]} - 1))]}"

if [[ "${RUN_SFT}" == "1" ]]; then
    for batch in "${BATCH_ARRAY[@]}"; do
        if [[ -n "${START_AT}" && "${started}" == "0" ]]; then
            if [[ "${batch}" != "${START_AT}" ]]; then
                log "skip SFT batch=${batch}; START_AT=${START_AT}"
                continue
            fi
            started=1
        fi
        run_sft_batch "${batch}" "${current_model}"
        current_model="${LAST_SFT_CKPT}"
    done
fi

if [[ "${RUN_RL}" == "1" ]]; then
    if [[ "${RL_BATCH}" == "last" ]]; then
        RL_BATCH="${last_batch}"
    fi
    validate_batch "${RL_BATCH}"
    run_rl_batch "${RL_BATCH}" "${current_model}"
fi

log "ALL DONE. Final model/checkpoint seed for next stage: ${current_model}"
log "Logs: ${LOG_ROOT}"
