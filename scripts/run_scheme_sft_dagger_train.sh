#!/bin/bash
# Train a non-destructive SFT -> DAgger data -> DAgger SFT scheme.
#
# Default scheme:
#   data/agent_v5/batch2_sft125_dagger50_rl175_val75_test75
#
# This intentionally stops before RL by default. Start RL from the final
# DAgger checkpoint after checking the DAgger correction distribution.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

SCHEME_ROOT="${SCHEME_ROOT:-data/agent_v5/batch2_sft125_dagger50_rl175_val75_test75}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard}"
RUN_ID="${RUN_ID:-scheme-b2-sft125-dagger50-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/${RUN_ID}}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PARENT_DIR="$(dirname "${PROJECT_DIR}")"
THINKSTREAM_ENV="${THINKSTREAM_ENV:-${PARENT_DIR}/envs/thinkstream}"
PYTHON_BIN="${PYTHON_BIN:-${THINKSTREAM_ENV}/bin/python}"

BASE_MODEL="${BASE_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"

NPROC="${NPROC:-8}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_N="${EVAL_N:-128}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-16384}"
# Macro action balance: silent/response/recall/compress = 0.25 each.
# Recall is split into tool-call and post-recall no-tools answer rows in pass5,
# so the two recall subtypes share the 0.25 macro bucket.
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS:-silent=0.25,response=0.25,recall=0.125,post_recall=0.125,compress=0.25}"

DAGGER_TP_SIZE="${DAGGER_TP_SIZE:-8}"
DAGGER_ROLLOUT_BATCH_SIZE="${DAGGER_ROLLOUT_BATCH_SIZE:-8}"
DAGGER_MAX_NEW_TOKENS="${DAGGER_MAX_NEW_TOKENS:-256}"
DAGGER_COMPRESS_MAX_NEW_TOKENS="${DAGGER_COMPRESS_MAX_NEW_TOKENS:-512}"
DAGGER_GPU_MEM_UTIL="${DAGGER_GPU_MEM_UTIL:-0.45}"
DAGGER_MAX_MODEL_LEN="${DAGGER_MAX_MODEL_LEN:-16384}"
if [[ "${THINKSTREAM_RENDER_LAYOUT}" == "standard" ]]; then
    RENDERED_DIR="${SCHEME_ROOT}/rendered/${FRAME_PROTOCOL}"
else
    RENDERED_DIR="${SCHEME_ROOT}/rendered/${FRAME_PROTOCOL}_${THINKSTREAM_RENDER_LAYOUT}"
fi
DAGGER_OUT="${DAGGER_OUT:-${RENDERED_DIR}/train_sft_dagger_messages.jsonl}"
DAGGER_SOURCE="${DAGGER_SOURCE:-${SCHEME_ROOT}/final/train_sft_dagger_source_trajectories.jsonl}"

RUN_RL="${RUN_RL:-0}"

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

select_ckpt() {
    "${PYTHON_BIN}" - "$1" <<'PY'
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

summarize_dagger() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import collections, itertools, json, sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
sample_type = collections.Counter(row.get("sample_type") for row in rows)
passed = collections.Counter(str(bool((row.get("verification") or {}).get("passed"))) for row in rows)
reasons = collections.Counter(
    itertools.chain.from_iterable(
        ((row.get("dagger") or {}).get("selected_correction_reasons")
         or (row.get("dagger") or {}).get("correction_reasons")
         or [])
        for row in rows
    )
)
print("dagger_rows", len(rows))
print("dagger_sample_type", dict(sample_type))
print("dagger_verification", dict(passed))
print("dagger_reasons", dict(reasons.most_common(20)))
PY
}

[[ -x "${PYTHON_BIN}" ]] || die "missing python: ${PYTHON_BIN}"
[[ -f "${SCHEME_ROOT}/scheme.json" ]] || die "missing scheme.json in ${SCHEME_ROOT}"
[[ -f "${RENDERED_DIR}/train_sft_messages.jsonl" ]] || die "missing scheme SFT messages in ${RENDERED_DIR}"
[[ -f "${DAGGER_SOURCE}" ]] || die "missing DAgger source trajectories: ${DAGGER_SOURCE}"

log "run id: ${RUN_ID}"
log "log file: ${LOG_FILE}"
log "scheme root: ${SCHEME_ROOT}"
log "frame protocol: ${FRAME_PROTOCOL}"
log "render layout: ${THINKSTREAM_RENDER_LAYOUT}"
log "base model: ${BASE_MODEL}"
log "python: ${PYTHON_BIN}"
log "8-card SFT config: NPROC=${NPROC}, BSZ=${BSZ}, GRAD_ACCUM=${GRAD_ACCUM}"
log "DAgger decode: max_new_tokens=${DAGGER_MAX_NEW_TOKENS}, compress_max_new_tokens=${DAGGER_COMPRESS_MAX_NEW_TOKENS}"
cat "${SCHEME_ROOT}/scheme.json"

BASE_RUN_NAME="${BASE_RUN_NAME:-agent-sft-${RUN_ID}-base}"
BASE_OUTPUT="${PROJECT_DIR}/output/${BASE_RUN_NAME}"
log "stage 1/3: base teacher-forced SFT -> ${BASE_OUTPUT}"
WANDB_MODE="${WANDB_MODE:-offline}" \
THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
THINKSTREAM_FINAL_DIR="${RENDERED_DIR}" \
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
LLM="${BASE_MODEL}" \
RUN_NAME="${BASE_RUN_NAME}" \
NPROC="${NPROC}" \
BSZ="${BSZ}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
EPOCHS="${BASE_EPOCHS:-1}" \
LR="${BASE_LR:-2e-5}" \
EVAL_STEPS="${EVAL_STEPS}" \
EVAL_N="${EVAL_N}" \
SAVE_LIMIT="${SAVE_LIMIT}" \
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS}" \
INCLUDE_FAILED_VERIFICATION=False \
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS}" \
GROUP_BY_MODALITY=1 \
bash scripts/sft_per_timestep.sh

BASE_CKPT="$(select_ckpt "${BASE_OUTPUT}")"
echo "${BASE_CKPT}" > "${LOG_ROOT}/base_sft_checkpoint.txt"
log "base checkpoint: ${BASE_CKPT}"

log "stage 2/3: DAgger rollout correction data -> ${DAGGER_OUT}"
mkdir -p "$(dirname "${DAGGER_OUT}")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
PYTHONUNBUFFERED=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
"${PYTHON_BIN}" scripts/agent_data_v5/build_dagger_sft_vllm.py \
    --ckpt "${BASE_CKPT}" \
    --trajectories "${DAGGER_SOURCE}" \
    --out "${DAGGER_OUT}" \
    --data-dir "${SCHEME_ROOT}" \
    --frames-root "${SCHEME_ROOT}/frames" \
    --frame-protocol "${FRAME_PROTOCOL}" \
    --render-layout "${THINKSTREAM_RENDER_LAYOUT}" \
    --correction-only \
    --rollout-batch-size "${DAGGER_ROLLOUT_BATCH_SIZE}" \
    --max-new-tokens "${DAGGER_MAX_NEW_TOKENS}" \
    --compress-max-new-tokens "${DAGGER_COMPRESS_MAX_NEW_TOKENS}" \
    --tensor-parallel-size "${DAGGER_TP_SIZE}" \
    --gpu-memory-utilization "${DAGGER_GPU_MEM_UTIL}" \
    --max-model-len "${DAGGER_MAX_MODEL_LEN}" \
    --max-videos-per-prompt 2 \
    --log-every-steps 20
summarize_dagger "${DAGGER_OUT}" | tee "${LOG_ROOT}/dagger_summary.txt"

DAGGER_RUN_NAME="${DAGGER_RUN_NAME:-agent-sft-${RUN_ID}-dagger}"
DAGGER_OUTPUT="${PROJECT_DIR}/output/${DAGGER_RUN_NAME}"
log "stage 3/3: DAgger mixed SFT -> ${DAGGER_OUTPUT}"
WANDB_MODE="${WANDB_MODE:-offline}" \
THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
THINKSTREAM_FINAL_DIR="${RENDERED_DIR}" \
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
LLM="${BASE_CKPT}" \
RUN_NAME="${DAGGER_RUN_NAME}" \
NPROC="${NPROC}" \
BSZ="${BSZ}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
EPOCHS="${DAGGER_EPOCHS:-1}" \
LR="${DAGGER_LR:-1e-5}" \
EVAL_STEPS="${EVAL_STEPS}" \
EVAL_N="${EVAL_N}" \
SAVE_LIMIT="${SAVE_LIMIT}" \
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS}" \
INCLUDE_FAILED_VERIFICATION=False \
CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS}" \
GROUP_BY_MODALITY=1 \
bash scripts/sft_dagger.sh

DAGGER_CKPT="$(select_ckpt "${DAGGER_OUTPUT}")"
echo "${DAGGER_CKPT}" > "${LOG_ROOT}/dagger_sft_checkpoint.txt"
log "dagger checkpoint: ${DAGGER_CKPT}"

if [[ "${RUN_RL}" == "1" ]]; then
    log "RUN_RL=1: starting RL from ${DAGGER_CKPT}"
    WANDB_MODE="${WANDB_MODE:-offline}" \
    THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
    THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
    FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
    LLM="${DAGGER_CKPT}" \
    RUN_NAME="${RL_RUN_NAME:-grpo-${RUN_ID}}" \
    NPROC="${NPROC}" \
    BATCH_SIZE="${RL_BATCH_SIZE:-4}" \
    GROUP_SIZE="${RL_GROUP_SIZE:-8}" \
    MAX_CHUNKS="${RL_MAX_CHUNKS:-120}" \
    TP_SIZE="${RL_TP_SIZE:-2}" \
    GPU_MEM_UTIL="${RL_GPU_MEM_UTIL:-0.55}" \
    SAVE_FREQ="${RL_SAVE_FREQ:-50}" \
    TEST_FREQ="${RL_TEST_FREQ:-25}" \
    THINKSTREAM_OUTPUT_DIR="${PROJECT_DIR}/output/${RL_RUN_NAME:-grpo-${RUN_ID}}" \
    bash scripts/grpo_train_verl.sh
fi

log "DONE. Base=${BASE_CKPT}"
log "DONE. DAgger=${DAGGER_CKPT}"
