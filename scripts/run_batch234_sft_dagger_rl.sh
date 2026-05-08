#!/usr/bin/env bash
# End-to-end batch234 training:
#   base SFT -> DAgger correction data -> DAgger SFT -> verl GRPO.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

latest_checkpoint() {
  local run_dir="$1"
  "${PYTHON_BIN}" - "$run_dir" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"missing run dir: {root}")

state = root / "trainer_state.json"
if state.exists():
    try:
        best = json.loads(state.read_text()).get("best_model_checkpoint")
        if best and pathlib.Path(best).exists():
            print(best)
            raise SystemExit
    except Exception:
        pass

def step(path: pathlib.Path) -> int:
    m = re.match(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1

ckpts = [p for p in root.glob("checkpoint-*") if p.is_dir()]
if not ckpts:
    raise SystemExit(f"no checkpoints under {root}")
print(max(ckpts, key=step))
PY
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory is missing: ${path}" >&2
    exit 1
  fi
}

summarize_messages() {
  local path="$1"
  "${PYTHON_BIN}" - "$path" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
counts = collections.Counter()
total = 0
with path.open() as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        kind = row.get("meta", {}).get("sample_type") or row.get("sample_type") or "unknown"
        counts[kind] += 1
print(f"{path}: total={total} " + " ".join(f"{k}={counts[k]}" for k in sorted(counts)))
PY
}

RUN_ID="${RUN_ID:-batch234_strat_sft450_dagger300_rl550_$(timestamp)}"
SCHEME_ROOT="${SCHEME_ROOT:-data/agent_v5/batch234_bank_scheme_strat_sft450_dagger300_rl550_val100_test100}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard}"
PARENT_DIR="$(dirname "${PROJECT_DIR}")"
THINKSTREAM_ENV="${THINKSTREAM_ENV:-${PARENT_DIR}/envs/thinkstream}"
PYTHON_BIN="${PYTHON_BIN:-${THINKSTREAM_ENV}/bin/python}"
BASE_MODEL="${BASE_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_ID}}"
DAGGER_SOURCE="${DAGGER_SOURCE:-${SCHEME_ROOT}/final/train_sft_dagger_source_trajectories.jsonl}"
if [[ "${THINKSTREAM_RENDER_LAYOUT}" == "standard" ]]; then
  RENDERED_DIR="${SCHEME_ROOT}/rendered/${FRAME_PROTOCOL}"
else
  RENDERED_DIR="${SCHEME_ROOT}/rendered/${FRAME_PROTOCOL}_${THINKSTREAM_RENDER_LAYOUT}"
fi

if [[ -x "${PYTHON_BIN}" ]]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
  echo "Required python is missing or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== ThinkStream batch234 SFT + DAgger + RL =="
echo "run_id=${RUN_ID}"
echo "scheme_root=${SCHEME_ROOT}"
echo "frame_protocol=${FRAME_PROTOCOL}"
echo "render_layout=${THINKSTREAM_RENDER_LAYOUT}"
echo "base_model=${BASE_MODEL}"
echo "processor_model=${PROCESSOR_MODEL}"
echo "log_file=${LOG_FILE}"

require_dir "${SCHEME_ROOT}"
require_file "${SCHEME_ROOT}/scheme.json"
require_file "${RENDERED_DIR}/train_sft_messages.jsonl"
require_file "${RENDERED_DIR}/val_messages.jsonl"
require_file "${RENDERED_DIR}/test_messages.jsonl"
require_file "${DAGGER_SOURCE}"
require_file "${RENDERED_DIR}/train_rl_multi_q.parquet"

summarize_messages "${RENDERED_DIR}/train_sft_messages.jsonl"
summarize_messages "${RENDERED_DIR}/val_messages.jsonl"

# Current loss recipe used for this retry.
export CLASS_LOSS_TARGET_RATIOS="${CLASS_LOSS_TARGET_RATIOS:-silent=0.25,response=0.25,recall=0.125,post_recall=0.125,compress=0.25}"
export CLASS_LOSS_ALPHA="${CLASS_LOSS_ALPHA:-1.0}"
export CLASS_LOSS_MAX_WEIGHT="${CLASS_LOSS_MAX_WEIGHT:-8.0}"
export COMPRESS_TOKEN_WEIGHTING="${COMPRESS_TOKEN_WEIGHTING:-True}"
export COMPRESS_STRUCTURE_TOKEN_WEIGHT="${COMPRESS_STRUCTURE_TOKEN_WEIGHT:-2.0}"
export COMPRESS_BODY_TOKEN_WEIGHT="${COMPRESS_BODY_TOKEN_WEIGHT:-0.35}"
export COMPRESS_CLOSE_TOKEN_WEIGHT="${COMPRESS_CLOSE_TOKEN_WEIGHT:-4.0}"
export COMPRESS_CLOSE_TAIL_TOKENS="${COMPRESS_CLOSE_TAIL_TOKENS:-24}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export GROUP_BY_MODALITY="${GROUP_BY_MODALITY:-1}"
export INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION:-False}"
export MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-16384}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
export EVAL_N="${EVAL_N:-300}"
export SAVE_LIMIT="${SAVE_LIMIT:-2}"
export NPROC="${NPROC:-8}"
export BSZ="${BSZ:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"

RUN_BASE_SFT="${RUN_BASE_SFT:-1}"
RUN_DAGGER_BUILD="${RUN_DAGGER_BUILD:-1}"
RUN_DAGGER_SFT="${RUN_DAGGER_SFT:-1}"
RUN_RL="${RUN_RL:-1}"

BASE_EPOCHS="${BASE_EPOCHS:-2}"
DAGGER_EPOCHS="${DAGGER_EPOCHS:-1}"

BASE_RUN_NAME="${BASE_RUN_NAME:-agent-sft-${RUN_ID}-base}"
DAGGER_RUN_NAME="${DAGGER_RUN_NAME:-agent-sft-${RUN_ID}-dagger}"
RL_RUN_NAME="${RL_RUN_NAME:-agent-rl-${RUN_ID}}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${OUTPUT_ROOT}/${BASE_RUN_NAME}}"
DAGGER_OUTPUT_DIR="${DAGGER_OUTPUT_DIR:-${OUTPUT_ROOT}/${DAGGER_RUN_NAME}}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RL_RUN_NAME}}"
DAGGER_MESSAGES="${DAGGER_MESSAGES:-${RENDERED_DIR}/train_sft_dagger_messages.jsonl}"

if [[ "${RUN_BASE_SFT}" == "1" ]]; then
  echo "== Stage 1/4: base SFT =="
  THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
  THINKSTREAM_FINAL_DIR="${RENDERED_DIR}" \
  THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
  FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
  LLM="${BASE_MODEL}" \
  RUN_NAME="${BASE_RUN_NAME}" \
  OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
  EPOCHS="${BASE_EPOCHS}" \
  bash scripts/sft_per_timestep.sh
  BASE_CKPT="$(latest_checkpoint "${BASE_OUTPUT_DIR}")"
else
  BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT when RUN_BASE_SFT=0}"
fi
echo "${BASE_CKPT}" > "${LOG_ROOT}/base_sft_checkpoint.txt"
echo "base_ckpt=${BASE_CKPT}"

if [[ "${RUN_DAGGER_BUILD}" == "1" ]]; then
  echo "== Stage 2/4: build DAgger correction SFT data =="
  DAGGER_MODEL="${DAGGER_MODEL:-${BASE_CKPT}}"
  DAGGER_TP_SIZE="${DAGGER_TP_SIZE:-8}"
  DAGGER_ROLLOUT_BATCH_SIZE="${DAGGER_ROLLOUT_BATCH_SIZE:-32}"
  DAGGER_GPU_MEM_UTIL="${DAGGER_GPU_MEM_UTIL:-0.45}"
  DAGGER_MAX_MODEL_LEN="${DAGGER_MAX_MODEL_LEN:-16384}"
  DAGGER_MAX_NEW_TOKENS="${DAGGER_MAX_NEW_TOKENS:-256}"
  DAGGER_COMPRESS_MAX_NEW_TOKENS="${DAGGER_COMPRESS_MAX_NEW_TOKENS:-512}"
  DAGGER_CORRECTION_ONLY="${DAGGER_CORRECTION_ONLY:-1}"

  dagger_args=(
    --ckpt "${DAGGER_MODEL}"
    --trajectories "${DAGGER_SOURCE}"
    --out "${DAGGER_MESSAGES}"
    --data-dir "${SCHEME_ROOT}"
    --frames-root "${SCHEME_ROOT}/frames"
    --frame-protocol "${FRAME_PROTOCOL}"
    --render-layout "${THINKSTREAM_RENDER_LAYOUT}"
    --rollout-batch-size "${DAGGER_ROLLOUT_BATCH_SIZE}"
    --max-new-tokens "${DAGGER_MAX_NEW_TOKENS}"
    --compress-max-new-tokens "${DAGGER_COMPRESS_MAX_NEW_TOKENS}"
    --tensor-parallel-size "${DAGGER_TP_SIZE}"
    --gpu-memory-utilization "${DAGGER_GPU_MEM_UTIL}"
    --max-model-len "${DAGGER_MAX_MODEL_LEN}"
    --max-videos-per-prompt 2
    --log-every-steps 20
  )
  if [[ "${DAGGER_CORRECTION_ONLY}" == "1" ]]; then
    dagger_args+=(--correction-only)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  PYTHONUNBUFFERED=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${PYTHON_BIN}" scripts/agent_data_v5/build_dagger_sft_vllm.py "${dagger_args[@]}"
else
  require_file "${DAGGER_MESSAGES}"
fi
summarize_messages "${DAGGER_MESSAGES}"

if [[ "${RUN_DAGGER_SFT}" == "1" ]]; then
  echo "== Stage 3/4: DAgger SFT =="
  THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
  THINKSTREAM_FINAL_DIR="${RENDERED_DIR}" \
  THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
  FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
  LLM="${BASE_CKPT}" \
  RUN_NAME="${DAGGER_RUN_NAME}" \
  OUTPUT_DIR="${DAGGER_OUTPUT_DIR}" \
  EPOCHS="${DAGGER_EPOCHS}" \
  LR="${DAGGER_LR:-1e-5}" \
  bash scripts/sft_dagger.sh
  DAGGER_CKPT="$(latest_checkpoint "${DAGGER_OUTPUT_DIR}")"
else
  DAGGER_CKPT="${DAGGER_CKPT:?Set DAGGER_CKPT when RUN_DAGGER_SFT=0}"
fi
echo "${DAGGER_CKPT}" > "${LOG_ROOT}/dagger_sft_checkpoint.txt"
echo "dagger_ckpt=${DAGGER_CKPT}"

if [[ "${RUN_RL}" == "1" ]]; then
  echo "== Stage 4/4: verl GRPO =="
  THINKSTREAM_DATA_ROOT="${SCHEME_ROOT}" \
  THINKSTREAM_OUTPUT_DIR="${RL_OUTPUT_DIR}" \
  THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
  FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  LLM="${DAGGER_CKPT}" \
  RUN_NAME="${RL_RUN_NAME}" \
  NPROC="${RL_NPROC:-${NPROC}}" \
  BATCH_SIZE="${RL_BATCH_SIZE:-4}" \
  GROUP_SIZE="${RL_GROUP_SIZE:-8}" \
  MAX_CHUNKS="${RL_MAX_CHUNKS:-120}" \
  TP_SIZE="${RL_TP_SIZE:-2}" \
  GPU_MEM_UTIL="${RL_GPU_MEM_UTIL:-0.55}" \
  SAVE_FREQ="${RL_SAVE_FREQ:-50}" \
  TEST_FREQ="${RL_TEST_FREQ:-25}" \
  LR="${RL_LR:-5e-7}" \
  EPOCHS="${RL_EPOCHS:-1}" \
  MAX_STEPS="${RL_MAX_STEPS:-}" \
  bash scripts/grpo_train_verl.sh
else
  echo "RUN_RL=0, skip RL stage."
fi

echo "== Done =="
echo "base_ckpt=${BASE_CKPT}"
echo "dagger_ckpt=${DAGGER_CKPT}"
echo "rl_output_dir=${RL_OUTPUT_DIR}"
