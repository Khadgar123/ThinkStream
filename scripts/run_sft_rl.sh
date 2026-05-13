#!/usr/bin/env bash
# One-command ThinkStream SFT -> verl GRPO launcher.
#
# Minimal usage:
#   THINKSTREAM_DATA_ROOT=data/agent_v5/<batch_root> \
#   BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
#   bash scripts/run_sft_rl.sh
#
# Defaults target the current full-video OVO-style setting:
#   video_meta + standard_query_last, multi-question rows, recurrent RL.

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

RUN_ID="${RUN_ID:-sft_rl_$(timestamp)}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-video_meta}}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard_query_last}"
case "${THINKSTREAM_RENDER_LAYOUT}" in
  standard_query_last) ;;
  *)
    echo "ERROR: unsupported THINKSTREAM_RENDER_LAYOUT=${THINKSTREAM_RENDER_LAYOUT}" >&2
    exit 2
    ;;
esac
if [[ "${FRAME_PROTOCOL}" != "video_meta" ]]; then
  echo "ERROR: canonical SFT/RL uses FRAME_PROTOCOL=video_meta" >&2
  echo "       got FRAME_PROTOCOL=${FRAME_PROTOCOL} THINKSTREAM_RENDER_LAYOUT=${THINKSTREAM_RENDER_LAYOUT}" >&2
  exit 2
fi
export THINKSTREAM_MEMORY_POSITION="${THINKSTREAM_MEMORY_POSITION:-before_visual}"
THINKSTREAM_DATA_ROOT="${THINKSTREAM_DATA_ROOT:-${AGENT_DATA_DIR:-data/agent_v5}}"
if [[ "${THINKSTREAM_DATA_ROOT}" == */final ]]; then
  THINKSTREAM_DATA_ROOT="$(dirname "${THINKSTREAM_DATA_ROOT}")"
fi

PARENT_DIR="$(dirname "${PROJECT_DIR}")"
THINKSTREAM_ENV="${THINKSTREAM_ENV:-${PARENT_DIR}/envs/thinkstream}"
if [[ -x "${THINKSTREAM_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${THINKSTREAM_ENV}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi
export THINKSTREAM_ENV PYTHON_BIN

BASE_MODEL="${BASE_MODEL:-${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-${BASE_MODEL}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_ID}}"
mkdir -p "${LOG_ROOT}"

SFT_RENDERED_DIR="${THINKSTREAM_DATA_ROOT}/rendered/trajectory"

RUN_SFT="${RUN_SFT:-1}"
RUN_RL="${RUN_RL:-1}"
SFT_RUN_NAME="${SFT_RUN_NAME:-agent-trajectory-sft-${RUN_ID}}"
RL_RUN_NAME="${RL_RUN_NAME:-agent-rl-${RUN_ID}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${OUTPUT_ROOT}/${SFT_RUN_NAME}}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RL_RUN_NAME}}"

exec > >(tee -a "${LOG_ROOT}/run.log") 2>&1

echo "== ThinkStream SFT -> RL =="
echo "run_id=${RUN_ID}"
echo "data_root=${THINKSTREAM_DATA_ROOT}"
echo "sft_rendered_dir=${SFT_RENDERED_DIR}"
echo "base_model=${BASE_MODEL}"
echo "processor_model=${PROCESSOR_MODEL}"
echo "sft_output=${SFT_OUTPUT_DIR}"
echo "rl_output=${RL_OUTPUT_DIR}"

require_file "${SFT_RENDERED_DIR}/train_sft_trajectory.jsonl"
require_file "${SFT_RENDERED_DIR}/val_trajectory.jsonl"

if [[ "${RUN_SFT}" == "1" ]]; then
  echo "== Stage 1/2: SFT =="
  THINKSTREAM_DATA_ROOT="${THINKSTREAM_DATA_ROOT}" \
  THINKSTREAM_FINAL_DIR="${SFT_RENDERED_DIR}" \
  THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
  FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_PROCESSOR_PATH="${PROCESSOR_MODEL}" \
  LLM="${BASE_MODEL}" \
  RUN_NAME="${SFT_RUN_NAME}" \
  OUTPUT_DIR="${SFT_OUTPUT_DIR}" \
  bash scripts/sft_trajectory.sh
  SFT_CKPT="$(latest_checkpoint "${SFT_OUTPUT_DIR}")"
else
  SFT_CKPT="${SFT_CKPT:?Set SFT_CKPT when RUN_SFT=0}"
fi
echo "${SFT_CKPT}" > "${LOG_ROOT}/sft_checkpoint.txt"
echo "sft_ckpt=${SFT_CKPT}"

if [[ "${RUN_RL}" == "1" ]]; then
  echo "== Stage 2/2: verl GRPO RL =="
  THINKSTREAM_DATA_ROOT="${THINKSTREAM_DATA_ROOT}" \
  THINKSTREAM_OUTPUT_DIR="${RL_OUTPUT_DIR}" \
  THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT}" \
  FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
  THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}" \
  THINKSTREAM_ROLLOUT_ENGINE="${THINKSTREAM_ROLLOUT_ENGINE:-streaming}" \
  THINKSTREAM_RL_EPISODE_MODE="${THINKSTREAM_RL_EPISODE_MODE:-full}" \
  THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE="${THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE:-offline_pass2_boundaries}" \
  THINKSTREAM_RL_REWARD_PROFILE="${THINKSTREAM_RL_REWARD_PROFILE:-initial_outcome_time_format_decision}" \
  THINKSTREAM_ENABLE_STEP_ACTION_REWARD="${THINKSTREAM_ENABLE_STEP_ACTION_REWARD:-0}" \
  THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD="${THINKSTREAM_ENABLE_COMPRESS_ACTION_REWARD:-0}" \
  MULTI_Q="${MULTI_Q:-1}" \
  LLM="${SFT_CKPT}" \
  RUN_NAME="${RL_RUN_NAME}" \
  BATCH_SIZE="${RL_BATCH_SIZE:-${BATCH_SIZE:-4}}" \
  GROUP_SIZE="${RL_GROUP_SIZE:-${GROUP_SIZE:-8}}" \
  TP_SIZE="${RL_TP_SIZE:-${TP_SIZE:-2}}" \
  MAX_NEW_TOKEN="${RL_MAX_NEW_TOKEN:-${MAX_NEW_TOKEN:-4096}}" \
  MAX_CHUNKS="${RL_MAX_CHUNKS:-${MAX_CHUNKS:-420}}" \
  MAX_ACTION_TOKENS="${RL_MAX_ACTION_TOKENS:-${MAX_ACTION_TOKENS:-256}}" \
  MAX_COMPRESS_ACTION_TOKENS="${RL_MAX_COMPRESS_ACTION_TOKENS:-${MAX_COMPRESS_ACTION_TOKENS:-1024}}" \
  PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}" \
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}" \
  FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}" \
  DATA_SHUFFLE="${DATA_SHUFFLE:-false}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}" \
  bash scripts/grpo_train_verl.sh
else
  echo "RUN_RL=0, skip RL."
fi

echo "== Done =="
echo "sft_ckpt=${SFT_CKPT}"
echo "rl_output=${RL_OUTPUT_DIR}"
