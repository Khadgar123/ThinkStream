#!/bin/bash
#
# Run correctness monitoring on ThinkStream RL trajectory data through the
# production true-KV recurrent AgentLoop. This is the replacement for the old
# pre-RL simulated/vLLM rollout audit.

set -euo pipefail

CKPT=${CKPT:-}
SOURCE=${SOURCE:-}
FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}
OUT_DIR=${OUT_DIR:-output/rl_recurrent_audit}
MAX_QUESTIONS_PER_TRAJ=${MAX_QUESTIONS_PER_TRAJ:-16}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --frames-root|--frames_root) FRAMES_ROOT="$2"; shift 2 ;;
        --out-dir|--out_dir) OUT_DIR="$2"; shift 2 ;;
        --max-questions-per-traj|--max_questions_per_traj)
            MAX_QUESTIONS_PER_TRAJ="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${CKPT}" || -z "${SOURCE}" ]]; then
    echo "ERROR: --ckpt and --source are required" >&2
    exit 1
fi
if [[ ! -f "${SOURCE}" ]]; then
    echo "ERROR: source not found: ${SOURCE}" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

mkdir -p "${OUT_DIR}/input" "${OUT_DIR}/validation"

case "${SOURCE}" in
    *.parquet)
        PARQUET="${SOURCE}"
        TRAJ_JSONL="${TRAIN_JSONL:-}"
        ;;
    *.jsonl|*.jsonl.gz)
        TRAJ_JSONL="${SOURCE}"
        PARQUET="${OUT_DIR}/input/rl_multi_q.parquet"
        "${PYTHON_BIN:-python}" scripts/agent_data/build_verl_parquet.py \
            --jsonl "${TRAJ_JSONL}" \
            --out "${PARQUET}" \
            --max_questions_per_traj "${MAX_QUESTIONS_PER_TRAJ}" \
            --multi_q
        ;;
    *)
        echo "ERROR: source must be .parquet, .jsonl, or .jsonl.gz: ${SOURCE}" >&2
        exit 1
        ;;
esac

VAL_DUMP_DIR="${OUT_DIR}/validation/generations"
SUMMARY_OUT="${OUT_DIR}/summary.json"

echo "============================================================"
echo "RL recurrent trajectory audit"
echo "  ckpt:    ${CKPT}"
echo "  source:  ${SOURCE}"
echo "  parquet: ${PARQUET}"
[ -n "${TRAJ_JSONL}" ] && echo "  jsonl:   ${TRAJ_JSONL}"
[ -n "${FRAMES_ROOT}" ] && echo "  frames:  ${FRAMES_ROOT}"
echo "  out:     ${OUT_DIR}"
echo "============================================================"

export LLM="${CKPT}"
export TRAIN_PARQUET="${PARQUET}"
export VAL_PARQUET="${PARQUET}"
if [[ -n "${TRAJ_JSONL}" ]]; then
    export TRAIN_JSONL="${TRAJ_JSONL}"
    export VAL_JSONL="${TRAJ_JSONL}"
fi
if [[ -n "${FRAMES_ROOT}" ]]; then
    export FRAMES_ROOT="${FRAMES_ROOT}"
fi
export VAL_ONLY=true
export VAL_BEFORE_TRAIN=true
export VALIDATION_DATA_DIR="${VAL_DUMP_DIR}"
export DATA_SHUFFLE=false
export TEST_FREQ=-1
export SAVE_FREQ="${SAVE_FREQ:-1000000}"
export RUN_NAME="${RUN_NAME:-rl-recurrent-audit}"
export THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-streaming}"
export THINKSTREAM_ROLLOUT_ENGINE="${THINKSTREAM_ROLLOUT_ENGINE:-streaming}"
export FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
export THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard_query_last}"
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export PPO_MINI_BS="${PPO_MINI_BS:-${BATCH_SIZE}}"
export GROUP_SIZE="${GROUP_SIZE:-1}"
export NPROC="${NPROC:-8}"
export MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-4096}"
export MAX_CHUNKS="${MAX_CHUNKS:-2048}"
export OUTPUT_DIR="${OUT_DIR}/verl"
export THINKSTREAM_OUTPUT_DIR="${THINKSTREAM_OUTPUT_DIR:-${OUT_DIR}/verl}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${OUT_DIR}/runtime}"

bash scripts/grpo_train_verl.sh

GEN_JSONL="${VAL_DUMP_DIR}/0.jsonl"
if [[ -f "${GEN_JSONL}" ]]; then
    "${PYTHON_BIN:-python}" scripts/audit/summarize_rl_recurrent_validation.py \
        --generations "${GEN_JSONL}" \
        --out "${SUMMARY_OUT}"
    echo "Summary: ${SUMMARY_OUT}"
else
    echo "WARNING: validation dump not found: ${GEN_JSONL}" >&2
fi
