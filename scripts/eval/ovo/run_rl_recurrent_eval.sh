#!/bin/bash
#
# OVO-Bench eval through the same verl recurrent AgentLoop used by RL.
#
# This path is for correctness-sensitive monitoring: it first converts the
# original OVO JSON into ThinkStream multi-Q trajectory/parquet rows, then runs
# verl validation with VAL_ONLY=true. No PPO update is performed.
#
# Usage:
#   bash scripts/eval/ovo/run_rl_recurrent_eval.sh \
#     --ckpt output/agent-rl \
#     --benchmark_json /path/to/ovo_bench_new.json \
#     --frames_root /path/to/OVO-Bench/frames \
#     [--tasks CRR,SSR,REC]

set -euo pipefail

CKPT=${CKPT:-}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}
TASKS=${TASKS:-}
SCORING=${SCORING:-strict}
OUT_DIR=${OUT_DIR:-}
PREBUILT_TRAJ_JSONL=${PREBUILT_TRAJ_JSONL:-}
PREBUILT_PARQUET=${PREBUILT_PARQUET:-}
PREBUILT_BUILD_SUMMARY=${PREBUILT_BUILD_SUMMARY:-}
MAX_QUESTIONS_PER_TRAJECTORY=${MAX_QUESTIONS_PER_TRAJECTORY:-16}
MAX_SPAN_CHUNKS=${MAX_SPAN_CHUNKS:-512}
PRE_CONTEXT_CHUNKS=${PRE_CONTEXT_CHUNKS:-64}
POST_CONTEXT_CHUNKS=${POST_CONTEXT_CHUNKS:-2}
PACK_ACROSS_TASKS=${PACK_ACROSS_TASKS:-false}
SPLIT_POLICY=${SPLIT_POLICY:-strict25_45}
SHORT_MIN_SPAN_CHUNKS=${SHORT_MIN_SPAN_CHUNKS:-25}
SHORT_MAX_SPAN_CHUNKS=${SHORT_MAX_SPAN_CHUNKS:-45}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --benchmark_json|--benchmark-json) BENCHMARK_JSON="$2"; shift 2 ;;
        --frames_root|--frames-root) FRAMES_ROOT="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --scoring) SCORING="$2"; shift 2 ;;
        --out_dir|--out-dir) OUT_DIR="$2"; shift 2 ;;
        --traj_jsonl|--traj-jsonl) PREBUILT_TRAJ_JSONL="$2"; shift 2 ;;
        --parquet) PREBUILT_PARQUET="$2"; shift 2 ;;
        --build_summary|--build-summary) PREBUILT_BUILD_SUMMARY="$2"; shift 2 ;;
        --max_questions_per_trajectory|--max-questions-per-trajectory)
            MAX_QUESTIONS_PER_TRAJECTORY="$2"; shift 2 ;;
        --max_span_chunks|--max-span-chunks) MAX_SPAN_CHUNKS="$2"; shift 2 ;;
        --pre_context_chunks|--pre-context-chunks) PRE_CONTEXT_CHUNKS="$2"; shift 2 ;;
        --post_context_chunks|--post-context-chunks) POST_CONTEXT_CHUNKS="$2"; shift 2 ;;
        --pack_across_tasks|--pack-across-tasks) PACK_ACROSS_TASKS=true; shift ;;
        --split_policy|--split-policy) SPLIT_POLICY="$2"; shift 2 ;;
        --short_min_span_chunks|--short-min-span-chunks) SHORT_MIN_SPAN_CHUNKS="$2"; shift 2 ;;
        --short_max_span_chunks|--short-max-span-chunks) SHORT_MAX_SPAN_CHUNKS="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

USING_PREBUILT=false
if [[ -n "${PREBUILT_TRAJ_JSONL}" || -n "${PREBUILT_PARQUET}" ]]; then
    USING_PREBUILT=true
fi

if [[ -z "${CKPT}" || -z "${FRAMES_ROOT}" || ( "${USING_PREBUILT}" != "true" && -z "${BENCHMARK_JSON}" ) ]]; then
    echo "ERROR: --ckpt and --frames_root are required; --benchmark_json is required unless --traj_jsonl/--parquet are provided" >&2
    exit 1
fi
if [[ "${USING_PREBUILT}" != "true" && ! -f "${BENCHMARK_JSON}" ]]; then
    echo "ERROR: benchmark JSON not found: ${BENCHMARK_JSON}" >&2
    exit 1
fi
if [[ ! -d "${FRAMES_ROOT}" ]]; then
    echo "ERROR: frames root not found: ${FRAMES_ROOT}" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${CKPT}/eval/ovo_rl_recurrent"
    mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/ovo_rl_recurrent"
fi
mkdir -p "${OUT_DIR}/input" "${OUT_DIR}/validation"

TRAJ_JSONL="${PREBUILT_TRAJ_JSONL:-${OUT_DIR}/input/ovo_trajectories.jsonl}"
PARQUET="${PREBUILT_PARQUET:-${OUT_DIR}/input/ovo_rl_multi_q.parquet}"
SUMMARY_JSON="${PREBUILT_BUILD_SUMMARY:-${OUT_DIR}/input/build_summary.json}"
VAL_DUMP_DIR="${OUT_DIR}/validation/generations"
SUMMARY_OUT="${OUT_DIR}/summary.json"

BUILD_ARGS=()
if [[ -n "${TASKS}" ]]; then
    BUILD_ARGS+=(--tasks "${TASKS}")
fi
if [[ "${PACK_ACROSS_TASKS}" == "1" || "${PACK_ACROSS_TASKS}" == "true" ]]; then
    BUILD_ARGS+=(--pack-across-tasks)
fi

echo "============================================================"
echo "OVO eval via RL recurrent AgentLoop"
echo "  ckpt:       ${CKPT}"
echo "  benchmark:  ${BENCHMARK_JSON}"
echo "  frames:     ${FRAMES_ROOT}"
echo "  out:        ${OUT_DIR}"
echo "  scoring:    ${SCORING}"
echo "  max span:   ${MAX_SPAN_CHUNKS}"
echo "  pre ctx:    ${PRE_CONTEXT_CHUNKS}"
echo "  post ctx:   ${POST_CONTEXT_CHUNKS}"
echo "  pack tasks: ${PACK_ACROSS_TASKS}"
echo "  split:      ${SPLIT_POLICY}"
echo "  short span: ${SHORT_MIN_SPAN_CHUNKS}-${SHORT_MAX_SPAN_CHUNKS}"
if [[ "${USING_PREBUILT}" == "true" ]]; then
    echo "  prebuilt:   ${TRAJ_JSONL}"
    echo "  parquet:    ${PARQUET}"
    echo "  build sum:  ${SUMMARY_JSON}"
fi
[ -n "${TASKS}" ] && echo "  tasks:      ${TASKS}"
echo "============================================================"

if [[ "${USING_PREBUILT}" == "true" ]]; then
    if [[ ! -f "${TRAJ_JSONL}" ]]; then
        echo "ERROR: prebuilt trajectory JSONL not found: ${TRAJ_JSONL}" >&2
        exit 1
    fi
    if [[ ! -f "${PARQUET}" ]]; then
        echo "ERROR: prebuilt parquet not found: ${PARQUET}" >&2
        exit 1
    fi
    if [[ ! -f "${SUMMARY_JSON}" ]]; then
        echo "ERROR: prebuilt build summary not found: ${SUMMARY_JSON}" >&2
        exit 1
    fi
else
    "${PYTHON_BIN:-python}" scripts/eval/ovo/build_rl_trajectories.py \
        --benchmark-json "${BENCHMARK_JSON}" \
        --out-jsonl "${TRAJ_JSONL}" \
        --out-parquet "${PARQUET}" \
        --scoring "${SCORING}" \
        --max-questions-per-trajectory "${MAX_QUESTIONS_PER_TRAJECTORY}" \
        --max-span-chunks "${MAX_SPAN_CHUNKS}" \
        --pre-context-chunks "${PRE_CONTEXT_CHUNKS}" \
        --post-context-chunks "${POST_CONTEXT_CHUNKS}" \
        --split-policy "${SPLIT_POLICY}" \
        --short-min-span-chunks "${SHORT_MIN_SPAN_CHUNKS}" \
        --short-max-span-chunks "${SHORT_MAX_SPAN_CHUNKS}" \
        --summary-out "${SUMMARY_JSON}" \
        "${BUILD_ARGS[@]}"
fi

# MAX_CHUNKS is an absolute chunk cap in the AgentLoop, not the segment span.
# OVO contains late probes, so keep this above the largest absolute chunk id.
export MAX_CHUNKS="${MAX_CHUNKS:-2048}"
export LLM="${CKPT}"
export TRAIN_JSONL="${TRAJ_JSONL}"
export VAL_JSONL="${TRAJ_JSONL}"
export TRAIN_PARQUET="${PARQUET}"
export VAL_PARQUET="${PARQUET}"
export FRAMES_ROOT="${FRAMES_ROOT}"
export VAL_ONLY=true
export VAL_BEFORE_TRAIN=true
export VALIDATION_DATA_DIR="${VAL_DUMP_DIR}"
export DATA_SHUFFLE=false
export TEST_FREQ=-1
export SAVE_FREQ="${SAVE_FREQ:-1000000}"
export RUN_NAME="${RUN_NAME:-ovo-rl-recurrent-eval}"
export THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-streaming}"
export THINKSTREAM_ROLLOUT_ENGINE="${THINKSTREAM_ROLLOUT_ENGINE:-streaming}"
export FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
export THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard_query_last}"
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export THINKSTREAM_PREFER_PATH_FRAME_INDEX="${THINKSTREAM_PREFER_PATH_FRAME_INDEX:-0}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export PPO_MINI_BS="${PPO_MINI_BS:-${BATCH_SIZE}}"
export GROUP_SIZE="${GROUP_SIZE:-1}"
export NPROC="${NPROC:-8}"
export MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-4096}"
export OUTPUT_DIR="${OUT_DIR}/verl"
export THINKSTREAM_OUTPUT_DIR="${OUTPUT_DIR}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${OUT_DIR}/runtime}"

bash scripts/grpo_train_verl.sh

GEN_JSONL="${VAL_DUMP_DIR}/0.jsonl"
if [[ -f "${GEN_JSONL}" ]]; then
    "${PYTHON_BIN:-python}" scripts/audit/summarize_rl_recurrent_validation.py \
        --generations "${GEN_JSONL}" \
        --build-summary "${SUMMARY_JSON}" \
        --out "${SUMMARY_OUT}"
else
    echo "WARNING: validation dump not found: ${GEN_JSONL}" >&2
fi

echo ""
echo "Done."
echo "  OVO RL parquet:        ${PARQUET}"
echo "  Build summary:         ${SUMMARY_JSON}"
echo "  Validation generations:${VAL_DUMP_DIR}/0.jsonl"
echo "  Summary:               ${SUMMARY_OUT}"
