#!/bin/bash
#
# OVO-Bench full eval — SFT ckpt, system-triggered compression.
# Runs ALL 12 sub-tasks (RT/BT/FT) on the ORIGINAL ovo_bench_new.json
# in a single pass, dispatching per task family:
#
#   BT/RT (9 tasks):  single-realtime MCQ (A/B/C/D)
#   REC:              cumulative integer count, probed at multiple times
#   SSR:              per-probe Yes/No about a specific step
#   CRR:              ask-once at ask_time, probe with delay structure
#                     preserved (the formatted file destroys this)
#
# Locks --compress_mode system to match v11 SFT training. For self-pick
# compression (RL ckpt post-GDPO), use run_rl_full.sh.
#
# Reports per-task accuracy, per-category averages (RT / BT / FT),
# and overall (matches OVO paper Table 2 layout). CRR/SSR additionally
# get type=0/type=1 strict + lenient + fp_rate breakdown.
#
# Usage:
#   bash scripts/eval/ovo/run_sft_full.sh \
#     --ckpt output/agent-sft \
#     --benchmark_json /path/to/ovo_bench_new.json \
#     --video_root /path/to/videos \
#     [--frames_root /path/to/pre_extracted_frames] \
#     [--retriever hybrid] [--alpha 0.5] [--n_per_task 30]
#     [--tasks CRR,SSR,REC]   # subset for quick iteration

set -euo pipefail

CKPT=${CKPT:-}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}
TASKS=${TASKS:-}
N_PER_TASK=${N_PER_TASK:-}
RETRIEVER=${RETRIEVER:-bm25}
ALPHA=${ALPHA:-0.5}
SIGLIP_PATH=${SIGLIP_PATH:-google/siglip-base-patch16-224}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
PROFILE=${PROFILE:-16k}
SCORING=${SCORING:-strict}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-video_meta}}
RENDER_LAYOUT=${RENDER_LAYOUT:-${THINKSTREAM_RENDER_LAYOUT:-timeline_video_imagepad}}
COMPRESS_MODE=${COMPRESS_MODE:-system}
MEMORY_MODE=${MEMORY_MODE:-full}
SAVE_STEP_TRACE=${SAVE_STEP_TRACE:-0}
ENGINE=${ENGINE:-hf}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-}
VLLM_REPETITION_PENALTY=${VLLM_REPETITION_PENALTY:-1.0}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --benchmark_json)  BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root)      VIDEO_ROOT="$2"; shift 2 ;;
        --frames_root)     FRAMES_ROOT="$2"; shift 2 ;;
        --tasks)           TASKS="$2"; shift 2 ;;
        --n_per_task)      N_PER_TASK="$2"; shift 2 ;;
        --retriever)       RETRIEVER="$2"; shift 2 ;;
        --alpha)           ALPHA="$2"; shift 2 ;;
        --siglip_path)     SIGLIP_PATH="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --profile)         PROFILE="$2"; shift 2 ;;
        --scoring)         SCORING="$2"; shift 2 ;;
        --frame_protocol|--frame-protocol) FRAME_PROTOCOL="$2"; shift 2 ;;
        --render_layout|--render-layout) RENDER_LAYOUT="$2"; shift 2 ;;
        --compress_mode|--compress-mode) COMPRESS_MODE="$2"; shift 2 ;;
        --memory_mode|--memory-mode) MEMORY_MODE="$2"; shift 2 ;;
        --save_step_trace|--save-step-trace) SAVE_STEP_TRACE=1; shift ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --rollout_batch_size|--rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done
if [[ "${FRAME_PROTOCOL}" != "video_meta" || "${RENDER_LAYOUT}" != "timeline_video_imagepad" ]]; then
    echo "ERROR: canonical OVO SFT eval uses FRAME_PROTOCOL=video_meta RENDER_LAYOUT=timeline_video_imagepad" >&2
    echo "       got FRAME_PROTOCOL=${FRAME_PROTOCOL} RENDER_LAYOUT=${RENDER_LAYOUT}" >&2
    exit 2
fi
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
export THINKSTREAM_RENDER_LAYOUT="${RENDER_LAYOUT}"
export THINKSTREAM_EVAL_MEMORY_MODE="${MEMORY_MODE}"

if [[ -z "$CKPT" || -z "$BENCHMARK_JSON" || -z "$VIDEO_ROOT" ]]; then
    echo "ERROR: --ckpt, --benchmark_json, --video_root required" >&2; exit 1
fi
if [[ ! -f "$BENCHMARK_JSON" ]]; then
    echo "ERROR: $BENCHMARK_JSON not found" >&2; exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${CKPT}/eval/ovo_full"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/ovo_full"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/sft_${RETRIEVER}_compress-${COMPRESS_MODE}_memory-${MEMORY_MODE}_${FRAME_PROTOCOL}_${RENDER_LAYOUT}.json"

echo "============================================================"
echo "OVO full eval — SFT (compress=system, all 12 sub-tasks)"
echo "  ckpt:       ${CKPT}"
echo "  benchmark:  ${BENCHMARK_JSON}"
echo "  videos:     ${VIDEO_ROOT}"
[ -n "$FRAMES_ROOT" ] && echo "  frames:     ${FRAMES_ROOT}"
echo "  retriever:  ${RETRIEVER}$([ "$RETRIEVER" = "hybrid" ] && echo " (alpha=${ALPHA})")"
echo "  compress:   ${COMPRESS_MODE}"
echo "  memory:     ${MEMORY_MODE}"
echo "  profile:    ${PROFILE}"
echo "  scoring:    ${SCORING}"
echo "  protocol:   ${FRAME_PROTOCOL}"
echo "  layout:     ${RENDER_LAYOUT}"
echo "  engine:     ${ENGINE}"
[[ "${ENGINE}" == "vllm" ]] && echo "  batch:      ${ROLLOUT_BATCH_SIZE}"
[ -n "$TASKS" ] && echo "  tasks:      ${TASKS}"
[ -n "$N_PER_TASK" ] && echo "  n_per_task: ${N_PER_TASK}"
echo "  out:        ${OUT_JSON}"
echo "============================================================"

EXTRA=()
[ -n "$FRAMES_ROOT" ] && EXTRA+=("--frames_root" "$FRAMES_ROOT")
[ -n "$TASKS" ] && EXTRA+=("--tasks" "$TASKS")
[ -n "$N_PER_TASK" ] && EXTRA+=("--n_per_task" "$N_PER_TASK")
[[ "$SAVE_STEP_TRACE" == "1" ]] && EXTRA+=("--save_step_trace")
[[ -n "$TENSOR_PARALLEL_SIZE" ]] && EXTRA+=("--tensor_parallel_size" "$TENSOR_PARALLEL_SIZE")
[[ -n "$VLLM_MAX_MODEL_LEN" ]] && EXTRA+=("--vllm_max_model_len" "$VLLM_MAX_MODEL_LEN")
[[ -n "$VLLM_MM_PROCESSOR_CACHE_GB" ]] && EXTRA+=("--vllm_mm_processor_cache_gb" "$VLLM_MM_PROCESSOR_CACHE_GB")

python scripts/eval/ovo/eval_full.py \
    --ckpt "${CKPT}" \
    --benchmark_json "${BENCHMARK_JSON}" \
    --video_root "${VIDEO_ROOT}" \
    --retriever "${RETRIEVER}" \
    --alpha "${ALPHA}" \
    --siglip_path "${SIGLIP_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --profile "${PROFILE}" \
    --scoring "${SCORING}" \
    --frame-protocol "${FRAME_PROTOCOL}" \
    --render-layout "${RENDER_LAYOUT}" \
    --engine "${ENGINE}" \
    --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm_repetition_penalty "${VLLM_REPETITION_PENALTY}" \
    --compress_mode "${COMPRESS_MODE}" \
    --memory_mode "${MEMORY_MODE}" \
    --out "${OUT_JSON}" \
    "${EXTRA[@]}"

echo ""
echo "Done. Per-task / per-category report printed above."
echo "Full per-probe records: jq '.summary' ${OUT_JSON}"
