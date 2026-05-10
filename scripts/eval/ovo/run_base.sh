#!/bin/bash
# OVO-Bench full eval — base VLM without ThinkStream memory/recall/compress.
#
# Uses the original ovo_bench_new.json, matching run_sft_full.sh/run_rl_full.sh.
# FORM=offline_prefix samples uniformly from the visible prefix up to each
# question time; FORM=offline_full samples from the whole video as a content
# upper bound; FORM=streaming samples only the recent visual window.

set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}
FORM=${FORM:-offline_prefix}
MAX_FRAMES=${MAX_FRAMES:-64}
FPS=${FPS:-1}
VISUAL_WINDOW_SEC=${VISUAL_WINDOW_SEC:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TASKS=${TASKS:-}
N_PER_TASK=${N_PER_TASK:-}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-video_meta}
MIN_PIXELS=${MIN_PIXELS:-130000}
MAX_PIXELS=${MAX_PIXELS:-220000}
OUT=${OUT:-}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --benchmark_json) BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root) VIDEO_ROOT="$2"; shift 2 ;;
        --frames_root) FRAMES_ROOT="$2"; shift 2 ;;
        --form|--mode) FORM="$2"; shift 2 ;;
        --max_frames) MAX_FRAMES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --visual_window_sec) VISUAL_WINDOW_SEC="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --n_per_task) N_PER_TASK="$2"; shift 2 ;;
        --frame_protocol|--frame-protocol) FRAME_PROTOCOL="$2"; shift 2 ;;
        --min_pixels) MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels) MAX_PIXELS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${BENCHMARK_JSON}" || -z "${VIDEO_ROOT}" ]]; then
    echo "ERROR: --benchmark_json and --video_root are required" >&2
    exit 1
fi
if [[ ! -f "${BENCHMARK_JSON}" ]]; then
    echo "ERROR: benchmark json not found: ${BENCHMARK_JSON}" >&2
    exit 1
fi
case "${FORM}" in
    offline|offline_prefix|offline_full|streaming) ;;
    *) echo "ERROR: --form must be offline_prefix, offline_full, or streaming" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${CKPT}/eval/ovo_full"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/ovo_full"
mkdir -p "${OUT_DIR}"
if [[ -z "${OUT}" ]]; then
    OUT="${OUT_DIR}/base_${FORM}_${MAX_FRAMES}f_fps${FPS}_${FRAME_PROTOCOL}.json"
fi

EXTRA=()
[[ -n "${FRAMES_ROOT}" ]] && EXTRA+=("--frames_root" "${FRAMES_ROOT}")
[[ -n "${TASKS}" ]] && EXTRA+=("--tasks" "${TASKS}")
[[ -n "${N_PER_TASK}" ]] && EXTRA+=("--n_per_task" "${N_PER_TASK}")

echo "============================================================"
echo "OVO full eval — base VLM (${FORM}, ${MAX_FRAMES} frames)"
echo "  ckpt:      ${CKPT}"
echo "  benchmark: ${BENCHMARK_JSON}"
echo "  videos:    ${VIDEO_ROOT}"
[[ -n "${FRAMES_ROOT}" ]] && echo "  frames:    ${FRAMES_ROOT}"
echo "  protocol:  ${FRAME_PROTOCOL}"
echo "  fps/window:${FPS} fps / ${VISUAL_WINDOW_SEC}s"
echo "  pixels:    ${MIN_PIXELS}-${MAX_PIXELS}"
echo "  out:       ${OUT}"
echo "============================================================"

python scripts/eval/ovo/base.py \
    --ckpt "${CKPT}" \
    --benchmark_json "${BENCHMARK_JSON}" \
    --video_root "${VIDEO_ROOT}" \
    --mode "${FORM}" \
    --max_frames "${MAX_FRAMES}" \
    --fps "${FPS}" \
    --visual_window_sec "${VISUAL_WINDOW_SEC}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --frame-protocol "${FRAME_PROTOCOL}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --out "${OUT}" \
    "${EXTRA[@]}"
