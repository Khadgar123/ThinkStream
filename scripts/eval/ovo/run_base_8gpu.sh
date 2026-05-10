#!/bin/bash
# Sharded 8-GPU OVO-Bench base VLM eval.
set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
FORM=${FORM:-streaming}
MAX_FRAMES=${MAX_FRAMES:-24}
FPS=${FPS:-1}
VISUAL_WINDOW_SEC=${VISUAL_WINDOW_SEC:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
SCORING=${SCORING:-lenient}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-video_meta}
PREPROCESS=${PREPROCESS:-auto}
MIN_PIXELS=${MIN_PIXELS:-130000}
MAX_PIXELS=${MAX_PIXELS:-220000}
TOTAL_PIXELS=${TOTAL_PIXELS:-}
N_PER_TASK=${N_PER_TASK:-}
CHUNKED_DIR=${CHUNKED_DIR:-}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUT_DIR=${OUT_DIR:-output/ovo_base_8gpu}
NO_BF16=${NO_BF16:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"
mkdir -p "${OUT_DIR}/logs"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}

echo "============================================================"
echo "OVO base eval sharded"
echo "  ckpt:      ${CKPT}"
echo "  mode:      ${FORM}"
echo "  gpus:      ${GPUS}"
echo "  shards:    ${NUM_SHARDS}"
echo "  frames:    ${MAX_FRAMES}"
echo "  preprocess:${PREPROCESS}"
echo "  fps/window:${FPS} fps / ${VISUAL_WINDOW_SEC}s"
echo "  out_dir:   ${OUT_DIR}"
echo "============================================================"

PIDS=()
SHARDS=()
for IDX in "${!GPU_LIST[@]}"; do
    GPU="${GPU_LIST[$IDX]}"
    OUT="${OUT_DIR}/shard_${IDX}_of_${NUM_SHARDS}.json"
    LOG="${OUT_DIR}/logs/shard_${IDX}_gpu${GPU}.log"
    SHARDS+=("${OUT}")
    EXTRA=()
    if [[ "${NO_BF16}" == "1" ]]; then
        EXTRA+=("--no_bf16")
    fi
    if [[ -n "${N_PER_TASK}" ]]; then
        EXTRA+=("--n_per_task" "${N_PER_TASK}")
    fi
    if [[ -n "${CHUNKED_DIR}" ]]; then
        EXTRA+=("--chunked_dir" "${CHUNKED_DIR}")
    fi
    if [[ -n "${TOTAL_PIXELS}" ]]; then
        EXTRA+=("--total_pixels" "${TOTAL_PIXELS}")
    fi
    (
        CUDA_VISIBLE_DEVICES="${GPU}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
        TOKENIZERS_PARALLELISM=false \
        "${PYTHON_BIN}" scripts/eval/ovo/base.py \
            --ckpt "${CKPT}" \
            --benchmark_json "${BENCHMARK_JSON}" \
            --video_root "${VIDEO_ROOT}" \
            --frames_root "${FRAMES_ROOT}" \
            --mode "${FORM}" \
            --max_frames "${MAX_FRAMES}" \
            --fps "${FPS}" \
            --visual_window_sec "${VISUAL_WINDOW_SEC}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --scoring "${SCORING}" \
            --frame-protocol "${FRAME_PROTOCOL}" \
            --preprocess "${PREPROCESS}" \
            --min_pixels "${MIN_PIXELS}" \
            --max_pixels "${MAX_PIXELS}" \
            --num_shards "${NUM_SHARDS}" \
            --shard_index "${IDX}" \
            --out "${OUT}" \
            "${EXTRA[@]}"
    ) > "${LOG}" 2>&1 &
    PIDS+=("$!")
    echo "launched shard ${IDX}/${NUM_SHARDS} on GPU ${GPU}: ${OUT}"
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

if [[ "${FAIL}" != "0" ]]; then
    echo "One or more shards failed. Check ${OUT_DIR}/logs." >&2
    exit 1
fi

"${PYTHON_BIN}" scripts/eval/ovo/merge_base_shards.py \
    --out "${OUT_DIR}/merged_${FORM}_${MAX_FRAMES}f_fps${FPS}_${SCORING}.json" \
    "${SHARDS[@]}"
