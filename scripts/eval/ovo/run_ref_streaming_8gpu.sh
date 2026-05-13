#!/bin/bash
# Reference-style OVO streaming eval, sharded across independent GPUs.
set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
BENCHMARK_DIR=${BENCHMARK_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
MODEL_TYPE=${MODEL_TYPE:-qwen3vl}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUT_DIR=${OUT_DIR:-output/ovo_ref_streaming_8gpu}
PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}

FRAMES_PER_CHUNK=${FRAMES_PER_CHUNK:-2}
REMAINING_SECONDS=${REMAINING_SECONDS:-120}
MAX_LEN=${MAX_LEN:-24576}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30}
THINK_BUDGET=${THINK_BUDGET:-20}
SLACK_TIME=${SLACK_TIME:-3.0}
MIN_PIXELS=${MIN_PIXELS:-$((100352*2))}
MAX_PIXELS=${MAX_PIXELS:-$((100352*4))}
NUM_WORKERS=${NUM_WORKERS:-4}
SAMPLE=${SAMPLE:-}

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"
mkdir -p "${OUT_DIR}/logs"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}

echo "============================================================"
echo "OVO ref-style streaming eval sharded"
echo "  ckpt:          ${CKPT}"
echo "  benchmark_dir: ${BENCHMARK_DIR}"
echo "  gpus:          ${GPUS}"
echo "  shards:        ${NUM_SHARDS}"
echo "  pixels:        ${MIN_PIXELS}-${MAX_PIXELS}"
echo "  chunks/maxlen: ${FRAMES_PER_CHUNK} / ${REMAINING_SECONDS} / ${MAX_LEN}"
echo "  out_dir:       ${OUT_DIR}"
echo "============================================================"

PIDS=()
SHARDS=()
for IDX in "${!GPU_LIST[@]}"; do
    GPU="${GPU_LIST[$IDX]}"
    OUT="${OUT_DIR}/shard_${IDX}_of_${NUM_SHARDS}.json"
    LOG="${OUT_DIR}/logs/shard_${IDX}_gpu${GPU}.log"
    SHARDS+=("${OUT}")

    EXTRA=()
    if [[ -n "${SAMPLE}" ]]; then
        EXTRA+=("--sample" "${SAMPLE}")
    fi

    (
        CUDA_VISIBLE_DEVICES="${GPU}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
        TOKENIZERS_PARALLELISM=false \
        "${PYTHON_BIN}" scripts/eval/ovo/ref_streaming.py \
            --benchmark_dir "${BENCHMARK_DIR}" \
            --model_path "${CKPT}" \
            --model_type "${MODEL_TYPE}" \
            --out "${OUT}" \
            --num_shards "${NUM_SHARDS}" \
            --shard_index "${IDX}" \
            --frames_per_chunk "${FRAMES_PER_CHUNK}" \
            --remaining_seconds "${REMAINING_SECONDS}" \
            --max_len "${MAX_LEN}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --think_budget "${THINK_BUDGET}" \
            --slack_time "${SLACK_TIME}" \
            --min_pixels "${MIN_PIXELS}" \
            --max_pixels "${MAX_PIXELS}" \
            --num_workers "${NUM_WORKERS}" \
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

"${PYTHON_BIN}" scripts/eval/ovo/ref_streaming.py \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --model_path "${CKPT}" \
    --model_type "${MODEL_TYPE}" \
    --out "${OUT_DIR}/merged_ref_streaming.json" \
    --merge "${SHARDS[@]}"
