#!/bin/bash
# Run reference-style OVO streaming eval for Qwen3-VL 2B/4B/8B.
set -euo pipefail

BENCHMARK_DIR=${BENCHMARK_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUT_ROOT=${OUT_ROOT:-output/ovo_ref_streaming_qwen3vl_8gpu}

MODELS=(
  "qwen3vl2b:/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct"
  "qwen3vl4b:/home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct"
  "qwen3vl8b:/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct"
)

for ITEM in "${MODELS[@]}"; do
    NAME="${ITEM%%:*}"
    CKPT_PATH="${ITEM#*:}"
    echo "============================================================"
    echo "Running ${NAME}: ${CKPT_PATH}"
    echo "============================================================"
    CKPT="${CKPT_PATH}" \
    BENCHMARK_DIR="${BENCHMARK_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    GPUS="${GPUS}" \
    OUT_DIR="${OUT_ROOT}/${NAME}" \
    bash scripts/eval/ovo/run_ref_streaming_8gpu.sh
done
