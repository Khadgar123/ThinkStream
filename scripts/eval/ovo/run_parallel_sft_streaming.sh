#!/bin/bash
# Parallel OVO streaming eval across 8 GPUs
# v9.4.2 — fixed context overflow with SFT-aligned pixel budget.

set -euo pipefail

CKPT=${1:-output/agent-sft/checkpoint-100}
BENCHMARK_JSON=${2:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${3:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${4:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}

PYTHON=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python
SCRIPT=scripts/eval/ovo/eval_full.py

# Task assignments
TASKS_GPU0="EPM"
TASKS_GPU1="HLD,ASI"
TASKS_GPU2="OCR,ACR"
TASKS_GPU3="ATR,STU,FPD"
TASKS_GPU4="OJR"
TASKS_GPU5="REC"
TASKS_GPU6="SSR"
TASKS_GPU7="CRR"

COMMON_ARGS=(
    --ckpt "$CKPT"
    --benchmark_json "$BENCHMARK_JSON"
    --video_root "$VIDEO_ROOT"
    --frames_root "$FRAMES_ROOT"
    --retriever bm25
    --compress_mode system
    --max_new_tokens 128
    --scoring strict
)

OUT_BASE="${CKPT}/eval/ovo_streaming_parallel"
mkdir -p "$OUT_BASE"

launch() {
    local gpu=$1
    local tasks=$2
    local out="${OUT_BASE}/gpu${gpu}_${tasks//,/_}.json"
    echo "[GPU${gpu}] tasks=${tasks} -> ${out}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
        "${COMMON_ARGS[@]}" \
        --tasks "$tasks" \
        --out "$out" \
        > "${out%.json}.log" 2>&1 &
}

launch 0 "$TASKS_GPU0"
launch 1 "$TASKS_GPU1"
launch 2 "$TASKS_GPU2"
launch 3 "$TASKS_GPU3"
launch 4 "$TASKS_GPU4"
launch 5 "$TASKS_GPU5"
launch 6 "$TASKS_GPU6"
launch 7 "$TASKS_GPU7"

echo "All jobs launched. Monitor with: tail -f ${OUT_BASE}/*.log"
wait
echo "All jobs done."
