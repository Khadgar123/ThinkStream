#!/bin/bash
# Parallel OVO RT/BT single-step eval across 8 GPUs
# Uses eval_sft_rtbt.py (fixed-context, one forward pass per sample).

set -euo pipefail

CKPT=${1:-output/agent-sft/checkpoint-100}
BENCHMARK_JSON=${2:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${3:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${4:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}

PYTHON=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python
SCRIPT=scripts/eval/ovo/eval_sft_rtbt.py

# Task assignments (sample counts)
# EPM:297  ASI:148  HLD:186  OCR:149  ACR:109  ATR:116
# STU:178  FPD:101  OJR:184
TASKS_GPU0="EPM"
TASKS_GPU1="HLD,ASI"
TASKS_GPU2="OCR,ACR"
TASKS_GPU3="ATR,STU,FPD"
TASKS_GPU4="OJR"
TASKS_GPU5="EPM"      # overflow / duplicate for variance check
TASKS_GPU6="ASI,HLD"  # overflow
TASKS_GPU7="OCR,ACR"  # overflow

COMMON_ARGS=(
    --ckpt "$CKPT"
    --benchmark_json "$BENCHMARK_JSON"
    --video_root "$VIDEO_ROOT"
    --frames_root "$FRAMES_ROOT"
    --max_new_tokens 512
)

OUT_BASE="${CKPT}/eval/ovo_rtbt_parallel"
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

# Only launch overflow GPUs if they exist
if nvidia-smi -L | grep -q "GPU 5"; then
    launch 5 "$TASKS_GPU5"
fi
if nvidia-smi -L | grep -q "GPU 6"; then
    launch 6 "$TASKS_GPU6"
fi
if nvidia-smi -L | grep -q "GPU 7"; then
    launch 7 "$TASKS_GPU7"
fi

echo "All jobs launched. Monitor with: tail -f ${OUT_BASE}/*.log"
wait
echo "All jobs done."

# Aggregate
"$PYTHON" scripts/eval/ovo/aggregate_parallel.py \
    --inputs "${OUT_BASE}"/gpu*.json \
    --out "${OUT_BASE}/merged.json"
