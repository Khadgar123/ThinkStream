#!/bin/bash
# Parallel OVO streaming eval across 8 GPUs — LOAD BALANCED by total chunk count
# v9.5 — rebalanced based on per-task estimated chunk workload

set -euo pipefail

CKPT=${1:-output/agent-sft/checkpoint-100}
BENCHMARK_JSON=${2:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${3:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${4:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}

PYTHON=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python
SCRIPT=scripts/eval/ovo/eval_full.py

# Task assignments — balanced by total estimated chunk count (~24.8k per GPU)
# GPU0: EPM     (37,367 chunk, 297 samples) — heaviest single task
# GPU1: OJR     (29,511 chunk, 184 samples)
# GPU2: STU     (28,037 chunk, 178 samples)
# GPU3: HLD     (24,610 chunk, 186 samples)
# GPU4: ATR     (18,870 chunk, 116 samples)
# GPU5: OCR     (18,538 chunk, 149 samples)
# GPU6: ASI,CRR,REC,SSR (18,921 chunk, 320 samples) — four light FT+BT tasks
# GPU7: ACR,FPD (20,587 chunk, 210 samples)

TASKS_GPU0="EPM"
TASKS_GPU1="OJR"
TASKS_GPU2="STU"
TASKS_GPU3="HLD"
TASKS_GPU4="ATR"
TASKS_GPU5="OCR"
TASKS_GPU6="ASI,CRR,REC,SSR"
TASKS_GPU7="ACR,FPD"

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
