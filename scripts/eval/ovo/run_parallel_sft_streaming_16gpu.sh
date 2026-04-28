#!/bin/bash
# 16-process parallel eval: 2 processes per GPU, each runs 1/16 of the full dataset
# Balanced by chunk workload (~12,280 chunks per process)

set -euo pipefail

CKPT=${1:-output/agent-sft/checkpoint-100}
VIDEO_ROOT=${2:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${3:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}

PYTHON=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python
SCRIPT=scripts/eval/ovo/eval_full.py
SPLITS_DIR=/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/splits_16

COMMON_ARGS=(
    --ckpt "$CKPT"
    --video_root "$VIDEO_ROOT"
    --frames_root "$FRAMES_ROOT"
    --retriever bm25
    --compress_mode system
    --max_new_tokens 128
    --scoring strict
)

OUT_BASE="${CKPT}/eval/ovo_streaming_16gpu"
mkdir -p "$OUT_BASE"

launch() {
    local gpu=$1
    local split_idx=$2
    local split_name=$(printf "split_%02d" "$split_idx")
    local split_json="${SPLITS_DIR}/${split_name}.json"
    local out="${OUT_BASE}/gpu${gpu}_${split_name}.json"
    echo "[GPU${gpu}] ${split_name} -> ${out}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
        "${COMMON_ARGS[@]}" \
        --benchmark_json "$split_json" \
        --out "$out" \
        > "${out%.json}.log" 2>&1 &
}

# Launch 16 processes: 2 per GPU
for gpu in {0..7}; do
    launch $gpu $((gpu * 2))
    launch $gpu $((gpu * 2 + 1))
done

echo "16 jobs launched. Monitor with: tail -f ${OUT_BASE}/*.log"
wait
echo "All jobs done."
