#!/bin/bash
#
# OVO-Bench eval — BASE Qwen3-VL-Instruct (no SFT, no agent protocol).
#
# Two sub-modes:
#   FORM=offline   - 64 frames uniformly sampled from full video
#   FORM=streaming - chunk-by-chunk video delivery, no agent protocol
#                    (matches the streaming rate at which our SFT model
#                     would see the video; fairer for streaming-vs-offline
#                     gap analysis)
#
# Usage:
#   bash scripts/eval/ovo/run_base.sh \
#     --benchmark_dir /path/with/ovo-bench-formatted.jsonl \
#     [--ckpt Qwen/Qwen3-VL-8B-Instruct] \
#     [--ngpu 8] [--form offline|streaming]
#
# Result location: ${ckpt}/eval/ovo_bench/<filename>.json with per-task
# accuracy and the three category averages (RT / BT / FT).

set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
NGPU=${NGPU:-8}
MODEL_TYPE=${MODEL_TYPE:-qwen3vl}
FORM=${FORM:-offline}
MIN_PIXELS=${MIN_PIXELS:-$((100352*2))}
MAX_PIXELS=${MAX_PIXELS:-$((100352*4))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30}
MAX_FRAMES=${MAX_FRAMES:-64}
BENCHMARK_DIR=${BENCHMARK_DIR:-}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)           CKPT="$2"; shift 2 ;;
        --ngpu)           NGPU="$2"; shift 2 ;;
        --model_type)     MODEL_TYPE="$2"; shift 2 ;;
        --form)           FORM="$2"; shift 2 ;;
        --benchmark_dir)  BENCHMARK_DIR="$2"; shift 2 ;;
        --min_pixels)     MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels)     MAX_PIXELS="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --max_frames)     MAX_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "ERROR: --benchmark_dir is required (must contain ovo-bench-formatted.jsonl)." >&2
    exit 1
fi
if [[ ! -f "$BENCHMARK_DIR/ovo-bench-formatted.jsonl" ]]; then
    echo "ERROR: $BENCHMARK_DIR/ovo-bench-formatted.jsonl not found." >&2
    exit 1
fi

case "$FORM" in
    offline)
        ENTRY=thinkstream/eval/ovo_bench/eval_ovo_offline.py
        EXTRA_ARGS=("--max_frames" "${MAX_FRAMES}")
        echo "=== Base Qwen3-VL — OVO-Bench (OFFLINE, ${MAX_FRAMES} frames) ==="
        ;;
    streaming)
        ENTRY=thinkstream/eval/ovo_bench/eval_ovo_baseline.py
        EXTRA_ARGS=()
        echo "=== Base Qwen3-VL — OVO-Bench (STREAMING, no agent) ==="
        ;;
    *)
        echo "Unknown FORM=$FORM. Use offline | streaming." >&2; exit 1 ;;
esac

echo "Model: ${CKPT}"
echo "GPUs:  ${NGPU}"
echo "Bench: ${BENCHMARK_DIR}"
echo "Pixels: ${MIN_PIXELS}-${MAX_PIXELS}"
echo "============================================="

TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=${NGPU} ${ENTRY} \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --model_path "${CKPT}" \
    --model_type "${MODEL_TYPE}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    "${EXTRA_ARGS[@]}"
