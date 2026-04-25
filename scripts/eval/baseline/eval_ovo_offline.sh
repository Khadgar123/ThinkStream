#!/bin/bash
#
# Offline OVO-Bench evaluation (full video, N frames uniformly sampled).
# Matches "Open-source Offline Models" in paper Table 2.
#
# Usage:
#   bash scripts/eval/baseline/eval_ovo_offline.sh \
#     --ckpt Qwen/Qwen3-VL-8B \
#     --model_type qwen3vl \
#     --ngpu 8
#

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)         CKPT="$2"; shift 2 ;;
        --model_type)   MODEL_TYPE="$2"; shift 2 ;;
        --ngpu)         NGPU="$2"; shift 2 ;;
        --benchmark_dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --max_frames)   MAX_FRAMES="$2"; shift 2 ;;
        --min_pixels)   MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels)   MAX_PIXELS="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        *)              echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

NGPU=${NGPU:-8}
MODEL_TYPE=${MODEL_TYPE:-qwen3vl}
MAX_FRAMES=${MAX_FRAMES:-64}
MIN_PIXELS=${MIN_PIXELS:-$((100352*2))}
MAX_PIXELS=${MAX_PIXELS:-$((100352*4))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30}
BENCHMARK_DIR=${BENCHMARK_DIR:-/your/benchmark/dir}

echo "=== Offline OVO-Bench (${MAX_FRAMES} frames, no streaming) ==="
echo "Model: ${CKPT}"
echo "Type: ${MODEL_TYPE}"
echo "GPUs: ${NGPU}"
echo "Frames: ${MAX_FRAMES}"

TOKENIZERS_PARALLELISM=false \
torchrun \
--nproc_per_node=${NGPU} thinkstream/eval/ovo_bench/eval_ovo_offline.py \
--benchmark_dir "${BENCHMARK_DIR}" \
--model_path "${CKPT}" \
--model_type "${MODEL_TYPE}" \
--min_pixels "${MIN_PIXELS}" \
--max_pixels "${MAX_PIXELS}" \
--max_new_tokens "${MAX_NEW_TOKENS}" \
--max_frames "${MAX_FRAMES}"
