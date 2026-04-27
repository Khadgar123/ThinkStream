#!/bin/bash
#
# OVO-Bench eval — SFT ckpt with full agent protocol.
#
# Uses StreamingAgentLoop (recall + system-triggered compress, FIFO range).
# Compression range is fixed by the system trigger — the model only writes
# the <summary> text. This matches v11 SFT training (C1 only, no C2);
# range exploration is moved to the RL stage.
#
# Usage:
#   bash scripts/eval/ovo/run_sft.sh \
#     --ckpt output/agent-sft \
#     --benchmark_dir /path/with/ovo-bench-formatted.jsonl \
#     [--ngpu 8]

set -euo pipefail

CKPT=${CKPT:-}
NGPU=${NGPU:-8}
MODEL_TYPE=${MODEL_TYPE:-qwen3vl}
MIN_PIXELS=${MIN_PIXELS:-$((100352*2))}
MAX_PIXELS=${MAX_PIXELS:-$((100352*4))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
THINK_BUDGET=${THINK_BUDGET:-20}
BENCHMARK_DIR=${BENCHMARK_DIR:-}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --ngpu)            NGPU="$2"; shift 2 ;;
        --model_type)      MODEL_TYPE="$2"; shift 2 ;;
        --benchmark_dir)   BENCHMARK_DIR="$2"; shift 2 ;;
        --min_pixels)      MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels)      MAX_PIXELS="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --think_budget)    THINK_BUDGET="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "ERROR: --ckpt required (path to your SFT output dir)." >&2; exit 1
fi
if [[ -z "$BENCHMARK_DIR" || ! -f "$BENCHMARK_DIR/ovo-bench-formatted.jsonl" ]]; then
    echo "ERROR: --benchmark_dir must contain ovo-bench-formatted.jsonl." >&2; exit 1
fi

echo "=== SFT Agent — OVO-Bench (system-triggered compress) ==="
echo "Model:  ${CKPT}"
echo "GPUs:   ${NGPU}"
echo "Bench:  ${BENCHMARK_DIR}"
echo "Pixels: ${MIN_PIXELS}-${MAX_PIXELS}"
echo "==========================================================="

TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=${NGPU} thinkstream/eval/ovo_bench/eval_ovo.py \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --model_path "${CKPT}" \
    --model_type "${MODEL_TYPE}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --think_budget "${THINK_BUDGET}" \
    --use_agent_loop \
    --compress_mode system
