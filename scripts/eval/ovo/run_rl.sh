#!/bin/bash
#
# OVO-Bench eval — RL (post-GDPO) ckpt with self-decided compression.
#
# Differs from run_sft.sh in ONE place: --compress_mode self. The agent
# loop never inserts a system <compress_trigger>; the model itself must
# decide when to emit <action>compress</action> and what time_range to
# summarize. The reward (overflow_pen) trained during GDPO is what
# shapes that policy.
#
# IMPORTANT: do NOT run this with a pure-SFT ckpt. v11 SFT samples were
# all C1 (system-triggered fixed range), so under --compress_mode self
# a SFT-only model will likely never compress and the memory budget
# will overflow halfway through long videos.
#
# Usage:
#   bash scripts/eval/ovo/run_rl.sh \
#     --ckpt output/agent-rl \
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
    echo "ERROR: --ckpt required (path to your RL/GDPO output dir)." >&2; exit 1
fi
if [[ -z "$BENCHMARK_DIR" || ! -f "$BENCHMARK_DIR/ovo-bench-formatted.jsonl" ]]; then
    echo "ERROR: --benchmark_dir must contain ovo-bench-formatted.jsonl." >&2; exit 1
fi

echo "=== RL (post-GDPO) Agent — OVO-Bench (self-decided compress) ==="
echo "Model:  ${CKPT}"
echo "GPUs:   ${NGPU}"
echo "Bench:  ${BENCHMARK_DIR}"
echo "Pixels: ${MIN_PIXELS}-${MAX_PIXELS}"
echo "================================================================="

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
    --compress_mode self
