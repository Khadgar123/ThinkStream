#!/bin/bash
#
# Test set eval — BASE Qwen3-VL-Instruct (no SFT).
#
# WHY this is asymmetric (only generative, no teacher-forced):
#   Our test.jsonl uses the per-timestep agent protocol with special
#   tokens (<action>, <silent>, <query>, ...) that ONLY exist in the
#   tokenizer after register_special_tokens during SFT. A base ckpt
#   has not been trained on these tokens, so teacher-forced loss is
#   not informative (random embeddings for the new tokens). We run
#   generative-only and report how often base output happens to match
#   the gold action keyword (typically near 0% — that's the baseline).
#
# Result: a floor showing "what does the agent format add over base?"
#
# Usage:
#   bash scripts/eval/test_set/run_base.sh \
#     [--ckpt Qwen/Qwen3-VL-8B-Instruct] [--n 200]

set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
N_GEN=${N_GEN:-200}
TEST_JSONL=${TEST_JSONL:-}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT="$2"; shift 2 ;;
        --n)           N_GEN="$2"; shift 2 ;;
        --test_jsonl)  TEST_JSONL="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
TEST_JSONL=${TEST_JSONL:-${ROOT}/data/agent_v5/final/test.jsonl}

if [[ ! -f "$TEST_JSONL" ]]; then
    echo "ERROR: test jsonl not found: $TEST_JSONL" >&2; exit 1
fi

OUT_DIR="${CKPT}/eval/test_set_base"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/test_set_base"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Test set — BASE Qwen3-VL (generative only)"
echo "  ckpt:  ${CKPT}"
echo "  test:  ${TEST_JSONL}"
echo "  N:     ${N_GEN}"
echo "  out:   ${OUT_DIR}"
echo "=========================================="

python scripts/eval/sft_action_acc.py \
    --ckpt "${CKPT}" \
    --val "${TEST_JSONL}" \
    --n "${N_GEN}" \
    --out "${OUT_DIR}/gen_action.json"

echo ""
echo "Done. base ckpt action_accuracy is the floor — anything"
echo "significantly above this is what SFT contributed."
