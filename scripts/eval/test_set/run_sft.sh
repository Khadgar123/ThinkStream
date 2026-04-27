#!/bin/bash
#
# Test set eval — SFT ckpt. Two passes:
#
# Pass 1 (teacher-forced, multi-GPU): runs trainer.evaluate() on
#   test.jsonl. Reports the SAME metrics that wandb showed during
#   training (eval_loss, eval/action_accuracy + per class,
#   eval/silent_eos_rate). Same loss formula as training so the test
#   number is directly comparable to the val curve.
#
# Pass 2 (generative, single-GPU): runs real model.generate on
#   N_GEN samples, parses <action>...</action>, reports inference-time
#   accuracy. Lower bound on action accuracy compared to teacher-
#   forced — typical gap is 5-10pp.
#
# Usage:
#   bash scripts/eval/test_set/run_sft.sh \
#     --ckpt output/agent-sft \
#     [--ngpu 8] [--n_gen 200]

set -euo pipefail

CKPT=${CKPT:-}
NGPU=${NGPU:-8}
N_GEN=${N_GEN:-200}
DATASET=${DATASET:-stream_agent_test}
NO_GEN=0
NO_TF=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT="$2"; shift 2 ;;
        --ngpu)        NGPU="$2"; shift 2 ;;
        --n_gen)       N_GEN="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --no_gen)      NO_GEN=1; shift 1 ;;
        --no_tf)       NO_TF=1; shift 1 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "ERROR: --ckpt required (path to your SFT output dir)." >&2; exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
TEST_JSONL="${ROOT}/data/agent_v5/final/test.jsonl"
OUT_DIR="${CKPT}/eval/test_${DATASET}"

if [[ "$NO_TF" != "1" ]]; then
    echo "=========================================="
    echo "Pass 1 — teacher-forced (loss + L2 argmax)"
    echo "  ckpt:    ${CKPT}"
    echo "  dataset: ${DATASET}"
    echo "  ngpu:    ${NGPU}"
    echo "=========================================="

    TOKENIZERS_PARALLELISM=false \
    torchrun --nproc_per_node=${NGPU} scripts/eval/test_set_eval.py \
        --ckpt "${CKPT}" \
        --dataset "${DATASET}"
fi

if [[ "$NO_GEN" != "1" ]]; then
    echo ""
    echo "=========================================="
    echo "Pass 2 — generative (${N_GEN} samples, greedy)"
    echo "=========================================="

    python scripts/eval/sft_action_acc.py \
        --ckpt "${CKPT}" \
        --val "${TEST_JSONL}" \
        --n "${N_GEN}" \
        --out "${OUT_DIR}/gen_action.json"
fi

echo ""
echo "=========================================="
echo "Done. Reports under: ${OUT_DIR}/"
echo "  metrics.json     — pass 1 (teacher-forced)"
echo "  gen_action.json  — pass 2 (generative)"
echo "=========================================="
