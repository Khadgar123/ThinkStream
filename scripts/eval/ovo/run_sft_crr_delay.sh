#!/bin/bash
#
# OVO CRR delay-aware eval — SFT ckpt, system-triggered compression.
#
# Locks --compress_mode system to match v11 SFT training (C1 only,
# system inserts <compress_trigger range="X-Y"/> with a FIFO range,
# model only writes the <summary>). DO NOT use --compress_mode self
# with a pure-SFT ckpt — it will likely never compress and overflow
# on long videos. For self-pick mode, use run_rl_crr_delay.sh.
#
# Tests: "agent receives a question, must wait for evidence (clue_time),
# then answer Yes — without prematurely saying Yes before clue."
#
# Reports per-probe-type (type=0 before clue, type=1 after clue):
#   strict_acc  - matches OVO scoring (must explicitly say "No" / "Yes")
#   lenient_acc - silent at type=0 also counted correct (matches our
#                 SFT design intent: "wait for evidence")
#   fp_rate     - type=0 probes where agent said "Yes" (the failure
#                 mode strict eval cares about)
#
# Retriever modes:
#   --retriever bm25    keyword-only over <think> text (v11 baseline)
#   --retriever hybrid  BM25 + dense visual via SigLIP (alpha=0.5 default)
#
# Usage:
#   bash scripts/eval/ovo/run_sft_crr_delay.sh \
#     --ckpt output/agent-sft \
#     --benchmark_json /path/to/ovo_bench_new.json \
#     --video_root /path/to/videos \
#     [--retriever hybrid] [--alpha 0.5] [--n 30]

set -euo pipefail

CKPT=${CKPT:-}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
TASK=${TASK:-CRR}
RETRIEVER=${RETRIEVER:-bm25}
ALPHA=${ALPHA:-0.5}
SIGLIP_PATH=${SIGLIP_PATH:-google/siglip-base-patch16-224}
N=${N:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

# This script always uses system-triggered compression.
COMPRESS_MODE=system

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --benchmark_json)  BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root)      VIDEO_ROOT="$2"; shift 2 ;;
        --task)            TASK="$2"; shift 2 ;;
        --retriever)       RETRIEVER="$2"; shift 2 ;;
        --alpha)           ALPHA="$2"; shift 2 ;;
        --siglip_path)     SIGLIP_PATH="$2"; shift 2 ;;
        --n)               N="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" || -z "$BENCHMARK_JSON" || -z "$VIDEO_ROOT" ]]; then
    echo "ERROR: --ckpt, --benchmark_json, --video_root are required" >&2
    exit 1
fi
if [[ ! -f "$BENCHMARK_JSON" ]]; then
    echo "ERROR: $BENCHMARK_JSON not found" >&2; exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${CKPT}/eval/ovo_delay"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/ovo_delay"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/${TASK}_sft_${RETRIEVER}_compress-system.json"

echo "=========================================="
echo "OVO ${TASK} delay-aware eval — SFT (compress=system)"
echo "  ckpt:       ${CKPT}"
echo "  benchmark:  ${BENCHMARK_JSON}"
echo "  videos:     ${VIDEO_ROOT}"
echo "  retriever:  ${RETRIEVER}$([ "$RETRIEVER" = "hybrid" ] && echo " (alpha=${ALPHA})")"
echo "  compress:   ${COMPRESS_MODE} (system FIFO range; model only writes <summary>)"
[ -n "$N" ] && echo "  n:          ${N}"
echo "  out:        ${OUT_JSON}"
echo "=========================================="

EXTRA=()
[ -n "$N" ] && EXTRA+=("--n" "$N")

python scripts/eval/ovo/eval_delay.py \
    --ckpt "${CKPT}" \
    --benchmark_json "${BENCHMARK_JSON}" \
    --video_root "${VIDEO_ROOT}" \
    --task "${TASK}" \
    --retriever "${RETRIEVER}" \
    --alpha "${ALPHA}" \
    --siglip_path "${SIGLIP_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --compress_mode "${COMPRESS_MODE}" \
    --out "${OUT_JSON}" \
    "${EXTRA[@]}"

echo ""
echo "Done. Read with:"
echo "  jq '.by_bucket' ${OUT_JSON}"
echo ""
echo "Key signals:"
echo "  type0.fp_rate  > 0.20 → agent answers Yes prematurely (silent_eos failing)"
echo "  type0.lenient  > 0.80 → agent waits correctly (silent or No)"
echo "  type1.strict   > 0.70 → agent answers Yes once evidence arrives"
