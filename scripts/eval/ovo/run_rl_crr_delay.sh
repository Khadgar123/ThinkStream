#!/bin/bash
#
# OVO CRR delay-aware eval — RL (post-GDPO) ckpt, model-self-decided
# compression range.
#
# Locks --compress_mode self: the agent loop never inserts a system
# <compress_trigger>. The model decides autonomously when to emit
# <action>compress</action> AND which time_range to summarize. The
# overflow_pen reward in thinkstream/trainer/grpo.py is what trained
# the policy to make this decision well.
#
# DO NOT use this with a pure-SFT ckpt — v11 SFT samples were all C1
# (system-triggered fixed range), so under self-pick a SFT-only model
# will likely never compress and overflow on long videos. For SFT
# evaluation, use run_sft_crr_delay.sh.
#
# Same scoring as the SFT variant (strict / lenient / fp_rate per
# probe type) so the two reports are directly comparable. Diff against
# the SFT report at <SFT_CKPT>/eval/ovo_delay/CRR_sft_*.json to see
# what RL's compression policy added.
#
# Retriever modes:
#   --retriever bm25    keyword-only over <think> text (v11 baseline)
#   --retriever hybrid  BM25 + dense visual via SigLIP (alpha=0.5 default)
#
# Usage:
#   bash scripts/eval/ovo/run_rl_crr_delay.sh \
#     --ckpt output/agent-rl \
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

# This script always uses self-decided compression.
COMPRESS_MODE=self

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
OUT_JSON="${OUT_DIR}/${TASK}_rl_${RETRIEVER}_compress-self.json"

echo "=========================================="
echo "OVO ${TASK} delay-aware eval — RL (compress=self)"
echo "  ckpt:       ${CKPT}"
echo "  benchmark:  ${BENCHMARK_JSON}"
echo "  videos:     ${VIDEO_ROOT}"
echo "  retriever:  ${RETRIEVER}$([ "$RETRIEVER" = "hybrid" ] && echo " (alpha=${ALPHA})")"
echo "  compress:   ${COMPRESS_MODE} (model decides when AND which range)"
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
echo "What to look for vs SFT report:"
echo "  - Did RL preserve type0.lenient / type1.strict from SFT?"
echo "    (autonomous compress shouldn't hurt timing if reward shaped right)"
echo "  - Does RL avoid memory overflow on long-clue-time samples?"
echo "    (samples with last_probe - ask_time > 30s are the stress test)"
