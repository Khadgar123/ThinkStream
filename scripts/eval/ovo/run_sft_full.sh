#!/bin/bash
#
# OVO-Bench full eval — SFT ckpt, system-triggered compression.
# Runs ALL 12 sub-tasks (RT/BT/FT) on the ORIGINAL ovo_bench_new.json
# in a single pass, dispatching per task family:
#
#   BT/RT (9 tasks):  single-realtime MCQ (A/B/C/D)
#   REC:              cumulative integer count, probed at multiple times
#   SSR:              per-probe Yes/No about a specific step
#   CRR:              ask-once at ask_time, probe with delay structure
#                     preserved (the formatted file destroys this)
#
# Locks --compress_mode system to match v11 SFT training. For self-pick
# compression (RL ckpt post-GDPO), use run_rl_full.sh.
#
# Reports per-task accuracy, per-category averages (RT / BT / FT),
# and overall (matches OVO paper Table 2 layout). CRR/SSR additionally
# get type=0/type=1 strict + lenient + fp_rate breakdown.
#
# Usage:
#   bash scripts/eval/ovo/run_sft_full.sh \
#     --ckpt output/agent-sft \
#     --benchmark_json /path/to/ovo_bench_new.json \
#     --video_root /path/to/videos \
#     [--retriever hybrid] [--alpha 0.5] [--n_per_task 30]
#     [--tasks CRR,SSR,REC]   # subset for quick iteration

set -euo pipefail

CKPT=${CKPT:-}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
TASKS=${TASKS:-}
N_PER_TASK=${N_PER_TASK:-}
RETRIEVER=${RETRIEVER:-bm25}
ALPHA=${ALPHA:-0.5}
SIGLIP_PATH=${SIGLIP_PATH:-google/siglip-base-patch16-224}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

COMPRESS_MODE=system

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --benchmark_json)  BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root)      VIDEO_ROOT="$2"; shift 2 ;;
        --tasks)           TASKS="$2"; shift 2 ;;
        --n_per_task)      N_PER_TASK="$2"; shift 2 ;;
        --retriever)       RETRIEVER="$2"; shift 2 ;;
        --alpha)           ALPHA="$2"; shift 2 ;;
        --siglip_path)     SIGLIP_PATH="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" || -z "$BENCHMARK_JSON" || -z "$VIDEO_ROOT" ]]; then
    echo "ERROR: --ckpt, --benchmark_json, --video_root required" >&2; exit 1
fi
if [[ ! -f "$BENCHMARK_JSON" ]]; then
    echo "ERROR: $BENCHMARK_JSON not found" >&2; exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${CKPT}/eval/ovo_full"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/ovo_full"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/sft_${RETRIEVER}_compress-system.json"

echo "============================================================"
echo "OVO full eval — SFT (compress=system, all 12 sub-tasks)"
echo "  ckpt:       ${CKPT}"
echo "  benchmark:  ${BENCHMARK_JSON}"
echo "  videos:     ${VIDEO_ROOT}"
echo "  retriever:  ${RETRIEVER}$([ "$RETRIEVER" = "hybrid" ] && echo " (alpha=${ALPHA})")"
echo "  compress:   ${COMPRESS_MODE} (FIFO range; model writes <summary> only)"
[ -n "$TASKS" ] && echo "  tasks:      ${TASKS}"
[ -n "$N_PER_TASK" ] && echo "  n_per_task: ${N_PER_TASK}"
echo "  out:        ${OUT_JSON}"
echo "============================================================"

EXTRA=()
[ -n "$TASKS" ] && EXTRA+=("--tasks" "$TASKS")
[ -n "$N_PER_TASK" ] && EXTRA+=("--n_per_task" "$N_PER_TASK")

python scripts/eval/ovo/eval_full.py \
    --ckpt "${CKPT}" \
    --benchmark_json "${BENCHMARK_JSON}" \
    --video_root "${VIDEO_ROOT}" \
    --retriever "${RETRIEVER}" \
    --alpha "${ALPHA}" \
    --siglip_path "${SIGLIP_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --compress_mode "${COMPRESS_MODE}" \
    --out "${OUT_JSON}" \
    "${EXTRA[@]}"

echo ""
echo "Done. Per-task / per-category report printed above."
echo "Full per-probe records: jq '.summary' ${OUT_JSON}"
