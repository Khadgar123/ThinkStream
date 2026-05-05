#!/bin/bash
#
# Test set eval — RL (post-GDPO) ckpt. Identical structure to run_sft.sh.
#
# Why a separate script (vs reusing run_sft.sh with a different ckpt):
#   - Distinct output paths so SFT and RL reports don't clobber each
#     other (results land at <ckpt>/eval/test_<dataset>/...)
#   - Documents intent: this evaluates the RL-shaped policy on the
#     same per-timestep test set, useful for measuring whether RL
#     preserved (or regressed) per-class action accuracy from SFT
#   - test.jsonl is per-timestep and DOES NOT exercise the
#     compress_mode=self autonomous decision (that requires full-video
#     agent_loop runs — see scripts/eval/ovo/run_rl.sh for that)
#
# So this script measures:
#   - Did RL preserve teacher-forced action accuracy from SFT?
#   - Did RL improve or regress per-class action_acc?
#   - Did RL keep silent_eos_rate stable?
# It does NOT measure:
#   - Autonomous compression policy quality (use OVO eval for that)
#   - Long-horizon overflow behavior (also OVO/full-video eval)
#
# Usage:
#   bash scripts/eval/test_set/run_rl.sh \
#     --ckpt output/agent-rl \
#     [--ngpu 8] [--n_gen 200]

set -euo pipefail

CKPT=${CKPT:-}
NGPU=${NGPU:-8}
N_GEN=${N_GEN:-200}
DATASET=${DATASET:-stream_agent_test}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-ts_image}}
NO_GEN=0
NO_TF=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT="$2"; shift 2 ;;
        --ngpu)        NGPU="$2"; shift 2 ;;
        --n_gen)       N_GEN="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --frame_protocol|--frame-protocol) FRAME_PROTOCOL="$2"; shift 2 ;;
        --no_gen)      NO_GEN=1; shift 1 ;;
        --no_tf)       NO_TF=1; shift 1 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "ERROR: --ckpt required (path to your RL output dir)." >&2; exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
DATA_ROOT="${THINKSTREAM_DATA_ROOT:-${AGENT_DATA_DIR:-${ROOT}/data/agent_v5}}"
if [[ "${DATA_ROOT}" == */final ]]; then
    DATA_ROOT="$(dirname "${DATA_ROOT}")"
fi
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
if [[ -z "${THINKSTREAM_FINAL_DIR:-}" ]]; then
    if [[ -d "${DATA_ROOT}/rendered/${FRAME_PROTOCOL}" ]]; then
        export THINKSTREAM_FINAL_DIR="${DATA_ROOT}/rendered/${FRAME_PROTOCOL}"
    else
        export THINKSTREAM_FINAL_DIR="${DATA_ROOT}/final"
    fi
fi
TEST_JSONL="${TEST_JSONL:-${THINKSTREAM_FINAL_DIR}/test_messages.jsonl}"
OUT_DIR="${CKPT}/eval/test_${DATASET}_${FRAME_PROTOCOL}"

if [[ "$NO_TF" != "1" ]]; then
    echo "=========================================="
    echo "Pass 1 — teacher-forced (RL ckpt)"
    echo "  ckpt:    ${CKPT}"
    echo "  dataset: ${DATASET}"
    echo "  ngpu:    ${NGPU}"
    echo "  proto:   ${FRAME_PROTOCOL}"
    echo "  final:   ${THINKSTREAM_FINAL_DIR}"
    echo "=========================================="

    TOKENIZERS_PARALLELISM=false \
    torchrun --nproc_per_node=${NGPU} scripts/eval/test_set_eval.py \
        --ckpt "${CKPT}" \
        --dataset "${DATASET}" \
        --output_json "${OUT_DIR}/metrics.json"
fi

if [[ "$NO_GEN" != "1" ]]; then
    echo ""
    echo "=========================================="
    echo "Pass 2 — generative (RL ckpt, ${N_GEN} samples)"
    echo "=========================================="

    python scripts/eval/sft_action_acc.py \
        --ckpt "${CKPT}" \
        --val "${TEST_JSONL}" \
        --frame-protocol "${FRAME_PROTOCOL}" \
        --n "${N_GEN}" \
        --out "${OUT_DIR}/gen_action.json"
fi

echo ""
echo "=========================================="
echo "Done. RL ckpt test reports under: ${OUT_DIR}/"
echo ""
echo "Tip: diff against SFT report at \${SFT_CKPT}/eval/test_${DATASET}/"
echo "to see what RL changed per metric/class."
echo "=========================================="
