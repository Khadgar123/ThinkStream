#!/bin/bash
#
# Test set eval — BASE Qwen3-VL-Instruct, two visual-context modes.
#
# Both modes extract user question + gold from test.jsonl response /
# recall_response samples, run base offline-style (no agent protocol),
# and score Yes/No / integer / letter outputs OVO-style.
#
# --form offline:
#     full video [0, video_end], 64 frames uniformly sampled.
#     Reports the BASE-VLM CEILING under full information — what's
#     the best a strong offline VLM can do on our test distribution.
#
# --form streaming:
#     only the visual_window slice the streaming agent sees at decision
#     time (default 12 chunks × 2 sec = 24 sec, ~24 frames @ 1 fps).
#     This is the APPLES-TO-APPLES baseline against our agent: same
#     visual context, no agent protocol. Use this number to measure
#     "what does the agent protocol add over a naive sliding window?"
#
# Usage:
#   bash scripts/eval/test_set/run_base.sh --form offline
#   bash scripts/eval/test_set/run_base.sh --form streaming
#   bash scripts/eval/test_set/run_base.sh --form streaming --n 500

set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
N=${N:-200}
FORM=${FORM:-streaming}
TEST_JSONL=${TEST_JSONL:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
VWIN=${VWIN:-12}
CHUNK_SEC=${CHUNK_SEC:-2.0}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT="$2"; shift 2 ;;
        --n)           N="$2"; shift 2 ;;
        --form)        FORM="$2"; shift 2 ;;
        --test_jsonl)  TEST_JSONL="$2"; shift 2 ;;
        --video_root)  VIDEO_ROOT="$2"; shift 2 ;;
        --vwin)        VWIN="$2"; shift 2 ;;
        --chunk_sec)   CHUNK_SEC="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

case "$FORM" in
    offline|streaming) ;;
    *) echo "ERROR: --form must be offline | streaming" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
TEST_JSONL=${TEST_JSONL:-${ROOT}/data/agent_v5/final/test.jsonl}

if [[ ! -f "$TEST_JSONL" ]]; then
    echo "ERROR: test jsonl not found: $TEST_JSONL" >&2; exit 1
fi

OUT_DIR="${CKPT}/eval/test_base_${FORM}"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${ROOT}/output/test_base_${FORM}"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/results.json"

echo "=========================================="
echo "Test set — BASE (form=${FORM})"
echo "  ckpt:  ${CKPT}"
echo "  test:  ${TEST_JSONL}"
echo "  N:     ${N}"
if [[ "$FORM" == "streaming" ]]; then
    echo "  vwin:  ${VWIN} chunks × ${CHUNK_SEC}s = $(echo "${VWIN} * ${CHUNK_SEC}" | bc)s"
fi
echo "  out:   ${OUT_JSON}"
echo "=========================================="

EXTRA_ARGS=()
if [[ -n "${VIDEO_ROOT}" ]]; then
    EXTRA_ARGS+=("--video_root" "${VIDEO_ROOT}")
fi

python scripts/eval/test_set_base.py \
    --ckpt "${CKPT}" \
    --test_jsonl "${TEST_JSONL}" \
    --mode "${FORM}" \
    --n "${N}" \
    --visual_window_chunks "${VWIN}" \
    --agent_chunk_sec "${CHUNK_SEC}" \
    --out "${OUT_JSON}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Compare:"
echo "  - 'offline'   = base ceiling (full video buffer)"
echo "  - 'streaming' = base with same context as our agent (vwin=${VWIN})"
echo "  - SFT/RL agent results at \${CKPT}/eval/test_stream_agent_test/"
