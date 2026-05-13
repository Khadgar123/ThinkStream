#!/bin/bash
#
# OVO-Bench RL checkpoint eval through the production true-KV recurrent
# AgentLoop. The old standalone OVO HF/vLLM runner is no longer a current
# method test path.

set -euo pipefail

CKPT=${CKPT:-}
BENCHMARK_JSON=${BENCHMARK_JSON:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
FRAMES_ROOT=${FRAMES_ROOT:-${THINKSTREAM_FRAMES_ROOT:-}}
TASKS=${TASKS:-}
SCORING=${SCORING:-strict}
OUT_DIR=${OUT_DIR:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --benchmark_json|--benchmark-json) BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root|--video-root) VIDEO_ROOT="$2"; shift 2 ;;
        --frames_root|--frames-root) FRAMES_ROOT="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --scoring) SCORING="$2"; shift 2 ;;
        --out_dir|--out-dir) OUT_DIR="$2"; shift 2 ;;
        # Accepted for CLI compatibility; recurrent RL-path eval ignores them.
        --retriever|--alpha|--siglip_path|--siglip-path|--max_new_tokens|--max-new-tokens|--profile|--n_per_task|--n-per-task|--memory_position|--memory-position|--frame_protocol|--frame-protocol|--render_layout|--render-layout)
            shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${FRAMES_ROOT}" && -n "${VIDEO_ROOT}" && -d "${VIDEO_ROOT}/frames" ]]; then
    FRAMES_ROOT="${VIDEO_ROOT}/frames"
fi
if [[ -z "${OUT_DIR}" && -n "${CKPT}" ]]; then
    OUT_DIR="${CKPT}/eval/ovo_rl_recurrent"
fi

ARGS=(--ckpt "${CKPT}" --benchmark_json "${BENCHMARK_JSON}" --frames_root "${FRAMES_ROOT}" --scoring "${SCORING}")
if [[ -n "${TASKS}" ]]; then
    ARGS+=(--tasks "${TASKS}")
fi
if [[ -n "${OUT_DIR}" ]]; then
    ARGS+=(--out_dir "${OUT_DIR}")
fi

echo "OVO RL eval uses the true-KV recurrent RL AgentLoop."
bash scripts/eval/ovo/run_rl_recurrent_eval.sh "${ARGS[@]}"
