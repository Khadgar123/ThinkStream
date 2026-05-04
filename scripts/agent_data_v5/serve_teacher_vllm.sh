#!/bin/bash
# Launch the construction-time teacher vLLM server for ThinkStream.
#
# Important: ThinkStream sends pre-extracted JPEG frames as timestamped
# image/image_url lists. It does not send raw mp4 files for server-side
# decode/resampling, and it does not depend on vLLM's pre-sampled video
# metadata path for temporal anchors.
#
# Keep both image and video limits: image=64 covers the 32-frame sliding
# window plus recalled frames; video=2 is retained only for legacy/raw-video
# fallback paths. These limits are per prompt, not concurrency.
set -euo pipefail

MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MM_LIMIT="${MM_LIMIT:-{\"image\":64,\"video\":2}}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-512}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

EXTRA_ARGS=()
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  EXTRA_ARGS+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

exec vllm serve "${MODEL}" \
  "${EXTRA_ARGS[@]}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --limit-mm-per-prompt "${MM_LIMIT}" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}" \
  --port "${PORT}"
