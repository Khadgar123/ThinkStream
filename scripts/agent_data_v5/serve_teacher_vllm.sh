#!/bin/bash
# Launch the construction-time teacher vLLM server for ThinkStream.
#
# Important: pass2 sends pre-extracted JPEG frames as one Qwen3-VL video block
# with video_metadata; it does not send raw mp4 files for server-side decode.
# Keep both image and video limits because pass1a uses image_url while pass2
# uses video. The video limit is per prompt, not concurrency; pass2 uses one
# video block per request, while runtime/eval recall can use current-window +
# recalled-frame videos in one prompt.
set -euo pipefail

MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MM_LIMIT="${MM_LIMIT:-{\"image\":64,\"video\":2}}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-512}"
PORT="${PORT:-8000}"

exec vllm serve "${MODEL}" \
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
