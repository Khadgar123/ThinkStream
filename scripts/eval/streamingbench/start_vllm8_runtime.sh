#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct}"
export MODEL_NAME="${MODEL_NAME:-qwen3vl2b-video500-fps2-runtime}"
export PORT_BASE="${PORT_BASE:-18100}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export OUT_ROOT="${OUT_ROOT:-output/streamingbench_vllm8_runtime_servers}"

# StreamingBench runner sends file:// clips from the repo output tree.
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output}"

# vLLM 0.11 OpenAI requests accept video_url. Per-request media_io_kwargs are
# not part of the OpenAI protocol, so keep media settings at server startup.
# The eval runner clips each request to 8s at 2fps; num_frames=-1 preserves all
# clip frames under the default opencv loader.
export MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-{\"video\":{\"num_frames\":-1}}}"
export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":200704,\"max_pixels\":401408}}"
export LIMIT_IMAGE_PER_PROMPT="${LIMIT_IMAGE_PER_PROMPT:-128}"
export LIMIT_VIDEO_PER_PROMPT="${LIMIT_VIDEO_PER_PROMPT:-8}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":${LIMIT_IMAGE_PER_PROMPT},\"video\":${LIMIT_VIDEO_PER_PROMPT}}}"

bash scripts/audit/start_vllm8_qwen3vl_video500.sh
