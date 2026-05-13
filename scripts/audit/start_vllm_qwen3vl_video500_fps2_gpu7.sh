#!/usr/bin/env bash
set -euo pipefail

export GPUS="${GPUS:-7}"
export MODEL_NAME="${MODEL_NAME:-qwen3vl2b-video500-fps2}"
export OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/vllm_qwen3vl_video500_fps2_gpu7}"
export MEDIA_IO_KWARGS='{"video":{"fps":2,"max_duration":500}}'
export MM_PROCESSOR_KWARGS='{"min_pixels":65536,"max_pixels":100352}'

bash scripts/audit/start_vllm8_qwen3vl_video500.sh
