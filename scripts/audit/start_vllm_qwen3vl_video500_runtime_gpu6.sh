#!/usr/bin/env bash
set -euo pipefail

export GPUS="${GPUS:-6}"
export MODEL_NAME="${MODEL_NAME:-qwen3vl2b-video500-fps1-runtime}"
export OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/vllm_qwen3vl_video500_runtime_gpu6}"
export MEDIA_IO_KWARGS='{"video":{"fps":1,"max_duration":500}}'
export MM_PROCESSOR_KWARGS='{"min_pixels":200704,"max_pixels":401408}'

bash scripts/audit/start_vllm8_qwen3vl_video500.sh
