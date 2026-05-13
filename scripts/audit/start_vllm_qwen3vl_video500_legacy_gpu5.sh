#!/usr/bin/env bash
set -euo pipefail

export GPUS="${GPUS:-5}"
export MODEL_NAME="${MODEL_NAME:-qwen3vl2b-video500-fps1-legacy}"
export OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/vllm_qwen3vl_video500_legacy_gpu5}"
export MEDIA_IO_KWARGS='{"video":{"fps":1,"max_duration":500}}'
export MM_PROCESSOR_KWARGS='{"min_pixels":130000,"max_pixels":220000}'

bash scripts/audit/start_vllm8_qwen3vl_video500.sh
