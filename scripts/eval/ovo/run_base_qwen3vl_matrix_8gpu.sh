#!/bin/bash
# Run OVO-Bench base VLM eval for local Qwen3-VL 2B/4B/8B checkpoints.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
OUT_ROOT=${OUT_ROOT:-output/ovo_base_qwen3vl_8gpu}

COMMON_ENV=(
  BENCHMARK_JSON=/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json
  VIDEO_ROOT=/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench
  FRAMES_ROOT=/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames
  FORM=streaming
  MAX_FRAMES=24
  FPS=1
  VISUAL_WINDOW_SEC=16
  MAX_NEW_TOKENS=64
  SCORING=lenient
  FRAME_PROTOCOL=video_meta
  PREPROCESS=auto
  GPUS=0,1,2,3,4,5,6,7
  PYTHON_BIN="${PYTHON_BIN}"
)

run_one() {
  local name="$1"
  local ckpt="$2"
  echo "============================================================"
  echo "Running ${name}: ${ckpt}"
  echo "============================================================"
  env "${COMMON_ENV[@]}" \
    CKPT="${ckpt}" \
    OUT_DIR="${OUT_ROOT}/${name}" \
    bash scripts/eval/ovo/run_base_8gpu.sh
}

run_one qwen3vl2b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct
run_one qwen3vl4b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct
run_one qwen3vl8b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct
