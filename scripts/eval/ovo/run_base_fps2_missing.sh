#!/bin/bash
set -euo pipefail

BENCH=${BENCH:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
SCRIPT_PATH=${SCRIPT_PATH:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/scripts/eval/ovo/base.py}
PYTHON=${PYTHON:-/tmp/ovo_eval_env/bin/python}
LOG_DIR=${LOG_DIR:-/tmp/ovo_base_fps2_logs}
FPS=${FPS:-2}
N_PER_TASK=${N_PER_TASK:-20}
SCORING=${SCORING:-lenient}
HOLD_GPUS_AFTER=${HOLD_GPUS_AFTER:-0}
HOLD_SCRIPT=${HOLD_SCRIPT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/scripts/eval/ovo/hold_gpus.py}
HOLD_GPU_FRACTION=${HOLD_GPU_FRACTION:-0.85}
HOLD_LOG=${HOLD_LOG:-${LOG_DIR}/hold_gpus_missing.log}

GPUS_STR=${GPUS:-"0 1 2 3"}
read -r -a GPUS_ARR <<< "$GPUS_STR"

mkdir -p "$LOG_DIR"

JOBS=(
  "Qwen3.5-9B|/home/tione/notebook/gaozhenkun/model/Qwen3.5-9B|offline|256|16|/home/tione/notebook/gaozhenkun/model/Qwen3.5-9B/eval/ovo_base_fps2/offline_f256_fps2_lenient_n20.json|${LOG_DIR}/Qwen3.5-9B_offline_f256.log"
  "Qwen3-VL-2B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct|streaming|32|16|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct/eval/ovo_base_fps2/streaming_w16_f32_fps2_lenient_n20.json|${LOG_DIR}/Qwen3-VL-2B-Instruct_streaming_w16.log"
  "Qwen3-VL-8B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct|offline|64|16|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct/eval/ovo_base_fps2/offline_f64_fps2_lenient_n20.json|${LOG_DIR}/Qwen3-VL-8B-Instruct_offline_f64.log"
  "Qwen3-VL-8B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct|offline|256|16|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct/eval/ovo_base_fps2/offline_f256_fps2_lenient_n20.json|${LOG_DIR}/Qwen3-VL-8B-Instruct_offline_f256.log"
)

run_job() {
  local gpu=$1 job=$2
  IFS='|' read -r name ckpt mode max_frames visual_window out log <<< "$job"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "[$(date '+%F %T')] [GPU ${gpu}] SKIP existing ${out}"
    return 0
  fi

  echo "[$(date '+%F %T')] [GPU ${gpu}] START ${name} ${mode} frames=${max_frames} window=${visual_window}s -> ${out}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT_PATH" \
    --ckpt "$ckpt" \
    --benchmark_json "$BENCH" \
    --video_root "$VIDEO_ROOT" \
    --frames_root "$FRAMES_ROOT" \
    --mode "$mode" \
    --max_frames "$max_frames" \
    --visual_window_sec "$visual_window" \
    --fps "$FPS" \
    --n_per_task "$N_PER_TASK" \
    --scoring "$SCORING" \
    --out "$out" \
    > "$log" 2>&1
  echo "[$(date '+%F %T')] [GPU ${gpu}] DONE ${name} ${mode} frames=${max_frames} window=${visual_window}s"
}

echo "Missing fps2 jobs: ${#JOBS[@]}"
echo "GPUs: ${GPUS_ARR[*]}"

for idx in "${!JOBS[@]}"; do
  gpu="${GPUS_ARR[$((idx % ${#GPUS_ARR[@]}))]}"
  run_job "$gpu" "${JOBS[$idx]}" &
done

wait
echo "[$(date '+%F %T')] Missing fps2 jobs finished."

if [[ "$HOLD_GPUS_AFTER" == "1" ]]; then
  "$PYTHON" "$HOLD_SCRIPT" \
    --gpus "$GPUS_STR" \
    --fraction "$HOLD_GPU_FRACTION" \
    >> "$HOLD_LOG" 2>&1
fi
