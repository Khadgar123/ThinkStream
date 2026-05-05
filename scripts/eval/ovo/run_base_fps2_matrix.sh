#!/bin/bash
set -euo pipefail

BENCH=${BENCH:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
SCRIPT_PATH=${SCRIPT_PATH:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/scripts/eval/ovo/base.py}
PYTHON=${PYTHON:-/tmp/ovo_eval_env/bin/python}
N_PER_TASK=${N_PER_TASK:-20}
SCORING=${SCORING:-lenient}
FPS=${FPS:-2}
LOG_DIR=${LOG_DIR:-/tmp/ovo_base_fps2_logs}
HOLD_GPUS_AFTER=${HOLD_GPUS_AFTER:-1}
HOLD_SCRIPT=${HOLD_SCRIPT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/scripts/eval/ovo/hold_gpus.py}
HOLD_GPU_FRACTION=${HOLD_GPU_FRACTION:-0.85}
HOLD_LOG=${HOLD_LOG:-${LOG_DIR}/hold_gpus.log}

# Override as: GPUS="0 1 2 3".
GPUS_STR=${GPUS:-"0 1 2 3 4 5 6 7"}
read -r -a GPUS_ARR <<< "$GPUS_STR"

MODELS=(
  "Qwen3.5-4B|/home/tione/notebook/gaozhenkun/model/Qwen3.5-4B"
  "Qwen3.5-9B|/home/tione/notebook/gaozhenkun/model/Qwen3.5-9B"
  "Qwen3-VL-2B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct"
  "Qwen3-VL-4B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct"
  "Qwen3-VL-8B-Instruct|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct"
)

OFFLINE_FRAMES=(64 128 256)
ONLINE_WINDOWS=(8 16 32)

mkdir -p "$LOG_DIR"

JOBS=()
for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r name ckpt <<< "$model_entry"
  out_dir="${ckpt}/eval/ovo_base_fps2"

  for frames in "${OFFLINE_FRAMES[@]}"; do
    out="${out_dir}/offline_f${frames}_fps${FPS}_${SCORING}_n${N_PER_TASK}.json"
    log="${LOG_DIR}/${name}_offline_f${frames}.log"
    JOBS+=("${name}|${ckpt}|offline|${frames}|16|${out}|${log}")
  done

  for window in "${ONLINE_WINDOWS[@]}"; do
    frames=$((window * FPS))
    out="${out_dir}/streaming_w${window}_f${frames}_fps${FPS}_${SCORING}_n${N_PER_TASK}.json"
    log="${LOG_DIR}/${name}_streaming_w${window}.log"
    JOBS+=("${name}|${ckpt}|streaming|${frames}|${window}|${out}|${log}")
  done
done

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

worker() {
  local gpu=$1 start=$2 stride=$3
  local i
  for ((i = start; i < ${#JOBS[@]}; i += stride)); do
    run_job "$gpu" "${JOBS[$i]}"
  done
}

echo "Matrix: ${#MODELS[@]} models x (${#OFFLINE_FRAMES[@]} offline + ${#ONLINE_WINDOWS[@]} online) = ${#JOBS[@]} jobs"
echo "Tasks: all OVO tasks, N_PER_TASK=${N_PER_TASK}; FPS=${FPS}; logs=${LOG_DIR}"
echo "GPUs: ${GPUS_ARR[*]}"

for idx in "${!GPUS_ARR[@]}"; do
  worker "${GPUS_ARR[$idx]}" "$idx" "${#GPUS_ARR[@]}" &
done

wait
echo "[$(date '+%F %T')] All fps2 base eval jobs finished."

if [[ "$HOLD_GPUS_AFTER" == "1" ]]; then
  echo "[$(date '+%F %T')] Starting GPU holder on GPUs: ${GPUS_STR}; log=${HOLD_LOG}"
  "$PYTHON" "$HOLD_SCRIPT" \
    --gpus "$GPUS_STR" \
    --fraction "$HOLD_GPU_FRACTION" \
    >> "$HOLD_LOG" 2>&1
fi
