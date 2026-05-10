#!/bin/bash
# Focused OVO input-protocol / resolution sweep.
#
# This is intentionally narrower than run_window_sweep_then_hold.sh: it tests
# whether Qwen3-VL is being hurt by the visual carrier or pixel budget while
# keeping the fast streaming setting fixed.
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
OUT_ROOT=${OUT_ROOT:-output/ovo_input_protocol_sweep_${STAMP}}
PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
CHUNKED_DIR=${CHUNKED_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/chunked_videos}
GPUS=${GPUS:-1,3,4,5,7}
FPS=${FPS:-1}
SCORING=${SCORING:-lenient}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
PREPROCESS=${PREPROCESS:-auto}
BUSY_MEM_MIB=${BUSY_MEM_MIB:-2000}
BUSY_SLEEP_SEC=${BUSY_SLEEP_SEC:-60}

Q3_8B=${Q3_8B:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
Q25_7B=${Q25_7B:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output/models/Qwen2.5-VL-7B-Instruct-clean}

mkdir -p "${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/status.tsv"
MATRIX="${OUT_ROOT}/matrix_summary.tsv"
printf "time\tstate\tjob\tgpu\tdetail\n" > "${STATUS}"
printf "time\tjob\toverall_line\n" > "${MATRIX}"

# tag|ckpt|mode|frames|window|frame_protocol|min_pixels|max_pixels|n_per_task
JOBS=(
  "q3_video_meta_p360_streaming_w8_f16|${Q3_8B}|streaming|16|8|video_meta|130000|360000|"
  "q3_video_meta_p512_streaming_w8_f16|${Q3_8B}|streaming|16|8|video_meta|130000|512000|"
  "q3_ts_image_p220_streaming_w8_f16|${Q3_8B}|streaming|16|8|ts_image|130000|220000|"
  "q3_ts_image_p360_streaming_w8_f16|${Q3_8B}|streaming|16|8|ts_image|130000|360000|"
  "q3_ts_image_p512_streaming_w8_f16|${Q3_8B}|streaming|16|8|ts_image|130000|512000|"
  "q3_video_meta_p360_streaming_w16_f32|${Q3_8B}|streaming|32|16|video_meta|130000|360000|"
  "q25_ts_image_p220_streaming_w8_f16|${Q25_7B}|streaming|16|8|ts_image|130000|220000|"
  "q25_video_meta_p360_streaming_w8_f16|${Q25_7B}|streaming|16|8|video_meta|130000|360000|"
)

gpu_mem_used_mib() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_gpu_free() {
  local gpu="$1"
  local used
  while true; do
    used="$(gpu_mem_used_mib "${gpu}" || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < BUSY_MEM_MIB )); then
      return 0
    fi
    printf "%s\tWAIT_GPU\tgpu%s\t%s\tused_mib=%s\n" "$(date +%F_%T)" "${gpu}" "${gpu}" "${used}" >> "${STATUS}"
    sleep "${BUSY_SLEEP_SEC}"
  done
}

run_one() {
  local job="$1"
  local gpu="$2"
  IFS='|' read -r tag ckpt mode frames window protocol min_pixels max_pixels n_per_task <<< "${job}"
  local out_dir="${OUT_ROOT}/${tag}"
  local merged="${out_dir}/merged_${mode}_${frames}f_fps${FPS}_${SCORING}.json"
  local log="${OUT_ROOT}/logs/${tag}_gpu${gpu}.log"

  mkdir -p "${out_dir}"
  printf "%s\tSTART\t%s\t%s\t%s/%sf_%s_%s-%s_n%s\n" \
    "$(date +%F_%T)" "${tag}" "${gpu}" "${mode}" "${frames}" "${protocol}" "${min_pixels}" "${max_pixels}" "${n_per_task:-all}" >> "${STATUS}"

  if [[ -s "${out_dir}/summary_compact.json" ]]; then
    printf "%s\tSKIP\t%s\t%s\tsummary_exists\n" "$(date +%F_%T)" "${tag}" "${gpu}" >> "${STATUS}"
    return 0
  fi

  wait_gpu_free "${gpu}"

  (
    CKPT="${ckpt}" \
    BENCHMARK_JSON="${BENCHMARK_JSON}" \
    VIDEO_ROOT="${VIDEO_ROOT}" \
    FRAMES_ROOT="${FRAMES_ROOT}" \
    CHUNKED_DIR="${CHUNKED_DIR}" \
    FORM="${mode}" \
    MAX_FRAMES="${frames}" \
    FPS="${FPS}" \
    VISUAL_WINDOW_SEC="${window}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    SCORING="${SCORING}" \
    FRAME_PROTOCOL="${protocol}" \
    PREPROCESS="${PREPROCESS}" \
    MIN_PIXELS="${min_pixels}" \
    MAX_PIXELS="${max_pixels}" \
    N_PER_TASK="${n_per_task}" \
    GPUS="${gpu}" \
    OUT_DIR="${out_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/eval/ovo/run_base_8gpu.sh
  ) > "${log}" 2>&1
  local rc=$?

  if [[ "${rc}" != "0" || ! -s "${merged}" ]]; then
    printf "%s\tFAIL\t%s\t%s\trc=%s merged=%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${rc}" "${merged}" >> "${STATUS}"
    return 0
  fi

  local line
  line=$("${PYTHON_BIN}" scripts/eval/ovo/summarize_base_eval.py "${merged}" --out_dir "${out_dir}" 2>> "${log}")
  rc=$?
  if [[ "${rc}" != "0" ]]; then
    printf "%s\tSUM_FAIL\t%s\t%s\trc=%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${rc}" >> "${STATUS}"
    return 0
  fi
  printf "%s\t%s\t%s\n" "$(date +%F_%T)" "${tag}" "${line}" >> "${MATRIX}"
  printf "%s\tDONE\t%s\t%s\t%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${line}" >> "${STATUS}"
}

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  sleep "$((worker_idx * 5))"
  local i
  for i in "${!JOBS[@]}"; do
    if (( i % NUM_GPUS == worker_idx )); then
      run_one "${JOBS[$i]}" "${gpu}"
    fi
  done
}

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

echo "OVO input protocol sweep"
echo "  out_root: ${OUT_ROOT}"
echo "  gpus:     ${GPUS}"
echo "  jobs:     ${#JOBS[@]}"
echo "  preprocess: ${PREPROCESS}"
echo "  status:   ${STATUS}"

PIDS=()
for worker_idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_idx]}"
  run_worker "${worker_idx}" "${gpu}" > "${OUT_ROOT}/logs/worker_${worker_idx}_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

printf "%s\tALL_DONE\tall\t%s\tjobs=%s\n" "$(date +%F_%T)" "${GPUS}" "${#JOBS[@]}" >> "${STATUS}"
echo "All input-protocol sweep workers finished. Summary: ${MATRIX}"
