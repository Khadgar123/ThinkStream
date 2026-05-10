#!/bin/bash
# Run OVO window/frame sweep queues across GPUs, then hold all GPUs.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-output/ovo_window_sweep_${STAMP}}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
CHUNKED_DIR=${CHUNKED_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/chunked_videos}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
FPS=${FPS:-1}
SCORING=${SCORING:-lenient}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MIN_PIXELS=${MIN_PIXELS:-130000}
MAX_PIXELS=${MAX_PIXELS:-220000}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-video_meta}
BUSY_MEM_MIB=${BUSY_MEM_MIB:-2000}
BUSY_SLEEP_SEC=${BUSY_SLEEP_SEC:-60}
HOLD_AFTER=${HOLD_AFTER:-1}
HOLD_FRACTION=${HOLD_FRACTION:-0.90}
HOLD_RESERVE_GB=${HOLD_RESERVE_GB:-6}

Q3_8B=${Q3_8B:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
Q25_3B=${Q25_3B:-/home/tione/notebook/gaozhenkun/model/Qwen2.5-VL-3B-Instruct}
Q25_7B=${Q25_7B:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output/models/Qwen2.5-VL-7B-Instruct-clean}

mkdir -p "${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/status.tsv"
MATRIX="${OUT_ROOT}/matrix_summary.tsv"
printf "time\tstate\tjob\tgpu\tdetail\n" > "${STATUS}"
printf "time\tjob\toverall_line\n" > "${MATRIX}"

# tag|ckpt|mode|max_frames|visual_window_sec
JOBS=(
  "q3vl8b_streaming_w8_f16|${Q3_8B}|streaming|16|8"
  "q3vl8b_streaming_w32_f64|${Q3_8B}|streaming|64|32"
  "q3vl8b_streaming_w64_f128|${Q3_8B}|streaming|128|64"
  "q3vl8b_offline_prefix32|${Q3_8B}|offline_prefix|32|16"
  "q3vl8b_offline_prefix256|${Q3_8B}|offline_prefix|256|16"
  "q3vl8b_official_prompt_prefix32|${Q3_8B}|official_prompt_prefix|32|16"
  "q3vl8b_official_prompt_prefix256|${Q3_8B}|official_prompt_prefix|256|16"

  "q25vl7b_clean_streaming_w8_f16|${Q25_7B}|streaming|16|8"
  "q25vl7b_clean_streaming_w16_f32|${Q25_7B}|streaming|32|16"
  "q25vl7b_clean_streaming_w32_f64|${Q25_7B}|streaming|64|32"
  "q25vl7b_clean_streaming_w64_f128|${Q25_7B}|streaming|128|64"
  "q25vl7b_clean_offline_prefix32|${Q25_7B}|offline_prefix|32|16"
  "q25vl7b_clean_offline_prefix64|${Q25_7B}|offline_prefix|64|16"
  "q25vl7b_clean_offline_prefix128|${Q25_7B}|offline_prefix|128|16"
  "q25vl7b_clean_offline_prefix256|${Q25_7B}|offline_prefix|256|16"
  "q25vl7b_clean_official_prompt_prefix32|${Q25_7B}|official_prompt_prefix|32|16"
  "q25vl7b_clean_official_prompt_prefix64|${Q25_7B}|official_prompt_prefix|64|16"
  "q25vl7b_clean_official_prompt_prefix128|${Q25_7B}|official_prompt_prefix|128|16"
  "q25vl7b_clean_official_prompt_prefix256|${Q25_7B}|official_prompt_prefix|256|16"

  "q25vl3b_streaming_w8_f16|${Q25_3B}|streaming|16|8"
  "q25vl3b_streaming_w32_f64|${Q25_3B}|streaming|64|32"
  "q25vl3b_streaming_w64_f128|${Q25_3B}|streaming|128|64"
  "q25vl3b_offline_prefix32|${Q25_3B}|offline_prefix|32|16"
  "q25vl3b_offline_prefix128|${Q25_3B}|offline_prefix|128|16"
  "q25vl3b_offline_prefix256|${Q25_3B}|offline_prefix|256|16"
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
  IFS='|' read -r tag ckpt mode frames window <<< "${job}"
  local out_dir="${OUT_ROOT}/${tag}"
  local merged="${out_dir}/merged_${mode}_${frames}f_fps${FPS}_${SCORING}.json"
  local log="${OUT_ROOT}/logs/${tag}_gpu${gpu}.log"

  mkdir -p "${out_dir}"
  printf "%s\tSTART\t%s\t%s\t%s/%sf_window%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${mode}" "${frames}" "${window}" >> "${STATUS}"

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
    FRAME_PROTOCOL="${FRAME_PROTOCOL}" \
    MIN_PIXELS="${MIN_PIXELS}" \
    MAX_PIXELS="${MAX_PIXELS}" \
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

echo "OVO window sweep"
echo "  out_root: ${OUT_ROOT}"
echo "  gpus:     ${GPUS}"
echo "  jobs:     ${#JOBS[@]}"
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
echo "All sweep workers finished. Summary: ${MATRIX}"

if [[ "${HOLD_AFTER}" == "1" ]]; then
  printf "%s\tHOLD_START\tall\t%s\tfraction=%s reserve_gb=%s\n" "$(date +%F_%T)" "${GPUS}" "${HOLD_FRACTION}" "${HOLD_RESERVE_GB}" >> "${STATUS}"
  exec "${PYTHON_BIN}" scripts/hold_gpus.py \
    --gpus "${GPUS}" \
    --fraction "${HOLD_FRACTION}" \
    --reserve-gb "${HOLD_RESERVE_GB}"
fi
