#!/bin/bash
# Run a broad OVO baseline matrix with one independent queue per GPU.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-output/ovo_baseline_matrix_overnight_${STAMP}}
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

mkdir -p "${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/status.tsv"
MATRIX="${OUT_ROOT}/matrix_summary.tsv"
printf "time\tstate\tjob\tgpu\tdetail\n" > "${STATUS}"
printf "time\tjob\toverall_line\n" > "${MATRIX}"

Q3_2B=/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct
Q3_4B=/home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct
Q3_8B=/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct
Q25_3B=/home/tione/notebook/gaozhenkun/model/Qwen2.5-VL-3B-Instruct
Q25_7B=/home/tione/notebook/gaozhenkun/model/Qwen2.5-VL-7B-Instruct

# tag|ckpt|mode|max_frames|visual_window_sec|n_per_task
JOBS=(
  "qwen3vl8b_streaming32|${Q3_8B}|streaming|32|16|"
  "qwen25vl7b_streaming32|${Q25_7B}|streaming|32|16|"
  "qwen3vl4b_streaming32|${Q3_4B}|streaming|32|16|"
  "qwen25vl3b_streaming32|${Q25_3B}|streaming|32|16|"
  "qwen3vl2b_streaming32|${Q3_2B}|streaming|32|16|"

  "qwen3vl8b_offline_prefix64|${Q3_8B}|offline_prefix|64|16|"
  "qwen25vl7b_offline_prefix64|${Q25_7B}|offline_prefix|64|16|"
  "qwen3vl4b_offline_prefix64|${Q3_4B}|offline_prefix|64|16|"
  "qwen25vl3b_offline_prefix64|${Q25_3B}|offline_prefix|64|16|"
  "qwen3vl2b_offline_prefix64|${Q3_2B}|offline_prefix|64|16|"

  "qwen3vl8b_official_prompt_prefix64|${Q3_8B}|official_prompt_prefix|64|16|"
  "qwen25vl7b_official_prompt_prefix64|${Q25_7B}|official_prompt_prefix|64|16|"
  "qwen3vl4b_official_prompt_prefix64|${Q3_4B}|official_prompt_prefix|64|16|"
  "qwen25vl3b_official_prompt_prefix64|${Q25_3B}|official_prompt_prefix|64|16|"
  "qwen3vl2b_official_prompt_prefix64|${Q3_2B}|official_prompt_prefix|64|16|"

  "qwen3vl8b_offline_full64|${Q3_8B}|offline_full|64|16|"
  "qwen25vl7b_offline_full64|${Q25_7B}|offline_full|64|16|"
  "qwen3vl4b_offline_full64|${Q3_4B}|offline_full|64|16|"
  "qwen25vl3b_offline_full64|${Q25_3B}|offline_full|64|16|"
  "qwen3vl2b_offline_full64|${Q3_2B}|offline_full|64|16|"

  "qwen3vl8b_oracle_support64|${Q3_8B}|oracle_support|64|16|"
  "qwen25vl7b_oracle_support64|${Q25_7B}|oracle_support|64|16|"
  "qwen3vl4b_oracle_support64|${Q3_4B}|oracle_support|64|16|"
  "qwen25vl3b_oracle_support64|${Q25_3B}|oracle_support|64|16|"
  "qwen3vl2b_oracle_support64|${Q3_2B}|oracle_support|64|16|"

  "qwen3vl8b_official_prompt_prefix128|${Q3_8B}|official_prompt_prefix|128|16|"
  "qwen25vl7b_official_prompt_prefix128|${Q25_7B}|official_prompt_prefix|128|16|"
  "qwen3vl8b_offline_prefix128|${Q3_8B}|offline_prefix|128|16|"
  "qwen25vl7b_offline_prefix128|${Q25_7B}|offline_prefix|128|16|"
  "qwen3vl8b_offline_full128|${Q3_8B}|offline_full|128|16|"
  "qwen25vl7b_offline_full128|${Q25_7B}|offline_full|128|16|"

  "qwen3vl8b_official_offline64_smoke_n5|${Q3_8B}|official_offline|64|16|5"
  "qwen25vl7b_official_offline64_smoke_n5|${Q25_7B}|official_offline|64|16|5"
)

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

run_one() {
  local job="$1"
  local gpu="$2"
  IFS='|' read -r tag ckpt mode frames window n_per_task <<< "${job}"
  local out_dir="${OUT_ROOT}/${tag}"
  local merged="${out_dir}/merged_${mode}_${frames}f_fps${FPS}_${SCORING}.json"
  local log="${OUT_ROOT}/logs/${tag}_gpu${gpu}.log"

  mkdir -p "${out_dir}"
  printf "%s\tSTART\t%s\t%s\t%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${mode}/${frames}" >> "${STATUS}"

  if [[ -s "${out_dir}/summary_compact.json" ]]; then
    printf "%s\tSKIP\t%s\t%s\tsummary_exists\n" "$(date +%F_%T)" "${tag}" "${gpu}" >> "${STATUS}"
    return 0
  fi

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
    N_PER_TASK="${n_per_task}" \
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
  sleep "$((worker_idx * 8))"
  local i
  for i in "${!JOBS[@]}"; do
    if (( i % NUM_GPUS == worker_idx )); then
      run_one "${JOBS[$i]}" "${gpu}"
    fi
  done
}

echo "OVO baseline matrix"
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
echo "All workers finished. Summary: ${MATRIX}"
