#!/bin/bash
set -euo pipefail

# Full-video OVO agent ablations for the SFT/RL-style streaming loop.
# This does not use single-question RL segments. Each OVO sample starts at
# chunk 0 and runs forward to the task-specific probe horizon.

ROOT=${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}
PYTHON=${PYTHON:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
CKPT=${CKPT:-${ROOT}/output/agent-sft-v1259-timeline-video-imagepad-timeonly-bs2-fix2-20260508_1725}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}

TASKS=${TASKS:-}
N_PER_TASK=${N_PER_TASK:-20}
SCORING=${SCORING:-strict}
PROFILE=${PROFILE:-16k}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-video_meta}
RENDER_LAYOUT=${RENDER_LAYOUT:-timeline_video_imagepad}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
SIGLIP_PATH=${SIGLIP_PATH:-google/siglip-base-patch16-224}
USE_AGENT_VISION=${USE_AGENT_VISION:-0}
SAVE_STEP_TRACE=${SAVE_STEP_TRACE:-0}
GPUS_STR=${GPUS:-}
ENGINE=${ENGINE:-vllm}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-}
VLLM_REPETITION_PENALTY=${VLLM_REPETITION_PENALTY:-1.0}

LOG_DIR=${LOG_DIR:-${ROOT}/output/ovo_full_ablation_logs}
OUT_DIR=${OUT_DIR:-${CKPT}/eval/ovo_full_ablation}
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# name|retriever|alpha|compress_mode|memory_mode
VARIANTS_STR=${VARIANTS_STR:-"full_bm25|bm25|0.5|system|full
full_hybrid|hybrid|0.5|system|full
no_recall|none|0.5|system|no_recall
no_compress|bm25|0.5|off|full
no_memory_prompt|bm25|0.5|system|no_prompt
no_text_memory|none|0.5|off|none"}

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
export THINKSTREAM_RENDER_LAYOUT="${RENDER_LAYOUT}"

echo "OVO full-video SFT ablation matrix"
echo "  ckpt:      ${CKPT}"
echo "  benchmark: ${BENCHMARK_JSON}"
echo "  videos:    ${VIDEO_ROOT}"
echo "  frames:    ${FRAMES_ROOT}"
echo "  n/task:    ${N_PER_TASK}"
echo "  tasks:     ${TASKS:-all}"
echo "  protocol:  ${FRAME_PROTOCOL}"
echo "  layout:    ${RENDER_LAYOUT}"
echo "  engine:    ${ENGINE}"
echo "  batch:     ${ROLLOUT_BATCH_SIZE}"
echo "  out:       ${OUT_DIR}"
echo "  logs:      ${LOG_DIR}"
[[ -n "${GPUS_STR}" ]] && echo "  gpus:      ${GPUS_STR}"

readarray -t VARIANTS_ARR <<< "${VARIANTS_STR}"

run_variant() {
  local gpu="$1"
  local entry="$2"
  local name retriever alpha compress_mode memory_mode out log gpu_label
  local -a extra cmd
  IFS='|' read -r name retriever alpha compress_mode memory_mode <<< "${entry}"
  [[ -z "${name}" ]] && return 0
  out="${OUT_DIR}/${name}_${SCORING}_${PROFILE}_n${N_PER_TASK}.json"
  log="${LOG_DIR}/${name}_${SCORING}_${PROFILE}_n${N_PER_TASK}.log"
  if [[ -s "${out}" ]]; then
    echo "SKIP existing ${out}"
    return 0
  fi

  extra=()
  [[ -n "${FRAMES_ROOT}" ]] && extra+=("--frames_root" "${FRAMES_ROOT}")
  [[ -n "${TASKS}" ]] && extra+=("--tasks" "${TASKS}")
  [[ "${USE_AGENT_VISION}" == "1" ]] && extra+=("--use_agent_vision")
  [[ "${SAVE_STEP_TRACE}" == "1" ]] && extra+=("--save_step_trace")
  [[ -n "${TENSOR_PARALLEL_SIZE}" ]] && extra+=("--tensor_parallel_size" "${TENSOR_PARALLEL_SIZE}")
  [[ -n "${VLLM_MAX_MODEL_LEN}" ]] && extra+=("--vllm_max_model_len" "${VLLM_MAX_MODEL_LEN}")
  [[ -n "${VLLM_MM_PROCESSOR_CACHE_GB}" ]] && extra+=("--vllm_mm_processor_cache_gb" "${VLLM_MM_PROCESSOR_CACHE_GB}")

  gpu_label="${gpu:-default}"
  echo "START ${name}: gpu=${gpu_label} retriever=${retriever} compress=${compress_mode} memory=${memory_mode}"
  cmd=(
    "${PYTHON}" scripts/eval/ovo/eval_full.py
    --ckpt "${CKPT}"
    --benchmark_json "${BENCHMARK_JSON}"
    --video_root "${VIDEO_ROOT}"
    --retriever "${retriever}"
    --alpha "${alpha}"
    --siglip_path "${SIGLIP_PATH}"
    --compress_mode "${compress_mode}"
    --memory_mode "${memory_mode}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --profile "${PROFILE}"
    --scoring "${SCORING}"
    --frame-protocol "${FRAME_PROTOCOL}"
    --render-layout "${RENDER_LAYOUT}"
    --engine "${ENGINE}"
    --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --vllm_repetition_penalty "${VLLM_REPETITION_PENALTY}"
    --n_per_task "${N_PER_TASK}"
    --out "${out}"
    "${extra[@]}"
  )
  if [[ -n "${gpu}" ]]; then
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log}" 2>&1
  else
    env PYTHONUNBUFFERED=1 "${cmd[@]}" > "${log}" 2>&1
  fi
  echo "DONE  ${name}: ${out}"
}

if [[ -n "${GPUS_STR}" ]]; then
  read -r -a GPUS_ARR <<< "${GPUS_STR}"
  worker() {
    local gpu="$1"
    local start="$2"
    local stride="$3"
    local i
    for ((i = start; i < ${#VARIANTS_ARR[@]}; i += stride)); do
      run_variant "${gpu}" "${VARIANTS_ARR[$i]}"
    done
  }
  for idx in "${!GPUS_ARR[@]}"; do
    worker "${GPUS_ARR[$idx]}" "${idx}" "${#GPUS_ARR[@]}" &
  done
  wait
else
  for entry in "${VARIANTS_ARR[@]}"; do
    run_variant "" "${entry}"
  done
fi

echo "Done. Compare with:"
echo "  python scripts/eval/ovo/compare_runs.py ${OUT_DIR}/*.json"
