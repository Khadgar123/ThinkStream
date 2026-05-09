#!/bin/bash
# 8-process vLLM full-video OVO eval for SFT checkpoints.
# Each process owns one GPU and one TP1 vLLM engine; tasks are split by
# estimated full-video chunk workload to reduce wall-clock tail latency.

set -euo pipefail

CKPT=${CKPT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output/agent-sft-v1259-timeline-video-imagepad-timeonly-bs2-fix2-20260508_1725}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
OUT_BASE=${OUT_BASE:-output/ovo_sft_decoder_tp1x8_vllm}
LOG_DIR=${LOG_DIR:-${OUT_BASE}_logs}

N_PER_TASK=${N_PER_TASK:-20}
RETRIEVER=${RETRIEVER:-bm25}
COMPRESS_MODE=${COMPRESS_MODE:-system}
MEMORY_MODE=${MEMORY_MODE:-full}
PROFILE=${PROFILE:-16k}
SCORING=${SCORING:-lenient}
FRAME_PROTOCOL=${FRAME_PROTOCOL:-video_meta}
RENDER_LAYOUT=${RENDER_LAYOUT:-timeline_video_imagepad}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
COMPRESS_MAX_NEW_TOKENS=${COMPRESS_MAX_NEW_TOKENS:-512}
PROGRESS_EVERY=${PROGRESS_EVERY:-3}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --benchmark_json) BENCHMARK_JSON="$2"; shift 2 ;;
        --video_root) VIDEO_ROOT="$2"; shift 2 ;;
        --frames_root) FRAMES_ROOT="$2"; shift 2 ;;
        --out_base) OUT_BASE="$2"; LOG_DIR="${2}_logs"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --n_per_task) N_PER_TASK="$2"; shift 2 ;;
        --scoring) SCORING="$2"; shift 2 ;;
        --rollout_batch_size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
        --vllm_mm_processor_cache_gb) VLLM_MM_PROCESSOR_CACHE_GB="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

PYTHON=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python
SCRIPT=scripts/eval/ovo/eval_full.py

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"
export THINKSTREAM_RENDER_LAYOUT="${RENDER_LAYOUT}"
export THINKSTREAM_EVAL_MEMORY_MODE="${MEMORY_MODE}"

mkdir -p "${OUT_BASE}" "${LOG_DIR}"

# Balanced for n_per_task=20 estimated chunks:
# CRR 13406, STU 12511, ATR 12251, OJR 9226, HLD 8963,
# EPM+REC+SSR 9789, OCR+FPD 9851, ACR+ASI 7539.
TASKS_GPU0="CRR"
TASKS_GPU1="STU"
TASKS_GPU2="ATR"
TASKS_GPU3="OJR"
TASKS_GPU4="HLD"
TASKS_GPU5="EPM,REC,SSR"
TASKS_GPU6="OCR,FPD"
TASKS_GPU7="ACR,ASI"

echo "OVO SFT full-video eval via 8xTP1 vLLM"
echo "  ckpt:       ${CKPT}"
echo "  out:        ${OUT_BASE}"
echo "  logs:       ${LOG_DIR}"
echo "  n/task:     ${N_PER_TASK}"
echo "  scoring:    ${SCORING}"
echo "  batch:      ${ROLLOUT_BATCH_SIZE}"
echo "  mm cache:   ${VLLM_MM_PROCESSOR_CACHE_GB} GB"

launch() {
    local gpu=$1
    local tasks=$2
    local label="gpu${gpu}_${tasks//,/_}"
    local out="${OUT_BASE}/${label}.json"
    local log="${LOG_DIR}/${label}.log"
    echo "[GPU${gpu}] tasks=${tasks} -> ${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}" \
        --engine vllm \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --ckpt "${CKPT}" \
        --benchmark_json "${BENCHMARK_JSON}" \
        --video_root "${VIDEO_ROOT}" \
        --frames_root "${FRAMES_ROOT}" \
        --retriever "${RETRIEVER}" \
        --compress_mode "${COMPRESS_MODE}" \
        --memory_mode "${MEMORY_MODE}" \
        --profile "${PROFILE}" \
        --scoring "${SCORING}" \
        --frame-protocol "${FRAME_PROTOCOL}" \
        --render-layout "${RENDER_LAYOUT}" \
        --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --compress_max_new_tokens "${COMPRESS_MAX_NEW_TOKENS}" \
        --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
        --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --progress_every "${PROGRESS_EVERY}" \
        --n_per_task "${N_PER_TASK}" \
        --tasks "${tasks}" \
        --out "${out}" \
        > "${log}" 2>&1 &
}

launch 0 "${TASKS_GPU0}"
launch 1 "${TASKS_GPU1}"
launch 2 "${TASKS_GPU2}"
launch 3 "${TASKS_GPU3}"
launch 4 "${TASKS_GPU4}"
launch 5 "${TASKS_GPU5}"
launch 6 "${TASKS_GPU6}"
launch 7 "${TASKS_GPU7}"

echo "All jobs launched."
wait

"${PYTHON}" scripts/eval/ovo/aggregate_parallel.py \
    "${OUT_BASE}"/gpu*.json \
    --out "${OUT_BASE}/merged.json"

echo "Merged report: ${OUT_BASE}/merged.json"
