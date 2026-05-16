#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
MODEL_PATH="${MODEL_PATH:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3vl2b-video500}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/vllm8_qwen3vl_video500}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"

# 500s at fps=1 is up to 500 video frames. Qwen3-VL-2B advertises 262k text positions.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-8}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"

# Enable real video_url inputs. Keep images available for older probes/retrieval.
if [[ -z "${LIMIT_MM_PER_PROMPT+x}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":128,"video":8}'
fi

# vLLM's video reader defaults to max_duration=300. Raise it for long OVO videos.
if [[ -z "${MEDIA_IO_KWARGS+x}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"fps":1,"max_duration":500}}'
fi

# Keep frame token budget bounded. 100352 = 128 * 28 * 28.
if [[ -z "${MM_PROCESSOR_KWARGS+x}" ]]; then
  MM_PROCESSOR_KWARGS='{"min_pixels":65536,"max_pixels":100352}'
fi

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "Starting one-GPU Qwen3-VL video500 vLLM servers."
echo "model=$MODEL_PATH name=$MODEL_NAME port_base=$PORT_BASE out=$OUT_ROOT"
echo "gpus=$GPUS"
echo "vllm_bin=$VLLM_BIN"
echo "max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS"
[ -n "$MAX_NUM_BATCHED_TOKENS" ] && echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
echo "limit_mm_per_prompt=$LIMIT_MM_PER_PROMPT"
echo "media_io_kwargs=$MEDIA_IO_KWARGS"
echo "mm_processor_kwargs=$MM_PROCESSOR_KWARGS"
[ -n "$ALLOWED_LOCAL_MEDIA_PATH" ] && echo "allowed_local_media_path=$ALLOWED_LOCAL_MEDIA_PATH"

for gpu in $GPUS; do
  port=$((PORT_BASE + gpu))
  log="$OUT_ROOT/vllm_gpu${gpu}_port${port}.log"
  echo "GPU $gpu -> port $port log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
    EXTRA=()
    if [[ -n "$ALLOWED_LOCAL_MEDIA_PATH" ]]; then
      EXTRA+=(--allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH")
    fi
    if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
      EXTRA+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi
    exec "$VLLM_BIN" serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 1 \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
      --media-io-kwargs "$MEDIA_IO_KWARGS" \
      --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
      --trust-remote-code \
      --enable-prefix-caching \
      --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
      "${EXTRA[@]}" \
      --port "$port"
  ) >"$log" 2>&1 &
done

wait
