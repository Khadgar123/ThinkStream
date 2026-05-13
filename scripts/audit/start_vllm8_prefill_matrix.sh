#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
MODEL_PATH="${MODEL_PATH:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/vllm8_prefill_matrix}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-16}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "Starting 8 one-GPU vLLM servers."
echo "model=$MODEL_PATH name=$MODEL_NAME port_base=$PORT_BASE out=$OUT_ROOT"

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((PORT_BASE + gpu))
  log="$OUT_ROOT/vllm_gpu${gpu}_port${port}.log"
  echo "GPU $gpu -> port $port log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
    exec /usr/local/bin/vllm serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 1 \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --limit-mm-per-prompt '{"image":128,"video":0}' \
      --trust-remote-code \
      --enable-prefix-caching \
      --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
      --port "$port"
  ) >"$log" 2>&1 &
done

wait
