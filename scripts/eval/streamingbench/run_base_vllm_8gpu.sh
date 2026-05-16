#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench}
CSV_DIR=${CSV_DIR:-${DATA_ROOT}/StreamingBench}
VIDEO_ROOT=${VIDEO_ROOT:-${DATA_ROOT}/extracted}
FRAMES_ROOT=${FRAMES_ROOT:-${DATA_ROOT}/frames_fps2}
OUT_DIR=${OUT_DIR:-output/streamingbench_base_vllm_runtime}
CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-${OUT_DIR}/clips}

PORT_BASE=${PORT_BASE:-18100}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_NAME=${MODEL_NAME:-qwen3vl2b-video500-fps2-runtime}

# Current requested runtime profile:
#   type/video carrier: vLLM OpenAI HTTP uses video_url; local HF uses type=video.
#   chunk: 1s, 2 frames/chunk; KV/visual window: 32 chunks; pixels: runtime.
FPS=${FPS:-2}
WINDOW_SEC=${WINDOW_SEC:-32}
MIN_PIXELS=${MIN_PIXELS:-200704}
MAX_PIXELS=${MAX_PIXELS:-401408}

CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-8}
SAMPLE_PER_CSV=${SAMPLE_PER_CSV:-}
LIMIT=${LIMIT:-}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-}
MAX_FRAMES_PER_REQUEST=${MAX_FRAMES_PER_REQUEST:-0}
RECALL_IMAGE_FRAMES=${RECALL_IMAGE_FRAMES:-8}
RECALL_VIDEO_BLOCKS=${RECALL_VIDEO_BLOCKS:-4}
RECALL_BLOCK_FRAMES=${RECALL_BLOCK_FRAMES:-8}

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
ENDPOINTS=()
MODELS=()
for gpu in "${GPU_LIST[@]}"; do
  ENDPOINTS+=("http://127.0.0.1:$((PORT_BASE + gpu))/v1")
  MODELS+=("${MODEL_NAME}")
done

join_by_comma() {
  local IFS=,
  echo "$*"
}

EXTRA=()
if [[ -n "${SAMPLE_PER_CSV}" ]]; then
  EXTRA+=(--sample-per-csv "${SAMPLE_PER_CSV}")
fi
if [[ -n "${LIMIT}" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ -n "${SPLIT_MANIFEST}" ]]; then
  EXTRA+=(--split-manifest "${SPLIT_MANIFEST}")
fi

mkdir -p "${OUT_DIR}"

echo "StreamingBench base vLLM eval"
echo "  csv_dir:    ${CSV_DIR}"
echo "  video_root: ${VIDEO_ROOT}"
echo "  frames:     ${FRAMES_ROOT}"
echo "  out_dir:    ${OUT_DIR}"
echo "  clips:      ${CLIP_CACHE_DIR}"
echo "  endpoints:  $(join_by_comma "${ENDPOINTS[@]}")"
echo "  model_name: ${MODEL_NAME}"
echo "  window/fps: ${WINDOW_SEC}s / ${FPS}"
echo "  pixels:     ${MIN_PIXELS}/${MAX_PIXELS}"
echo "  max frames: ${MAX_FRAMES_PER_REQUEST}"
echo "  recall:     images=${RECALL_IMAGE_FRAMES} blocks=${RECALL_VIDEO_BLOCKS} block_frames=${RECALL_BLOCK_FRAMES}"
if [[ -n "${SPLIT_MANIFEST}" ]]; then
  echo "  manifest:   ${SPLIT_MANIFEST}"
fi

"${PYTHON_BIN}" scripts/eval/streamingbench/base_vllm.py \
  --csv-dir "${CSV_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --frames-root "${FRAMES_ROOT}" \
  --visual-source "${VISUAL_SOURCE:-frames_image}" \
  --out-dir "${OUT_DIR}" \
  --clip-cache-dir "${CLIP_CACHE_DIR}" \
  --endpoints "$(join_by_comma "${ENDPOINTS[@]}")" \
  --models "$(join_by_comma "${MODELS[@]}")" \
  --window-sec "${WINDOW_SEC}" \
  --fps "${FPS}" \
  --min-pixels "${MIN_PIXELS}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --max-frames-per-request "${MAX_FRAMES_PER_REQUEST}" \
  --recall-image-frames "${RECALL_IMAGE_FRAMES}" \
  --recall-video-blocks "${RECALL_VIDEO_BLOCKS}" \
  --recall-block-frames "${RECALL_BLOCK_FRAMES}" \
  "${EXTRA[@]}"
