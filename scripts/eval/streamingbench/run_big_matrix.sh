#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-output/streamingbench_big_matrix_${STAMP}}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT_BASE=${PORT_BASE:-18100}
MODEL_NAME=${MODEL_NAME:-qwen3vl8b-video500-fps2-runtime}
PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-}

mkdir -p "${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/status.tsv"
printf "time\tstate\tjob\tdetail\n" > "${STATUS}"

# tag|visual_source|window_sec|min_pixels|max_pixels|max_frames_per_request|recall_image_frames|recall_video_blocks|recall_block_frames
JOBS=(
  "strict25_45_online_images_w32|frames_image|32|200704|401408|0|8|4|8"
  "strict25_45_recall_images_w32_f8|recall_images|32|200704|401408|0|8|4|8"
  "strict25_45_recall_video_4blocks_w32|recall_video_blocks|32|200704|401408|0|8|4|2"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag visual_source window minp maxp maxf recall_image_frames recall_video_blocks recall_block_frames <<< "${job}"
  out_dir="${OUT_ROOT}/${tag}"
  log="${OUT_ROOT}/logs/${tag}.log"
  printf "%s\tSTART\t%s\tvisual=%s window=%s pixels=%s/%s maxf=%s recall=%s/%sx%s\n" \
    "$(date +%F_%T)" "${tag}" "${visual_source}" "${window}" "${minp}" "${maxp}" "${maxf}" \
    "${recall_image_frames}" "${recall_video_blocks}" "${recall_block_frames}" >> "${STATUS}"
  (
    OUT_DIR="${out_dir}" \
    GPUS="${GPUS}" \
    PORT_BASE="${PORT_BASE}" \
    MODEL_NAME="${MODEL_NAME}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
    VISUAL_SOURCE="${visual_source}" \
    WINDOW_SEC="${window}" \
    MIN_PIXELS="${minp}" \
    MAX_PIXELS="${maxp}" \
    MAX_FRAMES_PER_REQUEST="${maxf}" \
    RECALL_IMAGE_FRAMES="${recall_image_frames}" \
    RECALL_VIDEO_BLOCKS="${recall_video_blocks}" \
    RECALL_BLOCK_FRAMES="${recall_block_frames}" \
    bash scripts/eval/streamingbench/run_base_vllm_8gpu.sh
  ) > "${log}" 2>&1
  rc=$?
  if [[ "${rc}" == "0" ]]; then
    printf "%s\tDONE\t%s\t%s\n" "$(date +%F_%T)" "${tag}" "${out_dir}/summary.json" >> "${STATUS}"
  else
    printf "%s\tFAIL\t%s\trc=%s log=%s\n" "$(date +%F_%T)" "${tag}" "${rc}" "${log}" >> "${STATUS}"
    exit "${rc}"
  fi
done

printf "%s\tALL_DONE\tall\tjobs=%s\n" "$(date +%F_%T)" "${#JOBS[@]}" >> "${STATUS}"
echo "Done: ${OUT_ROOT}"
