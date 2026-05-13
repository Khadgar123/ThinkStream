#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tione/notebook/gaozhenkun/hzh/ThinkStream"
PY="/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python"
BASE_URL="http://127.0.0.1:18080/v1"
MODEL="qwen3vl2b-60chunk-probe"
PREFILL_SUMMARY="output/vllm_latest_frame_probe/noresp_stream_0_59/summary.json"
OUT_ROOT="output/vllm_latest_frame_probe/prefill_time_range_matrix"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

run_case() {
  local name="$1"
  local text_mode="$2"
  local text_count="$3"
  local visual_mode="$4"
  local visual_count="$5"
  local summary_count="$6"
  local layout="$7"
  local query="$8"

  local out="$OUT_ROOT/$name/$query"
  mkdir -p "$out"
  echo "=== time_range $name query=$query ==="
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_time_range_qa \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out-dir "$out" \
    --start-time 60 \
    --end-time 60 \
    --prefill-start 0 \
    --prefill-end 59 \
    --prefill-summary "$PREFILL_SUMMARY" \
    --time-query "$query" \
    --frames-per-turn 1 \
    --prefill-text-mode "$text_mode" \
    --prefill-text-count "$text_count" \
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-layout "$layout" \
    --temperature 0 \
    --max-tokens 120 \
    2>&1 | tee -a "$out/run.log"
}

configs=(
  "summary_only none 0 none 0 8 text_then_visual"
  "text_recent16 recent 16 none 0 8 text_then_visual"
  "text_uniform8 uniform 8 none 0 8 text_then_visual"
  "text_uniform12_visual_uniform8 uniform 12 uniform 8 8 text_then_visual"
  "text_recent16_visual_recent8 recent 16 recent 8 8 text_then_visual"
  "text_all_visual_none all -1 none 0 12 text_then_visual"
  "visual_all_text_none none 0 all -1 8 visual_only"
)

queries=(electric_mixer flour_measuring holding_bananas)

for cfg in "${configs[@]}"; do
  read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
  for query in "${queries[@]}"; do
    run_case "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout" "$query"
  done
done
