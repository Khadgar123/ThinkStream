#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tione/notebook/gaozhenkun/hzh/ThinkStream"
PY="/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python"
BASE_URL="http://127.0.0.1:18080/v1"
MODEL="qwen3vl2b-60chunk-probe"
PREFILL_SUMMARY="output/vllm_latest_frame_probe/noresp_stream_0_59/summary.json"
OUT_ROOT="output/vllm_latest_frame_probe/prefill_matrix_60_90"

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

  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  echo "=== continue $name ==="
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_continue \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out-dir "$out/continue" \
    --start-time 60 \
    --end-time 90 \
    --prefill-start 0 \
    --prefill-end 59 \
    --prefill-summary "$PREFILL_SUMMARY" \
    --frames-per-turn 1 \
    --prefill-text-mode "$text_mode" \
    --prefill-text-count "$text_count" \
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-layout "$layout" \
    --temperature 0 \
    --max-tokens 120 \
    2>&1 | tee -a "$out/continue.log"

  echo "=== history $name ==="
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_history_qa \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out-dir "$out/history_t50" \
    --start-time 60 \
    --end-time 60 \
    --prefill-start 0 \
    --prefill-end 59 \
    --prefill-summary "$PREFILL_SUMMARY" \
    --history-check-time 50 \
    --frames-per-turn 1 \
    --prefill-text-mode "$text_mode" \
    --prefill-text-count "$text_count" \
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-layout "$layout" \
    --temperature 0 \
    --max-tokens 120 \
    2>&1 | tee -a "$out/history.log"
}

run_case "summary_only" "none" "0" "none" "0" "8" "text_then_visual"
run_case "text_recent4" "recent" "4" "none" "0" "8" "text_then_visual"
run_case "text_recent8" "recent" "8" "none" "0" "8" "text_then_visual"
run_case "text_uniform8" "uniform" "8" "none" "0" "8" "text_then_visual"
run_case "text_uniform_recent12" "uniform_recent" "12" "none" "0" "8" "text_then_visual"
run_case "text_recent8_visual_recent4" "recent" "8" "recent" "4" "8" "text_then_visual"
run_case "text_uniform8_visual_uniform4" "uniform" "8" "uniform" "4" "8" "text_then_visual"
run_case "text_recent16_visual_none" "recent" "16" "none" "0" "8" "text_then_visual"
