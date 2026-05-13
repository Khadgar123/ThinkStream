#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PY="${PY:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
MODEL="${MODEL:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/prefill_duration_8_16_matrix}"
PREFILL_SUMMARY="${PREFILL_SUMMARY:-output/vllm_latest_frame_probe/noresp_stream_0_59/summary.json}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

ports=()
for gpu in 0 1 2 3 4 5 6 7; do
  ports+=("$((PORT_BASE + gpu))")
done

wait_server() {
  local port="$1"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "ready port=$port"
      return 0
    fi
    sleep 3
  done
  echo "server not ready: port=$port" >&2
  return 1
}

for port in "${ports[@]}"; do
  wait_server "$port" &
done
wait

configs=(
  "summary_only none 0 none 0 4 text_then_visual"
  "text_all all -1 none 0 0 text_then_visual"
  "text_all_summary4 all -1 none 0 4 text_then_visual"
  "text_recent8 recent 8 none 0 0 text_then_visual"
  "visual_all none 0 all -1 0 visual_only"
  "visual_recent8 none 0 recent 8 0 visual_only"
  "text_all_visual_all all -1 all -1 0 text_then_visual"
)

run_continue_case() {
  local port="$1"
  local dur="$2"
  local prefill_end="$3"
  local start_time="$4"
  local end_time="$5"
  local name="$6"
  local text_mode="$7"
  local text_count="$8"
  local visual_mode="$9"
  local visual_count="${10}"
  local summary_count="${11}"
  local layout="${12}"
  local out="$OUT_ROOT/continue/dur${dur}_${name}"
  mkdir -p "$out"
  echo "continue start port=$port dur=$dur config=$name"
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_continue \
    --base-url "http://127.0.0.1:${port}/v1" \
    --model "$MODEL" \
    --out-dir "$out" \
    --start-time "$start_time" \
    --end-time "$end_time" \
    --prefill-start 0 \
    --prefill-end "$prefill_end" \
    --prefill-summary "$PREFILL_SUMMARY" \
    --frames-per-turn 1 \
    --memory-lookback 2 \
    --prefill-text-mode "$text_mode" \
    --prefill-text-count "$text_count" \
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-layout "$layout" \
    --temperature 0 \
    --max-tokens 120 \
    >"$out/run.log" 2>&1
  echo "continue done port=$port dur=$dur config=$name"
}

run_time_case() {
  local port="$1"
  local dur="$2"
  local prefill_end="$3"
  local name="$4"
  local text_mode="$5"
  local text_count="$6"
  local visual_mode="$7"
  local visual_count="$8"
  local summary_count="$9"
  local layout="${10}"
  local query="${11}"
  local out="$OUT_ROOT/time_range/dur${dur}_${name}/$query"
  mkdir -p "$out"
  echo "time_range start port=$port dur=$dur config=$name query=$query"
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_time_range_qa \
    --base-url "http://127.0.0.1:${port}/v1" \
    --model "$MODEL" \
    --out-dir "$out" \
    --start-time "$((prefill_end + 1))" \
    --end-time "$((prefill_end + 1))" \
    --prefill-start 0 \
    --prefill-end "$prefill_end" \
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
    >"$out/run.log" 2>&1
  echo "time_range done port=$port dur=$dur config=$name query=$query"
}

job_count=0
running=0
launch_job() {
  local port="${ports[$((job_count % ${#ports[@]}))]}"
  local fn="$1"
  shift
  "$fn" "$port" "$@" &
  job_count=$((job_count + 1))
  running=$((running + 1))
  if (( running >= ${#ports[@]} )); then
    wait -n
    running=$((running - 1))
  fi
}

for dur_spec in "8 7 8 38" "16 15 16 46"; do
  read -r dur prefill_end start_time end_time <<<"$dur_spec"
  for cfg in "${configs[@]}"; do
    read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
    launch_job run_continue_case "$dur" "$prefill_end" "$start_time" "$end_time" "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout"
  done

  queries=(banana_fork electric_mixer sliced_bread_plate)
  if [[ "$dur" == "16" ]]; then
    queries+=(holding_bananas)
  fi
  for cfg in "${configs[@]}"; do
    read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
    for query in "${queries[@]}"; do
      launch_job run_time_case "$dur" "$prefill_end" "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout" "$query"
    done
  done
done

wait

"$PY" scripts/audit/summarize_prefill_parallel_matrix.py "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"
