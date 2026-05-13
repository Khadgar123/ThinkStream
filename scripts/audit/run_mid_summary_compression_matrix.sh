#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PY="${PY:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
MODEL="${MODEL:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/mid_summary_compression_8gpu}"
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

# name visual_count text_mode text_count summary_start summary_end summary_count
configs=(
  "v12_t8_s0 12 recent 8 8 51 0"
  "v12_t8_mid4 12 recent 8 8 51 4"
  "v12_t8_mid8 12 recent 8 8 51 8"
  "v12_t8_mid12 12 recent 8 8 51 12"
  "v12_t8_mid16 12 recent 8 8 51 16"
  "v12_t8_mid24 12 recent 8 8 51 24"
  "v12_t16_mid8 12 recent 16 8 51 8"
  "v12_t16_mid12 12 recent 16 8 51 12"
  "v12_t16_mid16 12 recent 16 8 51 16"
  "v8_t8_mid8 8 recent 8 8 51 8"
  "v8_t8_mid12 8 recent 8 8 51 12"
  "v8_t8_mid16 8 recent 8 8 51 16"
  "summary_mid8 0 none 0 8 51 8"
  "summary_mid12 0 none 0 8 51 12"
  "summary_mid16 0 none 0 8 51 16"
  "v12_t8_all8 12 recent 8 0 51 8"
  "v12_t8_all12 12 recent 8 0 51 12"
  "v12_t8_all16 12 recent 8 0 51 16"
)

queries=(
  electric_mixer
  banana_fork
  sliced_bread_plate
  holding_bananas
  flour_measuring
)

run_continue_case() {
  local port="$1"
  local name="$2"
  local visual_count="$3"
  local text_mode="$4"
  local text_count="$5"
  local summary_start="$6"
  local summary_end="$7"
  local summary_count="$8"
  local out="$OUT_ROOT/continue/$name"
  mkdir -p "$out"
  echo "continue start port=$port config=$name v=$visual_count text=${text_mode}:${text_count} summary=${summary_start}-${summary_end}:$summary_count"
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_continue \
    --base-url "http://127.0.0.1:${port}/v1" \
    --model "$MODEL" \
    --out-dir "$out" \
    --start-time 60 \
    --end-time 90 \
    --prefill-start 0 \
    --prefill-end 59 \
    --prefill-summary "$PREFILL_SUMMARY" \
    --frames-per-turn 1 \
    --memory-lookback 2 \
    --prefill-text-mode "$text_mode" \
    --prefill-text-count "$text_count" \
    --prefill-visual-mode recent \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-summary-start "$summary_start" \
    --prefill-summary-end "$summary_end" \
    --prefill-summary-style snippets \
    --prefill-layout text_then_visual \
    --temperature 0 \
    --max-tokens 120 \
    >"$out/run.log" 2>&1
  echo "continue done port=$port config=$name"
}

run_time_case() {
  local port="$1"
  local name="$2"
  local visual_count="$3"
  local text_mode="$4"
  local text_count="$5"
  local summary_start="$6"
  local summary_end="$7"
  local summary_count="$8"
  local query="$9"
  local out="$OUT_ROOT/time_range/$name/$query"
  mkdir -p "$out"
  echo "time_range start port=$port config=$name query=$query"
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_time_range_qa \
    --base-url "http://127.0.0.1:${port}/v1" \
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
    --prefill-visual-mode recent \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-summary-start "$summary_start" \
    --prefill-summary-end "$summary_end" \
    --prefill-summary-style snippets \
    --prefill-layout text_then_visual \
    --temperature 0 \
    --max-tokens 120 \
    >"$out/run.log" 2>&1
  echo "time_range done port=$port config=$name query=$query"
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

for cfg in "${configs[@]}"; do
  read -r name visual_count text_mode text_count summary_start summary_end summary_count <<<"$cfg"
  launch_job run_continue_case "$name" "$visual_count" "$text_mode" "$text_count" "$summary_start" "$summary_end" "$summary_count"
done

for cfg in "${configs[@]}"; do
  read -r name visual_count text_mode text_count summary_start summary_end summary_count <<<"$cfg"
  for query in "${queries[@]}"; do
    launch_job run_time_case "$name" "$visual_count" "$text_mode" "$text_count" "$summary_start" "$summary_end" "$summary_count" "$query"
  done
done

wait

"$PY" scripts/audit/summarize_prefill_parallel_matrix.py "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"
