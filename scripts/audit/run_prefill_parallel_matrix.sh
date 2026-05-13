#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PY="${PY:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
MODEL="${MODEL:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/prefill_matrix_8gpu}"
PREFILL_SUMMARY="${PREFILL_SUMMARY:-output/vllm_latest_frame_probe/noresp_stream_0_59/summary.json}"
RUN_TIME_RANGE="${RUN_TIME_RANGE:-1}"
RUN_CONTINUE="${RUN_CONTINUE:-1}"
START_TIME="${START_TIME:-60}"
END_TIME="${END_TIME:-90}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

ports=()
for gpu in 0 1 2 3 4 5 6 7; do
  ports+=("$((PORT_BASE + gpu))")
done

wait_server() {
  local port="$1"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "ready port=$port"
      return 0
    fi
    sleep 5
  done
  echo "server not ready: port=$port" >&2
  return 1
}

for port in "${ports[@]}"; do
  wait_server "$port" &
done
wait

configs=(
  "summary_only none 0 none 0 8 text_then_visual"
  "text_recent8 recent 8 none 0 8 text_then_visual"
  "text_recent16 recent 16 none 0 8 text_then_visual"
  "text_uniform8 uniform 8 none 0 8 text_then_visual"
  "text_uniform16 uniform 16 none 0 8 text_then_visual"
  "text_uniform_recent16 uniform_recent 16 none 0 8 text_then_visual"
  "text_all all -1 none 0 12 text_then_visual"
  "visual_uniform8 none 0 uniform 8 8 visual_only"
  "visual_recent8 none 0 recent 8 8 visual_only"
  "visual_all none 0 all -1 8 visual_only"
  "text_uniform12_visual_uniform8 uniform 12 uniform 8 8 text_then_visual"
  "text_recent16_visual_recent8 recent 16 recent 8 8 text_then_visual"
  "visual_then_text_uniform12v8 uniform 12 uniform 8 8 visual_then_text"
)

queries=(
  electric_mixer
  flour_measuring
  banana_fork
  holding_bananas
  sliced_bread_plate
)

run_time_range_case() {
  local port="$1"
  local name="$2"
  local text_mode="$3"
  local text_count="$4"
  local visual_mode="$5"
  local visual_count="$6"
  local summary_count="$7"
  local layout="$8"
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
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count "$summary_count" \
    --prefill-layout "$layout" \
    --temperature 0 \
    --max-tokens 120 \
    >"$out/run.log" 2>&1
  echo "time_range done port=$port config=$name query=$query"
}

run_continue_case() {
  local port="$1"
  local name="$2"
  local text_mode="$3"
  local text_count="$4"
  local visual_mode="$5"
  local visual_count="$6"
  local summary_count="$7"
  local layout="$8"
  local out="$OUT_ROOT/continue/$name"
  mkdir -p "$out"
  echo "continue start port=$port config=$name"
  "$PY" scripts/audit/vllm_prefill_memory_caption_probe.py \
    --mode prefill_continue \
    --base-url "http://127.0.0.1:${port}/v1" \
    --model "$MODEL" \
    --out-dir "$out" \
    --start-time "$START_TIME" \
    --end-time "$END_TIME" \
    --prefill-start 0 \
    --prefill-end 59 \
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
  echo "continue done port=$port config=$name"
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

if [[ "$RUN_TIME_RANGE" == "1" ]]; then
  for cfg in "${configs[@]}"; do
    read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
    for query in "${queries[@]}"; do
      launch_job run_time_range_case "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout" "$query"
    done
  done
fi

if [[ "$RUN_CONTINUE" == "1" ]]; then
  for cfg in "${configs[@]}"; do
    read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
    launch_job run_continue_case "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout"
  done
fi

wait

"$PY" scripts/audit/summarize_prefill_parallel_matrix.py "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"
