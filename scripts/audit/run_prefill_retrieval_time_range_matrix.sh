#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PY="${PY:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
MODEL="${MODEL:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/prefill_retrieval_time_range_8gpu}"
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
  "text_querytop4_nosummary query_top 4 none 0 0 text_then_visual"
  "text_querytop8_nosummary query_top 8 none 0 0 text_then_visual"
  "text_querytop12_nosummary query_top 12 none 0 0 text_then_visual"
  "visual_querytop4_nosummary none 0 query_top 4 0 visual_only"
  "visual_querytop8_nosummary none 0 query_top 8 0 visual_only"
  "text_querytop8_visual_querytop4_nosummary query_top 8 query_top 4 0 text_then_visual"
  "text_querytop8_visual_querytop8_nosummary query_top 8 query_top 8 0 text_then_visual"
)

queries=(
  electric_mixer
  flour_measuring
  banana_fork
  holding_bananas
  sliced_bread_plate
)

run_case() {
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
  echo "retrieval_time_range start port=$port config=$name query=$query"
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
  echo "retrieval_time_range done port=$port config=$name query=$query"
}

job_count=0
running=0
for cfg in "${configs[@]}"; do
  read -r name text_mode text_count visual_mode visual_count summary_count layout <<<"$cfg"
  for query in "${queries[@]}"; do
    port="${ports[$((job_count % ${#ports[@]}))]}"
    run_case "$port" "$name" "$text_mode" "$text_count" "$visual_mode" "$visual_count" "$summary_count" "$layout" "$query" &
    job_count=$((job_count + 1))
    running=$((running + 1))
    if (( running >= ${#ports[@]} )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

"$PY" scripts/audit/summarize_prefill_parallel_matrix.py "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"
