#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
PY="${PY:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
MODEL="${MODEL:-qwen3vl2b-prefill-matrix}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-output/vllm_latest_frame_probe/visual_text_retention_continue_8gpu}"
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
  "v0_t8 none 0 recent 8"
  "v0_t16 none 0 recent 16"
  "v0_t32 none 0 recent 32"
  "v0_tall none 0 all -1"
  "v8_t0 recent 8 none 0"
  "v8_t8 recent 8 recent 8"
  "v8_t16 recent 8 recent 16"
  "v8_t32 recent 8 recent 32"
  "v8_tall recent 8 all -1"
  "v12_t0 recent 12 none 0"
  "v12_t8 recent 12 recent 8"
  "v12_t16 recent 12 recent 16"
  "v12_t32 recent 12 recent 32"
  "v12_tall recent 12 all -1"
  "v16_t0 recent 16 none 0"
  "v16_t8 recent 16 recent 8"
  "v16_t16 recent 16 recent 16"
  "v16_t32 recent 16 recent 32"
  "v16_tall recent 16 all -1"
)

run_case() {
  local port="$1"
  local name="$2"
  local visual_mode="$3"
  local visual_count="$4"
  local text_mode="$5"
  local text_count="$6"
  local out="$OUT_ROOT/continue/$name"
  mkdir -p "$out"
  echo "continue start port=$port config=$name visual=${visual_mode}:${visual_count} text=${text_mode}:${text_count}"
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
    --prefill-visual-mode "$visual_mode" \
    --prefill-visual-count "$visual_count" \
    --prefill-summary-count 0 \
    --prefill-layout text_then_visual \
    --temperature 0 \
    --max-tokens 120 \
    >"$out/run.log" 2>&1
  echo "continue done port=$port config=$name"
}

job_count=0
running=0
for cfg in "${configs[@]}"; do
  read -r name visual_mode visual_count text_mode text_count <<<"$cfg"
  port="${ports[$((job_count % ${#ports[@]}))]}"
  run_case "$port" "$name" "$visual_mode" "$visual_count" "$text_mode" "$text_count" &
  job_count=$((job_count + 1))
  running=$((running + 1))
  if (( running >= ${#ports[@]} )); then
    wait -n
    running=$((running - 1))
  fi
done
wait

"$PY" scripts/audit/summarize_prefill_parallel_matrix.py "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"
