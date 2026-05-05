#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
SFT_RUN_NAME="${SFT_RUN_NAME:-agent-sft-batch1-video_meta_all-8807-20260505-fixedenv}"
SFT_OUT="${SFT_OUT:-$ROOT/output/$SFT_RUN_NAME}"
WAIT_FOR_SFT="${WAIT_FOR_SFT:-1}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"

DATA_ROOT="${DATA_ROOT:-data/agent_v5/batch1}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/eval/batch1_post_sft_$(date +%Y%m%d_%H%M%S)}"

# Full defaults. Set BASE_MAX_EVENTS_PER_SPLIT>0 only for a quick smoke run.
BASE_MAX_EVENTS_PER_SPLIT="${BASE_MAX_EVENTS_PER_SPLIT:-0}"
BASE_MODES="${BASE_MODES:-streaming text_memory streaming_text_memory recall_oracle offline_past offline_full}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-96}"

ACTION_N="${ACTION_N:-100000}"
SFT_GEN_N="${SFT_GEN_N:-0}"
AGENT_N="${AGENT_N:-0}"
AGENT_AUDIT_N="${AGENT_AUDIT_N:-120}"
RUN_OVO="${RUN_OVO:-1}"
OVO_TASKS="${OVO_TASKS:-EPM,ASI,HLD,OCR,ACR,ATR,STU,FPD,OJR}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/base_probe" "$OUT_ROOT/sft_eval"
echo "[post_sft_eval] out_root=$OUT_ROOT"

if [[ "$WAIT_FOR_SFT" == "1" ]]; then
  while pgrep -f "thinkstream/sft/train.py.*${SFT_RUN_NAME}" >/dev/null; do
    echo "[post_sft_eval] waiting for SFT run to finish: $SFT_RUN_NAME"
    sleep "$WAIT_INTERVAL"
  done
fi

if [[ ! -d "$SFT_OUT" ]]; then
  echo "[post_sft_eval] missing SFT output dir: $SFT_OUT" >&2
  exit 1
fi

SFT_CKPT="$("$PYTHON_BIN" - "$SFT_OUT" <<'PY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
ckpts = sorted(
    [p for p in out.glob("checkpoint-*") if p.is_dir()],
    key=lambda p: int(re.search(r"checkpoint-(\d+)$", p.name).group(1)) if re.search(r"checkpoint-(\d+)$", p.name) else -1,
)
best = None
for p in reversed(ckpts):
    state = p / "trainer_state.json"
    if not state.exists():
        continue
    try:
        value = json.loads(state.read_text()).get("best_model_checkpoint")
    except Exception:
        value = None
    if value:
        best = Path(value)
        break
print(best if best and best.exists() else (ckpts[-1] if ckpts else out))
PY
)"
echo "[post_sft_eval] sft_ckpt=$SFT_CKPT"

JOB_DIR="$OUT_ROOT/jobs"
QUEUE_FILE="$OUT_ROOT/jobs.queue"
LOCK_FILE="$OUT_ROOT/jobs.lock"
FAILED_FILE="$OUT_ROOT/failed_jobs.txt"
mkdir -p "$JOB_DIR"
: >"$QUEUE_FILE"
: >"$FAILED_FILE"
JOB_INDEX=0

add_job() {
  local label="$1"
  local script
  script="$JOB_DIR/$(printf '%03d' "$JOB_INDEX")_${label}.sh"
  JOB_INDEX=$((JOB_INDEX + 1))
  cat >"$script"
  chmod +x "$script"
  printf '%s\n' "$script" >>"$QUEUE_FILE"
}

add_base_job() {
  local name="$1"
  local ckpt="$2"
  local mode="$3"
  add_job "base_${name}_${mode}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.trajectory_base_probe \\
  --ckpt "$ckpt" \\
  --input "train_sft:${DATA_ROOT}/final/train_sft_trajectories.jsonl" \\
  --input "train_rl:${DATA_ROOT}/final/train_rl_trajectories.jsonl" \\
  --input "eval:${DATA_ROOT}/final/val_trajectories.jsonl" \\
  --input "test:${DATA_ROOT}/final/test_trajectories.jsonl" \\
  --modes "$mode" \\
  --max-events-per-split "$BASE_MAX_EVENTS_PER_SPLIT" \\
  --max-new-tokens "$BASE_MAX_NEW_TOKENS" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/base_probe/${name}_${mode}.json"
EOF
}

for spec in \
  "qwen3vl2b|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct" \
  "qwen3vl4b|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct" \
  "qwen3vl8b|/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct"
do
  name="${spec%%|*}"
  ckpt="${spec#*|}"
  for mode in $BASE_MODES; do
    add_base_job "$name" "$ckpt" "$mode"
  done
done

add_job "sft_test_action_acc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.sft_action_acc \\
  --ckpt "$SFT_CKPT" \\
  --val "${DATA_ROOT}/rendered/video_meta_all/test_messages.jsonl" \\
  --n "$ACTION_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/sft_eval/test_action_acc.json"
EOF

add_job "sft_test_answer_acc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_sft_gen \\
  --ckpt "$SFT_CKPT" \\
  --test_jsonl "${DATA_ROOT}/rendered/video_meta_all/test_messages.jsonl" \\
  --n "$SFT_GEN_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/sft_eval/test_answer_acc.json"
EOF

for compress_mode in system self; do
  add_job "sft_test_agent_${compress_mode}_bm25" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_agent \\
  --ckpt "$SFT_CKPT" \\
  --test_jsonl "${DATA_ROOT}/final/test.jsonl" \\
  --video_root / \\
  --frames_root "${DATA_ROOT}/frames" \\
  --retriever bm25 \\
  --compress_mode "$compress_mode" \\
  --max_results 4 \\
  --n "$AGENT_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/sft_eval/test_agent_${compress_mode}_bm25.json"
EOF
done

for split in val train_rl; do
  add_job "sft_${split}_agent_system_bm25_think_audit" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_agent \\
  --ckpt "$SFT_CKPT" \\
  --test_jsonl "${DATA_ROOT}/final/${split}.jsonl" \\
  --video_root / \\
  --frames_root "${DATA_ROOT}/frames" \\
  --retriever bm25 \\
  --compress_mode system \\
  --max_results 4 \\
  --n "$AGENT_AUDIT_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/sft_eval/${split}_agent_system_bm25_think_audit.json"
EOF
done

if [[ "$RUN_OVO" == "1" ]]; then
  add_job "sft_ovo_rtbt" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.ovo.eval_sft_rtbt \\
  --ckpt "$SFT_CKPT" \\
  --benchmark_json /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json \\
  --video_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench \\
  --frames_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames \\
  --tasks "$OVO_TASKS" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --out "$OUT_ROOT/sft_eval/ovo_rtbt.json"
EOF
fi

echo "[post_sft_eval] queued_jobs=$(wc -l < "$QUEUE_FILE")"

claim_job() {
  "$PYTHON_BIN" - "$QUEUE_FILE" "$LOCK_FILE" <<'PY'
import fcntl
import sys
from pathlib import Path

queue = Path(sys.argv[1])
lock = Path(sys.argv[2])
with lock.open("w") as lock_f:
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    lines = queue.read_text().splitlines() if queue.exists() else []
    if not lines:
        raise SystemExit(1)
    job = lines[0]
    queue.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
    print(job)
PY
}

worker() {
  local gpu="$1"
  local job
  local label
  while job="$(claim_job)"; do
    label="$(basename "$job" .sh)"
    echo "[post_sft_eval] START gpu=$gpu label=$label"
    if GPU_ID="$gpu" bash "$job" >"$OUT_ROOT/logs/${label}.log" 2>&1; then
      echo "[post_sft_eval] DONE  gpu=$gpu label=$label"
    else
      echo "[post_sft_eval] FAIL  gpu=$gpu label=$label" >&2
      printf '%s\n' "$label" >>"$FAILED_FILE"
    fi
  done
  echo "[post_sft_eval] worker gpu=$gpu idle: queue empty"
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

if [[ -s "$FAILED_FILE" ]]; then
  echo "[post_sft_eval] failed jobs:" >&2
  cat "$FAILED_FILE" >&2
  exit 1
fi

echo "[post_sft_eval] complete: $OUT_ROOT"
