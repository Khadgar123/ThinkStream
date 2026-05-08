#!/usr/bin/env bash
# Parameterized ThinkStream checkpoint evaluation for staged batch1 experiments.
#
# This launcher intentionally keeps the metric surface fixed across warmup SFT,
# DAgger SFT, and RL checkpoints:
#   - real generate action accuracy on per-step messages
#   - one-shot answer accuracy on response/recall rows
#   - full streaming agent-loop accuracy with BM25 recall and system/self compress
#   - think/memory behavior audits on val and train_rl trajectory rows
#   - optional OVO-Bench RTBT eval
#
# Required:
#   CKPT=/path/to/checkpoint bash scripts/eval/run_stage_eval.sh
#
# Common overrides:
#   DATA_ROOT=data/agent_v5/backups/batch1_aligned_20260506_1200
#   FINAL_DIR=$DATA_ROOT/experiments/ratio_dagger_aligned_20260506/warmup
#   OUT_ROOT=output/eval/<name>

set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "$ROOT"

CKPT="${CKPT:?CKPT=/path/to/checkpoint is required}"
PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/agent_v5/backups/batch1_aligned_20260506_1200}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-standard}"
if [[ -z "${FINAL_DIR:-}" ]]; then
  if [[ "$THINKSTREAM_RENDER_LAYOUT" == "standard" ]]; then
    FINAL_DIR="${DATA_ROOT}/rendered/${FRAME_PROTOCOL}"
  else
    FINAL_DIR="${DATA_ROOT}/rendered/${FRAME_PROTOCOL}_${THINKSTREAM_RENDER_LAYOUT}"
  fi
fi
FRAMES_ROOT="${FRAMES_ROOT:-${DATA_ROOT}/frames}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/eval/stage_$(basename "$CKPT")_$(date +%Y%m%d_%H%M%S)}"
TEST_TRAJ="${TEST_TRAJ:-}"
if [[ -z "$TEST_TRAJ" ]]; then
  if [[ -f "${DATA_ROOT}/final/test_trajectories.jsonl" ]]; then
    TEST_TRAJ="${DATA_ROOT}/final/test_trajectories.jsonl"
  else
    TEST_TRAJ="${DATA_ROOT}/final/test.jsonl"
  fi
fi
VAL_TRAJ="${VAL_TRAJ:-}"
if [[ -z "$VAL_TRAJ" ]]; then
  if [[ -f "${DATA_ROOT}/final/val_trajectories.jsonl" ]]; then
    VAL_TRAJ="${DATA_ROOT}/final/val_trajectories.jsonl"
  else
    VAL_TRAJ="${DATA_ROOT}/final/val.jsonl"
  fi
fi
TRAIN_RL_TRAJ="${TRAIN_RL_TRAJ:-}"
if [[ -z "$TRAIN_RL_TRAJ" ]]; then
  if [[ -f "${DATA_ROOT}/final/train_rl_trajectories.jsonl" ]]; then
    TRAIN_RL_TRAJ="${DATA_ROOT}/final/train_rl_trajectories.jsonl"
  else
    TRAIN_RL_TRAJ="${DATA_ROOT}/final/train_rl.jsonl"
  fi
fi
export THINKSTREAM_FRAME_PROTOCOL="$FRAME_PROTOCOL"
export THINKSTREAM_RENDER_LAYOUT="$THINKSTREAM_RENDER_LAYOUT"

ACTION_N="${ACTION_N:-0}"
SFT_GEN_N="${SFT_GEN_N:-0}"
AGENT_N="${AGENT_N:-0}"
AGENT_AUDIT_N="${AGENT_AUDIT_N:-120}"
RUN_OVO="${RUN_OVO:-1}"
OVO_TASKS="${OVO_TASKS:-EPM,ASI,HLD,OCR,ACR,ATR,STU,FPD,OJR}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/sft_eval"
echo "[stage_eval] ckpt=$CKPT"
echo "[stage_eval] data_root=$DATA_ROOT"
echo "[stage_eval] final_dir=$FINAL_DIR"
echo "[stage_eval] render_layout=$THINKSTREAM_RENDER_LAYOUT"
echo "[stage_eval] out_root=$OUT_ROOT"

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

add_job "test_action_acc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.sft_action_acc \\
  --ckpt "$CKPT" \\
  --val "$FINAL_DIR/test_messages.jsonl" \\
  --n "$ACTION_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --render-layout "$THINKSTREAM_RENDER_LAYOUT" \\
  --out "$OUT_ROOT/sft_eval/test_action_acc.json"
EOF

add_job "test_answer_acc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_sft_gen \\
  --ckpt "$CKPT" \\
  --test_jsonl "$FINAL_DIR/test_messages.jsonl" \\
  --n "$SFT_GEN_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --render-layout "$THINKSTREAM_RENDER_LAYOUT" \\
  --out "$OUT_ROOT/sft_eval/test_answer_acc.json"
EOF

for compress_mode in system self; do
  add_job "test_agent_${compress_mode}_bm25" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_agent \\
  --ckpt "$CKPT" \\
  --test_jsonl "$TEST_TRAJ" \\
  --video_root / \\
  --frames_root "$FRAMES_ROOT" \\
  --retriever bm25 \\
  --compress_mode "$compress_mode" \\
  --max_results 4 \\
  --n "$AGENT_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --render-layout "$THINKSTREAM_RENDER_LAYOUT" \\
  --out "$OUT_ROOT/sft_eval/test_agent_${compress_mode}_bm25.json"
EOF
done

for split in val train_rl; do
  if [[ "$split" == "val" ]]; then
    split_traj="$VAL_TRAJ"
  else
    split_traj="$TRAIN_RL_TRAJ"
  fi
  add_job "${split}_agent_system_bm25_think_audit" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.test_set_agent \\
  --ckpt "$CKPT" \\
  --test_jsonl "$split_traj" \\
  --video_root / \\
  --frames_root "$FRAMES_ROOT" \\
  --retriever bm25 \\
  --compress_mode system \\
  --max_results 4 \\
  --n "$AGENT_AUDIT_N" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --render-layout "$THINKSTREAM_RENDER_LAYOUT" \\
  --out "$OUT_ROOT/sft_eval/${split}_agent_system_bm25_think_audit.json"
EOF
done

if [[ "$RUN_OVO" == "1" ]]; then
  add_job "ovo_rtbt" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?}"
"$PYTHON_BIN" -m scripts.eval.ovo.eval_sft_rtbt \\
  --ckpt "$CKPT" \\
  --benchmark_json /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json \\
  --video_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench \\
  --frames_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames \\
  --tasks "$OVO_TASKS" \\
  --frame-protocol "$FRAME_PROTOCOL" \\
  --render-layout "$THINKSTREAM_RENDER_LAYOUT" \\
  --out "$OUT_ROOT/sft_eval/ovo_rtbt.json"
EOF
fi

echo "[stage_eval] queued_jobs=$(wc -l < "$QUEUE_FILE")"

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
    echo "[stage_eval] START gpu=$gpu label=$label"
    if GPU_ID="$gpu" bash "$job" >"$OUT_ROOT/logs/${label}.log" 2>&1; then
      echo "[stage_eval] DONE  gpu=$gpu label=$label"
    else
      echo "[stage_eval] FAIL  gpu=$gpu label=$label" >&2
      printf '%s\n' "$label" >>"$FAILED_FILE"
    fi
  done
  echo "[stage_eval] worker gpu=$gpu idle: queue empty"
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
  echo "[stage_eval] failed jobs:" >&2
  cat "$FAILED_FILE" >&2
  exit 1
fi

echo "[stage_eval] complete: $OUT_ROOT"
