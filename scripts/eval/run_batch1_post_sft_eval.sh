#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"
SFT_RUN_NAME="${SFT_RUN_NAME:-agent-sft-batch1-video_meta_all-8807-20260505-fixedenv}"
SFT_OUT="${SFT_OUT:-$ROOT/output/$SFT_RUN_NAME}"
WAIT_FOR_SFT="${WAIT_FOR_SFT:-1}"

DATA_ROOT="${DATA_ROOT:-data/agent_v5/batch1}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/eval/batch1_post_sft_$(date +%Y%m%d_%H%M%S)}"

# Pilot defaults. Set BASE_MAX_EVENTS_PER_SPLIT=0 for the full matrix.
BASE_MAX_EVENTS_PER_SPLIT="${BASE_MAX_EVENTS_PER_SPLIT:-40}"
BASE_MODES="${BASE_MODES:-streaming text_memory streaming_text_memory recall_oracle offline_past}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-96}"

ACTION_N="${ACTION_N:-100000}"
SFT_GEN_N="${SFT_GEN_N:-0}"
AGENT_N="${AGENT_N:-120}"
RUN_OVO="${RUN_OVO:-1}"
OVO_TASKS="${OVO_TASKS:-EPM,ASI,HLD,OCR,ACR,ATR,STU,FPD,OJR}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/base_probe" "$OUT_ROOT/sft_eval"
echo "[post_sft_eval] out_root=$OUT_ROOT"

if [[ "$WAIT_FOR_SFT" == "1" ]]; then
  while pgrep -f "thinkstream/sft/train.py.*${SFT_RUN_NAME}" >/dev/null; do
    echo "[post_sft_eval] waiting for SFT run to finish: $SFT_RUN_NAME"
    sleep 300
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

common_inputs=(
  --input "train_sft:${DATA_ROOT}/final/train_sft_trajectories.jsonl"
  --input "train_rl:${DATA_ROOT}/final/train_rl_trajectories.jsonl"
  --input "eval:${DATA_ROOT}/final/val_trajectories.jsonl"
  --input "test:${DATA_ROOT}/final/test_trajectories.jsonl"
)

run_base_probe() {
  local gpu="$1"
  local name="$2"
  local ckpt="$3"
  local out_json="$OUT_ROOT/base_probe/${name}.json"
  local log="$OUT_ROOT/logs/base_${name}.log"
  echo "[post_sft_eval] base probe start name=$name gpu=$gpu ckpt=$ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m scripts.eval.trajectory_base_probe \
    --ckpt "$ckpt" \
    "${common_inputs[@]}" \
    --modes $BASE_MODES \
    --max-events-per-split "$BASE_MAX_EVENTS_PER_SPLIT" \
    --max-new-tokens "$BASE_MAX_NEW_TOKENS" \
    --frame-protocol "$FRAME_PROTOCOL" \
    --out "$out_json" \
    >"$log" 2>&1
  echo "[post_sft_eval] base probe done name=$name out=$out_json"
}

pids=()
run_base_probe 0 qwen3vl2b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct &
pids+=("$!")
run_base_probe 1 qwen3vl4b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-4B-Instruct &
pids+=("$!")
run_base_probe 2 qwen3vl8b /home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct &
pids+=("$!")

base_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    base_status=1
  fi
done
if [[ "$base_status" != "0" ]]; then
  echo "[post_sft_eval] one or more base probes failed; continuing with SFT evals" >&2
fi

CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" -m scripts.eval.sft_action_acc \
  --ckpt "$SFT_CKPT" \
  --val "${DATA_ROOT}/rendered/video_meta_all/test_messages.jsonl" \
  --n "$ACTION_N" \
  --frame-protocol "$FRAME_PROTOCOL" \
  --out "$OUT_ROOT/sft_eval/test_action_acc.json" \
  >"$OUT_ROOT/logs/sft_action_acc.log" 2>&1

CUDA_VISIBLE_DEVICES=4 "$PYTHON_BIN" -m scripts.eval.test_set_sft_gen \
  --ckpt "$SFT_CKPT" \
  --test_jsonl "${DATA_ROOT}/rendered/video_meta_all/test_messages.jsonl" \
  --n "$SFT_GEN_N" \
  --frame-protocol "$FRAME_PROTOCOL" \
  --out "$OUT_ROOT/sft_eval/test_answer_acc.json" \
  >"$OUT_ROOT/logs/sft_answer_acc.log" 2>&1

for compress_mode in system self; do
  CUDA_VISIBLE_DEVICES=5 "$PYTHON_BIN" -m scripts.eval.test_set_agent \
    --ckpt "$SFT_CKPT" \
    --test_jsonl "${DATA_ROOT}/final/test.jsonl" \
    --video_root / \
    --frames_root "${DATA_ROOT}/frames" \
    --retriever bm25 \
    --compress_mode "$compress_mode" \
    --max_results 4 \
    --n "$AGENT_N" \
    --frame-protocol "$FRAME_PROTOCOL" \
    --out "$OUT_ROOT/sft_eval/test_agent_${compress_mode}_bm25.json" \
    >"$OUT_ROOT/logs/test_agent_${compress_mode}_bm25.log" 2>&1
done

if [[ "$RUN_OVO" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" -m scripts.eval.ovo.eval_sft_rtbt \
    --ckpt "$SFT_CKPT" \
    --benchmark_json /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json \
    --video_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench \
    --frames_root /home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames \
    --tasks "$OVO_TASKS" \
    --frame-protocol "$FRAME_PROTOCOL" \
    --out "$OUT_ROOT/sft_eval/ovo_rtbt.json" \
    >"$OUT_ROOT/logs/ovo_rtbt.log" 2>&1
fi

echo "[post_sft_eval] complete: $OUT_ROOT"
