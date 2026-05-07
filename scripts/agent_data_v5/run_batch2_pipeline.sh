#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/tione/notebook/gaozhenkun/hzh/ThinkStream"
BATCH_ROOT="${PROJECT_ROOT}/data/agent_v5/batch2"

cd "${PROJECT_ROOT}"

# Keep the batch root inside this child process so nohup/tmux launches cannot
# accidentally fall back to the historical data/agent_v5 root.
export THINKSTREAM_DATA_ROOT="${BATCH_ROOT}"
export AGENT_DATA_DIR="${BATCH_ROOT}"
export THINKSTREAM_BATCH="batch2"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.18.9"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.18.9"

API_BASE="${API_BASE:-http://10.16.18.9:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8}"
VIDEOS_JSONL="${VIDEOS_JSONL:-${PROJECT_ROOT}/data/agent_v5/batch2_videos.jsonl}"
NUM_VIDEOS="${NUM_VIDEOS:-500}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-32768}"
# The 4-GPU 122B service OOM-killed its EngineCore at 1024-way pressure
# (400GB+ RSS inside a 640GB memory cgroup). Keep batch2 below that ceiling;
# callers can still override these values explicitly when the service changes.
export THINKSTREAM_PASS1A_CONCURRENT="${THINKSTREAM_PASS1A_CONCURRENT:-512}"
export THINKSTREAM_PASS1B_CONCURRENT="${THINKSTREAM_PASS1B_CONCURRENT:-512}"
export THINKSTREAM_PASS2_ROLLOUT_CONCURRENT="${THINKSTREAM_PASS2_ROLLOUT_CONCURRENT:-512}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-512}"
export THINKSTREAM_PASS3B_VISIBILITY_CONCURRENT="${THINKSTREAM_PASS3B_VISIBILITY_CONCURRENT:-512}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-512}"
LOG_DIR="${BATCH_ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/pipeline.${RUN_ID}.log}"
ENV_SNAPSHOT="${LOG_DIR}/env.${RUN_ID}.txt"

printenv | sort > "${ENV_SNAPSHOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "project_root=${PROJECT_ROOT}"
echo "batch_root=${BATCH_ROOT}"
echo "api_base=${API_BASE}"
echo "model=${MODEL}"
echo "thinkstream_vllm_max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
echo "videos_jsonl=${VIDEOS_JSONL}"
echo "num_videos=${NUM_VIDEOS}"
echo "log_path=${LOG_PATH}"
echo "env_snapshot=${ENV_SNAPSHOT}"

python - <<'PY'
from pathlib import Path
from scripts.agent_data_v5 import config as c

expected = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/batch2")
print(f"resolved DATA_ROOT={c.DATA_ROOT}")
print(f"resolved FINAL_DIR={c.FINAL_DIR}")
if c.DATA_ROOT != expected:
    raise SystemExit(f"Refusing to run: DATA_ROOT={c.DATA_ROOT}, expected {expected}")
PY

API_BASE="${API_BASE}" MODEL="${MODEL}" python - <<'PY'
import json
import os
import urllib.request

api_base = os.environ["API_BASE"].rstrip("/")
model = os.environ["MODEL"]
url = f"{api_base}/models"
with urllib.request.urlopen(url, timeout=10) as resp:
    payload = json.loads(resp.read().decode("utf-8"))

models = [str(item.get("id", "")) for item in payload.get("data", [])]
print(f"available_models={models}")
if model not in models:
    raise SystemExit(
        f"Refusing to run: MODEL={model} is not listed by {url}"
    )
PY

args=(
  --api_base "${API_BASE}"
  --model "${MODEL}"
  --videos_jsonl "${VIDEOS_JSONL}"
  --num_videos "${NUM_VIDEOS}"
)
exec python -m scripts.agent_data_v5.pipeline run "${args[@]}" "$@"
