#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-1 2 3 4 5 6 7 8 9}"

cd "${PROJECT_ROOT}"

GLOBAL_LOG="${PROJECT_ROOT}/data/agent_v5/audits/pass3_rerun_batch1_9_8card_${RUN_ID}.log"
mkdir -p "$(dirname "${GLOBAL_LOG}")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-65536}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET="${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET:-64000000}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.12.175"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.12.175"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${GLOBAL_LOG}"
}

videos_file_for_batch() {
  local b="$1"
  local external="${PROJECT_ROOT}/data/agent_v5/batch${b}_videos.jsonl"
  local registry="${PROJECT_ROOT}/data/agent_v5/batch${b}/video_registry.jsonl"
  if [[ -f "${external}" ]]; then
    echo "${external}"
  elif [[ -f "${registry}" ]]; then
    echo "${registry}"
  else
    return 1
  fi
}

preflight_batch() {
  local batch_root="$1"
  local videos_jsonl="$2"
  local expected_n="$3"
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  VIDEOS_JSONL="${videos_jsonl}" \
  EXPECTED_N="${expected_n}" \
  python - <<'PY'
import os
from pathlib import Path

from scripts.agent_data import config as c

root = Path(os.environ["THINKSTREAM_DATA_ROOT"])
videos = Path(os.environ["VIDEOS_JSONL"])
expected_n = int(os.environ["EXPECTED_N"])
print(f"resolved DATA_ROOT={c.DATA_ROOT}")
print(f"resolved FINAL_DIR={c.FINAL_DIR}")
if c.DATA_ROOT != root:
    raise SystemExit(f"Refusing to run: DATA_ROOT={c.DATA_ROOT}, expected {root}")
if not videos.exists():
    raise SystemExit(f"Missing videos_jsonl: {videos}")

def count_json(path: Path) -> int:
    return len(list(path.glob("*.json"))) if path.exists() else 0

counts = {
    "videos": sum(1 for _ in videos.open()),
    "evidence_1a": count_json(root / "evidence_1a"),
    "evidence_1b": count_json(root / "evidence_1b"),
    "rollout": count_json(root / "rollout"),
}
print("preflight_counts=" + " ".join(f"{k}={v}" for k, v in counts.items()))
for key in ("videos", "evidence_1a", "evidence_1b", "rollout"):
    if counts[key] < expected_n:
        raise SystemExit(f"{key} incomplete: {counts[key]} < {expected_n}")
PY
}

log "START batch1-9 pass3 rerun on 8-card endpoint"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
log "batches=${BATCHES}"
log "global_log=${GLOBAL_LOG}"

for b in ${BATCHES}; do
  batch="batch${b}"
  batch_root="${PROJECT_ROOT}/data/agent_v5/${batch}"
  videos_jsonl="$(videos_file_for_batch "${b}")"
  num_videos="$(wc -l < "${videos_jsonl}")"
  batch_log="${batch_root}/logs/pass3_rerun_8card_${RUN_ID}.log"
  mkdir -p "${batch_root}/logs"

  log "${batch} START pass3a+downstream; videos=${num_videos}; videos_jsonl=${videos_jsonl}; log=${batch_log}"

  {
    echo "[$(date '+%F %T')] [${batch}] START pass3 rerun on 8-card"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "api_base=${API_BASE}"
    echo "model=${MODEL}"
    echo "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
    echo "videos_jsonl=${videos_jsonl}"
    echo "num_videos=${num_videos}"
    echo "data_root=${batch_root}"
    echo "run_id=${RUN_ID}"
    echo "force_rerun_from=3a"
    echo "skip_pass=1 2"
  } | tee "${batch_log}"

  set +e
  preflight_batch "${batch_root}" "${videos_jsonl}" "${num_videos}" 2>&1 | tee -a "${batch_log}"
  preflight_rc=${PIPESTATUS[0]}
  set -e
  if [[ ${preflight_rc} -ne 0 ]]; then
    log "${batch} PREFLIGHT_EXIT:${preflight_rc}"
    exit "${preflight_rc}"
  fi

  set +e
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  THINKSTREAM_BATCH="${batch}" \
  python -m scripts.agent_data.pipeline run \
    --api_base "${API_BASE}" \
    --model "${MODEL}" \
    --videos_jsonl "${videos_jsonl}" \
    --num_videos "${num_videos}" \
    --skip_pass 1 2 \
    --force_rerun_from 3a \
    2>&1 | tee -a "${batch_log}"
  rc=${PIPESTATUS[0]}
  set -e

  log "${batch} EXIT:${rc}"
  echo "[$(date '+%F %T')] [${batch}] EXIT:${rc}" | tee -a "${batch_log}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

log "COMPLETE batch1-9 pass3 rerun on 8-card endpoint"
