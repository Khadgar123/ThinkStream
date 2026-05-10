#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.18.9:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-1 3 6 8}"
FORCE_RERUN_FROM="${FORCE_RERUN_FROM-3a}"

cd "${PROJECT_ROOT}"

GLOBAL_LOG="${PROJECT_ROOT}/data/agent_v5/audits/pass3_rerun_batch1368_4card_${RUN_ID}.log"
mkdir -p "$(dirname "${GLOBAL_LOG}")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-40000}"
export THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO="${THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO:-0.95}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET="${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET:-64000000}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
if [[ -z "${FORCE_RERUN_FROM}" ]]; then
  export THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE="${THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE:-1}"
else
  export THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE="${THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE:-0}"
fi
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.18.9"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.18.9"

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

from scripts.agent_data_v5 import config as c

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

print(
    "context_check="
    f"max_model_len={c.VLLM_MAX_MODEL_LEN} "
    f"safe_tokens={c.max_safe_context_tokens()} "
    f"pass3a_est={c.estimated_request_tokens('pass3a')} "
    f"pass3c_est={c.estimated_request_tokens('pass3c')} "
    f"pass3a_concurrency={c.safe_concurrency_for_pass('pass3a')} "
    f"pass3c_concurrency={c.safe_concurrency_for_pass('pass3c')}"
)
if c.estimated_request_tokens("pass3a") > c.max_safe_context_tokens():
    raise SystemExit("pass3a estimated context exceeds configured safe context")
if c.safe_concurrency_for_pass("pass3a") < 1024:
    raise SystemExit("pass3a concurrency was clamped below 1024")
PY
}

log "START batch1368 pass3 rerun on 4-card endpoint"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
log "context_safety_ratio=${THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO}"
log "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
log "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
log "pass3a_concurrent=${THINKSTREAM_PASS3A_CONCURRENT}"
log "pass3c_concurrent=${THINKSTREAM_PASS3C_CONCURRENT}"
log "prefill_batch_token_budget=${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET}"
log "allow_partial_pass3a_cache=${THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE}"
log "force_rerun_from=${FORCE_RERUN_FROM:-<none>}"
log "batches=${BATCHES}"
log "global_log=${GLOBAL_LOG}"

for b in ${BATCHES}; do
  batch="batch${b}"
  batch_root="${PROJECT_ROOT}/data/agent_v5/${batch}"
  videos_jsonl="$(videos_file_for_batch "${b}")"
  num_videos="$(wc -l < "${videos_jsonl}")"
  batch_log="${batch_root}/logs/pass3_rerun_4card_${RUN_ID}.log"
  mkdir -p "${batch_root}/logs"

  log "${batch} START pass3a+downstream; videos=${num_videos}; videos_jsonl=${videos_jsonl}; log=${batch_log}"

  {
    echo "[$(date '+%F %T')] [${batch}] START pass3 rerun on 4-card"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "api_base=${API_BASE}"
    echo "model=${MODEL}"
    echo "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
    echo "context_safety_ratio=${THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO}"
    echo "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
    echo "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
    echo "pass3a_concurrent=${THINKSTREAM_PASS3A_CONCURRENT}"
    echo "pass3c_concurrent=${THINKSTREAM_PASS3C_CONCURRENT}"
    echo "allow_partial_pass3a_cache=${THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE}"
    echo "videos_jsonl=${videos_jsonl}"
    echo "num_videos=${num_videos}"
    echo "data_root=${batch_root}"
    echo "run_id=${RUN_ID}"
    echo "force_rerun_from=${FORCE_RERUN_FROM:-<none>}"
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
  force_args=()
  if [[ -n "${FORCE_RERUN_FROM}" ]]; then
    force_args+=(--force_rerun_from "${FORCE_RERUN_FROM}")
  fi
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  THINKSTREAM_BATCH="${batch}" \
  python -m scripts.agent_data_v5.pipeline run \
    --api_base "${API_BASE}" \
    --model "${MODEL}" \
    --videos_jsonl "${videos_jsonl}" \
    --num_videos "${num_videos}" \
    --skip_pass 1 2 \
    "${force_args[@]}" \
    2>&1 | tee -a "${batch_log}"
  rc=${PIPESTATUS[0]}
  set -e

  log "${batch} EXIT:${rc}"
  echo "[$(date '+%F %T')] [${batch}] EXIT:${rc}" | tee -a "${batch_log}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

log "COMPLETE batch1368 pass3 rerun on 4-card endpoint"
