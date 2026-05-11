#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-2 4 5 7 9}"
RESUME_FIRST_BATCH="${RESUME_FIRST_BATCH:-}"
RESUME_FIRST_BATCH_FORCE_FROM="${RESUME_FIRST_BATCH_FORCE_FROM-2}"

cd "${PROJECT_ROOT}"

GLOBAL_LOG="${PROJECT_ROOT}/data/agent_v5/audits/pass1b_rerun_batch24579_8card_${RUN_ID}.log"
mkdir -p "$(dirname "${GLOBAL_LOG}")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-65536}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET="${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET:-64000000}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
export THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE="${THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE:-0}"
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
    "evidence_1b_current": count_json(root / "evidence_1b"),
    "rollout_current": count_json(root / "rollout"),
}
print("preflight_counts=" + " ".join(f"{k}={v}" for k, v in counts.items()))
if counts["videos"] != expected_n:
    raise SystemExit(f"video count mismatch: {counts['videos']} != {expected_n}")
if counts["evidence_1a"] < expected_n:
    raise SystemExit(f"evidence_1a incomplete: {counts['evidence_1a']} < {expected_n}")
PY
}

log "START batch24579 rerun from pass1b on 8-card endpoint"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
log "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
log "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
log "batches=${BATCHES}"
log "resume_first_batch=${RESUME_FIRST_BATCH:-<none>}"
log "resume_first_batch_force_from=${RESUME_FIRST_BATCH_FORCE_FROM}"
log "global_log=${GLOBAL_LOG}"

for b in ${BATCHES}; do
  batch="batch${b}"
  batch_root="${PROJECT_ROOT}/data/agent_v5/${batch}"
  videos_jsonl="$(videos_file_for_batch "${b}")"
  num_videos="$(wc -l < "${videos_jsonl}")"
  batch_log="${batch_root}/logs/pass1b_rerun_8card_${RUN_ID}.log"
  force_stage="1b"
  resume_existing_1b=0
  if [[ -n "${RESUME_FIRST_BATCH}" && "${b}" == "${RESUME_FIRST_BATCH}" ]]; then
    force_stage="${RESUME_FIRST_BATCH_FORCE_FROM}"
    resume_existing_1b=1
  fi
  allow_partial_pass3a_cache=0
  if [[ -z "${force_stage}" ]]; then
    allow_partial_pass3a_cache=1
  fi
  mkdir -p "${batch_root}/logs"

  log "${batch} START from pass1b; force_rerun_from=${force_stage}; videos=${num_videos}; videos_jsonl=${videos_jsonl}; log=${batch_log}"

  {
    echo "[$(date '+%F %T')] [${batch}] START rerun from pass1b on 8-card"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "api_base=${API_BASE}"
    echo "model=${MODEL}"
    echo "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
    echo "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
    echo "pass3a_video_concurrent=${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
    echo "videos_jsonl=${videos_jsonl}"
    echo "num_videos=${num_videos}"
    echo "data_root=${batch_root}"
    echo "run_id=${RUN_ID}"
    echo "force_rerun_from=${force_stage:-<none>}"
    echo "resume_existing_1b=${resume_existing_1b}"
    echo "allow_partial_pass3a_cache=${allow_partial_pass3a_cache}"
    echo "skip_pass=<none>"
  } | tee "${batch_log}"

  set +e
  preflight_batch "${batch_root}" "${videos_jsonl}" "${num_videos}" 2>&1 | tee -a "${batch_log}"
  preflight_rc=${PIPESTATUS[0]}
  set -e
  if [[ ${preflight_rc} -ne 0 ]]; then
    log "${batch} PREFLIGHT_EXIT:${preflight_rc}"
    exit "${preflight_rc}"
  fi

  if [[ "${resume_existing_1b}" == "1" && -n "${force_stage}" && "${force_stage}" != "1b" ]]; then
    log "${batch} resume mode: preserving existing 1b JSON caches and clearing from ${force_stage}"
    THINKSTREAM_DATA_ROOT="${batch_root}" \
    AGENT_DATA_DIR="${batch_root}" \
    python -c "from scripts.agent_data.cache_version import write_stage_version; write_stage_version('1b')"
  fi

  set +e
  force_args=()
  if [[ -n "${force_stage}" ]]; then
    force_args+=(--force_rerun_from "${force_stage}")
  fi
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  THINKSTREAM_BATCH="${batch}" \
  THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE="${allow_partial_pass3a_cache}" \
  python -m scripts.agent_data.pipeline run \
    --api_base "${API_BASE}" \
    --model "${MODEL}" \
    --videos_jsonl "${videos_jsonl}" \
    --num_videos "${num_videos}" \
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

log "COMPLETE batch24579 rerun from pass1b on 8-card endpoint"
