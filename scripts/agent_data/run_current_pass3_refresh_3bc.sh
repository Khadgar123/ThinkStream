#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:?API_BASE is required}"
MODEL="${MODEL:?MODEL is required}"
BATCHES="${BATCHES:?BATCHES is required, e.g. 'batch1 batch2'}"
PLAN_JSONL="${PLAN_JSONL:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REFRESH_DECISIONS="${REFRESH_DECISIONS:-inspect,rerun_3a_then_3bc}"
REFRESH_FAMILIES="${REFRESH_FAMILIES:-}"
REFRESH_ALL_SELECTED="${REFRESH_ALL_SELECTED:-0}"
RUN_REFRESH="${RUN_REFRESH:-1}"
RUN_PASS45="${RUN_PASS45:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export THINKSTREAM_VLLM_MODEL="${MODEL}"
export THINKSTREAM_VLLM_MAX_MODEL_LEN="${THINKSTREAM_VLLM_MAX_MODEL_LEN:-65536}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="${THINKSTREAM_VLLM_MAX_CONCURRENT:-1024}"
export THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET="${THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET:-64000000}"
export THINKSTREAM_PASS3A_VIDEO_CONCURRENT="${THINKSTREAM_PASS3A_VIDEO_CONCURRENT:-40}"
export THINKSTREAM_PASS3A_CONCURRENT="${THINKSTREAM_PASS3A_CONCURRENT:-1024}"
export THINKSTREAM_PASS3C_CONCURRENT="${THINKSTREAM_PASS3C_CONCURRENT:-1024}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9,10.16.10.172,10.16.11.160"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9,10.16.10.172,10.16.11.160"

GLOBAL_LOG="${PROJECT_ROOT}/data/agent_v5/audits/current_pass3_refresh_3bc_${RUN_ID}.log"
mkdir -p "$(dirname "${GLOBAL_LOG}")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${GLOBAL_LOG}"
}

videos_file_for_batch() {
  local batch="$1"
  local suffix="${batch#batch}"
  local external="${PROJECT_ROOT}/data/agent_v5/batch${suffix}_videos.jsonl"
  local registry="${PROJECT_ROOT}/data/agent_v5/${batch}/video_registry.jsonl"
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
    "${PYTHON_BIN}" - <<'PY'
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
    "task_cards": count_json(root / "task_cards"),
}
print("preflight_counts=" + " ".join(f"{k}={v}" for k, v in counts.items()))
for key in ("videos", "evidence_1a", "evidence_1b", "rollout", "task_cards"):
    if counts[key] < expected_n:
        raise SystemExit(f"{key} incomplete: {counts[key]} < {expected_n}")
PY
}

preflight_imports() {
  "${PYTHON_BIN}" - <<'PY'
import thinkstream.data.agent_protocol as ap
from scripts.agent_data.placement import design

missing = []
if not hasattr(ap, "select_recall_chunks_uniform"):
    missing.append("thinkstream.data.agent_protocol.select_recall_chunks_uniform")
for name in (
    "QUESTION_WAY_SOURCE_DISCRIMINATION",
    "QUESTION_WAY_MULTIMODAL_ALIGNMENT",
    "TASK_SUBTYPE_TARGET_FRACTION",
):
    if not hasattr(design, name):
        missing.append(f"scripts.agent_data.placement.design.{name}")
if missing:
    raise SystemExit("missing required pass3 symbols: " + ", ".join(missing))
print(f"agent_protocol={ap.__file__}")
print(f"placement_design={design.__file__}")
PY
}

batch_csv="$(printf '%s\n' ${BATCHES} | paste -sd, -)"
log "START current pass3 refresh+3bc"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "batches=${BATCHES}"
log "plan_jsonl=${PLAN_JSONL}"
log "refresh=${RUN_REFRESH} run_pass45=${RUN_PASS45} all_selected=${REFRESH_ALL_SELECTED} decisions=${REFRESH_DECISIONS} families=${REFRESH_FAMILIES:-ALL}"
log "global_log=${GLOBAL_LOG}"
preflight_imports 2>&1 | tee -a "${GLOBAL_LOG}"

if [[ "${RUN_REFRESH}" != "0" ]]; then
  refresh_report="${PROJECT_ROOT}/data/agent_v5/audits/current_pass3a_refresh_${RUN_ID}_${batch_csv//,/__}.jsonl"
  log "PASS3A_REFRESH start report=${refresh_report}"
  refresh_args=(
    --root data/agent_v5
    --decisions "${REFRESH_DECISIONS}"
    --batches "${batch_csv}"
    --families "${REFRESH_FAMILIES}"
    --api-base "${API_BASE}"
    --model "${MODEL}"
    --max-concurrent "${THINKSTREAM_PASS3A_CONCURRENT}"
    --video-concurrent "${THINKSTREAM_PASS3A_VIDEO_CONCURRENT}"
    --report "${refresh_report}"
  )
  if [[ "${REFRESH_ALL_SELECTED}" == "1" ]]; then
    refresh_args+=(--all-selected)
  else
    if [[ -z "${PLAN_JSONL}" ]]; then
      echo "PLAN_JSONL is required unless REFRESH_ALL_SELECTED=1" >&2
      exit 2
    fi
    refresh_args+=(--plan-jsonl "${PLAN_JSONL}")
  fi
  "${PYTHON_BIN}" -m scripts.agent_data.refresh_pass3a_subset \
    "${refresh_args[@]}" \
    2>&1 | tee -a "${GLOBAL_LOG}"
  log "PASS3A_REFRESH done report=${refresh_report}"
fi

for batch in ${BATCHES}; do
  batch_root="${PROJECT_ROOT}/data/agent_v5/${batch}"
  videos_jsonl="$(videos_file_for_batch "${batch}")"
  num_videos="$(wc -l < "${videos_jsonl}")"
  batch_log="${batch_root}/logs/current_pass3_3bc_${RUN_ID}.log"
  mkdir -p "${batch_root}/logs"

  log "${batch} START pass3B/3C; videos=${num_videos}; log=${batch_log}"
  {
    echo "[$(date '+%F %T')] [${batch}] START current pass3B/3C"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "api_base=${API_BASE}"
    echo "model=${MODEL}"
    echo "videos_jsonl=${videos_jsonl}"
    echo "num_videos=${num_videos}"
    echo "data_root=${batch_root}"
    echo "run_id=${RUN_ID}"
    echo "force_rerun_from=3b"
    echo "skip_pass=1 2"
    echo "allow_stale_pass2_cache=1"
    echo "allow_stale_pass3_cache=1"
    echo "run_pass45=${RUN_PASS45}"
  } | tee "${batch_log}"

  preflight_batch "${batch_root}" "${videos_jsonl}" "${num_videos}" 2>&1 | tee -a "${batch_log}"

  pipeline_env=(
    "THINKSTREAM_DATA_ROOT=${batch_root}"
    "AGENT_DATA_DIR=${batch_root}"
    "THINKSTREAM_BATCH=${batch}"
    "THINKSTREAM_ALLOW_STALE_PASS2_CACHE=1"
    "THINKSTREAM_ALLOW_STALE_PASS3_CACHE=1"
  )
  if [[ "${RUN_PASS45}" != "1" ]]; then
    pipeline_env+=("SKIP_PASS45=1")
  fi
  env "${pipeline_env[@]}" \
    "${PYTHON_BIN}" -m scripts.agent_data.pipeline run \
      --api_base "${API_BASE}" \
      --model "${MODEL}" \
      --videos_jsonl "${videos_jsonl}" \
      --num_videos "${num_videos}" \
      --skip_pass 1 2 \
      --force_rerun_from 3b \
      2>&1 | tee -a "${batch_log}"
  rc=${PIPESTATUS[0]}
  log "${batch} EXIT:${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
done

log "COMPLETE current pass3 refresh+3bc"
