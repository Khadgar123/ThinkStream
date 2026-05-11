#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
API_BASE="${API_BASE:-http://10.16.12.175:8000/v1}"
MODEL="${MODEL:-/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-4 5 7 9}"
PRESERVE_1B_BATCHES="${PRESERVE_1B_BATCHES:-4 5}"

cd "${PROJECT_ROOT}"

GLOBAL_LOG="${PROJECT_ROOT}/data/agent_v5/audits/batch4579_sequential_8card_${RUN_ID}.log"
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

contains_batch() {
  local needle="$1"
  local item
  for item in ${PRESERVE_1B_BATCHES}; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

preflight_batch() {
  local batch_root="$1"
  local videos_jsonl="$2"
  local expected_n="$3"
  local preserve_1b="$4"
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  VIDEOS_JSONL="${videos_jsonl}" \
  EXPECTED_N="${expected_n}" \
  PRESERVE_1B="${preserve_1b}" \
  python - <<'PY'
import json
import os
from pathlib import Path

from scripts.agent_data import config as c

root = Path(os.environ["THINKSTREAM_DATA_ROOT"])
videos = Path(os.environ["VIDEOS_JSONL"])
expected_n = int(os.environ["EXPECTED_N"])
preserve_1b = os.environ["PRESERVE_1B"] == "1"
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
    "samples_3c": count_json(root / "samples_3c"),
}
print("preflight_counts=" + " ".join(f"{k}={v}" for k, v in counts.items()))
if counts["videos"] != expected_n:
    raise SystemExit(f"video count mismatch: {counts['videos']} != {expected_n}")
if counts["evidence_1a"] < expected_n:
    raise SystemExit(f"evidence_1a incomplete: {counts['evidence_1a']} < {expected_n}")

if preserve_1b:
    bad = []
    for p in sorted((root / "evidence_1b").glob("*.json")):
        try:
            ev = json.loads(p.read_text())
            ev1 = json.loads((root / "evidence_1a" / p.name).read_text())
        except Exception as exc:  # noqa: BLE001
            bad.append((p.name, repr(exc)))
            continue
        if not isinstance(ev, list) or len(ev) != len(ev1):
            bad.append((p.name, f"len/type mismatch: {len(ev) if isinstance(ev, list) else type(ev).__name__} vs {len(ev1)}"))
            continue
        for i, chunk in enumerate(ev):
            if not isinstance(chunk, dict) or chunk.get("chunk_idx") != i or "time" not in chunk or "think" not in chunk:
                bad.append((p.name, f"bad chunk schema at {i}"))
                break
    if bad:
        print("bad_1b_cache=" + repr(bad[:10]))
        raise SystemExit(f"bad partial 1b cache: {len(bad)} files")
PY
}

stamp_1b_version() {
  local batch_root="$1"
  THINKSTREAM_DATA_ROOT="${batch_root}" \
  AGENT_DATA_DIR="${batch_root}" \
  python - <<'PY'
from scripts.agent_data.cache_version import write_stage_version

write_stage_version("1a")
write_stage_version("1b")
PY
}

log "START sequential batch4579 on 8-card endpoint"
log "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "api_base=${API_BASE}"
log "model=${MODEL}"
log "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
log "max_concurrent=${THINKSTREAM_VLLM_MAX_CONCURRENT}"
log "batches=${BATCHES}"
log "preserve_1b_batches=${PRESERVE_1B_BATCHES}"
log "global_log=${GLOBAL_LOG}"

for b in ${BATCHES}; do
  batch="batch${b}"
  batch_root="${PROJECT_ROOT}/data/agent_v5/${batch}"
  videos_jsonl="$(videos_file_for_batch "${b}")"
  num_videos="$(wc -l < "${videos_jsonl}")"
  batch_log="${batch_root}/logs/sequential_8card_${RUN_ID}.log"
  mkdir -p "${batch_root}/logs"

  preserve_1b=0
  force_stage="1b"
  if contains_batch "${b}"; then
    preserve_1b=1
    force_stage="2"
  fi

  log "${batch} START; preserve_1b=${preserve_1b}; force_rerun_from=${force_stage}; videos=${num_videos}; log=${batch_log}"
  {
    echo "[$(date '+%F %T')] [${batch}] START sequential 8-card"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "api_base=${API_BASE}"
    echo "model=${MODEL}"
    echo "max_model_len=${THINKSTREAM_VLLM_MAX_MODEL_LEN}"
    echo "videos_jsonl=${videos_jsonl}"
    echo "num_videos=${num_videos}"
    echo "data_root=${batch_root}"
    echo "run_id=${RUN_ID}"
    echo "preserve_1b=${preserve_1b}"
    echo "force_rerun_from=${force_stage}"
  } | tee "${batch_log}"

  set +e
  preflight_batch "${batch_root}" "${videos_jsonl}" "${num_videos}" "${preserve_1b}" 2>&1 | tee -a "${batch_log}"
  preflight_rc=${PIPESTATUS[0]}
  set -e
  if [[ ${preflight_rc} -ne 0 ]]; then
    log "${batch} PREFLIGHT_EXIT:${preflight_rc}"
    exit "${preflight_rc}"
  fi

  if [[ "${preserve_1b}" == "1" ]]; then
    log "${batch} stamping 1a/1b version to preserve partial 1b cache"
    stamp_1b_version "${batch_root}" 2>&1 | tee -a "${batch_log}"
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
    --force_rerun_from "${force_stage}" \
    2>&1 | tee -a "${batch_log}"
  rc=${PIPESTATUS[0]}
  set -e

  log "${batch} EXIT:${rc}"
  echo "[$(date '+%F %T')] [${batch}] EXIT:${rc}" | tee -a "${batch_log}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

log "COMPLETE sequential batch4579 on 8-card endpoint"
