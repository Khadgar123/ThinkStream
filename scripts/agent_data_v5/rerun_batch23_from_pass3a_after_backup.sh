#!/usr/bin/env bash
# Wait for an external backup-complete marker, restart batch2 and batch3 from
# pass3a with the correct per-batch environment, then optionally start batch1
# after both reruns finish successfully.
#
# Typical use:
#   rm -f data/agent_v5/backups/batch23_pass3a_backup.done
#   BACKUP_DONE_FILE=data/agent_v5/backups/batch23_pass3a_backup.done \
#     bash scripts/agent_data_v5/rerun_batch23_from_pass3a_after_backup.sh
#
# The backup program should touch BACKUP_DONE_FILE after it has finished.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream}"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_DIR="${LAUNCH_DIR:-${PROJECT_ROOT}/data/agent_v5/audits/rerun_batch23_from3a_${RUN_ID}}"
BACKUP_DONE_FILE="${BACKUP_DONE_FILE:-${PROJECT_ROOT}/data/agent_v5/backups/batch23_pass3a_backup.done}"
BACKUP_PID="${BACKUP_PID:-}"
BACKUP_WAIT_TIMEOUT_SEC="${BACKUP_WAIT_TIMEOUT_SEC:-0}"
BACKUP_POLL_SEC="${BACKUP_POLL_SEC:-30}"
REQUIRE_FRESH_BACKUP_MARKER="${REQUIRE_FRESH_BACKUP_MARKER:-1}"
SKIP_BACKUP_WAIT="${SKIP_BACKUP_WAIT:-0}"

CHECK_MODELS="${CHECK_MODELS:-1}"
VALIDATE_JSON="${VALIDATE_JSON:-1}"
WAIT_FOR_VLLM_IDLE="${WAIT_FOR_VLLM_IDLE:-1}"
VLLM_IDLE_TIMEOUT_SEC="${VLLM_IDLE_TIMEOUT_SEC:-0}"
VLLM_IDLE_POLL_SEC="${VLLM_IDLE_POLL_SEC:-30}"
VLLM_IDLE_MAX_INFLIGHT="${VLLM_IDLE_MAX_INFLIGHT:-0}"
KEEP_TMUX_OPEN="${KEEP_TMUX_OPEN:-1}"
DRY_RUN="${DRY_RUN:-0}"

RUN_BATCH1_AFTER_BATCH23="${RUN_BATCH1_AFTER_BATCH23:-1}"
WAIT_BATCH23_POLL_SEC="${WAIT_BATCH23_POLL_SEC:-60}"
WAIT_BATCH23_TIMEOUT_SEC="${WAIT_BATCH23_TIMEOUT_SEC:-0}"
BATCH1_FORCE_RERUN_FROM="${BATCH1_FORCE_RERUN_FROM:-}"
BATCH1_SKIP_PASS="${BATCH1_SKIP_PASS:-}"
BATCH1_NUM_VIDEOS="${BATCH1_NUM_VIDEOS:-312}"
BATCH1_CONCURRENCY="${BATCH1_CONCURRENCY:-1024}"
BATCH1_MAX_MODEL_LEN="${BATCH1_MAX_MODEL_LEN:-65536}"
BATCH1_WAIT_FOR_VLLM_IDLE="${BATCH1_WAIT_FOR_VLLM_IDLE:-1}"

START_EPOCH="$(date +%s)"

usage() {
    sed -n '1,80p' "$0"
    cat <<'EOF'

Environment knobs:
  BACKUP_DONE_FILE              Marker touched by the backup program.
  BACKUP_PID                    Optional backup process PID; fail if it exits without marker.
  BACKUP_WAIT_TIMEOUT_SEC       0 means wait forever.
  REQUIRE_FRESH_BACKUP_MARKER   1 requires marker mtime >= this script start.
  SKIP_BACKUP_WAIT              1 skips marker waiting.
  WAIT_FOR_VLLM_IDLE            1 waits per endpoint until running+waiting <= max.
  VLLM_IDLE_MAX_INFLIGHT        Default 0.
  RUN_BATCH1_AFTER_BATCH23      1 starts batch1 after batch2+batch3 finish.
  BATCH1_FORCE_RERUN_FROM       Empty resumes; set 1a to overwrite batch1.
  BATCH1_SKIP_PASS              Optional skip_pass list for batch1, e.g. "1 2".
  DRY_RUN                       1 validates and writes runner scripts, but does not start tmux.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "${LAUNCH_DIR}"
LAUNCH_LOG="${LAUNCH_DIR}/launcher.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

count_json_files() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        echo 0
        return
    fi
    find "${dir}" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' '
}

wait_for_backup_marker() {
    if [[ "${SKIP_BACKUP_WAIT}" == "1" ]]; then
        log "SKIP_BACKUP_WAIT=1; not waiting for backup marker"
        return
    fi

    log "waiting for backup marker: ${BACKUP_DONE_FILE}"
    local wait_start now marker_mtime
    wait_start="$(date +%s)"
    while true; do
        if [[ -f "${BACKUP_DONE_FILE}" ]]; then
            marker_mtime="$(stat -c %Y "${BACKUP_DONE_FILE}")"
            if [[ "${REQUIRE_FRESH_BACKUP_MARKER}" != "1" || "${marker_mtime}" -ge "${START_EPOCH}" ]]; then
                log "backup marker accepted: ${BACKUP_DONE_FILE}"
                return
            fi
            log "found stale backup marker; waiting for fresh touch: ${BACKUP_DONE_FILE}"
        fi

        if [[ -n "${BACKUP_PID}" ]] && ! kill -0 "${BACKUP_PID}" 2>/dev/null; then
            die "backup PID ${BACKUP_PID} exited before a valid marker appeared"
        fi

        if [[ "${BACKUP_WAIT_TIMEOUT_SEC}" != "0" ]]; then
            now="$(date +%s)"
            if (( now - wait_start > BACKUP_WAIT_TIMEOUT_SEC )); then
                die "timeout waiting for backup marker: ${BACKUP_DONE_FILE}"
            fi
        fi
        sleep "${BACKUP_POLL_SEC}"
    done
}

ensure_no_pipeline_running() {
    local batch="$1"
    local videos_jsonl="$2"
    local running
    running="$(ps -ef | grep -F 'python -m scripts.agent_data_v5.pipeline run' | grep -F "${videos_jsonl}" | grep -v grep || true)"
    if [[ -n "${running}" ]]; then
        printf '%s\n' "${running}"
        die "${batch}: pipeline already running for ${videos_jsonl}"
    fi
}

validate_batch_inputs() {
    local batch="$1"
    local root="$2"
    local videos_jsonl="$3"
    local expected="$4"

    [[ -s "${videos_jsonl}" ]] || die "${batch}: missing videos_jsonl: ${videos_jsonl}"
    local n_videos
    n_videos="$(wc -l < "${videos_jsonl}" | tr -d ' ')"
    [[ "${n_videos}" == "${expected}" ]] || die "${batch}: videos_jsonl has ${n_videos}, expected ${expected}"

    local n_1b n_2
    n_1b="$(count_json_files "${root}/evidence_1b")"
    n_2="$(count_json_files "${root}/rollout")"
    [[ "${n_1b}" == "${expected}" ]] || die "${batch}: evidence_1b has ${n_1b}, expected ${expected}"
    [[ "${n_2}" == "${expected}" ]] || die "${batch}: rollout has ${n_2}, expected ${expected}"

    BATCH="${batch}" ROOT="${root}" EXPECTED="${expected}" VALIDATE_JSON="${VALIDATE_JSON}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from scripts.agent_data_v5.cache_version import STAGE_VERSIONS

batch = os.environ["BATCH"]
root = Path(os.environ["ROOT"])
expected = int(os.environ["EXPECTED"])
validate_json = os.environ.get("VALIDATE_JSON", "1") == "1"

checks = [
    ("1b", "evidence_1b"),
    ("2", "rollout"),
]
for stage, subdir in checks:
    path = root / subdir
    files = sorted(path.glob("*.json"))
    if len(files) != expected:
        raise SystemExit(f"{batch}: {subdir} count {len(files)} != {expected}")
    version_path = path / "_version"
    want = STAGE_VERSIONS[stage]
    got = version_path.read_text().strip() if version_path.exists() else ""
    if got != want:
        raise SystemExit(f"{batch}: {subdir}/_version {got!r} != {want!r}")
    if validate_json:
        empty = []
        bad = []
        for f in files:
            if f.stat().st_size == 0:
                empty.append(f.name)
                continue
            try:
                json.load(open(f))
            except Exception as exc:
                bad.append((f.name, str(exc)[:120]))
        if empty or bad:
            raise SystemExit(
                f"{batch}: {subdir} invalid json empty={len(empty)} bad={len(bad)} "
                f"examples={empty[:3] or bad[:3]}"
            )
print(f"{batch}: prerequisites ok ({expected} videos, evidence_1b+rollout valid)")
PY

    ensure_no_pipeline_running "${batch}" "${videos_jsonl}"
}

check_model_endpoint() {
    local batch="$1"
    local api_base="$2"
    local model="$3"
    if [[ "${CHECK_MODELS}" != "1" ]]; then
        log "${batch}: CHECK_MODELS=0; skipping /v1/models check"
        return
    fi
    BATCH="${batch}" API_BASE="${api_base}" MODEL="${model}" "${PYTHON}" - <<'PY'
import json
import os
import urllib.request

batch = os.environ["BATCH"]
api_base = os.environ["API_BASE"].rstrip("/")
model = os.environ["MODEL"]
url = f"{api_base}/models"
with urllib.request.urlopen(url, timeout=10) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
models = [str(item.get("id", "")) for item in payload.get("data", [])]
if model not in models:
    raise SystemExit(f"{batch}: model {model!r} not listed by {url}: {models}")
max_lens = {str(item.get("id", "")): item.get("max_model_len") for item in payload.get("data", [])}
print(f"{batch}: model endpoint ok; max_model_len={max_lens.get(model)}")
PY
}

write_runner() {
    local batch="$1"
    local run_script="$2"
    local batch_root="$3"
    local videos_jsonl="$4"
    local api_base="$5"
    local metrics_url="$6"
    local model="$7"
    local max_model_len="$8"
    local concurrency="$9"
    local runner_path="${LAUNCH_DIR}/run_${batch}_from3a.sh"
    local log_path="${batch_root}/logs/pass3a_rerun.${RUN_ID}.log"
    local runner_log="${batch_root}/logs/runner_from3a.${RUN_ID}.log"

    cat > "${runner_path}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"

RUN_ID="${RUN_ID}"
BATCH="${batch}"
BATCH_ROOT="${batch_root}"
RUN_SCRIPT="${run_script}"
VIDEOS_JSONL="${videos_jsonl}"
API_BASE="${api_base}"
METRICS_URL="${metrics_url}"
MODEL="${model}"
MAX_MODEL_LEN="${max_model_len}"
CONCURRENCY="${concurrency}"
LOG_PATH="${log_path}"
RUNNER_LOG="${runner_log}"
WAIT_FOR_VLLM_IDLE="${WAIT_FOR_VLLM_IDLE}"
VLLM_IDLE_TIMEOUT_SEC="${VLLM_IDLE_TIMEOUT_SEC}"
VLLM_IDLE_POLL_SEC="${VLLM_IDLE_POLL_SEC}"
VLLM_IDLE_MAX_INFLIGHT="${VLLM_IDLE_MAX_INFLIGHT}"
KEEP_TMUX_OPEN="${KEEP_TMUX_OPEN}"

export NO_PROXY="\${NO_PROXY:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9"
export no_proxy="\${no_proxy:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9"

mkdir -p "\${BATCH_ROOT}/logs"
exec > >(tee -a "\${RUNNER_LOG}") 2>&1

log() {
    echo "[\$(date '+%F %T')] [\${BATCH}] \$*"
}

metric_value() {
    local metric="\$1"
    curl -sS -m 5 "\${METRICS_URL}" | awk -v metric="\${metric}" -v model="\${MODEL}" '
        index(\$0, metric) && index(\$0, model) {value = \$NF}
        END {if (value != "") print value}
    '
}

wait_vllm_idle() {
    if [[ "\${WAIT_FOR_VLLM_IDLE}" != "1" ]]; then
        log "WAIT_FOR_VLLM_IDLE=0; starting immediately"
        return
    fi
    local start now running waiting total
    start="\$(date +%s)"
    while true; do
        running="\$(metric_value 'vllm:num_requests_running' || true)"
        waiting="\$(metric_value 'vllm:num_requests_waiting' || true)"
        if [[ -n "\${running}" && -n "\${waiting}" ]]; then
            total="\$(awk -v r="\${running}" -v w="\${waiting}" 'BEGIN { printf "%d", r + w }')"
            if (( total <= VLLM_IDLE_MAX_INFLIGHT )); then
                log "vLLM idle enough: running=\${running}, waiting=\${waiting}"
                return
            fi
            log "waiting for vLLM idle: running=\${running}, waiting=\${waiting}, max=\${VLLM_IDLE_MAX_INFLIGHT}"
        else
            log "metrics unavailable from \${METRICS_URL}; retrying"
        fi
        if [[ "\${VLLM_IDLE_TIMEOUT_SEC}" != "0" ]]; then
            now="\$(date +%s)"
            if (( now - start > VLLM_IDLE_TIMEOUT_SEC )); then
                log "ERROR: timeout waiting for vLLM idle"
                exit 1
            fi
        fi
        sleep "\${VLLM_IDLE_POLL_SEC}"
    done
}

wait_vllm_idle

export LOG_PATH
log "runner_log=\${RUNNER_LOG}"
log "pipeline_log=\${LOG_PATH}"
export API_BASE
export MODEL
export VIDEOS_JSONL
export NUM_VIDEOS=500
export THINKSTREAM_VLLM_MAX_MODEL_LEN="\${MAX_MODEL_LEN}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS1A_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS1B_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS2_ROLLOUT_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3A_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3B_VISIBILITY_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3C_CONCURRENT="\${CONCURRENCY}"

log "starting pass3a rerun: \${RUN_SCRIPT}"
set +e
bash "\${RUN_SCRIPT}" --skip_pass 1 2 --force_rerun_from 3a
rc="\$?"
set -e
log "EXIT:\${rc}"
if [[ "\${KEEP_TMUX_OPEN}" == "1" ]]; then
    exec bash
fi
exit "\${rc}"
EOF
    chmod +x "${runner_path}"
    echo "${runner_path}"
}

start_tmux_runner() {
    local session="$1"
    local runner="$2"
    if tmux has-session -t "${session}" 2>/dev/null; then
        die "tmux session already exists: ${session}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN=1; would start tmux session ${session}: ${runner}"
        return
    fi
    tmux new-session -d -s "${session}" "bash '${runner}'"
    log "started tmux session ${session}: ${runner}"
}

wait_for_runner_success() {
    local batch="$1"
    local session="$2"
    local runner_log="$3"
    local pipeline_log="$4"
    local start now
    start="$(date +%s)"
    log "waiting for ${batch} completion: session=${session}, runner_log=${runner_log}"
    while true; do
        if [[ -f "${pipeline_log}" ]] && grep -q 'PIPELINE COMPLETE' "${pipeline_log}"; then
            log "${batch} pipeline complete"
            return
        fi
        if [[ -f "${runner_log}" ]] && grep -q 'EXIT:0' "${runner_log}"; then
            log "${batch}: runner exited 0; waiting for PIPELINE COMPLETE marker in ${pipeline_log}"
        fi
        if [[ -f "${runner_log}" ]] && grep -Eq 'EXIT:([1-9][0-9]*|-[0-9]+)' "${runner_log}"; then
            tail -n 80 "${runner_log}" || true
            die "${batch}: runner exited non-zero; see ${runner_log}"
        fi
        if ! tmux has-session -t "${session}" 2>/dev/null; then
            tail -n 80 "${runner_log}" 2>/dev/null || true
            die "${batch}: tmux session disappeared before successful completion: ${session}"
        fi
        if [[ "${WAIT_BATCH23_TIMEOUT_SEC}" != "0" ]]; then
            now="$(date +%s)"
            if (( now - start > WAIT_BATCH23_TIMEOUT_SEC )); then
                die "${batch}: timeout waiting for completion"
            fi
        fi
        sleep "${WAIT_BATCH23_POLL_SEC}"
    done
}

count_nonempty_jsonl() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo 0
        return
    fi
    "${PYTHON}" - "$path" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open("rb") as f:
    print(sum(1 for line in f if line.strip()))
PY
}

validate_completed_batch() {
    local batch="$1"
    local root="$2"
    local n_train n_val n_test n_rl

    [[ -s "${root}/final/pipeline_stats.json" ]] || die "${batch}: missing final/pipeline_stats.json"
    [[ -s "${root}/final/train_sft_messages.jsonl" ]] || die "${batch}: missing final/train_sft_messages.jsonl"
    [[ -s "${root}/final/val_messages.jsonl" ]] || die "${batch}: missing final/val_messages.jsonl"
    [[ -s "${root}/final/test_messages.jsonl" ]] || die "${batch}: missing final/test_messages.jsonl"
    [[ -s "${root}/final/train_rl_trajectories.jsonl" ]] || die "${batch}: missing final/train_rl_trajectories.jsonl"
    [[ -s "${root}/rendered/ts_image/train_rl_multi_q.parquet" ]] || die "${batch}: missing ts_image train parquet"
    [[ -s "${root}/rendered/video_meta/train_rl_multi_q.parquet" ]] || die "${batch}: missing video_meta train parquet"

    n_train="$(count_nonempty_jsonl "${root}/final/train_sft_messages.jsonl")"
    n_val="$(count_nonempty_jsonl "${root}/final/val_messages.jsonl")"
    n_test="$(count_nonempty_jsonl "${root}/final/test_messages.jsonl")"
    n_rl="$(count_nonempty_jsonl "${root}/final/train_rl_trajectories.jsonl")"
    log "${batch} final counts: train_sft_messages=${n_train}, val_messages=${n_val}, test_messages=${n_test}, train_rl_trajectories=${n_rl}"
    [[ "${n_train}" -gt 0 ]] || die "${batch}: empty train_sft_messages"
    [[ "${n_val}" -gt 0 ]] || die "${batch}: empty val_messages"
    [[ "${n_test}" -gt 0 ]] || die "${batch}: empty test_messages"
    [[ "${n_rl}" -gt 0 ]] || die "${batch}: empty train_rl_trajectories"
}

validate_batch1_inputs() {
    local root="$1"
    local videos_jsonl="$2"
    [[ -s "${videos_jsonl}" ]] || die "batch1: missing videos_jsonl: ${videos_jsonl}"
    local n_videos
    n_videos="$(wc -l < "${videos_jsonl}" | tr -d ' ')"
    [[ "${n_videos}" == "${BATCH1_NUM_VIDEOS}" ]] || die "batch1: videos_jsonl has ${n_videos}, expected ${BATCH1_NUM_VIDEOS}"
    ensure_no_pipeline_running "batch1" "${videos_jsonl}"

    if [[ "${VALIDATE_JSON}" == "1" && -d "${root}/evidence_1a" ]]; then
        BATCH="batch1" ROOT="${root}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["ROOT"])
files = sorted((root / "evidence_1a").glob("*.json"))
empty = []
bad = []
for f in files:
    if f.stat().st_size == 0:
        empty.append(f.name)
        continue
    try:
        json.load(open(f))
    except Exception as exc:
        bad.append((f.name, str(exc)[:120]))
if empty or bad:
    raise SystemExit(f"batch1: evidence_1a invalid empty={len(empty)} bad={len(bad)} examples={empty[:3] or bad[:3]}")
print(f"batch1: resume cache ok; evidence_1a={len(files)}")
PY
    fi
}

write_batch1_runner() {
    local batch_root="$1"
    local videos_jsonl="$2"
    local api_base="$3"
    local metrics_url="$4"
    local model="$5"
    local runner_path="${LAUNCH_DIR}/run_batch1_after_batch23.sh"
    local log_path="${batch_root}/logs/after_batch23.${RUN_ID}.log"
    local runner_log="${batch_root}/logs/runner_after_batch23.${RUN_ID}.log"

    cat > "${runner_path}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"

RUN_ID="${RUN_ID}"
BATCH="batch1"
BATCH_ROOT="${batch_root}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/agent_data_v5/run_batch1_pipeline.sh"
VIDEOS_JSONL="${videos_jsonl}"
API_BASE="${api_base}"
METRICS_URL="${metrics_url}"
MODEL="${model}"
LOG_PATH="${log_path}"
RUNNER_LOG="${runner_log}"
MAX_MODEL_LEN="${BATCH1_MAX_MODEL_LEN}"
CONCURRENCY="${BATCH1_CONCURRENCY}"
BATCH1_FORCE_RERUN_FROM="${BATCH1_FORCE_RERUN_FROM}"
BATCH1_SKIP_PASS="${BATCH1_SKIP_PASS}"
NUM_VIDEOS="${BATCH1_NUM_VIDEOS}"
WAIT_FOR_VLLM_IDLE="${BATCH1_WAIT_FOR_VLLM_IDLE}"
VLLM_IDLE_TIMEOUT_SEC="${VLLM_IDLE_TIMEOUT_SEC}"
VLLM_IDLE_POLL_SEC="${VLLM_IDLE_POLL_SEC}"
VLLM_IDLE_MAX_INFLIGHT="${VLLM_IDLE_MAX_INFLIGHT}"
KEEP_TMUX_OPEN="${KEEP_TMUX_OPEN}"

export NO_PROXY="\${NO_PROXY:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9"
export no_proxy="\${no_proxy:-},127.0.0.1,localhost,10.16.12.175,10.16.18.9"

mkdir -p "\${BATCH_ROOT}/logs"
exec > >(tee -a "\${RUNNER_LOG}") 2>&1

log() {
    echo "[\$(date '+%F %T')] [\${BATCH}] \$*"
}

metric_value() {
    local metric="\$1"
    curl -sS -m 5 "\${METRICS_URL}" | awk -v metric="\${metric}" -v model="\${MODEL}" '
        index(\$0, metric) && index(\$0, model) {value = \$NF}
        END {if (value != "") print value}
    '
}

wait_vllm_idle() {
    if [[ "\${WAIT_FOR_VLLM_IDLE}" != "1" ]]; then
        log "WAIT_FOR_VLLM_IDLE=0; starting immediately"
        return
    fi
    local start now running waiting total
    start="\$(date +%s)"
    while true; do
        running="\$(metric_value 'vllm:num_requests_running' || true)"
        waiting="\$(metric_value 'vllm:num_requests_waiting' || true)"
        if [[ -n "\${running}" && -n "\${waiting}" ]]; then
            total="\$(awk -v r="\${running}" -v w="\${waiting}" 'BEGIN { printf "%d", r + w }')"
            if (( total <= VLLM_IDLE_MAX_INFLIGHT )); then
                log "vLLM idle enough: running=\${running}, waiting=\${waiting}"
                return
            fi
            log "waiting for vLLM idle: running=\${running}, waiting=\${waiting}, max=\${VLLM_IDLE_MAX_INFLIGHT}"
        else
            log "metrics unavailable from \${METRICS_URL}; retrying"
        fi
        if [[ "\${VLLM_IDLE_TIMEOUT_SEC}" != "0" ]]; then
            now="\$(date +%s)"
            if (( now - start > VLLM_IDLE_TIMEOUT_SEC )); then
                log "ERROR: timeout waiting for vLLM idle"
                exit 1
            fi
        fi
        sleep "\${VLLM_IDLE_POLL_SEC}"
    done
}

wait_vllm_idle

export LOG_PATH
log "runner_log=\${RUNNER_LOG}"
log "pipeline_log=\${LOG_PATH}"
export API_BASE
export MODEL
export VIDEOS_JSONL
export NUM_VIDEOS
export THINKSTREAM_VLLM_MAX_MODEL_LEN="\${MAX_MODEL_LEN}"
export THINKSTREAM_VLLM_MAX_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS1A_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS1B_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS2_ROLLOUT_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3A_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3B_VISIBILITY_CONCURRENT="\${CONCURRENCY}"
export THINKSTREAM_PASS3C_CONCURRENT="\${CONCURRENCY}"

args=()
if [[ -n "\${BATCH1_SKIP_PASS}" ]]; then
    read -r -a skip_parts <<< "\${BATCH1_SKIP_PASS}"
    args+=(--skip_pass "\${skip_parts[@]}")
fi
if [[ -n "\${BATCH1_FORCE_RERUN_FROM}" ]]; then
    args+=(--force_rerun_from "\${BATCH1_FORCE_RERUN_FROM}")
fi

log "starting batch1: \${RUN_SCRIPT} \${args[*]}"
set +e
bash "\${RUN_SCRIPT}" "\${args[@]}"
rc="\$?"
set -e
log "EXIT:\${rc}"
if [[ "\${KEEP_TMUX_OPEN}" == "1" ]]; then
    exec bash
fi
exit "\${rc}"
EOF
    chmod +x "${runner_path}"
    echo "${runner_path}"
}

main() {
    log "launch_dir=${LAUNCH_DIR}"
    log "backup_done_file=${BACKUP_DONE_FILE}"

    wait_for_backup_marker

    local batch2_root="${PROJECT_ROOT}/data/agent_v5/batch2"
    local batch3_root="${PROJECT_ROOT}/data/agent_v5/batch3"
    local batch1_root="${PROJECT_ROOT}/data/agent_v5/batch1"
    local batch2_videos="${PROJECT_ROOT}/data/agent_v5/batch2_videos.jsonl"
    local batch3_videos="${PROJECT_ROOT}/data/agent_v5/batch3_videos.jsonl"
    local batch1_videos="${PROJECT_ROOT}/data/agent_v5/batch1/selected_videos.jsonl"
    local batch2_api="http://10.16.12.175:8000/v1"
    local batch3_api="http://10.16.12.175:8000/v1"
    local batch1_api="http://10.16.12.175:8000/v1"
    local batch2_metrics="http://10.16.12.175:8000/metrics"
    local batch3_metrics="http://10.16.12.175:8000/metrics"
    local batch1_metrics="http://10.16.12.175:8000/metrics"
    local batch2_model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
    local batch3_model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
    local batch1_model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"

    validate_batch_inputs "batch2" "${batch2_root}" "${batch2_videos}" 500
    validate_batch_inputs "batch3" "${batch3_root}" "${batch3_videos}" 500
    check_model_endpoint "batch2" "${batch2_api}" "${batch2_model}"
    check_model_endpoint "batch3" "${batch3_api}" "${batch3_model}"
    if [[ "${RUN_BATCH1_AFTER_BATCH23}" == "1" ]]; then
        validate_batch1_inputs "${batch1_root}" "${batch1_videos}"
        check_model_endpoint "batch1" "${batch1_api}" "${batch1_model}"
    fi

    local batch2_runner batch3_runner batch1_runner batch2_session batch3_session batch1_session
    batch2_runner="$(write_runner \
        "batch2" \
        "${PROJECT_ROOT}/scripts/agent_data_v5/run_batch2_pipeline.sh" \
        "${batch2_root}" \
        "${batch2_videos}" \
        "${batch2_api}" \
        "${batch2_metrics}" \
        "${batch2_model}" \
        "65536" \
        "1024")"
    batch3_runner="$(write_runner \
        "batch3" \
        "${PROJECT_ROOT}/scripts/agent_data_v5/run_batch3_pipeline.sh" \
        "${batch3_root}" \
        "${batch3_videos}" \
        "${batch3_api}" \
        "${batch3_metrics}" \
        "${batch3_model}" \
        "65536" \
        "1024")"

    batch2_session="batch2_from3a_${RUN_ID}"
    batch3_session="batch3_from3a_${RUN_ID}"
    batch1_session="batch1_after_batch23_${RUN_ID}"
    if [[ "${RUN_BATCH1_AFTER_BATCH23}" == "1" ]]; then
        batch1_runner="$(write_batch1_runner \
            "${batch1_root}" \
            "${batch1_videos}" \
            "${batch1_api}" \
            "${batch1_metrics}" \
            "${batch1_model}")"
    fi

    cat > "${LAUNCH_DIR}/launch_manifest.txt" <<EOF
run_id=${RUN_ID}
backup_done_file=${BACKUP_DONE_FILE}
batch2_session=${batch2_session}
batch2_runner=${batch2_runner}
batch2_log=${batch2_root}/logs/pass3a_rerun.${RUN_ID}.log
batch2_runner_log=${batch2_root}/logs/runner_from3a.${RUN_ID}.log
batch3_session=${batch3_session}
batch3_runner=${batch3_runner}
batch3_log=${batch3_root}/logs/pass3a_rerun.${RUN_ID}.log
batch3_runner_log=${batch3_root}/logs/runner_from3a.${RUN_ID}.log
batch23_command_args=--skip_pass 1 2 --force_rerun_from 3a
run_batch1_after_batch23=${RUN_BATCH1_AFTER_BATCH23}
batch1_session=${batch1_session}
batch1_runner=${batch1_runner:-}
batch1_log=${batch1_root}/logs/after_batch23.${RUN_ID}.log
batch1_runner_log=${batch1_root}/logs/runner_after_batch23.${RUN_ID}.log
batch1_force_rerun_from=${BATCH1_FORCE_RERUN_FROM}
batch1_skip_pass=${BATCH1_SKIP_PASS}
EOF

    start_tmux_runner "${batch3_session}" "${batch3_runner}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        start_tmux_runner "${batch2_session}" "${batch2_runner}"
    else
        wait_for_runner_success \
            "batch3" \
            "${batch3_session}" \
            "${batch3_root}/logs/runner_from3a.${RUN_ID}.log" \
            "${batch3_root}/logs/pass3a_rerun.${RUN_ID}.log"
        validate_completed_batch "batch3" "${batch3_root}"
        start_tmux_runner "${batch2_session}" "${batch2_runner}"
    fi

    if [[ "${RUN_BATCH1_AFTER_BATCH23}" == "1" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            log "DRY_RUN=1; would wait for batch2+batch3 success before batch1"
            start_tmux_runner "${batch1_session}" "${batch1_runner}"
        else
            wait_for_runner_success \
                "batch2" \
                "${batch2_session}" \
                "${batch2_root}/logs/runner_from3a.${RUN_ID}.log" \
                "${batch2_root}/logs/pass3a_rerun.${RUN_ID}.log"
            validate_completed_batch "batch2" "${batch2_root}"
            start_tmux_runner "${batch1_session}" "${batch1_runner}"
        fi
    fi

    log "launch manifest: ${LAUNCH_DIR}/launch_manifest.txt"
}

main "$@"
