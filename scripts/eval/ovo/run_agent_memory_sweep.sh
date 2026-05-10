#!/bin/bash
# Training-free OVO agent-memory sweep.
#
# This runs base/SFT checkpoints through the full ThinkStream streaming loop:
# the model must emit <think> + action/tool calls, MemoryState is updated
# online, recall is executed by the controller, and system compression is
# triggered by the same memory rule as eval/RL.
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
OUT_ROOT=${OUT_ROOT:-output/ovo_agent_memory_sweep_${STAMP}}
PYTHON_BIN=${PYTHON_BIN:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python}
BENCHMARK_JSON=${BENCHMARK_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json}
VIDEO_ROOT=${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench}
FRAMES_ROOT=${FRAMES_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames}
BASE_CKPT=${BASE_CKPT:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}
SFT_CKPT=${SFT_CKPT:-}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TASKS=${TASKS:-}
SAMPLE_IDS=${SAMPLE_IDS:-}
JOB_FILTER=${JOB_FILTER:-}
JOB_MAX_PIXELS_OVERRIDE=${JOB_MAX_PIXELS_OVERRIDE:-}
MAX_JOB_CHUNK=${MAX_JOB_CHUNK:-}
MAX_AGENT_JOBS=${MAX_AGENT_JOBS:-}
PREFER_SHORT_JOBS=${PREFER_SHORT_JOBS:-0}
PREFER_LONG_JOBS=${PREFER_LONG_JOBS:-0}
RECENT_THINKS_TOKEN_BUDGET=${RECENT_THINKS_TOKEN_BUDGET:-}
RECALL_TEXT_MAX_CHARS=${RECALL_TEXT_MAX_CHARS:-}
SUMMARY_TOKENS_MAX=${SUMMARY_TOKENS_MAX:-}
N_PER_TASK=${N_PER_TASK:-1}
SCORING=${SCORING:-lenient}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
MM_CACHE_GB=${MM_CACHE_GB:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
COMPRESS_MAX_NEW_TOKENS=${COMPRESS_MAX_NEW_TOKENS:-512}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
SAVE_STEP_TRACE=${SAVE_STEP_TRACE:-1}
PROGRESS_EVERY=${PROGRESS_EVERY:-25}
REQUIRE_FRAME_CACHE=${REQUIRE_FRAME_CACHE:-1}
DROP_INCOMPLETE_FRAME_CACHE=${DROP_INCOMPLETE_FRAME_CACHE:-1}

mkdir -p "${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/status.tsv"
MATRIX="${OUT_ROOT}/matrix_summary.tsv"
printf "time\tstate\tjob\tgpu\tdetail\n" > "${STATUS}"
printf "time\tjob\toverall\tcontent\tstrict\ttargeted\tno_early\tno_late\ton_time\trecall_per_probe\trecall_hit\tcompress_per_probe\tcompress_success\tstable_think_rate\tformat_error_rate\taction_error_rate\tframe_cache_misses\tsteps\tseconds\tsteps_per_sec\tprompt_tokens_max\tthink_tokens_max\n" > "${MATRIX}"

# tag|ckpt|memory_mode|compress_mode|retriever|render_layout|frame_protocol|memory_position|max_pixels|window_chunks|frames_per_chunk
JOBS=(
  "base_none_w16|${BASE_CKPT}|none|none|none|standard_query_last|video_meta|before_visual|220000|16|2"
  "base_full_w16|${BASE_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|220000|16|2"
  "base_full_nocomp_w16|${BASE_CKPT}|full|none|bm25|standard_query_last|video_meta|before_visual|220000|16|2"
  "base_norecall_w16|${BASE_CKPT}|no_recall|system|none|standard_query_last|video_meta|before_visual|220000|16|2"
)

if [[ -n "${SFT_CKPT}" ]]; then
  JOBS+=(
    "sft_none_w16|${SFT_CKPT}|none|none|none|standard_query_last|video_meta|before_visual|220000|16|2"
    "sft_full_w16|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|220000|16|2"
    "sft_full_nocomp_w16|${SFT_CKPT}|full|none|bm25|standard_query_last|video_meta|before_visual|220000|16|2"
    "sft_norecall_w16|${SFT_CKPT}|no_recall|system|none|standard_query_last|video_meta|before_visual|220000|16|2"
    "sft_full_w8|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|220000|8|2"
    "sft_full_w32|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|220000|32|2"
    "sft_full_1fps_w16|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|220000|16|1"
    "sft_full_p180_w16|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|180000|16|2"
    "sft_full_p360_w16|${SFT_CKPT}|full|system|bm25|standard_query_last|video_meta|before_visual|360000|16|2"
  )
fi

if [[ -n "${JOB_FILTER}" ]]; then
  IFS=',' read -r -a FILTER_TAGS <<< "${JOB_FILTER}"
  FILTERED_JOBS=()
  for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    for want in "${FILTER_TAGS[@]}"; do
      if [[ "${tag}" == "${want}" ]]; then
        FILTERED_JOBS+=("${job}")
        break
      fi
    done
  done
  JOBS=("${FILTERED_JOBS[@]}")
fi

summarize_json() {
  local tag="$1"
  local out_json="$2"
  local seconds="$3"
  "${PYTHON_BIN}" - "$tag" "$out_json" "$seconds" <<'PY'
import json, sys
tag, path, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(path))
s = d.get("summary", {})
h = s.get("health", {})
a = h.get("answer", {})
r = h.get("recall", {})
c = h.get("compression", {})
ab = h.get("format_runtime", h.get("abnormal", {}))
steps = int(ab.get('steps', h.get('total_steps', 0)) or 0)
steps_per_sec = steps / seconds if seconds > 0 else 0.0
print("\t".join([
    tag,
    f"{s.get('overall', 0.0):.6f}",
    f"{a.get('content_acc', 0.0):.6f}",
    f"{a.get('strict_acc', 0.0):.6f}",
    f"{a.get('targeted_acc', 0.0):.6f}",
    f"{a.get('no_early_acc', 0.0):.6f}",
    f"{a.get('no_late_acc', 0.0):.6f}",
    f"{a.get('on_time_acc', 0.0):.6f}",
    f"{r.get('events_per_probe', 0.0):.6f}",
    f"{r.get('support_hit_rate', 0.0):.6f}",
    f"{c.get('events_per_probe', 0.0):.6f}",
    f"{c.get('success_rate', 0.0):.6f}",
    f"{ab.get('stable_think_sample_rate', 0.0):.6f}",
    f"{ab.get('format_violation_rate', 0.0):.6f}",
    f"{ab.get('action_space_error_rate', 0.0):.6f}",
    str(ab.get('frame_cache_misses', 0)),
    str(steps),
    f"{seconds:.1f}",
    f"{steps_per_sec:.3f}",
    str(ab.get('prompt_tokens_max', 0)),
    str(ab.get('think_tokens_max', 0)),
]))
PY
}

run_one() {
  local job="$1"
  local gpu="$2"
  IFS='|' read -r tag ckpt memory_mode compress_mode retriever layout protocol memory_position max_pixels window_chunks frames_per_chunk <<< "${job}"
  if [[ -n "${JOB_MAX_PIXELS_OVERRIDE}" ]]; then
    max_pixels="${JOB_MAX_PIXELS_OVERRIDE}"
  fi
  local out_dir="${OUT_ROOT}/${tag}"
  local out_json="${out_dir}/result.json"
  local log="${OUT_ROOT}/logs/${tag}_gpu${gpu}.log"
  mkdir -p "${out_dir}"

  printf "%s\tSTART\t%s\t%s\t%s %s %s mem=%s comp=%s ret=%s W=%s fpc=%s px=%s\n" \
    "$(date +%F_%T)" "${tag}" "${gpu}" "${layout}" "${protocol}" "${memory_position}" \
    "${memory_mode}" "${compress_mode}" "${retriever}" "${window_chunks}" "${frames_per_chunk}" "${max_pixels}" >> "${STATUS}"

  local extra=()
  if [[ -n "${TASKS}" ]]; then
    extra+=("--tasks" "${TASKS}")
  fi
  if [[ -n "${SAMPLE_IDS}" ]]; then
    extra+=("--sample_ids" "${SAMPLE_IDS}")
  fi
  if [[ "${SAVE_STEP_TRACE}" == "1" ]]; then
    extra+=("--save_step_trace")
  fi
  if [[ "${REQUIRE_FRAME_CACHE}" == "1" ]]; then
    extra+=("--require_frame_cache")
  fi
  if [[ "${DROP_INCOMPLETE_FRAME_CACHE}" == "1" ]]; then
    extra+=("--drop_incomplete_frame_cache")
  fi
  if [[ -n "${MAX_JOB_CHUNK}" ]]; then
    extra+=("--max_job_chunk" "${MAX_JOB_CHUNK}")
  fi
  if [[ -n "${MAX_AGENT_JOBS}" ]]; then
    extra+=("--max_agent_jobs" "${MAX_AGENT_JOBS}")
  fi
  if [[ "${PREFER_SHORT_JOBS}" == "1" ]]; then
    extra+=("--prefer_short_jobs")
  fi
  if [[ "${PREFER_LONG_JOBS}" == "1" ]]; then
    extra+=("--prefer_long_jobs")
  fi
  if [[ -n "${RECENT_THINKS_TOKEN_BUDGET}" ]]; then
    extra+=("--recent_thinks_token_budget" "${RECENT_THINKS_TOKEN_BUDGET}")
  fi
  if [[ -n "${RECALL_TEXT_MAX_CHARS}" ]]; then
    extra+=("--recall_text_max_chars" "${RECALL_TEXT_MAX_CHARS}")
  fi
  if [[ -n "${SUMMARY_TOKENS_MAX}" ]]; then
    extra+=("--summary_tokens_max" "${SUMMARY_TOKENS_MAX}")
  fi

  local t0 t1 elapsed
  t0=$(date +%s)
  set +e
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${PYTHON_BIN}" scripts/eval/ovo/eval_full.py \
      --ckpt "${ckpt}" \
      --benchmark_json "${BENCHMARK_JSON}" \
      --video_root "${VIDEO_ROOT}" \
      --frames_root "${FRAMES_ROOT}" \
      --engine vllm \
      --tensor_parallel_size 1 \
      --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --vllm_mm_processor_cache_gb "${MM_CACHE_GB}" \
      --vllm_max_images_per_prompt 128 \
      --vllm_max_videos_per_prompt 64 \
      --vllm_repetition_penalty "${REPETITION_PENALTY}" \
      --n_per_task "${N_PER_TASK}" \
      --scoring "${SCORING}" \
      --compress_mode "${compress_mode}" \
      --memory_mode "${memory_mode}" \
      --retriever "${retriever}" \
      --frame-protocol "${protocol}" \
      --render-layout "${layout}" \
      --memory-position "${memory_position}" \
      --min_pixels 130000 \
      --max_pixels "${max_pixels}" \
      --visual_window_chunks "${window_chunks}" \
      --frames_per_chunk "${frames_per_chunk}" \
      --source_frames_per_chunk 0 \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --compress_max_new_tokens "${COMPRESS_MAX_NEW_TOKENS}" \
      --progress_every "${PROGRESS_EVERY}" \
      --out "${out_json}" \
      "${extra[@]}"
  ) > "${log}" 2>&1
  local rc=$?
  set -e
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [[ "${rc}" != "0" || ! -s "${out_json}" ]]; then
    printf "%s\tFAIL\t%s\t%s\trc=%s seconds=%s log=%s\n" \
      "$(date +%F_%T)" "${tag}" "${gpu}" "${rc}" "${elapsed}" "${log}" >> "${STATUS}"
    return 0
  fi

  local line
  line=$(summarize_json "${tag}" "${out_json}" "${elapsed}")
  printf "%s\t%s\n" "$(date +%F_%T)" "${line}" >> "${MATRIX}"
  printf "%s\tDONE\t%s\t%s\t%s\n" "$(date +%F_%T)" "${tag}" "${gpu}" "${line}" >> "${STATUS}"
}

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  sleep "$((worker_idx * 3))"
  local i
  for i in "${!JOBS[@]}"; do
    if (( i % NUM_GPUS == worker_idx )); then
      run_one "${JOBS[$i]}" "${gpu}"
    fi
  done
}

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

echo "OVO agent memory sweep"
echo "  out_root: ${OUT_ROOT}"
echo "  gpus:     ${GPUS}"
echo "  jobs:     ${#JOBS[@]}"
echo "  n/task:   ${N_PER_TASK}"
echo "  status:   ${STATUS}"

PIDS=()
for worker_idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_idx]}"
  run_worker "${worker_idx}" "${gpu}" > "${OUT_ROOT}/logs/worker_${worker_idx}_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

printf "%s\tALL_DONE\tall\t%s\tjobs=%s\n" "$(date +%F_%T)" "${GPUS}" "${#JOBS[@]}" >> "${STATUS}"
echo "All agent-memory sweep workers finished. Summary: ${MATRIX}"
