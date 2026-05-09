#!/bin/bash
# ThinkStream streaming-video GRPO via verl on 8× H20 (96 GB).
#
# Why this script (vs ThinkStream/scripts/grpo_train_verl.sh):
#   - That one assumed a top-level recipe/v12_grpo.yaml inside the
#     ThinkStream repo and shelled out to verl.trainer.main_ppo from there.
#   - This one is the "in-tree" verl recipe: lives next to deepeyes,
#     mirrors its layout, and only needs THINKSTREAM_HOME + the parquet
#     train/val files to drive an end-to-end run.
#
# Required env:
#   THINKSTREAM_HOME    — checkout of github.com/Khadgar123/ThinkStream
#                         (used for PYTHONPATH so reward_fn can import
#                          thinkstream.trainer.v12_rewards).
#   HF_MODEL_PATH       — Qwen3-VL-8B SFT checkpoint (HF format).
#   TRAIN_PARQUET       — flattened (video, question) parquet from
#                         scripts/agent_data_v5/build_verl_parquet.py.
#   VAL_PARQUET         — val split.
#   THINKSTREAM_DATA_ROOT — generated batch root containing final/ and frames/
#                           (optional, defaults to $THINKSTREAM_HOME/data/agent_v5).
#
# Optional env (defaults in [...]):
#   N_GPUS_PER_NODE [8] / NNODES [1]
#   GEN_TP [2]              vLLM tensor_model_parallel_size
#   GROUP_SIZE [8]          GRPO group size
#   BATCH_SIZE [4]          videos per step
#   PPO_MINI_BS [BATCH_SIZE]
#                           verl multiplies this by rollout.n internally
#                           for the generated-response mini-batch.
#   LR [5e-7]
#   EPOCHS [1]
#   MAX_PROMPT_LEN [16384]
#   MAX_RESP_LEN [32768]    total stitched response buffer
#   MAX_ACTION_TOKENS [256]
#                           per-action streaming/recall vLLM generation cap
#   MAX_COMPRESS_ACTION_TOKENS [512]
#                           per-action compression vLLM generation cap
#   MAX_TURNS [120]   (covers batch1 max=95 + headroom. Stitched ceiling
#                       ~180; for 240+ chunks see
#                       docs/v12.14_recurrent_design.md for the recurrent
#                       path that lifts this to 600+ without OOM.)
#   GPU_MEM_UTIL [0.55]
#   MM_CACHE_GB [auto]       vLLM CPU mm processor cache GB
#   THINKSTREAM_FRAME_PROTOCOL [video_meta]
#   THINKSTREAM_RENDER_LAYOUT [timeline_video_imagepad]
#   THINKSTREAM_RL_EPISODE_MODE [full] full | segment
#   THINKSTREAM_RECURRENT_MODE [recurrent] recurrent | stitched
#   LIMIT_IMAGES [64]       vLLM limit_mm_per_prompt.image for timestamped frames
#   LIMIT_VIDEOS [2]        vLLM limit_mm_per_prompt.video for video_meta blocks
#   PROJECT_NAME [thinkstream-v12]
#   EXPERIMENT_NAME [grpo-v12.26-verl-$THINKSTREAM_FRAME_PROTOCOL]
#   SAVE_DIR [./output/$EXPERIMENT_NAME]
#   SAVE_FREQ [50] / TEST_FREQ [25]
#   PARAM_OFFLOAD [true] / OPTIMIZER_OFFLOAD [true]
#   FREEZE_VISION_TOWER [true]
#   PPO_MAX_TOKEN_LEN_PER_GPU / LOG_PROB_MAX_TOKEN_LEN_PER_GPU [65536]
#   RUNTIME_ROOT [$THINKSTREAM_HOME/.runtime/$EXPERIMENT_NAME]
#                            local root for Ray temp, HF cache, Torch cache,
#                            Triton cache, and XDG cache.
#   ROLLOUT_DATA_DIR [""]    optional full verl generation dump, one JSONL
#                            file per step; can be large.
#   THINKSTREAM_RL_ROLLOUT_AUDIT_PATH [$SAVE_DIR/audit/rl_rollout_samples.jsonl]
#                            compact reward-time rollout audit sampler.
#   THINKSTREAM_RL_ROLLOUT_AUDIT_PROB [0.01]
#                            random sample rate; suspicious rows are always
#                            logged until THINKSTREAM_RL_ROLLOUT_AUDIT_MAX.

set -xeuo pipefail

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

THINKSTREAM_HOME=${THINKSTREAM_HOME:?"THINKSTREAM_HOME= required (path to ThinkStream checkout)"}
HF_MODEL_PATH=${HF_MODEL_PATH:?"HF_MODEL_PATH= required (Qwen3-VL-8B SFT ckpt)"}
TRAIN_PARQUET=${TRAIN_PARQUET:?"TRAIN_PARQUET= required"}
VAL_PARQUET=${VAL_PARQUET:?"VAL_PARQUET= required"}
THINKSTREAM_DATA_ROOT=${THINKSTREAM_DATA_ROOT:-${THINKSTREAM_HOME}/data/agent_v5}
if [[ "${THINKSTREAM_DATA_ROOT}" == */final ]]; then
    THINKSTREAM_DATA_ROOT="$(dirname "${THINKSTREAM_DATA_ROOT}")"
fi
if [[ -z "${THINKSTREAM_ENV:-}" ]]; then
    THINKSTREAM_PARENT="$(dirname "${THINKSTREAM_HOME}")"
    if [[ -x "${THINKSTREAM_PARENT}/envs/thinkstream/bin/python" ]]; then
        THINKSTREAM_ENV="${THINKSTREAM_PARENT}/envs/thinkstream"
    elif [[ -x "${THINKSTREAM_HOME}/envs/thinkstream/bin/python" ]]; then
        THINKSTREAM_ENV="${THINKSTREAM_HOME}/envs/thinkstream"
    else
        THINKSTREAM_ENV=""
    fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${THINKSTREAM_ENV}" && -x "${THINKSTREAM_ENV}/bin/python" ]]; then
        PYTHON_BIN="${THINKSTREAM_ENV}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-2}
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
PPO_MINI_BS=${PPO_MINI_BS:-${BATCH_SIZE}}
LR=${LR:-5e-7}
EPOCHS=${EPOCHS:-1}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-16384}
THINKSTREAM_RECURRENT_MODE="${THINKSTREAM_RECURRENT_MODE:-recurrent}"
case "${THINKSTREAM_RECURRENT_MODE}" in
    recurrent|stitched) ;;
    *)
        echo "ERROR: THINKSTREAM_RECURRENT_MODE must be recurrent or stitched, got ${THINKSTREAM_RECURRENT_MODE}" >&2
        exit 2
        ;;
esac
# v12.14 (2026-05-03): MAX_RESP_LEN is the STITCHED total across all chunk
# turns. With D1 chunk-internal multi-turn recall a chunk with one recall
# round costs ~3× a plain chunk; size with 1.5× headroom on top:
#
#   MAX_TURNS=60   → ~10K stitched   (smoke / very short videos only)
#   MAX_TURNS=120  → ~20K stitched   (default — covers current batch1's
#                                     60-95 chunk range with headroom)
#   MAX_TURNS=180  → ~30K stitched   (catches catalog 120-180s tier;
#                                     tight, needs use_dynamic_bsz)
#   MAX_TURNS=240+ → SWITCH TO v12.14 RECURRENT (docs/v12.14_recurrent_design.md)
#                    stitched training at 240+ chunks puts the actor
#                    forward at ~40K-60K seq + Qwen3-VL-8B FSDP shards
#                    → high OOM risk on 96GB H20.
#
# Real data context (2026-05-03):
#   batch1 actual rollouts (n=22): chunks min=60 p50=70 max=95
#   video_catalog_30s_plus.csv:
#     30-60s   = 32% (60 chunks)     ← stitched ok
#     60-120s  = 26% (60-120 chunks) ← stitched ok at MAX_TURNS=120
#     120-240s = 32% (120-240 chunks)← stitched OK ≤180; 240+ needs v12.14
#     240-600s = 7%                  ← needs v12.14
#     >=600s   = 3%                  ← needs v12.14
if [[ -z "${MAX_RESP_LEN:-}" ]]; then
    if [[ "${THINKSTREAM_RECURRENT_MODE}" == "recurrent" ]]; then
        MAX_RESP_LEN=4096
    else
        MAX_RESP_LEN=32768
    fi
fi
MAX_ACTION_TOKENS=${MAX_ACTION_TOKENS:-256}
MAX_COMPRESS_ACTION_TOKENS=${MAX_COMPRESS_ACTION_TOKENS:-512}
# Default 120 chunks comfortably covers all of current batch1 (max=95) and
# the lower tier of batch2's 120-240s videos. Bump to 180 for batch2
# coverage; for 240+ chunks switch to v12.14 recurrent rollout.
MAX_TURNS=${MAX_TURNS:-120}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
LIMIT_IMAGES=${LIMIT_IMAGES:-64}
LIMIT_VIDEOS=${LIMIT_VIDEOS:-2}
# v12.13: vLLM mm_processor_cache_gb (CPU-side image preprocessor cache).
# pass2's teacher run uses 512GB and gets 93.8% mm-cache hit on the same
# streaming-video workload. Default auto-sizes from MemAvailable so 2TiB
# H20 servers use 512GB while smaller/debug boxes stay conservative.
auto_mm_cache_gb() {
    local avail_kb avail_gb
    avail_kb="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)"
    avail_gb=$((avail_kb / 1024 / 1024))
    if (( avail_gb >= 1536 )); then
        echo 512
    elif (( avail_gb >= 768 )); then
        echo 256
    elif (( avail_gb >= 384 )); then
        echo 128
    elif (( avail_gb >= 128 )); then
        echo 64
    else
        echo 16
    fi
}
MM_CACHE_GB="${MM_CACHE_GB:-${THINKSTREAM_MM_CACHE_GB:-${VLLM_MM_PROCESSOR_CACHE_GB:-}}}"
if [[ -z "${MM_CACHE_GB}" ]]; then
    MM_CACHE_GB="$(auto_mm_cache_gb)"
fi
export MM_CACHE_GB
export THINKSTREAM_MM_CACHE_GB="${MM_CACHE_GB}"

PROJECT_NAME=${PROJECT_NAME:-thinkstream-v12}
THINKSTREAM_FRAME_PROTOCOL="${THINKSTREAM_FRAME_PROTOCOL:-video_meta}"
THINKSTREAM_RENDER_LAYOUT="${THINKSTREAM_RENDER_LAYOUT:-timeline_video_imagepad}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo-v12.26-verl-${THINKSTREAM_FRAME_PROTOCOL}}
SAVE_DIR=${SAVE_DIR:-./output/${EXPERIMENT_NAME}}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-true}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
MAX_STEPS=${MAX_STEPS:-}
DATA_SHUFFLE=${DATA_SHUFFLE:-true}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

RUNTIME_ROOT="${RUNTIME_ROOT:-${THINKSTREAM_HOME}/.runtime/${EXPERIMENT_NAME}}"
mkdir -p "${RUNTIME_ROOT}"/{tmp,ray,hf,torch,triton,xdg}
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_ROOT}/ray}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${RUNTIME_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg}"

# verl spawns Ray workers; each worker process inherits PYTHONPATH so the
# reward function can import thinkstream.trainer.v12_rewards.
export PYTHONPATH="${THINKSTREAM_HOME}:${PYTHONPATH:-}"
export THINKSTREAM_TRAJ_INDEX_PATH="${THINKSTREAM_TRAJ_INDEX_PATH:-${THINKSTREAM_DATA_ROOT}/final/train_rl_trajectories.jsonl}"
export THINKSTREAM_EXPERIMENT_NAME="${EXPERIMENT_NAME}"
export THINKSTREAM_RL_ROLLOUT_AUDIT_PATH="${THINKSTREAM_RL_ROLLOUT_AUDIT_PATH:-${SAVE_DIR}/audit/rl_rollout_samples.jsonl}"
export THINKSTREAM_RL_ROLLOUT_AUDIT_PROB="${THINKSTREAM_RL_ROLLOUT_AUDIT_PROB:-0.01}"
export THINKSTREAM_RL_ROLLOUT_AUDIT_MAX="${THINKSTREAM_RL_ROLLOUT_AUDIT_MAX:-2000}"

ROLLOUT_DATA_ARGS=()
if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
    ROLLOUT_DATA_ARGS=(trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}")
fi

# v12.13: ThinkStream-specific multi_turn config (verl's MultiTurnConfig
# rejects custom keys, so we pass them as env vars; streaming_agent_loop.py
# reads them in __init__ at line 316+). frames_root="" → text-only run.
export THINKSTREAM_FRAMES_ROOT="${THINKSTREAM_FRAMES_ROOT:-${THINKSTREAM_DATA_ROOT}/frames}"
export THINKSTREAM_FRAME_PROTOCOL
export THINKSTREAM_RENDER_LAYOUT
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export THINKSTREAM_VISUAL_WINDOW_CHUNKS="${THINKSTREAM_VISUAL_WINDOW_CHUNKS:-16}"
export THINKSTREAM_RECALL_STUB="${THINKSTREAM_RECALL_STUB:-(no relevant past observation found)}"
export THINKSTREAM_MAX_TOKENS_PER_ACTION="${THINKSTREAM_MAX_TOKENS_PER_ACTION:-${MAX_ACTION_TOKENS}}"
export THINKSTREAM_COMPRESS_MAX_TOKENS_PER_ACTION="${THINKSTREAM_COMPRESS_MAX_TOKENS_PER_ACTION:-${MAX_COMPRESS_ACTION_TOKENS}}"

# v12.13 (2026-05-02): visual-window mode for vLLM prefix cache.
# ─────────────────────────────────────────────────────────────────
# IMPORTANT — SFT/RL must agree on this value, else rollout-time
# visual context differs from training distribution.
#
#   sliding   — DEFAULT. window = [chunk-VWC+1 .. chunk]. Frame token
#               IDs shift every chunk → ~0% visual prefix-cache hit.
#               Use this if your SFT data was generated with the
#               legacy `max(0, chunk-VWC+1)` window (i.e. without
#               THINKSTREAM_VISUAL_WINDOW_MODE=expanding set when
#               running pass2_rollout.py / pass5_messages.py).
#
#   expanding — window = [(chunk//VWC)*VWC .. chunk]. Frames append
#               monotonically within a 16-chunk segment → ~94% (15/16)
#               visual prefix-cache hit; cache resets at segment
#               boundary. To use this:
#                 1. Regenerate SFT data with THINKSTREAM_VISUAL_WINDOW_MODE=expanding
#                    set during pass2_rollout.py + pass5_messages.py.
#                 2. Verify SFT loss + eval are healthy under the new layout.
#                 3. Then export THINKSTREAM_VISUAL_WINDOW_MODE=expanding
#                    for the RL run.
#               Without step 1+2 you'll have an SFT-RL distribution
#               mismatch (RL rollout shows different frame windows
#               than what the model trained on).
export THINKSTREAM_VISUAL_WINDOW_MODE="${THINKSTREAM_VISUAL_WINDOW_MODE:-sliding}"

# v12.14 Option B (2026-05-03): rollout output mode.
#   "stitched" (default) — one AgentLoopOutput per trajectory, all chunks
#                          stitched into one response. Matches all
#                          versions ≤ v12.13. Use this for batch1
#                          ≤120-chunk training.
#   "recurrent" (EXPERIMENTAL) — one AgentLoopOutput per assistant action.
#                          AgentLoopWorker (verl/verl/experimental/
#                          agent_loop/agent_loop.py Phase 1) flattens
#                          across the batch tagging sample_index +
#                          final_mask; ray_trainer Phase 4d
#                          (verl/verl/trainer/ppo/ray_trainer.py)
#                          extracts trajectory-level reward, computes 1D
#                          GRPO advantage by uid, broadcasts back to
#                          action rows via sample_index, and pads to
#                          actor world_size with response_mask=0 on
#                          padded rows. Required for >180-chunk training
#                          without OOM.
#
# EXPERIMENTAL — what's done, what's NOT:
#   ✓ math + DataProto integration tests pass (test_phase4_*.py)
#   ✗ NOT validated end-to-end on a real Ray/FSDP multi-GPU cluster
#   ✗ format/spam reward only comes from final action's solution_str
#     (NOT a stitched trajectory) — defensible compromise, see Phase 4d
#     comment in ray_trainer.py
#   ⚠ rollout.n must be ≥ 2 — n=1 collapses to singleton GRPO group
#     (advantage = raw score, no normalization)
#
# Activation (full v12.14, EXPERIMENTAL — start with 2-GPU debug):
#   THINKSTREAM_RECURRENT_MODE=recurrent \
#   MAX_RESP_LEN=4096 \                        # ← per-action cap, not stitched 32768
#   MULTI_Q=1 THINKSTREAM_MAX_RECALL_PER_CHUNK=1 \
#   bash verl/recipe_thinkstream/run_thinkstream_grpo.sh
#
# Why MAX_RESP_LEN must drop in recurrent mode:
#   In stitched, MAX_RESP_LEN=32768 caps the WHOLE trajectory's response.
#   In recurrent, AgentLoopWorker pads EACH action's response to
#   rollout.response_length (verl/verl/experimental/agent_loop/
#   agent_loop.py:736). So sum(K_i) actions × 32768 makes
#   responses/rm_scores/advantages/log_probs explode in dense memory and
#   cross-rank communication, even if attention runs no-padding. Set to
#   ~4096 (typical single-action upper bound for ThinkStream) or 6144
#   for headroom. The Phase 4 swap+pad path doesn't crash with 32768 —
#   it's purely a memory/throughput concern.
export THINKSTREAM_RECURRENT_MODE
export THINKSTREAM_RL_EPISODE_MODE="${THINKSTREAM_RL_EPISODE_MODE:-full}"
export THINKSTREAM_MAX_RECALL_PER_CHUNK="${THINKSTREAM_MAX_RECALL_PER_CHUNK:-1}"

# v12.13 P0: enable ReMemR1-style double-layer GRPO flags for telemetry and
# compatibility. The retired thinkstream/train.py grpo path no longer runs;
# verl's compute_score applies the same trajectory-scalar α-mix directly.
export THINKSTREAM_USE_STATE_ADVANTAGE="${THINKSTREAM_USE_STATE_ADVANTAGE:-1}"
export THINKSTREAM_ADVANTAGE_MODE="${THINKSTREAM_ADVANTAGE_MODE:-remem}"
export THINKSTREAM_STATE_REWARD_MODE="${THINKSTREAM_STATE_REWARD_MODE:-format_action}"
export THINKSTREAM_STATE_ADV_ALPHA="${THINKSTREAM_STATE_ADV_ALPHA:-0.7}"

mkdir -p "${SAVE_DIR}"

TRAINING_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
    TRAINING_STEPS_ARGS=(trainer.total_training_steps=${MAX_STEPS})
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="$(pwd)/recipe_thinkstream/configs" \
    --config-name='thinkstream_grpo' \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.val_batch_size=${BATCH_SIZE} \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.shuffle=${DATA_SHUFFLE} \
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
    ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR="${TMPDIR}" \
    ray_kwargs.ray_init.runtime_env.env_vars.RAY_TMPDIR="${RAY_TMPDIR}" \
    ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="${HF_HOME}" \
    ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    ray_kwargs.ray_init.runtime_env.env_vars.TORCH_HOME="${TORCH_HOME}" \
    ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.freeze_vision_tower=${FREEZE_VISION_TOWER} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.limit_images=${LIMIT_IMAGES} \
    actor_rollout_ref.rollout.limit_videos=${LIMIT_VIDEOS} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${MM_CACHE_GB} \
    actor_rollout_ref.rollout.response_length=${MAX_RESP_LEN} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    reward.custom_reward_function.path="recipe_thinkstream/thinkstream.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${EPOCHS} \
    "${TRAINING_STEPS_ARGS[@]}" \
    "${ROLLOUT_DATA_ARGS[@]}" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_DIR} 2>&1 | tee "${SAVE_DIR}/train.log"
