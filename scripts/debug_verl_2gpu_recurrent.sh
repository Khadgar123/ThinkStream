#!/bin/bash
# 2-GPU verl GRPO debug — recurrent mode (v12.14 EXPERIMENTAL).
# Verifies Phase 4 DataProto integration:
#   - sample_index tagging across actions
#   - final_mask (only last action per trajectory = True)
#   - trajectory-level reward broadcast back to action rows
#   - FSDP padding with response_mask=0 on padded rows
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}/verl"

# ── conda env (override via CONDA_ENV / skip via SKIP_CONDA=1).
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream}"
    fi
fi

LLM="${LLM:-${PROJECT_DIR}/output/agent-sft/checkpoint-200}"
if [[ ! -d "${LLM}" ]]; then
    LLM="${PROJECT_DIR}/output/agent-sft-debug/checkpoint-5"
fi
if [[ ! -d "${LLM}" ]]; then
    echo "ERROR: no SFT checkpoint at ${LLM}. Set LLM=/path/to/ckpt." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/agent-verl-recurrent-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
FRAME_PROTOCOL="${FRAME_PROTOCOL:-${THINKSTREAM_FRAME_PROTOCOL:-ts_image}}"
export THINKSTREAM_FRAME_PROTOCOL="${FRAME_PROTOCOL}"

TRAJ_DIR="${PROJECT_DIR}/data/test_rl"
TRAJ_JSONL="${TRAJ_DIR}/synthetic_trajectories.jsonl"
TRAIN_PARQUET="${TRAJ_DIR}/synthetic_train_${FRAME_PROTOCOL}.parquet"
mkdir -p "${TRAJ_DIR}"
if [[ ! -f "${TRAJ_JSONL}" || "${REGEN_DATA:-0}" == "1" ]]; then
    python -m scripts.test_rl.synthetic_traj --out "${TRAJ_JSONL}"
fi
if [[ ! -f "${TRAIN_PARQUET}" || "${REGEN_DATA:-0}" == "1" ]]; then
    python -m scripts.agent_data_v5.build_verl_parquet \
        --jsonl "${TRAJ_JSONL}" --out "${TRAIN_PARQUET}" \
        --frame-protocol "${FRAME_PROTOCOL}"
fi
VAL_PARQUET="${VAL_PARQUET:-${TRAIN_PARQUET}}"

export THINKSTREAM_TRAJ_INDEX_PATH="${THINKSTREAM_TRAJ_INDEX_PATH:-${TRAJ_JSONL}}"
export THINKSTREAM_FRAMES_ROOT="${THINKSTREAM_FRAMES_ROOT:-}"
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export THINKSTREAM_VISUAL_WINDOW_CHUNKS="${THINKSTREAM_VISUAL_WINDOW_CHUNKS:-8}"
export THINKSTREAM_VISUAL_WINDOW_MODE="${THINKSTREAM_VISUAL_WINDOW_MODE:-sliding}"
export THINKSTREAM_RECALL_STUB="${THINKSTREAM_RECALL_STUB:-(no relevant past observation found)}"
export THINKSTREAM_USE_STATE_ADVANTAGE="${THINKSTREAM_USE_STATE_ADVANTAGE:-1}"
export THINKSTREAM_ADVANTAGE_MODE="${THINKSTREAM_ADVANTAGE_MODE:-remem}"
export THINKSTREAM_STATE_REWARD_MODE="${THINKSTREAM_STATE_REWARD_MODE:-format_action}"
# v12.14: recurrent mode (EXPERIMENTAL)
export THINKSTREAM_RECURRENT_MODE="recurrent"
export THINKSTREAM_MAX_RECALL_PER_CHUNK="${THINKSTREAM_MAX_RECALL_PER_CHUNK:-1}"

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

# ── Recurrent capacity config ──
# Per-action cap (NOT stitched total). Each action's response is padded
# to this length. Typical single-action upper bound for ThinkStream.
MAX_MODEL_LEN=8192
RESP_LEN=4096
PROMPT_LEN=4096
MAX_TOK_LEN=$((MAX_MODEL_LEN))

echo "═══ ThinkStream verl GRPO 2-GPU recurrent ═══"
echo "  Checkpoint:    ${LLM}"
echo "  Train parquet: ${TRAIN_PARQUET}"
echo "  Protocol:      ${FRAME_PROTOCOL}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  response_len:  ${RESP_LEN}  (per-action cap)"
echo "  prompt_len:    ${PROMPT_LEN}"
echo "  max_turns:     120"
echo "  Output:        ${OUTPUT_DIR}"
echo "════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7} \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
    --config-path="${PROJECT_DIR}/verl/recipe_thinkstream/configs" \
    --config-name="thinkstream_grpo" \
    actor_rollout_ref.model.path="${LLM}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_LEN} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOK_LEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.limit_images=${LIMIT_IMAGES:-64} \
    actor_rollout_ref.rollout.limit_videos=${LIMIT_VIDEOS:-2} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${MM_CACHE_GB:-8} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.response_length=${RESP_LEN} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOK_LEN} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=120 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=120 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=1 \
    data.max_prompt_length=${PROMPT_LEN} \
    data.max_response_length=${RESP_LEN} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    reward.custom_reward_function.path="recipe_thinkstream/thinkstream.py" \
    reward.custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.experiment_name="verl-recurrent" \
    trainer.project_name="thinkstream-debug" \
    trainer.default_local_dir="${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/debug.log"

echo ""
echo "═══ ✓ verl recurrent debug complete ═══"
ls -la "${OUTPUT_DIR}"
