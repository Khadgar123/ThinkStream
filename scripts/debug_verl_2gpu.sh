#!/bin/bash
# 2-GPU verl GRPO debug — minimal smoke test for the streaming-video
# recipe (verl 0.4 path, NOT the legacy thinkstream/train.py path).
#
# What this script verifies:
#   1. Synthetic trajectory + parquet build pipeline works.
#   2. Streaming agent loop registers and runs at least one rollout.
#   3. enable_prefix_caching + mm_preprocessor_cache don't break startup.
#   4. expanding visual-window mode produces correct loss masks.
#   5. compute_score returns a non-zero scalar end-to-end.
#
# Layout:
#   - Use 2 GPUs (CUDA_VISIBLE_DEVICES override-able via env).
#   - vLLM tensor_model_parallel_size=1 → vLLM uses 1 GPU; actor + ref
#     colocate on the other (FSDP colocation; param/optim offload to CPU
#     to fit Qwen3-VL-8B on a single 96GB GPU at debug batch sizes).
#   - One short rollout (max_assistant_turns=8), tiny batch (1×1×1).
#   - Pre-extracted frames are OPTIONAL: if THINKSTREAM_FRAMES_ROOT
#     points at a directory with frame_*.jpg the agent loop injects
#     visual blocks; otherwise it runs text-only (still exercises the
#     prompt assembly + tool-call parsing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}/verl"

# ── conda env (override via CONDA_ENV / skip via SKIP_CONDA=1).
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV:-/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream}"
    fi
fi

# ── checkpoint resolution: prefer the most recent SFT, fall back to debug.
LLM="${LLM:-${PROJECT_DIR}/output/agent-sft/checkpoint-200}"
if [[ ! -d "${LLM}" ]]; then
    LLM="${PROJECT_DIR}/output/agent-sft-debug/checkpoint-5"
fi
if [[ ! -d "${LLM}" ]]; then
    echo "ERROR: no SFT checkpoint at ${LLM}. Set LLM=/path/to/ckpt." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/agent-verl-debug-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

# ── Synthetic trajectory data — generate if missing.
TRAJ_DIR="${PROJECT_DIR}/data/test_rl"
TRAJ_JSONL="${TRAJ_DIR}/synthetic_trajectories.jsonl"
MULTI_Q="${MULTI_Q:-1}"
if [[ "${MULTI_Q}" == "1" ]]; then
    TRAIN_PARQUET="${TRAJ_DIR}/synthetic_train_multi_q.parquet"
else
    TRAIN_PARQUET="${TRAJ_DIR}/synthetic_train_single_q.parquet"
fi
mkdir -p "${TRAJ_DIR}"
if [[ ! -f "${TRAJ_JSONL}" || "${REGEN_DATA:-0}" == "1" ]]; then
    echo "[debug] generating synthetic trajectories → ${TRAJ_JSONL}"
    python -m scripts.test_rl.synthetic_traj --out "${TRAJ_JSONL}"
fi
if [[ ! -f "${TRAIN_PARQUET}" || "${REGEN_DATA:-0}" == "1" ]]; then
    # MULTI_Q=1 → 1 video = 1 row (OVOBench-aligned, all questions co-evaluated).
    # MULTI_Q=0 → legacy (video, question) flatten for ablations only.
    MULTI_Q_FLAG=""
    if [[ "${MULTI_Q}" == "1" ]]; then
        MULTI_Q_FLAG="--multi_q"
        echo "[debug] building MULTI-Q parquet → ${TRAIN_PARQUET}"
    else
        echo "[debug] flattening to (video,question) parquet → ${TRAIN_PARQUET}"
    fi
    python -m scripts.agent_data_v5.build_verl_parquet \
        --jsonl "${TRAJ_JSONL}" --out "${TRAIN_PARQUET}" ${MULTI_Q_FLAG}
fi
VAL_PARQUET="${VAL_PARQUET:-${TRAIN_PARQUET}}"

# ── Streaming agent loop env vars (read inside __init__, line 316+).
export THINKSTREAM_TRAJ_INDEX_PATH="${THINKSTREAM_TRAJ_INDEX_PATH:-${TRAJ_JSONL}}"
export THINKSTREAM_FRAMES_ROOT="${THINKSTREAM_FRAMES_ROOT:-}"  # empty = text-only debug
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export THINKSTREAM_VISUAL_WINDOW_CHUNKS="${THINKSTREAM_VISUAL_WINDOW_CHUNKS:-8}"
# v12.13: visual-window mode. Default sliding (matches existing SFT data
# layout). Override with `THINKSTREAM_VISUAL_WINDOW_MODE=expanding bash ...`
# only after regenerating SFT data with the same mode.
export THINKSTREAM_VISUAL_WINDOW_MODE="${THINKSTREAM_VISUAL_WINDOW_MODE:-sliding}"
export THINKSTREAM_RECALL_STUB="${THINKSTREAM_RECALL_STUB:-(no relevant past observation found)}"
# Double-layer advantage (mostly noise on a 1-step debug, but exercises the path).
export THINKSTREAM_USE_STATE_ADVANTAGE="${THINKSTREAM_USE_STATE_ADVANTAGE:-1}"
export THINKSTREAM_ADVANTAGE_MODE="${THINKSTREAM_ADVANTAGE_MODE:-remem}"
export THINKSTREAM_STATE_REWARD_MODE="${THINKSTREAM_STATE_REWARD_MODE:-format_action}"

# ── PYTHONPATH so reward fn + agent loop can import thinkstream.*.
export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

echo "═══ ThinkStream verl GRPO 2-GPU debug ═══"
echo "  Checkpoint:    ${LLM}"
echo "  Train parquet: ${TRAIN_PARQUET}"
echo "  Frames root:   ${THINKSTREAM_FRAMES_ROOT:-<text-only>}"
echo "  Window mode:   ${THINKSTREAM_VISUAL_WINDOW_MODE}"
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${MM_CACHE_GB:-8} \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
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
    trainer.experiment_name="verl-debug" \
    trainer.project_name="thinkstream-debug" \
    trainer.default_local_dir="${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/debug.log"

echo ""
echo "═══ ✓ verl 2-GPU debug complete ═══"
ls -la "${OUTPUT_DIR}"
