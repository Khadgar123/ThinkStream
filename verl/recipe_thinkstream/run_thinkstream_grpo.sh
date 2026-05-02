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
#
# Optional env (defaults in [...]):
#   N_GPUS_PER_NODE [8] / NNODES [1]
#   GEN_TP [2]              vLLM tensor_model_parallel_size
#   GROUP_SIZE [8]          GRPO group size
#   BATCH_SIZE [8]          videos per step
#   PPO_MINI_BS [32]
#   LR [1e-6]
#   EPOCHS [1]
#   MAX_PROMPT_LEN [16384]
#   MAX_RESP_LEN [2048]
#   MAX_TURNS [360]
#   GPU_MEM_UTIL [0.55]
#   PROJECT_NAME [thinkstream-v12]
#   EXPERIMENT_NAME [grpo-v126-verl]
#   SAVE_DIR [./output/$EXPERIMENT_NAME]

set -xeuo pipefail

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

THINKSTREAM_HOME=${THINKSTREAM_HOME:?"THINKSTREAM_HOME= required (path to ThinkStream checkout)"}
HF_MODEL_PATH=${HF_MODEL_PATH:?"HF_MODEL_PATH= required (Qwen3-VL-8B SFT ckpt)"}
TRAIN_PARQUET=${TRAIN_PARQUET:?"TRAIN_PARQUET= required"}
VAL_PARQUET=${VAL_PARQUET:?"VAL_PARQUET= required"}

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-2}
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
PPO_MINI_BS=${PPO_MINI_BS:-32}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-16384}
MAX_RESP_LEN=${MAX_RESP_LEN:-2048}
MAX_TURNS=${MAX_TURNS:-360}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}

PROJECT_NAME=${PROJECT_NAME:-thinkstream-v12}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo-v126-verl}
SAVE_DIR=${SAVE_DIR:-./output/${EXPERIMENT_NAME}}

# verl spawns Ray workers; each worker process inherits PYTHONPATH so the
# reward function can import thinkstream.trainer.v12_rewards.
export PYTHONPATH="${THINKSTREAM_HOME}:${PYTHONPATH:-}"
export THINKSTREAM_TRAJ_INDEX_PATH="${THINKSTREAM_TRAJ_INDEX_PATH:-${THINKSTREAM_HOME}/data/agent_v5/final/train_rl_trajectories.jsonl}"

# v12.13: ThinkStream-specific multi_turn config (verl's MultiTurnConfig
# rejects custom keys, so we pass them as env vars; streaming_agent_loop.py
# reads them in __init__ at line 316+). frames_root="" → text-only run.
export THINKSTREAM_FRAMES_ROOT="${THINKSTREAM_FRAMES_ROOT:-${THINKSTREAM_HOME}/data/agent_v5/frames}"
export THINKSTREAM_FRAMES_PER_CHUNK="${THINKSTREAM_FRAMES_PER_CHUNK:-2}"
export THINKSTREAM_VISUAL_WINDOW_CHUNKS="${THINKSTREAM_VISUAL_WINDOW_CHUNKS:-16}"
export THINKSTREAM_RECALL_STUB="${THINKSTREAM_RECALL_STUB:-(no relevant past observation found)}"

mkdir -p "${SAVE_DIR}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$(pwd)/recipe_thinkstream/configs" \
    --config-name='thinkstream_grpo' \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.response_length=${MAX_RESP_LEN} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    custom_reward_function.path="recipe_thinkstream/thinkstream.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=${EPOCHS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_DIR} 2>&1 | tee "${SAVE_DIR}/train.log"
