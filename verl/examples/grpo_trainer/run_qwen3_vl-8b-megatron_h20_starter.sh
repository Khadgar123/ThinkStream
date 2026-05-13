#!/usr/bin/env bash
set -xeuo pipefail
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# dependency: vllm>=0.11.0, megatron-lm>=0.13, mbridge with qwen3vl_cp branch
# environment option1: use a stable container later than docker://verlai/verl:vllm011.dev6 
    # and install mbridge in it by following the instruction in the container
            # pip remove mbridge if you have installed it
            # pip install git+https://github.com/ISEEKYAN/mbridge.git@qwen3vl_cp # for correct mbridge
# environment option2: use container docker://verlai/verl:vllm011.dev_qwenvl_cp
 

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP


# Set local model path here (or export HF_MODEL_PATH before running).
HF_MODEL_PATH=${HF_MODEL_PATH:-"/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct"}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# Only log to wandb on rank 0 node
if [ "$NODE_RANK" != "0" ]; then
    export WANDB_MODE=disabled
fi

GEN_TP=${GEN_TP:-2}
CP=${CP:-2}
TP=${TP:-2}
PP=${PP:-2}

DATA_ROOT=${DATA_ROOT:-"/home/tione/notebook/gaozhenkun/hzh/data/geo3k"}
train_path=${TRAIN_PATH:-"$DATA_ROOT/train.parquet"}
test_path=${TEST_PATH:-"$DATA_ROOT/test.parquet"}

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_megatron_h20_starter' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@