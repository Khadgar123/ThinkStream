#!/bin/bash
# Agent RL Training Script (GRPO with 3-action protocol)
#
# Requires SFT-B checkpoint as starting point.
#
# Usage:
#   LLM=./output/agent-sft-b/checkpoint-best bash scripts/rl_agent.sh

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=0
NPROC_PER_NODE=8

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

deepspeed=./scripts/zero3.json

llm=${LLM:?'Set LLM to SFT-B checkpoint path'}
datasets=stream_agent_rl
lr=2e-7
batch_size=1
grad_accum_steps=1
entry_file=thinkstream/train.py
run_name="agent-rl"
output_dir=./output/${run_name}

echo "=== Agent RL (GRPO) ==="
echo "Model: ${llm}"
echo "Dataset: ${datasets}"
echo "LR: ${lr}"
echo "Output: ${output_dir}"

args="
    grpo \
    --args.train.num_train_epochs 1 \
    --args.train.output_dir ${output_dir} \
    --args.train.deepspeed ${deepspeed} \
    --args.train.per_device_train_batch_size ${batch_size} \
    --args.train.warmup_ratio 0.03 \
    --args.train.learning_rate ${lr} \
    --args.train.gradient_accumulation_steps ${grad_accum_steps} \
    --args.model.name_or_path ${llm} \
    --args.train.save_steps 300 \
    --args.data.dataset_use ${datasets} \
    --args.model.model_type qwen2.5vl \
    --args.train.group_size 8 \
    --args.train.micro_batch_size 8 \
    --args.train.rollout_max_pixels $((192*28*28)) \
    --args.train.rollout_min_pixels $((128*28*28))
"

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node-rank=${NODE_RANK} \
         --nnodes=${NNODES} \
         ${entry_file} ${args}
