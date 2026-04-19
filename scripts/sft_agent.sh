#!/bin/bash
# Agent SFT Training Script (3-action protocol)
#
# Two-stage training:
#   Stage A: Protocol alignment (stream_agent_sft_a)
#   Stage B: Recall bootstrap (stream_agent_sft_b, from Stage A ckpt)
#
# Usage:
#   # Stage A
#   STAGE=a bash scripts/sft_agent.sh
#   # Stage B (after Stage A completes)
#   STAGE=b LLM=./output/agent-sft-a/checkpoint-best bash scripts/sft_agent.sh

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=0
NPROC_PER_NODE=8

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

deepspeed=./scripts/zero3.json

# Stage selection
STAGE=${STAGE:-a}

if [ "$STAGE" = "a" ]; then
    llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
    datasets=stream_agent_sft_a
    lr=1e-5
    epochs=3
    run_name="agent-sft-a"
elif [ "$STAGE" = "b" ]; then
    llm=${LLM:?'Set LLM to Stage A checkpoint path'}
    datasets=stream_agent_sft_b
    lr=5e-6
    epochs=3
    run_name="agent-sft-b"
else
    echo "Invalid STAGE=$STAGE, use 'a' or 'b'"
    exit 1
fi

batch_size=8
grad_accum_steps=1
entry_file=thinkstream/train.py
output_dir=./output/${run_name}

echo "=== Agent SFT Stage ${STAGE} ==="
echo "Model: ${llm}"
echo "Dataset: ${datasets}"
echo "LR: ${lr}, Epochs: ${epochs}"
echo "Output: ${output_dir}"

args="
    sft \
    --args.train.deepspeed ${deepspeed} \
    --args.model.name_or_path ${llm} \
    --args.model.model_type qwen2.5vl \
    --args.model.max_length 32768 \
    --args.data.dataset_use ${datasets} \
    --args.data.flatten False \
    --args.data.video_max_pixels 150528 \
    --args.data.video_min_pixels 100352 \
    --args.train.bf16 True \
    --args.train.output_dir ${output_dir} \
    --args.train.num_train_epochs ${epochs} \
    --args.train.per_device_train_batch_size ${batch_size} \
    --args.train.gradient_accumulation_steps ${grad_accum_steps} \
    --args.train.save_steps 500 \
    --args.train.learning_rate ${lr} \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.torch_empty_cache_steps 1 \
    --args.train.dataloader.num_workers 4
"

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node-rank=${NODE_RANK} \
         --nnodes=${NNODES} \
         ${entry_file} ${args}
