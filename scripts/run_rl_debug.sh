#!/bin/bash
set -euo pipefail
cd /home/tione/notebook/gaozhenkun/hzh/ThinkStream
set +u
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /home/tione/notebook/gaozhenkun/hzh/envs/thinkstream
set -u
export AGENT_DATA_DIR=/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_current_backup/final
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 thinkstream/train.py grpo \
    --args.train.deepspeed scripts/zero3.json \
    --args.model.name_or_path output/agent-sft-debug \
    --args.model.model_type qwen3vl \
    --args.data.dataset_use stream_agent_rl_traj \
    --args.train.output_dir output/agent-grpo-debug \
    --args.train.num_train_epochs 1 \
    --args.train.max_steps 30 \
    --args.train.per_device_train_batch_size 1 \
    --args.train.gradient_accumulation_steps 1 \
    --args.train.learning_rate 5e-7 \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.save_steps 30 \
    --args.train.bf16 True \
    --args.train.group_size 4 \
    --args.train.micro_batch_size 2 \
    --args.train.beta 1e-3 \
    --args.train.rollout_max_new_tokens 128 \
    --args.train.rollout_max_think_tokens 60 \
    --args.train.rollout_temperature 1.0 \
    --args.train.rollout_top_k 50 \
    --args.train.rollout_top_p 0.95 \
    --args.train.rollout_max_chunks 100 \
    --args.train.rollout_min_pixels 100352 \
    --args.train.rollout_max_pixels 150528 \
    --args.train.rollout_fpc 2.0 \
    --args.train.time_reward_window 5 \
    --args.train.time_reward_slack 3.0 \
    --args.train.dataloader.num_workers 4 \
    --args.train.dataloader.pin_memory True
