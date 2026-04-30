#!/bin/bash
# End-to-end SFT + RL training pipeline for ThinkStream agent v12.5.
#
# Features:
#   - Uninterruptible (run inside tmux/screen)
#   - Auto-resume SFT from latest checkpoint if interrupted
#   - Auto-select best SFT checkpoint for RL (load_best_model_at_end)
#   - All stdout/stderr tee'd to timestamped log
#
# Usage:
#   tmux new-session -d -s thinkstream_train \
#       "bash scripts/train_end_to_end.sh 2>&1 | tee output/train_end_to_end_$(date +%Y%m%d_%H%M%S).log"
#
# To attach:  tmux attach -t thinkstream_train
# To detach:  Ctrl-b d

set -euo pipefail

# Activate conda environment (non-interactive shell needs explicit hook init)
set +u
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /home/tione/notebook/gaozhenkun/hzh/envs/thinkstream || { echo "Failed to activate thinkstream env"; exit 1; }
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# Data directory override (v12.5 backup)
export AGENT_DATA_DIR=${AGENT_DATA_DIR:-/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_current_backup/final}

# ------------------------------------------------------------------
# Configurable overrides (env vars)
# ------------------------------------------------------------------
NPROC=${NPROC:-8}
BSZ=${BSZ:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS_SFT=${EPOCHS_SFT:-2}
EPOCHS_RL=${EPOCHS_RL:-2}
LR_SFT=${LR_SFT:-2e-5}
LR_RL=${LR_RL:-5e-7}
BETA=${BETA:-1e-3}
GROUP_SIZE=${GROUP_SIZE:-4}
MICRO_BATCH=${MICRO_BATCH:-4}

SFT_RUN_NAME="agent-sft-v12.5"
RL_RUN_NAME="agent-grpo-v12.5"
SFT_OUTPUT="${PROJECT_DIR}/output/${SFT_RUN_NAME}"
RL_OUTPUT="${PROJECT_DIR}/output/${RL_RUN_NAME}"

LLM_BASE=${LLM:-/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct}

# ------------------------------------------------------------------
# Phase 1: SFT
# ------------------------------------------------------------------
echo "============================================"
echo "Phase 1: SFT"
echo "============================================"

# Detect existing checkpoint for resume
RESUME_FROM_CHECKPT=""
if [ -d "${SFT_OUTPUT}" ]; then
    latest_ckpt=$(find "${SFT_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
    if [ -n "${latest_ckpt}" ]; then
        echo "Found existing SFT checkpoint: ${latest_ckpt}"
        export RESUME_FROM_CHECKPOINT="${latest_ckpt}"
    else
        echo "No checkpoint found in ${SFT_OUTPUT}, starting from base model."
    fi
else
    echo "No existing SFT output dir, starting from base model."
fi

# Build SFT command
sft_cmd="PHASE=sft \
    LLM=${LLM_BASE} \
    NPROC=${NPROC} \
    BSZ=${BSZ} \
    GRAD_ACCUM=${GRAD_ACCUM} \
    EPOCHS=${EPOCHS_SFT} \
    LR=${LR_SFT} \
    bash ${SCRIPT_DIR}/sft_per_timestep.sh"

echo "SFT command:"
echo "${sft_cmd}"
echo ""

# Run SFT
eval ${sft_cmd}

# ------------------------------------------------------------------
# Phase 1→2 hand-off: find best SFT checkpoint
# ------------------------------------------------------------------
echo "============================================"
echo "Phase 1 complete. Selecting best checkpoint for RL."
echo "============================================"

# load_best_model_at_end saves the best checkpoint alongside the rolling window.
# If it exists, prefer it; otherwise fall back to the numerically latest checkpoint.
best_ckpt=""
if [ -d "${SFT_OUTPUT}" ]; then
    # HF Trainer with load_best_model_at_end writes a 'best' checkpoint link
    best_link="${SFT_OUTPUT}/checkpoint-best"
    if [ -L "${best_link}" ] && [ -d "$(readlink -f "${best_link}")" ]; then
        best_ckpt="$(readlink -f "${best_link}")"
        echo "Using best checkpoint (symlink): ${best_ckpt}"
    else
        best_ckpt=$(find "${SFT_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
        if [ -n "${best_ckpt}" ]; then
            echo "Using latest checkpoint: ${best_ckpt}"
        else
            echo "ERROR: No SFT checkpoint found in ${SFT_OUTPUT}"
            exit 1
        fi
    fi
else
    echo "ERROR: SFT output directory ${SFT_OUTPUT} does not exist."
    exit 1
fi

# ------------------------------------------------------------------
# Phase 2: RL (GRPO)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "Phase 2: RL (GRPO)"
echo "============================================"

rl_cmd="LLM=${best_ckpt} \
    NPROC=${NPROC} \
    GROUP_SIZE=${GROUP_SIZE} \
    MICRO_BATCH=${MICRO_BATCH} \
    LR=${LR_RL} \
    EPOCHS=${EPOCHS_RL} \
    BETA=${BETA} \
    RUN_NAME=${RL_RUN_NAME} \
    bash ${SCRIPT_DIR}/grpo_train.sh"

echo "RL command:"
echo "${rl_cmd}"
echo ""

eval ${rl_cmd}

echo ""
echo "============================================"
echo "End-to-end training complete!"
echo "SFT output: ${SFT_OUTPUT}"
echo "RL  output: ${RL_OUTPUT}"
echo "============================================"
