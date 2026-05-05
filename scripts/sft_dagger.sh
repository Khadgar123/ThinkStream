#!/bin/bash
# Continue per-step SFT on a clean + on-policy DAgger mixture.
#
# Expected DAgger input is produced by:
#   python -m scripts.agent_data_v5.build_dagger_sft \
#     --ckpt <rollout-policy> \
#     --trajectories data/agent_v5/batch1/final/train_sft_trajectories.jsonl \
#     --out data/agent_v5/batch1/rendered/video_meta/train_sft_dagger_messages.jsonl \
#     --data-dir data/agent_v5/batch1 \
#     --frames-root data/agent_v5/batch1/frames \
#     --frame-protocol video_meta
#
# The trainer and loss are the same as scripts/sft_per_timestep.sh: one
# training row per streaming step, prompt tokens masked, assistant tokens
# trained. DAgger only changes the prompt distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PHASE="${PHASE:-sft}"
export FRAME_PROTOCOL="${FRAME_PROTOCOL:-video_meta}"
export THINKSTREAM_DATA_ROOT="${THINKSTREAM_DATA_ROOT:-data/agent_v5/batch1}"

# Default to the practical correction pass: keep gold prompts as an anchor and
# oversample the on-policy prompts. Override DATASETS for ablations.
export DATASETS="${DATASETS:-stream_agent_sft%50,stream_agent_sft_dagger}"
export LR="${LR:-1e-5}"
export EPOCHS="${EPOCHS:-1}"
export RUN_NAME="${RUN_NAME:-agent-sft-dagger-${FRAME_PROTOCOL}}"
export INCLUDE_FAILED_VERIFICATION="${INCLUDE_FAILED_VERIFICATION:-False}"

exec bash "${SCRIPT_DIR}/sft_per_timestep.sh"
