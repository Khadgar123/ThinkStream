#!/bin/bash
# Legacy RL debug entry. The old thinkstream/train.py grpo path is retired;
# use the verl debug launcher so reward/parquet/rollout behavior matches
# production RL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/debug_verl_2gpu.sh" "$@"
