#!/bin/bash
# Compatibility wrapper. The legacy slyme GRPO trainer is retired; verl is the
# only supported RL backend for ThinkStream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "scripts/grpo_train.sh is deprecated; forwarding to scripts/grpo_train_verl.sh" >&2
exec "${SCRIPT_DIR}/grpo_train_verl.sh" "$@"
