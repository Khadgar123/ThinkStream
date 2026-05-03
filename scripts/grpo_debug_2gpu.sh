#!/bin/bash
# Compatibility wrapper for the retired legacy GRPO debug launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "scripts/grpo_debug_2gpu.sh is deprecated; using scripts/debug_verl_2gpu.sh" >&2
exec "${SCRIPT_DIR}/debug_verl_2gpu.sh" "$@"
