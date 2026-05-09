#!/usr/bin/env bash
# Build one canonical ThinkStream training/eval data root from multiple batches.
#
# Example:
#   bash scripts/prepare_training_data.sh \
#     --out data/agent_v5/scheme_v1 \
#     --batches data/agent_v5/batch1 data/agent_v5/batch2 data/agent_v5/batch3

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -m scripts.agent_data_v5.make_training_scheme "$@"
