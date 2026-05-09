#!/usr/bin/env bash
# Create the local ThinkStream Python environment used by SFT/RL launchers.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(dirname "${PROJECT_DIR}")"

ENV_DIR="${THINKSTREAM_ENV:-${PARENT_DIR}/envs/thinkstream}"
PYTHON="${PYTHON:-python3.12}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

echo "project=${PROJECT_DIR}"
echo "env=${ENV_DIR}"
echo "python=${PYTHON}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${PYTHON}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --extra-index-url "${CUDA_INDEX_URL}" -r "${PROJECT_DIR}/requirements.txt"

cat <<EOF

Environment ready.

Use:
  export THINKSTREAM_ENV=${ENV_DIR}
  export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/verl:\${PYTHONPATH:-}
EOF
