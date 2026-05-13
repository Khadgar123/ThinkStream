#!/bin/bash
#
# Compatibility notice.
#
# The old OVO agent-memory sweep used the standalone OVO HF/vLLM runner and is
# no longer a current ThinkStream method test. Use the recurrent RL AgentLoop
# OVO path instead.

set -euo pipefail

echo "run_agent_memory_sweep.sh is retired for current testing." >&2
echo "Use scripts/eval/ovo/run_rl_recurrent_eval.sh, which builds OVO RL" >&2
echo "trajectories/parquet and runs verl VAL_ONLY=true with true-KV recurrent rollout." >&2
exit 2
