#!/usr/bin/env bash
# Mini RL test harness — one-shot runner.
#
# 1. Generate synthetic pass4-format trajectories (covers all mechanisms +
#    320-chunk long video).
# 2. Run mock rollout: drives streaming agent loop logic with deterministic
#    fake outputs, asserts mechanism-aware behavior + reward outcome=1.0.
# 3. Validate trainer_verl/dataset.py preserves v12.13 ground_truth fields
#    (answer_chunks / per_emit_answers / options).
#
# Exits 0 if all stages pass.
set -e

cd "$(dirname "$0")/../.."

OUT=data/test_rl
mkdir -p "$OUT"

echo "═══ Step 1: Generate synthetic trajectories ═══"
python -m scripts.test_rl.synthetic_traj \
    --out "$OUT/synthetic_trajectories.jsonl"
echo

echo "═══ Step 2: Mock rollout + reward + prompt consistency ═══"
python -m scripts.test_rl.mock_rollout \
    --traj "$OUT/synthetic_trajectories.jsonl" \
    --report "$OUT/mock_report.json"
echo

echo "═══ Step 3: VERL dataset ground_truth field check ═══"
python -m scripts.test_rl.verl_dataset_check \
    --traj "$OUT/synthetic_trajectories.jsonl"
echo

echo "═══ Step 4: Multi-Q parquet + compute_score round-trip ═══"
python -m scripts.agent_data.build_verl_parquet \
    --jsonl "$OUT/synthetic_trajectories.jsonl" \
    --out "$OUT/synthetic_multi_q.parquet" \
    --multi_q
python -m scripts.test_rl.test_multi_q_score \
    --parquet "$OUT/synthetic_multi_q.parquet"
echo

echo "═══ ✓ Mini RL test passed ═══"
echo "Report: $OUT/mock_report.json"
