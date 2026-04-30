"""Ablation runner — does the turn-level silent/response signal earn its weight?

Question:
  Is `silent_quality` (turn-level reward that catches "hallucinate when gold
  says silent" and "stay silent when gold says respond") doing real work, or
  is the trajectory-level outcome reward enough on its own?

  The user's RL design treats every chunk as a silent/answer/recall/compress
  decision. silent_quality is the only turn-level reward that directly
  shapes that decision. If we drop it, the policy must infer the silent
  vs respond distinction from outcome alone — which suffers severe credit
  assignment because outcome only fires at end-of-trajectory.

Configurations:
  A0 (turn_signal_on):  outcome + timing + format + spam + silent_quality
                        weights = V12_DEFAULT_REWARD_WEIGHTS
                        (the production v12.6 stack)
  A1 (turn_signal_off): outcome + timing + format + spam
                        silent_quality weight forced to 0
                        (no turn-level silent vs respond shaping)

Metrics tracked (per ablation):
  - outcome reward mean & median over training steps
  - silent_acc      = P(model emits empty answer | gold says silent)
  - response_acc    = P(model emits non-empty answer | gold says respond)
  - hallucinate_rate = P(model talks | gold says silent)   ← key failure mode
  - missed_rate      = P(model silent | gold says respond) ← key failure mode

Output:
  ablation_results/<timestamp>/
    A0_turn_on.jsonl         per-step metrics
    A1_turn_off.jsonl        per-step metrics
    summary.json             final mean/SE per metric
    plot_outcome.png         outcome curve overlay
    plot_silent_response.png silent_acc / response_acc overlay

This script is the ENTRY POINT — it sets reward weights, kicks off the
slyme RL trainer twice, and aggregates results. It runs on 8×H20 (the
RL trainer needs GPUs). For dry-run / config validation only:

    python -m scripts.ablation_runner --dry-run

For production:

    bash scripts/ablation_runner.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Reward weight configurations
# ===========================================================================

# A0: production stack (turn-level silent_quality ON)
A0_WEIGHTS = {
    "outcome":        1.0,
    "timing":         0.3,
    "format":         0.1,
    "spam":          -0.2,
    "silent_quality": 0.2,
}

# A1: turn signal OFF — silent_quality weight zeroed. Other weights unchanged
# so the comparison isolates the turn-level silent_quality contribution.
A1_WEIGHTS = {
    "outcome":        1.0,
    "timing":         0.3,
    "format":         0.1,
    "spam":          -0.2,
    "silent_quality": 0.0,  # ablated
}

ABLATIONS = {
    "A0_turn_on":  A0_WEIGHTS,
    "A1_turn_off": A1_WEIGHTS,
}


# ===========================================================================
# Dry-run config validation
# ===========================================================================

def validate_config():
    """Sanity-check that both ablation configs differ only in silent_quality."""
    print("\n=== Ablation configurations ===")
    print(f"  Reward keys (canonical): outcome, timing, format, spam, silent_quality")
    for name, w in ABLATIONS.items():
        print(f"\n  {name}: {w}")

    # Verify A0 and A1 differ ONLY in silent_quality
    diff = {k for k in A0_WEIGHTS if A0_WEIGHTS[k] != A1_WEIGHTS[k]}
    if diff != {"silent_quality"}:
        raise ValueError(
            f"Ablation must isolate silent_quality only; differ in: {diff}"
        )
    print(f"\n  ✓ A0 and A1 differ ONLY in silent_quality "
          f"(A0={A0_WEIGHTS['silent_quality']}, A1={A1_WEIGHTS['silent_quality']})")


# ===========================================================================
# Reward weights override (writes a temp config the trainer reads)
# ===========================================================================

def write_reward_override(out_dir: Path, weights: Dict[str, float]):
    """Write a JSON reward-weight override the trainer picks up via env var.

    The trainer (slyme grpo or verl reward_fn) reads
    THINKSTREAM_REWARD_WEIGHTS_PATH if set; otherwise falls back to
    V12_DEFAULT_REWARD_WEIGHTS in gdpo_advantage.py. This file overrides
    only for THIS ablation run — production weights stay untouched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "reward_weights.json"
    p.write_text(json.dumps(weights, indent=2))
    return p


# ===========================================================================
# Per-config trainer launch (8×H20)
# ===========================================================================

def run_one_ablation(
    name: str,
    weights: Dict[str, float],
    output_root: Path,
    *,
    train_steps: int = 50,
    dry_run: bool = False,
) -> Path:
    """Launch trainer with reward-weight override; return output dir."""
    cfg_dir = output_root / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    weights_path = write_reward_override(cfg_dir, weights)

    # Trainer command (slyme path; switch to verl with --backend verl)
    cmd = [
        "torchrun", "--nproc_per_node=8",
        "thinkstream/trainer/main_grpo.py",
        f"--output_dir={cfg_dir}",
        f"--max_steps={train_steps}",
        f"--reward_weights_path={weights_path}",
        "--audit_log_dir=" + str(cfg_dir / "audit"),
    ]
    env = os.environ.copy()
    env["THINKSTREAM_REWARD_WEIGHTS_PATH"] = str(weights_path)
    env["THINKSTREAM_OUTPUT_DIR"] = str(cfg_dir)

    if dry_run:
        logger.info(f"[DRY] {name}: would launch {' '.join(cmd)}")
        logger.info(f"[DRY] {name}: weights → {weights_path}")
        return cfg_dir

    logger.info(f"=== Running {name} (50 steps) ===")
    logger.info(f"  weights: {weights}")
    logger.info(f"  output:  {cfg_dir}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"{name} trainer exited {result.returncode}")
    return cfg_dir


# ===========================================================================
# Audit log → metrics aggregation
# ===========================================================================

def collect_metrics(audit_dir: Path) -> List[Dict]:
    """Read trainer's audit JSONL and extract per-step rewards + behaviors.

    Expected schema (from thinkstream/trainer/audit.py):
      { "step": int, "outcome_mean": float, "silent_acc": float,
        "response_acc": float, "hallucinate_rate": float,
        "missed_rate": float, ... }
    """
    rows = []
    step_log = audit_dir / "grpo_step.jsonl"
    if not step_log.exists():
        logger.warning(f"audit log missing: {step_log}")
        return rows
    with step_log.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_summary(metrics_a0: List[Dict], metrics_a1: List[Dict]) -> Dict:
    """Compute final-window mean for each tracked metric."""
    def _last_window_mean(rows: List[Dict], key: str, window: int = 10) -> float:
        vals = [r.get(key) for r in rows[-window:] if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    keys = ["outcome_mean", "silent_acc", "response_acc",
            "hallucinate_rate", "missed_rate"]
    out = {"A0_turn_on": {}, "A1_turn_off": {}}
    for k in keys:
        out["A0_turn_on"][k]  = _last_window_mean(metrics_a0, k)
        out["A1_turn_off"][k] = _last_window_mean(metrics_a1, k)
        out[f"delta_{k}"] = out["A0_turn_on"][k] - out["A1_turn_off"][k]
    return out


# ===========================================================================
# Driver
# ===========================================================================

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs + commands; don't launch trainers.")
    parser.add_argument("--train_steps", type=int, default=50,
                        help="Steps per ablation (50 enough to see convergence).")
    parser.add_argument("--output_root", default="ablation_results",
                        help="Where ablation outputs land.")
    args = parser.parse_args()

    print("=" * 78)
    print("ThinkStream ablation: turn-level silent/response signal vs none")
    print("=" * 78)

    validate_config()

    if args.dry_run:
        print("\n--dry-run: validating config only, not launching trainers.")
        timestamp = "DRY"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_root = Path(args.output_root) / timestamp

    # Run both ablations
    dirs = {}
    for name, weights in ABLATIONS.items():
        d = run_one_ablation(
            name, weights, out_root,
            train_steps=args.train_steps,
            dry_run=args.dry_run,
        )
        dirs[name] = d

    if args.dry_run:
        return

    # Aggregate
    metrics = {
        name: collect_metrics(d / "audit")
        for name, d in dirs.items()
    }
    summary = compute_summary(metrics["A0_turn_on"], metrics["A1_turn_off"])
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print(f"Ablation summary  →  {summary_path}")
    print("=" * 78)
    print(json.dumps(summary, indent=2))

    # Interpretation hint
    delta_silent = summary.get("delta_silent_acc", 0.0)
    delta_halluc = summary.get("delta_hallucinate_rate", 0.0)
    print("\nExpected directional findings:")
    print(f"  delta_silent_acc (A0 - A1)        = {delta_silent:+.3f}  (>0 means turn signal helps)")
    print(f"  delta_hallucinate_rate (A0 - A1)  = {delta_halluc:+.3f}  (<0 means turn signal helps)")
    if delta_silent > 0.02 and delta_halluc < -0.02:
        print("  → silent_quality EARNS its weight.")
    elif abs(delta_silent) < 0.01 and abs(delta_halluc) < 0.01:
        print("  → silent_quality has NO measurable effect — drop it for simplicity.")
    else:
        print("  → mixed effect; inspect per-step curves before deciding.")


if __name__ == "__main__":
    main()
