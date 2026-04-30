"""verl GRPO entrypoint for ThinkStream — cross-validation against slyme path.

Usage (8×H20 production):
    python -m thinkstream.trainer_verl.main_grpo \
        --config recipe/v12_grpo.yaml

The config is a Hydra YAML (verl convention). See recipe/v12_grpo.yaml for
the production setting.

This module is intentionally THIN — most of the algorithm lives in
`thinkstream.trainer.v12_rewards` and `thinkstream.trainer.gdpo_advantage`,
both pytorch-only and shared with the slyme path.
"""
from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ThinkStream GRPO via verl")
    parser.add_argument(
        "--config",
        default="recipe/v12_grpo.yaml",
        help="Hydra config path (verl convention).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without launching training.",
    )
    args = parser.parse_args()

    # verl is GPU-only; keep import deferred so this module is importable
    # in CPU envs (for testing / structure validation).
    try:
        from verl.trainer.main_ppo import main as verl_main
    except ImportError as e:
        raise SystemExit(
            f"verl not installed: {e}\n"
            f"Install: pip install verl>=0.4.0\n"
            f"Repo:    https://github.com/volcengine/verl"
        )

    # Override verl defaults with our reward + dataset adapters.
    # verl 0.4 reads `reward_model.reward_manager` and instantiates from a
    # qualified path; ours points at `thinkstream.trainer_verl.reward_fn`.
    os.environ.setdefault("VERL_REWARD_FN", "thinkstream.trainer_verl.reward_fn:compute_thinkstream_reward")
    os.environ.setdefault("VERL_DATASET_FN", "thinkstream.trainer_verl.dataset:build_rl_dataset")

    if args.dry_run:
        logger.info("Dry run: resolved config + reward + dataset adapters")
        logger.info(f"  config: {args.config}")
        logger.info(f"  reward: {os.environ['VERL_REWARD_FN']}")
        logger.info(f"  dataset: {os.environ['VERL_DATASET_FN']}")
        return

    # Hand off to verl. verl reads --config arg from sys.argv and runs hydra.
    import sys
    sys.argv = ["verl-train", f"--config-path=.", f"--config-name={args.config}"]
    verl_main()


if __name__ == "__main__":
    main()
