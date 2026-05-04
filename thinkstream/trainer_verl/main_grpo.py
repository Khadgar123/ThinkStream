"""Retired verl GRPO shim.

Production RL now lives in the vendored verl recipe:

    THINKSTREAM_DATA_ROOT=data/agent_v5/<batch_id> \
    LLM=/path/to/sft/checkpoint \
    bash scripts/grpo_train_verl.sh

or directly:

    bash verl/recipe_thinkstream/run_thinkstream_grpo.sh

This module remains only so old imports fail with a clear migration message.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Retired ThinkStream GRPO shim")
    parser.add_argument(
        "--config",
        default=None,
        help="Ignored. Use scripts/grpo_train_verl.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration target and exit.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Use: bash scripts/grpo_train_verl.sh")
        print("Recipe: verl/recipe_thinkstream/")
        return

    raise SystemExit(
        "thinkstream.trainer_verl.main_grpo is retired. "
        "Use bash scripts/grpo_train_verl.sh."
    )


if __name__ == "__main__":
    main()
