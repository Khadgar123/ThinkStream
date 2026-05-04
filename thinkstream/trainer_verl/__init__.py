"""Archived compatibility adapters for early verl experiments.

Production GRPO no longer enters through ``thinkstream.trainer_verl``.
Use the vendored recipe instead:

    bash scripts/grpo_train_verl.sh

The active code is:

  verl/recipe_thinkstream/thinkstream.py
  verl/recipe_thinkstream/streaming_agent_loop.py
  verl/recipe_thinkstream/configs/thinkstream_grpo.yaml

This package remains importable for old tests and reward parity utilities.
"""
