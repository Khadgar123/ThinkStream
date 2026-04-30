"""verl-based RL trainer (parallel implementation to slyme `trainer/`).

Goal: cross-validate the slyme implementation by running the same algorithm
on the same data through the public verl framework. Algorithm modules
(rewards, advantages, state evolution) are SHARED across both:

  thinkstream/trainer/v12_rewards.py   ← pure pytorch, framework-agnostic
  thinkstream/trainer/gdpo_advantage.py← pure pytorch, framework-agnostic
  thinkstream/trainer/v12_rollout.py   ← state evolution, framework-agnostic

verl-specific glue (this directory):
  reward_fn.py        — verl-format reward function (wraps v12_rewards)
  multiturn_rollout.py — verl multi-turn rollout config (chunk-level)
  dataset.py          — verl dataset adapter (reads pass5 messages format)
  main_grpo.py        — verl entrypoint (train script)

Cross-validation invariant:
  Same trajectory_id × group_size × seed → same advantage vector ± numerical
  noise on both slyme and verl paths.
"""
