"""
Agent Data Construction Pipeline v12.15

Main data-construction stages:
  1. Teacher Evidence Graph
     1-A: Independent chunk annotation (2 frames, parallel)
     1-B: Entity alignment + state change detection
  2. Question-blind Streaming Rollout (thinks + compressions + snapshots)
  3. Task Mining + Sample Generation
     3-A: Task Card generation (per-family 397B calls)
     3-B: Placement + trajectory planning (pure program)
     3-C: Trajectory sample generation
     3-E: Verify + tag, retaining trajectory continuity
  4. Trajectory grouping
  5. Qwen3-VL messages conversion

Architecture:
  - Per-timestep samples plus trajectory rows for RL/eval
  - v12 Qwen tool protocol with answer / recall / compress actions
  - Queries zone for persistent question tracking
  - Mechanisms: direct / recall_demo / silent_then_response / multi_emit
"""
