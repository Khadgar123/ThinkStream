"""
Agent Data Construction Pipeline v8.0

4-stage pipeline:
  1. Teacher Evidence Graph
     1-A: Independent chunk annotation (2 frames, parallel)
     1-B: Entity alignment + state change detection (2 x 397B/video)
  2. Question-blind Streaming Rollout (thinks + compressions + snapshots)
  3. Task Mining + Sample Generation
     3-A: Task Card generation (per-family 397B calls)
     3-B: Placement + behavior sequence planning (pure program)
     3-C: Trajectory sample generation (397B response/query)
  4. Verify + Filter

Architecture:
  - Per-timestep independent training samples
  - 3 prompts: SYSTEM_PROMPT / POST_RECALL / COMPRESS
  - Queries zone for persistent question tracking
  - Behavior sequences: immediate_response / recall / event_watch / multi_response
"""
