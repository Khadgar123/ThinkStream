"""
Agent Data Construction Pipeline v5.0

5-stage pipeline:
  1. Teacher Evidence Graph (rich structured captions, hidden from student)
  2. Question-blind Streaming Rollout (student observations + compressions)
  3. Task Planning (per-type mining with action minimality)
  4. Question-aware Forks (generate final training samples)
  5. Verify + Filter (leakage, grounding, format, difficulty)

Architecture:
  - Per-timestep independent training samples (方案 A)
  - Three-layer text separation (teacher_caption / student_observation / compressed_summary)
  - Visibility matrix + action minimality for every sample
"""
