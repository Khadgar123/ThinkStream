# Benchmark Eval Schemes

Current ThinkStream method evaluation uses the verl recurrent AgentLoop with
`video_meta` frames, true-KV streaming state, runtime compression, recall tool
responses, and validation-only rollout. The canonical split policy is
`continuous_prefix`.

## Current Scheme

- `continuous_prefix`: group questions by source video, start each trajectory at
  chunk 0, and run continuously through the last answer slot plus
  post-context. Questions are injected at their original absolute chunks. If
  two questions would be open at the same time, the builder creates sibling
  prefix tracks to keep the runtime single-active-query contract.

Overlap handling is a pre-eval split, not a runtime fork. For each source video,
the builder sorts question active windows (`ask_chunk..answer/open_until`,
closed interval) and greedily places each question into the first sibling track
whose existing active windows do not overlap. If none fits, it creates a new
sibling track. Each sibling track is then evaluated as an independent
continuous-prefix trajectory from chunk 0; it does not share KV or compact
memory with sibling tracks.

Current compression eval uses planned compact-memory boundaries written into
`offline_compress_chunks`, with 25-45 new ordinary chunks per compact update.
The planner chooses a boundary in each 25-45 chunk band that avoids
answer-active windows and support/evidence windows when possible:

```bash
THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE=offline_pass2_boundaries
THINKSTREAM_COMPRESS_THRESHOLD=3200
THINKSTREAM_COMPRESS_RANGE_MIN=25
THINKSTREAM_COMPRESS_RANGE_MAX=45
```

`THINKSTREAM_COMPRESS_RANGE_*` documents the planned split policy; offline
mode triggers exactly at the JSONL/parquet boundaries. The old `64/4/6`
setting is only a stress/smoke setting for checking structure and re-prefill
plumbing. It is too frequent for benchmark scoring.

Current builders:

- OVO: `scripts/eval/ovo/build_rl_trajectories.py --split-policy continuous_prefix`
- StreamingBench: `scripts/eval/streamingbench/build_rl_trajectories.py --split-policy continuous_prefix`

The paired parquet is produced by the builders through
`scripts.agent_data.build_verl_parquet` in multi-Q mode, so one parquet row is
one recurrent trajectory.

## Legacy / Ablation Schemes

- `strict25_45`: one question per short 25-45s window. This is a short-window
  ablation and can remove long OVO questions.
- `strict25_45_stateful`: old OVO cut-plan for compress/re-prefill experiments.
  It emits context/scored parts and is not the current method-eval contract.
- `streamingbench/build_split_manifest.py` and `streamingbench/base_vllm.py`:
  base-model fixed-window VLM baselines, not ThinkStream recurrent eval.
- OVO `base.py`, `ref_streaming.py`, and matrix/sweep launchers: standalone
  baseline/debug paths. They are kept for comparisons, not current method
  evaluation.

## Existing Current Split Files

- `output/benchmark_splits/current/ovo/`
- `output/benchmark_splits/current/streamingbench/`

Each directory should contain a trajectory JSONL, a paired multi-Q parquet, and
a build summary. `build_summary.json` records the split policy and span stats.
