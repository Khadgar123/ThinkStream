# StreamingBench Eval

Current ThinkStream method evaluation uses recurrent RL trajectories, not the
fixed-window base VLM manifest.

Build the current split:

```bash
python scripts/eval/streamingbench/build_rl_trajectories.py \
  --csv-dir /path/to/StreamingBench/StreamingBench \
  --video-root /path/to/StreamingBench/extracted \
  --frames-root /path/to/StreamingBench/frames_fps2 \
  --out-jsonl output/benchmark_splits/current/streamingbench/streaming_trajectories.jsonl \
  --out-parquet output/benchmark_splits/current/streamingbench/streaming_rl_multi_q.parquet \
  --split-policy continuous_prefix \
  --sample-per-task-type 0 \
  --summary-out output/benchmark_splits/current/streamingbench/rl_build_summary.json
```

`continuous_prefix` groups questions by source video, starts each trajectory at
chunk 0, and runs through the last answer slot plus post-context. Overlapping
open-question windows are split into sibling prefix tracks.

Sibling tracks are created before evaluation: questions are greedily packed by
non-overlapping active windows (`ask_chunk..answer/open_until`, closed
interval). A question whose active window overlaps every existing sibling track
starts a new independent prefix track from chunk 0.

For ThinkStream compressed evaluation, the builder writes planned compact-memory
boundaries into `offline_compress_chunks`, with 25-45 new chunks per compact
update. The planner chooses the lowest-risk boundary in each 25-45 chunk band,
avoiding answer-active and support/evidence windows when possible:

```bash
THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE=offline_pass2_boundaries
THINKSTREAM_COMPRESS_THRESHOLD=3200
THINKSTREAM_COMPRESS_RANGE_MIN=25
THINKSTREAM_COMPRESS_RANGE_MAX=45
```

Do not use the earlier `64/4/6` smoke setting for benchmark scoring; it
compresses every few chunks and is only useful for plumbing checks.

Legacy/base paths:

- `build_split_manifest.py` freezes 25-45s fixed windows for `base_vllm.py`.
- `base_vllm.py`, `run_base_vllm_8gpu.sh`, `run_big_matrix.sh`, and
  `start_vllm8_runtime.sh` are base-model baselines, not current ThinkStream
  recurrent evaluation.
