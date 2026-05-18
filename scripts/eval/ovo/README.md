# OVO Recurrent Eval

Use the original `ovo_bench_new.json`. Do not convert to the old formatted
JSONL layout.

Current ThinkStream entry point:

```bash
bash scripts/eval/ovo/run_rl_recurrent_eval.sh \
  --ckpt output/agent-rl/checkpoint-... \
  --benchmark_json /path/to/ovo_bench_new.json \
  --frames_root /path/to/frames
```

`run_sft_full.sh` and `run_rl_full.sh` are compatibility wrappers around the
same command above. They no longer call the legacy standalone OVO HF/vLLM
runner. Current method evaluation always uses the same canonical RL rollout:
`FRAME_PROTOCOL=video_meta`, `RENDER_LAYOUT=standard_query_last`, true-KV
recurrent AgentLoop, and `VAL_ONLY=true`.

For RL-path monitoring or OVO evaluation that must match training/test rollout,
first render OVO into ThinkStream multi-Q trajectory rows and parquet:

```bash
python scripts/eval/ovo/build_rl_trajectories.py \
  --benchmark-json /path/to/ovo_bench_new.json \
  --tasks OCR \
  --out-jsonl output/benchmark_splits/current/ovo/OCR/ovo_trajectories.jsonl \
  --out-parquet output/benchmark_splits/current/ovo/OCR/ovo_rl_multi_q.parquet \
  --split-policy continuous_prefix \
  --post-context-chunks 2 \
  --summary-out output/benchmark_splits/current/ovo/OCR/build_summary.json
```

Current report subcategories are `OCR ACR ATR STU FPD OJR`, `EPM ASI HLD`, and
`REC SSR CRR`. Group averages are computed from those per-task runs.

The current method-eval contract is `continuous_prefix`: each trajectory starts
at source-video chunk 0 and runs continuously through the last answer slot plus
post-context. Questions are injected at their original absolute chunks inside
that trajectory. If one source video has overlapping open question windows, the
builder emits sibling prefix tracks so the runtime prompt still has only one
active query at a time.

Sibling tracks are a pre-eval split, not a runtime fork. The builder sorts
question active windows (`ask_chunk..answer/open_until`, closed interval) and
places each question into the first sibling track with no active-window
overlap. If none fits, it starts a new independent prefix track from chunk 0.

For compressed ThinkStream eval, the builder writes planned compact-memory
boundaries into `offline_compress_chunks`, with 25-45 new chunks per compact
update. The planner chooses the lowest-risk boundary in each 25-45 chunk band,
avoiding answer-active and support/evidence windows when possible:

```bash
THINKSTREAM_RL_COMPRESS_TRIGGER_SOURCE=offline_pass2_boundaries
THINKSTREAM_COMPRESS_THRESHOLD=3200
THINKSTREAM_COMPRESS_RANGE_MIN=25
THINKSTREAM_COMPRESS_RANGE_MAX=45
```

The `64/4/6` setting is only a smoke/stress setting for compression/re-prefill
plumbing. It is too frequent for benchmark scoring.

The resulting parquet is intended for the same verl recurrent AgentLoop
validation/test path as RL rollout, so KV window behaviour, recall payload
handling, compression, and reward parsing stay shared. The validation dump is
summarized by `scripts/audit/summarize_rl_recurrent_validation.py` into
`summary.json` under the output directory.

Split policies:

- `continuous_prefix`: current method-eval path. Source-video prefix track,
  grouped by non-overlapping question windows, no 25-45s truncation.
- `strict25_45`: legacy short-window ablation. One question per row in a strict
  25-45 second/chunk window; long OVO questions can be excluded.
- `strict25_45_stateful`: legacy cut-plan ablation for compress + re-prefill.
  It emits context/scored parts but does not represent the current continuous
  method-eval contract.

`summary.health` is the compact recurrent-rollout health block:

- `answer`: question-weighted content accuracy, all-question-correct rate,
  answered rate, and answer-decision score.
- `recall`: recall alignment telemetry from the RL reward path.
- `format_runtime`: format score, action-space score, and per-chunk action score.

Metric convention:

- `trajectory_mean_correct_question_weighted`: main OVO/RL content score.
- `task_macro_trajectory_mean_correct`: mean over task scores.
- REC and CRR keep one active query across multiple expected answer chunks.
- SSR probes are placed on continuous prefix tracks; overlapping probes are
  split into sibling tracks to preserve the single-active-query runtime
  contract.

Use `scripts/eval/ovo/compare_runs.py` to compare result JSON files.
