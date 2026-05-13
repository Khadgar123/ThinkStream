# OVO Full-Video Eval

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
  --out-jsonl data/ovo_rl/ovo_trajectories.jsonl \
  --out-parquet data/ovo_rl/ovo_rl_multi_q.parquet \
  --split-policy query_span \
  --max-span-chunks 512 \
  --pre-context-chunks 64 \
  --post-context-chunks 2
```

The converter packs non-overlapping questions from the same video and task into
one trajectory, splits overlapping active-query intervals, and keeps each
question's ask chunk and answer chunk in the same segment unless a single long
OVO probe interval already exceeds the soft span limit. Use
`--pack-across-tasks` only for throughput sweeps where mixed-task trajectory
metrics are acceptable. The resulting parquet is intended for the same verl
recurrent AgentLoop validation/test path as RL rollout, so KV window behaviour,
recall payload handling, and reward parsing stay shared. The validation dump is
summarized by `scripts/audit/summarize_rl_recurrent_validation.py` into
`summary.json` under the output directory.

Split policies:

- `query_span`: current default; packs non-overlapping questions from the same
  video/task and gives each group a dynamic segment around the required
  ask/answer interval.
- `casia`: matches the reference CASIA OVO conversion semantics: every
  formatted probe is an independent row, `video_start/video_end` define the
  segment, and the query fires at `video_end`.
- `short20_40`: one question per row; short questions target
  `--short-min-span-chunks` to `--short-max-span-chunks` chunks, while a single
  inherently long question is allowed to exceed that target rather than cutting
  out its required context.
- `short20_40_stateful`: cut-plan mode for long questions. It first applies
  `short20_40`, then splits any row above the short max into contiguous
  stateful parts. Context-only parts intentionally carry no scored question;
  this JSONL requires an evaluator/rollout that carries memory state across
  `stateful_split_parent_id` parts before scoring.

`summary.health` is the compact recurrent-rollout health block:

- `answer`: question-weighted content accuracy, all-question-correct rate,
  answered rate, and answer-decision score.
- `recall`: recall alignment telemetry from the RL reward path.
- `format_runtime`: format score, action-space score, and per-chunk action score.

Metric convention:

- `trajectory_mean_correct_question_weighted`: main OVO/RL content score.
- `task_macro_trajectory_mean_correct`: mean over task scores.
- REC and CRR keep one active query across multiple expected answer chunks;
  SSR is expanded into per-probe trajectories because simultaneous active step
  probes cannot be represented by the current single active-query state.

Use `scripts/eval/ovo/compare_runs.py` to compare result JSON files.
