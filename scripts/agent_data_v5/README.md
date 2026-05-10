# Agent Data v5 Pipeline

This package is the active data-construction surface. Run it as a module:

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
python -m scripts.agent_data_v5.pipeline run \
  --api_base http://HOST:8000/v1 \
  --model /path/to/Qwen3.5-397B-A17B-FP8 \
  --videos_jsonl data/agent_v5/batch2_videos.jsonl \
  --num_videos 500
```

Pass order:

```text
pass1a_evidence.py  -> evidence_1a/
pass1b_enrich.py    -> evidence_1b/
pass2_rollout.py    -> rollout/
pass3a_cards.py     -> task_cards/
pass3b_placement.py -> placements/
pass3c_samples.py   -> samples_3c/
pass3e_verify.py    -> verified/
pass4.py            -> final/*_trajectories.jsonl and flat splits
pass5_messages.py   -> final/*_messages.jsonl for SFT
```

The expensive teacher passes are protocol-neutral after frame extraction:
pass1/pass2/pass3/pass4 store frame paths, timestamps, memory, questions,
options, answer_form, accepted answers, and answer chunks. The student-facing
visual carrier is late-bound at render/eval time:

```bash
# Current robust protocol: explicit timestamp text + image items.
python -m scripts.agent_data_v5.pass5_messages \
  --final-dir data/agent_v5/batch2/final \
  --output-dir data/agent_v5/batch2/rendered/ts_image \
  --frame-protocol ts_image

# Native Qwen video-metadata protocol over the same pre-extracted frames.
python -m scripts.agent_data_v5.pass5_messages \
  --final-dir data/agent_v5/batch2/final \
  --output-dir data/agent_v5/batch2/rendered/video_meta \
  --frame-protocol video_meta
```

Both rendered variants keep the same sample schema, memory, queries,
visual_window, answers, options, and split assignment. The prompt semantics are
also aligned; only the system-prompt sentence describing the visual carrier and
the media content item differ. `format_queries_block()` renders an explicit
`Answer format:` line for the active question, so SFT/RL/eval all tell the model
whether to answer with a single MC letter, letter+text, text-only, number,
binary answer, short exact phrase, or descriptive text.

Build matching verl parquets from the same canonical trajectories:

```bash
python -m scripts.agent_data_v5.build_verl_parquet \
  --jsonl data/agent_v5/batch2/final/train_rl_trajectories.jsonl \
  --out data/agent_v5/batch2/rendered/ts_image/train_rl_multi_q.parquet \
  --multi_q --frame-protocol ts_image

python -m scripts.agent_data_v5.build_verl_parquet \
  --jsonl data/agent_v5/batch2/final/train_rl_trajectories.jsonl \
  --out data/agent_v5/batch2/rendered/video_meta/train_rl_multi_q.parquet \
  --multi_q --frame-protocol video_meta
```

Train/eval with the same protocol end to end:

```bash
# Pass45 emits rendered/video_meta_standard_query_last by default.
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
bash scripts/run_sft_rl.sh

bash scripts/eval/ovo/run_sft_full.sh \
  --ckpt output/agent-sft/checkpoint-... \
  --benchmark_json /path/to/ovo_bench_new.json \
  --video_root /path/to/videos \
  --frames_root data/agent_v5/batch2/frames
```

The old `ts_image` paths are archived. Current SFT/RL/eval launchers use
`video_meta_standard_query_last`.

`v2/` is the current pass3 design implementation, not a deprecated folder.
It owns the card taxonomy, placement rules, and LLM prompts used by
pass3a/pass3b/pass3c.

Every new batch should use its own `THINKSTREAM_DATA_ROOT`, which gives one
self-contained directory with frames, pass caches, audits, and final files.
The pipeline writes `batch_manifest.json` and `selected_videos.jsonl` at the
batch root for traceability.
