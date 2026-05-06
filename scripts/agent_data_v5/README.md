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
# SFT reads THINKSTREAM_FINAL_DIR when set; otherwise it auto-picks
# data root/rendered/$FRAME_PROTOCOL if that directory exists.
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
FRAME_PROTOCOL=ts_image \
bash scripts/sft_per_timestep.sh

THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
FRAME_PROTOCOL=ts_image \
LLM=output/agent-sft-v12.26-ts_image/checkpoint-best \
bash scripts/grpo_train_verl.sh

FRAME_PROTOCOL=ts_image \
bash scripts/eval/run_matrix.sh all
```

For the native-video AB run, switch only `FRAME_PROTOCOL=video_meta` and use
the matching rendered messages/parquets/checkpoint output names. The teacher
passes do not need to be re-run because both variants consume the same
pre-extracted frames and canonical trajectories.

`v2/` is the current pass3 design implementation, not a deprecated folder.
It owns the card taxonomy, placement rules, and LLM prompts used by
pass3a/pass3b/pass3c.

Every new batch should use its own `THINKSTREAM_DATA_ROOT`, which gives one
self-contained directory with frames, pass caches, audits, and final files.
The pipeline writes `batch_manifest.json` and `selected_videos.jsonl` at the
batch root for traceability.
