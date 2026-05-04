# ThinkStream Project Structure

This repository has four active surfaces. Keep new code in one of these
places unless it is explicitly an audit or paper artifact.

```text
scripts/agent_data_v5/       data construction CLI and pass implementations
thinkstream/sft/             SFT dataset loading and trainer entrypoint
verl/recipe_thinkstream/     production GRPO recipe and streaming agent loop
scripts/eval/ + thinkstream/eval/
                             evaluation entrypoints and shared eval engines
thinkstream/data/agent_protocol.py
                             shared prompt/content protocol for pass/SFT/RL/eval
```

## Data Construction

The canonical pipeline is:

```text
pass1a  evidence extraction per chunk        -> evidence_1a/
pass1b  evidence enrichment                  -> evidence_1b/
pass2   student-behavior rollout/thinks      -> rollout/
pass3a  question/card generation             -> task_cards/
pass3b  placement and timeline planning      -> placements/
pass3c  one-question-at-a-time samples       -> samples_3c/
pass3e  non-destructive verification tags    -> verified/
pass4   trajectory grouping and split files  -> final/
pass5   ShareGPT messages for SFT            -> final/*_messages.jsonl
```

`scripts/agent_data_v5/v2/` is not a deprecated project version. It is the
active pass3 card/placement design package imported by pass3a/pass3b/pass3c.
Do not delete it unless those imports are migrated.

## Batch Layout

Each generated batch should have one independent data root:

```text
data/agent_v5/batch2/
  batch_manifest.json
  selected_videos.jsonl
  video_registry.jsonl
  frames/<video_id>/frame_000001.jpg
  evidence_1a/<video_id>.json
  evidence_1b/<video_id>.json
  rollout/<video_id>.json
  task_cards/<video_id>.json
  placements/<video_id>.json
  samples_3c/<video_id>.json
  verified/<video_id>.json
  audits/*.json
  final/
    train_sft_messages.jsonl
    train_rl_trajectories.jsonl
    val_messages.jsonl
    test_messages.jsonl
    dataset_info.json
```

Use one of these equivalent selectors:

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2
AGENT_DATA_DIR=data/agent_v5/batch2
THINKSTREAM_BATCH=batch2
```

`THINKSTREAM_DATA_ROOT` is preferred. `AGENT_DATA_DIR` is kept for older
scripts; it now means the batch root, not a random final-file directory.
SFT still accepts `AGENT_DATA_DIR=data/agent_v5/batch2/final` for backward
compatibility.

For a pre-balanced batch list, pass it directly instead of letting the
pipeline reselect videos:

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
python -m scripts.agent_data_v5.pipeline run \
  --api_base http://HOST:8000/v1 \
  --model /path/to/Qwen3.5-397B-A17B-FP8 \
  --videos_jsonl data/agent_v5/batch2_videos.jsonl \
  --num_videos 500
```

## Shared Visual Protocol

All active pass/SFT/RL/eval paths use the same pre-extracted-frame protocol:

```text
Frame timestamp t=12.5s (latest chunk).
<image or image_url item>
```

We do not send pre-extracted JPEG frames as `video_url`/`data:video/jpeg` in
the hot path. vLLM still schedules the image tensors, and the visible
timestamp text supplies the temporal anchor consistently across Qwen3-VL,
SFT, GRPO rollout, and eval.

## SFT

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
bash scripts/sft_per_timestep.sh
```

SFT reads `final/train_sft_messages.jsonl` through
`thinkstream/sft/data_list.py`.

## RL

Only the vendored verl recipe is the production RL backend:

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
LLM=output/agent-sft-v12.22/checkpoint-... \
bash scripts/grpo_train_verl.sh
```

The old `scripts/grpo_train.sh` only forwards to the verl launcher.
`thinkstream/trainer/grpo.py` is retained for archived tests/parity helpers,
not as a supported training entrypoint.

## Eval

Use the wrappers under `scripts/eval/`. The active streaming eval engines
share the v12 answer matcher and timestamped-frame input construction with
SFT/RL.
