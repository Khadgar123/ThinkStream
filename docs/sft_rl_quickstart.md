# ThinkStream SFT + RL Quickstart

This checkout is meant to run from source. Do not `pip install verl`; the
custom verl fork is vendored under `verl/` and is injected through
`PYTHONPATH` by the launchers.

## 1. Environment

```bash
bash scripts/setup_thinkstream_env.sh
export THINKSTREAM_ENV=/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream
```

The current validated local environment uses Python 3.12, CUDA 12.6 wheels,
PyTorch 2.8.0, transformers 4.57.3, vLLM 0.11.0, Ray 2.55.1, deepspeed 0.17.1,
flash_attn 2.8.3, and flashinfer-python 0.6.7. Core versions are pinned in
`requirements.txt`.

## 2. Data Layout

Point `THINKSTREAM_DATA_ROOT` at a generated agent-v5 batch root. The direct
launcher expects rendered SFT messages and RL trajectories/parquets:

```text
data/agent_v5/<batch_root>/
  final/
    train_rl_trajectories.jsonl
    val_trajectories.jsonl
  frames/
  rendered/video_meta_timeline_video_imagepad/
    train_sft_messages.jsonl
    val_messages.jsonl
    train_rl_multi_q.parquet       # auto-built if missing
    val_rl_multi_q.parquet         # auto-built if missing
```

The supported project entry is intentionally fixed to
`FRAME_PROTOCOL=video_meta` and `THINKSTREAM_RENDER_LAYOUT=timeline_video_imagepad`.
Archived `ts_image` and standard block layouts are not produced by the main
pipeline or launchers.

## 3. Multi-Batch Training Scheme

Use this when several generated batches need to become one SFT/RL/eval/test
root. The split is video-disjoint and balances task family, recall frequency,
and compression frequency across SFT, RL, val, and test.

```bash
bash scripts/prepare_training_data.sh \
  --out data/agent_v5/scheme_v1 \
  --batches data/agent_v5/batch1 data/agent_v5/batch2 data/agent_v5/batch3 \
  --sft-videos 150 \
  --rl-videos 175 \
  --val-videos 50 \
  --test-videos 50 \
  --force
```

The output already contains canonical rendered SFT messages and RL parquet
files under `rendered/video_meta_timeline_video_imagepad/`.

## 4. One-Command SFT -> RL

```bash
THINKSTREAM_DATA_ROOT=data/agent_v5/<batch_root> \
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
bash scripts/run_sft_rl.sh
```

Outputs go to `output/agent-sft-$RUN_ID` and `output/agent-rl-$RUN_ID`; logs go
to `logs/$RUN_ID`.

To skip SFT and start RL from an existing checkpoint:

```bash
RUN_SFT=0 \
SFT_CKPT=output/agent-sft-.../checkpoint-... \
THINKSTREAM_DATA_ROOT=data/agent_v5/<batch_root> \
bash scripts/run_sft_rl.sh
```

## 5. RL Defaults

The default RL path uses full-video recurrent rollout:

```bash
THINKSTREAM_RECURRENT_MODE=recurrent
THINKSTREAM_RL_EPISODE_MODE=full
MULTI_Q=1
BATCH_SIZE=1
GROUP_SIZE=8
TP_SIZE=2
MAX_NEW_TOKEN=4096
MAX_CHUNKS=420
PPO_MAX_TOKEN_LEN_PER_GPU=65536
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=65536
FREEZE_VISION_TOWER=true
```

Runtime spill/cache files are kept under `.runtime/$RUN_NAME` so Ray/vLLM do
not fill `/tmp`. Override with `RUNTIME_ROOT=/path/to/runtime` if needed.

## 6. Monitoring

For recurrent RL, keep the compact audit enabled and monitor with:

```bash
python scripts/monitor_rl_recurrent.py \
  --run-dir output/agent-rl-$RUN_ID \
  --tail-audit 500
```

The monitor reports action rows, recall frequency, recall/support overlap,
answer timing, answer accuracy, compression parse health, JSON/action parse
errors, and time-range validity.

OVO full eval writes the same abnormal-behavior signals in one compact
`summary.health` block:

- `answer`: content accuracy, no-early/no-late/on-time accuracy, missing/early/late rates.
- `recall`: recall frequency, support-hit rate, recall-before-answer rate, accuracy with and without recall, blocked second-step recall rate.
- `compression`: compression frequency, success rate, and system trigger/range/calculation checks.
- `format_runtime`: step errors, format/action-space errors, stable-think rate, max prompt/think tokens, and action histogram.
