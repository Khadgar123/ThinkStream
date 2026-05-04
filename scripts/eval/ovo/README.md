# OVO-Bench evaluation matrix

Three eval entry points map to three checkpoint maturity levels.

| Script | Ckpt type | Recall | Compression | What it tests |
|--------|-----------|--------|-------------|---------------|
| `run_base.sh` | base Qwen3-VL-Instruct | ❌ | ❌ | Pure base-model OVO performance — the floor we must beat. Two sub-modes via `--form`: `offline` (full video, 64 frames) or `streaming` (chunk-by-chunk, no agent). |
| `run_sft.sh` | SFT ThinkStream | ✅ | **system-triggered** | Full v12 agent protocol with timestamped image frames. When memory pressure fires, the system inserts bare `<compress_trigger/>`; the model writes the summary from the current memory state. |
| `run_rl.sh` | post-verl GRPO ckpt | ✅ | **model self-decides** | Same timestamped-frame agent loop with `--compress_mode self`. The system never inserts a trigger; the model autonomously decides when to compress. Only meaningful with an RL-tuned ckpt — pure-SFT under this mode will overflow. |

## Why three different scripts (and why SFT does NOT self-pick)

The v12 design splits compression into two skills:

1. **Mechanism** (write a faithful summary from bounded memory) — taught by SFT compress samples
2. **Policy** (decide when to compress) — taught by the verl GRPO reward path

The old model-self-pick range SFT was removed because:
- All 8/8 same-era 2026 streaming-video papers do single-stage RL for policy
- Mixing C1 (fixed range) + C2 (self-pick) in one SFT pass introduces a distributional inconsistency the model has to resolve at inference
- Range exploration is a sequential decision, ill-suited to teacher-forcing

So during SFT eval, **always** use `--compress_mode system`. After GRPO finishes, switch to `--compress_mode self` to evaluate the RL-shaped policy.

## Quick start

Assuming your benchmark files live in `/data/ovo_bench/` and contain
`ovo-bench-formatted.jsonl` (the time-point-expanded version):

```bash
# 1) Base ckpt floor (offline, 64 frames):
bash scripts/eval/ovo/run_base.sh \
    --benchmark_dir /data/ovo_bench --form offline

# 2) Base ckpt floor (streaming, fairer to streaming agent):
bash scripts/eval/ovo/run_base.sh \
    --benchmark_dir /data/ovo_bench --form streaming

# 3) SFT ckpt (the recommended ThinkStream eval):
bash scripts/eval/ovo/run_sft.sh \
    --ckpt output/agent-sft \
    --benchmark_dir /data/ovo_bench

# 4) RL ckpt (only after GDPO has trained):
bash scripts/eval/ovo/run_rl.sh \
    --ckpt output/agent-rl \
    --benchmark_dir /data/ovo_bench
```

Results land at `${ckpt}/eval/ovo_bench/<filename>.json` with per-task
accuracy and the three category averages (Real-Time / Backward Tracing /
Forward Tracing). FT-SSR/CRR is the timing-sensitive sub-task — that's
where ThinkStream is supposed to beat the offline baseline.

## Knobs that matter

| Env / flag | Default | Effect |
|-----------|---------|--------|
| `--ngpu` | 8 | Distributed eval across N GPUs. Each rank takes a NoPad shard of the dataset. |
| `--max_new_tokens` | 30 (base) / 128 (sft+rl) | Generation budget. SFT/RL needs more because outputs include `<think>...</think><action>...</action><response>...</response>`. |
| `--think_budget` | 20 | Token budget allocated specifically for `<think>` content. Only used by streaming paths. |
| `--max_frames` | 64 | Only applies to `--form offline`. Match the paper's offline baseline row. |
| `--min_pixels`/`--max_pixels` | 130000 / 220000 | Runtime visual resolution. Same as SFT/RL defaults — do not change without a reason. |
