# OVO-Bench evaluation matrix

## Current full-video protocol

Use `eval_full.py` on the original `ovo_bench_new.json` for ChronoStream /
ThinkStream-style SFT and RL checkpoints. This path keeps the evaluation as a
full-video trajectory: each sample starts at chunk 0, advances in video order,
updates the same memory state, and scores each task at its annotated probe
time. It does not use the single-question RL segment mode.

The reported agent metrics separate several questions:

- `acc`: content accuracy. A correct answer still counts even if it was early
  or late.
- `noE`: correct and not earlier than the probe.
- `noL`: correct and not later than the probe.
- `onT`: correct exactly at the probe chunk.
- `early`, `late`, `miss`: timing error rates.
- `rec`, `r_hit`: recall count and best-effort overlap with annotated support.
- `comp`, `c_ok`: compression trigger count and parse success rate.
- `stable`: consecutive high-similarity think pairs, used to catch degenerate
  stable-think loops.

## Base VideoLLM baselines

`base.py` evaluates plain VideoLLMs without agent memory, recall, or
compression. These baselines are not meant to be weak:

- `offline`: uniformly sample frames from the entire causal prefix
  `[video_start, probe_time]`. The model sees all visual evidence that has
  appeared so far, never future frames. This is the paper's strong base-model
  setting because it gives the static VLM global observed context without a
  memory bottleneck.
- `streaming`: uniformly sample frames from `[probe_time - window, probe_time]`.
  This matches the short visual window available to the streaming agent and
  shows how much is lost when the plain VLM has no persistent memory.

For FT tasks, the base VLM is called independently at each annotated probe
time, so its timing-aware scores equal content accuracy by construction
(`acc = noE = noL = onT`, with `early = late = miss = 0`). This makes the base
strong on answer timing and isolates whether it can infer the answer from the
provided frames. ChronoStream is evaluated under the harder online trajectory:
it must carry state through the whole video and decide when to answer, while
only using the current window, text memory, compression, and optional recall.

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
