# Held-out test set evaluation matrix

The test set is `data/agent_v5/final/test.jsonl` — 1,600 per-timestep
samples, video-disjoint from `train_sft`, `train_rl`, and `val`. Three
scripts mirror the OVO eval matrix.

| Script | Ckpt type | Pass 1 (teacher-forced) | Pass 2 (generative) | What it measures |
|--------|-----------|-------------------------|----------------------|------------------|
| `run_base.sh` | base Qwen3-VL-Instruct | ❌ skipped (special tokens are untrained on base — loss is uninformative) | ✅ | Floor for action accuracy. Base will mostly produce free-form text, so action_acc near 0% expected. |
| `run_sft.sh` | SFT ckpt | ✅ same metrics as wandb in-loop eval | ✅ | Did the model learn the agent format on test (held-out from train_sft AND from val)? |
| `run_rl.sh` | post-GDPO RL ckpt | ✅ | ✅ | Did RL preserve / improve per-class action accuracy from SFT? **Does not** measure autonomous compress decisions (use `scripts/eval/ovo/run_rl.sh` for that). |

## Why teacher-forced + generative (two-pass)

| Metric | Pass 1 (TF) | Pass 2 (gen) | Why both |
|--------|-------------|--------------|----------|
| `eval/loss` | ✅ | ❌ | Same formula as training: monitors per-token weighted loss on test. |
| `eval/action_accuracy` | ✅ argmax-vs-gold | ✅ generate-then-parse | TF is optimistic upper bound (sees gold prefix); gen is real inference accuracy. Gap should be 5-10pp. |
| `eval/silent_eos_rate` | ✅ | ✅ (`post_continued` flag inverted) | TF tells you if the transition position would argmax to `<\|im_end\|>`; gen tells you if the model actually stops generating. |
| `eval/post_action_acc_<class>` | ✅ | implicit in gen | Per-class transition correctness. Only TF reports this directly. |

## Quick start (after SFT and/or RL finish)

```bash
# 1) Base floor — generative-only (skips Pass 1 by design)
bash scripts/eval/test_set/run_base.sh --n 200

# 2) SFT eval — the recommended post-train report
bash scripts/eval/test_set/run_sft.sh \
    --ckpt output/agent-sft \
    --ngpu 8 --n_gen 200

# 3) RL eval — same shape as SFT, output goes to RL ckpt's dir
bash scripts/eval/test_set/run_rl.sh \
    --ckpt output/agent-rl \
    --ngpu 8 --n_gen 200
```

Reports land at:
- `${CKPT}/eval/test_stream_agent_test/metrics.json`     (Pass 1)
- `${CKPT}/eval/test_stream_agent_test/gen_action.json`  (Pass 2)

## Knobs

- `--ngpu`: GPUs for Pass 1 teacher-forced eval. Pass 2 is single-GPU (autoregressive, can't parallelize a single sample).
- `--n_gen`: how many samples to run Pass 2 on. Default 200 (~2-5 min on 1 GPU). Set higher for tighter confidence.
- `--no_tf`: skip Pass 1 (e.g., to re-run only Pass 2)
- `--no_gen`: skip Pass 2 (e.g., to iterate quickly on Pass 1 reports)
- `--dataset`: registry name. Default `stream_agent_test`. Could swap for `stream_agent_val` to reproduce the wandb eval curve number on the same val set.

## How this differs from OVO eval

| | Test set eval (this dir) | OVO eval (`scripts/eval/ovo/`) |
|---|---|---|
| Input format | Per-timestep agent samples (one chunk = one decision) | MCQ benchmark (one Q&A at a `realtime`) |
| What's measured | Action keyword + transition correctness | MCQ accuracy (A/B/C/D, Yes/No, integer) |
| Compression | Not exercised (per-timestep slice doesn't span enough chunks) | Exercised in `--use_agent_loop` mode (system or self) |
| Use case | "Did SFT/RL learn the per-step decision?" | "Does the agent answer the right question at the right time on a real benchmark?" |

Run both. Test-set eval validates the per-timestep policy; OVO eval validates the assembled streaming behavior.
