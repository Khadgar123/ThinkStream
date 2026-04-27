# Held-out test set evaluation matrix

The test set is `data/agent_v5/final/test.jsonl` — 1,600 per-timestep
samples, video-disjoint from `train_sft`, `train_rl`, and `val`. Three
scripts mirror the OVO eval matrix.

| Script | Ckpt type | What runs | What it measures |
|--------|-----------|-----------|------------------|
| `run_base.sh --form offline` | base Qwen3-VL-Instruct | Extract Q + gold from test.jsonl response / recall_response samples, ask base offline (full video [0, video_end], 64 frames). | **Base ceiling** under full information. Yes/No, integer, letter scoring (OVO-style prefix/substring). |
| `run_base.sh --form streaming` | base Qwen3-VL-Instruct | Same extraction, but base sees only the visual_window slice the streaming agent has at decision time (12 chunks × 2 sec = 24 sec by default). | **Apples-to-apples baseline**: base with the SAME visual context our agent uses — measures what the agent protocol adds beyond a naive sliding window. |
| `run_sft.sh` | SFT ckpt | Pass 1: teacher-forced (loss + L2 argmax via trainer.evaluate). Pass 2: generative action keyword accuracy. | Did the model learn the agent format on held-out test? Pass 1 matches wandb in-loop eval byte-for-byte. |
| `run_rl.sh` | post-GDPO RL ckpt | Same two-pass as SFT but on the RL ckpt path. | Did RL preserve / improve per-class action accuracy? **Does not** measure autonomous compress decisions (test.jsonl is per-timestep slices that don't trigger memory pressure — use `scripts/eval/ovo/run_rl.sh` for that). |

## Why teacher-forced + generative (two-pass)

| Metric | Pass 1 (TF) | Pass 2 (gen) | Why both |
|--------|-------------|--------------|----------|
| `eval/loss` | ✅ | ❌ | Same formula as training: monitors per-token weighted loss on test. |
| `eval/action_accuracy` | ✅ argmax-vs-gold | ✅ generate-then-parse | TF is optimistic upper bound (sees gold prefix); gen is real inference accuracy. Gap should be 5-10pp. |
| `eval/silent_eos_rate` | ✅ | ✅ (`post_continued` flag inverted) | TF tells you if the transition position would argmax to `<\|im_end\|>`; gen tells you if the model actually stops generating. |
| `eval/post_action_acc_<class>` | ✅ | implicit in gen | Per-class transition correctness. Only TF reports this directly. |

## Quick start (after SFT and/or RL finish)

```bash
# 1a) Base ceiling — full video buffer (offline VLM upper bound)
bash scripts/eval/test_set/run_base.sh --form offline --n 200

# 1b) Base apples-to-apples — same 24-sec visual window as our agent
bash scripts/eval/test_set/run_base.sh --form streaming --n 200

# 2) SFT eval — the recommended post-train report
bash scripts/eval/test_set/run_sft.sh \
    --ckpt output/agent-sft \
    --ngpu 8 --n_gen 200

# 3) RL eval — same shape as SFT, output goes to RL ckpt's dir
bash scripts/eval/test_set/run_rl.sh \
    --ckpt output/agent-rl \
    --ngpu 8 --n_gen 200
```

**Reading the four numbers**:
- If `SFT generative > base streaming` → agent protocol is helping
- If `SFT generative ≈ base offline` → agent matches the offline ceiling on this distribution
- If `SFT generative < base streaming` → agent protocol is hurting (regressed) — investigate
- If `RL ≈ SFT` on test set → RL preserved per-step decisions; benefit shows up in OVO (long-horizon)

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
