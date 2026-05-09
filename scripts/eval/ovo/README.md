# OVO Full-Video Eval

Use the original `ovo_bench_new.json`. Do not convert to the old formatted
JSONL layout.

Supported entry points:

```bash
# Strong base VLM, no memory/recall/compression.
bash scripts/eval/ovo/run_base.sh \
  --ckpt /path/to/Qwen3-VL-8B-Instruct \
  --benchmark_json /path/to/ovo_bench_new.json \
  --video_root /path/to/videos \
  --frames_root /path/to/frames \
  --form offline

# SFT checkpoint, system-triggered compression.
bash scripts/eval/ovo/run_sft_full.sh \
  --ckpt output/agent-sft/checkpoint-... \
  --benchmark_json /path/to/ovo_bench_new.json \
  --video_root /path/to/videos \
  --frames_root /path/to/frames

# RL checkpoint, model-self compression.
bash scripts/eval/ovo/run_rl_full.sh \
  --ckpt output/agent-rl/checkpoint-... \
  --benchmark_json /path/to/ovo_bench_new.json \
  --video_root /path/to/videos \
  --frames_root /path/to/frames
```

All ThinkStream eval wrappers use the same canonical prompt contract as SFT/RL:
`FRAME_PROTOCOL=video_meta` and `RENDER_LAYOUT=timeline_video_imagepad`.

`summary.health` is the compact abnormal-behavior block:

- `answer`: content, no-early/no-late/on-time accuracy, missing/early/late rates.
- `recall`: frequency, support hit, before-answer usage, accuracy with/without recall.
- `compression`: frequency, success, and system trigger/range/calculation checks.
- `format_runtime`: parse/action errors, stable-think rate, token maxima, action histogram.

Use `scripts/eval/ovo/compare_runs.py` to compare result JSON files.
