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
  --form offline_full

# 8-GPU sharded base eval, no recall/compression.
FORM=streaming MAX_FRAMES=24 bash scripts/eval/ovo/run_base_8gpu.sh
FORM=offline_full MAX_FRAMES=256 bash scripts/eval/ovo/run_base_8gpu.sh

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
`FRAME_PROTOCOL=video_meta` and `RENDER_LAYOUT=standard_query_last`.

Base VLM context modes are intentionally separate:

- `streaming`: recent visual window only, no future frames.
- `offline_prefix`/legacy `offline`: uniform frames from video start to the
  question/probe time, no future frames.
- `offline_full`: uniform frames from the whole video; use as a conventional
  offline content upper bound, not as an online timing metric.

`summary.health` is the compact abnormal-behavior block:

- `answer`: main content accuracy plus `strict_acc` and `targeted_acc`;
  no-early/no-late/on-time accuracy; missing/early/late rates; count MAE for
  REC.
- `recall`: frequency, support hit, before-answer usage, accuracy with/without recall.
- `compression`: frequency, success, and system trigger/range/calculation checks.
- `format_runtime`: parse/action errors, stable-think rate, token maxima, action histogram.

Metric convention:

- `acc`: main score used for OVO-style task/category averages.
- `strict_acc`: task answer is correct and emitted at the exact target probe chunk.
- `targeted_acc`: task-specific scorer with reasonable non-speaking fallbacks
  such as SSR/CRR negative probes allowing silence as a valid No state.
- REC and CRR keep one active query across multiple expected answer chunks via
  `answer_chunks`/`per_emit_answers`; SSR is expanded into one full-prefix
  trajectory per step probe, matching the Streamo benchmark script.

Use `scripts/eval/ovo/compare_runs.py` to compare result JSON files.
