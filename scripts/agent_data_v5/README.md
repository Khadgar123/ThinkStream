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

`v2/` is the current pass3 design implementation, not a deprecated folder.
It owns the card taxonomy, placement rules, and LLM prompts used by
pass3a/pass3b/pass3c.

Every new batch should use its own `THINKSTREAM_DATA_ROOT`, which gives one
self-contained directory with frames, pass caches, audits, and final files.
The pipeline writes `batch_manifest.json` and `selected_videos.jsonl` at the
batch root for traceability.
