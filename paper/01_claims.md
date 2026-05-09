# Claim-Evidence Board

Use this file as the control surface for the paper. Do not write a strong paper claim unless it has an evidence row here.

| Claim | Evidence needed | Current source | Status |
|---|---|---|---|
| ChronoStream formulates streaming video understanding as agentic temporal reasoning rather than only response timing. | Problem taxonomy, related-work contrast, examples covering past/current/future information. | `paper/drafts/introduction_zh_v1.md`, `references/prior_work/notes/related_work_matrix.md`, `references/prior_work/DOWNLOAD_MANIFEST.md` | Drafted / needs related-work notes |
| ChronoStream separates user-facing interaction decisions from internal temporal memory operations. | Protocol examples, parser/runtime code, figure showing wait/answer vs revisit/memory-edit. | `docs/design.md`, `verl/recipe_thinkstream/streaming_agent_loop.py`, `paper/drafts/introduction_zh_v1.md` | Drafted / needs figure |
| Time-anchored memory revisiting recovers past context under streaming causality. | Recall protocol, MROPE temporal anchoring, qualitative examples and ablation. | `docs/design.md`, `docs/v12.14_integration_status.md`, `verl/recipe_thinkstream/streaming_agent_loop.py` | TODO |
| Snapshot-consistent recurrent RL gives non-terminal memory/revisit actions outcome credit while avoiding dense reward hacking. | Recurrent design, sample_index/final_mask path, reward ablation or stability evidence. | `docs/v12.14_integration_status.md`, `docs/v12.14_recurrent_design.md`, `paper/drafts/introduction_zh_v1.md` | Optional / needs validation |
| ThinkStream reduces the mismatch between streaming inference and training. | Prompt/sample construction proof, code path, ablation or training stability evidence. | `docs/design.md`, `scripts/agent_data_v5/`, `thinkstream/sft/` | TODO |
| Reasoning-compressed memory bounds context growth while preserving useful history. | Token budget, memory state design, long-video efficiency table. | `docs/design.md` | TODO |
| Watch-Think-Speak supports streaming response timing, recall, and silence decisions. | Protocol examples, parser/runtime code, qualitative case. | `docs/design.md`, `scripts/eval/ovo/` | TODO |
| ThinkStream outperforms online/open-source video baselines. | OVO-Bench and StreamingBench main table; include strong OVO offline base with uniform causal-prefix frame sampling and online streaming-window base. | `scripts/eval/ovo/base.py`, `scripts/eval/ovo/eval_full.py`, result logs TBD | Running |
| RL improves streaming-specific behavior beyond SFT. | SFT vs RL ablation, reward decomposition. | `thinkstream/trainer_verl/`, `scripts/test_rl/`, result logs TBD | TODO |
| Recurrent rollout is the path for very long trajectories. | Design/status doc, smoke test, memory comparison. | `docs/v12.14_recurrent_design.md`, `docs/v12.14_integration_status.md` | Optional / appendix |
