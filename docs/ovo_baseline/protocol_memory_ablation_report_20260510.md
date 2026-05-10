# OVO-Bench Baseline and ThinkStream Protocol Ablation Report

Date: 2026-05-10

This document summarizes the evidence used to choose the input protocol for subsequent SFT, RL, and OVO evaluation. It focuses on the specific risk raised in debugging: **large video context + large text memory** must still let the model attend to the newest frames, emit useful `<think>`, preserve historical context, and answer at the right time.

## Executive Decision

Recommended default for SFT/RL/eval agent rollouts:

```text
render_layout=standard_query_last
frame_protocol=video_meta
memory_position=before_visual
visual_window_chunks=16
frames_per_chunk=2
min_pixels=130000
max_pixels=220000
recent_thinks_token_budget=3200 or 4000
compress_mode=system
retriever=bm25/hybrid
```

Ordering:

```text
<user_input/event> -> <memory> + <compressed> -> current visual window -> <active_query>
```

Rationale: long text memory should **not** be placed after the latest visual window, because it increases prefill pressure, can dominate the latest visual evidence, and in the short order stress test caused a high format/action error rate. The active query is short and should be last so it re-anchors answer timing and answer format after the large visual block.

## Status of Questions

| Question | Current Answer | Evidence Level |
| --- | --- | --- |
| Can base model run as a training-free think/memory/compress agent? | Not yet a reliable conclusion. Direct-answer OVO baselines are complete, but base-agent memory sweeps launched at 09:53/10:40 did not produce DONE rows. Do not use those empty runs as evidence. | Incomplete |
| Can SFT model run the agent loop with text memory + visual + system compression? | Yes. SFT full-video agent eval runs completed with parseable actions, system compression telemetry, timing metrics, and stable-think diagnostics. | Completed small-sample |
| Does bigvideo alone solve speed/quality? | No. Bigvideo is fast, but stable-think remains high. The problem is bigvideo + long text + query anchoring. | Completed small-sample; long stress running |
| Best memory/visual/query order? | Prefer memory before visual and query last: memory -> video_meta visual window -> active_query. Avoid video -> memory -> query for training. | Completed short stress; long stress running |
| Best visual carrier for SFT/RL? | Use video_meta for SFT/RL throughput and consistency. Direct baseline can benefit from timestamped image lists, but SFT/RL rollout speed favors video_meta. | Completed direct and SFT sweeps |
| Best window/fps? | W=16 chunks, 2 frames/chunk is the default. W=32/64 usually adds cost without consistent gains. 1fps is cheaper but loses some accuracy/timing. | Completed sweeps |
| Best resolution? | 220k max pixels is a good default. 360k/512k did not show reliable gains for the tested settings and costs more. | Completed sweeps |
| How much text memory? | Avoid too-small budgets. 800/1600/2400 caused compression storms or format errors. 3200/4000 are safer. | Completed text budget sweep |

## Direct Baseline: Key Runs

These are direct OVO evaluations, not the ThinkStream agent loop. They measure base model visual-QA ability under offline/online/prefix/oracle variants.

| Run | Mode | Frames | FPS | Samples | Probes | Overall | RT | BT | FT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen25vl3b_streaming32 | streaming | 32 | 1.0 | 1638 | 3033 | 53.28 | 68.56 | 44.47 | 46.80 |
| qwen25vl7b_official_prompt_prefix128 | official_prompt_prefix | 128 | 1.0 | 1640 | 3035 | 49.85 | 61.77 | 39.58 | 48.21 |
| qwen25vl7b_offline_full128 | offline_full | 128 | 1.0 | 1640 | 3035 | 47.44 | 56.91 | 44.28 | 41.14 |
| qwen25vl7b_offline_prefix128 | offline_prefix | 128 | 1.0 | 1640 | 3035 | 48.68 | 60.72 | 39.46 | 45.85 |
| qwen25vl7b_oracle_support64 | oracle_support | 64 | 1.0 | 1638 | 3033 | 55.26 | 68.56 | 44.47 | 52.75 |
| qwen25vl7b_streaming32 | streaming | 32 | 1.0 | 1638 | 3033 | 53.28 | 68.56 | 44.47 | 46.80 |
| qwen3vl2b_streaming32 | streaming | 32 | 1.0 | 1638 | 3033 | 45.20 | 46.00 | 42.12 | 47.49 |
| qwen3vl4b_streaming32 | streaming | 32 | 1.0 | 1638 | 3033 | 43.75 | 54.34 | 37.84 | 39.08 |
| qwen3vl8b_official_prompt_prefix64 | official_prompt_prefix | 64 | 1.0 | 1640 | 3035 | 36.45 | 39.87 | 32.73 | 36.74 |
| qwen3vl8b_offline_full64 | offline_full | 64 | 1.0 | 1640 | 3035 | 37.43 | 41.19 | 33.63 | 37.46 |
| qwen3vl8b_offline_prefix64 | offline_prefix | 64 | 1.0 | 1640 | 3035 | 39.50 | 41.77 | 36.34 | 40.40 |
| qwen3vl8b_oracle_support64 | oracle_support | 64 | 1.0 | 1638 | 3033 | 45.73 | 51.30 | 40.08 | 45.80 |
| qwen3vl8b_streaming32 | streaming | 32 | 1.0 | 1638 | 3033 | 44.25 | 51.30 | 40.08 | 41.37 |

Observations:

- Qwen2.5-VL-7B/3B direct streaming32 is stronger than the first Qwen3-VL-8B direct streaming32 run in the overnight matrix. Later official/preprocess protocol sweeps improved Qwen3-VL substantially, which indicates protocol/preprocessing mattered more than raw model size in the earlier low scores.
- Offline full-video 64/128 is not automatically best. Prefix/offline variants can miss late evidence or dilute the relevant interval; oracle-support is a useful upper bound but is not a valid online setting.
- HLD and REC are consistently hard. HLD needs absence/waiting behavior; REC needs multi-answer counting and timing, not only visual recognition.

## Direct Baseline: Input Protocol and Window Findings

Protocol sweep using corrected/official preprocessing:

| Run | Mode | Frames | FPS | Samples | Probes | Overall | RT | BT | FT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q25_ts_image_p220_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 57.30 | 79.52 | 51.38 | 41.02 |
| q25_video_meta_p360_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 57.47 | 74.58 | 54.64 | 43.18 |
| q3_ts_image_p220_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 59.59 | 81.84 | 51.95 | 44.98 |
| q3_ts_image_p360_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 58.99 | 81.38 | 50.70 | 44.88 |
| q3_ts_image_p512_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 58.99 | 81.38 | 50.70 | 44.88 |
| q3_video_meta_p360_streaming_w16_f32 | streaming | 32 | 1.0 | 1638 | 3033 | 59.01 | 77.60 | 49.78 | 49.66 |
| q3_video_meta_p360_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 57.76 | 79.44 | 48.35 | 45.50 |
| q3_video_meta_p360_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 51.22 | 64.49 | 44.88 | 44.29 |
| q3_video_meta_p512_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 57.84 | 79.44 | 48.58 | 45.50 |
| q3_video_meta_p512_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 53.11 | 67.41 | 47.58 | 44.34 |

Window sweep:

| Run | Mode | Frames | FPS | Samples | Probes | Overall | RT | BT | FT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q25vl3b_offline_prefix128 | offline_prefix | 128 | 1.0 | 1640 | 3035 | 48.68 | 60.72 | 39.46 | 45.85 |
| q25vl3b_offline_prefix32 | offline_prefix | 32 | 1.0 | 1640 | 3035 | 48.42 | 58.48 | 42.16 | 44.63 |
| q25vl3b_streaming_w32_f64 | streaming | 64 | 1.0 | 1638 | 3033 | 54.19 | 67.06 | 45.55 | 49.94 |
| q25vl3b_streaming_w64_f128 | streaming | 128 | 1.0 | 1639 | 3034 | 53.45 | 63.87 | 45.80 | 50.68 |
| q25vl3b_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 52.28 | 69.91 | 43.80 | 43.12 |
| q25vl7b_clean_official_prompt_prefix128 | official_prompt_prefix | 128 | 1.0 | 1640 | 3035 | 48.91 | 61.39 | 44.69 | 40.67 |
| q25vl7b_clean_official_prompt_prefix32 | official_prompt_prefix | 32 | 1.0 | 1640 | 3035 | 47.46 | 61.19 | 42.32 | 38.88 |
| q25vl7b_clean_official_prompt_prefix64 | official_prompt_prefix | 64 | 1.0 | 1640 | 3035 | 48.07 | 61.98 | 41.90 | 40.33 |
| q25vl7b_clean_offline_prefix128 | offline_prefix | 128 | 1.0 | 1640 | 3035 | 53.59 | 65.34 | 54.55 | 40.89 |
| q25vl7b_clean_offline_prefix32 | offline_prefix | 32 | 1.0 | 1640 | 3035 | 50.91 | 62.51 | 51.17 | 39.06 |
| q25vl7b_clean_offline_prefix64 | offline_prefix | 64 | 1.0 | 1640 | 3035 | 51.80 | 64.40 | 51.33 | 39.66 |
| q25vl7b_clean_streaming_w16_f32 | streaming | 32 | 1.0 | 1638 | 3033 | 58.25 | 73.18 | 57.45 | 44.12 |
| q25vl7b_clean_streaming_w32_f64 | streaming | 64 | 1.0 | 1638 | 3033 | 57.04 | 70.14 | 56.50 | 44.47 |
| q25vl7b_clean_streaming_w64_f128 | streaming | 128 | 1.0 | 1639 | 3034 | 57.40 | 70.10 | 56.80 | 45.31 |
| q25vl7b_clean_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 57.63 | 74.39 | 55.50 | 43.01 |
| q3vl8b_official_prompt_prefix256 | official_prompt_prefix | 256 | 1.0 | 1640 | 3035 | 35.31 | 38.20 | 31.67 | 36.05 |
| q3vl8b_official_prompt_prefix32 | official_prompt_prefix | 32 | 1.0 | 1640 | 3035 | 38.32 | 42.15 | 34.89 | 37.93 |
| q3vl8b_offline_prefix256 | offline_prefix | 256 | 1.0 | 1640 | 3035 | 39.03 | 40.52 | 37.40 | 39.17 |
| q3vl8b_offline_prefix32 | offline_prefix | 32 | 1.0 | 1640 | 3035 | 40.50 | 41.81 | 38.37 | 41.32 |
| q3vl8b_streaming_w32_f64 | streaming | 64 | 1.0 | 1638 | 3033 | 41.38 | 44.64 | 38.52 | 40.98 |
| q3vl8b_streaming_w64_f128 | streaming | 128 | 1.0 | 1639 | 3034 | 39.04 | 41.55 | 36.83 | 38.74 |
| q3vl8b_streaming_w8_f16 | streaming | 16 | 1.0 | 1638 | 3033 | 48.10 | 60.31 | 42.31 | 41.69 |

Protocol conclusions:

- Direct baseline can prefer `ts_image` with explicit frame tags: `q3_ts_image_p220_streaming_w8_f16` reached 59.59 overall, higher than `q3_video_meta_p360_streaming_w8_f16` at 57.76 in the same official sweep.
- For SFT/RL agent rollout, `video_meta` is still preferred because it gives much higher rollout throughput with mm processor cache and avoids enormous image-list prompt fragmentation.
- Higher resolution is not monotonic. `p512` did not materially improve over `p360`, and `p220` timestamped images were competitive or better.
- Increasing window from W=8 to W=16 can help direct baseline slightly in some runs, but W=32/W=64 is not reliably better and is much more expensive.

## SFT Agent Protocol Sweeps

Small-sample SFT agent runs use full streaming loop: model emits `<think>` plus action/tool calls, memory updates online, recall is controller-executed, and compression is system-triggered.

| Run/Job | Overall | Content | Strict | NoEarly | NoLate | OnTime | Recall/Probe | Compress/Probe | CompressOK | StableThinkSample | FormatErr | Steps/s | PromptTokMax | Steps | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ovo_sft_protocol_design_20260510_1058/sft_full_bigvideo_w16 | 65.64 | 46.67 | 46.67 | 43.33 | 46.67 | 43.33 | 0 | 0.533 | 100.00 | 80.00 | 1.94 | 3.135 | 5325 | 1188 | 379.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkvideo_w16 | 54.53 | 43.33 | 43.33 | 40.00 | 43.33 | 40.00 | 0 | 0.967 | 58.62 | 76.67 | 1.01 | 1.936 | 5778 | 1189 | 614.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w8 | 60.09 | 46.67 | 43.33 | 43.33 | 43.33 | 40.00 | 0 | 1.600 | 47.92 | 76.67 | 2.18 | 1.773 | 5889 | 1195 | 674.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_1fps_w16 | 57.86 | 43.33 | 40.00 | 43.33 | 40.00 | 40.00 | 0 | 0.700 | 100.00 | 76.67 | 0.00 | 1.749 | 5809 | 1193 | 682.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_aftermem_w16 | 68.97 | 46.67 | 40.00 | 46.67 | 40.00 | 40.00 | 0 | 0.367 | 100.00 | 80.00 | 12.09 | 1.133 | 5886 | 1183 | 1044.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w16 | 60.09 | 46.67 | 43.33 | 43.33 | 43.33 | 40.00 | 0 | 0.833 | 80.00 | 76.67 | 0.50 | 1.130 | 5833 | 1192 | 1055.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_bigimg_w16 | 68.97 | 46.67 | 46.67 | 46.67 | 46.67 | 46.67 | 0 | 0.533 | 100.00 | 76.67 | 0.00 | 1.123 | 5820 | 1188 | 1058.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w32 | 60.09 | 46.67 | 43.33 | 43.33 | 43.33 | 40.00 | 0 | 0.967 | 68.97 | 63.33 | 0.84 | 0.725 | 6104 | 1192 | 1644.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_p360_w16 | 54.53 | 43.33 | 40.00 | 40.00 | 40.00 | 36.67 | 0 | 0.667 | 100.00 | 53.33 | 0.00 | 1.190 | 6017 | 1192 | 1002.0 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_p180_w16 | 60.09 | 46.67 | 43.33 | 43.33 | 43.33 | 40.00 | 0 | 0.600 | 100.00 | 70.00 | 0.08 | 1.127 | 5673 | 1190 | 1056.0 |
| ovo_sft_bigvideo_order_stress_20260510_1140/sft_full_bigvideo_querylast_w16 | 48.97 | 40.00 | 40.00 | 40.00 | 40.00 | 40.00 | 0 | 0.600 | 100.00 | 80.00 | 3.19 | 2.990 | 5376 | 1190 | 398.0 |
| ovo_sft_bigvideo_order_stress_20260510_1140/sft_full_bigvideo_visualmem_w16 | 60.09 | 43.33 | 40.00 | 43.33 | 40.00 | 40.00 | 0 | 0.333 | 100.00 | 76.67 | 15.14 | 2.282 | 5452 | 1182 | 518.0 |
| ovo_sft_protocol_confirm_20260510_1125/sft_full_bigvideo_w16 | 38.57 | 30.23 | 29.07 | 29.07 | 30.23 | 29.07 | 0 | 0.733 | 100.00 | 80.23 | 0.70 | 3.913 | 5561 | 3991 | 1020.0 |
| ovo_sft_text_budget_bigvideo_r800_20260510_1112/sft_full_bigvideo_w16 | 48.89 | 23.33 | 20.00 | 23.33 | 20.00 | 20.00 | 0 | 16.900 | 10.65 | 66.67 | 40.13 | 4.046 | 2567 | 1226 | 303.0 |
| ovo_sft_text_budget_bigvideo_r1600_20260510_1111/sft_full_bigvideo_w16 | 63.42 | 43.33 | 36.67 | 43.33 | 36.67 | 36.67 | 0 | 6.967 | 19.62 | 80.00 | 15.17 | 3.360 | 3172 | 1213 | 361.0 |
| ovo_sft_text_budget_bigvideo_r2400_20260510_1111/sft_full_bigvideo_w16 | 51.45 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 0 | 3.333 | 34.00 | 80.00 | 6.30 | 3.233 | 4020 | 1206 | 373.0 |
| ovo_sft_text_budget_bigvideo_r3200_20260510_1112/sft_full_bigvideo_w16 | 63.42 | 43.33 | 43.33 | 43.33 | 43.33 | 43.33 | 0 | 0.767 | 100.00 | 80.00 | 2.34 | 3.204 | 4494 | 1195 | 373.0 |

Important readings:

- `sft_full_bigvideo_w16` is fast: 3.135 steps/s in the first protocol design run and 3.913 steps/s in the N=3 confirm run. It is the best speed baseline for RL rollout.
- `sft_full_bigimg_w16` can match or exceed answer accuracy on tiny samples but is about 3x slower than bigvideo in the first sweep.
- `sft_full_chunkimg_w32` is slower and did not improve accuracy; W=32 is not a good default for training.
- `sft_full_chunkimg_1fps_w16` is faster than chunkimg W16 but slightly weaker in strict/on-time metrics; use it only if throughput is the bottleneck and accuracy loss is acceptable.
- `video -> memory -> query` (`sft_full_bigvideo_visualmem_w16`) looked numerically okay on a tiny short set but had 15.14% format/action error and lower throughput, so it is not a safe training protocol.

## Text Memory Budget Sweep

| Budget Setting | Overall | Content | Strict | Compress/Probe | CompressOK | StableThinkSample | FormatErr | Steps/s | PromptTokMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recent=800 | 48.89 | 23.33 | 20.00 | 16.900 | 10.65 | 66.67 | 40.13 | 4.046 | 2567 |
| recent=1600 | 63.42 | 43.33 | 36.67 | 6.967 | 19.62 | 80.00 | 15.17 | 3.360 | 3172 |
| recent=2400 | 51.45 | 33.33 | 33.33 | 3.333 | 34.00 | 80.00 | 6.30 | 3.233 | 4020 |
| recent=3200 | 63.42 | 43.33 | 43.33 | 0.767 | 100.00 | 80.00 | 2.34 | 3.204 | 4494 |

Conclusion: memory budget must not be too small. At 800/1600 the system over-compresses, many compress attempts fail, and format errors become large. 3200 is the lowest currently safe budget; 4000 remains the conservative default.

## Memory Ordering Stress

Short order stress results:

| Order | Run | Overall | Content | Strict | OnTime | Compress/Probe | StableThinkSample | StableThinkPair | FormatErr | Steps/s | PromptTokMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory -> query -> video | protocol_confirm/sft_full_bigvideo_w16 | 38.57 | 30.23 | 29.07 | 29.07 | 0.733 | 80.23 | 68.45 | 0.70 | 3.913 | 5561 |
| memory -> video -> query | order_stress/sft_full_bigvideo_querylast_w16 | 48.97 | 40.00 | 40.00 | 40.00 | 0.600 | 80.00 | 61.18 | 3.19 | 2.990 | 5376 |
| video -> memory -> query | order_stress/sft_full_bigvideo_visualmem_w16 | 60.09 | 43.33 | 40.00 | 40.00 | 0.333 | 76.67 | 68.95 | 15.14 | 2.282 | 5452 |

Long-trajectory stress currently running:

| Log | Done Jobs | Runner Chunks | Rate Chunks/s |
| --- | --- | --- | --- |
| sft_full_bigvideo_querylast_w16_gpu1.log | 0/2 | 1805 | 2.0 |
| sft_full_bigvideo_visualmem_w16_gpu4.log | 0/2 | 916 | 1.0 |
| sft_full_bigvideo_w16_gpu0.log | 0/2 | 1537 | 1.7 |

Interpretation so far:

- `video -> memory -> query` is disfavored despite a high tiny-set score because long text after visual increases risk that the model answers/formats from text state instead of current visual evidence; it is also the slowest in the long stress run so far.
- `memory -> query -> video` makes the current visual last, which helps latest-frame grounding, but the query is no longer the final anchor; answer timing/format can degrade.
- `memory -> video -> query` is the best compromise: memory is available, current visual is near the end, and the active query/answer format is the final instruction.

## Per-Task Baseline Tables

### Overnight Baseline Matrix

| Run | Overall | OCR | ACR | ATR | STU | FPD | OJR | EPM | ASI | HLD | REC | SSR | CRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen25vl3b_official_prompt_prefix64 | 49.20 | 65.77 | 51.38 | 75.00 | 47.19 | 66.34 | 57.61 | 47.81 | 56.76 | 15.05 | 24.36 | 60.89 | 56.25 |
| qwen25vl3b_offline_full64 | 46.73 | 59.06 | 44.95 | 68.10 | 41.01 | 67.33 | 51.09 | 48.48 | 64.86 | 18.28 | 20.06 | 60.57 | 42.50 |
| qwen25vl3b_offline_prefix64 | 48.18 | 67.11 | 49.54 | 72.41 | 42.13 | 72.28 | 58.70 | 45.79 | 53.38 | 18.28 | 27.65 | 62.00 | 45.42 |
| qwen25vl3b_oracle_support64 | 55.26 | 83.78 | 66.06 | 71.55 | 49.44 | 69.31 | 71.20 | 51.18 | 53.74 | 28.49 | 31.52 | 76.31 | 50.42 |
| qwen25vl3b_streaming32 | 53.28 | 83.78 | 66.06 | 71.55 | 49.44 | 69.31 | 71.20 | 51.18 | 53.74 | 28.49 | 21.63 | 70.43 | 48.33 |
| qwen25vl7b_official_offline64_smoke_n5 | 43.75 | 60.00 | 0.00 | 40.00 | 0.00 | 80.00 | 20.00 | 40.00 | 60.00 | 0.00 | 78.95 | 62.82 | 52.00 |
| qwen25vl7b_official_prompt_prefix128 | 49.85 | 70.47 | 55.05 | 74.14 | 44.94 | 67.33 | 58.70 | 48.15 | 53.38 | 17.20 | 27.22 | 60.73 | 56.67 |
| qwen25vl7b_official_prompt_prefix64 | 49.20 | 65.77 | 51.38 | 75.00 | 47.19 | 66.34 | 57.61 | 47.81 | 56.76 | 15.05 | 24.36 | 60.89 | 56.25 |
| qwen25vl7b_offline_full128 | 47.44 | 65.77 | 42.20 | 68.10 | 43.26 | 68.32 | 53.80 | 50.51 | 66.22 | 16.13 | 18.77 | 60.89 | 43.75 |
| qwen25vl7b_offline_full64 | 46.73 | 59.06 | 44.95 | 68.10 | 41.01 | 67.33 | 51.09 | 48.48 | 64.86 | 18.28 | 20.06 | 60.57 | 42.50 |
| qwen25vl7b_offline_prefix128 | 48.68 | 70.47 | 52.29 | 71.55 | 41.57 | 70.30 | 58.15 | 47.81 | 53.38 | 17.20 | 27.36 | 63.12 | 47.08 |
| qwen25vl7b_offline_prefix64 | 48.18 | 67.11 | 49.54 | 72.41 | 42.13 | 72.28 | 58.70 | 45.79 | 53.38 | 18.28 | 27.65 | 62.00 | 45.42 |
| qwen25vl7b_oracle_support64 | 55.26 | 83.78 | 66.06 | 71.55 | 49.44 | 69.31 | 71.20 | 51.18 | 53.74 | 28.49 | 31.52 | 76.31 | 50.42 |
| qwen25vl7b_streaming32 | 53.28 | 83.78 | 66.06 | 71.55 | 49.44 | 69.31 | 71.20 | 51.18 | 53.74 | 28.49 | 21.63 | 70.43 | 48.33 |
| qwen3vl2b_official_prompt_prefix64 | 38.19 | 30.87 | 24.77 | 46.55 | 35.39 | 47.52 | 30.98 | 36.36 | 43.24 | 13.98 | 22.35 | 67.25 | 52.50 |
| qwen3vl2b_offline_full64 | 39.27 | 36.24 | 22.94 | 41.38 | 35.39 | 50.50 | 31.52 | 32.32 | 43.92 | 39.25 | 15.47 | 65.18 | 48.33 |
| qwen3vl2b_offline_prefix64 | 40.60 | 36.91 | 30.28 | 44.83 | 30.90 | 50.50 | 31.52 | 33.67 | 45.95 | 38.71 | 19.48 | 65.98 | 49.17 |
| qwen3vl2b_oracle_support64 | 45.42 | 43.92 | 42.20 | 52.59 | 36.52 | 53.47 | 47.28 | 37.04 | 51.70 | 37.63 | 22.78 | 67.89 | 53.75 |
| qwen3vl2b_streaming32 | 45.20 | 43.92 | 42.20 | 52.59 | 36.52 | 53.47 | 47.28 | 37.04 | 51.70 | 37.63 | 20.77 | 68.36 | 53.33 |
| qwen3vl4b_official_prompt_prefix64 | 36.51 | 33.56 | 35.78 | 48.28 | 38.20 | 48.51 | 36.41 | 35.02 | 47.30 | 19.35 | 24.93 | 41.65 | 40.00 |
| qwen3vl4b_offline_full64 | 34.16 | 32.21 | 27.52 | 45.69 | 41.01 | 60.40 | 39.13 | 35.69 | 45.27 | 8.06 | 9.31 | 45.31 | 40.83 |
| qwen3vl4b_offline_prefix64 | 36.17 | 34.90 | 33.94 | 49.14 | 38.76 | 49.50 | 38.04 | 39.73 | 52.03 | 13.44 | 10.89 | 43.08 | 44.17 |
| qwen3vl4b_oracle_support64 | 44.20 | 45.27 | 56.88 | 61.21 | 47.75 | 64.36 | 50.54 | 41.08 | 55.78 | 16.67 | 13.47 | 60.73 | 47.08 |
| qwen3vl4b_streaming32 | 43.75 | 45.27 | 56.88 | 61.21 | 47.75 | 64.36 | 50.54 | 41.08 | 55.78 | 16.67 | 11.60 | 58.98 | 46.67 |
| qwen3vl8b_official_offline64_smoke_n5 | 40.82 | 40.00 | 40.00 | 20.00 | 40.00 | 80.00 | 0.00 | 20.00 | 80.00 | 20.00 | 47.37 | 50.00 | 40.00 |
| qwen3vl8b_official_prompt_prefix128 | 35.90 | 33.56 | 33.94 | 45.69 | 41.01 | 44.55 | 33.70 | 37.04 | 48.65 | 13.44 | 26.22 | 41.49 | 40.00 |
| qwen3vl8b_official_prompt_prefix64 | 36.45 | 34.23 | 36.70 | 47.41 | 39.89 | 44.55 | 36.41 | 37.04 | 49.32 | 11.83 | 26.50 | 43.72 | 40.00 |
| qwen3vl8b_offline_full128 | 37.01 | 30.20 | 33.94 | 44.83 | 39.89 | 50.50 | 36.96 | 40.07 | 52.03 | 16.67 | 14.76 | 47.22 | 44.17 |
| qwen3vl8b_offline_full64 | 37.43 | 30.20 | 33.03 | 48.28 | 39.89 | 54.46 | 41.30 | 38.38 | 50.68 | 11.83 | 17.19 | 49.76 | 45.42 |
| qwen3vl8b_offline_prefix128 | 39.43 | 31.54 | 38.53 | 50.00 | 38.20 | 53.47 | 34.78 | 39.06 | 54.05 | 21.51 | 22.21 | 48.97 | 45.83 |
| qwen3vl8b_offline_prefix64 | 39.50 | 35.57 | 35.78 | 50.86 | 38.76 | 50.50 | 39.13 | 37.37 | 53.38 | 18.28 | 23.35 | 51.19 | 46.67 |
| qwen3vl8b_oracle_support64 | 45.73 | 39.86 | 49.54 | 58.62 | 50.00 | 55.45 | 54.35 | 44.78 | 59.86 | 15.59 | 26.22 | 66.61 | 44.58 |
| qwen3vl8b_streaming32 | 44.25 | 39.86 | 49.54 | 58.62 | 50.00 | 55.45 | 54.35 | 44.78 | 59.86 | 15.59 | 21.20 | 57.07 | 45.83 |

### Window Sweep

| Run | Overall | OCR | ACR | ATR | STU | FPD | OJR | EPM | ASI | HLD | REC | SSR | CRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q25vl3b_offline_prefix128 | 48.68 | 70.47 | 52.29 | 71.55 | 41.57 | 70.30 | 58.15 | 47.81 | 53.38 | 17.20 | 27.36 | 63.12 | 47.08 |
| q25vl3b_offline_prefix32 | 48.42 | 69.13 | 48.62 | 67.24 | 42.70 | 68.32 | 54.89 | 47.81 | 56.08 | 22.58 | 23.93 | 61.21 | 48.75 |
| q25vl3b_streaming_w32_f64 | 54.19 | 85.14 | 60.55 | 67.24 | 46.63 | 73.27 | 69.57 | 52.19 | 59.18 | 25.27 | 24.07 | 68.68 | 57.08 |
| q25vl3b_streaming_w64_f128 | 53.45 | 79.87 | 53.21 | 67.24 | 46.07 | 73.27 | 63.59 | 52.53 | 61.22 | 23.66 | 24.64 | 67.41 | 60.00 |
| q25vl3b_streaming_w8_f16 | 52.28 | 85.14 | 69.72 | 71.55 | 51.12 | 71.29 | 70.65 | 52.53 | 53.06 | 25.81 | 17.48 | 69.79 | 42.08 |
| q25vl7b_clean_official_prompt_prefix128 | 48.91 | 72.48 | 55.05 | 66.38 | 50.56 | 67.33 | 56.52 | 48.48 | 66.22 | 19.35 | 27.79 | 50.87 | 43.33 |
| q25vl7b_clean_official_prompt_prefix32 | 47.46 | 66.44 | 55.05 | 69.83 | 52.25 | 70.30 | 53.26 | 44.78 | 62.84 | 19.35 | 28.37 | 47.85 | 40.42 |
| q25vl7b_clean_official_prompt_prefix64 | 48.07 | 71.81 | 55.05 | 68.97 | 51.12 | 67.33 | 57.61 | 46.46 | 61.49 | 17.74 | 28.08 | 50.40 | 42.50 |
| q25vl7b_clean_offline_prefix128 | 53.59 | 74.50 | 55.96 | 72.41 | 53.93 | 73.27 | 61.96 | 52.53 | 67.57 | 43.55 | 32.95 | 47.22 | 42.50 |
| q25vl7b_clean_offline_prefix32 | 50.91 | 68.46 | 54.13 | 73.28 | 53.37 | 69.31 | 56.52 | 44.44 | 60.14 | 48.92 | 29.80 | 47.38 | 40.00 |
| q25vl7b_clean_offline_prefix64 | 51.80 | 75.84 | 54.13 | 74.14 | 51.69 | 70.30 | 60.33 | 48.15 | 62.84 | 43.01 | 31.66 | 46.90 | 40.42 |
| q25vl7b_clean_streaming_w16_f32 | 58.25 | 86.49 | 72.48 | 74.14 | 63.48 | 71.29 | 71.20 | 51.85 | 61.90 | 58.60 | 24.36 | 63.43 | 44.58 |
| q25vl7b_clean_streaming_w32_f64 | 57.04 | 88.51 | 69.72 | 72.41 | 56.18 | 69.31 | 64.67 | 53.54 | 63.27 | 52.69 | 27.36 | 55.64 | 50.42 |
| q25vl7b_clean_streaming_w64_f128 | 57.40 | 87.25 | 64.22 | 75.00 | 55.62 | 73.27 | 65.22 | 52.53 | 67.35 | 50.54 | 29.94 | 50.56 | 55.42 |
| q25vl7b_clean_streaming_w8_f16 | 57.63 | 86.49 | 73.39 | 76.72 | 61.80 | 71.29 | 76.63 | 48.82 | 61.22 | 56.45 | 19.63 | 67.73 | 41.67 |
| q3vl8b_official_prompt_prefix256 | 35.31 | 35.57 | 33.03 | 44.83 | 37.64 | 45.54 | 32.61 | 37.37 | 47.97 | 9.68 | 26.50 | 41.65 | 40.00 |
| q3vl8b_official_prompt_prefix32 | 38.32 | 34.90 | 37.61 | 48.28 | 39.89 | 51.49 | 40.76 | 40.40 | 51.35 | 12.90 | 27.79 | 45.15 | 40.83 |
| q3vl8b_offline_prefix256 | 39.03 | 33.56 | 34.86 | 49.14 | 38.76 | 51.49 | 35.33 | 39.73 | 54.73 | 17.74 | 22.49 | 49.60 | 45.42 |
| q3vl8b_offline_prefix32 | 40.50 | 33.56 | 33.94 | 52.59 | 38.20 | 53.47 | 39.13 | 40.74 | 58.78 | 15.59 | 22.64 | 53.42 | 47.92 |
| q3vl8b_streaming_w32_f64 | 41.38 | 35.81 | 39.45 | 54.31 | 39.89 | 55.45 | 42.93 | 40.40 | 58.50 | 16.67 | 22.78 | 55.17 | 45.00 |
| q3vl8b_streaming_w64_f128 | 39.04 | 34.90 | 33.03 | 51.72 | 41.01 | 49.50 | 39.13 | 38.72 | 55.10 | 16.67 | 22.78 | 51.35 | 42.08 |
| q3vl8b_streaming_w8_f16 | 48.10 | 54.05 | 62.39 | 67.24 | 54.49 | 63.37 | 60.33 | 44.78 | 58.50 | 23.66 | 19.34 | 62.80 | 42.92 |

### Input Protocol Sweep

| Run | Overall | OCR | ACR | ATR | STU | FPD | OJR | EPM | ASI | HLD | REC | SSR | CRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q25_ts_image_p220_streaming_w8_f16 | 57.30 | 93.92 | 79.82 | 78.45 | 65.73 | 78.22 | 80.98 | 55.56 | 59.86 | 38.71 | 17.34 | 66.14 | 39.58 |
| q25_video_meta_p360_streaming_w8_f16 | 57.47 | 90.54 | 75.23 | 76.72 | 63.48 | 70.30 | 71.20 | 48.15 | 59.86 | 55.91 | 20.77 | 67.09 | 41.67 |
| q3_ts_image_p220_streaming_w8_f16 | 59.59 | 92.57 | 86.24 | 81.03 | 75.84 | 73.27 | 82.07 | 54.55 | 62.59 | 38.71 | 22.92 | 68.68 | 43.33 |
| q3_ts_image_p360_streaming_w8_f16 | 58.99 | 91.22 | 85.32 | 81.90 | 74.16 | 75.25 | 80.43 | 54.21 | 59.18 | 38.71 | 22.78 | 68.52 | 43.33 |
| q3_ts_image_p512_streaming_w8_f16 | 58.99 | 91.22 | 85.32 | 81.90 | 74.16 | 75.25 | 80.43 | 54.21 | 59.18 | 38.71 | 22.78 | 68.52 | 43.33 |
| q3_video_meta_p360_streaming_w16_f32 | 59.01 | 89.86 | 82.57 | 77.59 | 66.85 | 74.26 | 74.46 | 56.90 | 63.95 | 28.49 | 27.51 | 67.73 | 53.75 |
| q3_video_meta_p360_streaming_w8_f16 | 57.76 | 90.54 | 83.49 | 80.17 | 66.29 | 76.24 | 79.89 | 53.87 | 60.54 | 30.65 | 22.21 | 68.04 | 46.25 |
| q3_video_meta_p360_streaming_w8_f16 | 51.22 | 55.41 | 64.22 | 75.00 | 60.11 | 65.35 | 66.85 | 46.13 | 60.54 | 27.96 | 20.20 | 63.91 | 48.75 |
| q3_video_meta_p512_streaming_w8_f16 | 57.84 | 90.54 | 83.49 | 80.17 | 66.29 | 76.24 | 79.89 | 53.87 | 61.22 | 30.65 | 22.21 | 68.04 | 46.25 |
| q3_video_meta_p512_streaming_w8_f16 | 53.11 | 65.54 | 69.72 | 72.41 | 58.99 | 69.31 | 68.48 | 48.82 | 63.27 | 30.65 | 22.35 | 63.59 | 47.08 |

## Per-Task SFT Agent Protocol Table

| Job | Overall | Content | OCR | ACR | ATR | STU | FPD | OJR | EPM | ASI | HLD | REC | SSR | CRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ovo_sft_bigvideo_order_stress_20260510_1140/sft_full_bigvideo_querylast_w16 | 48.97 | 40.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_bigvideo_order_stress_20260510_1140/sft_full_bigvideo_visualmem_w16 | 60.09 | 43.33 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_confirm_20260510_1125/sft_full_bigvideo_w16 | 38.57 | 30.23 | 33.33 | 0.00 | 33.33 | 0.00 | 33.33 | 33.33 | 66.67 | 33.33 | 100.00 | 6.25 | 34.21 | 40.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_aftermem_w16 | 68.97 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 40.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_bigimg_w16 | 68.97 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 40.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_bigvideo_w16 | 65.64 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_1fps_w16 | 57.86 | 43.33 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 40.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_p180_w16 | 60.09 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_p360_w16 | 54.53 | 43.33 | 100.00 | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w16 | 60.09 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w32 | 60.09 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkimg_w8 | 60.09 | 46.67 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_protocol_design_20260510_1058/sft_full_chunkvideo_w16 | 54.53 | 43.33 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 30.77 | 60.00 |
| ovo_sft_text_budget_bigvideo_r1600_20260510_1111/sft_full_bigvideo_w16 | 63.42 | 43.33 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 40.00 |
| ovo_sft_text_budget_bigvideo_r2400_20260510_1111/sft_full_bigvideo_w16 | 51.45 | 33.33 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 23.08 | 40.00 |
| ovo_sft_text_budget_bigvideo_r3200_20260510_1112/sft_full_bigvideo_w16 | 63.42 | 43.33 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 30.77 | 40.00 |
| ovo_sft_text_budget_bigvideo_r800_20260510_1112/sft_full_bigvideo_w16 | 48.89 | 23.33 | 0.00 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 40.00 |

## Implications for SFT + RL + Eval

1. Use one canonical prompt/rendering path across SFT, RL, and eval. The current target is `standard_query_last + video_meta + memory before visual`.
2. Keep the answer format and options inside `<active_query>` and place `<active_query>` last for ordinary visual turns. This reduces the chance that long visual/text context overrides answer-format instructions.
3. Keep system compression as a controller event, not a model-initiated ordinary action. Metrics should check system trigger/range/calc correctness, not gold teacher compress alignment.
4. Monitor both answer quality and behavior: content/strict/targeted/no-early/no-late/on-time, recall frequency/support hit, compression success, format/action errors, prompt token max, stable-think sample rate, stable-think pair rate, and max stable run.
5. RL rollout speed should be optimized around B*S multi-trajectory vLLM rollout, `video_meta` caching, and keeping W/memory budget bounded. Larger W/fps/resolution should be justified by measured per-task gains, not assumed.
6. The SFT model still has a stable-think problem. The prompt says current chunk first, but metrics show high stable-think rates. SFT/RL should include explicit diagnostics and possibly shaping/DAgger examples for current-frame updating and answer timing.

## Remaining Gaps

- Base-model training-free agent memory/compress sweep did not complete successfully in the earlier runs; only direct-answer base OVO results are reliable right now.
- Long-trajectory SFT order stress is still running; update this report after `output/ovo_sft_long_memory_order_stress_20260510_1150/matrix_summary.tsv` receives DONE rows.
- Recall usefulness is not fully tested by the short SFT protocol sweeps because recall events were zero in those small samples. Need targeted historical-query eval where recall should fire.
- Need full-video SFT eval under the final selected protocol, not only N_PER_TASK=1/3 protocol sweeps.

## Source Outputs

- `output/ovo_baseline_matrix_overnight_20260509`
- `output/ovo_window_sweep_20260510_0340`
- `output/ovo_input_protocol_sweep_official_20260510_0807`
- `output/ovo_input_protocol_sweep_20260510_0750`
- `output/ovo_sft_protocol_design_20260510_1058`
- `output/ovo_sft_bigvideo_order_stress_20260510_1140`
- `output/ovo_sft_protocol_confirm_20260510_1125`
- `output/ovo_sft_text_budget_bigvideo_r800_20260510_1112`
- `output/ovo_sft_text_budget_bigvideo_r1600_20260510_1111`
- `output/ovo_sft_text_budget_bigvideo_r2400_20260510_1111`
- `output/ovo_sft_text_budget_bigvideo_r3200_20260510_1112`
- `output/ovo_sft_long_memory_order_stress_20260510_1150`
