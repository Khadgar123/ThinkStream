# Batch 2-15 Video Selection And Scale Plan

Generated on 2026-05-04 from:

`/Users/hzh/Downloads/candidate_pool_validated.jsonl`

Single-batch output:

`data/agent_v5/batch2_videos.jsonl`

Single-batch report:

`data/agent_v5/batch2_selection_report.json`

Multi-batch outputs:

- `data/agent_v5/batch2_videos.jsonl`
- `data/agent_v5/batch3_videos.jsonl`
- ...
- `data/agent_v5/batch15_videos.jsonl`
- `data/agent_v5/batch2_to_batch15_videos.jsonl`
- `data/agent_v5/batch_selection_summary.json`

## Streamo Reference

The local Streamo paper (`references/prior_work/papers/Streamo_2512.21334.pdf`)
constructs Streamo-Instruct-465K with multiple streaming task families:

- Real-time narration
- Event caption
- Action caption
- Event grounding
- Time-sensitive QA
- Offline QA

Its listed video sources include Koala, LLaVA-Video, ActivityNet, QVHighlight,
YouCook2, HACS, EgoTimeQA, DiDeMo, and COIN. Streamo-Bench samples 300 videos
from COIN, YouCook2, and ActivityNet and annotates each with grounding,
narration, caption, and time-sensitive QA.

Streamo's duration distribution over all 135,875 videos is:

| Duration | Count |
| --- | ---: |
| 0-30s | 68,273 |
| 30-60s | 19,153 |
| 60-120s | 21,834 |
| 120-240s | 20,529 |
| 240s+ | 6,086 |

Because our candidate pool starts at 30s and ThinkStream uses one video as one
trajectory, batch2 keeps Streamo-like source/task coverage but shifts upward in
duration to provide recall and compression supervision.

## Policy

Target size: 500 videos.

Duration policy:

| Bucket | Target |
| --- | ---: |
| 30-60s | 75 |
| 60-120s | 125 |
| 120-240s | 175 |
| 240-600s | 125 |

Source/function quotas:

| Source group | Kind | Quota |
| --- | --- | ---: |
| how_to_step | procedural_step | 40 |
| how_to_caption | procedural_step | 30 |
| Koala_raw | procedural_longform | 40 |
| Koala | procedural_short_clip | 20 |
| VideoMind/coin | procedural_benchmark | 35 |
| VideoMind/youcook2 | procedural_benchmark | 30 |
| VideoMind/qvhighlights | temporal_grounding | 40 |
| VideoMind/didemo | temporal_grounding | 25 |
| VideoMind/queryd | temporal_grounding | 20 |
| VideoMind/hirest | temporal_grounding | 30 |
| LLaVA-Video/youtube | open_world_qa | 45 |
| LLaVA-Video/academic | offline_video_qa | 25 |
| VideoMind/activitynet | activity_event | 50 |
| VideoMind/charades_sta | activity_action | 20 |
| tarsier2/VATEX | short_dynamic_scene | 35 |
| VideoMind/nextqa | short_dynamic_qa | 15 |

The selector excludes local prior selections and `is_thinkstream` rows, enforces
unique `video_id` and `video_path`, and prefers non-Streamo-overlap rows when a
source has enough alternatives. Some LLaVA/tarsier source copies are avoided
because they duplicate VideoMind `video_id`s, and pipeline caches are keyed by
`video_id`.

## Generated Batches

Actual combined result for batch2-batch15:

| Metric | Value |
| --- | ---: |
| Videos | 7,000 |
| Unique video_id | 7,000 |
| Unique video_path | 7,000 |
| Existing overlap by path | 0 |
| Existing overlap by id | 0 |
| Total duration | 292.908h |
| Min / p50 / p90 / p95 / max | 30.0s / 141.0s / 280.5s / 297.2s / 516.5s |
| Streamo overlap | 1,260 true / 5,740 false |

Actual duration buckets:

| Bucket | Count |
| --- | ---: |
| 30-60s | 1,224 |
| 60-120s | 1,871 |
| 120-240s | 2,322 |
| 240-600s | 1,583 |

Actual source groups:

| Source group | Count |
| --- | ---: |
| VideoMind/activitynet | 700 |
| LLaVA-Video/youtube | 630 |
| how_to_step | 560 |
| Koala_raw | 560 |
| VideoMind/qvhighlights | 560 |
| VideoMind/coin | 490 |
| tarsier2/VATEX | 490 |
| how_to_caption | 420 |
| VideoMind/youcook2 | 420 |
| VideoMind/hirest | 420 |
| VideoMind/didemo | 350 |
| LLaVA-Video/academic | 350 |
| Koala | 280 |
| VideoMind/queryd | 280 |
| VideoMind/charades_sta | 280 |
| VideoMind/nextqa | 210 |

## Prefix Comparison

Use these prefixes depending on generation budget:

| Corpus | Files | Videos | Hours | p50 / p90 / p95 | Max |
| --- | --- | ---: | ---: | --- | ---: |
| 3k | batch2-batch7 | 3,000 | 124.46h | 142.4s / 276.6s / 290.7s | 483.7s |
| 5k | batch2-batch11 | 5,000 | 208.03h | 141.3s / 279.8s / 295.7s | 483.7s |
| 7k | batch2-batch15 | 7,000 | 292.91h | 141.0s / 280.5s / 297.2s | 516.5s |

Duration distribution compared with Streamo videos >=30s:

| Corpus | 30-60s | 60-120s | 120-240s | 240-600s |
| --- | ---: | ---: | ---: | ---: |
| Streamo >=30s | 28.3% | 32.3% | 30.4% | 9.0% |
| 3k | 16.6% | 27.0% | 33.7% | 22.7% |
| 5k | 17.2% | 26.8% | 33.4% | 22.6% |
| 7k | 17.5% | 26.7% | 33.2% | 22.6% |

This is deliberately longer than Streamo: Streamo optimizes broad streaming
instruction coverage, while ThinkStream needs one-video trajectories with
recall and compression. The 240-600s band is therefore about 13.6 points
higher than Streamo.

Video kind distribution is stable across 3k/5k/7k because every 500-video
batch follows the same policy:

| Kind | Share |
| --- | ---: |
| temporal_grounding | 23% |
| procedural_step | 14% |
| procedural_benchmark | 13% |
| activity_event | 10% |
| open_world_qa | 9% |
| procedural_longform | 8% |
| short_dynamic_scene | 7% |
| offline_video_qa | 5% |
| procedural_short_clip | 4% |
| activity_action | 4% |
| short_dynamic_qa | 3% |

Dataset distribution is also stable:

| Dataset | Share |
| --- | ---: |
| VideoMind-Dataset | 53% |
| LLaVA-Video-178K | 14% |
| how_to_step | 8% |
| Koala_raw | 8% |
| tarsier2_unzip | 7% |
| how_to_caption | 6% |
| Koala | 4% |

## Scale Estimate

The current pipeline uses video-level splits:

- 70% train, 15% validation, 15% test.
- Train is split 50/50 into SFT-video and RL-video subsets.
- RL has 1 trajectory per video.
- The adaptive question cap gives roughly 6-14 questions per trajectory.
- SFT timestep rows are estimated from the local v2 simulation:
  34,470 timestep samples / 251 videos = 137.3 rows/video. For the selected
  150s average duration, this corresponds to about 0.9 rows/sec after patrol
  downsampling.

For the 3k/5k/7k prefixes, assuming a conservative 90% usable-video rate after
all passes:

| Estimate | Value |
| --- | ---: |
| 3k selected / usable | 3,000 / 2,700 |
| 3k SFT train videos / RL groups | 944 / 945 |
| 3k SFT train timestep rows | ~127k |
| 3k RL G=8 rollouts / questions | ~7.6k / ~9.7k |
| 3k val+test question instances | ~8.4k |
| 5k selected / usable | 5,000 / 4,500 |
| 5k SFT train videos / RL groups | 1,575 / 1,575 |
| 5k SFT train timestep rows | ~212k |
| 5k RL G=8 rollouts / questions | ~12.6k / ~16.2k |
| 5k val+test question instances | ~13.9k |
| 7k selected / usable | 7,000 / 6,300 |
| 7k SFT train videos / RL groups | 2,205 / 2,205 |
| 7k SFT train timestep rows | ~299k |
| 7k RL G=8 rollouts / questions | ~17.6k / ~22.6k |
| 7k val+test question instances | ~19.4k |

Recommended project scale:

| Scale | Use |
| --- | --- |
| 1,000 clean videos | Minimum debugging scale; enough to see SFT/RL run, but eval slices are thin. |
| 2,000 clean videos | Recommended first serious training scale: about 700 SFT videos, 700 RL videos, and 300+300 val/test videos. |
| 2,500 clean videos | Best current target: enough reserve for benchmark-hard subsets and scale ablation while keeping data generation cost manageable. |
| 3,000 clean videos | Strong first full corpus; already enough for stable SFT/RL and benchmark slices. |
| 5,000 clean videos | Preferred production corpus if generation cost is acceptable. |
| 7,000 clean videos | High-resource corpus and scale-ablation upper point. |

Use batch2-batch7 (3,000 videos) for the first serious full SFT+RL run if cost
is a concern. Use batch2-batch11 (5,000 videos) as the preferred production
corpus. Keep batch2-batch15 (7,000 videos) for scale ablation or if the first
5k run still shows underfitting.

Reproduce with:

```bash
python scripts/select_batch2.py \
  --candidate-pool /Users/hzh/Downloads/candidate_pool_validated.jsonl \
  --num-batches 14 \
  --start-batch 2 \
  --batch-size 500
```
