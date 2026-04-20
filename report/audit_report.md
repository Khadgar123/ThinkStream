# 数据审计报告

> 生成时间: 2026-04-19 17:22
> 总行数: 705,460
> 总唯一 origin: 285,764

---

## 12 个关键审计问题

- Q1. streamo 行数: 465832
- Q1. thinkstream 行数: 239628
- Q2. streamo 唯一 clip: 223174
- Q2. thinkstream 唯一 clip: 73433
- Q3. streamo 唯一原视频: 222525
- Q3. thinkstream 唯一原视频: 73368
- Q4. 跨数据集同源 origin 数: 10129 / 285764 (3.5%)
- Q5. streamo 底层来源 top3: unknown: 253925, llava_video: 144687, activitynet: 52428
- Q5. thinkstream 底层来源 top3: llava_video: 130112, kinetics700: 49436, activitynet: 32360
- Q6. streamo 每 origin 平均样本: 2.1, 中位数: 1, max: 305
- Q6. thinkstream 每 origin 平均样本: 3.3, 中位数: 2, max: 2378
- Q7. streamo 任务分布 top5: QA: 58.9%, Event_Grounding: 21.4%, Narration: 7.2%, Event_Caption: 6.7%, Action_Caption: 5.8%
- Q7. thinkstream 任务分布 top5: Real-time Dialogue: 50.1%, Event Trigger: 29.9%, Continuous Output: 20.1%
- Q8. streamo temporal_scope: : 100.0%
- Q8. thinkstream temporal_scope: Current: 59.0%, Past: 21.6%, Future: 19.4%
- Q9. streamo response_format: : 100.0%
- Q9. thinkstream response_format: Open-ended: 65.1%, Multiple Choice: 25.0%, Binary: 8.2%, Counting: 1.7%
- Q10. streamo 有时间戳: 100.0%, 有 support span: 100.0%
- Q10. thinkstream 有时间戳: 100.0%, 有 support span: 21.6%
- Q11. streamo 适合 recall: 49.7%
- Q11. thinkstream 适合 recall: 16.7%
- Q12. streamo 仅弱锚点: 0.0%
- Q12. thinkstream 仅弱锚点: 0.0%

## 表 1: 数据源总览

| dataset     |   num_rows |   num_unique_clips |   num_unique_origin_videos |   avg_samples_per_origin |   missing_video_path_pct |   missing_user_timestamp_pct |
|:------------|-----------:|-------------------:|---------------------------:|-------------------------:|-------------------------:|-----------------------------:|
| streamo     |     465832 |             223174 |                     222525 |                     2.09 |                        0 |                            0 |
| thinkstream |     239628 |              73433 |                      73368 |                     3.27 |                        0 |                            0 |
| TOTAL       |     705460 |             286509 |                     285764 |                     2.47 |                        0 |                            0 |

## 表 2: 来源分布

| source_dataset   | underlying_video_source   |   num_rows |   num_unique_origin_videos |   overlap_rows_with_other_dataset |
|:-----------------|:--------------------------|-----------:|---------------------------:|----------------------------------:|
| streamo          | unknown                   |     253925 |                     159831 |                                25 |
| streamo          | llava_video               |     144687 |                      43682 |                              1993 |
| streamo          | activitynet               |      52428 |                      14890 |                             32285 |
| streamo          | charades                  |      14791 |                       4445 |                              4046 |
| streamo          | kinetics700               |          1 |                          1 |                                 0 |
| thinkstream      | llava_video               |     130112 |                      40798 |                             18046 |
| thinkstream      | kinetics700               |      49436 |                      18193 |                                64 |
| thinkstream      | activitynet               |      32360 |                       9077 |                             31770 |
| thinkstream      | charades                  |      27720 |                       5300 |                              2858 |

## 表 3: 时间切片分布 (多 clip origin, top 50)

| origin_video_id   | source_datasets     |   num_clips |   clip_duration_mean |   clip_duration_std |
|:------------------|:--------------------|------------:|---------------------:|--------------------:|
| split_2           | streamo,thinkstream |        2660 |                 12.3 |                10.3 |
| split_0           | streamo,thinkstream |        2650 |                 11.9 |                 8.2 |
| split_1           | streamo,thinkstream |        2535 |                 12   |                 7.8 |
| split_3           | streamo,thinkstream |        2468 |                 12.1 |                 8   |
| split_4           | streamo,thinkstream |        2402 |                 12.1 |                 7.6 |
| split_5           | streamo,thinkstream |        1774 |                 12   |                 7.4 |
| split_6           | streamo,thinkstream |        1606 |                 12   |                 7.4 |
| split_7           | streamo,thinkstream |        1252 |                 11.9 |                 7.1 |
| split_8           | streamo,thinkstream |         950 |                 11.3 |                 6.7 |
| split_9           | streamo,thinkstream |         705 |                 11.2 |                 6.5 |
| split_10          | streamo,thinkstream |         435 |                 11.7 |                 8   |
| split_11          | streamo,thinkstream |         318 |                 10.1 |                 7   |
| s21-d42           | streamo             |         183 |                 96.2 |                54.3 |
| s21-d23           | streamo             |         168 |                 64.5 |                33.7 |
| s23-d21           | streamo             |         167 |                 84.9 |                47.2 |
| s23-d39           | streamo             |         152 |                 57.1 |                48   |
| s15-d70           | streamo             |         151 |                123.3 |                96.5 |
| s26-d70           | streamo             |         146 |                 79.7 |                61.7 |
| s23-d51           | streamo             |         145 |                113.1 |                79.7 |
| s21-d43           | streamo             |         143 |                 86.2 |                39.7 |

## 表 4: 任务标签分布

| source_dataset   | task_type          | temporal_scope   | response_format   | content_dimension                 |   count |   percentage |
|:-----------------|:-------------------|:-----------------|:------------------|:----------------------------------|--------:|-------------:|
| streamo          | QA                 |                  |                   |                                   |  274243 |        58.87 |
| streamo          | Event_Grounding    |                  |                   |                                   |   99727 |        21.41 |
| streamo          | Narration          |                  |                   |                                   |   33772 |         7.25 |
| streamo          | Event_Caption      |                  |                   |                                   |   31159 |         6.69 |
| streamo          | Action_Caption     |                  |                   |                                   |   26931 |         5.78 |
| thinkstream      | Continuous Output  | Current          | Open-ended        | Procedural State & Evolution      |   44528 |        18.58 |
| thinkstream      | Event Trigger      | Current          | Open-ended        | Global Scene & Context            |   44146 |        18.42 |
| thinkstream      | Event Trigger      | Future           | Open-ended        | Entity & Attribute Perception     |   22086 |         9.22 |
| thinkstream      | Real-time Dialogue | Current          | Multiple Choice   | Spatial & Geometric Relationships |   18230 |         7.61 |
| thinkstream      | Real-time Dialogue | Current          | Open-ended        | Causal & Logical Reasoning        |   18150 |         7.57 |
| thinkstream      | Real-time Dialogue | Past             | Open-ended        | Causal & Logical Reasoning        |   18148 |         7.57 |
| thinkstream      | Real-time Dialogue | Past             | Multiple Choice   | Optical Character Recognition     |   18064 |         7.54 |
| thinkstream      | Real-time Dialogue | Future           | Multiple Choice   | Procedural State & Evolution      |    9074 |         3.79 |
| thinkstream      | Real-time Dialogue | Future           | Open-ended        | Entity & Attribute Perception     |    9058 |         3.78 |
| thinkstream      | Real-time Dialogue | Current          | Binary            | Causal & Logical Reasoning        |    6872 |         2.87 |
| thinkstream      | Real-time Dialogue | Past             | Binary            | Procedural State & Evolution      |    6866 |         2.87 |
| thinkstream      | Real-time Dialogue | Future           | Binary            | Spatial & Geometric Relationships |    3444 |         1.44 |
| thinkstream      | Real-time Dialogue | Past             | Counting          | Entity & Attribute Perception     |    1266 |         0.53 |
| thinkstream      | Real-time Dialogue | Current          | Counting          | Action & Activity Semantics       |    1264 |         0.53 |
| thinkstream      | Real-time Dialogue | Future           | Counting          | Entity & Attribute Perception     |     580 |         0.24 |

## 表 5: 时序监督质量

| source_dataset   |   total_rows |   has_user_timestamp_pct |   has_assistant_timestamp_pct |   has_think_timestamp_pct |   has_support_span_pct |
|:-----------------|-------------:|-------------------------:|------------------------------:|--------------------------:|-----------------------:|
| streamo          |       465832 |                      100 |                           100 |                       0   |                  100   |
| thinkstream      |       239628 |                      100 |                           100 |                      92.5 |                   21.6 |

## 表 6: 可用性三分

| source_dataset   |   total_rows |   usable_for_protocol_sft_pct |   usable_for_recall_pct |   usable_for_timing_pct |   usable_for_multiturn_pct |   usable_for_rl_verifiable_pct |   only_as_weak_anchor_pct |   level_A_pct |   level_B_pct |   level_C_pct |
|:-----------------|-------------:|------------------------------:|------------------------:|------------------------:|---------------------------:|-------------------------------:|--------------------------:|--------------:|--------------:|--------------:|
| streamo          |       465832 |                           100 |                    49.7 |                     100 |                        9.3 |                            0   |                         0 |         100   |           0   |             0 |
| thinkstream      |       239628 |                           100 |                    16.7 |                     100 |                        0   |                           34.9 |                         0 |          95.6 |           4.4 |             0 |

## 表 7: 去重风险

| metric                           |   value |   percentage |
|:---------------------------------|--------:|-------------:|
| exact_duplicate_rows             |   31240 |         4.43 |
| origins_with_multiple_samples    |  146040 |        51.11 |
| cross_dataset_same_origin_videos |   10129 |         3.54 |
| cross_dataset_same_origin_rows   |   91087 |        12.91 |
| semantic_near_duplicate_rows     |       0 |         0    |
| visual_near_duplicate_rows       |       0 |         0    |
| over_sampling_cap_rows           |       0 |         0    |

## 表 8: 分层抽样质检

| bucket                |   total_available |   sampled_count | sample_asset_ids                                                                                                                                                                                                                          |
|:----------------------|------------------:|----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| high_recallability    |            271674 |              50 | ["thinkstream_0032553", "streamo_0011320", "streamo_0419583", "streamo_0319895", "streamo_0215579", "streamo_0190969", "streamo_0015321", "streamo_0448598", "streamo_0013846", "thinkstream_0215114"]...                                 |
| low_recallability     |            433786 |              50 | ["thinkstream_0219262", "streamo_0212222", "thinkstream_0088252", "streamo_0039875", "streamo_0233295", "streamo_0098769", "thinkstream_0140700", "thinkstream_0151071", "streamo_0295573", "streamo_0289678"]...                         |
| kinetics700           |             49437 |              50 | ["thinkstream_0141138", "thinkstream_0000584", "thinkstream_0060316", "thinkstream_0037239", "thinkstream_0145995", "thinkstream_0218783", "thinkstream_0010626", "thinkstream_0210414", "thinkstream_0102738", "thinkstream_0234270"]... |
| activitynet           |             84788 |              50 | ["streamo_0060508", "streamo_0282076", "streamo_0264139", "thinkstream_0188243", "thinkstream_0072727", "streamo_0287701", "thinkstream_0132854", "streamo_0193597", "thinkstream_0221913", "streamo_0329889"]...                         |
| multiturn             |             43112 |              50 | ["streamo_0325732", "streamo_0097973", "streamo_0409863", "streamo_0381365", "streamo_0389268", "streamo_0389282", "streamo_0412253", "streamo_0103993", "streamo_0359047", "streamo_0375884"]...                                         |
| cross_dataset_overlap |             91087 |              50 | ["thinkstream_0151550", "streamo_0299983", "streamo_0388499", "streamo_0166245", "thinkstream_0078275", "streamo_0335097", "thinkstream_0094883", "streamo_0059004", "streamo_0325975", "thinkstream_0224198"]...                         |
