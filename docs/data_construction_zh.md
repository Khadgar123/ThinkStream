# 流视频 Agent 数据构造方案

> 版本: v2.0 | 日期: 2026-04-20

## 1. 目标

为流视频 Agent 构造三动作格式 (`<think>...<action>silent|response|recall</action>`) 的 SFT + RL 训练数据。模型在每个 2s chunk 学会：
- **Silent**: 无需回答，增量观察
- **Response**: 证据充足时回答
- **Recall**: 证据已离开 recent window (24s)，需要检索历史帧

## 2. 基础设施

| 节点 | 硬件 | 用途 |
|------|------|------|
| AMD | MI300X × 8 (每卡 192GB) | 397B teacher 推理 |
| H20 | H20 96GB × 8 | 训练 + 验证 |

### vLLM 启动命令

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --port 8000 \
    --enable-prefix-caching
```

关键参数说明：
- `--max-model-len 65536`：Step 2d 每请求 24 帧 × 1500 token ≈ 37.5K，需要 ≥ 65536
- `--max-num-seqs 64`：GPU 同时处理的最大请求数，客户端并发不应超过此值
- `--enable-prefix-caching`：同视频不同 segment 共享 system prompt KV cache

### 压测结果（纯文本基线）

| 并发 | 吞吐 (req/s) | 延迟 | 状态 |
|------|-------------|------|------|
| 8 | 0.86 | 23.1s | 基线 |
| 16 | 8.52 | 4.7s | 良好 |
| 32 | 10.77 | 3.7s | 峰值 |
| 64 | 5.75 | 7.0s | 下降（显存压力） |

纯文本（16 input tokens）峰值在并发 32。实际带图片请求（6.8K–37.5K tokens/请求）最优并发更低。

### 各 Step 并发配置

| Step | tokens/请求 | 并发数 | max_tokens | temperature |
|------|------------|--------|------------|-------------|
| 2a（逐段标注） | ~6.8K（4 帧） | 32 | 512 | 0.3 |
| 2b（任务设计） | ~8K（纯文本） | 32 | 4096 | 0.7 |
| 2d（think 生成） | ~37.5K（24 帧） | 16 | 128 | 0.5 |

### Token 预算分析

```
Step 2a: 4 帧 × 1500 vision + 300 text = ~6.8K tokens/请求
Step 2b: ~3K 标注文本 + 1K prompt = ~8K tokens（输出 ~4K）
Step 2d: 24 帧 × 1500 vision + 1.5K text = ~37.5K tokens/请求
         → 需要 --max-model-len >= 65536
```

## 3. 任务分类体系

### 3.1 Action 分布（chunk 级目标）

| Action | 占比 | 说明 |
|--------|------|------|
| Silent | 58-65% | 大部分 chunk 静默观察 |
| Response | 23-30% | 有证据时回答 |
| Recall | 10-15% | 证据不在窗口内 |

### 3.2 子任务详细分类

**Silent 子任务**

| ID | 名称 | 说明 | 代码实现 |
|----|------|------|---------|
| S1 | 无问题观察 | 没有用户提问 | 所有 episode 自然包含 |
| S2 | 问题已答完 | 之前已回答 | 自然包含 |
| S3 | Standby 等待 | 问了但事件没发生 | S3_R2 ask→answer 之间的 silent chunks |
| S4 | Trigger 监控 | 条件未满足 | S4_R4 ask→answer 之间的 silent chunks |

**Response 子任务**

| ID | 名称 | 说明 | 代码实现 |
|----|------|------|---------|
| R1 | 即时回答 | 证据在当前帧 | `assemble_episode` 默认路径 |
| R2 | 延迟回答 | Standby 后事件发生 | S3_R2 的 answer chunk |
| R3 | 多次回答 | 答案随时间展开，多次 response | `_assemble_r3` |
| R4 | Trigger 触发 | 监控条件满足 | S4_R4 的 answer chunk |
| R5 | 连续解说 | 实时解说 | 未实现（延后） |
| R6 | Recall 后回答 | 检索后回答 | recall episode 的 response chunk |
| R7 | 假 recall 负样本 | 听起来要回忆但不需要 | `generate_triplets` 的 false_negative |

**Recall 子任务**

| ID | 名称 | 问什么 | query 特征 | 代码实现 |
|----|------|--------|-----------|---------|
| RC1 | 视觉细节回忆 | 颜色/形状/外观 | 实体+属性 | 通用 RC handler |
| RC2 | 精确数值回忆 | 价格/数字/文字 | OCR+数字 | 通用 RC handler |
| RC3 | 程序步骤回忆 | 第几步/顺序 | 动作+步骤 | 通用 RC handler |
| RC4 | 跨时间比较 | 和之前比变了什么 | 实体+状态 | 通用 RC handler |
| RC5 | 长程因果回忆 | 为什么会这样 | 事件+因果 | 通用 RC handler |
| RC6 | 实体再识别 | 是不是之前那个人/物 | 实体+外观 | 通用 RC handler |
| RC7 | 多轮 follow-up | 追问之前对话细节 | 对话+细节 | `_assemble_rc7` |

### 3.3 三合一绑定规则

每条 recall 任务 (RC1-RC6) 生成一个 triplet：
- 1 条 **no-recall control** (R1): 同问题在证据可见时提问
- 1 条 **false-recall negative** (R7): 同问题加 "Earlier/Before, ..." 措辞但证据可见

**例外**：RC7 不参与 triplet binding，因为 control 样本会引用不存在的 base 对话，导致模型学到幻觉。

### 3.4 每视频任务数量

| 任务类型 | 数量 | 条件 |
|---------|------|------|
| R1（即时回答） | 3 | 始终生成 |
| S3_R2（延迟回答） | 2 | 始终生成 |
| RC1（视觉细节） | 1-3 | `num_segments // 10` |
| RC2（数值回忆） | 2 | 有 OCR 的 segment |
| RC3（步骤回忆） | 2 | >10 个 segment |
| RC4（比较回忆） | 1 | >15 个 segment |
| RC5（因果回忆） | 1 | 始终生成 |
| RC6（实体再识别） | 1 | 始终生成 |
| RC7（多轮追问） | 1 | >15 个 segment |
| R3（多次回答） | 1 | >10 个 segment |
| S4_R4（Trigger） | 1 | 始终生成 |

### 3.5 视频类型 → 任务映射

| 视频类型 | 时长要求 | 适合任务 |
|---------|---------|---------|
| 教程/烹饪/装配 | >120s | RC1, RC3, RC5 |
| Vlog/长镜头 | >90s | RC4, RC6, R2, R3 |
| 屏幕录制/UI | >60s | RC2, R1 |
| 剧情/综艺 | >120s | RC5, RC7 |
| 体育/户外 | >90s | R4, RC4 |
| 短视频 (<30s) | 任意 | R1, S1（仅协议训练） |

## 4. 数据构造流程

### 4.1 全流程概览

```
Step 1:  选视频 + 抽帧                          [CPU, ~5min]
Step 2a: 397B 逐段标注                          [AMD vLLM, concurrent=32]
Step 2b: 397B 设计任务（引用 segment ID）        [AMD vLLM, concurrent=32]
Step 2c: 规则校验 + timing 修正                  [CPU, <1s]
Step 2d: 397B think 生成                         [AMD vLLM, concurrent=16]
Step 3:  三合一绑定                              [CPU, <1s]
Step 4:  展开为 chunk-level episode              [CPU, ~1min]
Step 5:  过滤                                    [CPU + 可选小模型]
Step 6:  按配比采样 + 最终组装                    [CPU, <1s]
```

### 4.2 各 Step 详细说明

**Step 1: 选视频 + 抽帧**
- 从 Streamo ≥60s 视频中按丰富度评分选取（时长 × 类型多样性 × 标注密度）
- 2fps 抽帧，短边缩放到 720px（~1500 vision tokens/帧）
- 每 4s 打包为一个 segment，选 4 张关键帧（1fps 覆盖）
- 单个视频抽帧失败会跳过（记日志，不崩溃）
- `duration_sec` 从 ffprobe 实际时长更新（不依赖标注时间戳）
- 输出: `video_registry.jsonl` + `frames/` + `segments/`

**Step 2a: 397B 逐段标注**
- 每个 segment 的 4 张关键帧 → 397B 用英文描述内容
- 每段输出: `action`（一句话）, `entities[]`, `visual_details[{entity, attributes}]`, `ocr`, `change`
- prompt 包含完整 JSON 示例和精确 schema
- 解析: 优先直接 JSON parse → 平衡括号匹配提取 → 回退到原始文本
- 输出: `segment_annotations.jsonl`

**Step 2b: 397B 设计任务**
- 一个视频的全部段标注（文本）→ 397B 设计 10-15 个任务
- prompt 指定每种类型的数量、answer_type 约束、3 个 few-shot 示例（RC1, R3, RC7）
- 引用 segment_id（不猜时间戳）
- 解析: 平衡括号匹配提取 JSON 数组
- 输出: `task_candidates_raw.jsonl`

**Step 2c: 规则校验**
- segment_id → 精确时间戳转换
- 校验顺序: RC7 → R3 → RC*（通用）→ S3_R2 → S4_R4 → 其他（R1）
- RC7: 校验 `base_ask_segment`、`base_question`、`base_answer`，间隔 ≥ 24s
- R3: 校验 `response_segments`（≥2 个，时间递增），补齐 `partial_answers`
- RC*: 校验 `support_segment`，间隔 ≥ 24s（不足时自动前推 ask_time）
- 通用: 拒绝空问题、空答案、ask > 视频时长
- 输出: `task_candidates_verified.jsonl`

**Step 2d: 397B Think 生成**
- 发送 24s recent window 内的 24 帧实际图片（不是文本标注）
- Token 预算: 24 × 1500 + 1500 = ~37.5K → 需要 `--max-model-len 65536`
- recall 任务额外生成 post-recall think（support 帧 + 当前帧）
- 输出: `task_pool.jsonl`（任务带 `think_at_ask` 和 `think_after_recall`）

**Step 3: 三合一绑定**
- 每条 RC1-RC6 recall 任务 → 3 个 episode:
  - `recall_positive`: 原始（需要 recall）
  - `control` (R1): 同问题，在证据可见时提问
  - `false_negative` (R7): 加 "Earlier/Before, ..." 措辞，证据仍可见
- RC7 排除（control 会引用不存在的 base 对话）
- R3 排除（不是 recall 任务）
- 输出: `task_triplets.jsonl`

**Step 4: Chunk-Level Episode 组装**

按任务类型分发到不同组装函数:

| 任务类型 | 组装函数 | Episode 结构 |
|---------|---------|-------------|
| R3 | `_assemble_r3` | 提问 → response₁ → silent → response₂ → ... → responseₙ |
| RC7 | `_assemble_rc7` | base Q&A → (间隔截断为 3 个上下文 chunk) → follow-up recall → response |
| RC1-RC6 | 默认（recall 路径） | silent... → 提问+recall → recall_result → response |
| R1 | 默认（response 路径） | silent... → 提问+response |
| S3_R2 | 默认（延迟路径） | silent... → 提问 → 等待 silent → response |
| S4_R4 | 默认（延迟路径） | silent... → 提问 → 监控 silent → response |

关键设计决策:
- Silent think: 使用 segment `action` 描述（标注文本），最长 50 字符
- S3_R2/S4_R4 等待 silent: "User asked about this, but the event hasn't occurred yet. Waiting."
- RC7 间隔截断: 每个关键事件前后只保留 3 个 silent chunk（避免 30+ 个填充 silent）
- `answer_type` 优先使用 397B 输出，回退到启发式（yes/no → yesno, 数字 → number）
- 未参与 triplet binding 的 recall 任务（RC7）自动标记为 `sample_type=recall_positive`

输出: `sft_episodes_raw.jsonl`

**Step 5: 过滤**
- 硬规则: ≥3 条消息，所有 assistant 消息匹配格式正则，答案非空
- 格式正则: `<think>.*</think><action>(silent|response|recall)</action>(<response>.*</response>|<query>{.*}</query>)?`
- 输出: `sft_episodes_filtered.jsonl`

**Step 6: 采样 + 最终组装**

| 数据集 | 组成 | 用途 |
|--------|------|------|
| `sft_a.jsonl` | simple + 25% recall + 25% control | 协议对齐 warmstart |
| `sft_b.jsonl` | 全部 recall + control + false_neg + 匹配数量的 simple | Recall 重点训练 |
| `rl_pool.jsonl` | 仅可验证答案（yesno, number, entity, slot, multiple_choice） | RL 训练 |

## 5. 持久化资产表

| 文件 | 内容 | 何时重新生成 |
|------|------|------------|
| `video_registry.jsonl` | 视频索引 | 加新视频时追加 |
| `segments/*.json` | 每视频帧路径 | 不重新生成（按视频缓存） |
| `segment_annotations.jsonl` | 逐段 397B 标注 | 不重新生成（最贵） |
| `task_candidates_raw.jsonl` | 397B 任务设计 | 删除后用新 prompt 重新生成 |
| `task_candidates_verified.jsonl` | 校验后的任务 | 自动重新生成 |
| `task_pool.jsonl` | 带 think 的任务 | 删除后重新生成 |
| `task_triplets.jsonl` | 三合一绑定 | 自动派生 |
| `sft_episodes_raw.jsonl` | 组装后的 episode | 改格式时重跑 |
| `sft_a.jsonl` / `sft_b.jsonl` | 训练数据 | 改配比时重跑 |
| `rl_pool.jsonl` | RL 训练池 | 改配比时重跑 |
| `pipeline_stats.json` | 运行统计 | 自动生成 |

## 6. Prompt 设计

所有 prompt 均为英文。关键设计:

- **SEGMENT_ANNOTATE_PROMPT**: 包含完整 JSON 示例和精确 schema，`visual_details` 为 `[{entity, attributes}]` 数组
- **TASK_DESIGN_PROMPT**: 每种任务指定 answer_type 约束，3 个 few-shot 示例（RC1, R3, RC7），query_candidates 规则
- **THINK_PROMPT**: 基于 token 数的长度约束（15-48 tokens），"Output English only"
- **AGENT_SYSTEM_PROMPT**: 三动作协议定义，用作所有 episode 的 system message

## 7. 错误处理

| 场景 | 行为 |
|------|------|
| ffprobe/ffmpeg 对某视频失败 | 跳过该视频，记日志，继续 |
| 397B 返回纯文本（无 JSON） | 用原始文本作为 `action`，其他字段为空 |
| 397B 返回 JSON 后有多余文本 | 平衡括号匹配提取（不用贪婪正则） |
| `query_candidates` 为 null | 回退到用 expected_answer 构造默认 query |
| 任务校验失败 | 拒绝该任务，记录数量 |
| vLLM 请求失败 | 指数退避重试 3 次（1s → 2s → 4s） |
| 所有视频都失败 | `print_statistics` 优雅处理空列表 |

## 8. 训练计划

```
SFT-A（协议对齐）:   datasets=sft_a, lr=1e-5, epochs=3
SFT-B（recall 重点）: datasets=sft_b, lr=5e-6, epochs=3, 基于 SFT-A ckpt
RL-A （action 校准）: datasets=rl_pool, lr=2e-7, 基于 SFT-B ckpt
```

## 9. 质量指标

| 指标 | 通过标准 |
|------|---------|
| 格式合规率 | ≥ 95% |
| OVO-Bench accuracy | 不低于原模型 -2% |
| RTVU accuracy | 不低于原模型 -2% |
| Recall precision | ≥ 70% |
| Recall specificity | ≥ 85% |

## 10. 使用方法

```bash
# 1. 在 AMD 节点启动 vLLM（见第 2 节）

# 2. 压测
python -m scripts.agent_data_pipeline.generate_data stress_test \
    --api_base http://AMD_IP:8000/v1 --max_concurrent 8 --num_requests 20

# 3. 运行完整 pipeline
python -m scripts.agent_data_pipeline.generate_data run \
    --api_base http://AMD_IP:8000/v1 \
    --streamo_dir /path/to/streamo \
    --video_root /path/to/videos \
    --output_dir data/agent \
    --max_concurrent 32 \
    --num_videos 200
```
