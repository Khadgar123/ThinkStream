# 流视频 Agent 数据构造方案

> 版本: v1.0 | 日期: 2026-04-19

## 1. 目标

为流视频 Agent 构造三动作格式 (`<think>...<action>silent|response|recall</action>`) 的 SFT + RL 训练数据。模型需要学会在每个 2s chunk：
- **Silent**: 没有需要回答的事，增量观察
- **Response**: 有足够证据时回答
- **Recall**: 证据已离开 recent window (24s)，需要检索历史帧

## 2. 基础设施

| 节点 | 硬件 | 用途 | vLLM 启动命令 |
|------|------|------|-------------|
| AMD | MI300X × 8 | 397B teacher | `vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --tensor-parallel-size 8 --mm-encoder-tp-mode data --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64 --port 8000` |
| H20 | H20 96GB × 8 | 训练 + 验证 | (可选) 部署 35B 做验证 |

**并发压测**：先用少量请求测 throughput，逐步增加 `--max-num-seqs` 直到显存占用 ~85%。

## 3. 任务分类体系

### 3.1 按 action 分（chunk 级目标分布）

| Action | 占比 | 说明 |
|--------|------|------|
| Silent | 58-65% | 大部分 chunk 不说话 |
| Response | 23-30% | 有证据时回答 |
| Recall | 10-15% | 证据不在窗口内 |

### 3.2 子任务详细分类

**Silent 子任务**

| ID | 名称 | 说明 | 边界区分 |
|----|------|------|---------|
| S1 | 无问题观察 | 没有用户问题 | 所有样本自然包含 |
| S2 | 问题已答完 | 之前已回答 | 自然包含 |
| S3 | Standby 等待 | 问了但事件没发生 | vs R2 (差 1 chunk) |
| S4 | Trigger 监控 | 条件未满足 | vs R4 (条件满足时) |

**Response 子任务**

| ID | 名称 | 说明 | 边界区分 |
|----|------|------|---------|
| R1 | 即时回答 | 证据在当前帧 | vs S3 (证据到没到) |
| R2 | 延迟回答 | Standby 后事件发生 | vs S3 (前一 chunk) |
| R3 | 多次回答 | 持续性问题分次答 | |
| R4 | Trigger 触发 | 监控条件满足 | vs S4 |
| R5 | 连续解说 | 实时解说 | |
| R6 | Recall 后回答 | 检索后回答 | 必须跟在 RC 后 |
| R7 | 假 recall 负样本 | 听起来要回忆但不需要 | vs RC1 (核心区分) |

**Recall 子任务**

| ID | 名称 | 问什么 | query 特征 |
|----|------|--------|-----------|
| RC1 | 视觉细节回忆 | 颜色/形状/外观/穿着 | 实体+属性 |
| RC2 | 精确数值回忆 | 价格/数字/文字 | OCR+数字 |
| RC3 | 程序步骤回忆 | 第几步/顺序 | 动作+步骤 |
| RC4 | 跨时间比较 | 和之前比变了什么 | 实体+状态 |
| RC5 | 长程因果回忆 | 为什么会这样 | 事件+因果 |
| RC6 | 实体再识别 | 是不是之前那个人/物 | 实体+外观 |
| RC7 | 多轮 follow-up | 追问之前对话细节 | 对话+细节 |

### 3.3 三合一绑定规则

每条 recall (RC1-RC7) 必须配套:
- 1 条 **no-recall control** (R1/R2): 同问题在证据可见时提问
- 1 条 **false-recall negative** (R7): 同问题加"之前"措辞但证据可见

### 3.4 任务配比（episode 级）

| 子任务 | 目标占比 | 首批最低数量 |
|--------|---------|-------------|
| R1 即时回答 | 16% | 300 |
| S3+R2 延迟回答 | 12% | 200 |
| RC1 视觉细节 | 10% | 200 |
| RC2-RC7 其他 recall | 15% | 300 |
| R7 假 recall | 与 RC 配对 | 自动 |
| S4+R4 Trigger | 5% | 80 |
| R3 多次回答 | 5% | 80 |
| Protocol (ThinkStream) | 25% | 2000 |

### 3.5 视频类型→任务映射

| 视频类型 | 时长要求 | 适合任务 |
|---------|---------|---------|
| 教程/烹饪/装配 | >120s | RC1, RC3, RC5 |
| Vlog/长镜头 | >90s | RC4, RC6, R2, R3 |
| 屏幕录制/UI | >60s | RC2, R1 |
| 剧情/综艺 | >120s | RC5, RC7 |
| 体育/户外 | >90s | R4, R5, RC4 |
| 短视频 (<30s) | any | R1, S1 (protocol only) |

## 4. 数据构造流程

### 4.1 全流程

```
Step 1: 选视频 + 抽帧                    [CPU, ~5min]
Step 2a: 397B 逐段标注                   [AMD vLLM, ~N min]
Step 2b: 397B 设计任务 (引用 segment ID)  [AMD vLLM, ~N min]
Step 2c: 规则校验修正 timing              [CPU, <1s]
Step 2d: 397B 写 think + 构造答案         [AMD vLLM, ~N min]
Step 3: 三合一绑定派生                    [CPU, <1s]
Step 4: 展开成 chunk-level 数据           [CPU, ~1min]
Step 5: 过滤                             [CPU + 可选小模型]
Step 6: 按配比采样 + 组装                 [CPU, <1s]
```

### 4.2 各 Step 详细说明

**Step 1: 选视频 + 抽帧**
- 从 Streamo >60s 视频中按类型分桶选取
- 2fps 抽帧，每 4s 打包为一个 segment
- 输出: `video_registry.jsonl` + 帧文件

**Step 2a: 397B 逐段标注**
- 每个 segment 的 2 张帧 → 397B 描述内容
- 输出每段: action, entities, visual_details, ocr_text, change_from_prev
- 并发: 全部 segment 一次发出，vLLM continuous batching
- 输出: `segment_annotations.jsonl`

**Step 2b: 397B 设计任务**
- 每个视频的全部段标注（文本）→ 397B 设计 8-15 个任务
- 引用 segment_id 而不是猜时间
- 输出: `task_candidates_raw.jsonl`

**Step 2c: 规则校验修正**
- segment_id → 精确时间转换
- 验证: support_time + 24s < ask_time (recall)
- 修正不合法的 timing
- 输出: `task_candidates_verified.jsonl`

**Step 2d: 397B 写 think + 答案**
- 对 ask chunk: 397B 看 recent window 帧写 think
- 对 recall 后: 397B 写拿到检索结果后的 think
- 构造 canonical_answer (可验证格式)
- 输出: `task_pool.jsonl`

**Step 3: 三合一绑定派生**
- 每条 recall → control + false-negative
- 纯规则，不调模型
- 输出: `task_triplets.jsonl`

**Step 4: 展开 chunk-level**
- 按 2s chunk 展开消息序列
- Silent chunk 用段标注填充 think
- Recall chunk 展开为 3 轮消息
- 输出: `sft_episodes_raw.jsonl`

**Step 5: 过滤**
- 硬规则: timing 合法, 答案非空, 格式正确
- 软规则 (可选): 小模型验证 recall 必要性
- 输出: `sft_episodes_filtered.jsonl`

**Step 6: 采样组装**
- 按目标配比采样
- 混合 ThinkStream 协议数据
- 输出: `sft_a.jsonl`, `sft_b.jsonl`, `rl_pool.jsonl`

## 5. 持久化资产表

| 表 | 内容 | 何时重新生成 |
|----|------|------------|
| `video_registry.jsonl` | 视频索引 | 加新视频时追加 |
| `segment_annotations.jsonl` | 逐段标注 | 不重新生成（最贵） |
| `task_pool.jsonl` | 完整任务 | 不重新生成 |
| `task_triplets.jsonl` | 三合一绑定 | 自动派生 |
| `sft_episodes_raw.jsonl` | 展开数据 | 改格式时重跑 |
| `sft_a/b.jsonl` | 训练数据 | 改配比时重跑 |

## 6. 训练计划

```
SFT-A (协议对齐):   datasets=sft_a, lr=1e-5, epochs=3
SFT-B (recall 重点): datasets=sft_b, lr=5e-6, epochs=3, 基于 SFT-A ckpt
RL-A (action 校准):  datasets=rl_pool, lr=2e-7, 基于 SFT-B ckpt
```

## 7. 质量验证

| 指标 | 通过标准 |
|------|---------|
| 格式合规率 | ≥ 95% |
| OVO-Bench accuracy | 不低于原模型 -2% |
| RTVU accuracy | 不低于原模型 -2% |
| Recall precision | ≥ 70% |
| Recall specificity | ≥ 85% |
