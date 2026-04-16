# 流视频 Agent 数据构造详细实施方案

> **版本**: v0.1  
> **日期**: 2026-04-15  
> **依赖文档**: `docs/agent数据构造文档.txt`（原始粗设计）  
> **核心原则**: 小规模优先、质量优先、增量扩展、首批端到端完整覆盖

---

## 目录

0. [数据审计与去重设计（前置步骤）](#0-数据审计与去重设计前置步骤)
1. [原始方案矛盾分析与解决方案](#1-原始方案矛盾分析与解决方案)
2. [硬件规格与吞吐量基线](#2-硬件规格与吞吐量基线)
3. [核心协议定义](#3-核心协议定义)
4. [六张数据表完整 Schema](#4-六张数据表完整-schema)
5. [数据构造流程 Stage 0–6](#5-数据构造流程-stage-06)
6. [首批小规模构造计划](#6-首批小规模构造计划)
7. [训练计划](#7-训练计划)
8. [质量保障体系](#8-质量保障体系)
9. [代码改造清单](#9-代码改造清单)
10. [附录 C: 源数据集阅读说明](#附录-c-源数据集阅读说明)

---

## 0. 数据审计与去重设计（前置步骤）

> **这一步必须在任何数据构造之前完成。** 没有审计结果，后续所有配比设计、SFT、RL 和评测都建立在不稳的地基上。

### 0.1 为什么必须先做审计

两个源数据集存在 **4 个核心风险**：

| # | 风险 | 具体表现 |
|---|------|---------|
| 1 | 同源视频跨数据集重叠 | 两个数据集都使用了 LLaVA-Video 作为视频来源；ThinkStream 额外使用 Tarsier2（含 Kinetics-700/Charades/ActivityNet），Streamo 也使用了 ActivityNet/Kinetics |
| 2 | 同一原视频被重复标注 | 同一 YouTube 视频可能在两个数据集中被切成不同 clip 并分别标注不同任务 |
| 3 | 原始分布未知 | 不清楚哪些视频适合 recall、哪些只适合短时问答 |
| 4 | 训练/验证泄漏 | 如果 train 和 val 来自同一原视频的不同时间段，长期记忆任务会被污染 |

### 0.2 源数据集概况

#### Streamo-Instruct-465K

| 维度 | 值 | 来源 |
|------|-----|------|
| 论文 | "Streaming Video Instruction Tuning", arXiv:2512.21334, CVPR 2026 |
| 总样本数 | **465,800** | 论文 Figure 3 |
| 唯一视频数 | **135,875** | 论文 Figure 3 |
| 平均每视频样本数 | ~3.4 | 465800/135875 |
| 视频来源 | LLaVA-Video, ActivityNet, Kinetics, ShareGPT4Video | 论文正文 |
| 标注模型 | Qwen2.5-VL-72B (narration), GLM-4.5V (TSQA), ARC-Hunyuan-Video-7B (event caption) | 论文 Section 4.1 |
| HuggingFace | `maifoundations/Streamo-Instruct-465K` (gated access, 621MB) |
| 训练框架 | 基于 ms-swift，1fps，multi-turn 对话 |

**任务分布**:

| 任务类型 | 占比 | 样本数 (估) | 适合 Agent 化程度 |
|---------|------|-----------|-----------------|
| Time-sensitive QA | 34.8% | ~162K | 高：有时间锚点，可派生 timing/recall |
| Event Grounding | 26.3% | ~122K | 高：有 event + temporal span，可派生 recall |
| Offline QA | 13.8% | ~64K | 低：无时间信息，只能做弱锚点 |
| Narration | 12.7% | ~59K | 中：连续描述，可做 continuous output |
| Event Caption | 6.7% | ~31K | 中：有事件边界，可做 timing |
| Action Caption | 5.8% | ~27K | 中：有动作步骤，可做 procedure |

**视频时长分布**:

| 时长范围 | 视频数 | 占比 | 适合 recall |
|---------|--------|------|------------|
| 0–30s | 68,273 | 50.25% | 否（太短，support 无法离开 recent window） |
| 30–60s | 19,153 | 14.1% | 边界（需要 recent_window < 24s） |
| 60–120s | 21,834 | 16.07% | 是 |
| 120–240s | 20,529 | 15.11% | 是（理想长度） |
| 240s+ | 6,086 | 4.48% | 是（最适合长程 recall） |

**原始数据格式**（`raw_data.json`）:

```json
{
  "video_name": "ytb_ls8tmkfD7S0.mp4",
  "video_path": "LLaVA_Video/1_2_m_youtube_v0_1/.../ytb_ls8tmkfD7S0.mp4",
  "task_type": "QA",
  "source": "LLaVA_Video",
  "question": [{"content": "What is happening?", "time": "5"}],
  "response": [{"content": "A person walks in.", "st_time": 5.0, "end_time": 6.0, "time": ""}]
}
```

**关键字段解读**:
- `question.time`: 问题出现的秒数（如 "5" 表示第 5 秒，对应 `<4s-5s>` 帧）
- `response.st_time`: 事件开始时间（Standby 开始）
- `response.end_time`: 事件结束时间（Response 触发点）
- `response.time`: 即时回答时间（无 Standby 的场景）
- `source`: 原始视频来源数据集（如 `LLaVA_Video`）

**训练格式**（stream_format.json）:
- 1fps 多轮对话，每帧一个 turn
- 特殊 token: `</Silence>` / `</Standby>` / `</Response>`
- `<stream>` token 占位符，训练时替换为 `<image>`
- Focal loss + frequency-based alpha 解决 token 不平衡

#### ThinkStream

| 维度 | 值 | 来源 |
|------|-----|------|
| 论文 | "Thinking in Streaming Video", arXiv:2603.12938 |
| Cold Start 样本 | **110,000** | 论文 Section 5 |
| RLVR 样本 | **9,000** | 论文 Section 5 |
| 总计 | **~119,000** |
| 视频来源 | LLaVA-Video-178K + Tarsier2-Recap-585K (含 Kinetics-700, Charades, ActivityNet) | 论文 Appendix B |
| 标注模型 | Qwen3-VL-235B-A22B-Instruct (dense caption) + synthesis pipeline | 论文 Section 5 |
| HuggingFace | `CASIA-IVA-Lab/ThinkStream` (MIT license, 299MB) |
| 训练框架 | 自研 (slyme + deepslyme)，2fps，FlexAttention |

**三维度标签体系**（Cartesian product → 39 有效组合）:

| 维度 | 值 | 说明 |
|------|-----|------|
| **interaction_mode** | Real-time Dialogue | 实时问答，问题和回答同一时间戳 |
| | Event Trigger | 提前设定监控规则，事件发生才触发 |
| | Continuous Output | 连续输出，动态描述变化 |
| **temporal_scope** | Past | 回忆性问题，依赖长期记忆 |
| | Current | 当前实时感知 |
| | Future | 预测性问题 |
| **content_dimension** | Entity & Attribute Perception | 实体识别和属性 |
| | Action & Activity Semantics | 动作和活动 |
| | Spatial & Geometric Relationships | 空间关系 |
| | Causal & Logical Reasoning | 因果推理 |
| | Procedural State & Evolution | 程序状态追踪 |
| | Global Scene & Context | 全局场景 |
| | Optical Character Recognition | OCR |

**response_format**: Open-ended, Multiple Choice, Binary (Yes/No), Counting

**数据格式**:

```json
{
  "video_path": "./datasets/tarsier2_unzip/Kinetics-700/videos/FSPKtVyji6s_000074_000084.mp4",
  "interaction_mode": "Real-time Dialogue",
  "content_dimension": "Action & Activity Semantics",
  "temporal_scope": "Current",
  "response_format": "Open-ended",
  "conversations": [
    {"role": "user", "content": "What is happening?", "timestamp": 5.0, "options": null},
    {"role": "assistant", "content": "A person is cooking.", "timestamp": 5.0}
  ],
  "thoughts": [
    {"think": "The person is standing at the stove...", "timestamp": 3.0},
    {"think": "Now they are adding ingredients...", "timestamp": 5.0}
  ]
}
```

**关键字段解读**:
- `video_path`: 包含原始数据集路径信息（如 `Kinetics-700`），**可解析出 origin_video_id**
- `conversations.timestamp`: 对话发生的精确时间（秒），**这是 agent 化的关键锚点**
- `thoughts.timestamp`: 思维过程的时间戳，**直接对应 chunk-level think**
- `thoughts.think`: 增量思维文本，**可直接复用为 agent think 内容**
- `options`: Multiple Choice 题目的选项列表

**Think token 长度分布**:

| 范围 | 占比 | 含义 |
|------|------|------|
| 0–100 | 1.0% | 极短 think |
| 100–200 | 30.1% | 短 think（1-2 个 chunk） |
| 200–300 | 42.0% | 中等（典型长度） |
| 300–500 | 20.4% | 较长（多 chunk 累积） |
| 500+ | 6.4% | 很长（复杂推理） |

**视频路径格式规律**（从 HF 预览解析）:
- 格式: `{youtube_id}_{start_frame:06d}_{end_frame:06d}.mp4`
- 示例: `FSPKtVyji6s_000074_000084.mp4` → origin_id=`FSPKtVyji6s`, clip=74-84s
- 来源路径包含数据集标识: `Kinetics-700/`, `Charades/`, `ActivityNet/`

### 0.3 重叠风险分析

#### 已知重叠来源

| 底层数据集 | Streamo 是否使用 | ThinkStream 是否使用 | 重叠风险 |
|-----------|----------------|-------------------|---------|
| LLaVA-Video | ✅ 直接使用 | ✅ 通过 LLaVA-Video-178K | **高** |
| Kinetics-700 | ✅ 通过 LLaVA-Video | ✅ 通过 Tarsier2 | **高** |
| ActivityNet | ✅ 直接使用 | ✅ 通过 Tarsier2 | **高** |
| Charades | ❓ 未明确 | ✅ 通过 Tarsier2 | 中 |
| ShareGPT4Video | ✅ 直接使用 | ❓ 未明确 | 低 |
| YouTube raw | ✅ 通过 LLaVA-Video | ✅ 通过 LLaVA-Video | **高** |

**初步判断**: 两个数据集的底层视频来源有 **大量重叠**（LLaVA-Video + Kinetics + ActivityNet 是共同来源）。必须做 origin_video_id 级别的去重。

### 0.4 统一资产表 video_asset_registry

在做任何数据构造之前，先将两个数据集统一到一张资产表：

```python
# video_asset_registry schema
{
    "asset_id": "str",                    # 全局唯一ID
    "source_dataset": "str",              # "streamo" | "thinkstream"
    "source_row_id": "int",               # 原始行号
    "video_locator_raw": "str",           # 原始视频路径（不清洗）
    "video_origin_id": "str",             # 归一化后的同源视频ID（核心键）
    "underlying_source": "str",           # "kinetics700" | "activitynet" | "llava_video_youtube" | "charades" | "unknown"
    "clip_start_sec": "float",            # clip 起始秒
    "clip_end_sec": "float",              # clip 结束秒
    "duration_sec": "float",              # clip 时长
    "video_available": "bool",            # 视频文件是否可访问
    # --- 任务信息 ---
    "task_type_raw": "str",               # Streamo: task_type; ThinkStream: interaction_mode
    "temporal_scope": "str",              # ThinkStream: temporal_scope; Streamo: 从 task_type 推断
    "content_dimension": "str",           # ThinkStream 原生; Streamo: 从 task_type 推断
    "response_format": "str",             # ThinkStream 原生; Streamo: 从数据推断
    "has_user_timestamp": "bool",         # 是否有用户提问时间
    "has_assistant_timestamp": "bool",    # 是否有回答时间
    "has_think_timestamp": "bool",        # 是否有 think 时间戳
    "has_support_span": "bool",           # 是否有 support 时间区间
    "num_conversations": "int",           # 对话轮数
    "num_thoughts": "int",               # think 条数
    # --- 可用性标签 ---
    "usable_for_protocol_sft": "bool",    # 可直接用于 agent 协议 SFT
    "usable_for_recall": "bool",          # 可派生 recall 样本
    "usable_for_timing": "bool",          # 可做 timing 校准
    "usable_for_multiturn": "bool",       # 可做多轮对话
    "usable_for_rl_verifiable": "bool",   # 可做 RL（可验证答案）
    "only_as_weak_anchor": "bool",        # 只能做弱锚点
    "usable_level": "str",               # "A" | "B" | "C" (见 0.7 节)
    # --- 去重信息 ---
    "dedup_group_id": "str",              # 同源视频分组ID
    "dedup_status": "str",                # "unique" | "same_origin_cross_dataset" | "same_clip" | "near_duplicate"
    "max_samples_from_origin": "int",     # 该 origin 的最大采样上限
}
```

### 0.5 四层同源识别

去重分 4 层执行，从最便宜到最贵：

#### 层 1: 字符串级同源（成本: ~0, 命中率: 高）

从 video_path 解析 origin_video_id：

```python
import re

def parse_origin_id(video_path: str) -> dict:
    """从视频路径解析同源信息"""
    result = {"origin_id": None, "clip_start": None, "clip_end": None, "source": "unknown"}

    # 模式 1: YouTube ID 格式 (ThinkStream Kinetics-700)
    # e.g., FSPKtVyji6s_000074_000084.mp4
    m = re.search(r'([A-Za-z0-9_-]{11})_(\d{6})_(\d{6})\.mp4', video_path)
    if m:
        result["origin_id"] = m.group(1)
        result["clip_start"] = int(m.group(2))
        result["clip_end"] = int(m.group(3))
        if "Kinetics" in video_path:
            result["source"] = "kinetics700"
        elif "Charades" in video_path:
            result["source"] = "charades"
        elif "ActivityNet" in video_path:
            result["source"] = "activitynet"
        return result

    # 模式 2: Streamo YouTube 格式
    # e.g., ytb_ls8tmkfD7S0.mp4
    m = re.search(r'ytb_([A-Za-z0-9_-]+)\.mp4', video_path)
    if m:
        result["origin_id"] = m.group(1)
        result["source"] = "llava_video_youtube"
        return result

    # 模式 3: LLaVA-Video 路径
    if "LLaVA_Video" in video_path or "llava" in video_path.lower():
        filename = video_path.split("/")[-1].replace(".mp4", "")
        result["origin_id"] = filename
        result["source"] = "llava_video"
        return result

    # 模式 4: ActivityNet 格式
    # e.g., v_xxxxx.mp4
    m = re.search(r'(v_[A-Za-z0-9_-]+)\.mp4', video_path)
    if m:
        result["origin_id"] = m.group(1)
        result["source"] = "activitynet"
        return result

    # 兜底: 用文件名作为 origin_id
    result["origin_id"] = video_path.split("/")[-1].replace(".mp4", "")
    return result
```

#### 层 2: 元数据级同源（成本: 低, 需要视频文件）

对层 1 未能匹配的样本，比较 `duration + resolution + file_size`：

```python
def metadata_match(asset_a, asset_b, duration_tol=0.5, size_tol=0.05):
    """元数据匹配：时长和文件大小接近"""
    duration_close = abs(asset_a["duration_sec"] - asset_b["duration_sec"]) < duration_tol
    size_close = abs(asset_a["file_size"] - asset_b["file_size"]) / max(asset_a["file_size"], 1) < size_tol
    return duration_close and size_close
```

#### 层 3: 视觉近重（成本: 中, 需要 GPU）

每个 clip 取 3 张关键帧 → SigLIP embedding → cosine similarity：

```python
def visual_near_duplicate(emb_a, emb_b, threshold=0.95):
    """视觉近重：embedding cosine similarity > 0.95"""
    sim = cosine_similarity(emb_a, emb_b)
    return sim > threshold
```

标签区分:
- `same_origin_possible`: 可能来自同一长视频的不同时段
- `near_duplicate_clip`: 几乎相同的片段（重编码/改名）

#### 层 4: 语义近重（成本: 低, 文本比较）

检查标注文本的重复度：

```python
def semantic_near_duplicate(text_a, text_b, threshold=0.85):
    """标注文本语义近重"""
    emb_a = text_encoder.encode(text_a)
    emb_b = text_encoder.encode(text_b)
    return cosine_similarity(emb_a, emb_b) > threshold
```

### 0.6 8 张审计报表

第一轮审计必须产出以下 8 张表，**没有这 8 张表不进入数据构造**。

#### 表 1: 数据源总览表

```
| dataset_name | num_rows | num_unique_clips | num_unique_origin_videos | avg_clips_per_origin | missing_video_path% | missing_timestamp% |
|-------------|----------|------------------|------------------------|---------------------|--------------------|--------------------|
| streamo     | 465800   | ?                | ?                      | ?                   | ?                  | ?                  |
| thinkstream | 119000   | ?                | ?                      | ?                   | ?                  | ?                  |
```

#### 表 2: 来源分布表

```
| source_dataset | underlying_video_source | num_rows | num_unique_origin_videos | overlap_with_other_dataset |
|---------------|------------------------|----------|------------------------|---------------------------|
| streamo       | llava_video_youtube    | ?        | ?                      | ?                         |
| streamo       | activitynet            | ?        | ?                      | ?                         |
| streamo       | kinetics               | ?        | ?                      | ?                         |
| thinkstream   | kinetics700            | ?        | ?                      | ?                         |
| thinkstream   | charades               | ?        | ?                      | ?                         |
| thinkstream   | activitynet            | ?        | ?                      | ?                         |
```

#### 表 3: 时间切片分布表

```
| origin_video_id | source_dataset | num_clips | clip_duration_mean | clip_duration_std | coverage_ratio | overlap_between_clips |
|-----------------|---------------|-----------|-------------------|------------------|---------------|----------------------|
```

#### 表 4: 任务标签分布表

```
| source_dataset | task_category | temporal_scope | response_format | count | percentage |
|---------------|--------------|---------------|----------------|-------|-----------|
```

#### 表 5: 时序监督质量表

```
| source_dataset | has_user_timestamp% | has_assistant_timestamp% | has_think_timestamp% | has_support_span% | temporal_consistency_ok% |
|---------------|--------------------|-----------------------|--------------------|-----------------|-----------------------|
```

#### 表 6: 可用性三分表

```
| source_dataset | usable_for_protocol_sft% | usable_for_recall% | usable_for_timing% | usable_for_rl_verifiable% | only_as_weak_anchor% | drop% |
|---------------|-------------------------|--------------------|-------------------|--------------------------|---------------------|-------|
```

#### 表 7: 去重风险表

```
| metric                              | value | percentage |
|-------------------------------------|-------|-----------|
| exact_duplicate_ratio               | ?     | ?         |
| same_origin_video_ratio (within)    | ?     | ?         |
| same_origin_cross_dataset_ratio     | ?     | ?         |  ← 最重要
| near_duplicate_clip_ratio           | ?     | ?         |
```

#### 表 8: 分层抽样质检表（人工）

每个桶随机抽 50 条人工检查:
- 高 recallability 视频
- 低 recallability 视频
- Kinetics-700 来源
- ActivityNet 来源
- 多轮对话样本
- suspected duplicate 样本

### 0.7 可用性三级分类

#### A 类: 直接可迁移

满足全部条件:
- 视频可访问
- 时间信息完整（has_user_timestamp + has_assistant_timestamp）
- 任务类型清楚
- 无严重同源泄漏
- 标注与视频内容一致

**用途**: 直接进入 agent SFT / retriever / reranker / RL 候选池

#### B 类: 只能当弱锚点

典型情况:
- 有 query + 大致 support 时间
- 但 response 文字质量一般
- 或时间边界不适合直接做实时输出
- 或 Offline QA 无时间信息但有事件锚点

**用途**: 
- `earliest_answerable_search` 的锚点
- recall support 锚点
- timing control 样本构造的参考

#### C 类: 丢弃

- 视频打不开
- OCR/ASR 严重错误
- 标注与画面不符
- 重复太多（同 origin 超过采样上限）
- 只有极弱的开放式描述且无时间信息

### 0.8 去重规则

#### 规则 1: train/val/test 按 origin_video_id 分割

```python
# 绝对不允许同一 origin_video_id 同时出现在 train 和 val/test 中
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, test_idx = next(splitter.split(X, groups=origin_video_ids))
```

#### 规则 2: 每个 origin_video_id 最大采样上限

```python
MAX_EPISODES_PER_ORIGIN = 12  # 避免单视频过度采样

def cap_per_origin(episodes):
    grouped = episodes.groupby("video_origin_id")
    capped = grouped.apply(lambda g: g.sample(min(len(g), MAX_EPISODES_PER_ORIGIN)))
    return capped
```

#### 规则 3: 跨数据集同源优先保留 ThinkStream

当两个数据集包含同一原视频时:
- ThinkStream 样本有 thoughts + timestamp → **优先保留**
- Streamo 样本降级为 B 类（弱锚点）
- 如果 Streamo 有更丰富的时间区间标注（st_time/end_time），则两者互补保留

#### 规则 4: near_duplicate_clip 只保留一条

```python
# 对 near_duplicate_clip 组，保留标注最丰富的那条
def dedup_near_duplicates(group):
    # 按标注丰富度排序：conversations + thoughts 数量
    return group.sort_values("annotation_richness", ascending=False).iloc[0]
```

### 0.9 审计脚本清单

实施前需要开发 3 个脚本：

#### 脚本 1: `scripts/audit/build_video_asset_registry.py`

```
输入: Streamo raw_data (json/parquet) + ThinkStream dataset (parquet from HF)
输出: data/audit/video_asset_registry.parquet
功能:
  - 读取两个数据集的原始数据
  - 解析 video_path → origin_video_id
  - 统一字段映射
  - 标记 underlying_source
  - 初始化可用性标签为 null
```

**GPU 需求**: 无，纯 CPU  
**预计耗时**: ~10 分钟（读取 + 解析 58.5 万行）

#### 脚本 2: `scripts/audit/resolve_dedup.py`

```
输入: data/audit/video_asset_registry.parquet
输出: data/audit/video_asset_registry_deduped.parquet + data/audit/dedup_report.json
功能:
  - 层 1: 字符串级同源匹配
  - 层 2: 元数据匹配（需要视频文件存在）
  - 层 3: 视觉近重（需要 GPU，可延后）
  - 层 4: 语义近重
  - 生成 dedup_group_id 和 dedup_status
```

**GPU 需求**: 层 1-2 无需 GPU；层 3 需 H20 2 卡（SigLIP），~2 小时  
**预计耗时**: 层 1-2 ~5 分钟；层 3 ~2 小时；层 4 ~30 分钟

#### 脚本 3: `scripts/audit/generate_audit_report.py`

```
输入: data/audit/video_asset_registry_deduped.parquet
输出: data/audit/audit_report.md + 8 张 CSV 表
功能:
  - 生成 8 张审计报表
  - 回答 12 个审计问题
  - 输出可用性分布和去重统计
```

**GPU 需求**: 无  
**预计耗时**: ~2 分钟

### 0.10 审计必须回答的 12 个问题

| # | 问题 | 信息来源 |
|---|------|---------|
| 1 | 两个数据集各有多少行？ | 表 1 |
| 2 | 各有多少唯一 clip？ | 表 1 |
| 3 | 各有多少唯一原视频？ | 表 1 |
| 4 | **跨数据集同源重叠比例是多少？** | 表 7（最重要） |
| 5 | 各自底层来源是什么？Kinetics-700 占多少？ | 表 2 |
| 6 | 每个原视频平均派生多少 clip/样本？ | 表 3 |
| 7 | 任务类别分布是什么？ | 表 4 |
| 8 | temporal scope 分布是什么？ | 表 4 |
| 9 | response format 分布是什么？ | 表 4 |
| 10 | 有多少样本具备可用的时间戳/支持区间？ | 表 5 |
| 11 | 哪些样本适合 recall，比例多少？ | 表 6 |
| 12 | 哪些样本只能做弱锚点，比例多少？ | 表 6 |

### 0.11 Streamo 到 Agent 协议的字段映射

| Streamo 原始字段 | Agent 用途 | 映射方式 |
|-----------------|----------|---------|
| `question.time` | `ask_time_ms` | `int(time) * 1000` |
| `response.st_time` | support_span 起点 | `st_time * 1000` |
| `response.end_time` | `earliest_response_time_ms` | `end_time * 1000` |
| `response.content` | `natural_response`（弱标签，不直接使用） | 作为参考 |
| `task_type` | `task_bucket` 初始分类 | 规则映射 |
| `source` | `underlying_source` | 直接使用 |
| `video_path` | 解析 `origin_video_id` | `parse_origin_id()` |

**Streamo 的 3 种 response 模式**:
1. `st_time + end_time` → Standby 到 Response 的区间 → **可派生 timing 和 recall 样本**
2. `time` 字段有值 → 即时回答 → **可派生 current_perception 样本**
3. 两者都有 → 多 response 场景 → **可派生一问多答样本**

### 0.12 ThinkStream 到 Agent 协议的字段映射

| ThinkStream 原始字段 | Agent 用途 | 映射方式 |
|--------------------|----------|---------|
| `conversations[role=user].timestamp` | `ask_time_ms` | `timestamp * 1000` |
| `conversations[role=assistant].timestamp` | `earliest_response_time_ms` | `timestamp * 1000` |
| `thoughts.timestamp` | chunk-level think 对齐 | 按 2s chunk 分桶 |
| `thoughts.think` | `<think>` 内容 | 直接复用（增量式） |
| `interaction_mode` | `task_bucket` | 直接映射 |
| `temporal_scope` | 决定是否 need_recall | Past → 可能 need_recall |
| `response_format` | 决定是否 usable_for_rl | MC/Binary/Counting → RL 可用 |
| `video_path` | 解析 `origin_video_id` + `underlying_source` | `parse_origin_id()` |

**ThinkStream 的 recall 样本派生方式**:
- `temporal_scope == "Past"` 且 `interaction_mode == "Real-time Dialogue"` → **recall 候选**
- 人为缩小 recent_window（如从 24s 缩到 8s）→ **forced-recall 样本**
- 直接复用 `thoughts` 作为 think 内容，只需 teacher 补充 `gold_query` + `recall 后 response`

### 0.13 审计→构造→训练的完整流程

```
Step 1: 下载两个数据集到本地
  ↓
Step 2: 运行 build_video_asset_registry.py
  ↓ 输出: video_asset_registry.parquet
Step 3: 运行 resolve_dedup.py
  ↓ 输出: deduped registry + dedup_report.json
Step 4: 运行 generate_audit_report.py
  ↓ 输出: 8 张审计表 + 12 个问题答案
Step 5: 人工审核审计报告，确认可用性分布
  ↓
Step 6: 基于审计结果选取首批 50 视频
  ↓ 从 A 类 + 高 recallability 中选取
Step 7: 进入 Stage 0-6 数据构造流程（第 5 章）
```

---

## 1. 原始方案矛盾分析与解决方案

对比 `docs/agent数据构造文档.txt` 与现有代码库，发现以下 **7 个核心矛盾**。每个矛盾必须在编码前解决。

### 1.1 Think Token 预算不匹配

| 维度 | 代码现状 | 文档提案 |
|------|---------|---------|
| 默认预算 | `rollout_max_think_tokens=20`（`scripts/demo.py:L26`） | 均值 24–40 token/chunk |
| 最大预算 | `max_think_tokens=512`（`inference.py:L1138` 默认参数） | burst 64–80 token |
| 奖励函数 | `_compute_think_length_factor(avg, target=20, step=5)` → target=20 时，24 token 已获满分 | 文档期望区分 24 与 40 的差异 |

**问题根源**：`demo.py` 和 RL 训练中使用的 `rollout_max_think_tokens=20` 是硬限制，会在 20 token 处强制插入 `</think>`（`think_budget_sample` 函数, `inference.py:1199-1206`）。文档提出的 24-40 均值在此限制下不可能实现。

**解决方案**：

- **训练阶段**：将 `rollout_max_think_tokens` 提升至 **48**（常规上限），burst 场景通过 `max_think_tokens=80` 覆盖
- **奖励函数**：修改 `_compute_think_length_factor` 的 `target_tokens` 从 20 → 36，`step_window` 从 5 → 8，使奖励在 20-48 区间有区分度
- **推理阶段**：`demo.py` 的 `MAX_THINK_TOKENS` 改为 48，与训练对齐
- **文档修正**：将 burst 从 64-80 降为 48-64，避免生成过长 think 导致延迟不可控

### 1.2 Chunk 时长：动态 vs 固定

| 维度 | 代码现状 | 文档提案 |
|------|---------|---------|
| 计算方式 | `video_chunk_size = ceil(duration / DEFAULT_MAX_CHUNKS)` 动态计算（`stream_data_processor.py:477`） | 固定 2s，高运动时降至 1s |
| 典型值 | 60s 视频 → 0.5s/chunk; 240s → 2s/chunk; 600s → 5s/chunk | 统一 2s |
| 自适应机制 | 不存在 | 根据运动/问答状态切换 |

**问题根源**：当前代码将视频均匀切成 `DEFAULT_MAX_CHUNKS=120` 个 chunk，chunk 时长随视频时长变化。文档假设固定 2s chunk，但这与代码逻辑完全不同。

**解决方案**：

- **Agent 数据构造**采用 **固定 2s chunk**：新增 `AGENT_CHUNK_SEC = 2.0` 常量，Agent 数据的 `_build_agent_messages()` 函数使用固定时长切分
- **保留原有动态切分**：不修改 `_build_messages()` 和 `DEFAULT_MAX_CHUNKS`，保证原有 SFT/RL 数据兼容
- **V1 不做自适应**：1s chunk 的自适应切换留到 V2，首版统一 2s 简化实现
- **影响**：60s 视频产生 30 个 chunk，120s 视频产生 60 个 chunk，240s 视频产生 120 个 chunk（与原有上限一致）

### 1.3 帧采样策略

| 维度 | 代码现状 | 文档提案 |
|------|---------|---------|
| 采样方式 | `FRAMES_PER_CHUNK = 2`，每 chunk 均匀 2 帧（`stream_data_processor.py:273`） | 16-20 帧：8 均匀 + 4 高运动 + 4 最新 + 4-6 recall |
| 场景检测 | 不存在 | 需要运动分析、场景切换检测 |

**问题根源**：文档提出的 16-20 帧方案需要复杂的自适应采样和场景检测，但项目中无此基础设施。

**解决方案**：

- **V1 保持 `FRAMES_PER_CHUNK=2` 不变**：每个 2s chunk 采样 2 帧 = 1fps，计算简单且与现有代码兼容
- **Recent visual buffer**：24s 窗口 = 12 chunks × 2 帧 = 24 帧，已足够
- **Recall 帧**：recall 命中时，从 keyframe 存储中取 top1-2 的 keyframe（已在 `segment_archive` 中预存），作为 `images` 字段传入
- **V2 再考虑自适应**：待首版验证后，再引入运动检测和 OCR 感知帧选取

### 1.4 Action 格式不兼容

| 维度 | 代码现状 | 文档提案 |
|------|---------|---------|
| 格式 | `<think>...</think><response>...` 或 `<think>...</think><silent>` | `<think>...</think><action>silent\|response\|recall</action>[<response>...\|<query>...]` |
| 正则 | `_CHUNK_FORMAT_RE = r"^<think>.*?</think>(?:<response>.*\|<silent>)<\|im_end\|>$"` (`grpo.py:41-44`) | 需要匹配 `<action>` 包裹 |
| 奖励 | `_compute_format_reward()` 只识别双格式（`grpo.py:52-57`） | 需识别三格式 + query JSON |

**问题根源**：新 action 格式与现有正则和奖励函数完全不兼容。用新格式数据训练后，所有 format_reward 都为 0。

**解决方案**：

- **更新正则** `_CHUNK_FORMAT_RE`：
  ```python
  _CHUNK_FORMAT_RE = re.compile(
      r"^<think>.*?</think>"
      r"<action>(?:silent|response|recall)</action>"
      r"(?:<response>.*?</response>|<query>\{.*?\}</query>)?"
      r"<\|im_end\|>$",
      re.DOTALL,
  )
  ```
- **扩展 format_reward**：增加 `_check_action_type()` 函数，验证 action 类型与后续内容的一致性（response 后必须有 `<response>`，recall 后必须有 `<query>`，silent 后无额外内容）
- **新增 `_RESPONSE_RE`**：从 `<response>(.*?)</response>` 提取答案（替换当前 `grpo.py:127` 的正则）
- **向后兼容**：SFT-A 阶段同时支持新旧格式，通过 config flag 切换

### 1.5 Recall 机制从零构建

| 维度 | 代码现状 | 文档提案 |
|------|---------|---------|
| Recall action | 不存在 | 10-15% chunk 动作 |
| 检索模块 | 不存在 | embedding 检索 + reranker |
| Recall reward | 不存在 | hit@k + support coverage |

**问题根源**：recall 是全新能力，涉及推理引擎（检测 recall action → 调用检索 → 注入结果 → 继续生成）、训练奖励（action 正确性 + 检索质量）、数据格式（tool call 伪装为 user 消息）三个层面。

**解决方案（分阶段实现）**：

- **SFT 阶段（优先）**：recall 以 **预构造的固定交互** 形式出现在训练数据中，不需要在线检索。`sft_final.jsonl` 中 recall 交互已经展开为 assistant→user(recall_result)→assistant 的多轮消息。模型只需学会输出正确格式。
- **RL 阶段（次优先）**：recall reward 基于离线预计算。`rl_pool.jsonl` 中已包含 `gold_query` 和 `reward_spec`，rollout 时检测到 recall action 后查预计算的检索结果表，不做在线检索。
- **推理阶段（最后）**：在 `streaming_video_chat()` 中增加 recall action 检测逻辑，调用外部检索接口。V1 可用简单的 embedding 近似搜索。

### 1.6 训练规模过大

| 维度 | 文档提案 | 用户修正 |
|------|---------|---------|
| SFT-A | 12万–18万 episodes | 几千条 |
| SFT-B | 8万–12万 episodes | 几千条 |
| SFT-C | 8万–12万 episodes | V1 不做 |
| SFT-D | 3万–6万 episodes | V1 不做 |
| RL-A | 未明确 | 几百条 |
| RL-B | 2万–4万 trajectories | 几百条 |

**解决方案**：

- **首批 50 视频**：产出 ~1000 SFT + ~300 RL 样本
- **V1 只做 4 个训练阶段**：SFT-A(协议) → SFT-B(recall) → RL-A(action/timing) → RL-B(recall utility, 可选)
- **合并 SFT-C/D 到 SFT-B**：将复合对话和 hardening 的关键样本混入 SFT-B，不单独设 stage
- **扩量节点**：首批验证通过后 → 500 视频（~10000 SFT + ~3000 RL）→ 2000 视频（~40000 SFT + ~12000 RL）

### 1.7 推理延迟不现实

| 维度 | 文档暗示 | 实际估算 |
|------|---------|---------|
| 目标 | 实时（<2s/chunk） | Qwen2.5-VL-3B ~30ms/token × (48 think + 80 response) = ~3.8s |
| 瓶颈 | - | autoregressive decode |

**解决方案**：

- **接受 3-5s/chunk 延迟**：对于 2s chunk，模型处理延迟约 3-5s，即有 1-3s 的缓冲延迟。在流式视频场景中，这意味着模型总是落后视频进度 1-2 个 chunk，但这是可接受的
- **优化方向（V2）**：
  - 投机解码（speculative decoding）
  - 量化至 INT4
  - 升级到更大但更高效的 MoE 模型
- **Think 预算控制**：通过 `think_budget_sample` 严格限制 think token 数量，避免 think 过长拖慢响应

---

## 2. 硬件规格与吞吐量基线

### 2.1 AMD 8 卡集群

**硬件**: MI300X 192GB HBM3 × 8（假设，需确认具体型号）  
**用途**: Qwen3.5-397B FP8 teacher 推理（8 卡 tensor parallel）  
**显存**: 192GB × 8 = 1536GB，397B FP8 模型约占 400GB，剩余用于 KV cache  

**吞吐量基线**（Qwen3.5-397B 为 MoE 架构，激活参数 ~70B）：

| 指标 | 估算值 | 计算依据 |
|------|--------|---------|
| Prefill 速度 | 3000–5000 tokens/s | 8 卡 TP，FP8 compute，MoE 稀疏激活 |
| Decode 速度 | 15–25 tokens/s | 受限于 memory bandwidth，MoE 路由开销 |
| 单次任务包生成 | 60–100 秒/视频 | 输入 ~2300 tok (prefill ~0.5s) + 输出 ~1500 tok (decode ~60-100s) |
| Query 生成/修复 | 8–13 秒/任务 | 输入 ~500 tok + 输出 ~200 tok |
| 争议仲裁 | 12–20 秒/样本 | 输入 ~1000 tok + 输出 ~300 tok |
| 显存占用（推理） | ~500GB (model) + ~200GB (KV cache) | 远低于 1536GB，有余量 |

**并发能力**：单实例 8 卡 TP，一次只能处理 1 个请求。可通过 continuous batching（如 vLLM）提升吞吐，但 V1 先用串行。

**不同规模的耗时**：

| 任务 | 50 视频 | 500 视频 | 2000 视频 |
|------|--------|---------|----------|
| Stage 2 任务包生成 | 50 × 80s = **67 min** | 500 × 80s = **11.1 h** | 2000 × 80s = **44.4 h** |
| Stage 4 query repair (~5%) | 8 × 12s = **1.6 min** | 75 × 12s = **15 min** | 300 × 12s = **60 min** |
| Stage 5 争议仲裁 (~10%) | 15 × 16s = **4 min** | 150 × 16s = **40 min** | 600 × 16s = **2.7 h** |
| **AMD 总计** | **~73 min** | **~12 h** | **~47 h** |

### 2.2 H20 8 卡集群

**硬件**: NVIDIA H20 96GB HBM3 × 8  
**用途**: 预处理 + 主训练 + 检索验证  
**显存**: 96GB × 8 = 768GB

**各任务吞吐量**：

| 任务 | 模型 | 显存/卡 | 单卡速度 | 8 卡并行速度 |
|------|------|--------|---------|-------------|
| Dense caption | Qwen3-VL-8B FP16 | ~18GB | ~2s/segment (输入 ~1000 visual tok, 输出 ~100 tok) | ~0.25s/segment |
| OCR 提取 | Qwen3-VL-8B FP16 | 共享 | ~1.5s/frame (输入 1 帧, 输出 ~50 tok) | ~0.19s/frame |
| ASR 转录 | Whisper-large-v3 | ~4GB | 实时 ×30 (4s 音频 → ~0.13s) | ~0.07s/segment (2 卡) |
| Embedding (visual) | SigLIP-SO400M | ~2GB | ~100 images/s | ~400 images/s (4 卡) |
| Embedding (text) | BGE-M3 / GTE | ~2GB | ~200 texts/s | ~400 texts/s (2 卡) |
| Reranker | BGE-Reranker-v2 | ~2GB | ~50 pairs/s | ~200 pairs/s (4 卡) |
| 小 VL 推理 (验证) | Qwen3-VL-8B | ~18GB | ~3-5s/sample | ~0.4-0.6s/sample |

**SFT 训练吞吐量**（Qwen2.5-VL-3B, ZeRO-3, 8 卡）：

| 参数 | 值 | 计算 |
|------|-----|------|
| 每卡 batch size | 8 | 当前 `sft.sh` 配置 |
| 全局 batch size | 64 | 8 × 8 |
| 单步耗时 | ~5-8s | 含前向、反向、ZeRO-3 通信 |
| 1000 样本/epoch | ~16 steps | 1000/64 ≈ 16 |
| 每 epoch 耗时 | ~80-128s ≈ **~2 min** | 16 × 5-8s |

**RL 训练吞吐量**（Qwen2.5-VL-3B, GRPO, 8 卡）：

| 参数 | 值 | 计算 |
|------|-----|------|
| 每卡 batch size | 1 | 当前 `rl.sh` 配置 |
| Group size | 8 | 每 sample 生成 8 条 rollout |
| Micro batch size | 8 | rollout 并行度 |
| 单步耗时 | ~25-35s | rollout (~20s) + update (~5-15s) |
| 200 trajectories | ~200 steps | 200/1 (per device) |
| 总耗时 | ~5000-7000s ≈ **~83-117 min** | 200 × 25-35s |

### 2.3 并行调度总览

AMD 8 卡和 H20 8 卡是**两个独立集群**，可以完全并行：

```
时间线（首批 50 视频）:

AMD 8卡:  |----- Stage 2: 任务包生成 (67min) -----|-S4 repair(2min)-|-S5 仲裁(4min)-|
           |                                        |                |               |
H20 8卡:  |-S0: 预处理(25min)-|-S1(10min)-|--------|-S3(15min)-|-S4 检索(3min)-|-S5 闸门(12min)-|-S6(5min)-|

端到端: ~73 分钟 (受 AMD Stage 2 限制)
```

**关键瓶颈**: Stage 2（397B 任务包生成）是全流程瓶颈，AMD 卡利用率最高。H20 在等待 Stage 2 完成期间可做其他预处理工作或 retriever 预训练。

### 2.4 存储需求估算

| 数据类型 | 50 视频 | 2000 视频 | 存储位置 |
|---------|--------|----------|---------|
| 原始视频 | ~5GB | ~200GB | SSD |
| Keyframes (JPEG, ~50KB/张) | ~125MB | ~5GB | SSD |
| Dense caption + OCR + ASR 文本 | ~5MB | ~200MB | JSONL |
| Embeddings (768d float32, ~3KB/segment) | ~4MB | ~150MB | NPY |
| segment_archive.jsonl | ~2MB | ~80MB | JSONL |
| event_timeline.jsonl | ~1MB | ~40MB | JSONL |
| episode_recall.jsonl | ~3MB | ~120MB | JSONL |
| sft_final.jsonl | ~5MB | ~200MB | JSONL |
| **总计** | **~5.2GB** | **~206GB** | - |

---

## 3. 核心协议定义

### 3.1 三种 Action 精确输出格式

**重要**: 以下格式是 **student 模型生成的输出格式**。结构化字段（milestones、verification 等）仅存在于数据构造管理层，不进入模型的输入或输出。

#### Action 1: 沉默（Silent）

当前 chunk 不需要回答用户，模型仅维护内部推理状态。

```
<think>当前画面显示厨师在切菜，暂无需要回答的问题。</think><action>silent</action>
```

- Think 内容: 增量式，只写当前 chunk 的新增判断，**不复述整段视频**
- Think 长度: 15–30 tokens（常规），无 burst
- 占比: chunk-level 58–65%

#### Action 2: 回答（Response）

当前 chunk 有足够证据回答用户问题。

```
<think>用户问现在在做什么，当前画面清楚看到厨师在翻炒蒜末。</think><action>response</action><response>厨师正在翻炒蒜末。</response>
```

- Think 内容: 写出回答的推理过程和证据来源
- Think 长度: 20–40 tokens（常规），48–64（burst，用于复杂推理）
- Response 长度: 40–140 tokens（简短即时答 40–80，解释型 80–140）
- 占比: chunk-level 23–30%

#### Action 3: 检索回忆（Recall）

当前 chunk 证据不足，需要检索历史片段。

```
<think>用户问前面先下锅的是什么，当前窗口内看不到早期加料步骤，需要检索历史。</think><action>recall</action><query>{"query":"蒜末 下锅 翻炒 第一步 调料","time_bias":"past_far","target":"procedure","topk":3}</query>
```

- Think 内容: 说明为什么当前窗口不够、需要什么信息
- Think 长度: 20–40 tokens
- Query 字段说明:
  - `query`: 检索关键词，**不是问句、不用代词**，包含实体锚点，8–20 tokens
  - `time_bias`: `past_recent`（近期 8s 外）| `past_far`（远期 >16s）| `current`（当前附近）
  - `target`: `entity` | `action` | `event` | `ocr` | `procedure` | `cause`
  - `topk`: 检索返回数量，固定为 3
- 占比: chunk-level 10–15%
- V1 限制: **每个 chunk 最多一次 recall**

### 3.2 Recall 两步交互协议

Recall 是一个两步交互：模型发出 recall → 系统返回结果 → 模型根据结果决定 silent 或 response。

在训练数据中，这个交互被展开为多轮消息（tool call 伪装为 user 消息）：

```
# Step 1: assistant 发出 recall
{"role": "assistant", "content": "<think>...</think><action>recall</action><query>{...}</query>"}

# Step 2: 系统返回 recall 结果（伪装为 user 消息）
{"role": "user", "content": "<recall_result>\n<item rank=\"1\" start=\"40.0\" end=\"44.0\">caption: 厨师把蒜末倒入锅中并开始翻炒。 asr: 先把蒜末下锅。</item>\n<item rank=\"2\" start=\"42.0\" end=\"46.0\">caption: 锅中继续翻炒蒜末。</item>\n</recall_result>\n继续按协议回答。"}

# Step 3: assistant 根据结果回答（只能 silent 或 response，不能再 recall）
{"role": "assistant", "content": "<think>检索命中蒜末下锅片段，证据充分。</think><action>response</action><response>前面先下锅的是蒜末。</response>"}
```

**硬性规则**：
- 收到 `<recall_result>` 后，assistant 只能输出 `silent` 或 `response`，**不能再次 recall**
- `<recall_result>` 中只放 top1-2 的 keyframes + 文本摘要，不放完整 clip
- Recall 结果以 `<item>` 标签组织，每项包含 `rank`、`start`（秒）、`end`（秒）和文本描述

### 3.3 新 System Prompt

替换 `thinkstream/data/stream_data_processor.py:431-438` 的 `SYSTEM_PROMPT` 常量。

**中文版（agent 训练用）**：

```python
AGENT_SYSTEM_PROMPT = (
    "你是流视频问答 agent。你会持续接收视频片段流。\n\n"
    "每次 assistant turn 必须严格输出以下三种形式之一：\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    "<query>{\"query\":\"...\",\"time_bias\":\"...\",\"target\":\"...\",\"topk\":3}</query>\n\n"
    "规则：\n"
    "- 每个 turn 只允许一个 action。\n"
    "- think 只写当前新增判断，不复述整段视频。\n"
    "- 若当前 recent window 证据已足够，直接 response，不要 recall。\n"
    "- 若问题依赖已离开 recent window 的历史内容，优先 recall。\n"
    "- recall query 必须短、可检索、非完整问句、避免代词和泛词。\n"
    "- 收到 <recall_result> 后，只能再输出 silent 或 response。\n"
    "- 若当前 chunk 不该说话，输出 silent。"
)
```

**英文版（兼容国际评测用）**：

```python
AGENT_SYSTEM_PROMPT_EN = (
    "You are a streaming video QA agent receiving a continuous stream of video chunks.\n\n"
    "Each assistant turn must output exactly one of these three forms:\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    "<query>{\"query\":\"...\",\"time_bias\":\"...\",\"target\":\"...\",\"topk\":3}</query>\n\n"
    "Rules:\n"
    "- Only one action per turn.\n"
    "- Think only about new observations in the current chunk. Do not restate the whole video.\n"
    "- If the recent window has enough evidence, respond directly without recall.\n"
    "- If the answer depends on content that has left the recent window, use recall.\n"
    "- Recall queries must be short, searchable keywords — not full questions, no pronouns.\n"
    "- After receiving <recall_result>, only output silent or response.\n"
    "- If no speech is needed for the current chunk, output silent."
)
```

### 3.4 消息格式设计

基于现有 `_build_messages()`（`stream_data_processor.py:468`）的结构扩展。

#### 3.4.1 User 消息结构

每个 chunk 的 user 消息包含视频片段和可选的文本内容：

```json
{
  "role": "user",
  "content": [
    {
      "type": "video",
      "video": "/path/to/video.mp4",
      "video_start": 40.0,
      "video_end": 42.0
    },
    {
      "type": "text",
      "text": "<dialogue_history>\n用户: 现在做到哪一步了？\n助手: 已经开始翻炒配料了。\n</dialogue_history>\n<think_memory>\n当前在炒锅阶段；早期准备步骤已离开 recent window。\n</think_memory>\n<question>\n前面先下锅的是什么？\n</question>"
    }
  ]
}
```

**字段说明**：

- `<dialogue_history>`: 最近 3-5 轮的对话摘要（可选，仅在多轮对话时出现）
  - 获取方式: 从前序 chunk 的 assistant response 和 user question 自动拼接
  - 最大长度: 200 tokens
- `<think_memory>`: 当前 think buffer 的压缩摘要（可选，仅在 think buffer 接近溢出时出现）
  - 获取方式: 对前序 think 内容的自动摘要（小模型或规则）
  - 最大长度: 100 tokens
- `<question>`: 用户在当前 chunk 时间窗口内提出的问题（可选）
  - 获取方式: 从原始标注的 conversations 中按时间戳匹配

#### 3.4.2 Recall 结果消息

```json
{
  "role": "user",
  "content": "<recall_result>\n<item rank=\"1\" start=\"40.0\" end=\"44.0\">caption: 厨师把蒜末倒入锅中并开始翻炒。 asr: 先把蒜末下锅。 ocr: 无。</item>\n<item rank=\"2\" start=\"42.0\" end=\"46.0\">caption: 锅中继续翻炒蒜末。 asr: 无。 ocr: 无。</item>\n</recall_result>\n继续按协议回答。"
}
```

- 每个 `<item>` 包含: rank、start(秒)、end(秒)、caption、asr、ocr
- 最多返回 top2（减少 token 消耗）
- 末尾加 "继续按协议回答。" 作为 generation prompt

### 3.5 Token 预算约束总表

| 组件 | 最小 | 典型 | 最大 | 硬限制 |
|------|------|------|------|--------|
| Think (silent chunk) | 10 | 20 | 30 | 48 |
| Think (response chunk) | 15 | 30 | 48 | 64 |
| Think (recall chunk) | 15 | 30 | 48 | 64 |
| Response (简短即时) | 10 | 40 | 80 | 140 |
| Response (解释型) | 40 | 80 | 140 | 200 |
| Recall query JSON | 30 | 50 | 70 | 100 |
| Recall result (per item) | 30 | 60 | 100 | 150 |
| Dialogue history | 0 | 50 | 150 | 200 |
| Think memory | 0 | 30 | 80 | 100 |

**单 chunk 最坏情况 token 消耗**：
- Recall 路径: think(64) + action+query(100) + recall_result(300) + think(64) + response(200) = **~728 tokens**
- 普通路径: think(48) + response(140) = **~188 tokens**

### 3.6 Action 分布目标

| Action | Chunk-level 占比 | Episode-level 含义 |
|--------|-----------------|-------------------|
| silent | 58–65% | 大部分 chunk 不需要说话 |
| response | 23–30% | 约 1/4 的 chunk 有回答 |
| recall | 10–15% | 约 1/10 的 chunk 触发检索 |

这个分布目标用于：
1. 训练数据构造时的配比检查
2. RL 训练中 action 分布偏差的惩罚项
3. 推理时异常检测（recall 频率持续 >25% 说明模型过度依赖检索）

---

## 4. 六张数据表完整 Schema

### 4.1 segment_archive.jsonl

**用途**: 外部 episodic memory 的底表。Recall 查询的目标索引。  
**产出阶段**: Stage 0（预处理资产化）  
**切分参数**: segment_sec=4, overlap_sec=2, raw_fps=2  
**存储路径**: `data/agent/segment_archive/{video_id}.jsonl`（每个视频一个文件）

| 字段 | 类型 | 必填 | 说明 | 获取方式 |
|------|------|------|------|---------|
| `video_id` | string | ✅ | 视频唯一标识 | 从原始数据集 |
| `segment_id` | string | ✅ | 格式: `{video_id}_{start_ms:08d}_{end_ms:08d}` | 脚本生成 |
| `start_ms` | int | ✅ | 起始时间（毫秒） | 切分计算 |
| `end_ms` | int | ✅ | 结束时间（毫秒） | 切分计算 |
| `scene_id` | string | ✅ | 所属场景ID | PySceneDetect |
| `raw_fps` | int | ✅ | 原始抽帧帧率，固定 2 | 常量 |
| `num_frames` | int | ✅ | 本 segment 帧数，通常为 8 (4s×2fps) | 计算 |
| `keyframe_paths` | list[string] | ✅ | 1-2 张代表帧路径 | 规则选取（最高运动/中间帧） |
| `ocr_text` | string | ✅ | OCR 提取文本，无则 "无" | Qwen3-VL-8B OCR prompt |
| `asr_text` | string | ✅ | ASR 转录文本，无则 "无" | Whisper-large-v3 |
| `dense_caption` | string | ✅ | 4s 片段的描述（1-2 句） | Qwen3-VL-8B captioning prompt |
| `entity_tags` | list[string] | ✅ | 实体标签 | 从 dense_caption NER 提取 |
| `action_tags` | list[string] | ✅ | 动作标签 | 从 dense_caption 动词提取 |
| `state_tags` | list[string] | ✅ | 状态标签 | 从 dense_caption 状态词提取 |
| `memory_keys` | list[string] | ✅ | 可检索短键（3-5 个） | 从 entity+action+OCR 压缩 |
| `visual_emb_path` | string | ✅ | 视觉 embedding 存储路径 | SigLIP encode keyframe |
| `text_emb_path` | string | ✅ | 文本 embedding 存储路径 | BGE/GTE encode caption+keys |
| `salience` | float | ✅ | 显著性分数 [0, 1] | 见下方公式 |
| `motion_score` | float | ✅ | 运动量分数 [0, 1] | 光流或帧差计算 |
| `has_ocr` | bool | ✅ | 是否有 OCR 文本 | `ocr_text != "无"` |
| `has_asr` | bool | ✅ | 是否有 ASR 文本 | `asr_text != "无"` |

**salience 计算公式**:

```python
salience = (
    0.30 * motion_score          # 运动量 [0,1]
    + 0.25 * entity_density      # len(entity_tags) / max_entities, clamp [0,1]
    + 0.20 * has_ocr_or_asr      # 1.0 if has_ocr or has_asr else 0.0
    + 0.15 * action_density      # len(action_tags) / max_actions, clamp [0,1]
    + 0.10 * scene_boundary      # 1.0 if this segment crosses scene boundary else 0.0
)
```

**memory_keys 生成规则**:

```python
def generate_memory_keys(entity_tags, action_tags, ocr_text, dense_caption):
    keys = []
    # 规则 1: 每个 entity + 其关联 action
    for entity in entity_tags[:3]:
        for action in action_tags[:2]:
            keys.append(f"{entity} {action}")
    # 规则 2: OCR 关键词
    if ocr_text != "无":
        keys.append(ocr_text[:20])  # 截断
    # 规则 3: dense_caption 前 5 个关键词
    keywords = extract_keywords(dense_caption, topk=5)
    keys.extend(keywords[:2])
    return keys[:5]  # 最多 5 个
```

**完整示例**:

```json
{
  "video_id": "vid_001",
  "segment_id": "vid_001_00040000_00044000",
  "start_ms": 40000,
  "end_ms": 44000,
  "scene_id": "scene_05",
  "raw_fps": 2,
  "num_frames": 8,
  "keyframe_paths": [
    "data/agent/keyframes/vid_001/frame_040000.jpg",
    "data/agent/keyframes/vid_001/frame_042000.jpg"
  ],
  "ocr_text": "无",
  "asr_text": "先把蒜末下锅",
  "dense_caption": "厨师把蒜末倒入锅中并开始翻炒。",
  "entity_tags": ["厨师", "锅", "蒜末"],
  "action_tags": ["倒入", "翻炒"],
  "state_tags": ["cooking", "garlic_added"],
  "memory_keys": ["蒜末 倒入", "厨师 翻炒", "蒜末下锅", "翻炒 第一步"],
  "visual_emb_path": "data/agent/embeddings/vid_001/visual_040000.npy",
  "text_emb_path": "data/agent/embeddings/vid_001/text_040000.npy",
  "salience": 0.82,
  "motion_score": 0.75,
  "has_ocr": false,
  "has_asr": true
}
```

### 4.2 event_timeline.jsonl

**用途**: 视频事件结构化表，397B teacher 的主输入。  
**产出阶段**: Stage 1（结构化时间线生成）  
**存储路径**: `data/agent/event_timeline/{video_id}.jsonl`

| 字段 | 类型 | 必填 | 说明 | 获取方式 |
|------|------|------|------|---------|
| `video_id` | string | ✅ | 视频ID | 继承 |
| `event_id` | string | ✅ | 格式: `evt_{video_id}_{seq:04d}` | 自增 |
| `start_ms` | int | ✅ | 事件起始时间 | 聚合 segment 边界 |
| `end_ms` | int | ✅ | 事件结束时间 | 聚合 segment 边界 |
| `support_segment_ids` | list[string] | ✅ | 支撑 segment 列表 | 时间重叠的 segments |
| `event_type` | string | ✅ | 事件类型枚举 | 分类规则（见下） |
| `summary` | string | ✅ | 事件一句话描述 | 聚合 dense_captions |
| `entities` | list[string] | ✅ | 涉及实体 | 合并 segment entity_tags |
| `preconditions` | list[string] | ✅ | 前置条件 | 小 VL 模型 + 因果规则 |
| `effects` | list[string] | ✅ | 结果影响 | 小 VL 模型 + 因果规则 |
| `evidence` | object | ✅ | 证据来源 | 聚合 visual/asr/ocr |
| `causal_links_prev` | list[string] | ✅ | 因果前驱事件ID | 时间邻近 + 实体共现 |
| `causal_links_next` | list[string] | ✅ | 因果后继事件ID | 时间邻近 + 实体共现 |
| `importance` | float | ✅ | 重要性 [0, 1] | 见下方公式 |

**event_type 枚举值**:

| 值 | 说明 | 判定规则 |
|----|------|---------|
| `procedure_step` | 流程步骤 | 动作序列中的一步 |
| `state_change` | 状态变化 | entity state_tags 发生变化 |
| `entity_action` | 实体动作 | 单个实体的独立动作 |
| `dialogue` | 对话/语音 | ASR 文本为主要内容 |
| `ocr_event` | 文字/数字出现 | OCR 文本为主要内容 |
| `scene_transition` | 场景切换 | scene_id 发生变化 |

**importance 计算公式**:

```python
importance = (
    0.25 * duration_norm              # (end_ms - start_ms) / max_event_duration, clamp [0,1]
    + 0.25 * len(entities) / 5.0      # 实体数量归一化
    + 0.25 * causal_depth / 5.0       # len(causal_links_prev) + len(causal_links_next)
    + 0.15 * has_asr_or_ocr           # 1.0 if evidence.asr or evidence.ocr else 0.0
    + 0.10 * avg_segment_salience     # 支撑 segments 的平均 salience
)
```

**因果链推断规则**:

```python
def infer_causal_links(events):
    """基于时间邻近和实体共现推断因果链"""
    for i, evt_a in enumerate(events):
        for j, evt_b in enumerate(events):
            if j <= i:
                continue
            # 条件 1: 时间邻近（B 在 A 结束后 8s 内开始）
            if evt_b["start_ms"] - evt_a["end_ms"] > 8000:
                continue
            # 条件 2: 实体共现（至少 1 个共同实体）
            shared = set(evt_a["entities"]) & set(evt_b["entities"])
            if not shared:
                continue
            # 条件 3: effect-precondition 语义匹配（简单字符串包含）
            effect_match = any(
                eff in pre
                for eff in evt_a.get("effects", [])
                for pre in evt_b.get("preconditions", [])
            )
            if shared or effect_match:
                evt_a["causal_links_next"].append(evt_b["event_id"])
                evt_b["causal_links_prev"].append(evt_a["event_id"])
```

**完整示例**:

```json
{
  "video_id": "vid_001",
  "event_id": "evt_vid_001_0017",
  "start_ms": 39800,
  "end_ms": 44600,
  "support_segment_ids": [
    "vid_001_00038000_00042000",
    "vid_001_00040000_00044000",
    "vid_001_00042000_00046000"
  ],
  "event_type": "procedure_step",
  "summary": "把蒜末下锅并翻炒",
  "entities": ["厨师", "锅", "蒜末"],
  "preconditions": ["锅已加热"],
  "effects": ["进入爆香步骤"],
  "evidence": {
    "visual": "手把蒜末倒入锅中",
    "asr": "先把蒜末下锅",
    "ocr": ""
  },
  "causal_links_prev": ["evt_vid_001_0016"],
  "causal_links_next": ["evt_vid_001_0018"],
  "importance": 0.88
}
```

### 4.3 episode_recall.jsonl

**用途**: Recall 训练的主样本表。SFT、retriever、reranker、RL 数据均从此派生。  
**产出阶段**: Stage 2-5（经 397B 生成 + 检索验证 + 6 闸门过滤）  
**存储路径**: `data/agent/episode_recall.jsonl`（全局单文件）

| 字段 | 类型 | 必填 | 说明 | 获取方式 |
|------|------|------|------|---------|
| `episode_id` | string | ✅ | 格式: `ep_recall_{seq:06d}` | 自增 |
| `source_dataset` | string | ✅ | `streamo` 或 `thinkstream` | 来源标记 |
| `video_id` | string | ✅ | 视频ID | 继承 |
| `task_type` | string | ✅ | 任务类型枚举（见下） | Stage 2 397B 生成 |
| `video_type` | string | ✅ | 视频类型 | 元数据标注 |
| `question` | string | ✅ | 用户问题 | Stage 2 397B 生成 |
| `dialogue_history` | list[object] | ⬜ | 前序对话 | Stage 2 生成或继承 |
| `ask_time_ms` | int | ✅ | 提问时间（毫秒） | Stage 2 生成 |
| `earliest_response_time_ms` | int | ✅ | 最早可答时间 | Stage 2 生成或 prefix sufficiency 检测 |
| `recent_window_sec` | int | ✅ | recent buffer 大小，固定 24 | 常量 |
| `recent_visible_segment_ids` | list[string] | ✅ | 当前可见 segment | 按 ask_time 和 recent_window 计算 |
| `support_event_ids` | list[string] | ✅ | 支撑事件ID | Stage 2 397B 指定 |
| `support_segment_ids` | list[string] | ✅ | 支撑 segment | 从 support_event_ids 展开 |
| `support_outside_recent` | bool | ✅ | 支撑是否在窗口外 | `overlap(recent, support) == 0` |
| `need_recall` | bool | ✅ | 是否需要 recall | Stage 2 + Stage 5 验证 |
| `recall_reason` | string | ⬜ | recall 原因说明 | Stage 2 397B 生成 |
| `gold_query` | object | ✅ | 最佳检索 query | Stage 4 验证选出 |
| `gold_retrieved_segment_ids` | list[string] | ✅ | 检索命中 segment | Stage 4 检索返回 |
| `canonical_answer` | object | ✅ | 标准答案（可验证） | Stage 2 397B 生成 |
| `natural_response` | string | ✅ | 自然语言回答 | Stage 2 397B 生成 |
| `sparse_think_milestones` | list[object] | ✅ | 稀疏 think 关键点 | Stage 2 397B 生成 |
| `difficulty` | string | ✅ | `easy`/`medium`/`hard`/`very_hard` | Stage 2 分类 |
| `verification` | object | ✅ | 6 闸门验证结果 | Stage 5 填充 |
| `provenance` | object | ✅ | 数据血缘 | 自动记录 |

**task_type 枚举值**:

| 值 | 说明 | 典型 need_recall |
|----|------|-----------------|
| `current_perception` | 当前感知即时答 | false |
| `short_temporal` | 短时序理解 | false |
| `retrospective_detail` | 长程细节回忆 | true |
| `procedural_state` | 程序状态回忆 | true |
| `compare_past_present` | 过去 vs 现在比较 | true |
| `long_causal` | 长程因果推理 | true |
| `ocr_history` | OCR 历史比对 | true |
| `multi_turn_followup` | 多轮追问 | true/false |
| `delayed_trigger` | 延迟触发 | false |
| `continuous_narration` | 连续解说 | false |

**canonical_answer 结构**:

```json
{
  "answer_type": "slot|entity|yesno|number|ordered_steps|span|multiple_choice",
  "value": {"ingredient": "蒜末"}
}
```

优先构造可验证答案类型（multiple_choice, yesno, number, slot），便于 RL 自动评测。

**verification 结构**:

```json
{
  "gate1_support_outside_recent": true,
  "gate2_retrieval_hit_at_3": true,
  "gate3_support_coverage": 0.67,
  "gate4_no_recall_fail": true,
  "gate4_no_recall_score": 0.15,
  "gate5_with_recall_pass": true,
  "gate5_with_recall_score": 0.92,
  "gate6_counterfactual_fail": true,
  "gate6_score_drop": 0.77,
  "all_gates_passed": true
}
```

**完整示例**: 见原始文档第 428-507 行（`docs/agent数据构造文档.txt`），此处不重复。

### 4.4 sft_final.jsonl

**用途**: Student 模型 SFT 训练的最终输入。遵循 Qwen3-VL 的 messages 格式。  
**产出阶段**: Stage 6（样本组装）  
**存储路径**: `data/agent/sft_final/sft_{stage}_{version}.jsonl`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 样本唯一ID |
| `messages` | list[object] | ✅ | Qwen3-VL 格式的多轮消息 |
| `videos` | list[string] | ✅ | 视频文件路径列表（只放 recent buffer clip） |
| `images` | list[string] | ⬜ | 图片路径列表（recall 命中的 keyframes） |
| `episode_id` | string | ✅ | 溯源到 episode_recall |
| `sample_type` | string | ✅ | `recall_positive`/`no_recall_control`/`false_recall_negative`/`protocol`/`easy_qa` |

**视频裁剪逻辑**:

```python
def get_recent_clip_path(video_path, ask_time_ms, recent_window_sec=24):
    """裁剪 recent buffer 对应的视频片段"""
    clip_start = max(0, ask_time_ms / 1000 - recent_window_sec)
    clip_end = ask_time_ms / 1000
    clip_path = f"data/agent/clips/{video_id}_{int(clip_start)}_{int(clip_end)}.mp4"
    # 使用 ffmpeg 裁剪: ffmpeg -ss {clip_start} -to {clip_end} -i {video_path} -c copy {clip_path}
    return clip_path
```

**Recall 样本完整示例**:

```json
{
  "id": "sft_recall_000123",
  "messages": [
    {
      "role": "system",
      "content": "你是流视频问答 agent。每次 assistant turn 必须严格输出 <think>...</think><action>silent|response|recall</action> 协议。"
    },
    {
      "role": "user",
      "content": "<video>\n<dialogue_history>\n用户: 现在做到哪一步了？\n助手: 已经开始翻炒配料了。\n</dialogue_history>\n<think_memory>\n当前在炒锅阶段；早期准备步骤已离开 recent window。\n</think_memory>\n<question>\n前面先下锅的是什么？\n</question>"
    },
    {
      "role": "assistant",
      "content": "<think>当前窗口内看不到早期加料顺序，需要检索历史步骤。</think><action>recall</action><query>{\"query\":\"蒜末 下锅 翻炒 第一步 调料\",\"time_bias\":\"past_far\",\"target\":\"procedure\",\"topk\":3}</query>"
    },
    {
      "role": "user",
      "content": "<recall_result>\n<item rank=\"1\" start=\"40.0\" end=\"44.0\">caption: 厨师把蒜末倒入锅中并开始翻炒。 asr: 先把蒜末下锅。 ocr: 无。</item>\n<item rank=\"2\" start=\"42.0\" end=\"46.0\">caption: 锅中继续翻炒蒜末。 asr: 无。 ocr: 无。</item>\n</recall_result>\n继续按协议回答。"
    },
    {
      "role": "assistant",
      "content": "<think>检索结果直接命中早期加料步骤，证据充分。</think><action>response</action><response>前面先下锅的是蒜末，先炒香之后才继续往后做。</response>"
    }
  ],
  "videos": ["data/agent/clips/vid_001_54_78.mp4"],
  "images": [
    "data/agent/keyframes/vid_001/frame_040000.jpg",
    "data/agent/keyframes/vid_001/frame_042000.jpg"
  ],
  "episode_id": "ep_recall_000123",
  "sample_type": "recall_positive"
}
```

**No-recall Control 样本**（同一问题，但 support 还在 recent window 内）:

```json
{
  "id": "sft_norecall_000123",
  "messages": [
    {"role": "system", "content": "..."},
    {
      "role": "user",
      "content": "<video>\n<question>\n刚才下锅的是什么？\n</question>"
    },
    {
      "role": "assistant",
      "content": "<think>刚才看到厨师把蒜末倒入锅中，当前窗口内就有答案。</think><action>response</action><response>刚下锅的是蒜末。</response>"
    }
  ],
  "videos": ["data/agent/clips/vid_001_36_46.mp4"],
  "images": [],
  "episode_id": "ep_recall_000123",
  "sample_type": "no_recall_control"
}
```

**False-recall Negative 样本**（表面像回忆，但当前窗口已足够）:

```json
{
  "id": "sft_false_recall_000123",
  "messages": [
    {"role": "system", "content": "..."},
    {
      "role": "user",
      "content": "<video>\n<question>\n之前放了什么调料？\n</question>"
    },
    {
      "role": "assistant",
      "content": "<think>虽然用户说"之前"，但当前窗口内刚好能看到加入蒜末的画面，不需要检索。</think><action>response</action><response>放了蒜末。</response>"
    }
  ],
  "videos": ["data/agent/clips/vid_001_38_46.mp4"],
  "images": [],
  "episode_id": "ep_recall_000123",
  "sample_type": "false_recall_negative"
}
```

### 4.5 retriever_train.jsonl

**用途**: Embedding retriever / reranker 训练。SWIFT 兼容格式。  
**产出阶段**: Stage 6（从 episode_recall 派生）  
**存储路径**: `data/agent/retriever_train.jsonl`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `messages` | list[object] | ✅ | Query 端（system + user） |
| `positive_messages` | list[list[object]] | ✅ | 正样本 segments |
| `negative_messages` | list[list[object]] | ✅ | 负样本 segments (4+条) |
| `episode_id` | string | ✅ | 溯源 |

**Hard Negatives 的 4 种类型及采样策略**:

| 类型 | 说明 | 采样方式 | 配比 |
|------|------|---------|------|
| temporal_near_miss | 同视频、同场景、相邻但非 support | 取 gold support 前后 1-2 个 segment | 2 份 |
| semantic_confounder | 同实体不同动作，或同动作不同对象 | 在 segment_archive 中按 entity_tags 模糊匹配 | 1 份 |
| ocr_confounder | 文本相似但数字/关键词不同 | 按 OCR 文本编辑距离 < 5 筛选 | 0.5 份 |
| same_video_random | 同视频中风格相近但无关的片段 | 随机采样，排除 support 和 recent window | 0.5 份 |

**完整示例**: 见原始文档第 562-605 行。

### 4.6 rl_pool.jsonl

**用途**: RL 训练数据，带 reward 元信息。  
**产出阶段**: Stage 6（从 episode_recall 派生）  
**存储路径**: `data/agent/rl_pool.jsonl`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `episode_id` | string | ✅ | 溯源 |
| `video_id` | string | ✅ | 视频ID |
| `question` | string | ✅ | 用户问题 |
| `ask_time_ms` | int | ✅ | 提问时间 |
| `recent_window_sec` | int | ✅ | recent buffer 大小 |
| `support_segment_ids` | list[string] | ✅ | 支撑 segments |
| `gold_query` | object | ⬜ | recall 最佳 query（need_recall=true 时） |
| `canonical_answer` | object | ✅ | 标准答案 |
| `reward_spec` | object | ✅ | 奖励计算规格 |

**reward_spec 结构与现有 grpo.py 的映射关系**:

```json
{
  "need_recall": true,
  "action_full_credit": "recall_then_response",
  "action_partial_credit": "direct_response_if_correct",
  "earliest_response_time_ms": 78000,
  "latest_full_credit_time_ms": 82000,
  "retrieval_hit_metric": "hit@3",
  "wrong_action_penalty": 1.0,
  "over_recall_penalty": 0.3,
  "long_think_penalty_after_tokens": 48
}
```

| reward_spec 字段 | 对应 grpo.py 奖励维度 | 说明 |
|-----------------|---------------------|------|
| `earliest_response_time_ms` | `_compute_time_reward(gt_chunk_idx)` | 转换为 chunk_idx 后传入 |
| `latest_full_credit_time_ms` | `time_reward_slack` | 满分区间上界 |
| `canonical_answer` | `_compute_correctness_reward(gt_content)` | 提取答案对比 |
| `long_think_penalty_after_tokens` | `_compute_think_length_factor(target)` | 设为 target |
| `need_recall` + `action_full_credit` | **新增** `_compute_action_reward()` | 需新增 |
| `retrieval_hit_metric` | **新增** `_compute_recall_utility_reward()` | 需新增 |
| `wrong_action_penalty` | **新增** | recall 时 response 或 response 时 recall 的惩罚 |
| `over_recall_penalty` | **新增** | 不需要 recall 却 recall 的惩罚 |

---

## 5. 数据构造流程 Stage 0–6

### 5.0 全流程概览

```
原始视频 (Streamo + ThinkStream)
    │
    ▼ Stage 0: 预处理资产化 [H20 8卡, ~25min/50vid]
segment_archive.jsonl + keyframes/ + embeddings/
    │
    ▼ Stage 1: 结构化时间线 [H20 2-4卡, ~10min/50vid]
event_timeline.jsonl
    │
    ▼ Stage 2: 397B 任务包生成 [AMD 8卡, ~67min/50vid]
episode_recall_raw.jsonl (未验证)
    │
    ▼ Stage 3: 稀疏→稠密 Think 展开 [H20 4卡, ~15min/50vid]
episode_recall_dense.jsonl (含 chunk-level 序列)
    │
    ▼ Stage 4: Query 生成与检索验证 [H20 2卡 + AMD 部分, ~4min/50vid]
episode_recall_verified.jsonl (query 已验证)
    │
    ▼ Stage 5: 6 闸门自动验证 [H20 4卡, ~12min/50vid]
episode_recall.jsonl (最终版，全部通过)
    │
    ▼ Stage 6: 样本组装 [CPU, ~5min]
sft_final.jsonl + retriever_train.jsonl + rl_pool.jsonl
```

### 5.1 Stage 0: 预处理资产化

**目标**: 将原始视频转化为可检索、可派生、可验证的结构化底座。

**输入**: 原始视频文件列表  
**输出**: `segment_archive.jsonl` + keyframe 图片 + embedding 文件

#### 5.1.1 子任务分解与 GPU 调度

Stage 0 包含 8 个子任务，按依赖关系分为 3 层：

```
Layer 1 (无依赖，可并行):
  ├─ [T0.1] 场景切割 + chunk 切分 (CPU)
  ├─ [T0.2] 2fps 抽帧 + keyframe 选取 (CPU/1卡)
  └─ [T0.3] ASR 转录 (H20 卡6-7, Whisper)

Layer 2 (依赖 T0.2 的抽帧结果):
  ├─ [T0.4] Dense caption (H20 卡0-3, Qwen3-VL-8B)
  ├─ [T0.5] OCR 提取 (H20 卡4-5, Qwen3-VL-8B)
  └─ [T0.6] Visual embedding (H20 卡6-7, SigLIP)

Layer 3 (依赖 T0.4/T0.5 的文本结果):
  ├─ [T0.7] 标签提取 + memory_keys 生成 (CPU, 规则脚本)
  ├─ [T0.8] Text embedding (H20 卡6-7, BGE)
  └─ [T0.9] salience + recallability 计算 (CPU, 规则脚本)
```

#### 5.1.2 每个子任务的实现细节

**[T0.1] 场景切割 + chunk 切分**

```python
# 工具: PySceneDetect
# 输入: video_path
# 输出: scene_boundaries.json + chunk_boundaries.json

from scenedetect import detect, ContentDetector

def detect_scenes(video_path):
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    return [(s.get_timecode(), e.get_timecode()) for s, e in scene_list]

def build_segments(video_duration_ms, segment_sec=4, overlap_sec=2):
    """生成 4s segment, 2s overlap"""
    step_ms = (segment_sec - overlap_sec) * 1000  # 2000ms 步进
    segments = []
    start = 0
    while start + segment_sec * 1000 <= video_duration_ms:
        segments.append({
            "start_ms": start,
            "end_ms": start + segment_sec * 1000,
        })
        start += step_ms
    return segments
```

- **耗时**: ~6s/视频（CPU），50 视频 ~5 min
- **输出**: `data/agent/meta/{video_id}_scenes.json`, `data/agent/meta/{video_id}_segments.json`

**[T0.2] 2fps 抽帧 + keyframe 选取**

```python
# 工具: torchcodec (项目已依赖, requirements.txt)
from torchcodec.decoders import VideoDecoder

def extract_frames(video_path, segment, fps=2):
    decoder = VideoDecoder(video_path)
    start_sec = segment["start_ms"] / 1000
    end_sec = segment["end_ms"] / 1000
    timestamps = [start_sec + i/fps for i in range(int((end_sec - start_sec) * fps))]
    frames = [decoder.get_frame_at(t).data for t in timestamps]
    return frames

def select_keyframes(frames, n=2):
    """选择代表帧: 中间帧 + 最高运动帧"""
    if len(frames) <= n:
        return list(range(len(frames)))
    mid = len(frames) // 2
    # 运动检测: 简单帧差
    diffs = [torch.abs(frames[i+1].float() - frames[i].float()).mean().item()
             for i in range(len(frames)-1)]
    max_motion_idx = max(range(len(diffs)), key=lambda i: diffs[i])
    return sorted(set([mid, max_motion_idx]))[:n]
```

- **耗时**: ~3s/视频（1 卡 GPU decode），50 视频 ~3 min
- **输出**: `data/agent/keyframes/{video_id}/frame_{start_ms:06d}.jpg`

**[T0.3] ASR 转录**

```python
# 工具: Whisper-large-v3
# GPU: H20 卡6-7 (2卡)
# 显存: ~4GB/卡

import whisper

def transcribe_segment(audio_path, start_sec, end_sec):
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path, language="zh",
                              clip_timestamps=[start_sec, end_sec])
    return result["text"].strip() or "无"
```

- **吞吐**: 实时 ×30，4s segment → ~0.13s/segment/卡
- **并行**: 2 卡，50 视频 × 25 segments = 1250 segments → ~80s = **~1.5 min**
- **输出**: 写入 segment 的 `asr_text` 字段

**[T0.4] Dense caption**

```python
# 模型: Qwen3-VL-8B (FP16, ~18GB/卡)
# GPU: H20 卡0-3 (4卡, 每卡独立实例)
# Prompt:

CAPTION_PROMPT = """请用1-2句中文描述这个4秒视频片段中发生了什么。
要求：
- 描述主要动作和参与实体
- 如果有明显的状态变化，请提及
- 不要推测画面外的内容
- 保持简洁"""

# 输入: 4s 视频片段的 2 张 keyframes
# 输出: 1-2 句描述文本
```

- **吞吐**: ~2s/segment/卡（输入 ~1000 visual tokens + 100 text tokens，输出 ~80 tokens）
- **并行**: 4 卡各处理 1/4 segments → 1250/4 = 312 segments/卡 → 312 × 2s = 624s = **~10 min**
- **50 视频实际**: 因为 Layer 2 可在 T0.2 完成后立即开始，且 4 卡并行 → **~5 min**（与 T0.5 时间重叠）

**[T0.5] OCR 提取**

```python
# 模型: Qwen3-VL-8B (共享 T0.4 的模型实例, 交替推理)
# GPU: H20 卡4-5 (2卡)
# Prompt:

OCR_PROMPT = """请提取这张图片中所有可见的文字内容。
如果没有可见文字，回复"无"。
只输出文字内容本身，不要添加解释。"""

# 输入: 每个 segment 的 1 张 keyframe
# 输出: OCR 文本或 "无"
```

- **吞吐**: ~1.5s/frame/卡
- **并行**: 2 卡，1250 segments × 1 frame/segment = 1250 frames → 1250/(2×0.67) = 933s = **~4 min**（与 T0.4 并行）

**[T0.6] Visual Embedding**

```python
# 模型: SigLIP-SO400M (~2GB)
# GPU: H20 卡6-7 (共享 ASR 完成后的卡)

from transformers import AutoModel, AutoProcessor

def compute_visual_embedding(keyframe_path):
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    image = Image.open(keyframe_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy()  # shape: (1, 768)
```

- **吞吐**: ~100 images/s/卡（batch=64）
- **并行**: 2 卡，2500 keyframes → 2500/(2×100) = 12.5s = **<1 min**

**[T0.7] 标签提取 + memory_keys**

```python
# 纯 CPU 规则脚本
import jieba
import jieba.posseg as pseg

def extract_tags(dense_caption):
    words = pseg.cut(dense_caption)
    entities = [w.word for w in words if w.flag in ('nr', 'ns', 'nz', 'n')]
    actions = [w.word for w in words if w.flag == 'v']
    states = [w.word for w in words if w.flag in ('a', 'ad')]
    return entities[:5], actions[:3], states[:3]
```

- **耗时**: ~0.01s/segment，1250 segments → ~12s = **<1 min**

**[T0.8] Text Embedding**

```python
# 模型: BGE-M3 (~2GB)
# GPU: H20 卡6-7

def compute_text_embedding(text):
    # text = dense_caption + " " + " ".join(memory_keys)
    model = FlagModel("BAAI/bge-m3")
    emb = model.encode(text)
    return emb  # shape: (1024,)
```

- **吞吐**: ~200 texts/s/卡
- **并行**: 2 卡，1250 texts → 1250/(2×200) = 3.1s = **<1 min**

**[T0.9] Salience + Recallability 计算**

```python
def compute_recallability_score(video_segments):
    """视频级 recallability 分数，用于筛选适合做 recall 的视频"""
    duration_sec = video_segments[-1]["end_ms"] / 1000
    duration_score = min(1.0, duration_sec / 120)  # 120s 以上满分

    entities = set()
    for seg in video_segments:
        entities.update(seg["entity_tags"])
    entity_recurrence = len([e for e in entities
                            if sum(1 for s in video_segments if e in s["entity_tags"]) >= 3]) / max(len(entities), 1)

    # event_chain_depth: 需要 event_timeline, 在 Stage 1 后补算
    # 暂时用 segment 数量近似
    event_chain_depth = min(1.0, len(video_segments) / 30)

    procedurality = sum(1 for s in video_segments if any(
        a in s["action_tags"] for a in ["加入", "倒入", "切", "搅拌", "打开", "关闭", "移动"]
    )) / max(len(video_segments), 1)

    ocr_density = sum(1 for s in video_segments if s["has_ocr"]) / max(len(video_segments), 1)
    asr_density = sum(1 for s in video_segments if s["has_asr"]) / max(len(video_segments), 1)

    # cut_rate: 场景切换频率（越高越不适合 recall）
    scene_ids = [s["scene_id"] for s in video_segments]
    cuts = sum(1 for i in range(1, len(scene_ids)) if scene_ids[i] != scene_ids[i-1])
    cut_rate = cuts / max(len(video_segments), 1)

    score = (
        0.25 * duration_score
        + 0.20 * entity_recurrence
        + 0.20 * event_chain_depth
        + 0.15 * procedurality
        + 0.10 * ocr_density
        + 0.10 * asr_density
        - 0.20 * cut_rate
    )
    return max(0.0, min(1.0, score))
```

- **耗时**: ~0.1s/视频，50 视频 → <1s

#### 5.1.3 Stage 0 GPU 调度甘特图（50 视频）

```
时间(min):  0    2    4    6    8   10   12   14   16   18   20   22   24   26
            |    |    |    |    |    |    |    |    |    |    |    |    |    |
H20 卡0-3:  [============ T0.4 Dense Caption (~10min) ============]
H20 卡4-5:  [========= T0.5 OCR (~8min) =========]
H20 卡6-7:  [T0.3 ASR(~2min)][T0.6 VisEmb(<1min)][T0.8 TxtEmb(<1min)]
CPU:        [T0.1 Scene(~5min)][T0.2 Frames(~3min)][T0.7 Tags(<1min)][T0.9 Score(<1s)]
```

**实际端到端**: T0.1 + T0.2 需要 ~8min → T0.4/T0.5 在第 8 分钟开始 → T0.4 在第 18 分钟完成 → T0.7/T0.8/T0.9 在第 19 分钟完成  
**Stage 0 总耗时**: **~20 分钟**（50 视频），**~5 小时**（2000 视频）

### 5.2 Stage 1: 结构化时间线生成

**目标**: 从 segment 聚合出事件，推断因果链。  
**输入**: `segment_archive.jsonl`  
**输出**: `event_timeline.jsonl`

#### 5.2.1 具体步骤

```python
def build_event_timeline(segments, video_id):
    """从 segments 聚合生成 event_timeline"""
    events = []
    current_event_segments = [segments[0]]

    for i in range(1, len(segments)):
        prev = segments[i-1]
        curr = segments[i]

        # 判断是否应该开始新事件
        should_split = (
            # 条件 1: 场景切换
            curr["scene_id"] != prev["scene_id"]
            # 条件 2: 实体完全不重叠
            or not set(curr["entity_tags"]) & set(prev["entity_tags"])
            # 条件 3: caption 语义相似度低（用 text embedding 余弦距离）
            or cosine_similarity(curr["text_emb"], prev["text_emb"]) < 0.7
            # 条件 4: 事件已累积超过 3 个 segment (12s)
            or len(current_event_segments) >= 3
        )

        if should_split:
            events.append(aggregate_event(current_event_segments, video_id, len(events)))
            current_event_segments = [curr]
        else:
            current_event_segments.append(curr)

    if current_event_segments:
        events.append(aggregate_event(current_event_segments, video_id, len(events)))

    # 推断因果链
    infer_causal_links(events)

    return events

def aggregate_event(segments, video_id, seq):
    """将多个 segment 聚合为一个 event"""
    return {
        "video_id": video_id,
        "event_id": f"evt_{video_id}_{seq:04d}",
        "start_ms": segments[0]["start_ms"],
        "end_ms": segments[-1]["end_ms"],
        "support_segment_ids": [s["segment_id"] for s in segments],
        "event_type": classify_event_type(segments),
        "summary": merge_captions([s["dense_caption"] for s in segments]),
        "entities": list(set(e for s in segments for e in s["entity_tags"])),
        "preconditions": [],  # 由小 VL 模型补充或留空
        "effects": [],        # 由小 VL 模型补充或留空
        "evidence": {
            "visual": segments[0]["dense_caption"],
            "asr": next((s["asr_text"] for s in segments if s["has_asr"]), ""),
            "ocr": next((s["ocr_text"] for s in segments if s["has_ocr"]), ""),
        },
        "causal_links_prev": [],
        "causal_links_next": [],
        "importance": 0.0,  # 后续计算
    }
```

#### 5.2.2 GPU 调度

- **主要计算**: cosine_similarity 需要加载 text embedding (numpy)，~0.01s/pair
- **小 VL 模型补充 preconditions/effects（可选）**: 2 卡 Qwen3-VL-8B
  - 每个 event ~2s（输入 caption + entities，输出 preconditions + effects）
  - 50 视频 × 5 events/video = 250 events → 250 × 2s / 2 卡 = 250s = **~4 min**
- **规则计算**: CPU 即可，~1s/视频

**Stage 1 总耗时**: **~5-10 分钟**（50 视频），**~3-4 小时**（2000 视频）

### 5.3 Stage 2: 397B Teacher 任务包生成

**目标**: 对每个视频，一次性生成 8-20 个候选任务（含 recall 和 non-recall）。  
**输入**: `event_timeline.jsonl` + `segment_archive` 摘要  
**输出**: `episode_recall_raw.jsonl`

#### 5.3.1 Prompt 构造

使用原始文档第 916-969 行的 teacher prompt 模板（第 5.1 节）。

**输入准备**:

```python
def prepare_teacher_input(video_id, event_timeline, segment_archive):
    """为 397B teacher 准备输入"""
    # 1. event_timeline 直接序列化（~500-1000 tokens）
    timeline_json = json.dumps(event_timeline, ensure_ascii=False, indent=2)

    # 2. segment_archive 压缩为摘要（~300-500 tokens）
    # 不发送 embedding 路径和原始帧路径，只保留文本信息
    archive_summary = []
    for seg in segment_archive:
        archive_summary.append({
            "id": seg["segment_id"],
            "t": f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}s",
            "cap": seg["dense_caption"][:50],
            "ent": seg["entity_tags"][:3],
            "act": seg["action_tags"][:2],
            "ocr": seg["ocr_text"][:20] if seg["has_ocr"] else "",
            "asr": seg["asr_text"][:20] if seg["has_asr"] else "",
            "sal": round(seg["salience"], 2),
        })

    return {
        "timeline_json": timeline_json,
        "archive_summary": json.dumps(archive_summary, ensure_ascii=False),
        "video_type": infer_video_type(event_timeline),
    }
```

**总输入 token 估算**: system prompt (~200) + timeline (~800) + archive_summary (~1000) + instruction (~300) = **~2300 tokens**  
**总输出 token 估算**: 10 tasks × ~150 tokens/task = **~1500 tokens**

#### 5.3.2 GPU 调度

- **模型**: Qwen3.5-397B FP8, 8 卡 tensor parallel
- **串行处理**: 每次处理 1 个视频
- **单视频耗时**: prefill ~0.5s + decode ~60-100s = **~80s/视频**（取中位数）

| 规模 | 视频数 | 耗时计算 | 总耗时 |
|------|--------|---------|--------|
| 首批 | 50 | 50 × 80s | **67 min** |
| 中批 | 500 | 500 × 80s | **11.1 h** |
| 大批 | 2000 | 2000 × 80s | **44.4 h** |

**优化方案（中批/大批）**:
- 使用 vLLM 的 continuous batching：batch=4 可提升吞吐约 2-3x → 2000 视频降至 ~15-20h
- 需要测试 MI300X 上 vLLM 的兼容性

#### 5.3.3 输出后处理

```python
def postprocess_teacher_output(raw_output, video_id):
    """后处理 397B 输出，分配难度等级"""
    tasks = json.loads(raw_output)["tasks"]
    for task in tasks:
        # 分配难度
        if not task["need_recall"] and task["task_type"] in ["current_perception", "short_temporal"]:
            task["difficulty"] = "easy"
        elif not task["need_recall"]:
            task["difficulty"] = "medium"
        elif task["task_type"] in ["retrospective_detail", "procedural_state", "compare_past_present"]:
            task["difficulty"] = "hard"
        else:
            task["difficulty"] = "very_hard"

        # 生成 episode_id
        task["episode_id"] = f"ep_recall_{uuid4().hex[:6]}"
        task["video_id"] = video_id
        task["source_dataset"] = infer_source(video_id)

    return tasks
```

**难度分布目标**: Easy 25% / Medium 35% / Hard 30% / Very hard 10%  
如果 397B 产出的分布偏离目标，在 Stage 6 样本组装时通过采样调整。

### 5.4 Stage 3: 稀疏→稠密 Think 展开

**目标**: 将 397B 生成的 sparse_think_milestones 展开为 chunk-level 的完整 action 序列。  
**输入**: `episode_recall_raw.jsonl`（含 sparse milestones）  
**输出**: `episode_recall_dense.jsonl`（含 chunk-level action 序列）

#### 5.4.1 展开逻辑

```python
def expand_sparse_to_dense(episode, segment_archive, chunk_sec=2):
    """将稀疏 milestone 展开为 chunk-level action 序列"""
    ask_time_ms = episode["ask_time_ms"]
    recent_start_ms = max(0, ask_time_ms - episode["recent_window_sec"] * 1000)

    # 1. 确定 chunk 范围: 从 recent window 开始到 ask_time + 若干 chunk
    chunk_start_ms = recent_start_ms
    chunk_end_ms = ask_time_ms + 4000  # 额外 2 个 chunk 的缓冲
    num_chunks = int((chunk_end_ms - chunk_start_ms) / (chunk_sec * 1000))

    # 2. 为每个 chunk 分配 action
    chunks = []
    milestones = {m["time_ms"]: m["text"] for m in episode["sparse_think_milestones"]}
    question_time = episode["ask_time_ms"]
    response_time = episode["earliest_response_time_ms"]

    for i in range(num_chunks):
        t_start = chunk_start_ms + i * chunk_sec * 1000
        t_end = t_start + chunk_sec * 1000
        t_mid = (t_start + t_end) / 2

        # 找到落在此 chunk 内的 milestone
        chunk_milestone = None
        for mt, mtext in milestones.items():
            if t_start <= mt < t_end:
                chunk_milestone = mtext
                break

        # 分配 action
        if episode["need_recall"] and abs(t_mid - question_time) < chunk_sec * 1000:
            # Recall chunk: 提问发生时触发 recall
            action = "recall"
            think = chunk_milestone or f"用户提问，当前窗口内找不到答案，需要检索历史片段。"
        elif abs(t_mid - response_time) < chunk_sec * 1000:
            # Response chunk
            action = "response"
            think = chunk_milestone or f"证据充足，可以回答。"
        else:
            # Silent chunk
            action = "silent"
            if chunk_milestone:
                think = chunk_milestone
            else:
                # 从最近的 segment 生成简短 think
                nearby_seg = find_nearest_segment(segment_archive, t_mid)
                think = generate_incremental_think(nearby_seg)

        chunks.append({
            "chunk_idx": i,
            "start_ms": t_start,
            "end_ms": t_end,
            "action": action,
            "think": think,
        })

    episode["chunk_sequence"] = chunks
    return episode

def generate_incremental_think(segment):
    """为 silent chunk 生成简短的增量 think (规则模板)"""
    templates = [
        f"画面中{segment['entity_tags'][0] if segment['entity_tags'] else '主体'}在{segment['action_tags'][0] if segment['action_tags'] else '活动'}，暂无需回答。",
        f"当前观察到{segment['dense_caption'][:20]}，继续监控。",
        f"没有新问题，保持关注。",
    ]
    return random.choice(templates)
```

#### 5.4.2 GPU 调度

- **主要计算**: 大部分是 CPU 规则逻辑
- **小 VL 模型辅助（可选）**: 对 silent chunk 的 think 质量要求不高时，用模板即可
- **如果需要 VL 模型生成 think**: H20 4 卡，~1s/chunk
  - 50 视频 × 500 任务 × ~15 chunks/任务 = 7500 chunks → 7500/(4×1) = 1875s = **~31 min**
  - **优化**: silent chunk 用模板（~80% chunk），只对 response/recall chunk 用 VL → ~1500 chunks → **~6 min**

**Stage 3 总耗时**: **~5-15 分钟**（50 视频），取决于是否用 VL 模型

### 5.5 Stage 4: Recall Query 生成与检索验证

**目标**: 验证 recall query 能否检索命中 gold support segments。  
**输入**: `episode_recall_dense.jsonl` + segment_archive embeddings  
**输出**: `episode_recall_verified.jsonl`

#### 5.5.1 具体步骤

```python
def verify_recall_queries(episode, segment_archive, embeddings_index):
    """验证 recall query 的检索命中率"""
    if not episode["need_recall"]:
        return episode  # 非 recall 样本跳过

    gold_support = set(episode["support_segment_ids"])
    best_query = None
    best_retrieved = None

    # Step 1: 测试 3 个 query candidates
    for candidate in episode["query_candidates"]:
        query_emb = encode_query(candidate["query"])  # BGE encode
        top20 = embeddings_index.search(query_emb, k=20)  # FAISS search
        top5 = rerank(candidate["query"], top20)  # BGE-Reranker
        top3 = top5[:3]

        # 检查是否命中 gold support
        retrieved_ids = set(r["segment_id"] for r in top3)
        hit = len(retrieved_ids & gold_support) > 0
        coverage = compute_temporal_overlap(retrieved_ids, gold_support)

        if hit and coverage >= 0.5:
            if best_query is None or len(candidate["query"]) < len(best_query["query"]):
                best_query = candidate
                best_retrieved = [r["segment_id"] for r in top3]

    # Step 2: 如果 3 个都没命中，尝试 query repair
    if best_query is None:
        repaired_queries = repair_query_with_small_model(episode, segment_archive)
        for rq in repaired_queries:
            # 重复上述检索验证...
            pass

    # Step 3: 如果 repair 也失败，标记为 needs_teacher_repair
    if best_query is None:
        episode["needs_teacher_repair"] = True
    else:
        episode["gold_query"] = best_query
        episode["gold_retrieved_segment_ids"] = best_retrieved

    return episode
```

#### 5.5.2 FAISS 索引构建

```python
import faiss
import numpy as np

def build_faiss_index(segment_archive):
    """为一个视频的 segment embeddings 构建 FAISS 索引"""
    embeddings = []
    segment_ids = []
    for seg in segment_archive:
        emb = np.load(seg["text_emb_path"])
        embeddings.append(emb)
        segment_ids.append(seg["segment_id"])

    embeddings = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (cosine after normalize)
    index.add(embeddings)

    return index, segment_ids
```

#### 5.5.3 GPU 调度

| 子任务 | 模型 | 卡分配 | 50 视频耗时 |
|--------|------|--------|-----------|
| Query encoding (BGE) | BGE-M3 | H20 卡6 | ~150 queries × 3 candidates × 0.01s = ~4.5s |
| FAISS search | CPU (FAISS) | CPU | ~450 searches × 0.001s = ~0.5s |
| Reranking | BGE-Reranker | H20 卡7 | ~450 × 20 pairs × 0.02s = ~180s = **~3 min** |
| Query repair (小模型, ~20%) | Qwen3-VL-8B | H20 卡0-1 | ~30 queries × 2s = ~60s = ~1 min |
| 397B repair (~5%) | Qwen3.5-397B | AMD 8卡 | ~8 queries × 12s = ~96s = **~1.5 min** |

**Stage 4 总耗时**: **~5 分钟**（50 视频），**~2 小时**（2000 视频）

### 5.6 Stage 5: 6 闸门自动验证

**目标**: 每条 recall-positive 样本必须过 6 个自动闸门，全过才收入最终数据集。  
**输入**: `episode_recall_verified.jsonl`  
**输出**: `episode_recall.jsonl`（最终版）

#### 5.6.1 六个闸门的实现

**闸门 1: Support 在 recent window 外**

```python
def gate1_support_outside_recent(episode):
    recent_start = episode["ask_time_ms"] - episode["recent_window_sec"] * 1000
    recent_end = episode["ask_time_ms"]
    for sid in episode["support_segment_ids"]:
        seg = get_segment(sid)
        overlap = max(0, min(recent_end, seg["end_ms"]) - max(recent_start, seg["start_ms"]))
        if overlap > 0:
            return False, 0.0
    return True, 1.0
```
- **耗时**: ~0.001s/样本（纯计算）

**闸门 2: Retrieval hit@3**

```python
def gate2_retrieval_hit(episode):
    gold = set(episode["support_segment_ids"])
    retrieved = set(episode["gold_retrieved_segment_ids"])
    hit = len(gold & retrieved) > 0
    return hit, 1.0 if hit else 0.0
```
- **耗时**: 0（Stage 4 已完成）

**闸门 3: Support coverage ≥ 0.5**

```python
def gate3_support_coverage(episode):
    gold_spans = [(get_segment(s)["start_ms"], get_segment(s)["end_ms"])
                  for s in episode["support_segment_ids"]]
    retrieved_spans = [(get_segment(s)["start_ms"], get_segment(s)["end_ms"])
                       for s in episode["gold_retrieved_segment_ids"]]
    coverage = max_temporal_overlap(retrieved_spans, gold_spans)
    return coverage >= 0.5, coverage
```
- **耗时**: ~0.001s/样本

**闸门 4: No-recall fail（小 VL 推理）**

```python
def gate4_no_recall_fail(episode, vl_model):
    """只给 recent window，不给 recall 结果，让小 VL 尝试回答"""
    recent_clip = get_recent_clip(episode)
    prompt = f"问题: {episode['question']}\n请直接回答。"
    answer = vl_model.generate(video=recent_clip, prompt=prompt)
    score = compute_answer_score(answer, episode["canonical_answer"])
    return score < 0.4, score  # 必须答错或不确定
```
- **耗时**: ~3-5s/样本（VL 推理）
- **GPU**: H20 4 卡 Qwen3-VL-8B

**闸门 5: With-recall pass（小 VL 推理）**

```python
def gate5_with_recall_pass(episode, vl_model):
    """给 recent window + recall 结果，让小 VL 回答"""
    recent_clip = get_recent_clip(episode)
    recall_context = format_recall_result(episode["gold_retrieved_segment_ids"])
    prompt = f"回忆信息:\n{recall_context}\n\n问题: {episode['question']}\n请回答。"
    answer = vl_model.generate(video=recent_clip, prompt=prompt)
    score = compute_answer_score(answer, episode["canonical_answer"])
    return score >= 0.8, score  # 必须答对
```
- **耗时**: ~3-5s/样本

**闸门 6: Counterfactual fail（小 VL 推理）**

```python
def gate6_counterfactual_fail(episode, vl_model):
    """给 recall 结果但删掉 gold support，看性能是否下降"""
    recent_clip = get_recent_clip(episode)
    # 用非 gold 的 retrieved segments
    fake_recall = format_recall_result(
        [s for s in episode["gold_retrieved_segment_ids"]
         if s not in episode["support_segment_ids"]]
    ) or "未找到相关片段。"
    prompt = f"回忆信息:\n{fake_recall}\n\n问题: {episode['question']}\n请回答。"
    answer = vl_model.generate(video=recent_clip, prompt=prompt)
    score = compute_answer_score(answer, episode["canonical_answer"])
    score_drop = episode["verification"]["gate5_with_recall_score"] - score
    return score_drop >= 0.5, score_drop
```
- **耗时**: ~3-5s/样本

#### 5.6.2 GPU 调度

闸门 1-3 是纯计算（<1s 总计），闸门 4-6 需要 VL 推理。

**关键优化**: 闸门 4/5/6 是 3 次独立的 VL 推理，可以 batch 化处理。

```
H20 卡0-1: 闸门 4 (no-recall fail)
H20 卡2-3: 闸门 5 (with-recall pass)
H20 卡4-5: 闸门 6 (counterfactual fail)
H20 卡6-7: 空闲（或做 retriever 训练准备）
```

| 闸门 | 样本数 (50 视频) | 每卡耗时 | 2 卡并行 |
|------|-----------------|---------|---------|
| 4 | ~150 recall 样本 | 150 × 4s / 2 = 300s | **5 min** |
| 5 | ~150 | 同上 | **5 min** |
| 6 | ~150 | 同上 | **5 min** |

**但 4/5/6 可以同时跑在不同卡上**，所以实际 **~5 分钟**（50 视频）

**争议样本送 397B 仲裁**:
- 预计 ~10-15% 样本在某个闸门上"边界"（score 接近阈值）
- 50 视频 × ~15 边界样本 → 送 AMD 397B 裁决，~15 × 16s = **~4 min**

**Stage 5 总耗时**: **~10 分钟**（50 视频），**~7 小时**（2000 视频, 8 卡并行）

### 5.7 Stage 6: 样本组装

**目标**: 从 episode_recall.jsonl 组装最终的 SFT/Retriever/RL 数据。  
**输入**: `episode_recall.jsonl`  
**输出**: `sft_final.jsonl` + `retriever_train.jsonl` + `rl_pool.jsonl`

#### 5.7.1 SFT 样本组装

```python
def assemble_sft_samples(episodes, segment_archive):
    sft_samples = []

    for ep in episodes:
        if ep["need_recall"] and ep["verification"]["all_gates_passed"]:
            # 1. Recall-positive 样本
            sft_samples.append(build_recall_sft(ep, segment_archive))
            # 2. No-recall control (同一问题，提前问)
            sft_samples.append(build_no_recall_control(ep, segment_archive))
            # 3. False-recall negative (表面像回忆，但窗口够)
            sft_samples.append(build_false_recall_negative(ep, segment_archive))
        else:
            # Non-recall 样本 (easy/medium)
            sft_samples.append(build_simple_sft(ep, segment_archive))

    return sft_samples
```

**三合一绑定**: 每条 recall-positive 必须同时生成 no-recall control 和 false-recall negative，确保模型学会"该 recall 时 recall，不该 recall 时不 recall"。

#### 5.7.2 RL 样本筛选

```python
def assemble_rl_samples(episodes):
    rl_samples = []
    for ep in episodes:
        # 只选可验证答案类型
        if ep["canonical_answer"]["answer_type"] in [
            "multiple_choice", "yesno", "number", "slot", "entity"
        ]:
            rl_samples.append(build_rl_sample(ep))
    return rl_samples
```

#### 5.7.3 GPU 调度

- **纯 CPU 脚本**，无 GPU 需求
- **耗时**: ~5 分钟（50 视频），~30 分钟（2000 视频，主要是视频裁剪 I/O）

---

## 6. 首批小规模构造计划

### 6.1 视频选取策略

| 来源 | 数量 | 选取标准 | 用途 |
|------|------|---------|------|
| Streamo-Instruct-465K | 30 | duration > 60s, recallability_score top 30% | recall 任务主力 |
| ThinkStream 现有数据 | 20 | 有 conversations + thoughts 标注 | 协议 warm start + 缩窗 forced-recall |

**Streamo 视频选取的具体筛选流程**:

```python
# 1. 从 Streamo 元数据中筛选
candidates = [v for v in streamo_videos
              if v["duration_sec"] > 60
              and v["num_questions"] >= 3]  # 至少 3 个标注问题

# 2. 计算 recallability_score (Stage 0.9 的公式)
for v in candidates:
    v["recallability"] = compute_recallability_score(v)

# 3. 取 top 30% 后随机选 30 个，确保视频类型多样
selected = stratified_sample(
    sorted(candidates, key=lambda v: v["recallability"], reverse=True)[:len(candidates)//3],
    strata={"tutorial": 8, "vlog": 6, "cooking": 5, "screen_record": 4,
            "sports": 3, "drama": 2, "other": 2},
    total=30
)
```

### 6.2 各阶段数量流转

```
输入: 50 视频 (30 Streamo + 20 ThinkStream)
  │
  ▼ Stage 0: 预处理资产化
  │ 30 Streamo: 30 × ~25 segments = ~750 segments
  │ 20 ThinkStream: 20 × ~10 segments = ~200 segments (短视频, 更少 segments)
  │ 总计: ~950 segments, ~1900 keyframes
  │
  ▼ Stage 1: 结构化时间线
  │ ~200 events (~4 events/Streamo视频 × 30 + ~2.5/ThinkStream × 20)
  │
  ▼ Stage 2: 397B 任务包生成
  │ Streamo: 30 × 12 tasks = ~360 候选任务
  │ ThinkStream: 20 × 8 tasks = ~160 候选任务 (短视频产出少)
  │ 总计: ~520 候选任务
  │ 难度分布: Easy ~130, Medium ~182, Hard ~156, VeryHard ~52
  │
  ▼ Stage 3: Think 展开
  │ ~520 episodes 展开为 chunk-level 序列
  │ 其中 ~156 Hard + ~52 VeryHard = ~208 recall-positive 候选
  │
  ▼ Stage 4: Query 验证
  │ ~208 recall 候选中:
  │   - 直接命中: ~145 (70%)
  │   - 小模型 repair 后命中: ~30 (15%)
  │   - 397B repair 后命中: ~15 (7%)
  │   - 无法命中（丢弃）: ~18 (8%)
  │ 保留: ~190 recall-positive (query 验证通过)
  │
  ▼ Stage 5: 6 闸门过滤
  │ ~190 recall-positive 中:
  │   - 全部通过: ~120-135 (63-71%)
  │   - 闸门 4 fail (no-recall 也能答对): ~20
  │   - 闸门 5 fail (recall 后仍答错): ~15
  │   - 闸门 6 fail (删 support 后不下降): ~10
  │   - 边界仲裁: ~10 (397B 裁决)
  │ 最终 recall-positive: ~125-140
  │
  ▼ Stage 6: 样本组装
  │
  │ SFT 样本:
  │   recall-positive: ~130
  │   no-recall control: ~130
  │   false-recall negative: ~130
  │   ThinkStream 协议样本: ~200 (20 视频 × 10 chunks)
  │   Easy/Medium 非 recall: ~312 (130+182 中抽 80%)
  │   ────────────────
  │   SFT 总计: ~900 条
  │
  │ Retriever 训练:
  │   ~130 条 (每条带 4+ hard negatives)
  │
  │ RL 样本:
  │   可验证答案 (MC/yesno/number/slot): ~520 × 60% = ~312
  │   RL 总计: ~300 条
```

### 6.3 总耗时估算（首批 50 视频）

#### 6.3.1 分 Stage 串行耗时

| Stage | AMD 8卡 | H20 8卡 | CPU | 阶段耗时 |
|-------|--------|--------|-----|---------|
| 0 预处理 | - | 20 min | 8 min (并行) | 20 min |
| 1 时间线 | - | 8 min | <1 min | 8 min |
| 2 任务包 | **67 min** | - | - | 67 min |
| 3 Think 展开 | - | 6 min | <1 min | 6 min |
| 4 Query 验证 | 1.5 min | 3 min | <1 min | 4 min |
| 5 6 闸门 | 4 min | 5 min | <1 min | 8 min |
| 6 样本组装 | - | - | 5 min | 5 min |
| **串行总计** | **73 min** | **42 min** | **15 min** | **118 min** |

#### 6.3.2 并行优化后

AMD 和 H20 是独立集群，可完全并行：

```
时间线 (分钟):
  0   10   20   30   40   50   60   70   80
  |    |    |    |    |    |    |    |    |

AMD: |----------- Stage 2 (67min) ------------|S4(2)|S5(4)|
H20: |S0(20)|S1(8)|     等待S2      |S3(6)|S4(3)|S5(5)|S6(5)|
```

**关键路径**: Stage 0+1 (28min) → 等待 Stage 2 完成 (67min 处) → Stage 3+4+5+6 (24min)  
但 H20 的 Stage 0+1 在 AMD Stage 2 运行期间完成，不占关键路径。

**实际端到端**: max(AMD 73min, H20 28min + 等待 + 24min) = **~73 分钟**

### 6.4 扩量时间估算

| 规模 | 视频 | SFT | RL | AMD 时间 | H20 时间 | 端到端 |
|------|------|-----|-----|---------|---------|--------|
| 首批 | 50 | ~900 | ~300 | 73 min | 42 min | **~73 min** |
| 中批 | 500 | ~9000 | ~3000 | 12 h | 8 h | **~12 h** |
| 大批 | 2000 | ~36000 | ~12000 | 47 h | 35 h | **~48 h** |

### 6.5 扩量前验证清单

以下每项必须通过才能从首批扩到中批：

| # | 验证项 | 方法 | 通过标准 |
|---|--------|------|---------|
| 1 | Dense caption 质量 | 人工抽检 20 条 | 准确率 ≥ 90% |
| 2 | Event timeline 因果链 | 人工抽检 10 条视频的完整链 | 无断裂、无错误因果 |
| 3 | 397B 任务包多样性 | 统计 task_type 分布 | 与目标配比偏差 < 10% |
| 4 | 难度分布 | 统计 difficulty 分布 | Easy 20-30%, Medium 30-40%, Hard 25-35%, VH 5-15% |
| 5 | Recall query hit@3 | 统计整体通过率 | ≥ 70% |
| 6 | 6 闸门整体通过率 | 统计 all_gates_passed | ≥ 60% |
| 7 | SFT 样本格式解析 | 用修改后的 `_build_agent_messages()` 加载全部样本 | 0 解析错误 |
| 8 | SFT-A 训练可行性 | 跑 500 步 | loss 稳定下降 |
| 9 | RL-A 训练可行性 | 跑 100 步 | reward 上升趋势 |
| 10 | Action 分布 | 统计 sft_final 中 action 分布 | silent 55-70%, response 20-35%, recall 8-18% |

---

## 7. 训练计划

### 7.1 总览

V1 只做 4 个训练阶段，小规模快速验证：

```
SFT-A (协议对齐) → SFT-B (recall bootstrap) → RL-A (action/timing) → RL-B (recall utility, 可选)
```

### 7.2 SFT-A: 协议对齐

**目标**: 学会三动作格式 `<think>+<action>{silent|response|recall}`，学会正确的 timing。

| 参数 | 值 |
|------|-----|
| 数据量 | ~500-800 episodes |
| 配比 | ThinkStream 协议 50% + Streamo timing 对比 30% + 通用 VL 保能力 20% |
| Recall 比例 | 0-5%（此阶段几乎不教 recall，只教格式） |
| 基础模型 | Qwen2.5-VL-3B-Instruct |
| 学习率 | 1e-5 |
| Batch size | 8/卡 × 8 卡 = 64 |
| Max length | 32768 |
| Epochs | 3 |
| 保存间隔 | 每 100 步 |

**训练耗时**:
- ~700 样本 / 64 batch = ~11 steps/epoch
- 3 epochs × 11 steps × ~6s/step = **~3.5 分钟**
- 含 warmup 和 I/O: **~5 分钟**

**关键代码改动**:
- 替换 `SYSTEM_PROMPT` → `AGENT_SYSTEM_PROMPT`
- 新增 `_build_agent_messages()` 函数
- 更新 `_CHUNK_FORMAT_RE` 正则
- 注册数据集 `stream_agent_sft_a`

**验证指标**:
- 格式合规率: ≥ 95%（生成文本匹配新正则）
- Silent/Response 时机: 在验证集上 timing accuracy ≥ 80%

### 7.3 SFT-B: Recall Bootstrap

**目标**: 学会什么时候该 recall，query 怎么发，回来后怎么答。

| 参数 | 值 |
|------|-----|
| 数据量 | ~400-600 episodes |
| 配比 | recall-positive 35% + no-recall control 20% + false-recall negative 15% + ThinkStream forced-recall 10% + OCR/ASR history 10% + 通用 VL 10% |
| Recall 比例 | 12-18% |
| 基础模型 | SFT-A 的 checkpoint |
| 学习率 | 5e-6（降低，避免遗忘） |
| Epochs | 3 |

**训练耗时**:
- ~500 样本 / 64 batch = ~8 steps/epoch
- 3 epochs × 8 steps × ~8s/step (recall 样本序列更长) = **~3 分钟**
- 含 warmup: **~5 分钟**

**SFT Loss 加权**（可选，需修改 trainer）:

```python
# 不同 token span 的 loss 权重
loss_weights = {
    "action_tokens": 4.0,    # <action>silent|response|recall</action>
    "query_tokens": 1.5,     # <query>...</query>
    "response_tokens": 1.0,  # <response>...</response>
    "think_tokens": 0.2,     # <think>...</think>
}
```

**验证指标**:
- Recall precision: need_recall=true 样本中正确触发 recall ≥ 70%
- Recall specificity: need_recall=false 样本中不触发 recall ≥ 85%
- Query 格式: 生成的 query JSON 合法率 ≥ 90%

### 7.4 RL-A: Action/Timing 校准

**目标**: 优化 action 选择的精确性和响应时机。

| 参数 | 值 |
|------|-----|
| 数据量 | ~150-200 trajectories |
| 数据类型 | 可验证单轮任务 + delayed trigger |
| 基础模型 | SFT-B 的 checkpoint |
| 学习率 | 2e-7 |
| Group size | 8 |
| Micro batch size | 8 |
| 保存间隔 | 每 50 步 |

**奖励函数设计（6 维）**:

```python
def calc_agent_rewards(chunk_results, gen_idx, tokenizer, reward_spec):
    # 维度 1: 格式奖励 (保留)
    format_r = _compute_format_reward(chunk_texts)

    # 维度 2: 时间奖励 (保留, 修改 gt_chunk_idx 来源)
    time_r = _compute_time_reward(response_chunk_idx, gt_chunk_idx, window=5)

    # 维度 3: 正确性奖励 (保留)
    correct_r = _compute_correctness_reward(model_answer, gt_content)

    # 维度 4: Think 长度奖励 (修改 target)
    think_r = _compute_think_length_factor(avg_think_len,
                                           target_tokens=reward_spec["long_think_penalty_after_tokens"])

    # 维度 5: Action 正确性奖励 (新增)
    action_r = _compute_action_reward(
        predicted_actions, reward_spec["need_recall"],
        reward_spec["action_full_credit"],
        reward_spec["wrong_action_penalty"],
        reward_spec["over_recall_penalty"])

    # 维度 6: Response 效率 (修改)
    efficiency_r = think_r * num_response_factor

    # 最终奖励
    total_reward = (
        0.15 * format_r
        + 0.20 * time_r
        + 0.25 * correct_r
        + 0.10 * think_r
        + 0.20 * action_r
        + 0.10 * efficiency_r
    )
    return total_reward
```

**新增 `_compute_action_reward`**:

```python
def _compute_action_reward(predicted_actions, need_recall, full_credit, wrong_penalty, over_recall_penalty):
    """
    评估 action 选择的正确性
    predicted_actions: List[str] - 每个 chunk 的 action ('silent'/'response'/'recall')
    """
    reward = 0.0

    if need_recall:
        # 应该 recall 的样本
        if "recall" in predicted_actions:
            recall_idx = predicted_actions.index("recall")
            if "response" in predicted_actions[recall_idx+1:]:
                reward = 1.0  # recall then response = full credit
            else:
                reward = 0.5  # recall but no response
        elif "response" in predicted_actions:
            reward = 0.3  # 直接 response，部分分（如果答对的话）
        else:
            reward = 0.0  # 全 silent
    else:
        # 不应该 recall 的样本
        if "recall" in predicted_actions:
            reward = -over_recall_penalty  # 过度 recall 惩罚
        elif "response" in predicted_actions:
            reward = 1.0  # 正确 response
        else:
            reward = 0.0  # 全 silent（可能是该 silent 的样本）

    return max(0.0, reward)
```

**训练耗时**:
- ~200 trajectories × ~30s/step = ~6000s = **~100 分钟**
- 含 checkpoint saving: **~110 分钟**

### 7.5 RL-B: Recall Utility（可选）

**前提**: RL-A 完成后，如果 recall 行为已基本正确，可跳过此阶段。

| 参数 | 值 |
|------|-----|
| 数据量 | ~100-150 trajectories |
| 数据类型 | recall-positive + compare/causal recall |
| 基础模型 | RL-A checkpoint |
| 额外奖励 | hit@k + support coverage + over-recall 惩罚 |

**训练耗时**: **~50-75 分钟**

### 7.6 训练总时间表

| 阶段 | 耗时 | 累计 |
|------|------|------|
| SFT-A | ~5 min | 5 min |
| SFT-B | ~5 min | 10 min |
| RL-A | ~110 min | 120 min |
| RL-B | ~60 min (可选) | 180 min |
| **总计** | | **~2-3 小时** |

---

## 8. 质量保障体系

### 8.1 自动验收指标（7 项）

| # | 指标 | 计算公式 | 阈值 | 监控频率 |
|---|------|---------|------|---------|
| 1 | 格式合规率 | `count(match_regex) / count(all_chunks)` | ≥ 95% | 每 stage |
| 2 | Action 分布偏差 | `KL(actual_dist ‖ target_dist)` | ≤ 0.1 nats | 每 stage |
| 3 | Recall 精度 | `count(correct_recall) / count(need_recall=true)` | ≥ 70% | Stage 5 |
| 4 | 时序因果性 | `count(answer_not_depend_future) / count(all)` | = 100% | Stage 5 |
| 5 | 答案稳定性 | `count(stable_for_2_chunks) / count(all)` | ≥ 90% | Stage 5 |
| 6 | No-recall 失败率 | `count(no_recall_fail) / count(need_recall=true)` | ≥ 80% | Stage 5 |
| 7 | Counterfactual 灵敏度 | `mean(score_drop) where need_recall=true` | ≥ 0.5 | Stage 5 |

### 8.2 人工抽检规范

| 阶段 | 抽检对象 | 首批比例 | 扩量比例 | 重点检查 |
|------|---------|---------|---------|---------|
| Stage 0 | dense_caption, OCR | 20 条 (1.6%) | 100 条 (0.5%) | 描述准确性、OCR 完整性 |
| Stage 1 | event_timeline | 10 条视频全链 | 20 条 | 因果链是否合理 |
| Stage 2 | 397B 任务包 | 30 个任务 (6%) | 100 个 (1%) | 问题合理性、时间点准确 |
| Stage 5 | 闸门边界样本 | 全部 (~15 条) | 50 条 | 仲裁是否正确 |
| Stage 6 | sft_final 样本 | 50 条 (5%) | 200 条 (0.5%) | 格式、内容、多轮逻辑 |

### 8.3 数据血缘追踪

每条最终样本包含完整 provenance，便于回溯和调试：

```json
{
  "provenance": {
    "source_dataset": "streamo",
    "video_id": "vid_001",
    "original_annotation_id": "streamo_12345",
    "pipeline_version": "v0.1",
    "stage0_timestamp": "2026-04-16T10:00:00",
    "stage0_models": {
      "caption": "Qwen3-VL-8B",
      "ocr": "Qwen3-VL-8B",
      "asr": "Whisper-large-v3",
      "visual_emb": "SigLIP-SO400M",
      "text_emb": "BGE-M3"
    },
    "stage2_teacher_model": "Qwen3.5-397B-FP8",
    "stage2_task_id": "task_vid_001_007",
    "stage4_query_method": "direct_hit",
    "stage5_gate_results": {
      "gate1": true, "gate2": true, "gate3": 0.67,
      "gate4": true, "gate4_score": 0.15,
      "gate5": true, "gate5_score": 0.92,
      "gate6": true, "gate6_drop": 0.77
    },
    "stage5_arbitration": false,
    "created_at": "2026-04-16T11:13:00"
  }
}
```

### 8.4 异常检测与回退策略

| 异常 | 检测方式 | 回退策略 |
|------|---------|---------|
| Dense caption 幻觉 | 同一 segment 两个模型生成结果差异 > 阈值 | 送人工审核或丢弃 |
| 397B 拒绝输出 JSON | 解析失败 | 重试 1 次，仍失败则跳过该视频 |
| Recall query 全部无法命中 | hit@3 = 0 对所有 candidates | 降级为 non-recall 样本 |
| 6 闸门通过率 < 50% | 单视频统计 | 检查该视频的 segment_archive 质量 |
| Action 分布严重偏离 | KL > 0.3 | 调整 Stage 2 的 task_type 配额后重新生成 |

---

## 9. 代码改造清单

### 9.1 优先级 P0（首批构造前必须完成）

| 文件 | 改动 | 具体内容 |
|------|------|---------|
| `thinkstream/data/__init__.py` | 注册 4 个新数据集 | 添加 `stream_agent_sft_a`, `stream_agent_sft_b`, `stream_agent_rl_a`, `stream_agent_rl_b` |
| `thinkstream/data/stream_data_processor.py:431` | 新增 AGENT_SYSTEM_PROMPT | 添加中/英文 agent system prompt 常量 |
| `thinkstream/data/stream_data_processor.py` | 新增 `_build_agent_messages()` | 解析 sft_final.jsonl 格式，支持 recall 多轮交互 |
| `thinkstream/trainer/grpo.py:41-44` | 更新 `_CHUNK_FORMAT_RE` | 匹配三动作 + `<action>` 包裹 + `<query>` |
| `thinkstream/trainer/grpo.py:127` | 更新 `_RESPONSE_RE` | 从 `<response>(.*?)</response>` 提取 |
| 新文件 `scripts/agent_data_pipeline/stage0_preprocess.py` | Stage 0 脚本 | 场景切割、抽帧、caption、OCR、ASR、embedding |
| 新文件 `scripts/agent_data_pipeline/stage1_timeline.py` | Stage 1 脚本 | 事件聚合、因果链推断 |
| 新文件 `scripts/agent_data_pipeline/stage2_teacher.py` | Stage 2 脚本 | 397B 任务包生成 prompt + 调用 |
| 新文件 `scripts/agent_data_pipeline/stage3_expand.py` | Stage 3 脚本 | 稀疏→稠密 think 展开 |
| 新文件 `scripts/agent_data_pipeline/stage4_verify_query.py` | Stage 4 脚本 | Query 检索验证 + FAISS 索引 |
| 新文件 `scripts/agent_data_pipeline/stage5_gates.py` | Stage 5 脚本 | 6 闸门自动验证 |
| 新文件 `scripts/agent_data_pipeline/stage6_assemble.py` | Stage 6 脚本 | 样本组装 |

### 9.2 优先级 P1（首批训练前完成）

| 文件 | 改动 | 具体内容 |
|------|------|---------|
| `thinkstream/trainer/grpo.py` | 新增 `_compute_action_reward()` | Action 正确性奖励 |
| `thinkstream/trainer/grpo.py` | 修改 `calc_rewards()` | 整合 6 维奖励 |
| `thinkstream/model/inference.py:1211` | 扩展 `think_budget_sample_restricted` | 支持 recall action token |
| `scripts/sft_agent.sh` | 新增 agent SFT 训练脚本 | 基于 sft.sh 修改 |
| `scripts/rl_agent.sh` | 新增 agent RL 训练脚本 | 基于 rl.sh 修改 |

### 9.3 优先级 P2（推理部署时完成）

| 文件 | 改动 | 具体内容 |
|------|------|---------|
| `thinkstream/model/inference.py` | `streaming_video_chat()` 支持 recall | 检测 recall action → 调用检索 → 注入结果 |
| 新文件 `thinkstream/retrieval/` | 检索模块 | FAISS 索引 + BGE encode + rerank |
| `scripts/demo.py` | 更新 demo | 支持 recall 交互展示 |

---

## 附录 A: 目录结构规范

```
data/agent/
├── segment_archive/        # Stage 0 输出
│   ├── vid_001.jsonl
│   └── vid_002.jsonl
├── keyframes/              # Stage 0 输出
│   ├── vid_001/
│   │   ├── frame_040000.jpg
│   │   └── frame_042000.jpg
│   └── vid_002/
├── embeddings/             # Stage 0 输出
│   ├── vid_001/
│   │   ├── visual_040000.npy
│   │   └── text_040000.npy
│   └── vid_002/
├── event_timeline/         # Stage 1 输出
│   ├── vid_001.jsonl
│   └── vid_002.jsonl
├── episode_recall_raw.jsonl      # Stage 2 输出
├── episode_recall_dense.jsonl    # Stage 3 输出
├── episode_recall_verified.jsonl # Stage 4 输出
├── episode_recall.jsonl          # Stage 5 输出 (最终版)
├── sft_final/              # Stage 6 输出
│   ├── sft_a_v0.1.jsonl
│   └── sft_b_v0.1.jsonl
├── retriever_train.jsonl   # Stage 6 输出
├── rl_pool.jsonl           # Stage 6 输出
├── clips/                  # Stage 6 裁剪的 recent buffer clips
│   ├── vid_001_54_78.mp4
│   └── ...
└── meta/                   # Stage 0 中间元数据
    ├── vid_001_scenes.json
    └── vid_001_segments.json
```

## 附录 B: 关键设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| V1 固定 2s chunk | 固定 | 简化实现，自适应留 V2 |
| V1 保持 FRAMES_PER_CHUNK=2 | 保持 | 与现有代码兼容 |
| Think 结构化仅用于数据管理 | 是 | Student 只生成自由文本 think |
| Recall 在 SFT 中预展开为多轮 | 是 | 避免在线检索的复杂性 |
| 首批 50 视频 | 50 | 端到端验证 ~73 分钟可完成 |
| RL 用离线预计算 recall 结果 | 是 | V1 不做在线检索 |
| Loss 加权可选 | 可选 | 首批先用均匀 loss 验证，效果不好再加权 |
| 数据审计先行 | 是 | 两个数据集来源重叠，必须先去重再构造 |

## 附录 C: 源数据集阅读说明

### C.1 Streamo-Instruct-465K 核心要点

**论文**: "Streaming Video Instruction Tuning", Xia et al., CVPR 2026, arXiv:2512.21334  
**代码**: `ThinkStream/Streamo/` (基于 ms-swift)  
**数据**: `maifoundations/Streamo-Instruct-465K` (HuggingFace, gated)

#### 核心设计

1. **端到端 streaming 训练**: 不使用辅助模型做 clip 分割，直接在视频流上训练。每秒 1 帧，每帧一个 turn。
2. **三状态协议**: `</Silence>` (不说话) / `</Standby>` (事件进行中) / `</Response>` (事件完成，回答)
3. **Focal Loss**: 解决 Silence 占比过高 (>80%) 导致的类别不平衡。使用 focal weighting + frequency-based alpha。
4. **多任务标注**: 同一视频标注 6 种不同任务，每种任务有独立的时间边界。

#### 6 种任务详解

| 任务 | 构造方式 | 时间信息质量 | Agent 化价值 |
|------|---------|------------|------------|
| **Real-time Narration** (12.7%) | Qwen2.5-VL-72B 逐秒描述 → GLM-4.5 后处理 | 每秒一条，密集 | 中：可做 continuous output 样本 |
| **Event Caption** (6.7%) | ARC-Hunyuan-Video-7B 分段 caption → 时间对齐 → 过滤 | 有精确事件边界 | 高：有 event boundary，可做 timing |
| **Action Caption** (5.8%) | 复用 event caption pipeline + 动作导向 prompt | 有精确动作边界 | 高：可做 procedure step |
| **Event Grounding** (26.3%) | 从 event caption 采样 → 改写为 grounding 格式 | 有 event span + caption | 高：最适合派生 recall |
| **Time-sensitive QA** (34.8%) | GLM-4.5V 检测变化点 → 生成 QA | 有变化点时间 | 高：最适合 timing 校准 |
| **Offline QA** (13.8%) | 传统 video QA | 无精确时间 | 低：只能做弱锚点 |

#### 读取数据的关键注意事项

1. **video_path 中的来源信息**: `LLaVA_Video/` 前缀表示来自 LLaVA-Video 数据集，可从路径推断原始来源
2. **同一视频多任务**: 同一 `video_name` 会出现多次，每次对应不同 task_type
3. **response.time vs st_time/end_time**: 两种时间标注机制，`time` 用于即时回答，`st_time/end_time` 用于延迟回答
4. **1fps 格式**: 每个 `<stream>` token 对应 1 秒 1 帧。Agent 化时需注意 ThinkStream 使用 2fps 的差异

### C.2 ThinkStream 核心要点

**论文**: "Thinking in Streaming Video", Liu et al., arXiv:2603.12938  
**代码**: `ThinkStream/thinkstream/` (自研框架)  
**数据**: `CASIA-IVA-Lab/ThinkStream` (HuggingFace, MIT)

#### 核心设计

1. **Watch-Think-Speak 范式**: 每个 video chunk 触发一次 think 更新，决定 speak 还是 silent
2. **RCSM (Reasoning-Compressed Streaming Memory)**: 用 think token 作为语义压缩，替代被淘汰的视频 token。关键参数: visual window=20 chunks
3. **Streaming RLVR**: GRPO 优化，三维奖励 (format + time + accuracy)
4. **2fps + 动态分辨率**: 与 Streamo 的 1fps 不同

#### 数据构造 Pipeline

```
视频 → PySceneDetect 分段 → Qwen3-VL-235B dense caption
  → Cartesian(interaction_mode × temporal_scope × content_dimension)
  → 过滤无效组合 → 39 个 scenario
  → 合成 QA pairs → 合成 time-grounded CoT
  → Cold Start 110K + RLVR 9K
```

#### 关键数据特征

1. **thoughts 字段极有价值**: 每个 thought 有精确 timestamp，内容是增量式推理（不是整段总结）。这直接对应 agent 的 `<think>` 输出。
2. **三维标签完整**: interaction_mode / temporal_scope / content_dimension 三维标签使得数据可以精确分桶。
3. **response_format 多样**: 包含 MC/Binary/Counting/Open-ended，前三种直接可用于 RL 可验证奖励。
4. **视频路径可解析**: 路径如 `Kinetics-700/FSPKtVyji6s_000074_000084.mp4` 可提取 YouTube ID + clip 时间，用于去重。

#### Ablation 结论对 Agent 设计的启示

| 论文结论 | 对 Agent 设计的意义 |
|---------|-------------------|
| Think token budget 最优 = 20 tok/s | Agent 的 think 预算应与此对齐（2s chunk → 40 tokens） |
| Visual window 最优 = 20s | 对应 recent_window_sec = 20-24s |
| Cold-start CoT >> Discrete caption as memory | 说明 think 作为语义压缩远优于纯 caption |
| RLVR 比纯 SFT 提升 4.3 分 | 确认 RL 阶段必要性 |

### C.3 两个数据集的对比总结

| 维度 | Streamo-Instruct-465K | ThinkStream |
|------|----------------------|-------------|
| 规模 | 465.8K 样本 / 135.8K 视频 | 119K 样本 |
| 帧率 | 1fps | 2fps |
| 动作协议 | `</Silence>/<Standby>/<Response>` | `<think>...<response>/<silent>` |
| Think 支持 | **无** (无 think/reasoning 过程) | **有** (thoughts 字段) |
| 时间信息 | question.time + response.st_time/end_time | conversations.timestamp + thoughts.timestamp |
| 任务类型 | 6 种（Narration/Caption/Grounding/QA） | 39 种组合（3×3×7 过滤） |
| 标注模型 | Qwen2.5-VL-72B + GLM-4.5V | Qwen3-VL-235B |
| 视频来源 | LLaVA-Video + ActivityNet + Kinetics + ShareGPT4Video | LLaVA-Video-178K + Tarsier2 (Kinetics-700/Charades/ActivityNet) |
| **Agent 化主要价值** | 时间锚点库 + timing 样本 + recall 派生 | 协议 warm-start + think 内容 + 结构化标签 |
| **主要局限** | 无 think 过程、Silence 占比极高 | 短视频为主、无 recall 机制 |

### C.4 数据集获取与本地部署

#### 下载 ThinkStream 数据集

```bash
# 方式 1: HuggingFace CLI
huggingface-cli download CASIA-IVA-Lab/ThinkStream --local-dir data/thinkstream_raw

# 方式 2: Python
from datasets import load_dataset
ds = load_dataset("CASIA-IVA-Lab/ThinkStream")
ds.to_parquet("data/thinkstream_raw/thinkstream.parquet")
```

#### 下载 Streamo-Instruct-465K 数据集

```bash
# 需要先在 HuggingFace 页面申请访问权限
huggingface-cli download maifoundations/Streamo-Instruct-465K --local-dir data/streamo_raw

# 注意: gated access，需要先登录
huggingface-cli login
```

#### 下载视频文件

```bash
# ThinkStream 视频来源
# 1. LLaVA-Video-178K videos
# 2. Tarsier2 videos (Kinetics-700 + Charades + ActivityNet 子集)
# 具体路径参考 ThinkStream README: "Prepare the video sources"

# Streamo 视频来源
# 参考 Streamo raw_data.json 中的 video_path 字段
# 大部分视频路径以 LLaVA_Video/ 开头
```
