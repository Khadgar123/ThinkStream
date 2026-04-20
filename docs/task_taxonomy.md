# 流视频 Agent 任务分类体系

> 版本: v0.2
> 日期: 2026-04-19
> 目的: 指导首批造数据——定义每个 action 类型下有哪些子任务、边界情况、prompt 模板、配比、区分度，以及视频类型→任务映射

---

## 1. 按 action 类型的大分类

模型在每个 chunk 只能输出三种 action 之一。但每种 action 下有多种**触发原因**，需要分别造数据：

### 1.1 Silent（58-65% of chunks）

| 子任务 ID | 名称 | 触发条件 | think 内容 | 区分度样本 |
|-----------|------|---------|-----------|-----------|
| S1 | 无问题观察 | 没有用户问题，场景正常 | 描述当前观察到的增量变化 | 不需要 |
| S2 | 问题已答完 | 之前已经回答了用户问题 | "问题已回答，继续观察" | 和 S4 对比 |
| S3 | Standby 等待 | 用户问了但事件还没发生 | "用户问了X，事件还没发生，继续等待" | 和 R1 对比（时机差 1 chunk） |
| S4 | Trigger 监控中 | 用户设置了触发条件，条件未满足 | "持续监控，目标条件尚未出现" | 和 R4 对比（条件满足时） |
| S5 | Recall 后沉默 | recall 回来了但证据不够，决定不答 | "检索结果不确定，暂不回答" | 和 RC1 区分 |

**S3 的边界情况最重要**：问题出来了但答案还没出现。模型必须学会等，不能看到问题就立刻答。
```
边界样本对:
  chunk t=30s (问题刚出来): → S3 silent "事件还没发生，等待"
  chunk t=35s (事件发生了): → R2 response "事件发生了，现在可以回答"
```

### 1.2 Response（23-30% of chunks）

| 子任务 ID | 名称 | 触发条件 | think 内容 | 区分度样本 |
|-----------|------|---------|-----------|-----------|
| R1 | 即时回答 | 问题出来，证据就在当前帧 | "当前画面可以看到X，直接回答" | 和 S3 对比 |
| R2 | 延迟回答 | Standby 结束，事件发生了 | "等待的事件发生了，现在回答" | 和 S3 对比（前一 chunk） |
| R3 | 多次回答中的一次 | 持续性问题，事件在展开 | "新的进展出现了，补充回答" | 和 R5 对比 |
| R4 | Trigger 触发 | 监控条件满足 | "目标条件满足了，触发回答" | 和 S4 对比 |
| R5 | 连续解说 | 用户要求实时解说 | 描述当前正在发生的事 | 和 S1 对比 |
| R6 | Recall 后回答 | recall 结果足够，现在可以答 | "检索命中，结合历史信息回答" | 必须在 RC 之后 |
| R7 | 不需要 recall 的回忆类问题 | 问"之前"但 text memory 够 | "虽然问的是之前，但 think 里记着" | 和 RC1 对比（假 recall 负样本） |

**R1 vs S3 的边界**是最关键的 timing 训练：
```
视频: 厨师正在准备材料，用户问 "什么时候开始炒？"
  chunk t=28s: S3 silent "还在切菜，还没开始炒"
  chunk t=30s: S3 silent "还在准备"
  chunk t=32s: R2 response "现在开始炒了"  ← 事件刚发生
```

**R7 是核心的 false-recall negative**：
```
用户: "之前放了什么调料？" (听起来需要回忆)
think: "虽然说了'之前'，但 3 秒前刚看到放了盐，还在 recent window 里"
action: response (不是 recall)
```

### 1.3 Recall（10-15% of chunks）

| 子任务 ID | 名称 | 触发条件 | think 内容 | query 特征 | 区分度样本 |
|-----------|------|---------|-----------|-----------|-----------|
| RC1 | 视觉细节回忆 | 问颜色/形状/外观等 text memory 记不住的 | "当前窗口看不到，需要检索视觉细节" | 实体+属性 | 和 R7 对比 |
| RC2 | 精确数值回忆 | 问价格/数量/尺寸等精确数字 | "具体数字需要回看原始画面" | OCR+数字 | 和 R1 对比（当前有数字时） |
| RC3 | 程序步骤回忆 | 问之前做到第几步/顺序 | "流程步骤的具体顺序需要回看" | 动作+步骤 | 和 R7 对比（简单步骤） |
| RC4 | 跨时间比较 | 问"和之前比有什么变化" | "需要回看之前的状态来比较" | 实体+状态 | |
| RC5 | 长程因果回忆 | 问"为什么会这样"，原因在很早以前 | "原因发生在很久之前" | 事件+因果 | |
| RC6 | 实体再识别 | 问"之前出现的那个人/物" | "需要确认是否是同一个实体" | 实体+外观 | |
| RC7 | 多轮 follow-up 回忆 | 追问之前对话中提到的细节 | "用户追问之前提到的内容" | 对话引用+细节 | |

**RC1 vs R7 是 recall 能力的核心边界**：
```
同一个问题 "之前用了什么调料？"

情况 A (R7, 不需要 recall):
  support 在 10s 前 → 还在 recent window
  text memory: "刚看到放了辣椒酱"
  → 直接 response

情况 B (RC1, 需要 recall):
  support 在 60s 前 → 早已离开 window
  text memory: "之前用了一些调料" (细节丢失)
  → recall → 看到 keyframe 里是辣椒酱 → response
  
情况 C (R7, false-recall negative):
  support 在 5s 前 → 就在眼前
  用户说了"之前" → 听起来像 recall
  → 直接 response (不要 recall)
```

---

## 2. 任务配比

### 2.1 按 action 分（chunk 级）

| Action | 占比 | 说明 |
|--------|------|------|
| silent | 58-65% | 大部分时间不说话 |
| response | 23-30% | 有足够证据时回答 |
| recall | 10-15% | 证据不在窗口内时检索 |

### 2.2 按子任务分（episode 级）

| 子任务 | 目标占比 | 最低数量/首批 | 优先级 |
|--------|---------|-------------|-------|
| **Silent 类** | | | |
| S1 无问题观察 | 自然产生 | - | 包含在所有样本中 |
| S2 问题已答完 | 自然产生 | - | 包含在所有样本中 |
| S3 Standby 等待 | 12% | 200 | P0 |
| S4 Trigger 监控中 | 5% | 80 | P1 |
| **Response 类** | | | |
| R1 即时回答 | 16% | 300 | P0 |
| R2 延迟回答 | 12% | 200 | P0 |
| R3 多次回答 | 5% | 80 | P1 |
| R4 Trigger 触发 | 5% | 80 | P1 |
| R5 连续解说 | 3% | 50 | P2 |
| R6 Recall 后回答 | 与 RC 配对 | - | 自动产生 |
| R7 假 recall 负样本 | 与 RC 配对 | - | 自动产生 |
| **Recall 类** | | | |
| RC1 视觉细节回忆 | 10% | 150 | P0 |
| RC2 精确数值回忆 | 5% | 80 | P0 |
| RC3 程序步骤回忆 | 5% | 80 | P0 |
| RC4 跨时间比较 | 4% | 60 | P1 |
| RC5 长程因果回忆 | 3% | 50 | P1 |
| RC6 实体再识别 | 3% | 50 | P1 |
| RC7 多轮 follow-up | 3% | 50 | P2 |

### 2.3 三合一绑定规则

**每条 recall 样本 (RC1-RC7) 必须配套生成：**
- 1 条 no-recall control (R1/R2: 同问题在证据可见时提问)
- 1 条 false-recall negative (R7: 同问题但当前窗口就有答案)

所以 recall 500 条 → 实际产出 1500 条（500 recall + 500 control + 500 false-negative）

### 2.4 难度分布

| 难度 | 占比 | 定义 |
|------|------|------|
| Easy | 25% | 无 recall，证据就在当前帧 (R1, S1) |
| Medium | 35% | 需要 think，但不需要 recall (R2, S3, R4) |
| Hard | 30% | 需要 recall (RC1-RC5) |
| Very Hard | 10% | 长程因果 / 多轮 / 多跳 (RC5, RC6, RC7) |

---

## 3. 视频类型 → 子任务映射

不同视频适合不同子任务。选视频时按类型分桶：

| 视频类型 | 时长要求 | 适合的子任务 | 从 Streamo 取 |
|---------|---------|------------|-------------|
| **教程/烹饪/装配** | >120s | RC3 程序步骤, RC1 视觉细节, RC5 因果, S3 等待 | duration>120s, task=QA+EventGrounding |
| **Vlog/长镜头** | >90s | RC4 比较, RC6 实体再识别, R2 延迟, R3 多次 | duration>90s, task=Narration |
| **屏幕录制/UI** | >60s | RC2 数值, OCR 回忆, R1 即时 | 有 OCR 的视频 |
| **剧情/综艺** | >120s | RC5 因果, RC7 多轮, RC6 实体 | duration>120s, task=EventGrounding |
| **体育/户外** | >90s | R4 Trigger, R5 连续解说, RC4 比较 | duration>90s |
| **短视频 (<30s)** | any | R1 即时, S1 观察 (protocol only, 不造 recall) | ThinkStream 短视频 |

**首批视频选取策略：**
```
教程/烹饪 >120s:  15 个 → RC3, RC1, RC5
Vlog >90s:        10 个 → RC4, RC6, R2
屏幕录制 >60s:    10 个 → RC2
剧情 >120s:       5 个  → RC5, RC7
ThinkStream 短:   20 个 → R1, S1 (protocol)
────────────────────────
总计: 60 个
```

---

## 4. 每种子任务的 Prompt 模板

### 4.1 给 397B 的任务生成 Prompt

**核心 prompt 结构**（给 397B 看完视频帧后用）：

```
你是流视频 agent 训练数据构造器。

你刚看完一个 {video_type} 类型的 {duration}s 视频。
请为这个视频生成训练任务。

要求：
1. 每个任务必须指定 task_type, ask_time_sec, answer_time_sec, need_recall
2. 对于 need_recall=true 的任务：
   - 问题必须考察 TEXT MEMORY 无法保留的视觉细节（颜色、数字、空间关系、具体外观）
   - ask_time 时，证据必须已经离开 recent window (>24s 前)
   - 提供 support_time_sec (证据出现的时间)
   - 提供 3 个 recall query 候选
3. 对于 need_recall=false 的任务：
   - 证据在 ask_time 时必须在 recent window 内
4. 必须包含以下任务类型（按指定数量）：
   {task_requirements}
5. 每个任务输出 canonical_answer (可验证) 和 natural_response (自然语言)
```

### 4.2 每种子任务的具体 prompt 补充

**RC1 视觉细节回忆**：
```
生成一个关于视觉细节的 recall 问题。
问题必须考察只有看到原始帧才能回答的细节：
  - 物体的颜色、形状、大小
  - 人的穿着、姿态
  - 空间位置关系
  - 背景中的特定物品
问题不能是"发生了什么"（text memory 能答），
必须是"具体长什么样"（必须看帧）。
```

**RC2 精确数值回忆**：
```
生成一个关于精确数字/文字的 recall 问题。
问题必须考察 OCR 级别的精确信息：
  - 价格标签上的数字
  - 屏幕上显示的文字
  - 仪表盘读数
  - 时间显示
这些信息 text memory 只会记"有个数字"，不会记精确值。
```

**RC3 程序步骤回忆**：
```
生成一个关于操作步骤顺序的 recall 问题。
问题考察之前的步骤细节：
  - 第几步做了什么
  - 某个步骤的具体操作方式
  - 步骤的先后顺序
  - 某个步骤是否已经完成
```

**S3 Standby 等待**：
```
生成一个需要等待才能回答的问题。
用户在事件发生前提问，模型需要：
  - 在问题出来的 chunk: silent (standby)
  - 在事件发生的 chunk: response
需要指定 ask_time 和 answer_time，两者之间至少差 4s。
```

**R4 Trigger 触发**：
```
生成一个监控触发任务。
用户设定一个条件："看到XX就告诉我"
模型需要：
  - 在条件设定后的若干 chunk: silent (监控中)
  - 在条件满足的 chunk: response (触发)
```

---

## 5. 数据过滤规则

### 5.1 自动过滤（hard rules）

| 规则 | 条件 | 处理 |
|------|------|------|
| 答案因果性 | canonical_answer 依赖 ask_time 之后的帧 | 丢弃 |
| Recall 窗口检查 | need_recall=true 但 support 在 recent window 内 | 丢弃 |
| 答案可验证性 | canonical_answer 为空或无法 parse | 丢弃 |
| 问题合理性 | question 为空或太短 (<5 字) | 丢弃 |
| Think 一致性 | think 内容与 action 矛盾 | 丢弃 |

### 5.2 质量过滤（soft rules, 用模型验证）

| 规则 | 方法 | 阈值 |
|------|------|------|
| No-recall 失败 | 只给 recent window，小模型尝试答 | 答对率 < 0.4 才是有效 recall |
| With-recall 通过 | 给 recall 结果后答 | 答对率 ≥ 0.8 |
| 区分度 | control 样本不 recall 也能答对 | 答对率 ≥ 0.8 |
| False-negative 区分 | 假 recall 样本不 recall 也能答对 | 答对率 ≥ 0.8 |

### 5.3 分布过滤

| 规则 | 条件 | 处理 |
|------|------|------|
| 同视频过采样 | 同一视频 >15 个任务 | 随机丢弃到 15 |
| 任务类型偏斜 | 某子任务占比 > 目标 ×2 | 下采样 |
| 难度偏斜 | Easy/Hard 偏离目标 >15% | 调整采样 |

---

## 6. 持久化资产表设计

### 6.1 为什么要资产表

造数据最贵的是 397B 调用。一次生成后应该把**中间结果全部存下来**，后续只需：
- 改过滤规则 → 重新过滤（不用重新生成）
- 改配比 → 重新采样（不用重新生成）
- 改格式 → 重新组装（不用重新生成）
- 加新视频 → 只生成新视频的（增量）

### 6.2 核心表结构

#### 表 1: `video_registry.jsonl`（视频级，一次性）

```json
{
  "video_id": "streamo_ytb_xxx",
  "video_path": "/path/to/video.mp4",
  "source_dataset": "streamo",
  "duration_sec": 135.0,
  "video_type": "tutorial",
  "recallability_score": 0.78,
  "num_tasks_generated": 12,
  "generation_timestamp": "2026-04-20T10:00:00",
  "generation_model": "Qwen3.5-397B-A17B-FP8"
}
```

#### 表 2: `task_pool.jsonl`（任务级，397B 生成的原始输出）

```json
{
  "task_id": "task_001",
  "video_id": "streamo_ytb_xxx",
  "task_type": "RC1_visual_detail",
  "sub_type": "color_recall",
  "question": "前面那个调料瓶是什么颜色的？",
  "ask_time_sec": 90.0,
  "answer_time_sec": 90.0,
  "support_time_sec": 20.0,
  "need_recall": true,
  "canonical_answer": {"answer_type": "entity", "value": {"color": "红色"}},
  "natural_response": "是红色包装的调料瓶。",
  "query_candidates": [
    {"query": "调料瓶 颜色 包装", "time_bias": "past_far", "target": "entity"},
    {"query": "红色 瓶子 厨房", "time_bias": "past_far", "target": "entity"},
    {"query": "调料 容器 外观", "time_bias": "past_far", "target": "entity"}
  ],
  "think_at_ask": "用户问调料瓶颜色，这是很久之前出现的，当前窗口内看不到",
  "think_after_recall": "检索到 t=20s 的帧，红色包装",
  "difficulty": "hard",
  "generation_model": "Qwen3.5-397B-A17B-FP8",
  "generation_timestamp": "2026-04-20T10:05:00"
}
```

#### 表 3: `task_triplets.jsonl`（三合一绑定，从 task_pool 派生）

```json
{
  "triplet_id": "tri_001",
  "recall_positive_task_id": "task_001",
  "no_recall_control": {
    "task_id": "task_001_ctrl",
    "ask_time_sec": 25.0,
    "think": "调料瓶刚出现在画面里，直接回答"
  },
  "false_recall_negative": {
    "task_id": "task_001_fn",
    "question": "之前那个调料瓶是什么颜色的？",
    "ask_time_sec": 25.0,
    "think": "虽然说'之前'，但调料瓶就在当前窗口里"
  }
}
```

#### 表 4: `verification_results.jsonl`（质量验证结果）

```json
{
  "task_id": "task_001",
  "no_recall_student_score": 0.12,
  "with_recall_student_score": 0.95,
  "control_student_score": 0.90,
  "false_negative_student_score": 0.88,
  "all_passed": true,
  "verification_model": "Qwen3.5-35B-A3B",
  "verification_timestamp": "2026-04-20T11:00:00"
}
```

#### 表 5: `sft_episodes.jsonl`（最终 SFT 训练数据，从表 2-4 组装）

已有的 Stage 6 输出格式，加上 `protocol_version: "3action"`。

### 6.3 数据复用流程

```
首次造数据:
  视频 → 397B → task_pool (存盘)
                → task_triplets (存盘)
                → verification_results (存盘)
                → sft_episodes (存盘)

改配比/改格式:
  task_pool (已有) → 重新采样/过滤 → 新的 sft_episodes
  不需要重新调 397B

加新视频:
  新视频 → 397B → 追加到 task_pool
               → 合并旧数据 → 重新采样 → sft_episodes

改验证规则:
  task_pool (已有) → 重新验证 → 新的 verification_results → 过滤 → sft_episodes
```

---

## 7. 首批造数据计划

### 7.1 数量目标

| 数据 | 数量 | 来源 |
|------|------|------|
| Recall 三合一 (RC1-RC7) | 500 recall + 500 control + 500 false-neg = **1500** | 397B 看长视频生成 |
| Response 非 recall (R1-R5) | **800** | 397B 生成 + ThinkStream 转换 |
| Silent + Standby (S3, S4) | **300** (episode 级) | 397B 生成 |
| Protocol warmstart | **2000** | ThinkStream 直接转格式 |
| **SFT 总计** | **~4600** | |
| RL 可验证 | **~600** | 从上面选 MC/Binary/Number |

### 7.2 需要的视频数量

- Recall 500 条 → 需要 ~50 个长视频 (>120s)，每个生成 ~10 recall 任务
- Response/Silent → 同一批视频的 non-recall 任务
- Protocol → ThinkStream 现有数据

### 7.3 397B 调用量估算

| 步骤 | 调用次数 | 输入 token | 输出 token |
|------|---------|-----------|-----------|
| 看视频生成任务 | 50 | ~60K vision + 3K text | ~5K |
| 总计 | 50 | | |

vLLM 并发 40 → **~5 分钟**完成全部生成。

验证阶段（可选 35B）：
| 步骤 | 调用次数 |
|------|---------|
| No-recall 验证 | 500 |
| With-recall 验证 | 500 |
| Control 验证 | 500 |
| False-neg 验证 | 500 |
| 总计 | 2000 |

vLLM 并发 → **~3 分钟**。
