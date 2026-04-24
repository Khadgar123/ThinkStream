# 流视频 Agent 数据构造方案

> 版本: v9.0 | 日期: 2026-04-24
>
> 核心设计：
> - Per-timestep 独立样本，每步重新构造完整 input，不依赖跨步 KV cache
> - `<think>` 每步立即入文本记忆（recall step 2 的分析 think 除外），文本记忆覆盖时间 > 视觉滑动窗口
> - 输入布局：**视频在前、文本在后**（`<visual_window>` → `<memory>` → `<user_input>`）
> - 文本 think 的 MROPE temporal position 对齐到对应视频帧的时间戳
> - 80% token 预算触发系统压缩，C1 系统指定范围 → C2 模型自选范围
> - 压缩时模型可同时利用 thinks 文本和当前可见的视觉帧生成更精确的 summary
> - Recall 拆为两步：step1 输出 query → 系统检索 → step2 输出 response/silent
> - 所有 action 输出统一为 `<think>...</think><action>...</action>` 格式（含 recall_response）
> - 3 个专用 Prompt：主循环(silent/response/recall) + post-recall(silent/response) + compress(summary)
>
> 配套文档：
> - **`sft_engineering.md`**：SFT 训练工程设计 v3.0
> - **`streaming_position_encoding.md`**：位置编码与 KV Cache 分区设计 v1.0

---

## 1. 架构原则

### 1.1 训练格式：Per-Timestep Re-render（v6.1 定稿）

**每个 2s chunk 构造为独立训练样本**（单轮 messages）。推理时每步重新构造完整 input，不依赖跨步 KV cache。

```
单条训练样本 = 推理时一步的精确快照 {
    messages: [
        {role: system, content: 4-action protocol},
        {role: user, content: [
            {type: video, video_start: 26, video_end: 50, nframes: 24},  // 视频在前 (Zone B)
            {type: text, text: "<memory>...<compressed>...</compressed>\n[40-42] think...</memory>"},  // 文本在后 (Zone C)
            {type: text, text: "<user_input>Question?</user_input>"},  // 用户输入 (Zone D)
        ]},
        {role: assistant, content: "<think>...</think><action>response</action><response>...</response>"}
    ]
}
```

只对 assistant content 计算 loss。Memory block 作为 user text input 的一部分。

**为什么用 per-timestep 而不是多轮对话**（v6.1 关键决策）：

1. **KV surgery 不可行**：多轮中压缩后旧 thinks 的 KV 仍在 cache，替换需要重算位置编码和后续所有 KV
2. **Mask≠删除**：attention mask=0 只遮蔽不释放，模型无动力写好 summary（直接看旧 thinks 更详细）
3. **KV 无限增长**：1小时视频 → ~1M tokens KV → 57GB，无法部署
4. **Re-render 足够快**：H20 ~580ms/step, H100 ~200ms/step（ViT帧缓存后），远在 2s 实时预算内
5. **压缩天然生效**：旧 thinks 不在 input 中，summary 是唯一信息源。模型被迫学会写好 summary

详见 `sft_engineering.md` §0（推理状态机规范）。

### 1.2 三层文本严格分离

```
A. teacher_caption (隐藏，不进 SFT):
   397B 的详细结构化事实，用于造题、验证、生成标准答案。
   学生模型永远看不到这层。

B. student_think (进 SFT 的 output):
   学生模型每个 timestep 生成的 40-60 token 增量视觉记忆。
   立即存入 memory，可被压缩。

C. compressed_summary (进 SFT):
   学生模型生成的压缩摘要，替换旧 thinks。
```

**三者不能混**：
- teacher_caption 不能出现在 student 的 input 或 output 中
- recall_result 不能包含 teacher_caption 的信息（除非学生 memory 中本来就有）
- 标准答案来自 teacher_caption，但 SFT output 不能照搬（需要基于 student 可见信息回答）

### 1.3 三 Prompt 协议

> v8.0: 从"4 action 同一 prompt"改为"3 个专用 prompt，每个 prompt 教一种行为"。
> 压缩不再是主循环的 action 之一，而是系统触发的独立步骤。

**3 个 Prompt，各负责一种决策：**

#### Prompt 1: SYSTEM_PROMPT — 主循环（每个时间步）

```
可选 action: silent / response / recall
输出格式:
  1) <think>...</think><action>silent</action>
  2) <think>...</think><action>response</action><response>...</response>
  3) <think>...</think><action>recall</action><query>{"query":"...","time_range":"..."}</query>
```

模型学什么：看视频 → 该观察、回答、还是检索？
think = 当前帧视觉观察 → **存入 memory**。

#### Prompt 2: SYSTEM_PROMPT_POST_RECALL — recall 返回后

```
可选 action: silent / response
输出格式:
  1) <think>...</think><action>silent</action>       (检索结果没用)
  2) <think>...</think><action>response</action><response>...</response>  (回答)
```

模型学什么：检索结果有用吗？→ 回答或沉默。
think = 检索结果分析 → **不存 memory**（非视觉观察）。

#### Prompt 3: SYSTEM_PROMPT_COMPRESS — 系统触发压缩（两步之间）

```
输出格式:
  C1: <summary>{"time_range":[s,e],"text":"..."}</summary>     (系统指定范围)
  C2: <summary>{"time_range":[s,e],"text":"..."}</summary>     (模型自选范围)
```

模型学什么：哪些 thinks 可以压缩？怎么写摘要？
**无 think** — 压缩不是视觉观察，不产生 memory 条目。
**不占用时间步** — 在两个时间步之间触发，不影响主循环。

#### 完整时序示例

```
chunk 21: [SYSTEM_PROMPT, 3-action]
  → <think>Chef covers pot</think><action>silent</action>
  → memory.add_think() → token count > 80%

──── 系统触发压缩（BETWEEN chunk 21 and 22）────
  [SYSTEM_PROMPT_COMPRESS]
  input: memory_timeline + visual_window
  C1: 系统指定 range="16-30" → 模型输出 summary
  C2: 模型自选 range → 模型输出 range + summary
  → memory.compress(summary, range)
────────────────────────────────────────────

chunk 22: [SYSTEM_PROMPT, 3-action]
  → <think>Chef walks to fridge</think><action>response</action><response>...</response>
  → memory(已压缩) 正常继续

──── 如果有 recall ────
chunk 25: [SYSTEM_PROMPT, 3-action]
  → <think>观察</think><action>recall</action><query>...</query>
  → memory.add_think()

  [SYSTEM_PROMPT_POST_RECALL, 2-action]
  input: memory + recall_result + recall_context
  → <think>分析结果</think><action>response</action><response>...</response>
  → think 不存 memory
────────────────────────────────────────────
```

**每个时间步只有一条视觉观察 think 进入 memory。压缩和 recall-response 都不写 memory。**

**Recalled frames 与视觉窗口分离**：recall 返回的历史帧追加在 visual_window 之后、memory 之前，有时间戳和 source 标注，防止模型混淆为当前画面。下一时间步不再出现。

### 1.4 Think 规范

| 参数 | 值 |
|------|------|
| 长度 | **40-60 tokens** |
| 内容 | 当前 2s 帧中的增量可观察事实 |
| 禁止 | 声音/气味/情绪/意图/元推理（如 "I need to recall..."） |
| 允许 | 实体+属性、动作、状态变化、OCR、空间关系 |

**正确**: `Chef places 4 Roma tomatoes on wooden board, deep red, medium sized.`
**错误**: `I notice the chef is preparing tomatoes. The sizzling sound suggests the oil is hot.`

### 1.5 Response 长度按问题类型

| 问题类型 | Response 长度 | 示例 |
|---------|-------------|------|
| Factoid (实体/颜色/数量) | 5-40 tokens | "Red apron." |
| Procedural (步骤/顺序) | 40-120 tokens | "First oil, then garlic, then tomatoes." |
| Summary (概括/描述) | 80-200 tokens | 详细描述 |

> v8.0: 删除 Uncertain 作为独立类型。模型信息不够时自然回答"无法确定"，这是 response 的内容，不是独立行为。

---

## 2. 记忆架构

### 2.1 记忆状态（统一时间线）

> v8.0 变更：取消 compressed_segments / recent_thinks 两个区的分离。
> 改为**一条统一的时间线列表** — summary 和 think 混排，按时间顺序。
> 压缩 = 原位替换（选中的 thinks 被 summary 替换，位置不变）。

每个 timestep，系统从外部拼装一份完整 input 给模型。模型看到：

```
模型看到的完整 input:
  visual_window: [最近 12 chunks 的帧]         ← FIFO，固定大小
  memory_timeline: [                            ← 一条时间线，按时间顺序
    <summary t="0-16">Chef prepared workspace, oil heated...</summary>,
    [16-18] Garlic browned in oil,              ← 未压缩的 think
    [18-20] Tomato quarters added to pot,
    <summary t="20-30">Seasoning added, sauce simmered...</summary>,
    [30-32] Chef covers pot with glass lid,     ← 未压缩的 think
    [32-34] Chef walks toward refrigerator,
  ]
  queries: [{question: "...", answers: [...]}]  ← 独立问答追踪区
  user_input: 当前新问题 / 空
```

**summary 和 think 在同一个列表里，按时间顺序排列。不存在"两个区时间顺序倒置"的问题。**

**压缩 = 原位替换**：

```
压缩前 memory_timeline:
  [0-2]  think_0
  [2-4]  think_1  ←─┐
  [4-6]  think_2     │ 压缩选了这 4 条
  [6-8]  think_3     │
  [8-10] think_4  ←─┘
  [10-12] think_5

压缩后 memory_timeline:
  [0-2]  think_0                                    ← 保留（高价值）
  <summary t="2-10">Chef chopped and seasoned...</summary>  ← 替换了 think_1~4
  [10-12] think_5                                   ← 保留

时间线始终连续：0-2 | 2-10 | 10-12，不管压缩选在哪里。
```

**核心设计：文本记忆覆盖时间 > 视觉窗口**。

```
时间线:  [== memory_timeline (summary + thinks 混排) ==]
                              [== visual_window ==]
         ↑ 最早                                    ↑ 当前
```

Think 每步生成后**立即追加**到 memory_timeline 末尾。视觉窗口由 FIFO 独立管理。

**压缩状态更新顺序**（不可违反）：
```
模型输入 = pre-action memory_timeline（不含当前 think）
系统触发压缩 (SYSTEM_PROMPT_COMPRESS):
  1. 选中 memory_timeline 中一段连续 thinks
  2. 模型输出 summary
  3. 系统用 summary 替换选中的 thinks（原位替换）
  4. 再 append 当前 think 到 timeline 末尾
```

**Recall = 系统侧检索**。模型输出 `recall` 后，系统从 _retrieval_archive（所有历史 thinks 原文）中查找。

**Summary 信息来源**：模型做压缩时能看到被压缩的 thinks 文本 + 当前 visual_window 中与压缩范围重叠的帧。

**Action 优先级**（主循环 SYSTEM_PROMPT 内）：
```
用户当前新问题 response/recall > query 触发 response > silent
```
compress 不在主循环中，不存在优先级冲突。

### 2.2 单条训练样本的 input 结构

```
[System prompt: 协议说明]                                           ← Zone A
[Visual window: 最近 12 chunks 帧]                                  ← Zone B (固定)
[Memory timeline:]                                                  ← Zone C (可变，按时间序)
  <summary t="0-16">Chef prepared workspace...</summary>
  [16-18] Garlic browned in oil.
  [18-20] Tomato quarters added.
  <summary t="20-30">Seasoning added, sauce simmered...</summary>
  [30-32] Chef covers pot with glass lid.
  [32-34] Chef walks toward refrigerator.
[Queries:]                                                          ← Zone Q
  <queries>
    <query t="20">Tell me when basil is added</query>
    <answer t="70">Basil leaves being torn over pot.</answer>
  </queries>
[User input: 当前新问题 / 空]                                       ← Zone D
```

**summary 用 `<summary t="X-Y">` 标签，think 用 `[X-Y]` 前缀。两者在同一个列表，按时间排序。**

### 2.3 压缩触发与范围选择

**触发条件**: `timeline 中 thinks 总 token ≥ RECENT_THINKS_TOKEN_BUDGET × 80%`（~480 tok）

具体行为取决于 think 长度：
- 长 think（~60 tok/条）→ ~8 条就触发（早）
- 中等 think（~50 tok/条）→ ~10 条触发
- 短 think（~40 tok/条）→ 12 条才触发（由 item 硬上限兜底）

**Hysteresis 机制**：压缩后 timeline 中 thinks 总 token 须降至 `RECENT_THINKS_TOKEN_BUDGET × 55%`（~330 tok）以下。
如果压缩一次不够（窗口太短），系统会扩大范围或连续触发。这防止频繁触发压缩导致 summary 质量下降。
见 config.py `COMPRESS_HYSTERESIS_RATIO = 0.55` / `COMPRESS_HYSTERESIS_THRESHOLD = 330`。

造数据时直接加载学生模型 tokenizer 精确计算 token 数（`get_tokenizer()` 自动加载）。
若 tokenizer 不可用则降级为 `chars/4` 估算。

**注意**：
- think 立即追加到 memory_timeline 末尾。视觉窗口独立管理。
- **recall step 2 和 compress 的输出不存入 memory_timeline**。每个时间步只有主循环的 think 进入。

**压缩在两步之间触发**：
系统检测到 memory_timeline 总 token 超阈值后，在当前步和下一步之间插入 SYSTEM_PROMPT_COMPRESS 调用。

**范围选择（C1: teacher 多维评分，归一化 [0,1]）**：
系统评估 memory_timeline 中所有合法连续范围，选择综合得分最低的：

All 5 dimensions normalized to [0, 1], then weighted (configurable):

| 维度 | 权重 | [0,1] 含义 |
|------|------|-----------|
| content | W=0.30 | 平均 per-item importance（entity/OCR/digits/state_change 加权） |
| merge | W=0.20 | max_merge_level / 3（re-compress penalty） |
| boundary | W=0.15 | 边界有 state_change → 0.5/1.0 |
| recency | W=0.20 | center_position / timeline_length（越近越保留） |
| token | W=0.15 | range_tokens / est_total（越多越优先压缩，subtracted） |

```
score = W_content × content + W_merge × merge + W_boundary × boundary + W_recency × recency - W_token × token_ratio

LOWER = better to compress. 权重可在 config 调整。
```

**合法范围约束**：
1. timeline 中连续 items（可跨 summary）
2. 至少含 2 个 thinks
3. summary 有 merge_penalty 抑制再压缩

**二级压缩**：当 memory_timeline 中 summary 段数超过 5 时，系统合并最老的两个相邻 summary（截断到 200 tok）。

**三阶段训练计划**：

| 阶段 | 触发方式 | 范围选择 | 目标 |
|------|---------|---------|------|
| **C1** | SYSTEM_PROMPT_COMPRESS + 指定范围 | teacher 指定最优范围 | 学会压缩行为 + 写好 summary |
| **C2** | SYSTEM_PROMPT_COMPRESS + 不指定范围 | 模型自选（gold = teacher policy） | 学会选择压缩窗口 |
| **C3**（未来） | 模型自主判断何时压缩 | 模型自选 | 完整记忆管理 |

### 2.4 Token 预算

| 组件 | Tokens | 说明 |
|------|--------|------|
| System prompt | ~150 | |
| 压缩段 (≤5 × 150) | ≤750 | |
| 未压缩 thinks (≤12 × 50) | ≤600 | 容量 12 条，80% 触发压缩 |
| Visual window (12 chunks) | ~1536 | min_pixels=100352 |
| User input (问题/trigger/result) | ~100 | |
| Recall result (含 4帧) | ~356 | 256 vision + 100 text |
| **输入总计** | **~3400** | |
| 输出: think + action | ~80 | silent/recall 时 |
| 输出: think + action + response | ~80-280 | factoid 5-40 / procedural 40-120 / summary 80-200 |
| 输出: think + action + summary | ~230 | compress ��（summary 100-180 tok） |
| **单样本总计** | **~3500-3700** | 远在 16K 内 |

**注**：需用目标 tokenizer/processor 实测验证。上述为草算。

---

## 3. 数据构造 Pipeline（4 阶段）

> v9.0 变更：
> - 3-A: 所有 family 并发调用（asyncio.gather），共享输出 schema，P1/C1/R1 双通路候选
> - 3-B: 贪心多样性选择替代旧分桶方案，5 条密集轨迹（每条 4-6 问题），关键词提取修复 binary/MC
> - 3-C: 新增 base sample 生成（5 类选择性采样），fork+base 合并为完整 episode
> - Pass 5 重命名为 Pass 4，新增 4 项轨迹/base 一致性检查（共 14 项）

```
阶段 1: Teacher Evidence Graph          [397B + 视频, ~5min]
  ├─ 1-A: 独立 chunk 标注              [397B, 2帧/chunk, 全并行]
  └─ 1-B: 实体链接                     [纯文本 397B, 2 call/video]
阶段 2: Question-blind Streaming Rollout [397B + 视频, ~1h]
阶段 3: Task Mining + Sample Generation  [397B + 规则, ~30min]
  ├─ 3-A: Task Card 生成               [397B, ~6-8 call/video, asyncio.gather 全并行, ~2s]
  ├─ 3-B: Placement + 行为序列规划     [纯程序, 0 call, 贪心多样性选择]
  └─ 3-C: 轨迹样本生成 + base 采样     [397B, ~15-25 call/video, fork+base 合并]
阶段 4: Verify + Filter                 [纯规则, 14 项检查, per-video 保存]
```

### 3.1 阶段 1: Teacher Evidence Graph

> v8.0 重构：每个 2s chunk 独立标注（2 帧，无前文 context），视频内可全并行。
> 实体一致性由轻量后处理解决，不再阻塞主流程。

**目标**: 为每个 2s chunk 生成详细结构化事实图。这是**隐藏教师信息**，不进入 SFT。

#### 1-A: 独立 chunk 标注（可全并行）

**397B 输入**: 仅当前 chunk 的 2 帧，无前文 caption、无滑窗。

**每个 chunk 只输出当前 2 帧的静态快照——"画面里有什么"，不输出"相比之前变了什么"。**

独立 2 帧能生成 vs 不能生成：

| 字段 | 能否独立生成 | 说明 |
|------|------------|------|
| `visible_entities` | **能** | 看 2 帧就知道画面里有什么 |
| `atomic_facts` | **能** | 2 帧中可观察的事实 |
| `ocr` | **能** | 2 帧里的文字 |
| `spatial` | **能** | 当前帧空间布局 |
| `confidence` | **能** | 画面清不清楚是当前帧的事 |
| ~~`state_changes`~~ | **不能** | "变化"需要前一个 chunk 做对比——独立 chunk 没有 |
| ~~`support_level`~~ | **不能** | `carried_from_previous` 需要前文 |
| ~~`relation_to_previous`~~ | **不能** | 需要前文 |

**397B 输出** (per chunk):
```json
{
  "time": [28, 30],
  "visible_entities": [
    {"desc": "person wearing red apron, short hair", "action": "sprinkling seasoning", "position": "center"},
    {"desc": "small white pinch bowl", "held_by": "person wearing red apron", "position": "center"},
    {"desc": "stainless steel pot", "position": "front-right burner"}
  ],
  "atomic_facts": [
    {"fact": "person sprinkles white granular seasoning from small bowl into pot",
     "confidence": 0.88, "target_resolution_visible": true},
    {"fact": "amount approximately one teaspoon",
     "confidence": 0.55, "target_resolution_visible": false}
  ],
  "ocr": [],
  "spatial": "person center, pot on right burner"
}
```

**不再输出 `state_changes`、`support_level`、`relation_to_previous`。**

**简化后的 prompt**:

```
你在标注一段 2 秒视频片段（t={start}-{end}s）。

根据上方 2 帧画面，输出 JSON:
{{
  "time": [{start}, {end}],
  "visible_entities": [
    {{"desc": "外观描述（穿着/颜色/材质/大小）", "action": "正在做什么", "position": "画面位置"}}
  ],
  "atomic_facts": [
    {{"fact": "可观察的事实陈述", "confidence": 0.0-1.0, "target_resolution_visible": true/false}}
  ],
  "ocr": ["画面中可见的文字"],
  "spatial": "简要空间布局"
}}

规则:
- 只描述这 2 帧中可见的内容（不需要和其他片段比较）
- 用外观描述实体（不要编号，不要假设身份）
- confidence < 0.7: 模糊/快速运动/部分遮挡
- target_resolution_visible: 小字/远处细节在训练分辨率下不可读时为 false
```

**与旧方案的区别**：

| | 旧方案 | 新方案 |
|--|--------|--------|
| 视觉输入 | 24 帧滑窗 | **2 帧** |
| 文本 context | 前 30 个 chunk 的 caption | **无** |
| 实体标识 | `"id": "chef_1"` (依赖前文) | `"desc": "person wearing red apron"` (自描述) |
| state_changes | 397B 生成（依赖前文） | **后处理推导**（见 1-B） |
| 视频内并发 | 串行 | **全并行** |
| 每请求 token | ~10K | **~1.5K** |

**并发**:
```
视频内全并行 + 视频间并行
每请求: 2帧 + prompt = ~1.5K input, max_tokens=16384, thinking=ON
每视频: 60 chunks × 1 call = 60 calls，全并行 → ~15s
300 视频 ÷ 64 ≈ ~5 min（vs 旧方案 ~70 min）
```

#### 1-B: 后处理（两次 397B 调用：实体对齐 + 变化检测）

1-A 输出的是每个 chunk 的**静态快照**。1-B 做跨 chunk 分析，拆为两次 397B 调用（避免单次 thinking 爆炸）。

##### 两次调用设计

```
Call 1: 实体对齐
  输入: ~1K tokens（unique desc 列表，~20-40 项，纯文本）
  输出: entity groups [{id: "person_1", descs: [...]}]
  → 写回 entity["id"]（粗筛 hint）

Call 2: 变化检测
  输入: ~3K tokens（chunk entity/action 摘要，纯文本）
  输出: state changes [{chunk: 3, change: "started pouring oil"}]
  → 写回 cap["state_changes"]
```

**两次都是纯文本输入，不发视频帧。**
1-A 已经用 2 帧详细描述了每个实体的外观/动作，1-B 基于这些文本描述做跨 chunk 推理足够。
如果未来发现纯文本不够（歧义场景），可以为 Call 1 附加少量代表帧（每个 unique desc 一帧）。

##### entity_id 是粗筛 hint，不是 ground truth

实体对齐即使用 397B 也可能出错（"hand stirring" 是 person_1 还是 person_2 的手？）。因此：

| 用途 | 怎么用 entity_id | 出错的影响 |
|------|-----------------|----------|
| `scan_opportunities` C1/R1 粗筛 | "同 id 在不同 chunk 出现 → 标记为候选" | 多标/漏标几个候选，无害 |
| Pass 3-A 397B 生成 Task Card | **不用 entity_id** — 397B 看完整 evidence 自己判断 | 不受影响 |
| Task Card 问题文本 | **必须用 desc 描述**，不用 id | 不受影响 |

```
问题文本：
  ✅ "穿红围裙的人现在和之前做的事一样吗？"（desc）
  ✅ "之前那个小白碗还在画面里吗？"（desc）
  ❌ "person_1 和之前比有什么变化？"（id）
  ❌ "bowl_1 还在画面里吗？"（id）
```

**模型在推理时看到的是视觉帧和 desc 文本，不是 id。训练数据的问题也必须用 desc。**

##### 并发与 context 控制

```
1-B 每视频 2 次纯文本 397B call
  300 videos × 2 calls = 600 calls
  每 call: ~1-3K input, max_tokens=16384 (含 thinking)
  并发: 由 asyncio.gather 自然控制（~300 video 并发）
  总耗时: ~15-30s
```

**用途**（不变）：
- 造任务（知道哪里有可问的细节）
- 生成标准答案（来自 atomic_facts）
- 验证 student think 是否遗漏关键信息
- 验证 compressed_summary 是否保留必要内容
- 目标分辨率可见性检查（confidence < 0.5 的 fact 不适合做任务）

### 3.2 阶段 2: Question-blind Streaming Rollout

**目标**: 模拟学生模型的真实流式体验（无问题介入），产出：
- 每个 timestep 的 student think (40-60 tok)
- 压缩决策和 summary
- 主动 recall 事件
- **每个 timestep 的 memory state snapshot**

**核心原则**: Question-blind。不知道未来会有什么问题。不能为未来问题优化 think 或 summary。

**397B 输入** (每个 timestep):
```
[memory_timeline (summary + thinks)] + [24帧视觉窗口] + prompt
```

**Rollout 产出数据结构**（与代码完全对齐）:

```python
# 每步产出的 think（存入 thinks 列表）
thinks[chunk_idx] = {
    "chunk_idx": 14,
    "time": [28, 30],
    "think": "Chef sprinkles seasoning from small white bowl into pot on right burner."
}

# 压缩事件（当 80% token 预算触发时）
compression_events.append({
    "trigger_chunk": 22,
    "summary": {"time_range": [0, 16], "text": "..."},
    "compressed_thinks_chunks": [0, 1, 2, 3, 4, 5],  # 被压缩的 chunk 列表
})

# 每步 pre-action snapshot（不含当前 think）
snapshots[chunk_idx] = {
    "chunk_idx": 14,
    "timeline": [...],          # summary + thinks 混排，按时间序
    # queries 区独立于 snapshot，由系统单独管理
    "visual_window_start": 3,
}
```

**关键**:
- 压缩 summary 基于**当前 thinks**（不是 teacher caption），且不知道未来问题
- 压缩范围由 `choose_optimal_compress_range(pre_action_thinks)` 从 pre-action snapshot 中选出
- 压缩事件中 `compressed_thinks_chunks` 记录的是真实被压缩的 chunk 列表（不依赖模型输出的 time_range）

**并发配置**:
```
视频内串行, 视频间 64 并行
每请求: ~10K input (24帧 + memory), max_tokens=16384, thinking=ON
每视频: 60×12s + 4×15s(compress) = 780s ≈ 13 min
300 视频 ÷ 64 ≈ 61 min ≈ 1h
```

### 3.3 阶段 3-A: Task Card 生成

> v8.0: Teacher 只负责找"可问什么"。不决定 ask_time 和 action。
> 拆成 opportunity 筛选 + 按 family 分批 397B call。

**目标**: 从 evidence 中提取"可问的内容"，生成 Task Card。

**输入**: evidence_1b（含 entity_id hint + state_changes）

**输出**: task_cards/{video_id}.json

#### Task Card 数据结构（精简版）

**只保留有明确消费者的字段，其余程序推导：**

```python
TaskCard = {
    "card_id": "vid001_F2_001",           # 唯一标识
    "family": "F2",                        # 问题家族 → 决定 retention_class / 3-B 逻辑
    "question": "围裙是什么颜色？ A.红色 B.蓝色 C.白色 D.绿色",  # MC 选项嵌入问题
    "canonical_answer": "A",               # MC→字母; binary→Yes/No; number→数字; exact→短文本
    "answer_form": "multiple_choice",      # binary/multiple_choice/number/short_exact/descriptive
    "support_chunks": [5],                 # 397B 从哪个 chunk 发现的证据
    "visibility_type": "persistent",       # persistent / transient
}
```

**7 个字段，每个有明确消费者：**

| 字段 | 谁消费 | 做什么 |
|------|--------|--------|
| `card_id` | 全流程 | 唯一标识 |
| `family` | 3-B, retention_class 推导 | 决定合法 availability + bitmap 阈值 |
| `question` | 3-C, SFT 样本 | 问题文本（MC 选项嵌入其中） |
| `canonical_answer` | 3-C response 约束, GRPO 验证 | 标准答案 |
| `answer_form` | 验证系统 | exact match / 容差 / LLM judge |
| `support_chunks` | 3-B placement 计算 | 证据位置 |
| `visibility_type` | 3-B availability 过滤 | persistent 只允许 in_visual |

**删除的字段及其替代：**

| 删除字段 | 替代方式 |
|---------|---------|
| `choices` | 嵌入 question 文本：`"A.红色 B.蓝色 C.白色 D.绿色"` |
| `answer_keywords` | 程序从 canonical_answer 提取：`extract_keywords("Red") → ["red"]` |
| `retention_class` | 从 family 直接映射：`F1/F2/F3→low, F4/P1/E2/C1/R1→medium, E1/S1/M1→high` |
| `support_facts` | 查 `evidence[support_chunks[0]]["atomic_facts"]` |
| `entities_involved` | 问题文本已经用 desc 引用实体，不需要单独字段 |

#### 各 answer_form 的问题格式

```
binary:
  question: "围裙是红色的吗？"
  canonical_answer: "Yes"

multiple_choice:
  question: "围裙是什么颜色？ A.红色 B.蓝色 C.白色 D.绿色"
  canonical_answer: "A"
  干扰选项来源：同视频其他 chunk 的同类 fact，不够时用 category 通用项

number:
  question: "加了几勺盐？"
  canonical_answer: "1"

short_exact:
  question: "围裙什么颜色？"
  canonical_answer: "Red"

descriptive:
  question: "描述当前场景"
  canonical_answer: "厨师在红色围裙下切菜，旁边有不锈钢锅"
```

#### visibility_type

| 类型 | 含义 | 例子 | 3-B 怎么用 |
|------|------|------|----------|
| **persistent** | 实体/属性持续可见 | 围裙颜色、锅的位置、人的穿着 | 只选 in_visual（答案一直在帧里，不可能 recall） |
| **transient** | 事件/数字一闪而过 | 加盐动作、屏幕 $4.99、切菜步骤 | 可选 in_visual + in_history_only（可造 response/recall 对照） |

397B 在生成 card 时标注 visibility_type，3-B 据此过滤不合法的 availability。

#### 生成流程：结构化筛选 + 分组 397B call

**Step 1: 结构化字段筛选 + 双通路候选（纯程序）**

`classify_chunks(evidence)` → `{family: [chunk_indices]}`。主路用 1-B 字段，备路用 1-A 直接视觉字段（减少因 1-B 文本推导漏标导致的 false negative）：

| Family | 主路筛选条件 | 备路（1-A fallback） |
|--------|------------|-------------------|
| F1 | `ocr` 非空 或 facts 含有效数字（`\d{2,}` 或货币/单位模式） | — |
| F2 | 有 `visible_entities` | — |
| F3 | facts 含有效数字（**独立于 F1，不互斥**） | — |
| F4 | `visible_entities` ≥ 2 | — |
| E1 | 有 `atomic_facts` 的 chunk 动态步长采样（保证 ≥ target×2 候选） | — |
| E2 | 有 `state_changes`（来自 1-B） | — |
| P1 | 连续 ≥ 3 个有 `state_changes` | 连续 ≥ 3 个 chunk 的主 entity `action` 各不同（1-A） |
| S1 | `visible_entities` ≥ 3 | — |
| C1 | 同 `entity_id` + 不同 chunk 有 `state_change` | desc 词重叠 ≥ 0.6 + action 不同（1-A） |
| R1 | 同 `entity_id` 消失后重现（gap ≥ 5） | desc 词重叠 ≥ 0.6 + chunk 间隔 ≥ 5（1-A） |
| M1 | 不按 chunk 筛选，给全视频摘要 | — |

数字过滤用正则 `\d{2,}|[\$€¥£]\d|\d\s*(kg|lb|ml|oz|cm|mm|g)\b`，排除 "person 1" 等噪声。
facts 过滤：confidence ≥ 0.7。

**Step 2: asyncio.gather 并发调 397B**

所有 family 调用完全独立（不看其他 family 的输出），用 `asyncio.gather` 并行发出。每视频 ~6-8 个有效 family call，~2-3 秒完成。每组 ~500-1000 tok input，~200-300 tok output。

**共享输出规范**（`_OUTPUT_SCHEMA`）：所有 11 个 family prompt 末尾拼接统一的 JSON schema + 格式约束：
- 5 个必须字段：`question, canonical_answer, answer_form, support_chunks, visibility_type`
- `canonical_answer` 格式：binary→"Yes"/"No"，MC→"A"/"B"/"C"/"D"，number→纯数字，short_exact→1-5 词，descriptive→1-3 句
- Entity 引用规则：必须用外观描述，禁止用 entity_id
- MC 选项嵌入问题文本，干扰项来自同视频同 family 的其他 fact

**关键词提取**：`extract_card_keywords(card)` 按 answer_form 智能提取（非统一用 canonical_answer）：

| answer_form | 提取来源 | 示例 |
|-------------|---------|------|
| binary | question 文本 | "Is the apron red?" → `['apron', 'red']` |
| multiple_choice | question 主体 + 正确选项文本 | "What color? A.Red ..." 答 "A" → `['color', 'apron', 'red']` |
| number | question + 数字 | "How many tomatoes?" 答 "3" → `['tomatoes', 'cut', '3']` |
| short_exact / descriptive | canonical_answer | 不变 |

**并发配置**: `concurrent: 512`（纯文本，batch budget 支持 ~1882），每视频 ~6-8 call 全并行 → ~2s/video。

### 3.4 阶段 3-B: Placement + 行为序列规划

> 纯程序，零 397B。不只选"在哪问"，还规划"问了之后的完整行为序列"。

**输入**: Task Cards + Rollout (snapshots, compression_events, evidence)

**输出**: placements/{video_id}.json — 行为序列蓝图列表

#### 核心概念：行为序列

一个问题从提出到完结，模型的行为不是单一 action，而是一条**行为序列**：

| 序列类型 | 触发条件 | 行为链 | 样本数 |
|---------|---------|--------|-------|
| **immediate_response** | 答案在帧/memory/summary 中 | response → post_silent | 2 |
| **recall_success** | 答案不在可见源，recall 返回有用结果 | recall → post_recall_response → post_silent | 3 |
| **recall_fail_then_found** | recall 失败，但后续帧里出现答案 | recall → post_recall_silent → ... → response | 4+ |
| **event_watch** | E2: 事件还没发生 | silent → ... → response (事件发生时) | 3+ |
| **multi_response** | M1: 持续解说 | response → silent → response → ... | 4+ |

#### availability → 行为序列类型映射

```python
def determine_sequence_type(card, availability):
    if card.family == "M1":     return "multi_response"
    if availability == "in_future":  return "event_watch"
    if availability in ("in_visual", "in_recent_thinks", "in_compressed"):
        return "immediate_response"
    if availability == "in_history_only":
        return "recall_success"  # recall_fail_then_found 由 rng.random()<0.3 额外生成
    return "immediate_response"
```

> v9.0: availability 名称统一为 `in_recent_thinks` / `in_compressed`（旧名 `in_timeline_think` / `in_timeline_summary` 已废弃）。
> `precompute_retention` 使用 `extract_card_keywords(card)` 而非 `extract_keywords(canonical_answer)`，修复 binary/MC 的 bitmap 全 False 问题。

#### Placement 数据结构（行为序列蓝图）

每条 Placement = `{card_id, ask_chunk, sequence_type, key_chunks}`。key_chunks 按序列类型不同：

| sequence_type | key_chunks 字段 | 行为链示例 |
|---------------|----------------|-----------|
| **immediate_response** | `ask` → `post_silent` | ask:20 → response → post_silent:21 |
| **recall_success** | `ask` → `post_recall` → `post_silent` | ask:20 → recall → post_recall:20(response) → post_silent:21 |
| **recall_fail_then_found** | `ask` → `post_recall` → `wait_silent[]` → `found_response` → `post_silent` | ask:45 → recall → post_recall:45(silent) → wait:46 → found:50(response) → post_silent:51 |
| **event_watch** | `ask` → `wait_silent[]` → `trigger` → `post_silent` | ask:30 → silent → wait:[32,35] → trigger:40(response) → post_silent:41 |
| **multi_response** | `ask` → `no_change_silent[]` → `followup_response[]` → `post_silent` | ask:10 → response → silent:[15,18] → response:[25,40] → post_silent:41 |

#### 计算 key_chunks 的逻辑

`compute_placement(card, ask_chunk, sequence_type, rollout, evidence)` 按序列类型填充 key_chunks：

- **immediate/recall_success**: post_silent = ask+1; recall_success 额外设 post_recall = ask
- **recall_fail_then_found**: `find_next_visible_chunk()` 找答案重现的 chunk → found_response; 找不到则 placement 无效
- **event_watch**: trigger = min(support_chunks), 均匀采样 2 个 wait_silent 点
- **multi_response**: 遍历后续 chunk，有 state_change → followup_response（最多 5 次），无变化 → no_change_silent（最多 2 个）

#### 轨迹规划：贪心多样性选择

> v9.0 重写：旧方案 30 条稀疏轨迹（1-3 问题/条）→ 5 条密集轨迹（4-6 问题/条），silent/response 比例从 80/8 改善到 ~60/30。

`plan_trajectories(placements, cards_map, target=5, max_placements_per_traj=5, min_chunk_gap=8)`：

**Phase 1: 贪心选择 placements**（budget = target × max_pp = 25）

每轮对所有候选打分，选最高分的 placement：

| 维度 | 分值 | 说明 |
|------|------|------|
| 答案可验证 | +1.0 | binary/MC/number/short_exact（支撑 RL reward） |
| 未见 family | +2.0 | 已选中没有这个 family → 多样性奖励 |
| 未见 sequence_type | +2.0 | 同上 |
| 时间分散 | +0~1.5 | 与已选 ask_chunks 的最小距离 / 10，cap 1.5 |
| support_inferred | -2.0 | 3-A 推断的 support_chunks → 质量惩罚 |

同一 card_id 不重复选择。使用 seeded RNG 打破平分。

**Phase 2: 组合为轨迹**

按 ask_chunk 排序后，间隔 ≥ 8 chunks（16 秒）的 placements 分入同一 trajectory（最多 5 个/条）。剩余未配对的作为单问题轨迹。

**效果**: 5 条轨迹覆盖 ≥ 5 种 family + ≥ 3 种 sequence_type，问题均匀分布在视频时间线上。

### 3.5 阶段 3-C: 轨迹样本生成

> v9.0 变更：新增 base sample 生成（5 类选择性采样），fork+base 合并为完整 episode。
> recall_fail_then_found 补 post_silent，multi_response followup 传 prior_answers 避免重复。

**输入**: trajectories + rollout + evidence + cards_map

**输出**: samples_3c/{video_id}.json（fork + base 合并，按 chunk_idx 排序）

#### Fork 样本：按行为序列逐 key_chunk 生成

维护 `queries_state[]` 演化 + `queries_state_at_chunks` 边界记录。

| 序列类型 | 生成的样本（按序） | 397B call |
|---------|------------------|-----------|
| immediate_response | ask→response, **post_silent** | 1 |
| recall_success | ask→recall, post_recall→response, **post_silent** | 2 |
| recall_fail_then_found | ask→recall, post_recall→silent, wait×N, found→response, **post_silent** | 3 |
| event_watch | ask→silent, wait×N, trigger→response | 1 |
| multi_response | ask→response, no_change_silent×N, followup→response×N（**含 prior_answers 去重**） | 1+N |

#### Base 样本：选择性采样（非每 chunk 都选）

`_select_base_chunks()` 从 60 chunks 中选 ~20-35 个有训练价值的，分 5 类：

| 类别 | 选择规则 | 训练目的 |
|------|---------|---------|
| **Warmup** | chunk 0-2 | 冷启动（memory 为空的观察） |
| **Evidence anchor** | support_chunks ± 2 | recall 证据所在位置（模型需学会观察/记住） |
| **Question window** | 每个 key_chunk ± 2/+3 | Q&A 前后上下文（非 min-max 全覆盖） |
| **Compress anchor** | trigger ± 1 + 被压缩 chunk 首尾各 2 | 压缩上下文（非全选被压缩 chunks） |
| **Long-silent patrol** | 间隔 > 10 chunks 的空白区每 5 chunk 采样 1 个 | 教模型"问题已答，持续 silent"（queries_state 非空） |

`generate_base_samples()` 为每个选中 chunk 插值正确的 `queries_state`（从最近的 fork key_chunk 边界继承）。

#### 完整 episode = fork + base，按 chunk_idx 排序

**action 比例**（5 问题 trajectory）：silent ~60%，response+recall ~30%，compress ~10%。

#### 397B 调用量

```
~5 轨迹 × ~3-5 calls = ~15-25 calls/视频
300 视频 = ~4,500-7,500 calls（旧方案 ~27,000 → 降 70%）
```

### 3.6 阶段 4: Verify + Filter（pass4_verify.py）

> v9.0 重写：旧"5 类验证"→ **14 项检查**（10 项旧有适配 + 4 项新增），按 trajectory 分组验证，per-video 保存。

**输入**: 3-C 的 fork + base 合并样本

**输出**: `verified/{video_id}.json`（通过的样本 + 统计）

#### 14 项检查

**Per-sample 检查（1-10 旧有，适配 base sample）**：

| # | 检查 | base sample 处理 |
|---|------|:---:|
| 1 | `information_flow`: 无未来信息泄漏 | 跳过（无 Q&A） |
| 2 | `action_minimality`: action 匹配 sequence_type | 始终通过（silent/compress by construction） |
| 3 | `grounding`: think 无声音/情绪/元推理 | base compress 允许空 think |
| 4 | `format`: tag 完整 + action 合法 + JSON 合法 | base compress 允许缺 think tags |
| 5 | `think_token_length`: think 40-60 tok ± 容差 | 标准 |
| 6 | `compression_ratio`: 压缩比 ≥ 2.5 | 无 input.memory 时跳过 |
| 7 | `summary_provenance`: summary 实体来自源 thinks | 无 source_texts 时跳过 |
| 8 | `summary_retention`: summary 保留关键实体/数字 | 标准 |
| 9 | `summary_no_current_think_leak`: summary 不含当前 think 独有事实 | 标准 |
| 10 | `question_answer_leakage`: 问题文本不泄漏答案 | 标准 |

**新增检查（11-14）**：

| # | 检查 | 内容 |
|---|------|------|
| 11 | `queries_state_temporal` | queries 列表每项必须是 `{question, answers}` dict |
| 12 | `trajectory_action_distribution` | 轨迹必须有 ≥1 active 样本；多问题轨迹 silent ≤ 90% |
| 13 | `base_sample_consistency` | base sample 只能 silent/compress，无 user_input |
| 14 | `recall_evidence_reachable` | recall 的 support_chunks 必须在 ask_chunk 之前 |

#### Difficulty 标签

基于 `sequence_type` + `action` 推导：

| 条件 | 难度 |
|------|------|
| base silent (queries 为空) | easy |
| base silent (queries 非空) / base compress | medium |
| fork silent (event_watch) | medium |
| immediate_response | easy |
| recall_success | medium |
| recall_fail_then_found | **hard** |
| multi_response | medium |

#### 数据切分

按 video_id 切分（不按 sample 随机切），同视频所有样本只在同一 split。300 videos → 240 train / 30 val / 30 test。

---

## 4. 问题家族与答案验证等级

> v8.0: 按问题本体分类（非 action 分类），同一问题在不同 ask_time 自动产生不同 action。

### 4.1 问题家族

| Family | 名称 | retention | 优先 answer_form |
|--------|------|-----------|-----------------|
| **F1** OCR/Number | 价格/分数/屏幕字段 | low | number, short_exact |
| **F2** Attribute | 颜色/材质/形状 | low | binary, multiple_choice |
| **F3** Count | 个数/份数/次数 | low-mid | number |
| **F4** Spatial | 相对位置/布局 | medium | binary, multiple_choice |
| **E1** Action/State | 当前在做什么 | high | short_exact, binary |
| **E2** Change | 开始/停止/变成 | medium | binary, short_exact |
| **P1** Procedure | 第几步/先后顺序 | medium | number, multiple_choice |
| **C1** Compare | 前后差异 | low | binary, short_factual |
| **R1** Re-id | 是否同一实体 | medium | binary |
| **S1** Summary | 短时段概括 | high | descriptive |
| **M1** Commentary | 持续解说（queries 机制） | high | short_factual |
| **D1** Follow-up | 追问细节 | — | **二版** |

M1 通过 queries 历史 answer 避免重复回答。D1 推迟。retention_class 不决定 action（由 availability 决定），用于 3-B 难度评估。

### 4.2 答案验证等级

V1 binary(≥15%) / V2 multiple_choice(≥20%) / V3 number(≥10%) / V4 short_exact(≥15%) = 可自动验证（支撑 RL reward），**一版目标 ≥60%**。V5 short_factual(≤25%, keyword+LLM judge) / V6 descriptive(≤15%, 仅 LLM judge)。
Multiple choice: 正确 + 3 干扰项（同视频同 family evidence），随机打乱。

### 4.3 Action 类型（由 availability 自动推导）

`in_visual` / `in_recent_thinks` / `in_compressed` → response；`in_history_only` → recall；`in_future` → silent（仅 E2）。
Recall 必要条件：答案不在帧/thinks/compressed 中，但存在于历史中。Compress 从 rollout 继承。

### 4.4 Queries 机制（替代旧 pending）

v8.0: queries 区独立于 memory（不被压缩），存问题原文 + 历史回答。即时可答不进 queries；需持续观察的（E2/M1）加入 queries，模型对比历史 answer 判断新内容；recall 不进 queries（同 chunk 完成）。一版不做 answer 压缩（~360 tok 可控），不主动删除 query。

### 4.5 各 Phase 配比（episode-level 控制）

P1: silent 65% / response 35%。P2: 50/20/20(recall)/10(special)。C1: 45/18/15/12(compress)/10。C2: 42/18/15/15/10。P5: 45/20/15/10/10。

### 4.6 负样本（v8.0 简化）

一版 3 类：不存在内容(2%) + recall 错误(3%) + 记忆画面冲突(2%)。歧义/纠正推迟二版。Unanswerable 在 3-C 末尾追加模板。

---

## 5. 训练样本格式

### 5.1 一条训练样本（recall_query 示例）

```json
{
  "sample_id": "vid001_t60_recall_query_42",  "video_id": "vid001",
  "sample_type": "recall_query",  "chunk_idx": 30,  "phase": "C1",
  "input": {
    "system": "You are a streaming video agent...",
    "memory": {"timeline": ["<summary t=\"0-20\">...</summary>", "[40-42] ...", "[42-44] ..."]},
    "queries": [{"question": "Describe each step", "answers": ["Chopping onions"]}],
    "visual_window": {"video_start": 36, "video_end": 60, "frames": 24},
    "user_input": "How much salt did the chef add?"
  },
  "output": "<think>Chef adjusting burner dial.</think><action>recall</action><query>{\"query\":\"seasoning amount pot\",\"time_range\":\"0-48\"}</query>",
  "metadata": {
    "gold_answer": {"answer": "~1 teaspoon", "source": "teacher_caption"},
    "gold_action": "recall",  "card_id": "vid001_F1_003",  "family": "F1",
    "availability": "in_history_only",
    "action_minimality": {"answer_in_frames": false, "answer_in_historical": true},
    "leakage_checks": {"query_contains_answer": false}
  }
}
```

### 5.2 Recall follow-up 样本

同 §5.1 input 结构，额外含 `recall_context`（原始问题+query）、`recalled_frames`（4帧）、`recall_result`（检索文本）。
Output: `<think>分析结果(20-40tok)</think><action>response</action><response>...</response>`。
Recall failure 时 response 表达"信息不足"。

### 5.3 Compress 样本

```json
{
  "sample_id": "vid001_t44_compress",
  "input": {
    "memory": {"timeline": ["<summary t=\"0-20\">...</summary>", "[20-22] Oil poured...", "...(8 thinks)"]},
    "visual_window": {"video_start": 20, "video_end": 44},
    "user_input": "<compress_trigger range=\"20-34\"/>"
  },
  "output": "<summary>{\"time_range\":[20,34],\"text\":\"Oil heated, garlic browned, tomato added, seasoning stirred. Entities: stainless pot, wooden spoon.\"}</summary>"
}
```

> Compress 无 think（独立触发）。Summary 只引用被选中 range 的 thinks，不能偷用未选中 thinks。

---

## 6. 质量保证

14 项自动检查（详见 §3.6），核心三条不变：
1. **Recall query 防泄漏**：query 只含问题已知信息 + 可观察锚点，禁止含未知答案值
2. **Compression question-blind**：压缩不能因未来问题特殊保留答案（同类动作需一致处理）
3. **Target-visibility**：只有 teacher caption confidence ≥ 0.7 的 fact 才能造任务

新增：queries_state 时序一致性、轨迹 action 分布、base sample 结构合法性、recall 证据可达性。

---

## 7. RL 阶段 (GRPO)

SFT Phase 5 完成后，在**独立的 RL 视频集**（75 条，不与 SFT 视频重叠）上进行 GRPO 训练。

### 7.1 RL 流程

1. 从 RL 视频池采样 (视频, task) 对（task 的 gold_answer 来自 Pass3）
2. 模型从视频开头 rollout 到 task 所在 chunk，自主决定每步 action
3. 对同一 (视频, task) 生成 G=4 条 rollout（GRPO 需要组内对比）
4. 计算每条 rollout 的 reward（6 个分量加权求和）
5. 组内 advantage = (reward - mean) / std → GRPO loss 更新

### 7.2 Reward 设计

| 分量 | 权重 | 信号 | 已有代码 |
|------|------|------|---------|
| R_format | 0.15 | think/action tag 格式匹配 | `grpo.py: _compute_format_reward` |
| R_action | 0.20 | 选对 action 类型 vs gold_action | `grpo.py: _compute_action_reward` |
| R_correctness | 0.30 | 答�� vs gold_answer（V1-V4 自动验证，V5-V6 用 LLM judge） | `grpo.py: _compute_correctness_reward` |
| R_timing | 0.15 | 在正确 chunk 响应 | `grpo.py: _compute_time_reward` |
| R_think_len | 0.10 | think 长度在 40-60 tok | `grpo.py: _compute_think_length_factor` |
| R_compress | 0.10 | 压缩后 entity retention | **需新增** |

R_compress 计算方式：压缩后从 summary 中能还原多少 entity 和 fact（与 teacher evidence 对比）。

### 7.3 RL 数据量与训练步数

| 指标 | 数值 |
|------|------|
| RL 视频 | 75 条 (短16 + 中50 + 长9) |
| (视频, task) 对 | ~2,050 |
| Group size | 4 |
| Global batch | 16 samples/step |
| Epochs | 2 |
| **总 steps** | **~256** |
| LR | 5e-7 |
| 基模型 | SFT Phase 5 checkpoint |

### 7.4 迁移要点

当前 `grpo.py` 的 `rollout()` 使用旧的 `streaming_video_chat`（多轮 KV cache 格式），需迁移到 `StreamingAgentLoop.step()`（per-timestep re-render），以保证 RL rollout 与 SFT 训练格式一致。

### 7.5 评估指标

| 维度 | 指标 |
|------|------|
| Action Selection | silent/response/recall/compress macro-F1 |
| Recall Necessity | need-recall 判断准确率 |
| Query Quality | recall@1, recall@3 |
| Compression | entity retention, QA-after-compress accuracy |
| Final QA | 按来源分: frame/memory/compressed/recall/unanswerable |
| System | token length p50/p90, latency/chunk, context overflow rate |

---

## 8. 配置常量

```python
THINK_TOKENS = (40, 60)
RECENT_THINKS_TOKEN_BUDGET = 600;  COMPRESS_TRIGGER_RATIO = 0.8  # threshold = 480
STUDENT_MODEL = "Qwen/Qwen3-VL-8B"
COMPRESS_RANGE_MIN = 2;  COMPRESS_RANGE_MAX = 8
MAX_COMPRESSED_SEGMENTS = 5;  SUMMARY_TOKENS = (100, 180);  COMPRESSION_RATIO_MIN = 2.5
VISUAL_WINDOW_CHUNKS = 12  # 24s, 24帧
RECALL_RETURN_FRAMES = 4
MAX_LENGTH = 16384  # 训练 max_length (单样本 ~3500 tok)
PASS1_CONCURRENT = 1024;  PASS2_CONCURRENT = 128;  PASS3A_CONCURRENT = 512;  PASS3C_CONCURRENT = 1024
THINKING = True;  MAX_TOKENS_VISION = 16384;  MAX_TOKENS_TEXT = 60000
# Special Tokens — 完整列表见 §13
```

### 8.1 vLLM 部署方案

统一 thinking=True，`--reasoning-parser qwen3` 分离 thinking/content。
视觉 Pass: max-model-len=65536, gpu-util=0.90, max-num-seqs=16, prefix-caching OFF。
文本 Pass: max-model-len=16384, gpu-util=0.95, max-num-seqs=64, prefix-caching ON。
推荐两套实例按序启动（显存冲突）。OOM 时降 max-num-seqs。

---

## 10. 核心设计约束（不可违反）

以下约束来自多轮迭代的核心决策，任何代码或文档修改都必须遵守：

| # | 约束 | 对应���节 | 代码执行 |
|---|------|---------|---------|
| 1 | 视觉观察 think 每步立即入文本记忆（recall step2 分析 think 不存） | §2.1 | `MemoryState.add_think()` |
| 2 | 文本记忆覆盖时间 > 视觉窗口，两者重叠 | §2.1 | `snapshot()` 中 `visual_window_start` 独立于 `timeline` |
| 3 | 80% token 预算触发压缩（精确 tokenizer） | §2.3 | `should_compress()` + `count_recent_tokens()` |
| 4 | compress range 只覆盖 INPUT 中的 timeline thinks，不含当前 think | §2.1 | `pre_action_thinks = snapshots[chunk_idx]["timeline"]` |
| 5 | 系统执行：先替换→再 append 当前 think | §2.1 | `memory.compress()` 后 think 已在 memory 中 |
| 6 | summary 只能引用 thinks，不能偷看视觉帧 | §2.1 | `verify_summary_provenance()` |
| 7 | C1 teacher 多维评分选最优范围，C2 模型自选 | §2.3 | `score_range_for_compression()` 5 维评分 |
| 8 | 主循环 action 优先级：用户问题 > query 触发 > silent（compress 不在主循环中） | §2.1 | pipeline Pass 3-C |
| 9 | recall_result 不含 teacher_caption、不含未来内容 | §3.5 | `_get_correct_result()` 只用 `observations[ec]['think']` |
| 10 | recall failure/distractor 时 response 内容表达信息不足，不用 gold_answer 强答 | §3.5 | `is_failed_recall` → "信息不足" response + `verify_information_flow()` |
| 11 | post-recall response 输出 think（分析检索结果），格式与所有 action 统一 | §1.3 | step2 有 `<think>` 标签 |
| 12 | query generator 不接收 gold_answer | §6 | `RECALL_QUERY_PROMPT` 仅含 question + visible_context |
| 13 | timeline 中 summary 超 5 段时合并最老两段（≤200 tok） | §2.1 | `MemoryState.compress()` + tokenizer 截断 |
| 14 | merge_compress 有对应训练样本 | §2.1 | `build_merge_compress_sample()` |
| 15 | 压缩比 ≥ 2.5:1 | §8 | `verify_compression_ratio()` |
| 16 | C1/C2 必须拆开训练，不合并 | §2.3 | C1 `range="X-Y"` / C2 `<compress_trigger/>` 独立样本 |
| 17 | teacher 信息只用于选 gold，不进 student 可见路径 | §1.2 | evidence 仅在 `score_range_for_compression` 中使用 |
| 18 | summary 不能包含当前 think 的独有事实 | §2.1 | `verify_summary_no_current_think_leak()` |
| 19 | compressed tag 使用 JSON-inside-tag 格式，不用 XML attribute | §13 | `format_for_prompt()` 输出 `<compressed>{json}</compressed>` |
| 20 | Pass2 必须 question-blind | §3.2 | 无 question/task/gold_answer 参数 |
| 21 | 压缩触发基于 pre-action memory，不含当前 think | §2.1 | `should_compress` 在 `add_think` 之前评估 |
| 22 | 每批数据必须跑 task coverage audit | §4.6 | `audit_task_coverage()` + `task_coverage_report.json` |

---

## 11. 数据源分析与视频选择策略

### 11.1 数据源总览

基于 `video_catalog_30s_plus.csv`（≥30s 全量索引）：VideoMind 395K(114s) + LLaVA 94K(88s) + Koala 67K(32s) + tarsier2 52K(51s) + Koala_raw 26K(547s) + how_to_* 35K(186s) = **总计 668K**，其中 ≥120s 280K。`used_in_streamo=1` 共 192K 条优先复用。

### 11.2 时长门槛与三档划分

最低时长：silent/response 24s → compress 20s → recall 30s → compress+recall 50s → 全覆盖 **≥120s**。
三档：<24s 不可用(~37万) | 30-89s Phase1/2 可用(~30万) | ≥90s 全 Phase 可用(~37万)。

### 11.3 分阶段选择

P1(≥30s,200) + P2(≥60s,200) + C1/C2/P5(≥120s,300) = ~700 视频 → ~33K samples。优先 streamo 已验证视频，按 dataset 分层抽样，同视频不跨 split。

---

## 12. 开放问题

1. **Per-timestep vs 短序列**: 先用 per-timestep baseline，缺连续性时改 3-step 滑动窗口。见 `sft_engineering.md` §3.4。
2. **Think 关键信息覆盖率**: Pass 2 后检查，覆盖率 < 0.6 的重新生成。
3. **RL rollout 迁移**: `grpo.py` 需迁移到 `StreamingAgentLoop.step()`。

---

## 13. SFT 工程集成要点

> 完整设计见 `sft_engineering.md`。样本字段契约见 §5。

**Special Token**：所有 tag 是 special token，JSON 放在 tag 内（不用 XML 属性）。
基础: `<think>` `<action>` `<silent>` `<response>` `<query>` `<recall_result>` 及对应闭合。
新增: `<memory>` `<compressed>` `<queries>` `<answer>` `<visual_window>` `<recalled_frames>` `<user_input>` `<summary>` `<compress_trigger>` 及对应闭合。

**Visual Window**: `video_start/end` 绝对秒数，frames=24，recalled frames 单独加载（4 帧）。

**Phase 分配**: Pass 4 验证后输出 `verified/{video_id}.json`，pipeline 汇总为 `final/train/val/test.jsonl`，按 `phase` 字段过滤。超参见 `sft_engineering.md` §7。
