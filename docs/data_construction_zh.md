# 流视频 Agent 数据构造方案

> 版本: v8.0 | 日期: 2026-04-23
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
  input: recent_thinks 列表 + visual_window
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

### 2.1 记忆状态（单层，模型视角）

每个 timestep，系统从外部拼装一份完整 input 给模型。模型看到四个区：

```
模型看到的完整 input:
  visual_window: [最近 12 chunks 的帧]                ← FIFO，固定大小
  memory:
    compressed_segments: [summary_0, summary_1, ...]  ← 最多 5 段
    recent_thinks: [think_0, think_1, ..., think_N]   ← 未被压缩的
  queries:                                             ← 独立的问答追踪区（v8.0 新增）
    [{question: "...", answers: ["...", "..."]}]       ← 未完结的问题 + 历史回答
  user_input: 当前步的新问题 / compress_trigger / 空
```

**核心设计：文本记忆覆盖时间 > 视觉窗口**。

Think 每步生成后**立即进入** `recent_thinks`，不需要等到 chunk 离开视觉窗口。
这意味着 `recent_thinks` 和 `visual_window` 在近期 chunks 上有**重叠**——近期 chunks
同时有视觉帧（在 visual_window）和文本记忆（在 recent_thinks），而更早的 chunks
只有文本记忆（压缩段或 recent_thinks），视觉帧已滑出。

```
时间线:  [=== compressed ===][=== recent_thinks ===]
                              [== visual_window ==]
         ↑ 最早               ↑ 重叠区域          ↑ 当前
```

这是**正确的**：
- 视觉窗口提供近期的原始帧细节
- 文本记忆提供更长时间的语义摘要
- 两者互补，不是互斥

**压缩状态更新顺序**（不可违反）：

```
模型输入 = pre-action snapshot（不含当前 think）
模型输出 = <think>X</think><action>compress</action><summary>Y</summary>
系统执行：
  1. 对 INPUT 中的 recent_thinks 指定 range 做替换（Y 替换选中段）
  2. 再 append 当前 think X 到 recent_thinks
```

**关键约束**：compress range 只能覆盖 INPUT 中已存在的 recent_thinks，
不能覆盖当前输出的 think。这确保模型只能基于它看到的信息做压缩决策。

**compressed_segments 超过 5 段时**：系统合并最老的两段（截断到 200 token，使用学生 tokenizer），保持 ≤5 段。

**Recall = 系统侧检索**。模型输出 `recall` 后，系统从历史存储中查找相关内容
（包括已被压缩移除的原始 thinks、压缩摘要、历史视频帧），将结果注入下一步 input。

**Summary 信息来源**：模型做压缩时能看到被压缩的 thinks 文本**和**当前 visual_window 中的帧。
- 被压缩的 thinks 覆盖的时间段如果和 visual_window 重叠，模型可以利用帧补充细节，写出更精确的 summary
- 被压缩的 thinks 覆盖的时间段如果已离开 visual_window，模型只能基于 thinks 文本做摘要
- 这和推理时的行为完全一致：推理时模型也同时看到帧和文本

造数据时（Pass2）：`build_compress_request_from_thinks` 自动检测压缩范围和 visual_window 的重叠部分，将重叠帧加入 compress prompt，使 397B 生成的 summary 和推理时模型能看到的信息一致。无重叠时仅用 thinks 文本生成 summary。

**Action 优先级**（主循环 SYSTEM_PROMPT 内）：
```
用户当前新问题 response/recall > query 触发 response > silent
```
compress 不在主循环中，不存在优先级冲突。
post-recall response 在独立的 SYSTEM_PROMPT_POST_RECALL 中，也不存在竞争。
query 触发 = 模型看到 `<queries>` 中的问题且当前帧有新内容时主动 response。

### 2.2 单条训练样本的 input 结构（v6.3 视频在前布局）

```
[System prompt: 协议说明, ~150 tok]                                 ← Zone A
[Visual window: 最近 12 chunks 的视频帧, ~1536 vision tok]          ← Zone B (固定大小)
[Memory block:]                                                     ← Zone C (可变)
  <compressed>{"time_range":[0,20],"text":"summary_0"}</compressed>
  <compressed>{"time_range":[20,40],"text":"summary_1"}</compressed>
  [40-42] think at t=40-42                               ← 未压缩的 think
  [42-44] think at t=42-44
  ...（recent_thinks 可能和 visual_window 重叠）
[Queries: 未完结的问题 + 历史回答]                                    ← Zone Q (v8.0 新增)
  <queries>
    <query t="20">Tell me when basil is added</query>
    <answer t="70">Basil leaves being torn over pot.</answer>
  </queries>
[User input: 当前新问题 / 空]                                      ← Zone D
```

这是学生模型**真实能看到的全部信息**。注意 recent_thinks 和 visual_window 在时间上可能重叠。

**v6.3 变更**：视频在前、文本在后。视频区大小固定（~1536 tok），放在前面使 position 稳定，便于推理时 KV cache 增量复用。文本 think 的 MROPE temporal position 对齐到对应帧的时间戳（详见 `streaming_position_encoding.md` §4, §8）。

### 2.3 压缩触发与范围选择

**触发条件**: `recent_thinks 总 token ≥ RECENT_THINKS_TOKEN_BUDGET × 80%`（~480 tok）

具体行为取决于 think 长度：
- 长 think（~60 tok/条）→ ~8 条就触发（早）
- 中等 think（~50 tok/条）→ ~10 条触发
- 短 think（~40 tok/条）→ 12 条才触发（由 item 硬上限兜底）

**Hysteresis 机制**：压缩后 recent_thinks 总 token 须降至 `RECENT_THINKS_TOKEN_BUDGET × 55%`（~330 tok）以下。
如果压缩一次不够（窗口太短），系统会扩大范围或连续触发。这防止频繁触发压缩导致 summary 质量下降。
见 config.py `COMPRESS_HYSTERESIS_RATIO = 0.55` / `COMPRESS_HYSTERESIS_THRESHOLD = 330`。

造数据时直接加载学生模型 tokenizer 精确计算 token 数（`get_tokenizer()` 自动加载）。
若 tokenizer 不可用则降级为 `chars/4` 估算。

**注意**：
- think 立即进入 recent_thinks，最早的 thinks 可能仍有对应帧在 visual_window 中。这不是问题——压缩只操作文本记忆，视频帧由 FIFO 独立管理。
- **recall step 2 和 compress 的输出不存入 memory**。每个时间步只有主循环的视觉观察 think 进入 memory。

**压缩在两步之间触发**（v8.0 变更）：
压缩不占用时间步，不在主循环 3-action prompt 中。系统在检测到 token 超阈值后，在当前步和下一步之间插入一个 SYSTEM_PROMPT_COMPRESS 调用。模型在这个独立 prompt 下只做压缩，不做观察/回答/检索。

**范围选择（C1: teacher 多维评分）**：
系统触发后，评估所有合法的连续范围（4-8 条），选择**综合得分最低**的范围。
不是简单"信息损失最小"，而是多维度打分：

```python
score(window) =
  + importance_lost          # 实体数、OCR/数字（丢了多少重要信息）
  + event_boundary_penalty   # 打断进行中的事件 → 罚
  - token_saving_gain        # 省的 token 越多越好
  - reconstructability_bonus # 内容重复/简单 → 容易压缩 → 加分

best_range = argmin(score)   # 综合最优
```

> v8.0 变更：删除 `pending_overlap_penalty`。Pass 2 是 question-blind，无 pending/queries，
> 该维度永远为 0。queries 区独立于 memory，不影响压缩决策。

目标不是"丢最少信息"，而是**最大化未来可回答性 / 最小化记忆 regret**。

**合法范围约束**：
1. 连续 4-8 条 `recent_thinks`
2. 尽量是事件边界完整段
3. 压缩后 token gain 足够
4. summary 能保留关键实体、OCR、数量

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

> v8.0 重构：旧阶段 3（Task Planning）和阶段 4（Question-aware Forks）合并为新阶段 3 的三步流程。
> 核心变化：**"问什么"和"什么时候问"彻底分离**，action 由 ask_time × support_time × snapshot 机械推出。

```
阶段 1: Teacher Evidence Graph          [397B + 视频, ~5min]
  ├─ 1-A: 独立 chunk 标注              [397B, 2帧/chunk, 全并行]
  └─ 1-B: 实体链接                     [规则 或 1 call/video]
阶段 2: Question-blind Streaming Rollout [397B + 视频, ~4h]
阶段 3: Task Mining + Sample Generation  [397B + 规则, ~2h]
  ├─ 3-A: Task Card 生成               [397B, ~11 call/video (按 family 分批)]
  ├─ 3-B: Placement 候选生成           [纯程序, 0 call]
  ├─ 3-B2: 轨迹规划                    [纯程序, 0 call]
  └─ 3-C: 轨迹样本生成                 [397B, ~60 call/video]
阶段 4: Verify + Filter                 [规则 + 小模型, ~30min]
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
[压缩段] + [recent_thinks] + [24帧视觉窗口] + prompt
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
    "compressed_segments": [...],
    "recent_thinks": [...],
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

#### Task Card 数据结构

```python
TaskCard = {
    "card_id": "vid001_F2_001",
    "family": "F2",
    "question": "What color is the chef\'s apron?",
    "canonical_answer": "Red",
    "answer_form": "short_exact",         # V1-V6 验证等级
    "choices": null,                       # multiple_choice 时为 ["Red","Blue","White","Green"]
    "support_chunks": [5],                 # 397B 发现答案的 chunk（证据来源）
    "support_facts": ["Person wearing red apron"],
    "visibility_type": "persistent",       # persistent（实体一直在）/ transient（事件瞬时）
    "retention_class": "low",              # high / medium / low
    "answer_keywords": ["red", "apron"],
    "entities_involved": ["person wearing red apron"],
}
```

**新增字段 `visibility_type`**：

| 类型 | 含义 | 例子 | 3-B 怎么用 |
|------|------|------|----------|
| **persistent** | 实体/属性持续可见 | 围裙颜色、锅的位置 | 只选 in_visual 的 ask_time（答案一直在帧里，不可能 recall） |
| **transient** | 事件/数字一闪而过 | 加了一勺盐、屏幕显示 $4.99 | 可选 in_visual + in_history_only（可造 response/recall 对照） |

**397B 在生成 card 时告诉我们"这个答案是持续可见还是瞬时的"，3-B 据此决定哪些 availability 合法。**

#### 生成流程：opportunity 筛选 + 分批 call

**Step 1: 程序预筛选（无 397B）**

```python
def scan_opportunities(evidence):
    """标记每个 chunk 可供哪些 family 使用"""
    opp = {}
    for cap in evidence:
        idx = cap["chunk_idx"]
        families = []
        for fact in cap.get("atomic_facts", []):
            if fact.get("confidence", 0) < 0.7:
                continue
            text = fact["fact"].lower()
            if cap.get("ocr") or any(c.isdigit() for c in text):
                families.append("F1")
            if has_visual_attribute(text):
                families.append("F2")
            if has_count_pattern(text):
                families.append("F3")
        if len(cap.get("visible_entities", [])) >= 2:
            families.append("F4")
        if cap.get("atomic_facts"):
            families.append("E1")
        if cap.get("state_changes"):
            families.append("E2")
        if len(cap.get("visible_entities", [])) >= 3:
            families.append("S1")
        opp[idx] = list(set(families))
    # 跨 chunk: P1(连续 state_changes ≥ 3), C1(同实体不同状态), R1(实体消失重现)
    add_cross_chunk_opportunities(evidence, opp)
    return opp
```

**Step 2: 按 family 分批 397B call（每 family 一次 call）**

不再一次 call 生成全部 25 cards（thinking 会爆）。按 family 分批，每次只给相关 chunks：

```
F1 call: 只输入含 OCR/数字的 ~5 chunks → 输出 3 个 F1 cards
F2 call: 只输入含视觉属性的 ~8 chunks → 输出 4 个 F2 cards
E2 call: 只输入有 state_changes 的 ~6 chunks → 输出 2 个 E2 cards
M1 call: 给全视频摘要 → 输出 2 个 M1 cards（需要看整体才知道怎么做连续解说）
...
```

每次 call 输入短（~500-1000 tokens）、输出短（~300 tokens）、任务单一、thinking 可控。

**每 family 的 prompt 模板不同**：

```
F1 prompt: "根据以下含 OCR/数字的片段，生成 {n} 个关于精确数值的问题。
            优先 binary/number 格式。输出 JSON 数组。"

E2 prompt: "根据以下状态变化，生成 {n} 个关于事件边界的问题。
            格式：event_watch（'看到X告诉我'）或 binary（'X是否已经开始？'）"

M1 prompt: "根据以下视频步骤摘要，生成 {n} 个适合持续解说的问题。
            格式：'描述每一步操作' / '告诉我接下来发生什么'"
```

**每视频 ~11 次 call（11 family），每次 ~500 tok input → 总 ~5.5K tokens。**

### 3.4 阶段 3-B: Placement 候选生成

> 纯程序，零 397B。输出全部合法候选，不做选择（选择在轨迹规划做）。

**输入**: Task Cards + Rollout (snapshots, compression_events)

**输出**: placements/{video_id}.json — 全部 (card, ask_chunk, availability) 候选

#### visibility_type 决定哪些 availability 合法

```python
def compute_placements(card, rollout, evidence):
    """
    对一个 card，输出所有合法的 (ask_chunk, availability) 候选。
    
    visibility_type 决定哪些 availability 类型是合法的：
    - persistent: 只有 in_visual（答案一直在帧里）
    - transient:  in_visual + in_recent_thinks + in_compressed + in_history_only
    """
    candidates = []
    support_end = max(card.support_chunks)
    num_chunks = rollout["num_chunks"]
    
    # ── in_visual: persistent 和 transient 都有 ──
    # persistent: 答案在很长时间内都可见，visual 区间更宽
    # transient:  答案只在 support_chunks 附近可见
    if card.visibility_type == "persistent":
        # 整个视频内任何时刻都可以问（答案一直在帧里）
        # 但只选几个代表位置
        for ask in [num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4]:
            if ask < num_chunks:
                candidates.append({"ask_chunk": ask, "availability": "in_visual"})
    else:
        # transient: 只在 support_chunks 附近的视觉窗口内
        visual_lo = support_end
        visual_hi = min(num_chunks - 1, support_end + VISUAL_WINDOW_CHUNKS - 1)
        if visual_lo <= visual_hi:
            candidates.append({"ask_chunk": (visual_lo + visual_hi) // 2, "availability": "in_visual"})
    
    # ── 以下只有 transient 才有 ──
    if card.visibility_type == "transient":
        bitmap = precompute_retention(card, rollout)
        
        # in_recent_thinks
        exit_visual = support_end + VISUAL_WINDOW_CHUNKS
        compressed_at = find_compression_for(support_end, rollout)
        if compressed_at:
            thinks_hi = compressed_at - 1
        else:
            thinks_hi = num_chunks - 1
        if exit_visual <= thinks_hi and any(bitmap.thinks_retained.values()):
            candidates.append({"ask_chunk": (exit_visual + thinks_hi) // 2, "availability": "in_recent_thinks"})
        
        # in_compressed
        for idx, event in enumerate(rollout["compression_events"]):
            if set(card.support_chunks) & set(event.get("compressed_thinks_chunks", [])):
                if bitmap.summary_retained.get(idx, False):
                    comp_lo = event["trigger_chunk"] + VISUAL_WINDOW_CHUNKS
                    if comp_lo < num_chunks:
                        candidates.append({"ask_chunk": comp_lo, "availability": "in_compressed"})
                    break
        
        # in_history_only
        history_lo = max(exit_visual, support_end + VISUAL_WINDOW_CHUNKS + 3)
        if history_lo < num_chunks:
            candidates.append({"ask_chunk": min(history_lo + 5, num_chunks - 1), "availability": "in_history_only"})
    
    # ── in_future: 仅 E2 ──
    if card.family == "E2":
        support_start = min(card.support_chunks)
        if support_start >= 5:
            candidates.append({"ask_chunk": max(0, support_start - 8), "availability": "in_future"})
    
    return candidates
```

**输出全部候选，不做选择。后续轨迹规划从中挑选。**

#### retention bitmap（仅 transient cards 需要）

persistent cards 不需要 bitmap（答案一直可见）。transient cards 用 retention_class 控制匹配阈值（同之前设计）。

### 3.5 阶段 3-B2: 轨迹规划

> 纯程序。从全部 placement 候选中组合出合理的多问题轨迹。

**输入**: 全部 placements + task cards

**输出**: trajectories/{video_id}.json

```python
def plan_trajectories(cards, placements, num_chunks, target_trajectories=30):
    """
    从 placement 候选中组合多问题轨迹。
    
    每条轨迹 = 1-3 个问题，分布在不同时间点。
    同一条轨迹内的问题至少间隔 5 chunks（10s）。
    
    轨迹类型:
    - 单问题轨迹: 1 个 card + 1 个 ask_time（大多数）
    - 多问题轨迹: 2-3 个 card + 各自的 ask_time（更真实）
    """
    trajectories = []
    used_placements = set()
    
    # 按 availability 多样性排序候选
    all_placements = sorted(placements, key=lambda p: (p["availability"], p["ask_chunk"]))
    
    # Phase 1: 单问题轨迹（保证每个 card 至少有一条）
    for card in cards:
        card_placements = [p for p in all_placements 
                          if p["card_id"] == card.card_id and id(p) not in used_placements]
        if card_placements:
            best = card_placements[0]  # 取第一个（availability 优先排序）
            trajectories.append({
                "trajectory_id": f"traj_{len(trajectories)}",
                "questions": [{"card_id": card.card_id, **best}],
            })
            used_placements.add(id(best))
    
    # Phase 2: 多问题轨迹（更真实的交互）
    remaining = [p for p in all_placements if id(p) not in used_placements]
    # 按 ask_chunk 排序，贪心组合不冲突的 placements
    remaining.sort(key=lambda p: p["ask_chunk"])
    
    current_traj_questions = []
    for p in remaining:
        if len(trajectories) >= target_trajectories:
            break
        if not current_traj_questions:
            current_traj_questions.append(p)
        elif p["ask_chunk"] - current_traj_questions[-1]["ask_chunk"] >= 5:
            current_traj_questions.append(p)
            if len(current_traj_questions) >= 3:
                trajectories.append({
                    "trajectory_id": f"traj_{len(trajectories)}",
                    "questions": [{"card_id": q.get("card_id", ""), **q} for q in current_traj_questions],
                })
                current_traj_questions = []
        else:
            # 冲突，跳过
            continue
    
    return trajectories
```

### 3.6 阶段 3-C: 轨迹样本生成

> 397B 生成 response/recall 文本。只对选中的轨迹执行。

**输入**: 选中的轨迹 + rollout + evidence

**输出**: fork_samples/{video_id}.json

#### 每条轨迹生成哪些样本？

```
轨迹: [
  {card: "围裙颜色?", ask_chunk: 15, availability: "in_visual"},
  {card: "看到basil告诉我", ask_chunk: 30, availability: "in_future"},
]

需要生成的样本:

chunk 15: 问题 1 到达
  input: queries=[], user_input="围裙什么颜色?"
  → 397B 生成 response → "Red"
  样本: {type: "question_arrival", action: "response"}

chunk 16: 问题 1 已答完
  input: queries=[{q:"围裙颜色?", answers:["Red"]}], user_input=""
  → silent（学会不重复回答）
  样本: {type: "post_answer_silent"}

chunk 30: 问题 2 到达
  input: queries=[{q:"围裙颜色?", answers:["Red"]}], user_input="看到basil告诉我"
  → silent（basil 还没出现）
  样本: {type: "event_watch_silent"}

chunk 35: basil 出现
  input: queries=[{q:"围裙颜色?", answers:["Red"]}, {q:"basil?", answers:[]}]
  → 397B 生成 response → "Basil is being torn"
  样本: {type: "event_watch_trigger", action: "response"}

chunk 36: event_watch 已答
  input: queries=[..., {q:"basil?", answers:["Basil is being torn"]}]
  → silent
  样本: {type: "post_answer_silent"}
```

#### M1 多次回答

```
轨迹: [{card: "描述每步操作", ask_chunk: 10, availability: "in_visual", family: "M1"}]

chunk 10: 问题到达 → response "正在切菜"
chunk 15: queries 有答案，无新 state_change → silent
chunk 25: state_change 出现（"开始炒菜"）→ response "开始炒菜"（追加回答）
chunk 30: 无新 state_change → silent
chunk 40: state_change 出现（"加调料"）→ response "加入调料"

M1 response 时机 = queries 区有活跃 M1 问题 + 当前 chunk 有 state_change
```

#### recall 样本

```
轨迹: [{card: "加了多少盐?", ask_chunk: 45, availability: "in_history_only"}]

chunk 45: 问题到达，答案不在任何可见源
  step 1 (SYSTEM_PROMPT): <think>视觉观察</think><action>recall</action><query>...</query>
  step 2 (SYSTEM_PROMPT_POST_RECALL): <think>分析结果</think><action>response</action><response>...</response>
  
  两个 step = 2 个样本（不同 prompt）
```

#### base samples（所有轨迹共享）

```
不带任何问题的 chunk = base rollout 的 silent 样本
  从 60 chunks 中采样 ~20% = ~12 个 silent 样本
  + compress 样本（从 compression_events 取，~3 个）
  这些在所有轨迹间共享，不需要重复造
```

#### 397B 调用量

```
每条轨迹:
  - question_arrival 的 response: 1 call（生成 response 文本）
  - recall 的 query + response: 2 calls
  - M1 followup: 每次 1 call
  平均 ~2 calls/轨迹

30 轨迹 × 2 calls = ~60 calls/视频
300 视频 = ~18,000 calls
```

### 3.6 阶段 4: Verify + Filter

#### 5 类验证

**第一类：信息流合法性（最重要）**
```
✓ 当前 action 是否只依赖当前可见信息
✓ response 有没有用到未来信息
✓ recall query 有没有包含未知答案
✓ compression summary 有没有因未来问题而特殊保留答案
✓ 问题出现前的 memory 是否 question-blind
✓ recall_result 是否来自学生可访问内容
```

**第二类：Action Minimality**
```
✓ 如果当前帧/obs/summary 能答 → 不应标 recall
✓ 如果需要历史证据 → 不应标 response
✓ 如果无证据 → 不应强答
```

**第三类：Grounding**
```
✓ think 是否被当前帧支持
✓ summary 是否只含原始 thinks 中的信息
✓ response 是否被 support evidence 支持
✓ 无声音/气味/情绪/意图推断
```

**第四类：格式与长度**
```
✓ think 40-60 tokens
✓ response 长度匹配问题类型
✓ query JSON 合法，无答案泄漏
✓ summary JSON 合法，压缩比合理
```

**第五类：难度标注**
```
标注: current_visible_response / memory_response / recall_required / unanswerable
用于后续按 phase 采样
```

---

## 4. 问题家族与答案验证等级

> v8.0 重构：不再按 action 分类（response_from_frames, recall, ...），
> 改为按**问题本体**分类。同一问题在不同 ask_time 自动产生不同 action。

### 4.1 问题家族 (Question Families)

| Family | 名称 | 问什么 | retention_class | 优先 answer_form | 一版包含 |
|--------|------|--------|----------------|-----------------|---------|
| **F1** | OCR / Number | 价格/时间/分数/按钮文案/屏幕字段/包装数字 | low | number, short_exact | 是 |
| **F2** | Fine-Grained Attribute | 颜色/材质/形状/服饰/器具细节 | low | binary, multiple_choice | 是 |
| **F3** | Count / Quantity | 个数/份数/次数/精确数量 | low-medium | number | 是 |
| **F4** | Spatial Relation | 左右前后/相对位置/布局 | medium | binary, multiple_choice | 是 |
| **E1** | Local Action / State | 当前 2-6s 在做什么/刚发生什么 | high | short_exact, binary | 是 |
| **E2** | State Change Boundary | 什么时候开始/停止/变成 | medium | binary, short_exact | 是 |
| **P1** | Procedure / Step Order | 第几步/先后顺序/某步是否完成 | medium | number, multiple_choice | 是 |
| **C1** | Cross-Time Comparison | 和之前比有什么变化/前后差异 | low | binary, short_factual | 是 |
| **R1** | Entity Re-identification | 之前那个物体/人是不是同一个 | medium | binary | 是 |
| **S1** | Local Summary / Description | 对一个短时间段做概括 | high | descriptive | 是 |
| **M1** | Continuous Commentary | 持续解说/事件监控，可多次回答 | high | short_factual, descriptive | **是** |
| **D1** | Follow-up / Multi-turn | 追问刚才提到的细节 | — | — | **二版** |

> **M1 进一版**（v8.0 变更）：推理时用户问一次后模型必须在后续 chunk 持续判断是否有新内容，
> 这是 `<queries>` 机制的核心场景。M1 通过 queries 区的历史 answer 避免重复回答。
> D1 仍推迟——追问需要对话状态管理，复杂度更高。

#### Retention Class

| Level | 含义 | 举例 |
|-------|------|------|
| **high** | think 几乎一定会记住 | 当前动作、刚发生的事 |
| **medium** | think 记粗粒度，压缩后可能保留 | 步骤顺序、空间布局 |
| **low** | think 大概率不记精确值，压缩后几乎必丢 | 精确数字、细粒度颜色/材质、OCR |

retention_class 不决定 action（action 由 availability 决定），用于 3-B 的难度评估和统计分析。

### 4.2 答案验证等级（Verifiability Hierarchy）

> **核心原则：优先构造可自动验证的问题，再逐步扩展到开放性问答。**
> 可验证问题直接支撑 RL reward 信号（GRPO R_correctness），开放性问题依赖 LLM judge。

| 等级 | answer_form | 验证方式 | 举例 | RL 可用 | 一版占比目标 |
|------|------------|---------|------|--------|------------|
| **V1** | `binary` | exact match (yes/no) | "围裙是红色的吗？" → "Yes" | **是** | ≥15% |
| **V2** | `multiple_choice` | exact match (A/B/C/D) | "围裙是什么颜色？A.红 B.蓝 C.白 D.绿" → "A" | **是** | ≥20% |
| **V3** | `number` | 数值容差匹配 (±10%) | "加了几勺盐？" → "1" | **是** | ≥10% |
| **V4** | `short_exact` | exact match (1-3 词) | "围裙什么颜色？" → "Red" | **是** | ≥15% |
| **V5** | `short_factual` | keyword overlap + LLM judge | "厨师在做什么？" → "Stirring the sauce" | 半自动 | ≤25% |
| **V6** | `descriptive` | 仅 LLM judge | "描述当前场景" → (长文) | 否 | ≤15% |

**一版整体目标：V1-V4 可自动验证问题 ≥ 60%。**

**每个 family 的验证等级分布指导**：

| Family | V1 binary | V2 choice | V3 number | V4 exact | V5 factual | V6 desc |
|--------|-----------|-----------|-----------|----------|-----------|---------|
| F1 OCR | — | — | **50%** | **50%** | — | — |
| F2 Attr | **30%** | **40%** | — | **20%** | 10% | — |
| F3 Count | — | — | **80%** | — | 20% | — |
| F4 Spatial | **40%** | **40%** | — | — | 20% | — |
| E1 Action | **20%** | **30%** | — | **30%** | 20% | — |
| E2 Change | **50%** | **30%** | — | — | 20% | — |
| P1 Step | — | **40%** | **30%** | — | 30% | — |
| C1 Compare | **50%** | **30%** | — | — | 20% | — |
| R1 Re-id | **80%** | — | — | — | 20% | — |
| S1 Summary | — | — | — | — | 30% | **70%** |

**Multiple choice 选项生成规则**：
- 正确答案 + 3 个干扰项
- 干扰项来自同视频的其他 evidence（相同 family 的其他 fact）
- 如果不够，用 category-level 的通用干扰项（如颜色类: red/blue/green/white）
- 选项随机打乱，answer 记录选项字母（A/B/C/D）

### 4.3 Action 类型（由 availability 自动推导）

| Availability | Gold Action | 说明 |
|-------------|-------------|------|
| `in_visual` | response | 证据在帧中 |
| `in_recent_thinks` | response | 证据在文本记忆中 |
| `in_compressed` | response | 证据在压缩摘要中 |
| `in_history_only` | recall | 证据不在任何可见源中 |
| `in_future` | silent | 事件还没发生（仅 E2 允许） |

> 如果 compressed_summary 已包含答案 → response，不是 recall！

Recall 的必要条件（不变）：
1. 答案不在当前视觉帧中
2. 答案不在 recent_thinks 中
3. 答案不在 compressed_summaries 中
4. 答案存在于历史 thinks / frames 中

Compress 行为不在问题系统内，直接从 rollout 压缩事件继承。

### 4.4 Queries 机制（替代旧 pending）

> v8.0 变更：删除 `<pending>` 机制，引入独立的 `<queries>` 区。
> 解决旧 pending 的三个问题：模型不知道自己答过什么、不能避免重复、不支持多次回答。

**核心设计：queries 区存问题原文 + 所有历史回答，模型自己判断有没有新内容。**

```
<queries>
  <query t="20">Tell me when basil is added</query>
  <answer t="70">Basil leaves being torn over pot.</answer>
  <answer t="80">Basil fully mixed into sauce.</answer>
</queries>
```

**模型的决策依据**：
- 看到 query + 无历史 answer + 当前帧无相关 → **silent**
- 看到 query + 无历史 answer + 当前帧有相关 → **response**（首次回答）
- 看到 query + 有历史 answer + 当前帧有**新**信息 → **response**（追加回答）
- 看到 query + 有历史 answer + 当前帧**无新**信息 → **silent**

**模型不需要元数据（responded 计数等），它通过阅读历史 answer 来判断什么是"新"的。**

**系统管理 query 生命周期**：
1. 用户提出问题 → 系统判断是即时可答还是需要持续观察
   - 即时可答：不进 queries，直接在当步 response
   - 需要观察：加入 queries（answer 列表为空）
2. 模型输出关联某 query 的 response → 系统追加 answer
3. 删除时机：用户取消、或视频结束（一版不主动删除）

**覆盖的场景**：

| 场景 | 旧方案 | 新方案 |
|------|--------|--------|
| E2 event_watch "看到 X 告诉我" | `<pending>` 3 样本 | query + 0/1 条 answer |
| M1 连续解说 "描述每步操作" | 不支持（推到二版） | query + N 条 answer |
| recall 后回答 | pending + awaiting_recall | 不变（recall 不跨 chunk，不用 queries） |

**answer 积累过多时的处理**：一版先不做 answer 压缩，实际 token 消耗预估：
120s 视频 × M1 每 ~10s 回答一次 → ~12 条 answer × ~30 tok = ~360 tok。在 4K input budget 内可控。

**Unanswerable → 极简处理**：
- 不进 task card 系统，在 3-C 末尾追加少量模板样本
- 统一行为：response + "从画面中无法确定这个信息"
- 每视频 2-3 条

**"未来不会发生的问题" → 不做**：
- 直接丢弃，不增加模型行为复杂度

### 4.5 从旧 task type 到新 family 的映射

| 旧 task type | 去向 | 说明 |
|-------------|------|------|
| `response_from_frames` | 删除 | family × in_visual 自动覆盖 |
| `response_from_memory` | 删除 | family × in_recent_thinks 自动覆盖 |
| `recall_*` (5 个 sub_type) | 删除 | F1-R1 吸收: ocr→F1, visual_detail→F2, procedural→P1, state_change→E2/C1 |
| `compress_recall` | 删除 | family × in_history_only (post-compression) 自动覆盖 |
| `compress_response` | 删除 | family × in_compressed 自动覆盖 |
| `unanswerable` | 大幅简化 | 不分 3 类，极简模板 |
| `pending_event_watch` | 简化为 E2 | 只保留 event_watch 一种 silent 触发 |
| `compress` | 不变 | 直接从 rollout 继承 |

### 4.6 各 Phase 配比

| Phase | Silent | Response | Recall | Compress | Special |
|-------|--------|----------|--------|----------|---------|
| 1 (协议) | 65% | 35% | 0% | 0% | — |
| 2 (Recall) | 50% | 20% | 20% | 0% | 10% |
| C1 (压缩) | 45% | 18% | 15% | 12% | 10% |
| C2 (自选) | 42% | 18% | 15% | 15% | 10% |
| 5 (混合) | 45% | 20% | 15% | 10% | 10% |

**分布按 episode-level 控制**，不是人工拉 turn-level 比例。

---

## 5. 训练样本格式

### 5.1 一条训练样本

```json
{
  "sample_id": "vid001_t60_recall_query_42",
  "video_id": "vid001",
  "sample_type": "recall_query",
  "chunk_idx": 30,
  "phase": "C1",
  
  "input": {
    "system": "You are a streaming video agent...",
    "memory": {
      "compressed": [
        {"time_range": [0, 20], "text": "Chef(red apron) prepared workspace..."},
        {"time_range": [20, 40], "text": "Tomatoes diced, added to pot, seasoning added..."}
      ],
      "recent_thinks": [
        "[40-42] Sauce simmering, chef covers pot with glass lid.",
        "[42-44] Chef retrieves basil from refrigerator.",
        "[44-46] Basil torn over pot, green leaves on sauce surface.",
        "[46-48] Chef wipes hands on towel, steps back from stove."
      ]
    },
    "visual_window": {
      "video_start": 36, "video_end": 60, "frames": 24,
      "chunk_indices": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
      "frame_indices": [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    },
    "user_input": "How much salt did the chef add?"
  },
  
  "output": "<think>Chef adjusting burner dial, checking sauce consistency through glass lid.</think><action>recall</action><query>{\"query\":\"seasoning amount pot cooking\",\"time_range\":\"0-48\"}</query>",
  
  "metadata": {
    "gold_answer": "Approximately 1 teaspoon",
    "gold_action": "recall",
    "action_minimality": {
      "answer_in_frames": false,
      "answer_in_recent_obs": false,
      "answer_in_compressed": false,
      "answer_in_historical": true,
      "note": "evidence at t=28-30 已离开 visual_window(36-60), summary 只写了 seasoning added 没写数量"
    },
    "visibility_matrix": {
      "compressed_mentions_salt": true,
      "compressed_mentions_amount": false,
      "why_recall": "Summary says 'seasoning added' but not the amount. Need historical frames."
    },
    "support": {
      "evidence_time": [28, 30],
      "evidence_fact": "approximately 1 teaspoon salt from small bowl",
      "source": "teacher_caption"
    },
    "leakage_checks": {
      "query_contains_answer": false,
      "summary_has_exact_answer": false
    }
  }
}
```

> **v5.0→v5.1 修正**: 将 ask_time 从 t=44 改为 t=60，确保 evidence(28-30) 已离开
> visual_window(36-60)。query 不再包含 answer_topic，仅基于问题 + 可见 memory 生成。

### 5.2 Recall 后的 follow-up 样本

```json
{
  "sample_id": "vid001_t60_post_recall",
  "input": {
    "system": "...",
    "memory": {
      "compressed": ["...same as above..."],
      "recent_thinks": ["...same as above..."]
    },
    "recall_context": {
      "original_question": "How much salt did the chef add?",
      "recall_query": {"query": "seasoning amount pot", "time_range": "0-48"}
    },
    "visual_window": {
      "video_start": 36, "video_end": 60, "frames": 24
    },
    "recalled_frames": {
      "time_range": [28, 30], "n_frames": 4, "source": "historical_frames",
      "frame_indices": [28, 29, 30, 31]
    },
    "recall_result": {
      "source": "historical_frames",
      "time": "28-30",
      "text_content": "Retrieved frames from t=28-30s.\nText memory: [28-30] Chef sprinkles seasoning from small white bowl into pot."
    },
    "user_input": "Continue following the protocol to respond."
  },
  
  "output": "<think>Retrieved frames show white granular seasoning from small bowl at t=28-30s, approximately one teaspoon quantity.</think><action>response</action><response>Based on the retrieved frames, the chef added approximately one teaspoon of white granular seasoning from a small bowl.</response>"
}
```

> **v5.0→v5.1→v5.2 修正**:
> - input 必须包含 `recall_context` 字段，携带原始问题和 recall query（v8.0: 替代旧 pending）
> - post-recall response **输出 think**（分析检索结果，20-40 tokens），格式与其他 action 统一
> - response 避免确定说 "salt"（仅可见 white granular seasoning），除非有 OCR/上下文支持
> - recall failure / distractor 时，response 内容应表达"信息不足"（不能用 gold_answer 强答）

### 5.3 Compress 样本

```json
{
  "sample_id": "vid001_t44_compress",
  "input": {
    "system": "...",
    "memory": {
      "compressed": [{"time_range": [0, 20], "text": "..."}],
      "recent_thinks": [
        "[20-22] Olive oil poured into stainless pot on right burner.",
        "[22-24] Minced garlic added to oil, small bubbles forming around garlic pieces.",
        "[24-26] Garlic pieces turning golden brown in oil.",
        "[26-28] Tomato quarters placed into pot from cutting board.",
        "[28-30] Chef sprinkles white granular seasoning from small bowl into pot.",
        "[30-32] Wooden spoon stirring contents in circular motion.",
        "[32-34] Sauce surface showing gentle bubbling, reddish color.",
        "[34-36] Chef places glass lid on pot.",
        "[36-38] Chef walks toward refrigerator at left side.",
        "[38-40] Chef pulls out fresh basil bunch from refrigerator shelf."
      ]
    },
    "visual_window": {"video_start": 20, "video_end": 44},
    "user_input": "<compress_trigger range=\"20-34\"/>"
  },
  
  "output": "<think>Chef tears basil leaves near covered pot, green fragments on counter surface.</think><action>compress</action><summary>{\"time_range\":[20,34],\"text\":\"[20-28] Oil heated in stainless pot, garlic browned, tomato quarters added. [28-34] White seasoning from small bowl added, stirred with wooden spoon, sauce reddish and bubbling. Entities: stainless pot(right burner), wooden spoon.\"}</summary>"
}
```

> **v5.0→v5.1 修正**:
> - think 改为当前帧的视觉观察（不是 "System triggered memory compression."）
> - 去掉 "sizzling"（声音），改为可见的 "small bubbles forming"
> - thinks 增加至 40-60 tokens 长度，包含更多视觉细节

---

## 6. 质量保证

### 6.1 Recall Query 防泄漏

```
问题: "How much salt did the chef add?"
答案: "approximately 1 teaspoon"

允许的 query: {"query": "salt seasoning amount pot", "time_range": "20-40"}
  ✓ 只包含问题中已知信息 + 可观察检索锚点

禁止的 query: {"query": "salt 1 teaspoon sauce", "time_range": "20-40"}
  ✗ 包含了未知答案值 "1 teaspoon"
```

### 6.2 Compression Question-blind 检查

```
如果未来 t=44 会问 "How much salt?"
那么 t=30 时的 compression (question-blind):

允许的 summary: "seasoning added from small white bowl"
  ✓ 没有因为未来问题而特殊保留 "1 teaspoon"

禁止的 summary: "salt added approximately 1 teaspoon from small bowl"  
  ✗ 过度保留了数量细节（如果其他类似动作没有保留数量，则不一致）
  
判断标准: 同类型动作（加调料）是否在其他地方也保留了同等细节？
如果是 → 允许（通用 policy 保留数量）
如果否 → 可能被未来问题污染
```

### 6.3 Target-Visibility Check

```
Teacher caption 标注: "timer reads 08:30" (confidence 0.55)
Student 训练分辨率: min_pixels=100352 (~317×317)

在该分辨率下，小字体 OCR 可能不可读 → 不适合做 OCR 类任务
→ 只有 confidence >= 0.7 的 fact 才能用来造任务
```

### 6.4 数据对象完整结构

每条样本存储完整 provenance（用于 debug，不用于训练）：

```json
{
  "sample": { "input": {...}, "output": "..." },
  "metadata": {
    "task_type": "...",
    "gold_action": "...",
    "gold_answer": {...},
    "action_minimality": {...},
    "visibility_matrix": {...},
    "leakage_checks": {...},
    "support_set": [...],
    "quality_scores": {"grounding": 0.91, "difficulty": "medium"}
  }
}
```

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
# Think
THINK_TOKENS = (40, 60)                # think 长度

# 压缩
RECENT_THINKS_TOKEN_BUDGET = 600        # recent_thinks 总 token 预算
COMPRESS_TRIGGER_RATIO = 0.8            # 80% token 预算触发
COMPRESS_TOKEN_THRESHOLD = 480          # = 600 × 0.8
STUDENT_MODEL = "Qwen/Qwen3-VL-8B"     # 学生模型 tokenizer（精确计算 token）
COMPRESS_RANGE_MIN = 4                  # C1 最少压缩 4 条
COMPRESS_RANGE_MAX = 8                  # C1 最多压缩 8 条
MAX_COMPRESSED_SEGMENTS = 5             # 最多 5 段，超限合并最老两段
SUMMARY_TOKENS = (100, 180)             # summary 长度
COMPRESSION_RATIO_MIN = 2.5             # 最低可接受压缩比（验证用），典型 ~3.5

# 视觉
VISUAL_WINDOW_CHUNKS = 12              # 24s, 24帧
RECALL_RETURN_FRAMES = 4               # 4s 历史帧

# 上下文
MAX_LENGTH = 16384                     # 训练 max_length (但单样本 ~3500 tok)

# 并发（所有 Pass 统一 64 路，由 safe_concurrency_for_pass() 动态 clamp）
PASS1_CONCURRENT = 1024                # Evidence Graph (chunk 全并行，2帧/请求)
PASS2_CONCURRENT = 64                  # Rollout
PASS3_CONCURRENT = 64                  # Task Mining + Sample Gen (3-A/B/C)

# 397B（Qwen3.5 /no_think 无效，所有 Pass 均 thinking=True）
# max_tokens = thinking + content 总预算，防止 thinking 阶段截断导致无有效输出
# 任务实际输出远小于此值（evidence ~1K, obs ~128, summary ~512, tasks ~2K, forks ~512）
THINKING = True                        # 所有 Pass（见 qwen35_output_format_analysis.md）
MAX_TOKENS_VISION_PASSES = 16384       # Pass 1/2: 视觉 Pass，受 GPU KV cache 限制
MAX_TOKENS_TEXT_PASSES = 60000         # Pass 3-A/C: 纯文本 Pass (= max_model_len 65536 - input ~4K - margin)

# Special Tokens（SFT 代码 init_processor 中添加）
SPECIAL_TOKENS_BASE = [
    "<silent>", "<response>", "<think>", "</think>",
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
]
SPECIAL_TOKENS_PER_TIMESTEP = [
    "<compressed>", "</compressed>",
    "<queries>", "</queries>",       # v8.0: 替代 <pending>
    "<query>", "</query>",
    "<answer>", "</answer>",
    "<compress_trigger>", "</compress_trigger>",
    "<summary>", "</summary>",
]
```

### 8.1 vLLM 部署方案（两套配置）

各 Pass 对 vLLM 的需求差异：

| Pass | 输入类型 | Thinking | max_tokens（含 thinking） | 任务实际��出 | 并发 | 单请���图片 |
|------|---------|----------|--------------------------|-------------|------|-----------|
| **Pass 1-A** Evidence | 视觉 (2 帧) | ON | 16384 | ~1K (JSON) | 1024 | 2 帧 |
| **Pass 1-B** Entity Link | 纯文本（或规则） | — | — | — | — | 0 |
| **Pass 2a** Observation | ���觉 (24 帧) + 记忆 | ON | 16384 | ~128 (think) | 64 | 24 帧 |
| **Pass 2b** Compress | 纯文本 | ON | 16384 | ~512 (summary) | 64 | 0 |
| **Pass 3-A** Cards | 纯文本 | ON | 60000 | ~2K (cards JSON) | 64 | 0 |
| **Pass 3-B** AskTime | 纯程序 | — | — | — | — | — |
| **Pass 3-C** Samples | 纯文本 | ON | 60000 | ~512 (output) | 64 | 0 |
| **Pass 4** Verify | 规则检查，无 LLM | — | — | — | — | — |

> **注**：Qwen3.5 �� `/no_think` 前缀实测无效（thinking 内容混入 content），
> 因此所有 Pass 统一 thinking=True，由 `--reasoning-parser qwen3` 在 vLLM 侧分离
> thinking 到 `reasoning_content`，`content` 为干净输出。
>
> **max_tokens 与任务输出的区别**：`max_tokens` 是 vLLM 生成上限，包含 thinking token + content token 总量。
> Qwen3.5 的 thinking 可达数千到数万 token，因此 max_tokens 必须远大于任务实际输出，
> 否则会在 thinking 阶段就截断，导致无法产出有效 content。
> 视觉 Pass 设 16384（受 GPU KV cache 限制），纯文本 Pass 设 60000（= max_model_len 65536 - 输入 ~4K - margin）���

**关键差异**：视觉 Pass（1-A, 2a）需要多模态处理能力。Pass 1-A 每请求仅 2 帧（~100KB），可高并发；
Pass 2a 每请求 24 帧（~1.2MB），需要更大 `max-model-len` 和更低并发；纯文本 Pass（2b, 3-A/C）只有 ~1K token 输入。

| 参数 | 视觉 Pass (1, 2a) | 文本 Pass (2b, 3, 4) |
|------|-------------------|---------------------|
| `--limit-mm-per-prompt` | `image=24`（Pass 2a 需要；Pass 1-A 只用 2 帧） | 不需要 |
| `--max-model-len` | `65536`（容纳帧 token） | `16384` 即可 |
| `--gpu-memory-utilization` | `0.90`（图片占显存） | `0.95`（纯文本更高效） |
| `--max-num-seqs` | `16`（图片大，限制并发） | `64`（文本小，可加大） |
| `--enable-prefix-caching` | OFF（图片变化��，命中率低） | ON（system prompt 共��） |

**方案 A：两套 vLLM 实例（推荐）**

视觉 Pass 和文本 Pass 分别启动不同配置的 vLLM 实例，最大化各自吞吐。
pipeline 按 pass 切换 `--api_base`。

```bash
# 实例 1: 视觉模式 (Pass 1-A + Pass 2a)
# 需要多模态支持；Pass 1-A 只需 2 帧/请求，可开高并发
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt image=24 \
  --enforce-eager \
  --trust-remote-code \
  --port 8000

# 实例 2: 纯文本模式 (Pass 2b + Pass 3-A/C)
# 无需多模态，短 context，高并发
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --trust-remote-code \
  --port 8001
```

**方案 B：单实例（GPU 不足时）**

用视觉模式配置跑所有 Pass。文本 Pass 吞吐受限但功能正常。

```bash
# 单实例: 视觉��式兼容所有 Pass
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt image=24 \
  --enforce-eager \
  --trust-remote-code \
  --port 8000
```

**注意事项**:
- 两套实例不能同时运行在同一组 GPU 上（显存冲突）。建议 Pass 1+2a 跑完后关闭实例 1，启动实例 2 跑 Pass 3+4。
- `--enforce-eager` 在视觉模式下可能必需（避免 CUDA graph 与动态图片 shape 冲突）。
- 如果 OOM，视觉 Pass 优先降 `--max-num-seqs` 到 12，再降到 8。
- 纯文本 Pass 的 `--max-num-seqs 64` 可以更激进，因为每请求只用 ~1-3K token。
- `/no_think` 前缀在 prompt 层控制，不需要改 vLLM 启动参数。Thinking ON/OFF 是请求级别控制。

**各 Pass 上下文安全估算**（input + thinking + content，区分 thinking 与有效输出）：

```
                输入        thinking(估计)  有效输出     单请求合计
Pass 1-A:      ~1.5K tok   ~2-5K           ~1K         ~8K      × 1024 并发 (chunk 全并行)
Pass 2a:       ~10K tok    ~2-5K           ~128        ~15K     × 64 并发
Pass 2b:       ~650 tok    ~2-5K           ~512        ~6K      × 64 并发
Pass 3-A:      ~4K tok     ~2-10K          ~2K         ~16K     × 64 并发 (1 call/video)
Pass 3-B:      纯程序，无 397B 调用
Pass 3-C:      ~2K tok     ~2-5K           ~512        ~8K      × 64 并发 (~20 call/video)

由 safe_concurrency_for_pass() 根据 batch_token_budget (2M) 动态 clamp 并发数。
Thinking 长度变化大（2K-10K+），估算取保守值。
max_tokens 必须覆盖 thinking + content 总量，否则 thinking 截断后无有效输出。
```

---

## 9. 补充设计（解决已知缺口）

### 9.1 Queries 区设计（v8.0，替代旧 pending）

> v8.0: 删除 `<pending>` 机制，引入独立的 `<queries>` 区。
> Queries 区存问题原文 + 历史回答，模型通过阅读历史回答判断是否有新内容。

#### 为什么替代 pending

旧 pending 的三个问题：
1. 模型不知道自己之前回答了什么 → 无法避免重复
2. 不支持多次回答（M1 连续解说）
3. 推理时必须每步都有问题可见（不能"在正确时间点重新注入"）

#### Queries 区在 input 中的位置

```
[Visual window: 帧]                                        ← Zone B
[Memory block:]                                            ← Zone C
  <compressed>...</compressed>
  [recent thinks]
[Queries:]                                                 ← Zone Q（独立于 memory）
  <queries>
    <query t="20">Tell me when basil is added</query>
    <answer t="70">Basil leaves being torn over pot.</answer>
  </queries>
[User input: 新问题 / 空]                                   ← Zone D
```

#### 训练样本类型（4 种组合）

```
1. query 在 queries 区，无历史 answer，当前帧无相关内容
   → silent

2. query 在 queries 区，无历史 answer，当前帧有相关内容
   → response（首次回答）

3. query 在 queries 区，有历史 answer，当前帧有新信息
   → response（追加回答，模型对比历史 answer 判断"新"）

4. query 在 queries 区，有历史 answer，当前帧无新信息
   → silent
```

#### E2 event_watch 示例

```
chunk 20 input:
  queries: []
  user_input: "Tell me when basil is added"
  → output: silent（系统将问题加入 queries）

chunk 25 input:
  queries: [{query: "Tell me when basil is added", answers: []}]
  → output: silent（basil 没出现）

chunk 35 input:
  queries: [{query: "Tell me when basil is added", answers: []}]
  当前帧: basil 出现
  → output: response "Basil is being torn over the pot."
  → 系统追加 answer

chunk 40 input:
  queries: [{query: "...", answers: ["Basil is being torn over the pot."]}]
  当前帧: basil 被搅拌进酱汁
  → output: response "Basil fully mixed into sauce."（新信息）
```

#### M1 连续解说示例

```
chunk 10 input:
  user_input: "描述后面每一步操作"
  → output: silent（系统加入 queries）

chunk 15 input:
  queries: [{query: "描述后面每一步操作", answers: []}]
  → output: response "正在切菜"

chunk 25 input:
  queries: [{query: "...", answers: ["正在切菜"]}]
  → output: response "开始炒菜"（新步骤）

chunk 30 input:
  queries: [{query: "...", answers: ["正在切菜", "开始炒菜"]}]
  → output: silent（还在炒，没有新步骤）
```

#### 与 recall 的关系

Recall 的 "pending" 和 queries 是不同的东西：
- Recall pending = 同一 chunk 内的两个样本（query → system recall → response），不跨 chunk
- Queries = 跨 chunk 的持久化问答追踪

Recall 不进 `<queries>` 区。recall_result 仍然通过 `<recall_result>` 注入 input。

#### Special tokens 变更

```
删除: <pending> </pending>
新增: <queries> </queries> <query> </query> <answer> </answer>
```

### 9.2 Recalled Frames 的精确训练格式

Recall 返回帧如何在 input 中呈现，必须和正常视觉窗口区分：

```
[Visual window: 最近 12 chunks 帧]                        ← Zone B, 正常窗口
[Recalled frames: 4帧, time=28-32, source=historical]    ← Zone B 追加, 明确标注为 recalled
[Memory block: ...]                                       ← Zone C
[User input: "Continue following the protocol."]          ← Zone D
```

**关键**：
- Recalled frames 有时间戳标注和 source 标注
- 防止模型把 recalled frames 当作"当前画面"
- 下一个 timestep 中 recalled frames 不再出现

### 9.3 检索方案设计

**线上检索系统**（推理时使用）：

```python
def retrieve(query, memory_state, historical_frames_db):
    """多路检索 + 分数归一化"""
    
    # 路径 1: 文本记忆检索 (BM25)
    text_candidates = bm25_search(
        query["query"], 
        corpus=memory_state["recent_thinks"] + 
               [s["text"] for s in memory_state["compressed_segments"]]
    )
    
    # 路径 2: 视觉帧检索 (CLIP embedding)
    frame_candidates = clip_search(
        query["query"],
        frame_index=historical_frames_db,
        time_filter=parse_time_range(query["time_range"])
    )
    
    # 分数归一化 (min-max per path, then weighted merge)
    text_scores = normalize(text_candidates.scores)    # [0, 1]
    frame_scores = normalize(frame_candidates.scores)  # [0, 1]
    
    # 加权合并
    merged = merge_candidates(
        text=(text_candidates, text_scores, weight=0.4),
        frame=(frame_candidates, frame_scores, weight=0.4),
        time_bonus=(time_overlap_bonus, weight=0.2),
    )
    
    return merged.top_k(4)  # 返回 top-4
```

**训练时检索模拟**：不使用在线系统，而是基于已知 support evidence 模拟结果：
- 70% oracle: 正确 evidence 在 rank 1
- 20% noisy: 正确在 rank 2-4，rank 1 是 distractor
- 5% all-wrong: 只有 distractors
- 5% failure: 返回空

**Distractor 来源**：
- 同视频时间相近但内容不同的 think/frame
- 同视频中相似实体但不同时间的 think
- 文本相似但语义不同的片段

### 9.4 补充负样本任务（v8.0 简化）

> v8.0: 大幅简化。Unanswerable 不进 task card 系统，在 3-C 末尾追加少量模板。
> 歧义、纠正等复杂交互任务推迟到二版。

| 类型 | 描述 | 正确行为 | 占比 | 一版是否包含 |
|------|------|---------|------|------------|
| **不存在内容** | 用户问视频中从未出现的事物 | response: "从画面中无法确定" | 2% | 是（少量模板） |
| **Recall 返回错误** | 检索结果不含答案 | response（内容表达信息不足） | 3% | 是（recall 噪声模拟） |
| **记忆与画面冲突** | 画面变了但记忆还是旧的 | response 基于当前画面 | 2% | 是（天然产生） |
| **多候选歧义** | "哪个人？"但有多个人 | response(clarify) | — | 否，二版 |
| **用户纠正** | "不对，我问的是另一个" | response: 重新回答 | — | 否，二版 |
| **计数任务** | "出现了几次" | response: 基于 thinks 计数 | — | F3 family 覆盖 |

### 9.5 Compression 自适应长度

**问题**：固定 10 条 × 50 tok = 500 tok 输入，输出 100-180 tok，比率变化大。

**解决**：根据输入内容复杂度自适应：

```python
def determine_summary_length(thinks_to_compress):
    """根据内容复杂度决定 summary 长度"""
    n_entities = count_unique_entities(thinks_to_compress)
    has_ocr = any("ocr" in obs for obs in thinks_to_compress)
    has_interaction = any("Q:" in obs or "response" in obs for obs in thinks_to_compress)
    n_state_changes = count_state_changes(thinks_to_compress)
    
    base = 80  # 最短
    base += n_entities * 10   # 每个实体 +10 tok
    base += 30 if has_ocr else 0
    base += 40 if has_interaction else 0
    base += n_state_changes * 8
    
    return min(base, 250)  # 上限 250 tok
```

**效果**：
- 纯观察（少实体）：80-100 tok summary
- 复杂场景（多实体+OCR+交互）：180-250 tok summary
- 保证压缩比 ≥ 2:1

### 9.6 Compression 与 Queries 的关系

> v8.0: Queries 区独立于 memory，压缩只操作 memory 中的 recent_thinks，不涉及 queries。

**压缩是 question-blind 的**：
- Pass 2 rollout 中没有 queries（question-blind），压缩评分不考虑问题
- 推理时即使有 queries，压缩也只看 memory 中 thinks 的信息密度
- Queries 区的 question + answer 历史不参与压缩、不被压缩

**不再有 pending-aware compression**。旧的 `pending_overlap_penalty` 已删除（§2.3）。
如果某个 think 恰好包含 query 相关信息且被压缩掉了 → 不影响 queries 区，模型仍然能看到完整的问答历史。

### 9.7 Grounding 验证具体方法

```python
def verify_grounding(think_text, frames, teacher_caption):
    """检查 think 是否被帧支持"""
    
    # 方法 1: 基于 teacher_caption 的覆盖检查
    think_entities = extract_entities(think_text)
    caption_entities = teacher_caption["visible_entities"]
    coverage = len(think_entities & caption_entities) / max(len(think_entities), 1)
    
    # 方法 2: 用另一个 VLM (如 7B) 做 entailment 判断
    # "Given these frames, is this statement supported?"
    entailment_score = small_vlm_verify(frames, think_text)
    
    # 方法 3: 黑名单关键词
    blacklist = ["sound", "smell", "aroma", "feels", "seems to", "probably", 
                 "likely", "intend", "want to", "emotion", "happy", "sad"]
    has_blacklist = any(w in think_text.lower() for w in blacklist)
    
    return {
        "entity_coverage": coverage,       # >= 0.7 通过
        "entailment": entailment_score,    # >= 0.6 通过
        "no_blacklist": not has_blacklist,  # True 通过
    }
```

**执行时机**: 阶段 2 (Rollout) 产出的每条 think 都验证。不通过的丢弃并重新生成。

### 9.8 数据切分与 Token 验证

**数据切分规则**:
```
训练 / 验证 / 测试 按 video_id 切分（不按 sample 随机切）
300 videos → 240 train / 30 val / 30 test
同一视频的所有样本只出现在同一个 split 中
```

**Token 验证计划**:
```
阶段 2 产出后，对 100 条样本用目标 tokenizer 精确统计:
  - input_tokens 分布 (p50/p90/p99)
  - output_tokens 分布
  - 视觉 token 实际数量 vs 草算值
  - 超过 max_length 的比例
如果 p99 > max_length * 0.9 → 调整参数
```

---

## 10. 核心设计约束（不可违反）

以下约束来自多轮迭代的核心决策，任何代码或文档修改都必须遵守：

| # | 约束 | 对应���节 | 代码执行 |
|---|------|---------|---------|
| 1 | 视觉观察 think 每步立即入文本记忆（recall step2 分析 think 不存） | §2.1 | `MemoryState.add_think()` |
| 2 | 文本记忆覆盖时间 > 视觉窗口，两者重叠 | §2.1 | `snapshot()` 中 `visual_window_start` 独立于 `recent_thinks` |
| 3 | 80% token 预算触发压缩（精确 tokenizer） | §2.3 | `should_compress()` + `count_recent_tokens()` |
| 4 | compress range 只覆盖 INPUT ���的 recent_thinks，不含当前 think | §2.1 | `pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]` |
| 5 | 系���执行：先替换→再 append 当前 think | §2.1 | `memory.compress()` 后 think 已在 memory 中 |
| 6 | summary 只能引用 thinks，不能偷看视觉帧 | §2.1 | `verify_summary_provenance()` |
| 7 | C1 teacher 多维评分选最优范围，C2 模型自选 | §2.3 | `score_range_for_compression()` 5 维评分 |
| 8 | 主循环 action 优先级：用户问题 > query 触发 > silent（compress 不在主循环中） | §2.1 | pipeline Pass 3-C |
| 9 | recall_result 不含 teacher_caption、不含未来内容 | §3.5 | `_get_correct_result()` 只用 `observations[ec]['think']` |
| 10 | recall failure/distractor 时 response 内容表达信息不足，不用 gold_answer 强答 | §3.5 | `is_failed_recall` → "信息不足" response + `verify_information_flow()` |
| 11 | post-recall response 输出 think（分析检索结果），格式与所有 action 统一 | §1.3 | step2 有 `<think>` 标签 |
| 12 | query generator 不接收 gold_answer | §6.1 | `RECALL_QUERY_PROMPT` 仅含 question + visible_context |
| 13 | compressed_segments 超 5 段时合并最老两段（≤200 tok） | §2.1 | `MemoryState.compress()` + tokenizer 截断 |
| 14 | merge_compress 有对应训练样本 | §9.5 | `build_merge_compress_sample()` |
| 15 | 压缩比 ≥ 2.5:1 | §8 | `verify_compression_ratio()` |
| 16 | C1/C2 必须拆开训练，不合并 | §2.3 | C1 `range="X-Y"` / C2 `<compress_trigger/>` 独立样本 |
| 17 | teacher 信息只用于选 gold，不进 student 可见路径 | §1.2 | evidence 仅在 `score_range_for_compression` 中使用 |
| 18 | summary 不能包含当前 think 的独有事实 | §2.1 | `verify_summary_no_current_think_leak()` |
| 19 | compressed tag 使用 JSON-inside-tag 格式，不用 XML attribute | §13.2 | `format_for_prompt()` 输出 `<compressed>{json}</compressed>` |
| 20 | Pass2 必须 question-blind | §3.2 | 无 question/task/gold_answer 参数 |
| 21 | 压缩触发基于 pre-action memory，不含当前 think | §2.1 | `should_compress` 在 `add_think` 之前评估 |
| 22 | 每批数据必须跑 task coverage audit | §9.4 | `audit_task_coverage()` + `task_coverage_report.json` |

---

## 11. 数据源分析与视频选择策略

### 11.1 数据源总览

数据审计（2026-04-20）基于 `video_catalog_30s_plus.csv`（≥30s 视频全量索引）。

| 数据集 | ≥30s 视频数 | 中位时长 | ≥90s | ≥120s | 主要内容 |
|--------|-----------|---------|------|-------|---------|
| VideoMind-Dataset | 394,833 | 114s | 250,959 | 184,794 | 多来源混合（ActivityNet/cosmo_cap/internvid 等） |
| LLaVA-Video-178K | 93,853 | 88s | 45,834 | 28,995 | 学术+YouTube，按时长分桶 |
| Koala | 66,834 | 32s | 0 | 0 | 30s 切片段，多样性好 |
| tarsier2_unzip | 51,602 | 51s | 16,990 | 13,603 | ActivityNet/VATEX/Charades 等 |
| Koala_raw | 25,813 | 547s | 25,523 | 25,335 | 原始长视频（烹饪/手工/教学） |
| how_to_step | 18,831 | 194s | 17,342 | 15,609 | 教程步骤视频 |
| how_to_caption | 15,844 | 179s | 13,849 | 11,930 | 教程描述视频 |
| **总计** | **667,611** | — | **370,498** | **280,267** | — |

**时长分布（≥30s 全量）**：

```
  30-60s:  211,352 (31.7%)  ████████████████
  60-90s:   85,761 (12.8%)  ██████
 90-120s:   90,231 (13.5%)  ███████
120-180s:  166,373 (24.9%)  ████████████
180-300s:   63,156 ( 9.5%)  █████
   300s+:   50,738 ( 7.6%)  ████
```

**Streamo 已用视频标记**：`used_in_streamo=1` 共 192,342 条。这些视频已在先前项目中验证过质量，
可优先复用。其中 Koala 占 6.7 万（全部 ~32s 短片段），LLaVA 5.4 万（中位 104s），
VideoMind 3.7 万（中位 150s），how_to_* 3.5 万（中位 180s+）。

### 11.2 Pipeline 时长门槛

各训练任务对视频时长的硬性要求，由 pipeline 参数推导：

```
AGENT_CHUNK_SEC = 2s
VISUAL_WINDOW_CHUNKS = 12  → 视觉窗口 = 24s
首次压缩触发 ≈ COMPRESS_TOKEN_THRESHOLD / THINK_TOKEN_AVG ≈ 480/50 ≈ 10 chunks = 20s
```

| 任务类型 | 最低时长 | 推导 |
|---------|---------|------|
| silent + response(from_frames) | **24s** | 填满一个视觉窗口 |
| compress（首次触发） | **20s** | ~10 thinks 达 80% token 预算 |
| recall | **30s** | evidence 必须离开视觉窗口（12 chunks gap） |
| response_from_memory | **30s** | think 在 recent_thinks 但帧已滑出 |
| compress + recall | **50s** | 压缩后 + 窗口滑出 + recall gap |
| 多次压缩 + compress_recall | **90s** | 2+ 压缩事件 + 压缩后 recall |
| **全任务覆盖** | **≥120s** | 所有任务类型均可产出 |

### 11.3 数据集三档划分

**不可用（<24s）**：TGIF(5s), LSMDC(5s), SSV2(10s), Kinetics-700(10s), VATEX(10s),
TREC-VTT(6s), Oops(15s), LLaVA_0_30s(15s) —— 约 37 万条。
连一个完整视觉窗口都填不满，无法产出任何有效训练样本。

**Phase 1/2 可用（30-89s）**：Koala(32s), Charades(32s), internvid_vtime(33s),
vid_morp(34s), LLaVA_30_60s(45s), cosmo_cap 部分 —— 约 30 万条。
可产出 silent、response(from_frames)、response(from_memory)、recall，
但不能触发完整压缩链。适合 Phase 1（协议对齐）和 Phase 2（recall 学习）。

**全 Phase 可用（≥90s）**：ActivityNet, cosmo_cap 长视频, LLaVA_1_2m/2_3m,
how_to_step/caption, youcook2, Koala_raw, ego4d —— 约 37 万条。
可产出所有任务类型，包括 compress + compress_recall 联合任务。
是 C1/C2/Phase 5 训练的核心数据源。

### 11.4 分阶段视频选择策略

**原则**：
1. 每个 phase 选用最匹配其任务需求的时长区间，不浪费长视频在简单任务上
2. 优先用 `used_in_streamo=1` 的已验证视频
3. 分层抽样保证数据集来源多样性（按 `dataset/subdataset` 分层）
4. 同一视频的所有样本只出现在同一个 train/val/test split

| Phase | 时长要求 | 选取数量 | 主要来源 | 选择逻辑 |
|-------|---------|---------|---------|---------|
| **Phase 1** | ≥30s | 200 视频 | Koala(32s), Charades, LLaVA_30_60s | 短视频池，最大化内容多样性 |
| **Phase 2** | ≥60s | 200 视频 | cosmo_cap, LLaVA_1_2m, VideoMind(60-120s) | 中等时长，需要 recall gap |
| **C1/C2** | ≥120s | 300 视频 | Koala_raw(547s), ActivityNet, how_to_*, LLaVA_2_3m | 长视频，多次压缩 + compress_recall |
| **Phase 5** | ≥120s | 与 C1/C2 共享 | 同上 | 混合任务全覆盖 |

**总计约 700 个视频**（有重叠：≥120s 的视频同时服务于 P2/C1/C2/P5）。

**各 Phase 产出样本量估算**：

```
Phase 1: 200 视频 × 20 samples/video ≈ 4,000 samples
         (silent ~2,600 + response ~1,400)

Phase 2: 200 视频 × 30 samples/video ≈ 6,000 samples
         (silent ~3,000 + response ~1,200 + recall ~1,200 + special ~600)

C1:      300 视频 × 50 samples/video ≈ 15,000 samples
         (silent ~6,750 + response ~2,700 + recall ~2,250 + compress ~1,800 + special ~1,500)

C2:      共享 C1 视频，每视频追加 ~10 C2 compress samples ≈ 3,000 samples

Phase 5:  共享 C1 视频，按混合比例采样 ≈ 5,000 samples

总计: ~33,000 samples
```

### 11.5 视频选择实现（`select_videos` 函数）

```python
def select_videos_by_phase(catalog_csv, phase, num_videos, seed=42):
    """从 video_catalog_30s_plus.csv 按 phase 需求选择视频。

    Args:
        catalog_csv: video_catalog_30s_plus.csv 路径
        phase: "P1" | "P2" | "C1" | "C2" | "P5"
        num_videos: 目标数量
        seed: 随机种子

    选择逻辑:
        1. 按 phase 过滤时长区间
        2. 优先选 used_in_streamo=1 的视频（质量已验证）
        3. 按 dataset/subdataset 分层抽样（避免单一来源）
        4. 每层内按时长降序排列（更长 = 更多样本产出）
    """
    PHASE_DURATION = {
        "P1": (30, 90),      # 30-89s：短视频，协议对齐
        "P2": (60, 180),     # 60-179s：中等，recall 学习
        "C1": (120, None),   # ≥120s：长视频，压缩训练
        "C2": (120, None),   # ≥120s：同 C1
        "P5": (120, None),   # ≥120s：混合全覆盖
    }
    min_dur, max_dur = PHASE_DURATION[phase]

    # 1. 从 CSV 读取并过滤
    candidates = []
    for row in csv.DictReader(open(catalog_csv)):
        dur = float(row["duration_sec"])
        if dur < min_dur:
            continue
        if max_dur and dur >= max_dur:
            continue
        candidates.append(row)

    # 2. 分为 streamo 优先 / 其他
    streamo = [r for r in candidates if r.get("used_in_streamo") == "1"]
    others = [r for r in candidates if r.get("used_in_streamo") != "1"]

    # 3. 分层抽样（按 dataset/subdataset）
    def stratified_sample(pool, n, seed):
        groups = defaultdict(list)
        for r in pool:
            key = f'{r["dataset"]}/{r["subdataset"]}'
            groups[key].append(r)
        # 每组内按时长降序
        for g in groups.values():
            g.sort(key=lambda x: -float(x["duration_sec"]))
        # Round-robin
        random.seed(seed)
        keys = sorted(groups.keys())
        random.shuffle(keys)
        selected = []
        per_group = max(1, n // len(keys))
        for key in keys:
            take = min(per_group, len(groups[key]), n - len(selected))
            selected.extend(groups[key][:take])
            if len(selected) >= n:
                break
        # Fill remaining
        if len(selected) < n:
            remaining = [r for g in keys for r in groups[g] if r not in selected]
            selected.extend(remaining[:n - len(selected)])
        return selected[:n]

    # 4. 先从 streamo 选，不足则从 others 补
    target = num_videos
    selected = stratified_sample(streamo, min(target, len(streamo)), seed)
    if len(selected) < target:
        extra = stratified_sample(others, target - len(selected), seed + 1)
        selected.extend(extra)

    return selected[:num_videos]
```

**选择结果验证清单**：
- [ ] 每个 phase 的视频时长全部满足下限
- [ ] 无单个 subdataset 占比超过 30%（多样性检查）
- [ ] train/val/test 按 video_id 切分，同视频不跨 split
- [ ] C1 视频平均产出 ≥2 次压缩事件
- [ ] P2 视频平均产出 ≥3 个 recall 任务

### 11.6 不可用数据集处理

以下数据集不进入 pipeline，但可用于其他用途：

| 数据集 | 原因 | 可替代用途 |
|--------|------|-----------|
| TGIF (~94K, 5s GIF) | 太短 | 视觉编码器预训练 |
| Kinetics-700 (~50K, 10s) | 太短 | 动作识别预训练 |
| SSV2 (~10K, 10s) | 太短 | 动作识别预训练 |
| LSMDC (~108K, 5s) | 太短 | caption 预训练 |
| VATEX (~22K, 10s) | 太短 | caption 预训练 |
| WebVid-10M (解压中) | 多数 <15s | 预训练，不用于 agent 数据 |
| Koala-36M | 仅元数据 | 不可用 |

---

## 12. 开放问题（已缩减）

1. **Per-timestep vs 短序列**: 当前方案 A 每步独立。如果实验中发现模型缺少连续性，可改为 3-step 滑动窗口（loss 只在最后一步）。先用纯 per-timestep 做 baseline。详见 `sft_engineering.md` §3.4。
2. **Think 关键信息保留策略**: 如果 think 漏了关键细节（如盐的量），后续 recall 就找不到。建议：阶段 2 产出后，用 teacher caption 检查 thinks 的"关键信息覆盖率"。覆盖率 < 0.6 的重新生成。
3. **RL rollout 迁移**: `grpo.py` 的 rollout 需从 `streaming_video_chat` 迁移到 `StreamingAgentLoop.step()`，确保 RL rollout 与 SFT 训练格式一致（per-timestep re-render + 显式 memory 管理）。

---

## 13. SFT 工程集成要点

> 完整设计见 `sft_engineering.md`。本节定义数据构造 pipeline 与 SFT 训练代码之间的接口。

### 13.1 Pipeline 输出 → SFT 输入的契约

Pipeline Pass 3-C 产出的每条样本必须包含��下字段，SFT `PerTimestepDataset` 依赖这些字段构建 model inputs：

```json
{
  "sample_id": "vid001_t30_silent_42",
  "video_id": "vid001",
  "video_path": "path/to/video.mp4",
  "sample_type": "silent | response | recall_query | recall_response | compress | merge_compress",
  "chunk_idx": 15,
  "phase": "1 | 2 | C1 | C2 | 5",
  
  "input": {
    "system": "You are a streaming video agent...",
    "memory": {
      "compressed": [{"time_range": [0, 20], "text": "..."}],
      "recent_thinks": ["[20-22] ...", "[22-24] ..."]
    },
    "queries": [
      {"question": "Tell me when basil is added", "since": 20,
       "answers": ["Basil leaves being torn over pot."]}
    ],
    "visual_window": {
      "video_start": 8.0,
      "video_end": 32.0,
      "frames": 24,
      "chunk_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    },
    "user_input": "What color is the apron?",
    "recalled_frames": {
      "time_range": [2, 6],
      "n_frames": 4,
      "source": "historical_frames"
    },
    "recall_result": {
      "source": "student_think",
      "time": "2-6",
      "text_content": "Retrieved: [2-4] Chef in red apron enters kitchen..."
    }
  },
  
  "output": "<think>...</think><action>response</action><response>Red.</response>",
  
  "metadata": {
    "gold_action": "response",
    "gold_answer": {"answer": "Red", "source": "teacher_caption"},
    "card_id": "vid001_F2_001",
    "family": "F2",
    "answer_form": "short_exact",
    "verifiable": true,
    "availability": "in_visual",
    "difficulty_margin": 0.12,
    "action_minimality": {"answer_in_frames": true},
    "visibility_matrix": {...},
    "leakage_checks": {"query_contains_answer": false}
  }
}
```

**Compress 样本的额外 metadata 字段（P1-1）**：

```json
{
  "sample_type": "compress",
  "metadata": {
    "gold_action": "compress",
    "compressed_chunks": [4, 5, 6, 7],
    "compressed_range": [8, 16],
    "compressed_source_texts": [
      "[8-10] Oil heated in stainless pot.",
      "[10-12] Garlic browned in oil.",
      "[12-14] Tomato quarters placed into pot.",
      "[14-16] White seasoning from small bowl added."
    ]
  }
}
```

验证时 summary provenance 和 compression ratio 只使用 `compressed_source_texts`（被选中的 range），
不使用 `input.memory.recent_thinks` 全量。这防止 summary 偷用未选中 thinks 的信息。

**字段约束**：

| 字段 | 必须 | 格式约束 |
|------|------|---------|
| `input.memory.compressed` | 是 | 列表，可为空 `[]` |
| `input.memory.recent_thinks` | 是 | 列表，可为空 `[]`，每条格式 `"[ts-te] text"` |
| `input.queries` | 否 | 有活跃 query 时，列表含 question + answers 历史 |
| `input.visual_window.video_start/end` | 是 | 秒，float |
| `input.visual_window.frames` | 是 | 整数，= chunk_count × 2 |
| `input.recalled_frames` | 否 | 仅 recall_response 类型有；**顶层字段，不嵌套在 visual_window 内** |
| `input.recall_result` | 否 | 仅 recall_response 类型有 |
| `input.user_input` | 否 | silent 类型无问题时可省略或为 `""` |
| `output` | 是 | 严格匹配四动作格式之一（方案 B: JSON inside tags） |
| `metadata.gold_action` | 是 | `silent \| response \| recall \| compress \| merge_compress` |
| `metadata.compressed_source_texts` | 仅 compress | 被选中 range 的原始 thinks 列表 |

### 13.2 Special Token 对齐

Pipeline 产出的 output 字段中使用以下 token，SFT 代码必须在 `init_processor` 中添加。

**方案 B（P0-7）**：所有 tag 是精确的 special token，属性数据以 JSON 放在 tag 内部。
不使用 `<compressed time="0-20">` 这种 XML 属性写法（tokenizer 无法精确匹配）。

```
基础 token（已有）：<think> </think> <silent> <response> </response>
                    <action> </action> <query> </query>
                    <recall_result> </recall_result>

新增 token（per-timestep 格式需要）：
    Input 结构 tag:
    <memory> </memory>                   — 包裹整个记忆块
    <compressed> </compressed>           — memory 内：压缩段
    <queries> </queries>                — 独立问答追踪区（v8.0 替代 pending）
    <query> </query>                   — queries 内：问题
    <answer> </answer>                 — queries 内：历史回答
    <visual_window> </visual_window>     — 视觉窗口 header (JSON inside)
    <recalled_frames> </recalled_frames> — 回忆帧 header
    <user_input> </user_input>           — 包裹用户输入文本
    
    Output payload tag:
    <summary> </summary>                — compress action 的 payload
    
    User input trigger:
    <compress_trigger> </compress_trigger> — 系统触发压缩

示例：
    <memory>
    <compressed>{"time_range":[0,20],"text":"Chef prepared workspace..."}</compressed>
    [20-22] Oil heated in pot.
    <queries>
    <query t="44">Tell me when basil added</query>
    <answer t="70">Basil leaves being torn over pot.</answer>
    </queries>
    </memory>
    <visual_window>{"start":20,"end":44,"frames":24,"current_time":[42,44]}</visual_window>
    <user_input><compress_trigger>{"range":[20,34]}</compress_trigger></user_input>
```

### 13.3 Visual Window 加载约定

SFT 代码通过 `load_video_frames()` 加载视觉帧。Pipeline 产出的 `visual_window` 字段遵循以下约定：

- `video_start` / `video_end`：绝对秒数，SFT 代码直接传给 `load_video_frames`
- `frames`：总帧数 = `VISUAL_WINDOW_CHUNKS × FRAMES_PER_CHUNK` = 12 × 2 = 24
- 帧加载使用 ghost message 模式，一次调用 `process_vision_info` 加载全部 24 帧
- Recalled frames 单独加载（另一次 `process_vision_info` 调用，4 帧）

### 13.4 Phase 分配与数据集切分

Pipeline Pass 4 (Verify) 输出 `train.jsonl` / `val.jsonl` / `test.jsonl`。
SFT 训练时按 `phase` 字段过滤样本：

```python
# Phase 1 训练：只加载 phase="1" 的样本
samples = [s for s in all_samples if s["phase"] == "1"]
```

各 Phase 的训练顺序、学习率、epoch 数等超参见 `sft_engineering.md` §7。
