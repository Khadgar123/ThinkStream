# 流视频 Agent 数据构造方案

> 版本: v8.0 | 日期: 2026-04-23
>
> 核心设计：
> - Per-timestep 独立样本，每步重新构造完整 input，不依赖跨步 KV cache
> - `<think>` 每步立即入文本记忆，文本记忆覆盖时间 > 视觉滑动窗口
> - 输入布局：**视频在前、文本在后**（`<visual_window>` → `<memory>` → `<user_input>`）
> - 文本 think 的 MROPE temporal position 对齐到对应视频帧的时间戳
> - 80% token 预算触发系统压缩，C1 系统指定范围 → C2 模型自选范围
> - 压缩时模型可同时利用 thinks 文本和当前可见的视觉帧生成更精确的 summary
> - Recall 拆为两步：step1 输出 query → 系统检索 → step2 输出 response/silent
> - 所有 action 输出统一为 `<think>...</think><action>...</action>` 格式（含 recall_response）
> - Action 优先级：post-recall response > 用户问题 > compress > silent
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

### 1.3 四动作协议

```
1) <think>...</think><action>silent</action>
2) <think>...</think><action>response</action><response>...</response>
3) <think>...</think><action>recall</action><query>{"query":"...","time_range":"..."}</query>
4) <think>...</think><action>compress</action><summary>{"time_range":[s,e],"text":"..."}</summary>
```

post-recall response 是特殊情况：**不输出 think**（因为同一 chunk 的 think 已在 recall query turn 输出），只输出：
```
<action>response</action><response>...</response>
```

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

**Action 优先级**（同一步多条件同时满足时）：
```
post-recall response > 用户当前新问题 response/recall > query 触发 response > compress > silent
```
用户交互优先，compress 可延后一帧。query 触发 = 模型看到 `<queries>` 中的问题且当前帧有新内容时主动 response。

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
[User input: 当前新问题 / compress_trigger / recall_result / 空]    ← Zone D
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

**注意**：think 立即进入 recent_thinks，最早的 thinks 可能仍有对应帧在 visual_window 中。
这不是问题——压缩只操作文本记忆，视频帧由 FIFO 独立管理。

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
| **C1** | 系统触发 `<compress_trigger range="X-Y"/>` | teacher 指定最优范围 | 学会压缩行为 + 压缩后记忆推理 |
| **C2** | 系统触发（无指定范围） | 模型自选（gold = teacher policy） | 学会选择压缩窗口 |
| **C3**（未来） | 模型自主判断 | 模型自选 | 完整记忆管理 |

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
  ├─ 3-A: Task Card 生成               [397B, 1 call/video]
  ├─ 3-B: Ask-Time 搜索               [纯程序, 0 call]
  └─ 3-C: Action 映射 + SFT 样本生成   [397B, ~20 call/video]
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

#### 1-B: 后处理（变化检测 + 可选实体链接）

1-A 输出的是每个 chunk 的**静态快照**。1-B 做跨 chunk 后处理，全部纯程序。

##### 下游到底需要什么？

| 消费者 | 用法 | 真正需要的信号 | 需要实体一致 ID？ |
|--------|------|-------------|----------------|
| Pass 2 `_teacher_fact_value` | `len(state_changes) * 0.3` | "这个 chunk 有没有变化" (bool/count) | **不需要** |
| Pass 3-A E2 检测 | `if cap.get("state_changes")` | "这个 chunk 有变化吗" (bool) | **不需要** |
| Pass 3-A P1 步骤序列 | 连续 ≥3 个有变化的 chunk | "连续变化的区间" | **不需要** |
| Pass 3-A C1 比较 | 同实体跨时间的状态差异 | "entity X 在 t1 vs t2" | **需要** |
| Pass 3-A R1 再识别 | 实体消失→重现模式 | "entity X 出现/消失时间线" | **需要** |

**结论：大多数场景只需要"有没有变化"这个信号，不需要精确的实体 ID。C1 和 R1 用 desc 描述实体即可（见下方），不需要跨 chunk 一致 ID。**

##### 1-B-1: 变化检测（所有视频必做，不需要实体 ID）

不依赖实体链接，直接比较相邻 chunk 的 **action 词集合** 变化：

```python
def detect_chunk_changes(evidence: List[Dict]):
    """
    比较相邻 chunk 的 action 词集合，检测变化。
    
    不做实体链接——只看"这个 chunk 的动作词集合和上一个 chunk 是否不同"。
    输出: evidence[i]["state_changes"] = ["action changed"] 或 []
    
    这足以满足 Pass 2 评分和 Pass 3-A 的 E2/P1 检测。
    """
    if not evidence:
        return
    evidence[0]["state_changes"] = []
    
    for i in range(1, len(evidence)):
        prev_actions = _extract_action_words(evidence[i - 1])
        curr_actions = _extract_action_words(evidence[i])
        
        # 实体数量变化
        prev_n = len(evidence[i - 1].get("visible_entities", []))
        curr_n = len(evidence[i].get("visible_entities", []))
        entity_count_changed = abs(prev_n - curr_n) >= 1
        
        # action 词集合变化
        if not prev_actions and not curr_actions:
            action_changed = False
        elif not prev_actions or not curr_actions:
            action_changed = True
        else:
            overlap = len(prev_actions & curr_actions) / max(len(prev_actions | curr_actions), 1)
            action_changed = overlap < 0.5  # 动作词重叠 < 50% → 认为有变化
        
        changes = []
        if action_changed:
            changes.append("action_changed")
        if entity_count_changed:
            changes.append("entity_count_changed")
        
        evidence[i]["state_changes"] = changes


def _extract_action_words(cap: Dict) -> set:
    """提取一个 chunk 中所有 entity 的 action 关键词"""
    stop = {"the", "a", "an", "is", "in", "on", "at", "to", "of", "with", "and"}
    words = set()
    for entity in cap.get("visible_entities", []):
        action = entity.get("action", "")
        words |= {w.lower() for w in action.split() if w.lower() not in stop and len(w) > 2}
    return words
```

**为什么不依赖实体 ID**：
- Pass 2 压缩评分只看 `len(state_changes)` → 有几项变化，不关心谁变了
- E2 只看 `if cap.get("state_changes")` → 有没有变化
- P1 只看连续 chunk 有无变化 → bool 信号
- action 词集合变化比逐实体比较 action 字符串更鲁棒（不怕 "stirring sauce" vs "stirring the pot" 的措辞差异）

##### 1-B-2: C1/R1 不需要实体链接

C1（跨时间比较）和 R1（再识别）**不需要跨 chunk 的实体 ID**。
问题用外观描述（desc）引用实体，模型靠视觉理解匹配，比编号更自然：

```
C1 问题用 desc：
  ✅ "锅里的东西和刚才比有什么变化？"（用位置/外观描述）
  ✅ "穿红围裙的人现在和之前做的事一样吗？"
  ❌ "pot_1 和之前比有什么变化？"（需要知道 pot_1 是谁）

R1 问题用 desc：
  ✅ "之前那个小白碗还在画面里吗？"（用外观描述）
  ❌ "bowl_1 还在画面里吗？"
```

**Pass 3-A 的 `scan_opportunities` 自动检测 C1/R1 机会时**，也用 desc 词重叠而非 entity ID：
- C1：相邻 chunk 的 state_changes 非空 + 有相似 desc 的实体 → 可问比较
- R1：某个 desc 在视频前半段出现、中间消失、后半段重现 → 可问再识别

397B 在生成 Task Card 时看到全视频的 evidence 摘要，直接判断哪些实体是"同一个"并生成比较/再识别问题，比程序做实体链接更可靠。

##### 1-B 执行流程

```
1-B-1 变化检测 → state_changes    （纯程序，~0.1ms/视频）
C1/R1 的实体匹配由 Pass 3-A 的 397B 直接处理（不需要 1-B 做实体链接）
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

> v8.0 新设计：Teacher 只负责找"可问什么"，不决定 ask_time 和 action。

**目标**: 从 evidence graph 中提取所有"可问的内容"，生成 Task Card。每张 card 只定义问题本体，不带 ask_time。

**输入**: Teacher evidence graph（来自阶段 1）

**输出**: Task Card 列表

#### Task Card 数据结构

```python
TaskCard = {
    "card_id": "vid001_F2_001",        # 唯一标识
    "family": "F2",                     # 问题家族（见 §4.1）
    "question": "What color is the chef's apron?",
    "canonical_answer": "Red",          # 简短标准答案（1-15 words）
    "answer_form": "short_exact",       # 答案形式（见 §4.2 验证等级）
    "verifiable": true,                 # 是否可自动验证
    "support_spans": [[5, 5]],          # 证据 chunk 区间 [(start, end), ...]
    "support_facts": ["Chef wearing a red apron"],
    "retention_class": "low",           # high | medium | low
    "answer_keywords": ["red", "apron", "color"],
    "entities_involved": ["chef_1"],
}
```

#### 两阶段生成

**阶段 1：程序预筛选（无 397B）**

遍历 evidence_graph，标记每个 chunk 可供哪些问题家族使用：

```python
def scan_opportunities(evidence):
    """返回 opportunity_map[chunk_idx] = [family_ids]"""
    opp = {}
    for cap in evidence:
        idx = cap["chunk_idx"]
        families = []
        for fact in cap.get("atomic_facts", []):
            if fact["confidence"] < 0.7 or fact["support_level"] != "direct_current_chunk":
                continue
            text = fact["fact"].lower()
            if cap.get("ocr") or any(c.isdigit() for c in text):
                families.append("F1")  # OCR/Number
            if has_visual_attribute(text):
                families.append("F2")  # Fine-grained
            if has_count_pattern(text):
                families.append("F3")  # Count
        if len(cap.get("visible_entities", [])) >= 2 and cap.get("spatial"):
            families.append("F4")  # Spatial
        if cap.get("atomic_facts"):
            families.append("E1")  # Local action
        if cap.get("state_changes"):
            families.append("E2")  # State change
        if len(cap.get("visible_entities", [])) >= 3:
            families.append("S1")  # Summary
        opp[idx] = list(set(families))
    
    # 跨 chunk 分析
    add_procedural_opportunities(evidence, opp)    # P1: 连续 state_changes ≥ 3 步
    add_comparison_opportunities(evidence, opp)    # C1: 同实体不同时间的 state_change 对
    add_reidentification_opportunities(evidence, opp)  # R1: 实体消失后重现
    return opp
```

**阶段 2：397B 批量生成 Task Cards**

对预筛选出的 opportunity，**一次 397B 调用生成全视频的 task cards**（不再每 fact 单独调用）：

```python
TASK_CARD_PROMPT = """你是流视频 agent 训练数据构造器。

以下是视频的 teacher 证据摘要：
{evidence_summary}

请为以下问题家族生成 Task Cards：
{family_requirements}

每个 Task Card 输出：
{{
  "family": "F1|F2|F3|F4|E1|E2|P1|C1|R1|S1",
  "question": "自然语言问题",
  "canonical_answer": "简短标准答案 (1-15 words)",
  "answer_form": "见下方验证等级",
  "support_chunks": [chunk_idx, ...],
  "support_facts": ["原始 fact 文本"],
  "retention_class": "high|medium|low",
  "answer_keywords": ["keyword1", "keyword2"],
  "entities_involved": ["entity_id"]
}}

answer_form 验证等级（优先生成 V1-V4）：
- "binary":          是/否问题 — 自动验证
- "multiple_choice":  选择题（附带选项）— 自动验证
- "number":          数值/计数 — 自动验证（容差匹配）
- "short_exact":     精确短答 1-3 词 — 自动验证（exact match）
- "short_factual":   事实短答 1-10 词 — 半自动（keyword + LLM judge）
- "descriptive":     描述性回答 — 需 LLM judge

要求：
- 优先生成可自动验证的问题（binary > multiple_choice > number > short_exact）
- 每种 family 至少 60% 的问题必须是 V1-V4（可自动验证）
- 不要在 question 里泄漏 answer
- 一个 fact 只生成一个 task card

输出 JSON 数组:"""
```

**每视频 family 目标数量**：

```python
FAMILY_TARGETS = {
    "F1": 3,   # OCR/Number — 天然可验证 (number/short_exact)
    "F2": 4,   # Fine-grained — 优先 binary/multiple_choice
    "F3": 2,   # Count — 天然可验证 (number)
    "F4": 2,   # Spatial — 优先 binary/multiple_choice
    "E1": 3,   # Local action — 主要做 response 对照
    "E2": 2,   # State change — event_watch + queries 持续追踪
    "P1": 2,   # Procedure — 优先 multiple_choice/number
    "C1": 2,   # Comparison — 优先 binary，用 desc 引用实体
    "R1": 1,   # Re-identification — 天然 binary，用 desc 引用实体
    "S1": 2,   # Summary — 唯一允许 descriptive 的 family
    "M1": 2,   # Continuous commentary — 通过 queries 区持续回答
}
# 每视频目标 ~25 个 task cards
```

**397B 调用量**：~1 call/video（vs 旧方案 ~30 calls/video）

### 3.4 阶段 3-B: Ask-Time 搜索

> 纯程序，零 397B 调用。替代旧 Pass 3 的 8 个 mine_*() 函数。

**目标**: 对每个 task card，高效判定每个候选 ask_chunk 的 availability，选择最优提问时间点。

**输入**: Task Cards（来自 3-A）+ Rollout（来自阶段 2）

**输出**: Placed Tasks（每张 card 绑定 1-3 个 ask_chunk）

#### 为什么不能靠关键词逐 chunk 扫描

旧方案对每个 (card, chunk) 组合都调 `semantic_overlap()` 做关键词/embedding 匹配，有三个根本问题：

1. **O(cards × chunks²) 复杂度** — 23 cards × 60 chunks × 每次扫 recent_thinks(~10 条) = 上万次匹配
2. **误判率高** — "red" 出现在无关 think 里是 false positive；paraphrase 不匹配是 false negative
3. **多数判定不需要看内容** — `in_visual`、`in_future` 完全是结构位置关系；`in_recent_thinks` 主要取决于"think 还在不在窗口里"而不是"think 写了什么"

#### 新方案：预计算 + 结构查表

**核心思路：把昂贵的内容匹配做一次预计算，把逐 chunk 的 availability 判定变成 O(1) 查表。**

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 预计算 retention bitmap（每个 card 做一次）      │
│  ─────────────────────────────────────────                │
│  对每个 card 的 support_chunks，检查：                     │
│    thinks_retained[chunk_idx] = think 是否记住了答案      │
│    summary_retained[comp_event_idx] = summary 是否保留了   │
│                                                          │
│  只做 O(cards × support_chunks) 次匹配                    │
│  而不是 O(cards × all_chunks) 次                          │
│                                                          │
│  Step 2: 逐 chunk 查表（纯结构判定）                       │
│  ──────────────────────────────                           │
│  对每个 ask_chunk，用 rollout 的结构信息判定 availability   │
│  全部是 index 比较和 bitmap lookup，零内容匹配              │
└─────────────────────────────────────────────────────────┘
```

#### Step 1: 预计算 Retention Bitmap

```python
def precompute_retention(card, rollout):
    """
    对一个 task card，预计算其答案在 rollout 各处的留存状态。
    
    只在 support_chunks 和 compression_events 上做内容匹配，
    不遍历所有 chunk。复杂度: O(|support_chunks| + |compression_events|)
    """
    observations = rollout.get("thinks", [])
    compression_events = rollout["compression_events"]
    
    # ── 哪些 support chunk 的 think 记住了答案？ ──
    thinks_retained = {}  # chunk_idx -> bool
    for span_start, span_end in card.support_spans:
        for chunk_idx in range(span_start, span_end + 1):
            if chunk_idx < len(observations):
                think_text = observations[chunk_idx]["think"]
                # retention_class 作为先验：low retention 用更严格的阈值
                threshold = {
                    "low": 0.5,     # 精确数字/细粒度颜色 → 要求更强匹配才算记住
                    "medium": 0.35,
                    "high": 0.2,    # 粗粒度动作 → 低阈值就算记住
                }[card.retention_class]
                thinks_retained[chunk_idx] = (
                    keyword_overlap(think_text, card.answer_keywords) > threshold
                )
            else:
                thinks_retained[chunk_idx] = False
    
    # ── 哪些 compression summary 保留了答案？ ──
    summary_retained = {}  # comp_event_idx -> bool
    for idx, event in enumerate(compression_events):
        compressed_chunks = event.get("compressed_thinks_chunks", [])
        # 只检查包含 support_chunks 的压缩事件
        if any(c in compressed_chunks for span in card.support_spans
               for c in range(span[0], span[1] + 1)):
            summary_text = event["summary"].get("text", "")
            summary_retained[idx] = (
                keyword_overlap(summary_text, card.answer_keywords) > 0.3
            )
    
    return RetentionBitmap(
        thinks_retained=thinks_retained,        # {5: True, 6: False}
        summary_retained=summary_retained,      # {0: True, 2: False}
        support_chunks=set(
            c for s in card.support_spans for c in range(s[0], s[1] + 1)
        ),
    )
```

**关键：retention_class 作为匹配阈值的先验。**

| retention_class | 含义 | threshold |
|----------------|------|-----------|
| **low** (F1 OCR, F2 细粒度) | think 大概率没记精确值 | 0.5（要求强匹配才算记住） |
| **medium** (F4, P1, E2) | think 可能记了粗粒度 | 0.35 |
| **high** (E1, S1) | think 几乎一定记住 | 0.2（弱匹配就算记住） |

这样 low-retention 的问题（精确数字/颜色）更容易判定为 `in_history_only` → recall，
高 retention 的问题更容易判定为 `in_recent_thinks` → response。
**用 retention_class 替代了大量不可靠的逐 chunk 关键词匹配。**

#### Step 2: 利用 snapshot 结构化字段做纯查表

Pass 2 的 `MemoryState.snapshot()` 不是文本拼接，而是结构化 dict：

```python
snapshots[chunk_idx] = {
    "chunk_idx": 25,
    "visual_window_start": 14,                         # 纯数字
    "compressed_segments": [                            # 结构化列表
        {"time_range": [0, 16], "text": "...", "merge_level": 1}
    ],
    "recent_thinks": [                                  # 每条带 chunk 索引
        {"chunk": 8,  "time": "16-18", "text": "..."},
        {"chunk": 9,  "time": "18-20", "text": "..."},
        # ...
    ],
}
# queries 区独立于 snapshot，由系统单独管理（见 §9.1）
```

**关键：`recent_thinks` 每条都有 `chunk` 字段（int 索引），不需要解析文本。**
所以判定 `in_recent_thinks` 只需要 set 交集：

```python
def classify_availability(card, ask_chunk, rollout, bitmap):
    """
    纯结构判定，零内容匹配。
    
    利用 snapshot 的结构化字段 + 预计算的 retention bitmap。
    每次调用 O(1)（set lookup + index 比较）。
    """
    support_start = min(s[0] for s in card.support_spans)
    support_end = max(s[1] for s in card.support_spans)
    snapshot = rollout["snapshots"][ask_chunk]
    
    # ═══════════════════════════════════════════════════
    # Case 1: in_future — 纯数字比较
    # ═══════════════════════════════════════════════════
    if support_start > ask_chunk:
        return "in_future"
    
    # ═══════════════════════════════════════════════════
    # Case 2: in_visual — 纯数字比较
    # ═══════════════════════════════════════════════════
    # visual_window_start 直接从 snapshot 取，不用算
    window_start = snapshot["visual_window_start"]
    window_end = snapshot["chunk_idx"]
    if any(window_start <= c <= window_end for c in bitmap.support_chunks):
        return "in_visual"
    
    # ═══════════════════════════════════════════════════
    # Case 3: in_recent_thinks — set 交集 + bitmap lookup
    # ═══════════════════════════════════════════════════
    # snapshot["recent_thinks"] 每条自带 chunk 索引，直接提取
    recent_chunks = {item["chunk"] for item in snapshot["recent_thinks"]}
    
    # support_chunk 还在 recent_thinks 里 且 think 记住了答案？
    retained_and_present = bitmap.support_chunks & recent_chunks
    if any(bitmap.thinks_retained.get(c, False) for c in retained_and_present):
        return "in_recent_thinks"
    
    # ═══════════════════════════════════════════════════
    # Case 4: in_compressed — 区间包含 + bitmap lookup
    # ═══════════════════════════════════════════════════
    # snapshot["compressed_segments"] 每段自带 time_range
    # 但我们需要知道是哪个 compression_event → 用 bitmap.summary_retained
    for idx, event in enumerate(rollout["compression_events"]):
        if event["trigger_chunk"] > ask_chunk:
            break  # 按时间顺序，后面的都没发生
        compressed_chunks = set(event.get("compressed_thinks_chunks", []))
        if bitmap.support_chunks & compressed_chunks:
            if bitmap.summary_retained.get(idx, False):
                return "in_compressed"
    
    # ═══════════════════════════════════════════════════
    # Case 5: in_history_only — 证据在过去，但不在任何可见源中
    # ═══════════════════════════════════════════════════
    # 包含两种情况：
    # a) think 在 recent_thinks 里但没记住答案（retained_and_present 非空但都是 False）
    # b) think 已被压缩掉但 summary 也没保留
    # 两种都需要 recall 回历史帧才能看到
    if support_end < ask_chunk:
        return "in_history_only"
    
    return "unavailable"
```

**整个函数不碰任何 `.text` 字段。** 数据流：

```
snapshot 结构化字段                          判定方式
──────────────────                          ──────────
chunk_idx, visual_window_start     ─────→   in_visual / in_future (数字比较)
recent_thinks[*].chunk             ─────→   in_recent_thinks (set 交集)
compressed_segments[*].time_range  ─────→   in_compressed (区间包含)
                                            ↑
预计算 retention_bitmap            ─────→   thinks_retained[c] / summary_retained[idx] (bool lookup)
```

#### 复杂度对比

| | 旧方案 | 新方案 |
|--|--------|--------|
| **内容匹配次数** | O(cards × chunks × recent_thinks_size) ≈ 23 × 60 × 10 = **~13,800** | 预计算 O(cards × support_chunks + cards × comp_events) ≈ **~115**；逐 chunk **零** |
| **匹配方法** | semantic_overlap（embedding + keyword） | keyword_overlap（仅预计算时） |
| **逐 chunk 判定** | 每次调 semantic_overlap | set 交集 + bool lookup, **O(1)** |
| **总耗时** | embedding 加载 + 万次匹配 | 百次 keyword 匹配 + 万次查表 |

#### Ask-Time 选择策略

同一张 card 不枚举所有 chunk，而是**从 rollout 结构直接计算各 availability 的区间边界**：

```python
def select_ask_times(card, rollout, bitmap):
    """
    直接计算每种 availability 的起止 chunk，各取一个代表。
    
    利用 rollout 的三个结构信息定位区间边界：
    1. support_end + VISUAL_WINDOW_CHUNKS → support 离开视觉窗口的时刻
    2. compression_events[*].compressed_thinks_chunks → support 的 think 被压缩的时刻
    3. bitmap.thinks_retained / summary_retained → think/summary 有没有记住答案
    """
    support_start = min(s[0] for s in card.support_spans)
    support_end = max(s[1] for s in card.support_spans)
    num_chunks = rollout["num_chunks"]
    selected = []
    
    # ── in_visual 区间 ──
    # support_chunk 在 ask_chunk 的视觉窗口内 ⟺ ask_chunk ∈ [support_end, support_end + 11]
    visual_lo = support_end
    visual_hi = min(num_chunks - 1, support_end + VISUAL_WINDOW_CHUNKS - 1)
    if visual_lo <= visual_hi:
        selected.append(PlacedTask(card, (visual_lo + visual_hi) // 2, "in_visual"))
    
    # ── in_recent_thinks 区间 ──
    # support 离开视觉窗口后，think 仍在 recent_thinks 里，直到被压缩掉
    # 压缩时刻从 compression_events 直接查：哪个事件的 compressed_thinks_chunks 包含 support_end？
    exit_visual = support_end + VISUAL_WINDOW_CHUNKS
    compressed_at = None
    for event in rollout["compression_events"]:
        if support_end in event.get("compressed_thinks_chunks", []):
            compressed_at = event["trigger_chunk"]
            break
    
    if compressed_at is not None:
        thinks_lo, thinks_hi = exit_visual, compressed_at - 1
    else:
        thinks_lo, thinks_hi = exit_visual, num_chunks - 1  # 从未被压缩
    
    if thinks_lo <= thinks_hi:
        # 前提：think 确实记住了答案（bitmap 预计算）
        if any(bitmap.thinks_retained.get(c, False) for c in bitmap.support_chunks):
            selected.append(PlacedTask(card, (thinks_lo + thinks_hi) // 2, "in_recent_thinks"))
    
    # ── in_compressed 区间 ──
    # support 被压缩后，summary 保留了答案 → 从压缩发生到 summary 被 merge 掉
    for idx, event in enumerate(rollout["compression_events"]):
        compressed_chunks = set(event.get("compressed_thinks_chunks", []))
        if bitmap.support_chunks & compressed_chunks and bitmap.summary_retained.get(idx, False):
            comp_lo = event["trigger_chunk"] + VISUAL_WINDOW_CHUNKS
            comp_hi = num_chunks - 1  # 简化：假设 summary 存活到视频结束
            if comp_lo <= comp_hi:
                selected.append(PlacedTask(card, (comp_lo + comp_hi) // 2, "in_compressed"))
                break
    
    # ── in_history_only 区间 ──
    # 所有可见源都没了
    last_visible = max(
        thinks_hi + 1 if thinks_lo <= thinks_hi else exit_visual,
        support_end + VISUAL_WINDOW_CHUNKS + 3,
    )
    history_lo = last_visible
    history_hi = min(num_chunks - 1, support_end + VISUAL_WINDOW_CHUNKS + 20)
    if history_lo <= history_hi:
        selected.append(PlacedTask(card, (history_lo + history_hi) // 2, "in_history_only"))
    
    # ── in_future: 仅 E2 ──
    if card.family == "E2" and support_start >= 5:
        selected.append(PlacedTask(card, max(0, support_start - 8), "in_future"))
    
    return selected
```

**不枚举 chunk，不扫描 snapshot。** 区间边界全部从 rollout 的结构化字段直接算出：

| 区间边界 | 数据来源 |
|---------|---------|
| support 离开视觉窗口 | `support_end + VISUAL_WINDOW_CHUNKS`（常量运算） |
| think 被压缩的时刻 | `compression_events[*].compressed_thinks_chunks` 包含 support_end 的事件的 `trigger_chunk` |
| summary 生效的区间 | 压缩事件的 `trigger_chunk` 到视频结束（或下一次 merge） |

总复杂度：O(cards × compression_events)，一般 ~23 × 3 = ~69 次结构查找。


#### 对照样本自然产生

```
Task Card: F2 "厨师围裙是什么颜色?" 
  support_span=[5,5], retention_class=low

预计算: thinks_retained[5] = False (think 只写了 "chef in apron"，没写 red)
        summary_retained = {} (还没被压缩过)

直接计算区间:
  in_visual:        ask ∈ [5, 16]   → 选 ask=10  → response
  in_recent_thinks: 跳过 (thinks_retained[5]=False，think 没记住颜色)
  in_compressed:    跳过 (没被压缩 或 summary 也没保留)
  in_history_only:  ask ∈ [18, 25]  → 选 ask=21  → recall

结果: 2 个样本（response + recall），无误判的 in_recent_thinks
```

```
Task Card: E1 "厨师在做什么?"
  support_span=[5,5], retention_class=high

预计算: thinks_retained[5] = True (think 写了 "chef chopping tomatoes")
        summary_retained[0] = True (summary 保留了 "chef chopped tomatoes")

直接计算区间:
  in_visual:        ask ∈ [5, 16]   → 选 ask=10  → response
  in_recent_thinks: ask ∈ [17, 22]  → 选 ask=19  → response (from memory)
  in_compressed:    ask ∈ [25, 40]  → 选 ask=32  → response (from summary)
  in_history_only:  跳过 (summary 一直保留着)

结果: 3 个 response 样本，覆盖三种 response 来源
```

**retention_class=low 的 F1/F2 问题自然倾向于产生 recall，
retention_class=high 的 E1/S1 问题自然倾向于产生 response。
不是靠不可靠的关键词匹配判断，而是靠预计算 + 结构位置。**

#### Difficulty Margin

每个 placed task 计算难度分数 [0, 1]，用于后续统计和难度梯度控制：

```python
base_difficulty = {
    "in_visual": 0.1, "in_recent_thinks": 0.3, "in_compressed": 0.5,
    "in_history_only": 0.8, "in_future": 0.2,
}
# + 时间衰减 (离证据越远越难)
# + retention_class 调整 (low → +0.1, high → -0.1)
# + 压缩次数惩罚
```

### 3.5 阶段 3-C: Action 映射 + SFT 样本生成

> 合并旧阶段 3 的 action 判定 + 旧阶段 4 的样本生成。

**目标**: availability → action 确定性映射，然后生成 SFT 训练样本。

#### Action 映射（一行查表，替代旧 determine_gold_action）

```python
ACTION_MAP = {
    "in_visual":        ("response", "evidence_visible_in_frames"),
    "in_recent_thinks": ("response", "answer_in_text_memory"),
    "in_compressed":    ("response", "answer_in_compressed_summary"),
    "in_history_only":  ("recall",   "evidence_left_all_memory"),
    "in_future":        ("silent",   "evidence_not_yet_occurred"),
}

def map_action(placed_task):
    return ACTION_MAP[placed_task.availability]
```

**不再有优先级冲突**：availability 在 3-B 已确定，action 是确定性的。

#### Visibility Matrix（保留，适配新结构）

```python
visibility = {
    "at_chunk": 45,
    "at_time": 90,
    "availability": "in_history_only",     # 新增：来自 3-B
    "visual_window": [68, 92],
    "evidence_in_window": false,
    "answer_in_recent_obs": false,
    "answer_in_compressed": false,
    "difficulty_margin": 0.82,             # 新增：来自 3-B
}
```

#### SFT 样本生成

**response 样本**：

```python
async def generate_response_sample(placed, rollout, client):
    snapshot = rollout["snapshots"][placed.ask_chunk]
    current_obs = rollout["thinks"][placed.ask_chunk]["think"]
    
    # 根据 availability 决定给 397B 什么证据
    if placed.availability == "in_visual":
        evidence = "Visible in current frames."  # + 附带帧
    elif placed.availability == "in_recent_thinks":
        evidence = get_matching_thinks(snapshot, placed.card.answer_keywords)
    elif placed.availability == "in_compressed":
        evidence = get_matching_compressed(snapshot, placed.card.answer_keywords)
    
    response_text = await call_397b_response(
        question=placed.card.question,
        evidence=evidence,
        gold_answer=placed.card.canonical_answer,
        answer_form=placed.card.answer_form,
    )
    
    output = f"<think>{current_obs}</think><action>response</action><response>{response_text}</response>"
    return build_sample(placed, snapshot, output)
```

**recall 样本**（仍拆为 query + response 两条独立样本）：

与旧阶段 4 的 recall 生成逻辑相同（§5.2），关键约束不变：
- recall 拆为 2 条独立样本
- query generator 不接收 gold_answer
- recall_result 只用学生可访问内容（student think / compressed summary / historical frames）
- post-recall 不输出 think
- recall failure / distractor → response 内容表达"检索结果不足以确定"

**Recall Result 来源**（不变）：

| 来源 | 内容 | 适用 |
|------|------|------|
| student think | 学生自己写过的 think 文本 | 文本细节 recall |
| compressed summary | 学生自己写的压缩摘要 | 压缩记忆 recall |
| historical frames | 4s 的原始视频帧 | 视觉细节 recall |

**绝不使用 teacher_caption 作为 recall_result**。

**event_watch 样本**（仅 E2 × in_future，简化版）：

```python
def generate_event_watch_samples(placed, rollout):
    """只产生 2 个样本（不再有中间 pending_silent）"""
    # Sample 1: silent at ask_time（事件还没发生）
    ask_obs = rollout["thinks"][placed.ask_chunk]["think"]
    sample1 = {
        "output": f"<think>{ask_obs}</think><action>silent</action>",
        "metadata": {"pending_question": placed.card.question},
    }
    
    # Sample 2: response at trigger_time（事件发生了）
    trigger_chunk = placed.card.support_spans[0][0]
    trigger_obs = rollout["thinks"][trigger_chunk]["think"]
    sample2 = {
        "output": (f"<think>{trigger_obs}</think>"
                   f"<action>response</action>"
                   f"<response>{placed.card.canonical_answer}</response>"),
    }
    return [sample1, sample2]
```

#### 标准答案来源（不变）

```python
gold_answer = {
    "answer": "Approximately 1 teaspoon",
    "source": "teacher_caption",
    "support_facts": ["chef sprinkles...approximately one teaspoon"],
    "confidence": 0.55,
    "answer_form": "short_factual",     # 验证等级
    "verifiable": false,                # 非精确匹配
}
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
  
  "output": "<action>response</action><response>Based on the retrieved frames, the chef added approximately one teaspoon of white granular seasoning from a small bowl.</response>"
}
```

> **v5.0→v5.1→v5.2 修正**:
> - input 必须包含 `recall_context` 字段，携带原始问题和 recall query（v8.0: 替代旧 pending）
> - post-recall response **不输出 think**（think 已在 recall_query 样本中输出，避免 runtime memory 重复写入）
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
  
  "output": "<think>Chef tears basil leaves near covered pot, green fragments on counter surface.</think><action>compress</action><summary>{\"time_range\":[20,34],\"text\":\"[20-28] Oil heated in stainless pot, garlic browned, tomato quarters added. [28-34] White seasoning from small bowl added, stirred with wooden spoon, sauce reddish and bubbling. Entities: pot_1(stainless, right burner), wooden_spoon.\"}</summary>"
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
| 1 | think 每步立即入文本记忆，不延迟 | §2.1 | `MemoryState.add_think()` |
| 2 | 文本记忆覆盖时间 > 视觉窗口，两者重叠 | §2.1 | `snapshot()` 中 `visual_window_start` 独立于 `recent_thinks` |
| 3 | 80% token 预算触发压缩（精确 tokenizer） | §2.3 | `should_compress()` + `count_recent_tokens()` |
| 4 | compress range 只覆盖 INPUT ���的 recent_thinks，不含当前 think | §2.1 | `pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]` |
| 5 | 系���执行：先替换→再 append 当前 think | §2.1 | `memory.compress()` 后 think 已在 memory 中 |
| 6 | summary 只能引用 thinks，不能偷看视觉帧 | §2.1 | `verify_summary_provenance()` |
| 7 | C1 teacher 多维评分选最优范围，C2 模型自选 | §2.3 | `score_range_for_compression()` 5 维评分 |
| 8 | action 优先���：用户交互 > query 触发 > compress > silent | §2.1 | pipeline Pass 3-C `interaction_chunks` 优先级过滤 |
| 9 | recall_result 不含 teacher_caption、不含未来内容 | §3.5 | `_get_correct_result()` 只用 `observations[ec]['think']` |
| 10 | recall failure/distractor 时 response 内容表达信息不足，不用 gold_answer 强答 | §3.5 | `is_failed_recall` → "信息不足" response + `verify_information_flow()` |
| 11 | post-recall response 不输出 think（避免重复��入 memory） | §1.3 | `sample2_output` 无 `<think>` 标签 |
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
