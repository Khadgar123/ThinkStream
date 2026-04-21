# 流视频 Agent 数据构造方案

> 版本: v6.1 | 日期: 2026-04-21
>
> 核心设计：
> - Per-timestep 独立样本，三层文本隔离（teacher / think / summary）
> - `<think>` 每步立即入文本记忆，文本记忆覆盖时间 > 视觉滑动窗口
> - 80% token 预算触发压缩，teacher 多维评分选最优范围
> - C1 系统指定范围 → C2 模型自选范围 → C3 模型自主触发（渐进训练）
> - Action 优先级：用户交互 > pending > compress > silent
> - Summary 只能引用 thinks 内容，不能偷看视觉帧补充细节
>
> 配套文档：
> - **`sft_engineering.md`**：SFT 训练工程设计（注意力掩码、数据加载、多模型兼容、训练课程）
> - `TRAINING.md`：旧版冷启动训练说明（已过时，保留供参考）
>
> v5.6 变更：
> - 新增 §13 SFT 工程集成要点（与 `sft_engineering.md` 的接口定义）
> - 修正 §5.2 post-recall 样本的 pending 字段一致性说明
> - 修正 §8 配置常量：补充压缩相关 special token 列表
> - 删除 §12 开放问题 #3（已在 §2.3 合法范围约束中解决）

---

## 1. 架构原则

### 1.1 训练格式：Per-Timestep Re-render（v6.1 定稿）

**每个 2s chunk 构造为独立训练样本**（单轮 messages）。推理时每步重新构造完整 input，不依赖跨步 KV cache。

```
单条训练样本 = 推理时一步的精确快照 {
    messages: [
        {role: system, content: 4-action protocol},
        {role: user, content: [
            {type: video, video_start: 26, video_end: 50, nframes: 24},
            {type: text, text: "<compressed>...</compressed>\n[40-42] think...\nQuestion?"}
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
| Uncertain (不确定) | 20-60 tokens | "I cannot confirm the exact amount..." |

---

## 2. 记忆架构

### 2.1 记忆状态（单层，模型视角）

每个 timestep，系统从外部拼装一份完整 input 给模型。模型只看到这一层：

```
模型看到的记忆状态:
  compressed_segments: [summary_0, summary_1, ...]   ← 最多 5 段
  recent_thinks: [think_0, think_1, ..., think_N]    ← 未被压缩的
  pending_questions: [...]                            ← 用户待回答的问题（如有）
  visual_window: [最近 12 chunks 的帧]                ← FIFO
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

**Summary provenance 约束**：compressed_summary 只能引用被压缩的 recent_thinks 中的信息。
即使视觉帧仍在 visual_window 中，summary 也不能偷看帧补充 thinks 中没有的细节。
否则 summary 变成"重新看视频写摘要"，破坏三层隔离。

**Action 优先级**（同一步多条件同时满足时）：
```
post-recall response > 用户当前问题 response/recall > pending trigger response > compress > silent
```
用户交互优先，compress 可延后一帧。训练数据构造时按此优先级确定 gold_action。

### 2.2 单条训练样本的 input 结构

```
[System prompt: 协议说明, ~150 tok]
[Memory block:]
  <compressed>{"time_range":[0,20],"text":"summary_0"}</compressed>    ← 压缩段
  <compressed>{"time_range":[20,40],"text":"summary_1"}</compressed>
  [40-42] think at t=40-42                               ← 未压缩的 think
  [42-44] think at t=42-44
  ...（recent_thinks 可能和 visual_window 重叠）
[Visual window: 最近 12 chunks 的视频帧, ~1536 vision tok]
[User input: 当前问题 / compress_trigger / recall_result / 空]
```

这是学生模型**真实能看到的全部信息**。注意 recent_thinks 和 visual_window 在时间上可能重叠。

### 2.3 压缩触发与范围选择

**触发条件**: `recent_thinks 总 token ≥ RECENT_THINKS_TOKEN_BUDGET × 80%`（~480 tok）

具体行为取决于 think 长度：
- 长 think（~60 tok/条）→ ~8 条就触发（早）
- 中等 think（~50 tok/条）→ ~10 条触发
- 短 think（~40 tok/条）→ 12 条才触发（由 item 硬上限兜底）

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
  + pending_overlap_penalty  # 覆盖 pending 相关内容 → 重罚
  + event_boundary_penalty   # 打断进行中的事件 → 罚
  - token_saving_gain        # 省的 token 越多越好
  - reconstructability_bonus # 内容重复/简单 → 容易压缩 → 加分

best_range = argmin(score)   # 综合最优
```

目标不是"丢最少信息"，而是**最大化未来可回答性 / 最小化记忆 regret**。

**合法范围约束**：
1. 连续 4-8 条 `recent_thinks`
2. 尽量是事件边界完整段
3. 不能包含 pending question 强相关的 thinks
4. 压缩后 token gain 足够
5. summary 能保留关键实体、OCR、数量

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

## 3. 数据构造 Pipeline（5 阶段）

```
阶段 1: Teacher Evidence Graph         [397B + 视频, ~4h]
阶段 2: Question-blind Streaming Rollout [397B + 视频, ~4h]
阶段 3: Task Planning                   [397B + evidence, ~1h]
阶段 4: Question-aware Forks            [397B, ~2h]
阶段 5: Verify + Filter                 [规则 + 小模型, ~30min]
```

### 3.1 阶段 1: Teacher Evidence Graph

**目标**: 为每个 2s chunk 生成详细结构化事实图。这是**隐藏教师信息**，不进入 SFT。

**397B 输入**: 最近 12 chunks 的 24 帧滑窗 + 前面所有 chunk 的 caption 文本（保持上下文一致性）

> 注：使用 24 帧滑窗（而非仅当前 2 帧），与推理模型的视觉窗口一致，确保 teacher 看到的上下文不少于 student。

**397B 输出** (per chunk):
```json
{
  "time": [28, 30],
  "visible_entities": [
    {"id": "chef_1", "attributes": ["red apron", "short hair"], "action": "sprinkling seasoning"},
    {"id": "bowl_1", "attributes": ["small", "white", "pinch bowl"], "held_by": "chef_1"},
    {"id": "pot_1", "attributes": ["stainless", "front-right burner"]}
  ],
  "atomic_facts": [
    {"fact": "chef sprinkles white granular seasoning from small bowl into pot", 
     "confidence": 0.88, "support_level": "direct_current_chunk",
     "target_resolution_visible": true},
    {"fact": "amount approximately one teaspoon", 
     "confidence": 0.55, "support_level": "direct_current_chunk",
     "target_resolution_visible": false, "uncertainty": "visual estimate"}
  ],
  "state_changes": ["seasoning added to sauce"],
  "ocr": [],
  "spatial": "chef center, pot on right burner",
  "not_observable": ["sizzling sound", "aroma"],
  "relation_to_previous": "continuation of cooking sequence"
}
```

**用途**：
- 造任务（知道哪里有可问的细节）
- 生成标准答案（来自 atomic_facts）
- 验证 student think 是否遗漏关键信息
- 验证 compressed_summary 是否保留必要内容
- 生成 hard negatives / distractors
- 目标分辨率可见性检查（confidence < 0.5 的 fact 不适合做任务）

**并发**:
```
视频内串行（需要前文 context）, 视频间并行 16 路
每请求: 24帧 + text context = ~38K input, max_tokens=1024, thinking=ON
每视频: 60 chunks × 15s = 900s = 15 min
300 视频 ÷ 16 = ~280 min ≈ 4.7h
```

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
    "pending_questions": [...],
    "visual_window_start": 3,
}
```

**关键**:
- 压缩 summary 基于**当前 thinks**（不是 teacher caption），且不知道未来问题
- 压缩范围由 `choose_optimal_compress_range(pre_action_thinks)` 从 pre-action snapshot 中选出
- 压缩事件中 `compressed_thinks_chunks` 记录的是真实被压缩的 chunk 列表（不依赖模型输出的 time_range）

**并发配置**:
```
视频内串行, 视频间 16 并行
每请求: ~38K input (24帧 + memory), max_tokens=256, thinking=OFF
每视频: 60×12s + 4×15s(compress) = 780s ≈ 13 min
300 视频 ÷ 16 = ~244 min ≈ 4h
```

### 3.3 阶段 3: Task Planning

**目标**: 基于 teacher evidence graph + student rollout，设计任务并确定标准答案。

**输入**:
- Teacher evidence graph (全部 atomic_facts)
- Student rollout (所有 thinks + summaries + memory snapshots)
- 视觉帧（验证可见性）

**对每种任务模式，独立挖掘**:

#### Action Minimality 判断（最重要）

每个候选任务必须先判断 **gold_action**:

```python
def determine_gold_action(task, snapshot_at_ask_time):
    """确定正确的 action：最小动作原则"""
    answer_keywords = extract_keywords(task["gold_answer"])
    
    # 1. 答案在当前视觉窗口的帧中可见？
    if answer_visible_in_frames(task, snapshot_at_ask_time["visual_window_range"]):
        return "response"  # 直接回答
    
    # 2. 答案在 recent_thinks 中？
    for obs in snapshot_at_ask_time["recent_thinks"]:
        if keyword_overlap(obs, answer_keywords) > threshold:
            return "response"  # 从文本记忆直接回答
    
    # 3. 答案在 compressed_summary 中？
    for seg in snapshot_at_ask_time["compressed_segments"]:
        if keyword_overlap(seg["text"], answer_keywords) > threshold:
            return "response"  # 从压缩记忆直接回答（不需要 recall!）
    
    # 4. 答案不在任何当前可见信息中 → 需要 recall
    # 但 recall 只能检索学生自己的 memory 或历史帧
    if answer_in_historical_thinks_or_frames(task):
        return "recall"
    
    # 5. 答案完全不可得 → action 仍是 response（uncertain 类型）
    return "response"  # answerability = "unanswerable"
```

**这解决了关键问题**：如果 compressed_summary 已经包含答案（如 "added salt ~1tsp"），gold_action 应该是 response 而不是 recall。

#### 标准答案来源

```python
gold_answer = {
    "answer": "Approximately 1 teaspoon",
    "source": "teacher_caption",  # 来源
    "support_facts": ["chef sprinkles...approximately one teaspoon"],
    "support_frames": ["f28", "f29"],
    "confidence": 0.55,  # 如果 confidence 低，标为 uncertain 类型
    "allowed_variants": ["about 1 tsp", "roughly a teaspoon"],
    "forbidden_claims": ["exactly 1 teaspoon"],  # 过度确定
}
```

#### Visibility Matrix

每个任务必须记录：
```python
visibility = {
    "at_time": 44,
    "student_can_see": {
        "visual_frames": ["36-38", "38-40", "40-42", "42-44"],
        "compressed_summaries": ["[0-20] ...", "[20-40] ..."],
        "recent_thinks": ["[40-42] ...", "[42-44] ..."],
    },
    "answer_location": "compressed_summary_1",  # 或 "not_visible" 需要 recall
    "gold_action": "response",  # 因为 summary 中有答案
}
```

#### 各模式挖掘（简述）

| 模式 | 挖掘条件 | gold_action | 特殊要求 |
|------|---------|-------------|---------|
| A1 | 答案在帧中可见 | response | 无 |
| A5-A11 | 答案不在任何可见信息中 | recall | evidence 在历史帧/obs 中 |
| B1-B5 | 答案被压缩丢失，只能从历史帧 recall | recall | summary 中无答案 |
| B8 | 压缩后，答案在 summary 中 | response | 验证 summary 包含答案 |
| 不可回答 | 视频中从未出现该信息 | response(uncertain) | 回答"无法确定" |

### 3.4 阶段 4: Question-aware Forks

**目标**: 从缓存的 memory snapshot 分叉，注入问题，生成最终训练样本。

**核心约束**: 
- **过去的 memory 不可重写**（question-blind 原则）
- 只有 ask_time 及之后的 action 受问题影响
- 标准答案作为 anchor 约束 397B 生成

**流程**:
```python
def generate_fork(task, snapshots, teacher_evidence):
    ask_chunk = task["ask_chunk"]
    
    # 从缓存 snapshot 直接取 memory state（不重新生成！）
    memory_state = snapshots[ask_chunk]
    
    # 构造训练样本的 input
    sample_input = build_input(
        system_prompt=SYSTEM_PROMPT,
        compressed=memory_state["compressed_segments"],
        recent_thinks=memory_state["recent_thinks"],
        visual_window=get_frames(memory_state["visual_window_range"]),
        user_input=task["question"],
    )
    
    # 构造训练样本的 output（受标准答案约束）
    if task["gold_action"] == "response":
        # 包括 uncertain 类型（answerability="unanswerable" 时生成不确定回答）
        output = generate_response_output(task, memory_state)
    elif task["gold_action"] == "recall":
        output = generate_recall_output(task, memory_state)
    
    return {"input": sample_input, "output": output, "metadata": task}
```

**对于 recall 任务**（拆为 2 条独立 per-timestep 样本）:

```python
def generate_recall_samples(task, memory_state):
    # Step 1: 生成 recall query
    # 注意: query generator 只接收问题 + 可见 memory context，不接收 gold_answer
    query = generate_query(task["question"], visible_context=memory_state)
    validate_no_answer_in_query(query, task["gold_answer"])
    
    # Step 2: 模拟检索结果（只用学生可访问的信息）
    # 注意：retrieval archive 是系统侧维护的，不在模型可见的 snapshot 中
    recall_result = simulate_retrieval(
        query=query,
        all_past_thinks=thinks_up_to_ask_time,  # 系统侧历史
        student_summaries=memory_state["compressed_segments"],
        historical_frames=available_frames,
        noise_level=sample_noise(),  # 70/20/5/5 分布
    )
    
    # Step 3: 基于检索结果生成 response
    post_recall_response = generate_response_from_result(
        question=task["question"],
        recall_result=recall_result,
        gold_answer=task["gold_answer"],  # 仅作为约束，不作为生成输入
    )
    
    # 样本 1: recall query（模型发起检索）
    sample1 = {
        "input": build_input(memory_state, user_input=task["question"]),
        "output": f'<think>当前视觉obs</think><action>recall</action><query>{query}</query>',
    }
    
    # 样本 2: post-recall response（系统返回结果后模型回答）
    # 关键: pending question 必须出现在 input 中
    memory_with_pending = add_pending(memory_state, task["question"], "recall")
    sample2 = {
        "input": build_input(memory_with_pending, 
                            user_input="Continue following the protocol to respond.",
                            recall_result=recall_result),
        "output": f'<action>response</action><response>{post_recall_response}</response>',
    }
    
    return [sample1, sample2]
```

**重要变更（相对 v5.0）**：
- recall 必须拆为 2 条独立样本，不能打包成一个 output
- `recall_result` 是系统注入的 input，不出现在模型 output 中
- post-recall sample 的 input 必须包含 `pending question`（原始问题 + recall query）
- post-recall sample **不输出 think**（避免与 sample1 重复写入 memory）
- query generator 不接收 gold_answer，只用问题 + 可见 memory context 生成 query
- recall failure / distractor 时不给 gold_answer，生成 uncertain response

**Recall Result 的三种来源**（只用学生可访问内容）:

| 来源 | 内容 | 适用 |
|------|------|------|
| student think | 学生自己写过的 think 文本 | 文本细节 recall |
| compressed summary | 学生自己写的压缩摘要 | 压缩记忆 recall |
| historical frames | 4s 的原始视频帧 | 视觉细节 recall |

**绝不使用 teacher_caption 作为 recall_result**。

### 3.5 阶段 5: Verify + Filter

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

## 4. 任务分类与分布

### 4.1 Action 类型（基于 Action Minimality）

| Gold Action | 触发条件 | 说明 |
|-------------|---------|------|
| **silent** | 无问题，正常观察 | 输出 think |
| **response (from frames)** | 答案在当前视觉帧中 | 直接从画面回答 |
| **response (from memory)** | 答案在 recent_thinks 中（帧已滑出） | 从文本记忆回答 |
| **response (from compressed)** | 答案在 compressed_summary 中 | 从压缩摘要回答（不需要 recall） |
| **response (uncertain)** | 无可靠证据 | 回答"不确定" |
| **recall** | 答案不在任何可见信息中 | 需要检索历史 |
| **compress (C1)** | 系统触发 + 指定范围 | 按 teacher 指定范围压缩 |
| **compress (C2)** | 系统触发，无指定范围 | 模型自选范围压缩 |
| **merge_compress** | compressed_segments > 5 段 | 合并最老两段（摘要的摘要） |

### 4.2 Recall 的必要条件

只有同时满足以下全部条件时，gold_action 才是 recall：
1. 答案不在当前视觉帧中
2. 答案不在 recent_thinks 中
3. 答案不在 compressed_summaries 中
4. 答案存在于历史 thinks / frames 中（可检索到）

**如果 compressed_summary 已包含答案 → response，不是 recall！**

### 4.3 补充任务类型

| 类型 | 说明 | 占比 |
|------|------|------|
| 不可回答 | 视频中无证据 | 5% |
| 歧义问题 | 多个可能目标 | 3% |
| Recall 失败后的 uncertain response | 检索无结果 | 3% |
| 长时间计数 | "出现了几次" | 2% |
| 时间定位 | "什么时候发生的" | 2% |

### 4.4 各 Phase 配比

| Phase | Silent | Response | Recall | Compress | Special |
|-------|--------|----------|--------|----------|---------|
| 1 (协议) | 65% | 30% | 0% | 0% | 5% uncertain |
| 2 (Recall) | 50% | 20% | 20% | 0% | 10% |
| C1 (压缩) | 45% | 18% | 15% | 12% | 10% |
| C2 (自选) | 42% | 18% | 15% | 15% | 10% |
| 5 (混合) | 45% | 20% | 15% | 10% | 10% |

**分布按 episode-level 控制**（有无 recall / 有无 compress 的 episode 比例），不是人工拉 turn-level 比例。

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
      "recent_thinks": ["...same as above..."],
      "pending": [
        {
          "question": "How much salt did the chef add?",
          "since": 60,
          "type": "awaiting_recall_response"
        }
      ]
    },
    "visual_window": {
      "video_start": 36, "video_end": 60, "frames": 24,
      "recalled_frames": {
        "time_range": [28, 30], "n_frames": 4, "source": "historical_frames",
        "frame_indices": [28, 29, 30, 31]
      }
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
> - input 必须包含 `pending` 字段，携带原始问题和 recall query
> - post-recall response **不输出 think**（think 已在 recall_query 样本中输出，避免 runtime memory 重复写入）
> - response 避免确定说 "salt"（仅可见 white granular seasoning），除非有 OCR/上下文支持
> - recall failure / distractor 时，response 必须是 uncertain（不能用 gold_answer 强答）

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

## 7. 后续增强

### 7.1 Student Rollout Augmentation (DAgger)

第一轮 SFT 后，用训练后的 student 模型跑视频：
- Student 生成的 thinks 质量更低（短、漏信息）
- 基于 student 的真实 memory 构造第二批 recall/compress 数据
- 解决"训练时 memory 质量高，推理时质量低"的分布偏移

### 7.2 评估指标

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

# 并发
PASS1_CONCURRENT = 16                  # Evidence Graph
PASS2_CONCURRENT = 16                  # Rollout
PASS3_CONCURRENT = 32                  # Task Planning (文本为主)
PASS4_CONCURRENT = 32                  # Forks (文本为主)

# 397B
THINKING_ON = ["pass1_evidence", "pass3_task_plan"]
THINKING_OFF = ["pass2_rollout", "pass4_forks"]
MAX_TOKENS_EVIDENCE = 1024
MAX_TOKENS_OBSERVATION = 128
MAX_TOKENS_COMPRESS = 512
MAX_TOKENS_TASK = 2048

# Special Tokens（SFT 代码 init_processor 中添加）
SPECIAL_TOKENS_BASE = [
    "<silent>", "<response>", "<think>", "</think>",
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
]
SPECIAL_TOKENS_PER_TIMESTEP = [
    "<compressed>", "</compressed>",
    "<pending>", "</pending>",
    "<compress_trigger>", "</compress_trigger>",
    "<summary>", "</summary>",
]
```

### 8.1 vLLM 部署方案（两套配置）

各 Pass 对 vLLM 的需求差异：

| Pass | 输入类型 | Thinking | 输出上限 | 并发 | 单请求图片 |
|------|---------|----------|---------|------|-----------|
| **Pass 1** Evidence | 视觉 (24 帧) + 文本 | ON | 1024 | 16 | 24 帧 |
| **Pass 2a** Observation | 视觉 (24 帧) + 记忆 | OFF (`/no_think`) | 128 | 16 | 24 帧 |
| **Pass 2b** Compress | 纯文本 | OFF (`/no_think`) | 512 | 16 | 0 |
| **Pass 3** Tasks | 纯文本 | ON | 2048 | 32 | 0 |
| **Pass 4** Forks | 纯文本 | OFF (`/no_think`) | 512 | 32 | 0 |
| **Pass 5** Verify | 规则检查，无 LLM | — | — | — | — |

**关键差异**：视觉 Pass（1, 2a）每请求发送 24 帧 base64 图片（~1.2MB），需要多模态处理能力、
更大 `max-model-len`、和更低并发；纯文本 Pass（2b, 3, 4）只有 ~1K token 输入，可以开更高并发。

| 参数 | 视觉 Pass (1, 2a) | 文本 Pass (2b, 3, 4) |
|------|-------------------|---------------------|
| `--limit-mm-per-prompt` | `image=24`（必须） | 不需要 |
| `--max-model-len` | `65536`（容纳帧 token） | `16384` 即可 |
| `--gpu-memory-utilization` | `0.90`（图片占显存） | `0.95`（纯文本更高效） |
| `--max-num-seqs` | `16`（图片大，限制并发） | `64`（文本小，可加大） |
| `--enable-prefix-caching` | OFF（图片变化��，命中率低） | ON（system prompt 共��） |

**方案 A：两套 vLLM 实例（推荐）**

视觉 Pass 和文本 Pass 分别启动不同配置的 vLLM 实例，最大化各自吞吐。
pipeline 按 pass 切换 `--api_base`。

```bash
# 实例 1: 视觉模式 (Pass 1 + Pass 2a)
# 需要多模态支持，大 context，低并发
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt image=24 \
  --enforce-eager \
  --trust-remote-code \
  --port 8000

# 实例 2: 纯文本模式 (Pass 2b + Pass 3 + Pass 4)
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

**各 Pass 上下文安全估算**：

```
Pass 1:  输入 ~2,736 tok + 输出 1,024 = ~3,760  × 16 并发 = 60K (vs 65K 上限 ⚠️)
Pass 2a: 输入 ~3,136 tok + 输出 128   = ~3,264  × 16 并发 = 52K ✓
Pass 2b: 输入 ~650 tok   + 输出 512   = ~1,162  无视觉 ✓
Pass 3:  输入 ~1,300 tok + 输出 2,048 = ~3,348  × 32 并发 = 107K (文本 batch) ✓
Pass 4:  输入 ~650 tok   + 输出 512   = ~1,162  × 32 并发 = 37K ✓
```

---

## 9. 补充设计（解决已知缺口）

### 9.1 Pending Question 状态表示

S3/S4 任务中，用户提问后模型需要在多个后续 timestep "记住在等什么"。

**解决**：在 memory block 中增加 `<pending>` 字段：

```
[Memory block:]
  <compressed>{"time_range":[0,20],"text":"..."}</compressed>
  [20-22] think...
  [22-24] think...
  <pending since="24">Tell me when the chef adds the basil.</pending>   ← 待解决问题
[Visual window: ...]
[User input: (无新问题，只有新视频)]
```

模型输出：
- 如果事件未发生：`<think>Sauce still simmering, no basil yet.</think><action>silent</action>`
- 如果事件发生：`<think>Chef tearing basil leaves over pot.</think><action>response</action><response>The chef is now adding basil to the sauce.</response>`

**pending 状态生命周期**：
1. **pending_start**：用户提出 event-watch 请求时，模型输出 `silent`（不要马上答）。系统在该 output 后将问题加入 pending。
2. **pending_silent**：后续 timestep 中，pending 字段出现在 input 里，事件未发生时模型继续 `silent`。
3. **pending_response**：事件发生时，模型输出 `response` 回答 pending 问题。系统删除已解决的 pending。

训练数据为每个 pending 任务生成 3 条样本（start + mid-silent + trigger-response）。

### 9.2 Recalled Frames 的精确训练格式

Recall 返回帧如何在 input 中呈现，必须和正常视觉窗口区分：

```
[Visual window: 最近 12 chunks 帧]                        ← 正常窗口
[Recalled frames: 4帧, time=28-32, source=historical]    ← 明确标注为 recalled
[User input: "Continue following the protocol."]
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

### 9.4 补充负样本任务

| 类型 | 描述 | 正确行为 | 占比 |
|------|------|---------|------|
| **不存在内容** | 用户问视频中从未出现的事物 | response(uncertain): "I don't see evidence of..." | 3% |
| **多候选歧义** | "哪个人？"但有多个人 | response(clarify): "There are multiple people..." | 2% |
| **记忆与画面冲突** | 画面变了但记忆还是旧的 | response 基于当前画面（最新优先） | 2% |
| **Recall 返回错误** | 检索结果不含答案 | response(uncertain): "The retrieved info doesn't show..." | 3% |
| **用户纠正前一个回答** | "不对，我问的是另一个" | response: 基于新理解重新回答 | 2% |
| **计数任务** | "出现了几次" | response: 基于 thinks 中的出现次数 | 2% |

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

### 9.6 Pending-question-aware vs Question-blind Compression

两种压缩场景必须分开：

**场景 A: Question-blind compression**（用户还没问问题）
```
[thinks 0-20]    → compress → summary 按通用重要性保留
不知道未来问什么，所有类似动作用相同粒度压缩
```

**场景 B: Pending-question-aware compression**（有未回答的问题）
```
<pending>"Tell me when basil is added"</pending>
[thinks 20-40]   → compress → summary 必须保留 basil 相关信息

因为有 pending question，模型应该知道保留与 basil 相关的 think。
如果某条 obs 含 "basil" 或相关实体 → 保留为未压缩，不压缩它。
```

**训练数据中两种比例**：
- Question-blind: 70%（更常见）
- Pending-aware: 30%

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
| 8 | action 优先���：用户交互 > pending > compress > silent | §2.1 | pipeline Pass 4 `interaction_chunks` 优先级过滤 |
| 9 | recall_result 不含 teacher_caption、不含未来内容 | §3.4 | `_get_correct_result()` 只用 `observations[ec]['think']` |
| 10 | recall failure/distractor 必须生成 uncertain response | §3.4 | `is_failed_recall` → uncertain answer + `verify_information_flow()` |
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
         (silent ~2,600 + response ~1,200 + uncertain ~200)

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
3. **DAgger 执行计划**: Phase 1-2 完成后训练 v0 模型 → v0 跑 100 视频 → 基于 v0 memory 造 2000 条补充数据 → 混入 Phase C1 训练。

---

## 13. SFT 工程集成要点

> 完整设计见 `sft_engineering.md`。本节定义数据构造 pipeline 与 SFT 训练代码之间的接口。

### 13.1 Pipeline 输出 → SFT 输入的契约

Pipeline Pass 4 产出的每条样本必须包含以下字段，SFT `PerTimestepDataset` 依赖这些字段构建 model inputs：

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
      "recent_thinks": ["[20-22] ...", "[22-24] ..."],
      "pending": [{"question": "...", "since": 24, "type": "awaiting_recall_response"}]
    },
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
| `input.memory.pending` | 否 | 仅 recall_response / pending_response 类型有 |
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
    <pending> </pending>                 — memory 内：待解决问题
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
    <pending>{"since":44,"question":"Tell me when basil added"}</pending>
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

Pipeline Pass 5 输出 `train.jsonl` / `val.jsonl` / `test.jsonl`。
SFT 训练时按 `phase` 字段过滤样本：

```python
# Phase 1 训练：只加载 phase="1" 的样本
samples = [s for s in all_samples if s["phase"] == "1"]
```

各 Phase 的训练顺序、学习率、epoch 数等超参见 `sft_engineering.md` §7。
