# 流视频 Agent 数据构造方案

> 版本: v5.0 | 日期: 2026-04-21
>
> 基于 v4.1 的全面重构。核心变更：
> - 训练格式改为 per-timestep 独立样本（方案 A）
> - 三层文本严格分离（teacher_caption / student_observation / compressed_summary）
> - 5 阶段 pipeline（Evidence Graph → Rollout → Task Plan → Fork → Verify）
> - Think 改名 observation，压缩到 40-60 tokens
> - 引入 visibility matrix 和 action minimality 验证

---

## 1. 架构原则

### 1.1 训练格式：Per-Timestep 独立样本（方案 A）

**不再使用完整多轮对话作为单条 SFT 样本**。原因：
- 旧 chunk 没有视频输入，但如果对其 assistant 算 loss → 模型学会"无图编描述"
- 压缩在 causal conversation 中无法真正替换（前文 token 仍可 attend）
- 推理时 memory 是外部状态管理，不是 chat history

**方案 A**：每个 timestep 构造为独立训练样本：

```
单条训练样本 = {
    input:  [system] + [memory_state] + [visual_window] + [user_input]
    output: <observation>...</observation><action>X</action>[附加]
}
```

只对 **当前这一步的 output** 计算 loss。历史信息作为 input context，不参与 loss。

### 1.2 三层文本严格分离

```
A. teacher_caption (隐藏，不进 SFT):
   397B 的详细结构化事实，用于造题、验证、生成标准答案。
   学生模型永远看不到这层。

B. student_observation (进 SFT 的 output):
   学生模型需要生成的 40-60 token 简短记忆。
   后续存入 memory，可被压缩。

C. compressed_summary (进 SFT):
   学生模型生成的压缩摘要，替换旧 observations。
```

**三者不能混**：
- teacher_caption 不能出现在 student 的 input 或 output 中
- recall_result 不能包含 teacher_caption 的信息（除非学生 memory 中本来就有）
- 标准答案来自 teacher_caption，但 SFT output 不能照搬（需要基于 student 可见信息回答）

### 1.3 四动作协议

```
1) <observation>...</observation><action>silent</action>
2) <observation>...</observation><action>response</action><response>...</response>
3) <observation>...</observation><action>recall</action><query>{"query":"...","time_range":"..."}</query>
4) <observation>...</observation><action>compress</action><summary>{"time_range":[s,e],"text":"..."}</summary>
```

**observation（原 think）改名原因**：
- "think" 暗示推理/元认知，容易诱导 "I need to recall..." 这种输出
- "observation" 明确是可观察事实的记录

### 1.4 Observation 规范

| 参数 | 值 |
|------|------|
| 长度 | **40-60 tokens** |
| 内容 | 当前 2s 帧中的增量可观察事实 |
| 禁止 | 声音/气味/情绪/意图/推理/元认知 |
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

### 2.1 单层替换

```
推理时每个 timestep 的记忆状态:
  compressed_segments: [summary_0, summary_1, ...]   ← 替换了原始 observations
  recent_observations: [obs_20, obs_21, ..., obs_N]  ← 未被压缩的
  visual_window: [最近 12 chunks 的帧]                ← FIFO

压缩发生时:
  recent_observations[:10] → compressed_summary → 加入 compressed_segments
  recent_observations = recent_observations[10:]      ← 真正移除
```

### 2.2 单条训练样本的 input 结构

```
[System prompt: 协议说明, ~150 tok]
[Memory block:]
  <compressed time="0-20">summary_0</compressed>         ← 压缩段
  <compressed time="20-40">summary_1</compressed>
  [40-42] observation at t=40-42                         ← 未压缩的 observation
  [42-44] observation at t=42-44
  ...
[Visual window: 最近 12 chunks 的视频帧, ~1536 vision tok]
[User input: 当前问题 / compress_trigger / recall_result / 空]
```

这是学生模型**真实能看到的全部信息**。

### 2.3 压缩触发与合法候选

**触发条件**: `len(recent_observations) >= 10`

**合法压缩范围约束**：
1. 必须是过去内容，不能包含视觉窗口内的新 observations
2. 尽量是事件边界完整段（不切断进行中的动作）
3. 不能切断 pending question / unresolved trigger
4. 压缩后 token gain 足够
5. summary 能保留关键实体、OCR、数量、交互

**Phase C1**: 系统指定 canonical 范围（最早 10 条）
**Phase C2**: 模型从合法候选集中选择（训练 gold 来自规则化 policy）

### 2.4 Token 预算

| 组件 | Tokens | 说明 |
|------|--------|------|
| System prompt | ~150 | |
| 压缩段 (≤5 × 150) | ≤750 | |
| 未压缩 observations (≤10 × 50) | ≤500 | |
| Visual window (12 chunks) | ~1536 | min_pixels=100352 |
| User input (问题/trigger/result) | ~100 | |
| Recall result (含 4帧) | ~356 | 256 vision + 100 text |
| **输入总计** | **~3400** | |
| 输出 (observation + action) | ~80 | |
| **单样本总计** | **~3500** | 远在 16K 内 |

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

**397B 输入**: 当前 chunk 的 2 帧 + 前面所有 chunk 的 caption（保持上下文一致性）

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
     "confidence": 0.88},
    {"fact": "amount approximately one teaspoon", 
     "confidence": 0.55, "uncertainty": "visual estimate"}
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
- 验证 student observation 是否遗漏关键信息
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
- 每个 timestep 的 student observation (40-60 tok)
- 压缩决策和 summary
- 主动 recall 事件
- **每个 timestep 的 memory state snapshot**

**核心原则**: Question-blind。不知道未来会有什么问题。不能为未来问题优化 observation 或 summary。

**397B 输入** (每个 timestep):
```
[压缩段] + [recent_observations] + [24帧视觉窗口] + prompt
```

**397B 输出**:
```json
{"observation": "Chef sprinkles seasoning from small white bowl into pot on right burner.",
 "action": "silent"}
```

或（主动 recall, ~5% 频率）:
```json
{"observation": "Person in red hat enters again, similar to earlier.",
 "action": "recall",
 "query": {"query": "red hat person store", "time_range": "0-20"}}
```

或（压缩, 当 recent_observations >= 10）:
```json
{"observation": "...",
 "action": "compress",
 "compress_range": [0, 20],
 "summary": {"time_range": [0, 20], "text": "..."}}
```

**关键**: 压缩的 summary 是基于**当前 observations**（不是 teacher caption），且不知道未来问题。

**每个 timestep 缓存 memory snapshot**:
```python
snapshots[chunk_idx] = {
    "compressed_segments": copy(compressed),
    "recent_observations": copy(recent_obs),
    "visual_window_range": (window_start, chunk_idx),
}
```

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
- Student rollout (所有 observations + summaries + memory snapshots)
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
    
    # 2. 答案在 recent_observations 中？
    for obs in snapshot_at_ask_time["recent_observations"]:
        if keyword_overlap(obs, answer_keywords) > threshold:
            return "response"  # 从文本记忆直接回答
    
    # 3. 答案在 compressed_summary 中？
    for seg in snapshot_at_ask_time["compressed_segments"]:
        if keyword_overlap(seg["text"], answer_keywords) > threshold:
            return "response"  # 从压缩记忆直接回答（不需要 recall!）
    
    # 4. 答案不在任何当前可见信息中 → 需要 recall
    # 但 recall 只能检索学生自己的 memory 或历史帧
    if answer_in_historical_observations_or_frames(task):
        return "recall"
    
    # 5. 答案完全不可得
    return "unanswerable"
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
        "recent_observations": ["[40-42] ...", "[42-44] ..."],
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
        recent_obs=memory_state["recent_observations"],
        visual_window=get_frames(memory_state["visual_window_range"]),
        user_input=task["question"],
    )
    
    # 构造训练样本的 output（受标准答案约束）
    if task["gold_action"] == "response":
        output = generate_response_output(task, memory_state)
    elif task["gold_action"] == "recall":
        output = generate_recall_output(task, memory_state)
    elif task["gold_action"] == "unanswerable":
        output = generate_uncertain_output(task)
    
    return {"input": sample_input, "output": output, "metadata": task}
```

**对于 recall 任务**:

```python
def generate_recall_output(task, memory_state):
    # Step 1: 生成 recall query（约束：不能包含答案）
    query = generate_query(task["question"], task["gold_answer"])
    validate_no_answer_in_query(query, task["gold_answer"])
    
    # Step 2: 模拟检索结果（只用学生可访问的信息）
    recall_result = simulate_retrieval(
        query=query,
        student_observations=all_past_observations,
        student_summaries=memory_state["compressed_segments"],
        historical_frames=available_frames,
        noise_level=sample_noise(),  # 70/20/5/5 分布
    )
    
    # Step 3: 基于检索结果生成 response
    # 注意: response 只能依赖 recall_result 中的信息，不能用 teacher caption
    post_recall_response = generate_response_from_result(
        question=task["question"],
        recall_result=recall_result,
        gold_answer=task["gold_answer"],  # 作为约束，不作为输入
    )
    
    return {
        "recall_turn": f'<observation>...</observation><action>recall</action><query>{query}</query>',
        "result_turn": recall_result,  # 系统注入
        "response_turn": f'<observation>...</observation><action>response</action><response>{post_recall_response}</response>',
    }
```

**Recall Result 的三种来源**（只用学生可访问内容）:

| 来源 | 内容 | 适用 |
|------|------|------|
| student observation | 学生自己写过的 obs 文本 | 文本细节 recall |
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
✓ observation 是否被当前帧支持
✓ summary 是否只含原始 observations 中的信息
✓ response 是否被 support evidence 支持
✓ 无声音/气味/情绪/意图推断
```

**第四类：格式与长度**
```
✓ observation 40-60 tokens
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
| **silent** | 无问题，正常观察 | 输出 observation |
| **response (from frames)** | 答案在当前视觉帧中 | 直接从画面回答 |
| **response (from memory)** | 答案在 obs/summary 中 | 从文本记忆回答 |
| **response (uncertain)** | 无可靠证据 | 回答"不确定" |
| **recall** | 答案不在任何可见信息中 | 需要检索历史 |
| **compress** | 系统触发 / 模型决策 | 压缩旧 observations |

### 4.2 Recall 的必要条件

只有同时满足以下全部条件时，gold_action 才是 recall：
1. 答案不在当前视觉帧中
2. 答案不在 recent_observations 中
3. 答案不在 compressed_summaries 中
4. 答案存在于历史 observations / frames 中（可检索到）

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
  "sample_id": "vid001_t44_recall",
  "video_id": "vid001",
  "phase": "C1",
  "task_type": "recall_from_compressed",
  
  "input": {
    "system": "You are a streaming video agent...",
    "memory": {
      "compressed": [
        {"time_range": [0, 20], "text": "Chef(red apron) prepared workspace..."},
        {"time_range": [20, 40], "text": "Tomatoes diced, added to pot, seasoning added..."}
      ],
      "recent_observations": [
        "[40-42] Sauce simmering, chef covers pot with glass lid.",
        "[42-44] Chef retrieves basil from refrigerator."
      ]
    },
    "visual_window": {"video_start": 20, "video_end": 44, "frames": 24},
    "user_input": "How much salt did the chef add?"
  },
  
  "output": "<observation>Basil being torn over pot, green leaves on red sauce surface.</observation><action>recall</action><query>{\"query\":\"salt seasoning amount pot\",\"time_range\":\"20-40\"}</query>",
  
  "metadata": {
    "gold_answer": "Approximately 1 teaspoon",
    "gold_action": "recall",
    "action_minimality": {
      "answer_in_frames": false,
      "answer_in_recent_obs": false,
      "answer_in_compressed": false,
      "answer_in_historical": true
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

### 5.2 Recall 后的 follow-up 样本

```json
{
  "sample_id": "vid001_t44_post_recall",
  "input": {
    "system": "...",
    "memory": { "...same as above..." },
    "visual_window": {"video_start": 20, "video_end": 44, "frames": 24},
    "recall_result": {
      "source": "historical_frames",
      "time": [28, 30],
      "frames": 4,
      "text_context": "[28-30] Chef sprinkles seasoning from small white bowl into pot."
    },
    "user_input": "Continue following the protocol to respond."
  },
  
  "output": "<observation>Retrieved frames show chef adding white granular seasoning from small bowl, approximately one teaspoon amount visible.</observation><action>response</action><response>The chef added approximately one teaspoon of salt from a small white bowl.</response>"
}
```

### 5.3 Compress 样本

```json
{
  "sample_id": "vid001_t44_compress",
  "input": {
    "system": "...",
    "memory": {
      "compressed": [{"time_range": [0, 20], "text": "..."}],
      "recent_observations": [
        "[20-22] Olive oil poured into pot.",
        "[22-24] Minced garlic added to oil, sizzling.",
        "[24-26] Garlic turning golden.",
        "[26-28] Tomato quarters dumped into pot.",
        "[28-30] Chef sprinkles seasoning from small bowl.",
        "[30-32] Stirring with wooden spoon.",
        "[32-34] Sauce bubbling gently.",
        "[34-36] Chef covers pot with lid.",
        "[36-38] Moves to refrigerator.",
        "[38-40] Retrieves fresh basil bunch."
      ]
    },
    "visual_window": {"video_start": 20, "video_end": 44},
    "user_input": "<compress_trigger/>"
  },
  
  "output": "<observation>System triggered memory compression.</observation><action>compress</action><summary>{\"time_range\":[20,40],\"text\":\"[20-28] Oil heated, garlic sauteed golden, tomato quarters added to pot. [28-34] Seasoning added from small white bowl, stirred with wooden spoon, sauce bubbling. [34-40] Pot covered with lid, chef retrieved fresh basil from fridge. Entities: pot(stainless), garlic, tomatoes, wooden spoon, glass lid, basil.\"}</summary>"
}
```

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
- Student 生成的 observations 质量更低（短、漏信息）
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
# 观察
OBSERVATION_TOKENS = (40, 60)          # observation 长度

# 压缩
COMPRESS_THRESHOLD = 10                 # 10 条 recent_obs 触发
COMPRESS_RANGE = 10                     # 每次压缩 10 条
SUMMARY_TOKENS = (100, 180)             # summary 长度
COMPRESSION_RATIO = 3.5                 # 500 tok → ~140 tok

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
```

---

## 9. 补充设计（解决已知缺口）

### 9.1 Pending Question 状态表示

S3/S4 任务中，用户提问后模型需要在多个后续 timestep "记住在等什么"。

**解决**：在 memory block 中增加 `<pending>` 字段：

```
[Memory block:]
  <compressed time="0-20">...</compressed>
  [20-22] observation...
  [22-24] observation...
  <pending since="24">Tell me when the chef adds the basil.</pending>   ← 待解决问题
[Visual window: ...]
[User input: (无新问题，只有新视频)]
```

模型输出：
- 如果事件未发生：`<observation>Sauce still simmering, no basil yet.</observation><action>silent</action>`
- 如果事件发生：`<observation>Chef tearing basil leaves over pot.</observation><action>response</action><response>The chef is now adding basil to the sauce.</response>`

训练时 pending 字段从用户提问那个 timestep 开始出现，直到被回答后消失。

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
        corpus=memory_state["recent_observations"] + 
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
- 同视频时间相近但内容不同的 observation/frame
- 同视频中相似实体但不同时间的 observation
- 文本相似但语义不同的片段

### 9.4 补充负样本任务

| 类型 | 描述 | 正确行为 | 占比 |
|------|------|---------|------|
| **不存在内容** | 用户问视频中从未出现的事物 | response(uncertain): "I don't see evidence of..." | 3% |
| **多候选歧义** | "哪个人？"但有多个人 | response(clarify): "There are multiple people..." | 2% |
| **记忆与画面冲突** | 画面变了但记忆还是旧的 | response 基于当前画面（最新优先） | 2% |
| **Recall 返回错误** | 检索结果不含答案 | response(uncertain): "The retrieved info doesn't show..." | 3% |
| **用户纠正前一个回答** | "不对，我问的是另一个" | response: 基于新理解重新回答 | 2% |
| **计数任务** | "出现了几次" | response: 基于 observations 中的出现次数 | 2% |

### 9.5 Compression 自适应长度

**问题**：固定 10 条 × 50 tok = 500 tok 输入，输出 100-180 tok，比率变化大。

**解决**：根据输入内容复杂度自适应：

```python
def determine_summary_length(observations_to_compress):
    """根据内容复杂度决定 summary 长度"""
    n_entities = count_unique_entities(observations_to_compress)
    has_ocr = any("ocr" in obs for obs in observations_to_compress)
    has_interaction = any("Q:" in obs or "response" in obs for obs in observations_to_compress)
    n_state_changes = count_state_changes(observations_to_compress)
    
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
[observations 0-20]    → compress → summary 按通用重要性保留
不知道未来问什么，所有类似动作用相同粒度压缩
```

**场景 B: Pending-question-aware compression**（有未回答的问题）
```
<pending>"Tell me when basil is added"</pending>
[observations 20-40]   → compress → summary 必须保留 basil 相关信息

因为有 pending question，模型应该知道保留与 basil 相关的 observation。
如果某条 obs 含 "basil" 或相关实体 → 保留为未压缩，不压缩它。
```

**训练数据中两种比例**：
- Question-blind: 70%（更常见）
- Pending-aware: 30%

### 9.7 Grounding 验证具体方法

```python
def verify_grounding(observation, frames, teacher_caption):
    """检查 observation 是否被帧支持"""
    
    # 方法 1: 基于 teacher_caption 的覆盖检查
    obs_entities = extract_entities(observation)
    caption_entities = teacher_caption["visible_entities"]
    coverage = len(obs_entities & caption_entities) / max(len(obs_entities), 1)
    
    # 方法 2: 用另一个 VLM (如 7B) 做 entailment 判断
    # "Given these frames, is this statement supported?"
    entailment_score = small_vlm_verify(frames, observation)
    
    # 方法 3: 黑名单关键词
    blacklist = ["sound", "smell", "aroma", "feels", "seems to", "probably", 
                 "likely", "intend", "want to", "emotion", "happy", "sad"]
    has_blacklist = any(w in observation.lower() for w in blacklist)
    
    return {
        "entity_coverage": coverage,       # >= 0.7 通过
        "entailment": entailment_score,    # >= 0.6 通过
        "no_blacklist": not has_blacklist,  # True 通过
    }
```

**执行时机**: 阶段 2 (Rollout) 产出的每条 observation 都验证。不通过的丢弃并重新生成。

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

## 10. 已解决问题对照表

| 问题来源 | 编号 | 状态 | 对应章节 |
|---------|------|------|---------|
| 问题.txt | #1 loss 问题 | ✅ | §1.1 方案 A |
| 问题.txt | #2 压缩替换 | ✅ | §2.1 |
| 问题.txt | #3 格式对齐 | ✅ | §9.1 pending + §9.2 recalled frames |
| 问题.txt | #4 Token 验证 | ✅ | §9.8 |
| 问题.txt | #5 分布 | ✅ | §4.4 episode-level |
| 问题.txt | #6-8 observation | ✅ | §1.3-1.4 |
| 问题.txt | #9 response 长度 | ✅ | §1.5 |
| 问题.txt | #10 query 泄漏 | ✅ | §6.1 |
| 问题.txt | #11 检索方案 | ✅ | §9.3 |
| 问题.txt | #12 负样本 | ✅ | §9.4 |
| 问题.txt | #13 压缩比 | ✅ | §9.5 自适应 |
| 问题.txt | #14 grounding | ✅ | §9.7 |
| 问题.txt | #16 任务缺失 | ✅ | §9.4 |
| 问题.txt | #17 格式验证 | ✅ | §9.8 tokenizer |
| 问题.txt | #20 评估 | ✅ | §7.2 |
| 修改.txt | #1 压缩合法候选 | ✅ | §2.3 |
| 修改.txt | #12 pending compression | ✅ | §9.6 |
| 修改.txt | #13 DAgger | ✅ | §7.1 |
| 修改.txt | #14 provenance | ✅ | §6.4 |

---

## 11. 开放问题（已缩减）

1. **Per-timestep vs 短序列**: 当前方案 A 每步独立。如果实验中发现模型缺少连续性，可改为 3-step 滑动窗口（loss 只在最后一步）。先用纯 per-timestep 做 baseline。
2. **Observation 关键信息保留策略**: 如果 observation 漏了关键细节（如盐的量），后续 recall 就找不到。建议：阶段 2 产出后，用 teacher caption 检查 observations 的"关键信息覆盖率"。覆盖率 < 0.6 的重新生成。
3. **Canonical compression range 选择 policy**: 当前规则：最早 10 条。进阶：按事件边界切分（检测 state_change 断点），选择最完整的事件段。
4. **DAgger 执行计划**: Phase 1-2 完成后训练 v0 模型 → v0 跑 100 视频 → 基于 v0 memory 造 2000 条补充数据 → 混入 Phase C1 训练。
