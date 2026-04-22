# 流视频 Agent SFT 工程设计

> 版本: v3.0 | 日期: 2026-04-22
>
> 配套文档：`data_construction_zh.md` v7.0, `streaming_position_encoding.md` v1.0
>
> 本文档定义 **推理状态机 → SFT 训练格式 → 数据构造** 的完整链路。
> 一切以 §0 推理状态机为源头，SFT 格式必须精确匹配推理行为。
>
> 核心设计：
> - Per-timestep re-render：每步重新构造完整 input，不依赖跨步 KV cache 累积
> - 输入布局：**视频在前、文本在后**（`<visual_window>` → `<memory>` → `<user_input>`）
>   - 视频区固定大小（~1536 tok），position 稳定，便于增量 KV cache 复用
> - 时间对齐 MROPE：文本 think 的 temporal position 对齐到对应视频帧时间戳
> - 4-action protocol：silent / response / recall / compress，`protocol_version: "4action"`
> - 所有 action 输出统一为 `<think>...</think><action>...</action>` 格式（含 recall_response）
> - 共享协议模块 `thinkstream/data/agent_protocol.py`：训练/推理共用输入构造代码
> - 推理端 Agent Loop `thinkstream/model/agent_loop.py`：MemoryState + 压缩触发 + recall 编排

---

## 0. 推理状态机规范（一切格式的源头）

> **核心原则：SFT 训练样本 = 推理时一步的精确快照。推理怎么构造 input，训练就怎么构造。**

### 0.1 系统侧维护的状态（外部 Python 对象，非 KV cache）

```python
memory_state = {
    "compressed_segments": [             # ≤5 段，每段 ≤150 tok
        {"time_range": [0, 20], "text": "Chef prepared workspace..."},
    ],
    "recent_thinks": [                   # ≤12 条，总 ≤600 tok
        {"time": "40-42", "text": "Sauce simmering, chef covers pot."},
        {"time": "42-44", "text": "Chef retrieves basil from fridge."},
    ],
    "pending_questions": [],             # 待解决的 event-watch 请求
    "_retrieval_archive": [...],         # 系统侧全部历史 thinks（模型不可见）
}
```

### 0.2 每步推理流程

```python
def inference_step(chunk_idx, video_path, memory_state, user_question=None):
    """一步推理 = 一条 SFT 训练样本。"""
    
    # ── 1. 构造文本（memory block + 触发 + 问题）──
    text_parts = []
    for seg in memory_state["compressed_segments"]:
        text_parts.append(f'<compressed>{json.dumps(seg)}</compressed>')
    for think in memory_state["recent_thinks"]:
        text_parts.append(f'[{think["time"]}] {think["text"]}')
    for pq in memory_state["pending_questions"]:
        text_parts.append(f'<pending since="{pq["since"]}">{pq["question"]}</pending>')
    
    # 压缩触发（C1: 带 range; C2: 不带）
    if should_compress(memory_state):
        text_parts.append('<compress_trigger range="40-54"/>')
    
    if user_question:
        text_parts.append(user_question)
    
    # 如果上一步是 recall，注入检索结果
    if recall_result_pending:
        text_parts.append(f'<recall_result>{recall_result}</recall_result>')
    
    memory_text = "\n".join(text_parts)
    
    # ── 2. 构造 messages（单轮，视频在前、文本在后）──
    # 布局: [system] [visual_window + 帧] [memory_text] [user_input]
    # 理由: 视频区固定大小(~1536 tok)，放前面 position 稳定，便于 KV cache 复用
    # 详见 streaming_position_encoding.md §1
    chunk_start = chunk_idx * 2.0
    window_start = max(0.0, chunk_start - 22.0)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_4ACTION},
        {"role": "user", "content": [
            {"type": "video", "video": video_path,
             "video_start": window_start, "video_end": chunk_start + 2.0,
             "nframes": 24},                          # 视频在前（Zone B）
            {"type": "text", "text": memory_text},    # 文本在后（Zone C）
        ]},
    ]
    
    # ── 3. 模型 forward（每步独立，不复用 KV cache）──
    output = model.generate(messages, max_new_tokens=128)
    
    # ── 4. 解析 output + 更新 memory_state ──
    think = parse_think(output)
    action = parse_action(output)
    
    if action == "compress":
        summary = parse_summary(output)
        memory_state = replace_range(memory_state, summary)  # 先替换
        memory_state = append_think(memory_state, think)     # 再追加
    elif action == "recall":
        memory_state = append_think(memory_state, think)
        # 系统异步检索，结果注入下一步的 user content
    else:  # silent / response
        memory_state = append_think(memory_state, think)
    
    return output, memory_state
```

### 0.3 为什么不用 KV cache 跨步累积

| 方案 | 优势 | 问题 |
|------|------|------|
| **KV 累积（多轮对话）** | 增量编码快 | KV 无限增长；压缩需要 KV surgery（位置编码错乱、后续 KV 依赖旧前缀、视觉/文本混合幽灵状态）|
| **Per-step re-render** | 内存恒定；压缩天然生效；训推完全一致 | 每步重新编码 |

Per-step re-render 的时延在实时预算内：

| 硬件 | 无帧缓存 | ViT帧缓存 | +Prefix cache | 2s 预算 |
|------|---------|----------|--------------|---------|
| H100 | 640ms | 451ms | ~300ms | ✅ 15% |
| H20 | 2044ms | 781ms | ~580ms | ✅ 29% |

### 0.4 SFT 样本 = 推理快照

```
推理构造的 messages  ==  SFT 样本的 messages
推理模型的 output    ==  SFT 样本的 assistant content (gold)
不多不少，完全一致。
```

---

## 1. 为什么需要 Per-Timestep Re-render

> 完整推理流程见 §0。本节解释为什么不用多轮对话。

### 1.1 多轮对话的根本问题

多轮方案（`_build_agent_messages`）将完整视频作为一条长对话。在 KV cache 中：

| 问题 | 描述 |
|------|------|
| **KV surgery 不可行** | 压缩后旧 thinks 的 KV 仍在 cache。替换需要重算后续所有 KV（位置编码错乱、hidden state 依赖旧前缀）|
| **KV cache 无限增长** | 1小时视频 → ~1M tokens KV → ~57GB 显存。即使 attention mask=0，KV 物理不释放 |
| **Mask ≠ 删除** | attention mask=0 只是不 attend，KV 物理存在。模型无动力写好 summary（直接看旧 thinks 更详细）|
| **训推不一致** | 训练时有 KV 残留，推理时（如果做 re-render）没有。分布偏移 |

### 1.2 Per-Timestep Re-render

每步重新构造完整 input（见 §0.2），不依赖跨步 KV cache：

- **压缩天然生效**：旧 thinks 不在 input 中，summary 是唯一信息源
- **内存恒定**：无论视频多长，每步 input ≤ 8K tokens
- **训推完全一致**：§0.2 的 `inference_step` = 训练样本构造
- **时延可控**：H20 ~580ms/step，H100 ~200ms/step（见 §0.3）

---

## 2. 训练样本结构

### 2.1 Input 编码（定稿排列 v3.0）

每条样本的 user content 按以下顺序拼装。**视频在前、文本在后**（v3.0 变更，详见 `streaming_position_encoding.md`）。每个 block 有明确的开闭 tag，属性以 JSON 放内部（方案 B）。

```
┌─────────────────────────────────────────────────────────────────┐
│ Zone A: [System Prompt]                              ~150 tok  │
│ 协议说明 + 四动作格式                                            │
├─────────────────────────────────────────────────────────────────┤
│ Zone B: 视觉区（固定大小，position 稳定）                        │
│ <visual_window>{"start":20,"end":44,"frames":24,    ~40 tok   │
│   "current_time":[42,44]}</visual_window>                      │
│ [VIDEO: 12 chunks × 2 frames = 24 frames]           ~1536 vis │
│                                                                │
│ (仅 recall_response 时追加)                                     │
│ <recalled_frames>{"time_range":[2,6],                ~260 vis  │
│   "source":"historical_frames","n_frames":4}                   │
│ </recalled_frames>                                             │
│ [VIDEO: 4 recalled frames]                                     │
├─────────────────────────────────────────────────────────────────┤
│ Zone C: 文本记忆区（可变大小，追加式增长）                        │
│ <memory>                                             ~600-1350 │
│   <compressed>{"time_range":[0,20],"text":"..."}</compressed>  │
│   <compressed>{"time_range":[20,40],"text":"..."}</compressed> │
│   [40-42] Chef places tomatoes on board...                     │
│   [42-44] Salt sprinkled from small white bowl...              │
│   <pending>{"since":24,"type":"awaiting_recall_response",      │
│             "question":"Tell me when basil added"}</pending>   │
│ </memory>                                                      │
│                                                                │
│ (仅 recall_response 时)                              ~100 tok  │
│ <recall_result>{"source":"student_think","time":"2-6",         │
│   "text":"Retrieved: [2-4] Chef in red apron..."}</recall_result>│
├─────────────────────────────────────────────────────────────────┤
│ Zone D: 用户输入区（每步变化）                                   │
│ <user_input>What color is the apron?</user_input>    ~50 tok   │
│ 或 <user_input><compress_trigger>{"range":[20,34]}             │
│    </compress_trigger></user_input>                            │
│ 或 <user_input>Continue following the protocol.</user_input>   │
│ 或 (silent 无问题时省略此 block)                                │
└─────────────────────────────────────────────────────────────────┘
```

**关键约定**（不可违反）：

| # | 约定 | 来源 |
|---|------|------|
| 1 | 顺序：`<visual_window>` + 帧 → `<recalled_frames>` + 帧 → `<memory>` → `<recall_result>` → `<user_input>` | v3.0 定稿 |
| 2 | 视频在前、文本在后：视频区(Zone B)固定大小，position 稳定，便于 KV cache 增量复用 | `streaming_position_encoding.md` §1 |
| 3 | `recalled_frames` 在 `visual_window` 之后、`memory` 之前（视觉区内部追加） | P0-2 |
| 4 | `recall_result` 必须在 `user_input` 之前 | P0-1 |
| 5 | `<visual_window>` JSON 含 `current_time` 字段，确保 think 只写当前 2s | P0-3 |
| 6 | 文本 think 的 MROPE temporal position 对齐到对应帧的时间戳（见 `streaming_position_encoding.md` §4） | v3.0 |
| 7 | 视觉帧优先用 `frame_paths`/`frame_indices` 加载（见 §4.5） | P1-3 |

### 2.2 Output 编码

根据 gold_action 有四种 output 格式。所有结构化数据使用 **方案 B（P0-7）**：JSON 放在 special token 内部，special token 精确匹配。

```python
# silent
"<think>Chef adjusting burner dial...</think><action>silent</action>"

# response
"<think>Answer visible in current frames...</think><action>response</action><response>The apron is red.</response>"

# recall — query 内容为 JSON
"<think>Evidence not in window...</think><action>recall</action><query>{\"query\":\"apron color chef\",\"time_range\":\"0-20\"}</query>"

# compress — summary 内容为 JSON (C1: range 由 trigger 指定; C2: 模型自选)
"<think>Current visual observation...</think><action>compress</action><summary>{\"time_range\":[20,34],\"text\":\"Oil heated, garlic browned...\"}</summary>"
```

**方案 B 的好处**：`<compressed>`, `</compressed>`, `<pending>`, `</pending>`, `<summary>`, `</summary>` 等 tag 作为 special token 可被 tokenizer 精确匹配为单个 token。属性（如 `time_range`, `since`）以 JSON 形式放在 tag 内部，避免 `<compressed time="0-20">` 这种含属性的 opening tag 被 tokenizer 拆成多个 sub-token。

**特殊情况 — post-recall response**（同一 chunk 内 recall 后的回答）：
```python
# 无 <think>（避免同一 chunk 两次写入 memory）
"<action>response</action><response>Based on retrieved info...</response>"
```

### 2.3 Tokenization 映射

Pipeline JSON 样本 → 模型 input_ids 的转换流程：

```
pipeline_sample.json
    │
    ├── input.system          → [System message]
    ├── input.memory          → [Memory block text]  (纯文本 token)
    ├── input.visual_window   → load_video_frames()  (vision token)
    ├── input.user_input      → [User message text]
    ├── input.recalled_frames → load_video_frames()  (vision token, 可选)
    │
    ▼  apply_chat_template() + processor()
    
    input_ids:  [system_tok... memory_tok... <|vision_start|> video_pad... <|vision_end|> user_tok... ]
    labels:     [IGNORE...     IGNORE...     IGNORE...                                    IGNORE...   ]
    
    output_ids: [think_tok... action_tok... payload_tok... <|im_end|>]
    labels:     [think_tok... action_tok... payload_tok... <|im_end|>]
```

**Loss 只覆盖 output 部分**（上图第二行 labels）。整个 input 部分的 labels 全部为 `IGNORE_INDEX = -100`。

---

## 3. 注意力掩码设计

### 3.1 现有 Flex-Attention 回顾

ThinkStream 已有基于 PyTorch `flex_attention` 的视频滑动窗口 CUDA kernel（`streaming_attention.py`）：

```python
def sliding_window_mod(b, h, q_idx, kv_idx):
    is_valid   = attention_mask[b, kv_idx] > 0
    is_causal  = q_idx >= kv_idx
    k_is_video = video_mask[b, kv_idx]
    diff       = block_ids[b, q_idx] - block_ids[b, kv_idx]
    is_in_window = (~k_is_video) | (diff < window_size_n)
    return is_valid & is_causal & is_in_window
```

**效果**：video token 只能看到当前 block 和前 N-1 个 block；text token 无限制。

### 3.2 Per-Timestep 样本的简化

**关键洞察**：在 per-timestep 独立样本中，**不需要**跨 chunk 的压缩遮蔽。

原因：
- 旧方案（多轮对话）中，压缩后仍有旧 think token 在序列中，需要用 4D mask 遮蔽
- 新方案中，input 的 memory block 就是**压缩后的状态**——旧 thinks 已被 summary 替换，根本不在序列中
- 视觉滑动窗口由 input 构造时就只包含最近 12 chunks，旧帧不在序列中

因此，per-timestep 方案**不能复用**旧的 `sliding_window_mod`，需要实现 `per_timestep_mask_mod`：

```python
def per_timestep_mask_mod(b, h, q_idx, kv_idx):
    """Per-timestep attention: causal + padding, 无视频滑动窗口。
    
    所有 video token（visual window + recalled frames）对所有后续 token 可见。
    不使用 block_ids 滑动窗口约束——per-timestep 样本中只有一个"时间步"
    的视频帧，不存在需要滑出的旧 block。
    """
    is_valid   = attention_mask[b, kv_idx] > 0
    is_causal  = q_idx >= kv_idx
    return is_valid & is_causal
```

**P0-6：为什么不能复用旧 `sliding_window_mod`**

旧 mask 按 `block_ids` 计算 `diff = q_block - k_block`，对 `diff >= window_size_n` 的 video token 遮蔽。
Recalled frames 来自很早的时间点，它们的 block_id 远小于 output token 的 block_id，
`diff` 会超过 `window_size_n`，导致 **output token 无法 attend to recalled frames**。

Per-timestep 样本中，所有视频帧（包括 recalled）都应该对所有后续 token 可见。
唯一需要的限制是 causal + padding，不需要视频滑动窗口。

### 3.3 Token 序列、Attention 与 MROPE（v3.0 新布局）

```
Token 序列 (v3.0 视频在前):
[system] [visual_window: 24 frames] [recalled: 4 frames] [memory_text] [user_input] [output]
         ├────── video_mask = True ──────────────────────┤

video_mask:     [0 0 ... 0  1 1 1 ... 1  1 1 1 1  0 0 ... 0  0 0 ... 0  0 ...]
```

**视频区(Zone B)在前、文本区(Zone C)在后。** 视频区固定大小，position 稳定。

visual_window 和 recalled frames 的 `video_mask` 都为 True。
区分靠 **MROPE 时间编码**，不靠 attention mask：
- visual_window frames: `second_per_grid_ts` 基于 video_chunk_size (2s)
- recalled frames: `second_per_grid_ts` 基于 recalled time range (如 4s)

**MROPE 时间对齐** (详见 `streaming_position_encoding.md` §4, §8)：
- 文本 think 的 temporal position = 对应帧的时间戳（如 think about t=10s → temporal_pos = encode(10)）
- 帧已 evict 但 think 仍在时：think 保留原始时间戳，正确表达"这是旧信息"
- recall 帧回来时：recalled frames 和 old think 的 temporal position 重新对齐
- h/w 维度区分模态：视频用空间网格(0,1,2..)，文本用递增序号(远离视频区间)

### 3.4 验收测试（必须在训练前通过）

```python
def test_recalled_frames_attention():
    """P0-6 验收：output token 必须能 attend to recalled frame token。
    
    v3.0 布局: [system ~50] [visual_window ~120] [recalled ~10] [memory_text ~15] [output ~5]
    """
    video_mask = torch.zeros(1, 200)
    video_mask[0, 50:170] = 1  # visual window (Zone B start)
    video_mask[0, 170:180] = 1  # recalled frames (Zone B end)
    attention_mask = torch.ones(1, 200)
    
    mask_mod = generate_per_timestep_mask_mod(video_mask, attention_mask)
    # output token (pos 195) attend to recalled frame (pos 175) ✅
    assert mask_mod(0, 0, 195, 175) == True
    # output token (pos 195) attend to visual window (pos 100) ✅
    assert mask_mod(0, 0, 195, 100) == True
    # memory text (pos 185) attend to visual window (pos 100) ✅
    assert mask_mod(0, 0, 185, 100) == True
    # causal: 不能看到未来
    assert mask_mod(0, 0, 100, 195) == False
```

### 3.4 未来扩展：短序列滑动窗口方案

如果实验发现 per-timestep 独立样本缺少连续性（开放问题 #1），可切换为 3-step 滑动窗口：

```
训练样本 = {
    input:  [System] + [Visual_window_at_t-2] + [Memory_at_t-2]
    turn 1: [User: chunk_t-2] → [Asst: think_t-2]  (loss OFF)
    turn 2: [User: chunk_t-1] → [Asst: think_t-1]  (loss OFF)
    turn 3: [User: chunk_t]   → [Asst: think_t]    (loss ON, 只在最后一步)
}
```

此时需要 flex-attention 的 compress_mask 来处理连续 3 步中可能出现的压缩事件。但 **先以纯 per-timestep 做 baseline**。

---

## 4. 数据加载与转换

### 4.1 新 Dataset 类：`PerTimestepDataset`

替代现有的 `LazySupervisedDataset`，专门处理 per-timestep 格式：

```python
class PerTimestepDataset(Dataset):
    """Dataset for per-timestep independent SFT samples.
    
    Each sample is a single 2s chunk with:
    - Memory state (compressed segments + recent thinks + pending)
    - Visual window (12 chunks, 24 frames)
    - Optional recalled frames
    - User input (question / compress trigger / recall result / empty)
    - Output (think + action + payload)
    """
    
    def __init__(self, processor, data_args):
        # 加载 pipeline 产出的 train.jsonl
        # 按 phase 过滤（如只加载 Phase 1 数据）
        # update_processor_pixels() 设置视觉分辨率
        pass
    
    def __getitem__(self, i):
        sample = self.samples[i]
        
        # 1. 构建 chat messages
        messages = self._build_messages(sample)
        
        # 2. 加载视觉帧
        visual_frames = self._load_visual_window(sample)
        recalled_frames = self._load_recalled_frames(sample)  # 可选
        
        # 3. Tokenize
        model_inputs = process_per_timestep_inputs(
            messages=messages,
            visual_frames=visual_frames,
            recalled_frames=recalled_frames,
            processor=self.processor,
            model_type=self.model_type,
        )
        
        # 4. Labels: 只在 output 部分计算 loss
        model_inputs["labels"] = self._build_labels(model_inputs)
        
        # 5. Position IDs (MROPE)
        model_inputs["position_ids"] = compute_position_ids(
            model_inputs, self.processor, self.model_type,
        )
        
        return model_inputs
```

### 4.2 Message 构建

将 pipeline JSON 转为 chat template 可消费的 messages。严格遵循 §2.1 定稿排列（v3.0: 视频在前）。

实现代码位于 `thinkstream/data/agent_protocol.py`（共享协议模块），训练和推理共用。

```python
def _build_messages(self, sample):
    """Convert pipeline sample to chat messages.
    
    顺序（不可违反，v3.0）：
    <visual_window> + 帧 → <recalled_frames> + 帧
    → <memory> → <recall_result> → <user_input>
    """
    inp = sample["input"]
    
    # System prompt
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    user_content = []
    
    # ── 1. <visual_window> + 视频帧 (Zone B, 固定大小) ──
    # ── 2. <recalled_frames> (如有) ──
    # ── 3. <memory> block (Zone C, 可变大小) ──
    memory_text = self._format_memory_block(inp["memory"])
    user_content.append({"type": "text", "text": f"<memory>\n{memory_text}\n</memory>"})
    
    # ── 2. <visual_window> header + frames ──
    vw = inp["visual_window"]
    chunk_idx = sample["chunk_idx"]
    current_start = chunk_idx * AGENT_CHUNK_SEC
    current_end = current_start + AGENT_CHUNK_SEC
    vw_json = json.dumps({
        "start": vw["video_start"], "end": vw["video_end"],
        "frames": vw["frames"], "current_time": [current_start, current_end],
    })
    user_content.append({"type": "text", "text": f"\n<visual_window>{vw_json}</visual_window>"})
    
    # 帧加载：优先 frame_paths，fallback 到 video_start/end（见 §4.5）
    user_content.append(self._make_video_entry(sample, vw))
    
    # ── 3. <recalled_frames> + frames（仅 recall_response）──
    if "recalled_frames" in inp:
        rf = inp["recalled_frames"]
        rf_json = json.dumps({
            "time_range": rf["time_range"],
            "source": rf.get("source", "historical_frames"),
            "n_frames": rf["n_frames"],
        })
        user_content.append({"type": "text", "text": f"\n<recalled_frames>{rf_json}</recalled_frames>"})
        user_content.append(self._make_recalled_video_entry(sample, rf))
    
    # ── 4. <recall_result>（仅 recall_response, P0-1: 必须在 user_input 之前）──
    if inp.get("recall_result"):
        rr = inp["recall_result"]
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", ""),
        }, ensure_ascii=False)
        user_content.append({"type": "text", "text": f"\n<recall_result>{rr_json}</recall_result>"})
    
    # ── 5. <user_input>（问题 / compress_trigger / "Continue..." / 省略）──
    if inp.get("user_input"):
        user_content.append({"type": "text", "text": f"\n<user_input>{inp['user_input']}</user_input>"})
    
    messages.append({"role": "user", "content": user_content})
    
    # Assistant output（训练目标）
    messages.append({"role": "assistant", "content": sample["output"]})
    
    return messages
```

### 4.3 Memory Block 格式化

**方案 B（P0-7）**：属性放 JSON 内部，special token 精确匹配。
不使用 `<compressed time="0-20">` 这种 XML 属性写法（tokenizer 无法精确匹配含属性的 opening tag）。

```python
def _format_memory_block(self, memory):
    """Format memory state as text for model input.
    
    使用方案 B: JSON inside tags。所有 opening/closing tag 是精确的 special token，
    属性数据以 JSON 形式放在 tag 内部。
    """
    parts = []
    
    # Compressed segments
    for seg in memory.get("compressed", []):
        seg_json = json.dumps({"time_range": seg["time_range"], "text": seg["text"]},
                               ensure_ascii=False)
        parts.append(f'<compressed>{seg_json}</compressed>')
    
    # Recent thinks (纯文本，格式: "[40-42] Chef places tomatoes...")
    for think in memory.get("recent_thinks", []):
        parts.append(think)
    
    # Pending questions
    for pq in memory.get("pending", []):
        pq_json = json.dumps({"since": pq["since"], "question": pq["question"]},
                              ensure_ascii=False)
        parts.append(f'<pending>{pq_json}</pending>')
    
    return "\n".join(parts)
```

### 4.4 Labels 构建

```python
def _build_labels(self, model_inputs):
    """Only compute loss on the output (assistant turn)."""
    input_ids = model_inputs["input_ids"]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    
    # find_assistant_spans 定位 assistant turn 的 token 范围
    for start, end in find_assistant_spans(input_ids[0].tolist(), self.tokenizer):
        labels[0, start:end] = input_ids[0, start:end]
    
    return labels
```

**与旧方案的区别**：旧方案多轮对话中有多个 assistant span，都计算 loss。新方案只有**一个** assistant span（当前 output），且这是正确的——历史 assistant turn 不在序列中。

### 4.5 帧加载：优先 frame_paths（P1-3）

SFT loader 从视频加载帧时，优先使用 Pass4 产出的 `frame_paths` / `frame_indices`，
而非根据 `video_start/video_end/nframes` 重新采样视频。

**问题**：重新采样可能因 fps rounding、ffmpeg 版本、视频编码差异导致帧不一致。
如果 SFT 看到的帧和造 think / answer 时看到的帧不同，模型学到的 think 就和真实视觉输入对不上。

```python
def _make_video_entry(self, sample, vw):
    """构造 visual window 的 video entry。优先用 frame_paths。"""
    # 优先 1: frame_paths（Pass1 抽帧结果，jpg 文件列表）
    if "frame_paths" in vw:
        return {
            "type": "video",
            "video": vw["frame_paths"],  # List[str], 直接传图片路径列表
        }
    # 优先 2: frame_indices（在原始视频中的帧号）
    if "frame_indices" in vw:
        return {
            "type": "video",
            "video": sample["video_path"],
            "video_start": vw["video_start"],
            "video_end": vw["video_end"],
            "nframes": len(vw["frame_indices"]),
        }
    # Fallback: 按时间范围重新采样（仅在 frame_paths 不可用时）
    return {
        "type": "video",
        "video": sample["video_path"],
        "video_start": vw["video_start"],
        "video_end": vw["video_end"],
        "nframes": vw["frames"],
    }

def _make_recalled_video_entry(self, sample, rf):
    """构造 recalled frames 的 video entry。同样优先 frame_paths。"""
    if "frame_paths" in rf:
        return {"type": "video", "video": rf["frame_paths"]}
    return {
        "type": "video",
        "video": sample["video_path"],
        "video_start": rf["time_range"][0],
        "video_end": rf["time_range"][1],
        "nframes": rf["n_frames"],
    }
```

**Pipeline 侧**：Pass4 的 `build_*_sample()` 函数必须把 `frame_paths` 写入样本：

```python
# pass4_forks.py 中
sample["input"]["visual_window"]["frame_paths"] = [
    frame_paths[i] for i in snapshot["visual_window_frame_indices"]
]
```

---

## 5. Data Collator

### 5.1 核心变更

Per-timestep 样本长度更短更均匀（~3500 tok vs 旧方案 ~20000 tok），collation 更高效。

**P0-4：不在 collator 中静默截断**。超长样本在 Dataset 初始化时过滤，不在 collator 里右截断（右截断会砍掉 output labels，模型变成只看 input 不学 output）。

```python
@dataclass
class PerTimestepDataCollator:
    """Collate per-timestep samples into training batch.
    
    不做 truncation。超长样本在 PerTimestepDataset.__init__ 中预过滤。
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances):
        # 1. Pad input_ids, labels, video_masks
        input_ids = pad_sequence([inst["input_ids"].squeeze(0) for inst in instances],
                                  batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([inst["labels"].squeeze(0) for inst in instances],
                               batch_first=True, padding_value=IGNORE_INDEX)
        
        # ⚠️ 不做 truncation。Dataset 已保证所有样本 < max_length。
        # 如果出现超长，说明 token budget 验证没通过，应排查数据，不应静默截断。
        
        # 2. Concatenate vision tensors
        videos = [inst["pixel_values_videos"] for inst in instances if "pixel_values_videos" in inst]
        video_grid_thw = [inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst]
        
        # 3. Position IDs (MROPE: 3D)
        position_ids = pad_and_cat([inst["position_ids"] for inst in instances])
        
        # 4. Attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # 5. Video mask (for flex-attention)
        video_masks = pad_sequence([inst["video_mask"].squeeze(0) for inst in instances],
                                    batch_first=True, padding_value=0)
        
        # 6. Per-sample loss weight (P0-5)
        sample_weights = torch.tensor(
            [inst["sample_weight"] for inst in instances], dtype=torch.float32
        )
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "video_mask": video_masks,
            "pixel_values_videos": torch.cat(videos, dim=0) if videos else None,
            "video_grid_thw": torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
            "position_ids": position_ids,
            "sample_weights": sample_weights,
        }
        
        return batch
```

**超长样本预过滤**（在 Dataset 初始化时执行）：

```python
# PerTimestepDataset.__init__ 中
self.samples = [s for s in all_samples if s.get("num_tokens", 0) < max_length]
if len(all_samples) - len(self.samples) > 0:
    logger.warning(f"Filtered {len(all_samples) - len(self.samples)} samples exceeding max_length={max_length}")
```

### 5.2 Per-Sample Loss Weight（P0-5 修正）

**不使用 vocab-level CE weight**（`torch.ones(vocab_size)` 那种），因为 per-timestep 中每个样本只有一个 action type，需要的是 **per-sample scalar weight**。

```python
# 在 Dataset.__getitem__ 中设置
ACTION_WEIGHTS = {
    "silent": 0.5,
    "response": 1.0,
    "recall_query": 1.5,
    "recall_response": 1.0,
    "compress": 1.5,
    "merge_compress": 1.5,
}
sample_weight = ACTION_WEIGHTS.get(sample["sample_type"], 1.0)
model_inputs["sample_weight"] = sample_weight
```

**在训练 forward 中使用**：

```python
# 训练步中的 loss 计算
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), 
                        ignore_index=IGNORE_INDEX, reduction="none")
# reshape to (B, L)，然后按 sample 加权
loss = loss.view(B, L)
valid_mask = (labels != IGNORE_INDEX).float()
per_sample_loss = (loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
weighted_loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
```

等 action accuracy 稳定后，再考虑更细粒度的 per-token weight（如 summary span 加权）。

---

## 6. 多模型兼容

### 6.1 当前支持矩阵

| 组件 | Qwen2.5-VL | Qwen3-VL | Qwen3.5（预留） |
|------|-----------|---------|--------------|
| `MODEL_CLS` | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` | 待添加 |
| `ROPE_INDEX_FN` | `get_rope_index_25` | `get_rope_index_3` | 待添加 |
| `attn_implementation` | `streaming_attention` | `streaming_attention` | 待适配 |
| `vision_attn_implementation` | `flash_attention_2` | `flash_attention_2` | 待确认 |
| `<think>` token | 需 `add_tokens` | 原生支持 | 原生支持 |
| `process_vision_info` | 标准调用 | + `return_video_metadata` | 待确认 |
| `video_metadata` (per-frame) | 无 | `frames_indices` | 待确认 |

### 6.2 分派机制

所有模型差异通过 `model_type` 字符串分派，**不需要修改框架代码**：

```python
# thinkstream/model/__init__.py — 添加新模型只需增加 entry
MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl":   Qwen3VLForConditionalGeneration,
    # "qwen3.5":  Qwen3_5ForConditionalGeneration,
}

# thinkstream/data/rope2d.py — RoPE 差异
ROPE_INDEX_FN = {
    "qwen2.5vl": get_rope_index_25,
    "qwen3vl":   get_rope_index_3,
    # "qwen3.5":  get_rope_index_35,
}

# thinkstream/trainer/sft.py → init_processor — special token 差异
if model_type == "qwen3vl":
    processor.tokenizer.add_tokens(["<silent>", "<response>"])
else:
    processor.tokenizer.add_tokens(["<silent>", "<response>", "<think>", "</think>"])
# 四动作格式 + per-timestep 结构 tag（方案 B: 精确匹配, JSON inside）
processor.tokenizer.add_tokens([
    # Action protocol
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
    # Input structure tags
    "<memory>", "</memory>",
    "<compressed>", "</compressed>",
    "<pending>", "</pending>",
    "<visual_window>", "</visual_window>",
    "<recalled_frames>", "</recalled_frames>",
    "<user_input>", "</user_input>",
    # Output payload
    "<summary>", "</summary>",
    # User input trigger
    "<compress_trigger>", "</compress_trigger>",
])
```

### 6.3 Vision Token 差异处理

```python
# load_video_frames 已处理差异：
if is_qwen3vl:
    pvi_kwargs["return_video_metadata"] = True
    # big_video_tensor 与 video_metadata 成对返回
    big_video_tensor, video_metadata = video_inputs_list[0]
else:
    big_video_tensor = video_inputs_list[0]
    video_metadata = None

# processor 调用差异：
if is_qwen3vl:
    processor_call_kwargs["do_resize"] = False
    processor_call_kwargs["video_metadata"] = chunk_metadatas
```

### 6.4 添加新模型的 Checklist

添加 Qwen3.5 或其他新模型时，需要：

1. [ ] `model/__init__.py`: 添加 `MODEL_CLS["qwen3.5"]`
2. [ ] `data/rope2d.py`: 实现 `get_rope_index_35()` 并注册
3. [ ] `model/patch.py`: 添加 `_lce_forward_qwen3_5` forward patch
4. [ ] `model/streaming_attention.py`: 确认 flex-attention 兼容性
5. [ ] `trainer/sft.py` → `init_processor`: 确认 special token 需求
6. [ ] `data/stream_data_processor.py` → `load_video_frames`: 处理 vision info 差异
7. [ ] 测试：运行 `tests/test_agent_sft.py` 全部通过

---

## 7. 训练课程（Curriculum）

### 7.1 五阶段课程

| 阶段 | 数据集 | 样本量 | 训练目标 | 超参 |
|------|--------|--------|---------|------|
| **Phase 1** 协议对齐 | P1: silent + response(from_frames) | ~4,000 | 学会 think + action 格式 | lr=1e-5, epochs=3 |
| **Phase 2** Recall 学习 | P2: + recall + pending + uncertain | ~6,000 | 学会判断 recall 时机 | lr=5e-6, epochs=3, from P1 ckpt |
| **Phase C1** 压缩行为 | C1: + compress(system指定) + compress_recall | ~15,000 | 学会按指定范围压缩 | lr=3e-6, epochs=2, from P2 ckpt |
| **Phase C2** 自选压缩 | C2: + compress(自选范围) | ~3,000 | 学会自选压缩窗口 | lr=2e-6, epochs=2, from C1 ckpt |
| **Phase 5** 混合训练 | P5: 所有类型按比例混合 | ~5,000 | 综合能力对齐 | lr=1e-6, epochs=1, from C2 ckpt |

### 7.2 数据集文件与注册（P1-4）

**Pipeline 同时输出两种文件**：

```
data/agent/final/
├── train.jsonl              # 全量训练集（所有 phase 混合）
├── val.jsonl                # 全量验证集
├── test.jsonl               # 全量测试集
├── phase1_train.jsonl       # 按 phase 拆分的训练集
├── phase2_train.jsonl
├── c1_train.jsonl
├── c2_train.jsonl
├── phase5_train.jsonl
└── pipeline_stats.json      # 含每个 phase 的样本量统计
```

Pipeline Pass 5 结尾新增拆分逻辑：

```python
# pipeline.py Pass 5 结尾
for phase in ["1", "2", "C1", "C2", "5"]:
    phase_samples = [s for s in train_samples if s["phase"] == phase]
    phase_name = {"1": "phase1", "2": "phase2", "C1": "c1", "C2": "c2", "5": "phase5"}[phase]
    path = FINAL_DIR / f"{phase_name}_train.jsonl"
    with open(path, "w") as f:
        for s in phase_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"  {phase_name}: {len(phase_samples)} samples → {path}")
    stats[f"{phase_name}_count"] = len(phase_samples)
```

**SFT 数据集注册**（`data_list.py`）：

```python
STREAM_AGENT_P1 = {
    "annotation_path": ".../agent/final/phase1_train.jsonl",
    "data_path": "./",
}
STREAM_AGENT_P2 = {
    "annotation_path": ".../agent/final/phase2_train.jsonl",
    "data_path": "./",
}
STREAM_AGENT_C1 = {
    "annotation_path": ".../agent/final/c1_train.jsonl",
    "data_path": "./",
}
STREAM_AGENT_C2 = {
    "annotation_path": ".../agent/final/c2_train.jsonl",
    "data_path": "./",
}
STREAM_AGENT_P5 = {
    "annotation_path": ".../agent/final/phase5_train.jsonl",
    "data_path": "./",
}
# 全量（用于调试或混合训练）
STREAM_AGENT_ALL = {
    "annotation_path": ".../agent/final/train.jsonl",
    "data_path": "./",
}
```

训练脚本直接 `--args.data.dataset_use stream_agent_p1`，不需要在代码中过滤 phase 字段。

### 7.3 训练脚本

```bash
#!/bin/bash
# scripts/sft_per_timestep.sh
# 用法: PHASE=1 bash scripts/sft_per_timestep.sh

PHASE=${PHASE:-1}
NPROC_PER_NODE=8
deepspeed=./scripts/zero3.json
entry_file=thinkstream/train.py

case $PHASE in
    1)
        llm=${LLM:-Qwen/Qwen2.5-VL-3B-Instruct}
        datasets=stream_agent_p1
        lr=1e-5; epochs=3
        run_name="agent-phase1"
        ;;
    2)
        llm=${LLM:?'Set LLM to Phase 1 checkpoint'}
        datasets=stream_agent_p2
        lr=5e-6; epochs=3
        run_name="agent-phase2"
        ;;
    C1)
        llm=${LLM:?'Set LLM to Phase 2 checkpoint'}
        datasets=stream_agent_c1
        lr=3e-6; epochs=2
        run_name="agent-c1"
        ;;
    C2)
        llm=${LLM:?'Set LLM to C1 checkpoint'}
        datasets=stream_agent_c2
        lr=2e-6; epochs=2
        run_name="agent-c2"
        ;;
    5)
        llm=${LLM:?'Set LLM to C2 checkpoint'}
        datasets=stream_agent_p5
        lr=1e-6; epochs=1
        run_name="agent-phase5"
        ;;
esac

output_dir=./output/${run_name}
batch_size=${BSZ:-8}
grad_accum_steps=${GRAD_ACCUM:-1}

args="
    sft \
    --args.train.deepspeed ${deepspeed} \
    --args.model.name_or_path ${llm} \
    --args.model.model_type ${MODEL_TYPE:-qwen2.5vl} \
    --args.model.max_length 16384 \
    --args.data.dataset_use ${datasets} \
    --args.data.flatten False \
    --args.data.video_max_pixels 150528 \
    --args.data.video_min_pixels 100352 \
    --args.train.bf16 True \
    --args.train.output_dir ${output_dir} \
    --args.train.num_train_epochs ${epochs} \
    --args.train.per_device_train_batch_size ${batch_size} \
    --args.train.gradient_accumulation_steps ${grad_accum_steps} \
    --args.train.save_steps 500 \
    --args.train.learning_rate ${lr} \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.torch_empty_cache_steps 1 \
    --args.train.dataloader.num_workers 4
"

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         ${entry_file} ${args}
```

### 7.4 max_length 降低

per-timestep 样本约 3500 tok（见 `data_construction_zh.md` §2.4），远低于旧方案的 32768。

| 设置 | 旧方案 (多轮) | 新方案 (per-timestep) |
|------|-------------|---------------------|
| `max_length` | 32768 | **16384** |
| 单样本实际长度 | 10,000-25,000 | **3,000-4,500** |
| 每卡 batch_size | 4 | **8-16** |
| 显存利用率 | 低（padding 多） | 高（样本长度均匀） |

---

## 8. Token 预算验证

### 8.1 理论预算（来自 data_construction_zh.md §2.4）

| 组件 | Tokens | 说明 |
|------|--------|------|
| System prompt | ~150 | 协议说明 |
| 压缩段 (≤5 × 150) | ≤750 | 最早期的记忆 |
| 未压缩 thinks (≤12 × 50) | ≤600 | 80% 触发压缩 |
| Visual window (12 chunks) | ~1536 | min_pixels=100352, 24 frames |
| User input | ~100 | 问题/trigger/result |
| Recall result (含 4 帧) | ~356 | 256 vision + 100 text |
| **输入总计** | **~3400** | |
| 输出: think + action + payload | ~80-280 | 按 action 类型变化 |
| **单样本总计** | **~3500-3700** | |

### 8.2 实测验证计划

Pipeline 产出后，执行以下验证：

```python
def verify_token_budget(samples, processor, model_type, max_length=16384):
    """Verify all samples fit within token budget."""
    lengths = []
    overflow = 0
    
    for sample in samples:
        messages = build_messages(sample)
        model_inputs = process_per_timestep_inputs(messages, ...)
        seq_len = model_inputs["input_ids"].shape[-1]
        lengths.append(seq_len)
        if seq_len > max_length:
            overflow += 1
    
    lengths = np.array(lengths)
    print(f"Token length: p50={np.median(lengths):.0f}, "
          f"p90={np.percentile(lengths, 90):.0f}, "
          f"p99={np.percentile(lengths, 99):.0f}")
    print(f"Overflow (>{max_length}): {overflow}/{len(samples)} ({overflow/len(samples):.1%})")
    
    # Vision token 实测
    vision_lengths = []
    for sample in samples[:100]:
        model_inputs = process_per_timestep_inputs(...)
        n_vision = model_inputs["video_mask"].sum().item()
        vision_lengths.append(n_vision)
    print(f"Vision tokens: mean={np.mean(vision_lengths):.0f}, "
          f"max={np.max(vision_lengths):.0f}")
```

**通过标准**：
- p99 < max_length × 0.9
- overflow < 0.5%
- vision tokens mean 与理论值偏差 < 20%

---

## 9. 训练循环集成

### 9.1 Builder 注册

在 `thinkstream/trainer/builder.py` 中注册新的 per-timestep SFT builder：

```python
def build_per_timestep_sft():
    """Build training pipeline for per-timestep agent SFT."""
    return train_pipeline.override(
        nodes=[
            # 模型加载
            with_hf_deepspeed_context.bind(
                wrapped=load_model,
            ),
            configure_model_gradients,
            init_processor,
            # 数据集（使用新的 PerTimestepDataset）
            init_per_timestep_dataset,
            # 训练循环（复用现有节点）
            align_special_tokens,
            set_gradient_checkpointing,
            build_optimizer_kwargs,
            # ... 优化器、调度器、训练步 ...
        ],
    )

TRAINER_BUILDERS["per_timestep_sft"] = build_per_timestep_sft
```

### 9.2 init_per_timestep_dataset 节点

```python
@node
def init_per_timestep_dataset(ctx, /, *, processor, model_type, data_dataset_use, 
                                model_max_length, vocab_size, train_dataset, data_collator, **kwargs):
    """Initialize per-timestep dataset and collator."""
    data_args = DataArgs()
    data_args.dataset_use = data_dataset_use
    data_args.model_type = model_type
    data_args.model_max_length = model_max_length
    # ... 设置其他视觉参数 ...
    
    dataset = PerTimestepDataset(processor, data_args)
    collator = PerTimestepDataCollator(processor.tokenizer, vocab_size=vocab_size)
    
    return ctx.update({
        train_dataset: dataset,
        data_collator: collator,
    })
```

### 9.3 Model Forward Patch

**需要新增 `per_timestep_mask_mod`**（见 §3.2）。不能复用旧的 `sliding_window_mod`，
因为旧 mask 的 block_ids 滑动窗口会遮蔽 recalled frames（P0-6）。

在 `streaming_attention.py` 中新增：
```python
def generate_per_timestep_mask_mod(video_mask, attention_mask):
    """Per-timestep: causal + padding only, no video sliding window."""
    B_limit = video_mask.shape[0]
    L_limit = video_mask.shape[1]
    
    def per_timestep_mod(b, h, q_idx, kv_idx):
        b_c = torch.clamp(b, 0, B_limit - 1)
        q_c = torch.clamp(q_idx, 0, L_limit - 1)
        k_c = torch.clamp(kv_idx, 0, L_limit - 1)
        in_bounds = (b < B_limit) & (q_idx < L_limit) & (kv_idx < L_limit)
        is_valid = attention_mask[b_c, k_c] > 0
        is_causal = q_c >= k_c
        return in_bounds & is_valid & is_causal
    
    return per_timestep_mod
```

在 `patch.py` 中，per-timestep SFT 的 forward 使用这个 mask_mod 替代 `sliding_window_mod`。
通过 `model.config` 中的 flag（如 `per_timestep_mode=True`）分派。

---

## 10. 评估策略

### 10.1 验证集

Pipeline 按 video_id 切分 train/val/test (80/10/10)。验证集用于：

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| **Action Accuracy** | 预测的 action 是否匹配 gold_action | ≥85% |
| **Think Quality** | think 中实体覆盖率（vs teacher caption） | ≥70% |
| **Response Correctness** | response 答案 vs gold_answer | ≥75% |
| **Compression Ratio** | 生成的 summary token 数 / 输入 thinks token 数 | 2.5-4.0 |
| **Query Quality** | recall query 不含答案 + 检索命中率 | 无泄漏 + recall@3 ≥60% |
| **Format Compliance** | 输出格式严格匹配四动作之一 | ≥95% |

### 10.2 评估脚本

```python
def evaluate_per_timestep(model, processor, val_dataset, model_type):
    """Evaluate per-timestep SFT model on validation set."""
    model.eval()
    metrics = defaultdict(list)
    
    for sample in val_dataset:
        # 构建 input（不含 output）
        messages = build_messages(sample, include_output=False)
        model_inputs = process_per_timestep_inputs(
            messages, ..., add_generation_prompt=True,
        )
        
        # Generate
        output_ids = model.generate(**model_inputs, max_new_tokens=300)
        output_text = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse and evaluate
        parsed = parse_action_output(output_text)
        gold = sample["metadata"]
        
        metrics["action_accuracy"].append(parsed["action"] == gold["gold_action"])
        if parsed["action"] == "response" and gold["gold_action"] == "response":
            metrics["response_correct"].append(
                fuzzy_match(parsed["response"], gold["gold_answer"]["answer"])
            )
        # ... 其他指标 ...
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### 10.3 Smoke Test 验收标准（训练前必须通过）

**最小验收**：随机抽 50 条样本（覆盖 silent / response / recall_query / recall_response / compress），
手动或脚本确认 **模型实际 input 可见内容 = 推理时状态机可见内容**。

```python
def smoke_test(samples, processor, model_type, n=50):
    """训练前验收：抽样检查 SFT input 和推理时状态机的一致性。"""
    import random
    random.seed(42)
    
    # 按 sample_type 分层抽样，确保覆盖所有类型
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["sample_type"]].append(s)
    
    selected = []
    required = ["silent", "response", "recall_query", "recall_response", "compress"]
    for st in required:
        pool = by_type.get(st, [])
        selected.extend(random.sample(pool, min(10, len(pool))))
    
    errors = []
    for s in selected:
        messages = build_messages(s)
        
        # 检查 1: recall_response 样本必须有 <recall_result> 在 user content 中
        if s["sample_type"] == "recall_response":
            user_text = str(messages[-2]["content"])  # user message
            if "<recall_result>" not in user_text:
                errors.append(f'{s["sample_id"]}: recall_response missing <recall_result>')
            if s["input"].get("recalled_frames") and "<recalled_frames>" not in user_text:
                errors.append(f'{s["sample_id"]}: recall_response missing <recalled_frames>')
        
        # 检查 2: visual_window 含 current_time
        user_text = str(messages[-2]["content"])
        if "<visual_window>" in user_text:
            if "current_time" not in user_text:
                errors.append(f'{s["sample_id"]}: visual_window missing current_time')
        
        # 检查 3: compress 样本的 output 含 <summary>
        if s["sample_type"] == "compress":
            if "<summary>" not in s["output"]:
                errors.append(f'{s["sample_id"]}: compress output missing <summary>')
        
        # 检查 4: post-recall response 不含 <think>
        if s["sample_type"] == "recall_response":
            if "<think>" in s["output"]:
                errors.append(f'{s["sample_id"]}: recall_response should not have <think>')
        
        # 检查 5: recalled_frames 不嵌套在 visual_window 内
        if "recalled_frames" in s.get("input", {}).get("visual_window", {}):
            errors.append(f'{s["sample_id"]}: recalled_frames nested in visual_window (P0-2)')
        
        # 检查 6: tokenize 后 labels 不全是 IGNORE_INDEX
        model_inputs = process_per_timestep_inputs(messages, processor=processor, model_type=model_type)
        labels = model_inputs["labels"]
        n_valid = (labels != IGNORE_INDEX).sum().item()
        if n_valid == 0:
            errors.append(f'{s["sample_id"]}: all labels are IGNORE_INDEX, no training signal')
        
        # 检查 7: sample_type 和 output format 匹配
        output = s["output"]
        if s["sample_type"] == "silent" and "<action>silent</action>" not in output:
            errors.append(f'{s["sample_id"]}: silent sample but output not silent action')
    
    if errors:
        for e in errors:
            logger.error(f"SMOKE TEST FAIL: {e}")
        raise AssertionError(f"Smoke test failed with {len(errors)} errors")
    
    logger.info(f"Smoke test passed: {len(selected)} samples OK")
```

**通过标准**：0 errors。任何 error 都阻塞训练，必须修复数据或 SFT loader。

---

## 11. 从旧方案迁移

### 11.1 代码变更清单

| 文件 | 变更 | 优先级 |
|------|------|--------|
| `thinkstream/data/stream_data_processor.py` | 新增 `PerTimestepDataset` + `PerTimestepDataCollator` | P0 |
| `thinkstream/data/stream_data_processor.py` | 新增 `process_per_timestep_inputs()` | P0 |
| `thinkstream/data/data_list.py` | 添加 Phase 1-5 数据集注册 | P0 |
| `thinkstream/trainer/sft.py` | 新增 `init_per_timestep_dataset` 节点 | P0 |
| `thinkstream/trainer/builder.py` | 注册 `per_timestep_sft` builder | P0 |
| `scripts/sft_per_timestep.sh` | 新的训练启动脚本 | P0 |
| `thinkstream/trainer/sft.py` → `init_processor` | 添加压缩相关 special tokens | P1 |
| `tests/test_agent_sft.py` | 添加 per-timestep 格式测试 | P1 |
| `scripts/agent_data_v5/pass4_forks.py` | 确保输出格式匹配 §2 | P1 |

### 11.2 向后兼容

**保留旧代码**。新的 per-timestep pipeline 与旧的多轮 pipeline 通过 builder name 和数据集格式自动分派：

```python
# 旧方案：torchrun ... train.py sft --args.data.dataset_use stream_cold_start
# 新方案：torchrun ... train.py per_timestep_sft --args.data.dataset_use stream_agent_p1
```

`LazySupervisedDataset` 通过 `protocol_version` 字段区分冷启动 vs 3-action 数据。
`PerTimestepDataset` 只处理 v5.5+ per-timestep 格式。

---

## 12. 核心设计约束（SFT 专有）

| # | 约束 | 原因 | 来源 |
|---|------|------|------|
| 1 | 每条样本只有一个 assistant turn | per-timestep 定义：一个 chunk 一个 action | 设计 |
| 2 | Loss 只覆盖 output，input 全部 IGNORE_INDEX | 防止在无视觉帧的历史 memory 上学习 | 设计 |
| 3 | Memory block 是 input 的一部分，不是 chat history | 匹配推理时的外部 memory 管理 | 设计 |
| 4 | Visual window 帧通过 ghost message 加载 | 统一的视频加载路径 | 设计 |
| 5 | Recalled frames 与 window frames 使用不同的 MROPE 时间编码 | 区分"当前"和"历史回忆" | 设计 |
| 6 | post-recall response 无 `<think>` | 避免同一 chunk 两次写入 text memory | 设计 |
| 7 | Compress 样本的 think 是当前帧观察，不是 "Compressing memory..." | think = 增量视觉记忆，与 action 无关 | 设计 |
| 8 | 超长样本在 Dataset 预过滤，collator 不做截断 | 右截断砍 output labels | P0-4 |
| 9 | 五阶段课程各 phase 的 ckpt 必须继承 | 不能从 base model 直接训 C1 | 设计 |
| 10 | 新旧方案通过 builder name 共存 | 不破坏旧实验的可复现性 | 设计 |
| 11 | recall_result 必须在 user_input 之前序列化进 input | 否则 post-recall response 看不到检索结果 | P0-1 |
| 12 | recalled_frames 是 `input` 顶层字段，不嵌套在 visual_window 内 | 否则 SFT loader 读不到 | P0-2 |
| 13 | Visual window header 显式标注当前 chunk 时间 | 否则 think 可能写整个 24s 描述 | P0-3 |
| 14 | 使用 per_timestep_mask_mod，不复用 sliding_window_mod | sliding_window 会遮蔽 recalled frames | P0-6 |
| 15 | Special token 使用方案 B（JSON inside tags） | XML 属性无法被 tokenizer 精确匹配 | P0-7 |
| 16 | Per-sample scalar loss weight，不用 vocab CE weight | labels 是 token id 不能乘权重 | P0-5 |
| 17 | 视觉帧优先用 frame_paths 加载，fallback 到 video 重采样 | 避免 fps/rounding 不一致 | P1-3 |
| 18 | Pipeline 同时输出分 phase 文件 + 统一文件 | 训练脚本最简单 | P1-4 |
| 19 | 训练前必须通过 50 条 smoke test | 验证 input 可见内容 = 推理时状态机 | 验收 |
| 20 | `recalled_frames` 必须在 `input` 顶层，pipeline 代码需修复嵌套 bug | pass4_forks.py 当前错误嵌套在 `visual_window` 内 | v2.1 审计 |
| 21 | `build_per_timestep_messages` 为死代码，`build_sample_input` 是唯一正确路径 | pass4_forks.py 中定义但从未调用 | v2.1 审计 |
| 22 | Pipeline 所有 Pass 均为 thinking=True（Qwen3.5 `/no_think` 无效） | 实测确认，见 `qwen35_output_format_analysis.md` | v2.1 审计 |
| 23 | 压缩触发使用 hysteresis：触发阈值 80%，压缩后须降至 55% 以下 | 防止频繁触发/窗口过短。见 config.py `COMPRESS_HYSTERESIS_RATIO` | v2.1 |

---

## 参考文件

| 文件 | 说明 |
|------|------|
| `docs/data_construction_zh.md` | 数据构造方案（per-timestep 格式定义） |
| `thinkstream/data/stream_data_processor.py` | 数据加载 + tokenization |
| `thinkstream/model/streaming_attention.py` | flex-attention CUDA kernel |
| `thinkstream/model/patch.py` | 模型 forward patch + video_block_mask |
| `thinkstream/data/rope2d.py` | MROPE 位置编码 |
| `thinkstream/trainer/sft.py` | SFT 训练节点 |
| `thinkstream/trainer/builder.py` | 训练 pipeline builder 注册 |
| `thinkstream/model/__init__.py` | MODEL_CLS 注册表 |
| `scripts/agent_data_v5/pipeline.py` | 数据构造 pipeline |
| `scripts/agent_data_v5/config.py` | Pipeline 配置常量 |
