# 流视频 Agent SFT 工程设计

> 版本: v1.1 | 日期: 2026-04-21
>
> 配套文档：`data_construction_zh.md`（数据构造方案 v5.6）
>
> 本文档定义 **如何将 pipeline 产出的 per-timestep 训练样本转化为可训练的 SFT batch**，
> 涵盖注意力掩码、数据加载、多模型兼容、训练课程等全部工程细节。

---

## 1. 为什么需要新的 SFT 方案

### 1.1 旧方案的问题

旧方案（`_build_messages` / `_build_agent_messages`）将完整视频的多轮对话作为一条训练样本：

```
[System] → [User: chunk0] → [Asst: think0] → [User: chunk1] → [Asst: think1] → ... → [User: chunkN] → [Asst: thinkN]
```

**三个根本性缺陷**：

| 问题 | 描述 |
|------|------|
| **无图编描述** | 旧 chunk 的视觉帧已被滑动窗口淘汰，但 loss 仍覆盖其 assistant turn → 模型学会"没有图也能编" |
| **压缩不可执行** | 在 causal attention 中，"用 summary 替换旧 thinks" 无法真正实现——旧 token 仍在序列中可 attend |
| **训推不一致** | 训练时 memory 是 chat history；推理时 memory 是外部状态管理。gap 越大，性能越差 |

### 1.2 新方案：Per-Timestep 独立样本

每个 2s chunk 构造为**独立训练样本**：

```
单条样本 = {
    input:  [System] + [Memory Block] + [Visual Window] + [User Input]
    output: <think>...</think><action>X</action>[payload]
}
```

**只对当前这一步的 output 计算 loss**。历史 memory 作为 input context，不参与梯度。

这完美匹配推理时的信息流：模型每一步看到的就是 system prompt + 外部管理的 memory state + 当前视觉窗口 + 用户输入。

---

## 2. 训练样本结构

### 2.1 Input 编码

```
┌─────────────────────────────────────────────────────┐
│ [System Prompt]                          ~150 tok    │
│ 协议说明 + 四动作格式                                  │
├─────────────────────────────────────────────────────┤
│ [Memory Block]                           ~600-1350 tok │
│ <compressed>{"time_range":[0,20],"text":"..."}</compressed>  │
│ <compressed>{"time_range":[20,40],"text":"..."}</compressed> │
│ [40-42] Chef places tomatoes on board...             │
│ [42-44] Salt sprinkled from small white bowl...      │
│ <pending>{"since":44,"question":"Tell me when basil added"}</pending> │
├─────────────────────────────────────────────────────┤
│ [Visual Window Header]                   ~40 tok     │
│ <visual_window start="20" end="44" current="42-44"> │
│ Write <think> only for the current 2s chunk (42-44s).│
│ Earlier frames are context only.                     │
│ </visual_window>                                     │
├─────────────────────────────────────────────────────┤
│ [Visual Window Frames]                   ~1536 vision tok │
│ 12 chunks × 2 frames = 24 frames                    │
│ 使用 ghost message 统一加载                            │
├─────────────────────────────────────────────────────┤
│ [Recalled Frames] (可选, 仅 post-recall)  ~260 vision tok │
│ 4 frames from historical time range                  │
│ 前置文本: [Recalled frames from t=28-32s]            │
├─────────────────────────────────────────────────────┤
│ [Recall Result Text] (可选, 仅 post-recall) ~100 tok │
│ <recall_result>{"source":"student_think","time":"28-32",│
│  "text_content":"..."}</recall_result>               │
├─────────────────────────────────────────────────────┤
│ [User Input]                             ~50 tok     │
│ 问题 / <compress_trigger>{"range":[20,34]}</compress_trigger> │
│ / "Continue following the protocol to respond." / (空) │
└─────────────────────────────────────────────────────┘
```

**关键顺序约定**（不可违反）：
1. Memory block → Visual window header → Visual frames → Recalled frames → Recall result → User input
2. `recall_result` 必须在 `user_input` 之前，否则 post-recall response 看不到检索结果（**P0-1**）
3. `recalled_frames` 是 `input` 的顶层字段，不嵌套在 `visual_window` 内（**P0-2**）
4. Visual window header 显式标注当前 chunk 时间，确保 think 只写当前 2s（**P0-3**）

### 2.2 Output 编码

根据 gold_action 有四种 output 格式：

```python
# silent
"<think>Chef adjusting burner dial...</think><action>silent</action>"

# response
"<think>Answer visible in current frames...</think><action>response</action><response>The apron is red.</response>"

# recall
"<think>Evidence not in window...</think><action>recall</action><query>{\"query\":\"apron color chef\",\"time_range\":\"0-20\"}</query>"

# compress (C1: range 由 trigger 指定; C2: 模型自选)
"<think>Current visual observation...</think><action>compress</action><summary>{\"time_range\":[20,34],\"text\":\"Oil heated, garlic browned...\"}</summary>"
```

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

因此，per-timestep 方案的 attention mask 是**标准 causal mask**，加上视觉滑动窗口：

```python
def per_timestep_mask_mod(b, h, q_idx, kv_idx):
    is_valid   = attention_mask[b, kv_idx] > 0
    is_causal  = q_idx >= kv_idx
    # 视觉窗口内的所有 video token 都可见（只有 12 chunks）
    # text token 全部可见（memory block + system + user）
    return is_valid & is_causal
```

**但仍需要 flex-attention** 的原因是 recalled frames：
- Recalled frames 作为额外的 vision token 出现在 input 中
- 它们来自不同时间范围，需要正确的 MROPE 时间编码
- flex-attention 的 video_mask 区分 visual_window frames 和 recalled frames

### 3.3 Recalled Frames 的 Attention 处理

```
Token 序列:
[system] [memory_text] [visual_window: 24 frames] [recalled: 4 frames] [user_input] [output]
                       ├── video_mask = True ──┤   ├── recalled_mask ──┤

video_mask:     [0 0 ... 0  1 1 1 ... 1  1 1 1 1  0 0 ... 0  0 0 ... 0]
recalled_mask:  [0 0 ... 0  0 0 0 ... 0  1 1 1 1  0 0 ... 0  0 0 ... 0]
```

两组 vision token 都 attend to 所有位置（标准 causal），但 **MROPE 时间编码不同**：
- visual_window frames: `second_per_grid_ts` 基于 video_chunk_size (2s)
- recalled frames: `second_per_grid_ts` 基于 recalled time range

这确保模型知道 recalled frames 来自过去的不同时间点。

### 3.4 未来扩展：短序列滑动窗口方案

如果实验发现 per-timestep 独立样本缺少连续性（开放问题 #1），可切换为 3-step 滑动窗口：

```
训练样本 = {
    input:  [System] + [Memory_at_t-2] + [Visual_window_at_t-2]
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

将 pipeline JSON 转为 chat template 可消费的 messages：

```python
def _build_messages(self, sample):
    """Convert pipeline sample to chat messages.
    
    顺序不可违反：memory → visual_header → visual_frames → recalled_frames
    → recall_result → user_input。见 §2.1 关键顺序约定。
    """
    inp = sample["input"]
    
    # System prompt
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    # User message: memory + visual window + recall + user input
    user_content = []
    
    # 1. Memory block (纯文本)
    memory_text = self._format_memory_block(inp["memory"])
    user_content.append({"type": "text", "text": memory_text})
    
    # 2. Visual window header — 告诉模型当前 chunk 是哪 2 秒 (P0-3)
    vw = inp["visual_window"]
    chunk_idx = sample["chunk_idx"]
    current_start = chunk_idx * 2  # AGENT_CHUNK_SEC = 2
    current_end = current_start + 2
    user_content.append({"type": "text", "text": (
        f'\n<visual_window start="{vw["video_start"]}" end="{vw["video_end"]}" '
        f'current="{current_start}-{current_end}">\n'
        f'Write <think> only for the current 2s chunk ({current_start}-{current_end}s). '
        f'Earlier frames are context only.\n</visual_window>'
    )})
    
    # 3. Visual window frames: 12 chunks × 2 frames = 24 frames
    user_content.append({
        "type": "video",
        "video": sample["video_path"],
        "video_start": vw["video_start"],
        "video_end": vw["video_end"],
        "nframes": vw["frames"],
    })
    
    # 4. Recalled frames (可选, 仅 post-recall; 顶层字段, 不嵌套在 visual_window 中, P0-2)
    if "recalled_frames" in inp:
        rf = inp["recalled_frames"]
        user_content.append({"type": "text", "text": (
            f'\n[Recalled frames from t={rf["time_range"][0]}-{rf["time_range"][1]}s]'
        )})
        user_content.append({
            "type": "video",
            "video": sample["video_path"],
            "video_start": rf["time_range"][0],
            "video_end": rf["time_range"][1],
            "nframes": rf["n_frames"],
        })
    
    # 5. Recall result text (可选, 仅 post-recall, P0-1)
    #    必须在 user_input 之前，否则模型看不到检索结果
    if inp.get("recall_result"):
        rr = inp["recall_result"]
        user_content.append({"type": "text", "text": (
            f'\n<recall_result>'
            f'{{"source":"{rr.get("source", "")}","time":"{rr.get("time", "")}",'
            f'"text_content":"{rr.get("text_content", "")}"}}'
            f'</recall_result>'
        )})
    
    # 6. User input (question / compress_trigger / "Continue..." / empty)
    if inp.get("user_input"):
        user_content.append({"type": "text", "text": "\n" + inp["user_input"]})
    
    messages.append({"role": "user", "content": user_content})
    
    # Assistant output (训练目标)
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
# 四动作格式的额外 token（所有模型都需要）
processor.tokenizer.add_tokens([
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
    "<compressed>", "</compressed>", "<pending>", "</pending>",
    "<compress_trigger>", "</compress_trigger>",
    "<summary>", "</summary>",
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
| **Phase C1** 压缩行为 | C1: + compress(system指定) + compress_recall | ~15,000 | 学会按指定范围压缩 | lr=5e-6, epochs=2, from P2 ckpt |
| **Phase C2** 自选压缩 | C2: + compress(自选范围) | ~3,000 | 学会自选压缩窗口 | lr=2e-6, epochs=2, from C1 ckpt |
| **Phase 5** 混合训练 | P5: 所有类型按比例混合 | ~5,000 | 综合能力对齐 | lr=2e-6, epochs=1, from C2 ckpt |

### 7.2 数据集注册

```python
# thinkstream/data/data_list.py 中添加：

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
```

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
        lr=5e-6; epochs=2
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
        lr=2e-6; epochs=1
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

模型的 forward patch（`patch.py`）无需修改 — 已有的 `build_video_block_mask` 基于 `video_mask` 构建 flex-attention mask，per-timestep 样本的 `video_mask` 格式完全兼容。

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

| # | 约束 | 原因 |
|---|------|------|
| 1 | 每条样本只有一个 assistant turn | per-timestep 定义：一个 chunk 一个 action |
| 2 | Loss 只覆盖 output，input 全部 IGNORE_INDEX | 防止在无视觉帧的历史 memory 上学习 |
| 3 | Memory block 是 input 的一部分，不是 chat history | 匹配推理时的外部 memory 管理 |
| 4 | Visual window 帧通过 ghost message 加载 | 统一的视频加载路径 |
| 5 | Recalled frames 与 window frames 使用不同的 MROPE 时间编码 | 区分"当前"和"历史回忆" |
| 6 | post-recall response 无 `<think>` | 避免同一 chunk 两次写入 text memory |
| 7 | Compress 样本的 think 是当前帧观察，不是 "Compressing memory..." | think = 增量视觉记忆，与 action 无关 |
| 8 | max_length=16384 足够 | 单样本 ~3500 tok，4.7x 余量 |
| 9 | 五阶段课程各 phase 的 ckpt 必须继承 | 不能从 base model 直接训 C1 |
| 10 | 新旧方案通过 builder name 共存 | 不破坏旧实验的可复现性 |

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
