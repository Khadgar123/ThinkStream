# 流式 Agent 位置编码与 KV Cache 分区设计

> 版本: v1.0 | 日期: 2026-04-22
>
> 配套文档：`sft_engineering.md`（SFT 工程设计 v2.1）, `data_construction_zh.md`（数据构造方案 v6.2）
>
> 本文档定义流式视频 Agent 的 **输入布局 → 位置编码 → KV Cache 分区** 方案。
> 目标：在保证训练-推理格式一致的前提下，实现高效增量推理。

---

## 0. 问题陈述

### 0.1 流式 Agent 的输入特性

每个推理步（2 秒一个 chunk），模型看到的输入由四部分组成：

| 区域 | 内容 | 变化模式 | 大小 |
|------|------|----------|------|
| 视觉窗口 | 最近 12 chunks 的 24 帧 | 每步 +2 帧 -2 帧（sliding window） | ~1536 tok（固定） |
| 压缩摘要 | 历史时段的文本摘要 | 偶尔新增一段，极少变化 | ~750 tok（上限） |
| 近期 thinks | 最近 N 步的观察文本 | 每步追加 ~50 tok，压缩时批量删除 | ~600 tok（预算） |
| 用户输入 | 问题 / 压缩触发 / 空 | 每步变化 | ~50 tok |

关键矛盾：
- **视频帧**是固定窗口大小，滑动式更新
- **文本 thinks** 是追加式增长，周期性压缩（删除旧 thinks + 新增摘要）
- **检索帧**临时出现（recall action），用完即移除
- 三者在时间上是**交错的**（chunk 5 的帧和 chunk 5 的 think 描述同一时刻），但在 token 序列中是**分离的**

### 0.2 当前方案的问题

当前布局（`agent_protocol.py`）：
```
[system] → [<memory> compressed + thinks + pending </memory>] → [<visual_window> 24帧 </visual_window>] → [user_input]
```

问题：

1. **位置漂移**：每步 thinks 增长 ~50 tok，导致视频 token 的 position IDs 右移。同一帧在不同 step 的 position 不同。
2. **时间-位置脱耦**：think_5 描述 t=10-12s，视频帧 chunk_5 也对应 t=10-12s，但它们在 position space 中相距 ~1000+，RoPE 远距离衰减削弱了时间关联。
3. **KV 无法复用**：文本区变化导致后面的视频区 position 全部失效，每步必须 full prefill ~3000 tok。
4. **压缩不可增量**：压缩时文本区内容大幅变化（多条 thinks → 一条摘要），KV cache 中旧 thinks 的 hidden states 不能被 summary 的 hidden states 替换（token 数不同、语义不同）。

---

## 1. 方案：分区布局 + 时间对齐 MROPE

### 1.1 新 Token 序列布局

```
┌──────────┬───────────────────────┬──────────────────────────────────────┬──────────┐
│ Zone A   │ Zone B                │ Zone C                                │ Zone D   │
│ system   │ visual_window         │ memory_text                           │ user     │
│ prompt   │ (固定 24帧)           │ compressed + thinks + pending + recall│ input    │
│ ~150 tok │ ~1536 tok (固定)      │ ~600-1350 tok (可变)                  │ ~50 tok  │
└──────────┴───────────────────────┴──────────────────────────────────────┴──────────┘
```

**核心改动：视频帧移到文本前面。**

理由：
- Zone B 大小固定（24 帧 sliding window），position 区间稳定
- Zone C 在末尾追加式增长，不影响前面的 position
- 压缩只影响 Zone C，Zone A+B 的 KV 完全可复用

### 1.2 时间对齐 MROPE

Qwen-VL 使用 3D MROPE 位置编码：每个 token 有 (temporal, height, width) 三个独立 position ID。

**当前行为**：
- 视频 token: temporal 按真实时间间隔编码（通过 `second_per_grid_ts`）
- 文本 token: temporal = height = width = 递增序号（退化为 1D）

**新方案：让文本 think 的 temporal position 和对应视频帧对齐**

```
视频帧 chunk_5 (t=10-12s):
  temporal_pos = encode(10)    ← 真实时间
  height_pos   = spatial_grid
  width_pos    = spatial_grid

文本 think_5 (描述 t=10-12s 的观察):
  temporal_pos = encode(10)    ← 和视频帧相同的时间编码!
  height_pos   = 递增序号       ← 退化为 1D（文本无空间结构）
  width_pos    = 递增序号

压缩摘要 [0-16s]:
  temporal_pos = encode(0)     ← 摘要覆盖的起始时间
  height_pos   = 递增序号
  width_pos    = 递增序号
```

效果：即使 think_5 在 token 序列中的绝对位置是 2000，视频帧 chunk_5 的绝对位置是 500，它们在 MROPE 的 **temporal 维度**上共享相同值，RoPE 会在时间维度上拉近它们的 attention 权重。

---

## 2. 技术可行性分析

### 2.1 MROPE 支持独立时间维度

Qwen2-VL 论文 (arXiv:2409.12191) 明确指出 M-RoPE 将旋转嵌入分解为三个独立维度。"Ordering and position indexing are orthogonal design choices" — 位置索引和 token 排列顺序是正交的设计选择。

**这意味着文本 token 的 temporal position 可以独立于其序列位置进行设置，不违反架构约束。**

### 2.2 相关工作支持

| 来源 | 关键发现 | 对本方案的支持 |
|------|----------|----------------|
| Qwen2-VL (2024) | MROPE 3 维解耦：temporal/height/width 独立编码 | 架构原生支持独立 temporal 赋值 |
| VRoPE (arXiv:2502.11664) | 低频应分配给 temporal 维度，避免振荡 | 语义时间 position 在低频区间自然平滑 |
| Interleaved-MRoPE (Qwen3-VL) | position IDs "增长比 vanilla RoPE 更慢"，为非顺序赋值留出空间 | 非递增 temporal position 在已验证的范围内安全 |
| VideoRoPE (2025) | positional attention bias 集中在空间区域 | temporal 对齐可增强时间粒度的 attention |

### 2.3 Qwen2.5-VL vs Qwen3-VL 的差异

| 特性 | Qwen2.5-VL | Qwen3-VL |
|------|------------|----------|
| 时间编码方式 | `second_per_grid_ts` 控制 temporal position 间距 | timestamp token 嵌入在序列中，temporal dim = 0 |
| 文本 temporal | 递增序号（退化 1D） | 递增序号（退化 1D） |
| 定制难度 | **低**：修改 `get_rope_index_25` 中文本区的 temporal 赋值 | **中**：需要绕过 timestamp token 机制 |

**推荐优先在 Qwen2.5-VL 上实现**，因为其 temporal position 是显式数值（`second_per_grid_ts` 直接控制），定制最直接。

### 2.4 潜在风险

1. **长度外推**：如果时间戳超出预训练范围（如 t=3600s），temporal RoPE 可能外推失败。**缓解**：使用归一化时间 `t / video_duration` 而非绝对秒数。
2. **位置碰撞**：文本 temporal position 和视频 temporal position 相同时，模型可能混淆模态。**缓解**：文本的 height/width 维度使用不同数值区间（如从 max_spatial+1 开始递增），利用空间维度区分模态。
3. **注意力偏移**：temporal 对齐可能导致模型过度关注时间匹配的帧而忽略空间细节。**缓解**：通过 ablation 验证效果，必要时可回退为弱对齐（使用粗粒度时间区间而非精确时间戳）。

---

## 3. KV Cache 分区管理

### 3.1 四区 KV Cache

```
                KV Cache Layout
┌─────────────────────────────────────────────┐
│ Zone A: system prompt     (~150 tok)        │  永不变，永远复用
├─────────────────────────────────────────────┤
│ Zone B: visual window     (~1536 tok)       │  固定大小 sliding window
│   [chunk_N-11] [chunk_N-10] ... [chunk_N]   │  每步: evict oldest + append newest
├─────────────────────────────────────────────┤
│ Zone C: memory text       (~600-1350 tok)   │  追加式增长, 压缩时重算
│   [compressed_0] ... [think_K] ... [think_N]│
├─────────────────────────────────────────────┤
│ Zone D: user input        (~50 tok)         │  每步重算
└─────────────────────────────────────────────┘
```

### 3.2 各场景的 KV 操作

**正常步（无压缩、无 recall）：**
```
Zone A: 不变 → 复用                              0 tok prefill
Zone B: evict 最旧 chunk KV, append 新 chunk KV   ~128 tok prefill
Zone C: append 新 think tokens                    ~50 tok prefill
Zone D: 重新 prefill                              ~50 tok prefill
───────────────────────────────────────────
总计:                                             ~228 tok prefill
```

**压缩步：**
```
Zone A: 不变 → 复用                              0 tok prefill
Zone B: evict + append                           ~128 tok prefill
Zone C: 内容大幅变化 → 全部重新 prefill           ~500 tok prefill
Zone D: 重新 prefill                              ~50 tok prefill
───────────────────────────────────────────
总计:                                             ~678 tok prefill
```

**Recall 步（step 2, recall_response）：**
```
Zone A: 不变 → 复用                              0 tok prefill
Zone B: 正常 evict/append + 追加 recalled frames ~128 + 256 tok prefill
Zone C: pending_questions 变化 + recall_result 追加
        → 部分重新 prefill                       ~200 tok prefill
Zone D: 重新 prefill                              ~50 tok prefill
───────────────────────────────────────────
总计:                                             ~634 tok prefill
```

**对比当前全量 prefill：~3086 tok/步 → 平均降低 ~90%。**

### 3.3 实现依赖

分区 KV cache 需要以下能力：

| 能力 | 当前状态 | 实现方式 |
|------|----------|----------|
| 视频 KV evict/append | ✅ `StreamingWindowInferenceEngine` 已实现 | 复用 `CacheEviction` |
| 文本 KV 增量 append | ⚠️ `StreamingInferenceEngine.prefill` 支持追加 | 在分区 KV 管理器中调用 |
| 文本 KV 全量重算 | ✅ 标准 prefill | 压缩步触发 |
| 分区 position IDs | ❌ 需要实现 | 新增 `compute_zone_position_ids` |
| 跨区 causal attention | ✅ flash_attn 自动处理 | KV 按 zone 顺序排列，causal 天然成立 |

---

## 4. Position ID 计算细节

### 4.1 Qwen2.5-VL 实现方案

在 `get_rope_index_25` 基础上扩展，核心改动是**文本 token 的 temporal position 注入时间戳**：

```python
def get_rope_index_agent(
    spatial_merge_size, input_ids, image_grid_thw, video_grid_thw,
    second_per_grid_ts,
    text_temporal_positions=None,  # 新参数: List[Tuple[int, int, float]]
                                  # [(start_token_idx, end_token_idx, timestamp_sec), ...]
    attention_mask=None,
):
    """Agent-specific MROPE: 支持文本 token 的 temporal position 对齐。
    
    text_temporal_positions 指定哪些文本 span 应使用语义时间戳：
    - compressed_segment [0-16s] → temporal_pos = encode(0)
    - think about t=10-12s → temporal_pos = encode(10)
    - 其他文本 (system, user_input) → 标准递增
    
    只修改 temporal 维度, height/width 维度保持递增。
    """
    # 1. 调用标准 get_rope_index_25 获得基础 position_ids
    position_ids, deltas = get_rope_index_25(
        spatial_merge_size, input_ids, image_grid_thw,
        video_grid_thw, second_per_grid_ts, attention_mask,
    )
    
    # 2. 覆盖指定文本 span 的 temporal position
    if text_temporal_positions is not None:
        for batch_idx, spans in enumerate(text_temporal_positions):
            for (tok_start, tok_end, timestamp_sec) in spans:
                # 将时间戳转换为和视频帧相同的 temporal position 尺度
                # Qwen2.5-VL: temporal_pos = time_sec * 2 / second_per_grid_ts
                # 简化: 使用 tokens_per_second 缩放
                temporal_pos = int(timestamp_sec * 2)  # 匹配视频 temporal 编码
                
                # 只修改 temporal 维度 (dim=0)
                # height (dim=1) 和 width (dim=2) 保持原始递增值
                position_ids[0, batch_idx, tok_start:tok_end] = temporal_pos
    
    return position_ids, deltas
```

### 4.2 Qwen3-VL 实现方案

Qwen3-VL 使用 timestamp token 编码时间（temporal 维度恒为 0），需要不同策略：

```python
# Qwen3-VL 的时间信息通过 <t=10.0> 这样的 timestamp token 传递。
# 方案：在文本 think 前注入对应的 timestamp token
# 
# 改前: <memory> [compressed] [10-12] chef stirs pot... [12-14] ... </memory>
# 改后: <memory> [compressed] <t=10.0> [10-12] chef stirs pot... <t=12.0> ... </memory>
#
# 这样 Qwen3-VL 的标准 RoPE 逻辑会自动处理时间对齐，无需定制 rope 函数。
```

### 4.3 输入构造中的时间标注

`build_user_content` 需要传递时间信息给 RoPE 计算：

```python
def build_user_content(memory_text, chunk_idx, video_path, ...):
    # 视频部分（Zone B）: 正常
    user_content.append({"type": "video", ...})
    
    # 文本部分（Zone C）: 附带时间标注给 position ID 计算
    # 标注格式: 每段文本带上 temporal_anchor
    #   compressed [0-16s] → temporal_anchor = 0
    #   think [10-12s]     → temporal_anchor = 10
    #   pending             → temporal_anchor = current_time (最新时间)
    user_content.append({
        "type": "text",
        "text": memory_text,
        "temporal_anchors": [  # 新字段，供 RoPE 计算使用
            {"token_range": [0, 30], "timestamp": 0},     # compressed_0
            {"token_range": [31, 80], "timestamp": 10},    # think_5
            ...
        ],
    })
```

---

## 5. 训练侧 Attention Mask

### 5.1 Flex Attention 适配

当前 `streaming_attention.py` 的 `generate_video_sliding_window_mask_mod` 对视频 token 做 sliding window mask。新布局需要确保：

1. **视频 token 在前**：mask_mod 中 video_mask 的检测逻辑不变（检测 video_token_id）
2. **文本 token 可 attend 到所有视频 token**：现有逻辑已满足（`(~k_is_video)` 条件使文本 KV 对所有 query 可见）
3. **视频 token 只 attend 到 window 内**：现有逻辑已满足

**结论：flex_attention mask 不需要改动，因为它基于 token type（video vs text）而非 position 做判断。**

### 5.2 Causal Mask

新布局 `[video] [text]` 中，视频在前、文本在后。Causal mask 保证：
- 文本 token 可以 attend 到所有前面的视频 token ✅
- 视频 token 不能 attend 到后面的文本 token ✅（causal）
- 同一 Zone 内保持 causal ✅

这和当前的 causal 行为一致。

---

## 6. 对造数据代码的改动

### 6.1 不影响 Pass1-Pass3

Pass1（evidence）、Pass2（rollout）、Pass3（tasks）的核心逻辑不变：
- 它们生成 observations、compression events、tasks 等**语义内容**
- Token 布局和 position encoding 是下游（Pass4 + SFT）的关注点

### 6.2 Pass4 改动：新布局 + 时间标注

`pass4_forks.py` 中的 `build_per_timestep_messages` 和 `build_sample_input` 需要改为新布局：

```python
# 改前 (pass4_forks.py:build_per_timestep_messages):
user_content = [
    {"type": "video", "video_start": ..., "video_end": ..., "nframes": 24},  # 视频
    {"type": "text", "text": memory_text},  # 文本在后
]

# 改后:
user_content = [
    {"type": "video", "video_start": ..., "video_end": ..., "nframes": 24},  # 视频（不变，已在前）
    {"type": "text", "text": memory_text, "temporal_anchors": anchors},  # 文本在后 + 时间标注
]
```

> **已完成**：`pass4_forks.py`、`agent_protocol.py`、`sft/data_processor.py` 三处均已统一为视频在前。

### 6.3 agent_protocol.py（已完成）

`build_user_content` 的 content 顺序已改为：
```
<visual_window> → <recalled_frames> → <memory> → <recall_result> → <user_input>
```

### 6.4 data_processor.py（已完成）

`build_per_timestep_messages` 已同步调整为视频在前。`_get_item` 待接入 `get_rope_index_agent` 传入 `text_temporal_positions`（需要 tokenization 后的 span 映射）。

### 6.5 Pass5 验证不受影响

Pass5 验证的是输出内容（think 长度、provenance、grounding 等），不检查输入 token 顺序或 position IDs。

### 6.6 改动完成状态

| 文件 | 改动内容 | 状态 |
|------|----------|------|
| `thinkstream/data/agent_protocol.py` | 新建。`build_user_content` 视频在前 | ✅ 已完成 |
| `thinkstream/sft/data_processor.py` | `build_per_timestep_messages` 视频在前 | ✅ 已完成 |
| `thinkstream/data/rope2d.py` | 新增 `get_rope_index_agent` | ✅ 已完成 |
| `thinkstream/sft/rope2d.py` | re-export `get_rope_index_agent` | ✅ 已完成 |
| `thinkstream/model/agent_loop.py` | 新建。单步推理 + BM25 检索 + 引擎适配 | ✅ 已完成 |
| `thinkstream/eval/eval_common.py` | 新增 `mcq_predict_agent_loop` | ✅ 已完成 |
| `thinkstream/model/inference.py` | `streaming_video_chat` 标记 deprecated | ✅ 已完成 |
| `thinkstream/trainer/grpo.py` | 添加迁移注释 | ✅ 已完成 |
| `scripts/agent_data_v5/pass4_forks.py` | recall_response 加 think + C2 variant | ✅ 已完成 |
| `scripts/agent_data_v5/pass2_rollout.py` | compress prompt 加入重叠帧 | ✅ 已完成 |
| `scripts/agent_data_v5/pass5_verify.py` | provenance 放宽 + grounding 统一 | ✅ 已完成 |
| `thinkstream/model/streaming_attention.py` | 无改动（mask 基于 token type） | — |
| `thinkstream/sft/data_processor.py:_get_item` | 接入 temporal_positions | ⏳ 待实现（需 tokenization span 映射） |
| `thinkstream/model/inference.py:ZonedKVCache` | 分区 KV cache | ⏳ Phase 3 优化 |

---

## 7. 实施路线

### Phase 1: 布局统一 ✅ 已完成

1. ✅ 统一所有 `build_user_content` / `build_per_timestep_messages` 的顺序为视频在前
2. Pass4 输出已是视频在前，无需重新生成
3. ✅ 训练 + 推理格式验证（test_per_timestep_sft.py 更新）

### Phase 2: 时间对齐 MROPE（部分完成）

1. ✅ 实现 `get_rope_index_agent`（rope2d.py，支持 Qwen2.5-VL 和 Qwen3-VL）
2. ⏳ SFT data_processor `_get_item` 中提取 temporal spans 并传入（需要 tokenization 后的 token→时间 映射）
3. ⏳ Ablation：对齐 vs 不对齐的 QA 准确率对比

### Phase 3: 增量 KV Cache（待实现）

1. 实现分区 KV 管理器
2. 推理端增量 prefill（正常步 ~228 tok vs 全量 ~3086 tok）
3. 压缩步全量重算 Zone C
4. 延迟评估 + 吞吐对比

---

## 8. 帧 Evict 后文本仍存在时的 Temporal Position 分析

### 8.1 核心场景

文本记忆的时间跨度远大于视觉窗口，这是**正常且频繁的**状态。

假设当前 chunk_idx = 31，视觉窗口 = chunk 20-31（t=40-64s）：

```
视频帧 (Zone B): 只有 t=40-64s 的帧
                 chunk 5 的帧早在 step 13 就被 evict 了

文本记忆 (Zone C):
  compressed_0: [0-16s] 的摘要       ← 帧在很久前就没了
  compressed_1: [16-32s] 的摘要      ← 帧也没了
  think_16: [32-34s] 的观察          ← 帧没了
  think_17: [34-36s]                 ← 帧没了
  think_18: [36-38s]                 ← 帧没了
  think_19: [38-40s]                 ← 帧没了
  think_20: [40-42s]                 ← 帧还在! 可以对齐
  ...
  think_31: [62-64s]                 ← 帧还在! 可以对齐
```

Temporal position 的实际分布：

```
Token 序列:
  [system]  [video: t=40-64s]  [compressed_0] [compressed_1] [think_16...think_31] [user]

Temporal position (dim=0):
  [0..149]  [40...........64]  [0]            [16]           [32 34 36...62]       [65]
                ↑                ↑              ↑              ↑           ↑
              有帧对应          无帧           无帧          无帧对应    有帧对应
```

### 8.2 结论：不需要 remap，保持原始时间戳即是最正确的设计

文本 think 和对应帧的关系有三种状态，temporal position 在三种状态下**都语义正确**：

#### 状态 1：帧还在 + think 在（共存对齐）

```
think_25 (t=50-52s): temporal_pos = 50
video chunk_25 帧:    temporal_pos = 50   ← 完美对齐
```

RoPE temporal 距离 = 0，cross-modal attention 最强。模型可以同时看帧和文本，互相验证。

#### 状态 2：帧已 evict + think 仍在（孤立文本）

```
think_16 (t=32-34s): temporal_pos = 32
最近的视频帧:         temporal_pos = 40   ← 差 8 个时间单位
```

Temporal position 传达了正确的语义："这条观察是关于 8 秒前的事"。RoPE 自然衰减使得模型对旧文本的 attention 较弱，符合"越旧越不确定"的直觉。文本记忆的全部意义就是**在帧消失后保留信息**，temporal position 准确反映了信息的时间戳。

#### 状态 3：recall 帧回来 + think 在（重新对齐）

```
think_5 (t=10-12s):    temporal_pos = 10
recalled frames (t=10-14s): temporal_pos = 10-14   ← 重新对齐!
```

**这是 temporal 对齐最有价值的场景。** Recall 检索回来的旧帧和当时的 think 在 temporal 维度上重新匹配，模型可以轻松关联"这段旧文本描述的就是这些 recall 回来的帧"。

### 8.3 Temporal 层次结构与记忆衰减

```
temporal position 轴:
0────────────16──────────32────────40─────────────64
│             │           │         │              │
compressed_0  compressed_1 older    ←─ overlap ─→  current
[0-16s摘要]  [16-32s摘要]  thinks   thinks+frames  thinks+frames
                          (无帧)    (有帧,对齐)     (有帧,对齐)

←──── 过去 (信息逐级压缩) ────→←── 现在 (高保真) ──→
```

模型从 temporal position 就能理解：
- **低 temporal position** = 远古信息，只有压缩摘要，精度低
- **中 temporal position** = 较旧信息，有原始 think 但无帧，精度中
- **高 temporal position** = 当前信息，有 think + 帧，精度高

这和人类记忆衰减规律一致：近期记忆清晰（图像+文字），远期记忆模糊（只剩概要）。

### 8.4 模态区分：h/w 维度防止混淆

Think_25 和 video_chunk_25 的 temporal position 都是 50，但不会混淆，因为 height/width 维度天然区分了模态：

```
video chunk_25 的帧:  temporal=50, height=0/1/2.., width=0/1/2..   (2D 空间网格)
think_25 文本:        temporal=50, height=1700,     width=1700      (递增序号，远离视频空间区间)
```

三个维度中只有 temporal 对齐，h/w 维度的数值差异确保模型不会把文本 token 误认为视频 token。

### 8.5 全场景正确性汇总

| 场景 | temporal position 行为 | 正确性 | 作用 |
|------|----------------------|--------|------|
| 帧在 + think 在 | 对齐（相同 temporal） | ✅ | cross-modal attention 增强 |
| 帧已 evict + think 在 | think 保留原始时间戳，比当前帧时间小 | ✅ | 正确表达"这是旧信息"，RoPE 衰减合理 |
| 帧已 evict + 压缩摘要 | 摘要用覆盖范围的起始时间 | ✅ | 正确表达"这是更早的摘要" |
| recall 帧回来 + think 在 | 重新对齐 | ✅ | 老帧和老 think 匹配，recall 最有价值场景 |
| 压缩后新摘要 | 摘要 temporal = 被压缩 thinks 的最早时间 | ✅ | 摘要在时间轴上替代了被删除的 thinks 位置 |

---

## 附录 A：Position ID 示例

### A.1 当前布局（文本在前）

```
Token:    [sys...] [<memory>think_5 think_6...</memory>] [<vw>V V V V V V</vw>] [Q]
Seq pos:  0...149  150.........................500         501........2036         2037
MROPE-t:  0...149  150.........................500         501=0*Δt...2036=23*Δt  2037
MROPE-h:  同上                                            501,502,501,502...      同上
MROPE-w:  同上                                            501,501,502,502...      同上
```

think_5(t=10-12s) 的 temporal pos ≈ 200
视频 chunk_5(t=10-12s) 的 temporal pos ≈ 600
**temporal 距离 = 400** → RoPE 衰减显著

### A.2 新布局（视频在前 + 时间对齐）

```
Token:    [sys...] [<vw>V V V V V V</vw>] [<memory>think_5 think_6...</memory>] [Q]
Seq pos:  0...149  150........1686         1687.........................2037       2038
MROPE-t:  0...149  150=0*Δt...1686=23*Δt  1687=T5_time...2037=TN_time  2038
                               ↑ chunk_5 的帧 temporal≈300               ↑ think_5 temporal=300(对齐!)
MROPE-h:  同上     150,151,150,151...       1687,1688,...               同上
MROPE-w:  同上     150,150,151,151...       1687,1688,...               同上
```

think_5(t=10-12s) 的 temporal pos = 300 (对齐到帧)
视频 chunk_5(t=10-12s) 的 temporal pos = 300
**temporal 距离 = 0** → RoPE 完美匹配

---

## 附录 B：参考文献

1. **Qwen2-VL** (Wang et al., 2024) - M-RoPE: Multimodal Rotary Position Embedding. arXiv:2409.12191
2. **VRoPE** (2025) - Rotary Position Embedding for Video Large Language Models. arXiv:2502.11664
3. **Interleaved-MRoPE** (Qwen3-VL) - Position IDs grow slower than vanilla RoPE.
4. **Revisiting Multimodal Positional Encoding** (2025) - Cross-modal position continuity. arXiv:2510.23095
