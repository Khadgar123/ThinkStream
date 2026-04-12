# ThinkStream 代码阅读指南

本文档面向希望深入理解 ThinkStream 代码细节的读者。按照"从入口到底层"的顺序，逐层解释每个模块的设计意图、关键抽象和核心逻辑，并给出具体的函数/行号引用，方便对照源码阅读。

---

## 目录

1. [项目依赖与整体架构](#1-项目依赖与整体架构)
2. [理解训练框架：slyme / deepslyme](#2-理解训练框架slyme--deepslyme)
3. [代码入口：train.py](#3-代码入口trainpy)
4. [配置层：trainer/scope.py](#4-配置层trainerrscopepy)
5. [图构建层：trainer/builder.py](#5-图构建层trainerbuildepy)
6. [SFT 节点层：trainer/sft.py](#6-sft-节点层trainersftpy)
7. [RL 节点层：trainer/grpo.py](#7-rl-节点层trainergrpopy)
8. [数据层：data/stream_data_processor.py](#8-数据层datastream_data_processorpy)
9. [位置编码：data/rope2d.py](#9-位置编码datarope2dpy)
10. [模型注册：model/__init__.py](#10-模型注册model__init__py)
11. [注意力机制：model/streaming_attention.py](#11-注意力机制modelstreaming_attentionpy)
12. [模型补丁：model/patch.py](#12-模型补丁modelpatchpy)
13. [推理引擎：model/inference.py](#13-推理引擎modelinferencepy)
14. [完整调用链追踪：SFT 一次前向](#14-完整调用链追踪sft-一次前向)
15. [完整调用链追踪：GRPO 一步迭代](#15-完整调用链追踪grpo-一步迭代)
16. [关键设计模式总结](#16-关键设计模式总结)
17. [调试与扩展指引](#17-调试与扩展指引)

---

## 1. 项目依赖与整体架构

### 1.1 外部框架依赖

ThinkStream 构建在两个非公开框架之上，理解这两个框架是读懂代码的前提：

| 框架 | 安装方式 | 作用 |
|------|---------|------|
| `slyme` | `requirements.txt` 中列出 | 提供 `Context`、`Ref`、`Node`、`@node`、`@wrapper`、`Registry` 等核心抽象 |
| `deepslyme` | `requirements.txt` 中列出 | 基于 slyme 的深度学习专用节点库，提供 DeepSpeed 集成、DataLoader、优化器等 |

这两个库实现了一种**计算图（Node Graph）训练范式**：训练流程不是传统的命令式代码，而是一个由节点（Node）构成的有向图，每个节点声明自己的输入依赖（`Auto[T]`）和输出（`Ref[T]`），框架自动完成依赖注入和执行。

### 1.2 整体模块依赖图

```
scripts/sft.sh  ──► thinkstream/train.py
scripts/rl.sh   ──►      │
                         ▼
                  trainer/builder.py  (注册并构建 Node 图)
                  ├─────────────────────────────────────────┐
                  │                                         │
              build_hf_deepspeed_train()             build_grpo_train()
                  │                                         │
        trainer/scope.py (Ref 映射)            trainer/scope.py (grpo_scope)
                  │                                         │
        trainer/sft.py   (SFT Nodes)          trainer/grpo.py (GRPO Nodes)
                  │                                         │
                  └──────────────┬──────────────────────────┘
                                 ▼
                   data/stream_data_processor.py
                   ├── _build_messages()          ← SFT 数据构建
                   ├── process_messages_to_model_inputs()  ← 共用
                   ├── compute_position_ids()     ← 共用
                   ├── LazySupervisedDataset      ← SFT
                   └── LazyRawDataset             ← GRPO
                                 │
                                 ▼
                   data/rope2d.py  (MROPE 位置编码)
                   data/__init__.py (路径注册表)
                                 │
                                 ▼
                   model/__init__.py  (MODEL_CLS 注册)
                   model/patch.py     (模型 forward 补丁)
                   model/streaming_attention.py  (Flex Attention)
                   model/inference.py (StreamingInferenceEngine)
```

### 1.3 阅读顺序建议

**第一次阅读（建立全局认知）：** `train.py` → `trainer/builder.py` → `trainer/scope.py`

**第二次阅读（理解数据流）：** `data/__init__.py` → `data/stream_data_processor.py` → `data/rope2d.py`

**第三次阅读（理解模型改造）：** `model/__init__.py` → `model/streaming_attention.py` → `model/patch.py`

**第四次阅读（理解 RL）：** `trainer/grpo.py` → `model/inference.py`

---

## 2. 理解训练框架：slyme / deepslyme

在读任何具体代码前，必须先理解以下四个概念：

### 2.1 `Context`（训练状态容器）

`Context` 是一个不可变的键值存储（类似于函数式编程中的状态 monad）。训练过程中所有的共享状态（模型、优化器、数据集、步数、损失等）都存在 `Context` 中。

```python
# 每个 @node 函数接受 ctx: Context，返回更新后的 ctx
@node
def load_model(ctx: Context, /, *, model_name_or_path: Auto[str], model: Ref[PreTrainedModel]) -> Context:
    lmm = ...load...
    return ctx.update({model: lmm})   # 把 model 写入 ctx
```

### 2.2 `Ref`（惰性引用）

`Ref` 是一个带名字的"指针"，指向 `Context` 中的某个键。它支持 `key_path` 深层访问（用 `P.attr` 语法）。

```python
# scope.py 中
"model": Ref("model"),                          # ctx["model"]
"hidden_size": Ref("model", key_path=tuple(P.config.text_config.hidden_size)),  
# 相当于 ctx["model"].config.text_config.hidden_size
```

### 2.3 `Auto[T]` vs `Ref[T]`

在 `@node` 函数的参数列表中：
- `Auto[T]`：**读取**（输入），框架从 `scope` 中查找对应的 `Ref`，再从 `Context` 中解析值
- `Ref[T]`：**写入**（输出），用于 `ctx.set(ref, value)` 或 `ctx.update({ref: value})`

```python
@node
def compute_loss(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],    # 读取 model
    step_inputs: Auto[dict],          # 读取 step_inputs
    step_loss: Ref[torch.Tensor],    # 写入 step_loss
) -> Context:
    loss = model(**step_inputs).loss
    return ctx.set(step_loss, loss)
```

### 2.4 `@node`、`@wrapper`、`@expression` 装饰器

| 装饰器 | 返回类型 | 用途 |
|--------|---------|------|
| `@node` | `Context` | 有副作用的训练步骤（加载模型、前向、更新权重等） |
| `@wrapper` | `Context`（包装其他节点） | 条件执行、计时、梯度边界等 |
| `@expression` | `dict`（指标字典） | 纯计算，返回指标值供 `collect_metrics` 收集 |

```python
@wrapper  
def check_should_save(ctx, wrapped, call_next, /, *, save_steps, state_global_step) -> Context:
    if state_global_step % save_steps == 0:
        return call_next(ctx)   # 调用被包裹的节点
    return ctx                  # 跳过
```

---

## 3. 代码入口：train.py

**文件：** `thinkstream/train.py`

```
thinkstream/train.py  [1-46行]
```

入口逻辑极简：

```
sys.argv[1]  →  builder_name  ("sft" 或 "grpo")
     │
     ▼
TRAINER_BUILDERS.get(builder_name)  →  builder_fn
     │
     ▼
builder_fn()  →  train_node  (一个完整的 Node 图)
     │
     ▼
Context()  +  parse_and_inject(...)  →  ctx（注入 CLI 参数）
     │
     ▼
train_node.prepare()(ctx)  →  开始执行训练图
```

**关键：** `register_streaming_attention()` 在 `train.py` 最顶部被调用（第 12 行），这会在 `transformers` 的 `AttentionInterface` 中注册一个名为 `"streaming_attention"` 的自定义注意力实现，这样之后加载模型时 `attn_implementation="streaming_attention"` 才能生效。

---

## 4. 配置层：trainer/scope.py

**文件：** `thinkstream/trainer/scope.py`

scope 是一个 Python `dict`，把"语义名称"（如 `"learning_rate"`）映射到 `Ref`（如 `Ref("args.train.learning_rate", ...)`）。

**读法：** 每个条目的格式为：
```python
"语义名称": Ref[类型]("ctx中的键路径", metadata={ARG: Arg(default=默认值, required=是否必需)})
```

`ARG: Arg(...)` 元数据告诉 `parse_and_inject` 这个 `Ref` 对应的 CLI 参数。例如：
```python
"learning_rate": Ref[float]("args.train.learning_rate", metadata={ARG: Arg(default=2e-7)})
```
→ CLI 参数 `--args.train.learning_rate 1e-5`，默认值 `2e-7`，写入 `ctx["args"]["train"]["learning_rate"]`。

`grpo_scope()` 继承 `default_scope()` 并追加 GRPO 专用配置（`group_size`、`rollout_*` 等）。

**主要分组：**

| 前缀 | 说明 | 来源 |
|------|------|------|
| `args.model.*` | 模型路径、类型、序列长度 | CLI |
| `args.train.*` | 学习率、batch size、epoch 等 | CLI |
| `args.data.*` | 数据集名、像素范围、fps | CLI |
| `state.train.*` | 训练状态（step、epoch、日志） | 运行时写入 |
| `distributed.*` | 分布式环境（进程号、设备） | deepspeed 初始化写入 |
| `step.*` | 单步数据（inputs、loss、advantages） | 每步更新 |

---

## 5. 图构建层：trainer/builder.py

**文件：** `thinkstream/trainer/builder.py`

这是"把所有节点串联成一个可执行图"的地方。

### 5.1 SFT 图结构（`build_hf_deepspeed_train`，第 89-172 行）

```
初始化阶段（顺序执行）：
  deepspeed_init_distributed       ← 初始化 NCCL / 进程组
  set_seed                         ← 固定随机数种子
  deepspeed_config_init            ← 解析 zero3.json，填写 auto 字段
  setup_dtype                      ← 设置 bf16/fp16
  build_optimizer_kwargs           ← 构造 AdamW 参数
  load_model                       ← 加载 Qwen2.5-VL（含 ZeRO-3 上下文包裹）
  update_deepspeed_config_by_hidden_size  ← 按 hidden_size 填 bucket 大小
  configure_model_gradients        ← 冻结 ViT 主体，只训 merger+LLM
  init_processor                   ← 加载 tokenizer，添加特殊 token
  init_dataset                     ← 构建 LazySupervisedDataset
  apply_liger_kernel               ← 融合 CE Loss（速度优化）
  align_special_tokens             ← 同步 model.config 的 pad/bos token_id
  prepare_distributed_dataloader   ← 创建分布式 DataLoader
  init_training_state              ← 初始化 global_step、epoch 等状态
  create_optimizer / create_scheduler  ← AdamW + 余弦调度
  set_gradient_checkpointing       ← 激活梯度检查点
  deepspeed_initialize             ← 用 deepspeed.initialize() 包装 model/optimizer

训练循环（epoch_loop → dataloader_loop）：
  每个 mini_step（对应梯度累积中的一个 micro batch）：
    deepspeed_set_grad_acc_boundary   ← 判断是否到了同步点
    prepare_inputs                    ← 把 batch 数据移到 GPU
    compute_loss                      ← 模型前向 + 加权 CE loss
    collect_metrics (sft_mini_metrics) ← 记录 loss
    clean_step_inputs                 ← 释放 batch tensor
    empty_cache                       ← 定期清理 GPU 缓存
    deepspeed_backward                ← loss.backward()
  每个 global_step（梯度累积完成后）：
    deepspeed_step                    ← optimizer.step() + lr_scheduler.step()
    collect_metrics (sft_global_metrics) ← 记录 grad_norm、lr
    reduce_and_log_metrics            ← all_reduce 指标，打印日志
    update_progress                   ← 更新 tqdm / wandb
    hf_deepspeed_save_model [被 check_should_save 包裹]  ← 按步保存

结束：
  hf_deepspeed_save_model           ← 训练结束强制保存一次
  destroy_progress                  ← 关闭 tqdm
```

### 5.2 GRPO 图结构（`build_grpo_train`，第 175-281 行）

相比 SFT 的主要差异在 `dataloader_loop_with_micro_steps` 中的节点：

```
每个 mini_step：
  [rollout + unwrap_model_for_generation 包裹]
    unwrap_model_for_generation  ← 暂时将 DeepSpeed engine 解包为原始 model
    rollout                      ← streaming_video_chat 生成 G 个候选回答
  calc_rewards                   ← 计算 4 维奖励分数
  calc_grpo_advantages           ← 组内相对优势 (reward - mean) / std
  prepare_grpo_micro_batches     ← 把 B×G 个生成结果切成 micro batches

每个 micro_step（对单个 micro batch）：
  build_grpo_inputs              ← 用生成的 token 重建 input_ids + labels
  prepare_inputs                 ← 移到 GPU
  [compute_grpo_loss + timer 包裹]
    compute_grpo_loss            ← GRPO loss（via LigerFusedLinearGRPOLoss）
  clean_step_inputs
  empty_cache
  [deepspeed_backward + timer 包裹]
    deepspeed_backward           ← loss.backward()
  collect_metrics (grpo_micro_metrics)
```

---

## 6. SFT 节点层：trainer/sft.py

**文件：** `thinkstream/trainer/sft.py`

### 6.1 `load_model`（第 246-273 行）

```python
attn_implementation = "streaming_attention"      # 使用自定义 Flex Attention
vision_attn_implementation = "flash_attention_2" # ViT 仍用 flash attention
lmm = MODEL_CLS[model_type].from_pretrained(...)
```

**注意：** 整个 `load_model` 节点被 `with_hf_deepspeed_context` wrapper 包裹（builder.py 第 109 行），这是为了在 ZeRO-3 模式下安全地加载模型——`HfDeepSpeedConfig` 需要在 `from_pretrained` 之前生效。

### 6.2 `configure_model_gradients`（第 276-299 行）

梯度策略：

```
ViT (model.model.visual)：        requires_grad = False   ← 冻结视觉编码器
ViT Merger (visual.merger)：      requires_grad = True    ← 训练合并层
LLM (model.model.language_model)：requires_grad = True    ← 训练语言模型
lm_head：                         requires_grad = True    ← 训练输出层
```

这体现了 ThinkStream 的设计选择：视觉特征提取是固定的，只训练"如何把视觉特征与时序推理结合"的部分。

### 6.3 `init_processor`（第 302-318 行）

```python
lmm_processor.tokenizer.add_tokens(["<silent>", "<response>", "<think>", "</think>"])
```

ThinkStream 的特殊 token 需要在原始 tokenizer 基础上追加。`Qwen3VL` 的 tokenizer 已经内置 `<think>/<think>`（因为 Qwen3 本身支持 CoT），所以只追加 `<silent>` 和 `<response>`。

### 6.4 `hf_deepspeed_save_model`（第 136-186 行）

ZeRO-3 下模型参数被分片到各 GPU，直接 `model.state_dict()` 只能得到本进程的分片。这里用：
```python
state_dict = model_for_training._zero3_consolidated_16bit_state_dict()
```
手动合并所有分片，然后只在 `process_index == 0` 的进程上写磁盘，避免写冲突。

---

## 7. RL 节点层：trainer/grpo.py

**文件：** `thinkstream/trainer/grpo.py`

这是项目最复杂的文件（1076 行），分为几个逻辑区块：

### 7.1 奖励函数（第 40-197 行）

**格式奖励 `_compute_format_reward`（第 52-57 行）：**

正则匹配单个 chunk 的生成文本：
```python
_CHUNK_FORMAT_RE = re.compile(
    r"^<think>.*?</think>(?:<response>.*|<silent>)<\|im_end\|>$",
    re.DOTALL,
)
```
奖励 = 格式正确的 chunk 数 / 总 chunk 数。

**时间奖励 `_compute_time_reward`（第 155-168 行）：**

线性衰减窗口：
```
|response_chunk - gt_chunk| <= slack           → 1.0
|response_chunk - gt_chunk| <= slack + window  → 线性衰减到 0
                                              > → 0.0
```

**正确性奖励 `_compute_correctness_reward`（第 171-177 行）：**

`_extract_literal_answer()` 解析答案格式（A/B/C/D/E、yes/no、0-9），然后精确匹配。

**回应效率奖励 `_compute_num_response_reward`（第 180-197 行）：**

阶梯式：恰好 1 次回应 = 1.0，多次回应按 `step_window=3` 阶梯衰减，超过 `max_responses=10` = 0.0。

**思维长度因子 `_compute_think_length_factor`（第 91-109 行）：**

离散阶梯奖励：思维链长度越接近目标 `rollout_max_think_tokens`，奖励越高。与 `num_response_reward` 相乘得到 `response_efficiency_reward`。

### 7.2 `rollout` 节点（第 ~380-480 行）

这是 GRPO 的核心——对每条原始数据生成 G 个候选回答：

```python
# 1. 暂时切换到推理模式（由 unwrap_model_for_generation wrapper 完成）
# 2. 调用流式推理引擎
for result in streaming_video_chat(
    model, processor,
    queries=[raw_sample],         # 1条原始数据
    num_generations=group_size,   # G=8 个候选
    ...
):
    chunk_results.append(result)
# 3. 返回格式：[{chunk_idx, window_start, window_end, generated_tokens: List[G]}]
```

`streaming_video_chat` 来自 `model/inference.py`，是真正执行"一边看视频一边生成"的函数。

### 7.3 `calc_rewards` 节点（第 492-598 行）

遍历 `rollout_data`（B 条样本），对每条样本的 G 个候选分别计算四维奖励，最终拼成 shape `[B×G]` 的 tensor。

关键：`gt_msg = conversations[1]` 取第二条消息作为 GT（RL 数据固定格式：`conversations[0]` 是 user，`conversations[1]` 是 assistant GT）。

### 7.4 `build_grpo_inputs` 节点（第 ~600-800 行）

把 rollout 产生的生成 tokens 与原始问题重新拼成完整的 `input_ids`，同时：
- 用 `find_assistant_spans` 构建 `completion_mask`（只有生成部分参与 loss）
- 计算 `old_per_token_logps`（rollout 时的对数概率，用于重要性采样）
- 调用 `process_messages_to_model_inputs` 重建所有模型输入（像素、位置编码等）

### 7.5 `compute_grpo_loss` 节点

调用 `model.forward(..., advantages=advantages, old_per_token_logps=..., completion_mask=...)`，触发 `model/patch.py` 中注册的 `grpo_lce_forward_qwen2_5_vl`，内部使用 `LigerFusedLinearGRPOLoss` 计算 GRPO 目标：

```
L_GRPO = -E[A_i * log(π_θ(a|s))] + β * KL(π_θ || π_ref)
```

---

## 8. 数据层：data/stream_data_processor.py

**文件：** `thinkstream/data/stream_data_processor.py`（1244 行）

这是项目最重要的文件之一，SFT 和 GRPO 的数据处理逻辑都在这里。

### 8.1 视频加载的统一入口：`load_video_frames`（第 53-183 行）

所有视频帧的加载都经过这个函数，它使用"ghost message"模式：

```python
ghost_message = [{"role": "user", "content": [
    {"type": "video", "video": path, "video_start": t0, "video_end": t1, "nframes": N}
]}]
_, video_inputs_list, video_kwargs = process_vision_info(ghost_message, ...)
big_tensor = video_inputs_list[0]          # 所有帧合并成一个大 tensor
split_videos = torch.split(big_tensor, frames_per_chunk)  # 按 chunk 切分
```

**为什么用 ghost message？** `process_vision_info`（来自 `qwen_vl_utils`）是 Qwen-VL 官方提供的视频帧提取工具，它需要消息格式的输入。用 ghost message 可以复用官方工具，无需自己实现 resize/pad 逻辑。

### 8.2 系统提示：`SYSTEM_PROMPT`（第 431-438 行）

```
"You are a helpful assistant. You will see a continuous stream of video chunks.
Based on the user's query and the video content, first output your internal reasoning
enclosed in <think>...</think> tags. Then, if you determine that a response is needed
at this moment, output <response> followed by the content. If no response is needed,
output <silent>. Your generated thoughts and responses should be continuous and fluent
across the video chunks."
```

这段 prompt 是模型行为的"程序规范"，训练和推理时完全一致。

### 8.3 SFT 消息构建：`_build_messages`（第 468-606 行）

**输入：** 一条 JSONL 样本（含 `video_path`、`conversations`、`thoughts`）

**输出：** `{messages, video_meta, video_chunk_size}`

**算法流程：**

```
1. 计算 video_chunk_size = ceil(duration / 120)
2. 三个排好序的队列：
   - user_queue:     [(timestamp, content), ...]  按时间排序的用户问题
   - assistant_queue: [(timestamp, content), ...] 按时间排序的 GT 回答
   - thoughts_queue:  [(timestamp, think), ...]   按时间排序的思维链
3. 确定截止 chunk（last_assist_ts 所在 chunk + remaining_video_chunks=3）
4. 对每个 chunk_idx [0, target_stop_idx]：
   a. 构建 user 消息：video 片段 + 落在该窗口内的用户问题
   b. 从 thoughts_queue 取该窗口内的 think 文本，拼成 <think>...</think>
   c. 从 assistant_queue 取该窗口内的回答，决定是 <response>文本 还是 <silent>
   d. 拼接：<think>...</think><response>内容  或  <think>...</think><silent>
```

**关键边界处理：**
- `is_last_chunk` 时，不做窗口截断（防止最后一帧丢失数据）
- `video_end <= video_start` 时，自动扩展一个微小窗口（第 113-114 行）

### 8.4 `process_messages_to_model_inputs`（第 609-706 行）

SFT、GRPO 重建输入、推理三条路径共用的函数，返回字典包含：

| 键 | 类型 | 含义 |
|----|------|------|
| `input_ids` | `[1, L]` | token id 序列 |
| `pixel_values_videos` | `Tensor` | 所有帧的像素值 |
| `video_grid_thw` | `[num_chunks, 3]` | 每 chunk 的 T×H×W 网格 |
| `video_mask` | `[1, L]` | True 处为视频 token（用于 Flex Attention） |
| `video_chunk_size` | `float` | 传递给 RoPE 计算 |

`video_mask` 的生成逻辑（第 699-701 行）：
```python
video_token_id = processor.tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
full_result["video_mask"] = input_ids == video_token_id
```
即：所有 `<|video_pad|>` token 位置为 `True`，其余（文本）为 `False`。

### 8.5 `find_assistant_spans`（第 714-745 行）

在扁平化的 token 序列中定位所有 assistant 轮次：

```
扫描 input_ids_1d，找到 assistant token
     ↓
assistant 内容从 pos+2 开始（跳过 "assistant\n" 两个 token）
     ↓
扫到 <|im_end|> 为止（+2，包含后面的 "\n"）
```

这个函数用于：
- SFT：构建 `labels`（只有 assistant span 的 labels 不是 -100）
- GRPO：构建 `completion_mask`（只有生成部分参与 GRPO loss）

### 8.6 `LazySupervisedDataset.__getitem__`（第 935-996 行）

**三级容错机制：**

```
第1级：重试当前样本最多3次（等待1秒，应对临时 I/O 故障）
     ↓失败
第2级：随机替换为其他样本重试最多30次（绕过坏样本）
     ↓失败
第3级：最后再尝试原样本一次，失败则抛出异常
```

### 8.7 `DataCollatorForSupervisedDataset`（第 1013-1100 行）

**加权 CE Loss（第 1088-1098 行）：**

```python
response_id = tokenizer.convert_tokens_to_ids(["<response>"])[0]
silent_id   = tokenizer.convert_tokens_to_ids(["<silent>"])[0]
n_response = (batch["labels"] == response_id).sum()
n_silent   = (batch["labels"] == silent_id).sum()
total_n = n_response + n_silent

ce_weight[silent_id]   = total_n / (2 * n_silent + eps)    # 类频率倒数
ce_weight[response_id] = total_n / (2 * n_response + eps)  # 类频率倒数
ce_weight = torch.clamp(ce_weight, 0, 20)  # 上界防止极端值
```

**为什么需要这个权重？** 流式视频中，绝大多数 chunk 模型应输出 `<silent>`（不需要回答），只有少数 chunk 需要 `<response>`。若不加权，模型会因为 `<silent>` 的频率更高而过拟合到"总是沉默"。这个平衡权重让模型同等程度地学习"何时沉默"和"何时回应"。

---

## 9. 位置编码：data/rope2d.py

**文件：** `thinkstream/data/rope2d.py`

实现了 **MROPE（Multimodal Rotary Position Embedding）**，对 Qwen2.5-VL 和 Qwen3-VL 分别实现了 `get_rope_index_25` 和 `get_rope_index_3`。

### 9.1 为什么需要 MROPE？

标准 RoPE 只有一维位置编码（序列位置）。对于多模态视频，我们需要：
- **时间维度（T）：** 当前帧在视频时间轴上的位置
- **高度维度（H）：** 像素 patch 在帧中的纵向位置
- **宽度维度（W）：** 像素 patch 在帧中的横向位置

MROPE 输出 `position_ids` 的 shape 为 `(3, B, L)`，三个维度分别对应 T/H/W。

### 9.2 关键参数：`second_per_grid_ts`

在 `compute_position_ids`（`stream_data_processor.py` 第 787-791 行）：

```python
second_per_grid_ts = [
    video_chunk_size * processor.video_processor.temporal_patch_size / processor.video_processor.fps
] * len(video_grid_thw)
```

**含义：** 每个时间网格格子（temporal grid cell）对应的真实时间（秒）。这让 RoPE 的时间维度编码反映真实的视频时序，而不仅仅是 token 的排列顺序。

对于 Qwen2.5-VL（`get_rope_index_25` 第 305-311 行）：
```python
time_tensor = expanded_range * second_per_grid_t * 2
```
时间坐标 = patch 索引 × 每格秒数 × 2（×2 是 Qwen 的超参）。

### 9.3 `ROPE_INDEX_FN` 注册表（第 376-379 行）

```python
ROPE_INDEX_FN: Dict[str, Callable] = {
    "qwen2.5vl": get_rope_index_25,
    "qwen3vl":   get_rope_index_3,
}
```

所有使用 RoPE 的地方（Dataset、GRPO 重建输入、推理）都通过这个注册表获取对应函数，确保一致性。

---

## 10. 模型注册：model/\_\_init\_\_.py

**文件：** `thinkstream/model/__init__.py`

```python
MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl":   Qwen3VLForConditionalGeneration,
}
DEFAULT_VIDEO_FLEX_WINDOW_SIZE = 20
```

`MODEL_CLS` 是模型类型到 HuggingFace 模型类的映射。通过 `model_type` 参数（CLI 传入）选择对应的类。

`DEFAULT_VIDEO_FLEX_WINDOW_SIZE = 20` 是滑动窗口注意力的默认窗口大小（20个视频块），可通过 `model.config.video_flex_window_size` 覆盖。

---

## 11. 注意力机制：model/streaming_attention.py

**文件：** `thinkstream/model/streaming_attention.py`

### 11.1 整体设计

ThinkStream 对 LLM 的注意力做了三层修改：

```
标准因果注意力
    +
padding mask（attention_mask）
    +
视频 token 滑动窗口（video sliding window）
    =
streaming_attention（Flex Attention 实现）
```

ViT 部分保留 `flash_attention_2`，不做修改。

### 11.2 `generate_video_sliding_window_mask_mod`（第 10-65 行）

这是核心算法，生成一个 `mask_mod` 函数，被 PyTorch `flex_attention` 编译成 CUDA kernel。

**`mask_mod(b, h, q_idx, kv_idx) → bool`**

返回 True 表示 query 位置 `q_idx` 可以 attend to key 位置 `kv_idx`。规则如下：

```python
# 1. 有效性：必须在 attention_mask 范围内
is_valid_pair = in_bounds & q_is_valid & k_is_valid

# 2. 因果性：只能看到过去的 token
is_causal = (q_idx >= kv_idx)

# 3. 视频滑动窗口：视频 token 只能看到最近 window_size 个块的视频 token
#    文本 token 不受限制（~k_is_video 为 True 时 is_in_window 直接为 True）
diff = q_block - k_block   # block_id 差值（通过 cumsum 预计算）
is_in_window = (~k_is_video) | (diff < window_size_n)

return is_valid_pair & is_causal & is_in_window
```

**为什么视频 token 需要滑动窗口？** 视频流可能非常长（120 chunks × 每 chunk 数百个 token = 数万 token），如果所有视频 token 都相互 attend，内存开销是 O(L²)。滑动窗口将视频部分的注意力限制在"当前块 + 历史 N 块"，降低为接近 O(N×L)，同时保留足够的时序上下文。

### 11.3 Block ID 的计算（第 30-35 行）

```python
shifted = F.pad(video_mask[:, :-1], (1, 0), value=False)  # 左移一位
block_starts = video_mask & (~shifted)                      # 视频块的起始位置
block_ids = block_starts.long().cumsum(dim=-1)             # 前缀和 = 块编号
```

这是一个"把连续的 True 片段赋予同一个 ID"的技巧：
- `block_starts[i]` = True 当且仅当位置 i 是视频块的第一个 token
- 前缀累加后，同一个视频块内所有 token 有相同的 block_id

### 11.4 `register_streaming_attention`（第 145-148 行）

```python
AttentionInterface.register("streaming_attention", flex_attention_forward)
```

这行代码在 `transformers` 的全局注意力接口注册表中注册了 `"streaming_attention"` 实现。之后加载模型时，`attn_implementation="streaming_attention"` 就会使用 `flex_attention_forward` 函数替代默认的 sdpa/flash attention。

### 11.5 单例编译优化：`_BaseCompiledSingleton`（第 91-118 行）

`WrappedFlexAttention` 是 `flex_attention` 的编译单例，使用 `torch.compile` 只编译一次，之后复用同一个编译好的函数对象。避免每次前向都重新触发编译。

---

## 12. 模型补丁：model/patch.py

**文件：** `thinkstream/model/patch.py`

这个文件做了两件事：

### 12.1 SFT LCE Forward 补丁（第 62-99 行）

使用 `liger_kernel` 的 fused cross-entropy（LCE）替换标准的两步（logits → CE loss）：

```python
# 原始 liger lce_forward 不接受 video_block_mask，需要 patch
def _LigerForCausalLMLoss(*args, **kwargs):
    kwargs.pop("video_block_mask")  # 去掉 flex attention 的额外参数
    return LigerForCausalLMLoss(*args, **kwargs)
```

然后在 `_lce_forward_qwen2_5_vl` 中：
1. 调用 `build_video_block_mask` 从 `video_mask` 构建 Flex Attention 的 `BlockMask`
2. 把 `video_block_mask` 传给 backbone（`self.model(...)`）

### 12.2 GRPO LCE Forward（第 186-468 行）

`_grpo_lce_forward_common` 是 GRPO 训练时的前向函数，当传入 `advantages` 参数时，走 GRPO loss 路径：

```python
if advantages is not None:
    # 1. 移位（把 hidden_states 和 labels 对齐）
    shifted_hs = hidden_states[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    
    # 2. Pack：只保留 completion_mask==1 的位置（去掉 prompt 和 padding 中间的 gap）
    packed = _pack_by_completion_mask(shifted_hs, shifted_labels, ...)
    
    # 3. 调用 LigerFusedLinearGRPOLoss
    grpo_loss_fn = LigerFusedLinearGRPOLoss(beta=grpo_beta, ...)
    loss = grpo_loss_fn(_input=packed_hs, lin_weight=self.lm_head.weight, ...)
```

**`_pack_by_completion_mask`（第 118-183 行）** 是一个关键的优化：GRPO 的序列中只有 assistant 生成部分（`completion_mask==1`）参与 loss，但序列中间可能有大量视频 token 片段（`completion_mask==0`）隔断。Pack 操作把每条序列中的有效位置紧凑排列，避免 LigerFusedLinearGRPOLoss 处理大量无效位置。

---

## 13. 推理引擎：model/inference.py

**文件：** `thinkstream/model/inference.py`（约 1770 行）

这是 GRPO rollout 和交互式推理共用的引擎，包含多个精心设计的组件。

### 13.1 `StreamingCache`（第 90-201 行）

静态分配的 KV Cache，预先分配最大长度的 tensor：

```python
self.k_cache[layer_idx] = torch.zeros(
    (batch_size, num_key_value_heads, max_len, head_dim)
)
self.cache_seqlens = torch.zeros((num_hidden_layers, batch_size), dtype=torch.int32)
```

`cache_seqlens[layer_idx, batch]` 记录第 `batch` 条样本在第 `layer_idx` 层已填充的 KV 数量，支持**逐层异步更新**（CUDA Graph 需要固定大小的内存地址）。

更新方法 `update`（第 154-192 行）区分两种路径：
- **Prefill（seq_len > 1）：** 使用 scatter 批量填充
- **Decode（seq_len == 1）：** 单步填充，被 CUDA Graph 捕获加速

### 13.2 `GraphDecoder`（第 207-337 行）

CUDA Graph 加速的解码器：

```python
# capture 阶段（只执行一次）
with torch.cuda.graph(self.graph):
    logits = model(input_ids=self.static_input_ids, ...).logits
    self.static_logits.copy_(logits)

# step 阶段（每个 decode step）
self.static_input_ids.copy_(input_ids)    # in-place 覆盖 static buffer
self.graph.replay()                        # 重放捕获的 kernel 序列
return self.static_logits
```

CUDA Graph 的原理：第一次 `capture` 时记录 GPU 指令序列（kernel launch），之后每次 `replay` 直接重新执行这些指令，跳过 CPU 调度开销，decode 速度大幅提升（对 1-token decode 效果最显著）。

### 13.3 `StreamingWindowInferenceEngine`

提供流式视频推理的上层接口，管理 prefill（视频帧 + 上下文）和 decode（生成 token）的完整流程。

### 13.4 `streaming_video_chat`

GRPO rollout 的直接调用函数，返回一个 **generator**：

```python
for result in streaming_video_chat(model, processor, queries, num_generations=8, ...):
    # result = {chunk_idx, window_start, window_end, generated_tokens: [G]}
    chunk_results.append(result)
```

每次 `yield` 对应一个视频 chunk 处理完毕，包含该 chunk 下 G 个生成序列的 token。

### 13.5 `think_budget_sample`

替代标准 top-k/top-p 采样的特殊采样策略，在 `<think>` 内部用 `rollout_max_think_tokens` 控制最大生成长度，在 `<response>/<silent>` 决策时用普通采样。

---

## 14. 完整调用链追踪：SFT 一次前向

从 DataLoader 取出一条数据到 `loss.backward()` 的完整路径：

```
DataLoader.__getitem__(i)
  └─► LazySupervisedDataset._get_item(sources)
        └─► preprocess_qwen_visual(sources, processor, model_type)
              ├─► _build_messages(item, base_path)
              │     ├─► _get_duration()                    ← 读视频元数据
              │     ├─► 构建 user/assistant/thoughts 三个队列
              │     └─► 遍历 chunk，组装 messages 列表
              └─► process_messages_to_model_inputs(messages, video_meta, ...)
                    ├─► load_video_frames(...)             ← 用 ghost message 加载帧
                    ├─► processor.apply_chat_template(messages)  ← 转为文本
                    ├─► processor(text, videos, ...)       ← tokenize + 提取像素
                    └─► 计算 video_mask
              └─► 计算 labels（find_assistant_spans + 掩码）
              └─► compute_position_ids(...)                ← MROPE
              └─► data_dict["attention_mask"] = [seq_len]  ← 仅序列长度（动态 padding）

DataCollatorForSupervisedDataset.__call__(instances)
  ├─► pad_sequence(input_ids)，pad_sequence(labels)
  ├─► pad_and_cat(position_ids)
  └─► 计算 ce_weight（<response>/<silent> 均衡）

compute_loss(ctx, model, step_inputs, step_loss)  [deepslyme node]
  └─► model(**batch)                             ← 触发 _lce_forward_qwen2_5_vl
        ├─► build_video_block_mask(video_mask)   ← Flex Attention mask
        ├─► self.model(input_ids, pixel_values_videos, video_block_mask, ...)
        └─► LigerForCausalLMLoss(logits, labels, ce_weight=ce_weight)

deepspeed_backward(ctx, step_loss)
  └─► model_for_training.backward(loss)
```

---

## 15. 完整调用链追踪：GRPO 一步迭代

```
DataLoader.__getitem__(i)
  └─► LazyRawDataset.__getitem__(i)
        └─► preload_video(abs_path, ...)          ← 并行预加载视频帧（worker 进程）
              → item["_preloaded_video"] = {split_videos, video_kwargs, ...}

rollout(ctx)  [包裹在 unwrap_model_for_generation 中]
  └─► streaming_video_chat(model, processor, [raw_sample], num_generations=8)
        ├─► 使用预加载的视频帧（从 _preloaded_video 取，避免重复 I/O）
        ├─► 对每个视频 chunk：
        │     ├─► Prefill：把 chunk 帧 + 上下文 token 送入模型
        │     └─► Decode：GraphDecoder.step() 循环，直到 <|im_end|>
        └─► yield {chunk_idx, window_start, window_end, generated_tokens: [8]}
  → all_rollout_results[i] = {raw_sample, chunk_results}

calc_rewards(ctx)
  ├─► 取 gt_msg = conversations[1]
  ├─► 对每条样本的 8 个候选：
  │     ├─► _compute_format_reward(chunk_texts)
  │     ├─► _scan_responses_for_answer(...)  → (answer, chunk_idx, num_responses)
  │     ├─► _compute_time_reward(response_chunk_idx, gt_chunk_idx, window=5)
  │     ├─► _compute_correctness_reward(model_answer, gt_content)
  │     └─► _compute_num_response_reward(num_responses) × _compute_think_length_factor(...)
  └─► rewards: Tensor [B×8]

calc_grpo_advantages(ctx)  [deepslyme 提供]
  └─► advantages = (rewards - mean(rewards)) / (std(rewards) + eps)  # 组内标准化

prepare_grpo_micro_batches
  └─► 把 B×8 分成若干 micro batch

build_grpo_inputs(ctx)  [对每个 micro batch]
  ├─► 从 chunk_results 重建完整 messages（_build_rollout_messages）
  ├─► process_messages_to_model_inputs(...)  ← 重新 tokenize（用 preloaded_video）
  ├─► find_assistant_spans → completion_mask
  └─► 计算 old_per_token_logps（重新跑一次 prefill，得到 rollout 时的 log prob）

compute_grpo_loss(ctx)
  └─► model(**inputs, advantages=adv, old_per_token_logps=...)
        └─► _grpo_lce_forward_common(...)
              ├─► build_video_block_mask
              ├─► backbone forward → hidden_states
              ├─► _pack_by_completion_mask(shifted_hs, ...)  ← 去掉 prompt 间隔
              └─► LigerFusedLinearGRPOLoss(packed_hs, lm_head.weight, advantages, ...)

deepspeed_backward
  └─► model_for_training.backward(loss)
```

---

## 16. 关键设计模式总结

### 16.1 注册表模式（Registry）

项目大量使用注册表将"类型名称"映射到"实现"：

| 注册表 | 文件 | 键示例 |
|--------|------|--------|
| `TRAINER_BUILDERS` | `trainer/builder.py` | `"sft"`, `"grpo"` |
| `MODEL_CLS` | `model/__init__.py` | `"qwen2.5vl"`, `"qwen3vl"` |
| `ROPE_INDEX_FN` | `data/rope2d.py` | `"qwen2.5vl"`, `"qwen3vl"` |
| `GRPO_LCE_FORWARD` | `model/patch.py` | `"qwen2.5vl"`, `"qwen3vl"` |
| `AttentionInterface` | transformers | `"streaming_attention"`, `"flash_attention_2_infer"` |
| `data_dict` | `data/__init__.py` | `"stream_cold_start"`, `"stream_rlvr"` |

**阅读建议：** 当看到 `model_type` 参数时，可以通过这些注册表快速找到对应的实现。

### 16.2 "消费后删除"的 `video_chunk_size`

在 `compute_position_ids`（`stream_data_processor.py` 第 787 行）：
```python
second_per_grid_ts = [processor_output.pop("video_chunk_size", 1) * ...]
```
`video_chunk_size` 在计算 MROPE 时被 `pop` 消耗掉，不会出现在最终的 batch 字典中。这是一种临时传参技巧，避免引入额外的参数通道。

### 16.3 训练与推理共用数据处理代码

`process_messages_to_model_inputs`、`compute_position_ids`、`find_assistant_spans` 被 SFT、GRPO 重建阶段和推理三处调用，代码完全相同。这是一个重要的设计决策：**训练时看到的 position_ids 和推理时完全一致**，避免了 train/test mismatch。

### 16.4 "Scope + Node"的依赖注入

```python
# 在 scope 中声明 "group_size" 对应 CLI 参数
"group_size": Ref[int]("args.train.group_size", metadata={ARG: Arg(default=8)})

# 在 @node 中使用
@node
def rollout(ctx, /, *, group_size: Auto[int], ...) -> Context:
    # group_size 的值自动从 ctx 中解析
```

这个模式让每个节点都是"无状态"的纯函数（除了对 ctx 的读写），便于单独测试和替换。

---

## 17. 调试与扩展指引

### 17.1 如何添加新数据集

1. 在 `thinkstream/data/__init__.py` 中追加：
```python
MY_DATASET = {
    "annotation_path": "./datasets/my_dataset.jsonl",
    "data_path": "./",
}
data_dict["my_dataset"] = MY_DATASET
```

2. 训练时传入 `--args.data.dataset_use my_dataset`

### 17.2 如何修改奖励函数

所有奖励函数都在 `trainer/grpo.py` 的第 40-197 行，是独立的纯函数。
修改权重：`DEFAULT_REWARD_WEIGHTS`（第 484-489 行）。
添加新奖励维度：在 `calc_rewards` 中追加计算，并更新 `REWARD_DICT_KEYS`。

### 17.3 如何支持新模型

1. `model/__init__.py`：`MODEL_CLS["new_model"] = NewModelClass`
2. `data/rope2d.py`：`ROPE_INDEX_FN["new_model"] = get_rope_index_new`
3. `model/patch.py`：`GRPO_LCE_FORWARD["new_model"] = grpo_lce_forward_new`
4. `trainer/sft.py:init_processor`：如有特殊 token 需求，添加条件分支

### 17.4 如何查看训练日志

训练指标通过 `reduce_and_log_metrics` 打印/记录，SFT 的指标定义在 `builder.py` 第 154-161 行，GRPO 的指标在第 254-270 行。GRPO 记录了：
- `loss`：GRPO 目标函数
- `reward_mean`/`reward_var`：总奖励的均值/方差
- `avg_think_len`：平均思维链长度（token 数）
- `reward_format_mean`、`reward_time_mean`、`reward_correctness_mean`、`reward_response_efficiency_mean`：各维度奖励

### 17.5 常见问题定位

| 现象 | 可能原因 | 查看位置 |
|------|---------|---------|
| 视频帧加载失败 | 视频路径或 data_path 错误 | `_make_abs_paths` (stream_data_processor.py:311) |
| position_ids shape 不匹配 | MROPE 计算逻辑问题 | `compute_position_ids` (stream_data_processor.py:748) |
| GRPO rewards 全为 0 | GT 答案格式不匹配 | `_extract_literal_answer` (grpo.py:112) |
| OOM in rollout | 像素预算太大 | `rollout_max_pixels`/`rollout_min_pixels` 参数 |
| 模型只输出 `<silent>` | `<response>` 权重太低 | `DataCollatorForSupervisedDataset.ce_weight` |
| Flex Attention 编译报错 | `video_mask` 维度不匹配 | `build_video_block_mask` (patch.py:32) |

---

## 附录：关键函数快速索引

| 函数/类 | 文件 | 行号 | 一句话说明 |
|---------|------|------|-----------|
| `register_streaming_attention` | `model/streaming_attention.py` | 145 | 注册 Flex Attention 到 transformers |
| `generate_video_sliding_window_mask_mod` | `model/streaming_attention.py` | 10 | 视频滑动窗口 mask 生成 |
| `build_video_block_mask` | `model/patch.py` | 32 | 构建 Flex Attention BlockMask |
| `_grpo_lce_forward_common` | `model/patch.py` | 186 | GRPO 模型前向（含 packing） |
| `load_video_frames` | `data/stream_data_processor.py` | 53 | Ghost message 视频帧加载 |
| `_build_messages` | `data/stream_data_processor.py` | 468 | SFT 消息列表构建 |
| `process_messages_to_model_inputs` | `data/stream_data_processor.py` | 609 | tokenize + 像素提取（三路共用） |
| `find_assistant_spans` | `data/stream_data_processor.py` | 714 | 定位 assistant token span |
| `compute_position_ids` | `data/stream_data_processor.py` | 748 | MROPE 位置编码（三路共用） |
| `DataCollatorForSupervisedDataset` | `data/stream_data_processor.py` | 1013 | SFT batch 组装 + CE 权重 |
| `get_rope_index_25` | `data/rope2d.py` | 165 | Qwen2.5-VL MROPE 实现 |
| `rollout` | `trainer/grpo.py` | ~380 | GRPO 推理采样节点 |
| `calc_rewards` | `trainer/grpo.py` | 492 | 四维奖励计算节点 |
| `build_grpo_inputs` | `trainer/grpo.py` | ~600 | GRPO 训练输入重建节点 |
| `StreamingCache` | `model/inference.py` | 90 | 静态 KV Cache |
| `GraphDecoder` | `model/inference.py` | 207 | CUDA Graph 解码器 |
| `streaming_video_chat` | `model/inference.py` | ~800+ | 流式视频推理主函数 |
| `default_scope` | `trainer/scope.py` | 13 | SFT 参数 Ref 映射表 |
| `grpo_scope` | `trainer/scope.py` | 181 | GRPO 参数 Ref 映射表 |
| `build_hf_deepspeed_train` | `trainer/builder.py` | 89 | 构建 SFT 训练图 |
| `build_grpo_train` | `trainer/builder.py` | 175 | 构建 GRPO 训练图 |
