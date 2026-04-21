# ThinkStream 训练流程详解

> **注意**：本文档描述的是旧版冷启动 SFT + GRPO 训练流程（多轮对话格式）。
> 最新的 per-timestep 独立样本训练方案请参见：
> - `sft_engineering.md` — SFT 工程设计（注意力掩码、数据加载、训练课程）
> - `data_construction_zh.md` — 数据构造方案 v5.6
>
> 本文档保留供冷启动实验复现参考。

本文档说明 ThinkStream 的完整训练流程，包括两阶段训练策略（SFT 冷启动 + RL 强化学习）、数据格式规范以及系统中的核心优化设计。

---

## 目录

1. [总体架构](#1-总体架构)
2. [第一阶段：SFT 冷启动](#2-第一阶段sft-冷启动)
3. [第二阶段：RL 强化学习（GRPO）](#3-第二阶段rl-强化学习grpo)
4. [数据格式](#4-数据格式)
  - [SFT 冷启动数据格式](#41-sft-冷启动数据格式-streaming_cot_cold_processed_5_20jsonl)
  - [RL RLVR 数据格式](#42-rl-rlvr-数据格式-streaming_rlvr_processedjsonl)
5. [系统优化](#5-系统优化)
6. [数据集注册机制](#6-数据集注册机制)
7. [快速开始](#7-快速开始)

---

## 1. 总体架构

ThinkStream 的训练采用**两阶段**策略，使 Qwen2.5-VL 多模态语言模型获得对流式视频的实时推理与回答能力：

```
原始模型 (Qwen2.5-VL-3B-Instruct)
        │
        ▼  [阶段一: SFT 冷启动]
冷启动模型 (具备流式思维链格式)
        │
        ▼  [阶段二: RL GRPO]
ThinkStream 模型 (具备时间感知的实时流式视频理解能力)
```


| 阶段  | 方法            | 数据集 key             | 主要目标                                         |
| --- | ------------- | ------------------- | -------------------------------------------- |
| 一   | SFT（监督微调）     | `stream_cold_start` | 教会模型输出 `<think>...<response>/<silent>` 的流式格式 |
| 二   | RL（GRPO 强化学习） | `stream_rlvr`       | 优化模型在正确时刻作出正确回答，并控制思维链长度                     |


---

## 2. 第一阶段：SFT 冷启动

### 2.1 训练目标

SFT 阶段的目标是让模型学会**流式思维链（Streaming CoT）格式**：在每个视频时间窗口（chunk）上输出：

```
<think>[此窗口的内部推理过程]</think><response>[回答内容]
```

或在无需回答时输出：

```
<think>[此窗口的内部推理过程]</think><silent>
```

模型需要在连续的视频片段流中保持思维和回应的连贯性。

### 2.2 启动命令

```bash
bash scripts/sft.sh
```

关键超参数（详见 `scripts/sft.sh`）：


| 参数                            | 值                             | 说明           |
| ----------------------------- | ----------------------------- | ------------ |
| `model`                       | `Qwen/Qwen2.5-VL-3B-Instruct` | 基础模型         |
| `model_type`                  | `qwen2.5vl`                   | RoPE / 处理器适配 |
| `learning_rate`               | `1e-5`                        | 学习率          |
| `num_train_epochs`            | `1`                           | 训练轮数         |
| `per_device_train_batch_size` | `8`                           | 每卡批次         |
| `max_length`                  | `32768`                       | 最大序列长度       |
| `video_min_pixels`            | `100352`                      | 视频最小像素预算     |
| `video_max_pixels`            | `150528`                      | 视频最大像素预算     |
| `warmup_ratio`                | `0.03`                        | 学习率预热比例      |
| `lr_scheduler_type`           | `cosine`                      | 调度策略         |
| `deepspeed`                   | `scripts/zero3.json`          | ZeRO-3 分布式策略 |


### 2.3 SFT 数据处理流程

```
JSONL 标注文件 (conversations + thoughts)
        │
        ▼  _build_messages()  [stream_data_processor.py]
多轮 Chat Messages（system + 交替的 user[视频chunk+问题] / assistant[<think>+<response>/<silent>]）
        │
        ▼  process_messages_to_model_inputs()
model inputs: input_ids, pixel_values_videos, video_grid_thw, position_ids
        │
        ▼  find_assistant_spans()  →  labels（仅 assistant 轮次有梯度）
训练 batch
```

**标签掩码**：只有 assistant 轮次的 token 参与损失计算，user / system 部分 label 填充为 `-100`。

### 2.4 加权交叉熵

在 `DataCollatorForSupervisedDataset` 中，对 `<response>` 和 `<silent>` 两个特殊 token 进行了**动态权重均衡**，防止高频的 `<silent>` 主导梯度：

```python
# ce_weight[token_id] = total_count / (2 * class_count)
ce_weight[silent_id]   = total_n / (2 * n_silent + eps)
ce_weight[response_id] = total_n / (2 * n_response + eps)
ce_weight = torch.clamp(ce_weight, 0, 20)  # 上界防止过拟合
```

---

## 3. 第二阶段：RL 强化学习（GRPO）

### 3.1 训练目标

RL 阶段以 SFT 冷启动模型为起点，使用 **GRPO（Group Relative Policy Optimization）** 方法进行强化学习。

GRPO 的核心思想：对同一个问题采样多个回答（group），通过组内奖励的相对排名来计算梯度，而不需要额外的 Critic 网络。

### 3.2 启动命令

```bash
bash scripts/rl.sh
```

关键超参数（详见 `scripts/rl.sh`）：


| 参数                   | 值            | 说明                |
| -------------------- | ------------ | ----------------- |
| `model`              | SFT 冷启动检查点路径 | 起点模型              |
| `learning_rate`      | `2e-7`       | 比 SFT 更小的学习率      |
| `group_size`         | `8`          | 每个问题的采样组大小 G      |
| `micro_batch_size`   | `8`          | 滚动推理的 micro batch |
| `rollout_max_pixels` | `192×28×28`  | 滚动推理的最大像素         |
| `rollout_min_pixels` | `128×28×28`  | 滚动推理的最小像素         |
| `save_steps`         | `300`        | 保存间隔              |


### 3.3 GRPO 训练流程

```
raw_sample（video_path + conversations + GT 时间戳）
        │
        ▼  streaming_video_chat()  [rollout，采样 G 个回答]
chunk_results: List[{chunk_idx, window_start, window_end, generated_tokens: [G]}]
        │
        ▼  calc_rewards()  →  rewards: [B×G]
        │
        ▼  build_grpo_inputs()  →  重建训练输入（input_ids + 生成的 tokens）
        │
        ▼  compute_grpo_loss()  →  GRPO loss（基于组内优势）
        │
        ▼  optimizer.step()
```

### 3.4 奖励函数设计

RL 阶段使用四个维度的复合奖励，总奖励为加权求和：

```python
total_reward = 0.2 * format_reward
             + 0.2 * time_reward
             + 0.4 * correctness_reward
             + 0.2 * response_efficiency_reward
```

#### format_reward（格式奖励）

检查每个 chunk 的生成文本是否严格匹配正则：

```
^<think>.*?</think>(?:<response>.*|<silent>)<|im_end|>$
```

奖励值 = 格式正确的 chunk 数 / 总 chunk 数，范围 `[0, 1]`。

#### correctness_reward（正确性奖励）

从所有 chunk 的 `<response>` 标签中提取第一个有效答案，与 GT 标注比较：

- 支持选择题（A–E）、是非题（yes/no）、计数题（0–9）
- 完全匹配得 1.0，否则得 0.0

#### time_reward（时间奖励）

鼓励模型在接近 GT 标注时刻作出回答，允许一定容差：

```python
diff = |response_chunk_idx - gt_chunk_idx|
if diff <= slack_window:         reward = 1.0
elif diff <= slack_window + W:   reward = 1.0 - (diff - slack) / W
else:                            reward = 0.0
```

#### response_efficiency_reward（回应效率奖励）

由两个子奖励的乘积组成，平衡思维深度与回应简洁性：

- **think_length_factor**：思维链长度接近目标 `rollout_max_think_tokens` 时得分更高（阶梯式奖励，防止空思维）
- **num_response_factor**：恰好回应 1 次得满分，多次回应按阶梯衰减（防止重复输出）

---

## 4. 数据格式

所有数据集的路径通过 `thinkstream/data/__init__.py` 中的注册表统一管理：

```python
from thinkstream.data import data_list

configs = data_list(["stream_cold_start", "stream_rlvr"])
# configs[0]["annotation_path"] → "./datasets/streaming_cot_cold_processed_5_20.jsonl"
# configs[1]["annotation_path"] → "./datasets/streaming_rlvr_processed.jsonl"
```

### 4.1 SFT 冷启动数据格式 (`streaming_cot_cold_processed_5_20.jsonl`)

每一行是一个 JSON 对象，字段如下：

```jsonc
{
  // 视频文件相对路径（相对于 data_path，即 "./"）
  "video_path": "videos/sample_001.mp4",

  // 对话轮次：user 和 assistant 交替出现，每条均带时间戳（单位：秒）
  "conversations": [
    {
      "role": "user",
      "content": "What is the person doing right now?",
      "timestamp": 12.5          // 该问题在视频中的提出时刻
    },
    {
      "role": "assistant",
      "content": "cooking pasta", // GT 回答文本（无需包含 <think>/<response> 标签，由代码拼接）
      "timestamp": 15.0           // GT 回答的期望时刻
    }
  ],

  // 思维链标注：每条对应一个视频时间窗口，由标注者提供的内部推理文本
  "thoughts": [
    { "timestamp": 10.0, "think": "The person is in the kitchen, facing the stove." },
    { "timestamp": 14.0, "think": "Steam is rising from the pot; they are stirring." }
  ],

  // [可选] 预计算的序列长度，用于 DataLoader 的 LengthGroupedSampler
  "num_tokens": 4096
}
```

**数据处理逻辑（`_build_messages`）**：

1. 将视频按时间窗口（`video_chunk_size ≈ ceil(duration / 120)`）切割成若干 chunk
2. 对每个 chunk 构建：
  - **user 消息**：包含该 chunk 的视频片段（`{"type": "video", ...}`）+ 落在该窗口内的用户问题
  - **assistant 消息**：`<think>[该窗口的 thoughts]</think>` + `<response>[该窗口的 GT 回答]` 或 `<silent>`
3. 拼接成完整的多轮 Chat 消息列表，首条为 `system` 提示

最终 assistant 消息格式示例：

```
<think>Steam is rising from the pot; they are stirring.</think><response>cooking pasta
```

或在无回答的 chunk：

```
<think>The person picks up a knife and starts chopping.</think><silent>
```

### 4.2 RL RLVR 数据格式 (`streaming_rlvr_processed.jsonl`)

每一行是一个 JSON 对象，相比 SFT 数据**不含 `thoughts` 字段**：

```jsonc
{
  "video_path": "videos/sample_042.mp4",

  // conversations 结构与 SFT 相同，但 assistant 内容是最终 GT 答案
  // GRPO rollout 代码使用 conversations[1] 作为 GT 标注
  "conversations": [
    {
      "role": "user",
      "content": "Has the step 'add salt to the bowl' been completed?",
      "timestamp": 30.0
    },
    {
      "role": "assistant",
      "content": "Yes",           // GT 答案（选择题 A-E / 是非题 yes/no / 计数题 0-9）
      "timestamp": 35.0          // GT 期望回答时刻，用于 time_reward 计算
    }
  ]

  // 可能含有的其他字段（取决于数据集来源）
  // "task": "SSR",               // 任务类型（CRR / SSR / REC / OQA 等）
  // "id": "ovo_bench_001",       // 样本 ID
}
```

**GRPO 如何使用该数据**：

```python
gt_msg = raw_sample["conversations"][1]    # 取第二条（assistant）消息
gt_timestamp = float(gt_msg["timestamp"]) # 用于计算 time_reward
gt_content   = gt_msg["content"]          # 用于计算 correctness_reward
```

---

## 5. 系统优化

### 5.1 流式视频分块处理（Streaming Chunking）

ThinkStream 的核心创新是将视频切分为时间窗口序列，模拟"实时流"环境。关键参数：

- `frames_per_chunk = 2`：每个时间窗口提取 2 帧
- `max_chunks = 120`：单个视频最多 120 个 chunk
- `video_chunk_size = ceil(duration / max_chunks)`：动态计算每个 chunk 的时长

这一设计使模型在训练和推理阶段保持一致的时序感知，避免了"看完整视频再回答"的信息泄露。

### 5.2 Ghost Message 视频加载模式

所有视频帧的加载（SFT、GRPO rollout、推理）均通过统一的 `load_video_frames()` 函数，使用"ghost message"模式：

```python
ghost_message = [{"role": "user", "content": [
    {"type": "video", "video": path, "video_start": t0, "video_end": t1, "nframes": N}
]}]
big_tensor = process_vision_info(ghost_message)     # 一次性加载所有帧
split_videos = torch.split(big_tensor, frames_per_chunk)  # 再切分
```

优点：避免重复 I/O，单次调用提取全部所需帧后再按 chunk 切分。

### 5.3 DataLoader 并行预加载视频

`LazyRawDataset`（RL 数据集）在 `__getitem__` 中调用 `preload_video()`，将视频解码提前到 DataLoader 的 worker 进程中执行：

```python
# num_workers=4 时，4 个 worker 并行解码视频，主进程直接消费 tensor
item["_preloaded_video"] = preload_video(abs_video_path, ...)
```

### 5.4 Multimodal RoPE 位置编码（MROPE）

为保持视频帧与文本 token 的时间对齐，在 `compute_position_ids()` 中精确计算每个视频 chunk 的 `second_per_grid_ts`：

```python
second_per_grid_ts = video_chunk_size * temporal_patch_size / fps
```

同一套 RoPE 计算逻辑被 SFT、GRPO 重建阶段和推理三处共用（`ROPE_INDEX_FN` 注册表），确保位置编码的完全一致性。

### 5.5 ZeRO-3 分布式训练

使用 DeepSpeed ZeRO Stage 3（`scripts/zero3.json`），将优化器状态、梯度、模型参数均分片到所有 GPU，显著降低每卡显存占用：

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

配合 `bf16` 混合精度训练，可在 8×A100-40G 上训练 3B 规模模型。

### 5.6 容错 Dataset `__getitem__`

`LazySupervisedDataset.__getitem__` 实现了三级容错：

1. 先重试当前样本 3 次（应对短暂 I/O 故障）
2. 再随机选取其他样本重试 30 次（避免少数坏样本卡死训练）
3. 最终再尝试原样本一次并抛出异常

### 5.7 数据集采样率

数据集名称支持 `%{N}` 后缀，在不修改代码的情况下灵活控制训练数据量：

```python
# 只使用 50% 的冷启动数据
data_list(["stream_cold_start%50"])
```

### 5.8 流式注意力（Streaming Attention）

模型层面通过 `thinkstream/model/streaming_attention.py` 注册了自定义的流式因果注意力，并在 `thinkstream/model/patch.py` 中构建 `video_block_mask`，使不同时间窗口的视频 token 只能看到本 chunk 和之前 chunk 的内容，实现因果性约束。

### 5.9 Think Budget 采样（RL 推理优化）

在 GRPO rollout 阶段使用 `think_budget_sample` 策略，根据 `rollout_max_think_tokens` 动态控制每个 chunk 中 `<think>` 部分的生成长度，防止推理时的 token 浪费，同时与 `response_efficiency_reward` 的奖励设计形成闭环。

---

## 6. 数据集注册机制

所有数据路径通过 `thinkstream/data/__init__.py` 集中管理：

```python
# thinkstream/data/__init__.py

STREAM_COLD_START = {
    "annotation_path": "./datasets/streaming_cot_cold_processed_5_20.jsonl",
    "data_path": "./",
}

STREAM_RLVR = {
    "annotation_path": "./datasets/streaming_rlvr_processed.jsonl",
    "data_path": "./",
}

data_dict = {
    "stream_cold_start": STREAM_COLD_START,
    "stream_rlvr":       STREAM_RLVR,
}
```

训练脚本通过 `--args.data.dataset_use stream_cold_start` 参数传入名称，由 `data_list()` 解析为路径配置后供 `LazySupervisedDataset` / `LazyRawDataset` 使用。

---

## 7. 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 下载数据集

从 Hugging Face 下载数据集并放置到项目根目录的 `datasets/` 目录下：

- `datasets/streaming_cot_cold_processed_5_20.jsonl` — SFT 冷启动标注
- `datasets/streaming_rlvr_processed.jsonl`          — RL 奖励学习标注

### 分析数据集

```bash
# 分析两个数据集的统计信息
python scripts/analyze_data.py

# 只分析某一个数据集并展示样本示例
python scripts/analyze_data.py --dataset stream_cold_start --show-sample

# 使用 50% 采样
python scripts/analyze_data.py --dataset stream_rlvr%50
```

### 训练

```bash
# 第一阶段：SFT 冷启动
bash scripts/sft.sh

# 第二阶段：RL 强化学习
# 先修改 scripts/rl.sh 中的 llm 变量为 SFT 输出路径
bash scripts/rl.sh
```

### 评估

```bash
bash scripts/eval/eval.sh
```

---

## 参考文件


| 文件                                          | 说明                     |
| ------------------------------------------- | ---------------------- |
| `thinkstream/data/__init__.py`              | 数据集路径注册表               |
| `thinkstream/data/stream_data_processor.py` | 数据加载、消息构建、tokenize     |
| `thinkstream/trainer/sft.py`                | SFT 训练主逻辑              |
| `thinkstream/trainer/grpo.py`               | GRPO rollout、奖励计算、loss |
| `thinkstream/trainer/builder.py`            | 训练节点图注册（SFT / GRPO）    |
| `thinkstream/model/streaming_attention.py`  | 流式注意力注册                |
| `thinkstream/model/patch.py`                | video_block_mask 构建    |
| `thinkstream/data/rope2d.py`                | MROPE 位置编码实现           |
| `scripts/sft.sh`                            | SFT 启动脚本               |
| `scripts/rl.sh`                             | RL 启动脚本                |
| `scripts/zero3.json`                        | DeepSpeed ZeRO-3 配置    |
| `scripts/analyze_data.py`                   | 数据集统计分析工具              |


