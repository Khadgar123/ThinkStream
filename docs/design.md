# ThinkStream 设计（当前状态：v12.5）

> Source of Truth: `git log` + `tests/` + `memory/`. 本文档是 **快照**——当
> 代码与本文档分歧时以代码为准。最新 commit 见 `git log --oneline -10`。
>
> 关键节奏：**1 秒 = 1 chunk × 2 frames** （v12.5，2026-04-29 起）。所有
> chunk_sec / 视觉窗 / token 预算都按这个单位推算。配置规范权威：
> `scripts/agent_data_v5/config.py`；运行时常量从 `agent_protocol.py`
> 读出（它再 fallback 到 config）。

---

## 1. 项目目标

流式视频 Agent：每 1 秒（一个 chunk）做一次决策，输出 `silent / response /
recall / compress` 之一，配合内部记忆状态在长视频中持续做问答。

**数据 + 训练 + 推理三端共用 per-timestep snapshot 格式**：训练样本 =
推理某一步的精确 input。teacher-forcing 让 SFT 训练分布严格等于 inference
分布。

## 2. 数据流水线

```
pass1a  per-chunk evidence (vision-LLM)             → evidence_1a/{vid}.json
pass1b  entity alignment + state changes            → evidence_1b/{vid}.json
pass2   streaming rollout (memory evolution)        → rollout/{vid}.json
pass3a  card generation (22 families)               → task_cards/{vid}.json
pass3b  placement + trajectory plan                 → placements/{vid}.json
pass3c  per-timestep sample generation              → samples_3c/{vid}.json
pass3e  verification (tag-only, non-destructive)    → verified/{vid}.json
pass4   trajectory grouping + flat split files      → final/{train_sft|train_rl|val|test}*.jsonl
pass5   ShareGPT messages format conversion         → final/{train_sft|val|test}_messages.jsonl
                                                      + final/dataset_info.json
```

**缓存版本控制**：`scripts/agent_data_v5/cache_version.py`。改 prompt → bump
3a；改 placement → bump 3b；改 verify → bump 4。pass1a/1b/2 是慢的，**改时
尽量不动**。

**pass5 是新增层（v12.5）**：把 pass4 的 `{input, output}` 单步样本翻译成
LLaMA-Factory ShareGPT messages 格式，单步 SFT 直接吃。recall_query 多轮 +
inter-chunk compress 的形态都在 pass5 里被正确渲染。

## 3. Memory 模型

**`MemoryState.timeline`**（pass2_rollout.py:48-180）= 单一按时间排序的列表，
混合两类项：

```
think     {"type":"think",   "chunk":N, "time":"a-b", "text":"..."}
summary   {"type":"summary", "time_range":[a,b], "text":"...", "source_chunks":[...]}
```

派生属性：
- `recent_thinks` = `[item for item in timeline if type=="think"]`
- `compressed_segments` = `[item for item in timeline if type=="summary"]`

**触发**（v12.5 配置）：

| 阈值 | 值 | 计算 |
|---|---|---|
| `RECENT_THINKS_TOKEN_BUDGET` | 4000 | ≈ 57 thinks @ 70 tok |
| `COMPRESS_TOKEN_THRESHOLD` (80%) | 3200 | trigger |
| `COMPRESS_HYSTERESIS_THRESHOLD` (55%) | 2200 | post-compress 必须降到此线下 |
| `COMPRESS_RANGE_MIN/MAX` | 8 / 24 | 单次压缩跨 thinks 数（= 秒数） |
| `MAX_COMPRESSED_SEGMENTS` | 5 | 5 × 280 tok = 1400 tok |
| `SUMMARY_TOKENS_MAX` | 280 | 单段摘要硬上限 |

**Text horizon**（设计预算）：57 个 active thinks ≈ 57 秒 + 5 段压缩 × ~16
think/段 ≈ 80 秒 → 总 history ~137 秒，远超 visual horizon 16 秒。

## 4. Visual 模型

```
AGENT_CHUNK_SEC      = 1
FRAMES_PER_CHUNK     = 2          # 2 fps
VISUAL_WINDOW_CHUNKS = 16         # 16 chunks × 1s = 16s 滑窗
VISUAL_WINDOW_FRAMES = 32
VISUAL_TOKENS_PER_CHUNK = 128     # at min_pixels=100352
VISUAL_WINDOW_TOKENS = 16 × 128 = 2048
```

视觉窗每 chunk 滑动：第 N 个 chunk 看 `[max(0,N-15), N]`，共 32 帧。

## 5. Recall 模型

assistant 输出 `<tool_call>{"name":"recall","arguments":{"query","time_range"}}</tool_call>`
→ system 检索历史 think → 下一个 user message 注入 `<recalled_frames>` +
帧 + `<recall_result>` → assistant 给最终 answer。**单 chunk 内 2-turn**，
在数据里是 shape B（`v12_assistant_turn_1/_2`）。

帧的 MROPE 时间位置 = 帧首次出现时的 timestamp（不是当前 chunk 的位置）——
通过 chat_template 的 `video_start/video_end` 参数设置。

## 6. 输入布局（位置编码）

**MROPE 当前实现现状**（不是严格对齐，避免误判）：

- `thinkstream/sft/rope2d.py` 的 `get_rope_index_3` 用 timestamps 编码视频
  temporal 维度，但 SFT/inference 入口直接调它（`data_processor.py:559`），
  视觉 temporal index 在 t=1 之后基本压成 0
- `thinkstream/data/rope2d.py:376` 有一个 text temporal override helper，**目
  前没被 SFT 入口调用**——文本和视觉位置编码并未做严格对齐
- recall 帧的"原始 MROPE 时间复用"目前主要靠 user content 里的
  `<recalled_frames time_range="...">` 文本 tag 和 `video_start/end`
  metadata 表达，**不是 position id 严格复用**——模型通过文本上下文知道
  这些是历史帧，但 KV 里的位置编码并不是它们首次出现时的位置
- 这是已知 gap，不影响 SFT 收敛（只要训练/推理一致即可），但要在 RL 阶
  段做严格 KV 复用时需要把 text temporal override helper 接进数据流

每步消息布局（视频在前、文本在后；位置编码按当前 chat_template 默认行为
渲染——并未做 agent-specific 对齐）：

```
[system + tools]                                  ← chat_template 自动渲染 <tools>
<visual_window>{json header}</visual_window>  +  video_block (32 frames @ 16s window)
<recalled_frames>{header}</recalled_frames>   +  recalled_video    (recall 二次 pass 才有)
<memory>
  <compressed t="a-b">{json}</compressed> × n_seg              (≤ 5)
  [time] think text × n_recent                                  (≤ ~57)
</memory>
<queries>...</queries>                                          (历史已答 Q&A)
<recall_result>{json}</recall_result>                           (recall 二次 pass 才有)
<user_input>{question or compress_trigger}</user_input>
```

inter-chunk compress 样本（`v12_inter_chunk=True`）**没有 visual_window 也
没有 video frames**——压缩是 chunk 之间的事件，不消耗一个视觉时间步。

## 7. 训练流水线

```
pretrained Qwen3-VL-8B-Instruct → SFT (3 epoch) → GDPO RL (1 epoch)
```

**SFT 损失**（v12.5）：vanilla CE on assistant span（DeepEyesV2 / Qwen-VL
官方约定）。**没有 token-span 加权**，**没有类别加权采样**——v12 协议没有
`<action>` 这种单一关键词位置可加权，且 v12.5 数据自然分布够平衡。

**SFT 数据三种 shape**：
- A. **单步**（silent / response / lonely recall）：[system, user, assistant]
- B. **recall 多轮**（chunk 内）：[system, user, assistant(tool_call), user(tool_result), assistant(answer)]
- C. **inter-chunk compress**：[system, user(memory + compress_trigger), assistant(tool_call)]

loss-mask 实现（与 VST / LiveCC / Qwen-VL 官方一致）：扫 `<|im_start|>assistant`
到 `<|im_end|>`，仅这段开 loss。

**RL（GDPO，v11.4 设计 + v12.0 协议适配）**：reward 8 路 column，权重和 = 1.0：

| reward | 权重 | 说明 |
|---|---|---|
| correctness | 0.30 | response 主结果 |
| silent_quality | 0.18 | streaming-specific dense reward |
| timing | 0.18 | 用 last response chunk 距 gold ask_chunk |
| recall_quality | 0.05 | query JSON 格式 + 无 leakage |
| recall_hit_rate | 0.07 | `\|returned ∩ support\| / \|support\|`，失败显式 -0.2 |
| range_tightness | 0.03 | `(1 - range_width/duration) × coverage`，coverage=0 显式 -0.2 |
| format | 0.09 | 协议合规率（含 v12 tool_call JSON） |
| overflow_pen | 0.10 | compress timing soft penalty |

**Per-reward group-norm**（GDPO）独立做 mean-only 不归 σ，权重控制相对拉力。

## 8. Inference / Eval 框架

**Streaming inference loop**（fresh-KV-per-chunk，对应 SFT 训练分布）：

```python
for chunk_idx in range(num_chunks):
    user_content = build_user(memory_state, queries, visual_window[N-15:N], frame_paths)
    response = vllm.generate(prompt=system + user_content, ...)
    parsed = parse_v12(response.text)
    if parsed.kind == "recall":
        recall_result = retrieve(parsed.tool_call.arguments)
        final_response = vllm.generate(prompt + response.text + recall_user, ...)
        # ... handle answer
    elif parsed.kind == "compress":
        memory_state.apply_compress(parsed.tool_call.arguments)
    elif parsed.kind == "answer":
        record_response(...)
    # 每 chunk KV 重建（state 在 Python 层维护）
```

每个 chunk 的 user content 自带完整历史（memory text + queries text）——
KV 不需要跨 chunk 复用，prefix cache 命中 system+tools 即可。memory text >
visual KV horizon（57s vs 16s），保证被踢出 KV 的老 think 仍在下一轮 user
content 里。

**两个 profile**（`scripts/eval/eval_profiles.py`）：

| | 16k 默认 | 32k 扩展 |
|---|---|---|
| model_max_length | 16384 | 32768 |
| QUERIES_HISTORY_CAP | 8 | 24 |
| RECALL_TEXT_MAX_CHARS | 800 | 3000 |
| max_new_tokens | 128 | 256 |

profile 通过 `apply_profile()` runtime 改 agent_protocol globals，**不影响
SFT-baked 常量**。

**Streaming 测试维度**（每 ckpt × bench 跑 32 配置）：
- retriever ∈ {bm25, hybrid}，hybrid 可加 `--use_agent_vision`
- compress_mode ∈ {system, self}
- scoring ∈ {strict, lenient}
- profile ∈ {16k, 32k}

## 9. Telemetry

`agent_loop.step()` 返回 dict 含：
```
prompt_text_token_count    每步 prompt 文本部分总 tokens
think_token_count          模型输出的 <think> tokens
format_ok                  bool: 是否合规协议格式
compress_succeeded         bool/None: 触发后 summary 是否合规
compress_telemetry         {thinks_count_at_trigger, compressed_chunks, ...}
recall_returned_chunks     recall 返回的 chunk 列表
memory_token_count         post-step recent_thinks tokens
```

`walk_and_score` 聚合到 per-sample：
```
n_compress_events / compress_thinks_at_trigger
n_partial_compress (summary < COMPRESS_RANGE_MIN)
compress_chunk_count / n_chunks_revisited
n_recall_events / recall_events
recall_hit_fractions
n_premature_responses / n_late_responses / response_offset_chunks
n_format_violations
prompt_tokens_per_step / think_tokens_per_step
```

## 10. 关键 quick-reference 常量（v12.5）

| 类型 | 名 | 值 | 来源 |
|---|---|---|---|
| chunk | AGENT_CHUNK_SEC | **1** | config.py |
| chunk | FRAMES_PER_CHUNK | 2 | config.py |
| chunk | VISUAL_WINDOW_CHUNKS | **16** | config.py |
| memory | RECENT_THINKS_TOKEN_BUDGET | **4000** | config.py |
| memory | COMPRESS_TOKEN_THRESHOLD | 3200 | config.py |
| memory | COMPRESS_HYSTERESIS_THRESHOLD | 2200 | config.py |
| memory | COMPRESS_RANGE_MIN/MAX | 8 / 24 | config.py |
| memory | MAX_COMPRESSED_SEGMENTS | 5 | config.py |
| memory | SUMMARY_TOKENS_MAX | **280** | config.py |
| think | THINK_TOKENS | (40, 80) | config.py |
| think | THINK_TOKEN_AVG | 60 | config.py |
| visual | video_min/max_pixels | 100352 / 150528 | sft/argument.py |
| visual | VISUAL_TOKENS_PER_CHUNK | 128 | config.py |
| budget | MAX_SAMPLE_TOKENS | **16384** | config.py |
| budget | SYSTEM_PROMPT_TOKENS | 400 | config.py |
| eval | QUERIES_HISTORY_CAP | 8 (16k) / 24 (32k) | profile |
| eval | RECALL_TEXT_MAX_CHARS | 800 / 3000 | profile |
| eval | model_max_length | 16384 / 32768 | profile |
| eval | max_new_tokens | 128 / 256 | profile |
| pass1 perf | pass1a concurrent | 1024 | config.py |
| pass1 perf | httpx max_connections | 2048 | vllm_client.py |

**SFT 训练框架**：Qwen-VL 官方 finetune fork（`thinkstream/sft/`），不依赖
LLaMA-Factory（但数据 messages 格式兼容 LLaMA-Factory CLI）。
**RL 训练框架**：verl + 自定义 ChunkLevelRolloutLoop。

## 11. v12.5 vs v11.x 变化一览

**v12.5（2026-04-29）核心改动**：
- chunk_sec 2 → 1（FPS 1 → 2，2 frames/chunk 不变）
- 视觉窗 12 chunks (24s) → 16 chunks (16s)
- text memory 600 token (~14s) → 4000 token (~57s) → 文本 horizon > 视觉
- COMPRESS_RANGE_MIN/MAX 4/12 → 8/24（按秒等价 8-24s 不变）
- COMPRESS_RANGE_MIN/MAX 4/12 → 8/24（按秒等价 8-24s 不变）
- SUMMARY_TOKENS_MAX 180 → 280（p80 实测中位数 169，原 180 切了 33% 摘要）
- MAX_SAMPLE_TOKENS 4096 → 16384（与 teacher 一致）
- pass5 messages 转换层新增

**v12.0（vs v11）核心改动**：
- 协议从 `<action>X</action>` 单 token 切到官方 `<tool_call>JSON</tool_call>` + `<answer>...</answer>`
- 模型从 Qwen2.5-VL 切到 Qwen3-VL（v12 强制依赖，2.5-VL 的 chat_template 不支持 tools）
- 删除所有 v11 vocab 扩展（register_special_tokens / smart_init / SPECIAL_TOKENS_AGENT）
- 删除 SPAN_WEIGHTS / focal+alpha / ClassBalancedDistributedSampler——vanilla CE 即可

**与同期工作对齐**：
- DeepEyesV2 / Qwen-VL 官方 / VST / LiveCC 都用 system+user+assistant 三角色（无 tool/observation 角色），tool 输出注入下一 user content
- loss-mask 都是扫 `<|im_start|>assistant` 到 `<|im_end|>`，仅此段开 loss
- 我们的 `data_processor.preprocess_per_timestep` 与官方实现行为一致
