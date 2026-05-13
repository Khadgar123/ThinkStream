# ThinkStream 设计（当前状态：v12.14, 2026-05-03）

> Source of Truth: `git log` + `tests/` + `memory/`. 本文档是**快照** — 当代码与
> 本文档分歧时以代码为准。最新 commit 见 `git log --oneline -10`。
>
> 历史版本详细变更见：
> - `v12.0_protocol_migration_design.md` — Qwen3-VL tool protocol 迁移
> - `v12.1_batch2_tuning.md` — batch2 调优
> - `v12.14_recurrent_design.md` — recurrent rollout 设计（>180 chunk）
> - `v12.14_integration_status.md` — recurrent Phase 1-4 落地状态
> - `v11.3_changelog.md` / `v11.4_rl_design.md` — 老版本（仅作历史参考）
>
> **关键节奏**：1 秒 = 1 chunk × 2 frames（v12.5 起，v12.14 不变）。所有
> chunk_sec / 视觉窗 / token 预算都按这个单位推算。配置规范权威：
> `scripts/agent_data_v5/config.py`；运行时常量从 `agent_protocol.py` 读出
> （它再 fallback 到 config）。

---

## 1. 项目目标

流式视频 Agent：每 1 秒（一个 chunk）做一次决策，输出 `silent / response /
recall / compress` 之一，配合内部记忆状态在长视频中持续做问答。

**3 端共用 per-timestep snapshot 格式**：训练样本 = 推理某一步的精确 input。
SFT teacher-forcing 让训练分布严格等于 inference 分布；RL rollout 使用同样
的 per-chunk prompt 结构。

**多问答 (multi-Q) 是一等公民**：1 video → N 个 MCQ 在不同时间点提问
（OVOBench 风格）。answer 必须归到对的 ask_chunk，timing reward 监督这件事。

---

## 2. 数据流水线（5 pass + parquet 转换）

```
pass1a  per-chunk evidence (vision-LLM)             → evidence_1a/{vid}.json
pass1b  entity alignment + state changes            → evidence_1b/{vid}.json
pass2   streaming rollout (memory evolution)        → rollout/{vid}.json
pass3a  card generation (10 families × 5 forms)     → task_cards/{vid}.json
pass3b  placement + ask-time + trajectory plan      → placements/{vid}.json
pass3c  per-timestep sample generation              → samples_3c/{vid}.json
pass3e  verification (tag-only, non-destructive)    → verified/{vid}.json
pass4   trajectory grouping + split files           → final/{train_sft|train_rl|val|test}*.jsonl
pass5   multi-turn trajectory rendering             → rendered/trajectory/{train_sft|val|test}_trajectory.jsonl

scripts/agent_data_v5/build_verl_parquet.py — 把 pass4 output 转 verl 输入：
  默认               → thinkstream_v12_streaming  （per-question rows）
  --multi_q          → thinkstream_v12_streaming_multi_q（per-video rows，N Qs）
```

**缓存版本控制**：`scripts/agent_data_v5/cache_version.py`。改 prompt → bump
3a；改 placement → bump 3b；改 verify → bump 4。pass1a/1b/2 是慢的，**改时
尽量不动**。

**Disjoint pool**（v12 起）：pass4 输出 `train_sft.jsonl`（199 vids）和
`train_rl.jsonl`（50 vids）**互斥**，避免 RL 在 SFT 已经记住的 prompt 上
reward-hack。`PHASE=mixed` 走单池 baseline。

**问题家族（v12 task_taxonomy）**：10 个 family —
F1 OCR/Number, F2 Fine-grained, F3 Count, F4 Spatial, E1 Local Action,
E2 State Change, P1 Procedure, C1 Comparison, R1 Re-id, S1 Summary。
D1 (multi-turn) 和 M1 (commentary) 推迟到二版。

**5 种 answer_form**（按可验证度排）：
| form | 例 | RL matcher (`thinkstream/trainer/outcome_match.py`) |
|---|---|---|
| binary | yes/no | `match_binary` — yes/no token 字面匹配 |
| multiple_choice | A/B/C/D | `match_mcq_answer` — liberal letter+text matching |
| number | 3 / 25.4 | `match_number` — 数值容差 |
| short_exact | "red" | `match_short_exact` — 字面匹配 + 大小写归一 |
| descriptive | 短句 | `match_descriptive` — fuzzy substring fallback |

**`build_verl_parquet --multi_q`**（v12.14 引入）：1 video = 1 row 携带
`questions: [{q, ask_chunk, options, answer, answer_form}, ...]`。trainer 端
`ts_per_q_*` extras 是这些 Q 的最终归属（chunk + text）。

---

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

**触发**（v12.14 配置，与 pass2 / RL / eval 严格一致）：

| 阈值 | 值 | 计算 |
|---|---|---|
| `RECENT_THINKS_TOKEN_BUDGET` | 4000 | ≈ 57 thinks @ 70 tok |
| `COMPRESS_TOKEN_THRESHOLD` (80%) | 3200 | trigger |
| `COMPRESS_HYSTERESIS_THRESHOLD` (55%) | 2200 | post-compress 必须降到此线下 |
| `COMPRESS_RANGE_MIN/MAX` | 8 / 24 | 单次压缩跨 thinks 数（= 秒数） |
| `MAX_COMPRESSED_SEGMENTS` | 5 | 5 × 280 tok = 1400 tok |
| `SUMMARY_TOKENS_MAX` | 280 | 单段摘要硬上限 |

**Text horizon**（设计预算）：57 个 active thinks ≈ 57 秒 + 5 段压缩 × ~16
think/段 ≈ 80 秒 → 总 history ~137 秒，远超 visual horizon 8 秒。

---

## 4. Visual 模型（v12.14 提升 token-per-frame）

```
AGENT_CHUNK_SEC          = 1
FRAMES_PER_CHUNK         = 2          # 2 fps
# v12.15 (vs v12.14):
VISUAL_WINDOW_CHUNKS     = 8          # 8 chunks × 1s = 8s 滑窗
VISUAL_WINDOW_FRAMES     = 16
min_pixels (RUNTIME)     = 200_704    # 256 × 28 × 28
max_pixels (RUNTIME)     = 401_408    # 512 × 28 × 28
VISUAL_TOKENS_PER_FRAME  = 235        # conservative runtime estimate
VISUAL_TOKENS_PER_CHUNK  = 470        # 235 × 2 (was 128)
VISUAL_WINDOW_TOKENS     = 3,760      # 8 × 470

# Recall (二次 pass) 更高分辨率：
min_pixels (RECALL)      = 200_000    # ~480p area floor
max_pixels (RECALL)      = 1_500_000  # 1280×720 fully preserved
RECALL_VISION_TOKENS     = 940        # 4 frames × 235
```

视觉窗每 chunk 滑动：第 N 个 chunk 看 `[max(0,N-15), N]`，共 32 帧。

---

## 5. Recall 模型（v12.13 D1 chunk-internal multi-turn）

**SFT 数据形态 B**：assistant 输出 `<tool_call>{"name":"recall","arguments":
{"query","time_range"}}</tool_call>` → system 检索历史 think → 下一个 user
message 注入 `<recalled_frames>` + 帧 + `<recall_result>` → assistant 给最终
answer。**单 chunk 内 2-turn**，loss-mask 同时覆盖两个 assistant span。

**RL 端 (v12.13)**：`streaming_agent_loop.py` 实现 D1 chunk-internal recall
完整在线流程，包括：
- 历史帧的 MROPE 锚定到原 chunk × FRAMES_PER_CHUNK（不是当前 chunk）
- TF-IDF retriever 选 top-k chunks 的 think
- 限制每 chunk 最多 1 次 recall（`THINKSTREAM_MAX_RECALL_PER_CHUNK=1`）

`recalled_frames time_range` 文本 tag + `video_start/video_end` metadata
让模型在语义层面理解这些是历史帧。**MROPE position id 严格复用**已在 RL
recall 路径上实现（`historical_frames_indices` + 原始 anchor）。

---

## 6. 输入布局（位置编码）

**MROPE 现状（RL 路径）**：
- 视觉 temporal index 用 timestamps 编码
- D1 recall 历史帧 `frames_indices` = 原 chunk × FRAMES_PER_CHUNK（已对齐）
- 文本 temporal override：`thinkstream/data/rope2d.py:376` helper 已接入
  RL streaming_agent_loop（v12.14）

**每步消息布局**（视频在前、文本在后）：

```
[system + tools]                                          ← chat_template 自动渲染 <tools>
<visual_window>{json header}</visual_window>  +  video_block (16 frames @ 8s window)
<recalled_frames>{header}</recalled_frames>   +  recalled_video    (recall 二次 pass 才有)
<memory>
  <compressed t="a-b">{json}</compressed> × n_seg              (≤ 5)
  [time] think text × n_recent                                  (≤ ~57)
</memory>
<queries>...</queries>                                          (历史已答 Q&A)
<recall_result>{json}</recall_result>                           (recall 二次 pass 才有)
<user_input>{question or compress_trigger}</user_input>
```

**inter-chunk compress 样本**（`v12_inter_chunk=True`）**没有 visual_window
也没有 video frames** — 压缩是 chunk 之间的事件，不消耗一个视觉时间步。

---

## 7. SFT（单阶段）

```
pretrained Qwen3-VL-8B-Instruct → SFT (3 epoch, train_sft.jsonl)
                                ↓
                   SFT ckpt → GRPO RL (1 epoch, train_rl.jsonl)
```

**v9.2 决策（不可逆）**：废弃 5-stage（SFT + C1 + C1-D + C2 + C2-D），改为
**1 SFT + 1 GRPO RL**。调研 8 篇 2026 同期工作（ReMemR1 / Mem-α / MARC /
AgeMem / SUMER / Memex / Dispider / StreamingVLM）：0/8 用多阶段 RL，0/8 用
DAgger，5/8 内存类工作直接 0 SFT + 1 RL。

**SFT 损失**（v12.5 起）：vanilla CE on assistant span（DeepEyesV2 / Qwen-VL
官方约定）。**没有 token-span 加权**，**没有类别加权采样**。v12 协议没有
`<action>` 这种单一关键词位置可加权，且 v12.5+ 数据自然分布够平衡。

**SFT 数据三种 shape**（`thinkstream/sft/data_processor.py`
`build_per_timestep_messages_v12`）：
- A. **单步**（silent / response / lonely recall）：`[system, user, assistant]`
- B. **recall 多轮**（chunk 内）：`[system, user, assistant(tool_call),
   user(tool_result), assistant(answer)]`
- C. **inter-chunk compress**：`[system, user(memory + compress_trigger),
   assistant(tool_call)]`

loss-mask 实现（与 VST / LiveCC / Qwen-VL 官方一致）：扫
`<|im_start|>assistant` 到 `<|im_end|>`，仅这段开 loss。形态 B 允许 2 个
assistant span 都开 loss。

**Qwen3-VL 强制依赖**：v12 协议依赖 Qwen3-VL 官方 `<tool_call>` 特殊 token
（151657/151658）。Qwen2.5-VL 的 `chat_template.json` 没有 `<tools>` 渲染、
没有 `<tool_call>` block，会**静默失败**。`train.py` 有硬 guard：
`protocol_version=v12 AND model_type != qwen3vl → RuntimeError`。

---

## 8. RL（verl 0.4 + thinkstream recipe）

### 8.1 框架选择

**verl 0.4** + 自定义 `recipe_thinkstream/`（不再是 v11 的 `slyme/`）：
- `verl/recipe_thinkstream/streaming_agent_loop.py` (~1500 行) — 流视频
  agent loop，注册为 verl 的 `AgentLoopBase` 子类
- `verl/recipe_thinkstream/thinkstream.py` — `compute_score` 入口
- `verl/recipe_thinkstream/configs/thinkstream_grpo.yaml` — config
- `verl/recipe_thinkstream/run_thinkstream_grpo.sh` — 启动脚本

### 8.2 默认 stitched 模式

`MAX_TURNS=120` 覆盖 batch1 max=95 chunks + lower batch2。
`MAX_RESP_LEN=32768` 是 trajectory 总 response 长度（所有 chunk stitched）。

**单 trajectory** = 整段视频的所有 chunk → 一个 `AgentLoopOutput`，
response_ids = 所有 assistant action stitched 起来。Reward 在 trajectory
末端整体打分，标准 GRPO 走 group_by uid。

### 8.3 Multi-Q 模式（OVOBench 对齐）

启用：`MULTI_Q=1` + parquet 用 `--multi_q` 构造。

每个 trajectory 对应**一个视频** + N 个 MCQ。answer 归属算法（3 阶段）：
1. **Window**: ask_chunk ± window 内 emit 的 answer 优先归到该 Q
2. **LIFO**: 同 chunk 多 emit → 后 emit 优先（最新答案覆盖）
3. **FIFO 兜底**: 跨 window emit 按到达顺序分给未填 Q

最终结果通过 `ts_per_q_answer_chunk[]` + `ts_per_q_answer_text[]` extras
传给 `compute_score`，后者按 `answer_form` 分发到 5 种 matcher。

### 8.4 Reward 8 路 column（v11.4 设计 + v12 协议适配）

| reward | 权重 | 说明 |
|---|---|---|
| correctness | 0.30 | response 主结果（5 种 form-aware matcher） |
| silent_quality | 0.18 | streaming-specific dense reward |
| timing | 0.18 | last response chunk 距 gold ask_chunk（last，不是 first） |
| recall_quality | 0.05 | query JSON 格式 + 无 leakage |
| recall_hit_rate | 0.07 | `\|returned ∩ support\| / \|support\|`，失败显式 -0.2 |
| range_tightness | 0.03 | `(1 - range_width/duration) × coverage`，coverage=0 显式 -0.2 |
| format | 0.09 | 协议合规率（含 v12 tool_call JSON） |
| overflow_pen | 0.10 | compress timing soft penalty |

**Per-reward group-norm**（GDPO）独立做 mean-only 不归 σ，权重控制相对拉力。
Bimodal 分布（format ∈ {0,1}）下 σ 不稳，归 σ 反而注入噪音。

**Mask=0 vs mask=1+负值**（v11.4 关键修复）：失败模式（recall fired 但
retriever 返回 ∅；query time_range 覆盖 0 gold chunks）返回 -0.2 with
mask=1，让 policy gradient 能惩罚它们；不可应用的组件才返回 mask=0。

### 8.5 v12.14 Recurrent 模式（EXPERIMENTAL）

**触发条件**：>180 chunk 视频，stitched 路径会遇到 token 容量上限或 OOM。

启用：
```bash
THINKSTREAM_RECURRENT_MODE=recurrent \
MAX_RESP_LEN=4096 \    # ← per-action cap，不是 stitched 的 32768
MULTI_Q=1 \
bash verl/recipe_thinkstream/run_thinkstream_grpo.sh
```

**架构**（对齐 MemAgent / ReMemR1）：
- `streaming_agent_loop.py` Phase 3 dispatch：每个 assistant action 输出一个
  `AgentLoopOutput`，trajectory → `list[AgentLoopOutput]`
- `AgentLoopWorker.generate_sequences` Phase 1：flatten list，写入
  `sample_index` (LongTensor) + `final_mask` (BoolTensor) 到 batch
- `ray_trainer.fit()` Phase 4d：
  1. swap：`batch = expanded gen_batch_output`（sum(K_i) 行），non_tensor_batch
     通过 sample_index 重索引
  2. 保留 `original_batch`（B*n 行，含 uid）
  3. 抽 trajectory 级 reward：从 `final_mask=True` 的行抽 rm_scores，按
     `reverse_indices(sample_index[final_mask])` 重排回输入顺序 → `[B*n, R]`
  4. 在 `actor_world_size` 维度 pad 一次（pad_dataproto_to_divisor），
     reward_tensor + reward_extra_infos_dict 用 head replication 同步 pad，
     padded rows 的 `response_mask=0` 让它们在 loss/adv 中归零
  5. **不 unpad**：padded batch 一路穿过 old_log_prob / ref / values /
     update_critic / update_actor，FSDP `DataProto.chunk()` 整除有保证
  6. `compute_1D_grpo_advantage`(reward_traj_tensor, original_batch.uid) →
     `[B*n]` scalar advantage；通过 sample_index broadcast 回 action rows，
     tile across response_length × response_mask

**不严格等价 stitched**：
- format/spam 只算 final action 的 solution_str（不是 stitched trajectory）
- non-final actions 的 score 进 telemetry（`recurrent/nonfinal_action_score_*`）
  但不参与 advantage
- 必须 `rollout.n ≥ 2`，n=1 退化为 singleton GRPO 组

**测试覆盖**：
- `scripts/test_rl/test_phase4_recurrent_advantage.py` — 数学
- `scripts/test_rl/test_phase4_dataproto_integration.py` — 真 DataProto
  swap / pad / final_batch / 1D adv broadcast
- `scripts/test_rl/run_all.sh` — stitched-default mini RL harness

**未验证**：真 Ray/FSDP 多 GPU actor/ref dispatch、actor loss 在 broadcasted
advantage + masked padded rows 下的稳定性。

---

## 9. Inference / Eval

### 9.1 Streaming inference loop

`thinkstream/eval/streaming_vllm.py`（fresh-KV-per-chunk，对应 SFT 训练分布）：

```python
for chunk_idx in range(num_chunks):
    user_content = build_user(memory_state, queries, visual_window[N-15:N], frame_paths)
    response = vllm.generate(prompt=system + user_content, ...)
    parsed = parse_v12(response.text)
    if parsed.kind == "recall":
        recall_result = retrieve(parsed.tool_call.arguments)
        final_response = vllm.generate(prompt + response.text + recall_user, ...)
    elif parsed.kind == "compress":
        memory_state.apply_compress(parsed.tool_call.arguments)
    elif parsed.kind == "answer":
        record_response(...)
```

每个 chunk 的 user content 自带完整历史（memory text + queries text）→
KV 不需要跨 chunk 复用，prefix cache 命中 system+tools 即可（实测 93.8%
visual KV 命中）。memory text > visual KV horizon（57s vs 16s），保证被
踢出 KV 的老 think 仍在下一轮 user content 里。

### 9.2 Eval 入口（4 个）

```
thinkstream/eval/
  ovo_bench/
    eval_ovo.py                 — wrapper
    eval_ovo_baseline.py        — non-streaming baseline
    eval_ovo_offline.py         — streaming offline (transformers)
    eval_ovo_offline_vllm.py    — streaming offline (vLLM, fast)
    eval_ovo_streaming_vllm.py  — true streaming (vLLM)
  rtvu/
    eval_rtvu.py                — RTVU streaming eval
  streaming_vllm.py             — shared streaming engine
  vllm_engine.py                — vLLM wrapper
```

### 9.3 两个 profile（`scripts/eval/eval_profiles.py`）

| | 16k 默认 | 32k 扩展 |
|---|---|---|
| `model_max_length` | 16384 | 32768 |
| `QUERIES_HISTORY_CAP` | 8 | 24 |
| `RECALL_TEXT_MAX_CHARS` | 800 | 3000 |
| `max_new_tokens` | 128 | 256 |

profile 通过 `apply_profile()` runtime 改 agent_protocol globals，**不影响
SFT-baked 常量**（pass2/3 数据生成时已经焊死的字段不变）。

### 9.4 统一 matcher（v12.14 收敛）

**RL `compute_score`、SFT-eval、OVOBench、RTVU 共用一份**
`thinkstream/trainer/outcome_match.py`。这避免了「RL 给 0，eval 给 1」的
reward gap（v11.x 时反复出现）。

### 9.5 Streaming 测试维度（每 ckpt × bench 跑 32 配置）

- retriever ∈ {bm25, hybrid}，hybrid 可加 `--use_agent_vision`
- compress_mode ∈ {system, self}
- scoring ∈ {strict, lenient}
- profile ∈ {16k, 32k}

---

## 10. Telemetry

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

**v12.14 Recurrent 专属**：
```
recurrent/swap_fired                = 1.0 confirms swap path
recurrent/expanded_rows             = sum(K_i)
recurrent/n_trajectories            = B*n
recurrent/avg_actions_per_traj
recurrent/pad_size
recurrent/traj_reward_mean / _std
recurrent/nonfinal_action_score_*   informational, not used for adv
recurrent/adv_mean / _std
```

---

## 11. 关键 quick-reference 常量（v12.14）

| 类型 | 名 | 值 | 来源 |
|---|---|---|---|
| chunk | `AGENT_CHUNK_SEC` | **1** | config.py |
| chunk | `FRAMES_PER_CHUNK` | 2 | config.py |
| chunk | `VISUAL_WINDOW_CHUNKS` | **8** | config.py |
| memory | `RECENT_THINKS_TOKEN_BUDGET` | **4000** | config.py |
| memory | `COMPRESS_TOKEN_THRESHOLD` | 3200 | config.py |
| memory | `COMPRESS_HYSTERESIS_THRESHOLD` | 2200 | config.py |
| memory | `COMPRESS_RANGE_MIN/MAX` | 8 / 24 | config.py |
| memory | `MAX_COMPRESSED_SEGMENTS` | 5 | config.py |
| memory | `SUMMARY_TOKENS_MAX` | **280** | config.py |
| think | `THINK_TOKENS` | (40, 80) | config.py |
| think | `THINK_TOKEN_AVG` | 60 | config.py |
| visual | RUNTIME `min_pixels` / `max_pixels` | **200704 / 401408** | config.py (v12.15) |
| visual | RECALL `min_pixels` / `max_pixels` | 200k / 1.5M | config.py |
| visual | `VISUAL_TOKENS_PER_FRAME_RUNTIME` | **235** | config.py (v12.14, was 128) |
| visual | `VISUAL_TOKENS_PER_CHUNK` | **470** | config.py (v12.14, was 128) |
| visual | `VISUAL_WINDOW_TOKENS` | **3,760** | config.py (v12.15) |
| visual | `RECALL_VISION_TOKENS` | 940 | config.py |
| budget | `MAX_SAMPLE_TOKENS` | **16384** | config.py |
| budget | `SYSTEM_PROMPT_TOKENS` | 400 | config.py |
| eval | `QUERIES_HISTORY_CAP` | 8 (16k) / 24 (32k) | profile |
| eval | `RECALL_TEXT_MAX_CHARS` | 800 / 3000 | profile |
| eval | `model_max_length` | 16384 / 32768 | profile |
| eval | `max_new_tokens` | 128 / 256 | profile |
| RL | `MAX_TURNS` (stitched) | 120 | run_thinkstream_grpo.sh |
| RL | `MAX_RESP_LEN` (stitched) | 32768 | run_thinkstream_grpo.sh |
| RL | `MAX_RESP_LEN` (recurrent) | **4096** | per-action cap |
| RL | `THINKSTREAM_MAX_RECALL_PER_CHUNK` | 1 | run_thinkstream_grpo.sh |
| pass1 perf | pass1a concurrent | 1024 | config.py |
| pass1 perf | httpx max_connections | 2048 | vllm_client.py |

**SFT 训练框架**：Qwen-VL 官方 finetune fork (`thinkstream/sft/`)，不依赖
LLaMA-Factory（但数据 messages 格式兼容 LLaMA-Factory CLI）。
**RL 训练框架**：verl 0.4 + `recipe_thinkstream/` (multi-Q + D1 recall +
v12.14 recurrent dispatch)。

---

## 12. 版本时间轴

| 版本 | 主要变更 |
|---|---|
| **v12.14** (2026-05-03) | Recurrent rollout (Phase 1-4); pad-once-mask-through FSDP; 1D GRPO advantage by uid; sample_index broadcast; `discarded_nonfinal` → `nonfinal_action_score_*` |
| **v12.15** (2026-05-12) | Visual/KV window aligned to 8 video chunks; runtime video pixel range restored to 200704/401408 |
| **v12.13** (2026-05-02) | D1 chunk-internal recall + 历史 MROPE; Multi-Q 3-stage attribution; 5 form-aware matchers (binary/MC/number/short/desc); 统一 matcher 闭 RL/eval gap |
| **v12.12** (2026-04-30) | RUNTIME mm_processor_kwargs profile; visual token-per-frame 128 → 235; `VISUAL_WINDOW_TOKENS` 2048 → 7520 |
| **v12.5** (2026-04-29) | chunk_sec 2 → 1; 视觉窗 12 → 16 chunks; text memory 600 → 4000 token; pass5 messages 转换层 |
| **v12.0** (2026-04-26) | Qwen3-VL 官方 tool protocol（`<tool_call>` JSON + `<answer>`）；删除 v11 vocab 扩展；vanilla CE |
| **v11.4** (2026-04-22) | GDPO mean-only group-norm; mask=0 vs negative-with-mask=1 区分; timing 用 last_response |
| **v11.3** (2026-04-20) | SFT rebalance: compress 0.8 → 2.5, smoothing 0.7 → 1.0 |
| **v9.2** (2026-04-15) | 5-stage → 1 SFT + 1 GRPO RL（依据 8 篇 2026 同期工作） |

---

## 13. 与同期工作对齐

- **DeepEyesV2 / Qwen-VL 官方 / VST / LiveCC**：system+user+assistant 三角色
  （无 tool/observation 角色），tool 输出注入下一 user content；loss-mask
  扫 `<|im_start|>assistant` 到 `<|im_end|>`，仅此段开 loss。我们的
  `data_processor.preprocess_per_timestep` 与官方实现行为一致。
- **MemAgent / ReMemR1** (recurrent 范式)：v12.14 Recurrent 路径核心
  （swap + 1D GRPO + sample_index broadcast）byte-equivalent 对齐。
  ThinkStream 是该框架的第一个 streaming-video 用户（MemAgent / ReMemR1
  都是 text-only）。
- **OVOBench**：multi-Q 模式（1 video → N MCQ at different timepoints）
  原生支持，统一 matcher 让 RL reward 和 OVOBench scoring 严格一致。
