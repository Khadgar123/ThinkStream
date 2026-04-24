# 流视频 Agent 数据构造方案

> 版本: v9.1 | 日期: 2026-04-24
>
> 核心设计：Per-timestep 独立样本 | 统一时间线记忆 | 视频在前文本在后 | MROPE 时间对齐
> 80% token 触发压缩 (C1→C2) | Recall 两步 (query→response) | 3 专用 Prompt
>
> 配套：`sft_engineering.md` (SFT v3.0) | `streaming_position_encoding.md` (位置编码 v1.0)

---

## 1. 架构原则

### 1.1 训练格式：Per-Timestep Re-render

每个 2s chunk 构造为独立训练样本（单轮 messages），只对 assistant content 计算 loss：

```
messages: [
  {role: system, content: 3-action protocol},
  {role: user, content: [
    {type: video, video_start: 26, video_end: 50, nframes: 24},   // Zone B 视频在前
    {type: text, text: "<memory>...</memory>"},                    // Zone C 文本在后
    {type: text, text: "<user_input>Question?</user_input>"},      // Zone D
  ]},
  {role: assistant, content: "<think>...</think><action>response</action><response>...</response>"}
]
```

**为什么不用多轮对话**：(1) 压缩后旧 KV 无法替换 (2) mask≠删除，模型不学 summary (3) KV 无限增长 (4) Re-render H100 ~200ms/step，远在 2s 内 (5) 旧 thinks 不在 input，summary 成唯一信息源

### 1.2 三层文本分离

| 层 | 内容 | 是否进 SFT |
|---|------|:---:|
| teacher_caption | 397B 详细结构化事实（造题/验证/标准答案） | **否** |
| student_think | 每步 40-60 tok 增量视觉记忆 → 存 memory | 是（output） |
| compressed_summary | 压缩摘要，替换旧 thinks | 是（input） |

三者不混：teacher 不进 student 路径；recall_result 不含 teacher 信息；SFT output 基于 student 可见信息回答。

### 1.3 三 Prompt 协议

| Prompt | 可选 action | think 存 memory | 触发方式 |
|--------|-----------|:---:|---------|
| **SYSTEM_PROMPT**（主循环） | silent / response / recall | **是** | 每个时间步 |
| **POST_RECALL_PROMPT** | silent / response | **否**（分析结果） | recall 返回后 |
| **COMPRESS_PROMPT** | — (输出 summary) | **否** | 系统触发（两步之间） |

时序：`chunk N → think → [compress?] → chunk N+1`。Recall 在同 chunk 内完成（query → 系统检索 → response）。

### 1.4 Think 与 Response 规范

| 参数 | 规则 |
|------|------|
| Think 长度 | 40-60 tokens |
| Think 内容 | 当前 2s 帧的增量可观察事实（实体/动作/OCR/空间） |
| Think 禁止 | 声音/气味/情绪/意图/元推理 |
| Response 长度 | factoid 5-40 / procedural 40-120 / summary 80-200 tok |

---

## 2. 记忆架构

### 2.1 统一时间线

summary 和 think 混排，按时间顺序。压缩 = 原位替换（选中 thinks 被 summary 替换）。

```
memory_timeline = [
  <summary t="0-16">Chef prepared workspace...</summary>,   ← 压缩段
  [16-18] Garlic browned in oil,                             ← 未压缩 think
  [18-20] Tomato quarters added to pot,
  <summary t="20-30">Seasoning added...</summary>,
  [30-32] Chef covers pot with glass lid,
]
queries = [{question: "...", answers: [...]}]                ← 独立追踪区
```

**核心**：文本记忆覆盖时间 > 视觉窗口。Think 每步立即追加到末尾。Recall = 系统侧检索 retrieval_archive。

### 2.2 压缩

**触发**：thinks 总 token ≥ 80% budget (~480 tok)。Hysteresis: 压缩后降至 55% (~330 tok)。

**C1 范围选择**：5 维归一化评分（content 0.30 / merge 0.20 / boundary 0.15 / recency 0.20 / token 0.15），选最低分范围压缩。C2: 模型自选范围。

**二级压缩**：summary > 5 段时合并最老两段（≤200 tok）。

### 2.3 Token 预算

| 组件 | Tokens |
|------|--------|
| System prompt + 压缩段 (≤5×150) + 未压缩 thinks (≤12×50) | ~1500 |
| Visual window (12 chunks) + User input | ~1636 |
| Recall result (含 4 帧) | ~356 |
| **输入总计 ~3400** · 输出 ~80-280 · **单样本 ~3500-3700** | 远在 16K 内 |

---

## 3. 数据构造 Pipeline

```
阶段 1: Teacher Evidence Graph          [397B + 视频, ~5min]
  ├─ 1-A: 独立 chunk 标注              [397B, 2帧/chunk, 全并行]
  └─ 1-B: 实体对齐 + 变化检测          [纯文本 397B, 2 call/video]
阶段 2: Question-blind Streaming Rollout [397B + 视频, ~1h]
阶段 3: Task Mining + Sample Generation  [397B + 规则, ~30min]
  ├─ 3-A: Task Card 生成               [397B, ~6-8 call/video, asyncio.gather, ~2s]
  ├─ 3-B: Placement + 行为序列规划     [纯程序, 贪心多样性选择]
  └─ 3-C: 轨迹样本生成 + base 采样     [397B, ~15-25 call/video, fork+base 合并]
阶段 4: Verify + Filter                 [纯规则, 14 项检查, per-video 保存]
```

### 3.1 阶段 1: Teacher Evidence Graph

**1-A**：每 chunk 独立标注（2 帧，无前文），输出 `visible_entities` / `atomic_facts` / `ocr` / `spatial`。视频内全并行，每请求 ~1.5K tok。

**1-B**：两次纯文本 397B call — (1) 实体对齐→写回 `entity["id"]`（粗筛 hint） (2) 变化检测→写回 `state_changes`。entity_id 可能出错，下游只做候选粗筛，Task Card 问题文本必须用 desc 不用 id。

### 3.2 阶段 2: Question-blind Rollout

模拟学生流式体验（无问题介入），产出 thinks + compression_events + snapshots。视频内串行，视频间 128 并行，~1h/300 videos。

### 3.3 阶段 3-A: Task Card 生成

**目标**：从 evidence 提取"可问什么"，生成 Task Card。只管 WHAT，不管 WHEN/HOW。

#### Task Card 结构

```python
TaskCard = {
    "card_id": "vid001_F2_001",
    "family": "F2",                                    # → retention_class / 3-B 逻辑
    "question": "围裙什么颜色？ A.红 B.蓝 C.白 D.绿",   # MC 选项嵌入问题
    "canonical_answer": "A",                            # binary→Yes/No, MC→字母, number→数字
    "answer_form": "multiple_choice",
    "support_chunks": [5],
    "visibility_type": "persistent",                    # persistent→只 in_visual; transient→可 recall
}
```

#### 问题家族

| Family | 名称 | retention | 优先 answer_form |
|--------|------|:---------:|-----------------|
| F1 | OCR/Number | low | number, short_exact |
| F2 | Attribute | low | MC, binary |
| F3 | Count | low | number |
| F4 | Spatial | medium | binary, MC |
| E1 | Action | high | binary, short_exact |
| E2 | Change | medium | binary |
| P1 | Procedure | medium | number, MC |
| C1 | Compare | medium | binary |
| R1 | Re-id | medium | binary |
| S1 | Summary | high | descriptive |
| M1 | Commentary | high | descriptive |

#### Step 1: 结构化筛选（双通路）

`classify_chunks(evidence)` → `{family: [chunk_indices]}`。主路用 1-B 字段，P1/C1/R1 增加 1-A 直接视觉字段备路：

| Family | 主路 | 备路（1-A fallback） |
|--------|------|-------------------|
| F1 | `ocr` 非空 或 facts 含有效数字 `\d{2,}` | — |
| F3 | facts 含有效数字（**独立于 F1**） | — |
| E1 | 有 atomic_facts 的 chunk 动态步长采样 | — |
| P1 | 连续 ≥3 有 `state_changes` | 连续 ≥3 chunk 的 entity `action` 各不同 |
| C1 | 同 `entity_id` + `state_change` | desc 词重叠 ≥0.6 + action 不同 |
| R1 | 同 `entity_id` gap ≥5 | desc 词重叠 ≥0.6 + gap ≥5 |

#### Step 2: asyncio.gather 并发调 397B

所有 family 独立并行（~6-8 call/video，~2s）。共享 `_OUTPUT_SCHEMA` 尾部：JSON schema + canonical_answer 格式规则 + entity ID 禁令。

**关键词提取** `extract_card_keywords(card)`：binary→从 question 提取，MC→question 主体+正确选项，number→question+数字，short_exact/descriptive→canonical_answer。

并发 512（纯文本，batch budget 支持 ~1882）。

### 3.4 阶段 3-B: Placement + 行为序列规划

纯程序，零 397B。输入 cards + rollout，输出 placements + trajectories。

#### 行为序列

| 序列类型 | 行为链 | fork 样本数 |
|---------|--------|:---------:|
| immediate_response | response → post_silent | 2 |
| recall_success | recall → post_recall_response → post_silent | 3 |
| recall_fail_then_found | recall → post_recall_silent → wait×N → found_response → **post_silent** | 5+ |
| event_watch | silent → wait×N → trigger_response | 3+ |
| multi_response | response → no_change_silent×N → followup_response×N（**含 prior_answers 去重**） | 4+ |

#### availability 分类

`classify_availability` 查 snapshot 判断答案可达性。`precompute_retention` 使用 `extract_card_keywords(card)` 匹配 think/summary 文本（修复了 binary/MC 的 keyword 全空问题）。

`in_visual` / `in_recent_thinks` / `in_compressed` → immediate_response；`in_history_only` → recall_success；`in_future` → event_watch。

#### 轨迹规划：贪心多样性选择

`plan_trajectories(placements, cards_map, target=5, max_pp=5, min_gap=8)`：

**Phase 1: 贪心选择**（budget=25）— 每轮选最高分 placement：

| 维度 | 分值 |
|------|------|
| 可自动验证的 answer_form | +1.0 |
| 未见 family / sequence_type | +2.0 each |
| 时间分散 (距已选最小距离/10) | +0~1.5 |
| support_inferred | -2.0 |

同 card_id 不重复。**Phase 2: 组合**— 按 ask_chunk 排序，间隔 ≥8 chunks 分入同 trajectory（最多 5 个/条）。

### 3.5 阶段 3-C: 轨迹样本生成

输入 trajectories + rollout + evidence + cards_map → 输出 `samples_3c/{video_id}.json`

#### Fork 样本

遍历 key_chunks，维护 `queries_state[]` 演化。397B 生成 response/query 文本。Recall 噪声：70% oracle / 20% noisy / 5% distractor / 5% failure。

#### Base 样本：5 类选择性采样

`_select_base_chunks()` 从 60 chunks 选 ~20-35 个（非全选）：

| 类别 | 规则 | 目的 |
|------|------|------|
| Warmup | chunk 0-2 | 冷启动 |
| Evidence anchor | support_chunks ± 2 | recall 证据位置 |
| Question window | 每个 key_chunk ± 2/+3 | Q&A 上下文 |
| Compress anchor | trigger ± 1 + 被压缩首尾各 2 | 压缩上下文 |
| Long-silent patrol | 间隔 >10 chunks 每 5 个采 1 | 持续 silent 行为 |

`generate_base_samples()` 从最近 fork 边界插值 `queries_state`。Fork + base 合并排序。

**action 比例**（5Q trajectory）：silent ~60% / response+recall ~30% / compress ~10%。

**调用量**：~5 traj × ~3-5 calls = ~15-25 calls/video（旧 ~90 → 降 75%）。

### 3.6 阶段 4: Verify + Filter

`pass4_verify.py` — 14 项检查，按 trajectory 分组，per-video 保存 `verified/{video_id}.json`。

**Per-sample 检查 (1-10)**：information_flow / action_minimality / grounding / format / think_token_length / compression_ratio / summary_provenance / summary_retention / summary_no_current_think_leak / question_answer_leakage — 均适配 base sample（跳过或放宽）。

**新增检查 (11-14)**：queries_state_temporal（结构合法）/ trajectory_action_distribution（多 Q trajectory silent ≤90%）/ base_sample_consistency（silent/compress only）/ recall_evidence_reachable（support < ask）。

**Difficulty**：easy (base silent 无 queries / immediate_response) → medium (有 queries 的 silent / recall_success / compress) → hard (recall_fail_then_found)。

**数据切分**：按 video_id 切分（同视频不跨 split），300 → 240 train / 30 val / 30 test。

---

## 4. RL 阶段 (GRPO)

SFT Phase 5 完成后，在独立 RL 视频集（75 条）上 GRPO 训练。

**Reward**：R_format(0.15) + R_action(0.20) + R_correctness(0.30) + R_timing(0.15) + R_think_len(0.10) + R_compress(0.10)。答案验证 V1-V4 自动（binary/MC/number/short_exact ≥60%），V5-V6 LLM judge。

**配置**：75 videos / ~2050 (video,task) 对 / G=4 / batch=16 / 2 epochs / ~256 steps / LR=5e-7。

**迁移**：`grpo.py` rollout 需从多轮 KV cache 迁移到 per-timestep re-render。

**评估**：action macro-F1 / recall@1,3 / entity retention / QA accuracy (frame/memory/compressed/recall) / latency/chunk。

---

## 5. 配置与部署

### 5.1 核心常量

```python
THINK_TOKENS = (40, 60);  VISUAL_WINDOW_CHUNKS = 12  # 24s, 24帧
RECENT_THINKS_TOKEN_BUDGET = 600;  COMPRESS_TRIGGER_RATIO = 0.8  # threshold=480
COMPRESS_RANGE = (2, 8);  MAX_COMPRESSED_SEGMENTS = 5;  COMPRESSION_RATIO_MIN = 2.5
SUMMARY_TOKENS = (100, 180);  RECALL_RETURN_FRAMES = 4;  MAX_LENGTH = 16384
PASS1_CONCURRENT = 1024;  PASS2_CONCURRENT = 128
PASS3A_CONCURRENT = 512;  PASS3C_CONCURRENT = 1024
STUDENT_MODEL = "Qwen/Qwen3-VL-8B";  TEACHER_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
```

详细参数以 `config.py` 为准。

### 5.2 vLLM 部署

统一 `thinking=True, --reasoning-parser qwen3`。视觉 Pass: max-model-len=65536, max-num-seqs=16。文本 Pass: max-model-len=16384, max-num-seqs=64, prefix-caching ON。两套实例按序启动。

### 5.3 数据源

基于 `video_catalog_30s_plus.csv`（≥30s 全量 668K）。三档：<24s 不可用 / 30-89s Phase1-2 / ≥90s 全 Phase。P1(200) + P2(200) + C1/C2/P5(300) = ~700 视频 → ~33K samples。按 video_id 切分，同视频不跨 split。

### 5.4 Special Tokens

基础: `<think>` `<action>` `<silent>` `<response>` `<query>` `<recall_result>` + 闭合。
扩展: `<memory>` `<compressed>` `<queries>` `<answer>` `<visual_window>` `<recalled_frames>` `<user_input>` `<summary>` `<compress_trigger>` + 闭合。
JSON 放在 tag 内，不用 XML attribute。

---

## 6. 附录

### 6.1 核心约束（不可违反）

| # | 约束 | 代码执行 |
|---|------|---------|
| 1 | think 每步立即入 memory（recall step2 不存） | `MemoryState.add_think()` |
| 2 | 文本记忆覆盖时间 > 视觉窗口 | `snapshot().visual_window_start` |
| 3 | 80% token 预算触发压缩 | `should_compress() + count_recent_tokens()` |
| 4 | compress range 不含当前 think | `pre_action_thinks` |
| 5 | 先替换→再 append 当前 think | `memory.compress()` 后 `add_think()` |
| 6 | summary 只引用 thinks + 可见帧 | `verify_summary_provenance()` |
| 7 | C1 teacher 评分选范围，C2 模型自选 | `score_range_for_compression()` |
| 8 | action 优先级：用户问题 > query 触发 > silent | Pass 3-C |
| 9 | recall_result 不含 teacher/未来内容 | `_simulate_recall_result()` |
| 10 | recall fail 时 response 表达不确定 | `verify_information_flow()` |
| 11 | query generator 不接收 gold_answer | `RECALL_QUERY_PROMPT` |
| 12 | summary >5 段时合并最老两段 | `MemoryState.compress()` |
| 13 | 压缩比 ≥ 2.5 | `verify_compression_ratio()` |
| 14 | C1/C2 拆开训练 | 独立 prompt |
| 15 | teacher 信息只用于选 gold | evidence 不进 student 路径 |
| 16 | summary 不含当前 think 独有事实 | `verify_summary_no_current_think_leak()` |
| 17 | Pass 2 question-blind | 无 question/task 参数 |
| 18 | base sample 只能 silent/compress | `verify_base_sample_consistency()` |
| 19 | queries_state 不含未来问题 | `verify_queries_state_temporal()` |
| 20 | recall support_chunks < ask_chunk | `verify_recall_evidence_reachable()` |

### 6.2 开放问题

1. Per-timestep vs 短序列：先用 per-timestep baseline，缺连续性时改 3-step 滑动窗口
2. Think 关键信息覆盖率：Pass 2 后检查，覆盖率 < 0.6 重新生成
3. RL rollout 迁移：`grpo.py` → `StreamingAgentLoop.step()`
