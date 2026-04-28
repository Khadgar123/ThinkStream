# ThinkStream 设计（当前状态：v9.5 + v11.3 + v11.4 增量）

> Source of Truth: `git log` + `tests/` + `memory/`. 本文档是 **快照**——
> 当代码与本文档分歧时以代码为准。最新 commit 见 `git log --oneline -10`。
>
> **v11.4 RL 增量见 [`docs/v11.4_rl_design.md`](v11.4_rl_design.md)** ——
> RL 数学原理（GRPO vs GDPO 公式推导）、3 处 reward bug 修复（timing 用
> last response、recall 失败显式 -0.2、range coverage=0 显式 -0.2）、
> `aggregate_grpo` 作 DeepSeekMath baseline 可选模式、vLLM batched
> streaming rollout（chunk-lockstep × group_size，opt-in）。
>
> **v11.3 增量见 [`docs/v11.3_changelog.md`](v11.3_changelog.md)** ——
> SFT 权重重平衡（compress 0.8→2.5 等）、wandb 每类监控、token 驱动
> 压缩范围、pass3 5 处非 think、vLLM eval（offline + streaming
> chunk-lockstep）、多方案 recall 检索 audit、JSON 解析鲁棒化、
> 0-chunk 过滤 + unique-think-rate 加权。本节以下仍是 v9.5 快照。
>
> v9.4.2 → v9.5 主要变化：
> - 22 family（新增 F7/CR5/CR6/CR7 对齐 OVO SSR/CRR + STAR/PerceptionTest）
> - pass1a strict parse + retry + thinking off + 字段瘦身（去 confidence/target_resolution_visible）
> - pass1b 合并 LLM 调用（entity_link + state_change 一次）
> - 多 probe 进度感知答案（F5/F7/CR5 multi_response progressive_answers）
> - RL reward 8 路（拆 recall_quality / hit_rate / range_tightness）
> - eval retriever 默认 hybrid + 可选 agent 自身 vision encoder
> - retriever 加 time_range 过滤
> - debug_streaming.py 已删（telemetry 嵌入 walk_and_score）

## 1. 项目目标

流式视频 Agent：每 2 秒（一个 chunk）做一次决策，输出 `silent / response / recall / compress` 之一，配合内部记忆状态在长视频中持续做问答。

数据 + 训练 + 推理三端共用 **per-timestep snapshot 格式**：训练样本 = 推理某一步的精确 input。

## 2. 数据流水线（8 步）

```
pass1a  per-chunk evidence (vision-LLM)        → evidence_1a/{vid}.json
pass1b  entity alignment + state changes       → evidence_1b/{vid}.json   (合并 LLM 调用)
pass2   streaming rollout (memory evolution)   → rollout/{vid}.json
pass3a  card generation (22 families)           → task_cards/{vid}.json
pass3b  placement + trajectory plan            → placements/{vid}.json   (multi-probe progressive_answers)
pass3c  per-timestep sample generation         → samples_3c/{vid}.json
pass3d  IFD score + submodular select + OVO quota → selected/
pass4   15 verification checks                 → verified/{vid}.json
render  → final/train_sft.jsonl, train_rl.jsonl, test.jsonl, val.jsonl
```

缓存版本控制：`scripts/agent_data_v5/cache_version.py`。改 prompt → bump 3a；改 placement → bump 3b；改 verify → bump 4。pass1a/1b/2 是慢的，**改时尽量不动**。

### 2.1 Pass1 v9.5 关键改动

- **Strict parse contract**：`parse_evidence_result` 要求 `visible_entities / atomic_facts / ocr / spatial` 至少非空一个，否则 `parse_success=False` + 保留 `_raw`。修了 v9.4 的 silent-empty bug（46.4% chunks 是 valid JSON 但全空）。
- **Retry-on-empty**：silent-empty 或 parse-fail 时重试一次（temp 0.7），打破 deterministic "I see nothing" 失败模式。
- **Thinking off** + **raw httpx POST**：pass1a thinking=False，通过 raw POST 把 `chat_template_kwargs.enable_thinking=false` 直接放在请求体顶层（OpenAI SDK extra_body 在某些 vLLM server 被静默忽略）。
- **字段瘦身**：`atomic_facts` 从 list[dict{fact,confidence,target_resolution_visible}] 简化为 list[str]；parser 兼容旧 dict 形式。
- **Pass1b 合并调用**：entity_linking + state_change_detection → 一个 LLM 调用，节省 ~40% pass1b 时间。
- **Concurrent 1024**：thinking off 后 pass1a 单请求负担降，httpx 连接池上限 100→2048 解决之前的 deadlock。

## 3. 22 Family 分类

| Layer | Family | OVO 任务 | answer_form | 派发 |
|---|---|---|---|---|
| 0 perception | F1 OCR | OCR | MC + number | immediate / event_watch |
| 0 | F2 ATR/OJR | ATR, OJR | MC | immediate |
| 0 | F3 OJR-count | OJR | number | immediate |
| 0 | F4 STU | STU | MC | immediate |
| 1 single-step | E1 ACR | ACR | MC | immediate |
| 1 | E2 EPM/CRR binary | EPM, CRR | binary | event_watch |
| 1 | P1 ASI/SSR | ASI, SSR | MC | immediate |
| 1 | C1 比较 | EPM | MC | immediate |
| 1 | R1 重现 | OJR, EPM | MC | recall_success / immediate |
| 1 | S1 描述 | ATR | descriptive | immediate |
| 1 | M1 commentary | ACR | descriptive | **multi_response** |
| 1 | F5 REC 累积计数 | REC | number | **multi_response + progressive count** |
| 1 | F6 FPD | FPD | MC | immediate |
| 1 | **F7 SSR step-progress (v9.5)** | SSR | binary | **multi_response + No→Yes flip** |
| 1 | N1 HLD | HLD | MC（含 "Unable to answer"） | immediate |
| 2 reasoning | CR1 因果 Why | CRR/EPM | MC | event_watch |
| 2 | CR2 时序排序 | ASI/SSR | MC | immediate |
| 2 | CR3 意图/目标 | ASI | MC | immediate |
| 2 | CR4 组合 AND/OR | OJR/CRR | MC | immediate |
| 2 | **CR5 CRR descriptive (v9.5)** | CRR | descriptive | **multi_response + clue 延迟** |
| 2 | **CR6 STAR-Feasibility (v9.5)** | （非 OVO） | MC | immediate |
| 2 | **CR7 PerceptionTest 永恒性 (v9.5)** | （非 OVO） | MC | immediate（ask_chunk 强制 = resolve_chunk） |

**FAMILY_TARGETS 总和 = 36，post-verify mean ~18-20 cards/video。**

`FAMILY_FORCE_ATTEMPT`（每视频强制尝试，绕过 classify_chunks 启发式）：
`F5, F6, F7, N1, F3, E2, S1, CR1, CR2, CR3, CR4, CR5, CR6, CR7`

**N1/HLD 格式**：4 MC 选项 = 3 个 plausible-but-absent + 1 个字面量 `"Unable to answer"`；canonical_answer = `"Unable to answer"` 那个字母。OVO 186/186 样本验证。

## 4. 记忆 + 压缩模型

```python
class MemoryState:
    compressed_segments: List[Dict]   # ≤5 段, 每段 ≤180 tok (SFT-baked)
    recent_thinks:        List[Dict]   # token-budget 600
    _retrieval_archive:   List[Dict]   # 全部原始 think，永不压缩
    _queries:             List[Dict]   # Q/A 历史
```

**压缩触发**（system-triggered，模型只写 summary）：
- **单条件**：`tokens >= 480 (80%) AND n_thinks >= 4` → trigger
  （v9.4.2 的紧急触发 720+2 在 v9.5 已删除——OOD vs SFT，且 frame_paths bug 修后无必要）
- 触发动作：系统注入 `<compress_trigger range="t_start-t_end"/>` 到 user_input；模型输出 `<action>compress</action><summary>{"time_range":[s,e],"text":"..."}</summary>`
- 压入 compressed_segments 时 text 截断至 **180 tok（SFT `SUMMARY_TOKENS_MAX`）**
- compressed_segments 超过 5 段 → merge oldest 2，再 cap 180 tok

**Compression 的 SFT-baked 常量**（改了必须重训）：
```
RECENT_THINKS_TOKEN_BUDGET = 600
COMPRESS_TOKEN_THRESHOLD   = 480
COMPRESS_RANGE_MIN         = 4
MAX_COMPRESSED_SEGMENTS    = 5
SUMMARY_TOKENS_MAX         = 180
```

## 5. Recall 机制

模型输出 `<action>recall</action><query>{"query":"keywords","time_range":"start-end"}</query>` → agent_loop：

1. **Time-range 过滤**（v9.5）：`filter_archive_by_time_range` 把 archive 限定在 query.time_range 重叠的 chunks（缺/坏 range → 全 archive fallback）。
2. retriever（BM25 或 hybrid=BM25⊕vision-encoder）查 `_retrieval_archive` 取 top-4。
3. 重新 build prompt 带 `<recalled_frames>` + 历史帧 + `<recall_result>` → 二次 generate。
4. 二次输出含 final response。

### 5.1 Retriever 选项

- **BM25**：纯文本 baseline，零成本。
- **Hybrid (default)**：BM25 + 视觉 embedding，α=0.5。
  - 默认用 SigLIP encoder（`google/siglip-base-patch16-224`）
  - **`--use_agent_vision`**（v9.5）：复用 Qwen3-VL 自身 visual tower，省 ~600MB 加载，embedding space 与 agent 训练时一致

### 5.2 Recall query 双 schema（v9.5）

SFT 数据 70% 用 `with_time_range` schema（含 `time_range` 字段，narrow window 由 `_compute_recall_time_range(card.support_chunks, slack=4s)` 派生），30% 用 `keyword_only` schema（不含 time_range）。两套并存：retriever 见 time_range 就过滤、不见就全 archive。RL 训窄范围质量。

### 5.3 Recall 帧持久化

视频帧只活在 step() 的二次 generate prompt 里，**不持久化**到下个 chunk。只有 think text 通过 `_retrieval_archive` 永久保留。SFT 数据中 recall_response 样本现在带 `recalled_frames.frame_paths`（v9.4 缺失，导致 train/eval mismatch；v9.5 通过 `render_samples._build_recalled_frames` 注入）。

## 6. 输入布局（位置编码）

每步消息布局（视频在前、文本在后；MROPE 时间对齐）：

```
[system]
<visual_window>{json header}</visual_window>  +  video_block (24 frames)
<recalled_frames>{header}</recalled_frames>  +  recalled_video  (recall 二次 pass 才有)
<memory>
  <compressed>{json}</compressed> × n
  [time] think text × n
</memory>
<queries>...</queries>
<recall_result>{json}</recall_result>            (recall 二次 pass 才有)
<user_input>{question or compress_trigger}</user_input>
```

## 7. 训练流水线

```
pretrained Qwen3-VL → SFT (1 epoch) → GDPO RL (1 epoch)
```

**SFT 损失加权**（per-timestep）：
- `<think>` tokens × 0.3
- `<action>` tokens × 8.0   ← 关键决策
- `<response>` tokens × 2.0
- 其他 tokens × 1.0

**RL reward (v9.5, 8 路 column，权重和 = 1.0)**：
| reward | 权重 | 说明 |
|---|---|---|
| correctness | 0.30 | response 主结果 |
| silent_quality | 0.18 | streaming-specific dense reward |
| timing | 0.18 | response 时机距 gold ask_chunk |
| recall_quality | 0.05 | query JSON 格式 + 无 leakage |
| **recall_hit_rate** | 0.07 | `\|returned ∩ support\| / \|support\|` (v9.5 拆出) |
| **range_tightness** | 0.03 | `(1 - range_width/duration) × coverage` (v9.5 新加) |
| format | 0.09 | 协议合规率 |
| overflow_pen | 0.10 | compress timing soft penalty |

**Per-reward group-norm**（GDPO）独立做（mean-only，不归 σ），故权重控制相对拉力。

## 8. Eval 框架

4 entry × 2 bench × 多模型：

|        | test.jsonl                      | ovo_bench.json              |
|--------|--------------------------------|----------------------------|
| base   | `scripts/eval/test_set_base.py`        | `scripts/eval/ovo/base.py`        |
| agent  | `scripts/eval/test_set_agent.py`       | `scripts/eval/ovo/eval_full.py`   |

驱动：`scripts/eval/run_matrix.sh {base|agent|all}`

**两个 profile**（`scripts/eval/eval_profiles.py`）：

| | 16k 默认 | 32k 扩展 |
|---|---|---|
| model_max_length | 16384 | 32768 |
| QUERIES_HISTORY_CAP | 8 | 24 |
| RECALL_TEXT_MAX_CHARS | **1600** (v9.5) | 3000 |
| max_new_tokens | 128 | 256 |

profile 通过 `apply_profile()` runtime 改 agent_protocol globals，**不影响 SFT-baked 常量**。

**Streaming 测试维度**（每 ckpt × bench 跑 32 配置）：
- retriever ∈ {bm25, hybrid}，hybrid 可加 `--use_agent_vision`
- compress_mode ∈ {system, self}
- scoring ∈ {strict, lenient}
- profile ∈ {16k, 32k}

## 9. Telemetry（每步 + 聚合）

agent_loop.step() 返回 dict 含：
```
prompt_text_token_count   每步 prompt 文本部分总 tokens
think_token_count          模型输出的 <think> tokens
format_ok                  bool: 是否合规协议格式
compress_succeeded         bool/None: 触发后 summary 是否合规
compress_telemetry         {thinks_count_at_trigger, compressed_chunks, ...}
recall_returned_chunks     recall 返回的 chunk 列表
memory_token_count         post-step recent_thinks tokens
```

walk_and_score 聚合到 per-sample（v9.5 加强）：
```
n_compress_events / compress_thinks_at_trigger (list)
n_partial_compress         事件数：summary 覆盖 chunks < COMPRESS_RANGE_MIN
compress_chunk_count       chunk_idx → 被 roll 进 summary 的次数（重复压缩检测）
n_chunks_revisited         至少被压 2 次的 chunk 数
n_recall_events / recall_events
  含 hit_support (bool), schema (with_time_range/keyword_only), hit_fraction
n_recall_schema_with_time_range / _hit
n_recall_schema_keyword_only / _hit
recall_hit_fractions       per-recall fractional 分布
n_premature_responses / n_late_responses / response_offset_chunks
n_format_violations
prompt_tokens_per_step / think_tokens_per_step (list, p95/max)
```

报告输出三块表：
1. **ACCURACY/COMPRESS/RECALL/TIMING**：acc, cmp/s, cmp_thk, cmp_ok, rec/s, rec_hit, off, late, pre, noresp
2. **PROMPT/THINK/FORMAT**：pt_avg/p50/p95/max, th_avg/p95, fmt_err%
3. **RECALL SCHEMA / COMPRESS QUALITY (v9.5)**：rec_w_tr / hit_w_tr / rec_kw / hit_kw / hit_frac / cmp_par / revisit

## 10. 关键 quick-reference 常量

| 类型 | 名 | 值 | 来源 |
|---|---|---|---|
| SFT-baked | RECENT_THINKS_TOKEN_BUDGET | 600 | agent_loop.py |
| SFT-baked | COMPRESS_TOKEN_THRESHOLD | 480 | agent_loop.py |
| SFT-baked | COMPRESS_RANGE_MIN | 4 | agent_loop.py |
| SFT-baked | MAX_COMPRESSED_SEGMENTS | 5 | agent_loop.py |
| SFT-baked | **SUMMARY_TOKENS_MAX** | **180** | agent_loop.py + config.py |
| SFT-baked | VISUAL_WINDOW_CHUNKS | 12 | agent_protocol.py |
| SFT-baked | FRAMES_PER_CHUNK | 2 | agent_protocol.py |
| SFT-baked | video_min/max_pixels | 100352/150528 | sft/argument.py |
| SFT-baked | RECALL_TIME_RANGE_FRACTION | 0.7 | pass3c_samples.py |
| eval-side | QUERIES_HISTORY_CAP | 8 (16k) / 24 (32k) | agent_protocol.py + profile |
| eval-side | RECALL_TEXT_MAX_CHARS | **1600** (16k) / 3000 (32k) | agent_protocol.py + profile |
| eval-side | model_max_length | 16384 / 32768 | profile |
| eval-side | max_new_tokens | 128 / 256 | profile |
| pass1 perf | pass1a concurrent | 1024 | config.py:PASS_CONFIG |
| pass1 perf | pass1a thinking | False | config.py |
| pass1 perf | httpx max_connections | 2048 | vllm_client.py |

## 11. v9.5 改动一览（commit 时间序）

| commit | 主题 |
|---|---|
| `86d0091` | SFT-align caps + multimodal recall + dual time_range schema + hit-rate reward |
| `2d0660e` | RL reward 8 路拆分 + streaming-eval 质量指标 + agent vision retriever |
| `aa952ec` | metadata.gold_compress_chunks |
| `8eb9a11` | pass1 silent-empty parse + retry + thinking-off via raw POST + 1b 合并 + chunk_has_evidence |
| `78b1608` | pass1a concurrent 64→1024，httpx pool 提升 |
| `4d6bd86` | pass3b F5/E2 multi_response 派发（后撤回 E2，留 F5） |
| `f05f5fc` | F7 (SSR) + CR5 (CRR) family + multi_response progressive_answers |
| `0729f66` | CR6 (STAR-Feas) + CR7 (PerceptionTest) family |
| `69f3155` | 死代码 + 字段对不上 + silent skip 4 处修复 |

## 12. 历史文档

- `docs/multi_probe_design_v95.md`：v9.5 多 probe 设计文档（**已落地为 F7/CR5 + progressive_answers**，原方案"在 F5/E2/P1 加 probes 字段"未采用）。保留作历史。
- `docs/batch1_video_selection.md`：batch1 视频清单选择策略，仍有效。
