# ThinkStream 设计（当前状态：v9.4.2）

> Source of Truth: `git log` + `tests/` + `memory/`. 本文档是 **快照**——
> 当代码与本文档分歧时以代码为准。最新 commit 见 `git log --oneline -10`。

## 1. 项目目标

流式视频 Agent：每 2 秒（一个 chunk）做一次决策，输出 `silent / response / recall / compress` 之一，配合内部记忆状态在长视频中持续做问答。

数据 + 训练 + 推理三端共用 **per-timestep snapshot 格式**：训练样本 = 推理某一步的精确 input。

## 2. 数据流水线（8 步）

```
pass1a  per-chunk evidence (vision-LLM)        → evidence_1a/{vid}.json
pass1b  entity alignment + state changes       → evidence_1b/{vid}.json
pass2   streaming rollout (memory evolution)   → rollout/{vid}.json
pass3a  card generation (18 families)          → task_cards/{vid}.json
pass3b  placement + trajectory plan            → placements/{vid}.json
pass3c  per-timestep sample generation          → samples_3c/{vid}.json
pass3d  IFD score + submodular select + OVO quota → selected/
pass4   14 verification checks                 → verified/{vid}.json
render  → final/train_sft.jsonl, train_rl.jsonl, test.jsonl, val.jsonl
```

缓存版本控制：`scripts/agent_data_v5/cache_version.py`。改 prompt → bump 3a；改 placement → bump 3b；改 verify → bump 4。pass1a/1b/2 是慢的，**改时尽量不动**。

## 3. 18 family 分类

| Layer | Family | OVO 任务 | answer_form |
|---|---|---|---|
| 0 perception | F1 OCR | OCR | MC + number |
| 0 | F2 ATR | ATR | MC |
| 0 | F3 OJR-count | OJR | number |
| 0 | F4 STU | STU | MC |
| 1 single-step | E1 ACR | ACR | MC + short_exact |
| 1 | E2 EPM/CRR | EPM, CRR | binary + MC |
| 1 | P1 ASI/SSR | ASI, SSR | MC |
| 1 | C1 EPM | EPM | MC |
| 1 | R1 OJR | OJR | MC |
| 1 | S1 描述 | ATR | descriptive |
| 1 | M1 commentary | ACR | descriptive |
| 1 | F5 REC | REC | number |
| 1 | F6 FPD | FPD | MC |
| 1 | N1 HLD | HLD | MC（见下） |
| **2 reasoning** | **CR1 因果** | CRR/EPM | MC |
| 2 | **CR2 顺序** | ASI/SSR | MC |
| 2 | **CR3 意图** | ASI | MC |
| 2 | **CR4 组合 AND** | OJR/CRR | MC |

**N1/HLD 格式（关键）**：4 MC 选项 = 3 个 plausible-but-absent + 1 个字面量 `"Unable to answer"`；canonical_answer = `"Unable to answer"` 那个字母。OVO 186/186 样本验证。

## 4. 记忆 + 压缩模型

```python
class MemoryState:
    compressed_segments: List[Dict]   # ≤5 段, 每段 ≤200 tok
    recent_thinks:        List[Dict]   # token-budget 600
    _retrieval_archive:   List[Dict]   # 全部原始 think，永不压缩
    _queries:             List[Dict]   # Q/A 历史
```

**压缩触发**（system-triggered，模型只写 summary）：
- 标准：`tokens >= 480 (80%) AND n_thinks >= 4` → trigger
- 紧急（v9.4.2，话痨防御）：`tokens >= 720 (1.5×) AND n_thinks >= 2` → trigger
- 触发动作：系统注入 `<compress_trigger range="t_start-t_end"/>` 到 user_input；模型按 SFT 训出的格式输出 `<action>compress</action><summary>{"time_range":[s,e],"text":"..."}</summary>`
- 压入 compressed_segments 时 text 截断至 200 tok（匹配 SFT `SUMMARY_TOKENS_MAX=180` + 余量）
- compressed_segments 超过 5 段 → merge oldest 2，再 cap 200 tok

**Compression 的 SFT-baked 常量**（改了必须重训）：
```
RECENT_THINKS_TOKEN_BUDGET = 600
COMPRESS_TOKEN_THRESHOLD   = 480
COMPRESS_RANGE_MIN         = 4
MAX_COMPRESSED_SEGMENTS    = 5
```

## 5. Recall 机制

模型输出 `<action>recall</action><query>{"query":"keywords","time_range":"start-end"}</query>` → agent_loop：
1. retriever（BM25 或 hybrid=BM25⊕siglip）查 `_retrieval_archive` 取 top-4
2. 重新 build prompt 带 `<recalled_frames>` + `<recall_result>` → 二次 generate
3. 二次输出含 final response

**重要**：recall 视频帧只活在 step() 的二次 generate prompt 里，**不持久化**到下个 chunk。只有 think text 通过 `_retrieval_archive` 永久保留。

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

**RL reward**（6 路，权重和 = 1.0）：
- correctness 0.30 + silent_quality 0.20 + timing 0.20 + recall_quality 0.10 + format 0.10 + overflow_pen 0.10

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
| RECALL_TEXT_MAX_CHARS | 800 | 3000 |
| max_new_tokens | 128 | 256 |

profile 通过 `apply_profile()` runtime 改 agent_protocol globals，**不影响 SFT-baked 常量**。

**Streaming 测试维度**（每 ckpt × bench 跑 32 配置）：
- retriever ∈ {bm25, hybrid}
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

walk_and_score 聚合到 per-sample：
```
n_compress_events / compress_thinks_at_trigger (list)
n_recall_events / recall_events (含 hit_support 标志)
n_premature_responses / n_late_responses
response_offset_chunks
n_format_violations
prompt_tokens_per_step (list, p95/max 看 model_max_length 占用率)
```

报告输出两块表：
1. **ACCURACY/COMPRESS/RECALL/TIMING**：acc, cmp/s, cmp_thk, cmp_ok, rec/s, rec_hit, off, late, pre, noresp
2. **PROMPT/THINK/FORMAT**：pt_avg/p50/p95/max, th_avg/p95, fmt_err%

## 10. Debug 模式

context overflow 怀疑时：

```bash
python scripts/eval/debug_streaming.py \
    --ckpt $CKPT \
    --test_jsonl data/agent_v5/final/test.jsonl \
    --video_root $VID_ROOT \
    --frames_root data/agent_v5/frames \
    --sample_idx 0 --profile 16k \
    [--dump_json /tmp/trace.json]
```

每个 chunk 打印 zone 分解（system/visual/memory/queries/recall_result/user_input），标记 ⚠️/🚨 当 zone 超预算或总 prompt > 90% × max_length。

## 11. 关键 quick-reference 常量

| 类型 | 名 | 值 | 来源 |
|---|---|---|---|
| SFT-baked | RECENT_THINKS_TOKEN_BUDGET | 600 | agent_loop.py |
| SFT-baked | COMPRESS_TOKEN_THRESHOLD | 480 | agent_loop.py |
| SFT-baked | COMPRESS_RANGE_MIN | 4 | agent_loop.py |
| SFT-baked | MAX_COMPRESSED_SEGMENTS | 5 | agent_loop.py |
| SFT-baked | SUMMARY_TOKENS_MAX | 180 | config.py |
| SFT-baked | VISUAL_WINDOW_CHUNKS | 12 | agent_protocol.py |
| SFT-baked | FRAMES_PER_CHUNK | 2 | agent_protocol.py |
| SFT-baked | video_min/max_pixels | 100352/150528 | sft/argument.py |
| eval-side | QUERIES_HISTORY_CAP | 8 (16k) / 24 (32k) | agent_protocol.py + profile |
| eval-side | RECALL_TEXT_MAX_CHARS | 800 / 3000 | agent_protocol.py + profile |
| eval-side | model_max_length | 16384 / 32768 | profile |
| eval-side | max_new_tokens | 128 / 256 | profile |

## 12. 下一步计划

未实现：见 `docs/multi_probe_design_v95.md`——OVO REC/CRR/SSR 多探针 schema 扩展（F5/E2/P1 加 `probes` 字段）+ OJR holding（F2 prompt 加强）。Phase A-C 各 ~1 天工作量。
