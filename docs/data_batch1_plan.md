# 第一批数据构造计划

> 日期: 2026-04-23（更新: 2026-04-23）
> 目标: 319 个视频，SFT + RL 两阶段训练，覆盖所有 task type

---

## 1. 视频总览

| 指标 | 数值 |
|------|------|
| 视频总数 | 319（原 320，移除 1 条坏记录） |
| 总视频时长 | ~14.1 小时 |
| 预估 397B API 调用 | ~50,000（仅 SFT 视频需要） |
| 预估造数据耗时（1024 并发） | ~18 小时 |

### 1.1 时长分布

| 时长区间 | 数量 | 占比 | 能力覆盖 |
|---------|------|------|---------|
| 60-120s（短） | 96 | 30% | think + silent + response，无压缩/recall |
| 120-240s（中） | 192 | 60% | **全部 4 个 action**，多次压缩，recall |
| 240-400s（长） | 32 | 10% | 深度压缩 + merge，长 pending，深 recall |

---

## 2. 视频级划分: train / val / test → SFT / RL

**原则**：视频是原子单位，同一视频的所有样本只属于一个 split。

```
319 个视频
├── test:  30 条 (短10 + 中16 + 长4)   ← 最终评估，训练全程不碰
├── val:   30 条 (短10 + 中16 + 长4)   ← 训练中 eval loss / reward
└── train: 259 条
    ├── SFT: 184 条 (短60 + 中110 + 长14)
    │   用 teacher (397B) 轨迹构造 per-timestep 样本
    │   覆盖 Phase 1 → Phase 2 → C1 → C2 → Phase 5
    │
    └── RL:  75 条 (短16 + 中50 + 长9)
        模型自己 rollout，不用 teacher 轨迹
        用 gold answer 算 reward，GRPO 优化
```

**SFT 和 RL 用不同视频**的原因：
- RL 需要 on-policy rollout（模型自己的轨迹），如果用 SFT 同样的视频，模型倾向记忆 teacher 轨迹
- RL 视频中长占比高（50+9=59 条，79%），因为中长视频决策点更多，reward 信号更丰富

### 2.1 划分明细

| Split | 短 (60-120s) | 中 (120-240s) | 长 (240-400s) | 合计 | 用途 |
|-------|-------------|---------------|---------------|------|------|
| test | 10 | 16 | 4 | **30** | 最终评估 |
| val | 10 | 16 | 4 | **30** | 训练中 eval |
| SFT train | 60 | 110 | 14 | **184** | teacher 轨迹 SFT |
| RL train | 16 | 50 | 9 | **75** | 模型 rollout + GRPO |
| **总计** | 96 | 192 | 31 | **319** | |

---

## 3. SFT 数据量（精确到视频类型）

### 3.1 Per-video 控制

```python
MAX_CANDIDATES_PER_VIDEO = {
    "response_from_frames": 8,
    "response_from_memory": 5,
    "recall": 5,
    "compress_recall": 3,
    "compress_response": 3,
    "unanswerable": 5,
    "pending": 3,
}
MAX_SAMPLES_PER_VIDEO = 50  # 最终样本数封顶（按类型等额预算）
```

### 3.2 SFT 每类视频明细

| 视频类型 | 条数 | 平均时长 | 平均 chunks | 总 chunks | 样本/条 | 总样本 | Pass5 后 |
|---------|------|---------|------------|----------|--------|--------|---------|
| 短 (60-120s) | 60 | 90s | 45 | 2,700 | 33 | 1,980 | 1,680 |
| 中 (120-240s) | 110 | 175s | 87 | 9,570 | 50 | 5,500 | 4,620 |
| 长 (240-400s) | 14 | 310s | 155 | 2,170 | 50 | 700 | 588 |
| **合计** | **184** | | | **14,440** | | **8,180** | **6,888** |

短视频无 compress/recall（memory 不够长），样本以 silent + response + pending 为主。
中长视频覆盖全部 4 action，cap 到 50 条/视频。

### 3.3 SFT 样本类型分布（Pass5 后 ~6,888 条）

| 类型 | 样本数 | 占比 | Phase |
|------|--------|------|-------|
| response | ~1,631 | 23.7% | Phase 1+2 |
| compress (C1+C2) | ~1,362 | 19.8% | C1+C2 |
| recall (query+response) | ~1,179 | 17.1% | Phase 2 |
| silent | ~1,123 | 16.3% | Phase 1 |
| pending (start+mid+trigger) | ~955 | 13.9% | Phase 1+2 |
| unanswerable | ~634 | 9.2% | Phase 2 |

### 3.4 SFT Phase 样本量

| Phase | 样本数 | 占比 |
|-------|--------|------|
| Phase 1 (协议对齐) | ~2,066 | 30% |
| Phase 2 (Recall 学习) | ~3,168 | 46% |
| C1 (指定范围压缩) | ~964 | 14% |
| C2 (自选压缩) | ~757 | 11% |
| Phase 5 (全量混合) | ~6,888 | 100% |

---

## 4. RL 数据量（精确到视频类型）

RL 不预造 per-timestep 样本。模型对 RL 视频做完整 rollout，用 gold answer 算 reward。

| 视频类型 | 条数 | 平均时长 | 平均 chunks | 总 chunks | task/条 | 总 tasks |
|---------|------|---------|------------|----------|--------|---------|
| 短 (60-120s) | 16 | 90s | 45 | 720 | 21 | 336 |
| 中 (120-240s) | 50 | 175s | 87 | 4,350 | 33 | 1,650 |
| 长 (240-400s) | 9 | 310s | 155 | 1,395 | 40 | 360 |
| **合计** | **75** | | | **6,465** | | **2,346** |

- 每条 rollout = 模型跑完整个视频的所有 chunk（自主决策每步 action）
- GRPO group_size=4 → 每个 task 跑 4 条 rollout
- 总推理步数: 2,346 × 4 × ~87 = ~816K 次推理
- task 的 gold answer 来自 Pass3（同一个 pipeline，只是模型自己 rollout）

---

## 5. 训练步数（v9.2 简化为 1 SFT + 1 GDPO RL）

> 旧 5 阶段 SFT (P1→P2→C1→C2→P5) 已合并为单次混合 SFT。依据见 `data_construction_zh.md` §4 + `sft_engineering.md` §7.1。Phase 数据文件仍产出，仅用于 per-category 诊断 eval。

### 5.1 SFT（8×H100, global_batch=64, one-shot mixed）

| 阶段 | 样本数 | epochs | steps | LR | 基模型 |
|------|--------|--------|-------|----|--------|
| **Stage 1: SFT** (P1+P2+C1+C2+P5 全混合) | 6,888 | 2 | 215 | 1e-5 → 1e-6 cosine | Qwen2.5-VL-3B |
| **SFT 小计** | | | **215** | | **~3 min** |

### 5.2 RL/GDPO（8×H100, global_batch=16, group_size=4）

| 阶段 | (视频,task) 对 | epochs | steps | LR | 基模型 |
|------|---------------|--------|-------|----|--------|
| **Stage 2: GDPO** | 2,346 | 2 | 292 | 5e-7 | ← Stage 1 ckpt |
| **RL 小计** | | | **292** | | **~39 min** |

**总训练**: 507 steps, ~42 min（RL rollout 仍是时间瓶颈）。

### 5.3 GDPO Reward 设计（v11，6 路）

| 分量 | 权重 | 信号 | 应用条件 (mask) |
|------|------|------|-----------------|
| R_correctness | 0.30 | 答案 vs gold_answer (exact match for V1-V4) | 仅有 gold_answer 的样本 |
| R_silent_quality | 0.20 | 该 silent 是否 silent / 该 respond 是否 respond | 始终 |
| R_timing | 0.20 | 在正确时间步响应（4 chunk 容忍窗口） | 仅 trajectory 含 response action |
| R_recall_quality | 0.10 | recall query 格式 + 无答案泄漏 | 仅 trajectory 含 recall action |
| R_format | 0.10 | think/action tag 格式正确 | 始终（不再硬门控） |
| R_overflow_pen | 0.10 | memory 溢出（自选 compress range 失败）→ -1.0 | 始终 |

**Aggregation (NVIDIA GDPO 模式)**：per-reward group-norm（mean-only） → weighted sum → batch-whiten。
代码：`thinkstream/trainer/gdpo_advantage.py` + `compute_gdpo_advantages` in `grpo.py`。

---

## 6. Eval 策略

### 6.1 Eval 时间点

### 6.0 val / test 明细

| Split | 短 (60-120s) | 中 (120-240s) | 长 (240-400s) | 合计 | 总 chunks | 样本数 |
|-------|-------------|---------------|---------------|------|----------|--------|
| val | 10 | 16 | 4 | 30 | 2,462 | ~1,120 |
| test | 10 | 16 | 4 | 30 | 2,462 | ~1,120 |

val/test 按视频粒度划分，同一视频的所有样本只在一个 split。按长度分层确保覆盖全 action type。

### 6.1 Eval 时间点（v9.2 简化）

| 时间点 | 数据 | 评什么 |
|--------|------|--------|
| SFT 25%/50%/75% | val 30 条 | per-phase 分桶 F1（silent/response/recall/compress 各自） + 压缩质量 |
| SFT 结束 | val 30 条 | 全量指标 → **决定是否进 RL** |
| RL 每 50 steps | val 30 条 | reward 均值 + action F1 + `mask_*_rate` from `grpo_step.jsonl` |
| RL 结束 | val 30 条 | 全量指标 |
| **最终** | test 30 条 | 全量 + 端到端 rollout |

### 6.2 Eval 指标

| 维度 | 指标 |
|------|------|
| Action Selection | silent/response/recall/compress macro-F1 |
| Recall | need-recall 判断准确率, recall@1/3 |
| QA | 按来源分: frame / memory / compressed / recall / unanswerable |
| 压缩 | entity retention, QA-after-compress accuracy |
| Think | 长度 p50/p90, 视觉 grounding rate |
| 系统 | latency/chunk, context overflow rate |

---

## 7. 数据集来源

从 CSV catalog (`data/video_catalog_30s_plus.csv`) 中按 dataset 分层采样：

| 数据集 | 视频数 | 内容特点 |
|--------|--------|----------|
| VideoMind-Dataset | ~55 | 多样化（动作识别、高光、问答） |
| how_to_step | ~55 | 教学步骤（烹饪、修理） |
| how_to_caption | ~55 | 教学+字幕 |
| tarsier2_unzip | ~53 | 视频理解 |
| LLaVA-Video-178K | ~50 | 通用视频问答 |
| Koala_raw | ~52 | 日常场景 |

---

## 8. 执行

```bash
# 造数据（仅 SFT 184 条视频需要跑 pipeline）
rm -f data/agent_v5/video_registry.jsonl
python -m scripts.agent_data_v5 run \
    --api_base http://AMD_IP:8000/v1 \
    --video_root /path/to/videos \
    --num_videos 319

# SFT (v9.2: one-shot, 全混合数据 + class-balanced sampler)
PHASE=mixed bash scripts/sft_per_timestep.sh   # → output/agent-mixed/

# RL/GDPO (从 SFT checkpoint 继续；audit log 自动开到 OUTPUT/audit/)
LLM=output/agent-mixed bash scripts/grpo_train.sh   # → output/agent-grpo/

# 训练中 tail GDPO 诊断：
tail -f output/agent-grpo/audit/grpo_step.jsonl | jq .
```

---

## 9. 第二批扩充（如需）

v0 模型评估后，如果某类 task 效果不好：
- 追加 300 个 120-240s 视频（200 SFT + 100 RL）
- 调高对应 type 的 MAX_CANDIDATES_PER_VIDEO 限额
- 只重跑 Pass3+4+5
