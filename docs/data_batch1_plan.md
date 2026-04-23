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

## 3. SFT 数据量

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

### 3.2 SFT 样本量预估

| 视频类型 | 视频数 | 平均样本/视频 | 样本小计 |
|---------|--------|-------------|---------|
| 短 | 60 | ~25 | ~1,500 |
| 中 | 110 | ~42 | ~4,620 |
| 长 | 14 | ~42 | ~588 |
| **SFT 总计** | **184** | | **~6,700** |
| Pass5 过滤后 (~85%) | | | **~5,700** |

### 3.3 SFT Phase 分布（仅 train）

| Phase | 样本类型 | 样本数 | 占比 |
|-------|---------|--------|------|
| Phase 1 | silent + response(frames) + pending_silent | ~1,700 | 30% |
| Phase 2 | recall + response(mem) + unanswerable + pending_response | ~2,600 | 46% |
| C1 | compress(指定范围) + compress_recall/response | ~800 | 14% |
| C2 | compress(自选范围) | ~630 | 11% |
| Phase 5 | 全量混合 | ~5,700 | 100% |

---

## 4. RL 数据量

RL 不需要预造 per-timestep 样本。模型对 RL 视频做完整 rollout，用 gold answer 算 reward。

| 视频类型 | 视频数 | 平均 task/视频 | (视频, task) 对 |
|---------|--------|--------------|----------------|
| 短 | 16 | ~15 | ~240 |
| 中 | 50 | ~30 | ~1,500 |
| 长 | 9 | ~35 | ~315 |
| **RL 总计** | **75** | | **~2,050** |

RL 的 task 来自 Pass3 挖掘的 gold answer（同一个 pipeline，只是模型自己 rollout 而不是用 teacher 轨迹）。

---

## 5. 训练步数

### 5.1 SFT（8×H100, global_batch=64）

| 阶段 | 样本数 | epochs | steps | LR | 基模型 |
|------|--------|--------|-------|----|--------|
| Phase 1 | ~1,700 | 3 | 79 | 1e-5 | Qwen2.5-VL-3B |
| Phase 2 | ~2,600 | 3 | 121 | 5e-6 | ← P1 ckpt |
| Phase C1 | ~800 | 2 | 25 | 3e-6 | ← P2 ckpt |
| Phase C2 | ~630 | 2 | 19 | 2e-6 | ← C1 ckpt |
| Phase 5 | ~5,700 | 1 | 89 | 1e-6 | ← C2 ckpt |
| **SFT 小计** | | | **333** | | ~4 min |

### 5.2 RL/GRPO（8×H100, global_batch=16, group_size=4）

| 阶段 | (视频,task) 对 | epochs | steps | LR | 基模型 |
|------|---------------|--------|-------|----|--------|
| GRPO | ~2,050 | 2 | 256 | 5e-7 | ← P5 ckpt |
| **RL 小计** | | | **256** | | ~34 min |

**总训练**: 589 steps, ~38 min（RL 的 rollout 是时间瓶颈）

### 5.3 GRPO Reward 设计

| 分量 | 权重 | 信号 |
|------|------|------|
| R_format | 0.15 | think/action tag 格式正确 |
| R_action | 0.20 | 选对 action 类型 vs gold_action |
| R_correctness | 0.30 | 答案 vs gold_answer |
| R_timing | 0.15 | 在正确时间步响应 |
| R_think_len | 0.10 | think 长度在 40-60 tok |
| R_compress | 0.10 | 压缩后 entity retention |

---

## 6. Eval 策略

### 6.1 Eval 时间点

| 时间点 | 数据 | 评什么 |
|--------|------|--------|
| SFT P1 结束 | val 30 条 | action F1 (silent/response) |
| SFT P2 结束 | val 30 条 | action F1 + recall 准确率 |
| SFT C2 结束 | val 30 条 | 压缩质量 (entity retention) |
| SFT P5 结束 | val 30 条 | 全量指标 → **决定是否进 RL** |
| RL 每 50 steps | val 30 条 | reward 均值 + action F1 |
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

# SFT
PHASE=1 bash scripts/sft_per_timestep.sh
PHASE=2 LLM=output/agent-phase1 bash scripts/sft_per_timestep.sh
PHASE=C1 LLM=output/agent-phase2 bash scripts/sft_per_timestep.sh
PHASE=C2 LLM=output/agent-c1 bash scripts/sft_per_timestep.sh
PHASE=5 LLM=output/agent-c2 bash scripts/sft_per_timestep.sh

# RL (从 Phase 5 checkpoint 继续)
# TODO: 迁移 grpo.py rollout 到 per-timestep agent_loop
```

---

## 9. 第二批扩充（如需）

v0 模型评估后，如果某类 task 效果不好：
- 追加 300 个 120-240s 视频（200 SFT + 100 RL）
- 调高对应 type 的 MAX_CANDIDATES_PER_VIDEO 限额
- 只重跑 Pass3+4+5
