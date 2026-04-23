# 第一批数据构造计划

> 日期: 2026-04-23
> 目标: 覆盖所有训练任务类型的最小可用数据集

---

## 1. 选取规模

**500 个视频, 预计 ~41K 训练样本**

| 指标 | 数值 |
|------|------|
| 视频总数 | 500 |
| 总视频时长 | 22.2 小时 |
| 总 chunks | 39,651 |
| 预估训练样本 | ~41,000 |
| 预估 397B API 调用 | ~147,000 |
| 预估耗时（1024 并发） | ~54 小时（~2.3 天） |

---

## 2. 视频时长分布

| 时长区间 | 数量 | 占比 | 作用 |
|---------|------|------|------|
| 60-120s（短） | 150 | 30% | 简单 memory 状态，学基础 think + response。无压缩，几乎无 recall。提供干净的协议对齐信号。 |
| 120-240s（中） | 300 | 60% | **主力**。3-5 次压缩，recall 任务充分，memory 状态包含 compressed_segments + recent_thinks。覆盖全部 task type。 |
| 240-400s（长） | 50 | 10% | 深度压缩（多次 merge，4-5 段 compressed_segments）。训练模型处理复杂记忆层次结构。 |

为什么不全用长视频: 短视频提供简单 memory 的训练样本，如果模型只见过复杂 memory，推理早期阶段（视频刚开始、memory 为空时）的行为可能不稳定。

---

## 3. 数据集来源分布

| 数据集 | 数量 | 内容特点 |
|--------|------|----------|
| VideoMind-Dataset | ~86 | 多样化：动作识别、高光检测、视频理解 |
| how_to_step | ~84 | 教学/操作：烹饪、手工、维修步骤 |
| how_to_caption | ~84 | 教学+字幕：流程性内容 |
| Koala_raw | ~86 | 日常场景、实体交互 |
| tarsier2_unzip | ~83 | 视频理解、动作描述 |
| LLaVA-Video-178K | ~76 | 通用视频问答 |

**6 个来源均匀分布**，确保内容多样性（不被单一来源主导）。

---

## 4. 预估样本分布

| 样本类型 | 预估数量 | Phase | 说明 |
|---------|---------|-------|------|
| response (from_frames) | ~5,000 | 1 | 答案在当前帧中，直接回答 |
| response (from_memory) | ~4,000 | 2 | 答案在 thinks 文本中，帧已滑走 |
| recall_query | ~6,000 | 2 | 答案不在可见范围，输出 recall query |
| recall_response | ~6,000 | 2 | recall 返回结果后回答（无 think） |
| compress (C1+C2) | ~6,700 | C1/C2 | 系统触发压缩，生成 summary |
| silent | ~6,700 | 1 | 无问题，只输出观察（20% 采样） |
| pending (start+mid+trigger) | ~6,400 | 1/2 | event-watch 生命周期 |
| unanswerable | ~6,400 | 2 | 不可回答的问题 |
| **总计** | **~41,000** | | |

Phase 分布:
- Phase 1 (silent + response_from_frames): ~12K
- Phase 2 (recall + memory + pending + unanswerable): ~23K
- Phase C1/C2 (compress): ~6K

如果做**混合训练**（不分阶段），41K 样本对 8B 模型做 2-3 epoch 是够的。

---

## 5. 执行步骤

```bash
# 1. 删除旧的 video_registry（如果有）
rm -f data/agent_v5/video_registry.jsonl

# 2. 运行 pipeline（自动从 CSV catalog 选取 500 视频）
python -m scripts.agent_data_v5 run \
    --api_base http://AMD_IP:8000/v1 \
    --video_root /path/to/videos \
    --num_videos 500

# 视频选取逻辑:
# - 自动加载 data/video_catalog_30s_plus.csv
# - 筛选 60-400s，按 30/60/10 比例分桶
# - 每桶按 dataset 来源分层采样
# - 保存到 data/agent_v5/video_registry.jsonl（缓存）
```

---

## 6. 资源预估

### 397B API 调用分解

| Pass | 调用/视频(短) | 调用/视频(中) | 调用/视频(长) | 总计 |
|------|-------------|-------------|-------------|------|
| Pass1 evidence | 30 | 90 | 150 | ~42K |
| Pass2 rollout | 33 | 98 | 168 | ~47K |
| Pass3 questions | 15 | 120 | 200 | ~40K |
| Pass4 responses | 5 | 40 | 80 | ~18K |
| **总计** | **83** | **348** | **598** | **~147K** |

### 时间预估

| 并发 | 吞吐 | 预计耗时 |
|------|------|----------|
| 64 | 945 tok/s | ~216 小时 (9 天) |
| 256 | ~2500 tok/s | ~81 小时 (3.4 天) |
| 1024 | ~3780 tok/s | ~54 小时 (2.3 天) |

---

## 7. 质量控制

### Pass5 过滤

预期 pass rate ~85%:
- information_flow: 去除 query 包含 answer 的样本
- grounding: 去除 think 中有 sound/emotion 的样本
- format: 去除格式不符的样本
- provenance: 去除 summary 引用了 thinks 之外内容的样本
- leakage: 去除 question 包含 answer 的样本

### 过滤后

~41K × 85% ≈ **~35K 可用训练样本**

### 是否够用

| 训练方式 | 需要 | 本批 | 够？ |
|---------|------|------|------|
| 混合训练 2 epoch | ~35K × 2 = 70K 次优化步 | 35K | ✅ 够 |
| 分阶段 Phase 1 | ~12K × 3 epoch | 12K | ✅ 够 |
| 分阶段 Phase 2 | ~23K × 3 epoch | 23K | ✅ 够 |
| 分阶段 C1/C2 | ~6K × 2 epoch | 6K | ⚠️ 偏少，考虑第二批扩充 |

---

## 8. 第二批扩充计划（如需）

训完 v0 后评估，如果 compress 或 recall 效果不够好:
- 追加 500 个 120-300s 视频（纯 medium）
- 只跑 Pass1-5，得到 ~50K 额外样本
- 重点补充 compress + recall 样本
