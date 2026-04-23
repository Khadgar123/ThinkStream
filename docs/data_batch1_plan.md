# 第一批数据构造计划

> 日期: 2026-04-23（更新: 2026-04-23）
> 目标: ~11K 训练样本，覆盖所有 task type，控制 API 成本

---

## 1. 选取规模

**319 个视频，预计 ~11K 训练样本（过滤后 ~9.5K）**

| 指标 | 数值 |
|------|------|
| 视频总数 | 319（原 320，移除 1 条坏记录） |
| 总视频时长 | ~14.1 小时 |
| 预估训练样本 | ~11,000 |
| Pass5 过滤后 | ~9,500 |
| 预估 397B API 调用 | ~50,000 |
| 预估耗时（1024 并发） | ~18 小时 |

---

## 2. 视频时长分布

| 时长区间 | 数量 | 占比 | 作用 |
|---------|------|------|------|
| 60-120s（短） | 96 | 30% | 简单 memory，学基础 think + response，无压缩 |
| 120-240s（中） | 192 | 60% | **主力**，覆盖全部 task type |
| 240-400s（长） | 32 | 10% | 深度压缩（多次 merge） |

---

## 3. Per-video candidate 限额

控制每个视频的样本数量，避免单个视频产出过多同类样本（如 90 个 response_from_frames）。

```python
MAX_CANDIDATES_PER_VIDEO = {
    "response_from_frames": 8,    # 不需要 90 个"看帧回答"
    "response_from_memory": 5,
    "recall": 5,
    "compress_recall": 3,
    "compress_response": 3,
    "unanswerable": 5,
    "pending": 3,
    # compress: 不限制，由实际压缩事件决定
}
```

限额后每个中等视频候选 ~35 个 task。最终样本数由 `MAX_SAMPLES_PER_VIDEO=50` 封顶
（recall 产 2 条、pending 产 3 条、compress 产 C1+C2 两条，实际样本数 > 候选数）。

---

## 4. 预估样本分布

| 样本类型 | 数量 | 占比 | Phase |
|---------|------|------|-------|
| response (from_frames) | ~1,800 | 16% | 1 |
| response (from_memory) | ~1,100 | 10% | 2 |
| recall_query | ~1,200 | 11% | 2 |
| recall_response | ~1,200 | 11% | 2 |
| compress (C1+C2) | ~1,400 | 13% | C1/C2 |
| silent | ~1,400 | 13% | 1 |
| pending (start+mid+trigger) | ~1,500 | 14% | 1/2 |
| unanswerable | ~1,400 | 13% | 2 |
| **总计** | **~11,000** | | |

**分布均衡**：每种 task type 约 1,000-1,800 个样本，没有严重偏科。

---

## 5. 数据集来源

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

## 6. 执行

```bash
# 删除旧 registry
rm -f data/agent_v5/video_registry.jsonl

# 运行（自动从 CSV catalog 选 320 视频）
python -m scripts.agent_data_v5 run \
    --api_base http://AMD_IP:8000/v1 \
    --video_root /path/to/videos \
    --num_videos 320
```

---

## 7. 资源预估

| 阶段 | API 调用 | 说明 |
|------|---------|------|
| Pass1 (evidence) | ~20,000 | 每 chunk 1 次 |
| Pass2 (rollout) | ~22,000 | observation + compress |
| Pass3 (questions) | ~5,000 | 限额后 unique facts ~15/video |
| Pass4 (responses) | ~3,000 | 限额后 ~10/video |
| **总计** | **~50,000** | |

| 并发 | 预计耗时 |
|------|---------|
| 256 | ~36 小时 |
| 1024 | ~18 小时 |

---

## 8. 训练够不够

| 场景 | 需要 | 本批 | 评估 |
|------|------|------|------|
| 混合训练 3 epoch (8B) | 9.5K × 3 = 28.5K step | 9.5K | ✅ 够 |
| Phase 1 单独 3 epoch | ~3K × 3 = 9K step | ~3K | ✅ 够 |
| Phase 2 单独 3 epoch | ~5K × 3 = 15K step | ~5K | ✅ 够 |
| C1/C2 单独 2 epoch | ~1.4K × 2 = 2.8K step | ~1.4K | ⚠️ 偏少 |

C1/C2 样本偏少，但混合训练时不是问题（compress 样本 loss weight = 1.5，等效更多）。如果单独训 C1/C2 phase 效果不好，第二批补充。

---

## 9. 第二批扩充（如需）

v0 模型评估后，如果某类 task 效果不好：
- 追加 300 个 120-240s 视频
- 调高对应 type 的 MAX_CANDIDATES_PER_VIDEO 限额
- 只重跑 Pass3+4+5
