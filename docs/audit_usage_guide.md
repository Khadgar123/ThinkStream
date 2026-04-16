# 数据审计工具使用指南

本文档说明如何使用 `scripts/audit/` 目录下的脚本完成数据审计与去重。

**审计必须在任何数据构造（Stage 0-6）之前完成。**

---

## 架构设计: 先标记, 再处理

审计流程分为两个阶段:

```
Phase 1 — 标记 (贵, 只跑一次)          Phase 2 — 决策 (便宜, 反复调)
┌─────────────────────┐               ┌──────────────────────┐
│ build_video_asset_   │               │ apply_audit_policy.py│
│ registry.py          │               │                      │
│  · 字段提取          │    标记结果    │  · 可用性分级 A/B/C  │
│  · origin_id 解析    │──────────────→│  · 去重决策 keep/drop│
│  · 原始特征标记      │    .parquet   │  · 采样上限          │
├─────────────────────┤               │  · train/val 分割    │
│ resolve_dedup.py     │               │                      │
│  · 同源分组          │    +          │  读取 ↑              │
│  · 相似度分数        │──────────────→│  audit_policy.yaml   │
│  · 跨数据集标记      │               │  (可配置策略)        │
└─────────────────────┘               └──────────────────────┘
                                               │
                                      ┌────────┴────────┐
                                      │ generate_audit_  │
                                      │ report.py        │
                                      │  · 8 张审计表    │
                                      │  · 12 个问题     │
                                      └─────────────────┘
```

**好处**:
- 标记只跑一次（~58 万行，~10 分钟）
- 修改策略后只需重跑 Phase 2（~1 分钟）
- 策略可版本化追踪（YAML 文件纳入 git）

---

## 前置条件

### 安装依赖

```bash
pip install pandas pyarrow tabulate pyyaml
```

### 下载数据集

```bash
# ThinkStream (MIT license, 直接下载)
huggingface-cli download CASIA-IVA-Lab/ThinkStream \
    --local-dir data/thinkstream_raw --repo-type dataset

# Streamo-Instruct-465K (gated, 需申请)
huggingface-cli login
huggingface-cli download maifoundations/Streamo-Instruct-465K \
    --local-dir data/streamo_raw --repo-type dataset
```

---

## 四步审计流程

### Step 1: 构建统一资产表 (Phase 1)

```bash
python scripts/audit/build_video_asset_registry.py \
    --streamo-path data/streamo_raw \
    --thinkstream-path data/thinkstream_raw \
    --output data/audit/video_asset_registry.parquet
```

**做了什么** (纯标记, 不做判断):
- 读取两个数据集所有行
- 从 `video_path` 解析 `video_origin_id`（YouTube ID、ActivityNet ID 等）
- 推断底层来源（Kinetics-700、LLaVA-Video 等）
- 提取时间特征（has_user_timestamp、has_support_span 等原始布尔值）
- 提取文本预览和数量统计

**输出**: `data/audit/video_asset_registry.parquet`  
**耗时**: ~2-10 分钟

### Step 2: 四层同源标记 (Phase 1)

```bash
python scripts/audit/resolve_dedup.py \
    --input data/audit/video_asset_registry.parquet \
    --output data/audit/video_asset_registry_marked.parquet
```

**做了什么** (纯标记, 不做判断):
- 层 1: 按 `origin_id` 分组 → `dedup_group_id`, `dedup_group_size`, `is_cross_dataset_overlap`
- 层 2: 无 origin_id 的行按文件名模糊匹配 → `metadata_match_target`, `metadata_match_score`
- 层 3: 视觉 embedding 相似度 → `visual_sim_max`, `visual_sim_partner` (可选, 需 `--embedding-dir`)
- 层 4: 同 origin 内标注文本相似度 → `semantic_sim_max`, `semantic_sim_partner`

**输出**: `data/audit/video_asset_registry_marked.parquet` + `dedup_marks_summary.json`  
**耗时**: ~5-15 分钟（不含层 3）

### Step 3: 策略处理 (Phase 2)

```bash
python scripts/audit/apply_audit_policy.py \
    --input data/audit/video_asset_registry_marked.parquet \
    --policy scripts/audit/audit_policy.yaml \
    --output data/audit/video_asset_registry_final.parquet
```

**做了什么** (所有决策在此执行):
1. **可用性分级**: 根据策略中的条件, 分配 `usable_level` (A/B/C) 和子标签
2. **去重决策**: 根据标记的相似度分数, 标记 `dedup_action` (keep/drop_semantic_dup/drop_visual_dup/drop_over_cap)
3. **采样上限**: 每个 origin 最多保留 N 条（默认 12）
4. **Train/Val 分割**: 按 origin_video_id 分割, 保证不泄漏

**输出**:
- `data/audit/video_asset_registry_final.parquet` — 最终资产表
- `data/audit/train_registry.parquet` — train 子集
- `data/audit/policy_report.json` — 策略执行报告

**耗时**: ~1-2 分钟

**修改策略后只需重跑此步**:

```bash
# 编辑策略
vim scripts/audit/audit_policy.yaml

# 重跑 Phase 2 (无需重跑 Step 1-2)
python scripts/audit/apply_audit_policy.py \
    --input data/audit/video_asset_registry_marked.parquet \
    --policy scripts/audit/audit_policy.yaml \
    --output data/audit/video_asset_registry_final.parquet
```

### Step 4: 生成审计报告

```bash
python scripts/audit/generate_audit_report.py \
    --input data/audit/video_asset_registry_final.parquet \
    --output-dir data/audit/report
```

**输出**:

```
data/audit/report/
├── audit_report.md              # 主报告 (Markdown, 重点查看)
├── audit_summary.json
├── table1_overview.csv          # 数据源总览
├── table2_source_distribution.csv  # 底层来源分布
├── table3_clip_distribution.csv # 多 clip origin 分析
├── table4_task_distribution.csv # 任务标签分布
├── table5_temporal_quality.csv  # 时序监督质量
├── table6_usability.csv         # 可用性三分表
├── table7_dedup_risk.csv        # 去重风险
└── table8_sampling_checklist.csv  # 分层抽样质检
```

---

## 快速一键运行

```bash
mkdir -p data/audit

# Phase 1: 标记 (一次性)
python scripts/audit/build_video_asset_registry.py \
    --streamo-path data/streamo_raw \
    --thinkstream-path data/thinkstream_raw \
    --output data/audit/video_asset_registry.parquet \
&& python scripts/audit/resolve_dedup.py \
    --input data/audit/video_asset_registry.parquet \
    --output data/audit/video_asset_registry_marked.parquet

# Phase 2: 决策 (可反复调)
python scripts/audit/apply_audit_policy.py \
    --input data/audit/video_asset_registry_marked.parquet \
    --policy scripts/audit/audit_policy.yaml \
    --output data/audit/video_asset_registry_final.parquet \
&& python scripts/audit/generate_audit_report.py \
    --input data/audit/video_asset_registry_final.parquet \
    --output-dir data/audit/report

echo "审计完成! 查看: data/audit/report/audit_report.md"
```

---

## 策略配置详解

策略文件 `scripts/audit/audit_policy.yaml` 包含 5 个部分:

### 1. usability — 可用性分级

```yaml
usability:
  level_A:
    conditions:
      any_of:              # 满足任意一组即为 A
        - has_think_timestamp: true
          has_user_timestamp: true
        - has_support_span: true
          min_estimated_duration_sec: 32
```

**调参建议**: 首批数据不够时, 可降低 `min_estimated_duration_sec` (如从 32 降到 24)。

### 2. usability_tags — 可用性子标签

定义每个 `usable_for_*` 标签的判定条件。影响后续数据构造时的样本分流。

### 3. dedup — 去重阈值

```yaml
dedup:
  semantic_sim_drop_threshold: 0.95   # 降低此值 → 更激进去重
  visual_sim_drop_threshold: 0.98
  cross_dataset_policy: "mark_only"   # 改为 "prefer_thinkstream" → 自动丢弃重叠的 Streamo
```

### 4. sampling — 采样上限

```yaml
sampling:
  max_per_origin: 12      # 降低 → 更多样化; 升高 → 更多样本
  priority_order:
    - thinkstream          # 超限时优先保留 ThinkStream (有 think)
    - streamo
```

### 5. split — Train/Val 分割

```yaml
split:
  val_ratio: 0.05          # 5% 用于验证
  split_by: "video_origin_id"  # 保证同源不泄漏
```

---

## 典型调参场景

### 场景 1: 首批数据太少

```yaml
# 放宽 A 类标准
usability:
  level_A:
    conditions:
      any_of:
        - has_user_timestamp: true    # 只要有时间戳就是 A
```

### 场景 2: 跨数据集重叠太多

```yaml
# 自动丢弃重叠的 Streamo 样本
dedup:
  cross_dataset_policy: "prefer_thinkstream"
```

### 场景 3: 需要更多 recall 样本

```yaml
# 降低 recall 时长门槛
usability_tags:
  usable_for_recall:
    all_of:
      min_estimated_duration_sec: 20    # 从 32 降到 20
    override_for_thinkstream:
      min_estimated_duration_sec: 6     # 从 8 降到 6
```

---

## 报告解读

### 重点关注指标

| 指标 | 位置 | 健康范围 | 异常处理 |
|------|------|---------|---------|
| 跨数据集同源比例 | 表7 | < 30% | 过高 → 调 `cross_dataset_policy` |
| 有时间戳比例 | 表5 | > 80% | 过低 → 大量 C 类, 需补充数据 |
| A 类比例 | 表6 | > 30% | 过低 → 放宽 A 类条件 |
| 适合 recall 比例 | 表6 | > 15% | 过低 → 降低时长门槛 |
| 语义近重 | 表7 | < 5% | 过高 → 降低 `semantic_sim_drop_threshold` |
| 最终 train 行数 | policy_report.json | > 50K | 过低 → 放宽去重和上限 |

---

## 文件结构总览

```
scripts/audit/
├── __init__.py
├── build_video_asset_registry.py   # Phase 1: 字段提取 (纯标记)
├── resolve_dedup.py                # Phase 1: 同源标记 (纯标记)
├── apply_audit_policy.py           # Phase 2: 策略处理 (决策)
├── audit_policy.yaml               # Phase 2: 策略配置
└── generate_audit_report.py        # 报告生成

data/audit/
├── video_asset_registry.parquet           # Step 1 输出
├── video_asset_registry.csv               # Step 1 预览
├── video_asset_registry_marked.parquet    # Step 2 输出
├── dedup_marks_summary.json               # Step 2 统计
├── video_asset_registry_final.parquet     # Step 3 输出
├── train_registry.parquet                 # Step 3 train 子集
├── policy_report.json                     # Step 3 策略报告
└── report/                                # Step 4 输出
    ├── audit_report.md
    ├── audit_summary.json
    └── table[1-8]_*.csv
```

---

## 下一步

审计通过后:

1. 从 `train_registry.parquet` 中按 `usable_level == "A"` + `usable_for_recall == True` 筛选首批 50 视频
2. 进入 `docs/agent_data_construction_detail.md` 第 5 章 Stage 0-6 数据构造流程
