"""
数据审计脚本 3: 生成审计报告

从去重后的 video_asset_registry 生成 8 张审计报表和 12 个审计问题答案。

Usage:
    python scripts/audit/generate_audit_report.py \
        --input data/audit/video_asset_registry_deduped.parquet \
        --output-dir data/audit/report
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def table1_overview(df: pd.DataFrame) -> pd.DataFrame:
    """表 1: 数据源总览表"""
    rows = []
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        n = len(sub)
        n_clips = sub["video_name"].nunique()
        n_origins = sub["video_origin_id"].nunique()
        avg_per = round(n / max(n_origins, 1), 2)
        miss_path = round((sub["video_locator_raw"].isna() | (sub["video_locator_raw"] == "")).mean() * 100, 1)
        miss_ts = round((~sub["has_user_timestamp"]).mean() * 100, 1)
        rows.append({
            "dataset": ds,
            "num_rows": n,
            "num_unique_clips": n_clips,
            "num_unique_origin_videos": n_origins,
            "avg_samples_per_origin": avg_per,
            "missing_video_path_pct": miss_path,
            "missing_user_timestamp_pct": miss_ts,
        })
    # 合计行
    total = len(df)
    rows.append({
        "dataset": "TOTAL",
        "num_rows": total,
        "num_unique_clips": df["video_name"].nunique(),
        "num_unique_origin_videos": df["video_origin_id"].nunique(),
        "avg_samples_per_origin": round(total / max(df["video_origin_id"].nunique(), 1), 2),
        "missing_video_path_pct": round((df["video_locator_raw"].isna() | (df["video_locator_raw"] == "")).mean() * 100, 1),
        "missing_user_timestamp_pct": round((~df["has_user_timestamp"]).mean() * 100, 1),
    })
    return pd.DataFrame(rows)


def table2_source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """表 2: 来源分布表"""
    rows = []
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        for src, grp in sub.groupby("underlying_source"):
            # 检查该来源在另一个数据集中是否存在
            other_ds = [d for d in df["source_dataset"].unique() if d != ds]
            overlap_origins = set()
            for od in other_ds:
                other = df[df["source_dataset"] == od]
                overlap_origins = grp["video_origin_id"].isin(other["video_origin_id"]).sum()

            rows.append({
                "source_dataset": ds,
                "underlying_video_source": src,
                "num_rows": len(grp),
                "num_unique_origin_videos": grp["video_origin_id"].nunique(),
                "overlap_rows_with_other_dataset": int(overlap_origins),
            })
    return pd.DataFrame(rows).sort_values(["source_dataset", "num_rows"], ascending=[True, False])


def table3_clip_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """表 3: 时间切片分布表 (按 origin_video_id 聚合)"""
    rows = []
    for oid, grp in df.groupby("video_origin_id"):
        if len(grp) < 2:
            continue  # 只关注多 clip 的 origin
        durations = grp["duration_sec"].dropna()
        rows.append({
            "origin_video_id": oid,
            "source_datasets": ",".join(sorted(grp["source_dataset"].unique())),
            "num_clips": len(grp),
            "clip_duration_mean": round(durations.mean(), 1) if len(durations) > 0 else None,
            "clip_duration_std": round(durations.std(), 1) if len(durations) > 1 else None,
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values("num_clips", ascending=False)
    return result


def table4_task_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """表 4: 任务标签分布表"""
    rows = []
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        for _, grp in sub.groupby(["task_type_raw", "temporal_scope", "response_format"]):
            row = grp.iloc[0]
            rows.append({
                "source_dataset": ds,
                "task_type": row["task_type_raw"],
                "temporal_scope": row["temporal_scope"],
                "response_format": row["response_format"],
                "content_dimension": row["content_dimension"],
                "count": len(grp),
                "percentage": round(len(grp) / len(sub) * 100, 2),
            })
    return pd.DataFrame(rows).sort_values(["source_dataset", "count"], ascending=[True, False])


def table5_temporal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """表 5: 时序监督质量表"""
    rows = []
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        n = len(sub)
        rows.append({
            "source_dataset": ds,
            "total_rows": n,
            "has_user_timestamp_pct": round(sub["has_user_timestamp"].mean() * 100, 1),
            "has_assistant_timestamp_pct": round(sub["has_assistant_timestamp"].mean() * 100, 1),
            "has_think_timestamp_pct": round(sub["has_think_timestamp"].mean() * 100, 1),
            "has_support_span_pct": round(sub["has_support_span"].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def table6_usability(df: pd.DataFrame) -> pd.DataFrame:
    """表 6: 可用性三分表"""
    rows = []
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        n = len(sub)
        rows.append({
            "source_dataset": ds,
            "total_rows": n,
            "usable_for_protocol_sft_pct": round(sub["usable_for_protocol_sft"].mean() * 100, 1),
            "usable_for_recall_pct": round(sub["usable_for_recall"].mean() * 100, 1),
            "usable_for_timing_pct": round(sub["usable_for_timing"].mean() * 100, 1),
            "usable_for_multiturn_pct": round(sub["usable_for_multiturn"].mean() * 100, 1),
            "usable_for_rl_verifiable_pct": round(sub["usable_for_rl_verifiable"].mean() * 100, 1),
            "only_as_weak_anchor_pct": round(sub["only_as_weak_anchor"].mean() * 100, 1),
            "level_A_pct": round((sub["usable_level"] == "A").mean() * 100, 1),
            "level_B_pct": round((sub["usable_level"] == "B").mean() * 100, 1),
            "level_C_pct": round((sub["usable_level"] == "C").mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def table7_dedup_risk(df: pd.DataFrame) -> pd.DataFrame:
    """表 7: 去重风险表"""
    n = len(df)
    rows = []

    # 精确重复 (同 origin + 同 clip 时间)
    if "clip_start_sec" in df.columns:
        dup_key = df[["video_origin_id", "clip_start_sec", "clip_end_sec"]].dropna()
        exact_dup = dup_key.duplicated().sum()
    else:
        exact_dup = 0

    rows.append({"metric": "exact_duplicate_rows", "value": exact_dup, "percentage": round(exact_dup / n * 100, 2)})

    # 同 origin 多样本
    multi_origin = df.groupby("video_origin_id").size()
    multi_rows = (multi_origin > 1).sum()
    rows.append({
        "metric": "origins_with_multiple_samples",
        "value": int(multi_rows),
        "percentage": round(multi_rows / max(df["video_origin_id"].nunique(), 1) * 100, 2),
    })

    # 跨数据集同源
    if "is_cross_dataset_overlap" in df.columns:
        cross_origins = df[df["is_cross_dataset_overlap"]]["video_origin_id"].nunique()
        cross_rows = df["is_cross_dataset_overlap"].sum()
    else:
        cross_origins = 0
        cross_rows = 0
    rows.append({
        "metric": "cross_dataset_same_origin_videos",
        "value": int(cross_origins),
        "percentage": round(cross_origins / max(df["video_origin_id"].nunique(), 1) * 100, 2),
    })
    rows.append({
        "metric": "cross_dataset_same_origin_rows",
        "value": int(cross_rows),
        "percentage": round(cross_rows / n * 100, 2),
    })

    # 语义近重
    if "is_semantic_near_dup" in df.columns:
        sem_dup = df["is_semantic_near_dup"].sum()
    else:
        sem_dup = 0
    rows.append({"metric": "semantic_near_duplicate_rows", "value": int(sem_dup), "percentage": round(sem_dup / n * 100, 2)})

    # 视觉近重
    if "is_visual_near_dup" in df.columns:
        vis_dup = df["is_visual_near_dup"].sum()
    else:
        vis_dup = 0
    rows.append({"metric": "visual_near_duplicate_rows", "value": int(vis_dup), "percentage": round(vis_dup / n * 100, 2)})

    # 超限降级
    if "is_over_cap" in df.columns:
        cap = df["is_over_cap"].sum()
    else:
        cap = 0
    rows.append({"metric": "over_sampling_cap_rows", "value": int(cap), "percentage": round(cap / n * 100, 2)})

    return pd.DataFrame(rows)


def table8_sampling_checklist(df: pd.DataFrame, n_per_bucket: int = 50) -> pd.DataFrame:
    """表 8: 分层抽样质检表 (生成抽样 ID 列表)"""
    buckets = {
        "high_recallability": df[df["usable_for_recall"]],
        "low_recallability": df[~df["usable_for_recall"] & (df["usable_level"] != "C")],
        "kinetics700": df[df["underlying_source"] == "kinetics700"],
        "activitynet": df[df["underlying_source"] == "activitynet"],
        "multiturn": df[df["usable_for_multiturn"]],
        "cross_dataset_overlap": df[df.get("is_cross_dataset_overlap", pd.Series(False, index=df.index)) == True]
        if "is_cross_dataset_overlap" in df.columns else df.head(0),
    }

    rows = []
    for bucket_name, bucket_df in buckets.items():
        n_available = len(bucket_df)
        n_sample = min(n_per_bucket, n_available)
        if n_sample > 0:
            sampled = bucket_df.sample(n_sample, random_state=42)
            sample_ids = sampled["asset_id"].tolist()
        else:
            sample_ids = []
        rows.append({
            "bucket": bucket_name,
            "total_available": n_available,
            "sampled_count": n_sample,
            "sample_asset_ids": json.dumps(sample_ids[:10]) + ("..." if len(sample_ids) > 10 else ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12 个审计问题
# ---------------------------------------------------------------------------

def answer_12_questions(df: pd.DataFrame) -> list:
    """回答 12 个关键审计问题。"""
    answers = []

    # Q1
    for ds in sorted(df["source_dataset"].unique()):
        n = len(df[df["source_dataset"] == ds])
        answers.append(f"Q1. {ds} 行数: {n}")

    # Q2
    for ds in sorted(df["source_dataset"].unique()):
        n = df[df["source_dataset"] == ds]["video_name"].nunique()
        answers.append(f"Q2. {ds} 唯一 clip: {n}")

    # Q3
    for ds in sorted(df["source_dataset"].unique()):
        n = df[df["source_dataset"] == ds]["video_origin_id"].nunique()
        answers.append(f"Q3. {ds} 唯一原视频: {n}")

    # Q4
    if "is_cross_dataset_overlap" in df.columns:
        cross = df[df["is_cross_dataset_overlap"]]["video_origin_id"].nunique()
        total_origins = df["video_origin_id"].nunique()
        answers.append(f"Q4. 跨数据集同源 origin 数: {cross} / {total_origins} ({cross / max(total_origins, 1) * 100:.1f}%)")
    else:
        answers.append("Q4. 未计算跨数据集重叠 (需运行 resolve_dedup.py)")

    # Q5
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        dist = sub["underlying_source"].value_counts()
        top3 = ", ".join(f"{k}: {v}" for k, v in dist.head(3).items())
        answers.append(f"Q5. {ds} 底层来源 top3: {top3}")

    # Q6
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        sizes = sub.groupby("video_origin_id").size()
        answers.append(f"Q6. {ds} 每 origin 平均样本: {sizes.mean():.1f}, 中位数: {sizes.median():.0f}, max: {sizes.max()}")

    # Q7
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        dist = sub["task_type_raw"].value_counts(normalize=True).head(5)
        top = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in dist.items())
        answers.append(f"Q7. {ds} 任务分布 top5: {top}")

    # Q8
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        dist = sub["temporal_scope"].value_counts(normalize=True)
        out = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in dist.items())
        answers.append(f"Q8. {ds} temporal_scope: {out}")

    # Q9
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        dist = sub["response_format"].value_counts(normalize=True)
        out = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in dist.items())
        answers.append(f"Q9. {ds} response_format: {out}")

    # Q10
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        ts_pct = sub["has_user_timestamp"].mean() * 100
        span_pct = sub["has_support_span"].mean() * 100
        answers.append(f"Q10. {ds} 有时间戳: {ts_pct:.1f}%, 有 support span: {span_pct:.1f}%")

    # Q11
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        recall_pct = sub["usable_for_recall"].mean() * 100
        answers.append(f"Q11. {ds} 适合 recall: {recall_pct:.1f}%")

    # Q12
    for ds in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == ds]
        weak_pct = sub["only_as_weak_anchor"].mean() * 100
        answers.append(f"Q12. {ds} 仅弱锚点: {weak_pct:.1f}%")

    return answers


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def generate_markdown_report(
    df: pd.DataFrame,
    tables: dict,
    questions: list,
    output_path: Path,
) -> None:
    """生成 Markdown 格式的审计报告。"""
    lines = [
        "# 数据审计报告",
        "",
        f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 总行数: {len(df):,}",
        f"> 总唯一 origin: {df['video_origin_id'].nunique():,}",
        "",
        "---",
        "",
        "## 12 个关键审计问题",
        "",
    ]
    for q in questions:
        lines.append(f"- {q}")
    lines.append("")

    for name, tbl in tables.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(tbl.to_markdown(index=False))
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown 报告: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成数据审计报告")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="去重后的 video_asset_registry parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/audit/report",
        help="报告输出目录",
    )
    args = parser.parse_args()

    logger.info(f"读取: {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"总行数: {len(df)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成 8 张表
    tables = {}

    logger.info("生成表 1: 数据源总览...")
    t1 = table1_overview(df)
    t1.to_csv(out_dir / "table1_overview.csv", index=False)
    tables["表 1: 数据源总览"] = t1

    logger.info("生成表 2: 来源分布...")
    t2 = table2_source_distribution(df)
    t2.to_csv(out_dir / "table2_source_distribution.csv", index=False)
    tables["表 2: 来源分布"] = t2

    logger.info("生成表 3: 时间切片分布 (top 50)...")
    t3 = table3_clip_distribution(df)
    t3.head(50).to_csv(out_dir / "table3_clip_distribution.csv", index=False)
    tables["表 3: 时间切片分布 (多 clip origin, top 50)"] = t3.head(20)

    logger.info("生成表 4: 任务标签分布...")
    t4 = table4_task_distribution(df)
    t4.to_csv(out_dir / "table4_task_distribution.csv", index=False)
    tables["表 4: 任务标签分布"] = t4

    logger.info("生成表 5: 时序监督质量...")
    t5 = table5_temporal_quality(df)
    t5.to_csv(out_dir / "table5_temporal_quality.csv", index=False)
    tables["表 5: 时序监督质量"] = t5

    logger.info("生成表 6: 可用性三分...")
    t6 = table6_usability(df)
    t6.to_csv(out_dir / "table6_usability.csv", index=False)
    tables["表 6: 可用性三分"] = t6

    logger.info("生成表 7: 去重风险...")
    t7 = table7_dedup_risk(df)
    t7.to_csv(out_dir / "table7_dedup_risk.csv", index=False)
    tables["表 7: 去重风险"] = t7

    logger.info("生成表 8: 分层抽样质检...")
    t8 = table8_sampling_checklist(df)
    t8.to_csv(out_dir / "table8_sampling_checklist.csv", index=False)
    tables["表 8: 分层抽样质检"] = t8

    # 12 个问题
    logger.info("回答 12 个审计问题...")
    questions = answer_12_questions(df)

    # Markdown 报告
    generate_markdown_report(df, tables, questions, out_dir / "audit_report.md")

    # JSON 汇总
    summary = {
        "total_rows": len(df),
        "total_origins": int(df["video_origin_id"].nunique()),
        "questions": questions,
    }
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 汇总: {out_dir / 'audit_summary.json'}")

    logger.info("✓ 审计报告生成完成")


if __name__ == "__main__":
    main()
