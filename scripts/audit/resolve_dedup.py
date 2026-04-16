"""
数据审计脚本 2: 四层同源识别与标记

【Phase 1 — 纯标记】
只标记去重相关的分数和分组信息，不做任何取舍决策。
决策（保留/丢弃/降级/采样上限）由 apply_audit_policy.py 执行。

四层标记:
  层 1: 字符串级同源 → dedup_group_id, is_cross_dataset_overlap
  层 2: 元数据级同源 → metadata_match_target (匹配到的 origin_id)
  层 3: 视觉近重 → visual_sim_max (最大相似度分数)
  层 4: 语义近重 → semantic_sim_max (最大语义相似度分数)

Usage:
    python scripts/audit/resolve_dedup.py \
        --input data/audit/video_asset_registry.parquet \
        --output data/audit/video_asset_registry_marked.parquet
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 层 1: 字符串级同源 → 分组 + 跨数据集标记
# ---------------------------------------------------------------------------

def layer1_string_match(df: pd.DataFrame) -> pd.DataFrame:
    """为每个 origin_id 分配 dedup_group_id, 标记跨数据集重叠。"""
    logger.info("=== 层 1: 字符串级同源标记 ===")

    origin_ids = df["video_origin_id"].fillna("__NULL__")
    unique_origins = sorted(origin_ids.unique())
    origin_to_group = {oid: f"grp_{i:06d}" for i, oid in enumerate(unique_origins)}

    df["dedup_group_id"] = origin_ids.map(origin_to_group)
    df["dedup_group_size"] = df.groupby("dedup_group_id")["dedup_group_id"].transform("count")

    # 跨数据集标记
    cross_origins: Set[str] = set()
    for oid, grp in df.groupby("video_origin_id"):
        if grp["source_dataset"].nunique() > 1:
            cross_origins.add(oid)

    df["is_cross_dataset_overlap"] = df["video_origin_id"].isin(cross_origins)

    n_groups = df["dedup_group_id"].nunique()
    n_cross = len(cross_origins)
    n_cross_rows = int(df["is_cross_dataset_overlap"].sum())
    logger.info(f"  唯一 origin 数: {n_groups}")
    logger.info(f"  跨数据集同源 origin 数: {n_cross} (涉及 {n_cross_rows} 行)")

    return df


# ---------------------------------------------------------------------------
# 层 2: 元数据级同源 → 匹配目标
# ---------------------------------------------------------------------------

def layer2_metadata_match(
    df: pd.DataFrame,
    duration_tol: float = 1.0,
    name_sim_threshold: float = 0.8,
) -> pd.DataFrame:
    """对无有效 origin_id 的行, 尝试通过元数据匹配并标记。"""
    logger.info("=== 层 2: 元数据级同源标记 ===")

    df["metadata_match_target"] = None  # 匹配到的 origin_id
    df["metadata_match_score"] = 0.0

    no_origin_mask = (
        df["video_origin_id"].isna()
        | (df["video_origin_id"] == "")
        | (df["video_origin_id"] == "__NULL__")
    )
    n_no_origin = int(no_origin_mask.sum())
    logger.info(f"  无有效 origin_id 的行: {n_no_origin}")

    if n_no_origin == 0:
        logger.info("  全部已有 origin_id, 跳过层 2")
        return df

    has_origin = df[~no_origin_mask]
    matched = 0

    for idx in df[no_origin_mask].index:
        row = df.loc[idx]
        dur = row.get("estimated_video_duration_sec")
        name = row.get("video_name", "")
        if dur is None or not name:
            continue

        candidates = has_origin[
            has_origin["estimated_video_duration_sec"].notna()
            & ((has_origin["estimated_video_duration_sec"] - dur).abs() < duration_tol)
        ]

        best_score = 0.0
        best_target = None
        for _, cand in candidates.iterrows():
            sim = _name_similarity(name, cand.get("video_name", ""))
            if sim > best_score:
                best_score = sim
                best_target = cand["video_origin_id"]

        if best_score >= name_sim_threshold and best_target is not None:
            df.at[idx, "metadata_match_target"] = best_target
            df.at[idx, "metadata_match_score"] = best_score
            matched += 1

    logger.info(f"  元数据匹配成功: {matched} 行")
    return df


def _name_similarity(a: str, b: str) -> float:
    """文件名 trigram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    a = a.lower().replace(".mp4", "").replace("_", "").replace("-", "")
    b = b.lower().replace(".mp4", "").replace("_", "").replace("-", "")
    if a == b:
        return 1.0
    tri_a = {a[i : i + 3] for i in range(len(a) - 2)}
    tri_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not tri_a or not tri_b:
        return 0.0
    return len(tri_a & tri_b) / len(tri_a | tri_b)


# ---------------------------------------------------------------------------
# 层 3: 视觉近重 → 相似度分数
# ---------------------------------------------------------------------------

def layer3_visual_similarity(
    df: pd.DataFrame,
    embedding_dir: Optional[str] = None,
) -> pd.DataFrame:
    """计算同 group 内的视觉 embedding 相似度, 记录最大值。"""
    logger.info("=== 层 3: 视觉近重标记 ===")

    df["visual_sim_max"] = 0.0
    df["visual_sim_partner"] = None

    if not embedding_dir or not Path(embedding_dir).exists():
        logger.info("  embedding_dir 不存在, 跳过层 3 (所有 visual_sim_max = 0)")
        return df

    emb_dir = Path(embedding_dir)
    emb_files = sorted(emb_dir.rglob("*.npy"))
    if not emb_files:
        logger.info("  未找到 .npy 文件, 跳过层 3")
        return df

    logger.info(f"  加载 {len(emb_files)} 个 embedding")
    emb_map: Dict[str, np.ndarray] = {}
    for f in emb_files:
        emb_map[f.stem] = np.load(f)

    checked = 0
    for gid, grp in df.groupby("dedup_group_id"):
        if len(grp) <= 1:
            continue
        ids = grp["asset_id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                emb_i = emb_map.get(ids[i])
                emb_j = emb_map.get(ids[j])
                if emb_i is None or emb_j is None:
                    continue
                sim = _cosine_sim(emb_i, emb_j)
                checked += 1
                # 记录每个样本的最大相似度
                idx_i = grp.index[grp["asset_id"] == ids[i]][0]
                idx_j = grp.index[grp["asset_id"] == ids[j]][0]
                if sim > df.at[idx_i, "visual_sim_max"]:
                    df.at[idx_i, "visual_sim_max"] = sim
                    df.at[idx_i, "visual_sim_partner"] = ids[j]
                if sim > df.at[idx_j, "visual_sim_max"]:
                    df.at[idx_j, "visual_sim_max"] = sim
                    df.at[idx_j, "visual_sim_partner"] = ids[i]

    logger.info(f"  比较了 {checked} 对 embedding")
    high_sim = (df["visual_sim_max"] > 0.95).sum()
    logger.info(f"  visual_sim_max > 0.95 的行数: {high_sim}")
    return df


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 层 4: 语义近重 → 标注文本相似度分数
# ---------------------------------------------------------------------------

def layer4_semantic_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """同 origin 内的标注文本相似度, 记录最大值。"""
    logger.info("=== 层 4: 语义近重标记 ===")

    df["semantic_sim_max"] = 0.0
    df["semantic_sim_partner"] = None

    for oid, grp in df.groupby("video_origin_id"):
        if len(grp) <= 1:
            continue

        indices = grp.index.tolist()
        texts = [_make_text_key(df.loc[i]) for i in indices]

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = _token_jaccard(texts[i], texts[j])
                if sim > df.at[indices[i], "semantic_sim_max"]:
                    df.at[indices[i], "semantic_sim_max"] = sim
                    df.at[indices[i], "semantic_sim_partner"] = df.at[indices[j], "asset_id"]
                if sim > df.at[indices[j], "semantic_sim_max"]:
                    df.at[indices[j], "semantic_sim_max"] = sim
                    df.at[indices[j], "semantic_sim_partner"] = df.at[indices[i], "asset_id"]

    high_sim = (df["semantic_sim_max"] > 0.90).sum()
    logger.info(f"  semantic_sim_max > 0.90 的行数: {high_sim}")
    return df


def _make_text_key(row: pd.Series) -> str:
    """拼接标注文本用于比较。"""
    parts = [
        str(row.get("task_type_raw", "")),
        str(row.get("question_text_preview", "")),
        str(row.get("response_text_preview", "")),
    ]
    return " ".join(parts).lower()


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# 汇总标记统计
# ---------------------------------------------------------------------------

def summarize_marks(df: pd.DataFrame, output_dir: Path) -> None:
    """输出标记统计摘要。"""
    summary = {
        "total_rows": len(df),
        "total_origin_videos": int(df["video_origin_id"].nunique()),
        "total_dedup_groups": int(df["dedup_group_id"].nunique()),
        "layer1_cross_dataset": {
            "origin_count": int(df[df["is_cross_dataset_overlap"]]["video_origin_id"].nunique())
            if "is_cross_dataset_overlap" in df.columns else 0,
            "row_count": int(df["is_cross_dataset_overlap"].sum())
            if "is_cross_dataset_overlap" in df.columns else 0,
        },
        "layer2_metadata_matched": int((df["metadata_match_target"].notna()).sum())
        if "metadata_match_target" in df.columns else 0,
        "layer3_visual_high_sim": int((df["visual_sim_max"] > 0.95).sum())
        if "visual_sim_max" in df.columns else 0,
        "layer4_semantic_high_sim": int((df["semantic_sim_max"] > 0.90).sum())
        if "semantic_sim_max" in df.columns else 0,
        "group_size_distribution": df["dedup_group_size"].describe().to_dict()
        if "dedup_group_size" in df.columns else {},
    }

    report_path = output_dir / "dedup_marks_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"标记统计摘要: {report_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: 四层同源识别与标记 (纯标记, 不做决策)"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="data/audit/video_asset_registry_marked.parquet",
    )
    parser.add_argument("--embedding-dir", type=str, default=None,
                        help="视觉 embedding 目录 (层 3, 可选)")
    args = parser.parse_args()

    logger.info(f"读取: {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"总行数: {len(df)}")

    df = layer1_string_match(df)
    df = layer2_metadata_match(df)
    df = layer3_visual_similarity(df, args.embedding_dir)
    df = layer4_semantic_similarity(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"标记后资产表: {out_path}")

    summarize_marks(df, out_path.parent)


if __name__ == "__main__":
    main()
