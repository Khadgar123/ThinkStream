"""
数据审计脚本 3: 策略处理

【Phase 2 — 决策】
读取 Phase 1 的标记结果 + 策略配置文件 (YAML/JSON)，执行:
  1. 可用性分级 (A/B/C)
  2. 去重决策 (keep/drop)
  3. 采样上限
  4. train/val 按 origin 分割

策略可反复调整重跑, 无需重新执行 Phase 1 的标记脚本。

Usage:
    python scripts/audit/apply_audit_policy.py \
        --input data/audit/video_asset_registry_marked.parquet \
        --policy scripts/audit/audit_policy.yaml \
        --output data/audit/video_asset_registry_final.parquet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 策略配置加载
# ---------------------------------------------------------------------------

def load_policy(path: str) -> Dict[str, Any]:
    """加载策略配置。支持 YAML 和 JSON。"""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("需要安装 pyyaml: pip install pyyaml")
    elif p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的策略文件格式: {p.suffix} (支持 .yaml/.json)")


DEFAULT_POLICY = {
    # --- 可用性分级规则 ---
    "usability": {
        "level_A": {
            "description": "直接可迁移: 有完整时间信息, 可 agent 化",
            "conditions": {
                # 满足任意一组即为 A 类
                "any_of": [
                    # 组 1: 有 think 时间戳 (ThinkStream 强项)
                    {"has_think_timestamp": True, "has_user_timestamp": True},
                    # 组 2: 有 support span + 足够时长 (Streamo recall 候选)
                    {"has_support_span": True, "min_estimated_duration_sec": 32},
                    # 组 3: 有完整时间 + timing 信息
                    {"has_user_timestamp": True, "has_assistant_timestamp": True, "has_support_span": True},
                ]
            }
        },
        "level_B": {
            "description": "部分可用: 有时间戳但不完整, 或短视频",
            "conditions": {
                "any_of": [
                    {"has_user_timestamp": True},
                    {"has_assistant_timestamp": True},
                ]
            }
        },
        # 既不是 A 也不是 B → C
        "level_C": {
            "description": "弱锚点或丢弃: 无时间信息或质量太低",
        }
    },

    # --- 可用性子标签规则 ---
    "usability_tags": {
        "usable_for_protocol_sft": {
            "any_of": [
                {"has_user_timestamp": True, "has_think_timestamp": True},
                {"has_user_timestamp": True, "has_assistant_timestamp": True},
            ]
        },
        "usable_for_recall": {
            "all_of": {
                "has_support_span": True,
                "min_estimated_duration_sec": 32,
            },
            "override_for_thinkstream": {
                "temporal_scope_raw_lower": "past",
                "min_estimated_duration_sec": 8,
            }
        },
        "usable_for_timing": {
            "all_of": {
                "has_user_timestamp": True,
                "has_assistant_timestamp": True,
            }
        },
        "usable_for_rl_verifiable": {
            "response_format_raw_in": ["Binary", "Multiple Choice", "Counting",
                                        "binary", "multiple_choice", "counting",
                                        "Multiple choice", "Yes/No"],
        },
    },

    # --- 去重决策 ---
    "dedup": {
        "semantic_sim_drop_threshold": 0.95,   # > 此阈值: 丢弃较差的一条
        "visual_sim_drop_threshold": 0.98,     # > 此阈值: 丢弃较差的一条
        "cross_dataset_policy": "mark_only",   # "mark_only" | "prefer_thinkstream" | "prefer_streamo"
        "metadata_match_merge": True,          # 层 2 匹配成功的是否合并 origin_id
    },

    # --- 采样上限 ---
    "sampling": {
        "max_per_origin": 12,
        "priority_order": ["thinkstream", "streamo"],  # 超限时优先保留的来源
        "level_priority": ["A", "B", "C"],             # 超限时优先保留的等级
    },

    # --- train/val 分割 ---
    "split": {
        "val_ratio": 0.05,
        "split_by": "video_origin_id",  # 按 origin 分割, 不按行分割
        "random_seed": 42,
    },
}


# ---------------------------------------------------------------------------
# 可用性分级
# ---------------------------------------------------------------------------

def _check_condition_group(row: pd.Series, conditions: list) -> bool:
    """检查一组条件 (any_of 中的一个)。"""
    for cond in conditions:
        match = True
        for key, expected in cond.items():
            if key == "min_estimated_duration_sec":
                val = row.get("estimated_video_duration_sec")
                if val is None or val < expected:
                    match = False
                    break
            elif key.endswith("_lower"):
                actual_key = key.replace("_lower", "")
                val = str(row.get(actual_key, "")).lower().strip()
                if val != str(expected).lower():
                    match = False
                    break
            elif key.endswith("_in"):
                actual_key = key.replace("_in", "")
                val = row.get(actual_key, "")
                if val not in expected:
                    match = False
                    break
            else:
                if row.get(key) != expected:
                    match = False
                    break
        if match:
            return True
    return False


def apply_usability(df: pd.DataFrame, policy: Dict) -> pd.DataFrame:
    """根据策略分配可用性等级和子标签。"""
    logger.info("=== 可用性分级 ===")

    usability = policy.get("usability", DEFAULT_POLICY["usability"])
    tags_policy = policy.get("usability_tags", DEFAULT_POLICY["usability_tags"])

    # 分级
    levels = []
    for _, row in df.iterrows():
        # 检查 A
        a_conds = usability.get("level_A", {}).get("conditions", {}).get("any_of", [])
        if a_conds and _check_condition_group(row, a_conds):
            levels.append("A")
            continue
        # 检查 B
        b_conds = usability.get("level_B", {}).get("conditions", {}).get("any_of", [])
        if b_conds and _check_condition_group(row, b_conds):
            levels.append("B")
            continue
        levels.append("C")

    df["usable_level"] = levels

    # 子标签: usable_for_protocol_sft
    tag = tags_policy.get("usable_for_protocol_sft", {})
    conds = tag.get("any_of", [])
    df["usable_for_protocol_sft"] = df.apply(
        lambda r: _check_condition_group(r, conds) if conds else False, axis=1
    )

    # 子标签: usable_for_recall
    recall_tag = tags_policy.get("usable_for_recall", {})
    recall_all = recall_tag.get("all_of", {})
    recall_ts_override = recall_tag.get("override_for_thinkstream", {})

    def check_recall(row):
        # 默认规则
        if (row.get("has_support_span", False)
                and row.get("estimated_video_duration_sec") is not None
                and row["estimated_video_duration_sec"] >= recall_all.get("min_estimated_duration_sec", 32)):
            return True
        # ThinkStream 特殊: 短视频可缩窗 forced-recall
        if row.get("source_dataset") == "thinkstream":
            ts = str(row.get("temporal_scope_raw", "")).lower().strip()
            dur = row.get("estimated_video_duration_sec")
            min_dur = recall_ts_override.get("min_estimated_duration_sec", 8)
            if ts == recall_ts_override.get("temporal_scope_raw_lower", "past") and dur is not None and dur >= min_dur:
                return True
        return False

    df["usable_for_recall"] = df.apply(check_recall, axis=1)

    # 子标签: usable_for_timing
    timing_tag = tags_policy.get("usable_for_timing", {})
    timing_all = timing_tag.get("all_of", {})
    df["usable_for_timing"] = (
        df["has_user_timestamp"].fillna(False)
        & df["has_assistant_timestamp"].fillna(False)
    )

    # 子标签: usable_for_rl_verifiable
    rl_tag = tags_policy.get("usable_for_rl_verifiable", {})
    rl_formats = rl_tag.get("response_format_raw_in", [])
    df["usable_for_rl_verifiable"] = df["response_format_raw"].isin(rl_formats)

    # 子标签: usable_for_multiturn
    df["usable_for_multiturn"] = df["num_questions"] > 1

    # 子标签: only_as_weak_anchor
    df["only_as_weak_anchor"] = (
        ~df["has_user_timestamp"].fillna(False)
        & ~df["has_think_timestamp"].fillna(False)
    )

    # 统计
    for level in ["A", "B", "C"]:
        cnt = (df["usable_level"] == level).sum()
        logger.info(f"  Level {level}: {cnt} ({cnt / len(df) * 100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 去重决策
# ---------------------------------------------------------------------------

def apply_dedup_decisions(df: pd.DataFrame, policy: Dict) -> pd.DataFrame:
    """根据标记分数和策略, 标记 dedup_action (keep/drop/downgrade)。"""
    logger.info("=== 去重决策 ===")

    dedup_policy = policy.get("dedup", DEFAULT_POLICY["dedup"])
    sem_thresh = dedup_policy.get("semantic_sim_drop_threshold", 0.95)
    vis_thresh = dedup_policy.get("visual_sim_drop_threshold", 0.98)
    merge_metadata = dedup_policy.get("metadata_match_merge", True)

    df["dedup_action"] = "keep"

    # 1. 语义近重: sim > threshold → 丢弃标注较少的一条
    if "semantic_sim_max" in df.columns:
        sem_dup_mask = df["semantic_sim_max"] > sem_thresh
        # 在同组内, 保留 num_conversations + num_thoughts 更多的
        for oid, grp in df[sem_dup_mask].groupby("video_origin_id"):
            if len(grp) <= 1:
                continue
            richness = grp["num_conversations"] + grp["num_thoughts"]
            sorted_idx = richness.sort_values(ascending=False).index
            # 保留最丰富的, 其余标记 drop
            for drop_idx in sorted_idx[1:]:
                if df.at[drop_idx, "semantic_sim_max"] > sem_thresh:
                    df.at[drop_idx, "dedup_action"] = "drop_semantic_dup"

    # 2. 视觉近重
    if "visual_sim_max" in df.columns:
        vis_dup_mask = (df["visual_sim_max"] > vis_thresh) & (df["dedup_action"] == "keep")
        n_vis_drop = 0
        for idx in df[vis_dup_mask].index:
            partner = df.at[idx, "visual_sim_partner"]
            if partner is not None:
                partner_rows = df[df["asset_id"] == partner]
                if not partner_rows.empty:
                    partner_idx = partner_rows.index[0]
                    # 保留来源优先级更高的
                    if df.at[idx, "source_dataset"] == "streamo" and df.at[partner_idx, "source_dataset"] == "thinkstream":
                        df.at[idx, "dedup_action"] = "drop_visual_dup"
                        n_vis_drop += 1
        if n_vis_drop > 0:
            logger.info(f"  视觉近重丢弃: {n_vis_drop}")

    # 3. 层 2 元数据匹配 → 合并 origin_id
    if merge_metadata and "metadata_match_target" in df.columns:
        merge_mask = df["metadata_match_target"].notna()
        merged = int(merge_mask.sum())
        if merged > 0:
            df.loc[merge_mask, "video_origin_id"] = df.loc[merge_mask, "metadata_match_target"]
            logger.info(f"  元数据匹配合并 origin_id: {merged} 行")

    # 统计
    action_counts = df["dedup_action"].value_counts()
    for action, cnt in action_counts.items():
        logger.info(f"  {action}: {cnt}")

    return df


# ---------------------------------------------------------------------------
# 采样上限
# ---------------------------------------------------------------------------

def apply_sampling_cap(df: pd.DataFrame, policy: Dict) -> pd.DataFrame:
    """每个 origin_video_id 最多保留 N 条, 超限标记 dedup_action=drop_over_cap。"""
    logger.info("=== 采样上限 ===")

    sampling = policy.get("sampling", DEFAULT_POLICY["sampling"])
    max_per = sampling.get("max_per_origin", 12)
    source_prio = sampling.get("priority_order", ["thinkstream", "streamo"])
    level_prio = sampling.get("level_priority", ["A", "B", "C"])

    source_rank = {s: i for i, s in enumerate(source_prio)}
    level_rank = {l: i for i, l in enumerate(level_prio)}

    df["_src_rank"] = df["source_dataset"].map(source_rank).fillna(99)
    df["_lvl_rank"] = df["usable_level"].map(level_rank).fillna(99)

    capped = 0
    # 只对 dedup_action=keep 的行应用上限
    keep_df = df[df["dedup_action"] == "keep"]
    for oid, grp in keep_df.groupby("video_origin_id"):
        if len(grp) <= max_per:
            continue
        sorted_idx = grp.sort_values(["_src_rank", "_lvl_rank"]).index
        over_idx = sorted_idx[max_per:]
        df.loc[over_idx, "dedup_action"] = "drop_over_cap"
        capped += len(over_idx)

    df.drop(columns=["_src_rank", "_lvl_rank"], inplace=True)

    logger.info(f"  每 origin 上限: {max_per}, 超限丢弃: {capped} 行")
    return df


# ---------------------------------------------------------------------------
# Train/Val 分割
# ---------------------------------------------------------------------------

def apply_split(df: pd.DataFrame, policy: Dict) -> pd.DataFrame:
    """按 origin_video_id 分割 train/val, 保证同源不泄漏。"""
    logger.info("=== Train/Val 分割 ===")

    split_cfg = policy.get("split", DEFAULT_POLICY["split"])
    val_ratio = split_cfg.get("val_ratio", 0.05)
    seed = split_cfg.get("random_seed", 42)

    # 只对 keep 的行分割
    keep_mask = df["dedup_action"] == "keep"
    keep_origins = df[keep_mask]["video_origin_id"].unique()

    rng = pd.np.random.default_rng(seed) if hasattr(pd.np, "random") else __import__("numpy").random.default_rng(seed)
    shuffled = rng.permutation(keep_origins)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_origins = set(shuffled[:n_val])

    df["split"] = "drop"
    df.loc[keep_mask & df["video_origin_id"].isin(val_origins), "split"] = "val"
    df.loc[keep_mask & ~df["video_origin_id"].isin(val_origins), "split"] = "train"

    n_train = (df["split"] == "train").sum()
    n_val_actual = (df["split"] == "val").sum()
    n_drop = (df["split"] == "drop").sum()
    logger.info(f"  train: {n_train}, val: {n_val_actual}, drop: {n_drop}")

    return df


# ---------------------------------------------------------------------------
# 输出报告
# ---------------------------------------------------------------------------

def generate_policy_report(df: pd.DataFrame, policy: Dict, output_dir: Path) -> None:
    """输出策略执行报告。"""
    report = {
        "total_rows": len(df),
        "policy_applied": True,
        "usable_level_distribution": df["usable_level"].value_counts().to_dict(),
        "dedup_action_distribution": df["dedup_action"].value_counts().to_dict(),
        "split_distribution": df["split"].value_counts().to_dict(),
        "usability_tags": {
            "usable_for_protocol_sft": int(df["usable_for_protocol_sft"].sum()),
            "usable_for_recall": int(df["usable_for_recall"].sum()),
            "usable_for_timing": int(df["usable_for_timing"].sum()),
            "usable_for_rl_verifiable": int(df["usable_for_rl_verifiable"].sum()),
            "usable_for_multiturn": int(df["usable_for_multiturn"].sum()),
            "only_as_weak_anchor": int(df["only_as_weak_anchor"].sum()),
        },
        "final_keep_count": int((df["dedup_action"] == "keep").sum()),
        "final_train_count": int((df["split"] == "train").sum()),
        "final_val_count": int((df["split"] == "val").sum()),
    }

    report_path = output_dir / "policy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"策略执行报告: {report_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: 读取标记结果 + 策略配置, 执行可用性分级/去重决策/采样上限/分割"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Phase 1 输出的 marked parquet")
    parser.add_argument("--policy", type=str, default=None,
                        help="策略配置文件 (YAML/JSON)。不指定则用内置默认策略。")
    parser.add_argument("--output", type=str,
                        default="data/audit/video_asset_registry_final.parquet")
    args = parser.parse_args()

    # 加载策略
    if args.policy:
        logger.info(f"加载策略配置: {args.policy}")
        policy = load_policy(args.policy)
    else:
        logger.info("使用内置默认策略")
        policy = DEFAULT_POLICY

    # 读取标记结果
    logger.info(f"读取标记后资产表: {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"总行数: {len(df)}")

    # Phase 2: 决策
    df = apply_usability(df, policy)
    df = apply_dedup_decisions(df, policy)
    df = apply_sampling_cap(df, policy)
    df = apply_split(df, policy)

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"最终资产表: {out_path}")

    # 报告
    generate_policy_report(df, policy, out_path.parent)

    # 导出 train 子集方便后续使用
    train_df = df[df["split"] == "train"]
    train_path = out_path.parent / "train_registry.parquet"
    train_df.to_parquet(train_path, index=False)
    logger.info(f"Train 子集: {train_path} ({len(train_df)} 行)")


if __name__ == "__main__":
    main()
