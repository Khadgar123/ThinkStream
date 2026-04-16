"""
数据审计脚本 1: 构建统一视频资产表 (video_asset_registry)

【Phase 1 — 纯标记】
只做字段提取、origin_id 解析、原始特征标记。
不做任何可用性判断或去重决策。

将 Streamo-Instruct-465K 和 ThinkStream 两个数据集统一映射到一张资产表，
为后续去重 (resolve_dedup.py) 和策略处理 (apply_audit_policy.py) 提供基础。

Usage:
    python scripts/audit/build_video_asset_registry.py \
        --streamo-path data/streamo_raw \
        --thinkstream-path data/thinkstream_raw \
        --output data/audit/video_asset_registry.parquet

输入支持的格式:
    - .parquet (HuggingFace 下载格式)
    - .jsonl
    - .json
    - HuggingFace 目录 (含 *.parquet 文件)
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Origin ID 解析 (纯提取, 无决策)
# ---------------------------------------------------------------------------

_CLIP_RE = re.compile(
    r"([A-Za-z0-9_-]{11})_(\d{5,6})_(\d{5,6})(?:\.mp4)?$"
)
_YTB_RE = re.compile(r"ytb_([A-Za-z0-9_-]+)\.mp4$")
_ANET_RE = re.compile(r"(v_[A-Za-z0-9_-]+)\.mp4$")


def _infer_underlying_source(video_path: str) -> str:
    """从路径关键词推断底层视频来源。"""
    p = video_path.lower()
    if "kinetics" in p:
        return "kinetics700"
    if "charades" in p:
        return "charades"
    if "activitynet" in p or "activity_net" in p:
        return "activitynet"
    if "llava_video" in p or "llava-video" in p:
        return "llava_video"
    if "sharegpt4video" in p or "sharegpt" in p:
        return "sharegpt4video"
    if "tarsier" in p:
        return "tarsier2"
    return "unknown"


def parse_origin_id(video_path: str) -> Dict[str, Any]:
    """从视频路径解析同源信息。纯提取, 不做任何判断。"""
    if not video_path:
        return {
            "origin_id": None,
            "clip_start_sec": None,
            "clip_end_sec": None,
            "underlying_source": "unknown",
        }

    underlying = _infer_underlying_source(video_path)
    filename = Path(video_path).stem

    # 模式 1: {yt_id}_{start}_{end}
    m = _CLIP_RE.search(filename)
    if m:
        return {
            "origin_id": m.group(1),
            "clip_start_sec": float(m.group(2)),
            "clip_end_sec": float(m.group(3)),
            "underlying_source": underlying if underlying != "unknown" else "kinetics700",
        }

    # 模式 2: ytb_{id}
    m = _YTB_RE.search(Path(video_path).name)
    if m:
        return {
            "origin_id": m.group(1),
            "clip_start_sec": None,
            "clip_end_sec": None,
            "underlying_source": underlying if underlying != "unknown" else "llava_video",
        }

    # 模式 3: v_{id}
    m = _ANET_RE.search(Path(video_path).name)
    if m:
        return {
            "origin_id": m.group(1),
            "clip_start_sec": None,
            "clip_end_sec": None,
            "underlying_source": underlying if underlying != "unknown" else "activitynet",
        }

    # 兜底
    return {
        "origin_id": filename,
        "clip_start_sec": None,
        "clip_end_sec": None,
        "underlying_source": underlying,
    }


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def _read_data(path: str) -> pd.DataFrame:
    """读取 parquet / jsonl / json 文件或 HuggingFace 目录。"""
    p = Path(path)

    if p.is_dir():
        parquets = sorted(p.rglob("*.parquet"))
        if parquets:
            logger.info(f"  找到 {len(parquets)} 个 parquet 文件")
            dfs = [pd.read_parquet(f) for f in parquets]
            return pd.concat(dfs, ignore_index=True)
        jsonls = sorted(p.rglob("*.jsonl"))
        if jsonls:
            dfs = [pd.read_json(f, lines=True) for f in jsonls]
            return pd.concat(dfs, ignore_index=True)
        jsons = sorted(p.rglob("*.json"))
        if jsons:
            dfs = []
            for f in jsons:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    dfs.append(pd.DataFrame(data))
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        raise FileNotFoundError(f"在 {path} 中未找到数据文件")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.json_normalize(data)

    raise ValueError(f"不支持的文件格式: {p.suffix}")


# ---------------------------------------------------------------------------
# Streamo 字段提取 (纯标记)
# ---------------------------------------------------------------------------

def _extract_streamo_row(row: pd.Series, idx: int) -> Dict[str, Any]:
    """从 Streamo 一行提取原始特征标记。不做可用性判断。"""
    video_path = row.get("video_path", "") or ""
    origin = parse_origin_id(video_path)

    questions = row.get("question", []) or []
    responses = row.get("response", []) or []

    # --- 原始时间信息标记 ---
    has_user_ts = False
    has_assistant_ts = False
    has_support_span = False
    ask_time_sec = None
    response_st_time = None
    response_end_time = None
    response_instant_time = None

    if isinstance(questions, list):
        for q in questions:
            if isinstance(q, dict) and q.get("time"):
                has_user_ts = True
                try:
                    ask_time_sec = float(q["time"])
                except (ValueError, TypeError):
                    pass

    if isinstance(responses, list):
        for r in responses:
            if isinstance(r, dict):
                if r.get("st_time") is not None and r.get("end_time") is not None:
                    has_assistant_ts = True
                    has_support_span = True
                    try:
                        response_st_time = float(r["st_time"])
                        response_end_time = float(r["end_time"])
                    except (ValueError, TypeError):
                        pass
                if r.get("time"):
                    has_assistant_ts = True
                    try:
                        response_instant_time = float(r["time"])
                    except (ValueError, TypeError):
                        pass

    # --- 原始标签 ---
    task_type = row.get("task_type", "") or ""
    source_tag = row.get("source", "") or ""

    # --- 文本内容特征 ---
    response_text = ""
    if isinstance(responses, list) and responses:
        r0 = responses[0] if isinstance(responses[0], dict) else {}
        response_text = r0.get("content", "") or ""

    question_text = ""
    if isinstance(questions, list) and questions:
        q0 = questions[0] if isinstance(questions[0], dict) else {}
        question_text = q0.get("content", "") or ""

    # --- 数值特征 (原始值, 不做判断) ---
    estimated_video_duration = None
    if response_end_time is not None:
        estimated_video_duration = response_end_time
    elif response_instant_time is not None:
        estimated_video_duration = response_instant_time + 5  # 粗估

    num_questions = len(questions) if isinstance(questions, list) else 0
    num_responses = len(responses) if isinstance(responses, list) else 0

    return {
        # 身份
        "asset_id": f"streamo_{idx:07d}",
        "source_dataset": "streamo",
        "source_row_id": idx,
        # 视频定位
        "video_locator_raw": video_path,
        "video_name": row.get("video_name", "") or "",
        "video_origin_id": origin["origin_id"],
        "underlying_source": origin["underlying_source"],
        "clip_start_sec": origin["clip_start_sec"],
        "clip_end_sec": origin["clip_end_sec"],
        # 原始标签
        "task_type_raw": task_type,
        "source_tag": source_tag,
        "temporal_scope_raw": "",  # Streamo 无此字段
        "content_dimension_raw": "",  # Streamo 无此字段
        "response_format_raw": "",  # Streamo 无此字段
        "interaction_mode_raw": "",  # Streamo 无此字段
        # 时间特征 (原始值)
        "has_user_timestamp": has_user_ts,
        "has_assistant_timestamp": has_assistant_ts,
        "has_think_timestamp": False,
        "has_support_span": has_support_span,
        "ask_time_sec": ask_time_sec,
        "response_st_time_sec": response_st_time,
        "response_end_time_sec": response_end_time,
        "response_instant_time_sec": response_instant_time,
        "estimated_video_duration_sec": estimated_video_duration,
        # 文本特征
        "question_text_preview": question_text[:200],
        "response_text_preview": response_text[:200],
        "response_text_length": len(response_text),
        # 数量特征
        "num_questions": num_questions,
        "num_responses": num_responses,
        "num_conversations": num_questions + num_responses,
        "num_thoughts": 0,
    }


# ---------------------------------------------------------------------------
# ThinkStream 字段提取 (纯标记)
# ---------------------------------------------------------------------------

def _extract_thinkstream_row(row: pd.Series, idx: int) -> Dict[str, Any]:
    """从 ThinkStream 一行提取原始特征标记。不做可用性判断。"""
    video_path = row.get("video_path", "") or ""
    origin = parse_origin_id(video_path)

    conversations = row.get("conversations", []) or []
    thoughts = row.get("thoughts", []) or []

    # --- 时间信息 ---
    has_user_ts = False
    has_assistant_ts = False
    has_think_ts = False
    ask_time_sec = None
    response_time_sec = None
    max_timestamp = 0.0

    user_count = 0
    assistant_count = 0

    if isinstance(conversations, list):
        for c in conversations:
            if not isinstance(c, dict):
                continue
            ts = c.get("timestamp")
            role = c.get("role", "")
            if ts is not None:
                try:
                    ts_f = float(ts)
                    max_timestamp = max(max_timestamp, ts_f)
                except (ValueError, TypeError):
                    ts_f = None

                if role == "user":
                    has_user_ts = True
                    user_count += 1
                    if ask_time_sec is None and ts_f is not None:
                        ask_time_sec = ts_f
                elif role == "assistant":
                    has_assistant_ts = True
                    assistant_count += 1
                    if response_time_sec is None and ts_f is not None:
                        response_time_sec = ts_f
            else:
                if role == "user":
                    user_count += 1
                elif role == "assistant":
                    assistant_count += 1

    think_count = 0
    total_think_chars = 0
    if isinstance(thoughts, list):
        for t in thoughts:
            if not isinstance(t, dict):
                continue
            think_count += 1
            if t.get("timestamp") is not None:
                has_think_ts = True
            think_text = t.get("think", "") or ""
            total_think_chars += len(think_text)

    # --- 原始标签 ---
    interaction_mode = row.get("interaction_mode", "") or ""
    temporal_scope = row.get("temporal_scope", "") or ""
    content_dimension = row.get("content_dimension", "") or ""
    response_format = row.get("response_format", "") or ""

    # --- 时长估算 ---
    estimated_duration = None
    if origin["clip_start_sec"] is not None and origin["clip_end_sec"] is not None:
        estimated_duration = origin["clip_end_sec"] - origin["clip_start_sec"]
    elif max_timestamp > 0:
        estimated_duration = max_timestamp + 2

    # --- 文本预览 ---
    question_text = ""
    response_text = ""
    if isinstance(conversations, list):
        for c in conversations:
            if isinstance(c, dict):
                if c.get("role") == "user" and not question_text:
                    question_text = c.get("content", "") or ""
                elif c.get("role") == "assistant" and not response_text:
                    response_text = c.get("content", "") or ""

    # --- 选项信息 ---
    has_options = False
    if isinstance(conversations, list):
        for c in conversations:
            if isinstance(c, dict) and c.get("options"):
                has_options = True
                break

    return {
        # 身份
        "asset_id": f"thinkstream_{idx:07d}",
        "source_dataset": "thinkstream",
        "source_row_id": idx,
        # 视频定位
        "video_locator_raw": video_path,
        "video_name": Path(video_path).name if video_path else "",
        "video_origin_id": origin["origin_id"],
        "underlying_source": origin["underlying_source"],
        "clip_start_sec": origin["clip_start_sec"],
        "clip_end_sec": origin["clip_end_sec"],
        # 原始标签
        "task_type_raw": interaction_mode,  # ThinkStream 用 interaction_mode 对应
        "source_tag": "",
        "temporal_scope_raw": temporal_scope,
        "content_dimension_raw": content_dimension,
        "response_format_raw": response_format,
        "interaction_mode_raw": interaction_mode,
        # 时间特征 (原始值)
        "has_user_timestamp": has_user_ts,
        "has_assistant_timestamp": has_assistant_ts,
        "has_think_timestamp": has_think_ts,
        "has_support_span": temporal_scope.lower().strip() == "past" and has_assistant_ts,
        "ask_time_sec": ask_time_sec,
        "response_st_time_sec": None,  # ThinkStream 无此字段
        "response_end_time_sec": None,
        "response_instant_time_sec": response_time_sec,
        "estimated_video_duration_sec": estimated_duration,
        # 文本特征
        "question_text_preview": question_text[:200],
        "response_text_preview": response_text[:200],
        "response_text_length": len(response_text),
        # 数量特征
        "num_questions": user_count,
        "num_responses": assistant_count,
        "num_conversations": len(conversations),
        "num_thoughts": think_count,
        # ThinkStream 特有
        "total_think_chars": total_think_chars,
        "has_options": has_options,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_registry(
    streamo_path: Optional[str],
    thinkstream_path: Optional[str],
) -> pd.DataFrame:
    """构建统一资产表 (纯标记, 不做任何决策)。"""
    rows: List[Dict[str, Any]] = []

    if streamo_path:
        logger.info(f"读取 Streamo 数据: {streamo_path}")
        df_s = _read_data(streamo_path)
        logger.info(f"  Streamo 行数: {len(df_s)}")
        for idx, row in df_s.iterrows():
            rows.append(_extract_streamo_row(row, idx))
        logger.info(f"  Streamo 提取完成: {len(rows)} 行")

    offset = len(rows)
    if thinkstream_path:
        logger.info(f"读取 ThinkStream 数据: {thinkstream_path}")
        df_t = _read_data(thinkstream_path)
        logger.info(f"  ThinkStream 行数: {len(df_t)}")
        for idx, row in df_t.iterrows():
            rows.append(_extract_thinkstream_row(row, idx))
        logger.info(f"  ThinkStream 提取完成: {len(rows) - offset} 行")

    if not rows:
        raise ValueError("未读取到任何数据。请检查输入路径。")

    registry = pd.DataFrame(rows)

    # 补齐 ThinkStream 特有列 (Streamo 行填 defaults)
    for col in ["total_think_chars", "has_options"]:
        if col in registry.columns:
            registry[col] = registry[col].fillna(0 if "chars" in col else False)

    logger.info(f"统一资产表总行数: {len(registry)}")
    for ds in registry["source_dataset"].unique():
        sub = registry[registry["source_dataset"] == ds]
        n_origin = sub["video_origin_id"].nunique()
        logger.info(
            f"  {ds}: {len(sub)} 行, "
            f"{n_origin} 个唯一 origin_video_id, "
            f"平均 {len(sub) / max(n_origin, 1):.1f} 样本/origin"
        )

    return registry


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: 构建统一视频资产表 (纯标记, 不做决策)"
    )
    parser.add_argument("--streamo-path", type=str, default=None)
    parser.add_argument("--thinkstream-path", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="data/audit/video_asset_registry.parquet",
    )
    args = parser.parse_args()

    if not args.streamo_path and not args.thinkstream_path:
        parser.error("至少需要指定 --streamo-path 或 --thinkstream-path 之一")

    registry = build_registry(args.streamo_path, args.thinkstream_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(out_path, index=False)
    logger.info(f"资产表已保存: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    csv_path = out_path.with_suffix(".csv")
    registry.head(100).to_csv(csv_path, index=False)
    logger.info(f"前100行预览: {csv_path}")


if __name__ == "__main__":
    main()
