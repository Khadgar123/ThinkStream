"""
首批视频选取脚本

从审计后的数据中选取首批 50 个视频用于 Pipeline 端到端验证。
选取策略:
  - Streamo 30 个: duration > 60s, recallability 高, 视频类型多样
  - ThinkStream 20 个: 有 think 时间戳, 有 conversations

依赖: 审计产出的 video_asset_registry (parquet)
如果没有 registry, 则从原始数据集的标注文件中直接选取。

Usage:
    python -m scripts.agent_data_pipeline.select_first_batch \
        --streamo_annotation /path/to/streamo/raw_data.json \
        --thinkstream_annotation /path/to/thinkstream/data.jsonl \
        --video_root /path/to/videos \
        --output data/agent/video_list.json \
        [--num_streamo 30] [--num_thinkstream 20]
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_streamo_candidates(annotation_path: str) -> List[Dict]:
    """从 Streamo 标注文件中提取视频候选列表。"""
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f) if annotation_path.endswith(".json") else [
            json.loads(l) for l in f if l.strip()
        ]

    # 按视频聚合
    video_map: Dict[str, Dict] = {}
    for row in data:
        vid_name = row.get("video_name", "")
        vid_path = row.get("video_path", "")
        if not vid_name:
            continue

        if vid_name not in video_map:
            video_map[vid_name] = {
                "video_id": f"streamo_{vid_name.replace('.mp4', '')}",
                "video_path": vid_path,
                "video_name": vid_name,
                "task_types": set(),
                "num_questions": 0,
                "has_support_span": False,
                "max_end_time": 0.0,
                "source": row.get("source", "unknown"),
            }

        entry = video_map[vid_name]
        entry["task_types"].add(row.get("task_type", ""))
        entry["num_questions"] += 1

        # 估算视频时长: 从 response 时间推断
        for resp in row.get("response", []):
            end_t = resp.get("end_time")
            st_t = resp.get("st_time")
            if end_t and float(end_t) > entry["max_end_time"]:
                entry["max_end_time"] = float(end_t)
            if st_t and float(st_t) > 0:
                entry["has_support_span"] = True

    # 转成 list 并计算简单 recallability
    candidates = []
    for v in video_map.values():
        v["task_types"] = list(v["task_types"])
        v["estimated_duration"] = v["max_end_time"]

        # 简单 recallability: 时长 + 多任务 + support span
        dur_score = min(1.0, v["estimated_duration"] / 120.0)
        task_score = min(1.0, len(v["task_types"]) / 5.0)
        span_score = 1.0 if v["has_support_span"] else 0.0
        q_score = min(1.0, v["num_questions"] / 10.0)
        v["recallability"] = 0.4 * dur_score + 0.2 * task_score + 0.2 * span_score + 0.2 * q_score

        candidates.append(v)

    return candidates


def load_thinkstream_candidates(annotation_path: str) -> List[Dict]:
    """从 ThinkStream 标注文件中提取视频候选列表。"""
    with open(annotation_path, encoding="utf-8") as f:
        if annotation_path.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(l) for l in f if l.strip()]

    video_map: Dict[str, Dict] = {}
    for row in data:
        vid_path = row.get("video_path", "")
        if not vid_path:
            continue

        vid_name = Path(vid_path).name
        if vid_name not in video_map:
            video_map[vid_name] = {
                "video_id": f"thinkstream_{vid_name.replace('.mp4', '')}",
                "video_path": vid_path,
                "video_name": vid_name,
                "interaction_modes": set(),
                "temporal_scopes": set(),
                "response_formats": set(),
                "num_conversations": 0,
                "num_thoughts": 0,
                "has_think": False,
            }

        entry = video_map[vid_name]
        entry["interaction_modes"].add(row.get("interaction_mode", ""))
        entry["temporal_scopes"].add(row.get("temporal_scope", ""))
        entry["response_formats"].add(row.get("response_format", ""))
        entry["num_conversations"] += len(row.get("conversations", []))
        thoughts = row.get("thoughts", [])
        entry["num_thoughts"] += len(thoughts)
        if thoughts:
            entry["has_think"] = True

    candidates = []
    for v in video_map.values():
        v["interaction_modes"] = list(v["interaction_modes"])
        v["temporal_scopes"] = list(v["temporal_scopes"])
        v["response_formats"] = list(v["response_formats"])
        candidates.append(v)

    return candidates


def select_streamo(
    candidates: List[Dict],
    num: int = 30,
    min_duration: float = 60.0,
) -> List[Dict]:
    """从 Streamo 候选中选取视频。

    策略: duration > 60s, recallability top 30%, 避开 split_* origin
    """
    # 过滤
    filtered = [
        c for c in candidates
        if c["estimated_duration"] > min_duration
        and not c["video_name"].startswith("split_")
        and c["has_support_span"]
    ]

    if not filtered:
        logger.warning("No Streamo videos > %ds with support span, relaxing criteria", min_duration)
        filtered = [c for c in candidates if c["estimated_duration"] > 30]

    # 按 recallability 排序, 取 top 30%
    filtered.sort(key=lambda x: x["recallability"], reverse=True)
    top_pool = filtered[:max(len(filtered) // 3, num * 2)]

    # 从 top pool 中随机选取, 尽量多样化 source
    if len(top_pool) <= num:
        selected = top_pool
    else:
        selected = random.sample(top_pool, num)

    logger.info("Streamo: %d candidates → %d filtered → %d selected",
                len(candidates), len(filtered), len(selected))
    return selected


def select_thinkstream(
    candidates: List[Dict],
    num: int = 20,
) -> List[Dict]:
    """从 ThinkStream 候选中选取视频。

    策略: 有 think 时间戳, 优先多样化 interaction_mode
    """
    # 优先有 think 的
    with_think = [c for c in candidates if c["has_think"]]
    if not with_think:
        with_think = candidates

    if len(with_think) <= num:
        selected = with_think
    else:
        # 优先选 RL 可验证的 (有 MC/Binary/Counting)
        verifiable = [
            c for c in with_think
            if any(f in c["response_formats"] for f in ["Multiple Choice", "Binary", "Counting"])
        ]
        non_verifiable = [c for c in with_think if c not in verifiable]

        # 60% 可验证, 40% 开放式
        n_ver = min(len(verifiable), int(num * 0.6))
        n_open = num - n_ver

        selected = random.sample(verifiable, n_ver) if len(verifiable) > n_ver else verifiable
        if len(non_verifiable) >= n_open:
            selected += random.sample(non_verifiable, n_open)
        else:
            selected += non_verifiable

    logger.info("ThinkStream: %d candidates → %d with_think → %d selected",
                len(candidates), len(with_think), len(selected))
    return selected


def build_video_list(
    selected: List[Dict],
    video_root: str,
) -> List[Dict]:
    """构建 pipeline 输入格式的 video_list."""
    result = []
    for v in selected:
        video_path = v["video_path"]
        # 如果是相对路径, 拼接 video_root
        if not Path(video_path).is_absolute():
            video_path = str(Path(video_root) / video_path)

        result.append({
            "video_id": v["video_id"],
            "video_path": video_path,
            "source_dataset": "streamo" if v["video_id"].startswith("streamo") else "thinkstream",
            "metadata": {
                k: v.get(k)
                for k in ["recallability", "estimated_duration", "num_questions",
                          "has_think", "num_thoughts", "interaction_modes",
                          "task_types", "response_formats"]
                if v.get(k) is not None
            },
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Select first batch of videos")
    parser.add_argument("--streamo_annotation", required=True)
    parser.add_argument("--thinkstream_annotation", required=True)
    parser.add_argument("--video_root", default="./")
    parser.add_argument("--output", default="data/agent/video_list.json")
    parser.add_argument("--num_streamo", type=int, default=30)
    parser.add_argument("--num_thinkstream", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    # Load candidates
    logger.info("Loading Streamo candidates from %s", args.streamo_annotation)
    streamo_candidates = load_streamo_candidates(args.streamo_annotation)
    logger.info("Loading ThinkStream candidates from %s", args.thinkstream_annotation)
    thinkstream_candidates = load_thinkstream_candidates(args.thinkstream_annotation)

    # Select
    streamo_selected = select_streamo(streamo_candidates, args.num_streamo)
    thinkstream_selected = select_thinkstream(thinkstream_candidates, args.num_thinkstream)

    # Build output
    video_list = build_video_list(
        streamo_selected + thinkstream_selected,
        args.video_root,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(video_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Selected {len(video_list)} videos → {output_path}")
    print(f"  Streamo:     {sum(1 for v in video_list if v['source_dataset'] == 'streamo')}")
    print(f"  ThinkStream: {sum(1 for v in video_list if v['source_dataset'] == 'thinkstream')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
