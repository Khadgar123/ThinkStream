"""
标注数据验证与清洗脚本

功能：
1. 读取指定的 JSONL 标注文件
2. 修复 LLaVA-Video 数据集中多余的 /data/ 中间路径
3. 基于数据集根目录检查视频文件是否存在
4. 不存在的视频对应的样本会被删除，并输出报告
5. 处理前自动备份原始文件

Usage:
    python scripts/validate_and_clean_data.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data/ThinkStream

    python scripts/validate_and_clean_data.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data/ThinkStream \
        --annotation-dir /home/tione/notebook/gaozhenkun/hzh/data/ThinkStream \
        --dry-run
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple


ANNOTATION_FILES = [
    "streaming_rlvr_processed.jsonl",
    "streaming_cot_cold_processed_5_20.jsonl",
]

LLAVA_VIDEO_PREFIX = "LLaVA-Video-178K"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 .jsonl 文件，返回样本列表。"""
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    """将样本列表写入 .jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def backup_file(path: Path) -> Path:
    """备份文件，返回备份文件路径。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(f".bak_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def fix_llava_video_path(video_path: str) -> str:
    """
    修复 LLaVA-Video 路径中多余的 /data/ 中间目录。

    例：
      ./datasets/LLaVA-Video-178K/0_30_s_academic_v0_1/data/academic_source/...
    → ./datasets/LLaVA-Video-178K/0_30_s_academic_v0_1/academic_source/...
    """
    if LLAVA_VIDEO_PREFIX not in video_path:
        return video_path

    parts = video_path.split("/")
    fixed_parts = []
    i = 0
    while i < len(parts):
        fixed_parts.append(parts[i])
        # 找到 LLaVA-Video-178K 后面的子目录（如 0_30_s_academic_v0_1），
        # 再跳过紧随其后的 "data"
        if parts[i].startswith(LLAVA_VIDEO_PREFIX) and i + 2 < len(parts):
            fixed_parts.append(parts[i + 1])  # 子目录
            if parts[i + 2] == "data":
                i += 3  # 跳过 "data"
                continue
            else:
                i += 1
                continue
        i += 1
    return "/".join(fixed_parts)


def validate_and_clean(
    annotation_path: Path,
    data_root: Path,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    验证并清洗单个标注文件。

    返回 (原始样本数, 路径修复数, 删除样本数)。
    """
    print(f"\n{'='*60}")
    print(f"  处理文件: {annotation_path.name}")
    print(f"{'='*60}")

    samples = read_jsonl(annotation_path)
    original_count = len(samples)
    print(f"  原始样本数: {original_count}")

    # --- 阶段 1：修复 LLaVA-Video 路径 ---
    fix_count = 0
    for sample in samples:
        vp = sample.get("video_path", "")
        fixed = fix_llava_video_path(vp)
        if fixed != vp:
            fix_count += 1
            sample["video_path"] = fixed

    print(f"  LLaVA-Video 路径修复数: {fix_count}")

    # --- 阶段 2：检查视频文件是否存在 ---
    valid_samples: List[Dict[str, Any]] = []
    missing_videos: List[str] = []

    for sample in samples:
        vp = sample.get("video_path", "")
        if not vp:
            valid_samples.append(sample)
            continue

        full_path = data_root / vp
        if full_path.exists():
            valid_samples.append(sample)
        else:
            missing_videos.append(vp)

    removed_count = original_count - len(valid_samples)

    if missing_videos:
        unique_missing = sorted(set(missing_videos))
        print(f"\n  [警告] 缺失视频 {len(missing_videos)} 条样本 "
              f"(涉及 {len(unique_missing)} 个唯一视频路径)")
        print(f"  缺失视频路径示例 (最多显示 20 个):")
        for vp in unique_missing[:20]:
            print(f"    ✗ {vp}")
        if len(unique_missing) > 20:
            print(f"    ... 及另外 {len(unique_missing) - 20} 个路径")
    else:
        print(f"  所有视频文件均存在")

    print(f"\n  清洗结果: 保留 {len(valid_samples)}/{original_count} 条样本, "
          f"删除 {removed_count} 条")

    # --- 阶段 3：备份 & 写入 ---
    if not dry_run and (fix_count > 0 or removed_count > 0):
        backup_path = backup_file(annotation_path)
        print(f"  备份已创建: {backup_path}")
        write_jsonl(annotation_path, valid_samples)
        print(f"  已写入清洗后的文件: {annotation_path}")
    elif dry_run:
        print(f"  [dry-run] 未实际修改文件")
    else:
        print(f"  文件无需修改，跳过写入")

    return original_count, fix_count, removed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证标注文件中的视频路径并清洗无效样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="数据集根目录，video_path 中的相对路径将基于此目录解析",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        help="标注文件所在目录 (默认与 --data-root 相同)",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="要处理的标注文件名列表 (默认处理预设的两个文件)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅检查，不实际修改文件",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else data_root
    filenames = args.files if args.files else ANNOTATION_FILES

    if not data_root.is_dir():
        print(f"[错误] 数据根目录不存在: {data_root}")
        sys.exit(1)
    if not annotation_dir.is_dir():
        print(f"[错误] 标注文件目录不存在: {annotation_dir}")
        sys.exit(1)

    print(f"数据根目录: {data_root}")
    print(f"标注文件目录: {annotation_dir}")
    if args.dry_run:
        print("[模式] dry-run — 仅检查，不修改文件")

    total_orig, total_fix, total_rm = 0, 0, 0
    for fname in filenames:
        ann_path = annotation_dir / fname
        if not ann_path.exists():
            print(f"\n[跳过] 文件不存在: {ann_path}")
            continue
        orig, fix, rm = validate_and_clean(ann_path, data_root, dry_run=args.dry_run)
        total_orig += orig
        total_fix += fix
        total_rm += rm

    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"  总样本数: {total_orig}")
    print(f"  路径修复: {total_fix}")
    print(f"  删除样本: {total_rm}")
    print(f"  最终保留: {total_orig - total_rm}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
