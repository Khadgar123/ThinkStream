"""
标注数据验证与清洗脚本

功能：
1. 从 thinkstream/data/__init__.py 中读取所有已注册数据集的标注路径
2. 修复 LLaVA-Video 数据集中多余的 /data/ 中间路径
3. 基于 --data-root 检查视频文件是否存在（video_path 相对于此目录解析）
4. 对仍找不到的 LLaVA-Video 路径，在对应第一级子目录下递归搜索文件名并修正路径
5. 不存在的视频对应的样本会被删除，并输出报告；缺失路径写入日志文件
6. 处理前自动备份原始文件

Usage:
    # 仅检查，不修改（推荐先运行）
    python scripts/validate_and_clean_data.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data \
        --dry-run

    # 执行清洗
    python scripts/validate_and_clean_data.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data

    # 只处理指定数据集
    python scripts/validate_and_clean_data.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data \
        --dataset stream_rlvr \
        --dry-run
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# 将项目根目录加入 sys.path，以便直接导入 thinkstream 包
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from thinkstream.data import data_dict  # noqa: E402

LLAVA_VIDEO_PREFIX = "LLaVA-Video-178K"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# 模糊搜索索引缓存：subdir_abs_path → {filename: relative_path_from_data_root}
_llava_index_cache: Dict[Path, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 阶段 1：修复 LLaVA-Video /data/ 多余层
# ---------------------------------------------------------------------------

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
    fixed_parts: List[str] = []
    i = 0
    while i < len(parts):
        fixed_parts.append(parts[i])
        # 在 LLaVA-Video-178K 后的第一个子目录之后，跳过紧随的 "data"
        if parts[i].startswith(LLAVA_VIDEO_PREFIX) and i + 2 < len(parts):
            fixed_parts.append(parts[i + 1])  # 第一级子目录（保留）
            if parts[i + 2] == "data":
                i += 3  # 跳过 "data"
                continue
            else:
                i += 1
                continue
        i += 1
    return "/".join(fixed_parts)


# ---------------------------------------------------------------------------
# 阶段 2.5：模糊路径解析（在 LLaVA 第一级子目录下递归搜索文件名）
# ---------------------------------------------------------------------------

def _get_llava_subdir_and_filename(
    video_path: str,
) -> Optional[Tuple[str, str]]:
    """
    从 LLaVA-Video 路径中提取 (第一级子目录名, 文件名)。

    例：
      ./datasets/LLaVA-Video-178K/0_30_s_academic_v0_1/garbage/foo/bar.mp4
      → ("0_30_s_academic_v0_1", "bar.mp4")
    """
    parts = video_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.startswith(LLAVA_VIDEO_PREFIX) and i + 1 < len(parts):
            subdir = parts[i + 1]
            filename = parts[-1]
            return subdir, filename
    return None


def _build_subdir_index(subdir_abs: Path, data_root: Path) -> Dict[str, str]:
    """
    递归遍历 subdir_abs，构建 {filename: 相对于 data_root 的路径} 索引。
    遇到重名文件时保留第一个（按 rglob 遍历顺序）。
    """
    index: Dict[str, str] = {}
    print(f"    [索引构建] 正在扫描 {subdir_abs} ...")
    for p in subdir_abs.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            if p.name not in index:
                rel = "./" + p.relative_to(data_root).as_posix()
                index[p.name] = rel
    print(f"    [索引构建] 完成，共索引 {len(index)} 个视频文件")
    return index


def fuzzy_resolve_llava_path(
    video_path: str,
    data_root: Path,
) -> Optional[str]:
    """
    对找不到的 LLaVA-Video 路径，在其第一级子目录下递归搜索文件名。
    找到则返回修正后的相对路径，否则返回 None。
    """
    result = _get_llava_subdir_and_filename(video_path)
    if result is None:
        return None
    subdir_name, filename = result

    llava_base = data_root / "datasets" / LLAVA_VIDEO_PREFIX
    subdir_abs = llava_base / subdir_name

    if not subdir_abs.is_dir():
        return None

    # 使用缓存避免重复遍历同一目录
    if subdir_abs not in _llava_index_cache:
        _llava_index_cache[subdir_abs] = _build_subdir_index(subdir_abs, data_root)

    index = _llava_index_cache[subdir_abs]
    return index.get(filename)


# ---------------------------------------------------------------------------
# 主清洗流程
# ---------------------------------------------------------------------------

def validate_and_clean(
    annotation_path: Path,
    data_root: Path,
    dry_run: bool = False,
    missing_log: Optional[Path] = None,
) -> Tuple[int, int, int, int]:
    """
    验证并清洗单个标注文件。

    返回 (原始样本数, 阶段1修复数, 模糊解析修复数, 删除样本数)。
    """
    print(f"\n{'='*60}")
    print(f"  处理文件: {annotation_path.name}")
    print(f"{'='*60}")

    samples = read_jsonl(annotation_path)
    original_count = len(samples)
    print(f"  原始样本数: {original_count}")

    # --- 阶段 1：修复 /data/ 多余层 ---
    fix1_count = 0
    for sample in samples:
        vp = sample.get("video_path", "")
        fixed = fix_llava_video_path(vp)
        if fixed != vp:
            fix1_count += 1
            sample["video_path"] = fixed

    print(f"  [阶段1] LLaVA-Video /data/ 路径修复: {fix1_count} 条")

    # --- 阶段 2：首次存在性检查 ---
    first_pass_missing: List[int] = []  # 找不到的样本在 samples 中的索引

    for idx, sample in enumerate(samples):
        vp = sample.get("video_path", "")
        if not vp:
            continue
        if not (data_root / vp).exists():
            first_pass_missing.append(idx)

    print(f"  [阶段2] 首次检查缺失: {len(first_pass_missing)} 条")

    # --- 阶段 2.5：对 LLaVA-Video 缺失路径做模糊递归搜索 ---
    fix2_count = 0
    still_missing_idx: List[int] = []

    if first_pass_missing:
        llava_missing = [
            idx for idx in first_pass_missing
            if LLAVA_VIDEO_PREFIX in samples[idx].get("video_path", "")
        ]
        non_llava_missing = [
            idx for idx in first_pass_missing
            if LLAVA_VIDEO_PREFIX not in samples[idx].get("video_path", "")
        ]

        if llava_missing:
            print(f"  [阶段2.5] 对 {len(llava_missing)} 条 LLaVA-Video 缺失路径进行模糊搜索...")
            for idx in llava_missing:
                vp = samples[idx]["video_path"]
                resolved = fuzzy_resolve_llava_path(vp, data_root)
                if resolved is not None:
                    samples[idx]["video_path"] = resolved
                    fix2_count += 1
                else:
                    still_missing_idx.append(idx)
            print(f"  [阶段2.5] 模糊搜索修复: {fix2_count} 条 / 仍缺失: {len(still_missing_idx)} 条")

        still_missing_idx.extend(non_llava_missing)

    # --- 阶段 3：最终过滤 ---
    still_missing_set = set(still_missing_idx)
    valid_samples = [s for i, s in enumerate(samples) if i not in still_missing_set]
    missing_videos = [samples[i]["video_path"] for i in still_missing_idx]
    removed_count = len(still_missing_idx)

    if missing_videos:
        unique_missing = sorted(set(missing_videos))
        print(f"\n  [警告] 最终缺失视频 {removed_count} 条样本 "
              f"(涉及 {len(unique_missing)} 个唯一路径)")
        print(f"  缺失视频路径示例 (最多显示 20 个):")
        for vp in unique_missing[:20]:
            print(f"    ✗ {vp}")
        if len(unique_missing) > 20:
            print(f"    ... 及另外 {len(unique_missing) - 20} 个路径")
        if missing_log is not None:
            with open(missing_log, "a", encoding="utf-8") as f:
                f.write(
                    f"# [{annotation_path.name}]  最终缺失样本 {removed_count} 条 / "
                    f"唯一路径 {len(unique_missing)} 个\n"
                )
                for vp in unique_missing:
                    f.write(vp + "\n")
                f.write("\n")
            print(f"  缺失路径已追加写入: {missing_log}")
    else:
        print(f"  所有视频文件均存在（或已通过模糊搜索修复）")

    total_fix = fix1_count + fix2_count
    print(f"\n  清洗结果: 保留 {len(valid_samples)}/{original_count} 条样本, "
          f"删除 {removed_count} 条  (路径修复合计 {total_fix} 条：阶段1={fix1_count}, 模糊={fix2_count})")

    # --- 阶段 4：备份 & 写入 ---
    if not dry_run and (total_fix > 0 or removed_count > 0):
        backup_path = backup_file(annotation_path)
        print(f"  备份已创建: {backup_path}")
        write_jsonl(annotation_path, valid_samples)
        print(f"  已写入清洗后的文件: {annotation_path}")
    elif dry_run:
        print(f"  [dry-run] 未实际修改文件")
    else:
        print(f"  文件无需修改，跳过写入")

    return original_count, fix1_count, fix2_count, removed_count


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

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
        help=(
            "视频数据集根目录，video_path 中的相对路径（./datasets/...）将基于此目录解析。"
            "例：/home/tione/notebook/gaozhenkun/hzh/data"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "只处理指定数据集，逗号分隔。"
            f"可选值: {', '.join(data_dict.keys())}。"
            "默认处理所有已注册数据集。"
        ),
    )
    parser.add_argument(
        "--missing-log",
        type=str,
        default="missing_videos.txt",
        help="将最终缺失视频路径写入的文件路径 (默认: missing_videos.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅检查，不实际修改文件",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    missing_log = Path(args.missing_log)
    if not data_root.is_dir():
        print(f"[错误] 数据根目录不存在: {data_root}")
        sys.exit(1)

    # 每次运行清空日志文件，避免追加到上次结果
    missing_log.write_text("", encoding="utf-8")

    # 从 thinkstream/data/__init__.py 获取所有已注册数据集的标注路径
    if args.dataset:
        selected_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
        unknown = [n for n in selected_names if n not in data_dict]
        if unknown:
            print(f"[错误] 未知数据集: {unknown}")
            print(f"可用数据集: {list(data_dict.keys())}")
            sys.exit(1)
        configs = {n: data_dict[n] for n in selected_names}
    else:
        configs = data_dict

    print(f"数据根目录 (video_path 相对于此解析): {data_root}")
    print(f"待处理数据集: {list(configs.keys())}")
    print(f"缺失路径日志: {missing_log.resolve()}")
    if args.dry_run:
        print("[模式] dry-run — 仅检查，不修改文件")

    total_orig = total_fix1 = total_fix2 = total_rm = 0
    for name, cfg in configs.items():
        ann_path = Path(cfg["annotation_path"])
        print(f"\n[数据集] {name}  →  {ann_path}")
        if not ann_path.exists():
            print(f"  [跳过] 标注文件不存在: {ann_path}")
            continue
        orig, fix1, fix2, rm = validate_and_clean(
            ann_path, data_root, dry_run=args.dry_run, missing_log=missing_log
        )
        total_orig += orig
        total_fix1 += fix1
        total_fix2 += fix2
        total_rm += rm

    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"  总样本数:          {total_orig}")
    print(f"  阶段1 路径修复:    {total_fix1}  (/data/ 层删除)")
    print(f"  阶段2.5 模糊修复:  {total_fix2}  (递归搜索文件名)")
    print(f"  删除样本:          {total_rm}")
    print(f"  最终保留:          {total_orig - total_rm}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
