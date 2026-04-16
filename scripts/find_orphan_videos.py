"""
孤立视频文件检测与清理脚本

找出存在于磁盘上、但未被任何标注文件引用的视频文件（"孤立文件"），
先用 --dry-run 查看报告，确认无误后再执行删除以释放空间。

功能：
1. 从 thinkstream/data/__init__.py 读取所有已注册数据集的标注路径
2. 收集所有标注文件中引用的视频路径（归一化为绝对路径）
3. 遍历 --scan-dirs 指定的目录（默认为 data_root/datasets/），枚举所有视频文件
4. 找出磁盘上存在但未被引用的孤立视频
5. --dry-run 模式：仅打印报告（含文件数 & 占用空间），不删除
6. 正式模式：删除孤立文件，并可选择删除由此产生的空目录

Usage:
    # 第一步：仅检查，不删除（推荐先运行）
    python scripts/find_orphan_videos.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data \
        --dry-run

    # 第二步：执行删除
    python scripts/find_orphan_videos.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data

    # 只扫描特定子目录
    python scripts/find_orphan_videos.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data \
        --scan-dirs datasets/LLaVA-Video-178K datasets/tarsier2_unzip \
        --dry-run

    # 删除孤立文件后同时清理空目录
    python scripts/find_orphan_videos.py \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data \
        --remove-empty-dirs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# 将项目根目录加入 sys.path，以便直接导入 thinkstream 包
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from thinkstream.data import data_dict  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def fmt_size(n_bytes: int) -> str:
    """将字节数格式化为人类可读的字符串。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def read_jsonl(path: Path) -> List[Dict]:
    """读取 .jsonl 文件，返回样本列表。"""
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def remove_empty_dirs(root: Path) -> int:
    """
    自底向上删除 root 目录树中的所有空目录（不删除 root 本身）。
    返回删除的目录数。
    """
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        try:
            p.rmdir()  # 只有目录为空时才会成功
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# 收集已引用路径
# ---------------------------------------------------------------------------

def collect_referenced_paths(
    data_root: Path,
    dataset_names: Optional[List[str]] = None,
) -> Set[Path]:
    """
    读取所有（或指定）数据集的标注文件，返回被引用视频的绝对路径集合。
    路径不存在的条目也会加入集合（用于对比，不影响孤立检测逻辑）。
    """
    configs = (
        {n: data_dict[n] for n in dataset_names}
        if dataset_names
        else data_dict
    )

    referenced: Set[Path] = set()
    for name, cfg in configs.items():
        ann_path = Path(cfg["annotation_path"])
        if not ann_path.exists():
            print(f"  [跳过标注] 文件不存在: {ann_path}")
            continue
        samples = read_jsonl(ann_path)
        for s in samples:
            vp = s.get("video_path", "")
            if vp:
                abs_path = (data_root / vp).resolve()
                referenced.add(abs_path)
        print(f"  [{name}] 读取 {len(samples)} 条标注，累计引用路径 {len(referenced)} 个")

    return referenced


# ---------------------------------------------------------------------------
# 扫描磁盘视频文件
# ---------------------------------------------------------------------------

def scan_video_files(scan_dirs: List[Path]) -> List[Path]:
    """
    递归扫描指定目录列表，返回所有视频文件的绝对路径列表。
    """
    found: List[Path] = []
    for d in scan_dirs:
        if not d.is_dir():
            print(f"  [跳过扫描] 目录不存在: {d}")
            continue
        print(f"  扫描目录: {d} ...")
        count_before = len(found)
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                found.append(p.resolve())
        print(f"    找到 {len(found) - count_before} 个视频文件")
    return found


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def find_orphans(
    data_root: Path,
    scan_dirs: List[Path],
    dataset_names: Optional[List[str]],
    report_file: Optional[Path],
    dry_run: bool,
    do_remove_empty_dirs: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"  第一步：收集标注文件中的引用路径")
    print(f"{'='*60}")
    referenced = collect_referenced_paths(data_root, dataset_names)
    print(f"\n  标注引用视频总计: {len(referenced)} 个唯一路径")

    print(f"\n{'='*60}")
    print(f"  第二步：扫描磁盘视频文件")
    print(f"{'='*60}")
    all_videos = scan_video_files(scan_dirs)
    print(f"\n  磁盘视频文件总计: {len(all_videos)} 个")

    print(f"\n{'='*60}")
    print(f"  第三步：比对，找出孤立文件")
    print(f"{'='*60}")
    orphans = [p for p in all_videos if p not in referenced]
    orphan_total_size = sum(p.stat().st_size for p in orphans if p.exists())

    print(f"  孤立文件数:   {len(orphans)}")
    print(f"  占用空间:     {fmt_size(orphan_total_size)}")
    print(f"  引用文件数:   {len(all_videos) - len(orphans)}")

    if not orphans:
        print("\n  未找到孤立视频文件，无需清理。")
        return

    # 按目录统计孤立文件分布
    dir_stats: Dict[Path, Dict] = {}
    for p in orphans:
        parent = p.parent
        if parent not in dir_stats:
            dir_stats[parent] = {"count": 0, "size": 0}
        dir_stats[parent]["count"] += 1
        dir_stats[parent]["size"] += p.stat().st_size if p.exists() else 0

    print(f"\n  孤立文件目录分布 (Top 20 by count):")
    for d, stat in sorted(dir_stats.items(), key=lambda x: -x[1]["count"])[:20]:
        rel = d.relative_to(data_root) if d.is_relative_to(data_root) else d
        print(f"    {rel}  →  {stat['count']} 个文件, {fmt_size(stat['size'])}")

    # 写入报告文件
    if report_file is not None:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# 孤立视频文件报告\n")
            f.write(f"# 孤立文件数: {len(orphans)}, 占用空间: {fmt_size(orphan_total_size)}\n\n")
            for p in sorted(orphans):
                rel = p.relative_to(data_root) if p.is_relative_to(data_root) else p
                f.write(str(rel) + "\n")
        print(f"\n  完整孤立文件列表已写入: {report_file.resolve()}")

    if dry_run:
        print(f"\n  [dry-run] 未删除任何文件。确认无误后去掉 --dry-run 执行删除。")
        return

    # 正式删除
    print(f"\n{'='*60}")
    print(f"  第四步：删除孤立文件")
    print(f"{'='*60}")
    deleted_count = 0
    deleted_size = 0
    failed: List[Path] = []

    for p in orphans:
        try:
            size = p.stat().st_size
            p.unlink()
            deleted_count += 1
            deleted_size += size
            if deleted_count % 500 == 0:
                print(f"    已删除 {deleted_count} / {len(orphans)} 个文件 ...")
        except OSError as e:
            print(f"    [错误] 无法删除 {p}: {e}")
            failed.append(p)

    print(f"\n  删除完成: {deleted_count} 个文件, 释放 {fmt_size(deleted_size)}")
    if failed:
        print(f"  删除失败: {len(failed)} 个文件（见上方错误信息）")

    # 清理空目录
    if do_remove_empty_dirs:
        print(f"\n{'='*60}")
        print(f"  第五步：清理空目录")
        print(f"{'='*60}")
        total_removed_dirs = 0
        for d in scan_dirs:
            if d.is_dir():
                n = remove_empty_dirs(d)
                total_removed_dirs += n
                print(f"  {d}  →  清理空目录 {n} 个")
        print(f"  共清理空目录: {total_removed_dirs} 个")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检测并清理磁盘上未被标注文件引用的孤立视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "数据集根目录，video_path 的相对路径基于此目录解析。"
            "例：/home/tione/notebook/gaozhenkun/hzh/data"
        ),
    )
    parser.add_argument(
        "--scan-dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "要扫描的目录（相对于 --data-root），支持多个。"
            "默认扫描 data_root/datasets/ 整个目录。"
            "例：--scan-dirs datasets/LLaVA-Video-178K datasets/tarsier2_unzip"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "只读取指定数据集的标注，逗号分隔。"
            f"可选值: {', '.join(data_dict.keys())}。"
            "默认读取所有已注册数据集。"
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default="orphan_videos.txt",
        help="将孤立文件列表写入的报告文件路径 (默认: orphan_videos.txt)",
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        default=False,
        help="删除孤立文件后，顺带清理产生的空目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅检查并生成报告，不实际删除任何文件",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"[错误] 数据根目录不存在: {data_root}")
        sys.exit(1)

    # 确定扫描目录
    if args.scan_dirs:
        scan_dirs = [data_root / d for d in args.scan_dirs]
    else:
        default_scan = data_root / "datasets"
        if not default_scan.is_dir():
            print(f"[错误] 默认扫描目录不存在: {default_scan}")
            print("请用 --scan-dirs 手动指定要扫描的目录。")
            sys.exit(1)
        scan_dirs = [default_scan]

    # 确定数据集名称
    dataset_names: Optional[List[str]] = None
    if args.dataset:
        dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
        unknown = [n for n in dataset_names if n not in data_dict]
        if unknown:
            print(f"[错误] 未知数据集: {unknown}")
            print(f"可用数据集: {list(data_dict.keys())}")
            sys.exit(1)

    report_file = Path(args.report)

    print(f"数据根目录:   {data_root}")
    print(f"扫描目录:     {[str(d) for d in scan_dirs]}")
    print(f"读取数据集:   {dataset_names or list(data_dict.keys())}")
    print(f"报告文件:     {report_file.resolve()}")
    if args.dry_run:
        print("[模式] dry-run — 仅检查，不删除文件")
    else:
        print("[模式] 正式运行 — 将删除孤立文件！")

    find_orphans(
        data_root=data_root,
        scan_dirs=scan_dirs,
        dataset_names=dataset_names,
        report_file=report_file,
        dry_run=args.dry_run,
        do_remove_empty_dirs=args.remove_empty_dirs,
    )

    print(f"\n{'='*60}")
    print(f"  完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
