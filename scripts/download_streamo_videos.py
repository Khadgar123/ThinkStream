"""
Streamo 缺失视频下载与切分脚本

功能：
1. generate   — 从 Streamo-Instruct-465K JSON 提取视频列表，生成下载 URL
2. download-howto  — 用 yt-dlp 下载 how_to_step / how_to_caption 视频
3. build-koala-index — 下载 Koala-36M CSV，建立 title→YouTube ID 映射
4. download-koala   — 按映射下载 Koala 原始视频
5. split-koala      — 按 segment 时间戳切分 Koala 视频
6. verify           — 验证视频覆盖率

Usage:
    python scripts/download_streamo_videos.py generate \
        --streamo-path /home/tione/notebook/gaozhenkun/hzh/data/Streamo-Instruct-465K \
        --output-dir /home/tione/notebook/gaozhenkun/hzh/data/download_lists

    python scripts/download_streamo_videos.py download-howto \
        --list-dir /home/tione/notebook/gaozhenkun/hzh/data/download_lists \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data

    python scripts/download_streamo_videos.py build-koala-index \
        --list-dir /home/tione/notebook/gaozhenkun/hzh/data/download_lists \
        --workers 8

    python scripts/download_streamo_videos.py download-koala \
        --list-dir /home/tione/notebook/gaozhenkun/hzh/data/download_lists \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data

    python scripts/download_streamo_videos.py split-koala \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data

    python scripts/download_streamo_videos.py verify \
        --streamo-path /home/tione/notebook/gaozhenkun/hzh/data/Streamo-Instruct-465K \
        --data-root /home/tione/notebook/gaozhenkun/hzh/data
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

KOALA_CSV_URLS = [
    f"https://huggingface.co/datasets/Koala-36M/Koala-36M-v1/resolve/main/Koala_36M_{i}.csv"
    for i in range(1, 11)
]

SEGMENT_RE = re.compile(
    r"^(.+)_segment_(\d{2})-(\d{2})-(\d{2})-(\d{3})_to_(\d{2})-(\d{2})-(\d{2})-(\d{3})\.mp4$"
)


def _parse_segment(filename: str) -> Optional[Tuple[str, float, float]]:
    """解析 Koala segment 文件名，返回 (title, start_sec, end_sec)。"""
    m = SEGMENT_RE.match(filename)
    if not m:
        return None
    title = m.group(1)
    start = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000
    end = int(m.group(6)) * 3600 + int(m.group(7)) * 60 + int(m.group(8)) + int(m.group(9)) / 1000
    return title, start, end


# ---------------------------------------------------------------------------
# Step 1: generate — 生成下载列表
# ---------------------------------------------------------------------------

def cmd_generate(args):
    streamo = Path(args.streamo_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    howto_step_ids: Set[str] = set()
    howto_caption_ids: Set[str] = set()
    koala_segments: Set[str] = set()  # segment filenames
    koala_titles: Set[str] = set()

    for subdir in streamo.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for jf in subdir.glob("*.json"):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for item in data:
                vp = item.get("video_path", "")
                if vp.startswith("how_to_step/"):
                    vid = vp.split("/")[-1].replace(".mp4", "")
                    howto_step_ids.add(vid)
                elif vp.startswith("how_to_caption/"):
                    vid = vp.split("/")[-1].replace(".mp4", "")
                    howto_caption_ids.add(vid)
                elif vp.startswith("Koala/"):
                    seg = vp.split("/", 1)[1]
                    koala_segments.add(seg)
                    parsed = _parse_segment(seg)
                    if parsed:
                        koala_titles.add(parsed[0])

    # how_to_step URLs
    step_file = output / "how_to_step_urls.txt"
    with open(step_file, "w") as f:
        for vid in sorted(howto_step_ids):
            f.write(f"https://www.youtube.com/watch?v={vid}\n")
    print(f"how_to_step: {len(howto_step_ids)} URLs → {step_file}")

    # how_to_caption URLs
    caption_file = output / "how_to_caption_urls.txt"
    with open(caption_file, "w") as f:
        for vid in sorted(howto_caption_ids):
            f.write(f"https://www.youtube.com/watch?v={vid}\n")
    print(f"how_to_caption: {len(howto_caption_ids)} URLs → {caption_file}")

    # Koala titles
    titles_file = output / "koala_titles.txt"
    with open(titles_file, "w", encoding="utf-8") as f:
        for t in sorted(koala_titles):
            f.write(t + "\n")
    print(f"Koala 唯一标题: {len(koala_titles)} → {titles_file}")

    # Koala segments (for split phase)
    segments_file = output / "koala_segments.json"
    # Group by title
    title_segments: Dict[str, List[Dict]] = defaultdict(list)
    for seg in koala_segments:
        parsed = _parse_segment(seg)
        if parsed:
            title, start, end = parsed
            title_segments[title].append({
                "filename": seg,
                "start": start,
                "end": end,
            })
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(title_segments, f, ensure_ascii=False, indent=2)
    print(f"Koala segments: {len(koala_segments)} 片段, {len(title_segments)} 视频 → {segments_file}")


# ---------------------------------------------------------------------------
# Step 2: download-howto
# ---------------------------------------------------------------------------

def cmd_download_howto(args):
    list_dir = Path(args.list_dir)
    data_root = Path(args.data_root)

    for name in ["how_to_step", "how_to_caption"]:
        url_file = list_dir / f"{name}_urls.txt"
        if not url_file.exists():
            print(f"[跳过] {url_file} 不存在，请先运行 generate")
            continue

        out_dir = data_root / "datasets" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n下载 {name} → {out_dir}")
        cmd = [
            "yt-dlp",
            "-a", str(url_file),
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            "-f", "best[height<=480]",
            "--no-playlist",
            "--retries", "3",
            "--ignore-errors",
            "--no-overwrites",
            "--console-title",
        ]
        subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Step 3: build-koala-index — 下载 CSV + 建立 title→ID 映射
# ---------------------------------------------------------------------------

def _get_title_for_id(yt_id: str) -> Optional[str]:
    """用 yt-dlp 获取单个视频的标题。"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-download", "--no-warnings",
             f"https://www.youtube.com/watch?v={yt_id}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def _normalize_title(title: str) -> str:
    """将 YouTube 标题归一化为 Koala 文件名格式（用于匹配）。"""
    # Koala 文件名中空格被替换为 _, 特殊字符被保留或替换
    t = title.replace(" ", "_")
    # 移除或替换文件系统不安全字符
    t = re.sub(r'[<>:"/\\|?*]', '_', t)
    return t


def cmd_build_koala_index(args):
    list_dir = Path(args.list_dir)
    workers = args.workers

    # 1. 读取需要的 Koala 标题
    titles_file = list_dir / "koala_titles.txt"
    if not titles_file.exists():
        print("[错误] koala_titles.txt 不存在，请先运行 generate")
        sys.exit(1)

    needed_titles = set()
    with open(titles_file, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                needed_titles.add(t)
    print(f"需要匹配的 Koala 标题: {len(needed_titles)}")

    # 2. 下载 Koala-36M CSV，提取唯一 YouTube ID
    csv_dir = list_dir / "koala_csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_yt_ids: Set[str] = set()
    for url in KOALA_CSV_URLS:
        fname = url.split("/")[-1]
        local = csv_dir / fname
        if not local.exists():
            print(f"下载 {fname} ...")
            subprocess.run(["curl", "-sL", "-o", str(local), url], check=True)
        else:
            print(f"已存在 {fname}")

        with open(local, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("videoID", "")
                # videoID 格式: lwAFSttEqno_96 → YouTube ID: lwAFSttEqno
                yt_id = vid.rsplit("_", 1)[0] if "_" in vid else vid
                if yt_id:
                    all_yt_ids.add(yt_id)

    print(f"Koala-36M 唯一 YouTube ID: {len(all_yt_ids)}")

    # 3. 检查已有的映射文件（支持断点续传）
    mapping_file = list_dir / "koala_id_to_title.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            id_to_title: Dict[str, str] = json.load(f)
        print(f"已有映射: {len(id_to_title)} 条")
    else:
        id_to_title = {}

    # 4. 用 yt-dlp 批量获取标题
    remaining = [yt_id for yt_id in all_yt_ids if yt_id not in id_to_title]
    print(f"需要获取标题: {remaining[:5]}... 共 {len(remaining)} 个")

    if remaining:
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_get_title_for_id, yt_id): yt_id for yt_id in remaining}
            for future in as_completed(futures):
                yt_id = futures[future]
                title = future.result()
                done += 1
                if title:
                    id_to_title[yt_id] = title
                else:
                    failed += 1

                if done % 500 == 0:
                    print(f"  进度: {done}/{len(remaining)}, 成功: {done - failed}, 失败: {failed}")
                    # 定期保存
                    with open(mapping_file, "w", encoding="utf-8") as f:
                        json.dump(id_to_title, f, ensure_ascii=False)

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(id_to_title, f, ensure_ascii=False)
        print(f"标题获取完成: 成功 {len(id_to_title)}, 失败 {failed}")

    # 5. 建立 normalized_title → yt_id 反向索引
    norm_to_id: Dict[str, str] = {}
    for yt_id, title in id_to_title.items():
        norm = _normalize_title(title)
        norm_to_id[norm] = yt_id

    # 6. 匹配 Streamo Koala 标题
    matched = 0
    unmatched = []
    title_to_id: Dict[str, str] = {}

    for needed in needed_titles:
        if needed in norm_to_id:
            title_to_id[needed] = norm_to_id[needed]
            matched += 1
        else:
            unmatched.append(needed)

    print(f"\n匹配结果: {matched}/{len(needed_titles)} ({matched/len(needed_titles)*100:.1f}%)")
    if unmatched:
        print(f"未匹配: {len(unmatched)} 个")
        for t in unmatched[:10]:
            print(f"  {t}")

    # 保存 title→ID 映射
    t2id_file = list_dir / "koala_title_to_id.json"
    with open(t2id_file, "w", encoding="utf-8") as f:
        json.dump(title_to_id, f, ensure_ascii=False, indent=2)
    print(f"映射已保存: {t2id_file}")

    # 生成下载 URL 列表
    url_file = list_dir / "koala_urls.txt"
    with open(url_file, "w") as f:
        for title, yt_id in sorted(title_to_id.items()):
            f.write(f"https://www.youtube.com/watch?v={yt_id}\n")
    print(f"Koala 下载 URL: {len(title_to_id)} → {url_file}")


# ---------------------------------------------------------------------------
# Step 4: download-koala
# ---------------------------------------------------------------------------

def cmd_download_koala(args):
    list_dir = Path(args.list_dir)
    data_root = Path(args.data_root)

    t2id_file = list_dir / "koala_title_to_id.json"
    if not t2id_file.exists():
        print("[错误] koala_title_to_id.json 不存在，请先运行 build-koala-index")
        sys.exit(1)

    with open(t2id_file, "r", encoding="utf-8") as f:
        title_to_id: Dict[str, str] = json.load(f)

    raw_dir = data_root / "datasets" / "Koala_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"下载 {len(title_to_id)} 个 Koala 原始视频 → {raw_dir}")

    done = 0
    failed = 0
    for title, yt_id in title_to_id.items():
        out_path = raw_dir / f"{title}.mp4"
        if out_path.exists():
            done += 1
            continue

        url = f"https://www.youtube.com/watch?v={yt_id}"
        cmd = [
            "yt-dlp", url,
            "-o", str(raw_dir / f"{title}.%(ext)s"),
            "-f", "best[height<=480]",
            "--no-playlist",
            "--retries", "3",
            "--ignore-errors",
            "--no-warnings",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        done += 1
        if result.returncode != 0:
            failed += 1

        if done % 100 == 0:
            print(f"  进度: {done}/{len(title_to_id)}, 失败: {failed}")

    print(f"下载完成: {done - failed} 成功, {failed} 失败")


# ---------------------------------------------------------------------------
# Step 5: split-koala
# ---------------------------------------------------------------------------

def cmd_split_koala(args):
    data_root = Path(args.data_root)
    list_dir = Path(args.list_dir)

    segments_file = list_dir / "koala_segments.json"
    if not segments_file.exists():
        print("[错误] koala_segments.json 不存在，请先运行 generate")
        sys.exit(1)

    with open(segments_file, "r", encoding="utf-8") as f:
        title_segments: Dict[str, List[Dict]] = json.load(f)

    raw_dir = data_root / "datasets" / "Koala_raw"
    out_dir = data_root / "datasets" / "Koala"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"切分 Koala 视频: {len(title_segments)} 个原始视频")

    total_segments = 0
    done_segments = 0
    skipped = 0
    failed_videos = 0

    for title, segments in title_segments.items():
        total_segments += len(segments)

        # 查找原始视频（可能是 mp4 或 webm 等）
        raw_path = None
        for ext in [".mp4", ".webm", ".mkv"]:
            candidate = raw_dir / f"{title}{ext}"
            if candidate.exists():
                raw_path = candidate
                break

        if raw_path is None:
            failed_videos += 1
            continue

        for seg in segments:
            out_path = out_dir / seg["filename"]
            if out_path.exists():
                done_segments += 1
                skipped += 1
                continue

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-i", str(raw_path),
                "-c", "copy",
                "-avoid_negative_ts", "1",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                done_segments += 1
            else:
                # 尝试重编码
                cmd_reencode = [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-to", str(seg["end"]),
                    "-i", str(raw_path),
                    "-c:v", "libx264", "-c:a", "aac",
                    str(out_path),
                ]
                result2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
                if result2.returncode == 0:
                    done_segments += 1

        if (failed_videos + done_segments) % 500 == 0:
            print(f"  进度: {done_segments}/{total_segments} 片段, 跳过: {skipped}, 缺失原始视频: {failed_videos}")

    print(f"\n切分完成:")
    print(f"  总片段: {total_segments}")
    print(f"  成功: {done_segments}")
    print(f"  跳过(已存在): {skipped}")
    print(f"  缺失原始视频: {failed_videos} 个标题")


# ---------------------------------------------------------------------------
# Step 6: verify
# ---------------------------------------------------------------------------

def cmd_verify(args):
    streamo = Path(args.streamo_path)
    data_root = Path(args.data_root)

    # 路径映射规则
    VM = data_root / "datasets" / "VideoMind-Dataset"
    LLAVA = data_root / "datasets" / "LLaVA-Video-178K"
    NAME_MAP = {
        "LLaVA_Video": str(LLAVA),
        "QVHighlight": str(VM / "qvhighlights" / "videos"),
        "didemo": str(VM / "didemo" / "videos"),
        "ActivityNet": str(VM / "activitynet" / "videos"),
        "coin": str(VM / "coin" / "videos"),
        "queryd": str(VM / "queryd" / "videos"),
        "YouCookv2": str(VM / "youcook2" / "videos"),
        "tacos": str(VM / "tacos" / "videos"),
        "ego_timeqa": str(VM / "ego_timeqa" / "videos"),
        "Koala": str(data_root / "datasets" / "Koala"),
        "how_to_step": str(data_root / "datasets" / "how_to_step"),
        "how_to_caption": str(data_root / "datasets" / "how_to_caption"),
    }

    from collections import Counter
    top_stats: Dict[str, Dict[str, int]] = {}

    for subdir in streamo.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for jf in subdir.glob("*.json"):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for item in data:
                vp = item.get("video_path", "")
                if not vp:
                    continue
                top = vp.split("/")[0]
                if top not in top_stats:
                    top_stats[top] = {"total": 0, "found": 0}
                top_stats[top]["total"] += 1

                mapped_base = NAME_MAP.get(top)
                if mapped_base is None:
                    continue

                if top == "LLaVA_Video":
                    rest = "/".join(vp.split("/")[1:])
                    full = Path(mapped_base) / rest
                elif top in ("Koala", "how_to_step", "how_to_caption"):
                    filename = vp.split("/")[-1]
                    full = Path(mapped_base) / filename
                elif top == "YouCookv2":
                    filename = vp.rstrip("/").split("/")[-1]
                    if not filename.endswith(".mp4"):
                        filename += ".mp4"
                    full = Path(mapped_base) / filename
                else:
                    filename = vp.split("/")[-1]
                    full = Path(mapped_base) / filename

                if full.exists():
                    top_stats[top]["found"] += 1

    print(f"\n{'来源':<20} {'总数':>8} {'找到':>8} {'覆盖率':>8}")
    print("-" * 50)
    total_all = 0
    found_all = 0
    for top, stats in sorted(top_stats.items(), key=lambda x: -x[1]["total"]):
        total = stats["total"]
        found = stats["found"]
        pct = found / total * 100 if total > 0 else 0
        print(f"{top:<20} {total:>8} {found:>8} {pct:>7.1f}%")
        total_all += total
        found_all += found
    print("-" * 50)
    print(f"{'总计':<20} {total_all:>8} {found_all:>8} {found_all/total_all*100:>7.1f}%")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Streamo 缺失视频下载与切分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="生成下载列表")
    p_gen.add_argument("--streamo-path", required=True)
    p_gen.add_argument("--output-dir", required=True)

    # download-howto
    p_howto = sub.add_parser("download-howto", help="下载 how_to 系列视频")
    p_howto.add_argument("--list-dir", required=True)
    p_howto.add_argument("--data-root", required=True)

    # build-koala-index
    p_idx = sub.add_parser("build-koala-index", help="下载 Koala CSV 并建立 title→ID 映射")
    p_idx.add_argument("--list-dir", required=True)
    p_idx.add_argument("--workers", type=int, default=8)

    # download-koala
    p_koala = sub.add_parser("download-koala", help="下载 Koala 原始视频")
    p_koala.add_argument("--list-dir", required=True)
    p_koala.add_argument("--data-root", required=True)

    # split-koala
    p_split = sub.add_parser("split-koala", help="切分 Koala 视频")
    p_split.add_argument("--list-dir", required=True)
    p_split.add_argument("--data-root", required=True)

    # verify
    p_verify = sub.add_parser("verify", help="验证视频覆盖率")
    p_verify.add_argument("--streamo-path", required=True)
    p_verify.add_argument("--data-root", required=True)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "download-howto":
        cmd_download_howto(args)
    elif args.command == "build-koala-index":
        cmd_build_koala_index(args)
    elif args.command == "download-koala":
        cmd_download_koala(args)
    elif args.command == "split-koala":
        cmd_split_koala(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
