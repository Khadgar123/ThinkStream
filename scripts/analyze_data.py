"""
ThinkStream 数据集分析脚本

读取并分析 SFT（冷启动）和 RL（RLVR）训练数据集的统计信息。
数据路径直接从 thinkstream/data/__init__.py 中注册的路径获取。

Usage:
    python scripts/analyze_data.py
    python scripts/analyze_data.py --dataset stream_cold_start
    python scripts/analyze_data.py --dataset stream_rlvr
    python scripts/analyze_data.py --dataset stream_cold_start,stream_rlvr
"""

import sys
import json
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# 将项目根目录加入 sys.path，以便直接导入 thinkstream 包
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from thinkstream.data import data_list, data_dict


# ---------------------------------------------------------------------------
# 文件读取工具
# ---------------------------------------------------------------------------

def read_annotations(path: str) -> List[Dict[str, Any]]:
    """读取 .jsonl 或 .json 格式的标注文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"标注文件不存在: {path}\n"
            "请先下载数据集，参考 README.md 中的数据集链接。"
        )
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# 通用统计工具
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, float]:
    """计算基础统计量：count / min / max / mean / median。"""
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "count": n,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "mean": sum(sorted_v) / n,
        "median": sorted_v[n // 2],
    }


def _print_stats(label: str, values: List[float], unit: str = "") -> None:
    s = _stats(values)
    u = f" {unit}" if unit else ""
    print(
        f"  {label}: count={s['count']}, "
        f"min={s['min']:.2f}{u}, max={s['max']:.2f}{u}, "
        f"mean={s['mean']:.2f}{u}, median={s['median']:.2f}{u}"
    )


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# SFT（冷启动）数据分析
# ---------------------------------------------------------------------------

def analyze_sft(samples: List[Dict[str, Any]], data_path: str) -> None:
    """分析 stream_cold_start 数据集的结构和统计信息。"""
    _print_section("SFT 冷启动数据集分析 (stream_cold_start)")
    print(f"  标注文件: {data_path}")
    print(f"  样本总数: {len(samples)}")

    # ------------------------------------------------------------------
    # 字段覆盖率
    # ------------------------------------------------------------------
    _print_subsection("字段覆盖率")
    field_counts: Counter = Counter()
    for s in samples:
        for k in s:
            field_counts[k] += 1
    for field, cnt in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(samples) * 100
        print(f"  {field:<25} {cnt:>6} ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 对话轮次统计
    # ------------------------------------------------------------------
    _print_subsection("对话 (conversations) 统计")
    conv_lens, user_lens, asst_lens = [], [], []
    role_counter: Counter = Counter()
    ts_user, ts_asst = [], []

    for s in samples:
        convs = s.get("conversations", [])
        conv_lens.append(len(convs))
        for c in convs:
            role = c.get("role", "unknown")
            role_counter[role] += 1
            text = c.get("content", "")
            word_cnt = len(str(text).split())
            if role == "user":
                user_lens.append(word_cnt)
                if "timestamp" in c:
                    ts_user.append(float(c["timestamp"]))
            elif role == "assistant":
                asst_lens.append(word_cnt)
                if "timestamp" in c:
                    ts_asst.append(float(c["timestamp"]))

    _print_stats("每条样本对话轮数", conv_lens, "turns")
    print(f"  角色分布: {dict(role_counter)}")
    _print_stats("用户消息词数", user_lens, "words")
    _print_stats("助手消息词数", asst_lens, "words")
    if ts_user:
        _print_stats("用户消息时间戳", ts_user, "s")
    if ts_asst:
        _print_stats("助手消息时间戳", ts_asst, "s")

    # ------------------------------------------------------------------
    # Thoughts（思维链）统计
    # ------------------------------------------------------------------
    has_thoughts = [s for s in samples if "thoughts" in s and s["thoughts"]]
    _print_subsection(f"Thoughts（思维链）统计  — 覆盖 {len(has_thoughts)}/{len(samples)} 条样本")
    if has_thoughts:
        thought_counts, thought_word_lens, thought_ts = [], [], []
        for s in has_thoughts:
            thoughts = s["thoughts"]
            thought_counts.append(len(thoughts))
            for t in thoughts:
                text = t.get("think", "")
                thought_word_lens.append(len(str(text).split()))
                if "timestamp" in t:
                    thought_ts.append(float(t["timestamp"]))
        _print_stats("每条样本 thoughts 数量", thought_counts)
        _print_stats("每条 thought 词数", thought_word_lens, "words")
        if thought_ts:
            _print_stats("Thought 时间戳", thought_ts, "s")

    # 输出格式预览
    _print_subsection("SFT 助手输出格式分析")
    response_count = 0
    silent_count = 0
    for s in samples:
        for c in s.get("conversations", []):
            if c.get("role") == "assistant":
                text = str(c.get("content", ""))
                if "<response>" in text:
                    response_count += 1
                if "<silent>" in text:
                    silent_count += 1
    print(f"  含 <response> 的助手轮次: {response_count}")
    print(f"  含 <silent>   的助手轮次: {silent_count}")
    total_asst = role_counter.get("assistant", 0)
    if total_asst > 0:
        print(f"  response/silent 比例: {response_count/(response_count+silent_count+1e-9):.2%} / "
              f"{silent_count/(response_count+silent_count+1e-9):.2%}")

    # ------------------------------------------------------------------
    # Video 路径统计
    # ------------------------------------------------------------------
    _print_subsection("视频路径统计")
    video_paths = [s.get("video_path", "") for s in samples]
    unique_videos = set(p for p in video_paths if p)
    print(f"  唯一视频数: {len(unique_videos)}")
    ext_counter: Counter = Counter()
    for vp in video_paths:
        ext = Path(vp).suffix.lower() if vp else "(无)"
        ext_counter[ext] += 1
    print(f"  视频格式分布: {dict(ext_counter)}")

    # num_tokens（预计算序列长度）
    if any("num_tokens" in s for s in samples):
        _print_subsection("预计算序列长度 (num_tokens)")
        token_lens = [s["num_tokens"] for s in samples if "num_tokens" in s]
        _print_stats("num_tokens", token_lens, "tokens")
        buckets = [0, 1024, 2048, 4096, 8192, 16384, 32768, math.inf]
        dist: Counter = Counter()
        for v in token_lens:
            for i in range(len(buckets) - 1):
                if buckets[i] <= v < buckets[i + 1]:
                    label = f"[{int(buckets[i])}, {int(buckets[i+1]) if buckets[i+1]!=math.inf else '∞'})"
                    dist[label] += 1
                    break
        for label, cnt in dist.items():
            print(f"  {label:<25} {cnt:>5} ({cnt/len(token_lens)*100:5.1f}%)")


# ---------------------------------------------------------------------------
# RL（RLVR）数据分析
# ---------------------------------------------------------------------------

def analyze_rl(samples: List[Dict[str, Any]], data_path: str) -> None:
    """分析 stream_rlvr 数据集的结构和统计信息。"""
    _print_section("RL RLVR 数据集分析 (stream_rlvr)")
    print(f"  标注文件: {data_path}")
    print(f"  样本总数: {len(samples)}")

    # ------------------------------------------------------------------
    # 字段覆盖率
    # ------------------------------------------------------------------
    _print_subsection("字段覆盖率")
    field_counts: Counter = Counter()
    for s in samples:
        for k in s:
            field_counts[k] += 1
    for field, cnt in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(samples) * 100
        print(f"  {field:<25} {cnt:>6} ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 对话结构
    # ------------------------------------------------------------------
    _print_subsection("对话 (conversations) 统计")
    conv_lens: List[int] = []
    role_counter: Counter = Counter()
    user_ts, asst_ts = [], []
    gt_contents: List[str] = []
    asst_content_type: Counter = Counter()

    for s in samples:
        convs = s.get("conversations", [])
        conv_lens.append(len(convs))
        for idx, c in enumerate(convs):
            role = c.get("role", "unknown")
            role_counter[role] += 1
            if role == "user" and "timestamp" in c:
                user_ts.append(float(c["timestamp"]))
            if role == "assistant":
                if "timestamp" in c:
                    asst_ts.append(float(c["timestamp"]))
                content = str(c.get("content", ""))
                gt_contents.append(content)
                # 分析答案类型
                import re
                if re.fullmatch(r"[A-E]", content.strip()):
                    asst_content_type["选择题(A-E)"] += 1
                elif content.strip().lower() in {"yes", "no"}:
                    asst_content_type["是非题(yes/no)"] += 1
                elif re.fullmatch(r"[0-9]+", content.strip()):
                    asst_content_type["数值题(int)"] += 1
                else:
                    asst_content_type["其他"] += 1

    _print_stats("每条样本对话轮数", conv_lens, "turns")
    print(f"  角色分布: {dict(role_counter)}")
    if user_ts:
        _print_stats("用户消息时间戳", user_ts, "s")
    if asst_ts:
        _print_stats("GT 标注时间戳（答题时刻）", asst_ts, "s")
    _print_subsection("GT 答案类型分布")
    total_gt = sum(asst_content_type.values())
    for atype, cnt in sorted(asst_content_type.items(), key=lambda x: -x[1]):
        print(f"  {atype:<20} {cnt:>5} ({cnt/max(total_gt,1)*100:5.1f}%)")

    # ------------------------------------------------------------------
    # 任务类型分布（如果有 task 字段）
    # ------------------------------------------------------------------
    task_counter: Counter = Counter()
    for s in samples:
        task = s.get("task", s.get("type", None))
        if task:
            task_counter[task] += 1
    if task_counter:
        _print_subsection("任务类型分布")
        for task, cnt in sorted(task_counter.items(), key=lambda x: -x[1]):
            print(f"  {task:<30} {cnt:>5} ({cnt/len(samples)*100:5.1f}%)")

    # ------------------------------------------------------------------
    # 视频路径统计
    # ------------------------------------------------------------------
    _print_subsection("视频路径统计")
    video_paths = [s.get("video_path", "") for s in samples]
    unique_videos = set(p for p in video_paths if p)
    print(f"  唯一视频数: {len(unique_videos)}")
    ext_counter: Counter = Counter()
    for vp in video_paths:
        ext = Path(vp).suffix.lower() if vp else "(无)"
        ext_counter[ext] += 1
    print(f"  视频格式分布: {dict(ext_counter)}")

    # ------------------------------------------------------------------
    # GT 时间戳分布（便于了解问答时刻集中在视频的哪个位置）
    # ------------------------------------------------------------------
    if asst_ts:
        _print_subsection("GT 答题时刻分布（按 20s 分桶）")
        bucket_size = 20.0
        bucket_counter: Counter = Counter()
        for t in asst_ts:
            bucket = int(t // bucket_size) * int(bucket_size)
            bucket_counter[f"{bucket}s~{bucket+int(bucket_size)}s"] += 1
        for label, cnt in sorted(bucket_counter.items(), key=lambda x: int(x[0].split("s")[0])):
            bar = "#" * min(cnt, 40)
            print(f"  {label:<15} {cnt:>4}  {bar}")


# ---------------------------------------------------------------------------
# 综合对比
# ---------------------------------------------------------------------------

def compare_datasets(
    sft_samples: Optional[List[Dict]] = None,
    rl_samples: Optional[List[Dict]] = None,
) -> None:
    if sft_samples is None or rl_samples is None:
        return
    _print_section("SFT vs RL 数据集对比")
    print(f"  {'指标':<30} {'SFT (cold-start)':>20} {'RL (RLVR)':>20}")
    print(f"  {'-'*72}")

    def fmt(v):
        return f"{v:>20}" if isinstance(v, str) else f"{v:>20.2f}"

    print(f"  {'样本总数':<30} {len(sft_samples):>20} {len(rl_samples):>20}")

    sft_has_thoughts = sum(1 for s in sft_samples if s.get("thoughts"))
    rl_has_thoughts = sum(1 for s in rl_samples if s.get("thoughts"))
    print(f"  {'含 thoughts 字段样本数':<30} {sft_has_thoughts:>20} {rl_has_thoughts:>20}")

    sft_conv_lens = [len(s.get("conversations", [])) for s in sft_samples]
    rl_conv_lens = [len(s.get("conversations", [])) for s in rl_samples]
    sft_avg = sum(sft_conv_lens) / max(len(sft_conv_lens), 1)
    rl_avg = sum(rl_conv_lens) / max(len(rl_conv_lens), 1)
    print(f"  {'平均对话轮数':<30} {sft_avg:>20.2f} {rl_avg:>20.2f}")

    sft_videos = len(set(s.get("video_path", "") for s in sft_samples if s.get("video_path")))
    rl_videos = len(set(s.get("video_path", "") for s in rl_samples if s.get("video_path")))
    print(f"  {'唯一视频数':<30} {sft_videos:>20} {rl_videos:>20}")

    if any("num_tokens" in s for s in sft_samples):
        sft_toks = [s["num_tokens"] for s in sft_samples if "num_tokens" in s]
        avg_tok = sum(sft_toks) / max(len(sft_toks), 1)
        print(f"  {'SFT 平均序列长度(tokens)':<30} {avg_tok:>20.0f} {'N/A':>20}")


# ---------------------------------------------------------------------------
# 示例样本打印
# ---------------------------------------------------------------------------

def print_sample(sample: Dict[str, Any], title: str = "样本示例", max_text: int = 120) -> None:
    """打印单条样本的可读摘要。"""
    _print_section(title)

    def truncate(text: str) -> str:
        text = str(text)
        return text[:max_text] + "..." if len(text) > max_text else text

    print(f"  video_path : {sample.get('video_path', 'N/A')}")
    print(f"  data_path  : {sample.get('data_path', 'N/A')}")
    if "num_tokens" in sample:
        print(f"  num_tokens : {sample['num_tokens']}")
    print()

    for i, c in enumerate(sample.get("conversations", [])):
        role = c.get("role", "?")
        ts = c.get("timestamp", "")
        ts_str = f" [t={ts}s]" if ts != "" else ""
        content = truncate(str(c.get("content", "")))
        print(f"  conversations[{i}] role={role}{ts_str}")
        print(f"    content: {content}")

    thoughts = sample.get("thoughts", [])
    if thoughts:
        print()
        for i, t in enumerate(thoughts[:3]):
            ts = t.get("timestamp", "")
            think = truncate(str(t.get("think", "")))
            print(f"  thoughts[{i}] t={ts}s  → {think}")
        if len(thoughts) > 3:
            print(f"  ... (共 {len(thoughts)} 条 thought)")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThinkStream 训练数据集分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stream_cold_start,stream_rlvr",
        help=(
            "要分析的数据集名称，逗号分隔。"
            "可选: stream_cold_start, stream_rlvr。"
            "支持采样后缀，例如: stream_cold_start%%50"
        ),
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        default=False,
        help="打印每个数据集的第一条样本",
    )
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]

    # 通过 data_list() 获取所有数据集的路径配置（与训练代码完全一致）
    try:
        configs = data_list(dataset_names)
    except ValueError as e:
        print(f"[错误] {e}")
        print(f"可用数据集: {list(data_dict.keys())}")
        sys.exit(1)

    loaded: Dict[str, List[Dict]] = {}

    for name, cfg in zip(dataset_names, configs):
        # 去掉可能的采样后缀以判断类型
        import re as _re
        base_name = _re.sub(r"%\d+$", "", name)
        annotation_path = cfg["annotation_path"]
        data_path = cfg["data_path"]

        print(f"\n[加载] {base_name} ← {annotation_path}")
        try:
            samples = read_annotations(annotation_path)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            continue

        # 把 data_path 注入到每条样本（与训练代码保持一致）
        for s in samples:
            s["data_path"] = data_path

        # 采样率
        sampling_rate = cfg.get("sampling_rate", 1.0)
        if sampling_rate < 1.0:
            import random
            samples = random.sample(samples, int(len(samples) * sampling_rate))
            print(f"  采样后剩余: {len(samples)} 条")

        loaded[base_name] = samples

        if args.show_sample and samples:
            print_sample(samples[0], title=f"{base_name} — 第一条样本")

        if base_name == "stream_cold_start":
            analyze_sft(samples, annotation_path)
        elif base_name == "stream_rlvr":
            analyze_rl(samples, annotation_path)
        else:
            # 未知数据集：只打印通用字段统计
            _print_section(f"通用数据集分析: {base_name}")
            print(f"  样本总数: {len(samples)}")
            field_counts: Counter = Counter()
            for s in samples:
                for k in s:
                    field_counts[k] += 1
            for field, cnt in sorted(field_counts.items(), key=lambda x: -x[1]):
                print(f"  {field:<25} {cnt:>6}")

    # 综合对比
    if "stream_cold_start" in loaded and "stream_rlvr" in loaded:
        compare_datasets(loaded["stream_cold_start"], loaded["stream_rlvr"])

    print("\n" + "=" * 60)
    print("  分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
