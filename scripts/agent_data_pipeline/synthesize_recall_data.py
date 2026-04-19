"""
轻量 Recall 数据合成脚本

核心思路: 不改动现有训练/推理框架 (grpo.py / inference.py),
用 **现有双动作格式** + **多轮消息** 实现 recall 能力。

                ┌─────────────────────────────────────────┐
                │  现有格式 (不变, 训练框架直接兼容)        │
                │  <think>...</think><response>...         │
                │  <think>...</think><silent>              │
                └─────────────────────────────────────────┘

方法: 从 ThinkStream 现有标注中合成 3 类数据:

  Type 1: Protocol (协议热身) — 直接复用现有 cold_start, 不改动
  Type 2: Recall-positive    — 把 Past 类型样本的 recent_window 缩小,
           让答案证据"离开"可见窗口, 在 think 中植入 "需要回忆" 的推理,
           然后以多轮消息注入 recall_result, 再 response
  Type 3: No-recall control  — 同一问题在证据可见时提问, 直接 response

输出格式完全兼容 thinkstream/data/stream_data_processor.py 的 _build_messages(),
可以直接注册到 data/__init__.py 然后用 sft.sh / rl.sh 训练。

Usage:
    python -m scripts.agent_data_pipeline.synthesize_recall_data \
        --cold_start_path /path/to/streaming_cot_cold_processed.jsonl \
        --rlvr_path /path/to/streaming_rlvr_processed.jsonl \
        --output_dir data/agent/synthesized \
        [--num_recall 2000] [--num_protocol 5000] [--seed 42]
"""

import argparse
import copy
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 缩窗后的 recent_window: 原始视频从 question 时刻往前只保留这么多秒
# 设得越小, 越多证据被"推出"可见窗口, recall 越有意义
SHRUNK_WINDOW_SEC = 4.0

# 回忆注入模板 (作为 user 消息追加到 think-recall 之后)
RECALL_INJECT_TEMPLATE = (
    "以下是从历史片段中检索到的信息，请结合这些信息回答:\n"
    "{evidence}"
)

# Think 中植入的 recall 推理模板
RECALL_THINK_TEMPLATES = [
    "用户问的内容发生在更早的时间，当前画面中看不到相关信息。回忆之前的视频内容: {hint}。",
    "这个问题需要回忆之前看到的内容。根据之前的记忆: {hint}。",
    "当前窗口内找不到答案的直接证据，但之前的片段中: {hint}。",
    "需要结合之前的视频内容来回答。回忆: {hint}。",
]

# 不需要 recall 时的 think 模板
NO_RECALL_THINK_TEMPLATES = [
    "当前画面中可以看到相关信息，直接回答。",
    "答案在当前可见的视频片段中，不需要回忆。",
    "当前窗口内有充分证据。",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_annotations(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


def save_jsonl(items: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 分析 ThinkStream 样本的时间结构
# ---------------------------------------------------------------------------


def analyze_sample(item: Dict) -> Dict:
    """提取一个 ThinkStream 样本的关键时间信息。"""
    conversations = item.get("conversations", [])
    thoughts = item.get("thoughts", [])

    user_turns = [(float(c.get("timestamp", 0)), c.get("content", ""))
                  for c in conversations if c.get("role") == "user"]
    assistant_turns = [(float(c.get("timestamp", 0)), c.get("content", ""))
                       for c in conversations if c.get("role") == "assistant"]
    think_turns = [(float(t.get("timestamp", 0)), t.get("think", ""))
                   for t in thoughts]

    user_turns.sort(key=lambda x: x[0])
    assistant_turns.sort(key=lambda x: x[0])
    think_turns.sort(key=lambda x: x[0])

    # 提问时间
    ask_time = user_turns[-1][0] if user_turns else 0.0
    # 回答时间
    answer_time = assistant_turns[-1][0] if assistant_turns else ask_time
    # 最早 think 时间 (证据开始出现的时间)
    earliest_think = think_turns[0][0] if think_turns else 0.0
    # 问题文本
    question = user_turns[-1][1] if user_turns else ""
    # 回答文本
    answer = assistant_turns[-1][1] if assistant_turns else ""
    # 所有 think 文本拼接 (用于 recall hint)
    all_thinks = " ".join(t[1] for t in think_turns if t[1])

    return {
        "ask_time": ask_time,
        "answer_time": answer_time,
        "earliest_think": earliest_think,
        "question": question,
        "answer": answer,
        "all_thinks": all_thinks,
        "think_turns": think_turns,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "temporal_scope": item.get("temporal_scope", "Current"),
        "response_format": item.get("response_format", "Open-ended"),
        "interaction_mode": item.get("interaction_mode", ""),
        "content_dimension": item.get("content_dimension", ""),
        "video_path": item.get("video_path", ""),
    }


# ---------------------------------------------------------------------------
# 判断是否适合做 recall 样本
# ---------------------------------------------------------------------------


def is_recall_candidate(item: Dict, analysis: Dict) -> bool:
    """判断一个样本是否可以通过缩窗变成 recall 样本。

    条件:
    1. temporal_scope == Past (问的是过去发生的事)
    2. 有至少 2 条 think (有推理过程)
    3. earliest_think 比 ask_time 早至少 SHRUNK_WINDOW_SEC + 2s
       (缩窗后证据确实在窗口外)
    4. 有明确的 answer
    """
    if analysis["temporal_scope"] != "Past":
        return False
    if len(analysis["think_turns"]) < 2:
        return False
    if not analysis["answer"]:
        return False
    # 证据要足够早, 缩窗后才能推出可见区域
    time_gap = analysis["ask_time"] - analysis["earliest_think"]
    if time_gap < SHRUNK_WINDOW_SEC + 2.0:
        return False
    return True


def is_rl_verifiable(item: Dict) -> bool:
    """判断是否可用于 RL 可验证奖励。"""
    rf = item.get("response_format", "")
    return rf in ("Multiple Choice", "Binary", "Counting")


# ---------------------------------------------------------------------------
# 合成 recall-positive 样本
# ---------------------------------------------------------------------------


def synthesize_recall_positive(item: Dict, analysis: Dict) -> Dict:
    """合成一条 recall-positive 样本。

    策略:
    1. 保留原视频路径
    2. 缩小 video_end 到 ask_time - SHRUNK_WINDOW_SEC 之后的窗口
       (让早期 think 对应的帧不在视频输入中)
    3. 在最后的 think 中植入 recall 推理
    4. 在 conversations 中插入 recall_result 轮次
    5. 保持原有的 response 不变
    """
    new_item = copy.deepcopy(item)

    # 构造 recall hint: 用早期 think 内容的关键信息
    early_thinks = [t for t in analysis["think_turns"]
                    if t[0] < analysis["ask_time"] - SHRUNK_WINDOW_SEC]
    hint = " ".join(t[1][:50] for t in early_thinks[:3] if t[1])
    if not hint:
        hint = analysis["all_thinks"][:100]

    # 构造 recall evidence (模拟检索到的历史信息)
    evidence_parts = []
    for t_time, t_text in early_thinks[:3]:
        if t_text:
            evidence_parts.append(f"[{t_time:.1f}s] {t_text[:80]}")
    evidence = "\n".join(evidence_parts) if evidence_parts else hint

    # 修改 thoughts: 保留缩窗后可见的 thinks + 在提问时刻加一条 recall think
    visible_thinks = [t for t in item.get("thoughts", [])
                      if float(t.get("timestamp", 0)) >= analysis["ask_time"] - SHRUNK_WINDOW_SEC]

    # 插入 recall think (在提问时刻)
    recall_think_text = random.choice(RECALL_THINK_TEMPLATES).format(hint=hint[:60])
    recall_think = {
        "timestamp": analysis["ask_time"],
        "think": recall_think_text,
    }
    visible_thinks.append(recall_think)
    visible_thinks.sort(key=lambda t: float(t.get("timestamp", 0)))
    new_item["thoughts"] = visible_thinks

    # 在 conversations 中, 在 user 提问和 assistant 回答之间插入一轮 recall
    new_conversations = []
    recall_injected = False
    for conv in item.get("conversations", []):
        new_conversations.append(conv)
        # 在最后一条 user 消息之后, assistant 回答之前, 注入 recall
        if (conv.get("role") == "user"
                and float(conv.get("timestamp", 0)) == analysis["ask_time"]
                and not recall_injected):
            # 注入 recall_result 作为额外 user 消息
            recall_msg = {
                "role": "user",
                "content": RECALL_INJECT_TEMPLATE.format(evidence=evidence),
                "timestamp": analysis["ask_time"] + 0.1,
                "is_recall_result": True,
            }
            new_conversations.append(recall_msg)
            recall_injected = True

    new_item["conversations"] = new_conversations

    # 标记为 recall 样本
    new_item["sample_type"] = "recall_positive"
    new_item["recall_meta"] = {
        "original_earliest_think": analysis["earliest_think"],
        "shrunk_window_sec": SHRUNK_WINDOW_SEC,
        "evidence_source": "early_thinks",
    }

    return new_item


# ---------------------------------------------------------------------------
# 合成 no-recall control 样本
# ---------------------------------------------------------------------------


def synthesize_no_recall_control(item: Dict, analysis: Dict) -> Dict:
    """合成一条 no-recall control: 同样的问题, 但在证据可见时提问。

    做法: 把 ask_time 移到 earliest_think + 2s 处,
    这样所有 think 证据都在可见窗口内, 不需要 recall。
    """
    new_item = copy.deepcopy(item)

    # 提前提问: 在最早 think 之后 2s
    early_ask_time = analysis["earliest_think"] + 2.0

    # 修改 conversations 中的 user 时间戳
    for conv in new_item.get("conversations", []):
        if conv.get("role") == "user":
            conv["timestamp"] = early_ask_time
        elif conv.get("role") == "assistant":
            conv["timestamp"] = early_ask_time + 0.5

    # 修改 thoughts: 在 early_ask_time 处加入 no-recall think
    no_recall_think = {
        "timestamp": early_ask_time,
        "think": random.choice(NO_RECALL_THINK_TEMPLATES),
    }
    # 只保留 early_ask_time 前后的 thinks
    visible_thinks = [t for t in item.get("thoughts", [])
                      if float(t.get("timestamp", 0)) <= early_ask_time + 2.0]
    visible_thinks.append(no_recall_think)
    visible_thinks.sort(key=lambda t: float(t.get("timestamp", 0)))
    new_item["thoughts"] = visible_thinks

    new_item["sample_type"] = "no_recall_control"
    return new_item


# ---------------------------------------------------------------------------
# 从 cold_start 中筛选 protocol 样本 (直接复用, 轻微增强)
# ---------------------------------------------------------------------------


def prepare_protocol_samples(
    cold_start: List[Dict],
    num: int = 5000,
) -> List[Dict]:
    """从 cold_start 中选取 protocol 热身样本。

    策略: 按 interaction_mode 分层采样, 保留原始格式不做修改。
    """
    # 按 interaction_mode 分桶
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for item in cold_start:
        mode = item.get("interaction_mode", "unknown")
        buckets[mode].append(item)

    # 分层采样
    selected = []
    modes = list(buckets.keys())
    per_mode = max(1, num // max(len(modes), 1))

    for mode in modes:
        pool = buckets[mode]
        n = min(per_mode, len(pool))
        sampled = random.sample(pool, n)
        for s in sampled:
            s["sample_type"] = "protocol"
        selected.extend(sampled)

    # 补足
    if len(selected) < num:
        remaining = [s for s in cold_start if s not in selected]
        extra = random.sample(remaining, min(num - len(selected), len(remaining)))
        for s in extra:
            s["sample_type"] = "protocol"
        selected.extend(extra)

    random.shuffle(selected)
    return selected[:num]


# ---------------------------------------------------------------------------
# 准备 RL 数据 (从所有样本中选可验证的)
# ---------------------------------------------------------------------------


def prepare_rl_samples(
    all_samples: List[Dict],
    rlvr_data: List[Dict],
    num: int = 1000,
) -> List[Dict]:
    """准备 RL 训练数据。

    优先用:
    1. 原始 RLVR 数据 (已有可验证答案)
    2. 合成样本中 response_format 为 MC/Binary/Counting 的
    """
    rl_pool = []

    # 原始 RLVR 数据
    for item in rlvr_data:
        item["sample_type"] = "rl_original"
        rl_pool.append(item)

    # 合成样本中可验证的
    for item in all_samples:
        if is_rl_verifiable(item):
            rl_item = copy.deepcopy(item)
            rl_item["sample_type"] = "rl_synthesized"
            rl_pool.append(rl_item)

    random.shuffle(rl_pool)
    return rl_pool[:num]


# ---------------------------------------------------------------------------
# 主合成流程
# ---------------------------------------------------------------------------


def synthesize(
    cold_start_path: str,
    rlvr_path: str,
    output_dir: str,
    num_recall: int = 2000,
    num_protocol: int = 5000,
    num_rl: int = 1000,
    seed: int = 42,
) -> Dict[str, int]:
    """主合成函数。"""
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 加载数据
    logger.info("Loading cold_start from %s", cold_start_path)
    cold_start = load_annotations(cold_start_path)
    logger.info("  → %d samples", len(cold_start))

    logger.info("Loading RLVR from %s", rlvr_path)
    rlvr = load_annotations(rlvr_path)
    logger.info("  → %d samples", len(rlvr))

    # =========================================================
    # Step 1: 从 cold_start 中找 recall 候选
    # =========================================================
    logger.info("\n=== Step 1: Finding recall candidates ===")
    recall_candidates = []
    for item in cold_start:
        analysis = analyze_sample(item)
        if is_recall_candidate(item, analysis):
            recall_candidates.append((item, analysis))

    logger.info("Recall candidates: %d / %d (%.1f%%)",
                len(recall_candidates), len(cold_start),
                100.0 * len(recall_candidates) / max(len(cold_start), 1))

    # 分析 recall 候选的分布
    scope_dist = Counter(a["temporal_scope"] for _, a in recall_candidates)
    format_dist = Counter(item.get("response_format", "?") for item, _ in recall_candidates)
    logger.info("  temporal_scope: %s", dict(scope_dist))
    logger.info("  response_format: %s", dict(format_dist))

    # =========================================================
    # Step 2: 合成 recall-positive + no-recall control
    # =========================================================
    logger.info("\n=== Step 2: Synthesizing recall pairs ===")

    # 从候选中选取
    selected_recall = random.sample(
        recall_candidates,
        min(num_recall, len(recall_candidates)),
    )

    recall_positive = []
    no_recall_control = []

    for item, analysis in selected_recall:
        # Recall-positive
        rp = synthesize_recall_positive(item, analysis)
        recall_positive.append(rp)

        # No-recall control (配对)
        nrc = synthesize_no_recall_control(item, analysis)
        no_recall_control.append(nrc)

    logger.info("Recall-positive: %d", len(recall_positive))
    logger.info("No-recall control: %d", len(no_recall_control))

    # =========================================================
    # Step 3: 准备 protocol 样本
    # =========================================================
    logger.info("\n=== Step 3: Preparing protocol samples ===")

    # 从 cold_start 中排除已用于 recall 的样本
    recall_ids = set(id(item) for item, _ in selected_recall)
    remaining_cold = [item for item in cold_start if id(item) not in recall_ids]

    protocol = prepare_protocol_samples(remaining_cold, num_protocol)
    logger.info("Protocol samples: %d", len(protocol))

    # =========================================================
    # Step 4: 组装 SFT 数据
    # =========================================================
    logger.info("\n=== Step 4: Assembling SFT data ===")

    sft_all = protocol + recall_positive + no_recall_control
    random.shuffle(sft_all)

    # 分阶段输出
    # SFT-A: 协议热身 (protocol 80% + recall 20%)
    sft_a_recall_n = min(len(recall_positive), len(protocol) // 4)
    sft_a = protocol + recall_positive[:sft_a_recall_n] + no_recall_control[:sft_a_recall_n]
    random.shuffle(sft_a)

    # SFT-B: recall 重点 (recall 50% + control 25% + protocol 25%)
    sft_b_proto_n = min(len(protocol), len(recall_positive) // 2)
    sft_b = (recall_positive + no_recall_control
             + random.sample(protocol, min(sft_b_proto_n, len(protocol))))
    random.shuffle(sft_b)

    save_jsonl(sft_a, out / "sft_a.jsonl")
    save_jsonl(sft_b, out / "sft_b.jsonl")
    save_jsonl(sft_all, out / "sft_all.jsonl")

    logger.info("SFT-A: %d samples (protocol + light recall)", len(sft_a))
    logger.info("SFT-B: %d samples (recall-heavy)", len(sft_b))

    # =========================================================
    # Step 5: 准备 RL 数据
    # =========================================================
    logger.info("\n=== Step 5: Preparing RL data ===")

    rl_samples = prepare_rl_samples(sft_all, rlvr, num_rl)
    save_jsonl(rl_samples, out / "rl_pool.jsonl")
    logger.info("RL pool: %d samples", len(rl_samples))

    # =========================================================
    # 统计报告
    # =========================================================
    stats = {
        "cold_start_total": len(cold_start),
        "rlvr_total": len(rlvr),
        "recall_candidates": len(recall_candidates),
        "recall_positive": len(recall_positive),
        "no_recall_control": len(no_recall_control),
        "protocol": len(protocol),
        "sft_a": len(sft_a),
        "sft_b": len(sft_b),
        "rl_pool": len(rl_samples),
    }

    # 保存统计
    with open(out / "synthesis_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Type 分布
    type_counter = Counter(s.get("sample_type", "?") for s in sft_all)
    logger.info("\nSFT sample type distribution:")
    for st, count in type_counter.most_common():
        logger.info("  %s: %d (%.1f%%)", st, count, 100.0 * count / len(sft_all))

    # RL type 分布
    rl_type_counter = Counter(s.get("sample_type", "?") for s in rl_samples)
    logger.info("\nRL sample type distribution:")
    for st, count in rl_type_counter.most_common():
        logger.info("  %s: %d", st, count)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize recall training data from ThinkStream annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m scripts.agent_data_pipeline.synthesize_recall_data \\
        --cold_start_path /home/.../streaming_cot_cold_processed_5_20.jsonl \\
        --rlvr_path /home/.../streaming_rlvr_processed.jsonl \\
        --output_dir data/agent/synthesized \\
        --num_recall 2000 --num_protocol 5000 --num_rl 1000

Output files:
    data/agent/synthesized/
    ├── sft_a.jsonl          # SFT-A: 协议热身 (protocol 主体 + 少量 recall)
    ├── sft_b.jsonl          # SFT-B: recall 重点训练
    ├── sft_all.jsonl        # 全量 SFT
    ├── rl_pool.jsonl        # RL 训练池
    └── synthesis_stats.json # 合成统计
""",
    )
    parser.add_argument("--cold_start_path", required=True,
                        help="ThinkStream cold start annotation file (.jsonl)")
    parser.add_argument("--rlvr_path", required=True,
                        help="ThinkStream RLVR annotation file (.jsonl)")
    parser.add_argument("--output_dir", default="data/agent/synthesized")
    parser.add_argument("--num_recall", type=int, default=2000,
                        help="Number of recall pairs to synthesize")
    parser.add_argument("--num_protocol", type=int, default=5000,
                        help="Number of protocol warmup samples")
    parser.add_argument("--num_rl", type=int, default=1000,
                        help="Number of RL samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = synthesize(
        cold_start_path=args.cold_start_path,
        rlvr_path=args.rlvr_path,
        output_dir=args.output_dir,
        num_recall=args.num_recall,
        num_protocol=args.num_protocol,
        num_rl=args.num_rl,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("SYNTHESIS COMPLETE")
    print(f"{'='*60}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
