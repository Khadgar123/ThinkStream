"""
分阶段质量检查脚本

每个 Stage 完成后运行, 输出质量指标, 判断是否可以进入下一阶段。
也可以一次性检查全部已完成的阶段。

Usage:
    # 检查特定阶段
    python -m scripts.agent_data_pipeline.quality_check --stage 0
    python -m scripts.agent_data_pipeline.quality_check --stage 2

    # 检查所有已完成阶段
    python -m scripts.agent_data_pipeline.quality_check --all

    # 详细模式 (打印样本示例)
    python -m scripts.agent_data_pipeline.quality_check --stage 6 --verbose
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    ACTION_DIST_TARGET,
    DIFFICULTY_DIST_TARGET,
    EPISODE_DENSE_PATH,
    EPISODE_FINAL_PATH,
    EPISODE_RAW_PATH,
    EPISODE_VERIFIED_PATH,
    EVENT_TIMELINE_DIR,
    GATE_THRESHOLDS,
    RETRIEVER_TRAIN_PATH,
    RL_POOL_PATH,
    SEGMENT_ARCHIVE_DIR,
    SFT_DIR,
)
from .utils import read_jsonl

logger = logging.getLogger(__name__)

# 新 agent 格式正则
_AGENT_FORMAT_RE = re.compile(
    r"^<think>.*?</think>"
    r"<action>(?:silent|response|recall)</action>"
    r"(?:<response>.*?</response>|<query>\{.*?\}</query>)?$",
    re.DOTALL,
)


def _pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100.0 * count / total:.1f}%"


# ===================================================================
# Stage 0: Segment Archive Quality
# ===================================================================


def check_stage0(verbose: bool = False) -> Tuple[bool, Dict]:
    """检查 Stage 0 输出质量。"""
    print("\n" + "=" * 60)
    print("Stage 0: Segment Archive Quality Check")
    print("=" * 60)

    archives = list(SEGMENT_ARCHIVE_DIR.glob("*.jsonl"))
    if not archives:
        print("  NO DATA - 没有找到 segment archive 文件")
        return False, {}

    total_segments = 0
    total_videos = len(archives)
    issues = {"no_caption": 0, "no_keyframes": 0, "no_entity_tags": 0, "no_embedding": 0}
    recallabilities = []

    for archive in archives:
        segments = read_jsonl(archive)
        total_segments += len(segments)

        for seg in segments:
            if not seg.get("dense_caption"):
                issues["no_caption"] += 1
            if not seg.get("keyframe_paths"):
                issues["no_keyframes"] += 1
            if not seg.get("entity_tags"):
                issues["no_entity_tags"] += 1
            if not seg.get("text_emb_path"):
                issues["no_embedding"] += 1

        # 粗算 recallability
        if segments:
            from .utils import compute_recallability
            r = compute_recallability(segments)
            recallabilities.append(r)

    print(f"\n  Videos processed:  {total_videos}")
    print(f"  Total segments:    {total_segments}")
    print(f"  Avg segs/video:    {total_segments / max(total_videos, 1):.1f}")

    print(f"\n  Coverage issues:")
    for issue, count in issues.items():
        pct = _pct(count, total_segments)
        status = "OK" if count / max(total_segments, 1) < 0.1 else "WARN"
        print(f"    {issue}: {count} ({pct}) [{status}]")

    if recallabilities:
        avg_r = sum(recallabilities) / len(recallabilities)
        print(f"\n  Recallability: mean={avg_r:.3f}, "
              f"min={min(recallabilities):.3f}, max={max(recallabilities):.3f}")

    # 抽样展示
    if verbose and archives:
        sample_seg = read_jsonl(archives[0])[0]
        print(f"\n  Sample segment from {archives[0].stem}:")
        for k in ["segment_id", "dense_caption", "entity_tags", "action_tags",
                   "ocr_text", "asr_text", "salience", "memory_keys"]:
            print(f"    {k}: {sample_seg.get(k, 'N/A')}")

    caption_rate = 1.0 - issues["no_caption"] / max(total_segments, 1)
    passed = caption_rate >= 0.9 and total_segments > 0
    print(f"\n  Caption coverage: {caption_rate:.1%} [threshold ≥ 90%] → {_pass_fail(passed)}")

    return passed, {"videos": total_videos, "segments": total_segments, "caption_rate": caption_rate}


# ===================================================================
# Stage 1: Event Timeline Quality
# ===================================================================


def check_stage1(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 1: Event Timeline Quality Check")
    print("=" * 60)

    timelines = list(EVENT_TIMELINE_DIR.glob("*.jsonl"))
    if not timelines:
        print("  NO DATA")
        return False, {}

    total_events = 0
    type_counter = Counter()
    causal_depths = []
    importances = []

    for tl in timelines:
        events = read_jsonl(tl)
        total_events += len(events)
        for evt in events:
            type_counter[evt.get("event_type", "unknown")] += 1
            depth = len(evt.get("causal_links_prev", [])) + len(evt.get("causal_links_next", []))
            causal_depths.append(depth)
            importances.append(evt.get("importance", 0))

    print(f"\n  Videos with timelines: {len(timelines)}")
    print(f"  Total events:          {total_events}")
    print(f"  Avg events/video:      {total_events / max(len(timelines), 1):.1f}")

    print(f"\n  Event type distribution:")
    for etype, count in type_counter.most_common():
        print(f"    {etype}: {count} ({_pct(count, total_events)})")

    if causal_depths:
        avg_depth = sum(causal_depths) / len(causal_depths)
        has_causal = sum(1 for d in causal_depths if d > 0)
        print(f"\n  Causal links: {_pct(has_causal, len(causal_depths))} events have links, "
              f"avg depth={avg_depth:.2f}")

    if verbose and timelines:
        sample_events = read_jsonl(timelines[0])[:2]
        print(f"\n  Sample events from {timelines[0].stem}:")
        for evt in sample_events:
            print(f"    {evt.get('event_id')}: [{evt.get('start_ms')}-{evt.get('end_ms')}ms] "
                  f"{evt.get('event_type')} - {evt.get('summary', '')[:50]}")

    passed = total_events > 0 and len(type_counter) >= 2
    print(f"\n  Overall: {total_events} events, {len(type_counter)} types → {_pass_fail(passed)}")
    return passed, {"events": total_events, "types": len(type_counter)}


# ===================================================================
# Stage 2: Teacher Task Pack Quality
# ===================================================================


def check_stage2(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 2: Teacher Task Pack Quality Check")
    print("=" * 60)

    if not EPISODE_RAW_PATH.exists():
        print("  NO DATA")
        return False, {}

    episodes = read_jsonl(EPISODE_RAW_PATH)
    print(f"\n  Total episodes: {len(episodes)}")

    # Task type distribution
    type_counter = Counter(ep.get("task_type", "unknown") for ep in episodes)
    print(f"\n  Task type distribution:")
    for tt, count in type_counter.most_common():
        print(f"    {tt}: {count} ({_pct(count, len(episodes))})")

    # Difficulty distribution
    diff_counter = Counter(ep.get("difficulty", "unknown") for ep in episodes)
    print(f"\n  Difficulty distribution (target: E25/M35/H30/VH10):")
    for diff in ["easy", "medium", "hard", "very_hard"]:
        actual = diff_counter.get(diff, 0)
        target = DIFFICULTY_DIST_TARGET.get(diff, 0)
        actual_pct = actual / max(len(episodes), 1)
        delta = abs(actual_pct - target)
        status = "OK" if delta < 0.15 else "WARN"
        print(f"    {diff}: {actual} ({actual_pct:.1%}) target={target:.0%} Δ={delta:.1%} [{status}]")

    # Recall ratio
    recall_count = sum(1 for ep in episodes if ep.get("need_recall"))
    print(f"\n  Need recall: {recall_count} ({_pct(recall_count, len(episodes))})")

    # Query candidates
    has_queries = sum(1 for ep in episodes if ep.get("query_candidates"))
    print(f"  Has query candidates: {has_queries} ({_pct(has_queries, recall_count)})")

    # Answer type distribution
    ans_counter = Counter()
    for ep in episodes:
        ca = ep.get("canonical_answer", {})
        ans_counter[ca.get("answer_type", "missing")] += 1
    print(f"\n  Answer type distribution:")
    for at, count in ans_counter.most_common():
        print(f"    {at}: {count} ({_pct(count, len(episodes))})")

    if verbose and episodes:
        print(f"\n  Sample episode:")
        ep = episodes[0]
        for k in ["episode_id", "task_type", "difficulty", "question",
                   "need_recall", "ask_time_ms", "natural_response"]:
            val = ep.get(k, "N/A")
            if isinstance(val, str) and len(val) > 80:
                val = val[:80] + "..."
            print(f"    {k}: {val}")

    passed = len(episodes) > 0 and recall_count > 0 and has_queries > 0
    print(f"\n  Overall → {_pass_fail(passed)}")
    return passed, {"episodes": len(episodes), "recall": recall_count}


# ===================================================================
# Stage 3: Think Expansion Quality
# ===================================================================


def check_stage3(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 3: Think Expansion Quality Check")
    print("=" * 60)

    if not EPISODE_DENSE_PATH.exists():
        print("  NO DATA")
        return False, {}

    episodes = read_jsonl(EPISODE_DENSE_PATH)
    has_chunks = [ep for ep in episodes if ep.get("chunk_sequence")]
    print(f"\n  Total episodes: {len(episodes)}")
    print(f"  With chunk_sequence: {len(has_chunks)} ({_pct(len(has_chunks), len(episodes))})")

    # Action distribution across all chunks
    action_counter = Counter()
    total_chunks = 0
    for ep in has_chunks:
        for chunk in ep["chunk_sequence"]:
            action_counter[chunk["action"]] += 1
            total_chunks += 1

    print(f"\n  Total chunks: {total_chunks}")
    print(f"  Action distribution (target: S58-65/R23-30/Re10-15):")
    for action in ["silent", "response", "recall"]:
        count = action_counter.get(action, 0)
        actual_pct = count / max(total_chunks, 1)
        lo, hi = ACTION_DIST_TARGET.get(action, (0, 1))
        in_range = lo <= actual_pct <= hi
        status = "OK" if in_range else "WARN"
        print(f"    {action}: {count} ({actual_pct:.1%}) range=[{lo:.0%}-{hi:.0%}] [{status}]")

    if verbose and has_chunks:
        ep = has_chunks[0]
        print(f"\n  Sample chunk sequence ({ep.get('episode_id')}):")
        for c in ep["chunk_sequence"][:5]:
            print(f"    [{c['chunk_idx']}] {c['start_ms']}-{c['end_ms']}ms "
                  f"action={c['action']} think={c['think'][:40]}...")

    passed = len(has_chunks) > 0 and total_chunks > 0
    print(f"\n  Overall → {_pass_fail(passed)}")
    return passed, {"episodes_with_chunks": len(has_chunks), "total_chunks": total_chunks}


# ===================================================================
# Stage 4: Query Verification Quality
# ===================================================================


def check_stage4(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 4: Query Verification Quality Check")
    print("=" * 60)

    if not EPISODE_VERIFIED_PATH.exists():
        print("  NO DATA")
        return False, {}

    episodes = read_jsonl(EPISODE_VERIFIED_PATH)
    recall_eps = [ep for ep in episodes if ep.get("need_recall")]
    print(f"\n  Total episodes: {len(episodes)}")
    print(f"  Recall episodes: {len(recall_eps)}")

    # Verification status
    status_counter = Counter(ep.get("query_verification", "unknown") for ep in recall_eps)
    print(f"\n  Query verification results:")
    for status, count in status_counter.most_common():
        print(f"    {status}: {count} ({_pct(count, len(recall_eps))})")

    # Hit rate
    hit_count = sum(1 for ep in recall_eps
                    if ep.get("query_verification") in ("direct_hit", "small_model_repair", "teacher_repair"))
    hit_rate = hit_count / max(len(recall_eps), 1)
    print(f"\n  Overall hit rate: {hit_rate:.1%} [threshold ≥ 70%] → {_pass_fail(hit_rate >= 0.7)}")

    # Coverage stats
    coverages = [ep.get("query_coverage", 0) for ep in recall_eps if ep.get("query_coverage")]
    if coverages:
        print(f"  Coverage: mean={sum(coverages)/len(coverages):.3f}, "
              f"min={min(coverages):.3f}, max={max(coverages):.3f}")

    passed = hit_rate >= 0.7
    print(f"\n  Overall → {_pass_fail(passed)}")
    return passed, {"recall_eps": len(recall_eps), "hit_rate": hit_rate}


# ===================================================================
# Stage 5: 6-Gate Verification Quality
# ===================================================================


def check_stage5(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 5: 6-Gate Verification Quality Check")
    print("=" * 60)

    if not EPISODE_FINAL_PATH.exists():
        print("  NO DATA")
        return False, {}

    episodes = read_jsonl(EPISODE_FINAL_PATH)
    recall_eps = [ep for ep in episodes if ep.get("need_recall") and ep.get("verification")]
    non_recall = [ep for ep in episodes if not ep.get("need_recall")]

    print(f"\n  Total episodes: {len(episodes)}")
    print(f"  Non-recall: {len(non_recall)}")
    print(f"  Recall with verification: {len(recall_eps)}")

    # Gate pass rates
    if recall_eps:
        gate_keys = [
            ("gate1_support_outside_recent", "Support outside recent"),
            ("gate2_retrieval_hit_at_3", "Retrieval hit@3"),
            ("gate3_support_coverage", "Support coverage ≥ 0.5"),
            ("gate4_no_recall_fail", "No-recall baseline fails"),
            ("gate5_with_recall_pass", "With-recall passes"),
            ("gate6_counterfactual_fail", "Counterfactual degrades"),
        ]
        print(f"\n  Gate pass rates:")
        for key, label in gate_keys:
            # gate3 is a float (coverage), others are bool
            if key == "gate3_support_coverage":
                passed_count = sum(1 for ep in recall_eps
                                   if ep["verification"].get(key, 0) >= GATE_THRESHOLDS["gate3_coverage_min"])
            else:
                passed_count = sum(1 for ep in recall_eps if ep["verification"].get(key, False))
            print(f"    {label}: {_pct(passed_count, len(recall_eps))}")

        all_passed = sum(1 for ep in recall_eps if ep["verification"].get("all_gates_passed"))
        pass_rate = all_passed / max(len(recall_eps), 1)
        print(f"\n  All gates passed: {all_passed}/{len(recall_eps)} ({pass_rate:.1%}) "
              f"[threshold ≥ 60%] → {_pass_fail(pass_rate >= 0.6)}")
    else:
        pass_rate = 0
        all_passed = 0

    passed = len(episodes) > 0
    print(f"\n  Overall → {_pass_fail(passed)}")
    return passed, {"recall_verified": len(recall_eps), "all_gates_passed": all_passed, "pass_rate": pass_rate}


# ===================================================================
# Stage 6: Final Assembly Quality
# ===================================================================


def check_stage6(verbose: bool = False) -> Tuple[bool, Dict]:
    print("\n" + "=" * 60)
    print("Stage 6: Final Assembly Quality Check")
    print("=" * 60)

    # SFT
    sft_files = list(SFT_DIR.glob("sft_*.jsonl"))
    sft_samples = []
    for f in sft_files:
        sft_samples.extend(read_jsonl(f))

    # Retriever
    retriever_samples = read_jsonl(RETRIEVER_TRAIN_PATH) if RETRIEVER_TRAIN_PATH.exists() else []

    # RL
    rl_samples = read_jsonl(RL_POOL_PATH) if RL_POOL_PATH.exists() else []

    print(f"\n  SFT samples:       {len(sft_samples)}")
    print(f"  Retriever samples: {len(retriever_samples)}")
    print(f"  RL samples:        {len(rl_samples)}")

    if not sft_samples:
        print("  NO SFT DATA")
        return False, {}

    # SFT sample type distribution
    type_counter = Counter(s.get("sample_type", "unknown") for s in sft_samples)
    print(f"\n  SFT sample types:")
    for st, count in type_counter.most_common():
        print(f"    {st}: {count} ({_pct(count, len(sft_samples))})")

    # Format compliance check
    format_ok = 0
    format_fail = 0
    action_counter = Counter()
    for s in sft_samples:
        for msg in s.get("messages", []):
            if msg["role"] == "assistant":
                content = msg["content"]
                if _AGENT_FORMAT_RE.match(content):
                    format_ok += 1
                else:
                    format_fail += 1
                # Count actions
                for act in ["silent", "response", "recall"]:
                    if f"<action>{act}</action>" in content:
                        action_counter[act] += 1

    total_assistant = format_ok + format_fail
    format_rate = format_ok / max(total_assistant, 1)
    print(f"\n  Format compliance: {format_rate:.1%} ({format_ok}/{total_assistant}) "
          f"[threshold ≥ 95%] → {_pass_fail(format_rate >= 0.95)}")

    # Action distribution
    total_actions = sum(action_counter.values())
    if total_actions > 0:
        print(f"\n  Action distribution in SFT:")
        for action in ["silent", "response", "recall"]:
            count = action_counter.get(action, 0)
            pct = count / total_actions
            lo, hi = ACTION_DIST_TARGET.get(action, (0, 1))
            status = "OK" if lo <= pct <= hi else "WARN"
            print(f"    {action}: {count} ({pct:.1%}) [{status}]")

    # RL answer types
    if rl_samples:
        rl_types = Counter(s.get("canonical_answer", {}).get("answer_type", "?") for s in rl_samples)
        print(f"\n  RL answer types:")
        for at, count in rl_types.most_common():
            print(f"    {at}: {count}")

    # Retriever hard negatives
    if retriever_samples:
        avg_neg = sum(len(s.get("negative_messages", [])) for s in retriever_samples) / len(retriever_samples)
        print(f"\n  Retriever avg hard negatives: {avg_neg:.1f}")

    if verbose and sft_samples:
        # Print one recall sample
        recall_samples = [s for s in sft_samples if s.get("sample_type") == "recall_positive"]
        if recall_samples:
            s = recall_samples[0]
            print(f"\n  Sample recall SFT ({s['id']}):")
            for msg in s["messages"]:
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"    [{msg['role']}] {content}")

    passed = len(sft_samples) > 0 and format_rate >= 0.95
    print(f"\n  Overall → {_pass_fail(passed)}")
    return passed, {
        "sft": len(sft_samples),
        "retriever": len(retriever_samples),
        "rl": len(rl_samples),
        "format_rate": format_rate,
    }


# ===================================================================
# Full check
# ===================================================================


_STAGE_CHECKERS = {
    0: check_stage0,
    1: check_stage1,
    2: check_stage2,
    3: check_stage3,
    4: check_stage4,
    5: check_stage5,
    6: check_stage6,
}


def check_all(verbose: bool = False) -> Dict[int, bool]:
    results = {}
    for stage, checker in _STAGE_CHECKERS.items():
        try:
            passed, _ = checker(verbose)
            results[stage] = passed
        except Exception as exc:
            logger.error("Stage %d check failed: %s", stage, exc)
            results[stage] = False

    print("\n" + "=" * 60)
    print("QUALITY CHECK SUMMARY")
    print("=" * 60)
    for stage, passed in results.items():
        status = _pass_fail(passed) if passed is not None else "SKIP"
        print(f"  Stage {stage}: {status}")
    print("=" * 60)
    return results


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Pipeline quality check")
    parser.add_argument("--stage", type=int, default=None, help="Check specific stage (0-6)")
    parser.add_argument("--all", action="store_true", help="Check all stages")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.stage is not None:
        if args.stage not in _STAGE_CHECKERS:
            print(f"Invalid stage {args.stage}. Valid: 0-6")
            sys.exit(1)
        passed, _ = _STAGE_CHECKERS[args.stage](args.verbose)
        sys.exit(0 if passed else 1)
    elif args.all:
        results = check_all(args.verbose)
        all_ok = all(v for v in results.values() if v is not None)
        sys.exit(0 if all_ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
