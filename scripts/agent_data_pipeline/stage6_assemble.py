"""
Stage 6: Final Sample Assembly

Assembles training data from verified episodes:
  1. sft_final.jsonl  — SFT training (recall-positive + no-recall control +
                         false-recall negative + protocol + easy QA)
  2. retriever_train.jsonl — Retriever/reranker training with hard negatives
  3. rl_pool.jsonl — RL training with reward specs

Usage:
    python -m scripts.agent_data_pipeline.stage6_assemble \
        [--version v0.1] \
        [--clip_videos]  # actually cut recent-buffer clips with ffmpeg
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from .config import (
    AGENT_SYSTEM_PROMPT_ZH,
    CLIP_DIR,
    DEFAULT_REWARD_SPEC,
    EPISODE_FINAL_PATH,
    HARD_NEGATIVE_RATIOS,
    RECENT_WINDOW_SEC,
    RETRIEVER_TRAIN_PATH,
    RL_POOL_PATH,
    SFT_DIR,
    VERIFIABLE_ANSWER_TYPES,
    ensure_dirs,
)
from .utils import (
    extract_video_clip,
    format_recall_result,
    load_segment_archive,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# SFT sample builders
# ===================================================================


def build_recall_sft(
    episode: Dict,
    segments: List[Dict],
    clip_videos: bool = False,
) -> Dict:
    """Build a recall-positive SFT sample (full recall interaction)."""
    video_id = episode["video_id"]
    ask_time_ms = episode["ask_time_ms"]
    seg_map = {s["segment_id"]: s for s in segments}

    # Prepare recent clip path
    recent_start = max(0, ask_time_ms / 1000 - RECENT_WINDOW_SEC)
    recent_end = ask_time_ms / 1000
    clip_path = str(CLIP_DIR / f"{video_id}_{int(recent_start)}_{int(recent_end)}.mp4")

    # Prepare recall keyframe paths
    retrieved_segs = [
        seg_map[sid] for sid in episode.get("gold_retrieved_segment_ids", [])
        if sid in seg_map
    ]
    recall_images = []
    for seg in retrieved_segs[:2]:
        for kf in seg.get("keyframe_paths", [])[:1]:
            recall_images.append(kf)

    # Build dialogue history text
    dialogue_text = ""
    if episode.get("dialogue_history"):
        lines = []
        for turn in episode["dialogue_history"]:
            role = "用户" if turn.get("role") == "user" else "助手"
            lines.append(f"{role}: {turn.get('text', '')}")
        dialogue_text = f"<dialogue_history>\n" + "\n".join(lines) + "\n</dialogue_history>\n"

    # Build user question message
    question_text = f"<question>\n{episode['question']}\n</question>"
    user_content = f"<video>\n{dialogue_text}{question_text}"

    # Build recall query
    gold_query = episode.get("gold_query", {})
    query_json = json.dumps(gold_query, ensure_ascii=False)

    # Build recall result
    recall_result_text = format_recall_result(retrieved_segs)

    # Get think texts from chunk_sequence
    chunks = episode.get("chunk_sequence", [])
    recall_think = "当前窗口内找不到答案，需要检索历史片段。"
    response_think = "检索到相关信息，可以回答。"
    for c in chunks:
        if c["action"] == "recall":
            recall_think = c["think"]
        elif c["action"] == "response":
            response_think = c["think"]

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_ZH},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": (
                f"<think>{recall_think}</think>"
                f"<action>recall</action>"
                f"<query>{query_json}</query>"
            ),
        },
        {
            "role": "user",
            "content": f"{recall_result_text}\n继续按协议回答。",
        },
        {
            "role": "assistant",
            "content": (
                f"<think>{response_think}</think>"
                f"<action>response</action>"
                f"<response>{episode.get('natural_response', '')}</response>"
            ),
        },
    ]

    return {
        "id": f"sft_recall_{uuid4().hex[:8]}",
        "messages": messages,
        "videos": [clip_path],
        "images": recall_images,
        "episode_id": episode["episode_id"],
        "sample_type": "recall_positive",
        "protocol_version": "3action",
    }


def build_no_recall_control(
    episode: Dict,
    segments: List[Dict],
) -> Dict:
    """Build a no-recall control: same question asked earlier when support
    is still in recent window, so recall is not needed."""
    video_id = episode["video_id"]

    # Move ask_time back so support is in recent window
    support_ids = episode.get("support_segment_ids", [])
    seg_map = {s["segment_id"]: s for s in segments}
    support_segs = [seg_map[sid] for sid in support_ids if sid in seg_map]

    if support_segs:
        # Place ask_time right after the latest support segment
        latest_support_end = max(s["end_ms"] for s in support_segs)
        early_ask_time = latest_support_end + 2000  # 2s after support
    else:
        early_ask_time = episode["ask_time_ms"] // 2

    recent_start = max(0, early_ask_time / 1000 - RECENT_WINDOW_SEC)
    recent_end = early_ask_time / 1000
    clip_path = str(CLIP_DIR / f"{video_id}_{int(recent_start)}_{int(recent_end)}.mp4")

    # Find think from chunk_sequence closest to early_ask
    think_text = "当前窗口内就有答案，直接回答。"
    for c in episode.get("chunk_sequence", []):
        if c["action"] == "response":
            think_text = c.get("think", think_text)
            break

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_ZH},
        {
            "role": "user",
            "content": f"<video>\n<question>\n{episode['question']}\n</question>",
        },
        {
            "role": "assistant",
            "content": (
                f"<think>{think_text}</think>"
                f"<action>response</action>"
                f"<response>{episode.get('natural_response', '')}</response>"
            ),
        },
    ]

    return {
        "id": f"sft_norecall_{uuid4().hex[:8]}",
        "messages": messages,
        "videos": [clip_path],
        "images": [],
        "episode_id": episode["episode_id"],
        "sample_type": "no_recall_control",
        "protocol_version": "3action",
    }


def build_false_recall_negative(
    episode: Dict,
    segments: List[Dict],
) -> Dict:
    """Build a false-recall negative: question looks like it needs recall
    (uses words like '之前'/'前面') but support is in recent window."""
    video_id = episode["video_id"]

    # Rewrite question to sound like a recall question
    question = episode["question"]
    if not any(kw in question for kw in ["之前", "前面", "刚才", "earlier", "before"]):
        prefixes = ["之前", "前面", "刚才"]
        question = f"{random.choice(prefixes)}{question}"

    # Same timing as no-recall control
    support_ids = episode.get("support_segment_ids", [])
    seg_map = {s["segment_id"]: s for s in segments}
    support_segs = [seg_map[sid] for sid in support_ids if sid in seg_map]

    if support_segs:
        latest_end = max(s["end_ms"] for s in support_segs)
        ask_time = latest_end + 2000
    else:
        ask_time = episode["ask_time_ms"] // 2

    recent_start = max(0, ask_time / 1000 - RECENT_WINDOW_SEC)
    recent_end = ask_time / 1000
    clip_path = str(CLIP_DIR / f"{video_id}_{int(recent_start)}_{int(recent_end)}.mp4")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_ZH},
        {
            "role": "user",
            "content": f"<video>\n<question>\n{question}\n</question>",
        },
        {
            "role": "assistant",
            "content": (
                "<think>虽然用户说了\"之前\"，但当前窗口内就能看到相关信息，不需要检索。</think>"
                "<action>response</action>"
                f"<response>{episode.get('natural_response', '')}</response>"
            ),
        },
    ]

    return {
        "id": f"sft_false_recall_{uuid4().hex[:8]}",
        "messages": messages,
        "videos": [clip_path],
        "images": [],
        "episode_id": episode["episode_id"],
        "sample_type": "false_recall_negative",
        "protocol_version": "3action",
    }


def build_simple_sft(
    episode: Dict,
    segments: List[Dict],
) -> Dict:
    """Build a simple SFT sample with multi-chunk streaming (silent + response).

    Unlike recall samples, this shows the full chunk-level streaming pattern:
    multiple silent chunks followed by the response chunk when the question
    arrives. This teaches the model the core streaming protocol.
    """
    video_id = episode["video_id"]
    ask_time_ms = episode["ask_time_ms"]

    recent_start = max(0, ask_time_ms / 1000 - RECENT_WINDOW_SEC)
    recent_end = ask_time_ms / 1000
    clip_path = str(CLIP_DIR / f"{video_id}_{int(recent_start)}_{int(recent_end)}.mp4")

    chunks = episode.get("chunk_sequence", [])
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT_ZH}]

    if chunks:
        # Build full chunk-level multi-turn: silent chunks + response chunk
        for c in chunks:
            # Each chunk = one user turn (video) + one assistant turn (action)
            user_parts = f"<video>"
            # Attach question text to the chunk where question arrives
            if c["action"] == "response":
                user_parts += f"\n<question>\n{episode['question']}\n</question>"

            messages.append({"role": "user", "content": user_parts})

            if c["action"] == "silent":
                messages.append({
                    "role": "assistant",
                    "content": f"<think>{c['think']}</think><action>silent</action>",
                })
            elif c["action"] == "response":
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"<think>{c['think']}</think>"
                        f"<action>response</action>"
                        f"<response>{episode.get('natural_response', '')}</response>"
                    ),
                })
                break  # Stop after response
    else:
        # Fallback: single-turn Q&A
        messages.append({
            "role": "user",
            "content": f"<video>\n<question>\n{episode['question']}\n</question>",
        })
        messages.append({
            "role": "assistant",
            "content": (
                "<think>当前画面有足够信息回答。</think>"
                "<action>response</action>"
                f"<response>{episode.get('natural_response', '')}</response>"
            ),
        })

    return {
        "id": f"sft_simple_{uuid4().hex[:8]}",
        "messages": messages,
        "videos": [clip_path],
        "images": [],
        "episode_id": episode["episode_id"],
        "sample_type": "easy_qa" if episode.get("difficulty") == "easy" else "protocol",
        "protocol_version": "3action",
    }


# ===================================================================
# Retriever training sample builder
# ===================================================================


def build_retriever_sample(
    episode: Dict,
    segments: List[Dict],
) -> Optional[Dict]:
    """Build a retriever training sample with hard negatives."""
    if not episode.get("need_recall"):
        return None

    gold_query = episode.get("gold_query", {})
    if not gold_query:
        return None

    seg_map = {s["segment_id"]: s for s in segments}
    gold_support_ids = set(episode.get("support_segment_ids", []))
    retrieved_ids = set(episode.get("gold_retrieved_segment_ids", []))

    # Build positive messages
    positive_segs = [seg_map[sid] for sid in gold_support_ids if sid in seg_map]
    if not positive_segs:
        return None

    positive_messages = []
    for seg in positive_segs[:2]:
        text = (
            f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}秒: "
            f"{seg.get('dense_caption', '')} "
            f"ASR: {seg.get('asr_text', '无')}. "
            f"OCR: {seg.get('ocr_text', '无')}."
        )
        positive_messages.append([{"role": "user", "content": text}])

    # Build hard negatives
    negative_messages = []

    # Type 1: Temporal near-miss (segments adjacent to gold support)
    for seg in segments:
        if seg["segment_id"] in gold_support_ids:
            continue
        for gold_seg in positive_segs:
            time_gap = abs(seg["start_ms"] - gold_seg["end_ms"])
            if time_gap <= 4000:  # Within 4s
                text = f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}秒: {seg.get('dense_caption', '')}"
                negative_messages.append([{"role": "user", "content": text}])
                break
        if len(negative_messages) >= 2:
            break

    # Type 2: Semantic confounder (same entities, different actions)
    gold_entities = set(e for s in positive_segs for e in s.get("entity_tags", []))
    for seg in segments:
        if seg["segment_id"] in gold_support_ids:
            continue
        seg_entities = set(seg.get("entity_tags", []))
        if seg_entities & gold_entities and seg["segment_id"] not in retrieved_ids:
            text = f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}秒: {seg.get('dense_caption', '')}"
            negative_messages.append([{"role": "user", "content": text}])
            if len(negative_messages) >= 3:
                break

    # Type 3: Random same-video (fill remaining)
    recent_start = max(0, episode["ask_time_ms"] - RECENT_WINDOW_SEC * 1000)
    for seg in random.sample(segments, min(len(segments), 5)):
        if (seg["segment_id"] not in gold_support_ids
                and seg["start_ms"] < recent_start
                and len(negative_messages) < 4):
            text = f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}秒: {seg.get('dense_caption', '')}"
            negative_messages.append([{"role": "user", "content": text}])

    if not negative_messages:
        return None

    return {
        "messages": [
            {"role": "system", "content": "检索最相关的历史视频片段，用于回答当前流视频中的回忆问题。"},
            {"role": "user", "content": gold_query.get("query", "")},
        ],
        "positive_messages": positive_messages,
        "negative_messages": negative_messages[:4],
        "episode_id": episode["episode_id"],
    }


# ===================================================================
# RL sample builder
# ===================================================================


def build_rl_sample(episode: Dict) -> Optional[Dict]:
    """Build an RL training sample with reward specification."""
    canonical = episode.get("canonical_answer", {})
    answer_type = canonical.get("answer_type", "span")

    # Only verifiable answer types
    if answer_type not in VERIFIABLE_ANSWER_TYPES:
        return None

    need_recall = episode.get("need_recall", False)

    reward_spec = {
        "need_recall": need_recall,
        "action_full_credit": "recall_then_response" if need_recall else "direct_response",
        "action_partial_credit": "direct_response_if_correct",
        "earliest_response_time_ms": episode.get("earliest_response_time_ms", 0),
        "latest_full_credit_time_ms": episode.get("earliest_response_time_ms", 0) + 4000,
        "retrieval_hit_metric": "hit@3",
        **DEFAULT_REWARD_SPEC,
    }

    return {
        "episode_id": episode["episode_id"],
        "video_id": episode["video_id"],
        "question": episode["question"],
        "ask_time_ms": episode["ask_time_ms"],
        "recent_window_sec": RECENT_WINDOW_SEC,
        "support_segment_ids": episode.get("support_segment_ids", []),
        "gold_query": episode.get("gold_query", {}),
        "canonical_answer": canonical,
        "reward_spec": reward_spec,
    }


# ===================================================================
# Main assembly
# ===================================================================


def assemble_all(
    version: str = "v0.1",
    clip_videos: bool = False,
) -> Dict[str, int]:
    """Assemble all final training data from verified episodes.

    Returns counts of samples created.
    """
    ensure_dirs()

    if not EPISODE_FINAL_PATH.exists():
        logger.error("No final episodes at %s", EPISODE_FINAL_PATH)
        return {}

    episodes = read_jsonl(EPISODE_FINAL_PATH)
    logger.info("Stage 6: Assembling from %d episodes", len(episodes))

    # Cache segments
    segment_cache: Dict[str, List[Dict]] = {}

    sft_samples = []
    retriever_samples = []
    rl_samples = []

    for ep in episodes:
        video_id = ep["video_id"]
        if video_id not in segment_cache:
            segment_cache[video_id] = load_segment_archive(video_id)
        segments = segment_cache[video_id]

        verification = ep.get("verification", {})

        if ep.get("need_recall") and verification.get("all_gates_passed", False):
            # Recall-positive: generate 3 bound samples
            sft_samples.append(build_recall_sft(ep, segments, clip_videos))
            sft_samples.append(build_no_recall_control(ep, segments))
            sft_samples.append(build_false_recall_negative(ep, segments))

            # Retriever training
            rt = build_retriever_sample(ep, segments)
            if rt:
                retriever_samples.append(rt)
        else:
            # Non-recall or failed recall: simple SFT
            sft_samples.append(build_simple_sft(ep, segments))

        # RL samples (from all verified episodes with verifiable answers)
        rl = build_rl_sample(ep)
        if rl:
            rl_samples.append(rl)

    # Write outputs
    sft_path = SFT_DIR / f"sft_{version}.jsonl"
    write_jsonl(sft_samples, sft_path)
    write_jsonl(retriever_samples, RETRIEVER_TRAIN_PATH)
    write_jsonl(rl_samples, RL_POOL_PATH)

    stats = {
        "sft_total": len(sft_samples),
        "sft_recall_positive": sum(1 for s in sft_samples if s["sample_type"] == "recall_positive"),
        "sft_no_recall_control": sum(1 for s in sft_samples if s["sample_type"] == "no_recall_control"),
        "sft_false_recall_negative": sum(1 for s in sft_samples if s["sample_type"] == "false_recall_negative"),
        "sft_easy_qa": sum(1 for s in sft_samples if s["sample_type"] == "easy_qa"),
        "sft_protocol": sum(1 for s in sft_samples if s["sample_type"] == "protocol"),
        "retriever": len(retriever_samples),
        "rl": len(rl_samples),
    }

    logger.info("Stage 6 results: %s", stats)

    # Print action distribution in SFT
    action_counts = {"silent": 0, "response": 0, "recall": 0}
    for s in sft_samples:
        for msg in s.get("messages", []):
            if msg["role"] == "assistant":
                content = msg["content"]
                if "<action>recall</action>" in content:
                    action_counts["recall"] += 1
                elif "<action>response</action>" in content:
                    action_counts["response"] += 1
                elif "<action>silent</action>" in content:
                    action_counts["silent"] += 1
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        logger.info("Action distribution in SFT:")
        for action, count in sorted(action_counts.items()):
            logger.info("  %s: %d (%.1f%%)", action, count, 100.0 * count / total_actions)

    return stats


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Sample assembly")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--clip_videos", action="store_true",
                        help="Actually cut video clips with ffmpeg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = assemble_all(version=args.version, clip_videos=args.clip_videos)

    print(f"\nStage 6 Summary:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
