"""
Stage 3: Sparse → Dense Think Expansion

Expands 397B-generated sparse_think_milestones into chunk-level action
sequences. Each episode gets a full sequence of (chunk_idx, action, think)
covering the recent window → ask_time → response_time range.

Usage:
    python -m scripts.agent_data_pipeline.stage3_expand
"""

import argparse
import logging
import random
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EPISODE_DENSE_PATH,
    EPISODE_RAW_PATH,
    RECENT_WINDOW_SEC,
    ensure_dirs,
)
from .utils import (
    find_nearest_segment,
    load_segment_archive,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Incremental think generation (rule-based templates)
# ===================================================================

_SILENT_TEMPLATES_ZH = [
    "画面中{entity}在{action}，暂无需回答。",
    "当前观察到{caption}，继续监控。",
    "没有新问题，保持关注当前场景。",
    "场景正常进行中，无需回应。",
    "继续关注画面变化。",
]


def generate_silent_think(segment: Optional[Dict]) -> str:
    """Generate a brief incremental think for a silent chunk."""
    if segment is None:
        return random.choice(_SILENT_TEMPLATES_ZH[-2:])

    entity = segment.get("entity_tags", ["主体"])[0] if segment.get("entity_tags") else "主体"
    action = segment.get("action_tags", ["活动"])[0] if segment.get("action_tags") else "活动"
    caption = segment.get("dense_caption", "")[:20]

    template = random.choice(_SILENT_TEMPLATES_ZH)
    return template.format(entity=entity, action=action, caption=caption)


# ===================================================================
# Core expansion logic
# ===================================================================


def expand_episode(
    episode: Dict,
    segments: List[Dict],
    chunk_sec: float = AGENT_CHUNK_SEC,
) -> Dict:
    """Expand a single episode's sparse milestones into chunk-level actions.

    The chunk sequence covers:
        [recent_window_start, ask_time + 4s buffer]

    Each chunk is assigned an action (silent / recall / response) and
    a think text.
    """
    ask_time_ms = episode["ask_time_ms"]
    response_time_ms = episode.get("earliest_response_time_ms", ask_time_ms)
    need_recall = episode.get("need_recall", False)
    recent_start_ms = max(0, ask_time_ms - RECENT_WINDOW_SEC * 1000)

    # Determine chunk range
    chunk_start_ms = recent_start_ms
    chunk_end_ms = response_time_ms + int(4000)  # 2 extra chunks buffer
    chunk_ms = int(chunk_sec * 1000)
    num_chunks = max(1, int((chunk_end_ms - chunk_start_ms) / chunk_ms))

    # Build milestone lookup: time_ms → text
    milestones = {}
    for m in episode.get("sparse_think_milestones", []):
        milestones[m["time_ms"]] = m["text"]

    # Assign actions to each chunk
    chunks = []
    recall_placed = False
    response_placed = False

    for i in range(num_chunks):
        t_start = chunk_start_ms + i * chunk_ms
        t_end = t_start + chunk_ms
        t_mid = (t_start + t_end) / 2

        # Find milestone in this chunk
        chunk_milestone = None
        for mt, mtext in milestones.items():
            if t_start <= mt < t_end:
                chunk_milestone = mtext
                break

        # Determine action
        # Priority: recall at question time, response at earliest_response_time
        is_question_chunk = (t_start <= ask_time_ms < t_end) or abs(t_mid - ask_time_ms) < chunk_ms
        is_response_chunk = (t_start <= response_time_ms < t_end) or abs(t_mid - response_time_ms) < chunk_ms

        if need_recall and is_question_chunk and not recall_placed:
            action = "recall"
            think = chunk_milestone or "用户提问，当前窗口内找不到答案，需要检索历史片段。"
            recall_placed = True
        elif is_response_chunk and not response_placed:
            # If recall was placed, this is the post-recall response
            # If no recall needed, this is the direct response
            if need_recall and not recall_placed:
                # Recall should come before response — place recall first
                action = "recall"
                think = chunk_milestone or "需要检索历史信息才能回答。"
                recall_placed = True
            else:
                action = "response"
                think = chunk_milestone or "证据充足，可以回答。"
                response_placed = True
        else:
            action = "silent"
            if chunk_milestone:
                think = chunk_milestone
            else:
                nearby_seg = find_nearest_segment(segments, t_mid)
                think = generate_silent_think(nearby_seg)

        chunks.append({
            "chunk_idx": i,
            "start_ms": t_start,
            "end_ms": t_end,
            "action": action,
            "think": think,
        })

    # If recall was placed but response wasn't, add response in next chunk
    if need_recall and recall_placed and not response_placed:
        recall_idx = next(i for i, c in enumerate(chunks) if c["action"] == "recall")
        # Place response in the chunk right after recall
        if recall_idx + 1 < len(chunks):
            chunks[recall_idx + 1]["action"] = "response"
            chunks[recall_idx + 1]["think"] = "检索到相关历史信息，可以回答。"
        else:
            # Extend with one more chunk
            last_end = chunks[-1]["end_ms"]
            chunks.append({
                "chunk_idx": len(chunks),
                "start_ms": last_end,
                "end_ms": last_end + chunk_ms,
                "action": "response",
                "think": "检索到相关历史信息，可以回答。",
            })

    # If not a recall episode but response hasn't been placed
    if not need_recall and not response_placed and chunks:
        # Find the chunk closest to ask_time and make it the response
        ask_chunk_idx = min(
            range(len(chunks)),
            key=lambda i: abs((chunks[i]["start_ms"] + chunks[i]["end_ms"]) / 2 - ask_time_ms)
        )
        chunks[ask_chunk_idx]["action"] = "response"
        chunks[ask_chunk_idx]["think"] = "当前画面有足够信息回答。"

    episode["chunk_sequence"] = chunks
    return episode


# ===================================================================
# Batch processing
# ===================================================================


def expand_all_episodes() -> int:
    """Read episode_recall_raw.jsonl, expand each, write to episode_recall_dense.jsonl."""
    ensure_dirs()

    if not EPISODE_RAW_PATH.exists():
        logger.error("No raw episodes found at %s", EPISODE_RAW_PATH)
        return 0

    episodes = read_jsonl(EPISODE_RAW_PATH)
    logger.info("Stage 3: Expanding %d episodes", len(episodes))

    # Cache segment archives per video
    segment_cache: Dict[str, List[Dict]] = {}
    expanded = []

    for ep in episodes:
        video_id = ep["video_id"]
        if video_id not in segment_cache:
            segment_cache[video_id] = load_segment_archive(video_id)

        try:
            ep = expand_episode(ep, segment_cache[video_id])
            expanded.append(ep)
        except Exception as exc:
            logger.warning("Failed to expand %s: %s", ep.get("episode_id"), exc)
            expanded.append(ep)  # Keep without chunk_sequence

    write_jsonl(expanded, EPISODE_DENSE_PATH)

    # Print distribution stats
    total_chunks = sum(len(ep.get("chunk_sequence", [])) for ep in expanded)
    action_counts = {"silent": 0, "response": 0, "recall": 0}
    for ep in expanded:
        for chunk in ep.get("chunk_sequence", []):
            action_counts[chunk["action"]] = action_counts.get(chunk["action"], 0) + 1

    logger.info("Stage 3: %d episodes → %d chunks", len(expanded), total_chunks)
    if total_chunks > 0:
        for action, count in sorted(action_counts.items()):
            logger.info("  %s: %d (%.1f%%)", action, count, 100.0 * count / total_chunks)

    return len(expanded)


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Sparse→Dense think expansion")
    parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    count = expand_all_episodes()
    print(f"\nStage 3: Expanded {count} episodes → {EPISODE_DENSE_PATH}")


if __name__ == "__main__":
    main()
