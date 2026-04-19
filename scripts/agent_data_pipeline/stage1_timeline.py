"""
Stage 1: Structured Event Timeline Generation

Aggregates segment archives into event timelines with:
- Segment-to-event grouping (scene, entity overlap, semantic similarity)
- Event type classification
- Causal link inference
- Importance scoring

Usage:
    python -m scripts.agent_data_pipeline.stage1_timeline \
        --video_ids vid_001,vid_002 \
        [--all]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (
    EVENT_TIMELINE_DIR,
    EVENT_TYPES,
    SEGMENT_ARCHIVE_DIR,
    ensure_dirs,
)
from .utils import (
    cosine_similarity_np,
    load_embedding,
    load_segment_archive,
    make_event_id,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Event type classification
# ===================================================================


def classify_event_type(segments: List[Dict]) -> str:
    """Classify event type based on segment content."""
    has_asr = any(s.get("has_asr") for s in segments)
    has_ocr = any(s.get("has_ocr") for s in segments)

    # Scene transition: segments span different scenes
    scene_ids = set(s.get("scene_id", "") for s in segments)
    if len(scene_ids) > 1:
        return "scene_transition"

    # OCR event
    if has_ocr and not has_asr:
        return "ocr_event"

    # Dialogue
    if has_asr:
        return "dialogue"

    # Check for procedural actions
    all_actions = [a for s in segments for a in s.get("action_tags", [])]
    procedural_kw = ["加入", "倒入", "切", "搅拌", "打开", "关闭", "移动",
                     "拿起", "放下", "安装", "拆卸", "按", "点击"]
    if any(kw in a for a in all_actions for kw in procedural_kw):
        return "procedure_step"

    # State change: entity tags differ between first and last segment
    if len(segments) >= 2:
        first_states = set(segments[0].get("state_tags", []))
        last_states = set(segments[-1].get("state_tags", []))
        if first_states != last_states and (first_states or last_states):
            return "state_change"

    return "entity_action"


# ===================================================================
# Caption merging
# ===================================================================


def merge_captions(captions: List[str]) -> str:
    """Merge multiple segment captions into an event summary."""
    valid = [c for c in captions if c]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]
    # Concatenate unique captions, avoid duplicates
    seen = set()
    merged = []
    for c in valid:
        c_stripped = c.strip().rstrip("。").rstrip(".")
        if c_stripped not in seen:
            seen.add(c_stripped)
            merged.append(c_stripped)
    return "；".join(merged[:3]) + "。"


# ===================================================================
# Event aggregation
# ===================================================================


def aggregate_event(
    segments: List[Dict],
    video_id: str,
    seq: int,
) -> Dict:
    """Aggregate multiple segments into a single event."""
    all_entities = list(set(e for s in segments for e in s.get("entity_tags", [])))
    all_captions = [s.get("dense_caption", "") for s in segments]

    # Gather evidence
    visual_evidence = next(
        (s["dense_caption"] for s in segments if s.get("dense_caption")), ""
    )
    asr_evidence = next(
        (s["asr_text"] for s in segments if s.get("has_asr")), ""
    )
    ocr_evidence = next(
        (s["ocr_text"] for s in segments if s.get("has_ocr")), ""
    )

    return {
        "video_id": video_id,
        "event_id": make_event_id(video_id, seq),
        "start_ms": segments[0]["start_ms"],
        "end_ms": segments[-1]["end_ms"],
        "support_segment_ids": [s["segment_id"] for s in segments],
        "event_type": classify_event_type(segments),
        "summary": merge_captions(all_captions),
        "entities": all_entities[:10],
        "preconditions": [],  # Can be filled by small VL model later
        "effects": [],
        "evidence": {
            "visual": visual_evidence,
            "asr": asr_evidence,
            "ocr": ocr_evidence,
        },
        "causal_links_prev": [],
        "causal_links_next": [],
        "importance": 0.0,  # Computed after causal links
    }


def should_split_event(
    prev_seg: Dict,
    curr_seg: Dict,
    current_group_size: int,
    similarity_threshold: float = 0.7,
) -> bool:
    """Decide whether to start a new event at curr_seg."""
    # Condition 1: Scene change
    if curr_seg.get("scene_id") != prev_seg.get("scene_id"):
        return True

    # Condition 2: No entity overlap
    prev_entities = set(prev_seg.get("entity_tags", []))
    curr_entities = set(curr_seg.get("entity_tags", []))
    if prev_entities and curr_entities and not (prev_entities & curr_entities):
        return True

    # Condition 3: Text embedding semantic distance
    prev_emb_path = prev_seg.get("text_emb_path", "")
    curr_emb_path = curr_seg.get("text_emb_path", "")
    if prev_emb_path and curr_emb_path:
        try:
            prev_emb = load_embedding(prev_emb_path)
            curr_emb = load_embedding(curr_emb_path)
            sim = cosine_similarity_np(prev_emb, curr_emb)
            if sim < similarity_threshold:
                return True
        except Exception:
            pass

    # Condition 4: Event already has 3+ segments (~12s)
    if current_group_size >= 3:
        return True

    return False


# ===================================================================
# Causal link inference
# ===================================================================


def infer_causal_links(
    events: List[Dict],
    time_gap_threshold_ms: int = 8000,
) -> None:
    """Infer causal links between events based on temporal proximity
    and entity co-occurrence. Modifies events in place."""
    for i, evt_a in enumerate(events):
        for j in range(i + 1, len(events)):
            evt_b = events[j]

            # Condition 1: Temporal proximity (B starts within 8s of A ending)
            gap = evt_b["start_ms"] - evt_a["end_ms"]
            if gap > time_gap_threshold_ms:
                break  # Events are sorted by time, no need to check further

            # Condition 2: Entity co-occurrence
            shared_entities = set(evt_a["entities"]) & set(evt_b["entities"])
            if not shared_entities:
                continue

            # Condition 3: Effect-precondition match (simple string containment)
            effect_match = any(
                eff.lower() in pre.lower()
                for eff in evt_a.get("effects", [])
                for pre in evt_b.get("preconditions", [])
            )

            if shared_entities or effect_match:
                evt_a["causal_links_next"].append(evt_b["event_id"])
                evt_b["causal_links_prev"].append(evt_a["event_id"])


# ===================================================================
# Importance scoring
# ===================================================================


def compute_importance(event: Dict, all_segments: List[Dict], max_event_duration_ms: int) -> float:
    """Compute event importance score [0, 1]."""
    duration = event["end_ms"] - event["start_ms"]
    duration_norm = min(1.0, duration / max(max_event_duration_ms, 1))

    entity_score = min(1.0, len(event["entities"]) / 5.0)

    causal_depth = len(event["causal_links_prev"]) + len(event["causal_links_next"])
    causal_score = min(1.0, causal_depth / 5.0)

    evidence = event.get("evidence", {})
    has_asr_or_ocr = 1.0 if (evidence.get("asr") or evidence.get("ocr")) else 0.0

    # Average salience of support segments
    support_ids = set(event["support_segment_ids"])
    support_segs = [s for s in all_segments if s["segment_id"] in support_ids]
    avg_salience = (
        sum(s.get("salience", 0) for s in support_segs) / max(len(support_segs), 1)
    )

    return (
        0.25 * duration_norm
        + 0.25 * entity_score
        + 0.25 * causal_score
        + 0.15 * has_asr_or_ocr
        + 0.10 * avg_salience
    )


# ===================================================================
# Main: build timeline for a single video
# ===================================================================


def build_event_timeline(video_id: str) -> List[Dict]:
    """Build structured event timeline from segment archive."""
    segments = load_segment_archive(video_id)
    if not segments:
        logger.warning("No segments found for %s", video_id)
        return []

    # Sort by start time
    segments.sort(key=lambda s: s["start_ms"])

    # Group segments into events
    events = []
    current_group = [segments[0]]

    for i in range(1, len(segments)):
        if should_split_event(segments[i - 1], segments[i], len(current_group)):
            events.append(aggregate_event(current_group, video_id, len(events)))
            current_group = [segments[i]]
        else:
            current_group.append(segments[i])

    if current_group:
        events.append(aggregate_event(current_group, video_id, len(events)))

    # Infer causal links
    infer_causal_links(events)

    # Compute importance
    max_duration = max((e["end_ms"] - e["start_ms"]) for e in events) if events else 1
    for evt in events:
        evt["importance"] = compute_importance(evt, segments, max_duration)

    # Save
    output_path = EVENT_TIMELINE_DIR / f"{video_id}.jsonl"
    write_jsonl(events, output_path)

    logger.info("Stage 1: %s → %d events", video_id, len(events))
    return events


# ===================================================================
# Batch processing
# ===================================================================


def process_batch(video_ids: Optional[List[str]] = None) -> Dict[str, int]:
    """Process multiple videos through Stage 1.

    If video_ids is None, process all videos with segment archives.
    """
    ensure_dirs()

    if video_ids is None:
        # Find all segment archives
        archives = list(SEGMENT_ARCHIVE_DIR.glob("*.jsonl"))
        video_ids = [p.stem for p in archives]

    results = {}
    for vid in video_ids:
        # Skip if already done
        output_path = EVENT_TIMELINE_DIR / f"{vid}.jsonl"
        if output_path.exists():
            logger.info("Skipping %s (timeline exists)", vid)
            continue

        try:
            events = build_event_timeline(vid)
            results[vid] = len(events)
        except Exception as exc:
            logger.error("Failed to build timeline for %s: %s", vid, exc, exc_info=True)
            results[vid] = -1

    return results


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Event timeline generation")
    parser.add_argument("--video_ids", default=None, help="Comma-separated video IDs")
    parser.add_argument("--all", action="store_true", help="Process all available videos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video_ids = None
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",")]
    elif not args.all:
        parser.error("Specify --video_ids or --all")

    results = process_batch(video_ids)

    print(f"\nStage 1 Summary: {len(results)} videos")
    valid = {k: v for k, v in results.items() if v >= 0}
    if valid:
        counts = list(valid.values())
        print(f"  Events: total={sum(counts)}, mean={sum(counts)/len(counts):.1f}")
    failed = [k for k, v in results.items() if v < 0]
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
