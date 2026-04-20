"""
Shared utility functions for the Agent Data Construction Pipeline.

Covers: video I/O, embedding helpers, scoring, segment lookup,
temporal overlap computation, answer evaluation, and JSONL I/O.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import (
    KEYFRAMES_PER_SEGMENT,
    PROCEDURAL_KEYWORDS_ZH,
    RECALLABILITY_WEIGHTS,
    SALIENCE_WEIGHTS,
    SEGMENT_ARCHIVE_DIR,
)

logger = logging.getLogger(__name__)

# ===================================================================
# JSONL I/O
# ===================================================================


def read_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ===================================================================
# Video frame extraction (torchcodec)
# ===================================================================


def extract_frames_for_segment(
    video_path: str,
    start_ms: int,
    end_ms: int,
    fps: int = 2,
) -> List[torch.Tensor]:
    """Extract frames from a video segment at given fps using torchcodec."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path)
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    duration = end_sec - start_sec
    n_frames = max(1, int(duration * fps))
    timestamps = [start_sec + i / fps for i in range(n_frames)]
    frames = []
    for ts in timestamps:
        try:
            frame = decoder.get_frame_at(ts)
            frames.append(frame.data)
        except Exception:
            logger.warning("Failed to decode frame at %.2fs in %s", ts, video_path)
    return frames


def select_keyframes(
    frames: List[torch.Tensor], n: int = KEYFRAMES_PER_SEGMENT
) -> List[int]:
    """Select representative keyframes: middle + highest-motion frame."""
    if len(frames) <= n:
        return list(range(len(frames)))
    mid = len(frames) // 2
    # Simple motion detection via frame differences
    diffs = []
    for i in range(len(frames) - 1):
        diff = torch.abs(frames[i + 1].float() - frames[i].float()).mean().item()
        diffs.append(diff)
    max_motion_idx = max(range(len(diffs)), key=lambda i: diffs[i])
    return sorted(set([mid, max_motion_idx]))[:n]


def save_keyframe(frame: torch.Tensor, path: Path) -> None:
    """Save a tensor frame as JPEG."""
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    # frame shape: (C, H, W), values 0-255 uint8 → normalize to [0,1] float
    img = frame.float() / 255.0
    save_image(img, str(path))


# ===================================================================
# Motion score
# ===================================================================


def compute_motion_score(frames: List[torch.Tensor]) -> float:
    """Simple motion score from mean frame differences, normalized to [0,1]."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(len(frames) - 1):
        diff = torch.abs(frames[i + 1].float() - frames[i].float()).mean().item()
        diffs.append(diff)
    raw = sum(diffs) / len(diffs)
    # Normalize: empirically ~0-50 range → clamp to [0,1]
    return min(1.0, raw / 50.0)


# ===================================================================
# Salience score
# ===================================================================


def compute_salience(
    motion_score: float,
    entity_tags: List[str],
    action_tags: List[str],
    has_ocr: bool,
    has_asr: bool,
    is_scene_boundary: bool,
    max_entities: int = 5,
    max_actions: int = 3,
) -> float:
    w = SALIENCE_WEIGHTS
    entity_density = min(1.0, len(entity_tags) / max(max_entities, 1))
    action_density = min(1.0, len(action_tags) / max(max_actions, 1))
    has_ocr_or_asr = 1.0 if (has_ocr or has_asr) else 0.0
    scene_boundary = 1.0 if is_scene_boundary else 0.0

    return (
        w["motion"] * motion_score
        + w["entity_density"] * entity_density
        + w["has_ocr_or_asr"] * has_ocr_or_asr
        + w["action_density"] * action_density
        + w["scene_boundary"] * scene_boundary
    )


# ===================================================================
# Recallability score (video-level)
# ===================================================================


def compute_recallability(
    segments: List[Dict],
    recent_window_sec: int = 24,
) -> float:
    if not segments:
        return 0.0

    duration_sec = segments[-1]["end_ms"] / 1000.0
    w = RECALLABILITY_WEIGHTS

    # Duration score
    duration_score = min(1.0, duration_sec / 120.0)

    # Entity recurrence: fraction of entities appearing in ≥3 segments
    all_entities: Dict[str, int] = {}
    for seg in segments:
        for e in seg.get("entity_tags", []):
            all_entities[e] = all_entities.get(e, 0) + 1
    total_entities = max(len(all_entities), 1)
    recurring = sum(1 for c in all_entities.values() if c >= 3)
    entity_recurrence = recurring / total_entities

    # Event chain depth proxy
    event_chain_depth = min(1.0, len(segments) / 30.0)

    # Procedurality
    proc_count = sum(
        1
        for seg in segments
        if any(kw in a for a in seg.get("action_tags", []) for kw in PROCEDURAL_KEYWORDS_ZH)
    )
    procedurality = proc_count / max(len(segments), 1)

    # OCR / ASR density
    ocr_density = sum(1 for s in segments if s.get("has_ocr")) / max(len(segments), 1)
    asr_density = sum(1 for s in segments if s.get("has_asr")) / max(len(segments), 1)

    # Cut rate
    scene_ids = [s.get("scene_id", "") for s in segments]
    cuts = sum(1 for i in range(1, len(scene_ids)) if scene_ids[i] != scene_ids[i - 1])
    cut_rate = cuts / max(len(segments), 1)

    score = (
        w["duration"] * duration_score
        + w["entity_recurrence"] * entity_recurrence
        + w["event_chain_depth"] * event_chain_depth
        + w["procedurality"] * procedurality
        + w["ocr_density"] * ocr_density
        + w["asr_density"] * asr_density
        + w["cut_rate"] * cut_rate  # negative weight
    )
    return max(0.0, min(1.0, score))


# ===================================================================
# Temporal overlap computation
# ===================================================================


def temporal_overlap(
    spans_a: List[Tuple[int, int]], spans_b: List[Tuple[int, int]]
) -> float:
    """Compute max temporal overlap ratio: intersection / union of spans."""
    if not spans_a or not spans_b:
        return 0.0

    def merge(spans):
        spans = sorted(spans)
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    merged_a = merge(spans_a)
    merged_b = merge(spans_b)

    intersection = 0
    for sa, ea in merged_a:
        for sb, eb in merged_b:
            overlap = max(0, min(ea, eb) - max(sa, sb))
            intersection += overlap

    total_a = sum(e - s for s, e in merged_a)
    total_b = sum(e - s for s, e in merged_b)
    union = total_a + total_b - intersection

    return intersection / max(union, 1)


def spans_overlap_any(
    recent_start_ms: int,
    recent_end_ms: int,
    support_spans: List[Tuple[int, int]],
) -> bool:
    """Check if any support span overlaps with the recent window."""
    for s, e in support_spans:
        if max(recent_start_ms, s) < min(recent_end_ms, e):
            return True
    return False


# ===================================================================
# Segment lookup helpers
# ===================================================================


def load_segment_archive(video_id: str) -> List[Dict]:
    path = SEGMENT_ARCHIVE_DIR / f"{video_id}.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def find_nearest_segment(segments: List[Dict], time_ms: float) -> Optional[Dict]:
    """Find the segment whose midpoint is closest to time_ms."""
    if not segments:
        return None
    return min(segments, key=lambda s: abs((s["start_ms"] + s["end_ms"]) / 2 - time_ms))


def get_recent_segment_ids(
    segments: List[Dict], ask_time_ms: int, recent_window_sec: int = 24
) -> List[str]:
    """Return segment IDs within the recent window at ask_time_ms."""
    recent_start_ms = max(0, ask_time_ms - recent_window_sec * 1000)
    return [
        s["segment_id"]
        for s in segments
        if s["end_ms"] > recent_start_ms and s["start_ms"] < ask_time_ms
    ]


# ===================================================================
# Tag extraction (jieba-based, lightweight NLP)
# ===================================================================


def extract_tags_from_caption(
    dense_caption: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Extract entity, action, and state tags from a dense caption.

    Falls back to simple regex if jieba is not available.
    """
    try:
        import jieba.posseg as pseg

        words = list(pseg.cut(dense_caption))
        entities = [w.word for w in words if w.flag in ("nr", "ns", "nz", "n") and len(w.word) > 1]
        actions = [w.word for w in words if w.flag == "v" and len(w.word) > 1]
        states = [w.word for w in words if w.flag in ("a", "ad") and len(w.word) > 1]
        return entities[:5], actions[:3], states[:3]
    except ImportError:
        # Fallback: simple keyword extraction
        logger.warning("jieba not installed, using fallback tag extraction")
        words = dense_caption.split()
        return words[:5], words[:3], words[:3]


def generate_memory_keys(
    entity_tags: List[str],
    action_tags: List[str],
    ocr_text: str,
    dense_caption: str,
) -> List[str]:
    """Generate searchable memory keys for a segment."""
    keys = []
    # Rule 1: entity + action combinations
    for entity in entity_tags[:3]:
        for action in action_tags[:2]:
            keys.append(f"{entity} {action}")
    # Rule 2: OCR keywords
    if ocr_text and ocr_text != "无":
        keys.append(ocr_text[:20])
    # Rule 3: caption keywords (first few meaningful words)
    try:
        import jieba

        kw = list(jieba.cut(dense_caption))
        kw = [w for w in kw if len(w) > 1][:3]
        keys.extend(kw)
    except ImportError:
        keys.extend(dense_caption.split()[:3])

    # Deduplicate and limit
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:5]


# ===================================================================
# Answer scoring helpers
# ===================================================================


def compute_answer_score(predicted: str, canonical: Dict) -> float:
    """Compute match score between predicted text and canonical answer."""
    answer_type = canonical.get("answer_type", "span")
    value = canonical.get("value", {})

    predicted_lower = predicted.lower().strip()

    if answer_type == "yesno":
        gold = str(value.get("answer", "")).lower()
        if gold in predicted_lower:
            return 1.0
        return 0.0

    if answer_type == "number":
        gold_num = str(value.get("answer", ""))
        if gold_num in predicted:
            return 1.0
        return 0.0

    if answer_type == "multiple_choice":
        gold_choice = str(value.get("answer", "")).upper()
        if gold_choice in predicted.upper():
            return 1.0
        return 0.0

    if answer_type in ("slot", "entity"):
        # Check if all slot values appear in predicted
        matches = 0
        total = max(len(value), 1)
        for k, v in value.items():
            if str(v).lower() in predicted_lower:
                matches += 1
        return matches / total

    if answer_type == "ordered_steps":
        steps = value.get("steps", [])
        if not steps:
            return 0.0
        found = sum(1 for s in steps if s.lower() in predicted_lower)
        return found / len(steps)

    # Default: span / open-ended → simple F1
    return _compute_f1(predicted_lower, str(value).lower())


def _compute_f1(predicted: str, gold: str) -> float:
    pred_tokens = set(predicted.split())
    gold_tokens = set(gold.split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ===================================================================
# Embedding helpers
# ===================================================================


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_embedding(path: str) -> np.ndarray:
    return np.load(path)


def save_embedding(emb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), emb)


# ===================================================================
# Video clip extraction (ffmpeg)
# ===================================================================


def extract_video_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    """Extract a video clip using ffmpeg (stream copy for speed)."""
    import subprocess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", str(video_path),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ===================================================================
# Recall result formatting
# ===================================================================


def format_recall_result(
    retrieved_segments: List[Dict],
    max_items: int = 2,
) -> str:
    """Format retrieved segments into <recall_result> XML for training data."""
    lines = ["<recall_result>"]
    for i, seg in enumerate(retrieved_segments[:max_items]):
        start_s = seg["start_ms"] / 1000.0
        end_s = seg["end_ms"] / 1000.0
        caption = seg.get("dense_caption", "")
        asr = seg.get("asr_text", "无")
        ocr = seg.get("ocr_text", "无")
        lines.append(
            f'<item rank="{i+1}" start="{start_s:.1f}" end="{end_s:.1f}">'
            f"caption: {caption} asr: {asr} ocr: {ocr}</item>"
        )
    lines.append("</recall_result>")
    return "\n".join(lines)


# ===================================================================
# Video type inference
# ===================================================================


def infer_video_type(
    event_timeline: List[Dict],
    segments: Optional[List[Dict]] = None,
) -> str:
    """Heuristic inference of video type from event/segment content."""
    all_entities = set()
    all_actions = set()
    ocr_count = 0
    proc_count = 0

    for evt in event_timeline:
        all_entities.update(evt.get("entities", []))
        evidence = evt.get("evidence", {})
        if evidence.get("ocr"):
            ocr_count += 1
        summary = evt.get("summary", "")
        for kw in PROCEDURAL_KEYWORDS_ZH:
            if kw in summary:
                proc_count += 1
                break

    n = max(len(event_timeline), 1)

    if ocr_count / n > 0.3:
        return "screen_record"
    if proc_count / n > 0.4:
        return "tutorial"
    if any(kw in str(all_entities) for kw in ["锅", "食材", "调料", "刀"]):
        return "cooking"
    if len(event_timeline) > 10:
        return "vlog"
    return "other"


# ===================================================================
# Scene detection wrapper
# ===================================================================


def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect scene boundaries using PySceneDetect.

    Returns list of (start_sec, end_sec) tuples.
    Falls back to single-scene if scenedetect is not installed.
    """
    try:
        from scenedetect import detect, ContentDetector

        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        return [
            (s.get_seconds(), e.get_seconds())
            for s, e in scene_list
        ]
    except ImportError:
        logger.warning("scenedetect not installed, treating entire video as one scene")
        return []
    except Exception as exc:
        logger.warning("Scene detection failed for %s: %s", video_path, exc)
        return []


# ===================================================================
# Video duration helper
# ===================================================================


def get_video_duration_ms(video_path: str) -> int:
    """Get video duration in milliseconds using torchcodec."""
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path)
        metadata = decoder.metadata
        duration_sec = metadata.duration_seconds
        return int(duration_sec * 1000)
    except Exception:
        # Fallback: ffprobe
        import subprocess

        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        return int(float(result.stdout.strip()) * 1000)


# ===================================================================
# Segment ID builder
# ===================================================================


def make_segment_id(video_id: str, start_ms: int, end_ms: int) -> str:
    return f"{video_id}_{start_ms:08d}_{end_ms:08d}"


def make_event_id(video_id: str, seq: int) -> str:
    return f"evt_{video_id}_{seq:04d}"
