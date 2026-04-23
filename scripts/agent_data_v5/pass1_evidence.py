"""
Pass 1: Teacher Evidence Graph

1-A: Independent per-chunk annotation (2 frames, no context, fully parallel).
1-B: Post-processing (change detection via action word-set comparison).

This is TEACHER-ONLY information — never enters SFT training data.

Input: Video frames (2 frames per chunk)
Output: Per-chunk structured JSON with entities (desc), facts, OCR, spatial.
        + state_changes derived by 1-B (not from 397B).

Processing: All chunks from all videos fully parallel (shared semaphore).
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EVIDENCE_DIR,
    EVIDENCE_GRAPH_PROMPT,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1-A: Independent Chunk Annotation
# ---------------------------------------------------------------------------


def build_evidence_request(
    chunk_idx: int,
    frame_paths: List[str],
    video_id: str,
) -> Dict:
    """Build evidence request for one chunk (independent, 2 frames only).

    No sliding window, no previous captions. Each chunk is self-contained.
    Entities described by appearance (desc), not by ID.
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC

    prompt = EVIDENCE_GRAPH_PROMPT.format(
        start=int(start),
        end=int(end),
    )

    chunk_frame_paths = get_chunk_frame_paths(frame_paths, chunk_idx)

    return {
        "messages": [{"role": "user", "content": build_vision_content(prompt, chunk_frame_paths)}],
        "max_tokens": PASS_CONFIG["pass1_evidence"]["max_tokens"],
        "temperature": PASS_CONFIG["pass1_evidence"]["temperature"],
        "id": f"{video_id}_evidence_{chunk_idx}",
        "_meta": {"video_id": video_id, "chunk_idx": chunk_idx, "time": [start, end]},
    }


def parse_evidence_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse 397B evidence graph output.

    Expected fields from 397B: visible_entities, atomic_facts, ocr, spatial.
    NOT expected: state_changes (derived in 1-B), not_observable (removed).
    """
    default = {
        "time": meta["time"],
        "visible_entities": [],
        "atomic_facts": [],
        "ocr": [],
        "spatial": "",
        "parse_success": False,
    }

    if raw is None:
        return default

    # Strip thinking block if present
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract JSON — try direct parse first, then find LAST complete JSON object.
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        positions = [i for i, c in enumerate(raw) if c == '{']
        if not positions:
            default["_raw"] = raw[:4000]
            return default
        parsed = None
        for start_idx in reversed(positions):
            depth = 0
            for i in range(start_idx, len(raw)):
                if raw[i] == '{':
                    depth += 1
                elif raw[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(raw[start_idx:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            if parsed is not None:
                break
        if parsed is None:
            default["_raw"] = raw[:4000]
            return default

    # Normalize
    raw_facts = parsed.get("atomic_facts", [])
    normalized_facts = [_normalize_atomic_fact(f) for f in raw_facts]

    raw_entities = parsed.get("visible_entities", [])
    normalized_entities = [_normalize_entity(e) for e in raw_entities]

    return {
        "time": parsed.get("time", meta["time"]),
        "visible_entities": normalized_entities,
        "atomic_facts": normalized_facts,
        "ocr": parsed.get("ocr", []),
        "spatial": parsed.get("spatial", ""),
        "parse_success": True,
    }


def _normalize_atomic_fact(fact) -> dict:
    """Normalize atomic_fact to ensure consistent schema.

    v8.0: no support_level (each chunk is independent, all facts are direct).
    """
    if isinstance(fact, str):
        return {
            "fact": fact,
            "confidence": 0.5,
            "target_resolution_visible": False,
            "parse_repaired": True,
        }
    if not isinstance(fact, dict):
        return {
            "fact": str(fact),
            "confidence": 0.0,
            "target_resolution_visible": False,
            "parse_repaired": True,
        }
    fact.setdefault("confidence", 0.5)
    fact.setdefault("target_resolution_visible", True)
    return fact


def _normalize_entity(entity) -> dict:
    """Normalize visible_entity to ensure consistent schema.

    Pass 1-A uses 'desc' (appearance description) not 'id'.
    Falls back to 'id' for backward compat with old evidence files.
    """
    if isinstance(entity, str):
        return {"desc": entity, "action": "", "position": ""}
    if not isinstance(entity, dict):
        return {"desc": str(entity), "action": "", "position": ""}
    if "desc" not in entity and "id" in entity:
        entity["desc"] = entity["id"]
    entity.setdefault("desc", "unknown")
    return entity


# ---------------------------------------------------------------------------
# 1-B: Post-processing (change detection)
# ---------------------------------------------------------------------------


def detect_chunk_changes(evidence: List[Dict]):
    """Detect changes between adjacent chunks via action word-set comparison.

    Does NOT require entity linking — compares the set of action keywords
    across all entities in adjacent chunks.

    Outputs: evidence[i]["state_changes"] = ["action_changed", ...] or []
    """
    _stop = {"the", "a", "an", "is", "in", "on", "at", "to", "of", "with", "and"}

    def _action_words(cap: Dict) -> set:
        words = set()
        for entity in cap.get("visible_entities", []):
            action = entity.get("action", "")
            words |= {w.lower() for w in action.split()
                       if w.lower() not in _stop and len(w) > 2}
        return words

    if not evidence:
        return
    evidence[0]["state_changes"] = []

    for i in range(1, len(evidence)):
        prev_actions = _action_words(evidence[i - 1])
        curr_actions = _action_words(evidence[i])

        prev_n = len(evidence[i - 1].get("visible_entities", []))
        curr_n = len(evidence[i].get("visible_entities", []))
        entity_count_changed = abs(prev_n - curr_n) >= 1

        if not prev_actions and not curr_actions:
            action_changed = False
        elif not prev_actions or not curr_actions:
            action_changed = True
        else:
            overlap = len(prev_actions & curr_actions) / max(
                len(prev_actions | curr_actions), 1
            )
            action_changed = overlap < 0.5

        changes = []
        if action_changed:
            changes.append("action_changed")
        if entity_count_changed:
            changes.append("entity_count_changed")

        evidence[i]["state_changes"] = changes


# ---------------------------------------------------------------------------
# Main Entry Points
# ---------------------------------------------------------------------------


async def run_pass1_single_video(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Run evidence graph generation for a single video.

    1-A: All chunks annotated in parallel (shared semaphore controls concurrency).
    1-B: Post-processing — change detection.

    Args:
        semaphore: Shared across all videos. If None, no concurrency limit.
    """
    async def annotate_chunk(chunk_idx):
        request = build_evidence_request(
            chunk_idx=chunk_idx,
            frame_paths=frame_paths,
            video_id=video_id,
        )
        if semaphore:
            async with semaphore:
                result = await client._call_one(
                    messages=request["messages"],
                    max_tokens=request["max_tokens"],
                    temperature=request["temperature"],
                    request_id=request["id"],
                )
        else:
            result = await client._call_one(
                messages=request["messages"],
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
                request_id=request["id"],
            )
        caption = parse_evidence_result(result, request["_meta"])
        caption["chunk_idx"] = chunk_idx
        caption["video_id"] = video_id
        return caption

    # 1-A: Launch all chunks (semaphore controls actual concurrency)
    tasks = [annotate_chunk(i) for i in range(num_chunks)]
    captions = await asyncio.gather(*tasks)
    captions = sorted(captions, key=lambda c: c["chunk_idx"])

    logger.info(f"  [{video_id}] Evidence 1-A: {num_chunks} chunks done")

    # 1-B: Change detection (pure computation)
    detect_chunk_changes(captions)

    return captions


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_evidence(video_id: str, captions: List[Dict], output_dir: Path = EVIDENCE_DIR):
    """Save evidence graph for one video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def load_evidence(video_id: str, evidence_dir: Path = EVIDENCE_DIR) -> Optional[List[Dict]]:
    """Load cached evidence graph if available."""
    path = evidence_dir / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers (used by other passes too)
# ---------------------------------------------------------------------------


def get_chunk_frame_paths(all_frame_paths: List[str], chunk_idx: int) -> List[str]:
    """Get frame paths for a specific chunk index."""
    start = chunk_idx * FRAMES_PER_CHUNK
    end = start + FRAMES_PER_CHUNK
    return all_frame_paths[start:end] if start < len(all_frame_paths) else []


def build_vision_content(text: str, image_paths: List[str]) -> list:
    """Build OpenAI-format multimodal content."""
    from scripts.agent_data_pipeline.vllm_client import encode_image_base64

    content = []
    for img_path in image_paths:
        if Path(img_path).exists():
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_base64(img_path)},
            })
    content.append({"type": "text", "text": text})
    return content
