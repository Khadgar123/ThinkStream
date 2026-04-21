"""
Pass 1: Teacher Evidence Graph

Generates detailed structured captions for every 2s chunk.
This is TEACHER-ONLY information — never enters SFT training data.

Input: Video frames (24-frame sliding window + text context)
Output: Per-chunk structured JSON with entities, facts, OCR, state changes.

Processing: Sequential per video (needs prior context), parallel across videos.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EVIDENCE_DIR,
    EVIDENCE_GRAPH_PROMPT,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)


def build_evidence_request(
    chunk_idx: int,
    frame_paths: List[str],
    previous_captions: List[Dict],
    video_id: str,
) -> Dict:
    """Build a single evidence graph request for one chunk.

    397B sees: 24-frame sliding window + all previous captions as text context.
    This matches the inference model's visual window for consistency.
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC

    # Format previous captions as context (text only, for entity consistency)
    caption_lines = []
    for cap in previous_captions[-30:]:  # limit context to last 30 for token budget
        t = cap["time"]
        entities = [e["id"] for e in cap.get("visible_entities", [])]
        caption_lines.append(f"[{t[0]}-{t[1]}] entities: {entities}")

    previous_text = "\n".join(caption_lines) if caption_lines else "(first chunk)"

    prompt = EVIDENCE_GRAPH_PROMPT.format(
        previous_captions=previous_text,
        start=int(start),
        end=int(end),
    )

    # Visual input: sliding window of up to 24 frames
    window_start_chunk = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    window_frame_paths = []
    for c in range(window_start_chunk, chunk_idx + 1):
        chunk_frames = get_chunk_frame_paths(frame_paths, c)
        window_frame_paths.extend(chunk_frames)

    return {
        "messages": [{"role": "user", "content": build_vision_content(prompt, window_frame_paths)}],
        "max_tokens": PASS_CONFIG["pass1_evidence"]["max_tokens"],
        "temperature": PASS_CONFIG["pass1_evidence"]["temperature"],
        "id": f"{video_id}_evidence_{chunk_idx}",
        "_meta": {"video_id": video_id, "chunk_idx": chunk_idx, "time": [start, end]},
    }


def parse_evidence_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse 397B evidence graph output.

    Handles thinking mode: strips <think>...</think> if present.
    Falls back gracefully on parse failure.
    """
    import re

    default = {
        "time": meta["time"],
        "visible_entities": [],
        "atomic_facts": [],
        "state_changes": [],
        "ocr": [],
        "spatial": "",
        "not_observable": [],
        "parse_success": False,
    }

    if raw is None:
        return default

    # Strip thinking block if present
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract JSON
    try:
        # Try direct parse
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Find balanced JSON object
        start_idx = raw.find("{")
        if start_idx < 0:
            default["_raw"] = raw[:4000]
            return default
        depth = 0
        for i in range(start_idx, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start_idx:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        default["_raw"] = raw[:4000]
                        return default
                    break
        else:
            default["_raw"] = raw[:4000]
            return default

    # Validate and normalize
    raw_facts = parsed.get("atomic_facts", [])
    normalized_facts = [_normalize_atomic_fact(f) for f in raw_facts]

    raw_entities = parsed.get("visible_entities", [])
    normalized_entities = [_normalize_entity(e) for e in raw_entities]

    result = {
        "time": parsed.get("time", meta["time"]),
        "visible_entities": normalized_entities,
        "atomic_facts": normalized_facts,
        "state_changes": parsed.get("state_changes", []),
        "ocr": parsed.get("ocr", []),
        "spatial": parsed.get("spatial", ""),
        "not_observable": parsed.get("not_observable", []),
        "parse_success": True,
    }

    return result


def _normalize_atomic_fact(fact) -> dict:
    """Normalize atomic_fact to ensure consistent schema.

    Handles cases where 397B outputs a string instead of a dict.
    """
    if isinstance(fact, str):
        return {
            "fact": fact,
            "confidence": 0.5,
            "support_level": "unknown",
            "target_resolution_visible": False,
            "parse_repaired": True,
        }
    if not isinstance(fact, dict):
        return {
            "fact": str(fact),
            "confidence": 0.0,
            "support_level": "unknown",
            "target_resolution_visible": False,
            "parse_repaired": True,
        }
    # Ensure required fields with safe defaults
    fact.setdefault("confidence", 0.5)
    fact.setdefault("support_level", "unknown")
    fact.setdefault("target_resolution_visible", True)
    return fact


def _normalize_entity(entity) -> dict:
    """Normalize visible_entity to ensure consistent schema."""
    if isinstance(entity, str):
        return {"id": entity, "attributes": [], "action": "", "position": ""}
    if not isinstance(entity, dict):
        return {"id": str(entity), "attributes": [], "action": "", "position": ""}
    entity.setdefault("id", "unknown")
    entity.setdefault("attributes", [])
    return entity


async def run_pass1_single_video(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,  # VLLMClient
) -> List[Dict]:
    """Run evidence graph generation for a single video (sequential).

    Each chunk depends on previous captions for entity consistency.
    """
    captions = []

    for chunk_idx in range(num_chunks):
        request = build_evidence_request(
            chunk_idx=chunk_idx,
            frame_paths=frame_paths,
            previous_captions=captions,
            video_id=video_id,
        )

        # Single call (sequential within video)
        result = await client._call_one(
            messages=request["messages"],
            max_tokens=request["max_tokens"],
            temperature=request["temperature"],
            request_id=request["id"],
        )

        caption = parse_evidence_result(result, request["_meta"])
        caption["chunk_idx"] = chunk_idx
        caption["video_id"] = video_id
        captions.append(caption)

        if (chunk_idx + 1) % 10 == 0:
            logger.info(f"  [{video_id}] Evidence: {chunk_idx+1}/{num_chunks} chunks")

    return captions


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
# Helpers
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
