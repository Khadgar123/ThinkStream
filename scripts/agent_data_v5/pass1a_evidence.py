"""
Pass 1-A: Independent Per-Chunk Annotation

Each 2s chunk annotated independently with 2 frames.
No sliding window, no prior context, fully parallel.

Output: Per-chunk JSON with visible_entities(desc), atomic_facts, ocr, spatial.
        No state_changes (derived in 1-B), no entity IDs (assigned in 1-B).

Saved to: data/agent_v5/evidence_1a/{video_id}.json
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EVIDENCE_1A_DIR,
    EVIDENCE_GRAPH_PROMPT,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Parse
# ---------------------------------------------------------------------------


def build_evidence_request(
    chunk_idx: int,
    frame_paths: List[str],
    video_id: str,
) -> Dict:
    """Build evidence request for one chunk (independent, 2 frames only)."""
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC

    prompt = EVIDENCE_GRAPH_PROMPT.format(start=int(start), end=int(end))
    chunk_frame_paths = get_chunk_frame_paths(frame_paths, chunk_idx)

    return {
        "messages": [{"role": "user", "content": build_vision_content(prompt, chunk_frame_paths)}],
        "max_tokens": PASS_CONFIG["pass1a"]["max_tokens"],
        "temperature": PASS_CONFIG["pass1a"]["temperature"],
        "id": f"{video_id}_1a_{chunk_idx}",
        "_meta": {"video_id": video_id, "chunk_idx": chunk_idx, "time": [start, end]},
    }


def parse_evidence_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse 397B evidence output. No state_changes expected (comes from 1-B)."""
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

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

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

    raw_facts = parsed.get("atomic_facts", [])
    raw_entities = parsed.get("visible_entities", [])

    return {
        "time": parsed.get("time", meta["time"]),
        "visible_entities": [_normalize_entity(e) for e in raw_entities],
        "atomic_facts": [_normalize_atomic_fact(f) for f in raw_facts],
        "ocr": parsed.get("ocr", []),
        "spatial": parsed.get("spatial", ""),
        "parse_success": True,
    }


def _normalize_atomic_fact(fact) -> dict:
    if isinstance(fact, str):
        return {"fact": fact, "confidence": 0.5, "target_resolution_visible": False, "parse_repaired": True}
    if not isinstance(fact, dict):
        return {"fact": str(fact), "confidence": 0.0, "target_resolution_visible": False, "parse_repaired": True}
    fact.setdefault("confidence", 0.5)
    fact.setdefault("target_resolution_visible", True)
    return fact


def _normalize_entity(entity) -> dict:
    if isinstance(entity, str):
        return {"desc": entity, "action": "", "position": ""}
    if not isinstance(entity, dict):
        return {"desc": str(entity), "action": "", "position": ""}
    if "desc" not in entity and "id" in entity:
        entity["desc"] = entity["id"]
    entity.setdefault("desc", "unknown")
    return entity


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_pass1a(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Annotate all chunks in parallel. Returns sorted captions list."""

    async def annotate_chunk(chunk_idx):
        request = build_evidence_request(chunk_idx, frame_paths, video_id)
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

    tasks = [annotate_chunk(i) for i in range(num_chunks)]
    captions = await asyncio.gather(*tasks)
    captions = sorted(captions, key=lambda c: c["chunk_idx"])

    n_ok = sum(1 for c in captions if c.get("parse_success"))
    logger.info(f"  [{video_id}] 1-A done: {n_ok}/{num_chunks} parsed ok")
    return captions


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_1a(video_id: str, captions: List[Dict], output_dir: Path = EVIDENCE_1A_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def load_1a(video_id: str, evidence_dir: Path = EVIDENCE_1A_DIR) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("1a"):
        return None
    path = evidence_dir / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers (used by other passes too)
# ---------------------------------------------------------------------------


def get_chunk_frame_paths(all_frame_paths: List[str], chunk_idx: int) -> List[str]:
    start = chunk_idx * FRAMES_PER_CHUNK
    end = start + FRAMES_PER_CHUNK
    return all_frame_paths[start:end] if start < len(all_frame_paths) else []


def build_vision_content(text: str, image_paths: List[str]) -> list:
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
