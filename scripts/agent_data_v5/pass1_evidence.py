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
# 1-B: Post-processing (change detection via 397B)
# ---------------------------------------------------------------------------

ENTITY_LINK_PROMPT = """Below are entity descriptions from different chunks of the same video.
Group descriptions that refer to the SAME entity (same person/object, even if action or angle differs).

{entity_list}

Output JSON array:
[
  {{"id": "person_1", "descs": ["person wearing red apron, standing", "person in red apron, chopping", "hand stirring pot"]}},
  {{"id": "pot_1", "descs": ["stainless steel pot", "pot with reddish sauce"]}}
]"""

STATE_CHANGE_PROMPT = """Below is a per-chunk entity/action summary for a video.
Identify chunks where a meaningful state change occurs compared to the previous chunk.

{chunk_summary}

Rules:
- Ignore paraphrase ("chopping" ≈ "cutting" = no change)
- Ignore camera angle changes without actual scene change
- Focus on: new actions starting, entities appearing/disappearing, step transitions

Output JSON array (only chunks WITH changes):
[{{"chunk": 3, "change": "started pouring oil into pot"}}, ...]"""


def _build_chunk_summary(evidence: List[Dict]) -> str:
    """Build a condensed summary of all chunks for 397B change detection."""
    lines = []
    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        t = cap.get("time", [idx * AGENT_CHUNK_SEC, (idx + 1) * AGENT_CHUNK_SEC])
        entities = []
        for e in cap.get("visible_entities", []):
            desc = e.get("desc", "unknown")
            action = e.get("action", "")
            if action:
                entities.append(f"{desc} ({action})")
            else:
                entities.append(desc)
        entity_str = ", ".join(entities) if entities else "(empty)"
        lines.append(f"chunk {idx} [{t[0]}-{t[1]}s]: {entity_str}")
    return "\n".join(lines)


async def run_pass1b(evidence: List[Dict], client, video_id: str):
    """Pass 1-B: Entity alignment + state change detection (two 397B calls).

    Split into two calls to avoid thinking explosion in a single long-context call:
    - Call 1: Entity linking (~1K input: unique desc list)
    - Call 2: State change detection (~3K input: chunk summary)

    Each call is shorter, thinking more controlled.
    """
    if not evidence:
        return

    for cap in evidence:
        cap["state_changes"] = []

    if len(evidence) <= 1:
        return

    # --- Call 1: Entity linking ---
    all_descs = set()
    for cap in evidence:
        for entity in cap.get("visible_entities", []):
            desc = entity.get("desc", "")
            if desc and desc != "unknown":
                all_descs.add(desc)

    desc_to_id = {}
    if len(all_descs) >= 2:
        entity_list = "\n".join(f"- {d}" for d in sorted(all_descs))
        raw_el = await client._call_one(
            messages=[{"role": "user", "content": ENTITY_LINK_PROMPT.format(entity_list=entity_list)}],
            max_tokens=PASS_CONFIG["pass1_evidence"]["max_tokens"],
            temperature=0.3,
            request_id=f"{video_id}_entity_link",
        )
        if raw_el:
            raw_el = re.sub(r'<think>.*?</think>', '', raw_el, flags=re.DOTALL).strip()
            groups = _parse_json_array(raw_el)
            if groups:
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    group_id = group.get("id", "unknown")
                    for desc in group.get("descs", []):
                        desc_to_id[desc] = group_id

    # Apply entity IDs
    if desc_to_id:
        for cap in evidence:
            for entity in cap.get("visible_entities", []):
                desc = entity.get("desc", "")
                if desc in desc_to_id:
                    entity["id"] = desc_to_id[desc]

    n_groups = len(set(desc_to_id.values())) if desc_to_id else 0

    # --- Call 2: State change detection ---
    chunk_summary = _build_chunk_summary(evidence)
    raw_sc = await client._call_one(
        messages=[{"role": "user", "content": STATE_CHANGE_PROMPT.format(chunk_summary=chunk_summary)}],
        max_tokens=PASS_CONFIG["pass1_evidence"]["max_tokens"],
        temperature=0.3,
        request_id=f"{video_id}_state_changes",
    )

    if raw_sc:
        raw_sc = re.sub(r'<think>.*?</think>', '', raw_sc, flags=re.DOTALL).strip()
        changes_list = _parse_json_array(raw_sc)
        if changes_list:
            chunk_map = {cap.get("chunk_idx", i): cap for i, cap in enumerate(evidence)}
            for item in changes_list:
                if not isinstance(item, dict):
                    continue
                chunk_idx = item.get("chunk")
                change = item.get("change", "state_changed")
                if chunk_idx is not None and chunk_idx in chunk_map:
                    chunk_map[chunk_idx]["state_changes"].append(change)
        else:
            _detect_changes_fallback(evidence)
    else:
        logger.warning(f"  [{video_id}] 1-B state changes: empty response, using fallback")
        _detect_changes_fallback(evidence)

    n_changes = sum(1 for cap in evidence if cap["state_changes"])
    logger.info(
        f"  [{video_id}] 1-B done: {n_groups} entity groups, "
        f"{n_changes}/{len(evidence)} chunks with state changes"
    )


def _parse_json_array(raw: str) -> Optional[List]:
    """Extract a JSON array from raw text."""
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _detect_changes_fallback(evidence: List[Dict]):
    """Fallback change detection when 397B call fails.

    Simple heuristic: entity count change between adjacent chunks.
    """
    for i in range(1, len(evidence)):
        prev_n = len(evidence[i - 1].get("visible_entities", []))
        curr_n = len(evidence[i].get("visible_entities", []))
        if abs(prev_n - curr_n) >= 2:
            evidence[i]["state_changes"] = ["entity_count_changed"]


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

    # 1-B: Entity alignment + change detection (one 397B call per video)
    await run_pass1b(captions, client, video_id)

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
