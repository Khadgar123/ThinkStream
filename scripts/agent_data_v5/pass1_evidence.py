"""
Pass 1: Teacher Evidence Graph

1-A: Independent per-chunk annotation (2 frames, no context).
1-B: Post-processing (entity linking + state_changes derivation).

This is TEACHER-ONLY information — never enters SFT training data.

Input: Video frames (2 frames per chunk)
Output: Per-chunk structured JSON with entities, facts, OCR.
        + cross-chunk entity IDs and state_changes (from 1-B).

Processing: Fully parallel within and across videos (1-A).
            Sequential post-processing (1-B, pure computation).
"""

import json
import logging
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


def build_evidence_request(
    chunk_idx: int,
    frame_paths: List[str],
    video_id: str,
) -> Dict:
    """Build evidence graph request for one chunk (independent, 2 frames only).

    No sliding window, no previous captions. Each chunk is self-contained.
    Entity consistency handled in post-processing (pass1b).
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC

    prompt = EVIDENCE_GRAPH_PROMPT.format(
        start=int(start),
        end=int(end),
    )

    # Only current chunk's 2 frames
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

    # Extract JSON — try direct parse first, then find LAST complete JSON object.
    # "Last" because if thinking leaks into content, thinking text comes first
    # and the actual JSON is at the end.
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Find all '{' positions, try from last to first
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
    """Run evidence graph generation for a single video.

    1-A: All chunks annotated in parallel (independent, 2 frames each).
    1-B: Post-processing — entity linking + state_changes derivation.
    """
    import asyncio

    # --- 1-A: Parallel chunk annotation ---
    async def annotate_chunk(chunk_idx):
        request = build_evidence_request(
            chunk_idx=chunk_idx,
            frame_paths=frame_paths,
            video_id=video_id,
        )
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

    logger.info(f"  [{video_id}] Evidence 1-A: {num_chunks} chunks annotated (parallel)")

    # --- 1-B: Post-processing ---
    link_entities(captions)
    derive_state_changes(captions)

    logger.info(f"  [{video_id}] Evidence 1-B: entity linking + state_changes done")

    return captions


# ---------------------------------------------------------------------------
# 1-B: Post-processing
# ---------------------------------------------------------------------------


def link_entities(evidence: List[Dict]):
    """Link entity descriptions across chunks via word-overlap clustering.

    Assigns consistent IDs (person_1, pot_1, ...) to visible_entities
    based on appearance description similarity.
    """
    # Collect all unique descs
    all_descs = set()
    for cap in evidence:
        for entity in cap.get("visible_entities", []):
            desc = entity.get("desc", entity.get("id", ""))
            if desc:
                all_descs.add(desc)

    if not all_descs:
        return

    # Greedy clustering by word overlap (longest desc first)
    stop = {"the", "a", "an", "is", "in", "on", "at", "of", "with", "and"}
    clusters = []  # [(canonical_desc, [desc1, desc2, ...])]

    for desc in sorted(all_descs, key=len, reverse=True):
        words = set(desc.lower().split()) - stop
        if not words:
            continue
        merged = False
        for canonical, members in clusters:
            canonical_words = set(canonical.lower().split()) - stop
            overlap = len(words & canonical_words) / max(len(words | canonical_words), 1)
            if overlap > 0.5:
                members.append(desc)
                merged = True
                break
        if not merged:
            clusters.append((desc, [desc]))

    # Assign IDs
    desc_to_id = {}
    type_counters = {}
    for canonical, members in clusters:
        prefix = _infer_type_prefix(canonical)
        type_counters[prefix] = type_counters.get(prefix, 0) + 1
        entity_id = f"{prefix}_{type_counters[prefix]}"
        for d in members:
            desc_to_id[d] = entity_id

    # Write back
    for cap in evidence:
        for entity in cap.get("visible_entities", []):
            desc = entity.get("desc", entity.get("id", ""))
            entity["id"] = desc_to_id.get(desc, "unknown")


def _infer_type_prefix(desc: str) -> str:
    """Infer entity type prefix from description text."""
    dl = desc.lower()
    for keyword, prefix in [
        ("person", "person"), ("man", "person"), ("woman", "person"),
        ("chef", "person"), ("hand", "person"), ("child", "person"),
        ("pot", "pot"), ("pan", "pan"), ("bowl", "bowl"), ("plate", "plate"),
        ("knife", "tool"), ("spoon", "tool"), ("fork", "tool"),
        ("bottle", "container"), ("cup", "container"), ("glass", "container"),
        ("screen", "screen"), ("monitor", "screen"), ("phone", "screen"),
    ]:
        if keyword in dl:
            return prefix
    return "object"


def derive_state_changes(evidence: List[Dict]):
    """Derive state_changes by comparing adjacent chunks' entities.

    Detects: entity appeared/disappeared, action changed.
    Must run AFTER link_entities (needs consistent IDs).
    """
    if not evidence:
        return
    evidence[0]["state_changes"] = []

    for i in range(1, len(evidence)):
        prev_entities = {
            e.get("id", e.get("desc", "")): e
            for e in evidence[i - 1].get("visible_entities", [])
        }
        curr_entities = {
            e.get("id", e.get("desc", "")): e
            for e in evidence[i].get("visible_entities", [])
        }
        changes = []

        # New entities
        for eid in curr_entities:
            if eid not in prev_entities and eid != "unknown":
                changes.append(f"{eid} appeared")

        # Disappeared entities
        for eid in prev_entities:
            if eid not in curr_entities and eid != "unknown":
                changes.append(f"{eid} disappeared")

        # Action changes
        for eid in curr_entities:
            if eid in prev_entities:
                prev_action = prev_entities[eid].get("action", "")
                curr_action = curr_entities[eid].get("action", "")
                if prev_action and curr_action and prev_action != curr_action:
                    changes.append(f"{eid}: {prev_action} -> {curr_action}")

        evidence[i]["state_changes"] = changes


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
