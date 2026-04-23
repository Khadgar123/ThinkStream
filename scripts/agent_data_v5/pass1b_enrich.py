"""
Pass 1-B: Entity Alignment + State Change Detection

Two 397B text-only calls per video (no video frames):
  Call 1: Entity linking — group descs that refer to the same entity
  Call 2: State change detection — identify meaningful changes between chunks

Input:  1-A evidence (per-chunk captions)
Output: Enriched evidence with entity IDs (hint) + state_changes

entity_id is a COARSE HINT for scan_opportunities, not ground truth.
All question text must use desc, never entity_id.

Saved to: data/agent_v5/evidence_1b/{video_id}.json
          (1-A original preserved in evidence_1a/ for comparison)
"""

import asyncio
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EVIDENCE_1B_DIR,
    PASS_CONFIG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_pass1b(
    evidence: List[Dict],
    client,
    video_id: str,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Enrich 1-A evidence with entity IDs and state_changes.

    Returns a deep copy — does NOT mutate the input.
    Two 397B text-only calls (no frames).

    Args:
        semaphore: Shared concurrency limit for 1-B calls.
    """
    enriched = deepcopy(evidence)

    # Initialize state_changes
    for cap in enriched:
        cap["state_changes"] = []

    if len(enriched) <= 1:
        return enriched

    # --- Call 1: Entity linking ---
    all_descs = set()
    for cap in enriched:
        for entity in cap.get("visible_entities", []):
            desc = entity.get("desc", "")
            if desc and desc != "unknown":
                all_descs.add(desc)

    desc_to_id = {}
    if len(all_descs) >= 2:
        entity_list = "\n".join(f"- {d}" for d in sorted(all_descs))
        prompt = ENTITY_LINK_PROMPT.format(entity_list=entity_list)

        raw_el = await _call_with_semaphore(
            client, prompt,
            PASS_CONFIG["pass1b"]["max_tokens"], 0.3,
            f"{video_id}_entity_link", semaphore,
        )
        if raw_el:
            groups = _parse_json_array(raw_el)
            if groups:
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    group_id = group.get("id", "unknown")
                    for desc in group.get("descs", []):
                        desc_to_id[desc] = group_id

    # Apply entity IDs (hint only, not used in question text)
    if desc_to_id:
        for cap in enriched:
            for entity in cap.get("visible_entities", []):
                desc = entity.get("desc", "")
                if desc in desc_to_id:
                    entity["id"] = desc_to_id[desc]

    n_groups = len(set(desc_to_id.values())) if desc_to_id else 0

    # --- Call 2: State change detection ---
    chunk_summary = _build_chunk_summary(enriched)
    prompt = STATE_CHANGE_PROMPT.format(chunk_summary=chunk_summary)

    raw_sc = await _call_with_semaphore(
        client, prompt,
        PASS_CONFIG["pass1b"]["max_tokens"], 0.3,
        f"{video_id}_state_changes", semaphore,
    )

    if raw_sc:
        changes_list = _parse_json_array(raw_sc)
        if changes_list:
            chunk_map = {cap.get("chunk_idx", i): cap for i, cap in enumerate(enriched)}
            for item in changes_list:
                if not isinstance(item, dict):
                    continue
                chunk_idx = item.get("chunk")
                change = item.get("change", "state_changed")
                if chunk_idx is not None and chunk_idx in chunk_map:
                    chunk_map[chunk_idx]["state_changes"].append(change)
        else:
            _detect_changes_fallback(enriched)
    else:
        logger.warning(f"  [{video_id}] 1-B state changes: empty response, using fallback")
        _detect_changes_fallback(enriched)

    n_changes = sum(1 for cap in enriched if cap["state_changes"])
    logger.info(
        f"  [{video_id}] 1-B: {n_groups} entity groups, "
        f"{n_changes}/{len(enriched)} chunks with state changes"
    )

    return enriched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MAX_MODEL_LEN = 65536
INPUT_MARGIN = 1000  # safety margin for tokenizer differences


def _safe_max_tokens(prompt: str, configured_max: int) -> int:
    """Dynamically cap max_tokens so input + max_tokens <= MAX_MODEL_LEN.

    For short inputs (~1K), max_tokens ≈ configured_max (60K).
    For long inputs (~5K), max_tokens drops to ~59K.
    Prevents 400 error on long videos.
    """
    # Estimate input tokens: ~1 token per 3.5 chars for mixed EN/CJK
    estimated_input = len(prompt) // 3 + INPUT_MARGIN
    available = MAX_MODEL_LEN - estimated_input
    return max(8192, min(configured_max, available))  # floor 8K for thinking room


async def _call_with_semaphore(client, prompt, max_tokens, temperature, request_id, semaphore,
                               max_retries: int = 2):
    """Call 397B with optional semaphore. Retries on empty response (thinking explosion).

    Dynamically caps max_tokens based on input length to avoid 400 errors.
    When thinking consumes all max_tokens, content is empty.
    Retry with higher temperature (encourages shorter thinking) as fallback.
    """
    max_tokens = _safe_max_tokens(prompt, max_tokens)
    for attempt in range(max_retries + 1):
        temp = temperature if attempt == 0 else min(temperature + 0.2 * attempt, 1.0)
        if semaphore:
            async with semaphore:
                raw = await client._call_one(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temp,
                    request_id=f"{request_id}_r{attempt}" if attempt > 0 else request_id,
                )
        else:
            raw = await client._call_one(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp,
                request_id=f"{request_id}_r{attempt}" if attempt > 0 else request_id,
            )

        if raw:
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        if raw:
            return raw

        # Empty response — likely thinking explosion ate all max_tokens
        if attempt < max_retries:
            logger.warning(
                f"  [{request_id}] empty response (attempt {attempt+1}/{max_retries+1}), "
                f"retrying with temperature={temp + 0.2:.1f}"
            )

    logger.warning(f"  [{request_id}] all {max_retries+1} attempts returned empty")
    return None


def _build_chunk_summary(evidence: List[Dict]) -> str:
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


def _parse_json_array(raw: str) -> Optional[List]:
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
    """Fallback: entity count change between adjacent chunks."""
    for i in range(1, len(evidence)):
        prev_n = len(evidence[i - 1].get("visible_entities", []))
        curr_n = len(evidence[i].get("visible_entities", []))
        if abs(prev_n - curr_n) >= 2:
            evidence[i]["state_changes"] = ["entity_count_changed"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_1b(video_id: str, evidence: List[Dict], output_dir: Path = EVIDENCE_1B_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)


def load_1b(video_id: str, evidence_dir: Path = EVIDENCE_1B_DIR) -> Optional[List[Dict]]:
    path = evidence_dir / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
