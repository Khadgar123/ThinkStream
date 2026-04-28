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
Each line lists a description AND the chunk indices where it appeared.

Group descriptions that refer to the SAME entity (same person/object), even if
the wording, action, or camera angle differs. This is CRITICAL for downstream
recall: if the same person appears in chunks 0, 5, 12, all three descriptions
MUST end up in one group with one id.

{entity_list}

Aggressive merging rules:
- Same body / clothing / hair color in nearby chunks → same person, even if
  the wording differs ("man in black polo" / "person in dark shirt").
- Hands, arms, or partial views of an already-named person → merge into that
  person's group (use a consistent id like person_chef_1, not hand_1).
- Same object with stable color / shape across chunks → same id (e.g.
  pot_silver_1) even if action differs ("stirring pot" / "pot on stove").
- DO NOT merge across visual categories. Text overlay graphics, arrows, and
  shape annotations are NOT the same entity even if they share style. Use
  separate ids: text_overlay_1 vs arrow_1 vs shape_outline_1.
- Only create a NEW id when the entity is genuinely new (different visible
  attributes, not just different wording).
- An entity that is mentioned in ≥2 chunks should always end up grouped —
  isolated single-chunk ids are a sign of under-merging.

Use stable, descriptive ids: person_chef_1, person_blonde_1, pot_silver_1,
chair_wooden_1. Avoid generic ids like person_1, object_1.

Output JSON array (group everything; do not omit any input desc):
[
  {{"id": "person_chef_1", "descs": ["person wearing red apron, standing", "person in red apron, chopping", "hand stirring pot"]}},
  {{"id": "pot_silver_1", "descs": ["stainless steel pot", "pot with reddish sauce"]}}
]"""

# v9.5: combined entity-link + state-change in ONE LLM call. Saves the
# round-trip overhead of pass1b's second call (was ~40% of pass1b time).
# The model sees both pieces of context (entity table + chunk_summary)
# and outputs both pieces in one structured response.
COMBINED_LINK_AND_STATE_PROMPT = """You are processing a video's per-chunk
captions. Do TWO tasks in one response:

TASK 1 — Entity linking. Below are entity descriptions and the chunks they
appear in. Group descriptions that refer to the SAME entity.

{entity_list}

Linking rules:
- Same body / clothing / hair color across chunks → same person.
- Hands, arms, partial views of an already-named person → merge into that
  person's group (use ids like person_chef_1, not hand_1).
- Same object with stable color/shape → same id, even if action differs.
- DO NOT merge across visual categories (text overlay vs arrow vs object
  must stay separate).
- An entity mentioned in ≥2 chunks should always be grouped.
- Use stable descriptive ids: person_chef_1, pot_silver_1, etc.

TASK 2 — State change detection. Below is a per-chunk action/entity summary.
Identify chunks where a meaningful state change occurs vs the previous chunk.

{chunk_summary}

State-change rules:
- Ignore paraphrase ("chopping" ≈ "cutting" = no change).
- Ignore camera angle changes without scene change.
- Focus on: new actions starting, entities appearing/disappearing, step
  transitions, completion of one action.

Output a SINGLE JSON object with both fields:
{{
  "entity_groups": [
    {{"id": "person_chef_1", "descs": ["...", "..."]}},
    {{"id": "pot_silver_1",  "descs": ["..."]}}
  ],
  "state_changes": [
    {{"chunk": 3, "change": "started pouring oil into pot"}},
    {{"chunk": 7, "change": "first batch removed from oven"}}
  ]
}}"""

# Legacy single-call prompts kept for the run_pass1b_legacy path (not the
# default), so anyone needing to reproduce v9.4 data can flip a flag.
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

    v9.5: ONE combined LLM call (entity_link + state_change). Saves the
    extra round-trip plus the redundant prompt-prefix tokens of two
    separate calls. Each task is independent so combining loses no
    information; the model sees both contexts and produces both outputs.

    Returns a deep copy — does NOT mutate the input.
    """
    enriched = deepcopy(evidence)

    for cap in enriched:
        cap["state_changes"] = []

    if len(enriched) <= 1:
        return enriched

    # Build desc → sorted chunk-index list (for entity linking)
    desc_chunks: Dict[str, List[int]] = {}
    for cap in enriched:
        cidx = cap.get("chunk_idx", -1)
        for entity in cap.get("visible_entities", []):
            desc = entity.get("desc", "")
            if desc and desc != "unknown":
                desc_chunks.setdefault(desc, []).append(cidx)

    if len(desc_chunks) < 2:
        # No linking possible. Still try state_change with chunk_summary.
        _run_state_change_only(enriched, client, video_id, semaphore)
        return enriched

    entity_list = "\n".join(
        f"- chunks={sorted(set(chunks))} | desc: {desc}"
        for desc, chunks in sorted(desc_chunks.items())
    )
    chunk_summary = _build_chunk_summary(enriched)
    prompt = COMBINED_LINK_AND_STATE_PROMPT.format(
        entity_list=entity_list,
        chunk_summary=chunk_summary,
    )

    raw = await _call_with_semaphore(
        client, prompt,
        PASS_CONFIG["pass1b"]["max_tokens"], 0.3,
        f"{video_id}_link_and_state", semaphore,
    )

    desc_to_id: Dict[str, str] = {}
    state_changes_applied = False

    if raw:
        parsed = _parse_combined_json(raw)
        if parsed:
            # Entity groups
            for group in parsed.get("entity_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                gid = group.get("id", "unknown")
                for desc in group.get("descs", []) or []:
                    desc_to_id[desc] = gid
            # State changes
            chunk_map = {cap.get("chunk_idx", i): cap
                         for i, cap in enumerate(enriched)}
            for item in parsed.get("state_changes", []) or []:
                if not isinstance(item, dict):
                    continue
                cidx = item.get("chunk")
                change = item.get("change", "state_changed")
                if cidx is not None and cidx in chunk_map:
                    chunk_map[cidx]["state_changes"].append(change)
                    state_changes_applied = True

    if desc_to_id:
        for cap in enriched:
            for entity in cap.get("visible_entities", []):
                desc = entity.get("desc", "")
                if desc in desc_to_id:
                    entity["id"] = desc_to_id[desc]

    if not state_changes_applied:
        # 1A-derived fallback (entity-count change + primary-action change)
        _detect_changes_fallback(enriched)

    n_groups = len(set(desc_to_id.values())) if desc_to_id else 0
    n_changes = sum(1 for cap in enriched if cap["state_changes"])
    logger.info(
        f"  [{video_id}] 1-B: {n_groups} entity groups, "
        f"{n_changes}/{len(enriched)} chunks with state changes "
        f"(combined call)"
    )

    return enriched


async def _run_state_change_only(enriched, client, video_id, semaphore):
    """Fallback path when only 0-1 unique descs exist (no linking to do)."""
    chunk_summary = _build_chunk_summary(enriched)
    prompt = STATE_CHANGE_PROMPT.format(chunk_summary=chunk_summary)
    raw = await _call_with_semaphore(
        client, prompt,
        PASS_CONFIG["pass1b"]["max_tokens"], 0.3,
        f"{video_id}_state_changes_only", semaphore,
    )
    if raw:
        changes = _parse_json_array(raw)
        if changes:
            chunk_map = {cap.get("chunk_idx", i): cap
                         for i, cap in enumerate(enriched)}
            for item in changes:
                if not isinstance(item, dict):
                    continue
                cidx = item.get("chunk")
                change = item.get("change", "state_changed")
                if cidx is not None and cidx in chunk_map:
                    chunk_map[cidx]["state_changes"].append(change)
            return
    _detect_changes_fallback(enriched)


def _parse_combined_json(raw: str):
    """Parse the combined link+state JSON object. Returns dict or None."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: extract first {...} block
    start = raw.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    return None
                break
    return None


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
    # v9.5: pass1b honours PASS_CONFIG["pass1b"]["thinking"] (default True
    # to preserve current behaviour). Routed through raw httpx when False.
    enable_thinking = bool(PASS_CONFIG.get("pass1b", {}).get("thinking", True))
    for attempt in range(max_retries + 1):
        temp = temperature if attempt == 0 else min(temperature + 0.2 * attempt, 1.0)
        if semaphore:
            async with semaphore:
                raw = await client._call_one(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temp,
                    request_id=f"{request_id}_r{attempt}" if attempt > 0 else request_id,
                    enable_thinking=enable_thinking,
                )
        else:
            raw = await client._call_one(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp,
                request_id=f"{request_id}_r{attempt}" if attempt > 0 else request_id,
                enable_thinking=enable_thinking,
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
    """Fallback: detect state changes when LLM did not provide them.

    v9.1: tighten from "any 2-entity swing" to either:
      (a) entity count changed by >= 3 (strong scene-level change), OR
      (b) primary entity DESCRIPTION changed (different desc text), AND
          count delta >= 1.

    Avoids over-classifying flickering entity detection (a tomato briefly
    occluded counts as a change) while still catching real procedural steps.
    """
    for i in range(1, len(evidence)):
        prev_entities = evidence[i - 1].get("visible_entities", [])
        curr_entities = evidence[i].get("visible_entities", [])
        prev_n = len(prev_entities)
        curr_n = len(curr_entities)
        delta = abs(prev_n - curr_n)

        if delta >= 3:
            evidence[i]["state_changes"] = ["entity_count_changed_significantly"]
            continue

        # Check if dominant entity description actually changed (any count delta).
        prev_desc = (prev_entities[0].get("desc", "") if prev_entities else "")
        curr_desc = (curr_entities[0].get("desc", "") if curr_entities else "")
        if prev_desc and curr_desc and prev_desc != curr_desc:
            pw = set(prev_desc.lower().split())
            cw = set(curr_desc.lower().split())
            if pw and cw:
                overlap = len(pw & cw) / min(len(pw), len(cw))
                if overlap < 0.5:
                    evidence[i]["state_changes"] = ["primary_entity_changed"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_1b(video_id: str, evidence: List[Dict], output_dir: Path = EVIDENCE_1B_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)


def load_1b(video_id: str, evidence_dir: Path = EVIDENCE_1B_DIR) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("1b"):
        return None
    path = evidence_dir / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
