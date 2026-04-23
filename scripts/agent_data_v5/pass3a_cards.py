"""
Pass 3-A: Task Card Generation

Generates question cards from teacher evidence.
Each card defines WHAT to ask (not WHEN or HOW to act).

Two steps:
1. classify_chunks: structural filtering (pure program)
2. generate_cards: per-family 397B calls

Output: task_cards/{video_id}.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    TASK_CARDS_DIR,
    PASS_CONFIG,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)

# Family targets per video
FAMILY_TARGETS = {
    "F1": 3, "F2": 4, "F3": 2, "F4": 2,
    "E1": 3, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 2,
}

# Retention class derived from family (not from 397B)
RETENTION_CLASS = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
}

# Per-family prompt templates
FAMILY_PROMPTS = {
    "F1": """Based on the following video chunks containing OCR text or numbers,
generate {n} questions about precise values (price, count, text on screen).
Prefer answer_form: number or short_exact.

{evidence}

Output JSON array of cards, each with: question, canonical_answer, answer_form, support_chunks, visibility_type.
visibility_type: "transient" for momentary values, "persistent" for always-visible text.""",

    "F2": """Based on the following video chunks, generate {n} questions about
visual attributes (color, material, shape, clothing).
Prefer answer_form: multiple_choice (embed A/B/C/D choices in question) or binary.

{evidence}

Output JSON array. For multiple_choice, put choices in the question text.
visibility_type: "persistent" for always-visible attributes, "transient" for brief appearances.""",

    "F3": """Generate {n} questions about counts/quantities from these chunks.
answer_form: number.

{evidence}

Output JSON array. visibility_type: usually "transient" (counts change).""",

    "F4": """Generate {n} questions about spatial relationships from these chunks.
Prefer binary ("Is X to the left of Y?") or multiple_choice.

{evidence}

Output JSON array. visibility_type: "persistent" for stable layouts.""",

    "E1": """Generate {n} questions about current actions from these chunks.
Prefer binary ("Is the person stirring?") or short_exact.

{evidence}

Output JSON array. visibility_type: "transient" (actions change).""",

    "E2": """Generate {n} event-watch or state-change questions from these chunks.
Format: "Tell me when X starts" or "Has X started yet?"
Prefer binary or short_exact.

{evidence}

Output JSON array. visibility_type: "transient".""",

    "P1": """Generate {n} questions about procedure/step order from these chunks.
Prefer number ("Which step is this?") or multiple_choice.

{evidence}

Output JSON array. visibility_type: "transient".""",

    "C1": """Generate {n} comparison questions: how has something changed over time?
Use entity descriptions (not IDs). Prefer binary ("Has X changed?").

{evidence}

Output JSON array. visibility_type: "transient".""",

    "R1": """Generate {n} re-identification questions: is a previously seen entity
still present? Use appearance descriptions. answer_form: binary.

{evidence}

Output JSON array. visibility_type: "transient".""",

    "S1": """Generate {n} descriptive questions about the scene.
answer_form: descriptive.

{evidence}

Output JSON array. visibility_type: "persistent".""",

    "M1": """Based on this full video summary, generate {n} questions suitable
for continuous commentary (e.g., "Describe each step as it happens").
answer_form: descriptive.

{evidence}

Output JSON array. visibility_type: "transient".""",
}


# ---------------------------------------------------------------------------
# Step 1: Structural chunk classification
# ---------------------------------------------------------------------------


def classify_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """Classify chunks by family using structural fields only.

    No keyword matching on fact text — uses ocr, visible_entities count,
    state_changes, entity_id for cross-chunk analysis.
    """
    fc = {f: [] for f in FAMILY_TARGETS}

    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        entities = cap.get("visible_entities", [])
        facts = [f for f in cap.get("atomic_facts", [])
                 if f.get("confidence", 0) >= 0.7]

        if cap.get("ocr"):
            fc["F1"].append(idx)
        elif any(any(c.isdigit() for c in f.get("fact", "")) for f in facts):
            fc["F1"].append(idx)
            fc["F3"].append(idx)

        if entities:
            fc["F2"].append(idx)
        if len(entities) >= 2:
            fc["F4"].append(idx)

        if cap.get("state_changes"):
            fc["E2"].append(idx)

        if len(entities) >= 3:
            fc["S1"].append(idx)

    # E1: subsample (almost all chunks qualify)
    all_chunks = [cap["chunk_idx"] for cap in evidence if cap.get("atomic_facts")]
    fc["E1"] = all_chunks[::3]

    # P1: consecutive state_changes >= 3
    consecutive = []
    for cap in evidence:
        if cap.get("state_changes"):
            consecutive.append(cap["chunk_idx"])
        else:
            if len(consecutive) >= 3:
                fc["P1"].extend(consecutive)
            consecutive = []
    if len(consecutive) >= 3:
        fc["P1"].extend(consecutive)

    # C1/R1: cross-chunk entity tracking
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}
    entity_appearances = {}
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            eid = e.get("id", e.get("desc", ""))
            if eid and eid != "unknown":
                entity_appearances.setdefault(eid, []).append(cap["chunk_idx"])

    for eid, chunks in entity_appearances.items():
        state_chunks = [c for c in chunks
                        if ev_by_idx.get(c, {}).get("state_changes")]
        if len(state_chunks) >= 2:
            fc["C1"].extend(state_chunks[-2:])
        for i in range(1, len(chunks)):
            if chunks[i] - chunks[i - 1] >= 5:
                fc["R1"].append(chunks[i])

    # Deduplicate
    for f in fc:
        fc[f] = sorted(set(fc[f]))

    return fc


# ---------------------------------------------------------------------------
# Step 2: Per-family 397B card generation
# ---------------------------------------------------------------------------


def _format_evidence_for_prompt(evidence: List[Dict], chunk_indices: List[int]) -> str:
    """Format selected chunks' evidence into a compact prompt string."""
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}
    lines = []
    for idx in chunk_indices[:10]:  # limit to 10 chunks per call
        cap = ev_by_idx.get(idx)
        if not cap:
            continue
        t = cap.get("time", [idx * AGENT_CHUNK_SEC, (idx + 1) * AGENT_CHUNK_SEC])
        entities = [e.get("desc", "?") for e in cap.get("visible_entities", [])]
        facts = [f["fact"] for f in cap.get("atomic_facts", [])
                 if f.get("confidence", 0) >= 0.7]
        ocr = cap.get("ocr", [])
        sc = cap.get("state_changes", [])
        parts = [f"chunk {idx} [{t[0]}-{t[1]}s]"]
        if entities:
            parts.append(f"entities: {entities}")
        if facts:
            parts.append(f"facts: {facts}")
        if ocr:
            parts.append(f"ocr: {ocr}")
        if sc:
            parts.append(f"changes: {sc}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _parse_cards_response(raw: Optional[str], family: str, video_id: str) -> List[Dict]:
    """Parse 397B response into card dicts."""
    if not raw:
        return []
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Try parse as JSON array
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract array
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"  [{video_id}] 3-A {family}: failed to parse cards")
    return []


async def generate_cards(
    video_id: str,
    evidence: List[Dict],
    client,
) -> List[Dict]:
    """Generate task cards for one video via per-family 397B calls.

    Returns list of TaskCard dicts.
    """
    family_chunks = classify_chunks(evidence)
    all_cards = []
    card_counter = 0

    for family, chunk_list in family_chunks.items():
        target_n = FAMILY_TARGETS[family]

        # Skip families with no candidate chunks (except M1 which uses full video)
        if family != "M1" and not chunk_list:
            continue

        # Build evidence input
        if family == "M1":
            ev_text = _format_evidence_for_prompt(evidence, list(range(len(evidence))))
        else:
            ev_text = _format_evidence_for_prompt(evidence, chunk_list)

        if not ev_text.strip():
            continue

        prompt_template = FAMILY_PROMPTS.get(family)
        if not prompt_template:
            continue

        prompt = prompt_template.format(n=target_n, evidence=ev_text)

        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=PASS_CONFIG.get("pass3a", {}).get("max_tokens", 16384),
            temperature=0.7,
            request_id=f"{video_id}_3a_{family}",
        )

        cards = _parse_cards_response(raw, family, video_id)

        for card in cards:
            if not isinstance(card, dict) or not card.get("question"):
                continue
            card_counter += 1
            card["card_id"] = f"{video_id}_{family}_{card_counter:03d}"
            card["family"] = family
            card.setdefault("answer_form", "short_exact")
            card.setdefault("visibility_type", "transient")
            card.setdefault("support_chunks", chunk_list[:1] if chunk_list else [0])
            # Ensure support_chunks is a list of ints
            if isinstance(card["support_chunks"], int):
                card["support_chunks"] = [card["support_chunks"]]
            all_cards.append(card)

    logger.info(f"  [{video_id}] 3-A: {len(all_cards)} cards "
                f"({', '.join(f'{f}:{sum(1 for c in all_cards if c[\"family\"]==f)}' for f in FAMILY_TARGETS if any(c['family']==f for c in all_cards))})")

    return all_cards


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_cards(video_id: str, cards: List[Dict]):
    TASK_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = TASK_CARDS_DIR / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)


def load_cards(video_id: str) -> Optional[List[Dict]]:
    path = TASK_CARDS_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from answer text for retention bitmap."""
    stop = {"the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
            "and", "or", "it", "yes", "no"}
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in stop and len(w) > 1 and w not in seen:
            seen.add(w)
            result.append(w)
    return result
