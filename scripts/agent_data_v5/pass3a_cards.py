"""
Pass 3-A: Task Card Generation

Generates question cards from teacher evidence.
Each card defines WHAT to ask (not WHEN or HOW to act).

Two steps:
1. classify_chunks: structural filtering (pure program)
2. generate_cards: per-family 397B calls

Output: task_cards/{video_id}.json
"""

import asyncio
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

# Shared output schema appended to every family prompt.
# Solves: (1) inconsistent field names across families,
#          (2) uncontrolled canonical_answer format,
#          (3) entity_id leaking into question text.
_OUTPUT_SCHEMA = """

Output a JSON array. Each element MUST have exactly these fields:
{{
  "question": "...",
  "canonical_answer": "...",
  "answer_form": "binary|multiple_choice|number|short_exact|descriptive",
  "support_chunks": [chunk_idx, ...],
  "visibility_type": "persistent|transient"
}}

canonical_answer format rules:
- binary: exactly "Yes" or "No" (English, capitalized)
- multiple_choice: exactly one letter "A", "B", "C", or "D"
- number: digits only, no units (e.g. "3" not "3个")
- short_exact: 1-5 English words, no articles
- descriptive: 1-3 sentences

Entity reference rules:
- ALWAYS refer to entities by visual appearance ("person wearing red apron"),
  NEVER by ID ("person_1") — the model cannot see IDs at inference time.
- For multiple_choice, embed "A. ... B. ... C. ... D. ..." in the question text.
  Distractors should come from the same video (other chunks' facts of same type).
  Randomize the position of the correct answer among A-D."""

# Per-family prompt templates
FAMILY_PROMPTS = {
    "F1": """Based on the following video chunks containing OCR text or numbers,
generate {n} questions about precise values (price, count, text on screen).
Prefer answer_form: number or short_exact.
visibility_type: "transient" for momentary values, "persistent" for always-visible text.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F2": """Based on the following video chunks, generate {n} questions about
visual attributes (color, material, shape, clothing).
Prefer answer_form: multiple_choice or binary.
visibility_type: "persistent" for always-visible attributes, "transient" for brief appearances.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F3": """Generate {n} questions about counts/quantities from these chunks.
Prefer answer_form: number.
visibility_type: usually "transient" (counts change).

{evidence}
""" + _OUTPUT_SCHEMA,

    "F4": """Generate {n} questions about spatial relationships from these chunks.
Prefer answer_form: binary ("Is X to the left of Y?") or multiple_choice.
visibility_type: "persistent" for stable layouts, "transient" for moving objects.

{evidence}
""" + _OUTPUT_SCHEMA,

    "E1": """Generate {n} questions about current actions from these chunks.
Prefer answer_form: binary ("Is the person stirring?") or short_exact.
visibility_type: "transient" (actions change).

{evidence}
""" + _OUTPUT_SCHEMA,

    "E2": """Generate {n} event-watch or state-change questions from these chunks.
Question format: "Tell me when X starts" or "Has X started yet?"
Prefer answer_form: binary or short_exact.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "P1": """Generate {n} questions about procedure/step order from these chunks.
Prefer answer_form: number ("Which step is this?") or multiple_choice.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "C1": """Generate {n} comparison questions: how has something changed over time?
Prefer answer_form: binary ("Has X changed since earlier?").
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "R1": """Generate {n} re-identification questions: is a previously seen entity
still present? Describe the entity by appearance.
answer_form: binary.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "S1": """Generate {n} descriptive questions about the scene.
answer_form: descriptive.
visibility_type: "persistent".

{evidence}
""" + _OUTPUT_SCHEMA,

    "M1": """Based on this full video summary, generate {n} questions suitable
for continuous commentary (e.g., "Describe each step as it happens").
answer_form: descriptive.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,
}


# ---------------------------------------------------------------------------
# Step 1: Structural chunk classification
# ---------------------------------------------------------------------------


def _desc_overlap(desc_a: str, desc_b: str) -> float:
    """Word-level overlap ratio between two entity descriptions."""
    words_a = set(re.findall(r'[a-zA-Z]{2,}', desc_a.lower()))
    words_b = set(re.findall(r'[a-zA-Z]{2,}', desc_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _get_primary_action(cap: Dict) -> str:
    """Extract the primary entity action from a chunk (1-A direct field)."""
    for e in cap.get("visible_entities", []):
        action = e.get("action", "")
        if action:
            return action.lower().strip()
    return ""


def classify_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """Classify chunks by family using structural fields.

    Primary path: 1-A direct fields (ocr, visible_entities, atomic_facts).
    Fallback path for P1/C1/R1: 1-A action/desc fields when 1-B fields
    (state_changes, entity_id) are missing, to reduce false negatives.
    """
    fc = {f: [] for f in FAMILY_TARGETS}

    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        entities = cap.get("visible_entities", [])
        facts = [f for f in cap.get("atomic_facts", [])
                 if f.get("confidence", 0) >= 0.7]

        has_digit_facts = any(
            re.search(r'\d{2,}|[\$€¥£]\d|\d\s*(?:kg|lb|ml|oz|cm|mm|g)\b', f.get("fact", ""))
            for f in facts
        )

        if cap.get("ocr") or has_digit_facts:
            fc["F1"].append(idx)
        if has_digit_facts:
            fc["F3"].append(idx)

        if entities:
            fc["F2"].append(idx)
        if len(entities) >= 2:
            fc["F4"].append(idx)

        if cap.get("state_changes"):
            fc["E2"].append(idx)

        if len(entities) >= 3:
            fc["S1"].append(idx)

    # E1: subsample — guarantee at least FAMILY_TARGETS["E1"] candidates
    all_chunks = [cap["chunk_idx"] for cap in evidence if cap.get("atomic_facts")]
    target_e1 = FAMILY_TARGETS.get("E1", 3)
    step = max(1, len(all_chunks) // max(target_e1 * 2, 1))
    fc["E1"] = all_chunks[::step]

    # ------------------------------------------------------------------
    # P1: procedure detection
    #   Primary: consecutive state_changes >= 3 (from 1-B)
    #   Fallback: consecutive chunks with distinct primary actions >= 3
    #             (from 1-A visible_entities[].action, direct visual field)
    # ------------------------------------------------------------------
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}

    # Primary path: 1-B state_changes
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

    # Fallback path: 1-A action diversity
    action_run = []
    seen_actions = set()
    for cap in evidence:
        action = _get_primary_action(cap)
        if action and action not in seen_actions:
            action_run.append(cap["chunk_idx"])
            seen_actions.add(action)
        else:
            if len(action_run) >= 3:
                fc["P1"].extend(action_run)
            action_run = []
            seen_actions = {action} if action else set()
            if action:
                action_run = [cap["chunk_idx"]]
            else:
                action_run = []
    if len(action_run) >= 3:
        fc["P1"].extend(action_run)

    # ------------------------------------------------------------------
    # C1/R1: cross-chunk entity tracking
    #   Primary: entity_id from 1-B alignment
    #   Fallback: desc word-overlap from 1-A (direct visual field)
    # ------------------------------------------------------------------

    # Primary path: 1-B entity_id
    entity_appearances = {}
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            eid = e.get("id", "")
            if eid and eid != "unknown":
                entity_appearances.setdefault(eid, []).append(cap["chunk_idx"])

    for eid, chunks in entity_appearances.items():
        # C1: same entity in different chunks with state_change
        state_chunks = [c for c in chunks
                        if ev_by_idx.get(c, {}).get("state_changes")]
        if len(state_chunks) >= 2:
            fc["C1"].extend(state_chunks[-2:])
        # R1: same entity reappears after gap >= 5
        for i in range(1, len(chunks)):
            if chunks[i] - chunks[i - 1] >= 5:
                fc["R1"].append(chunks[i])

    # Fallback path: 1-A desc similarity (covers 1-B entity_id misses)
    # Build desc→chunks index from 1-A direct visual field
    desc_chunks = {}  # desc_text -> [(chunk_idx, action)]
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            desc = e.get("desc", "")
            if desc and len(desc) > 5:
                desc_chunks.setdefault(desc, []).append(
                    (cap["chunk_idx"], e.get("action", ""))
                )

    # Match desc pairs with high overlap but different literal desc
    descs = list(desc_chunks.keys())
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            if _desc_overlap(descs[i], descs[j]) < 0.6:
                continue
            chunks_i = desc_chunks[descs[i]]
            chunks_j = desc_chunks[descs[j]]
            all_cidx = sorted(set(c for c, _ in chunks_i + chunks_j))

            # C1 fallback: same-looking entity with different actions
            actions_by_chunk = {}
            for c, a in chunks_i + chunks_j:
                actions_by_chunk.setdefault(c, set()).add(a)
            action_change_chunks = [c for c in all_cidx
                                    if len(actions_by_chunk.get(c, set())) >= 1]
            distinct_actions = set()
            for c in action_change_chunks:
                distinct_actions |= actions_by_chunk[c]
            if len(distinct_actions) >= 2 and len(action_change_chunks) >= 2:
                fc["C1"].extend(action_change_chunks[-2:])

            # R1 fallback: similar desc reappears after gap >= 5
            for k in range(1, len(all_cidx)):
                if all_cidx[k] - all_cidx[k - 1] >= 5:
                    fc["R1"].append(all_cidx[k])

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


async def _generate_family_cards(
    family: str, chunk_list: List[int],
    evidence: List[Dict], client, video_id: str,
) -> List[Dict]:
    """Generate cards for a single family. Called concurrently per family."""
    target_n = FAMILY_TARGETS[family]

    if family == "M1":
        ev_text = _format_evidence_for_prompt(evidence, [cap["chunk_idx"] for cap in evidence])
    else:
        ev_text = _format_evidence_for_prompt(evidence, chunk_list)

    if not ev_text.strip():
        return []

    prompt_template = FAMILY_PROMPTS.get(family)
    if not prompt_template:
        return []

    prompt = prompt_template.format(n=target_n, evidence=ev_text)

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3a", {}).get("max_tokens", 16384),
        temperature=0.7,
        request_id=f"{video_id}_3a_{family}",
    )

    cards = _parse_cards_response(raw, family, video_id)

    valid = []
    for card in cards:
        if not isinstance(card, dict) or not card.get("question"):
            continue
        card["family"] = family
        card.setdefault("answer_form", "short_exact")
        card.setdefault("visibility_type", "transient")
        if "support_chunks" not in card or not card["support_chunks"]:
            card["support_chunks"] = chunk_list[:1] if chunk_list else [0]
            card["_support_inferred"] = True
            logger.warning(
                f"  [{video_id}] 3-A {family}: "
                f"support_chunks missing, inferred {card['support_chunks']}")
        if isinstance(card["support_chunks"], int):
            card["support_chunks"] = [card["support_chunks"]]
        valid.append(card)
    return valid


async def generate_cards(
    video_id: str,
    evidence: List[Dict],
    client,
) -> List[Dict]:
    """Generate task cards for one video via per-family 397B calls.

    All family calls are independent and run concurrently via asyncio.gather.
    Returns list of TaskCard dicts.
    """
    family_chunks = classify_chunks(evidence)

    # Build tasks for families that have candidates
    tasks = []
    family_order = []
    for family, chunk_list in family_chunks.items():
        if family != "M1" and not chunk_list:
            continue
        tasks.append(_generate_family_cards(family, chunk_list, evidence, client, video_id))
        family_order.append(family)

    # Fire all family calls concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect and assign sequential card_ids
    all_cards = []
    card_counter = 0
    for family, result in zip(family_order, results):
        if isinstance(result, Exception):
            logger.error(f"  [{video_id}] 3-A {family}: call failed: {result}")
            continue
        for card in result:
            card_counter += 1
            card["card_id"] = f"{video_id}_{family}_{card_counter:03d}"
            all_cards.append(card)

    family_counts = {f: sum(1 for c in all_cards if c["family"] == f)
                     for f in FAMILY_TARGETS if any(c["family"] == f for c in all_cards)}
    counts_str = ", ".join(f"{f}:{n}" for f, n in family_counts.items())
    logger.info(f"  [{video_id}] 3-A: {len(all_cards)} cards ({counts_str})")

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


_STOP_WORDS = frozenset({
    # Articles / pronouns / prepositions / conjunctions
    "the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
    "and", "or", "it", "yes", "no", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "there", "here", "not", "but", "if", "so", "than", "then",
    "just", "about", "up", "out", "its", "his", "her", "my", "your",
    "their", "our", "me", "him", "them", "us", "we", "they",
    "you", "he", "she", "with", "for", "from", "by", "as",
    # Interrogatives / question function words — never appear in thinks
    "what", "which", "who", "whom", "whose", "how", "when", "where",
    "many", "much", "any", "some", "other", "tell", "describe",
    "currently", "now", "still", "yet", "ever", "already",
})


def extract_keywords(text: str) -> List[str]:
    """Extract content keywords from a text string, filtering stop words."""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and len(w) > 1 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _extract_mc_choice_text(question: str, answer_letter: str) -> str:
    """Extract the text of the correct MC choice from the question.

    Supports formats:
      "... A.Red B.Blue C.White D.Green"
      "... A. Red B. Blue C. White D. Green"
      "... A) Red B) Blue"
    """
    answer_letter = answer_letter.strip().upper()
    # Build pattern: "A.Red" or "A. Red" or "A) Red", capture until next choice or end
    pattern = rf'(?:^|\s){answer_letter}[\.\)]\s*(.+?)(?:\s+[B-Z][\.\)]|$)'
    m = re.search(pattern, question, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_card_keywords(card: Dict) -> List[str]:
    """Extract discriminative keywords from a card for retention matching.

    The canonical_answer alone is useless for binary ("Yes") and MC ("A").
    This function extracts keywords that would actually appear in a think
    or summary if the student observed the relevant evidence.

    Strategy by answer_form:
      binary:          question keywords (the predicate being judged)
      multiple_choice: question subject + correct choice text
      number:          question keywords + the number itself
      short_exact:     canonical_answer keywords (already informative)
      descriptive:     canonical_answer keywords (already informative)
    """
    answer_form = card.get("answer_form", "short_exact")
    question = card.get("question", "")
    canonical = card.get("canonical_answer", "")

    if answer_form == "binary":
        # "Is the apron red?" → ["apron", "red"]
        return extract_keywords(question)

    elif answer_form == "multiple_choice":
        # "What color? A.Red B.Blue C.White D.Green", answer="A"
        # → question subject ["color"] + correct choice ["red"]
        # Strip choices from question to get the subject part
        q_base = re.split(r'\s+A[\.\)]', question, maxsplit=1)[0]
        q_kw = extract_keywords(q_base)
        choice_text = _extract_mc_choice_text(question, canonical)
        c_kw = extract_keywords(choice_text)
        # Deduplicate preserving order
        seen = set()
        result = []
        for w in q_kw + c_kw:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result

    elif answer_form == "number":
        # "How many tomatoes were cut?" answer="3"
        # → ["tomatoes", "cut", "3"]
        q_kw = extract_keywords(question)
        num = canonical.strip()
        if num and num not in {kw for kw in q_kw}:
            q_kw.append(num)
        return q_kw

    else:
        # short_exact / descriptive: canonical_answer is already informative
        return extract_keywords(canonical)


# ---------------------------------------------------------------------------
# Card Verification (397B, independent per card, high concurrency)
# ---------------------------------------------------------------------------

VERIFY_CARD_PROMPT = """Verify this video question card against the source evidence.

Evidence chunks:
{evidence}

Card:
- question: "{question}"
- canonical_answer: "{canonical_answer}"
- answer_form: {answer_form}
- support_chunks: {support_chunks}
- visibility_type: {visibility_type}

Check:
1. Is the question answerable from the evidence above?
2. Does the question REQUIRE visual observation to answer? Reject pure common-sense
   questions that could be answered without watching the video (e.g. "Is water wet?").
3. Does the question text leak the answer? (e.g. "The red apron is what color?")
4. Which chunk indices actually contain the answer evidence?
5. Is the answer always visible throughout the video (persistent) or only momentarily (transient)?
6. Is canonical_answer correctly formatted?
   - binary: exactly "Yes" or "No"
   - multiple_choice: exactly one letter "A"/"B"/"C"/"D"
   - number: digits only, no units
   - short_exact: 1-5 English words, no articles

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "persistent|transient", "canonical_answer": "..."}}
If ANY check fails (not answerable, not visual-dependent, or answer leaked), output:
{{"valid": false}}"""


def _parse_verify_response(raw: Optional[str]) -> Optional[Dict]:
    """Parse verification response JSON."""
    if not raw:
        return None
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def _verify_one_card(
    card: Dict, evidence: List[Dict], client, video_id: str,
) -> Dict:
    """Verify one card independently. No cross-card dependency."""
    support = card.get("support_chunks", [])
    if not support:
        card["_verified"] = False
        return card

    # Include support chunks ± 2 for context
    search_range = set()
    for sc in support:
        for c in range(max(0, sc - 2), sc + 3):
            search_range.add(c)

    ev_text = _format_evidence_for_prompt(evidence, sorted(search_range))
    if not ev_text.strip():
        card["_verified"] = False
        return card

    prompt = VERIFY_CARD_PROMPT.format(
        evidence=ev_text,
        question=card.get("question", ""),
        canonical_answer=card.get("canonical_answer", ""),
        answer_form=card.get("answer_form", "short_exact"),
        support_chunks=card.get("support_chunks", []),
        visibility_type=card.get("visibility_type", "transient"),
    )

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3a_verify", {}).get("max_tokens", 2048),
        temperature=0.1,
        request_id=f"{video_id}_verify_{card.get('card_id', '')}",
    )

    result = _parse_verify_response(raw)
    if not result or not result.get("valid", False):
        card["_verified"] = False
        return card

    # Apply fixes from verification
    fixed_sc = result.get("support_chunks")
    if isinstance(fixed_sc, list) and fixed_sc:
        card["support_chunks"] = fixed_sc
    if result.get("visibility_type") in ("persistent", "transient"):
        card["visibility_type"] = result["visibility_type"]
    fixed_answer = result.get("canonical_answer")
    if isinstance(fixed_answer, str) and fixed_answer:
        card["canonical_answer"] = fixed_answer

    card["_verified"] = True
    return card


async def verify_cards(
    video_id: str, cards: List[Dict], evidence: List[Dict], client,
) -> List[Dict]:
    """Verify all cards concurrently. Each card is an independent 397B call.

    Drops invalid cards, fixes support_chunks/visibility_type/canonical_answer.
    Returns only verified cards.
    """
    if not cards:
        return []

    tasks = [_verify_one_card(card, evidence, client, video_id)
             for card in cards]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    dropped = 0
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"  [{video_id}] verify card failed: {result}")
            dropped += 1
            continue
        if result.get("_verified", False):
            verified.append(result)
        else:
            dropped += 1

    logger.info(f"  [{video_id}] 3-A verify: {len(verified)} passed, {dropped} dropped")
    return verified
