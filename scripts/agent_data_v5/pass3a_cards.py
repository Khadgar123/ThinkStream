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

# Family targets per video.
# v9.0: F5/F6/N1 added for OVOBench REC/FPD/HLD coverage.
FAMILY_TARGETS = {
    "F1": 3, "F2": 4, "F3": 2, "F4": 2,
    "E1": 3, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 2,
    "F5": 2,  # repetition counting (OVO REC)
    "F6": 2,  # future prediction (OVO FPD)
    "N1": 2,  # hallucination negative (OVO HLD)
}

# Families that MUST be attempted on every video, even when classify_chunks
# returns zero chunks for them. v9.1 audit found F5=4 cards across 312 videos
# because most videos lack a 3-chunk same-action run; here we let the teacher
# look at the whole video and decide whether it can construct a question.
# Without this, OVOBench REC/FPD/HLD coverage is structurally absent.
FAMILY_FORCE_ATTEMPT = {"F5", "F6", "N1", "F3", "E2", "S1"}

# Retention class derived from family (not from 397B)
RETENTION_CLASS = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
    "F5": "low",      # exact count must persist
    "F6": "medium",   # process-aware
    "N1": "low",      # specific entity absence
}

# Families whose canonical_answer="No" indicates expected silent/refusal.
# Used by GRPO silent_quality reward and SFT label routing.
NEGATIVE_FAMILIES = {"N1"}

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
Question format: "Describe what is happening" / "What entities are present and what are they doing?"
answer_form: descriptive.
visibility_type: "persistent".

ANSWER FORMAT (strict): canonical_answer must be 30-80 words containing
3-5 SPECIFIC observations (entity descriptions + actions + spatial layout).
DO NOT write meta-commentary like "the video shows" or "we can see".
Each observation must be a concrete visual fact, not interpretation.

{evidence}
""" + _OUTPUT_SCHEMA,

    "M1": """Based on this full video summary, generate {n} questions suitable
for continuous commentary (e.g., "Describe each step as it happens").
answer_form: descriptive.
visibility_type: "transient".

ANSWER FORMAT (strict): canonical_answer must be 30-80 words describing
3-5 distinct moments/steps with timestamps when visible. Format like:
"At [t1], X happens; at [t2], Y begins; ..."
DO NOT write meta narration. Each clause must reference a concrete visible event.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F5": """Based on the following video chunks, generate UP TO {n} questions about
ACTION REPETITION COUNT (OVOBench REC).

The chunks below come from the SAME video. Your job is to SCAN them and find
any action that visibly repeats — a person doing the same gesture/movement
on multiple distinct occasions, an object being struck/poured/wiped multiple
times, a recurring step in a procedure, etc. The repetitions need NOT be in
adjacent chunks; they can be spaced out across the timeline.

Question format: "How many times did the person {{verb}}?" or
"How many {{action}} occurrences appear in the video?"
answer_form: number (digits only, e.g. "3").
visibility_type: "transient" (count is only complete after the last repetition).
support_chunks MUST list EVERY chunk where the repetition is observed
(at least 2 chunks; ideally 3+ for a meaningful count).

CRITICAL — output an EMPTY JSON array `[]` ONLY IF you genuinely cannot find
any repeating action across the chunks. If you find even ONE repetition pair,
generate a question for it. Quality > quantity but DO try.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F6": """Based on the following video chunks, generate UP TO {n} questions about
FUTURE PREDICTION (OVOBench FPD).

The chunks below come from the SAME video, ordered by time. SCAN them for
any moment where an in-progress process strongly implies what happens next
(e.g. "person grabs knife and a tomato" → next step is cutting; "pours batter
into pan on stove" → next step is cooking/setting). The predicted event must
be observable in a LATER chunk so we can verify the gold answer.

Question format: "What will the person do next?" or "What is about to happen?"
Prefer answer_form: multiple_choice (4 plausible options, one is the actual
continuation; distractors are plausible alternatives from the same domain —
other steps in the procedure or visually adjacent objects).
visibility_type: "transient".
support_chunks: the chunks BEFORE the predicted event (the setup), NOT the
chunks where the event actually happens.

CRITICAL — output an EMPTY JSON array `[]` ONLY IF no chunk pair in the
evidence shows a setup → continuation relationship you are confident about.
If the continuation is genuinely ambiguous, SKIP that case — do NOT fabricate.
A wrong future-prediction gold answer poisons training. Quality > quantity.
But DO scan thoroughly: a typical 2-3 minute procedural video has 1-3 valid
prediction points.

{evidence}
""" + _OUTPUT_SCHEMA,

    "N1": """Based on the following video chunks, generate {n} HALLUCINATION-NEGATIVE
questions (OVOBench HLD).
The question must ask about an entity, action, or attribute that is SEMANTICALLY PLAUSIBLE
for this video genre but DOES NOT actually appear in the video.
Examples:
- Video shows kitchen cooking → "Is there a microwave being used?" (No, only stovetop)
- Video shows outdoor sports → "Is the person wearing a helmet?" (No, no helmet visible)
Question format: "Is/Does ...?" or "What color is the X?" where X never appears.
answer_form: binary (canonical_answer MUST be "No"), or short_exact ("not present"/"never appears").
visibility_type: "persistent" (absence holds throughout).
support_chunks: pick 2-3 representative chunks that show the SCENE CONTEXT (not the absent entity).

CRITICAL: the asked-about entity/action/attribute must NOT appear in any chunk's evidence above.

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


_ACTION_HEAD_RE = re.compile(r"^([a-zA-Z]+)")


def _action_verb_lemma(action: str) -> str:
    """Extract the head verb from an action phrase (lemma-ish, no NLP dep).

    "stirring the pot" → "stir"; "stirs vigorously" → "stir"; "stirred" → "stir".
    Used to compare actions across chunks for F5 (repetition) without being
    tripped by tense / objects / adverbs. Handles the doubled-consonant
    variant ("stirring" → "stirr" → "stir") which is common in cooking
    captions. Does NOT handle silent-e drop ("hated" → "hat" not "hate"),
    but as long as both chunks use the same tense the comparison still works.
    """
    if not action:
        return ""
    a = action.lower().strip()
    m = _ACTION_HEAD_RE.match(a)
    if not m:
        return a
    head = m.group(1)
    # Strip a single inflectional suffix, longest first.
    for suf, doubled in (("ing", True), ("ed", True), ("s", False)):
        if head.endswith(suf) and len(head) > len(suf) + 2:
            stem = head[: -len(suf)]
            # "stirring" → "stirr" → "stir": collapse doubled trailing
            # consonant introduced by -ing / -ed gerund formation.
            if (doubled and len(stem) >= 2
                    and stem[-1] == stem[-2]
                    and stem[-1] not in "aeiou"):
                stem = stem[:-1]
            return stem
    return head


def _get_primary_action(cap: Dict) -> str:
    """Extract the primary entity action from a chunk (1-A direct field).

    Returns the lemma-ish head verb so consecutive chunks of "stirring" /
    "stirs" / "stirred" all match for F5 repetition detection.
    """
    for e in cap.get("visible_entities", []):
        action = e.get("action", "")
        if action:
            return _action_verb_lemma(action)
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

    # ------------------------------------------------------------------
    # F5: action repetition count (OVO REC)
    #   Detect runs of SAME primary_action across >= 3 consecutive chunks.
    #   Distinct from P1 which detects runs of DIFFERENT actions.
    # ------------------------------------------------------------------
    rep_run = []
    rep_action = ""
    for cap in evidence:
        action = _get_primary_action(cap)
        if action and action == rep_action:
            rep_run.append(cap["chunk_idx"])
        else:
            if len(rep_run) >= 3:
                fc["F5"].extend(rep_run)
            rep_run = [cap["chunk_idx"]] if action else []
            rep_action = action
    if len(rep_run) >= 3:
        fc["F5"].extend(rep_run)

    # ------------------------------------------------------------------
    # F6: future prediction (OVO FPD)
    #   Pick chunks that (a) have state_changes or sequential action, AND
    #   (b) have at least 2 chunks of evidence remaining after them
    #   so the predicted continuation is observable.
    # ------------------------------------------------------------------
    if evidence:
        max_idx = max(cap.get("chunk_idx", 0) for cap in evidence)
        for cap in evidence:
            idx = cap.get("chunk_idx", 0)
            if idx > max_idx - 2:
                continue  # need future evidence
            if cap.get("state_changes") or _get_primary_action(cap):
                fc["F6"].append(idx)

    # ------------------------------------------------------------------
    # N1: hallucination negative (OVO HLD)
    #   Pick chunks with rich entity context (>=2 entities) so the teacher
    #   has enough scene grounding to construct plausible-but-absent
    #   negative questions. Distribute across the video timeline.
    # ------------------------------------------------------------------
    n1_candidates = [
        cap["chunk_idx"] for cap in evidence
        if len(cap.get("visible_entities", [])) >= 2
    ]
    n1_target = FAMILY_TARGETS.get("N1", 2) * 3  # 3x oversample for teacher selection
    if n1_candidates:
        step = max(1, len(n1_candidates) // max(n1_target, 1))
        fc["N1"] = n1_candidates[::step][:n1_target]

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

    # Whole-video fallback chunk list for families in FAMILY_FORCE_ATTEMPT
    # (when their structural classification yields nothing). We sample evenly
    # across the timeline so the teacher sees a representative cross-section.
    # Note: F5/F6 need to *scan* a wide span to find repetition / process
    # setups, so this fallback set caps at the prompt's 10-chunk format limit
    # (see _format_evidence_for_prompt) but spans the whole video.
    all_chunk_idxs = [cap["chunk_idx"] for cap in evidence]
    fallback_chunks: List[int] = []
    if all_chunk_idxs:
        n_total = len(all_chunk_idxs)
        step = max(1, n_total // 10)  # 10 evenly-spaced chunks
        fallback_chunks = sorted(set(all_chunk_idxs[::step]))[:10]

    # Build tasks for families that have candidates OR are force-attempt
    tasks = []
    family_order = []
    for family in FAMILY_TARGETS:
        chunk_list = family_chunks.get(family, [])
        if not chunk_list and family != "M1":
            if family in FAMILY_FORCE_ATTEMPT and fallback_chunks:
                # Use whole-video fallback so teacher can decide if a question
                # is possible (e.g. F5 needs ≥3 same-action chunks somewhere).
                chunk_list = fallback_chunks
            else:
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
    from .cache_version import stage_version_ok
    if not stage_version_ok("3a"):
        return None
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

# N1 (hallucination) needs an inverted check: the asked-about entity must NOT
# appear in the evidence. Standard verify would mark these "not answerable".
VERIFY_N1_PROMPT = """Verify this HALLUCINATION-NEGATIVE question card.
The card claims the asked-about entity/action/attribute is ABSENT from the video.

Evidence chunks (ALL chunks of the video, or a representative subset):
{evidence}

Card:
- question: "{question}"
- canonical_answer: "{canonical_answer}"  (must be "No" or equivalent absence statement)

Check:
1. Does the question ask about something specific (entity / action / attribute)?
2. Is that thing genuinely ABSENT from every chunk of evidence above? (If it appears
   in any chunk, the card is INVALID — answer should be Yes, not No.)
3. Is the asked-about thing semantically plausible for this video genre (otherwise
   the negative is too easy and not discriminative)?
4. Is canonical_answer exactly "No" (binary) or a clear absence phrase?

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "persistent", "canonical_answer": "No"}}
If the asked-about thing actually appears, or the question is trivial, output:
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

    family = card.get("family", "")
    is_negative = family in NEGATIVE_FAMILIES

    if is_negative:
        # N1: verify against the WHOLE video (or a wide sample), not just
        # support_chunks ± 2 — we need to confirm absence everywhere.
        all_idx = sorted(cap.get("chunk_idx", 0) for cap in evidence)
        # Sample up to 12 evenly-spaced chunks for context
        if len(all_idx) > 12:
            step = len(all_idx) // 12
            sampled = all_idx[::step][:12]
        else:
            sampled = all_idx
        ev_text = _format_evidence_for_prompt(evidence, sampled)
    else:
        # Include support chunks ± 2 for context
        search_range = set()
        for sc in support:
            for c in range(max(0, sc - 2), sc + 3):
                search_range.add(c)
        ev_text = _format_evidence_for_prompt(evidence, sorted(search_range))

    if not ev_text.strip():
        card["_verified"] = False
        return card

    if is_negative:
        prompt = VERIFY_N1_PROMPT.format(
            evidence=ev_text,
            question=card.get("question", ""),
            canonical_answer=card.get("canonical_answer", ""),
        )
    else:
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
    num_chunks = max((cap.get("chunk_idx", 0) for cap in evidence), default=0) + 1
    fixed_sc = result.get("support_chunks")
    if isinstance(fixed_sc, list) and fixed_sc:
        # Bounds check: discard out-of-range chunk indices
        fixed_sc = [c for c in fixed_sc if isinstance(c, int) and 0 <= c < num_chunks]
        if not fixed_sc:
            logger.warning(f"  [{video_id}] verify {card.get('card_id')}: "
                           f"all support_chunks out of range, dropping card")
            card["_verified"] = False
            return card
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
