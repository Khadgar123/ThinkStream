"""Heuristic card generator from evidence_1b — used by simulator.

Production pass3a will use 397B + the new prompts (TBD), but for distribution
analysis we derive cards deterministically from evidence so the simulator
runs without LLM calls. The family / answer_form / question_type mix below
is calibrated to roughly match what 397B's pass3a was producing in v9.5.

Schema produced matches v2.design.Card.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

from ..stable_hash import stable_mod, stable_seed
from .design import (
    CRR1_ADOPT_RATE,
    Card,
    F5_ADOPT_RATE,
    F7_ADOPT_RATE,
    GoldEmit,
    MULTI_EMIT_ADOPT_RATE,
    PN1_ADOPT_RATE,
)

MC_OPTION_LETTERS = "ABCDE"


def _ovo_option_count(*keys) -> int:
    bucket = stable_mod(*keys, "OPTION_COUNT", modulo=100)
    if bucket < 3:
        return 2
    if bucket < 6:
        return 3
    if bucket < 7:
        return 5
    return 4


# Family taxonomy aligned with OVOBench (MC-dominant) + 3-bucket profile.
# Total target ~14 cards/video (was 20.9) — gives the trajectory selector
# a small headroom over MAX_QUESTIONS_PER_TRAJECTORY for diversity scoring
# without wasting LLM budget on cards that get dropped.
#
# Bucket allocation (matches PLACEMENT_PROFILE in design.py):
#   backward (recall-heavy):  N1, P1, HLD1, CR1, CR2, CR4, CR5, M1
#   forward  (silent-then-respond): E2
#   realtime (immediate):     CR3, CR7, R1, ACR1, STU1, OJR1, C1
#   streaming multi_emit:     F5, F7, CRR1, PN1
#   ────────────────────────────────────────────────────────────────
#   total target                                                     20 cards
FAMILY_BUDGET = {
    # backward MC
    "N1":  1, "P1":  1, "HLD1": 1, "CR1": 1, "CR2": 1, "CR4": 1, "CR5": 1,
    # forward MC + binary
    "E2":  1, "F6":  1, "F7":  1, "CRR1": 1,
    # realtime MC + number + short_exact
    "CR3": 1, "CR7": 1, "R1":  1, "ACR1": 1, "STU1": 1, "OJR1": 1,
    "F5":  1, "C1":  2,
    # multi_emit (PN1 50% adopt; F5 already realtime above)
    "PN1": 1,
    # backward descriptive
    "M1":  1,
}

MC_FAMILIES = {"N1", "P1", "CR1", "CR2", "CR3", "CR4", "CR5", "CR7",
               "E2", "F6", "R1", "ACR1", "STU1", "OJR1"}


def _hash_id(*parts) -> str:
    s = "|".join(str(p) for p in parts)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _canonical_short(text: str, max_words: int = 6) -> str:
    """Trim a fact to a short answer."""
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip().rstrip(".")
    return " ".join(words[:max_words]).rstrip(".")


def _entity_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """Map entity_id → sorted list of chunks it appears in."""
    out: Dict[str, List[int]] = {}
    for cap in evidence:
        c = cap.get("chunk_idx", 0)
        for ent in cap.get("visible_entities", []) or []:
            eid = ent.get("id") or ent.get("desc", "")[:30]
            if not eid:
                continue
            out.setdefault(eid, []).append(c)
    return {k: sorted(set(v)) for k, v in out.items()}


def _state_change_chunks(evidence: List[Dict]) -> List[Tuple[int, str]]:
    """Return [(chunk, state_change_text), ...]."""
    out = []
    for cap in evidence:
        for sc in cap.get("state_changes", []) or []:
            sc_text = sc if isinstance(sc, str) else (sc.get("text") or sc.get("change") or str(sc))
            out.append((cap.get("chunk_idx", 0), sc_text))
    return out


def _ocr_chunks(evidence: List[Dict]) -> List[Tuple[int, str]]:
    """Collect (chunk_idx, ocr_text) pairs.

    v12.12 (2026-05-02): pass1a now emits OCR as
    `[{"text": "EXIT", "confidence": 0.95}, ...]` (structured dicts).
    Earlier evidence may have plain strings. Handle both.
    """
    out = []
    for cap in evidence:
        for o in cap.get("ocr", []) or []:
            if not o:
                continue
            # Structured: {"text": "...", "confidence": ...}
            if isinstance(o, dict):
                text = o.get("text", "")
            else:
                text = str(o)
            if text and text.strip():
                out.append((cap.get("chunk_idx", 0), text.strip()))
    return out


def _evidence_blob(evidence: List[Dict]) -> str:
    parts: List[str] = []
    for cap in evidence:
        for e in (cap.get("visible_entities") or []):
            parts.append(str(e.get("desc", "")))
            parts.append(str(e.get("id", "")))
        for f in (cap.get("atomic_facts") or []):
            if isinstance(f, dict):
                parts.append(str(f.get("fact", "")))
        for _c, text in _ocr_chunks([cap]):
            parts.append(text)
        if cap.get("think"):
            parts.append(str(cap.get("think", "")))
    return " ".join(parts).lower()


def _cap_blob(cap: Dict) -> str:
    return _evidence_blob([cap])


# ---------------------------------------------------------------------------
# Per-family generators
# ---------------------------------------------------------------------------


def gen_pn1_narration(evidence: List[Dict], video_id: str) -> List[Card]:
    """One PN1 card per video. Emits at state_change chunks ONLY.

    Density target: LiveCC sits at ~1 narration/12s → for a 1s/chunk video
    that's 1 emit per 12 chunks. PN1 is for "real-time awareness" training,
    not for OVOBench QA; OVOBench is dominated by MC questions which are
    handled by single_emit families below. PN1 should stay sparse so it
    doesn't dominate the response budget.
    """
    event_chunks: List[Tuple[int, str]] = []
    last = -100
    for cap in evidence:
        c = cap.get("chunk_idx", 0)
        if not cap.get("state_changes"):
            continue
        if c - last < 12:                    # was 6 → tightened to LiveCC density
            continue
        change = cap.get("state_changes")[0]
        if isinstance(change, dict):
            change_text = change.get("text") or change.get("change") or str(change)
        else:
            change_text = str(change)
        event_chunks.append((c, change_text))
        last = c
    if not event_chunks:
        return []
    # Hard cap: never more than 6 narration emits per video, regardless of
    # video length. Beyond 6 the model just learns "narrate every 12s" which
    # isn't what OVOBench measures.
    event_chunks = event_chunks[:6]
    emits = [GoldEmit(chunk=c, value=text) for c, text in event_chunks]
    return [Card(
        card_id=f"{video_id}_PN1_{_hash_id(video_id, 'PN1')}",
        family="PN1",
        question="Describe each important event as it happens.",
        answer_form="descriptive",
        question_type="multi_emit",
        gold_emits=emits,
        grounding_frames=[c for c, _text in event_chunks],
    )]


def _action_repetition_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """v12.13 (P1-7): detect repeated ACTIONS (not entity appearances).

    OVO Bench REC = action repetition counting (e.g. "how many times does
    the person wave?"). Looks at atomic_facts and groups by lemmatized
    verb-phrase. A "repetition" requires:
      - same verb-phrase fires in NON-ADJACENT chunks (gap ≥ 2),
        otherwise it's one continuous action being re-described
      - confidence ≥ 0.7 to drop noise
      - ≥ 3 distinct occurrences and span ≥ 6 chunks (real countable event)

    Returns {action_phrase: [chunk_idx, ...]}.
    """
    from collections import defaultdict
    import re as _re

    # Simple verb-phrase normalization: take first 3 tokens of fact text,
    # lowercase, strip punctuation. Skips common cluttering words.
    STOPS = {"the", "a", "an", "is", "are", "was", "were", "of"}
    def normalize(text: str) -> str:
        if not text:
            return ""
        toks = _re.findall(r"[a-zA-Z]+", text.lower())
        toks = [t for t in toks if t not in STOPS]
        return " ".join(toks[:3])    # first 3 content words

    by_action: Dict[str, List[int]] = defaultdict(list)
    for cap in evidence:
        for f in (cap.get("atomic_facts") or []):
            if not isinstance(f, dict):
                continue
            if f.get("confidence", 0) < 0.7:
                continue
            phrase = normalize(f.get("fact", ""))
            if len(phrase) < 5:    # need at least one verb + object
                continue
            by_action[phrase].append(int(cap.get("chunk_idx", 0)))

    # Filter by repetition criteria
    out: Dict[str, List[int]] = {}
    for phrase, chunks in by_action.items():
        chunks = sorted(set(chunks))
        # Drop adjacent-only repetitions (same continuous event being re-described)
        non_adjacent = [chunks[0]]
        for c in chunks[1:]:
            if c - non_adjacent[-1] >= 2:
                non_adjacent.append(c)
        if (len(non_adjacent) >= 3
                and (non_adjacent[-1] - non_adjacent[0]) >= 6):
            out[phrase] = non_adjacent
    return out


def gen_f5_counting(evidence: List[Dict], video_id: str) -> List[Card]:
    """F5 cards: ACTION repetition counting (OVO REC alignment, v12.13).

    Old behavior: counted entity appearances across chunks — meaningless
    because entities persist (a person visible in chunks 1-50 isn't "appearing
    50 times"). Aligns with OVO REC family which asks "how many times does
    [action] happen?" and expects cumulative counts at each occurrence.

    multi_emit: at each occurrence chunk, model emits the running count
    (1, 2, 3, ...). Reward (P0-2) scores per-emit with these per-chunk golds.
    """
    actions = _action_repetition_chunks(evidence)
    cards = []
    for phrase, chunks in actions.items():
        occurrences = chunks[:6]    # cap at 6 emits to keep per-emit window tractable
        emits = [GoldEmit(chunk=c, value=str(i + 1))
                 for i, c in enumerate(occurrences)]
        cards.append(Card(
            card_id=f"{video_id}_F5_{_hash_id(video_id, phrase)}",
            family="F5",
            question=(
                f"How many times does \"{phrase}\" happen so far in the video?"
            ),
            answer_form="number",
            question_type="multi_emit",
            gold_emits=emits,
            grounding_frames=occurrences,
        ))
        if len(cards) >= FAMILY_BUDGET["F5"]:
            break
    return cards


def gen_f7_status_flip(evidence: List[Dict], video_id: str) -> List[Card]:
    """F7 cards: real-time Yes/No status (OVO SSR alignment, v12.13).

    Old behavior: single_emit at change chunk with gold "Yes". This was
    "wait silent until event happens, then answer Yes once" — that's a
    forward task, not OVO SSR. SSR ("Same Sample Reasoning" / Status
    Reasoning) expects the model to answer Yes/No at MULTIPLE chunks in
    the trajectory, with the gold flipping at the change point.

    New: multi_emit binary card. Asks "Has X happened?" at every chunk
    in [change_chunk - K, change_chunk + K]:
      - chunks BEFORE change: gold = "No"
      - chunks AT or AFTER change: gold = "Yes"
    Per-emit reward (P0-2) scores each chunk's Yes/No against its gold.

    K is small (default 4) so the multi_emit doesn't span too many chunks
    (= many parallel pending queries simultaneously).
    """
    scs = _state_change_chunks(evidence)
    if not evidence:
        return []
    n_chunks = max((c.get("chunk_idx", 0) for c in evidence), default=0) + 1
    cards = []
    K = 4    # window radius around change chunk
    for c, text in scs[:FAMILY_BUDGET["F7"]]:
        if c < 1:
            continue
        lo = max(0, c - K)
        hi = min(n_chunks - 1, c + K)
        if hi - lo < 4:    # need ≥ 5 chunks for multi_emit to be meaningful
            continue
        # Build per-chunk emits: "No" before change, "Yes" at and after.
        emits = []
        for ci in range(lo, hi + 1):
            value = "No" if ci < c else "Yes"
            emits.append(GoldEmit(chunk=ci, value=value))
        values = {e.value for e in emits}
        if not {"No", "Yes"}.issubset(values):
            continue
        cards.append(Card(
            card_id=f"{video_id}_F7_{_hash_id(video_id, c, text)}",
            family="F7",
            question=f"Has \"{text[:40]}\" happened by now?",
            answer_form="binary",
            question_type="multi_emit",   # was single_emit
            gold_emits=emits,
            grounding_frames=[c],
        ))
    return cards


def gen_crr_event_status(evidence: List[Dict], video_id: str) -> List[Card]:
    """CRR1 cards: repeated Yes/No probes around an event becoming true."""
    scs = _state_change_chunks(evidence)
    if not evidence:
        return []
    n_chunks = max((c.get("chunk_idx", 0) for c in evidence), default=0) + 1
    cards = []
    for c, text in scs:
        if c < 8 or n_chunks - c < 2:
            continue
        pre_far = max(0, c - min(96, max(10, c // 2)))
        pre_near = max(0, c - min(12, max(4, c // 4)))
        post_near = min(n_chunks - 1, c + 2)
        post_far = min(
            n_chunks - 1,
            c + min(96, max(24, (n_chunks - 1 - c) // 2)),
        )
        probe_chunks = sorted({pre_far, pre_near, c, post_near, post_far})
        emits = [
            GoldEmit(chunk=pc, value=("No" if pc < c else "Yes"))
            for pc in probe_chunks
        ]
        values = {e.value for e in emits}
        if len(emits) < 3 or not {"No", "Yes"}.issubset(values):
            continue
        event = str(text or "").strip().rstrip(".")
        if not event:
            continue
        cards.append(Card(
            card_id=f"{video_id}_CRR1_{_hash_id(video_id, c, event)}",
            family="CRR1",
            question=f"Has \"{event[:70]}\" happened yet?",
            answer_form="binary",
            question_type="multi_emit",
            gold_emits=emits,
            grounding_frames=[c],
        ))
        if len(cards) >= FAMILY_BUDGET["CRR1"]:
            break
    return cards


def gen_hld_unanswerable(evidence: List[Dict], video_id: str) -> List[Card]:
    """HLD1 cards: explicit "Unable to answer" MC negatives.

    These are answerable only by abstaining. The requested object is chosen
    from a conservative pool and filtered against the evidence text so the
    generator does not accidentally ask about something that was observed.
    """
    if not evidence:
        return []
    blob = _evidence_blob(evidence)
    # Use neutral object names. Asking "what color was the red umbrella" leaks
    # a color through the question itself; HLD1 must be unanswerable because the
    # subject is absent, not because the annotation ignored visible evidence.
    absent_pool = [
        ("umbrella", ["umbrella"]),
        ("backpack", ["backpack"]),
        ("delivery truck", ["delivery", "truck"]),
        ("exit sign", ["exit", "sign"]),
        ("laptop", ["laptop"]),
        ("safety helmet", ["safety", "helmet"]),
        ("suitcase", ["suitcase"]),
        ("cardboard package", ["cardboard", "package"]),
        ("bicycle basket", ["bicycle", "basket"]),
        ("parking meter", ["parking", "meter"]),
        ("mailbox", ["mailbox"]),
        ("fire extinguisher", ["extinguisher"]),
        ("shopping cart", ["shopping", "cart"]),
        ("tripod", ["tripod"]),
        ("remote control", ["remote", "control"]),
        ("coffee mug", ["coffee", "mug"]),
        ("tennis racket", ["tennis", "racket"]),
        ("guitar case", ["guitar", "case"]),
        ("traffic cone", ["traffic", "cone"]),
        ("water bottle", ["water", "bottle"]),
    ]
    absent = ""
    for candidate, tokens in absent_pool:
        if all(t not in blob for t in tokens):
            absent = candidate
            break
    if not absent:
        return []

    option_pool = [
        "Blue", "Red", "Green", "Yellow", "Black", "White", "Orange",
        "Purple", "Pink", "Brown", "Gray", "Silver", "Gold", "Turquoise",
        "Magenta", "Cyan", "Violet", "Maroon", "Beige", "Ivory",
        "Navy", "Teal", "Lavender",
    ]
    option_count = _ovo_option_count(video_id, "HLD1")
    distractor_count = option_count - 1
    # Prefer options absent from the whole video evidence. With the larger pool
    # this normally succeeds; if not, fall back to support-local absence below.
    options = [o for o in option_pool if o.lower() not in blob][:distractor_count]
    if len(options) < distractor_count:
        options = option_pool[:distractor_count]

    chunks = [
        int(cap.get("chunk_idx", 0)) for cap in evidence
        if (
            cap.get("visible_entities") or cap.get("atomic_facts") or cap.get("ocr")
        )
        and not any(o.lower() in _cap_blob(cap) for o in options)
    ]
    if not chunks:
        chunks = [
            int(cap.get("chunk_idx", 0)) for cap in evidence
            if (cap.get("visible_entities") or cap.get("atomic_facts") or cap.get("ocr"))
        ]
    if not chunks:
        chunks = [int(evidence[-1].get("chunk_idx", 0))]
    chunks = sorted(set(chunks))
    if len(chunks) > 6:
        step = max(1, len(chunks) // 6)
        grounding = chunks[::step][:6]
    else:
        grounding = chunks
    emit_chunk = max(grounding)

    correct_pos = MC_OPTION_LETTERS[stable_mod(video_id, "HLD1", modulo=option_count)]
    options.insert(ord(correct_pos) - ord("A"), "Unable to answer")
    opts_with_letter = [f"{chr(65+j)}) {o}" for j, o in enumerate(options)]
    return [Card(
        card_id=f"{video_id}_HLD1_{_hash_id(video_id, absent)}",
        family="HLD1",
        question=f"What color was the {absent} in the video?",
        answer_form="multiple_choice",
        question_type="single_emit",
        gold_emits=[GoldEmit(chunk=emit_chunk, value=correct_pos)],
        grounding_frames=grounding,
        options=opts_with_letter,
        correct_option=correct_pos,
    )]


def gen_ocr(evidence: List[Dict], video_id: str) -> List[Card]:
    """C1 cards: OVO-style MC questions about visible OCR text."""
    ocr = _ocr_chunks(evidence)
    cards = []
    if not ocr:
        return []
    pool = []
    for _c, text in ocr:
        val = _canonical_short(text, max_words=4)
        if val and val not in pool:
            pool.append(val)
    generic = ["OPEN", "MENU", "EXIT", "START", "SALE", "STOP", "INFO"]
    for g in generic:
        if g.lower() not in {p.lower() for p in pool}:
            pool.append(g)
    for i, (c, text) in enumerate(ocr[:FAMILY_BUDGET["C1"]]):
        ans = _canonical_short(text, max_words=4)
        if not ans:
            continue
        candidates_d = [p for p in pool if p.strip().lower() != ans.strip().lower()]
        if len(candidates_d) < 3:
            continue
        chunk_rng = random.Random(_hash_id(video_id, "C1", c, text))
        distractors = chunk_rng.sample(candidates_d, 3)
        correct_pos = MC_OPTION_LETTERS[
            (i + stable_mod(video_id, "C1", modulo=4)) % 4
        ]
        options = list(distractors)
        options.insert(ord(correct_pos) - ord("A"), ans)
        options = options[:4]
        opts_with_letter = [f"{chr(65+j)}) {o}" for j, o in enumerate(options)]
        emits = [GoldEmit(chunk=c, value=correct_pos)]
        cards.append(Card(
            card_id=f"{video_id}_C1_{_hash_id(video_id, c, text)}",
            family="C1",
            question="Which exact text is visible in the scene?",
            answer_form="multiple_choice",
            question_type="single_emit",
            gold_emits=emits,
            grounding_frames=[c],
            options=opts_with_letter,
            correct_option=correct_pos,
        ))
    return cards


def gen_mc_card(
    evidence: List[Dict],
    video_id: str,
    family: str,
    budget: int,
    rng: random.Random,
) -> List[Card]:
    """Generic MC generator for OVOBench-style families.

    Picks `budget` chunks (stratified across video timeline), produces
    one MC card per chunk with rotated correct option for
    dataset-level balance.
    """
    candidates = []
    for cap in evidence:
        c = cap.get("chunk_idx", 0)
        ents = cap.get("visible_entities") or []
        facts = [f for f in (cap.get("atomic_facts") or [])
                 if isinstance(f, dict) and f.get("confidence", 0) >= 0.7]
        if not (ents or facts):
            continue
        candidates.append((c, ents, facts))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    n = len(candidates)
    bins = []
    seen = set()
    for i in range(budget):
        idx = (i * n) // budget + (n // (budget * 2))
        idx = min(idx, n - 1)
        if candidates[idx][0] not in seen:
            bins.append(candidates[idx])
            seen.add(candidates[idx][0])

    # v12.12 fix (P0-3): collect distractor pool from OTHER chunks' entities
    # / facts so the heuristic can emit plausible MC choices instead of the
    # placeholder string "distractor placeholder" (which made every card a
    # giveaway: 3/4 options are obviously wrong, MC reduces to "pick the
    # only non-placeholder text"). Fallback to short_exact when the pool
    # is too thin to pick 3 unique distractors.
    distractor_pool: List[str] = []
    for cap in evidence:
        for e in (cap.get("visible_entities") or []):
            d = e.get("desc", "")
            if d:
                distractor_pool.append(d[:40])
        for f in (cap.get("atomic_facts") or []):
            if isinstance(f, dict) and f.get("confidence", 0) >= 0.7:
                txt = _canonical_short(f.get("fact", ""), 6)
                if txt:
                    distractor_pool.append(txt)
    # de-dup while preserving order
    seen_d = set()
    distractor_pool = [
        x for x in distractor_pool if not (x in seen_d or seen_d.add(x))
    ]

    rotation = list(MC_OPTION_LETTERS)
    cards = []
    for i, (c, ents, facts) in enumerate(bins):
        if family in {"ACR1", "STU1", "OJR1", "R1", "CR1", "CR3", "CR4"} and facts:
            correct_text = _canonical_short(facts[0].get("fact", ""), 6)
        elif ents:
            correct_text = ents[0].get("desc", "entity")[:40]
        else:
            correct_text = _canonical_short(facts[0].get("fact", ""), 6)
        if not correct_text:
            continue

        option_count = _ovo_option_count(video_id, family, c)
        family_offset = stable_mod(video_id, family, modulo=option_count)

        # Sample distractors that aren't the correct answer.
        candidates_d = [d for d in distractor_pool
                        if d.strip().lower() != correct_text.strip().lower()]
        if len(candidates_d) < option_count - 1:
            # Heuristic pool too thin → fall back to short_exact (entity name)
            # so reward+eval stay valid (no MC letter without real options).
            cards.append(Card(
                card_id=f"{video_id}_{family}_{_hash_id(video_id, family, c)}",
                family=family,
                question=f"[{family}] What is the key visible entity in the scene?",
                answer_form="short_exact",
                question_type="single_emit",
                gold_emits=[GoldEmit(chunk=c, value=correct_text)],
                grounding_frames=[c],
                options=None,
                correct_option=None,
            ))
            continue

        # Deterministic distractor pick by chunk hash (stable across runs)
        chunk_rng = random.Random(_hash_id(video_id, family, c))
        distractors = chunk_rng.sample(candidates_d, option_count - 1)
        correct_pos = rotation[(i + family_offset) % option_count]
        options = list(distractors)
        options.insert(ord(correct_pos) - ord("A"), correct_text)
        options = options[:option_count]
        opts_with_letter = [f"{chr(65+j)}) {o}" for j, o in enumerate(options)]
        emits = [GoldEmit(chunk=c, value=correct_pos)]
        question_by_family = {
            "N1": "Which entity is visible in the relevant moment?",
            "P1": "Which visual attribute or state is shown in the relevant moment?",
            "CR1": "Which visible fact best explains what is happening?",
            "CR2": "Which event is observed in the relevant moment?",
            "CR3": "What is the actor most likely doing?",
            "CR4": "Which observation helps connect the events?",
            "CR5": "Which clue is visible in the relevant moment?",
            "CR7": "Which tracked object or location is visible?",
            "E2": "Which event becomes visible next?",
            "F6": "Which state becomes visible next?",
            "R1": "Which statement is true of the scene?",
            "ACR1": "What action is visible?",
            "STU1": "Which spatial, count, or direction statement is visible?",
            "OJR1": "Which object relation is visible?",
        }
        cards.append(Card(
            card_id=f"{video_id}_{family}_{_hash_id(video_id, family, c)}",
            family=family,
            question=question_by_family.get(family, "What is visible in the scene?"),
            answer_form="multiple_choice",
            question_type="single_emit",
            gold_emits=emits,
            grounding_frames=[c],
            options=opts_with_letter,
            correct_option=correct_pos,
        ))
    return cards


def gen_m1_summary(evidence: List[Dict], video_id: str) -> List[Card]:
    """M1: one big summary card. single_emit at the last evidence chunk."""
    if not evidence:
        return []
    chunks_with_facts = [c.get("chunk_idx", 0) for c in evidence
                         if c.get("atomic_facts")]
    if not chunks_with_facts:
        return []
    last = max(chunks_with_facts)
    grounding = chunks_with_facts[-min(8, len(chunks_with_facts)):]
    facts: List[str] = []
    grounding_set = set(grounding)
    for cap in evidence:
        if cap.get("chunk_idx", 0) not in grounding_set:
            continue
        for fact in cap.get("atomic_facts") or []:
            if isinstance(fact, dict) and fact.get("fact"):
                facts.append(str(fact["fact"]).strip().rstrip("."))
                break
    summary = "; ".join(facts[:4]) or "video summary"
    return [Card(
        card_id=f"{video_id}_M1_{_hash_id(video_id, 'M1')}",
        family="M1",
        question="Summarize the video.",
        answer_form="descriptive",
        question_type="single_emit",
        gold_emits=[GoldEmit(chunk=last, value=summary)],
        grounding_frames=grounding,
    )]


# ---------------------------------------------------------------------------
# Top-level: generate all cards for one video
# ---------------------------------------------------------------------------


def generate_cards(evidence: List[Dict], video_id: str, seed: int = 42) -> List[Card]:
    rng = random.Random(stable_seed(seed, video_id, modulo=10**6))
    cards: List[Card] = []
    # MC families (OVOBench bulk) — one per family per video
    for fam in sorted(MC_FAMILIES):
        cards += gen_mc_card(evidence, video_id, fam, FAMILY_BUDGET[fam], rng)
    # Type-specific (always-emit) families
    cards += gen_hld_unanswerable(evidence, video_id)    # MC Unable-to-answer
    if stable_mod(video_id, "F7_ADOPT", modulo=100) < int(F7_ADOPT_RATE * 100):
        cards += gen_f7_status_flip(evidence, video_id)  # binary SSR status flip
    if stable_mod(video_id, "CRR1_ADOPT", modulo=100) < int(CRR1_ADOPT_RATE * 100):
        cards += gen_crr_event_status(evidence, video_id)  # CRR-like before/after probes
    cards += gen_ocr(evidence, video_id)                 # MC OCR (realtime)
    cards += gen_m1_summary(evidence, video_id)          # descriptive (backward)
    # Active-responding non-MCQ: keep REC/counting common, but narration
    # exploratory and small. MULTI_EMIT_ADOPT_RATE remains as the legacy
    # fallback default if explicit env rates are unset.
    if rng.random() < (F5_ADOPT_RATE or MULTI_EMIT_ADOPT_RATE):
        cards += gen_f5_counting(evidence, video_id)     # number multi_emit
    if rng.random() < (PN1_ADOPT_RATE or MULTI_EMIT_ADOPT_RATE):
        cards += gen_pn1_narration(evidence, video_id)   # narration multi_emit
    return cards
