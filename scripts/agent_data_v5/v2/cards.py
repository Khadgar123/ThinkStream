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
from typing import Dict, List, Optional, Tuple

from ..stable_hash import stable_mod, stable_seed
from .design import Card, GoldEmit, MULTI_EMIT_ADOPT_RATE


# Family taxonomy aligned with OVOBench (MC-dominant) + 3-bucket profile.
# Total target ~14 cards/video (was 20.9) — gives the trajectory selector
# a small headroom over MAX_QUESTIONS_PER_TRAJECTORY for diversity scoring
# without wasting LLM budget on cards that get dropped.
#
# Bucket allocation (matches PLACEMENT_PROFILE in design.py):
#   backward (recall-heavy):  N1 1, P1 1, CR1 1, CR2 1, CR4 1, CR5 1 = 6 cards
#   forward  (silent-then-respond): E2 1, F6 1, F7 1               = 3 cards
#   realtime (immediate):     CR3 1, CR7 1, R1 1, F5 1, C1 1, PN1 1 = 6 cards
#   summary  (backward, descriptive): M1 1                          = 1 card
#   ────────────────────────────────────────────────────────────────
#   total                                                            16 cards
FAMILY_BUDGET = {
    # backward MC
    "N1":  1, "P1":  1, "CR1": 1, "CR2": 1, "CR4": 1, "CR5": 1,
    # forward MC + binary
    "E2":  1, "F6":  1, "F7":  1,
    # realtime MC + number + short_exact
    "CR3": 1, "CR7": 1, "R1":  1, "F5":  1, "C1":  1,
    # multi_emit (PN1 50% adopt; F5 already realtime above)
    "PN1": 1,
    # backward descriptive
    "M1":  1,
}

MC_FAMILIES = {"N1", "P1", "CR1", "CR2", "CR3", "CR4", "CR5", "CR7",
               "E2", "F6", "R1"}


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
    event_chunks: List[int] = []
    last = -100
    for cap in evidence:
        c = cap.get("chunk_idx", 0)
        if not cap.get("state_changes"):
            continue
        if c - last < 12:                    # was 6 → tightened to LiveCC density
            continue
        event_chunks.append(c)
        last = c
    if not event_chunks:
        return []
    # Hard cap: never more than 6 narration emits per video, regardless of
    # video length. Beyond 6 the model just learns "narrate every 12s" which
    # isn't what OVOBench measures.
    event_chunks = event_chunks[:6]
    emits = [GoldEmit(chunk=c, value=f"event@{c}") for c in event_chunks]
    return [Card(
        card_id=f"{video_id}_PN1_{_hash_id(video_id, 'PN1')}",
        family="PN1",
        question="",  # implicit narration mode
        answer_form="descriptive",
        question_type="multi_emit",
        gold_emits=emits,
        grounding_frames=event_chunks,
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
        lo = max(0, c - K)
        hi = min(n_chunks - 1, c + K)
        if hi - lo < 4:    # need ≥ 5 chunks for multi_emit to be meaningful
            continue
        # Build per-chunk emits: "No" before change, "Yes" at and after.
        emits = []
        for ci in range(lo, hi + 1):
            value = "No" if ci < c else "Yes"
            emits.append(GoldEmit(chunk=ci, value=value))
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


def gen_single_answer_factoid(evidence: List[Dict], video_id: str, family: str, budget: int) -> List[Card]:
    """Single_emit cards from atomic_facts. Emit chunks SPREAD across video.

    Production 397B picks question-worthy moments throughout the video; we
    mimic that by quantile-stratified sampling instead of always taking
    the earliest qualifying chunk.
    """
    candidates = []
    for cap in evidence:
        facts = [f for f in (cap.get("atomic_facts") or [])
                 if isinstance(f, dict) and f.get("confidence", 0) >= 0.7]
        if not facts:
            continue
        candidates.append((cap.get("chunk_idx", 0), facts[0]))
    if not candidates:
        return []
    # Stratified pick: divide chunks into `budget` quantile bins, pick one per bin.
    candidates.sort(key=lambda x: x[0])
    n = len(candidates)
    bins: List[Tuple[int, Dict]] = []
    for i in range(budget):
        idx = (i * n) // budget + (n // (budget * 2))
        idx = min(idx, n - 1)
        if not bins or bins[-1][0] != candidates[idx][0]:
            bins.append(candidates[idx])
    cards = []
    for c, f in bins[:budget]:
        ans = _canonical_short(f.get("fact", ""), max_words=6)
        if not ans:
            continue
        emits = [GoldEmit(chunk=c, value=ans)]
        cards.append(Card(
            card_id=f"{video_id}_{family}_{_hash_id(video_id, family, c)}",
            family=family,
            question=f"[{family}] question about chunk {c}?",
            answer_form="short_exact",
            question_type="single_emit",
            gold_emits=emits,
            grounding_frames=[c],
        ))
    return cards


def gen_compositional(evidence: List[Dict], video_id: str, family: str, budget: int) -> List[Card]:
    """CR4-style: pair two non-adjacent fact chunks; emit at max(grounding)."""
    fact_chunks = []
    for cap in evidence:
        if [f for f in (cap.get("atomic_facts") or [])
            if isinstance(f, dict) and f.get("confidence", 0) >= 0.7]:
            fact_chunks.append(cap.get("chunk_idx", 0))
    cards = []
    for i in range(len(fact_chunks) - 1):
        a, b = fact_chunks[i], fact_chunks[i + 1]
        if b - a < 6:  # need spread
            continue
        emits = [GoldEmit(chunk=b, value="A then B")]
        cards.append(Card(
            card_id=f"{video_id}_{family}_{_hash_id(video_id, family, a, b)}",
            family=family,
            question=f"[{family}] composite about chunks {a} and {b}?",
            answer_form="descriptive",
            question_type="single_emit",
            gold_emits=emits,
            grounding_frames=[a, b],
        ))
        if len(cards) >= budget:
            break
    return cards


def gen_ocr(evidence: List[Dict], video_id: str) -> List[Card]:
    """C1 cards: questions about OCR text."""
    ocr = _ocr_chunks(evidence)
    cards = []
    for c, text in ocr[:FAMILY_BUDGET["C1"]]:
        ans = _canonical_short(text, max_words=4)
        emits = [GoldEmit(chunk=c, value=ans)]
        cards.append(Card(
            card_id=f"{video_id}_C1_{_hash_id(video_id, c, text)}",
            family="C1",
            question=f"What text appears at chunk {c}?",
            answer_form="short_exact",
            question_type="single_emit",
            gold_emits=emits,
            grounding_frames=[c],
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
    one MC card per chunk with rotated correct option (A/B/C/D) for
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

    rotation = ["A", "B", "C", "D"]
    family_offset = stable_mod(video_id, family, modulo=4)
    cards = []
    for i, (c, ents, facts) in enumerate(bins):
        if ents:
            correct_text = ents[0].get("desc", "entity")[:40]
        else:
            correct_text = _canonical_short(facts[0].get("fact", ""), 6)
        if not correct_text:
            continue

        # Sample 3 distractors that aren't the correct answer.
        candidates_d = [d for d in distractor_pool
                        if d.strip().lower() != correct_text.strip().lower()]
        if len(candidates_d) < 3:
            # Heuristic pool too thin → fall back to short_exact (entity name)
            # so reward+eval stay valid (no MC letter without real options).
            cards.append(Card(
                card_id=f"{video_id}_{family}_{_hash_id(video_id, family, c)}",
                family=family,
                question=f"[{family}] What is the key entity at chunk {c}?",
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
        distractors = chunk_rng.sample(candidates_d, 3)
        correct_pos = rotation[(i + family_offset) % 4]
        options = list(distractors)
        options.insert(ord(correct_pos) - ord("A"), correct_text)
        # Drop the surplus item that pushed list to length 5 (insert grew it)
        options = options[:4]
        opts_with_letter = [f"{chr(65+j)}) {o}" for j, o in enumerate(options)]
        emits = [GoldEmit(chunk=c, value=correct_pos)]
        cards.append(Card(
            card_id=f"{video_id}_{family}_{_hash_id(video_id, family, c)}",
            family=family,
            question=f"[{family}] MC question about chunk {c}?",
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
    return [Card(
        card_id=f"{video_id}_M1_{_hash_id(video_id, 'M1')}",
        family="M1",
        question="Summarize the video.",
        answer_form="descriptive",
        question_type="single_emit",
        gold_emits=[GoldEmit(chunk=last, value="full summary")],
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
    cards += gen_f7_status_flip(evidence, video_id)      # binary (forward profile)
    cards += gen_ocr(evidence, video_id)                 # short_exact (realtime)
    cards += gen_m1_summary(evidence, video_id)          # descriptive (backward)
    # Multi_emit / narration — adopt only in MULTI_EMIT_ADOPT_RATE of videos
    # to bring multi_emit % of placements into target ~5% range.
    if rng.random() < MULTI_EMIT_ADOPT_RATE:
        cards += gen_f5_counting(evidence, video_id)     # number multi_emit
    if rng.random() < MULTI_EMIT_ADOPT_RATE:
        cards += gen_pn1_narration(evidence, video_id)   # narration multi_emit
    return cards
