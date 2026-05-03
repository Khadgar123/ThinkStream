"""Pass3 v2 — model-agnostic design.

Three concepts, three fields:
  question_type    — single_emit | multi_emit
  gold_emits       — list of (chunk, value) pairs; defines gold function
  grounding_frames — necessary evidence frames; defines recall oracle

All decisions (gold action, placement, trajectory mechanism) are pure
functions of (ask_chunk, gold_emits, grounding_frames) — never of rollout
state. Difficulty is the gap between ask and grounding, not what the
question-blind rollout happened to remember.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from ..config import MAX_QUESTIONS_PER_TRAJECTORY as CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
from ..stable_hash import stable_mod


# ---------------------------------------------------------------------------
# Constants (mirrors agent_data_v5/config.py + adds new ones)
# ---------------------------------------------------------------------------

VISUAL_WINDOW_CHUNKS = 16        # frames still in visual prompt
RECENT_THINKS_HORIZON = 60       # ~4000 tok / 70 tok-per-think — pre-compress horizon
RECALL_OK_RATE = 0.90            # pure-oracle recall demo
RECALL_NOISY_RATE = 0.05         # oracle ⊕ distractor frames
RECALL_FAILURE_RATE = 0.05       # empty result → wait for next grounding chunk

# ── ask placement: STRATIFIED tier ranges (3 difficulty bands per profile) ──
# Each profile picks one band per placement; multi-placement profiles
# (forward, backward) explicitly cover multiple bands to give the model
# difficulty variety.
#
# Forward stratification (silent_then_response lead time):
SE_LEAD_SHORT  = (4, 10)            # quick wait — easier to maintain pending
SE_LEAD_MEDIUM = (10, 18)           # standard
SE_LEAD_LONG   = (18, 32)           # long wait — hard, tests pending persistence
#
# Real-time / direct fresh stratification (gap from emit):
SE_FRESH_TRIVIAL = (0, 2)            # ask at evidence — trivial
SE_FRESH_EASY    = (3, 8)            # in visual window, recent
SE_FRESH_MEDIUM  = (9, VISUAL_WINDOW_CHUNKS)    # at edge of visual window
#
# Backward stratification (recall_demo gap):
SE_RECALL_NEAR   = (VISUAL_WINDOW_CHUNKS + 1, VISUAL_WINDOW_CHUNKS + 24)         # ~17-40, just out of visual
SE_RECALL_MID    = (VISUAL_WINDOW_CHUNKS + 25, VISUAL_WINDOW_CHUNKS + RECENT_THINKS_HORIZON)  # ~41-76, in recent_thinks
SE_RECALL_DEEP   = (VISUAL_WINDOW_CHUNKS + RECENT_THINKS_HORIZON + 1, 999)        # 77+, fully compressed
#
# Backward mid-band direct/recall mix
BACKWARD_MID_RECALL_PROB = 0.6      # 60% recall, 40% direct in mid band

# multi_emit ask runway before first emit
ME_LEAD_RANGE = (2, 8)

# Production trajectory caps (config.py is the source of truth for max cap)
MAX_QUESTIONS_PER_TRAJECTORY = CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
MIN_QUESTIONS_PER_TRAJECTORY = 6     # floor for very short videos
MAX_TRAJECTORIES_PER_VIDEO = 1
AGENT_CHUNK_SEC = 1                  # seconds per chunk

def adaptive_q_count(num_chunks: int) -> int:
    """Question count scales with video length, ~1 question per 12s.

    Short videos (≤60 chunks): 6-7 questions  (q-interval ~10s)
    Medium (60-120): 8-10 questions             (q-interval ~12s)
    Long (120-200): 10-12 questions             (q-interval ~15s)
    Very long (200+): 12-14 questions           (q-interval ~17s)
    """
    target = max(MIN_QUESTIONS_PER_TRAJECTORY,
                 min(MAX_QUESTIONS_PER_TRAJECTORY, num_chunks // 12))
    return target

# ── OVOBench-aligned family→profile mapping ─────────────────────────
#   backward  : evidence in past, often compressed → recall_demo dominant
#   forward   : evidence in future → silent_then_response dominant
#   realtime  : evidence right now → direct dominant
PLACEMENT_PROFILE = {
    # backward (memory / recall)
    "CR1": "backward", "CR2": "backward", "CR4": "backward",
    "CR5": "backward", "N1":  "backward", "P1":  "backward",
    "M1":  "backward",
    # forward (anticipation / wait)
    "E2":  "forward",  "F6":  "forward",
    # realtime (immediate)
    "CR3": "realtime", "CR7": "realtime", "R1":  "realtime",
    "F5":  "realtime", "C1":  "realtime",
    "PN1": "realtime",
    # v12.13 (P1-7): F7 moved forward → realtime. New F7 is OVO SSR-style
    # multi_emit Yes/No across [change-K, change+K]; not "wait then answer".
    "F7":  "realtime",
}

# Multi_emit families adoption rate per video (each tossed independently).
# Lower → fewer videos carry narration/counting → multi_emit % drops.
MULTI_EMIT_ADOPT_RATE = 0.5

# ── data-level information-density tuning ───────────────────────────
# Patrol = silent samples for chunks NOT covered by any active placement.
# These teach trivial "no active question → silent". Keeping 100% of them
# bloats SFT data with low-signal samples. Keep a stratified 1/3:
#   - chunks with state_changes/new entities: keep at higher rate (richer)
#   - empty chunks: keep at lower rate (trivial)
# Net keep ≈ PATROL_KEEP_RATE_AVG.
PATROL_KEEP_RATE_RICH = 0.50     # chunk has state_change / new entity
PATROL_KEEP_RATE_EMPTY = 0.20    # chunk is purely background / static

# Forward families (E2/F6/F7) generate 2 placements with different leads
# to give the model variety in wait-time training. Doubles hard-silent
# (silent_then_response) sample count.
FORWARD_DOUBLE_PLACEMENT = True

# Question type literal
QuestionType = Literal["single_emit", "multi_emit"]

# Action literal (gold output at any chunk)
GoldKind = Literal["silent", "response"]

# Trajectory placement mechanism
PlacementMechanism = Literal[
    "silent_then_response",  # ask < emit (single_emit only)
    "direct",                # gap small, no recall
    "recall_demo",           # gap large, insert tool_call
    "multi_emit",            # multi-trigger (counting / narration)
]


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GoldEmit:
    chunk: int
    value: str


@dataclass
class Card:
    card_id: str
    family: str                       # taxonomy from pass3a (F5/F7/M1/PN1/...)
    question: str
    answer_form: str                  # binary | multiple_choice | number | short_exact | descriptive
    question_type: QuestionType
    gold_emits: List[GoldEmit]
    grounding_frames: List[int]
    # OPTIONAL pre-generated by pass3a so pass3c doesn't re-call LLM:
    recall_query: Optional[Dict] = None  # {"query": str, "time_range": [int, int]}
    # MC-specific:
    options: Optional[List[str]] = None  # ["A) ...", "B) ...", ...]
    correct_option: Optional[str] = None  # "A" | "B" | "C" | "D"


@dataclass
class Placement:
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    # Per-chunk gold actions inside this placement's window
    # (chunk -> (kind, value or "")):
    chunk_actions: Dict[int, Tuple[GoldKind, str]] = field(default_factory=dict)
    # For recall_demo: which emit chunks have recall inserted, and which
    # noise type (oracle/noisy/failure):
    recall_at: Dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gold function — the ONE source of truth for benchmark and SFT label
# ---------------------------------------------------------------------------


def gold_action_at(
    c: int, ask_chunk: int, emits: List[GoldEmit], question_type: QuestionType,
) -> Tuple[GoldKind, str]:
    """Pure function. Returns (kind, value).

    single_emit rule:
      response_chunk = max(ask_chunk, emits[0].chunk)
      c == response_chunk → response(value); else silent
      → covers (a) ask after evidence: respond at ask
              (b) ask before evidence: respond at emit
              (c) deep history (recall demo): respond at ask, mechanism inserts recall

    multi_emit rule:
      c == any emit.chunk AND c >= ask → response(emit.value)
      else silent
    """
    if c < ask_chunk:
        return ("silent", "")
    if question_type == "single_emit":
        if not emits:
            return ("silent", "")
        response_chunk = max(ask_chunk, emits[0].chunk)
        if c == response_chunk:
            return ("response", emits[0].value)
        return ("silent", "")
    # multi_emit
    for e in emits:
        if e.chunk == c:
            return ("response", e.value)
    return ("silent", "")


def gold_window_for_card(card: Card, ask_chunk: int, num_chunks: int) -> Dict[int, Tuple[GoldKind, str]]:
    """Materialize gold_action over the chunks where this card is 'active'.

    single_emit: [ask, response_chunk + 1] where response_chunk = max(ask, emit)
    multi_emit:  [ask, last_emit + 1]
    The +1 gives one trailing silent for post-response supervision.
    """
    if not card.gold_emits:
        return {}
    if card.question_type == "single_emit":
        response_chunk = max(ask_chunk, card.gold_emits[0].chunk)
        end = min(num_chunks - 1, response_chunk + 1)
    else:
        last_emit = max(e.chunk for e in card.gold_emits)
        end = min(num_chunks - 1, last_emit + 1)
    actions: Dict[int, Tuple[GoldKind, str]] = {}
    for c in range(ask_chunk, end + 1):
        actions[c] = gold_action_at(c, ask_chunk, card.gold_emits, card.question_type)
    return actions


# ---------------------------------------------------------------------------
# Placement — model-agnostic; pure function of (card, num_chunks, rng)
# ---------------------------------------------------------------------------


def _gap_for_emit(emit_chunk: int, grounding_frames: List[int]) -> int:
    """Distance from this emit's most recent grounding frame to the emit chunk.

    For single_emit: emit.chunk == max(grounding) typically, gap >= 0.
    For multi_emit (counting/narration): each emit IS the grounding at that
    chunk → gap = 0.
    """
    relevant = [g for g in grounding_frames if g <= emit_chunk]
    if not relevant:
        return 0
    return emit_chunk - max(relevant)


def _classify_mechanism(card: Card, ask_chunk: int) -> PlacementMechanism:
    """Decide trajectory mechanism deterministically from ask & grounding."""
    if card.question_type == "multi_emit":
        return "multi_emit"
    # single_emit
    emit = card.gold_emits[0].chunk
    if ask_chunk < emit:
        return "silent_then_response"
    # ask >= emit
    gap = ask_chunk - emit
    if gap <= VISUAL_WINDOW_CHUNKS:
        return "direct"
    if gap <= VISUAL_WINDOW_CHUNKS + RECENT_THINKS_HORIZON:
        return "direct"  # mixed at render time (see render_placement); 50% will swap to recall
    return "recall_demo"


def _make_placement(card: Card, ask: int, num_chunks: int, mech: PlacementMechanism) -> Placement:
    return Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism=mech,
        chunk_actions=gold_window_for_card(card, ask, num_chunks),
    )


def _randint_safe(rng: random.Random, lo: int, hi: int) -> int:
    """Inclusive randint with safe handling when lo >= hi."""
    if hi <= lo:
        return lo
    return rng.randint(lo, hi)


def _ask_from_band(emit: int, band: Tuple[int, int], num_chunks: int,
                   rng: random.Random, sign: int) -> Optional[int]:
    """Pick an ask chunk inside the given (lo, hi) gap band.

    sign = +1 → ask = emit + gap (placement after emit)
    sign = -1 → ask = emit - lead (placement before emit)
    Returns None if band has no feasible gap.
    """
    lo, hi = band
    if sign > 0:
        max_feasible = num_chunks - 1 - emit
        hi_eff = min(hi, max_feasible)
        if hi_eff < lo:
            return None
        gap = _randint_safe(rng, lo, hi_eff)
        return emit + gap
    else:
        max_feasible = emit
        hi_eff = min(hi, max_feasible)
        if hi_eff < lo:
            return None
        lead = _randint_safe(rng, lo, hi_eff)
        return emit - lead


def place_single_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Profile-driven STRATIFIED placement.

    forward   → 2 placements: 1 short-lead (4-10), 1 medium/long (10-32)
    backward  → 2-3 placements: near recall + mid (direct or recall) + deep recall
    realtime  → 1 placement: stratified 3-band fresh gap
    """
    if not card.gold_emits:
        return []
    profile = PLACEMENT_PROFILE.get(card.family, "realtime")
    emit = card.gold_emits[0].chunk
    placements: List[Placement] = []

    if profile == "forward":
        # Always try to emit 2 placements with different lead bands
        # for difficulty variety (silent_then_response).
        for band in (SE_LEAD_SHORT, SE_LEAD_LONG):
            ask = _ask_from_band(emit, band, num_chunks, rng, sign=-1)
            if ask is not None:
                placements.append(_make_placement(
                    card, ask, num_chunks, "silent_then_response"))
        if not placements:
            # video too short for any standard lead → fallback short
            ask = _ask_from_band(emit, (4, 8), num_chunks, rng, sign=-1)
            if ask is not None:
                placements.append(_make_placement(
                    card, ask, num_chunks, "silent_then_response"))
        return placements

    if profile == "realtime":
        # Stratified fresh gap — sample one band (rotated by card id for spread)
        band_choice = (SE_FRESH_TRIVIAL, SE_FRESH_EASY, SE_FRESH_MEDIUM)[
            stable_mod(card.card_id, modulo=3)
        ]
        ask = _ask_from_band(emit, band_choice, num_chunks, rng, sign=+1)
        if ask is None:
            ask = _ask_from_band(emit, SE_FRESH_TRIVIAL, num_chunks, rng, sign=+1)
        if ask is not None:
            placements.append(_make_placement(card, ask, num_chunks, "direct"))
        return placements

    # ── backward profile: NEAR + MID + DEEP (3 difficulty bands) ─────
    # NEAR (just out of visual): always recall_demo (model already lost direct)
    ask_near = _ask_from_band(emit, SE_RECALL_NEAR, num_chunks, rng, sign=+1)
    if ask_near is not None:
        placements.append(_make_placement(card, ask_near, num_chunks, "recall_demo"))

    # MID (in recent_thinks): mixed direct/recall by BACKWARD_MID_RECALL_PROB
    ask_mid = _ask_from_band(emit, SE_RECALL_MID, num_chunks, rng, sign=+1)
    if ask_mid is not None:
        mech_mid = "recall_demo" if rng.random() < BACKWARD_MID_RECALL_PROB else "direct"
        placements.append(_make_placement(card, ask_mid, num_chunks, mech_mid))

    # DEEP (compressed): always recall_demo
    ask_deep = _ask_from_band(emit, SE_RECALL_DEEP, num_chunks, rng, sign=+1)
    if ask_deep is not None:
        placements.append(_make_placement(card, ask_deep, num_chunks, "recall_demo"))

    if not placements:
        # very short video — fallback to direct
        gap = _randint_safe(rng, 0, max(0, num_chunks - 1 - emit))
        placements.append(_make_placement(card, emit + gap, num_chunks, "direct"))

    return placements


def place_multi_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Generate ONE placement: ask = first_emit - random lead in [2, 8]."""
    if not card.gold_emits:
        return []
    first = min(e.chunk for e in card.gold_emits)
    lead = _randint_safe(rng, ME_LEAD_RANGE[0], min(ME_LEAD_RANGE[1], first))
    ask = max(0, first - lead)
    actions = gold_window_for_card(card, ask, num_chunks)
    return [Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism="multi_emit",
        chunk_actions=actions,
    )]


def place_card(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    if card.question_type == "single_emit":
        return place_single_emit(card, num_chunks, rng)
    return place_multi_emit(card, num_chunks, rng)


# ---------------------------------------------------------------------------
# Trajectory selection — pick MAX_QUESTIONS_PER_TRAJECTORY placements per video
# ---------------------------------------------------------------------------


def select_trajectory(
    cards: List[Card],
    placements_by_card: Dict[str, List[Placement]],
    num_chunks: int,
    rng: random.Random,
    max_q: int = MAX_QUESTIONS_PER_TRAJECTORY,
) -> List[Placement]:
    """Select up to max_q placements for ONE trajectory per video.

    Greedy diversity scoring:
      +3 unseen family
      +2 unseen mechanism
      +1 unseen answer_form
      +1 unseen card (one placement per card max)
      +spread: distance to nearest already-picked ask_chunk / 10 (cap 1.5)

    Strict constraint: at most ONE placement per card_id.
    """
    cards_by_id = {c.card_id: c for c in cards}
    pool: List[Tuple[Placement, Card]] = []
    for cid, plcs in placements_by_card.items():
        card = cards_by_id.get(cid)
        if not card:
            continue
        for p in plcs:
            pool.append((p, card))
    if not pool:
        return []

    selected: List[Placement] = []
    seen_families: set = set()
    seen_mechs: set = set()
    seen_aforms: set = set()
    seen_cards: set = set()
    used_ask: List[int] = []

    while len(selected) < max_q and pool:
        best_score = -1e9
        best_idx = -1
        for i, (p, card) in enumerate(pool):
            if card.card_id in seen_cards:
                continue
            s = 0.0
            if card.family not in seen_families:
                s += 3.0
            if p.mechanism not in seen_mechs:
                s += 2.0
            if card.answer_form not in seen_aforms:
                s += 1.0
            s += 1.0  # base for new card
            # spread bonus
            if used_ask:
                min_dist = min(abs(p.ask_chunk - x) for x in used_ask)
                s += min(min_dist / 10.0, 1.5)
            else:
                s += 1.5
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            break
        p, card = pool.pop(best_idx)
        selected.append(p)
        seen_families.add(card.family)
        seen_mechs.add(p.mechanism)
        seen_aforms.add(card.answer_form)
        seen_cards.add(card.card_id)
        used_ask.append(p.ask_chunk)

    return selected


# ---------------------------------------------------------------------------
# Recall augmentation — independent of rollout
# ---------------------------------------------------------------------------


def assign_recall_noise(placements: List[Placement], rng: random.Random) -> None:
    """For each recall_demo emit chunk, draw oracle/noisy/failure label.

    Mutates placement.recall_at in place. Rates are GLOBAL constants
    (RECALL_OK_RATE / RECALL_NOISY_RATE / RECALL_FAILURE_RATE) — they do
    NOT depend on rollout state.
    """
    for p in placements:
        if p.mechanism != "recall_demo":
            continue
        for c, (kind, _) in p.chunk_actions.items():
            if kind != "response":
                continue
            r = rng.random()
            if r < RECALL_OK_RATE:
                p.recall_at[c] = "oracle"
            elif r < RECALL_OK_RATE + RECALL_NOISY_RATE:
                p.recall_at[c] = "noisy"
            else:
                p.recall_at[c] = "failure"


# ---------------------------------------------------------------------------
# Trajectory rendering — produces SFT samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    chunk_idx: int
    sample_kind: str               # silent | response | recall+response | recall+silent
    placement_id: str
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    response_text: str = ""        # empty for silent
    recall_query: Optional[Dict] = None
    recall_result_kind: Optional[str] = None  # oracle/noisy/failure
    extra: Dict = field(default_factory=dict)


def render_placement(
    card: Card,
    placement: Placement,
) -> List[Sample]:
    """Walk the placement's gold window and emit one Sample per active chunk."""
    samples: List[Sample] = []
    pid = f"{card.card_id}@{placement.ask_chunk}"
    for c in sorted(placement.chunk_actions.keys()):
        kind, value = placement.chunk_actions[c]
        if kind == "silent":
            samples.append(Sample(
                chunk_idx=c,
                sample_kind="silent",
                placement_id=pid,
                card_id=card.card_id,
                ask_chunk=placement.ask_chunk,
                mechanism=placement.mechanism,
            ))
            continue
        # response chunk
        if placement.mechanism == "recall_demo" and c in placement.recall_at:
            rkind = placement.recall_at[c]
            if rkind == "failure":
                # silent at this chunk; the next emit (if any) will retry,
                # but we still emit a recall+silent sample for THIS chunk
                # to teach "recall returned nothing → don't fabricate"
                samples.append(Sample(
                    chunk_idx=c,
                    sample_kind="recall+silent",
                    placement_id=pid,
                    card_id=card.card_id,
                    ask_chunk=placement.ask_chunk,
                    mechanism=placement.mechanism,
                    recall_query=card.recall_query,
                    recall_result_kind=rkind,
                ))
            else:
                samples.append(Sample(
                    chunk_idx=c,
                    sample_kind="recall+response",
                    placement_id=pid,
                    card_id=card.card_id,
                    ask_chunk=placement.ask_chunk,
                    mechanism=placement.mechanism,
                    response_text=value,
                    recall_query=card.recall_query,
                    recall_result_kind=rkind,
                ))
        else:
            samples.append(Sample(
                chunk_idx=c,
                sample_kind="response",
                placement_id=pid,
                card_id=card.card_id,
                ask_chunk=placement.ask_chunk,
                mechanism=placement.mechanism,
                response_text=value,
            ))
    return samples


def render_video_samples(
    cards: List[Card],
    placements_by_card: Dict[str, List[Placement]],
    num_chunks: int,
    evidence: Optional[List[Dict]] = None,
    rng: Optional[random.Random] = None,
    compression_event_chunks: Optional[List[int]] = None,
) -> List[Sample]:
    """Render placements + STRATIFIED-DOWNSAMPLED patrol.

    Patrol downsampling rules (data-level info-density boost):
      - chunks with state_change OR new entity (rich):  keep at PATROL_KEEP_RATE_RICH
      - chunks empty / static:                          keep at PATROL_KEEP_RATE_EMPTY

    Multi-placement priority: response > recall+response > recall+silent > silent.
    """
    if rng is None:
        rng = random.Random(0)
    per_chunk: Dict[int, List[Tuple[str, str, Placement, Card]]] = {}
    cards_by_id = {c.card_id: c for c in cards}
    for cid, plcs in placements_by_card.items():
        card = cards_by_id[cid]
        for p in plcs:
            for s in render_placement(card, p):
                per_chunk.setdefault(s.chunk_idx, []).append(
                    (s.sample_kind, s.response_text, p, card)
                )

    PRIORITY = {
        "response": 5,
        "recall+response": 4,
        "recall+silent": 3,
        "silent": 2,
        "compress_silent": 1,    # NEW silent situation type
        "patrol": 0,
    }

    # Pre-compute richness flag per chunk for stratified patrol sampling
    chunk_rich: Dict[int, bool] = {}
    if evidence:
        seen_entities: set = set()
        for cap in evidence:
            c = cap.get("chunk_idx", 0)
            if cap.get("state_changes"):
                chunk_rich[c] = True
                continue
            new_entity = False
            for ent in cap.get("visible_entities", []) or []:
                eid = ent.get("id") or ent.get("desc", "")[:30]
                if eid and eid not in seen_entities:
                    seen_entities.add(eid)
                    new_entity = True
            chunk_rich[c] = new_entity

    compress_set = set(compression_event_chunks or [])

    all_samples: List[Sample] = []
    for c in range(num_chunks):
        candidates = per_chunk.get(c, [])
        if not candidates:
            # NEW silent situation: chunk where memory compression triggered.
            # Model should observe (silent) but not respond. This is a rare,
            # specific silent context the model otherwise never sees.
            if c in compress_set:
                all_samples.append(Sample(
                    chunk_idx=c,
                    sample_kind="compress_silent",
                    placement_id="",
                    card_id="",
                    ask_chunk=-1,
                    mechanism="multi_emit",
                    extra={"role": "compress_event"},
                ))
                continue
            # Patrol candidate — apply stratified downsampling
            keep_rate = (PATROL_KEEP_RATE_RICH if chunk_rich.get(c, False)
                         else PATROL_KEEP_RATE_EMPTY)
            if rng.random() >= keep_rate:
                continue                # drop — do NOT emit this patrol sample
            all_samples.append(Sample(
                chunk_idx=c,
                sample_kind="patrol",
                placement_id="",
                card_id="",
                ask_chunk=-1,
                mechanism="multi_emit",  # placeholder
                extra={"role": "patrol_rich" if chunk_rich.get(c) else "patrol_empty"},
            ))
            continue
        candidates.sort(key=lambda t: -PRIORITY[t[0]])
        kind, value, p, card = candidates[0]
        all_samples.append(Sample(
            chunk_idx=c,
            sample_kind=kind,
            placement_id=f"{card.card_id}@{p.ask_chunk}",
            card_id=card.card_id,
            ask_chunk=p.ask_chunk,
            mechanism=p.mechanism,
            response_text=value,
            recall_query=card.recall_query if "recall" in kind else None,
            recall_result_kind=p.recall_at.get(c) if "recall" in kind else None,
        ))
    return all_samples


# ---------------------------------------------------------------------------
# Sanity helpers (used by simulator)
# ---------------------------------------------------------------------------


def is_response_kind(k: str) -> bool:
    return k in ("response", "recall+response")


def is_silent_kind(k: str) -> bool:
    return k in ("silent", "patrol", "recall+silent", "compress_silent")
