"""Pass3 v2 — model-agnostic design.

Three concepts, three fields:
  question_type    — single_emit | multi_emit
  gold_emits       — list of (chunk, value) pairs; defines gold function
  grounding_frames — necessary evidence frames; defines recall oracle

Card families describe the benchmark skill being asked; they are not tied to
one availability bucket. A single family may produce current/direct,
memory-direct, forward/wait, and historical-recall placements. Difficulty is
primarily the gap between ask and grounding plus whether the first-turn prompt
still contains enough clear evidence to answer without visual recall.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
import os
import re
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from ..config import MAX_QUESTIONS_PER_TRAJECTORY as CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
from ..stable_hash import stable_mod


# ---------------------------------------------------------------------------
# Constants (mirrors agent_data_v5/config.py + adds new ones)
# ---------------------------------------------------------------------------

VISUAL_WINDOW_CHUNKS = 16        # frames still in visual prompt
RECENT_THINKS_HORIZON = 60       # ~4000 tok / 70 tok-per-think — pre-compress horizon
RECALL_OK_RATE = 0.95            # pure-oracle recall demo
RECALL_NOISY_RATE = 0.05         # oracle ⊕ distractor frames
RECALL_WAIT_PROBE_MAX = 3        # max recall+silent probes inside one wait
RECALL_WAIT_MIN_LEAD = 6         # do not recall immediately for very short waits
RECALL_MULTI_WAIT_PROBE_MAX = 2  # recall+silent probes while cumulative questions remain open
MEMORY_DIRECT_RECALL_PROBE_RATE = float(
    os.environ.get("THINKSTREAM_MEMORY_DIRECT_RECALL_PROBE_RATE", "0.55")
)
MEMORY_DIRECT_RECALL_FAMILY_RATE = {
    # Exact visual evidence is much better than text memory for these OVO
    # weaknesses: OCR, spatial/temporal relations, object relation/state,
    # action recognition, future/held state, and cross-event reasoning.
    "C1": 0.85,
    "STU1": 0.70,
    "OJR1": 0.70,
    "ACR1": 0.65,
    "CR7": 0.60,
    "F6": 0.60,
    "CR5": 0.60,
    "CR4": 0.58,
    "CR1": 0.55,
    "CR2": 0.50,
    "M1": 0.50,
    "HLD1": 0.45,
}

# ── ask placement: STRATIFIED tier ranges (3 difficulty bands per profile) ──
# Each profile picks one band per placement; multi-placement profiles
# (forward, backward) explicitly cover multiple bands to give the model
# difficulty variety.
#
# Forward stratification (silent_then_response lead time):
SE_LEAD_SHORT  = (4, 10)            # quick wait — easier to maintain pending
SE_LEAD_LONG   = (18, 32)           # long wait — hard, tests pending persistence
SE_LEAD_OVO_CRR = ((10, 24), (24, 64), (64, 120))  # OVO CRR clue delays
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
# multi_emit ask runway before first emit
ME_LEAD_RANGE = (2, 8)
MAX_MULTI_EMIT_ACTIVE_SPAN = 16
MAX_MULTI_EMIT_RESPONSES = 5
MAX_REC_EMIT_RESPONSES = 5
STATUS_FAR_AFTER_MIN_GAP = int(os.environ.get(
    "THINKSTREAM_STATUS_FAR_AFTER_MIN_GAP",
    str(VISUAL_WINDOW_CHUNKS + 1),
))
STATUS_FAR_AFTER_MAX_GAP = int(os.environ.get(
    "THINKSTREAM_STATUS_FAR_AFTER_MAX_GAP",
    "80",
))
RECALL_TARGET_FRACTION = 0.65

# Families whose answers often require multi-frame temporal/causal reasoning
# or fine visual verification. Exact text memory can still answer some of
# these, but selection should preferentially keep their historical placements
# as recall candidates rather than collapsing them to memory_direct.
HARD_RECALL_FAMILIES = {
    "CR1", "CR2", "CR4", "CR5", "M1",
    "C1", "STU1", "OJR1", "CR7", "ACR1", "F6", "HLD1",
}
SIMPLE_MEMORY_FAMILIES = {
    "N1", "P1", "R1", "CR3", "ACR1", "HLD1",
}

# Production trajectory caps (config.py is the source of truth for max cap)
MAX_QUESTIONS_PER_TRAJECTORY = CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
MIN_QUESTIONS_PER_TRAJECTORY = 8     # floor for very short videos
MAX_TRAJECTORIES_PER_VIDEO = 1
AGENT_CHUNK_SEC = 1                  # seconds per chunk
MIN_QUESTION_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_MIN_QUESTION_ASK_GAP_CHUNKS", "4"))
SHORT_VIDEO_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_SHORT_VIDEO_ASK_GAP_CHUNKS", "3"))
LONG_VIDEO_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_LONG_VIDEO_ASK_GAP_CHUNKS", "4"))
LONG_VIDEO_ASK_GAP_AT_CHUNKS = int(os.environ.get("THINKSTREAM_LONG_VIDEO_ASK_GAP_AT_CHUNKS", "180"))
SPREAD_SCORE_DENOM = 8.0
SPREAD_SCORE_CAP = 3.0

def adaptive_q_count(num_chunks: int) -> int:
    """Question count scales with video length, ~1 question per 10s.

    Short videos (≤60 chunks): 8 questions       (q-interval ~7s)
    Medium (60-120): 8-12 questions              (q-interval ~10s)
    Long (120-200): 12-20 questions              (q-interval ~10-12s)
    Very long (200+): 20 questions cap           (q-interval varies)
    """
    target = max(MIN_QUESTIONS_PER_TRAJECTORY,
                 min(MAX_QUESTIONS_PER_TRAJECTORY, num_chunks // 10))
    return target


def question_ask_gap_floor(num_chunks: int) -> int:
    """Minimum spacing between independent user questions on one timeline."""
    if num_chunks < 64:
        return SHORT_VIDEO_ASK_GAP_CHUNKS
    if num_chunks >= LONG_VIDEO_ASK_GAP_AT_CHUNKS:
        return LONG_VIDEO_ASK_GAP_CHUNKS
    return MIN_QUESTION_ASK_GAP_CHUNKS

# ── OVOBench-aligned family→profile mapping ─────────────────────────
#   backward  : evidence in past, often compressed → recall_demo dominant
#   forward   : evidence in future → silent_then_response dominant
#   realtime  : evidence right now → direct dominant
PLACEMENT_PROFILE = {
    # backward (memory / recall)
    "CR1": "backward", "CR2": "backward", "CR4": "backward",
    "CR5": "backward", "N1":  "backward", "P1":  "backward",
    "HLD1": "backward", "M1":  "backward",
    # forward (anticipation / wait)
    "E2":  "forward",
    # realtime (immediate)
    "CR3": "realtime", "CR7": "realtime", "R1":  "realtime",
    "ACR1": "realtime", "STU1": "realtime", "OJR1": "realtime",
    "F5":  "realtime", "C1":  "realtime",
    "PN1": "realtime",
    "F6":  "realtime",
    "CRR1": "realtime",
    # v12.13 (P1-7): F7 moved forward → realtime. New F7 is OVO SSR-style
    # multi_emit Yes/No across [change-K, change+K]; not "wait then answer".
    "F7":  "realtime",
}

# Multi_emit families adoption rate per video (each tossed independently).
# Lower → fewer videos carry narration/counting → multi_emit % drops.
MULTI_EMIT_ADOPT_RATE = 0.5

# F7/SSR should be present, but not every video should carry a progress-status
# card. Batch3 landed below OVO SSR scale, so use a higher adoption rate and
# let selection/overlap constraints decide whether the card fits each video.
F7_ADOPT_RATE = 0.6

# CRR1 is the OVO-CRR-like repeated probe around a clue/event becoming true.
# Keep it below every-video generation because each selected card can occupy a
# long active span and would otherwise squeeze out regular QA slots.
CRR1_ADOPT_RATE = 0.45

# Rare benchmark-aligned families can lose greedy selection because their
# active span is longer (F7) or because recall slots are already saturated
# (HLD1). Boosting selection, not generation volume, keeps the card pool
# balanced while making selected trajectories carry the intended coverage.
FAMILY_SELECTION_BOOST = {
    "F7": 6.0,     # target SSR-like status rows at roughly OVO scale
    "F5": 5.0,     # REC-style cumulative counting otherwise loses to recall
    "CRR1": 5.0,   # clue-before/after multi-probe status
    "CR5": 4.0,    # CRR-style clue waits should survive selection
    "OJR1": 6.0,   # OVO has a large object-joint-reasoning slice
    "STU1": 6.0,   # state transition understanding is under-selected
    "HLD1": 4.0,   # explicit negative/holdout questions are OVO-heavy
    "C1": 4.0,     # MC OCR should survive selection
    "ACR1": 2.5,   # action/causal reasoning should stay near OVO scale
}

# Keep HLD / "Unable to answer" abstention negatives near the previous
# reasonable family share, but do not force them into every feasible video.
HLD_ABSTENTION_RESERVE_PERCENT = 84

# ── data-level information-density tuning ───────────────────────────
# Patrol = silent samples for chunks NOT covered by any active placement.
# These teach trivial "no active question → silent". Keeping 100% of them
# bloats SFT data with low-signal samples. Keep a stratified 1/3:
#   - chunks with state_changes/new entities: keep at higher rate (richer)
#   - empty chunks: keep at lower rate (trivial)
# Net keep ≈ PATROL_KEEP_RATE_AVG.
PATROL_KEEP_RATE_RICH = 0.32     # chunk has state_change / new entity
PATROL_KEEP_RATE_EMPTY = 0.08    # chunk is purely background / static

# Question type literal
QuestionType = Literal["single_emit", "multi_emit"]

# Action literal (gold output at any chunk)
GoldKind = Literal["silent", "response"]

# Trajectory placement mechanism
PlacementMechanism = Literal[
    "silent_then_response",  # ask < emit (single_emit only)
    "direct",                # gap small, no recall
    "memory_direct",         # historical support, text memory is enough
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
    correct_option: Optional[str] = None  # "A" | "B" | ... matching options


@dataclass
class Placement:
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    difficulty_mode: str = ""
    recall_need: str = ""
    # Per-chunk gold actions inside this placement's window
    # (chunk -> (kind, value or "")):
    chunk_actions: Dict[int, Tuple[GoldKind, str]] = field(default_factory=dict)
    # Chunks where the assistant should call recall before the final action.
    # recall_demo response chunks use oracle/noisy historical frames.
    # recall_demo response chunks use oracle/noisy historical frames. Forward
    # silent_then_response waits are plain silent turns; answer support is in
    # the future, so a recall call cannot be the minimal action.
    recall_at: Dict[int, str] = field(default_factory=dict)
    # Optional explanation label for recall_at chunks. This is used only to
    # shape the recall tool-call think text; it does not affect gold timing.
    recall_reason_at: Dict[int, str] = field(default_factory=dict)


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


def _make_placement(
    card: Card,
    ask: int,
    num_chunks: int,
    mech: PlacementMechanism,
    *,
    difficulty_mode: str = "",
    recall_need: str = "",
) -> Placement:
    return Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism=mech,
        difficulty_mode=difficulty_mode,
        recall_need=recall_need,
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


def _dedupe_placements(placements: List[Placement]) -> List[Placement]:
    out: List[Placement] = []
    seen: set = set()
    for p in placements:
        key = (p.card_id, p.ask_chunk, p.mechanism)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def place_single_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Generate availability variants for one single-emit card.

    Family no longer decides whether a question is recall-only or direct-only.
    The same question type can be placed at several ask times:
      - direct/current: answer support is inside the visual window
      - memory_direct: historical support is outside vision but recent text
        memory should be enough for a plain response
      - recall_demo: historical support is far/ambiguous enough to call recall
      - silent_then_response: selected temporal families may also ask before
        evidence appears, preserving wait/silent supervision
    """
    if not card.gold_emits:
        return []
    emit = card.gold_emits[0].chunk
    placements: List[Placement] = []

    # Fresh/current direct placement for every family. This preserves the
    # answer-without-tool side of each benchmark skill.
    band_choice = (SE_FRESH_TRIVIAL, SE_FRESH_EASY, SE_FRESH_MEDIUM)[
        stable_mod(card.card_id, "direct", modulo=3)
    ]
    ask = _ask_from_band(emit, band_choice, num_chunks, rng, sign=+1)
    if ask is None:
        ask = _ask_from_band(emit, SE_FRESH_TRIVIAL, num_chunks, rng, sign=+1)
    if ask is not None:
        placements.append(_make_placement(
            card, ask, num_chunks, "direct",
            difficulty_mode="current_direct",
            recall_need="current_window",
        ))

    # Historical response from text memory: outside the visual window but not
    # necessarily requiring visual recall. The verifier/selector can keep this
    # as a response-side hard negative against over-calling recall.
    ask_mem = _ask_from_band(emit, SE_RECALL_NEAR, num_chunks, rng, sign=+1)
    if ask_mem is not None:
        placements.append(_make_placement(
            card, ask_mem, num_chunks, "memory_direct",
            difficulty_mode="memory_direct",
            recall_need="memory_text_enough",
        ))
        placements.append(_make_placement(
            card, ask_mem, num_chunks, "recall_demo",
            difficulty_mode="recall_near",
            recall_need="fine_visual_verification",
        ))

    # Historical visual-recall variants for every single-emit family. This is
    # what lets OCR/spatial/action/object-relation tasks appear as recall
    # tasks when the support is no longer visually present.
    ask_mid = _ask_from_band(emit, SE_RECALL_MID, num_chunks, rng, sign=+1)
    if ask_mid is not None:
        placements.append(_make_placement(
            card, ask_mid, num_chunks, "recall_demo",
            difficulty_mode="recall_mid",
            recall_need="historical_visual",
        ))
    ask_deep = _ask_from_band(emit, SE_RECALL_DEEP, num_chunks, rng, sign=+1)
    if ask_deep is not None:
        placements.append(_make_placement(
            card, ask_deep, num_chunks, "recall_demo",
            difficulty_mode="recall_deep",
            recall_need="compressed_history",
        ))

    profile = PLACEMENT_PROFILE.get(card.family, "realtime")
    if profile == "forward" or card.family in {"CR2", "CR5", "E2"}:
        # Keep explicit wait/silent supervision for naturally temporal
        # families without making every static attribute question a future
        # prediction prompt.
        lead_bands = [SE_LEAD_SHORT, SE_LEAD_LONG]
        if card.family == "CR5":
            lead_bands.extend(SE_LEAD_OVO_CRR)
        for band in lead_bands:
            ask_forward = _ask_from_band(emit, band, num_chunks, rng, sign=-1)
            if ask_forward is not None:
                placements.append(_make_placement(
                    card, ask_forward, num_chunks, "silent_then_response",
                    difficulty_mode=(
                        "ovo_crr_wait" if card.family == "CR5" and band in SE_LEAD_OVO_CRR
                        else "future_wait"
                    ),
                    recall_need="future_not_available",
                ))
        if not any(p.mechanism == "silent_then_response" for p in placements):
            ask_forward = _ask_from_band(emit, (4, 8), num_chunks, rng, sign=-1)
            if ask_forward is not None:
                placements.append(_make_placement(
                    card, ask_forward, num_chunks, "silent_then_response",
                    difficulty_mode="future_wait",
                    recall_need="future_not_available",
                ))

    if not placements:
        # very short video — fallback to direct
        gap = _randint_safe(rng, 0, max(0, num_chunks - 1 - emit))
        placements.append(_make_placement(
            card, emit + gap, num_chunks, "direct",
            difficulty_mode="current_direct",
            recall_need="fallback",
        ))

    return _dedupe_placements(placements)


def _select_multi_emit_subset(card: Card) -> List[GoldEmit]:
    """Pick a compact local subset for one active multi-answer episode."""
    emits = sorted(card.gold_emits, key=lambda e: e.chunk)
    if card.family == "F5":
        if len(emits) <= MAX_REC_EMIT_RESPONSES:
            return emits
        idxs = {
            round(i * (len(emits) - 1) / (MAX_REC_EMIT_RESPONSES - 1))
            for i in range(MAX_REC_EMIT_RESPONSES)
        }
        return [emits[i] for i in sorted(idxs)]

    if card.family in {"F7", "CRR1"}:
        first_yes_idx = next(
            (i for i, e in enumerate(emits)
             if str(e.value).strip().lower() == "yes"),
            None,
        )
        if first_yes_idx is not None and first_yes_idx > 0:
            first_yes = emits[first_yes_idx]
            no_before = [
                e for e in emits[:first_yes_idx]
                if str(e.value).strip().lower() == "no"
            ][-2:]
            yes_after = [
                e for e in emits[first_yes_idx + 1:]
                if str(e.value).strip().lower() == "yes"
            ]
            far_yes = next(
                (
                    e for e in yes_after
                    if STATUS_FAR_AFTER_MIN_GAP <= e.chunk - first_yes.chunk <= STATUS_FAR_AFTER_MAX_GAP
                ),
                None,
            )
            if far_yes is not None:
                middle_yes = next(
                    (
                        e for e in yes_after
                        if e.chunk < far_yes.chunk
                        and e.chunk - first_yes.chunk <= VISUAL_WINDOW_CHUNKS
                    ),
                    None,
                )
                subset = no_before + [first_yes]
                if middle_yes is not None:
                    subset.append(middle_yes)
                subset.append(far_yes)
                subset = sorted(
                    {int(e.chunk): e for e in subset}.values(),
                    key=lambda e: e.chunk,
                )[:MAX_MULTI_EMIT_RESPONSES]
            else:
                start = max(0, first_yes_idx - 2)
                end = min(len(emits), start + MAX_MULTI_EMIT_RESPONSES)
                if end <= first_yes_idx:
                    end = min(len(emits), first_yes_idx + 1)
                subset = emits[start:end]
            vals = {str(e.value).strip().lower() for e in subset}
            if {"no", "yes"}.issubset(vals):
                return subset

    if len(emits) <= MAX_MULTI_EMIT_RESPONSES:
        if emits[-1].chunk - emits[0].chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN:
            return emits

    best: List[GoldEmit] = []
    best_key: Tuple[int, int, int] = (-1, 10**9, 10**9)
    for i, start in enumerate(emits):
        cur = [
            e for e in emits[i:]
            if e.chunk - start.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
        ][:MAX_MULTI_EMIT_RESPONSES]
        if len(cur) < 2:
            continue
        span = cur[-1].chunk - cur[0].chunk
        key = (len(cur), -span, -cur[0].chunk)
        if key > best_key:
            best_key = key
            best = cur
    return best


def place_multi_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Generate ONE compact multi-answer placement.

    A multi-answer question is allowed, but it must be one local episode. A
    question spanning half the video would suppress too many independent
    Q/A episodes and make the training target ambiguous.
    """
    if not card.gold_emits:
        return []
    emits = _select_multi_emit_subset(card)
    if len(emits) < 2:
        return []
    first = min(e.chunk for e in emits)
    last = max(e.chunk for e in emits)
    difficulty_mode = "multi_emit"
    if card.family == "F5":
        # OVO REC asks the cumulative counting question from the beginning and
        # probes later. Keep the active query open from c0 instead of asking a
        # few seconds before the first counted occurrence.
        ask = 0
        difficulty_mode = "ovo_rec_cumulative"
    elif card.family in {"F7", "CRR1"}:
        # Status probes are immediate Yes/No checks at probe time, not a
        # persistent wait-before-first-answer question.
        ask = first
        difficulty_mode = "status_probe"
    else:
        lead = _randint_safe(rng, ME_LEAD_RANGE[0], min(ME_LEAD_RANGE[1], first))
        ask = max(0, first - lead)
    emit_by_chunk = {e.chunk: e.value for e in emits}
    end = min(num_chunks - 1, last + 1)
    actions: Dict[int, Tuple[GoldKind, str]] = {}
    for c in range(ask, end + 1):
        if c in emit_by_chunk:
            actions[c] = ("response", emit_by_chunk[c])
        else:
            actions[c] = ("silent", "")
    return [Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism="multi_emit",
        difficulty_mode=difficulty_mode,
        chunk_actions=actions,
    )]


def place_card(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    if card.question_type == "single_emit":
        return place_single_emit(card, num_chunks, rng)
    return place_multi_emit(card, num_chunks, rng)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "best", "by", "chunk",
    "did", "does", "for", "from", "has", "in", "is", "it", "of", "on",
    "or", "the", "this", "to", "was", "what", "which", "with",
}


def _strip_mc_label(text: str) -> str:
    return re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][\).:])\s*", "", str(text or "")).strip()


def _card_answer_text(card: Card) -> str:
    if card.answer_form == "multiple_choice" and card.options and card.correct_option:
        idx = ord(str(card.correct_option).strip().upper()[:1] or "A") - ord("A")
        if 0 <= idx < len(card.options):
            return _strip_mc_label(card.options[idx])
    if card.gold_emits:
        return str(card.gold_emits[-1].value or "")
    return ""


def _tokens(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(str(text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _evidence_text_for_chunks(evidence_by_chunk: Dict[int, Dict], chunks: Iterable[int]) -> str:
    parts: List[str] = []
    for c in chunks:
        cap = evidence_by_chunk.get(int(c)) or {}
        parts.append(str(cap.get("think", "")))
        for ent in cap.get("visible_entities") or []:
            if isinstance(ent, dict):
                parts.append(str(ent.get("desc", "")))
                parts.append(str(ent.get("action", "")))
            else:
                parts.append(str(ent))
        for fact in cap.get("atomic_facts") or []:
            parts.append(str(fact.get("fact", "")) if isinstance(fact, dict) else str(fact))
        for ocr in cap.get("ocr") or []:
            parts.append(str(ocr.get("text", "")) if isinstance(ocr, dict) else str(ocr))
        parts.append(str(cap.get("spatial", "")))
    return " ".join(parts).lower()


def _answer_terms_present(card: Card, text: str) -> bool:
    answer = _card_answer_text(card)
    answer_norm = " ".join(_tokens(answer))
    if not answer_norm:
        return False
    text_l = str(text or "").lower()
    answer_tokens = _tokens(answer)
    if card.answer_form in {"short_exact", "number", "binary"}:
        return answer_norm in " ".join(_tokens(text_l))
    if card.answer_form == "multiple_choice":
        # Need at least two meaningful option tokens unless the answer is an
        # exact OCR-like string. This avoids relabeling on common words.
        if len(answer_tokens) == 1:
            return answer_tokens[0] in _tokens(text_l) and len(answer_tokens[0]) >= 4
        hit = sum(1 for t in set(answer_tokens) if t in _tokens(text_l))
        return hit >= min(2, len(set(answer_tokens)))
    hit = sum(1 for t in set(answer_tokens) if t in _tokens(text_l))
    return hit >= max(2, min(4, len(set(answer_tokens))))


def _correct_option_text(card: Card) -> str:
    correct = str(card.correct_option or "").strip().upper()
    options = list(card.options or [])
    if len(correct) != 1 or correct < "A" or correct > "Z" or not options:
        return ""
    idx = ord(correct) - ord("A")
    if idx < 0 or idx >= len(options):
        return ""
    return _strip_mc_label(str(options[idx]))


def _is_unanswerable_card(card: Card) -> bool:
    if card.family == "HLD1":
        return True
    return _correct_option_text(card).strip().lower() == "unable to answer"


def _support_span(card: Card) -> int:
    support = [int(x) for x in (card.grounding_frames or [])]
    if not support:
        return 0
    return max(support) - min(support) + 1


def refine_placements_with_evidence(
    card: Card,
    placements: List[Placement],
    evidence: Optional[List[Dict]] = None,
) -> List[Placement]:
    """Low-cost first-turn availability refinement.

    This is deliberately heuristic and local. It prevents obvious bad recall
    labels when the answer is already visible in the current window or when a
    simple factual answer is explicitly present in recent text memory, while
    preserving recall candidates for temporal/causal/fine-grained families.
    The expensive semantic judge can be added later as an optional parallel
    pass over these already-filtered candidates.
    """
    if not evidence or card.question_type != "single_emit":
        return _dedupe_placements(placements)

    evidence_by_chunk = {
        int(cap.get("chunk_idx", -1)): cap
        for cap in evidence
        if cap.get("chunk_idx") is not None
    }
    refined: List[Placement] = []
    for p in placements:
        if p.mechanism != "recall_demo":
            refined.append(p)
            continue

        current_lo = max(0, int(p.ask_chunk) - VISUAL_WINDOW_CHUNKS + 1)
        current_text = _evidence_text_for_chunks(
            evidence_by_chunk, range(current_lo, int(p.ask_chunk) + 1)
        )
        if _answer_terms_present(card, current_text):
            p.mechanism = "memory_direct"
            p.difficulty_mode = "current_or_memory_direct"
            p.recall_need = "current_window_answerable"
            refined.append(p)
            continue

        memory_hi = max(0, int(p.ask_chunk) - VISUAL_WINDOW_CHUNKS)
        memory_text = _evidence_text_for_chunks(evidence_by_chunk, range(0, memory_hi))
        simple_memory_case = (
            card.family in SIMPLE_MEMORY_FAMILIES
            and _support_span(card) <= 4
            and _answer_terms_present(card, memory_text)
        )
        if simple_memory_case:
            p.mechanism = "memory_direct"
            p.difficulty_mode = "memory_direct"
            p.recall_need = "memory_text_exact"
            refined.append(p)
            continue

        # Keep as recall. Hard families and multi-support questions are the
        # main source of temporal/order/causal/fine-grained recall difficulty.
        if card.family in HARD_RECALL_FAMILIES or _support_span(card) > 4:
            p.recall_need = p.recall_need or "hard_historical_visual"
        refined.append(p)
    return _dedupe_placements(refined)


def placement_timing_verdict(card: Card, placement: Placement) -> Tuple[bool, str]:
    """Validate ask/answer timing against support availability.

    This is intentionally stricter than schema validation: pass3a decides
    what evidence supports the answer; pass3b must ensure the question is
    asked only at a time when the chosen mechanism is causally valid.
    """
    if not card.gold_emits:
        return False, "no_gold_emits"
    if not placement.chunk_actions:
        return False, "no_chunk_actions"

    response_chunks = [
        int(c) for c, (kind, _value) in placement.chunk_actions.items()
        if kind == "response"
    ]
    if not response_chunks:
        return False, "no_response_chunk"

    if card.question_type == "multi_emit":
        # Multi-emit placements may intentionally select a compact local
        # subset from a longer card. Validate against the selected placement's
        # first response chunk, not the card's global first emit.
        first_emit = min(response_chunks)
        if placement.ask_chunk > first_emit:
            return False, "multi_ask_after_first_emit"
        return True, "pass"

    emit = int(card.gold_emits[0].chunk)
    support = [int(g) for g in (card.grounding_frames or [emit])]
    max_support = max(support)
    first_response = min(response_chunks)
    if emit < max_support:
        return False, "emit_before_latest_support"
    if first_response < max_support:
        return False, "response_before_latest_support"

    if placement.mechanism == "silent_then_response":
        if placement.ask_chunk >= emit:
            return False, "forward_ask_not_before_emit"
        return True, "pass"

    if placement.mechanism == "recall_demo":
        if placement.ask_chunk <= max_support:
            return False, "recall_ask_before_support"
        if placement.ask_chunk - max_support <= VISUAL_WINDOW_CHUNKS:
            return False, "recall_support_still_visual"
        if first_response != placement.ask_chunk:
            return False, "recall_response_not_at_ask"
        return True, "pass"

    if placement.mechanism == "memory_direct":
        if placement.ask_chunk <= max_support:
            return False, "memory_ask_before_support"
        if placement.ask_chunk - max_support <= VISUAL_WINDOW_CHUNKS:
            return False, "memory_support_still_visual"
        if first_response != placement.ask_chunk:
            return False, "memory_response_not_at_ask"
        return True, "pass"

    if placement.mechanism == "direct":
        if placement.ask_chunk < max_support:
            return False, "direct_ask_before_support"
        if placement.ask_chunk - max_support > VISUAL_WINDOW_CHUNKS:
            return False, "direct_gap_requires_recall"
        return True, "pass"

    return True, "pass"


# ---------------------------------------------------------------------------
# Trajectory selection — pick MAX_QUESTIONS_PER_TRAJECTORY placements per video
# ---------------------------------------------------------------------------


def _placement_chunks(p: Placement) -> set:
    return {int(c) for c in p.chunk_actions.keys()}


def _placement_span_len(p: Placement) -> int:
    chunks = _placement_chunks(p)
    if not chunks:
        return 0
    return max(chunks) - min(chunks) + 1


def _recall_floor(max_q: int) -> int:
    if max_q <= 0:
        return 0
    return max(1, int(max_q * RECALL_TARGET_FRACTION + 0.999))


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
      +spread: distance to nearest already-picked ask_chunk / 8 (cap 3.0)

    Strict constraints:
      - at most ONE placement per card_id
      - at most ONE placement per family per trajectory
      - independent ask chunks must satisfy the video-length ask-gap floor
      - no overlapping placement chunks. This enforces a single active
        question at a time; multi-answer supervision is allowed only inside
        one multi_emit question, never as competing questions on the same
        answer chunk.
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
    used_chunks: set = set()
    used_ask: List[int] = []
    ask_gap_floor = question_ask_gap_floor(num_chunks)

    def feasible(p: Placement, card: Card) -> bool:
        if card.card_id in seen_cards:
            return False
        if card.family in seen_families:
            return False
        if used_ask and min(abs(int(p.ask_chunk) - x) for x in used_ask) < ask_gap_floor:
            return False
        return not (_placement_chunks(p) & used_chunks)

    def score(p: Placement, card: Card) -> float:
        s = 0.0
        if card.family not in seen_families:
            s += 3.0
        if p.mechanism not in seen_mechs:
            s += 2.0
        if card.answer_form not in seen_aforms:
            s += 1.0
        s += 1.0  # base for new card
        if p.mechanism == "recall_demo":
            if p.difficulty_mode == "recall_deep":
                s += 0.8
            elif p.difficulty_mode == "recall_mid":
                s += 0.4
            elif p.difficulty_mode == "recall_near":
                s += 0.2
        elif p.mechanism == "memory_direct":
            # Useful hard negatives against overusing recall, but do not let
            # them crowd out true recall slots.
            s += 0.15
        if used_ask:
            min_dist = min(abs(p.ask_chunk - x) for x in used_ask)
            s += min(min_dist / SPREAD_SCORE_DENOM, SPREAD_SCORE_CAP)
        else:
            s += SPREAD_SCORE_CAP
        # Long active spans are legitimate for one-question multi-answer
        # tasks, but they reduce the number of independent Q/A episodes
        # in a trajectory. Penalize them rather than banning them.
        s -= min(_placement_span_len(p) / 24.0, 2.0)
        s += FAMILY_SELECTION_BOOST.get(card.family, 0.0)
        return s

    def take_best(predicate) -> bool:
        best_score = -1e9
        best_idx = -1
        for i, (p, card) in enumerate(pool):
            if not feasible(p, card) or not predicate(p, card):
                continue
            s = score(p, card)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return False
        p, card = pool.pop(best_idx)
        selected.append(p)
        seen_families.add(card.family)
        seen_mechs.add(p.mechanism)
        seen_aforms.add(card.answer_form)
        seen_cards.add(card.card_id)
        used_chunks.update(_placement_chunks(p))
        used_ask.append(p.ask_chunk)
        return True

    unanswerable_reserve_key = min(
        (
            card.card_id
            for p, card in pool
            if p.mechanism != "recall_demo" and _is_unanswerable_card(card)
        ),
        default="",
    )
    reserve_unanswerable = bool(unanswerable_reserve_key) and (
        stable_mod(unanswerable_reserve_key, "abstention_reserve", modulo=100)
        < HLD_ABSTENTION_RESERVE_PERCENT
    )

    # Recall is sparse in row count (one tool-turn row per recall question).
    # Select a floor before filling realtime/current questions so SFT/RL see
    # enough tool-use supervision without allowing overlapping pending queries.
    # HLD / "Unable to answer" cards are recall-worthy abstention cases: the
    # model should inspect history before choosing the Unable option.
    if len(selected) < max_q:
        if not take_best(lambda p, card: p.mechanism == "multi_emit" and card.family == "F5"):
            if not take_best(lambda p, card: p.mechanism == "multi_emit" and card.family == "CRR1"):
                take_best(lambda p, card: p.mechanism == "multi_emit" and card.family == "F7")
    if len(selected) < max_q and max_q >= 10:
        take_best(
            lambda p, card: (
                p.mechanism == "multi_emit"
                and card.family in {"F5", "CRR1", "F7"}
            )
        )
    if len(selected) < max_q:
        if not take_best(
            lambda p, card: (
                p.mechanism == "silent_then_response"
                and card.family == "CR5"
                and p.difficulty_mode == "ovo_crr_wait"
                and _placement_span_len(p) >= 25
            )
        ):
            take_best(
                lambda p, card: (
                    p.mechanism == "silent_then_response"
                    and card.family == "CR5"
                    and p.difficulty_mode == "ovo_crr_wait"
                )
            )

    if len(selected) < max_q:
        take_best(lambda p, card: p.mechanism == "recall_demo" and _is_unanswerable_card(card))

    recall_capacity = max_q
    target_recall = min(recall_capacity, _recall_floor(max_q))
    while (
        len(selected) < max_q
        and sum(1 for p in selected if p.mechanism == "recall_demo") < target_recall
        and take_best(lambda p, _card: p.mechanism == "recall_demo")
    ):
        pass

    if reserve_unanswerable and len(selected) < max_q:
        take_best(lambda p, card: p.mechanism != "recall_demo" and _is_unanswerable_card(card))

    while len(selected) < max_q and pool:
        if not take_best(lambda _p, _card: True):
            break

    return selected


# ---------------------------------------------------------------------------
# Recall augmentation — independent of rollout
# ---------------------------------------------------------------------------


def assign_recall_noise(
    placements: List[Placement],
    rng: random.Random,
    cards_by_id: Optional[Dict[str, Card]] = None,
) -> None:
    """Assign recall demonstrations without creating no-answer questions.

    Mutates placement.recall_at in place.

    - recall_demo: recall at the response chunk with oracle/noisy evidence.
    - silent_then_response: optional recall+silent probes during the wait.
      These retrieve elapsed history and still keep the query open when the
      answer is not available yet.
    - F5 multi_emit: later cumulative count responses can recall previous
      occurrences before answering from current+historical evidence; long
      gaps before later occurrences can also recall history and stay silent
      because the final cumulative answer is not complete yet.
    - F7/CRR1 multi_emit: far-after-Yes probes may recall the event chunk when
      it has left the visual window.
    - memory_direct hard historical families: a sampled subset gets a recall
      probe candidate. Pass3B rollout filtering still removes it when clean
      memory already answers or when there is no real memory gap.

    Every selected question still has a grounded answer in the same
    trajectory. If failure-mode data is needed later, it should live in a
    separate diagnostic dataset, not in SFT/RL/eval training trajectories.
    """
    def mark_response_recall(p: Placement, c: int, reason: str) -> None:
        r = rng.random()
        p.recall_at[int(c)] = "oracle" if r < RECALL_OK_RATE else "noisy"
        p.recall_reason_at[int(c)] = reason

    def schedule_wait_recalls(p: Placement) -> None:
        response_chunks = sorted(
            int(c) for c, (kind, _value) in p.chunk_actions.items()
            if kind == "response"
        )
        if not response_chunks:
            return
        first_response = response_chunks[0]
        lead = first_response - int(p.ask_chunk)
        if lead < RECALL_WAIT_MIN_LEAD:
            return
        exact_candidates: List[Tuple[int, str]] = []
        if lead >= RECENT_THINKS_HORIZON + RECALL_WAIT_MIN_LEAD:
            exact_candidates.append((
                int(p.ask_chunk) + RECENT_THINKS_HORIZON + 1,
                "long_wait_history_check",
            ))
        if lead >= 36:
            fractions = (0.25, 0.50, 0.75)
        elif lead >= 16:
            fractions = (0.33, 0.67)
        else:
            fractions = (0.50,)
        reasons = ("memory_unclear", "related_history_check", "pre_answer_check")
        chosen: List[int] = []
        for c, reason in exact_candidates:
            c = min(first_response - 1, max(int(p.ask_chunk) + 1, int(c)))
            if c <= 0 or c in chosen:
                continue
            if p.chunk_actions.get(c, ("", ""))[0] != "silent":
                continue
            p.recall_at[c] = "not_yet"
            p.recall_reason_at[c] = reason
            chosen.append(c)
        for i, frac in enumerate(fractions[:RECALL_WAIT_PROBE_MAX]):
            c = int(p.ask_chunk) + max(2, round(lead * frac))
            c = min(first_response - 1, max(int(p.ask_chunk) + 1, c))
            if c <= 0 or c in chosen:
                continue
            if p.chunk_actions.get(c, ("", ""))[0] != "silent":
                continue
            p.recall_at[c] = "not_yet"
            p.recall_reason_at[c] = reasons[min(i, len(reasons) - 1)]
            chosen.append(c)

    def response_needs_historical_recall(
        card: Optional[Card],
        c: int,
    ) -> bool:
        if card is None:
            return False
        support = []
        for raw in card.grounding_frames or []:
            try:
                support.append(int(raw))
            except (TypeError, ValueError):
                continue
        if not support:
            return False
        visual_start = max(0, int(c) - VISUAL_WINDOW_CHUNKS + 1)
        return any(s < visual_start for s in support)

    for p in placements:
        if p.mechanism == "recall_demo":
            for c, (kind, _) in p.chunk_actions.items():
                if kind != "response":
                    continue
                mark_response_recall(p, int(c), "historical_answer")
        elif p.mechanism == "silent_then_response":
            schedule_wait_recalls(p)
            card = cards_by_id.get(p.card_id) if cards_by_id else None
            for c, (kind, _value) in p.chunk_actions.items():
                if kind != "response":
                    continue
                if response_needs_historical_recall(card, int(c)):
                    mark_response_recall(
                        p,
                        int(c),
                        "future_answer_historical_anchor",
                    )
        elif p.mechanism == "multi_emit":
            response_chunks = sorted(
                int(c) for c, (kind, _value) in p.chunk_actions.items()
                if kind == "response"
            )
            if not response_chunks:
                continue
            if p.difficulty_mode == "ovo_rec_cumulative":
                wait_added = 0
                for prev_c, next_c in zip(response_chunks, response_chunks[1:]):
                    lead = int(next_c) - int(prev_c)
                    if lead < RECENT_THINKS_HORIZON + RECALL_WAIT_MIN_LEAD:
                        continue
                    c = max(
                        int(prev_c) + RECENT_THINKS_HORIZON + 1,
                        int(prev_c) + max(2, round(lead * 0.55)),
                    )
                    c = min(int(next_c) - 1, max(int(prev_c) + 1, c))
                    if p.chunk_actions.get(c, ("", ""))[0] != "silent":
                        continue
                    p.recall_at[c] = "not_yet"
                    p.recall_reason_at[c] = "cumulative_waiting_more_events"
                    wait_added += 1
                    if wait_added >= RECALL_MULTI_WAIT_PROBE_MAX:
                        break
                for c in response_chunks[1:]:
                    mark_response_recall(p, c, "cumulative_history")
            elif p.difficulty_mode == "status_probe":
                yes_chunks = [
                    c for c in response_chunks
                    if str(p.chunk_actions[c][1]).strip().lower() == "yes"
                ]
                if not yes_chunks:
                    continue
                first_yes = min(yes_chunks)
                for c in yes_chunks:
                    if c - first_yes > VISUAL_WINDOW_CHUNKS:
                        mark_response_recall(p, c, "status_history")
        elif p.mechanism == "memory_direct" and cards_by_id:
            card = cards_by_id.get(p.card_id)
            if not card or card.family not in HARD_RECALL_FAMILIES:
                continue
            rate = MEMORY_DIRECT_RECALL_FAMILY_RATE.get(
                card.family,
                MEMORY_DIRECT_RECALL_PROBE_RATE,
            )
            if rng.random() >= rate:
                continue
            response_chunks = sorted(
                int(c) for c, (kind, _value) in p.chunk_actions.items()
                if kind == "response"
            )
            if not response_chunks:
                continue
            mark_response_recall(
                p,
                response_chunks[0],
                "memory_text_needs_visual_verification",
            )
            p.recall_need = "memory_direct_visual_verification"


# ---------------------------------------------------------------------------
# Trajectory rendering — produces SFT samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    chunk_idx: int
    sample_kind: str               # silent | response | recall+response
    placement_id: str
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    response_text: str = ""        # empty for silent
    recall_query: Optional[Dict] = None
    recall_result_kind: Optional[str] = None  # oracle/noisy
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
            if placement.recall_at.get(c) == "not_yet":
                samples.append(Sample(
                    chunk_idx=c,
                    sample_kind="recall+silent",
                    placement_id=pid,
                    card_id=card.card_id,
                    ask_chunk=placement.ask_chunk,
                    mechanism=placement.mechanism,
                    recall_query=card.recall_query,
                    recall_result_kind="not_yet",
                    extra={"recall_reason": placement.recall_reason_at.get(c, "")},
                ))
                continue
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
        if c in placement.recall_at:
            rkind = placement.recall_at[c]
            if rkind == "failure":
                raise ValueError(
                    f"recall failure is disabled in production trajectories: "
                    f"card={card.card_id} chunk={c}"
                )
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
                extra={"recall_reason": placement.recall_reason_at.get(c, "")},
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
    per_chunk: Dict[int, List[Tuple[str, str, Placement, Card, Dict]]] = {}
    cards_by_id = {c.card_id: c for c in cards}
    for cid, plcs in placements_by_card.items():
        card = cards_by_id[cid]
        for p in plcs:
            for s in render_placement(card, p):
                per_chunk.setdefault(s.chunk_idx, []).append(
                    (s.sample_kind, s.response_text, p, card, dict(s.extra or {}))
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
        if len(candidates) > 1:
            owners = [
                f"{card.card_id}@{p.ask_chunk}:{kind}"
                for kind, _value, p, card, _extra in candidates
            ]
            raise ValueError(
                f"overlapping question placements at chunk {c}: "
                + ", ".join(owners)
            )
        candidates.sort(key=lambda t: -PRIORITY[t[0]])
        kind, value, p, card, extra = candidates[0]
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
            extra=extra,
        ))
    return all_samples


# ---------------------------------------------------------------------------
# Sanity helpers (used by simulator)
# ---------------------------------------------------------------------------


def is_response_kind(k: str) -> bool:
    return k in ("response", "recall+response")


def is_silent_kind(k: str) -> bool:
    return k in ("silent", "patrol", "recall+silent", "compress_silent")
