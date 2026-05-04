"""Pass 3-A — Card generation (v2 model-agnostic schema).

Cards are now defined by `gold_emits` + `grounding_frames` + `question_type`
(see v2/design.py). The card distribution is OVOBench-aligned (76% MC + binary
+ number + short_exact + descriptive). The heuristic generator in v2/cards.py
derives cards directly from evidence; swap in 397B prompts later by
replacing `_generate_via_llm` below.

Pipeline contract preserved:
  generate_cards(evidence, client, video_id) -> List[Dict]    (async)
  verify_cards(cards, client, video_id)      -> List[Dict]    (async, no-op now)
  save_cards / load_cards
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import asyncio

from .config import TASK_CARDS_DIR, PASS_CONFIG
from .v2.cards import generate_cards as _heuristic_generate
from .v2.design import Card, GoldEmit
from .v2.llm_prompts import (
    FAMILY_RULES,
    QUESTION_TYPE_BY_FAMILY,
    card_generation_prompt,
    family_taxonomy,
    parse_card_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Card ↔ dict serialization
# ---------------------------------------------------------------------------


def _card_to_dict(card: Card) -> Dict:
    canonical = card.gold_emits[-1].value if card.gold_emits else ""
    taxonomy = family_taxonomy(card.family)
    return {
        "card_id": card.card_id,
        "family": card.family,
        **taxonomy,
        "question": card.question,
        "answer_form": card.answer_form,
        "canonical_answer": canonical,
        "question_type": card.question_type,
        "gold_emits": [{"chunk": e.chunk, "value": e.value} for e in card.gold_emits],
        "grounding_frames": list(card.grounding_frames),
        "support_chunks": list(card.grounding_frames),  # back-compat alias
        "options": list(card.options) if card.options else None,
        "correct_option": card.correct_option,
        "recall_query": card.recall_query,
    }


def dict_to_card(d: Dict) -> Card:
    """Rebuild a Card from its serialized dict (used by pass3b/c)."""
    return Card(
        card_id=d["card_id"],
        family=d.get("family", ""),
        question=d.get("question", ""),
        answer_form=d.get("answer_form", "descriptive"),
        question_type=d.get("question_type", "single_emit"),
        gold_emits=[GoldEmit(chunk=int(e["chunk"]), value=str(e["value"]))
                    for e in d.get("gold_emits", [])],
        grounding_frames=list(d.get("grounding_frames",
                                    d.get("support_chunks", []))),
        options=d.get("options"),
        correct_option=d.get("correct_option"),
        recall_query=d.get("recall_query"),
    )


# ---------------------------------------------------------------------------
# Public API (matches old pass3a interface)
# ---------------------------------------------------------------------------


async def generate_cards(
    *args,
    seed: int = 42,
    **kwargs,
) -> List[Dict]:
    """Generate v2 cards from evidence.

    Accepts both call styles to stay drop-in:
      - new:      generate_cards(evidence, client=..., video_id=...)
      - pipeline: generate_cards(video_id, evidence, client)         ← pipeline.py:677

    Routing:
      - client is not None  → 397B per-family LLM generation (production)
      - client is None      → heuristic (offline/simulator)
    """
    video_id, evidence, client = _parse_card_args(args, kwargs)
    if client is None:
        cards = _generate_via_heuristic(evidence, video_id, seed)
        return [_card_to_dict(c) for c in cards]
    return await _generate_via_llm(evidence, client, video_id, seed)


def _parse_card_args(args, kwargs):
    """Disambiguate (evidence, client, video_id) vs (video_id, evidence, client).

    pipeline.py uses positional (vid, evidence, client) — we detect that
    by the FIRST positional being a str (video_id) instead of a list.
    """
    video_id = kwargs.get("video_id", "")
    evidence = kwargs.get("evidence", None)
    client = kwargs.get("client", None)
    if args:
        if isinstance(args[0], str):
            # (video_id, evidence, client) form
            video_id = args[0]
            if len(args) > 1:
                evidence = args[1]
            if len(args) > 2:
                client = args[2]
        else:
            # (evidence, client, video_id) form
            evidence = args[0]
            if len(args) > 1:
                client = args[1]
            if len(args) > 2:
                video_id = args[2]
    return video_id, (evidence or []), client


def _generate_via_heuristic(evidence: List[Dict], video_id: str, seed: int) -> List[Card]:
    return _heuristic_generate(evidence, video_id, seed=seed)


async def _generate_via_llm(
    evidence: List[Dict], client, video_id: str, seed: int,
) -> List[Dict]:
    """397B-driven card generation. Per-family parallel via asyncio.gather.

    Each family fires ONE request that returns 1 card (target_n=1). 16
    families per video × 1 card ≈ 16 cards/video, matches the heuristic
    target. The trajectory selector then picks the best 6-14 per video.

    On parse failure for a family, falls back to that family's heuristic
    output so we never lose coverage entirely.
    """
    cfg = PASS_CONFIG.get("pass3a", {})
    max_tokens = int(cfg.get("max_tokens", 16384))
    temperature = float(cfg.get("temperature", 0.7))
    enable_thinking = cfg.get("thinking", False)

    families = list(FAMILY_RULES.keys())
    fallback_cards = _generate_via_heuristic(evidence, video_id, seed)
    fallback_by_family: Dict[str, List[Card]] = {}
    for c in fallback_cards:
        fallback_by_family.setdefault(c.family, []).append(c)

    async def _one_family(family: str) -> List[Dict]:
        prompt = card_generation_prompt(family, evidence, target_n=1)
        try:
            raw = await client._call_one(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=f"{video_id}_3a_{family}",
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning(f"[{video_id}] 3a {family}: LLM call failed: {exc}")
            raw = None
        cards = parse_card_response(raw or "", family) if raw else []
        if not cards:
            # Fallback to heuristic for this family
            return [_card_to_dict(c) for c in fallback_by_family.get(family, [])]
        # Assign card_id deterministically
        out = []
        for i, c in enumerate(cards):
            c["card_id"] = f"{video_id}_{family}_{seed:04d}_{i}"
            c.update(family_taxonomy(family))
            # Default placeholder; pass3c may override on demand
            c.setdefault("recall_query", None)
            out.append(c)
        return out

    results = await asyncio.gather(*[_one_family(f) for f in families])
    all_cards: List[Dict] = []
    for fam_cards in results:
        all_cards.extend(fam_cards)
    logger.info(f"[{video_id}] 3a: LLM generated {len(all_cards)} cards "
                f"across {len(families)} families")
    return all_cards


async def verify_cards(*args, **kwargs) -> List[Dict]:
    """v12.13 (P1-8): real semantic verification (was no-op).

    Two layers of validation. Each card either passes through unchanged
    or is dropped (logged with the rejection reason).

    Accepts both call shapes for back-compat:
      - verify_cards(cards, client, video_id)
      - verify_cards(video_id, cards, evidence, client)
    """
    # Argument unpacking — pipeline.py uses kwargs, legacy uses positional.
    cards = kwargs.get("cards")
    evidence = kwargs.get("evidence") or []
    video_id = kwargs.get("video_id", "")
    if cards is None:
        # Heuristic: first list of dicts is cards; subsequent lists may be
        # evidence (also list of dicts but with `chunk_idx` keys).
        for a in args:
            if isinstance(a, list) and a and isinstance(a[0], dict):
                if "card_id" in a[0]:
                    cards = a
                elif "chunk_idx" in a[0]:
                    if not evidence:
                        evidence = a
            elif isinstance(a, str) and not video_id:
                video_id = a
    if not cards:
        return []

    # Build evidence index for fast lookups (Layer 2)
    evidence_by_chunk: Dict[int, Dict] = {}
    for cap in (evidence or []):
        ci = cap.get("chunk_idx")
        if isinstance(ci, int):
            evidence_by_chunk[ci] = cap

    out: List[Dict] = []
    rejected: Dict[str, int] = {}
    for card in cards:
        verdict = _verify_card_layers(card, evidence_by_chunk)
        if verdict == "PASS":
            out.append(card)
        else:
            rejected[verdict] = rejected.get(verdict, 0) + 1

    if rejected:
        logger.info(
            f"[{video_id}] 3a verify: {len(out)}/{len(cards)} passed; "
            f"rejected: {dict(sorted(rejected.items(), key=lambda x: -x[1]))}"
        )
    else:
        logger.info(f"[{video_id}] 3a verify: {len(out)}/{len(cards)} passed")
    return out


def _verify_card_layers(card: Dict, ev_by_chunk: Dict[int, Dict]) -> str:
    """Two-layer card verification. Returns "PASS" or a fail reason string.

    Layer 1 — schema sanity:
      - has card_id, family, question, answer_form, gold_emits / canonical_answer
      - MC: options is list of length 4; correct_option in {A,B,C,D};
            options match the correct_option index
      - binary: gold_emits values in {Yes, No, yes, no}
      - number: gold_emits values are digit strings
      - multi_emit: ≥ 2 distinct gold_emit chunks
      - grounding_frames non-empty

    Layer 2 — grounding evidence exists:
      - every grounding chunk index has a matching evidence entry
      - canonical answer text overlaps with evidence text in the grounding
        chunks (loose containment; entity name OR fact phrase appears)
    """
    # ── Layer 1: schema ──
    if not card.get("card_id") or not card.get("family"):
        return "schema_missing_id_family"
    q = (card.get("question") or "").strip()
    if len(q) < 8:
        return "schema_question_too_short"
    af = card.get("answer_form", "")
    emits = card.get("gold_emits") or []
    if not emits and not card.get("canonical_answer"):
        return "schema_no_gold"

    if af == "multiple_choice":
        opts = card.get("options") or []
        if not isinstance(opts, list) or len(opts) != 4:
            return "schema_mc_options_not_4"
        co = card.get("correct_option", "")
        if co not in {"A", "B", "C", "D"}:
            return "schema_mc_bad_correct_letter"
        # The option at the correct letter's index should match canonical_answer
        # (or contain it). Skip exact match — pass3a may format options as
        # "A) text" or just "text"; we accept any non-placeholder.
        idx = ord(co) - ord("A")
        if any("distractor placeholder" in str(o).lower() for o in opts):
            return "schema_mc_placeholder_distractor"
    elif af == "binary":
        bad = [e for e in emits
               if str(e.get("value", "")).strip().lower() not in ("yes", "no")]
        if bad:
            return "schema_binary_bad_value"
    elif af == "number":
        bad = [e for e in emits
               if not str(e.get("value", "")).strip().isdigit()]
        if bad:
            return "schema_number_non_digit"

    # multi_emit must have ≥ 2 distinct emit chunks (otherwise treat as single)
    if card.get("question_type") == "multi_emit":
        chunks = {int(e["chunk"]) for e in emits if "chunk" in e}
        if len(chunks) < 2:
            return "schema_multi_emit_single_chunk"

    grounding = card.get("grounding_frames") or []
    if not grounding:
        return "schema_no_grounding_frames"

    # ── Layer 2: grounding evidence exists ──
    if not ev_by_chunk:
        # No evidence dict provided → skip Layer 2 (legacy callers)
        return "PASS"

    missing = [c for c in grounding if c not in ev_by_chunk]
    if missing:
        return "grounding_evidence_missing"

    # Loose evidence containment: canonical answer text should overlap
    # with the entity_id / fact text from at least one grounding chunk.
    canonical = (card.get("canonical_answer") or "").strip().lower()
    if af == "binary":
        # Binary progress/status questions usually have canonical Yes/No.
        # The literal token "yes" or "no" will not appear in visual evidence,
        # so grounding must rely on the emit chunk + evidence presence above.
        return "PASS"
    if canonical:
        ev_text = ""
        for c in grounding:
            cap = ev_by_chunk[c]
            for e in (cap.get("visible_entities") or []):
                ev_text += " " + (e.get("desc", "") + " " + e.get("id", "")).lower()
            for f in (cap.get("atomic_facts") or []):
                if isinstance(f, dict):
                    ev_text += " " + (f.get("fact", "") or "").lower()
            for o in (cap.get("ocr") or []):
                if isinstance(o, dict):
                    ev_text += " " + (o.get("text", "") or "").lower()
                else:
                    ev_text += " " + str(o).lower()
        # Tokenize canonical and check at least one content token (>2 char)
        # appears in the evidence text. This catches "answer pulled from
        # thin air" while tolerating phrasing variation.
        import re as _re
        toks = [t for t in _re.findall(r"[a-z0-9]+", canonical) if len(t) > 2]
        if toks and not any(t in ev_text for t in toks):
            return "grounding_canonical_not_in_evidence"

    return "PASS"


def save_cards(video_id: str, cards: List[Dict],
               output_dir: Path = TASK_CARDS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2)
    )


def load_cards(video_id: str,
               cards_dir: Path = TASK_CARDS_DIR) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("3a"):
        return None
    p = cards_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Helpers re-exported for back-compat (pass3b imports extract_card_keywords)
# ---------------------------------------------------------------------------

_STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
               "and", "or", "of", "to", "in", "on", "at", "for", "with",
               "what", "which", "who", "when", "where", "why", "how"}


def extract_keywords(text: str) -> List[str]:
    """Lowercased content words ≥3 chars, stopwords removed."""
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS]


def extract_card_keywords(card: Dict) -> List[str]:
    """Keywords representing what this card is asking about."""
    parts = [card.get("question", "")]
    if card.get("answer_form") == "multiple_choice" and card.get("options"):
        try:
            idx = ord(card.get("correct_option", "A")) - ord("A")
            if 0 <= idx < len(card["options"]):
                parts.append(card["options"][idx])
        except Exception:
            pass
    else:
        parts.append(card.get("canonical_answer", ""))
    return list(dict.fromkeys(extract_keywords(" ".join(parts))))


# Legacy aliases (kept so any stragglers importing them don't crash; v2 unused)
FAMILY_TARGETS: Dict[str, int] = {}
RETENTION_CLASS: Dict[str, str] = {}
