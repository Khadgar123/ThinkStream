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
    parse_card_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Card ↔ dict serialization
# ---------------------------------------------------------------------------


def _card_to_dict(card: Card) -> Dict:
    canonical = card.gold_emits[-1].value if card.gold_emits else ""
    return {
        "card_id": card.card_id,
        "family": card.family,
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
    """No-op in v2.

    Accepts both:
      - verify_cards(cards, client, video_id)                  (legacy)
      - verify_cards(video_id, cards, evidence, client)        (pipeline.py:679)
    Returns the cards list unchanged.
    """
    cards = kwargs.get("cards")
    if cards is None:
        # Find the first list arg (cards is always a list of dicts)
        for a in args:
            if isinstance(a, list):
                cards = a
                break
    return cards or []


def save_cards(video_id: str, cards: List[Dict],
               output_dir: Path = TASK_CARDS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2)
    )


def load_cards(video_id: str,
               cards_dir: Path = TASK_CARDS_DIR) -> Optional[List[Dict]]:
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
