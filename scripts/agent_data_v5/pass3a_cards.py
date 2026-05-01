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

from .config import TASK_CARDS_DIR
from .v2.cards import generate_cards as _heuristic_generate
from .v2.design import Card, GoldEmit

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
    evidence: List[Dict],
    client=None,                     # kept for API compat; LLM hook below
    video_id: str = "",
    seed: int = 42,
) -> List[Dict]:
    """Generate v2 cards from evidence.

    Currently uses heuristic from v2/cards.py. To switch to 397B:
    1. Build prompts that ask the teacher for {family, question, gold_emits,
       grounding_frames, options, correct_option} JSON.
    2. Replace the body of `_generate_via_llm` with the real call.
    3. Card schema stays identical — no downstream changes needed.
    """
    cards = _generate_via_heuristic(evidence, video_id, seed)
    return [_card_to_dict(c) for c in cards]


def _generate_via_heuristic(evidence: List[Dict], video_id: str, seed: int) -> List[Card]:
    return _heuristic_generate(evidence, video_id, seed=seed)


async def _generate_via_llm(
    evidence: List[Dict], client, video_id: str, seed: int,
) -> List[Card]:
    """Stub for 397B-driven card generation. Not yet wired."""
    raise NotImplementedError(
        "LLM-based card generation not yet implemented; "
        "use heuristic via generate_cards() default path."
    )


async def verify_cards(
    cards: List[Dict],
    client=None,
    video_id: str = "",
) -> List[Dict]:
    """No-op in v2: schema is teacher-validated at generation; pass3e still
    runs sanity tags on rendered samples. Kept for pipeline.py API compat."""
    return cards


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
