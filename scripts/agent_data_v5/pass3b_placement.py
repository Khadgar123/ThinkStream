"""Pass 3-B — Placement + Trajectory planning (v2 model-agnostic).

Replaces classify_availability / _check_visibility_one / sequence_type
machinery with profile-driven placement (backward / forward / realtime)
that depends ONLY on (ask, support_chunks). See v2/design.py for the
full rationale.

Pipeline contract preserved:
  compute_all_placements(cards, rollout, evidence, client, video_id) -> List[Dict]
  plan_trajectories(placements, cards_map, num_chunks, evidence, seed) -> List[Dict]
  save_placements / load_placements
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from .config import PLACEMENTS_DIR
from .pass3a_cards import dict_to_card
from .stable_hash import stable_seed
from .v2.design import (
    Placement,
    adaptive_q_count,
    assign_recall_noise,
    place_card,
    placement_timing_verdict,
    select_trajectory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placement ↔ dict serialization
# ---------------------------------------------------------------------------


def _placement_to_dict(p: Placement) -> Dict:
    return {
        "card_id": p.card_id,
        "ask_chunk": int(p.ask_chunk),
        "mechanism": p.mechanism,
        "chunk_actions": {str(k): list(v) for k, v in p.chunk_actions.items()},
        "recall_at": {str(k): v for k, v in p.recall_at.items()},
    }


def _dict_to_placement(d: Dict) -> Placement:
    return Placement(
        card_id=d["card_id"],
        ask_chunk=int(d["ask_chunk"]),
        mechanism=d["mechanism"],
        chunk_actions={int(k): tuple(v) for k, v in d.get("chunk_actions", {}).items()},
        recall_at={int(k): v for k, v in d.get("recall_at", {}).items()},
    )


# ---------------------------------------------------------------------------
# Public API (matches old pass3b interface)
# ---------------------------------------------------------------------------


async def compute_all_placements(
    cards: List[Dict],
    rollout: Dict,
    evidence: List[Dict],
    client=None,                     # kept for API compat (no LLM call now)
    video_id: str = "",
    seed: int = 42,
) -> List[Dict]:
    """Generate ALL candidate placements (multiple tiers per card).

    No LLM calls. Profile-driven, model-agnostic — depends only on each
    card's (gold_emits, grounding_frames) and the video length.
    """
    rng = random.Random(stable_seed(seed, video_id, modulo=1_000_000))
    num_chunks = int(rollout.get("num_chunks", 0))
    cards_obj = [dict_to_card(c) for c in cards]

    all_placements: List[Placement] = []
    rejected: Dict[str, int] = {}
    for card in cards_obj:
        for p in place_card(card, num_chunks, rng):
            ok, reason = placement_timing_verdict(card, p)
            if ok:
                all_placements.append(p)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
    if rejected:
        logger.info(
            f"[{video_id}] 3b placement timing rejected: "
            f"{dict(sorted(rejected.items(), key=lambda x: -x[1]))}"
        )
    return [_placement_to_dict(p) for p in all_placements]


def plan_trajectories(
    placements: List[Dict],
    cards_map: Dict[str, Dict] = None,
    num_chunks: int = 60,
    evidence: List[Dict] = None,     # unused in v2; kept for API compat
    seed: int = 42,
    **_,
) -> List[Dict]:
    """Select MAX_QUESTIONS_PER_TRAJECTORY placements for ONE trajectory.

    v2: 1 trajectory/video × adaptive 6-14 questions (q-interval ~14s).
    Diversity-weighted greedy selection across family / mechanism /
    answer_form / ask_chunk spread.
    """
    if not placements:
        return []
    cards_map = cards_map or {}
    rng = random.Random(seed)

    placement_objs = [_dict_to_placement(p) for p in placements]
    placements_by_card: Dict[str, List[Placement]] = {}
    for p in placement_objs:
        placements_by_card.setdefault(p.card_id, []).append(p)

    cards_obj = [dict_to_card(c) for c in cards_map.values()]
    cards_by_id = {c.card_id: c for c in cards_obj}
    filtered_by_card: Dict[str, List[Placement]] = {}
    rejected: Dict[str, int] = {}
    for cid, plcs in placements_by_card.items():
        card = cards_by_id.get(cid)
        if not card:
            continue
        for p in plcs:
            ok, reason = placement_timing_verdict(card, p)
            if ok:
                filtered_by_card.setdefault(cid, []).append(p)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
    if rejected:
        logger.info(
            "3b selected-placement input timing rejected: "
            f"{dict(sorted(rejected.items(), key=lambda x: -x[1]))}"
        )
    target_q = adaptive_q_count(num_chunks)
    selected = select_trajectory(
        cards_obj, filtered_by_card, num_chunks, rng, max_q=target_q,
    )
    assign_recall_noise(selected, rng)

    if not selected:
        return []
    # 1 trajectory per video — return single-element list to match
    # pipeline expectation that plan_trajectories yields a list.
    return [{
        "trajectory_id": f"traj_0",
        "placements": [_placement_to_dict(p) for p in selected],
        "n_questions": len(selected),
        "ask_chunks": sorted([p.ask_chunk for p in selected]),
    }]


def save_placements(video_id: str, data: Dict,
                    output_dir: Path = PLACEMENTS_DIR) -> None:
    """Save {placements: [...], trajectories: [...]} for a video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )


def load_placements(video_id: str,
                    placements_dir: Path = PLACEMENTS_DIR) -> Optional[Dict]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("3b"):
        return None
    p = placements_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Back-compat helpers (used by pass3e_verify)
# ---------------------------------------------------------------------------

import re as _re


def _keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in text. Used by pass3e_verify."""
    if not keywords:
        return 0.0
    text_words = set(_re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
    found = sum(1 for kw in keywords if kw in text_words)
    return found / len(keywords)
