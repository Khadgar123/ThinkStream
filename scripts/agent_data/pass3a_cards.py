"""Pass 3-A — Card generation (placement model-agnostic schema).

Cards are now defined by `gold_emits` + `grounding_frames` + `question_type`
(see placement/design.py). The card distribution is OVOBench-aligned (76% MC + binary
+ number + short_exact + descriptive). The heuristic generator in v2/cards.py
derives cards directly from evidence; swap in 397B prompts later by
replacing `_generate_via_llm` below.

Pipeline contract preserved:
  generate_cards(evidence, client, video_id) -> List[Dict]    (async)
  verify_cards(cards, client, video_id)      -> List[Dict]    (async, schema+grounding)
  save_cards / load_cards
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import asyncio

from .config import TASK_CARDS_DIR, PASS_CONFIG
from .stable_hash import stable_mod
from .placement.cards import generate_cards as _heuristic_generate
from .placement.design import (
    Card,
    F7_ADOPT_RATE,
    GoldEmit,
    EVIDENCE_TYPES,
    QUESTION_WAYS,
    SUPPORT_POLICIES,
    infer_card_answer_mode_constraints,
    infer_card_policy_fields,
    infer_card_semantic_fields,
)
from .placement.design import (
    QUESTION_STYLE_BENCHMARK_CORE,
    QUESTION_STYLE_BENCHMARK_VARIANT,
    QUESTION_STYLE_OURS_UNIQUE,
    QUESTION_STYLES,
)
from .placement.llm_prompts import (
    FAMILY_RULES,
    QUESTION_TYPE_BY_FAMILY,
    card_generation_prompt,
    family_taxonomy,
    parse_card_response,
)
from .pass3_slot_planner import (
    apply_slot_metadata_to_card,
    balanced_family_targets,
    build_pass3_slot_plan,
    card_matches_planned_slot,
    default_slot_metadata_for_family,
    filter_pass3_slot_plan,
    group_slots_by_family,
    slot_plan_summary,
)

logger = logging.getLogger(__name__)
ALLOW_HEURISTIC_FALLBACK = (
    os.environ.get("THINKSTREAM_PASS3A_ALLOW_HEURISTIC_FALLBACK", "")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
ENABLE_SLOT_PLANNING = (
    os.environ.get("THINKSTREAM_PASS3A_ENABLE_SLOT_PLANNING", "1")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)
SLOT_PLAN_DIR = os.environ.get("THINKSTREAM_PASS3A_SLOT_PLAN_DIR", "").strip()
SLOT_TARGET_MODE = os.environ.get("THINKSTREAM_PASS3A_SLOT_TARGET_MODE", "balanced").strip().lower()
STRICT_POST_VERIFY = (
    os.environ.get("THINKSTREAM_PASS3A_STRICT_POST_VERIFY", "0")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
BASIC_POST_VERIFY = (
    os.environ.get("THINKSTREAM_PASS3A_BASIC_POST_VERIFY", "1")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)
MAX_SLOT_RETRIES = int(os.environ.get("THINKSTREAM_PASS3A_MAX_SLOT_RETRIES", "1"))

MC_OPTION_LETTERS = "ABCDE"
MC_OPTION_COUNTS = {4, 5}
_OPTION_LABEL_RE = re.compile(r"^\s*(?:\([A-E]\)|[A-E][\).:])\s*")
_TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INTERNAL_TIME_REF_RE = re.compile(
    r"(?i)(?:\bchunks?\s*c?\d+\b|\bc\d+\b|\bframes?\s*\d+\b|"
    r"\bt\s*=\s*\d+|\baround\s+chunk\b|\bat\s+chunk\b)"
)
_QUESTION_RENDERING_LEAK_RE = re.compile(
    r"(?is)(?:"
    r"\boptions?\s*:"
    r"|(?:^|\n)\s*(?:\([A-E]\)|[A-E][\).:])\s+\S+"
    r"|\b(?:among|from)\s+(?:these|the\s+following)\s+options\b"
    r"|\banswer\s+(?:with|using|in|only)\b"
    r"|\breturn\s+(?:only\s+)?(?:the\s+)?(?:letter|answer|yes|no)\b"
    r"|\b(?:choose|select)\s+(?:one|from|the\s+correct|the\s+best)\b"
    r")"
)
_FUTURE_WAIT_SURFACE_RE = re.compile(
    r"\b(wait|until|once|as soon as|output\b|emit\b|"
    r"when\b[^?]{0,120}\b(output|emit|say|respond))\b",
    re.I,
)
_FUTURE_PRED_SURFACE_RE = re.compile(
    r"\b(about to|next|likely to|will\b|going to|expected to|what will|"
    r"what happens next|must happen|expected result)\b",
    re.I,
)
_PAST_SURFACE_RE = re.compile(
    r"\b(earlier|previously|before|did\b|was\b|were\b|had\b|initially|"
    r"at the beginning)\b",
    re.I,
)
F5_MAX_COUNT_VALUE = int(os.environ.get("THINKSTREAM_F5_MAX_COUNT_VALUE", "10"))
_CRR_HISTORICAL_STATUS_RE = re.compile(r"\b(has|have)\b.+\b(happened|occurred)\b.+\b(yet|by now)\b", re.I)


PASS3A_TARGETS_BY_FAMILY = {
    # HLD is useful as a small unanswerable/abstention slice, but
    # StreamingBench has no dedicated HLD-style task and OVO's large HLD
    # card-level share should not dominate our mixed training trajectories.
    "HLD1": 1,
    # OVO OCR is ~9%; one C1 per video landed at ~4.8% in batch3.
    # Ask the teacher/fallback for two OCR cards when the video has enough text.
    "C1": 2,
    # OVO-heavy MC skills. The selector still keeps at most one placement per
    # family in a trajectory, but extra candidates give pass3B a better chance
    # to choose an event-anchored historical/detail card that can become recall.
    "OJR1": 2,
    "STU1": 2,
    "ACR1": 2,
    # Active-responding non-MCQ families need extra candidates because their
    # placements are often longer and lose to compact MC cards. The selector
    # still keeps at most one placement per family in each trajectory.
    "F5": 2,
    "F7": 2,
    "CRR1": 2,
    # Our delayed-clue skill should survive selection as either wait/current
    # or historical recall, not only as an OVO-like ordering card.
    "CR5": 2,
    # Ask for one benchmark-core and one benchmark-compatible wording variant
    # when evidence supports both. Selection controls the final
    # benchmark_core / benchmark_variant / ours_unique mix.
    "N1": 2,
    "P1": 2,
    "CR1": 2,
    "CR2": 2,
    "CR3": 2,
    "CR4": 2,
    "CR7": 2,
    "R1": 2,
    "E2": 2,
    "F6": 2,
    "M1": 2,
}


def _strip_option_label(text: str) -> str:
    return _OPTION_LABEL_RE.sub("", str(text or "")).strip()


def _norm_text(text: str) -> str:
    return " ".join(_TEXT_TOKEN_RE.findall(str(text or "").lower()))


def _evidence_text(cap: Dict) -> str:
    parts: List[str] = []
    for e in cap.get("visible_entities") or []:
        if isinstance(e, dict):
            parts.extend([str(e.get("id", "")), str(e.get("desc", "")),
                          str(e.get("action", ""))])
        else:
            parts.append(str(e))
    for f in cap.get("atomic_facts") or []:
        parts.append(str(f.get("fact", "")) if isinstance(f, dict) else str(f))
    for o in cap.get("ocr") or []:
        parts.append(str(o.get("text", "")) if isinstance(o, dict) else str(o))
    spatial = cap.get("spatial")
    if spatial:
        parts.append(str(spatial))
    parts.append(str(cap.get("think", "")))
    return " ".join(parts)


def _mc_correct_text(card: Dict) -> str:
    opts = card.get("options") or []
    co = str(card.get("correct_option") or "").strip().upper()
    if isinstance(opts, list) and len(opts) in MC_OPTION_COUNTS:
        if co in MC_OPTION_LETTERS[:len(opts)]:
            return _strip_option_label(str(opts[ord(co) - ord("A")])).strip()
    return ""


def _relabel_mc_options(options: List[str]) -> List[str]:
    return [
        f"{chr(65 + i)}) {_strip_option_label(str(opt))}"
        for i, opt in enumerate(options)
    ]


def _normalize_hld1_unable_slot(card: Dict) -> None:
    if card.get("family") != "HLD1" or card.get("answer_form") != "multiple_choice":
        return
    opts = list(card.get("options") or [])
    if len(opts) not in MC_OPTION_COUNTS:
        return
    unable_idx = None
    for i, opt in enumerate(opts):
        if "unable to answer" in _strip_option_label(str(opt)).lower():
            unable_idx = i
            break
    if unable_idx is None:
        return

    key = str(card.get("card_id") or card.get("question") or "")
    target_idx = stable_mod(key, "HLD1_UNABLE_POS", modulo=len(opts))
    if unable_idx != target_idx:
        unable_opt = opts.pop(unable_idx)
        opts.insert(target_idx, unable_opt)
    card["options"] = _relabel_mc_options(opts)
    card["correct_option"] = chr(65 + target_idx)
    card["canonical_answer"] = "Unable to answer"
    emits = card.get("gold_emits") or []
    if emits:
        emits[0]["value"] = card["correct_option"]
        card["gold_emits"] = emits


def _normalize_non_hld_mc_slot(card: Dict) -> None:
    if card.get("family") == "HLD1" or card.get("answer_form") != "multiple_choice":
        return
    opts = list(card.get("options") or [])
    co = str(card.get("correct_option") or "").strip().upper()
    if len(opts) not in MC_OPTION_COUNTS or co not in MC_OPTION_LETTERS[:len(opts)]:
        return

    texts = [_strip_option_label(str(o)) for o in opts]
    if any(not t for t in texts) or len(set(texts)) != len(opts):
        return
    correct_text = texts[ord(co) - ord("A")]
    distractors = [t for i, t in enumerate(texts) if i != ord(co) - ord("A")]
    key = str(card.get("card_id") or card.get("question") or "")
    target_idx = stable_mod(
        key, str(card.get("family") or ""), "MC_CORRECT_POS", modulo=len(opts)
    )
    distractors = sorted(
        distractors,
        key=lambda t: stable_mod(key, str(card.get("family") or ""), "MC_DISTRACTOR", t, modulo=2**31),
    )

    ordered: List[str] = []
    it = iter(distractors)
    for i in range(len(opts)):
        ordered.append(correct_text if i == target_idx else next(it))
    card["options"] = _relabel_mc_options(ordered)
    card["correct_option"] = chr(65 + target_idx)
    card["canonical_answer"] = correct_text
    emits = card.get("gold_emits") or []
    if emits and card.get("question_type") == "single_emit":
        emits[0]["value"] = card["correct_option"]
        card["gold_emits"] = emits


def _normalize_card_in_place(card: Dict) -> None:
    """Repair deterministic schema fields before validation/rendering.

    - MC canonical_answer must be the correct option text, not a paraphrase.
    - Single-emit answer chunks cannot precede their full grounding support.
    """
    family = str(card.get("family") or "")
    for key, value in default_slot_metadata_for_family(family).items():
        card.setdefault(key, value)
    _normalize_hld1_unable_slot(card)
    _normalize_non_hld_mc_slot(card)

    if card.get("answer_form") == "multiple_choice":
        correct_text = _mc_correct_text(card)
        co = str(card.get("correct_option") or "").strip().upper()
        if correct_text:
            card["canonical_answer"] = correct_text
            if (
                card.get("question_type") == "single_emit"
                and co in MC_OPTION_LETTERS[:len(card.get("options") or [])]
            ):
                emits = card.get("gold_emits") or []
                if emits:
                    emits[0]["value"] = co
                    card["gold_emits"] = emits

    if card.get("question_type") == "single_emit":
        emits = card.get("gold_emits") or []
        grounding = card.get("grounding_frames") or card.get("support_chunks") or []
        if emits and grounding:
            max_ground = max(int(g) for g in grounding)
            emits[0]["chunk"] = max_ground
            card["gold_emits"] = emits
            card.setdefault("grounding_frames", list(grounding))
            card["support_chunks"] = list(card.get("grounding_frames") or grounding)

    policy = infer_card_policy_fields(card)
    if not card.get("target_ovo_task"):
        card["target_ovo_task"] = policy["target_ovo_task"]
    raw_ovo = str(card.get("ovo_task") or "").strip()
    if not raw_ovo or "/" in raw_ovo:
        card["ovo_task"] = policy["target_ovo_task"]
    if not card.get("temporal_role"):
        card["temporal_role"] = policy["temporal_role"]
    if str(card.get("support_policy") or "").strip().lower() not in SUPPORT_POLICIES:
        card["support_policy"] = policy["support_policy"]
    allowed = card.get("allowed_support_policies")
    if isinstance(allowed, str):
        allowed = [allowed]
    allowed_norm = [
        str(p or "").strip().lower()
        for p in (allowed or [])
        if str(p or "").strip().lower() in SUPPORT_POLICIES
    ]
    if not allowed_norm:
        card["allowed_support_policies"] = list(policy["allowed_support_policies"])
    else:
        card["allowed_support_policies"] = allowed_norm
    if "recall_eligible" not in card or card.get("recall_eligible") is None:
        card["recall_eligible"] = bool(policy["recall_eligible"])
    if "state_memory_required" not in card or card.get("state_memory_required") is None:
        card["state_memory_required"] = bool(policy["state_memory_required"])
    semantic = infer_card_semantic_fields(card)
    if str(card.get("question_way") or "").strip() not in QUESTION_WAYS:
        card["question_way"] = semantic["question_way"]
    if str(card.get("evidence_type") or "").strip() not in EVIDENCE_TYPES:
        card["evidence_type"] = semantic["evidence_type"]

    style = str(card.get("question_style") or "").strip()
    if style not in QUESTION_STYLES:
        taxonomy = family_taxonomy(str(card.get("family") or ""))
        if taxonomy.get("ours_unique"):
            style = QUESTION_STYLE_OURS_UNIQUE
        else:
            # Deterministic fallback for old/fallback cards. The prompt now
            # requests explicit styles; this keeps stale or heuristic cards
            # selectable under the same quota logic.
            key = str(card.get("card_id") or card.get("question") or "")
            style = (
                QUESTION_STYLE_BENCHMARK_VARIANT
                if stable_mod(key, "benchmark_variant_style", modulo=100) < 15
                else QUESTION_STYLE_BENCHMARK_CORE
            )
    card["question_style"] = style
    mode_policy = infer_card_answer_mode_constraints(card)
    legal = mode_policy.get("legal_answer_modes") or ()
    forbidden = mode_policy.get("forbidden_answer_modes") or ()
    if not card.get("legal_answer_modes"):
        card["legal_answer_modes"] = list(legal)
    if not card.get("required_answer_mode"):
        card["required_answer_mode"] = str(mode_policy.get("required_answer_mode") or "")
    if not card.get("forbidden_answer_modes"):
        card["forbidden_answer_modes"] = list(forbidden)
    if not card.get("answer_mode_reason"):
        card["answer_mode_reason"] = str(mode_policy.get("answer_mode_reason") or "")


# ---------------------------------------------------------------------------
# Card ↔ dict serialization
# ---------------------------------------------------------------------------


def _card_to_dict(card: Card) -> Dict:
    canonical = card.gold_emits[-1].value if card.gold_emits else ""
    if card.answer_form == "multiple_choice" and card.options and card.correct_option:
        idx = ord(str(card.correct_option).strip().upper()) - ord("A")
        if 0 <= idx < len(card.options):
            canonical = _strip_option_label(str(card.options[idx]))
    taxonomy = family_taxonomy(card.family)
    out = {
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
        "target_ovo_task": getattr(card, "target_ovo_task", ""),
        "temporal_role": getattr(card, "temporal_role", ""),
        "support_policy": getattr(card, "support_policy", ""),
        "allowed_support_policies": list(getattr(card, "allowed_support_policies", ()) or []),
        "recall_eligible": bool(getattr(card, "recall_eligible", False)),
        "state_memory_required": bool(getattr(card, "state_memory_required", False)),
        "question_style": getattr(card, "question_style", "") or (
            QUESTION_STYLE_OURS_UNIQUE
            if taxonomy.get("ours_unique")
            else QUESTION_STYLE_BENCHMARK_CORE
        ),
        "question_way": getattr(card, "question_way", ""),
        "evidence_type": getattr(card, "evidence_type", ""),
        "legacy_family_id": getattr(card, "legacy_family_id", ""),
        "task_family": getattr(card, "task_family", ""),
        "task_subtype": getattr(card, "task_subtype", ""),
        "timing_type": getattr(card, "timing_type", ""),
        "readable_task_name": getattr(card, "readable_task_name", ""),
        "slot_group": getattr(card, "slot_group", ""),
        "slot_subtype": getattr(card, "slot_subtype", ""),
        "temporal_bucket": getattr(card, "temporal_bucket", ""),
        "benchmark_source": getattr(card, "benchmark_source", ""),
        "benchmark_task": getattr(card, "benchmark_task", ""),
        "answer_behavior": getattr(card, "answer_behavior", ""),
        "question_goal": getattr(card, "question_goal", ""),
        "placement_hint": getattr(card, "placement_hint", ""),
        "legal_answer_modes": list(getattr(card, "legal_answer_modes", ()) or []),
        "required_answer_mode": getattr(card, "required_answer_mode", ""),
        "forbidden_answer_modes": list(getattr(card, "forbidden_answer_modes", ()) or []),
        "answer_mode_reason": getattr(card, "answer_mode_reason", ""),
        "options": list(card.options) if card.options else None,
        "correct_option": card.correct_option,
        "recall_query": card.recall_query,
    }
    _normalize_card_in_place(out)
    return out


def dict_to_card(d: Dict) -> Card:
    """Rebuild a Card from its serialized dict (used by pass3b/c)."""
    d = dict(d)
    _normalize_card_in_place(d)
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
        target_ovo_task=d.get("target_ovo_task", ""),
        temporal_role=d.get("temporal_role", ""),
        support_policy=d.get("support_policy", ""),
        allowed_support_policies=tuple(d.get("allowed_support_policies") or ()),
        recall_eligible=bool(d.get("recall_eligible", False)),
        state_memory_required=bool(d.get("state_memory_required", False)),
        question_style=d.get("question_style", ""),
        question_way=d.get("question_way", ""),
        evidence_type=d.get("evidence_type", ""),
        legacy_family_id=d.get("legacy_family_id", ""),
        task_family=d.get("task_family", ""),
        task_subtype=d.get("task_subtype", ""),
        timing_type=d.get("timing_type", ""),
        readable_task_name=d.get("readable_task_name", ""),
        slot_group=d.get("slot_group", ""),
        slot_subtype=d.get("slot_subtype", ""),
        temporal_bucket=d.get("temporal_bucket", ""),
        benchmark_source=d.get("benchmark_source", ""),
        benchmark_task=d.get("benchmark_task", ""),
        answer_behavior=d.get("answer_behavior", ""),
        question_goal=d.get("question_goal", ""),
        placement_hint=d.get("placement_hint", ""),
        legal_answer_modes=tuple(d.get("legal_answer_modes") or ()),
        required_answer_mode=d.get("required_answer_mode", ""),
        forbidden_answer_modes=tuple(d.get("forbidden_answer_modes") or ()),
        answer_mode_reason=d.get("answer_mode_reason", ""),
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
    source_row_targets = kwargs.get("source_row_targets")
    if client is None:
        cards = _generate_via_heuristic(evidence, video_id, seed)
        return [_card_to_dict(c) for c in cards]
    return await _generate_via_llm(
        evidence,
        client,
        video_id,
        seed,
        source_row_targets=source_row_targets,
    )


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
    evidence: List[Dict],
    client,
    video_id: str,
    seed: int,
    *,
    source_row_targets: Optional[Mapping[str, int]] = None,
) -> List[Dict]:
    """397B-driven card generation. Per-family parallel via asyncio.gather.

    Each family fires ONE request whose target count may vary by family. The
    family list is the current FAMILY_RULES order, so adding benchmark-aligned
    families changes coverage without touching pipeline orchestration. The
    trajectory selector then picks the best 6-14 per video.

    Production card text and MC options must come from the LLM. Heuristic
    cards are allowed only for offline simulation or when explicitly enabled
    through THINKSTREAM_PASS3A_ALLOW_HEURISTIC_FALLBACK=1.
    """
    cfg = PASS_CONFIG.get("pass3a", {})
    max_tokens = int(cfg.get("max_tokens", 16384))
    temperature = float(cfg.get("temperature", 0.7))
    enable_thinking = cfg.get("thinking", False)

    families = [
        family for family in FAMILY_RULES.keys()
        if not (
            family == "F7"
            and stable_mod(video_id, "F7_ADOPT", modulo=100) >= int(F7_ADOPT_RATE * 100)
        )
    ]
    slot_by_family: Dict[str, List[Dict]] = {}
    slot_audit: Dict[str, int] = {}
    if ENABLE_SLOT_PLANNING:
        slots = _load_external_slot_plan(video_id)
        if slots:
            slots = filter_pass3_slot_plan(evidence, slots)
        if not slots:
            family_targets = balanced_family_targets(video_id, SLOT_TARGET_MODE)
            family_targets = {
                family: int(family_targets.get(family, 0))
                for family in families
                if int(family_targets.get(family, 0)) > 0
            }
            slot_audit_counter = Counter()
            slots = build_pass3_slot_plan(
                evidence,
                video_id,
                family_targets,
                audit=slot_audit_counter,
                source_row_targets=source_row_targets,
                seed=seed,
            )
            slot_audit = dict(slot_audit_counter)
        slot_by_family = group_slots_by_family(slots)
        families = [family for family in families if slot_by_family.get(family)]
        logger.info("[%s] 3a slot plan: %s", video_id, slot_plan_summary(slots))
        if slot_audit:
            logger.info("[%s] 3a slot keep/drop: %s", video_id, slot_audit)
    fallback_cards = (
        _generate_via_heuristic(evidence, video_id, seed)
        if ALLOW_HEURISTIC_FALLBACK else []
    )
    fallback_by_family: Dict[str, List[Card]] = {}
    for c in fallback_cards:
        fallback_by_family.setdefault(c.family, []).append(c)
    evidence_by_chunk: Dict[int, Dict] = {}
    for cap in evidence:
        ci = cap.get("chunk_idx")
        if isinstance(ci, int):
            evidence_by_chunk[ci] = cap

    def _fallback_dicts(family: str) -> List[Dict]:
        return [_card_to_dict(c) for c in fallback_by_family.get(family, [])]

    async def _one_family(family: str) -> List[Dict]:
        planned_slots = slot_by_family.get(family, [])
        rejected_total: Counter = Counter()

        async def _request_slots(
            request_slots: List[Dict],
            *,
            request_suffix: str,
            card_offset: int,
        ) -> tuple[List[Dict], int]:
            target_n = (
                len(request_slots)
                if request_slots
                else int(PASS3A_TARGETS_BY_FAMILY.get(family, 1))
            )
            prompt = card_generation_prompt(
                family,
                evidence,
                target_n=target_n,
                planned_slots=request_slots,
            )
            try:
                raw = await client._call_one(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    request_id=f"{video_id}_3a_{family}{request_suffix}",
                    enable_thinking=enable_thinking,
                )
            except Exception as exc:
                if exc.__class__.__name__ == "TruncatedCompletionError":
                    logger.warning(f"[{video_id}] 3a {family}{request_suffix}: LLM output truncated: {exc}")
                    raw = None
                else:
                    logger.warning(f"[{video_id}] 3a {family}{request_suffix}: LLM call failed: {exc}")
                    raw = None
            cards = parse_card_response(raw or "", family) if raw else []
            if not cards:
                rejected_total["llm_no_cards"] += 1
                return [], 0

            out = []
            for i, c in enumerate(cards):
                c["card_id"] = f"{video_id}_{family}_{seed:04d}_{card_offset + i}"
                c.update(family_taxonomy(family))
                apply_slot_metadata_to_card(c, request_slots)
                # Default placeholder; pass3c may override on demand
                c.setdefault("recall_query", None)
                _normalize_card_in_place(c)
                out.append(c)
            verified = []
            rejected: Counter = Counter()
            for c in out:
                verdict = _verify_card_for_mode(c, evidence_by_chunk)
                if verdict == "PASS" and request_slots and not card_matches_planned_slot(c, request_slots):
                    verdict = "slot_plan_mismatch"
                if verdict == "PASS":
                    verified.append(c)
                else:
                    rejected[verdict] += 1
            verified, dedupe_rejected = _dedupe_cards_by_question_signature(verified)
            rejected.update(dedupe_rejected)
            rejected_total.update(rejected)
            return verified, len(out)

        verified, generated_count = await _request_slots(
            planned_slots,
            request_suffix="",
            card_offset=0,
        )
        all_verified = list(verified)
        total_generated = generated_count
        if planned_slots and MAX_SLOT_RETRIES > 0:
            seen_slot_ids = {
                str(c.get("slot_id") or "")
                for c in all_verified
                if str(c.get("slot_id") or "")
            }
            missing_slots = [
                slot
                for slot in planned_slots
                if str(slot.get("slot_id") or "") not in seen_slot_ids
            ]
            for retry_idx in range(max(0, MAX_SLOT_RETRIES)):
                if not missing_slots:
                    break
                retry_verified, retry_generated = await _request_slots(
                    missing_slots,
                    request_suffix=f"_retry{retry_idx + 1}",
                    card_offset=total_generated,
                )
                total_generated += retry_generated
                if retry_verified:
                    all_verified.extend(retry_verified)
                    all_verified, dedupe_rejected = _dedupe_cards_by_question_signature(all_verified)
                    rejected_total.update(dedupe_rejected)
                seen_slot_ids = {
                    str(c.get("slot_id") or "")
                    for c in all_verified
                    if str(c.get("slot_id") or "")
                }
                missing_slots = [
                    slot
                    for slot in planned_slots
                    if str(slot.get("slot_id") or "") not in seen_slot_ids
                ]
            if missing_slots:
                rejected_total["slot_retry_unfilled"] += len(missing_slots)

        if all_verified:
            if rejected_total:
                logger.info(
                    f"[{video_id}] 3a {family}: kept {len(all_verified)}/{total_generated}; "
                    f"rejected {dict(rejected_total)}"
                )
            return all_verified
        if rejected_total:
            logger.info(f"[{video_id}] 3a {family}: LLM cards rejected {dict(rejected_total)}")
        return _fallback_dicts(family)

    results = await asyncio.gather(*[_one_family(f) for f in families])
    all_cards: List[Dict] = []
    for fam_cards in results:
        all_cards.extend(fam_cards)
    all_cards, dedupe_rejected = _dedupe_cards_by_question_signature(all_cards)
    if dedupe_rejected:
        logger.info("[%s] 3a global dedupe rejected %s", video_id, dedupe_rejected)
    logger.info(f"[{video_id}] 3a: LLM generated {len(all_cards)} cards "
                f"across {len(families)} families")
    return all_cards


def _load_external_slot_plan(video_id: str) -> List[Dict]:
    if not SLOT_PLAN_DIR:
        return []
    path = Path(SLOT_PLAN_DIR) / f"{video_id}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("[%s] failed to read slot plan %s: %s", video_id, path, exc)
        return []
    if isinstance(data, dict):
        raw = data.get("slots") or []
    else:
        raw = data
    if not isinstance(raw, list):
        return []
    return [slot for slot in raw if isinstance(slot, dict)]


def _question_signature(card: Dict) -> str:
    q = str(card.get("question") or "").lower()
    q = re.sub(r"\b(?:[a-e][\).:])\s*", " ", q)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", q)
        if len(token) > 2 and token not in _STOP_WORDS
    ]
    return " ".join(tokens[:24])


def _dedupe_cards_by_question_signature(cards: List[Dict]) -> tuple[List[Dict], Dict[str, int]]:
    seen: set[str] = set()
    kept: List[Dict] = []
    rejected: Dict[str, int] = {}
    for card in cards:
        sig = _question_signature(card)
        if sig and sig in seen:
            rejected["question_signature_duplicate"] = rejected.get("question_signature_duplicate", 0) + 1
            continue
        if sig:
            seen.add(sig)
        kept.append(card)
    return kept, rejected


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
        _normalize_card_in_place(card)
        verdict = _verify_card_for_mode(card, evidence_by_chunk)
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


def _verify_card_for_mode(card: Dict, ev_by_chunk: Dict[int, Dict]) -> str:
    if STRICT_POST_VERIFY:
        return _verify_card_layers(card, ev_by_chunk, strict_semantic=True)
    if BASIC_POST_VERIFY:
        return _verify_card_layers(card, ev_by_chunk, strict_semantic=False)
    return "PASS"


def _verify_card_layers(
    card: Dict,
    ev_by_chunk: Dict[int, Dict],
    *,
    strict_semantic: bool = True,
) -> str:
    """Two-layer card verification. Returns "PASS" or a fail reason string.

    Layer 1 — schema sanity:
      - has card_id, family, question, answer_form, gold_emits / canonical_answer
      - MC: options is list of length 4-5; correct_option matches an option;
            options match the correct_option index
      - binary: gold_emits values in {Yes, No, yes, no}
      - number: gold_emits values are digit strings
      - multi_emit: ≥ 2 distinct gold_emit chunks
      - grounding_frames non-empty

    Layer 2 — grounding evidence exists:
      - every grounding chunk index has a matching evidence entry

    Strict semantic mode additionally checks:
      - canonical answer text overlaps with evidence text in the grounding
        chunks (loose containment; entity name OR fact phrase appears)
    """
    # ── Layer 1: schema ──
    if not card.get("card_id") or not card.get("family"):
        return "schema_missing_id_family"
    q = (card.get("question") or "").strip()
    if len(q) < 8:
        return "schema_question_too_short"
    if _INTERNAL_TIME_REF_RE.search(q):
        return "schema_question_internal_time_ref"
    if _QUESTION_RENDERING_LEAK_RE.search(q):
        return "schema_question_contains_options_or_answer_format"
    temporal_role = str(card.get("temporal_role") or "")
    temporal_bucket = str(card.get("temporal_bucket") or card.get("timing_type") or "")
    support_policy = str(card.get("support_policy") or "")
    if (
        temporal_role == "future_event_wait"
        or temporal_bucket == "future_delayed"
    ) and not _FUTURE_WAIT_SURFACE_RE.search(q):
        return "schema_future_wait_surface_missing"
    if (
        temporal_role == "future_current_cue"
        or temporal_bucket in {"current_future_prediction", "future_current_cue"}
    ) and not _FUTURE_PRED_SURFACE_RE.search(q):
        return "schema_future_prediction_surface_missing"
    if (
        support_policy == "historical_visual_recall"
        or temporal_bucket.startswith("past_")
    ) and not _PAST_SURFACE_RE.search(q):
        return "schema_past_visual_surface_missing"
    family = str(card.get("family") or "")
    af = card.get("answer_form", "")
    if af not in {"multiple_choice", "binary", "number", "short_exact", "descriptive"}:
        return "schema_bad_answer_form"
    emits = card.get("gold_emits") or []
    if not emits:
        return "schema_no_gold_emits"
    expected_af = str(FAMILY_RULES.get(family, {}).get("answer_form") or "")
    if expected_af and af != expected_af:
        return "schema_family_answer_form_mismatch"
    emit_chunks: List[int] = []
    for e in emits:
        if not isinstance(e, dict):
            return "schema_bad_gold_emit"
        if e.get("chunk") is None:
            return "schema_emit_missing_chunk"
        try:
            chunk = int(e.get("chunk"))
        except (TypeError, ValueError):
            return "schema_emit_bad_chunk"
        if not str(e.get("value", "")).strip():
            return "schema_emit_empty_value"
        emit_chunks.append(chunk)

    if family in {"HLD1", "C1", "ACR1", "STU1", "OJR1"} and af != "multiple_choice":
        return f"schema_{family.lower()}_must_be_mc"
    if family == "CRR1" and card.get("question_type") != "multi_emit":
        return "schema_status_not_multi_emit"

    correct_option_text = ""
    if af == "multiple_choice":
        opts = card.get("options") or []
        if not isinstance(opts, list) or len(opts) not in MC_OPTION_COUNTS:
            return "schema_mc_options_bad_count"
        if any(not _strip_option_label(str(opt)).strip() for opt in opts):
            return "schema_mc_empty_option"
        co = card.get("correct_option", "")
        if co not in MC_OPTION_LETTERS[:len(opts)]:
            return "schema_mc_bad_correct_letter"
        # The option at the correct letter's index should match canonical_answer
        # (or contain it). Skip exact match — pass3a may format options as
        # "A) text" or just "text"; we accept any non-placeholder.
        idx = ord(co) - ord("A")
        correct_option_text = _strip_option_label(str(opts[idx])).strip()
        if not correct_option_text:
            return "schema_mc_empty_correct_option"
        if any("distractor placeholder" in str(o).lower() for o in opts):
            return "schema_mc_placeholder_distractor"
        canonical_raw = str(card.get("canonical_answer") or "").strip()
        if canonical_raw != correct_option_text:
            return "schema_mc_canonical_not_correct_option"
        if family == "HLD1":
            if "unable to answer" not in correct_option_text.lower():
                return "schema_hld1_missing_unable_option"
            if (
                canonical_raw
                and canonical_raw.upper() not in set(MC_OPTION_LETTERS)
                and "unable to answer" not in canonical_raw.lower()
            ):
                return "schema_hld1_bad_canonical"
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
        if family == "F5":
            ordered_counts: List[int] = []
            for e in sorted(emits, key=lambda x: int(x.get("chunk", -1))):
                try:
                    ordered_counts.append(int(str(e.get("value", "")).strip()))
                except (TypeError, ValueError):
                    return "schema_f5_count_non_int"
            if not ordered_counts or len(ordered_counts) < 2:
                return "schema_f5_too_few_counts"
            if strict_semantic and max(ordered_counts) > F5_MAX_COUNT_VALUE:
                return "schema_f5_count_too_large"
            if any(b < a for a, b in zip(ordered_counts, ordered_counts[1:])):
                return "schema_f5_count_decreases"
            if strict_semantic and len(set(ordered_counts)) < 2:
                return "schema_f5_count_no_update"

    # multi_emit must have ≥ 2 distinct emit chunks (otherwise treat as single)
    if card.get("question_type") == "multi_emit":
        if len(set(emit_chunks)) < 2:
            return "schema_multi_emit_single_chunk"
    elif len(set(emit_chunks)) != 1:
        return "schema_single_emit_not_one_chunk"

    if family in {"F7", "CRR1"}:
        ordered = sorted(
            (int(e.get("chunk")), str(e.get("value", "")).strip().lower())
            for e in emits
        )
        vals = {v for _c, v in ordered}
        if family == "CRR1":
            if strict_semantic:
                if not {"no", "yes"}.issubset(vals):
                    return "schema_status_missing_no_yes"
                q_lower = q.lower()
                sufficiency_terms = (
                    "enough visual evidence",
                    "enough information",
                    "current visual",
                    "current video frames",
                    "current frames",
                    "current view",
                    "video frames confirm",
                    "visual content confirm",
                    "latest",
                    "can you answer",
                    "answer what",
                    "answer whether",
                    "provide enough",
                    "can you determine",
                    "now determine",
                    "currently identify",
                    "can you currently identify",
                )
                if not any(term in q_lower for term in sufficiency_terms):
                    return "schema_crr1_not_sufficiency_probe"
                if _CRR_HISTORICAL_STATUS_RE.search(q):
                    return "schema_crr1_historical_status_wording"
                seen_yes = False
                first_yes = None
                for c, v in ordered:
                    if v == "yes":
                        seen_yes = True
                        if first_yes is None:
                            first_yes = c
                    elif v == "no" and seen_yes:
                        return "schema_status_non_monotonic"
                if first_yes is None or not any(v == "no" and c < first_yes for c, v in ordered):
                    return "schema_status_no_before_yes"
        else:
            if card.get("question_type") != "single_emit":
                return "schema_f7_not_single_emit"
            if len(ordered) != 1:
                return "schema_f7_not_single_probe"
            if card.get("options"):
                return "schema_f7_unexpected_options"
            if strict_semantic:
                q_lower = q.lower()
                if "currently" not in q_lower and "right now" not in q_lower:
                    return "schema_f7_not_current_status"
                if any(term in q_lower for term in ("happened", "yet", "by now")):
                    return "schema_f7_historical_status_wording"

    grounding_raw = card.get("grounding_frames") or []
    if not grounding_raw:
        return "schema_no_grounding_frames"
    grounding: List[int] = []
    for c in grounding_raw:
        try:
            grounding.append(int(c))
        except (TypeError, ValueError):
            return "schema_grounding_bad_chunk"

    # ── Layer 2: grounding evidence exists ──
    if not ev_by_chunk:
        # No evidence dict provided → skip Layer 2 (legacy callers)
        return "PASS"

    missing = [c for c in grounding if c not in ev_by_chunk]
    if missing:
        return "grounding_evidence_missing"
    emit_missing = [c for c in emit_chunks if c not in ev_by_chunk]
    if emit_missing:
        return "emit_evidence_missing"

    if not strict_semantic:
        return "PASS"

    # Loose evidence containment: canonical answer text should overlap
    # with the entity_id / fact text from at least one grounding chunk.
    canonical = (card.get("canonical_answer") or "").strip().lower()
    if family == "HLD1":
        # HLD1 is a negative/abstention card. It is only valid when the
        # supplied grounding evidence does NOT support any concrete non-Unable
        # option. This catches bad cards such as asking for a shirt color when
        # "Blue" is both an option and explicitly present in the support chunks.
        support_text = " ".join(
            _evidence_text(ev_by_chunk[c]) for c in grounding if c in ev_by_chunk
        )
        support_norm = f" {_norm_text(support_text)} "
        concrete_hits: List[str] = []
        for opt in card.get("options") or []:
            opt_text = _strip_option_label(str(opt))
            if "unable to answer" in opt_text.lower():
                continue
            opt_norm = _norm_text(opt_text)
            if len(opt_norm) >= 3 and f" {opt_norm} " in support_norm:
                concrete_hits.append(opt_text)
        if concrete_hits:
            return f"hld1_concrete_option_supported: {concrete_hits[:2]}"
        return "PASS"
    if af == "binary":
        # Binary progress/status questions usually have canonical Yes/No.
        # The literal token "yes" or "no" will not appear in visual evidence,
        # so grounding must rely on the emit chunk + evidence presence above.
        return "PASS"
    if af == "multiple_choice" and correct_option_text:
        canonical = correct_option_text.strip().lower()
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
    truthy = {"1", "true", "yes", "on"}
    allow_partial = os.environ.get("THINKSTREAM_ALLOW_PARTIAL_PASS3A_CACHE", "").lower() in truthy
    allow_stale = os.environ.get("THINKSTREAM_ALLOW_STALE_PASS3_CACHE", "").lower() in truthy
    if not (allow_partial or allow_stale) and not stage_version_ok("3a"):
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


# Legacy compatibility helpers for older pass3 smoke tests. The production
# v2 card generator above no longer uses this family classifier, but keeping
# the small pure-Python surface avoids breaking historical regression tests.
_EV_PROMPT_MAX_CHUNKS = 10


def _select_prompt_chunks(chunk_indices: List[int], cap: int = _EV_PROMPT_MAX_CHUNKS) -> List[int]:
    ordered = list(dict.fromkeys(int(i) for i in chunk_indices))
    if len(ordered) <= cap:
        return ordered
    if cap <= 1:
        return [ordered[0]]
    selected = []
    last = len(ordered) - 1
    for j in range(cap):
        selected.append(ordered[round(j * last / (cap - 1))])
    return list(dict.fromkeys(selected))


def _format_evidence_for_prompt(evidence: List[Dict], chunk_indices: List[int]) -> str:
    by_idx = {int(cap.get("chunk_idx", i)): cap for i, cap in enumerate(evidence)}
    lines: List[str] = []
    for idx in _select_prompt_chunks(chunk_indices):
        cap = by_idx.get(idx, {})
        parts = [f"chunk {idx}"]
        if cap.get("time") is not None:
            parts.append(f"time={cap.get('time')}")
        facts = []
        for fact in cap.get("atomic_facts") or []:
            facts.append(str(fact.get("fact", "")) if isinstance(fact, dict) else str(fact))
        entities = []
        for ent in cap.get("visible_entities") or []:
            if isinstance(ent, dict):
                entities.append(str(ent.get("desc") or ent.get("id") or ""))
            else:
                entities.append(str(ent))
        ocr = []
        for item in cap.get("ocr") or []:
            ocr.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
        state_changes = [str(x) for x in (cap.get("state_changes") or [])]
        if entities:
            parts.append("entities=" + "; ".join(x for x in entities if x))
        if facts:
            parts.append("facts=" + "; ".join(x for x in facts if x))
        if ocr:
            parts.append("ocr=" + "; ".join(x for x in ocr if x))
        if state_changes:
            parts.append("state_changes=" + "; ".join(state_changes))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


FAMILY_TARGETS: Dict[str, int] = {
    "F1": 3, "F2": 4, "F3": 2, "F4": 2,
    "E1": 3, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 2,
    "PN1": 45,
}
RETENTION_CLASS: Dict[str, str] = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium", "PN1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
}
FAMILY_FORCE_ATTEMPT = {"PN1"}


def _entity_key(entity) -> str:
    if isinstance(entity, dict):
        return str(entity.get("id") or entity.get("desc") or "").strip().lower()
    return str(entity or "").strip().lower()


def _ocr_key(item) -> str:
    if isinstance(item, dict):
        text = item.get("text", "")
    else:
        text = item
    return str(text or "").strip().lower()


def classify_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """Legacy novelty classifier used by PN1 regression tests."""
    families = {family: [] for family in FAMILY_TARGETS}
    seen_entities: set[str] = set()
    seen_ocr: set[str] = set()
    pn1: List[int] = []
    for pos, cap in enumerate(evidence):
        idx = int(cap.get("chunk_idx", pos))
        novel = bool(cap.get("state_changes"))
        for entity in cap.get("visible_entities") or []:
            key = _entity_key(entity)
            if key and key not in seen_entities:
                novel = True
                seen_entities.add(key)
        for item in cap.get("ocr") or []:
            key = _ocr_key(item)
            if len(key) > 2 and key not in seen_ocr:
                novel = True
                seen_ocr.add(key)
        if novel:
            pn1.append(idx)
    families["PN1"] = pn1[:50]
    return families
