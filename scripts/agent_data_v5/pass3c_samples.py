"""Pass 3-C — Trajectory sample rendering (v2 model-agnostic).

For each trajectory's selected placements, walks every chunk in [0, num_chunks)
and emits ONE raw SFT sample per chunk (silent / response / recall+response /
recall+silent / patrol / compress_silent), using v2/design.py as the single
source of truth for gold actions.

Pipeline contract preserved:
  generate_trajectory_samples(trajectory, cards_map, rollout, evidence,
                               client, video_id) -> List[Dict]    (async)
  save_samples / load_samples
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from thinkstream.data.agent_protocol import (
    RECALL_RETURN_CHUNKS,
    build_assistant_content_v12,
    build_compress_trigger_user_input,
    format_memory_block,
    recall_time_string_for_chunks,
    select_recall_chunks,
)
from thinkstream.model.agent_loop import bm25_retrieve

from .config import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
    SAMPLES_3C_DIR,
    compute_visual_window_start,
)
from .pass3a_cards import dict_to_card
from .pass3b_placement import _dict_to_placement
from .stable_hash import stable_mod
from .v2.design import (
    Placement,
    render_video_samples as _design_render,
)
from .v2.llm_prompts import (
    family_taxonomy,
    parse_recall_query_response,
    recall_query_prompt,
    response_generation_prompt,
)

logger = logging.getLogger(__name__)


RECALL_MEMORY_OVERLAP_HARDEN_THRESHOLD = 0.50
RECALL_MEMORY_OVERLAP_ACCEPT_THRESHOLD = 0.35
RECALL_SUPPORT_OVERLAP_MIN = 0.40
RECALL_HARDEN_MAX_ATTEMPTS = 3
RECALL_HARDEN_CANDIDATES_PER_ATTEMPT = 3
RECALL_HARDEN_EVIDENCE_LINES = 56


# ---------------------------------------------------------------------------
# Helpers — think text + response/recall payload generation
# ---------------------------------------------------------------------------


def _think_for_chunk(rollout: Dict, chunk_idx: int) -> str:
    """Pull think text from question-blind rollout (pass2 output)."""
    for t in rollout.get("thinks", []):
        if int(t.get("chunk_idx", -1)) == chunk_idx:
            return str(t.get("think", "")).strip()
    return ""


def _recall_action_think(
    visual_think: str,
    *,
    final_action: str,
    reason: str = "",
) -> str:
    """Gold first-turn think for recall tool calls.

    Pass2 thinks are question-blind current-frame observations. For recall
    turns, keep that observation as this timestep's text memory, then add a
    short action decision so SFT learns why the recall tool is selected.
    """
    base = str(visual_think or "").strip()
    low = base.lower()
    if "visible evidence" in low and "recall" in low:
        return base
    reason = str(reason or "").strip()
    if final_action == "silent":
        if reason == "memory_unclear":
            decision = (
                "The active query depends on elapsed context, but the current "
                "view is not enough to answer. I will recall the earlier "
                "history once, and stay silent if it still does not contain "
                "the needed evidence."
            )
        elif reason == "related_history_check":
            decision = (
                "A related moment may have occurred earlier, so I should "
                "recall the elapsed history before deciding. If the retrieved "
                "history still lacks the answer, I will keep waiting."
            )
        elif reason == "pre_answer_check":
            decision = (
                "Before answering the pending query, I should check whether "
                "the answer already appeared in history. If it has not, the "
                "correct action is still an empty answer."
            )
        elif reason == "long_wait_history_check":
            decision = (
                "The query has stayed open long enough that earlier visual "
                "details may no longer be in current memory. I will recall "
                "elapsed history and keep waiting if the answer is still not "
                "supported."
            )
        else:
            decision = (
                "Current visible evidence is insufficient to answer the "
                "active query. The answer may not have appeared yet, so I "
                "will recall elapsed history once and stay silent if still "
                "unsupported."
            )
    else:
        if reason == "cumulative_history":
            decision = (
                "The current moment is relevant, but the answer also depends "
                "on earlier occurrences, so I will recall the prior window "
                "before giving the cumulative answer."
            )
        elif reason == "status_history":
            decision = (
                "The status question depends on an event that may have "
                "happened earlier, so I will recall that historical moment "
                "before answering."
            )
        else:
            decision = (
                "Current visible evidence is insufficient to answer the "
                "active query because the needed evidence is historical, so "
                "I will recall the earlier window rather than guess."
            )
    return f"{base} {decision}".strip()


MC_OPTION_LETTERS = "ABCDE"
MC_OPTION_COUNTS = {2, 3, 4, 5}
_OPTION_LABEL_RE = re.compile(r"^\s*(?:\([A-E]\)|[A-E][\).:])\s*")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MC_ANSWER_STYLES = ("letter_only", "letter_plus_text", "text_only")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "while", "what", "which", "who", "where", "when", "how", "is",
    "are", "was", "were", "be", "been", "being", "by", "from", "as", "it",
    "this", "that", "these", "those", "into", "onto", "there", "here", "his",
    "her", "their", "its", "your", "only", "answer", "option", "text",
    "letter", "video", "scene", "frame", "frames", "question",
}


def _strip_option_label(text: str) -> str:
    return _OPTION_LABEL_RE.sub("", str(text or "")).strip()


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _compact_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(ch for ch in text if ch.isalnum())


def _tokens(text: str) -> List[str]:
    return [
        t for t in _TOKEN_RE.findall(_norm_text(text))
        if len(t) >= 3 and t not in _STOPWORDS
    ]


def _token_overlap(answer: str, text: str) -> float:
    ans = set(_tokens(answer))
    if not ans:
        return 0.0
    return len(ans & set(_tokens(text))) / max(len(ans), 1)


def _text_contains_answer(answer: str, text: str) -> bool:
    answer_n = _norm_text(answer)
    text_n = _norm_text(text)
    if answer_n and len(answer_n) >= 3 and answer_n in text_n:
        return True
    answer_c = _compact_text(answer)
    text_c = _compact_text(text)
    return bool(answer_c and len(answer_c) >= 2 and answer_c in text_c)


def _answer_visible_in_text(answer: str, text: str, *, threshold: float) -> bool:
    if not str(answer or "").strip():
        return False
    return (
        _text_contains_answer(answer, text)
        or _token_overlap(answer, text) >= float(threshold)
    )


def _recall_query_leaks_answer(card: Dict, query: Dict) -> bool:
    """True when a recall search query exposes the target answer itself."""
    answer = _card_answer_text(card)
    if not answer or answer.strip().lower() == "unable to answer":
        return False
    return _answer_visible_in_text(
        answer,
        str((query or {}).get("query", "")),
        threshold=0.50,
    )


def _memory_overlap_score(
    text: str,
    memory_text: str,
    *,
    memory_tokens: Optional[set[str]] = None,
) -> float:
    toks = set(_tokens(text))
    if not toks:
        return 0.0
    mem = memory_tokens if memory_tokens is not None else set(_tokens(memory_text))
    if not mem:
        return 0.0
    return len(toks & mem) / max(len(toks), 1)


def _chunk_idx(cap: Dict, fallback: int = -1) -> int:
    try:
        return int(cap.get("chunk_idx", fallback))
    except (TypeError, ValueError):
        return fallback


def _evidence_text(cap: Dict) -> str:
    parts: List[str] = []
    for ent in cap.get("visible_entities") or []:
        if isinstance(ent, dict):
            parts.extend([
                str(ent.get("id", "")),
                str(ent.get("desc", "")),
                str(ent.get("action", "")),
            ])
        else:
            parts.append(str(ent))
    for fact in cap.get("atomic_facts") or []:
        parts.append(str(fact.get("fact", "")) if isinstance(fact, dict) else str(fact))
    for ocr in cap.get("ocr") or []:
        parts.append(str(ocr.get("text", "")) if isinstance(ocr, dict) else str(ocr))
    for sc in cap.get("state_changes") or []:
        parts.append(
            str(sc.get("text", sc.get("change", ""))) if isinstance(sc, dict)
            else str(sc)
        )
    if cap.get("spatial"):
        parts.append(str(cap.get("spatial")))
    if cap.get("think"):
        parts.append(str(cap.get("think")))
    return " ".join(p for p in parts if p).strip()


def _snapshot_for_chunk(rollout: Dict, chunk_idx: int) -> Dict:
    snapshots = rollout.get("snapshots") or {}
    return snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx)) or {}


def _memory_from_snapshot(snapshot: Dict) -> Dict:
    memory = {"compressed_segments": [], "recent_thinks": []}
    for seg in snapshot.get("compressed_segments") or []:
        if not isinstance(seg, dict):
            continue
        if "time_range" in seg and "text" in seg:
            memory["compressed_segments"].append({
                "time_range": seg.get("time_range"),
                "text": str(seg.get("text", "")),
            })
    for item in snapshot.get("recent_thinks") or []:
        if isinstance(item, dict):
            memory["recent_thinks"].append({
                "time": str(item.get("time", "")),
                "text": str(item.get("text", item.get("obs", ""))),
            })
        elif isinstance(item, str):
            memory["recent_thinks"].append(item)
    return memory


def _memory_text_for_chunk(rollout: Dict, chunk_idx: int) -> str:
    return format_memory_block(_memory_from_snapshot(_snapshot_for_chunk(rollout, chunk_idx)))


def _current_context_text_for_chunk(
    rollout: Dict,
    chunk_idx: int,
    *,
    memory_text: str = "",
) -> str:
    """Text visible before a recall tool call at the answer chunk."""
    parts = [memory_text, _think_for_chunk(rollout, chunk_idx)]
    return "\n".join(p for p in parts if str(p or "").strip())


def _card_answer_text(card: Dict) -> str:
    if card.get("answer_form") == "multiple_choice":
        _letter, text = _mc_correct_letter_text(card)
        if text:
            return text
    return str(card.get("canonical_answer") or "").strip()


def _support_evidence_text(evidence_by_chunk: Dict[int, Dict], chunks: List[int]) -> str:
    return "\n".join(
        f"[{c}-{c + 1}] {_evidence_text(evidence_by_chunk.get(int(c), {}))}"
        for c in sorted(set(int(x) for x in chunks))
        if _evidence_text(evidence_by_chunk.get(int(c), {}))
    )


def _mc_correct_letter_text(card: Dict, fallback: str = "") -> tuple[str, str]:
    """Return (correct_letter, correct_option_text) for an MC card."""
    options = list(card.get("options") or [])
    correct = str(card.get("correct_option") or "").strip().upper()
    if len(options) in MC_OPTION_COUNTS and correct in MC_OPTION_LETTERS[:len(options)]:
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(options):
            text = _strip_option_label(options[idx])
            if text:
                return correct, text
    canonical = str(card.get("canonical_answer") or "").strip()
    if canonical and canonical.upper() not in set(MC_OPTION_LETTERS):
        return correct, _strip_option_label(canonical)
    return correct, str(fallback or "").strip()


def _mc_answer_style_for_card(card: Dict, video_id: str = "") -> str:
    """Stable MC target-format mixture.

    SFT needs to learn OvO-compatible letter-only answering, but not only
    that format. The split is intentionally card-stable so all response /
    recall samples for the same question use one protocol:
      60% letter_only, 25% letter_plus_text, 15% text_only.
    """
    explicit = str(card.get("answer_style") or "").strip()
    if explicit in MC_ANSWER_STYLES:
        return explicit
    bucket = stable_mod(video_id, card.get("card_id", ""), card.get("question", ""),
                        modulo=100)
    if bucket < 60:
        return "letter_only"
    if bucket < 85:
        return "letter_plus_text"
    return "text_only"


def _mc_answer_instruction(style: str, options: Optional[List[str]] = None) -> str:
    if style == "letter_only":
        n_opts = len(list(options or []))
        labels = [chr(ord("A") + i) for i in range(max(2, min(n_opts or 4, 26)))]
        label_text = ", ".join(labels[:-1]) + f", or {labels[-1]}"
        return f"Answer format: one letter only ({label_text})."
    if style == "letter_plus_text":
        return "Answer format: letter plus option text, e.g. A) option text."
    return "Answer format: answer text only, no option letter."


def _mc_answer_text(card: Dict, fallback: str = "", style: Optional[str] = None) -> str:
    """Return the SFT target string for an MC card.

    Pass3A LLM cards may put a local evidence phrase in gold_emits while the
    actual answer is the canonical option. Use correct_option/options first so
    the response answers the question, then fall back to canonical_answer.

    `style` controls the output protocol. If absent, keep the old text-only
    behavior for offline tests and legacy cards.
    """
    correct, text = _mc_correct_letter_text(card, fallback)
    style = style or str(card.get("answer_style") or "text_only")
    if style == "letter_only" and correct:
        return correct
    if style == "letter_plus_text" and correct:
        return f"{correct}) {text}" if text else correct
    return text


def _clean_emit_text(value: str) -> str:
    """Humanize per-emit labels such as `pouring_butter@1`."""
    text = str(value or "").strip()
    text = re.sub(r"@\d+\s*$", "", text)
    text = text.replace("_", " ")
    return text.strip()


def _response_text_for(card: Dict, value: str) -> str:
    """Map gold_emit value → assistant response text (synchronous fast path).

    For MC: emits the card-stable target style (letter-only, letter+text,
    or text-only) using correct_option/options metadata.
    For binary/number/short_exact: emits the value directly.
    For descriptive single_emit: uses canonical_answer; for multi_emit:
    emits only the current per-event value to avoid future leakage.
    Use _response_text_via_llm when client is provided for richer descriptive text.
    """
    af = card.get("answer_form", "")
    if af == "multiple_choice":
        return _mc_answer_text(card, value, style=card.get("answer_style"))
    if af in ("binary", "number", "short_exact"):
        return value
    if card.get("question_type") == "multi_emit" and value:
        return _clean_emit_text(value)
    return card.get("canonical_answer", "") or value


async def _response_text_via_llm(card: Dict, value: str, client, video_id: str,
                                  chunk_idx: int) -> str:
    """397B-driven response text. Falls back to _response_text_for on failure.

    Only fires for descriptive single-emit answers. MC, short forms, and
    multi-emit narration/counting are deterministic so they cannot drift to
    a wrong option or leak future emits.
    """
    af = card.get("answer_form", "")
    if (
        af in ("multiple_choice", "binary", "number", "short_exact")
        or card.get("question_type") == "multi_emit"
    ):
        return _response_text_for(card, value)
    prompt = response_generation_prompt(card, chunk_idx)
    if not prompt:
        return _response_text_for(card, value)
    cfg = PASS_CONFIG.get("pass3c_response", PASS_CONFIG.get("pass3c", {}))
    try:
        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(cfg.get("max_tokens", 4096)),
            temperature=float(cfg.get("temperature", 0.3)),
            request_id=f"{video_id}_3c_resp_{card.get('card_id','?')}_{chunk_idx}",
            enable_thinking=cfg.get("thinking", False),
        )
    except Exception as exc:
        logger.warning(f"[{video_id}] 3c response LLM failed: {exc}")
        return _response_text_for(card, value)
    text = (raw or "").strip().strip('"').strip("'").strip()
    return text or _response_text_for(card, value)


def _query_keywords(question: str) -> str:
    keywords = " ".join(
        w.lower() for w in str(question or "").split() if len(w) > 3
    )[:80]
    return keywords or str(question or "").strip().lower()[:80]


_RECALL_TIME_RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$"
)


def _valid_recall_time_range(value: object) -> bool:
    """Return True for a non-empty past-time range like ``0-12``."""
    if not isinstance(value, str):
        return False
    m = _RECALL_TIME_RANGE_RE.fullmatch(value)
    if not m:
        return False
    start, end = float(m.group(1)), float(m.group(2))
    return end > start


def _valid_recall_query(query: Dict) -> bool:
    return bool((query or {}).get("query")) and _valid_recall_time_range(
        (query or {}).get("time_range")
    )


def _recall_range_end(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None
    m = _RECALL_TIME_RANGE_RE.fullmatch(value)
    if not m:
        return None
    return float(m.group(2))


def _recall_query_available(query: Dict, current_chunk: int) -> bool:
    """Recall queries may only search observations strictly before now."""
    if not _valid_recall_query(query):
        return False
    end = _recall_range_end((query or {}).get("time_range"))
    return end is not None and end <= current_chunk * AGENT_CHUNK_SEC


def _support_chunks(card: Dict) -> List[int]:
    chunks: List[int] = []
    for key in ("grounding_frames", "support_chunks"):
        for c in card.get(key) or []:
            try:
                chunks.append(int(c))
            except (TypeError, ValueError):
                continue
    if not chunks:
        for e in card.get("gold_emits") or []:
            try:
                chunks.append(int(e.get("chunk")))
            except (AttributeError, TypeError, ValueError):
                continue
    return sorted(set(chunks))


def _support_chunks_before(card: Dict, current_chunk: int) -> List[int]:
    return [
        c for c in _support_chunks(card)
        if c < int(current_chunk)
    ]


def _grounding_time_range_before(card: Dict, current_chunk: int) -> str:
    grounding = _support_chunks_before(card, current_chunk)
    if not grounding:
        return ""
    tr_start = min(grounding) * AGENT_CHUNK_SEC
    tr_end = (max(grounding) + 1) * AGENT_CHUNK_SEC
    return f"{int(tr_start)}-{int(tr_end)}"


def _hld_recall_query_for(card: Dict, current_chunk: int) -> Dict:
    """Recall query for HLD/Unable cases.

    HLD recall is an evidence check, not a search for the answer value. The
    query includes the requested target plus broad scene anchors so retrieval can
    return representative historical observations for verifying absence.
    """
    time_range = _grounding_time_range_before(card, current_chunk)
    if not time_range:
        end_s = max(0, int(current_chunk * AGENT_CHUNK_SEC))
        time_range = f"0-{end_s}" if end_s > 0 else ""
    banned = {
        "what", "which", "where", "when", "color", "material", "many",
        "video", "unable", "answer", "option", "did", "leave", "close",
        "open", "before", "after",
    }
    target_terms = [t for t in _tokens(card.get("question", "")) if t not in banned]
    terms = (target_terms[:3] + ["visible", "objects", "scene"])[:5]
    return {"query": " ".join(terms), "time_range": time_range}


def _recall_query_for(card: Dict, current_chunk: int) -> Dict:
    """Build recall_query (synchronous fast path).

    Returns card.recall_query if pre-generated, else heuristic.
    """
    if _is_unanswerable_card(card):
        return _hld_recall_query_for(card, current_chunk)
    if (
        card.get("question_type") != "multi_emit"
        and card.get("recall_query")
        and _recall_query_available(card["recall_query"], current_chunk)
        and not _recall_query_leaks_answer(card, card["recall_query"])
    ):
        return card["recall_query"]
    time_range = _grounding_time_range_before(card, current_chunk)
    q = card.get("question", "")
    keywords = _query_keywords(q)
    return {"query": keywords, "time_range": time_range}


def _repair_recall_query_for_response(
    card: Dict,
    query: Dict,
    current_chunk: int,
) -> Dict:
    """Return a legal recall query for an answerable recall-response sample.

    The normal path should already be legal because pass3b only selects recall
    placements with support in the past and outside the visual window. This
    helper repairs stale LLM/cache query ranges by rebuilding from support.
    """
    if (
        _recall_query_available(query, current_chunk)
        and not _recall_query_leaks_answer(card, query)
    ):
        return query
    repaired = _recall_query_for(card, current_chunk)
    if (
        _recall_query_available(repaired, current_chunk)
        and not _recall_query_leaks_answer(card, repaired)
    ):
        return repaired
    return {}


def _recall_wait_query_for(card: Dict, chunk_idx: int) -> Dict:
    """Recall query for forward wait-state samples.

    This deliberately ignores card.recall_query / grounding_frames because
    those point to the future answer evidence. At a real streaming timestep
    the student cannot know that future range. The query searches only the
    already elapsed history up to the current chunk; an empty result teaches
    "keep waiting", not "answer from future".
    """
    end_s = max(0, int(chunk_idx * AGENT_CHUNK_SEC))
    return {
        "query": _query_keywords(card.get("question", "")),
        "time_range": f"0-{end_s}" if end_s > 0 else "",
    }


async def _recall_query_via_llm(card: Dict, client, video_id: str,
                                  chunk_idx: int) -> Dict:
    """397B-driven recall_query. Caches result on card so we don't re-call."""
    if card.get("question_type") == "multi_emit":
        # Cumulative/status multi-emit recall depends on the current probe
        # chunk. A card-level cached teacher query can be too narrow for later
        # probes, so use the deterministic current-chunk range.
        return _recall_query_for(card, chunk_idx)
    if (
        card.get("recall_query")
        and _recall_query_available(card["recall_query"], chunk_idx)
        and not _recall_query_leaks_answer(card, card["recall_query"])
    ):
        return card["recall_query"]
    prompt = recall_query_prompt(card, current_chunk=chunk_idx, mode="answer")
    cfg = PASS_CONFIG.get("pass3c_recall_query", PASS_CONFIG.get("pass3c", {}))
    fallback_tr = _grounding_time_range_before(card, chunk_idx)
    try:
        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(cfg.get("max_tokens", 4096)),
            temperature=float(cfg.get("temperature", 0.3)),
            request_id=f"{video_id}_3c_rq_{card.get('card_id','?')}_{chunk_idx}",
            enable_thinking=cfg.get("thinking", False),
        )
    except Exception as exc:
        logger.warning(f"[{video_id}] 3c recall_query LLM failed: {exc}")
        return _recall_query_for(card, chunk_idx)
    rq = parse_recall_query_response(raw or "", fallback_time_range=fallback_tr)
    if (
        not _recall_query_available(rq, chunk_idx)
        or _recall_query_leaks_answer(card, rq)
    ):
        return _recall_query_for(card, chunk_idx)
    card["recall_query"] = rq      # cache for re-use within trajectory
    return rq


def _parse_json_candidates(raw: str) -> List[Dict]:
    """Parse one or more JSON object candidates from an LLM response."""
    if not raw:
        return []
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.replace("```json", "```", 1)
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    candidates = []
    obj_start, obj_end = text.find("{"), text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])
    list_start, list_end = text.find("["), text.rfind("]")
    if list_start >= 0 and list_end > list_start:
        candidates.append(text[list_start:list_end + 1])
    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            for key in ("candidates", "cards", "items"):
                wrapped = parsed.get(key)
                if isinstance(wrapped, list):
                    return [x for x in wrapped if isinstance(x, dict)]
            return [parsed]
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    return []


def _history_evidence_lines(
    evidence_by_chunk: Dict[int, Dict],
    card: Dict,
    current_chunk: int,
    *,
    memory_text: str = "",
    max_lines: int = RECALL_HARDEN_EVIDENCE_LINES,
) -> str:
    """Rich historical evidence pool for replacement recall-card generation.

    The pool is restricted to chunks that are outside the current visual
    window. The selected recall slot keeps its ask/response chunk fixed; this
    only gives the teacher grounded historical material from which it may build
    a harder question.
    """
    visual_start = compute_visual_window_start(int(current_chunk))
    original_support = set(_support_chunks_before(card, int(current_chunk)))
    memory_tokens = set(_tokens(memory_text)) if memory_text else set()
    scored = []
    for ci, cap in evidence_by_chunk.items():
        if ci < 0 or ci >= visual_start:
            continue
        text = _evidence_text(cap)
        if not text:
            continue
        score = 0
        if ci in original_support:
            score += 100
        score += 6 * len(cap.get("ocr") or [])
        score += 3 * len(cap.get("atomic_facts") or [])
        score += 3 * len(cap.get("state_changes") or [])
        if cap.get("spatial"):
            score += 2
        score += min(len(text) // 160, 4)
        if memory_text:
            novelty = 1.0 - _memory_overlap_score(
                text,
                memory_text,
                memory_tokens=memory_tokens,
            )
            score += int(max(0.0, novelty) * 24)
        scored.append((score, ci, text))
    scored = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_lines]
    return "\n".join(
        f"[c{ci} | t={ci * AGENT_CHUNK_SEC}-{(ci + 1) * AGENT_CHUNK_SEC}] {text[:700]}"
        for _score, ci, text in sorted(scored, key=lambda x: x[1])
    )


def _recall_hardening_prompt(
    card: Dict,
    *,
    current_chunk: int,
    memory_text: str,
    evidence_lines: str,
    answer_form: str,
    previous_error: str = "",
) -> str:
    options_doc = ""
    if answer_form == "multiple_choice":
        options_doc = """
  "options": ["A) ...", "B) ...", "..."],  # 2-5 options; default to 4, add "E) ..." only when useful
  "correct_option": "A" | "B" | "C" | "D" | "E","""
    prev = f"\nPrevious rejected candidate reason: {previous_error}\n" if previous_error else ""
    return f"""You are repairing ONE selected recall training slot.

The trajectory slot is fixed and MUST NOT change:
- family: {card.get('family', '')} / {card.get('family_name', '')}
- answer_form: {answer_form}
- question_type: single_emit
- ask/answer chunk: c{int(current_chunk)}

The current model-visible context at c{int(current_chunk)} is below, including
memory summaries and the current visual observation. The new question must NOT
be answerable from this context:
<current_memory>
{memory_text[:7000]}
</current_memory>

Historical evidence outside the current visual window is below. The new
question and answer MUST be fully verifiable from this evidence:
<historical_evidence>
{evidence_lines[:11000]}
</historical_evidence>
{prev}
Generate a replacement card for the same slot.

Rules:
- Keep the same family and answer_form. Do not change the ask/answer chunk.
- The answer must be present in historical_evidence but absent from current_memory.
- Prefer an answer with at least one specific content word that is not present
  in current_memory. Avoid generic answers that a strong text memory could infer.
- Prefer fine visual details, OCR text, object relations, before/after order,
  or cross-event details that compression summaries usually omit.
- Do not ask a question whose answer is "Unable to answer".
- The question must not contain or paraphrase the answer.
- grounding_frames must be the minimal historical chunk indices needed to
  verify the answer; every index must appear in historical_evidence and be
  before c{int(current_chunk)}.
- recall_query.query must contain search keywords only, not the answer value
  or any correct-option text. If the question asks for exact OCR/number/color/
  state/count, query for the surrounding object/action/location instead of
  that target value. Use neutral anchors from the question/event.
- Use compact closed-form video-QA wording. Prefer short, user-facing questions
  with natural event anchors such as before/after/while/when rather than
  internal chunks or long setup text.
- Keep most replacement cards answerable by one clear historical visual fact,
  but allow hard exploratory cases when historical_evidence supports them:
  separated clues, before/after state, object tracking, OCR, or evidence
  insufficiency boundaries.
- For multiple-choice replacements, options must be the same semantic type and
  similarly specific. Use plausible close distractors from historical_evidence,
  visually similar objects/actions, before/after states, or OCR-like snippets.
  Avoid random choices, all/none-of-the-above, length giveaways, and synonyms
  of the correct answer.

Output ONLY a JSON array of {RECALL_HARDEN_CANDIDATES_PER_ATTEMPT} distinct
candidate objects, best candidate first:
[
  {{
    "question": "...",
    "canonical_answer": "...",{options_doc}
    "grounding_frames": [int, ...],
    "recall_query": {{"query": "3-6 keywords", "time_range": "start-end"}}
  }}
]"""


def _candidate_to_recall_card(
    candidate: Dict,
    original: Dict,
    *,
    current_chunk: int,
) -> Dict:
    answer_form = str(original.get("answer_form") or "").strip()
    family = str(original.get("family") or "").strip()
    if not isinstance(candidate, dict) or not family or not answer_form:
        return {}
    question = str(candidate.get("question") or "").strip()
    if len(question) < 8:
        return {}
    grounding = []
    for raw in candidate.get("grounding_frames") or []:
        try:
            c = int(raw)
        except (TypeError, ValueError):
            continue
        if c < int(current_chunk):
            grounding.append(c)
    grounding = sorted(set(grounding))
    if not grounding:
        return {}

    def default_recall_query() -> Dict:
        return {
            "query": _query_keywords(question),
            "time_range": (
                f"{min(grounding) * AGENT_CHUNK_SEC}-"
                f"{(max(grounding) + 1) * AGENT_CHUNK_SEC}"
            ),
        }

    out = deepcopy(original)
    out["question"] = question
    out["family"] = family
    out["answer_form"] = answer_form
    out["question_type"] = "single_emit"
    out["grounding_frames"] = grounding
    out["support_chunks"] = grounding
    out.update(family_taxonomy(family))

    if answer_form == "multiple_choice":
        options = list(candidate.get("options") or [])
        correct = str(candidate.get("correct_option") or "").strip().upper()
        if len(options) not in MC_OPTION_COUNTS or correct not in MC_OPTION_LETTERS[:len(options)]:
            return {}
        relabelled = [
            f"{chr(65 + i)}) {_strip_option_label(str(opt)).strip()}"
            for i, opt in enumerate(options)
        ]
        correct_text = _strip_option_label(relabelled[ord(correct) - ord("A")]).strip()
        if not correct_text:
            return {}
        out["options"] = relabelled
        out["correct_option"] = correct
        out["canonical_answer"] = correct_text
        emit_value = correct
        answer_text = correct_text
    else:
        answer = str(candidate.get("canonical_answer") or "").strip()
        if not answer:
            return {}
        if answer_form == "binary" and answer not in {"Yes", "No"}:
            return {}
        if answer_form == "number" and not re.fullmatch(r"\d+", answer):
            return {}
        out["canonical_answer"] = answer
        out["options"] = None
        out["correct_option"] = None
        emit_value = answer
        answer_text = answer

    out["gold_emits"] = [{"chunk": max(grounding), "value": emit_value}]
    rq = candidate.get("recall_query") or {}
    if not isinstance(rq, dict):
        rq = {}
    if (
        not _valid_recall_query(rq)
        or _answer_visible_in_text(answer_text, str(rq.get("query", "")), threshold=0.50)
    ):
        rq = default_recall_query()
    out["recall_query"] = rq
    out["recall_hardened"] = True
    out["recall_hardened_from_card_id"] = original.get("card_id", "")
    return out


def _validate_hardened_recall_card(
    card: Dict,
    *,
    current_chunk: int,
    memory_text: str,
    evidence_by_chunk: Dict[int, Dict],
) -> tuple[bool, str]:
    answer = _card_answer_text(card)
    if not answer or answer.strip().lower() == "unable to answer":
        return False, "empty_or_unanswerable_answer"
    question = str(card.get("question") or "")
    if _answer_visible_in_text(answer, question, threshold=0.50):
        return False, "question_leaks_answer"
    if _answer_visible_in_text(
        answer, memory_text,
        threshold=RECALL_MEMORY_OVERLAP_ACCEPT_THRESHOLD,
    ):
        return False, "answer_still_visible_in_memory"

    visual_start = compute_visual_window_start(int(current_chunk))
    grounding = _support_chunks_before(card, int(current_chunk))
    if not grounding:
        return False, "no_past_grounding"
    not_historical = [c for c in grounding if c >= visual_start]
    if not_historical:
        return False, f"grounding_inside_visual_window:{not_historical[:5]}"
    missing = [c for c in grounding if c not in evidence_by_chunk]
    if missing:
        return False, f"grounding_missing_evidence:{missing[:5]}"
    support_text = _support_evidence_text(evidence_by_chunk, grounding)
    if not support_text:
        return False, "empty_support_text"
    if not (
        _text_contains_answer(answer, support_text)
        or _token_overlap(answer, support_text) >= RECALL_SUPPORT_OVERLAP_MIN
    ):
        return False, "answer_not_supported_by_evidence_text"

    rq = card.get("recall_query") or {}
    if not _recall_query_available(rq, int(current_chunk)):
        return False, "bad_recall_query_time"
    if _answer_visible_in_text(answer, str(rq.get("query", "")), threshold=0.50):
        return False, "recall_query_leaks_answer"
    return True, "pass"


def _needs_recall_hardening(card: Dict, memory_text: str) -> bool:
    answer = _card_answer_text(card)
    if not answer or answer.strip().lower() == "unable to answer":
        return False
    return _answer_visible_in_text(
        answer, memory_text,
        threshold=RECALL_MEMORY_OVERLAP_HARDEN_THRESHOLD,
    )


def _is_unanswerable_card(card: Dict) -> bool:
    if card.get("family") == "HLD1":
        return True
    return _card_answer_text(card).strip().lower() == "unable to answer"


def _set_placement_response_value(placement: Placement, value: str) -> None:
    for c, (kind, _old) in list(placement.chunk_actions.items()):
        if kind == "response":
            placement.chunk_actions[c] = ("response", str(value))


def _downgrade_recall_to_memory_direct(
    placement: Placement,
    card: Dict,
    *,
    reason: str,
) -> None:
    """Keep an easy historical slot answerable without emitting false recall."""
    placement.mechanism = "memory_direct"
    placement.difficulty_mode = "memory_direct_hardening_fallback"
    placement.recall_need = f"hardening_failed:{reason}"[:160]
    placement.recall_at.clear()
    card["recall_hardening_fallback"] = "memory_direct"
    card["recall_hardening_fallback_reason"] = str(reason)


async def _harden_one_recall_slot(
    *,
    card: Dict,
    placement: Placement,
    rollout: Dict,
    evidence_by_chunk: Dict[int, Dict],
    client,
    video_id: str,
) -> tuple[Dict, bool, str]:
    response_chunks = [
        int(c) for c, (kind, _value) in placement.chunk_actions.items()
        if kind == "response"
    ]
    if not response_chunks:
        return card, False, "no_response_chunk"
    current_chunk = max(response_chunks)
    memory_text = _memory_text_for_chunk(rollout, current_chunk)
    current_context_text = _current_context_text_for_chunk(
        rollout,
        current_chunk,
        memory_text=memory_text,
    )
    ok, reason = _validate_hardened_recall_card(
        card,
        current_chunk=current_chunk,
        memory_text=current_context_text,
        evidence_by_chunk=evidence_by_chunk,
    )
    if ok:
        return card, False, "already_hard"

    if reason in {"bad_recall_query_time", "recall_query_leaks_answer"}:
        repaired = deepcopy(card)
        repaired["recall_query"] = _recall_query_for(repaired, current_chunk)
        ok, repaired_reason = _validate_hardened_recall_card(
            repaired,
            current_chunk=current_chunk,
            memory_text=current_context_text,
            evidence_by_chunk=evidence_by_chunk,
        )
        if ok:
            return repaired, True, "hardened"
        reason = repaired_reason

    if client is None:
        return card, False, f"{reason}:no_client"

    answer_form = str(card.get("answer_form") or "")
    if answer_form not in {"multiple_choice", "descriptive", "number", "binary", "short_exact"}:
        return card, False, f"unsupported_answer_form:{answer_form}"
    evidence_lines = _history_evidence_lines(
        evidence_by_chunk, card, current_chunk, memory_text=current_context_text,
    )
    if not evidence_lines:
        return card, False, "no_historical_evidence_pool"

    cfg = PASS_CONFIG.get("pass3c_recall_hardening", PASS_CONFIG.get("pass3c", {}))
    previous_error = reason
    for attempt in range(RECALL_HARDEN_MAX_ATTEMPTS):
        prompt = _recall_hardening_prompt(
            card,
            current_chunk=current_chunk,
            memory_text=current_context_text,
            evidence_lines=evidence_lines,
            answer_form=answer_form,
            previous_error=previous_error,
        )
        try:
            raw = await client._call_one(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(cfg.get("max_tokens", 4096)),
                temperature=float(cfg.get("temperature", 0.4)),
                request_id=(
                    f"{video_id}_3c_recall_harden_"
                    f"{card.get('card_id','?')}_{current_chunk}_{attempt}"
                ),
                enable_thinking=cfg.get("thinking", False),
            )
        except Exception as exc:
            previous_error = f"llm_call_failed:{exc}"
            logger.warning("[%s] recall hardening LLM failed: %s", video_id, exc)
            continue
        reasons: List[str] = []
        for parsed in _parse_json_candidates(raw or ""):
            candidate = _candidate_to_recall_card(
                parsed,
                card,
                current_chunk=current_chunk,
            )
            ok, reason = _validate_hardened_recall_card(
                candidate,
                current_chunk=current_chunk,
                memory_text=current_context_text,
                evidence_by_chunk=evidence_by_chunk,
            ) if candidate else (False, "parse_or_schema_failed")
            if ok:
                return candidate, True, "hardened"
            reasons.append(reason)
        previous_error = "; ".join(reasons[:RECALL_HARDEN_CANDIDATES_PER_ATTEMPT])
        if not previous_error:
            previous_error = "parse_or_schema_failed"
    return card, False, previous_error or "hardening_failed"


async def _harden_selected_recall_slots(
    *,
    placements: List[Placement],
    cards_map: Dict[str, Dict],
    rollout: Dict,
    evidence: List[Dict],
    client,
    video_id: str,
) -> Dict[str, int]:
    """Replace easy selected recall cards without changing trajectory timing.

    This function mutates ``cards_map`` and response values inside selected
    recall placements. It never changes ask_chunk, response chunk keys, or
    trajectory length. If a memory-answerable recall slot cannot be hardened
    into a true recall task, it is downgraded to memory_direct so the sample
    remains correct without emitting a false tool call.
    """
    evidence_by_chunk = {
        _chunk_idx(cap): cap for cap in evidence
        if isinstance(cap, dict) and _chunk_idx(cap) >= 0
    }
    stats = {
        "recall_slots": 0,
        "already_hard": 0,
        "hardened": 0,
        "unanswerable_recall_kept": 0,
        "downgraded_memory_direct": 0,
        "failed": 0,
    }
    hardening_jobs = []
    for placement in placements:
        if placement.mechanism != "recall_demo":
            continue
        card = cards_map.get(placement.card_id)
        if not card:
            continue
        stats["recall_slots"] += 1
        if _is_unanswerable_card(card):
            response_chunks = [
                int(c) for c, (kind, _value) in placement.chunk_actions.items()
                if kind == "response"
            ]
            if response_chunks:
                rq = _recall_query_for(card, max(response_chunks))
                if _recall_query_available(rq, max(response_chunks)):
                    card["recall_query"] = rq
                    placement.recall_need = "hld_absence_evidence_check"
                    stats["unanswerable_recall_kept"] += 1
                    continue
            _downgrade_recall_to_memory_direct(
                placement,
                card,
                reason="unanswerable_invalid_recall_query",
            )
            stats["downgraded_memory_direct"] += 1
            continue
        before_ask = int(placement.ask_chunk)
        before_chunks = sorted(int(c) for c in placement.chunk_actions.keys())
        hardening_jobs.append((
            placement,
            card,
            before_ask,
            before_chunks,
            _harden_one_recall_slot(
                card=card,
                placement=placement,
                rollout=rollout,
                evidence_by_chunk=evidence_by_chunk,
                client=client,
                video_id=video_id,
            ),
        ))

    if hardening_jobs:
        hardening_results = await asyncio.gather(
            *(job[4] for job in hardening_jobs),
            return_exceptions=True,
        )
    else:
        hardening_results = []

    for (placement, card, before_ask, before_chunks, _task), result in zip(
        hardening_jobs,
        hardening_results,
    ):
        if isinstance(result, Exception):
            new_card, changed, reason = (
                card,
                False,
                f"hardening_exception:{result}",
            )
        else:
            new_card, changed, reason = result
        after_chunks = sorted(int(c) for c in placement.chunk_actions.keys())
        if int(placement.ask_chunk) != before_ask or after_chunks != before_chunks:
            raise RuntimeError(
                f"[{video_id}] recall hardening changed trajectory timing for "
                f"card={placement.card_id}: ask {before_ask}->{placement.ask_chunk}, "
                f"chunks {before_chunks}->{after_chunks}"
            )
        if changed:
            cards_map[placement.card_id].clear()
            cards_map[placement.card_id].update(new_card)
            answer_value = (
                str(new_card.get("correct_option"))
                if new_card.get("answer_form") == "multiple_choice"
                else str(new_card.get("canonical_answer") or "")
            )
            _set_placement_response_value(placement, answer_value)
            stats["hardened"] += 1
            continue
        if reason == "already_hard":
            stats["already_hard"] += 1
            continue
        stats["failed"] += 1
        response_chunks = [
            int(c) for c, (kind, _value) in placement.chunk_actions.items()
            if kind == "response"
        ]
        still_easy = False
        if response_chunks:
            memory_text = _memory_text_for_chunk(rollout, max(response_chunks))
            current_context_text = _current_context_text_for_chunk(
                rollout,
                max(response_chunks),
                memory_text=memory_text,
            )
            still_easy = _needs_recall_hardening(
                card, current_context_text,
            )
        if still_easy:
            _downgrade_recall_to_memory_direct(
                placement,
                card,
                reason=reason,
            )
            stats["downgraded_memory_direct"] += 1
            logger.warning(
                "[%s] recall hardening fallback: card=%s downgraded to "
                "memory_direct (%s)",
                video_id,
                placement.card_id,
                reason,
            )
    if stats["recall_slots"]:
        logger.info("[%s] recall hardening stats: %s", video_id, stats)
    return stats


def _recall_result_for(
    card: Dict,
    rollout: Dict,
    noise_kind: str,
    *,
    current_chunk: Optional[int] = None,
    recall_query: Optional[Dict] = None,
) -> Dict:
    """Build a recall_result matching the old pass3c noise vocabulary.

    noise_kind ∈ {oracle, noisy, not_yet, failure}.
    Production data uses oracle/noisy/not_yet; failure is legacy diagnostic.
    """
    def _archive_before_now() -> List[Dict]:
        archive = []
        for t in rollout.get("thinks", []):
            try:
                ci = int(t.get("chunk_idx", t.get("chunk", -1)))
            except (TypeError, ValueError):
                continue
            if ci < 0:
                continue
            if current_chunk is not None and ci >= int(current_chunk):
                continue
            text = str(t.get("think", t.get("text", "")) or "").strip()
            if not text:
                continue
            archive.append({
                "chunk": ci,
                "time": f"{int(ci * AGENT_CHUNK_SEC)}-"
                        f"{int((ci + 1) * AGENT_CHUNK_SEC)}",
                "text": text,
            })
        return archive

    def _archive_text_for_chunks(archive: List[Dict], chunks: List[int]) -> str:
        by_chunk = {int(item.get("chunk", -1)): item for item in archive}
        lines = []
        for c in chunks:
            item = by_chunk.get(int(c))
            if not item:
                continue
            lines.append(f"[{item.get('time', '')}] {item.get('text', '')}")
        return "\n".join(lines)

    if current_chunk is None:
        grounding = _support_chunks(card)
    else:
        grounding = _support_chunks_before(card, int(current_chunk))
    absence_check = _is_unanswerable_card(card)
    if noise_kind == "not_yet":
        archive = _archive_before_now()
        retrieved = bm25_retrieve(
            recall_query or {},
            archive,
            max_results=RECALL_RETURN_CHUNKS,
        )
        chunks = select_recall_chunks(retrieved.get("returned_chunks") or [])
        if not chunks and archive:
            chunks = select_recall_chunks(
                [item["chunk"] for item in archive[-RECALL_RETURN_CHUNKS:]]
            )
        if not chunks:
            return {
                "source": "failure",
                "text_content": "No matching historical frames found.",
                "returned_chunks": [],
                "time": "",
            }
        tr_text = recall_time_string_for_chunks(chunks)
        return {
            "source": "historical_frames",
            "text_content": retrieved.get("text_content") or (
                f"Retrieved {len(chunks) * FRAMES_PER_CHUNK} historical frames "
                f"from t={tr_text}s; they do not contain enough evidence to "
                "answer the pending question yet."
            ),
            "returned_chunks": chunks,
            "time": tr_text,
            "result_kind": "not_yet",
        }
    if noise_kind == "failure" or not grounding:
        return {
            "source": "failure",
            "text_content": "No matching results found.",
            "returned_chunks": [],
            "time": "",
        }
    chunks: List[int] = []
    text_content = ""
    if _valid_recall_query(recall_query or {}):
        archive = _archive_before_now()
        retrieved = bm25_retrieve(
            recall_query or {},
            archive,
            max_results=RECALL_RETURN_CHUNKS,
        )
        chunks = select_recall_chunks(retrieved.get("returned_chunks") or [])
        text_content = retrieved.get("text_content", "")

    # Fallback preserves answerable recall samples when the teacher query text
    # does not lexically match the pass2 memory even though gold support exists.
    if not chunks:
        chunks = sorted(int(c) for c in grounding)
        text_content = _archive_text_for_chunks(_archive_before_now(), chunks)
    elif grounding and not any(int(c) in set(int(g) for g in grounding) for c in chunks):
        # For SFT, a successful recall turn must return the answer support,
        # not merely any lexical neighbor. BM25 can retrieve a distractor when
        # the query is underspecified, so snap back to the grounded chunks.
        chunks = sorted(int(c) for c in grounding)
        text_content = _archive_text_for_chunks(_archive_before_now(), chunks)

    if noise_kind == "noisy":
        # Inject a distractor chunk near grounding
        max_c = max(0, int(rollout.get("num_chunks", 1)) - 1)
        if current_chunk is not None:
            max_c = min(max_c, max(0, int(current_chunk) - 1))
        distractor = min(chunks[-1] + 5, max_c)
        if len(chunks) >= RECALL_RETURN_CHUNKS:
            chunks = chunks[:max(0, RECALL_RETURN_CHUNKS - 1)] + [distractor]
        else:
            chunks = chunks + [distractor]
    chunks = select_recall_chunks(chunks)
    if not chunks:
        return {
            "source": "failure",
            "text_content": "No matching results found.",
            "returned_chunks": [],
            "time": "",
        }
    tr_text = recall_time_string_for_chunks(chunks)
    if absence_check:
        text_content = text_content or (
            f"Retrieved {len(chunks) * FRAMES_PER_CHUNK} historical frames "
            f"from t={tr_text}s for an absence/insufficient-evidence check. "
            "The retrieved evidence should be used to decide whether the "
            "active query is genuinely unanswerable."
        )
    return {
        "source": "historical_frames",
        "text_content": text_content or (
            f"Recalled {len(chunks) * FRAMES_PER_CHUNK} frames from t={tr_text}s."
        ),
        "returned_chunks": chunks,
        "time": tr_text,
        **({"result_kind": "absence_check"} if absence_check else {}),
    }


def _mech_to_sequence_type(mech: str) -> str:
    """Map v2 mechanism → legacy sequence_type for back-compat metadata."""
    return {
        "silent_then_response": "event_watch",
        "direct": "immediate_response",
        "memory_direct": "memory_response",
        "recall_demo": "recall_success",
        "multi_emit": "multi_response",
    }.get(mech, "immediate_response")


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------


def _silent_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    trajectory_id: str, *, card_id: str = "",
    sequence_type: str = "", base_role: str = "active_silent",
    sample_subtype: str = "silent", user_input: str = "",
) -> Dict:
    """Plain silent sample (silent / patrol).

    recall_silent uses _recall_silent_multiturn_sample so the model sees the
    recall tool call and the no-match result before staying silent.
    """
    if sample_subtype == "recall+silent":
        raise ValueError(
            "recall+silent must be rendered by _recall_silent_multiturn_sample, "
            "not _silent_sample."
        )
    sample_type = "silent"
    output_text = build_assistant_content_v12(
        think=think, kind="answer", answer_text="",
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": sample_type,
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "silent",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": None,
        "base_role": base_role,
    }


def _compress_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    trajectory_id: str, compress_event: Dict,
) -> Dict:
    """Compress sample — model emits <tool_call>{"name":"compress",...}</tool_call>.

    This is the SFT supervision signal that teaches the model to output a
    compression tool_call when the system signals memory pressure.

      user_input = <compress_trigger/> (legacy event marker only)
      output     = <think>...</think><tool_call>{compress with gold summary
                                                  INCLUDING time_range}</tool_call>
      action     = "compress"
      v12_inter_chunk = True (pass5 suppresses visual_window, query, and
      recalled-frame response context)

    The gold summary (time_range + text) comes from rollout's compression_events,
    which were generated by the question-blind pass2 rollout.

    v12.12 (2026-05-02): trigger no longer carries range='a-b'. Range stays
    ONLY in the gold tool_call output — model must learn to derive the range
    from <memory> contents (chunk timestamps + summary boundaries) rather
    than copy it from the system-injected trigger. SFT loss covers every
    token of the assistant turn, including the time_range field, so the
    range-selection policy is distilled from pass2's score_range oracle
    into the model. This unblocks the v12.12 RL upgrade where the trigger
    is removed entirely and the model decides timing AND range.
    """
    summary = compress_event.get("summary", {}) or {}
    tr = summary.get("time_range", [])
    trigger_tag = build_compress_trigger_user_input()
    chunks = sorted(int(c) for c in
        (summary.get("source_chunks")
         or compress_event.get("compressed_source_chunks")
         or compress_event.get("compressed_thinks_chunks")
         or []))
    if isinstance(tr, list) and len(tr) == 2:
        tr0, tr1 = int(tr[0]), int(tr[1])
    elif chunks:
        tr0, tr1 = min(chunks), max(chunks) + 1
    else:
        tr0, tr1 = 0, 1
    if tr1 <= tr0:
        tr1 = tr0 + 1
    summary_arg = {"time_range": [tr0, tr1],
                   "text": summary.get("text", "")}
    if summary.get("text", ""):
        compress_think = (
            f"Memory is over budget, so I should compress older observations "
            f"from t={tr0}-{tr1} into a concise summary."
        )
    else:
        compress_think = (
            "Memory is over budget, so I should compress older observations "
            f"from t={tr0}-{tr1} into a concise summary."
        )
    output_text = build_assistant_content_v12(
        think=compress_think, kind="compress", compress_summary=summary_arg,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "compress",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": "",
        "sequence_type": "compress_event",
        "action": "compress",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": trigger_tag,                           # system signal
        "recall_result": None,
        "base_role": "compress_action",
        "v12_inter_chunk": True,
        # ICAE-style aux target: original gold caption / source chunks
        "gold_caption": summary.get("text", ""),
        "gold_compress_chunks": chunks,
    }


def _response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "",
) -> Dict:
    output_text = build_assistant_content_v12(
        think=think, kind="answer", answer_text=response,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "response",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "response",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": None,
    }


def _recall_response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    recall_query: Dict, recall_result: Dict,
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "", recall_reason: str = "",
) -> Dict:
    """Multi-turn recall sample (v12 protocol).

    sample_type='recall' triggers pass5's two-turn render:
      assistant → tool_call(recall_query)
      tool      → recall_result + recalled_frames
      assistant → final answer
    """
    turn1 = build_assistant_content_v12(
        think=_recall_action_think(
            think, final_action="response", reason=recall_reason
        ),
        kind="recall",
        recall_query=recall_query,
    )
    turn2_think = (
        "The recalled frames provide the historical evidence needed for this "
        "pending question. I compare that retrieved moment with the question "
        "and give the grounded answer without adding unsupported details."
    )
    turn2 = build_assistant_content_v12(
        think=turn2_think, kind="answer", answer_text=response,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "recall",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "response",
        "output": turn2,
        "v12_assistant_turn_1": turn1,
        "v12_assistant_turn_2": turn2,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": recall_result,
    }


def _recall_silent_multiturn_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    recall_query: Dict, recall_result: Dict,
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "", recall_reason: str = "",
) -> Dict:
    """Recall followed by an empty answer while the query remains open.

    Production use is the forward/waiting case: after a question is asked,
    recall may show that the answer has not appeared in history yet. The
    model should keep <answer></answer> empty for this chunk, leave the query
    pending, and answer at a later response chunk.

    SFT rendering (pass5 shape B variant):
      assistant → tool_call(recall_query)        ← turn1: model attempts recall
      tool      → recalled_frames + recall_result ← system returns historical frames
      assistant → think + empty <answer>          ← turn2: wait, query stays open
    """
    turn1 = build_assistant_content_v12(
        think=_recall_action_think(
            think, final_action="silent", reason=recall_reason
        ),
        kind="recall",
        recall_query=recall_query,
    )
    turn2_think = (
        "The recalled frames do not provide enough evidence to answer the "
        "pending question yet. I should keep the question pending and leave "
        "the answer empty until the relevant future moment is visible."
    )
    turn2 = build_assistant_content_v12(
        think=turn2_think, kind="answer", answer_text="",  # ← silent
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "recall",   # same shape B as recall_response
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "silent",        # ← gold action is silent (not response)
        "output": turn2,
        "v12_assistant_turn_1": turn1,
        "v12_assistant_turn_2": turn2,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": recall_result,
        "base_role": "recall_silent",
    }


def _append_query_answer(
    queries_state: List[Dict],
    queries_idx_by_card: Dict[str, int],
    card_id: str,
    *,
    chunk_idx: int,
    text: str,
    status: str,
) -> None:
    """Update query lifecycle after rendering an answer sample.

    The current sample is rendered before this mutation, so it still sees the
    query as pending. Subsequent samples see the new answer state and cannot
    confuse this question with the next placement's pending query.
    """
    if card_id not in queries_idx_by_card:
        return
    q = queries_state[queries_idx_by_card[card_id]]
    q.setdefault("answers", []).append({
        "text": text,
        "time": chunk_idx * AGENT_CHUNK_SEC,
    })
    q["status"] = status


# ---------------------------------------------------------------------------
# Public API (matches old pass3c interface)
# ---------------------------------------------------------------------------


async def generate_trajectory_samples(
    trajectory: Dict,
    cards_map: Dict[str, Dict],
    rollout: Dict,
    evidence: List[Dict],
    client=None,                     # optional LLM for descriptive response / recall query text
    video_id: str = "",
) -> List[Dict]:
    """Render one trajectory's placements into raw per-chunk samples.

    Output samples carry every field render_samples.render_sample() needs
    (chunk_idx, sample_type, action, output, queries, user_input,
    recall_result, sequence_type, card_id, trajectory_id, optional
    v12_assistant_turn_*). Non-descriptive answers stay deterministic;
    descriptive answer text and recall queries can use `client` when
    provided. render_samples then adds the `input` dict + metadata for
    pass3e/4/5.
    """
    placements_dicts = trajectory.get("placements", [])
    placements = [_dict_to_placement(p) for p in placements_dicts]
    traj_id = trajectory.get("trajectory_id", f"{video_id}_traj0")
    num_chunks = int(rollout.get("num_chunks", 0))
    # pass3 pipeline renders trajectories from the same video concurrently and
    # passes the video-level card map to each task. Recall hardening may replace
    # a selected card for this trajectory, so keep all card mutations local.
    cards_map = {cid: deepcopy(card) for cid, card in cards_map.items()}

    compress_chunks = [int(e.get("trigger_chunk", -1))
                       for e in rollout.get("compression_events", [])
                       if e.get("trigger_chunk", -1) >= 0]
    # Map trigger_chunk → full event so we can inject <compress_trigger>
    compress_event_by_chunk: Dict[int, Dict] = {
        int(e.get("trigger_chunk", -1)): e
        for e in rollout.get("compression_events", [])
        if e.get("trigger_chunk", -1) >= 0
    }

    # Run design's gold-action pipeline (handles patrol stratification,
    # compress_silent, priority resolution).
    for cid, card in cards_map.items():
        if (card or {}).get("answer_form") == "multiple_choice":
            style = _mc_answer_style_for_card(card, video_id)
            card["answer_style"] = style
            card["answer_instruction"] = _mc_answer_instruction(
                style, options=card.get("options") or []
            )

    await _harden_selected_recall_slots(
        placements=placements,
        cards_map=cards_map,
        rollout=rollout,
        evidence=evidence,
        client=client,
        video_id=video_id,
    )

    cards_obj = [dict_to_card(c) for c in cards_map.values()]
    placements_by_card: Dict[str, List[Placement]] = {}
    for p in placements:
        placements_by_card.setdefault(p.card_id, []).append(p)
    rng = random.Random(stable_mod(video_id, traj_id, modulo=10**6))
    design_samples = _design_render(
        cards_obj, placements_by_card, num_chunks,
        evidence=evidence, rng=rng,
        compression_event_chunks=compress_chunks,
    )

    # Build queries_state evolution chronologically
    queries_state: List[Dict] = []
    queries_idx_by_card: Dict[str, int] = {}
    placements_sorted = sorted(placements, key=lambda p: p.ask_chunk)
    ask_chunk_by_card = {p.card_id: p.ask_chunk for p in placements_sorted}
    response_chunks_by_card: Dict[str, List[int]] = {}
    for p in placements_sorted:
        response_chunks_by_card[p.card_id] = sorted(
            int(c) for c, (kind, _value) in p.chunk_actions.items()
            if kind == "response"
        )
    open_until_by_card = {
        cid: max(chunks) for cid, chunks in response_chunks_by_card.items()
        if chunks
    }

    raw: List[Dict] = []
    for ds in design_samples:
        c = ds.chunk_idx
        # Add queries that became active by this chunk
        for p in placements_sorted:
            if p.ask_chunk <= c and p.card_id not in queries_idx_by_card:
                card = cards_map.get(p.card_id) or {}
                queries_idx_by_card[p.card_id] = len(queries_state)
                # v12.13 fix (P0-3): include options + answer_form so
                # format_queries_block can render MC choices for active
                # queries (forward responses fire AFTER ask, with no fresh
                # user_input — model sees only the active-query block).
                queries_state.append({
                    "card_id": p.card_id,
                    "question": card.get("question", ""),
                    "options": list(card.get("options") or []),
                    "answer_form": card.get("answer_form", ""),
                    "answer_style": card.get("answer_style", ""),
                    "answer_instruction": card.get("answer_instruction", ""),
                    "ask_time": p.ask_chunk * AGENT_CHUNK_SEC,
                    "open_until": open_until_by_card.get(p.card_id, p.ask_chunk)
                                  * AGENT_CHUNK_SEC,
                    "status": "open",
                    "answers": [],
                })
        pending_queries = [
            q for q in queries_state
            if str(q.get("status", "")).lower() in ("open", "pending", "active")
            or (not q.get("status") and not q.get("answers"))
        ]
        if len(pending_queries) > 1:
            pending_names = [str(q.get("question", ""))[:80] for q in pending_queries]
            raise ValueError(
                f"{video_id}/{traj_id} has {len(pending_queries)} concurrent "
                f"pending questions at chunk {c}: {pending_names}"
            )

        card_id = ds.card_id
        card = cards_map.get(card_id) if card_id else None
        sequence_type = _mech_to_sequence_type(ds.mechanism) if card_id else ""
        # user_input fires only at the ask_chunk for that card.
        #
        # v12.13 (2026-05-02): MC options live ONLY in the query-state block via
        # format_queries_block (queries_state carries options + answer_form;
        # active MC queries render an "Options: A) ... B) ..." line).
        # Putting options ALSO in user_input was duplicating ~30 tokens
        # per ask (model saw the same option list twice: once in query state
        # and once in <user_input>). user_input now carries just the
        # question text, parity with non-MC asks.
        user_input = ""
        if card_id and ask_chunk_by_card.get(card_id) == c and card:
            user_input = card.get("question", "")

        if ds.sample_kind == "patrol":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                base_role="patrol", sample_subtype="patrol",
            ))
        elif ds.sample_kind == "compress_silent":
            # SFT must teach the model to OUTPUT compress tool_calls, not
            # observe silently. Render as a compress action sample with
            # gold summary from the rollout's compression event.
            ce = compress_event_by_chunk.get(c)
            if ce:
                raw.append(_compress_sample(
                    c, _think_for_chunk(rollout, c), queries_state,
                    traj_id, ce,
                ))
            else:
                # Defensive fallback if event missing — emit silent
                raw.append(_silent_sample(
                    c, _think_for_chunk(rollout, c), queries_state,
                    traj_id, base_role="compress_event",
                ))
        elif ds.sample_kind == "silent":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                card_id=card_id or "",
                sequence_type=sequence_type, user_input=user_input,
            ))
        elif ds.sample_kind == "recall+silent":
            recall_reason = str((ds.extra or {}).get("recall_reason", ""))
            # Wait-state recall must not use the card's grounding_frames:
            # those point to the future answer chunk and would leak timing.
            rq = _recall_wait_query_for(card or {}, c)
            if not _valid_recall_query(rq):
                # At the first chunk there is no past interval to search.
                # Training a recall tool call with time_range="" teaches an
                # invalid API call; the correct behavior is to stay silent and
                # keep the query open for a later chunk.
                raw.append(_silent_sample(
                    c, _think_for_chunk(rollout, c), queries_state, traj_id,
                    card_id=card_id or "", sequence_type=sequence_type,
                    user_input=user_input, base_role="recall_wait_no_history",
                ))
                continue
            rr = _recall_result_for(card or {}, rollout,
                                     ds.recall_result_kind or "not_yet",
                                     current_chunk=c,
                                     recall_query=rq)
            raw.append(_recall_silent_multiturn_sample(
                c, _think_for_chunk(rollout, c), queries_state,
                rq, rr, traj_id, card_id or "", sequence_type,
                user_input=user_input, recall_reason=recall_reason,
            ))
            # Do not append an answer or close the query. recall+silent is a
            # wait state; a later response/recall+response sample must answer.
        elif ds.sample_kind == "response":
            if client is not None:
                resp = await _response_text_via_llm(
                    card or {}, ds.response_text, client, video_id, c)
            else:
                resp = _response_text_for(card or {}, ds.response_text)
            raw.append(_response_sample(
                c, _think_for_chunk(rollout, c), resp, queries_state,
                traj_id, card_id, sequence_type, user_input=user_input,
            ))
            status = (
                "answered" if c >= open_until_by_card.get(card_id, c)
                else "open"
            )
            _append_query_answer(
                queries_state, queries_idx_by_card, card_id,
                chunk_idx=c, text=resp, status=status,
            )
        elif ds.sample_kind == "recall+response":
            recall_reason = str((ds.extra or {}).get("recall_reason", ""))
            if client is not None:
                resp = await _response_text_via_llm(
                    card or {}, ds.response_text, client, video_id, c)
                if recall_reason == "memory_text_needs_visual_verification":
                    # These slots are created deterministically from selected
                    # memory_direct hard-visual questions. The point is to
                    # verify historical visual evidence rather than rewrite
                    # the card, so avoid an extra teacher call and use the
                    # support-grounded query/range.
                    rq = _recall_query_for(card or {}, c)
                else:
                    rq = await _recall_query_via_llm(
                        card or {}, client, video_id, c
                    )
            else:
                resp = _response_text_for(card or {}, ds.response_text)
                rq = _recall_query_for(card or {}, c)
            rq = _repair_recall_query_for_response(card or {}, rq, c)
            if not _recall_query_available(rq, c):
                raise ValueError(
                    f"[{video_id}] unrecoverable invalid recall+response "
                    f"query for card={card_id!r} chunk={c}: no past support "
                    "exists. Rerun/fix pass3a/pass3b instead of writing a "
                    "plain response into SFT."
                )
            rr = _recall_result_for(card or {}, rollout,
                                     ds.recall_result_kind or "oracle",
                                     current_chunk=c,
                                     recall_query=rq)
            if rr.get("source") == "failure" or not rr.get("returned_chunks"):
                raise ValueError(
                    f"[{video_id}] unrecoverable invalid recall+response "
                    f"result for card={card_id!r} chunk={c}: no historical "
                    "chunks could be returned. Rerun/fix pass3a/pass3b "
                    "instead of writing a plain response into SFT."
                )
            raw.append(_recall_response_sample(
                c, _think_for_chunk(rollout, c), resp, queries_state,
                rq, rr, traj_id, card_id, sequence_type, user_input=user_input,
                recall_reason=recall_reason,
            ))
            status = (
                "answered" if c >= open_until_by_card.get(card_id, c)
                else "open"
            )
            _append_query_answer(
                queries_state, queries_idx_by_card, card_id,
                chunk_idx=c, text=resp, status=status,
            )

    # v12.12 fix (P0-1): stamp every card-bearing sample with its REAL
    # ask_chunk from placement (not the answer chunk). pass4 currently
    # infers ask_chunk from response sample's chunk_idx, which collapses
    # forward/silent_then_response timing supervision. Now pass4 can read
    # `sample["ask_chunk"]` directly and preserve the silent-then-respond
    # pattern in `questions[*].ask_chunks`.
    #
    # v12.13 fix (P0-2): also stamp `per_emit_answers` from the card's
    # gold_emits. multi_emit cards (F5 counting / PN1 narration / F7 status)
    # have multiple expected answer chunks each with their OWN gold answer
    # — e.g. F5 emits "1" at chunk_5, "2" at chunk_10, "3" at chunk_15.
    # pass4 needs this to score multi-emit per-emit (not just at canonical_ask).
    for s in raw:
        cid = s.get("card_id")
        if not cid:
            continue
        if cid in ask_chunk_by_card:
            s["ask_chunk"] = int(ask_chunk_by_card[cid])
        card = cards_map.get(cid) or {}
        if card.get("answer_style"):
            s["answer_style"] = card.get("answer_style")
        if card.get("answer_instruction"):
            s["answer_instruction"] = card.get("answer_instruction")
        placement = next((p for p in placements if p.card_id == cid), None)
        if placement is not None:
            s["per_emit_answers"] = [
                {"chunk": int(c), "value": str(value)}
                for c, (kind, value) in sorted(placement.chunk_actions.items())
                if kind == "response"
            ]
        if card.get("recall_hardened"):
            s["hardened_card"] = deepcopy(card)
        if card.get("recall_hardening_fallback"):
            s["hardening_fallback"] = {
                "mode": card.get("recall_hardening_fallback"),
                "reason": card.get("recall_hardening_fallback_reason", ""),
            }

    return raw


def save_samples(video_id: str, samples: List[Dict],
                 output_dir: Path = SAMPLES_3C_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2)
    )


def load_samples(video_id: str,
                 samples_dir: Path = SAMPLES_3C_DIR) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("3c"):
        return None
    p = samples_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
