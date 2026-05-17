"""Pass 3-C — Trajectory sample rendering (placement model-agnostic).

For each trajectory's selected placements, walks every chunk in [0, num_chunks)
and emits ONE raw SFT sample per chunk (silent / response / recall+response /
recall+silent / compress_silent), using placement/design.py as the single
source of truth for gold actions.

Pipeline contract preserved:
  generate_trajectory_samples(trajectory, cards_map, rollout, evidence,
                               client, video_id) -> List[Dict]    (async)
  save_samples / load_samples
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import random
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from thinkstream.data.agent_protocol import (
    RECALL_RETURN_CHUNKS,
    append_timestamped_image_list,
    build_assistant_content,
    format_memory_block,
    recall_time_string_for_chunks,
    select_recall_chunks_uniform,
)

from .config import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
    SAMPLES_3C_DIR,
    VISUAL_WINDOW_CHUNKS,
)
from .pass3a_cards import dict_to_card
from .pass3b_placement import _dict_to_placement
from .stable_hash import stable_mod
from .placement.design import (
    Placement,
    render_video_samples as _design_render,
)
from .placement.llm_prompts import (
    family_taxonomy,
    response_generation_prompt,
)

logger = logging.getLogger(__name__)


RECALL_SUPPORT_OVERLAP_MIN = 0.40
RECALL_HARDEN_MAX_ATTEMPTS = 3
RECALL_HARDEN_CANDIDATES_PER_ATTEMPT = 3
RECALL_HARDEN_EVIDENCE_LINES = 56


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


PASS3C_ENABLE_LLM_RESPONSE = _env_flag("THINKSTREAM_PASS3C_ENABLE_LLM_RESPONSE", False)
PASS3C_ENABLE_LLM_POST_RECALL_THINK = _env_flag(
    "THINKSTREAM_PASS3C_ENABLE_LLM_POST_RECALL_THINK",
    True,
)
PASS3C_ENABLE_RECALL_HARDENING = _env_flag("THINKSTREAM_PASS3C_ENABLE_RECALL_HARDENING", False)
PASS3C_ALLOW_POST_RECALL_THINK_FALLBACK = _env_flag(
    "THINKSTREAM_PASS3C_ALLOW_POST_RECALL_THINK_FALLBACK",
    False,
)
PASS3C_POST_RECALL_THINK_ATTEMPTS = max(
    1,
    int(os.environ.get("THINKSTREAM_PASS3C_POST_RECALL_THINK_ATTEMPTS", "3")),
)


def _use_post_recall_think_fallback(
    *,
    card: Dict,
    recall_result: Dict,
    response: str,
    video_id: str,
    chunk_idx: int,
    action: str,
) -> str:
    """Deterministic fallback for post-recall visual thoughts.

    This is deliberately narrower than falling back from a bad recall query or
    empty recall result: the historical evidence has already been validated,
    only the optional teacher-generated bridge sentence failed cleaning.
    """
    card_id = str((card or {}).get("card_id") or "")
    fallback = _post_recall_think_for(
        card or {},
        recall_result or {},
        response,
        card_id=card_id,
        chunk_idx=chunk_idx,
    )
    logger.warning(
        "[%s] 3c post-recall think using deterministic fallback for "
        "card=%s chunk=%s action=%s",
        video_id,
        card_id,
        chunk_idx,
        action,
    )
    return fallback


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

    Pass2 thinks are question-blind current-frame observations. The following
    tool_call already supervises the recall action, so do not append a generic
    "I will recall..." rationale; that template was easy for SFT models to
    overfit and reduced recall-query diversity.
    """
    return _strip_recall_action_boilerplate(visual_think)


_RECALL_ACTION_BOILERPLATE_RE = re.compile(
    r"\s*(?:"
    r"Current visible evidence is insufficient|"
    r"The active query depends on elapsed context|"
    r"A related moment may have occurred earlier|"
    r"Before answering the pending query|"
    r"The query has stayed open long enough|"
    r"The current moment is relevant, but the answer also depends|"
    r"The status question depends on an event"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_recall_action_boilerplate(text: str) -> str:
    """Remove old recall-action rationale templates from first-turn thinks."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _RECALL_ACTION_BOILERPLATE_RE.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


MC_OPTION_LETTERS = "ABCDE"
MC_OPTION_COUNTS = {2, 3, 4, 5}
_OPTION_LABEL_RE = re.compile(r"^\s*(?:\([A-E]\)|[A-E][\).:])\s*")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MC_ANSWER_STYLES = ("letter_plus_text",)
_NUMBER_WORDS_BY_DIGIT = {
    "0": {"zero"},
    "1": {"one"},
    "2": {"two"},
    "3": {"three"},
    "4": {"four"},
    "5": {"five"},
    "6": {"six"},
    "7": {"seven"},
    "8": {"eight"},
    "9": {"nine"},
    "10": {"ten"},
}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "while", "what", "which", "who", "where", "when", "how", "is",
    "are", "was", "were", "be", "been", "being", "by", "from", "as", "it",
    "this", "that", "these", "those", "into", "onto", "there", "here", "his",
    "her", "their", "its", "your", "only", "answer", "option", "text",
    "letter", "video", "scene", "frame", "frames", "question", "can", "you",
    "now", "currently", "current", "whether", "many", "much", "times",
    "count", "number", "total", "far", "already", "yet", "latest",
    "action", "perform", "performed", "specific", "type", "kind", "variety",
    "category", "did", "does", "happen", "happened", "final", "chunk",
    "chunks", "before", "after",
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


def _standalone_token_in_text(token: str, text: str) -> bool:
    token = str(token or "").strip()
    if not token:
        return False
    return re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", str(text or ""), re.IGNORECASE) is not None


def _text_leaks_answer_value(
    card: Dict,
    text: str,
    *,
    threshold: float = 0.50,
    include_option_letter: bool = False,
) -> bool:
    """True when retrieval-facing text exposes the target answer itself."""
    answer = _card_answer_text(card)
    if not answer or answer.strip().lower() == "unable to answer":
        return False
    if _answer_visible_in_text(answer, text, threshold=threshold):
        return True
    answer_n = _norm_text(answer)
    if answer_n in {"yes", "no"} and _standalone_token_in_text(answer_n, text):
        return True
    if re.fullmatch(r"\d+", answer_n or ""):
        if _standalone_token_in_text(answer_n, text):
            return True
        for word in _NUMBER_WORDS_BY_DIGIT.get(answer_n, set()):
            if _standalone_token_in_text(word, text):
                return True
    if include_option_letter and card.get("answer_form") == "multiple_choice":
        letter, _text = _mc_correct_letter_text(card)
        if letter and _standalone_token_in_text(letter, text):
            return True
    return False


def _redact_answer_value(card: Dict, text: str) -> str:
    answer = _card_answer_text(card)
    out = str(text or "")
    if not answer or answer.strip().lower() == "unable to answer":
        return out
    if len(str(answer).strip()) >= 2:
        out = re.sub(re.escape(str(answer).strip()), " ", out, flags=re.IGNORECASE)
    answer_n = _norm_text(answer)
    if re.fullmatch(r"\d+", answer_n or ""):
        out = re.sub(rf"(?<![\w]){re.escape(answer_n)}(?![\w])", "the target count", out)
        for word in _NUMBER_WORDS_BY_DIGIT.get(answer_n, set()):
            out = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", "the target count", out, flags=re.IGNORECASE)
    return out


def _redact_answer_space(card: Dict, text: str) -> str:
    """Remove target-answer and MC option surface forms from teacher context."""
    out = _redact_answer_value(card, text)
    if (card or {}).get("answer_form") == "multiple_choice":
        letter, correct_text = _mc_correct_letter_text(card)
        if letter:
            out = re.sub(rf"(?<![\w]){re.escape(letter)}(?![\w])", "the target option", out)
        if correct_text and len(correct_text.strip()) >= 2:
            out = re.sub(
                re.escape(correct_text.strip()),
                "the target option",
                out,
                flags=re.IGNORECASE,
            )
        for opt in (card or {}).get("options") or []:
            opt_text = _strip_option_label(opt)
            if len(opt_text.strip()) >= 3:
                out = re.sub(
                    re.escape(opt_text.strip()),
                    "an option candidate",
                    out,
                    flags=re.IGNORECASE,
                )
    return re.sub(r"\s+", " ", out).strip()


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
    if (
        len(options) in MC_OPTION_COUNTS
        and len(correct) == 1
        and correct in MC_OPTION_LETTERS[:len(options)]
    ):
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
    """Project-wide MC target format.

    Keep MCQ prompts and SFT targets in the same OvO-compatible form across
    data construction, SFT, RL rollout, and eval. The scorer remains liberal
    and still accepts the bare option letter, but generated supervision should
    include option text so answer tokens carry semantic grounding.
    """
    return "letter_plus_text"


def _mc_answer_instruction(style: str, options: Optional[List[str]] = None) -> str:
    return "Answer format: letter plus option text, e.g. A) option text."


def _mc_answer_text(card: Dict, fallback: str = "", style: Optional[str] = None) -> str:
    """Return the SFT target string for an MC card.

    Pass3A LLM cards may put a local evidence phrase in gold_emits while the
    actual answer is the canonical option. Use correct_option/options first so
    the response answers the question, then fall back to canonical_answer.

    MC supervision is always rendered as letter plus option text. Legacy
    letter-only/text-only metadata is ignored here; old rows are normalized at
    rendering time by the shared query protocol.
    """
    correct, text = _mc_correct_letter_text(card, fallback)
    if correct:
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


def _recall_query_interval(query: object) -> Optional[tuple]:
    """Return ``(start_time, end_time)`` floats from a recall-query dict."""
    if not isinstance(query, dict):
        return None
    try:
        start = float(query.get("start_time"))
        end = float(query.get("end_time"))
    except (TypeError, ValueError):
        return None
    return start, end


def _valid_recall_query(query: Dict) -> bool:
    pair = _recall_query_interval(query or {})
    return pair is not None and pair[0] >= 0 and pair[1] >= pair[0]


def _recall_range_end(query: object) -> Optional[float]:
    pair = _recall_query_interval(query)
    return pair[1] if pair is not None else None


def _recall_query_available(query: Dict, current_chunk: int) -> bool:
    """Recall queries may only search observations strictly before now."""
    if not _valid_recall_query(query):
        return False
    end = _recall_range_end(query or {})
    return end is not None and end < current_chunk * AGENT_CHUNK_SEC


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


def _grounding_recall_args_before(card: Dict, current_chunk: int) -> Dict:
    """Return recall tool args for grounding evidence strictly before now."""
    grounding = _support_chunks_before(card, current_chunk)
    if not grounding:
        return {}
    start_time = int(min(grounding) * AGENT_CHUNK_SEC)
    end_time = int(max(grounding) * AGENT_CHUNK_SEC)
    return {"start_time": start_time, "end_time": end_time}


def _hld_recall_query_for(card: Dict, current_chunk: int) -> Dict:
    """Recall request for HLD/Unable cases.

    Recall is time-range only. HLD recall is an evidence check over older frames,
    not a lexical search for the missing target. Unlike answerable recall,
    HLD must check all previous context because the gold target is absence.
    """
    if int(current_chunk) <= 0:
        return {}
    return {
        "start_time": 0,
        "end_time": int((int(current_chunk) - 1) * AGENT_CHUNK_SEC),
    }


def _recall_query_for(card: Dict, current_chunk: int) -> Dict:
    """Build the time-range-only recall request for pass3c."""
    if _is_unanswerable_card(card):
        return _hld_recall_query_for(card, current_chunk)
    return _grounding_recall_args_before(card, current_chunk)


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
    if _recall_query_available(query, current_chunk):
        return {
            "start_time": (query or {}).get("start_time"),
            "end_time": (query or {}).get("end_time"),
        }
    repaired = _recall_query_for(card, current_chunk)
    return repaired if _recall_query_available(repaired, current_chunk) else {}


def _recall_wait_query_for(card: Dict, chunk_idx: int) -> Dict:
    """Recall query for forward wait-state samples.

    This deliberately ignores card.recall_query / grounding_frames because
    those point to the future answer evidence. At a real streaming timestep
    the student cannot know that future range. The query searches only history
    before the current 8s visual window. The tool still returns only the
    canonical 4s recall payload; an empty result teaches "keep waiting", not
    "answer from future".
    """
    visual_start_chunk = max(
        0,
        int(chunk_idx) - int(VISUAL_WINDOW_CHUNKS) + 1,
    )
    return (
        {
            "start_time": 0,
            "end_time": int((visual_start_chunk - 1) * AGENT_CHUNK_SEC),
        }
        if visual_start_chunk > 0
        else {}
    )


def _recall_archive_before(rollout: Dict, current_chunk: Optional[int]) -> List[Dict]:
    archive: List[Dict] = []
    for t in (rollout or {}).get("thinks", []):
        try:
            ci = int(t.get("chunk_idx", t.get("chunk", -1)))
        except (AttributeError, TypeError, ValueError):
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
    lines: List[str] = []
    for c in chunks:
        item = by_chunk.get(int(c))
        if not item:
            continue
        lines.append(f"[{item.get('time', '')}] {item.get('text', '')}")
    return "\n".join(lines)


def _recall_chunks_for_request(
    rollout: Dict,
    recall_query: Optional[Dict],
    current_chunk: Optional[int],
) -> List[int]:
    """Return time-range-only recall chunks, uniformly sampled to 8 frames."""
    tr = _recall_query_interval(recall_query or {})
    if tr is None:
        return []
    start_s, end_s = tr
    if end_s < start_s:
        return []
    candidates: List[int] = []
    for item in _recall_archive_before(rollout or {}, current_chunk):
        try:
            ci = int(item.get("chunk"))
        except (TypeError, ValueError):
            continue
        c_time = ci * float(AGENT_CHUNK_SEC)
        if start_s <= c_time <= end_s:
            candidates.append(ci)
    return select_recall_chunks_uniform(
        candidates,
        max_chunks=RECALL_RETURN_CHUNKS,
    )


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
    original_support = set(_support_chunks_before(card, int(current_chunk)))
    memory_tokens = set(_tokens(memory_text)) if memory_text else set()
    scored = []
    for ci, cap in evidence_by_chunk.items():
        if ci < 0 or int(current_chunk) - int(ci) <= VISUAL_WINDOW_CHUNKS:
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
- recall uses only start_time/end_time. Do not generate keyword queries; the
  tool will uniformly sample returned frames from the requested historical
  interval.
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
    "grounding_frames": [int, ...]
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
            "start_time": int(min(grounding) * AGENT_CHUNK_SEC),
            "end_time": int(max(grounding) * AGENT_CHUNK_SEC),
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
        if (
            len(options) not in MC_OPTION_COUNTS
            or len(correct) != 1
            or correct not in MC_OPTION_LETTERS[:len(options)]
        ):
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
    if not _valid_recall_query(rq):
        rq = default_recall_query()
    else:
        rq = {
            "start_time": rq.get("start_time"),
            "end_time": rq.get("end_time"),
        }
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
    grounding = _support_chunks_before(card, int(current_chunk))
    if not grounding:
        return False, "no_past_grounding"
    not_historical = [
        c for c in grounding
        if int(current_chunk) - int(c) <= VISUAL_WINDOW_CHUNKS
    ]
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
    return True, "pass"


def _needs_recall_hardening(card: Dict, memory_text: str) -> bool:
    """Memory text overlap no longer makes a recall slot invalid.

    Compact memory is a lossy state prior; the model still needs the recall
    tool to inspect old visual frames once support is outside the 8s KV window.
    Keep this shim for older call sites, but do not rewrite questions/answers
    based on memory-word visibility.
    """
    return False


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
    """Build a recall result from the explicit time range.

    ``noise_kind`` is retained for placement metadata, but retrieval itself no
    longer injects distractors: the tool returns uniform chunks from the
    requested window only.
    """
    if current_chunk is None:
        grounding = _support_chunks(card)
    else:
        grounding = _support_chunks_before(card, int(current_chunk))
    absence_check = _is_unanswerable_card(card)
    archive = _recall_archive_before(rollout, current_chunk)
    if noise_kind == "not_yet":
        chunks = _recall_chunks_for_request(rollout, recall_query or {}, current_chunk)
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
            "text_content": _archive_text_for_chunks(archive, chunks) or (
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
    chunks = _recall_chunks_for_request(rollout, recall_query or {}, current_chunk)
    text_content = _archive_text_for_chunks(archive, chunks)

    chunks = select_recall_chunks_uniform(chunks)
    text_content = _archive_text_for_chunks(archive, chunks)
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
    """Plain silent sample.

    recall_silent uses _recall_silent_multiturn_sample so the model sees the
    recall tool call and the no-match result before staying silent.
    """
    if sample_subtype == "recall+silent":
        raise ValueError(
            "recall+silent must be rendered by _recall_silent_multiturn_sample, "
            "not _silent_sample."
        )
    sample_type = "silent"
    output_text = build_assistant_content(
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


_COMPACT_M_LINE_RE = re.compile(
    r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>',
    re.DOTALL | re.IGNORECASE,
)


def _compact_safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip(), quote=False)


def _compact_range_from_item(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    chunks = item.get("source_chunks") or item.get("chunks") or []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) and chunks:
        try:
            vals = sorted(int(c) for c in chunks)
            return vals[0], vals[-1]
        except (TypeError, ValueError):
            pass
    tr = item.get("time_range") or item.get("time") or []
    if isinstance(tr, Sequence) and not isinstance(tr, (str, bytes)) and len(tr) >= 2:
        try:
            start = int(tr[0])
            end = int(tr[1])
        except (TypeError, ValueError):
            return None
        if end > start:
            # Legacy pass2 stores compressed ranges as half-open chunk ranges.
            end -= 1
        return start, max(start, end)
    return None


def _compact_segments_to_mlines(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        rng = _compact_range_from_item(seg)
        text = _compact_safe_text(seg.get("text") or seg.get("summary") or seg.get("think"))
        if rng is None or not text:
            continue
        start, end = rng
        lines.append(f'<m t="{start}-{end}">{text}</m>')
    return "\n".join(lines)


def _compact_chunks_from_event(compress_event: Dict[str, Any]) -> List[int]:
    summary = compress_event.get("summary") or {}
    for key in (
        "compressed_raw_think_chunks",
        "compressed_source_chunks",
        "compressed_thinks_chunks",
        "selected_indices",
    ):
        vals = compress_event.get(key) or []
        if vals:
            try:
                return sorted(set(int(c) for c in vals))
            except (TypeError, ValueError):
                pass
    vals = summary.get("source_chunks") or []
    if vals:
        try:
            return sorted(set(int(c) for c in vals))
        except (TypeError, ValueError):
            pass
    tr = summary.get("time_range") or []
    if isinstance(tr, Sequence) and not isinstance(tr, (str, bytes)) and len(tr) >= 2:
        try:
            start, end = int(tr[0]), int(tr[1])
            return list(range(start, max(start, end)))
        except (TypeError, ValueError):
            pass
    return []


def _compact_snapshot_for_event(
    rollout: Dict[str, Any],
    trigger_chunk: int,
    *,
    post: bool,
) -> Dict[str, Any]:
    snapshots = rollout.get("snapshots") or {}
    keys = [trigger_chunk + 1, trigger_chunk] if post else [trigger_chunk, trigger_chunk - 1]
    for key in keys:
        snap = snapshots.get(key) or snapshots.get(str(key))
        if isinstance(snap, dict):
            return snap
    if post and isinstance(rollout.get("final_memory"), dict):
        return rollout["final_memory"]
    return {}


def _compact_event_summary_segment(compress_event: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(compress_event.get("summary") or {})
    chunks = _compact_chunks_from_event(compress_event)
    if chunks:
        summary["source_chunks"] = chunks
        summary["time_range"] = [min(chunks), max(chunks) + 1]
    return summary


def _compact_post_memory_text(
    rollout: Dict[str, Any],
    compress_event: Dict[str, Any],
    trigger_chunk: int,
) -> str:
    post_snapshot = _compact_snapshot_for_event(rollout, trigger_chunk, post=True)
    mem_text = _compact_segments_to_mlines(post_snapshot.get("compressed_segments") or [])
    if mem_text:
        return mem_text

    pre_snapshot = _compact_snapshot_for_event(rollout, trigger_chunk, post=False)
    segments = list(pre_snapshot.get("compressed_segments") or [])
    segments.append(_compact_event_summary_segment(compress_event))
    return _compact_segments_to_mlines(segments)


def _compact_caption_block(rollout: Dict[str, Any], chunks: Sequence[int]) -> str:
    thinks: Dict[int, str] = {}
    for item in rollout.get("thinks") or []:
        try:
            chunk = int(item.get("chunk_idx", item.get("chunk")))
        except (AttributeError, TypeError, ValueError):
            continue
        text = str(item.get("think") or item.get("text") or "").strip()
        if text:
            thinks[chunk] = text
    lines = []
    for chunk in sorted(set(int(c) for c in chunks)):
        text = _compact_safe_text(thinks.get(chunk, ""))
        if text:
            lines.append(f'  <c t="{chunk}">{text}</c>')
    return "\n".join(lines)


def _compact_memory_update_input(
    rollout: Dict[str, Any],
    compress_event: Dict[str, Any],
    trigger_chunk: int,
) -> str:
    chunks = _compact_chunks_from_event(compress_event)
    if chunks:
        start, end = min(chunks), max(chunks)
    else:
        start = end = int(trigger_chunk)
    pre_snapshot = _compact_snapshot_for_event(rollout, trigger_chunk, post=False)
    old_lines = _compact_segments_to_mlines(pre_snapshot.get("compressed_segments") or [])
    old_body = f"  {old_lines.replace(chr(10), chr(10) + '  ')}" if old_lines else ""
    caption_body = _compact_caption_block(rollout, chunks) or "  (no source captions found in pass2 rollout)"
    return (
        "OLD_MEMORY:\n"
        "<MEM>\n"
        f"{old_body}\n"
        "</MEM>\n\n"
        "NEW_CAPTIONS:\n"
        "<NEW_CAPTIONS>\n"
        f"{caption_body}\n"
        "</NEW_CAPTIONS>\n\n"
        f"Covered latest span: t={start}-{end}\n"
        "Coverage check: preserve useful OLD_MEMORY and cover the listed "
        "NEW_CAPTIONS using their real timestamps.\n"
        "Return only compact-memory XML lines:\n"
        '<m t="start-end">one concise event or state.</m>\n'
        "Do not output NEW_MEMORY:, markdown, prose, analysis, or any text "
        "outside the <m> lines."
    )


def _compact_entries_from_mlines(mem_text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for match in _COMPACT_M_LINE_RE.finditer(mem_text or ""):
        start = int(match.group(1))
        end = int(match.group(2) if match.group(2) is not None else match.group(1))
        text = html.unescape(re.sub(r"\s+", " ", match.group(3)).strip())
        if text:
            entries.append({
                "t": f"{start}-{end}",
                "time_range": [start, end],
                "source_chunks": list(range(start, end + 1)),
                "text": text,
            })
    return entries


def _legacy_compact_sample_from_rollout(
    chunk_idx: int,
    queries: List[Dict],
    trajectory_id: str,
    compress_event: Dict,
    rollout: Optional[Dict],
) -> Optional[Dict]:
    if not rollout:
        return None
    try:
        trigger_chunk = int(compress_event.get("trigger_chunk", chunk_idx))
    except (TypeError, ValueError):
        trigger_chunk = int(chunk_idx)
    mem_text = _compact_post_memory_text(rollout, compress_event, trigger_chunk).strip()
    if not mem_text:
        return None
    chunks = sorted({
        chunk
        for entry in _compact_entries_from_mlines(mem_text)
        for chunk in entry.get("source_chunks", [])
    })
    update_input = _compact_memory_update_input(rollout, compress_event, trigger_chunk)
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "compress",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": "",
        "sequence_type": "compress_event",
        "action": "compress",
        "output": mem_text,
        "queries": deepcopy(queries),
        "user_input": "",
        "memory_update_input": update_input,
        "recall_result": None,
        "base_role": "compress_action",
        "inter_chunk": True,
        "gold_caption": mem_text,
        "gold_compress_chunks": chunks or _compact_chunks_from_event(compress_event),
        "gold_memory_entries": _compact_entries_from_mlines(mem_text),
        "memory_update_mode": "compact_mem",
    }


def _compress_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    trajectory_id: str, compress_event: Dict, *, rollout: Optional[Dict] = None,
) -> Dict:
    """Compact-memory update sample from a pass2 compression event.

    New data treats memory compaction as a standalone text-only system turn:
    user(old memory + recent captions) -> assistant(bare <m> lines).
    It is no longer a streaming tool_call or a <stage:compress> turn.
    """
    summary = compress_event.get("summary", {}) or {}
    tr = summary.get("time_range", [])
    chunks = sorted(int(c) for c in
        (summary.get("source_chunks")
         or compress_event.get("compressed_source_chunks")
         or compress_event.get("compressed_thinks_chunks")
         or []))
    mem_text = (summary.get("text") or "").strip()
    if summary.get("compact_memory_update") or mem_text.startswith("<MEM>") or summary.get("entries"):
        m_lines = re.findall(
            r'<m\s+t="[^"]+"\s*>.*?</m>',
            mem_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m_lines:
            mem_text = "\n".join(line.strip() for line in m_lines)
        else:
            lines = []
            for entry in summary.get("entries") or []:
                etr = entry.get("time_range") or []
                if not (isinstance(etr, list) and len(etr) == 2):
                    continue
                lines.append(
                    f'  <m t="{int(etr[0])}-{int(etr[1])}">{str(entry.get("text", "")).strip()}</m>'
                )
            mem_text = "\n".join(lines)
        return {
            "chunk_idx": chunk_idx,
            "sample_type": "compress",
            "prompt_type": "SYSTEM_PROMPT",
            "trajectory_id": trajectory_id,
            "card_id": "",
            "sequence_type": "compress_event",
            "action": "compress",
            "output": mem_text,
            "queries": deepcopy(queries),
            "user_input": "",
            "memory_update_input": str(compress_event.get("memory_update_input") or "").strip(),
            "recall_result": None,
            "base_role": "compress_action",
            "inter_chunk": True,
            "gold_caption": mem_text,
            "gold_compress_chunks": chunks,
            "gold_memory_entries": deepcopy(summary.get("entries") or []),
            "memory_update_mode": "compact_mem",
        }
    legacy_compact = _legacy_compact_sample_from_rollout(
        chunk_idx,
        queries,
        trajectory_id,
        compress_event,
        rollout,
    )
    if legacy_compact is not None:
        return legacy_compact
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
    output_text = build_assistant_content(
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
        "user_input": "",
        "recall_result": None,
        "base_role": "compress_action",
        "inter_chunk": True,
        # ICAE-style aux target: original gold caption / source chunks
        "gold_caption": summary.get("text", ""),
        "gold_compress_chunks": chunks,
    }


def _response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "",
) -> Dict:
    output_text = build_assistant_content(
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


def _recall_text_hint(card: Dict, recall_result: Dict, *, max_words: int = 18) -> str:
    """Extract a short non-answer visual anchor from internal recall text."""
    text = str(
        (recall_result or {}).get("text_content")
        or (recall_result or {}).get("text")
        or ""
    )
    if not text:
        return ""
    text = re.sub(r"\[[^\]]{0,40}\]", " ", text)
    text = re.sub(
        r"\bt\s*=\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*s?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = _redact_answer_value(card, text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _text_leaks_answer_value(card, text, threshold=0.45):
        return ""
    candidates = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    for cand in candidates:
        cand = re.sub(r"\s+", " ", cand).strip(" .")
        if len(cand) < 12:
            continue
        if _text_leaks_answer_value(card, cand, threshold=0.45):
            continue
        words = cand.split()
        return " ".join(words[:max_words]).strip(" .,;:")
    return ""


_POST_RECALL_META_DECISION_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|option|choice)\s*"
    r"(?:is|would be|should be|:)\s*(?:option\s+)?[A-E]\b|"
    r"\b(?:choose|select|pick)\s+(?:option\s+)?[A-E]\b",
    re.IGNORECASE,
)


def _capitalize_sentence_start(text: str) -> str:
    for idx, ch in enumerate(text):
        if ch.isalpha():
            return text[:idx] + ch.upper() + text[idx + 1:]
    return text


_POST_RECALL_SOURCE_SUBJECT_RE = re.compile(
    r"^\s*(?:the\s+)?(?:recalled|retrieved|returned|old|historical|earlier)\s+"
    r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)\s+"
    r"(?:show|shows|display|displays|reveal|reveals|contain|contains|"
    r"depict|depicts|include|includes|indicate|indicates)\b|"
    r"^\s*(?:the\s+)?frames?\s+"
    r"(?:show|shows|display|displays|reveal|reveals|contain|contains|"
    r"depict|depicts|include|includes|indicate|indicates)\b|"
    r"\b(?:tool|recall)\s+(?:result|response|output)\b",
    re.IGNORECASE,
)

_POST_RECALL_SOURCE_TERM_RE = re.compile(
    r"\b(?:recalled|retrieved|returned|old|historical)\s+"
    r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)\b|"
    r"\brecall(?:ed|ing)?\b",
    re.IGNORECASE,
)


def _strip_safe_post_recall_source_wrappers(text: str) -> str:
    """Remove only wrappers that leave a complete teacher sentence intact."""
    out = str(text or "").strip()
    prefix_patterns = [
        r"^\s*(?:in|within|from)\s+(?:the\s+)?"
        r"(?:recalled|retrieved|returned|old|historical|earlier)\s+"
        r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)\s*,\s*",
        r"^\s*after\s+(?:checking|viewing|seeing)\s+(?:the\s+)?"
        r"(?:recalled|retrieved|returned|old|historical|earlier)\s+"
        r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)\s*,\s*",
    ]
    suffix_patterns = [
        r"\s+(?:in|within|from)\s+(?:the\s+)?"
        r"(?:recalled|retrieved|returned|old|historical|earlier)\s+"
        r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)[.!?]?\s*$",
        r"\s+as\s+seen\s+in\s+(?:the\s+)?"
        r"(?:recalled|retrieved|returned|old|historical|earlier)\s+"
        r"(?:frames?|window|chunks?|visuals?|visual\s+window|evidence)[.!?]?\s*$",
    ]
    for pattern in prefix_patterns:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip(" ,;:")
    for pattern in suffix_patterns:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip(" ,;:")
    out = re.sub(r"\s+", " ", out).strip(" ,;:")
    return _capitalize_sentence_start(out.strip())


def _has_post_recall_source_framing(text: str) -> bool:
    out = str(text or "")
    return bool(
        _POST_RECALL_SOURCE_SUBJECT_RE.search(out)
        or _POST_RECALL_SOURCE_TERM_RE.search(out)
    )


def _extract_jsonish_post_recall_text(text: str) -> str:
    """Best-effort salvage for teachers that wrap the sentence in JSON."""
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "{[":
        return stripped
    try:
        parsed = json.loads(stripped)
    except Exception:
        return stripped
    stack = [parsed]
    preferred = {
        "sentence",
        "observation",
        "visual_observation",
        "visual_fact",
        "thought",
        "think",
        "text",
    }
    fallback = ""
    while stack:
        item = stack.pop(0)
        if isinstance(item, dict):
            for key in preferred:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in item.values():
                if isinstance(value, str) and value.strip() and not fallback:
                    fallback = value.strip()
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and item.strip() and not fallback:
            fallback = item.strip()
    return fallback or stripped


def _clean_post_recall_think(raw: str, card: Dict) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("```json", "```")
    if text.startswith("```") and text.count("```") >= 2:
        text = text.split("```", 2)[1].strip()
    text = _extract_jsonish_post_recall_text(text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?response>|</?answer>|</?silent>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<MEM>.*?</MEM>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*(?:thought|analysis|observation|result)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:Response|Silence|answer|response|silent)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \"'`")
    if not text:
        return ""
    if re.search(r"<tool_call|</?MEM\b", text, re.IGNORECASE):
        return ""
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    first = _strip_safe_post_recall_source_wrappers(first)
    if not first:
        return ""
    if _has_post_recall_source_framing(first):
        return ""
    if _POST_RECALL_META_DECISION_RE.search(first):
        return ""
    if not re.search(r"[.!?]$", first):
        first += "."
    return first


def _post_recall_think_prompt(
    card: Dict,
    recall_query: Dict,
    recall_result: Dict,
    *,
    action: str,
    template_id: int = 0,
    retry_feedback: str = "",
    response: str = "",
) -> str:
    question = str((card or {}).get("question") or "").strip()
    options = list((card or {}).get("options") or [])
    options_doc = "\n".join(str(opt).strip() for opt in options if str(opt).strip())
    if not options_doc:
        options_doc = "None"
    recall_window = {
        "start_time": (recall_query or {}).get("start_time"),
        "end_time": (recall_query or {}).get("end_time"),
    }
    retrieved_text = str(
        (recall_result or {}).get("text_content")
        or (recall_result or {}).get("text")
        or ""
    )
    chunks = ", ".join(str(c) for c in (recall_result or {}).get("returned_chunks") or [])
    if not retrieved_text:
        retrieved_text = "No textual summary is available; only visual segment metadata is available."
    if action == "response":
        target_text = _card_answer_text(card or {})
        target_doc = (
            "\nTeacher-only response target for alignment: "
            f"{str(response or '').strip() or target_text}"
        )
        if target_text and target_text != str(response or "").strip():
            target_doc += f" ({target_text})"
        target_doc += (
            "\nUse the target only to choose which visual fact to "
            "describe; do not write a meta answer decision."
        )
        mode_rule = (
            "The next assistant turn will answer separately. Your sentence should state "
            "the visual fact that supports that answer."
        )
    else:
        target_doc = ""
        mode_rule = (
            "The next assistant turn will stay silent because this old window does not "
            "settle the active query."
        )
    template = int(template_id or 0) % 3
    if template == 1:
        task_doc = """Write the assistant's direct local visual observation.

This sentence should sound like a concise visual note, not a generic
explanation of which tool was used."""
        style_doc = """- Mention a visible actor/object/action/context from the visual content.
- If the content is insufficient, state the missing visual fact directly."""
    elif template == 2:
        task_doc = """Write a one-sentence bridge from visual observation to the next action.

The bridge should summarize the visual information objectively. The
final answer will be inserted separately after the response token."""
        style_doc = """- Use concrete scene details from the visual notes when available.
- For response turns, include the answer-relevant visual fact if it is visible."""
    else:
        task_doc = """Write one short visual thought for a streaming video agent."""
        style_doc = """- Objectively describe what is visible or whether the needed fact is absent.
- Use a natural sentence, not a repeated stock phrase."""

    retry_doc = ""
    if retry_feedback:
        retry_doc = f"\nPrevious attempt was invalid: {retry_feedback}\n"

    return f"""{task_doc}

Active question: {question}
Options visible to the student, for identifying the relevant visual fact: {options_doc}
Internal time range: {recall_window}
Internal chunk IDs: {chunks or "none"}
Visual notes: {retrieved_text[:900]}
Attached images, when present, are the source of truth for the visual content.
Action after this thought: {action}
{target_doc}
{retry_doc}

{mode_rule}

Rules:
- Output exactly one short English sentence, 8-36 words.
{style_doc}
- Describe visible actors, objects, actions, scene relations, OCR text, counts,
  colors, or states inside the provided visual content when they are relevant.
- If the visual content contains the fact needed for the answer, mention that
  visual fact naturally. Do not avoid it.
- For response turns, prioritize the visual fact that is sufficient to answer
  the active question, not merely background from the same window.
- If the question asks what/which tool, action, text, color, count, or state,
  name that tool, action, text, color, count, or state when visible.
- For response turns, do not list absent distractor options; use the visual
  segment that contains the positive answering fact.
- Do not write a meta decision such as "the answer is A" or "choose option B";
  the separate response field will carry the answer format.
- Do not mention the retrieval/tool source. Avoid words or phrases like
  "recall", "recalled frames", "retrieved window", "returned chunks",
  "tool result", "old frames", or "the frames show".
- Start with the visual subject itself, such as "The person...", "The title
  card...", or "At 85 seconds...".
- Do not include protocol tags, tool calls, JSON, or multiple sentences.
- When the next action is response, do not say the content lacks, omits, or
  fails to specify the needed fact.
- Do not mention "I think" or "I notice".

Output the sentence only:"""


def _frame_paths_for_chunks(
    all_frame_paths: Optional[List[str]],
    chunks: List[int],
) -> List[str]:
    if not all_frame_paths or not chunks:
        return []
    from .pass1a_evidence import get_chunk_frame_paths

    paths: List[str] = []
    for chunk in chunks:
        paths.extend(get_chunk_frame_paths(all_frame_paths, int(chunk)))
    return paths


def _post_recall_teacher_messages(
    prompt: str,
    recall_result: Dict,
    *,
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    chunks = select_recall_chunks_uniform((recall_result or {}).get("returned_chunks") or [])
    frame_paths = _frame_paths_for_chunks(all_frame_paths, chunks)
    if not frame_paths:
        return [{"role": "user", "content": prompt}]
    from scripts.agent_data_pipeline.vllm_client import encode_image_base64

    content: List[Dict] = [{"type": "text", "text": prompt}]
    for chunk in chunks:
        chunk_paths = _frame_paths_for_chunks(all_frame_paths, [int(chunk)])
        if not chunk_paths:
            continue
        append_timestamped_image_list(
            content,
            chunk_paths,
            fps=float(FRAMES_PER_CHUNK / float(AGENT_CHUNK_SEC)),
            start_frame_index=int(chunk) * FRAMES_PER_CHUNK,
            total_num_frames=(int(chunk) + 1) * FRAMES_PER_CHUNK,
            context_label=f"visual segment c{int(chunk)}",
            image_key="image_url",
            image_url_encoder=encode_image_base64,
        )
    return [{"role": "user", "content": content}]


async def _post_recall_think_via_llm(
    card: Dict,
    recall_query: Dict,
    recall_result: Dict,
    response: str,
    client,
    video_id: str,
    chunk_idx: int,
    *,
    action: str,
    all_frame_paths: Optional[List[str]] = None,
) -> str:
    if client is None or not PASS3C_ENABLE_LLM_POST_RECALL_THINK:
        return ""
    cfg = PASS_CONFIG.get("pass3c_post_recall_think", PASS_CONFIG.get("pass3c", {}))
    attempts = max(
        1,
        int(cfg.get("attempts", PASS3C_POST_RECALL_THINK_ATTEMPTS)),
    )
    last_raw = ""
    retry_feedback = ""
    for attempt in range(attempts):
        prompt = _post_recall_think_prompt(
            card,
            recall_query,
            recall_result,
            action=action,
            retry_feedback=retry_feedback,
            response=response,
            template_id=stable_mod(
                video_id,
                str((card or {}).get("card_id", "")),
                chunk_idx,
                action,
                attempt,
                "post_recall_think_prompt",
                modulo=3,
            ),
        )
        try:
            raw = await client._call_one(
                messages=_post_recall_teacher_messages(
                    prompt,
                    recall_result,
                    all_frame_paths=all_frame_paths,
                ),
                max_tokens=int(cfg.get("max_tokens", 128)),
                temperature=float(cfg.get("temperature", 0.7)),
                request_id=(
                    f"{video_id}_3c_postrecall_"
                    f"{(card or {}).get('card_id','?')}_{chunk_idx}_{action}_try{attempt + 1}"
                ),
                enable_thinking=cfg.get("thinking", False),
            )
        except Exception as exc:
            logger.warning(f"[{video_id}] 3c post-recall think LLM failed: {exc}")
            retry_feedback = "the teacher request failed; return one grounded sentence only."
            continue
        last_raw = raw or ""
        cleaned = _clean_post_recall_think(last_raw, card or {})
        if not cleaned:
            retry_feedback = (
                "output must be a visual observation sentence, not a tool call, "
                "empty text, source-framing phrase, or option-letter answer decision."
            )
            logger.warning(
                "[%s] 3c post-recall think unusable for card=%s chunk=%s "
                "action=%s attempt=%s raw=%r",
                video_id,
                (card or {}).get("card_id", ""),
                chunk_idx,
                action,
                attempt + 1,
                last_raw[:180],
            )
            continue
        return cleaned
    logger.warning(
        "[%s] 3c post-recall think exhausted teacher attempts for card=%s "
        "chunk=%s action=%s raw=%r",
        video_id,
        (card or {}).get("card_id", ""),
        chunk_idx,
        action,
        last_raw[:180],
    )
    return ""


def _post_recall_think_for(
    card: Dict,
    recall_result: Dict,
    response: str,
    *,
    card_id: str,
    chunk_idx: int,
) -> str:
    """Deterministic but varied second-turn think after recall.

    Keep the text grounded in the action type instead of reusing one global
    sentence across every recall sample. Do not quote the answer directly here;
    the answer belongs in the response field.
    """
    af = str((card or {}).get("answer_form") or "")
    family = str((card or {}).get("family") or "")
    hint = _recall_text_hint(card or {}, recall_result)
    if hint:
        fact = _capitalize_sentence_start(hint).rstrip(" .")
        if af == "number":
            variants = [
                f"{fact}, linking the count to earlier visible events.",
                f"{fact}, keeping the numeric check tied to the observed action.",
                f"{fact}, providing the occurrence context for this count.",
            ]
        elif af == "binary":
            variants = [
                f"{fact}, providing the visual context for the status check.",
                f"{fact}, keeping the binary check tied to observed details.",
                f"{fact}, anchoring the status check to the scene.",
            ]
        elif af == "multiple_choice":
            variants = [
                f"{fact}, providing the visual detail needed for the comparison.",
                f"{fact}, grounding the comparison in the observed scene.",
                f"{fact}, giving the scene detail behind the response.",
            ]
        else:
            variants = [
                f"{fact}, providing the relevant visual context.",
                f"{fact}, anchoring the response to the observed moment.",
                f"{fact}, giving the scene context for this turn.",
            ]
        idx = stable_mod(f"{card_id}:{chunk_idx}", "post_recall_hint", modulo=len(variants))
        return variants[idx].strip()
    if str(response or "").strip().lower() == "unable to answer" or family == "HLD1":
        variants = [
            "The needed visual fact is absent from the checked scene.",
            "The checked scene keeps the absence case tied to visible content.",
            "The scene supports the unanswerable case without guessing from memory.",
        ]
    elif af == "multiple_choice":
        variants = [
            "The prior scene provides the visible detail needed for the comparison.",
            "The observed scene ties the comparison to a concrete visual moment.",
            "The scene detail anchors the response to visible content.",
        ]
    elif af == "binary":
        variants = [
            "The prior scene provides the visual context for the binary status check.",
            "The observed scene ties the status check to a concrete visual moment.",
            "The visible detail keeps the binary check grounded in the scene.",
        ]
    elif af == "number":
        variants = [
            "The scene provides the repeated-event context for the count.",
            "The visible occurrences tie the numeric check to observed events.",
            "The count stays grounded in visible repeated actions.",
        ]
    else:
        variants = [
            "The prior scene provides the relevant visual context for this turn.",
            "The observed detail anchors the response to a concrete visual moment.",
            "The response stays grounded in visible scene details.",
        ]
    idx = stable_mod(f"{card_id}:{chunk_idx}", "post_recall_think", modulo=len(variants))
    return variants[idx].strip()


def _recall_response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    recall_query: Dict, recall_result: Dict,
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "", recall_reason: str = "", card: Optional[Dict] = None,
    post_recall_think: str = "", allow_template_fallback: bool = True,
) -> Dict:
    """Multi-turn recall sample (v12 protocol).

    sample_type='recall' triggers pass5's two-turn render:
      assistant → tool_call(recall_query)
      tool      → recall_result + recalled_frames
      assistant → final answer
    """
    turn1 = build_assistant_content(
        think=_recall_action_think(
            think, final_action="response", reason=recall_reason
        ),
        kind="recall",
        recall_query=recall_query,
    )
    turn2_think = str(post_recall_think or "").strip()
    if not turn2_think and allow_template_fallback:
        turn2_think = _post_recall_think_for(
            card or {},
            recall_result,
            response,
            card_id=card_id,
            chunk_idx=chunk_idx,
        )
    turn2 = build_assistant_content(
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
    user_input: str = "", recall_reason: str = "", post_recall_think: str = "",
    allow_template_fallback: bool = True,
) -> Dict:
    """Recall followed by an empty answer while the query remains open.

    Production use is the forward/waiting case: after a question is asked,
    recall may show that the answer has not appeared in history yet. The
    model should emit </Silence> for this chunk, leave the query
    pending, and answer at a later response chunk.

    SFT rendering (pass5 shape B variant):
      assistant → tool_call(recall_query)        ← turn1: model attempts recall
      tool      → recalled_frames + recall_result ← system returns historical frames
      assistant → think + </Silence>             ← turn2: wait, query stays open
    """
    turn1 = build_assistant_content(
        think=_recall_action_think(
            think, final_action="silent", reason=recall_reason
        ),
        kind="recall",
        recall_query=recall_query,
    )
    turn2_think = str(post_recall_think or "").strip()
    if not turn2_think and allow_template_fallback:
        turn2_think = (
            "The checked scene does not contain the needed past evidence yet, so "
            "the query should stay open and the answer should remain empty."
        )
    turn2 = build_assistant_content(
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
    client=None,                     # optional LLM for descriptive response / post-recall think
    video_id: str = "",
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Render one trajectory's placements into raw per-chunk samples.

    Output samples carry every field render_samples.render_sample() needs
    (chunk_idx, sample_type, action, output, queries, user_input,
    recall_result, sequence_type, card_id, trajectory_id, optional
    v12_assistant_turn_*). Non-descriptive answers stay deterministic;
    descriptive answer text and post-recall visual notes can use `client`
    when provided. Recall itself is time-range-only. render_samples then adds
    the `input` dict + metadata for pass3e/4/5.
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
    # Map trigger_chunk -> full event so we can inject standalone compact-memory samples.
    compress_event_by_chunk: Dict[int, Dict] = {
        int(e.get("trigger_chunk", -1)): e
        for e in rollout.get("compression_events", [])
        if e.get("trigger_chunk", -1) >= 0
    }

    # Run design's gold-action pipeline (handles dense timeline silence,
    # compress_silent, priority resolution).
    for cid, card in cards_map.items():
        if (card or {}).get("answer_form") == "multiple_choice":
            style = _mc_answer_style_for_card(card, video_id)
            card["answer_style"] = style
            card["answer_instruction"] = _mc_answer_instruction(
                style, options=card.get("options") or []
            )

    if PASS3C_ENABLE_RECALL_HARDENING:
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
        # Add queries that became active by this chunk. Compact-memory update
        # happens before processing chunk c, so a question asked at c must not
        # be visible to the compact update row.
        query_activation_limit = c - 1 if ds.sample_kind == "compress_silent" else c
        for p in placements_sorted:
            if p.ask_chunk <= query_activation_limit and p.card_id not in queries_idx_by_card:
                card = cards_map.get(p.card_id) or {}
                queries_idx_by_card[p.card_id] = len(queries_state)
                # Include options + answer_form/answer_style so
                # format_queries_block can render choices and the canonical
                # Answer format line for active queries (forward responses fire
                # AFTER ask, with no fresh user_input — model sees only the
                # active-query block).
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
        # Options and output protocol live ONLY in the query-state block via
        # format_queries_block (queries_state carries options + answer_form).
        # Putting them ALSO in user_input duplicates prompt text and creates a
        # train/runtime mismatch. user_input carries just the bare question text,
        # matching non-MC asks.
        user_input = ""
        if card_id and ask_chunk_by_card.get(card_id) == c and card:
            user_input = card.get("question", "")

        if ds.sample_kind == "patrol":
            # Legacy cache compatibility. New design emits ordinary "silent"
            # for no-active-query timeline chunks.
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                base_role="legacy_patrol", sample_subtype="timeline_silent",
            ))
        elif ds.sample_kind == "compress_silent":
            # Render as a standalone compact-memory update sample with the
            # gold <m> lines from the rollout's compression event.
            ce = compress_event_by_chunk.get(c)
            if ce:
                raw.append(_compress_sample(
                    c, _think_for_chunk(rollout, c), queries_state,
                    traj_id, ce, rollout=rollout,
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
                # Training a recall tool call without start_time/end_time teaches an
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
            post_recall_think = ""
            if client is not None and PASS3C_ENABLE_LLM_POST_RECALL_THINK:
                post_recall_think = await _post_recall_think_via_llm(
                    card or {},
                    rq,
                    rr,
                    "",
                    client,
                    video_id,
                    c,
                    action="silent",
                    all_frame_paths=all_frame_paths,
                )
            if (
                not post_recall_think
                and client is not None
                and PASS3C_ENABLE_LLM_POST_RECALL_THINK
            ):
                post_recall_think = _use_post_recall_think_fallback(
                    card=card or {},
                    recall_result=rr,
                    response="",
                    video_id=video_id,
                    chunk_idx=c,
                    action="silent",
                )
            raw.append(_recall_silent_multiturn_sample(
                c, _think_for_chunk(rollout, c), queries_state,
                rq, rr, traj_id, card_id or "", sequence_type,
                user_input=user_input, recall_reason=recall_reason,
                post_recall_think=post_recall_think,
                allow_template_fallback=(
                    client is None
                    or not PASS3C_ENABLE_LLM_POST_RECALL_THINK
                    or PASS3C_ALLOW_POST_RECALL_THINK_FALLBACK
                ),
            ))
            # Do not append an answer or close the query. recall+silent is a
            # wait state; a later response/recall+response sample must answer.
        elif ds.sample_kind == "response":
            if client is not None and PASS3C_ENABLE_LLM_RESPONSE:
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
            if client is not None and PASS3C_ENABLE_LLM_RESPONSE:
                resp = await _response_text_via_llm(
                    card or {}, ds.response_text, client, video_id, c)
            else:
                resp = _response_text_for(card or {}, ds.response_text)
            rq = _repair_recall_query_for_response(
                card or {},
                _recall_query_for(card or {}, c),
                c,
            )
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
            post_recall_think = ""
            if client is not None and PASS3C_ENABLE_LLM_POST_RECALL_THINK:
                post_recall_think = await _post_recall_think_via_llm(
                    card or {},
                    rq,
                    rr,
                    resp,
                    client,
                    video_id,
                    c,
                    action="response",
                    all_frame_paths=all_frame_paths,
                )
            if (
                not post_recall_think
                and client is not None
                and PASS3C_ENABLE_LLM_POST_RECALL_THINK
            ):
                post_recall_think = _use_post_recall_think_fallback(
                    card=card or {},
                    recall_result=rr,
                    response=resp,
                    video_id=video_id,
                    chunk_idx=c,
                    action="response",
                )
            raw.append(_recall_response_sample(
                c, _think_for_chunk(rollout, c), resp, queries_state,
                rq, rr, traj_id, card_id, sequence_type, user_input=user_input,
                recall_reason=recall_reason, card=card or {},
                post_recall_think=post_recall_think,
                allow_template_fallback=(
                    client is None
                    or not PASS3C_ENABLE_LLM_POST_RECALL_THINK
                    or PASS3C_ALLOW_POST_RECALL_THINK_FALLBACK
                ),
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
            s["support_policy"] = getattr(placement, "support_policy", "") or card.get("support_policy", "")
            s["card_support_policy"] = card.get("support_policy", "")
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
    allow_stale = os.environ.get("THINKSTREAM_ALLOW_STALE_PASS3_CACHE", "").lower() in {"1", "true", "yes", "on"}
    if not allow_stale and not stage_version_ok("3c"):
        return None
    p = samples_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
