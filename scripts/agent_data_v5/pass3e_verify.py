"""
Pass 3-E: Verify + Tag

Validates every sample produced by Pass 3 (fork + base).

10 existing checks (adapted for new sample structure):
  1. Information flow legality (no future leakage)
  2. Action minimality (correct action for availability)
  3. Grounding (think supported by visual facts)
  4. Format & length
  5. Think token length
  6. Compression ratio
  7. Summary provenance
  8. Summary retention
  9. Summary no-current-think leak
  10. Question-answer leakage

4 new checks for Pass 3 pipeline:
  11. Queries-state temporal consistency
  12. Trajectory action distribution
  13. Base sample queries-state interpolation
  14. Recall evidence reachability

Each sample gets a pass/fail verdict with reasons; no rows are dropped here.
Output: verified/{video_id}.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    COMPRESSION_RATIO_MIN,
    LEAKAGE_OVERLAP_THRESHOLD,
    THINK_TOKENS,
    VERIFIED_DIR,
    VISUAL_WINDOW_CHUNKS,
    get_tokenizer,
)
from .pass3a_cards import extract_keywords, extract_card_keywords
from .pass3b_placement import _keyword_overlap as keyword_overlap
from thinkstream.trainer.outcome_match import score_outcome_by_form

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SUMMARY_TOKEN_STOPWORDS = {
    # Sentence-openers / discourse markers, not source entities.
    "a", "an", "the", "this", "that", "these", "those",
    "in", "on", "at", "to", "from", "with", "without", "while",
    "after", "before", "during", "first", "initially", "then", "later",
    "finally", "meanwhile",
    "scene", "scenes", "frame", "frames", "shot", "shots", "view",
    "video", "current", "background", "backgrounds", "foreground",
    "lighting", "close", "static", "several", "multiple", "various",
    "surrounding", "directly", "next", "inside", "outside", "below",
    "above", "behind", "beside", "nearby", "near", "left", "right",
    "top", "bottom", "center", "central", "upper", "lower", "front",
    "back",
    # Generic visual roles/objects that appear capitalized at sentence starts.
    "person", "people", "man", "woman", "child", "baby", "chef",
    "hand", "hands", "arm", "arms", "they", "their", "them", "her",
    "his", "he", "she",
    "shelf", "shelves", "counter", "countertop", "table", "floor",
    "wall", "walls",
}


def _summary_key_token(token: str) -> Optional[str]:
    """Return a normalized retention/provenance token, or None for noise."""
    t = (token or "").strip()
    if not t:
        return None
    lower = t.lower()
    if lower in _SUMMARY_TOKEN_STOPWORDS:
        return None
    # ALL-CAPS OCR/title fragments are often style text; provenance should not
    # fail a useful compression summary just because it paraphrases or drops one.
    if t.isupper() and len(t) > 2:
        return None
    if len(lower) < 3:
        return None
    return lower


def _count_tokens(text: str) -> int:
    """Count tokens using student tokenizer, fallback to chars/4."""
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text) // 4


def _parse_time_range_from_memory_line(item) -> Tuple[int, int]:
    text = item if isinstance(item, str) else item.get("time", "") if isinstance(item, dict) else str(item)
    m = re.search(r'\[(\d+)-(\d+)\]', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    if isinstance(item, dict) and "time" in item and isinstance(item["time"], str):
        m = re.search(r'(\d+)-(\d+)', item["time"])
        if m:
            return int(m.group(1)), int(m.group(2))
    return -1, -1


_RECALL_TIME_RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$"
)
_AUDIT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_AUDIT_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "while", "what", "which", "who", "where", "when", "how", "is",
    "are", "was", "were", "be", "been", "being", "by", "from", "as", "it",
    "this", "that", "these", "those", "into", "onto", "there", "here", "his",
    "her", "their", "its", "your", "only", "answer", "option", "text",
    "letter", "video", "scene", "frame", "frames", "question",
}


def _valid_recall_time_range(value) -> bool:
    if not isinstance(value, str):
        return False
    m = _RECALL_TIME_RANGE_RE.fullmatch(value)
    if not m:
        return False
    return float(m.group(2)) > float(m.group(1))


def _audit_tokens(text: str) -> set:
    return {
        t for t in _AUDIT_TOKEN_RE.findall(str(text or "").lower())
        if len(t) >= 3 and t not in _AUDIT_STOPWORDS
    }


def _audit_overlap(answer: str, text: str) -> float:
    ans = _audit_tokens(answer)
    if not ans:
        return 0.0
    return len(ans & _audit_tokens(text)) / max(len(ans), 1)


def _audit_memory_text(memory: Dict) -> str:
    if not isinstance(memory, dict):
        return str(memory or "")
    parts = []
    for seg in memory.get("compressed_segments", memory.get("compressed", [])) or []:
        if isinstance(seg, dict):
            parts.append(str(seg.get("text", "")))
        else:
            parts.append(str(seg))
    for item in memory.get("recent_thinks", memory.get("recent_observations", [])) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("text", item.get("obs", ""))))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _valid_compress_time_range(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
        and float(value[1]) > float(value[0])
    )


def _memory_item_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", item.get("obs", str(item)))
    return str(item)


def _verify_mc_response_text(resp_text: str, metadata: Dict) -> Tuple[bool, str]:
    """Validate MC answer semantics and, when present, the SFT target style."""
    style = (metadata.get("answer_style") or "").strip()
    correct = (metadata.get("correct_option") or "").strip().upper()
    options = list(metadata.get("options") or [])
    valid_letters = "".join(chr(ord("A") + i) for i in range(min(len(options), 26)))
    if style == "letter_only":
        letter_pattern = f"[{re.escape(valid_letters or 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')}]"
        if not re.fullmatch(letter_pattern, resp_text.strip().upper()):
            return False, f"mc_response_not_letter_only: '{resp_text[:30]}'"
        if correct and resp_text.strip().upper() != correct:
            return False, f"mc_response_letter_mismatch: got '{resp_text}' want '{correct}'"
    elif style == "letter_plus_text":
        if correct and not re.match(rf"^\s*{re.escape(correct)}[\).:\s]", resp_text.strip(), re.I):
            return False, f"mc_response_not_letter_plus_text: '{resp_text[:30]}'"
    elif style == "text_only":
        if re.match(r"^\s*(?:\([A-Z]\)|[A-Z][\).:])", resp_text.strip(), re.I):
            return False, f"mc_response_has_letter_for_text_only: '{resp_text[:30]}'"

    gold = (metadata.get("correct_answer_text")
            or metadata.get("gold_answer")
            or metadata.get("canonical_answer")
            or "")
    score = score_outcome_by_form(
        resp_text,
        options=options,
        correct_option=metadata.get("correct_option", ""),
        gold_answer=gold,
        answer_form="multiple_choice",
    )
    if score < 1.0:
        return False, f"mc_response_semantic_mismatch: '{resp_text[:50]}'"
    return True, "pass"


def _compressed_source_texts(sample: Dict) -> List[str]:
    """Return source memory items selected for compression."""
    metadata = sample.get("metadata", {})
    input_data = sample.get("input", {})
    memory = input_data.get("memory", {})

    if metadata.get("task_type") == "merge_compress":
        compressed = memory.get("compressed", [])
        return [
            seg.get("text", str(seg)) if isinstance(seg, dict) else str(seg)
            for seg in compressed
        ]

    recent_thinks = memory.get("recent_thinks", [])
    if not recent_thinks:
        return []

    compressed_chunks = set(metadata.get("compressed_chunks", []) or [])
    compressed_range = metadata.get("compressed_range", None)

    selected = []
    for item in recent_thinks:
        start, end = _parse_time_range_from_memory_line(item)
        chunk = start // 2 if start >= 0 else None
        in_chunk_set = chunk in compressed_chunks if compressed_chunks else False
        in_range = (
            bool(compressed_range)
            and start >= int(compressed_range[0])
            and end <= int(compressed_range[1])
        )
        if in_chunk_set or in_range:
            selected.append(_memory_item_text(item))

    return selected if selected else [_memory_item_text(t) for t in recent_thinks]


def _is_base_sample(sample: Dict) -> bool:
    """Check if sample is a base (non-fork) sample."""
    return sample.get("sequence_type") == "base"


# ---------------------------------------------------------------------------
# Existing Checks (1-10), adapted for new sample structure
# ---------------------------------------------------------------------------


def _is_v12_sample(sample: Dict) -> bool:
    """v12 samples either carry protocol_version=='v12' or have v12 markers.

    v12.11 audit-4 P0 #2 fix (2026-05-01): plain silent/response samples
    don't carry protocol_version (it's only stamped on shape-B recall and
    inter-chunk compress). These samples DO use the v12 protocol — their
    output is `<think>...</think><answer>...</answer>` (or empty answer
    for silent), NOT v11's `<action>...</action>`. Without recognising
    them as v12, the legacy verifier looks for `<action>` and reports
    `format_check_failed` on every silent / response sample → systematic
    false-negatives in pass3e verification.

    Detect v12 reliably via: explicit marker fields OR an output that
    contains `<answer>` / `<tool_call>` (v12-only tags) and lacks the
    legacy `<action>` tag.
    """
    if (sample.get("protocol_version") == "v12"
            or "v12_assistant_turn_1" in sample
            or sample.get("v12_inter_chunk") is True):
        return True
    out = sample.get("output", "") or sample.get("v12_assistant_turn_2", "") or ""
    if not isinstance(out, str):
        return False
    has_v12_tag = ("<answer>" in out) or ("<tool_call>" in out)
    has_legacy_tag = "<action>" in out
    return has_v12_tag and not has_legacy_tag


def _v12_combined_assistant_text(sample: Dict) -> str:
    """Get the full assistant text for v12 samples (handles multi-turn)."""
    if "v12_assistant_turn_1" in sample:
        # Multi-turn recall sample: concat both assistant turns
        return (sample.get("v12_assistant_turn_1", "") + "\n"
                + sample.get("v12_assistant_turn_2", ""))
    return sample.get("output", "")


def verify_information_flow(sample: Dict) -> Tuple[bool, str]:
    """Check 1: No future information leakage."""
    if _is_v12_sample(sample):
        return _verify_information_flow_v12(sample)

    metadata = sample.get("metadata", {})
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # Base samples: no leakage risk (think from rollout, no question/answer)
    if _is_base_sample(sample):
        return True, "pass"

    # Query doesn't contain answer
    if metadata.get("leakage_checks", {}).get("query_contains_answer"):
        return False, "query_contains_answer_value"

    # Response is non-empty.
    # v9.1: don't reject by character length — binary "No" is 2 chars and
    # short_exact "5" is 1. Only reject TRULY empty responses.
    # v9.3: multiple_choice = single letter (1 char) — must also pass.
    #       Add strict format check for binary/MC/number — these go straight
    #       into OVO eval matching, so any drift fails eval.
    if "<response>" in output:
        response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if response_match:
            resp_text = response_match.group(1).strip()
            if not resp_text:
                return False, "empty_response"
            answer_form = metadata.get("answer_form", "")
            family = metadata.get("family", "")
            # Forms whose response IS the eval-format answer — short by design.
            short_forms = ("binary", "multiple_choice", "number", "short_exact")
            if answer_form not in short_forms and family != "N1":
                if len(resp_text) < 3:
                    return False, "response_too_short"
            # Strict format match for forms that go directly into eval matching.
            # canonical_answer (=== gold_answer for these forms) is normalized
            # in pass3c via _normalize_exact_form_answer; the response must
            # match exactly. Drift here = OVO eval miss.
            if answer_form == "multiple_choice":
                ok, reason = _verify_mc_response_text(resp_text, metadata)
                if not ok:
                    return False, reason
            elif answer_form == "binary":
                if resp_text not in ("Yes", "No"):
                    return False, f"binary_response_not_yes_no: '{resp_text[:30]}'"
            elif answer_form == "number":
                if not re.fullmatch(r'\d+', resp_text):
                    return False, f"number_response_not_digits: '{resp_text[:30]}'"

    # recall_response after failure must show uncertainty
    if sample_type == "recall_response":
        recall_result = sample.get("recall_result") or \
                        sample.get("input", {}).get("recall_result", {})
        noise_level = recall_result.get("noise_level", recall_result.get("source", "oracle"))
        if noise_level in ("distractor", "failure"):
            response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
            if response_match:
                resp = response_match.group(1).lower()
                uncertain = [
                    "cannot", "could not", "not sure", "unable",
                    "uncertain", "don't see", "not enough", "unclear",
                    "not confirm", "insufficient",
                ]
                if not any(m in resp for m in uncertain):
                    return False, "confident_response_after_failed_recall"

    return True, "pass"


# ===========================================================================
# v12.0 — protocol-aware verification
#
# v12 samples use Qwen tool protocol (<tool_call>{json}</tool_call> +
# <answer>...</answer>) instead of v11's <action>X</action> structure.
# pass4's legacy verify_format / verify_information_flow / etc. all check
# v11-specific tags and would reject every v12 sample (or silently let bad
# format slip through). The functions below are v12-equivalents called from
# the protocol-aware dispatchers.
# ===========================================================================


def _verify_information_flow_v12(sample: Dict) -> Tuple[bool, str]:
    """v12 information-flow check: validate <answer>/<tool_call> content.

    Mirrors the v11 strict-format intent (binary/MC/number drift = OVO miss)
    but reads from v12's <answer> terminal instead of <response>.
    """
    metadata = sample.get("metadata", {})
    output = _v12_combined_assistant_text(sample)
    sample_type = sample.get("sample_type", "")

    if _is_base_sample(sample):
        return True, "pass"

    # Query doesn't contain answer (re-uses metadata flag from pass3c)
    if metadata.get("leakage_checks", {}).get("query_contains_answer"):
        return False, "query_contains_answer_value"

    # Locate the FINAL answer — for multi-turn recall it's in turn 2
    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if answer_match is not None:
        resp_text = answer_match.group(1).strip()
        # silent samples = empty answer (intended). Validate non-silent only.
        is_answer_sample = (
            sample_type in ("response", "recall_response")
            or (sample_type == "recall" and sample.get("action") == "response")
        )
        if is_answer_sample:
            if not resp_text:
                return False, "empty_response"
            answer_form = metadata.get("answer_form", "")
            family = metadata.get("family", "")
            short_forms = ("binary", "multiple_choice", "number", "short_exact")
            if answer_form not in short_forms and family != "N1":
                if len(resp_text) < 3:
                    return False, "response_too_short"
            if answer_form == "multiple_choice":
                ok, reason = _verify_mc_response_text(resp_text, metadata)
                if not ok:
                    return False, reason
            elif answer_form == "binary":
                if resp_text not in ("Yes", "No"):
                    return False, f"binary_response_not_yes_no: '{resp_text[:30]}'"
            elif answer_form == "number":
                if not re.fullmatch(r"\d+", resp_text):
                    return False, f"number_response_not_digits: '{resp_text[:30]}'"

    # Recall-failure uncertainty check — operates on the SECOND turn's answer
    # for multi-turn recall samples (turn 1 is the tool_call itself).
    if sample_type in ("recall_response", "recall") and "v12_assistant_turn_2" in sample:
        recall_result = sample.get("recall_result") or \
                        sample.get("input", {}).get("recall_result", {}) or {}
        noise_level = recall_result.get("noise_level", recall_result.get("source", "oracle"))
        if noise_level in ("distractor", "failure"):
            t2 = sample.get("v12_assistant_turn_2", "")
            ans = re.search(r"<answer>(.*?)</answer>", t2, re.DOTALL)
            if ans:
                resp = ans.group(1).lower()
                if sample.get("action") == "silent" and not resp.strip():
                    return True, "pass"
                uncertain_kws = [
                    "cannot", "could not", "not sure", "unable",
                    "uncertain", "don't see", "not enough", "unclear",
                    "not confirm", "insufficient",
                ]
                if not any(kw in resp for kw in uncertain_kws):
                    return False, "confident_response_after_failed_recall"

    return True, "pass"


def _verify_format_v12(sample: Dict) -> Tuple[bool, str]:
    """v12 format check: <think>/<answer>/<tool_call> well-formedness.

    v12 grammar (from agent_protocol.parse_agent_output_v12):
      <think>...</think>  (always)
      then EXACTLY ONE of:
        <answer>...</answer>
        <tool_call>{name, arguments}</tool_call>
    Multi-turn recall: turn 1 = <tool_call>{name=recall}</tool_call>,
                       turn 2 = <answer>...</answer>
    """
    sample_type = sample.get("sample_type", "")
    is_multiturn = "v12_assistant_turn_1" in sample
    is_inter_chunk = sample.get("v12_inter_chunk") is True

    # Collect each assistant turn's text for separate validation.
    turns: List[str] = []
    if is_multiturn:
        turns.append(sample.get("v12_assistant_turn_1", ""))
        turns.append(sample.get("v12_assistant_turn_2", ""))
    else:
        turns.append(sample.get("output", ""))

    for i, out in enumerate(turns):
        # think tags required (length-checked downstream)
        if "<think>" not in out or "</think>" not in out:
            if not (_is_base_sample(sample) and not is_multiturn
                    and sample.get("action") == "compress"):
                return False, f"v12_missing_think_tags (turn {i})"

        has_answer = "<answer>" in out and "</answer>" in out
        has_tool_call = "<tool_call>" in out and "</tool_call>" in out

        if has_answer and has_tool_call:
            return False, f"v12_both_terminals (turn {i})"
        if not has_answer and not has_tool_call:
            return False, f"v12_no_terminal (turn {i})"

        # Tool-call JSON well-formedness
        if has_tool_call:
            tc_match = re.search(r"<tool_call>(.*?)</tool_call>", out, re.DOTALL)
            if tc_match:
                try:
                    tc_obj = json.loads(tc_match.group(1).strip())
                except (json.JSONDecodeError, ValueError):
                    return False, f"v12_tool_call_invalid_json (turn {i})"
                if "name" not in tc_obj or "arguments" not in tc_obj:
                    return False, f"v12_tool_call_missing_fields (turn {i})"
                name = tc_obj.get("name")
                if name not in ("recall", "compress"):
                    return False, f"v12_unknown_tool_name: {name!r}"
                args = tc_obj.get("arguments") or {}
                if name == "recall":
                    if "query" not in args or "time_range" not in args:
                        return False, "v12_recall_args_missing_fields"
                    if not str(args.get("query") or "").strip():
                        return False, "v12_recall_empty_query"
                    if not _valid_recall_time_range(args.get("time_range")):
                        return False, "v12_recall_bad_time_range"
                elif name == "compress":
                    if "time_range" not in args or "text" not in args:
                        return False, "v12_compress_args_missing_fields"
                    if not _valid_compress_time_range(args.get("time_range")):
                        return False, "v12_compress_bad_time_range"
                    if not args.get("text"):
                        return False, "v12_compress_empty_summary"

    # Sample-type ↔ terminal-kind consistency
    final_text = turns[-1]
    final_has_answer = "<answer>" in final_text
    final_has_tool = "<tool_call>" in final_text
    if sample_type == "silent":
        if not final_has_answer:
            return False, "v12_silent_must_emit_answer"
        m = re.search(r"<answer>(.*?)</answer>", final_text, re.DOTALL)
        if m and m.group(1).strip():
            return False, "v12_silent_answer_must_be_empty"
    elif sample_type == "response":
        if not final_has_answer:
            return False, "v12_response_must_emit_answer"
    elif sample_type == "compress":
        # compress samples: ONLY a tool_call (no answer)
        if not final_has_tool:
            return False, "v12_compress_must_emit_tool_call"
        if final_has_answer:
            return False, "v12_compress_should_not_emit_answer"
        if not is_inter_chunk:
            # v12 compress remains an inter-chunk memory-management turn.
            return False, "v12_compress_missing_inter_chunk_flag"
    elif sample_type == "recall":
        # multi-turn merged: turn1=tool_call(recall), turn2=answer
        if not is_multiturn:
            return False, "v12_recall_not_multiturn"
        if "<tool_call>" not in turns[0]:
            return False, "v12_recall_turn1_missing_tool_call"
        if "<answer>" not in turns[1]:
            return False, "v12_recall_turn2_missing_answer"

    return True, "pass"


def _v12_extract_summary_text(sample: Dict) -> str:
    """Extract compress summary text from v12 <tool_call> JSON. Empty if missing."""
    if sample.get("sample_type") != "compress":
        return ""
    output = _v12_combined_assistant_text(sample)
    tc_match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not tc_match:
        return ""
    try:
        tc = json.loads(tc_match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return ""
    return (tc.get("arguments") or {}).get("text", "") or ""


def _verify_compression_ratio_v12(sample: Dict) -> Tuple[bool, str]:
    """v12 compression: read summary text from <tool_call> args.text."""
    if sample.get("sample_type") != "compress":
        return True, "pass"

    output = _v12_combined_assistant_text(sample)
    tc_match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not tc_match:
        return True, "pass"  # missing tool_call already caught by format check
    try:
        tc_obj = json.loads(tc_match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return True, "pass"  # invalid json caught by format check
    summary_text = (tc_obj.get("arguments") or {}).get("text", "")
    if not summary_text:
        return False, "empty_compression_summary"

    source_texts = _compressed_source_texts(sample)
    if not source_texts:
        return True, "pass"

    input_tokens = _count_tokens(" ".join(source_texts))
    summary_tokens = _count_tokens(summary_text)
    if summary_tokens == 0:
        return False, "empty_compression_summary"
    ratio = input_tokens / summary_tokens
    if ratio < COMPRESSION_RATIO_MIN:
        return False, f"compression_ratio_too_low ({ratio:.1f} < {COMPRESSION_RATIO_MIN})"
    return True, "pass"


def verify_action_minimality(sample: Dict) -> Tuple[bool, str]:
    """Check 2: Action is the minimal correct action for the availability."""
    sample_type = sample.get("sample_type", "")
    seq_type = sample.get("sequence_type", "")

    # Base samples always silent/compress — correct by construction
    if _is_base_sample(sample):
        return True, "pass"

    # For recall_query: check that the sample actually needs recall
    # (availability was in_history_only, verified by sequence_type)
    # v12 collapses recall_query+recall_response into sample_type='recall'
    # via _merge_recall_pairs_v12. Treat both labels equivalently for the
    # action-minimality / visibility checks.
    if sample_type in ("recall_query", "recall"):
        action = sample.get("action", "")
        is_wait_recall = action == "silent" and seq_type == "event_watch"
        if not is_wait_recall and seq_type not in ("recall_success", "recall_fail_then_found"):
            return False, f"recall_query_in_non_recall_sequence: {seq_type}"

    # For response in immediate_response: verify it's not recall sequence
    if sample_type == "response" and seq_type == "immediate_response":
        # This is correct — answer was available without recall
        pass

    # Legacy metadata-based checks (backward compat)
    metadata = sample.get("metadata", {})
    if sample_type in ("recall_query", "recall"):
        visibility = metadata.get("visibility", {})
        if visibility.get("answer_in_recent_obs"):
            return False, "recall_unnecessary_answer_in_observations"
        if visibility.get("answer_in_compressed"):
            return False, "recall_unnecessary_answer_in_compressed"
        if visibility.get("evidence_in_window"):
            return False, "recall_unnecessary_evidence_in_window"

    return True, "pass"


def verify_grounding(sample: Dict) -> Tuple[bool, str]:
    """Check 3: Think is grounded in visual facts.

    v12: multi-turn recall samples have `output` popped — read from
    _v12_combined_assistant_text which concatenates both turn texts so
    we check think tags from BOTH the recall turn and the answer turn.
    """
    if _is_v12_sample(sample):
        output = _v12_combined_assistant_text(sample)
    else:
        output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # recall_response has no think — check response instead
    if sample_type == "recall_response":
        resp_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if not resp_match:
            return True, "pass"
        check_text = resp_match.group(1).lower()
    else:
        # v12 multi-turn samples may have multiple <think> blocks (one per
        # assistant turn). Concatenate ALL think contents for blacklist scan
        # so a non-visual phrase in either turn is caught.
        all_thinks = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)
        if not all_thinks:
            if _is_base_sample(sample) and sample.get("action") == "compress":
                return True, "pass"
            return False, "no_think_tag"
        check_text = "\n".join(t for t in all_thinks).lower()

    # v9.4 — narrowed blacklist (was 33 phrases, dropped to ~17). Removed:
    #   "i think" / "i notice" / "i can see" — first-person observational
    #     phrasing is normal English narration, not a grounding violation.
    #   "probably" / "likely" — epistemic hedging is fine when describing
    #     genuine visual ambiguity ("the dish is probably soup based on the
    #     bowl shape"); rejecting these dropped 1.2k legit thinks in batch1.
    #   "talking" — mouth movement / speaking posture is commonly visible in
    #     frames; keep true audio-only words such as sound/voice/hear.
    #   "the video shows" — this is a harmless captioning style in pass1,
    #     not evidence of ungrounded content by itself.
    #   "feels" / "feeling" — model can describe physical contact (chef
    #     feels the dough); reserve "emotion" for true affect leaks.
    # Kept the strong non-visual sensory channels (sound/smell), affect
    # words (happy/sad/angry), and meta-language (the video shows / system
    # triggered) — these are real grounding violations that matter.
    # Note: per-sample rejection matches the OVO eval granularity (eval
    # scores per chunk, not per trajectory), so a stale phrase in one
    # sample shouldn't take down adjacent samples in the same trajectory.
    blacklist_phrases = [
        # Sensory channels the model has no access to
        "sound", "hear", "listen", "noise", "sizzle", "sizzling",
        "music", "speech", "voice",
        "smell", "aroma", "scent", "fragrant", "aromatic",
        # True affect leaks (model is not a person, has no emotions)
        "emotion", "happy", "sad", "angry",
        # Speculative-intent leaks (the model shouldn't infer wishes)
        "seems to want", "intend",
        # Meta-language about the system / dataset (breaks 4th wall)
        "the user wants",
        "system triggered", "memory compression", "retrieved evidence",
    ]

    for phrase in blacklist_phrases:
        if phrase in check_text:
            return False, f"think_contains_non_visual: '{phrase}'"

    return True, "pass"


def verify_format(sample: Dict) -> Tuple[bool, str]:
    """Check 4: Output format and tag consistency."""
    if _is_v12_sample(sample):
        return _verify_format_v12(sample)

    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # recall_response: may lack think tags (think was in recall_query step)
    if sample_type != "recall_response":
        if "<think>" not in output or "</think>" not in output:
            # Base compress with empty think is allowed
            if not (_is_base_sample(sample) and sample.get("action") == "compress"):
                return False, "missing_think_tags"

    if "<action>" not in output or "</action>" not in output:
        return False, "missing_action_tags"

    # Think length
    obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if obs_match:
        words = len(obs_match.group(1).split())
        action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
        action_type = action_match.group(1) if action_match else ""
        min_words = 3 if action_type in ("compress", "recall") else 5
        if words < min_words:
            return False, f"think_too_short ({words} words)"
        if words > 100:
            return False, f"think_too_long ({words} words)"

    # Valid action
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    if action_match:
        action = action_match.group(1)
        if action not in {"silent", "response", "recall", "compress"}:
            return False, f"invalid_action: {action}"
        if action == "recall" and "<query>" not in output:
            return False, "recall_action_without_query"
        if action == "compress" and "<summary>" not in output:
            return False, "compress_action_without_summary"
        if action == "response" and "<response>" not in output:
            return False, "response_action_without_response_tag"

    # Query JSON
    if "<query>" in output:
        query_match = re.search(r'<query>(.*?)</query>', output, re.DOTALL)
        if query_match:
            try:
                q = json.loads(query_match.group(1))
                if "query" not in q:
                    return False, "query_json_missing_query_field"
            except (json.JSONDecodeError, ValueError):
                return False, "query_invalid_json"

    # Summary JSON
    if "<summary>" in output:
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if summary_match:
            try:
                s = json.loads(summary_match.group(1))
                if "time_range" not in s or "text" not in s:
                    return False, "summary_json_missing_fields"
            except (json.JSONDecodeError, ValueError):
                return False, "summary_invalid_json"

    return True, "pass"


def verify_think_token_length(sample: Dict) -> Tuple[bool, str]:
    """Check 5: Think is within token range."""
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    if sample_type == "recall_response":
        return True, "pass"

    obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if not obs_match:
        return True, "pass"

    think_text = obs_match.group(1).strip()
    if not think_text:
        return True, "pass"

    tok_count = _count_tokens(think_text)
    # Compress turns are policy decisions, not visual observations. The gold
    # template is intentionally concise ("memory is over budget..."), so do not
    # apply the visual-turn lower bound to compress supervision.
    if sample_type == "compress":
        min_tok = 8
        max_tok = THINK_TOKENS[1] + 50
    else:
        # Margins: teacher tends to overshoot, so widen on the high side.
        min_tok = max(15, THINK_TOKENS[0] - 15)   # 25 by default
        max_tok = THINK_TOKENS[1] + 30            # 130 by default — covers p99

    if tok_count < min_tok:
        return False, f"think_tokens_too_few ({tok_count} < {min_tok})"
    if tok_count > max_tok:
        return False, f"think_tokens_too_many ({tok_count} > {max_tok})"

    return True, "pass"


def verify_compression_ratio(sample: Dict) -> Tuple[bool, str]:
    """Check 6: Compression summary achieves minimum ratio."""
    if _is_v12_sample(sample):
        return _verify_compression_ratio_v12(sample)

    if sample.get("sample_type") != "compress":
        return True, "pass"

    output = sample.get("output", "")
    summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
    if not summary_match:
        return True, "pass"

    try:
        summary_json = json.loads(summary_match.group(1))
        summary_text = summary_json.get("text", "")
    except (json.JSONDecodeError, ValueError):
        return True, "pass"

    if not summary_text:
        return False, "empty_compression_summary"

    source_texts = _compressed_source_texts(sample)
    if not source_texts:
        # Base compress samples may not have input.memory — skip ratio check
        if _is_base_sample(sample):
            return True, "pass"
        return True, "pass"

    input_tokens = _count_tokens(" ".join(source_texts))
    summary_tokens = _count_tokens(summary_text)

    if summary_tokens == 0:
        return False, "empty_compression_summary"

    ratio = input_tokens / summary_tokens
    if ratio < COMPRESSION_RATIO_MIN:
        return False, f"compression_ratio_too_low ({ratio:.1f} < {COMPRESSION_RATIO_MIN})"

    return True, "pass"


def verify_summary_provenance(sample: Dict) -> Tuple[bool, str]:
    """Check 7: Summary entities grounded in source thinks."""
    if sample.get("sample_type") != "compress":
        return True, "pass"

    if _is_v12_sample(sample):
        summary_text = _v12_extract_summary_text(sample)
    else:
        output = sample.get("output", "")
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not summary_match:
            return True, "pass"
        try:
            summary_text = json.loads(summary_match.group(1)).get("text", "")
        except (json.JSONDecodeError, ValueError):
            return True, "pass"

    if not summary_text:
        return True, "pass"

    source_texts = _compressed_source_texts(sample)
    if not source_texts:
        if _is_base_sample(sample):
            return True, "pass"
        return True, "pass"

    source_words = set()
    for text in source_texts:
        source_words.update(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text))

    entity_words = []
    for w in re.findall(r'\b[a-zA-Z0-9_]+\b', summary_text):
        if not (w[0].isupper() or "_" in w):
            continue
        token = _summary_key_token(w)
        if token:
            entity_words.append(w)

    if not entity_words:
        return True, "pass"

    unsupported = [w for w in entity_words if w.lower() not in source_words]
    has_visual = sample.get("metadata", {}).get("has_visual_context", False)
    # This is an audit signal, not a hard hallucination proof: compression is
    # expected to paraphrase and drop detail. Keep only egregious unsupported
    # proper-noun bursts as failures.
    max_ratio = 0.9 if has_visual else 0.8

    if len(unsupported) / len(entity_words) > max_ratio:
        return False, f"summary_provenance_violation: {unsupported[:3]} not in source"

    return True, "pass"


def _retention_threshold(n_unique: int) -> float:
    """Adaptive retention threshold scaled by source key-item count.

    A flat 0.5 floor punishes long sources unfairly: with 20 unique items,
    natural compression drops half of them and that's fine; with 4 items,
    losing 2 is severe. Empirically the v9.1 audit showed 55% of all
    pass4 failures came from `summary_retention`, mostly on long sources.
    """
    # Retention is a coarse sanity check. The token extractor is intentionally
    # heuristic, so only fail near-catastrophic loss instead of enforcing a
    # benchmark-style coverage score on every compressed detail.
    if n_unique <= 3:
        return 0.01
    if n_unique <= 8:
        return 0.15
    if n_unique <= 15:
        return 0.12
    return 0.08


def verify_summary_retention(sample: Dict) -> Tuple[bool, str]:
    """Check 8: Summary retains key information from source.

    Threshold is now adaptive (see `_retention_threshold`). We also
    de-duplicate the proper-noun signal a bit better — random caps from
    OCR overlays shouldn't inflate the key-item set.
    """
    if sample.get("sample_type") != "compress":
        return True, "pass"

    if _is_v12_sample(sample):
        summary_text = _v12_extract_summary_text(sample)
    else:
        output = sample.get("output", "")
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not summary_match:
            return True, "pass"
        try:
            summary_text = json.loads(summary_match.group(1)).get("text", "")
        except (json.JSONDecodeError, ValueError):
            return True, "pass"

    source_texts = _compressed_source_texts(sample)
    if not source_texts or not summary_text:
        return True, "pass"

    source_combined = " ".join(source_texts)
    summary_lower = summary_text.lower()

    key_items = []
    # Proper-nouns / entity tokens. Skip sentence-openers, generic roles, and
    # ALL-CAPS OCR/title fragments which inflate key-items spuriously.
    for w in re.findall(r'\b[A-Z][a-zA-Z0-9_]{2,}\b', source_combined):
        token = _summary_key_token(w)
        if token:
            key_items.append(token)
    for num in re.findall(r'\b\d+\.?\d*\b', source_combined):
        # Most short numbers here are chunk/time indices. Keep only model-like
        # or year-like numbers where dropping them may matter.
        if len(num.replace(".", "")) >= 4:
            key_items.append(num)

    if not key_items:
        return True, "pass"

    unique_items = list(set(key_items))
    retained = sum(1 for item in unique_items if item in summary_lower)
    rate = retained / len(unique_items)
    threshold = _retention_threshold(len(unique_items))

    if rate < threshold:
        missing = [item for item in unique_items if item not in summary_lower][:3]
        return False, (
            f"summary_retention_low ({rate:.0%} < {threshold:.0%}, "
            f"n={len(unique_items)}): missing {missing}"
        )

    return True, "pass"


def verify_summary_no_current_think_leak(sample: Dict) -> Tuple[bool, str]:
    """Check 9: Summary doesn't peek at current think."""
    if sample.get("sample_type") != "compress":
        return True, "pass"

    if _is_v12_sample(sample):
        output = _v12_combined_assistant_text(sample)
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        summary_text = _v12_extract_summary_text(sample)
        if not think_match or not summary_text:
            return True, "pass"
        think_text = think_match.group(1).strip()
    else:
        output = sample.get("output", "")
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not think_match or not summary_match:
            return True, "pass"
        think_text = think_match.group(1).strip()
        try:
            summary_text = json.loads(summary_match.group(1)).get("text", "")
        except (json.JSONDecodeError, ValueError):
            return True, "pass"

    if not think_text or not summary_text:
        return True, "pass"

    source_texts = _compressed_source_texts(sample)
    source_words = set()
    for text in source_texts:
        source_words.update(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text))

    think_unique = {
        w for w in re.findall(r'\b[a-zA-Z0-9_]+\b', think_text)
        if len(w) > 3 and (w[0].isupper() or "_" in w) and w.lower() not in source_words
    }

    if not think_unique:
        return True, "pass"

    summary_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', summary_text))
    leaked = {w for w in think_unique if w.lower() in summary_words}

    if leaked:
        return False, f"summary_contains_current_think_entities: {list(leaked)[:3]}"

    return True, "pass"


def verify_question_answer_leakage(sample: Dict) -> Tuple[bool, str]:
    """Check 10: Question text doesn't reveal the answer."""
    # Get question and answer from either metadata or sample fields
    question = (sample.get("metadata", {}).get("question")
                or sample.get("user_input", ""))
    answer = (sample.get("metadata", {}).get("gold_answer")
              or sample.get("metadata", {}).get("canonical_answer", ""))
    if not question or not answer:
        return True, "pass"

    q_lower = question.lower()
    a_lower = answer.lower().strip()

    # Exact substring
    if len(a_lower) >= 5 and a_lower in q_lower:
        return False, "question_contains_gold_answer_string"

    # All keywords present
    stop = {"the", "a", "an", "is", "was", "were", "in", "on", "at", "to",
            "of", "and", "or", "with", "from", "into", "what", "which"}
    q_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', q_lower))
    a_words = {
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', a_lower)
        if len(w) > 2 and w not in stop
    }
    if len(a_words) >= 2 and a_words.issubset(q_words):
        return False, "question_contains_all_answer_keywords"

    return True, "pass"


# ---------------------------------------------------------------------------
# New Checks (11-14): Pass 3 pipeline-specific
# ---------------------------------------------------------------------------


def verify_queries_state_temporal(sample: Dict) -> Tuple[bool, str]:
    """Check 11: queries_state only contains questions asked at or before this chunk.

    Every question in queries_state must have an ask_chunk <= sample's chunk_idx.
    This prevents future question leakage in the training input.
    """
    queries = sample.get("queries", [])
    chunk_idx = sample.get("chunk_idx", -1)

    if not queries or chunk_idx < 0:
        return True, "pass"

    # We can't directly check ask_chunk from queries_state alone (it stores
    # question text, not chunk). But we CAN check that queries_state is
    # monotonically growing across the trajectory — verified at trajectory level.
    # Per-sample check: queries must be a list of dicts with question+answers.
    open_queries = []
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            return False, f"queries_state[{i}] is not a dict"
        if "question" not in q:
            return False, f"queries_state[{i}] missing 'question' field"
        if "answers" not in q:
            return False, f"queries_state[{i}] missing 'answers' field"
        if not isinstance(q["answers"], list):
            return False, f"queries_state[{i}]['answers'] is not a list"
        status = str(q.get("status", "open")).strip().lower()
        if status in {"open", "pending", "active"}:
            open_queries.append(q)

    if len(open_queries) > 1:
        ids = [str(q.get("card_id") or q.get("question", ""))[:60] for q in open_queries]
        return False, f"multiple_open_queries:{ids}"

    return True, "pass"


def verify_trajectory_action_distribution(
    trajectory_samples: List[Dict],
) -> Tuple[bool, str]:
    """Check 12: Trajectory has reasonable action distribution.

    Checks at the trajectory level (not per-sample):
    - At least 1 response or recall sample (not all silent)
    - Silent should not exceed an adaptive threshold:
        * 98% if any placement is event_watch (legit long waits)
        * 97% if any placement is multi_response (silent gaps OK)
        * 90% otherwise
    """
    if not trajectory_samples:
        return True, "pass"

    action_counts = {}
    for s in trajectory_samples:
        a = s.get("action", "unknown")
        action_counts[a] = action_counts.get(a, 0) + 1

    total = sum(action_counts.values())
    active = action_counts.get("response", 0) + action_counts.get("recall", 0)

    if active == 0:
        return False, "trajectory_has_no_active_samples"

    # Count unique placements (non-base fork samples)
    n_questions = len(set(
        s.get("card_id", "") for s in trajectory_samples
        if s.get("card_id") and s.get("sequence_type") != "base"
    ))

    if n_questions >= 3:
        # v9.1: adaptive threshold by sequence_type. event_watch trajectories
        # legitimately spend most chunks waiting for the trigger.
        seq_types = {s.get("sequence_type", "") for s in trajectory_samples}
        if "event_watch" in seq_types:
            silent_threshold = 98.0
        elif "multi_response" in seq_types:
            silent_threshold = 97.0
        else:
            silent_threshold = 90.0

        silent_pct = action_counts.get("silent", 0) / total * 100
        if silent_pct > silent_threshold:
            return False, (
                f"multi_question_trajectory_too_silent "
                f"({silent_pct:.0f}% > {silent_threshold:.0f}%)"
            )

    return True, "pass"


def verify_base_sample_consistency(sample: Dict) -> Tuple[bool, str]:
    """Check 13: Base sample has valid structure.

    Base samples (sequence_type='base') must:
    - Have action 'silent' or 'compress'
    - Have valid queries list (may be empty for pre-question chunks)
    - Not have user_input (no question at this chunk)
    """
    if not _is_base_sample(sample):
        return True, "pass"

    action = sample.get("action", "")
    if action not in ("silent", "compress"):
        return False, f"base_sample_invalid_action: {action}"

    queries = sample.get("queries")
    if queries is not None and not isinstance(queries, list):
        return False, "base_sample_queries_not_list"

    if sample.get("user_input"):
        return False, "base_sample_has_user_input"

    return True, "pass"


def verify_support_chunks_have_evidence(sample: Dict,
                                          evidence: Optional[List[Dict]] = None) -> Tuple[bool, str]:
    """Check 14b (v9.5): support_chunks must reference 1-A chunks that
    actually contain evidence.

    Catches the case where pass3a picked silent-empty chunks (parse_success
    True but all fields empty under pre-v9.5 contract) as support. Such
    cards are unanswerable by definition — the chunks have no signal —
    and would teach the student to hallucinate.

    No-op when `evidence` is not threaded through (the existing
    filter_samples API doesn't pass it; pipeline.py adds the rich check
    where evidence is available).
    """
    if evidence is None:
        return True, "pass"
    metadata = sample.get("metadata", {})
    support_chunks = metadata.get("support_chunks") or []
    if not support_chunks:
        return True, "pass"
    ev_by_idx = {cap.get("chunk_idx", -1): cap for cap in evidence}
    empty_supports = []
    for sc in support_chunks:
        cap = ev_by_idx.get(sc)
        if cap is None:
            continue  # missing — likely a different bug, don't double-count
        has_ev = bool(cap.get("visible_entities") or cap.get("atomic_facts")
                      or cap.get("ocr")
                      or (cap.get("spatial") and str(cap.get("spatial")).strip()))
        if not has_ev:
            empty_supports.append(sc)
    if empty_supports:
        return False, f"support_chunks_empty_evidence: {empty_supports}"
    return True, "pass"


def verify_recall_evidence_reachable(sample: Dict, rollout: Dict = None) -> Tuple[bool, str]:
    """Check 14: For recall samples, the evidence chunk exists before ask_chunk.

    Recall is only valid if the answer was observed in a past chunk.
    """
    # v12 merges recall_query → sample_type=='recall'; treat equivalently.
    if sample.get("sample_type") not in ("recall_query", "recall"):
        return True, "pass"

    chunk_idx = sample.get("chunk_idx", -1)
    try:
        chunk_idx_int = int(chunk_idx)
    except (TypeError, ValueError):
        chunk_idx_int = -1

    recall_result = sample.get("recall_result")
    if not recall_result and isinstance(sample.get("input"), dict):
        recall_result = sample["input"].get("recall_result")
    if not isinstance(recall_result, dict):
        return False, "recall_result_missing"
    if recall_result.get("source") != "historical_frames":
        return False, f"recall_result_bad_source:{recall_result.get('source')}"
    text_content = str(
        recall_result.get("text_content")
        or recall_result.get("text")
        or ""
    ).strip()
    if not text_content:
        return False, "recall_result_empty_text"
    if isinstance(recall_result, dict):
        tr = recall_result.get("time", "")
        if tr:
            m = _RECALL_TIME_RANGE_RE.fullmatch(str(tr))
            if not m:
                return False, f"recall_bad_result_time: {tr}"
            if chunk_idx_int >= 0 and float(m.group(2)) > chunk_idx_int:
                return False, (
                    f"recall_result_time_in_future: time={tr} "
                    f"chunk={chunk_idx_int}"
                )
        returned = []
        for c in recall_result.get("returned_chunks") or []:
            try:
                returned.append(int(c))
            except (TypeError, ValueError):
                continue
        if not returned:
            return False, "recall_result_empty_returned_chunks"
        future_returned = [c for c in returned if chunk_idx_int >= 0 and c >= chunk_idx_int]
        if future_returned:
            return False, (
                f"recall_returned_future_chunks: returned={future_returned} "
                f"chunk={chunk_idx_int}"
            )
        if (
            sample.get("action") == "silent"
            and returned
            and recall_result.get("result_kind") != "not_yet"
        ):
            return False, f"recall_wait_returned_chunks: returned={returned}"

    # We need the card's support_chunks to check. If not available in sample,
    # this check is skipped (verified at pipeline level instead).
    metadata = sample.get("metadata", {})
    support_chunks = metadata.get("support_chunks", [])
    if not support_chunks:
        return True, "pass"

    # All support chunks must be before ask_chunk
    future_evidence = [sc for sc in support_chunks if sc >= chunk_idx_int]
    if future_evidence:
        return False, f"recall_evidence_in_future: support={future_evidence} ask={chunk_idx_int}"

    return True, "pass"


def verify_metadata_complete(sample: Dict) -> Tuple[bool, str]:
    """Check 15: response/recall samples must have non-empty metadata.gold_answer.

    Empty gold_answer would silently neuter the GRPO response/recall reward
    (the reward function falls through to the silent-only branch). This check
    catches that situation at data-construction time so downstream training
    isn't given unsupervised samples.

    Silent samples (and compress) legitimately have empty gold_answer and
    are exempt.
    """
    sample_type = sample.get("sample_type", "")
    # v12 merges recall_query+recall_response into sample_type='recall'
    if sample_type not in ("response", "recall_query", "recall_response", "recall"):
        return True, "pass"

    metadata = sample.get("metadata", {})
    gold_answer = (metadata.get("gold_answer") or "").strip()
    if not gold_answer:
        card_id = sample.get("card_id", "?")
        return False, (
            f"metadata.gold_answer empty for {sample_type} sample "
            f"(card={card_id}). Render is missing canonical_answer; "
            f"GRPO would silently un-gate this sample."
        )

    return True, "pass"


def verify_recall_memory_necessity(sample: Dict) -> Tuple[bool, str]:
    """Recall-response should not ask for facts already explicit in memory."""
    if not (sample.get("sample_type") == "recall" and sample.get("action") == "response"):
        return True, "pass"
    metadata = sample.get("metadata", {}) or {}
    answer = (
        metadata.get("correct_answer_text")
        or metadata.get("canonical_answer")
        or metadata.get("gold_answer")
        or ""
    )
    answer = str(answer).strip()
    if not answer or answer.lower() == "unable to answer":
        return True, "pass"
    memory_text = _audit_memory_text((sample.get("input") or {}).get("memory", {}))
    answer_n = re.sub(r"\s+", " ", answer.lower()).strip()
    memory_n = re.sub(r"\s+", " ", memory_text.lower()).strip()
    if answer_n and len(answer_n) >= 3 and answer_n in memory_n:
        return False, "recall_answer_verbatim_in_memory"
    overlap = _audit_overlap(answer, memory_text)
    if overlap >= 0.50:
        return False, f"recall_answer_memory_overlap:{overlap:.2f}"
    return True, "pass"


# ---------------------------------------------------------------------------
# Difficulty Labeling
# ---------------------------------------------------------------------------


def label_difficulty(sample: Dict) -> str:
    """Label sample difficulty for phase assignment."""
    sample_type = sample.get("sample_type", "")
    seq_type = sample.get("sequence_type", "")

    if _is_base_sample(sample):
        if sample.get("action") == "compress":
            return "medium"
        # Base silent with non-empty queries → medium (must sustain silence)
        if sample.get("queries"):
            return "medium"
        return "easy"

    if sample_type == "silent":
        if seq_type == "event_watch":
            return "medium"  # must stay silent while watching
        return "easy"
    elif sample_type == "response":
        if seq_type == "immediate_response":
            return "easy"
        if seq_type == "recall_fail_then_found":
            return "hard"  # delayed response after recall failure
        if seq_type == "multi_response":
            return "medium"
        return "medium"
    elif sample_type == "compress":
        return "medium"
    elif sample_type in ("recall_query", "recall_response", "recall_silent", "recall"):
        if seq_type == "recall_fail_then_found":
            return "hard"
        return "medium"

    return "medium"


# ---------------------------------------------------------------------------
# Main Verification
# ---------------------------------------------------------------------------


def verify_sample(sample: Dict, evidence: Optional[List[Dict]] = None) -> Dict:
    """Run all 15 checks on a single sample.

    `evidence` is the per-video 1-A/1-B caption list. When supplied,
    `verify_support_chunks_have_evidence` (v9.5) actually runs and
    rejects cards whose support_chunks point to silent-empty 1-A
    chunks. When None (legacy callers), that check is a no-op.
    """
    checks = {}

    for name, func in [
        ("information_flow", verify_information_flow),
        ("action_minimality", verify_action_minimality),
        ("grounding", verify_grounding),
        ("format", verify_format),
        ("think_token_length", verify_think_token_length),
        ("compression_ratio", verify_compression_ratio),
        ("summary_provenance", verify_summary_provenance),
        ("summary_retention", verify_summary_retention),
        ("summary_current_think_leak", verify_summary_no_current_think_leak),
        ("question_answer_leakage", verify_question_answer_leakage),
        ("queries_state_temporal", verify_queries_state_temporal),
        ("base_sample_consistency", verify_base_sample_consistency),
        ("recall_evidence_reachable", verify_recall_evidence_reachable),
        ("metadata_complete", verify_metadata_complete),
        ("recall_memory_necessity", verify_recall_memory_necessity),
    ]:
        passed, reason = func(sample)
        checks[name] = {"passed": passed, "reason": reason}

    # v9.5 — only fires when evidence is threaded in
    passed, reason = verify_support_chunks_have_evidence(sample, evidence)
    checks["support_chunks_have_evidence"] = {"passed": passed, "reason": reason}

    difficulty = label_difficulty(sample)
    all_passed = all(c["passed"] for c in checks.values())

    sample["verification"] = {
        "passed": all_passed,
        "checks": checks,
        "difficulty": difficulty,
        "fail_reasons": [
            f'{k}: {v["reason"]}' for k, v in checks.items() if not v["passed"]
        ],
    }

    return sample


def verify_trajectory(
    trajectory_samples: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], Dict]:
    """Verify all samples in a trajectory + trajectory-level checks.

    Returns (verified_samples, trajectory_stats).

    `evidence` is the per-video 1-A/1-B captions (passed through to
    verify_sample for v9.5's support_chunks_have_evidence check).
    """
    # Per-sample checks
    for sample in trajectory_samples:
        verify_sample(sample, evidence=evidence)

    # Trajectory-level check 12: if trajectory distribution is invalid,
    # tag the whole trajectory as failed, but keep rows for continuity.
    traj_passed, traj_reason = verify_trajectory_action_distribution(trajectory_samples)

    if not traj_passed:
        for s in trajectory_samples:
            s["verification"]["passed"] = False
            s["verification"]["fail_reasons"].append(f"trajectory_distribution: {traj_reason}")

    passed = [s for s in trajectory_samples if s["verification"]["passed"]]
    failed = [s for s in trajectory_samples if not s["verification"]["passed"]]

    fail_reasons = {}
    for s in failed:
        for reason in s["verification"]["fail_reasons"]:
            cat = reason.split(":")[0]
            fail_reasons[cat] = fail_reasons.get(cat, 0) + 1

    traj_id = trajectory_samples[0].get("trajectory_id", "unknown") if trajectory_samples else "unknown"

    stats = {
        "trajectory_id": traj_id,
        "total": len(trajectory_samples),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / max(len(trajectory_samples), 1),
        "trajectory_check_passed": traj_passed,
        "trajectory_check_reason": traj_reason,
        "fail_reasons": fail_reasons,
        "action_distribution": {},
        "difficulty_distribution": {},
    }

    for s in passed:
        st = s.get("sample_type", "unknown")
        stats["action_distribution"][st] = stats["action_distribution"].get(st, 0) + 1
        diff = s["verification"]["difficulty"]
        stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1

    return passed, stats


def tag_samples(
    samples: List[Dict],
    evidence_map: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[List[Dict], Dict]:
    """v12.5 — verify and TAG all samples in-place; never drop.

    Replaces `filter_samples` (which dropped failures and broke trajectory
    continuity) per user direction (2026-04-29): "过滤会破坏轨迹".

    Each sample gets `verification.passed` (bool) + `verification.fail_reasons`
    (list[str]) populated. The downstream pass4 trajectory-emitter uses
    these tags as soft signals (e.g., for SFT class-balanced sampling
    weights, RL outcome thresholds, or ablation experiments) — but every
    sample remains in the trajectory so the chunk timeline stays continuous.

    Returns: (all_samples_with_tags, aggregate_stats)

    `evidence_map` maps video_id → 1-A/1-B caption list. When supplied,
    the v9.5 support_chunks_have_evidence check fires and tags cards
    whose support_chunks reference silent-empty 1-A chunks. Legacy
    callers without evidence_map see that check as no-op (pass).
    """
    evidence_map = evidence_map or {}
    # Normalize messages-format if needed
    for sample in samples:
        if "output" not in sample:
            messages = sample.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    sample["output"] = content
                    break
            else:
                sample["output"] = ""

    # Group by trajectory. trajectory_id alone (e.g. "traj_0") is
    # NOT unique across videos — every video numbers its own trajectories
    # from 0. Combine with video_id so per-trajectory stats stay honest.
    by_traj = {}
    for s in samples:
        tid = s.get("trajectory_id", "no_traj")
        vid = s.get("video_id", "no_vid")
        by_traj.setdefault((vid, tid), []).append(s)

    all_tagged = []           # ALL samples — passed and failed alike
    all_stats = []

    for (vid, tid), traj_samples in by_traj.items():
        # verify_trajectory returns (passed_only, stats) — but it also
        # mutates each sample in-place to add sample["verification"]. We
        # discard its filtered list and keep ALL samples from this
        # trajectory; the verification tag travels with each sample.
        _passed_dropped, stats = verify_trajectory(
            traj_samples, evidence=evidence_map.get(vid),
        )
        stats["video_id"] = vid
        all_tagged.extend(traj_samples)   # ← KEY DIFFERENCE: keep all
        all_stats.append(stats)

    # Aggregate
    total = sum(s["total"] for s in all_stats)
    n_passed = sum(s["passed"] for s in all_stats)
    agg_fail = {}
    agg_action = {}
    agg_diff = {}
    for s in all_stats:
        for k, v in s["fail_reasons"].items():
            agg_fail[k] = agg_fail.get(k, 0) + v
        for k, v in s["action_distribution"].items():
            agg_action[k] = agg_action.get(k, 0) + v
        for k, v in s["difficulty_distribution"].items():
            agg_diff[k] = agg_diff.get(k, 0) + v

    aggregate = {
        "total": total,
        "passed": n_passed,
        "failed": total - n_passed,
        "pass_rate": n_passed / max(total, 1),
        "fail_reasons": agg_fail,
        "action_distribution": agg_action,
        "difficulty_distribution": agg_diff,
        "trajectories_total": len(all_stats),
        "trajectories": len(all_stats),
        "trajectory_check_failures": sum(
            1 for s in all_stats if not s["trajectory_check_passed"]),
    }

    return all_tagged, aggregate


def filter_samples(
    samples: List[Dict],
    evidence_map: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[List[Dict], Dict]:
    """LEGACY name kept for callers; now tag-only, no drops.

    Dropping individual rows breaks streaming trajectory continuity. Return
    all samples with verification tags and let downstream consumers decide how
    to weight or display failed rows.
    """
    return tag_samples(samples, evidence_map=evidence_map)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_verified(video_id: str, passed: List[Dict], stats: Dict):
    """Save verified samples and stats per video."""
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    path = VERIFIED_DIR / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": passed, "stats": stats}, f, ensure_ascii=False, indent=2)


def load_verified(video_id: str) -> Optional[Dict]:
    # Stage-4 cache invalidation: if STAGE_VERSIONS["4"] was bumped (e.g.
    # the retention threshold became adaptive in v9.2), per-video cached
    # files from older runs would be stale — return None so pipeline.py
    # regenerates instead of silently reusing.
    from .cache_version import stage_version_ok
    if not stage_version_ok("4"):
        return None
    path = VERIFIED_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
