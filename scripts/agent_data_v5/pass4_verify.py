"""
Pass 4: Verify + Filter

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

Each sample gets a pass/fail verdict with reasons.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _memory_item_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", item.get("obs", str(item)))
    return str(item)


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


def verify_information_flow(sample: Dict) -> Tuple[bool, str]:
    """Check 1: No future information leakage."""
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
                if not re.fullmatch(r'[A-D]', resp_text):
                    return False, f"mc_response_not_single_letter: '{resp_text[:30]}'"
                gold = (metadata.get("gold_answer", "") or "").strip().upper()
                if gold and resp_text != gold:
                    return False, f"mc_response_mismatch: got '{resp_text}' want '{gold}'"
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


def verify_action_minimality(sample: Dict) -> Tuple[bool, str]:
    """Check 2: Action is the minimal correct action for the availability."""
    sample_type = sample.get("sample_type", "")
    seq_type = sample.get("sequence_type", "")

    # Base samples always silent/compress — correct by construction
    if _is_base_sample(sample):
        return True, "pass"

    # For recall_query: check that the sample actually needs recall
    # (availability was in_history_only, verified by sequence_type)
    if sample_type == "recall_query":
        if seq_type not in ("recall_success", "recall_fail_then_found"):
            return False, f"recall_query_in_non_recall_sequence: {seq_type}"

    # For response in immediate_response: verify it's not recall sequence
    if sample_type == "response" and seq_type == "immediate_response":
        # This is correct — answer was available without recall
        pass

    # Legacy metadata-based checks (backward compat)
    metadata = sample.get("metadata", {})
    if sample_type == "recall_query":
        visibility = metadata.get("visibility", {})
        if visibility.get("answer_in_recent_obs"):
            return False, "recall_unnecessary_answer_in_observations"
        if visibility.get("answer_in_compressed"):
            return False, "recall_unnecessary_answer_in_compressed"
        if visibility.get("evidence_in_window"):
            return False, "recall_unnecessary_evidence_in_window"

    return True, "pass"


def verify_grounding(sample: Dict) -> Tuple[bool, str]:
    """Check 3: Think is grounded in visual facts."""
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # recall_response has no think — check response instead
    if sample_type == "recall_response":
        resp_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if not resp_match:
            return True, "pass"
        check_text = resp_match.group(1).lower()
    else:
        obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if not obs_match:
            # Base compress samples may have empty think
            if _is_base_sample(sample) and sample.get("action") == "compress":
                return True, "pass"
            return False, "no_think_tag"
        check_text = obs_match.group(1).lower()

    # v9.4 — narrowed blacklist (was 33 phrases, dropped to ~17). Removed:
    #   "i think" / "i notice" / "i can see" — first-person observational
    #     phrasing is normal English narration, not a grounding violation.
    #   "probably" / "likely" — epistemic hedging is fine when describing
    #     genuine visual ambiguity ("the dish is probably soup based on the
    #     bowl shape"); rejecting these dropped 1.2k legit thinks in batch1.
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
        "music", "speech", "talking", "voice",
        "smell", "aroma", "scent", "fragrant", "aromatic",
        # True affect leaks (model is not a person, has no emotions)
        "emotion", "happy", "sad", "angry",
        # Speculative-intent leaks (the model shouldn't infer wishes)
        "seems to want", "intend",
        # Meta-language about the system / dataset (breaks 4th wall)
        "the user wants", "the video shows",
        "system triggered", "memory compression", "retrieved evidence",
    ]

    for phrase in blacklist_phrases:
        if phrase in check_text:
            return False, f"think_contains_non_visual: '{phrase}'"

    return True, "pass"


def verify_format(sample: Dict) -> Tuple[bool, str]:
    """Check 4: Output format and tag consistency."""
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
    # Margins: teacher tends to overshoot, so widen on the high side.
    min_tok = max(15, THINK_TOKENS[0] - 15)   # 25 by default
    max_tok = THINK_TOKENS[1] + 30            # 130 by default — covers p99 of teacher

    if tok_count < min_tok:
        return False, f"think_tokens_too_few ({tok_count} < {min_tok})"
    if tok_count > max_tok:
        return False, f"think_tokens_too_many ({tok_count} > {max_tok})"

    return True, "pass"


def verify_compression_ratio(sample: Dict) -> Tuple[bool, str]:
    """Check 6: Compression summary achieves minimum ratio."""
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

    entity_words = [
        w for w in re.findall(r'\b[a-zA-Z0-9_]+\b', summary_text)
        if len(w) > 2 and (w[0].isupper() or "_" in w)
    ]

    if not entity_words:
        return True, "pass"

    unsupported = [w for w in entity_words if w.lower() not in source_words]
    has_visual = sample.get("metadata", {}).get("has_visual_context", False)
    # v9.1: relax thresholds. With visual_context, 397B legitimately refines
    # entity details (correct color/count) that may not be verbatim in thinks.
    # Old: 0.4/0.2 → New: 0.5/0.3.
    max_ratio = 0.5 if has_visual else 0.3

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
    if n_unique <= 3:
        return 0.34   # any 1 missing on a 3-item source is acceptable
    if n_unique <= 8:
        return 0.50   # current default — keep strict for short
    if n_unique <= 15:
        return 0.45
    return 0.40       # long lists: 40% retention is realistic


def verify_summary_retention(sample: Dict) -> Tuple[bool, str]:
    """Check 8: Summary retains key information from source.

    Threshold is now adaptive (see `_retention_threshold`). We also
    de-duplicate the proper-noun signal a bit better — random caps from
    OCR overlays shouldn't inflate the key-item set.
    """
    if sample.get("sample_type") != "compress":
        return True, "pass"

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
    # Proper-nouns / entity tokens. Skip ALL-CAPS words (likely OCR overlay
    # noise like "SAUSAGE" or "STREET") which inflate key-items spuriously
    # and pull retention rate down.
    for w in re.findall(r'\b[A-Z][a-zA-Z0-9_]{2,}\b', source_combined):
        if not w.isupper():
            key_items.append(w.lower())
    for num in re.findall(r'\b\d+\.?\d*\b', source_combined):
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
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            return False, f"queries_state[{i}] is not a dict"
        if "question" not in q:
            return False, f"queries_state[{i}] missing 'question' field"
        if "answers" not in q:
            return False, f"queries_state[{i}] missing 'answers' field"
        if not isinstance(q["answers"], list):
            return False, f"queries_state[{i}]['answers'] is not a list"

    return True, "pass"


def verify_trajectory_action_distribution(
    trajectory_samples: List[Dict],
) -> Tuple[bool, str]:
    """Check 12: Trajectory has reasonable action distribution.

    Checks at the trajectory level (not per-sample):
    - At least 1 response or recall sample (not all silent)
    - Silent should not exceed an adaptive threshold:
        * 95% if any placement is event_watch (legit long waits)
        * 92% if any placement is multi_response (silent gaps OK)
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
            silent_threshold = 95.0
        elif "multi_response" in seq_types:
            silent_threshold = 92.0
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
    if sample.get("sample_type") != "recall_query":
        return True, "pass"

    card_id = sample.get("card_id", "")
    chunk_idx = sample.get("chunk_idx", -1)

    # We need the card's support_chunks to check. If not available in sample,
    # this check is skipped (verified at pipeline level instead).
    metadata = sample.get("metadata", {})
    support_chunks = metadata.get("support_chunks", [])
    if not support_chunks:
        return True, "pass"

    # All support chunks must be before ask_chunk
    future_evidence = [sc for sc in support_chunks if sc >= chunk_idx]
    if future_evidence:
        return False, f"recall_evidence_in_future: support={future_evidence} ask={chunk_idx}"

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
    if sample_type not in ("response", "recall_query", "recall_response"):
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
    elif sample_type in ("recall_query", "recall_response", "recall_silent"):
        if seq_type == "recall_fail_then_found":
            return "hard"
        return "medium"

    return "medium"


# ---------------------------------------------------------------------------
# Main Verification
# ---------------------------------------------------------------------------


def verify_sample(sample: Dict) -> Dict:
    """Run all 14 checks on a single sample."""
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
    ]:
        passed, reason = func(sample)
        checks[name] = {"passed": passed, "reason": reason}

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
) -> Tuple[List[Dict], Dict]:
    """Verify all samples in a trajectory + trajectory-level checks.

    Returns (verified_samples, trajectory_stats).
    """
    # Per-sample checks
    for sample in trajectory_samples:
        verify_sample(sample)

    # Trajectory-level check 12: if trajectory distribution is invalid,
    # drop the ENTIRE trajectory (not just individual samples)
    traj_passed, traj_reason = verify_trajectory_action_distribution(trajectory_samples)

    if not traj_passed:
        # Entire trajectory is invalid — drop all samples
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


def filter_samples(samples: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Filter all samples through verification.

    Groups by trajectory_id, runs per-trajectory checks,
    then aggregates stats.

    Returns: (passed_samples, aggregate_stats)
    """
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

    all_passed = []
    all_stats = []

    for (vid, tid), traj_samples in by_traj.items():
        passed, stats = verify_trajectory(traj_samples)
        stats["video_id"] = vid
        all_passed.extend(passed)
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
        # Number of distinct (video_id, trajectory_id) buckets that ran
        # through verification. Old field "trajectories" used to count
        # only unique trajectory_id strings, which collapsed identical
        # IDs across videos to ~MAX_TRAJECTORIES_PER_VIDEO. Keep the old
        # name as an alias for backward compat but the meaningful stat
        # is `trajectories_total`.
        "trajectories_total": len(all_stats),
        "trajectories": len(all_stats),
        "trajectory_check_failures": sum(
            1 for s in all_stats if not s["trajectory_check_passed"]),
    }

    return all_passed, aggregate


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
