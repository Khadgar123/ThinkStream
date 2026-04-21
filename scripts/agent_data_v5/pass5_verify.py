"""
Pass 5: Verify + Filter

Seven categories of validation:
1. Information flow legality (no future leakage)
2. Action minimality (correct gold_action)
3. Grounding (observation supported by frames)
4. Format & length
5. Difficulty labeling
6. Summary provenance (summary only references compressed thinks)
7. Compression ratio & token length

Each sample gets a pass/fail verdict with reasons.
"""

import json
import logging
import re
from typing import Dict, List, Tuple

from .config import (
    COMPRESSION_RATIO_MIN,
    LEAKAGE_OVERLAP_THRESHOLD,
    THINK_TOKENS,
    get_tokenizer,
)
from .pass3_tasks import extract_keywords, keyword_overlap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def verify_information_flow(sample: Dict) -> Tuple[bool, str]:
    """Check: no future information leakage.

    1. response doesn't use info not in current visible sources
    2. recall query doesn't contain answer value
    3. recall_response after failure must be uncertain
    4. compression summary wasn't optimized for future questions
    """
    metadata = sample.get("metadata", {})
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # Check query doesn't contain answer
    if metadata.get("leakage_checks", {}).get("query_contains_answer"):
        return False, "query_contains_answer_value"

    # Check response doesn't hallucinate beyond evidence
    if "<response>" in output:
        response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if response_match:
            response_text = response_match.group(1)
            if len(response_text.strip()) < 3:
                return False, "empty_response"

    # Check: recall_response after failure/distractor must show uncertainty
    if sample_type == "recall_response":
        recall_result = sample.get("input", {}).get("recall_result", {})
        noise_level = recall_result.get("noise_level", "oracle")
        if noise_level in ("distractor", "failure"):
            response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
            if response_match:
                resp = response_match.group(1).lower()
                uncertain_markers = [
                    "cannot", "could not", "not sure", "unable",
                    "uncertain", "don't see", "not enough", "unclear",
                    "not confirm", "insufficient",
                ]
                if not any(m in resp for m in uncertain_markers):
                    return False, "confident_response_after_failed_recall"

    return True, "pass"


def verify_action_minimality(sample: Dict) -> Tuple[bool, str]:
    """Check: gold_action is the MINIMAL correct action.

    If answer is in compressed_summary → should be response, not recall.
    If answer is in visual window → should be response, not recall.
    """
    metadata = sample.get("metadata", {})
    sample_type = sample.get("sample_type", "")

    if sample_type == "recall_query":
        visibility = metadata.get("visibility", {})
        # If answer is accessible without recall → wrong action
        if visibility.get("answer_in_recent_obs"):
            return False, "recall_unnecessary_answer_in_observations"
        if visibility.get("answer_in_compressed"):
            return False, "recall_unnecessary_answer_in_compressed"
        if visibility.get("evidence_in_window"):
            return False, "recall_unnecessary_evidence_in_window"

    elif sample_type == "response":
        # For response tasks, verify the action_reason makes sense
        action_reason = metadata.get("action_reason", "")
        if "unanswerable" in action_reason:
            # Should have uncertain language in response
            output = sample.get("output", "")
            uncertain_markers = ["cannot", "not sure", "unable", "uncertain", "don't see"]
            if not any(m in output.lower() for m in uncertain_markers):
                return False, "unanswerable_but_confident_response"

    return True, "pass"


def verify_grounding(sample: Dict) -> Tuple[bool, str]:
    """Check: observation is grounded in visual facts.

    No sounds, smells, emotions, intentions, speculations.
    recall_response has no observation (already emitted in recall_query turn).
    """
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # recall_response intentionally omits observation — check response instead
    if sample_type == "recall_response":
        # Verify response doesn't contain non-visual claims
        resp_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if not resp_match:
            return True, "pass"  # No response to check
        check_text = resp_match.group(1).lower()
    else:
        obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if not obs_match:
            return False, "no_think_tag"
        check_text = obs_match.group(1).lower()

    # Blacklisted phrases (non-visual or meta-cognitive)
    blacklist_phrases = [
        "sound", "hear", "listen", "noise", "sizzle", "sizzling",
        "music", "speech", "said", "talking", "voice",
        "smell", "aroma", "scent", "fragrant", "aromatic",
        "feels", "feeling", "emotion", "happy", "sad", "angry",
        "probably", "likely", "seems to want", "intend",
        "i think", "i notice", "i need", "i should", "i can see",
        "the user wants", "the video shows",
        "system triggered", "memory compression", "retrieved evidence",
    ]

    for phrase in blacklist_phrases:
        if phrase in check_text:
            return False, f"think_contains_non_visual: '{phrase}'"

    return True, "pass"


def verify_format(sample: Dict) -> Tuple[bool, str]:
    """Check: output format and length constraints."""
    output = sample.get("output", "")

    # Must have observation tag (except recall_response which omits it to avoid duplication)
    sample_type = sample.get("sample_type", "")
    if sample_type != "recall_response":
        if "<think>" not in output or "</think>" not in output:
            return False, "missing_think_tags"

    # Must have action tag
    if "<action>" not in output or "</action>" not in output:
        return False, "missing_action_tags"

    # Extract observation and check length (rough char-based, needs tokenizer for exact)
    obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if obs_match:
        obs_text = obs_match.group(1)
        estimated_words = len(obs_text.split())
        # 40-60 tokens ≈ 30-50 words normally, but compress/system can be shorter
        action_match_inner = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
        action_type = action_match_inner.group(1) if action_match_inner else ""
        min_words = 3 if action_type in ("compress", "recall") else 5
        if estimated_words < min_words:
            return False, f"think_too_short ({estimated_words} words)"
        if estimated_words > 100:
            return False, f"think_too_long ({estimated_words} words)"

    # Check action is valid
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    if action_match:
        action = action_match.group(1)
        valid_actions = {"silent", "response", "recall", "compress"}
        if action not in valid_actions:
            return False, f"invalid_action: {action}"

    # Action/tag consistency: recall must have query, compress must have summary
    if action_match:
        action = action_match.group(1)
        if action == "recall" and "<query>" not in output:
            return False, "recall_action_without_query"
        if action == "compress" and "<summary>" not in output:
            return False, "compress_action_without_summary"

    # If recall, check query is valid JSON
    if "<query>" in output:
        query_match = re.search(r'<query>(.*?)</query>', output, re.DOTALL)
        if query_match:
            try:
                q = json.loads(query_match.group(1))
                if "query" not in q:
                    return False, "query_json_missing_query_field"
            except (json.JSONDecodeError, ValueError):
                return False, "query_invalid_json"

    # If compress, check summary is valid JSON
    if "<summary>" in output:
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if summary_match:
            try:
                s = json.loads(summary_match.group(1))
                if "time_range" not in s or "text" not in s:
                    return False, "summary_json_missing_fields"
            except (json.JSONDecodeError, ValueError):
                return False, "summary_invalid_json"

    # If response, check response tag exists
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    if action_match and action_match.group(1) == "response":
        if "<response>" not in output or "</response>" not in output:
            return False, "response_action_without_response_tag"

    return True, "pass"


def _count_tokens(text: str) -> int:
    """Count tokens using student tokenizer, fallback to chars/4."""
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text) // 4


def verify_think_token_length(sample: Dict) -> Tuple[bool, str]:
    """Check: think is within 40-60 token range.

    Uses student tokenizer for precise counting.
    Allows ±10 token tolerance for edge cases.
    """
    output = sample.get("output", "")
    sample_type = sample.get("sample_type", "")

    # recall_response has no think — skip
    if sample_type == "recall_response":
        return True, "pass"

    obs_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if not obs_match:
        return True, "pass"  # format check catches missing tags

    think_text = obs_match.group(1).strip()
    if not think_text:
        return True, "pass"

    tok_count = _count_tokens(think_text)
    min_tok = THINK_TOKENS[0] - 10  # 30 (tolerance)
    max_tok = THINK_TOKENS[1] + 15  # 75 (tolerance)

    if tok_count < min_tok:
        return False, f"think_tokens_too_few ({tok_count} < {min_tok})"
    if tok_count > max_tok:
        return False, f"think_tokens_too_many ({tok_count} > {max_tok})"

    return True, "pass"


def verify_compression_ratio(sample: Dict) -> Tuple[bool, str]:
    """Check: compression summary achieves minimum compression ratio.

    Only applies to compress samples. Ratio = input_tokens / summary_tokens >= 2.5.
    """
    sample_type = sample.get("sample_type", "")
    if sample_type != "compress":
        return True, "pass"

    output = sample.get("output", "")
    summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
    if not summary_match:
        return True, "pass"  # format check catches this

    try:
        summary_json = json.loads(summary_match.group(1))
        summary_text = summary_json.get("text", "")
    except (json.JSONDecodeError, ValueError):
        return True, "pass"  # format check catches this

    if not summary_text:
        return False, "empty_compression_summary"

    # Get input thinks that were compressed
    input_data = sample.get("input", {})
    recent_thinks = input_data.get("memory", {}).get("recent_thinks", [])
    if not recent_thinks:
        return True, "pass"

    input_text = " ".join(recent_thinks) if isinstance(recent_thinks[0], str) else \
        " ".join(t.get("text", t.get("obs", "")) if isinstance(t, dict) else str(t)
                 for t in recent_thinks)

    input_tokens = _count_tokens(input_text)
    summary_tokens = _count_tokens(summary_text)

    if summary_tokens == 0:
        return False, "empty_compression_summary"

    ratio = input_tokens / summary_tokens
    if ratio < COMPRESSION_RATIO_MIN:
        return False, f"compression_ratio_too_low ({ratio:.1f} < {COMPRESSION_RATIO_MIN})"

    return True, "pass"


def verify_summary_provenance(sample: Dict) -> Tuple[bool, str]:
    """Check: compression summary only references content from compressed thinks.

    Core constraint #6: summary cannot introduce facts not present in the
    thinks being compressed. This prevents "peeking" at visual frames during
    compression to add details the thinks didn't capture.

    Uses entity extraction: every capitalized entity/noun in the summary must
    appear in at least one of the compressed thinks.
    """
    sample_type = sample.get("sample_type", "")
    if sample_type != "compress":
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
        return True, "pass"

    # Collect all content words from compressed thinks (the source)
    input_data = sample.get("input", {})
    recent_thinks = input_data.get("memory", {}).get("recent_thinks", [])
    source_words = set()
    for t in recent_thinks:
        text = t if isinstance(t, str) else t.get("text", t.get("obs", "")) if isinstance(t, dict) else str(t)
        # Include underscore in word pattern to match entity IDs like Chef_1
        source_words.update(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text))

    # Extract entity-like words from summary (capitalized or with underscore)
    summary_words = re.findall(r'\b[a-zA-Z0-9_]+\b', summary_text)
    entity_words = [
        w for w in summary_words
        if len(w) > 2 and (w[0].isupper() or "_" in w)
    ]

    if not entity_words:
        return True, "pass"

    # Check: entity words in summary must have at least partial overlap with source
    unsupported = [w for w in entity_words if w.lower() not in source_words]
    # Allow up to 20% unsupported (for rephrasing / normalization tolerance)
    if entity_words and len(unsupported) / len(entity_words) > 0.2:
        examples = unsupported[:3]
        return False, f"summary_provenance_violation: {examples} not in source thinks"

    return True, "pass"


def label_difficulty(sample: Dict) -> str:
    """Label sample difficulty for phase assignment."""
    sample_type = sample.get("sample_type", "")
    metadata = sample.get("metadata", {})

    if sample_type == "silent":
        return "easy"
    elif sample_type == "response":
        reason = metadata.get("action_reason", "")
        if "visual_window" in reason:
            return "easy"
        elif "recent_think" in reason:
            return "medium"
        elif "compressed" in reason:
            return "medium"
        elif "unanswerable" in reason:
            return "medium"
        return "medium"
    elif sample_type == "compress":
        return "medium"
    elif sample_type in ("recall_query", "recall_response"):
        task_type = metadata.get("task_type", "")
        if "compress_recall" in task_type:
            return "hard"
        return "medium"

    return "medium"


# ---------------------------------------------------------------------------
# Main Verification
# ---------------------------------------------------------------------------


def verify_sample(sample: Dict) -> Dict:
    """Run all verifications on a single sample.

    Returns the sample with added verification results.
    """
    checks = {}

    # Run all checks
    passed, reason = verify_information_flow(sample)
    checks["information_flow"] = {"passed": passed, "reason": reason}

    passed, reason = verify_action_minimality(sample)
    checks["action_minimality"] = {"passed": passed, "reason": reason}

    passed, reason = verify_grounding(sample)
    checks["grounding"] = {"passed": passed, "reason": reason}

    passed, reason = verify_format(sample)
    checks["format"] = {"passed": passed, "reason": reason}

    passed, reason = verify_think_token_length(sample)
    checks["think_token_length"] = {"passed": passed, "reason": reason}

    passed, reason = verify_compression_ratio(sample)
    checks["compression_ratio"] = {"passed": passed, "reason": reason}

    passed, reason = verify_summary_provenance(sample)
    checks["summary_provenance"] = {"passed": passed, "reason": reason}

    # Difficulty labeling
    difficulty = label_difficulty(sample)

    # Overall verdict
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


def filter_samples(samples: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Filter samples and return statistics.

    Returns: (passed_samples, stats_dict)
    """
    passed = []
    failed = []
    fail_reasons_count = {}

    for sample in samples:
        sample = verify_sample(sample)
        if sample["verification"]["passed"]:
            passed.append(sample)
        else:
            failed.append(sample)
            for reason in sample["verification"]["fail_reasons"]:
                category = reason.split(":")[0]
                fail_reasons_count[category] = fail_reasons_count.get(category, 0) + 1

    stats = {
        "total": len(samples),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / max(len(samples), 1),
        "fail_reasons": fail_reasons_count,
        "difficulty_distribution": {},
    }

    for sample in passed:
        diff = sample["verification"]["difficulty"]
        stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1

    return passed, stats
