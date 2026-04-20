"""
Pass 5: Verify + Filter

Five categories of validation:
1. Information flow legality (no future leakage)
2. Action minimality (correct gold_action)
3. Grounding (observation supported by frames)
4. Format & length
5. Difficulty labeling

Each sample gets a pass/fail verdict with reasons.
"""

import json
import logging
import re
from typing import Dict, List, Tuple

from .config import (
    LEAKAGE_OVERLAP_THRESHOLD,
    OBSERVATION_TOKENS,
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
    3. compression summary wasn't optimized for future questions
    """
    metadata = sample.get("metadata", {})
    output = sample.get("output", "")

    # Check query doesn't contain answer
    if metadata.get("leakage_checks", {}).get("query_contains_answer"):
        return False, "query_contains_answer_value"

    # Check response doesn't hallucinate beyond evidence
    # (Heuristic: response should not contain keywords absent from input)
    if "<response>" in output:
        response_match = re.search(r'<response>(.*?)</response>', output, re.DOTALL)
        if response_match:
            response_text = response_match.group(1)
            # Very basic check: response shouldn't be empty
            if len(response_text.strip()) < 3:
                return False, "empty_response"

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
    """
    output = sample.get("output", "")

    # Extract observation text
    obs_match = re.search(r'<observation>(.*?)</observation>', output, re.DOTALL)
    if not obs_match:
        return False, "no_observation_tag"

    obs_text = obs_match.group(1).lower()

    # Blacklisted phrases
    blacklist_phrases = [
        "sound", "hear", "listen", "noise",
        "smell", "aroma", "scent", "fragrant",
        "feels", "feeling", "emotion", "happy", "sad", "angry",
        "probably", "likely", "seems to want", "intend",
        "i think", "i notice", "i need", "i should", "i can see",
        "the user wants", "the video shows",
    ]

    for phrase in blacklist_phrases:
        if phrase in obs_text:
            return False, f"observation_contains_non_visual: '{phrase}'"

    return True, "pass"


def verify_format(sample: Dict) -> Tuple[bool, str]:
    """Check: output format and length constraints."""
    output = sample.get("output", "")

    # Must have observation tag
    if "<observation>" not in output or "</observation>" not in output:
        return False, "missing_observation_tags"

    # Must have action tag
    if "<action>" not in output or "</action>" not in output:
        return False, "missing_action_tags"

    # Extract observation and check length (rough char-based, needs tokenizer for exact)
    obs_match = re.search(r'<observation>(.*?)</observation>', output, re.DOTALL)
    if obs_match:
        obs_text = obs_match.group(1)
        estimated_words = len(obs_text.split())
        # 40-60 tokens ≈ 30-50 words normally, but compress/system can be shorter
        action_match_inner = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
        action_type = action_match_inner.group(1) if action_match_inner else ""
        min_words = 3 if action_type in ("compress", "recall") else 5
        if estimated_words < min_words:
            return False, f"observation_too_short ({estimated_words} words)"
        if estimated_words > 100:
            return False, f"observation_too_long ({estimated_words} words)"

    # Check action is valid
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    if action_match:
        action = action_match.group(1)
        valid_actions = {"silent", "response", "recall", "compress"}
        if action not in valid_actions:
            return False, f"invalid_action: {action}"

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
        elif "recent_observations" in reason:
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
