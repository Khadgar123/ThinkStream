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


def _parse_time_range_from_memory_line(item) -> Tuple[int, int]:
    """Parse [start-end] prefix from a recent_thinks memory line."""
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
    """Return textual content from a memory item or formatted memory line."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", item.get("obs", str(item)))
    return str(item)


def _compressed_source_texts(sample: Dict) -> List[str]:
    """Return only the source memory items actually selected for compression.

    The verifier must not use all input recent_thinks as the provenance source
    when the gold range is a subset. Otherwise a summary could illegally copy
    facts from neighboring, uncompressed thinks and still pass provenance.
    """
    metadata = sample.get("metadata", {})
    input_data = sample.get("input", {})
    memory = input_data.get("memory", {})

    # Summary-of-summaries merge: source is compressed segments, not recent_thinks.
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

    # Fallback for old artifacts that did not store compressed_chunks/range.
    return selected if selected else [_memory_item_text(t) for t in recent_thinks]


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

    # Get ONLY the input thinks/segments that were actually compressed.
    source_texts = _compressed_source_texts(sample)
    if not source_texts:
        return True, "pass"

    input_text = " ".join(source_texts)

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

    # Collect content words from the exact compressed source range only.
    source_texts = _compressed_source_texts(sample)
    source_words = set()
    for text in source_texts:
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


def verify_summary_no_current_think_leak(sample: Dict) -> Tuple[bool, str]:
    """Check: compress summary does not contain facts from the current think.

    In autoregressive output, summary attends to the preceding <think>.
    If summary contains entity words unique to the current think (not in the
    compressed range), the model learned to "peek" at its own current output
    to enrich the summary — which breaks provenance at inference time.
    """
    sample_type = sample.get("sample_type", "")
    if sample_type != "compress":
        return True, "pass"

    output = sample.get("output", "")

    # Extract current think text
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if not think_match:
        return True, "pass"
    think_text = think_match.group(1).strip()

    # Extract summary text
    summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
    if not summary_match:
        return True, "pass"
    try:
        summary_json = json.loads(summary_match.group(1))
        summary_text = summary_json.get("text", "")
    except (json.JSONDecodeError, ValueError):
        return True, "pass"

    if not think_text or not summary_text:
        return True, "pass"

    # Get source range words (what the summary CAN reference)
    source_texts = _compressed_source_texts(sample)
    source_words = set()
    for text in source_texts:
        source_words.update(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text))

    # Find entity words in current think that are NOT in the compressed range
    think_words = set(re.findall(r'\b[a-zA-Z0-9_]+\b', think_text))
    think_unique = {
        w for w in think_words
        if len(w) > 3 and (w[0].isupper() or "_" in w) and w.lower() not in source_words
    }

    if not think_unique:
        return True, "pass"

    # Check if any of these unique-to-think entities leaked into summary
    summary_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', summary_text))
    leaked = {w for w in think_unique if w.lower() in summary_words}

    if leaked:
        return False, f"summary_contains_current_think_entities: {list(leaked)[:3]}"

    return True, "pass"


def verify_question_answer_leakage(sample: Dict) -> Tuple[bool, str]:
    """Check that the user-facing question does not reveal the gold answer.

    Two checks:
    1. Exact substring: gold_answer (≥5 chars) appears verbatim in question
    2. Keyword coverage: all meaningful answer keywords appear in question
    """
    metadata = sample.get("metadata", {})
    question = metadata.get("question") or sample.get("input", {}).get("user_input", "")
    answer = metadata.get("gold_answer", "")
    if not question or not answer:
        return True, "pass"

    q_lower = question.lower()
    a_lower = answer.lower().strip()

    # Check 1: exact substring
    if len(a_lower) >= 5 and a_lower in q_lower:
        return False, "question_contains_gold_answer_string"

    # Check 2: all meaningful answer keywords present in question
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

    passed, reason = verify_summary_no_current_think_leak(sample)
    checks["summary_current_think_leak"] = {"passed": passed, "reason": reason}

    passed, reason = verify_question_answer_leakage(sample)
    checks["question_answer_leakage"] = {"passed": passed, "reason": reason}

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
    """Filter samples (per-timestep or multi-turn conversations).

    Detects format automatically:
    - If sample has "messages" key → multi-turn conversation → verify_conversation
    - If sample has "output" key → per-timestep sample → verify_sample (legacy)

    Returns: (passed_samples, stats_dict)
    """
    passed = []
    failed = []
    fail_reasons_count = {}

    for sample in samples:
        if "messages" in sample:
            # Multi-turn conversation format
            sample = verify_conversation(sample)
        else:
            # Legacy per-timestep format
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
        "action_distribution": {},
    }

    # Count action types across all conversations
    for sample in passed:
        for action_type, count in sample.get("verification", {}).get("action_counts", {}).items():
            stats["action_distribution"][action_type] = \
                stats["action_distribution"].get(action_type, 0) + count

    return passed, stats


def verify_conversation(conversation: Dict) -> Dict:
    """Verify a multi-turn conversation (one video = one sample).

    Checks each assistant turn for grounding, format, and then checks
    conversation-level properties (action distribution, compression events).
    """
    messages = conversation.get("messages", [])
    issues = []
    action_counts = {"silent": 0, "response": 0, "recall": 0, "compress": 0}

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # Check format: must have valid action
        action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
        if not action_match:
            # Post-recall response may not have action tag in some formats
            if "<response>" in content:
                action_counts["response"] = action_counts.get("response", 0) + 1
                continue
            issues.append(f"turn_{i}: missing_action_tag")
            continue

        action = action_match.group(1)
        if action not in ("silent", "response", "recall", "compress"):
            issues.append(f"turn_{i}: invalid_action '{action}'")
            continue

        action_counts[action] = action_counts.get(action, 0) + 1

        # Check think grounding (no sounds/smells/emotions)
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).lower()
            blacklist = ["sound", "hear", "sizzle", "sizzling", "smell", "aroma",
                         "emotion", "happy", "sad", "i think", "i notice", "i need",
                         "system triggered", "memory compression"]
            for phrase in blacklist:
                if phrase in think_text:
                    issues.append(f"turn_{i}: think_non_visual '{phrase}'")
                    break

        # Check recall has query
        if action == "recall" and "<query>" not in content:
            issues.append(f"turn_{i}: recall_without_query")

        # Check compress has summary
        if action == "compress" and "<summary>" not in content:
            issues.append(f"turn_{i}: compress_without_summary")

    all_passed = len(issues) == 0

    conversation["verification"] = {
        "passed": all_passed,
        "action_counts": action_counts,
        "fail_reasons": issues,
        "num_assistant_turns": sum(action_counts.values()),
    }

    return conversation
