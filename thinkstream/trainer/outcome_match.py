"""Form-aware liberal outcome matchers — single source of truth.

This module is the SHARED matching logic for ThinkStream's RL reward,
SFT eval, and OVOBench / our_val eval adapters. Keeping all three in
sync prevents the classic train/eval reward gap (RL rewards "Yes." but
eval judges it wrong because the eval-side matcher requires exact "yes").

5 answer_form variants (matches scripts/agent_data_v5/pass3a_cards.py):

    multiple_choice  → match_mcq_answer       letter / text / gold fallback
    binary           → match_binary           polarity (yes/no/true/false, +zh)
    number           → match_number           regex extract + tolerance
    short_exact      → match_short_exact      article-stripped substring
    descriptive      → match_descriptive      bidirectional substring

Public entry point:
    score_outcome_by_form(model_answer, *, options, correct_option,
                          gold_answer, answer_form) -> float (0.0/1.0)

The dispatcher applies anti-spam guards (empty / >1000 chars → 0).

Usage in eval:
    from thinkstream.trainer.outcome_match import (
        score_outcome_by_form, match_mcq_answer,
    )

Usage in RL reward (verl/recipe_thinkstream/thinkstream.py):
    Re-exports under the leading-underscore names for backward compat.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional


__all__ = [
    "normalize_answer",
    "match_mcq_answer",
    "match_binary",
    "match_number",
    "match_short_exact",
    "match_descriptive",
    "score_outcome_by_form",
    "binary_polarity",
    "extract_first_number",
    "strip_articles",
    # Constants — exposed for tests / extensions.
    "BINARY_YES", "BINARY_NO", "STOPWORDS",
]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lowercase + strip + collapse whitespace + drop trailing punctuation
    + drop leading bracket-like chars. Used by every per-form matcher."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.\,\!\?\:\;\)\]\}]+$", "", s)
    s = re.sub(r"^[\(\[\{]+", "", s)
    return s


_OPTION_LABEL_RE = re.compile(r"^\s*[A-Z][\).:]\s*")


def _strip_option_label(s: str) -> str:
    return _OPTION_LABEL_RE.sub("", str(s or "")).strip()


# ---------------------------------------------------------------------------
# multiple_choice
# ---------------------------------------------------------------------------
def match_mcq_answer(
    model_answer: str,
    options: List[str],
    correct_option: Any,
    gold_answer: str = "",
) -> bool:
    """Liberal MCQ answer match. 4 strategies, short-circuit on first hit:

    1. Leading character is the correct letter (e.g., "C", "C.", "(C)",
       "C: option text"). Rejects "Cake" because the second char is alpha.
    2. Model output equals the correct option text, OR the correct
       option text appears as a substring of the model output.
       (One-way: ma in correct_text would let "b" match "ta**b**le".)
    3. Any option exactly matches the model output (or option text of
       length ≥ 4 appears in model output) — must be the correct one.
    4. gold_answer text fallback (some datasets use free text).
       Require len ≥ 2 to avoid single-char gold false positives.

    correct_option may be 0-indexed int OR letter A-Z.
    """
    if not model_answer:
        return False
    ma = normalize_answer(model_answer)
    if not ma:
        return False

    # Resolve correct option's letter + text.
    correct_idx: Optional[int] = None
    if isinstance(correct_option, int):
        correct_idx = int(correct_option)
    elif isinstance(correct_option, str):
        co = correct_option.strip().upper()
        if len(co) == 1 and "A" <= co <= "Z":
            correct_idx = ord(co) - ord("A")
        else:
            try:
                correct_idx = int(co)
            except ValueError:
                correct_idx = None
    correct_letter = (
        chr(ord("A") + correct_idx) if correct_idx is not None
        and 0 <= correct_idx < 26 else ""
    ).lower()
    correct_text = ""
    if correct_idx is not None and 0 <= correct_idx < len(options):
        correct_text = normalize_answer(_strip_option_label(options[correct_idx]))

    # Strategy 1: leading-letter match.
    leading = ma.lstrip("([").lstrip()
    if leading and correct_letter and leading[0] in "abcd":
        if len(leading) == 1 or not leading[1].isalpha():
            if leading[0] != correct_letter:
                return False
            return True

    # Strategy 2: equality / option-text-in-model-output (one-way only).
    if correct_text and (ma == correct_text or correct_text in ma):
        return True

    # Strategy 3: any option's text matches.
    for i, opt in enumerate(options):
        on = normalize_answer(_strip_option_label(opt))
        if on and (ma == on or (len(on) >= 4 and on in ma)):
            return i == correct_idx

    # Strategy 4: gold_answer text fallback.
    if gold_answer:
        ga = normalize_answer(_strip_option_label(gold_answer))
        if ga and len(ga) >= 2 and (ma == ga or ga in ma):
            return True
        if ga and len(ga) < 2 and ma == ga:
            return True

    return False


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------
BINARY_YES = {"yes", "y", "true", "1", "t", "是", "对"}
BINARY_NO = {"no", "n", "false", "0", "f", "否", "不"}


def binary_polarity(s: str) -> Optional[bool]:
    """Return True/False if `s` is a binary yes/no, else None.
    First whitespace-separated token, then strip its trailing punctuation."""
    if not s:
        return None
    norm = normalize_answer(s)
    if not norm:
        return None
    first = re.sub(r"[^\w]+$", "", norm.split()[0]) if norm.split() else ""
    if first in BINARY_YES:
        return True
    if first in BINARY_NO:
        return False
    return None


def match_binary(model_answer: str, gold_answer: str) -> bool:
    """yes/no/true/false matching with normalization. Falls through to
    short_exact equality so unusual gold values ("yeah" / "nope") still
    work conservatively."""
    ma_pol = binary_polarity(model_answer)
    ga_pol = binary_polarity(gold_answer)
    if ma_pol is None or ga_pol is None:
        return normalize_answer(model_answer) == normalize_answer(gold_answer)
    return ma_pol == ga_pol


# ---------------------------------------------------------------------------
# number
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_number(s: str) -> Optional[float]:
    """Pull the first numeric token (int or decimal) from `s`."""
    if not s:
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except (ValueError, TypeError):
        return None


def match_number(
    model_answer: str,
    gold_answer: str,
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 1e-6,
) -> bool:
    """Numeric match with tolerance. Default exact-equal on integers;
    bump rel_tol for noisy domains. Accepts trailing units / leading
    currency: "100" / "100." / "$100" / "100 mph"."""
    ma_n = extract_first_number(model_answer)
    ga_n = extract_first_number(gold_answer)
    if ma_n is None or ga_n is None:
        return False
    if abs(ma_n - ga_n) <= abs_tol:
        return True
    if rel_tol > 0 and ga_n != 0:
        return abs(ma_n - ga_n) / abs(ga_n) <= rel_tol
    return False


# ---------------------------------------------------------------------------
# short_exact
# ---------------------------------------------------------------------------
STOPWORDS = {"the", "a", "an", "of", "is", "are", "was", "were"}


def strip_articles(s: str) -> str:
    """Drop leading article-like stopwords. 'the apple' → 'apple'."""
    tokens = s.split()
    while tokens and tokens[0] in STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens)


def match_short_exact(model_answer: str, gold_answer: str) -> bool:
    """Entity / short-exact match: normalize, drop leading articles,
    accept bidirectional substring. Conservative: gold len ≥ 2 to
    avoid single-letter false positives."""
    ma = strip_articles(normalize_answer(model_answer))
    ga = strip_articles(normalize_answer(gold_answer))
    if not ma or not ga:
        return False
    if ma == ga:
        return True
    if len(ga) < 2:
        return ma == ga
    return ga in ma or ma in ga


# ---------------------------------------------------------------------------
# descriptive
# ---------------------------------------------------------------------------
def match_descriptive(model_answer: str, gold_answer: str) -> bool:
    """Free-text descriptive — bidirectional substring on normalized strings.
    Plug a real LLM judge here for production-grade scoring."""
    ma = normalize_answer(model_answer)
    ga = normalize_answer(gold_answer)
    if not ma or not ga:
        return False
    return ma == ga or ga in ma or ma in ga


# ---------------------------------------------------------------------------
# Form-aware dispatcher (the one place to call from RL / SFT eval / OVOBench)
# ---------------------------------------------------------------------------
def score_outcome_by_form(
    model_answer: str,
    *,
    options: Optional[List[str]] = None,
    correct_option: Any = "",
    gold_answer: str = "",
    answer_form: str = "",
) -> float:
    """Form-aware liberal outcome scoring. Returns 1.0 / 0.0.

    Dispatches by `answer_form`:
      - multiple_choice / mc           → match_mcq_answer
      - binary / yes_no                → match_binary
      - number / numeric               → match_number
      - short_exact / entity / literal → match_short_exact
      - descriptive (default)          → match_descriptive

    Anti-hacking: empty answer → 0; len > 1000 chars → 0
    (mirrors v12_rewards.compute_outcome_reward_v12 conventions).
    """
    if not model_answer or not str(model_answer).strip():
        return 0.0
    if len(str(model_answer)) > 1000:
        return 0.0

    af = (answer_form or "").lower()
    opts = options or []

    if af in ("multiple_choice", "mc") and opts:
        return 1.0 if match_mcq_answer(
            model_answer, opts, correct_option, gold_answer,
        ) else 0.0

    if af in ("binary", "yes_no"):
        return 1.0 if match_binary(model_answer, gold_answer) else 0.0

    if af in ("number", "numeric"):
        return 1.0 if match_number(model_answer, gold_answer) else 0.0

    if af in ("short_exact", "entity", "literal"):
        return 1.0 if match_short_exact(model_answer, gold_answer) else 0.0

    return 1.0 if match_descriptive(model_answer, gold_answer) else 0.0
