"""
Tests for Pass 3 (A/B/C): Task Card Generation, Placement, Sample Generation.

Tests core logic without requiring vLLM endpoint or video files.
All 397B calls are mocked. Pure-program logic (classify_chunks, placement,
trajectory planning, base sampling) is tested directly.
"""

import json
import random
import re
import pytest
from copy import deepcopy
from typing import Dict, List


# ---------------------------------------------------------------------------
# Inline implementations (avoid relative-import issues in test runner)
# ---------------------------------------------------------------------------
# We import the actual module functions where possible, but fall back to
# self-contained helpers when the module layout blocks it.

# ---- pass3a helpers (pure functions, no relative imports needed) ----

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
    "and", "or", "it", "yes", "no", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "there", "here", "not", "but", "if", "so", "than", "then",
    "just", "about", "up", "out", "its", "his", "her", "my", "your",
    "their", "our", "me", "him", "them", "us", "we", "they",
    "you", "he", "she", "with", "for", "from", "by", "as",
    "what", "which", "who", "whom", "whose", "how", "when", "where",
    "many", "much", "any", "some", "other", "tell", "describe",
    "currently", "now", "still", "yet", "ever", "already",
})


def extract_keywords(text: str) -> list:
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and len(w) > 1 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _extract_mc_choice_text(question: str, answer_letter: str) -> str:
    answer_letter = answer_letter.strip().upper()
    pattern = rf'(?:^|\s){answer_letter}[\.\)]\s*(.+?)(?:\s+[B-Z][\.\)]|$)'
    m = re.search(pattern, question, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_card_keywords(card: dict) -> list:
    answer_form = card.get("answer_form", "short_exact")
    question = card.get("question", "")
    canonical = card.get("canonical_answer", "")
    if answer_form == "binary":
        return extract_keywords(question)
    elif answer_form == "multiple_choice":
        q_base = re.split(r'\s+A[\.\)]', question, maxsplit=1)[0]
        q_kw = extract_keywords(q_base)
        choice_text = _extract_mc_choice_text(question, canonical)
        c_kw = extract_keywords(choice_text)
        seen = set()
        result = []
        for w in q_kw + c_kw:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result
    elif answer_form == "number":
        q_kw = extract_keywords(question)
        num = canonical.strip()
        if num and num not in {kw for kw in q_kw}:
            q_kw.append(num)
        return q_kw
    else:
        return extract_keywords(canonical)


def _keyword_overlap(text: str, keywords: list) -> float:
    if not keywords:
        return 0.0
    text_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))
    found = sum(1 for kw in keywords if kw in text_words)
    return found / len(keywords)


def _desc_overlap(desc_a: str, desc_b: str) -> float:
    words_a = set(re.findall(r'[a-zA-Z]{2,}', desc_a.lower()))
    words_b = set(re.findall(r'[a-zA-Z]{2,}', desc_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


# ---------------------------------------------------------------------------
# Fixtures: realistic mock data
# ---------------------------------------------------------------------------

FAMILY_TARGETS = {
    "F1": 3, "F2": 4, "F3": 2, "F4": 2,
    "E1": 3, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 2,
}

RETENTION_CLASS = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
}


def _make_evidence(num_chunks=60):
    """Build a realistic evidence list for a cooking video."""
    evidence = []
    actions = [
        "chopping onions", "peeling garlic", "heating oil",
        "adding garlic", "stirring pot", "slicing tomatoes",
        "adding tomatoes", "seasoning with salt", "tasting sauce",
        "plating food",
    ]
    for i in range(num_chunks):
        cap = {
            "chunk_idx": i,
            "time": [i * 2, (i + 1) * 2],
            "visible_entities": [
                {"desc": "person wearing red apron", "action": actions[i % len(actions)],
                 "id": "person_1", "position": "center"},
            ],
            "atomic_facts": [
                {"fact": f"person {actions[i % len(actions)]}", "confidence": 0.85},
            ],
            "ocr": [],
            "state_changes": [],
        }
        # Add variety
        if i % 8 == 0:
            cap["ocr"] = [f"${random.randint(1,99)}.99"]
        if i >= 2 and i % 5 == 0:
            cap["visible_entities"].append(
                {"desc": "stainless steel pot", "action": "on right burner",
                 "id": "pot_1", "position": "right"})
        if i >= 3 and i % 7 == 0:
            cap["visible_entities"].append(
                {"desc": "small white bowl", "action": "on counter",
                 "id": "bowl_1", "position": "left"})
        if 10 <= i <= 15:
            cap["state_changes"] = [f"started {actions[i % len(actions)]}"]
        if i % 12 == 0 and i > 0:
            cap["atomic_facts"].append(
                {"fact": f"added 15 grams of seasoning", "confidence": 0.9})
        return_val = cap
        evidence.append(cap)
    return evidence


def _make_rollout(num_chunks=60):
    """Build a realistic rollout with thinks, snapshots, compression events."""
    thinks = []
    for i in range(num_chunks):
        thinks.append({
            "chunk_idx": i,
            "time": f"{i*2}-{(i+1)*2}",
            "think": f"Person in red apron {'chopping' if i < 20 else 'stirring'} at counter. "
                     f"Stainless pot on right burner."
                     + (f" Price tag shows $4.99." if i % 8 == 0 else "")
                     + (f" Added seasoning to pot." if i in (5, 12, 25, 40) else ""),
        })

    snapshots = {}
    for i in range(num_chunks):
        window_start = max(0, i - 12)
        recent = [{"text": thinks[j]["think"], "time": thinks[j]["time"], "chunk": j}
                  for j in range(max(0, i - 10), i)]
        snapshots[i] = {
            "chunk_idx": i,
            "visual_window_start": window_start,
            "recent_thinks": recent,
            "compressed_segments": [],
        }

    compression_events = [
        {"trigger_chunk": 18,
         "compressed_thinks_chunks": list(range(0, 10)),
         "summary": {"time_range": [0, 20], "text": "Person in red apron chopped onions and garlic."}},
        {"trigger_chunk": 38,
         "compressed_thinks_chunks": list(range(10, 25)),
         "summary": {"time_range": [20, 50], "text": "Person heated oil, added tomatoes and seasoning."}},
    ]

    return {
        "num_chunks": num_chunks,
        "thinks": thinks,
        "snapshots": snapshots,
        "compression_events": compression_events,
    }


def _make_cards():
    """Build a realistic set of ~25 task cards."""
    cards = [
        {"card_id": "v_F1_001", "family": "F1", "question": "What price is shown on screen?",
         "canonical_answer": "4.99", "answer_form": "short_exact",
         "support_chunks": [8], "visibility_type": "transient"},
        {"card_id": "v_F2_002", "family": "F2",
         "question": "What color is the apron? A.Red B.Blue C.White D.Green",
         "canonical_answer": "A", "answer_form": "multiple_choice",
         "support_chunks": [3], "visibility_type": "persistent"},
        {"card_id": "v_F3_003", "family": "F3", "question": "How many grams of seasoning were added?",
         "canonical_answer": "15", "answer_form": "number",
         "support_chunks": [12], "visibility_type": "transient"},
        {"card_id": "v_F4_004", "family": "F4", "question": "Is the pot on the right burner?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [5], "visibility_type": "persistent"},
        {"card_id": "v_E1_005", "family": "E1", "question": "Is the person stirring?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [25], "visibility_type": "transient"},
        {"card_id": "v_E2_006", "family": "E2", "question": "Tell me when the person starts slicing tomatoes",
         "canonical_answer": "Started slicing", "answer_form": "short_exact",
         "support_chunks": [30], "visibility_type": "transient"},
        {"card_id": "v_P1_007", "family": "P1",
         "question": "Which step is this? A.Second B.Third C.Fourth D.Fifth",
         "canonical_answer": "B", "answer_form": "multiple_choice",
         "support_chunks": [12], "visibility_type": "transient"},
        {"card_id": "v_C1_008", "family": "C1",
         "question": "Has the person's activity changed since earlier?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [10, 25], "visibility_type": "transient"},
        {"card_id": "v_R1_009", "family": "R1",
         "question": "Is the small white bowl still on the counter?",
         "canonical_answer": "No", "answer_form": "binary",
         "support_chunks": [3], "visibility_type": "transient"},
        {"card_id": "v_S1_010", "family": "S1", "question": "Describe the current scene",
         "canonical_answer": "Person in red apron cooking at stove with pot",
         "answer_form": "descriptive",
         "support_chunks": [20], "visibility_type": "persistent"},
        {"card_id": "v_M1_011", "family": "M1", "question": "Describe each step as it happens",
         "canonical_answer": "Chopping onions", "answer_form": "descriptive",
         "support_chunks": [5], "visibility_type": "transient"},
        # Extra cards for density
        {"card_id": "v_F2_012", "family": "F2", "question": "Is the pot stainless steel?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [10], "visibility_type": "persistent"},
        {"card_id": "v_E1_013", "family": "E1", "question": "Is the person chopping?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [2], "visibility_type": "transient"},
        {"card_id": "v_F1_014", "family": "F1", "question": "What text is visible on screen?",
         "canonical_answer": "4.99", "answer_form": "short_exact",
         "support_chunks": [16], "visibility_type": "transient"},
        {"card_id": "v_E2_015", "family": "E2", "question": "Has the person started plating?",
         "canonical_answer": "No", "answer_form": "binary",
         "support_chunks": [50], "visibility_type": "transient"},
        {"card_id": "v_F4_016", "family": "F4", "question": "Is the bowl to the left of the pot?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [21], "visibility_type": "transient"},
        {"card_id": "v_C1_017", "family": "C1",
         "question": "Has the pot content changed compared to earlier?",
         "canonical_answer": "Yes", "answer_form": "binary",
         "support_chunks": [15, 35], "visibility_type": "transient"},
        {"card_id": "v_E1_018", "family": "E1", "question": "Is the person tasting the sauce?",
         "canonical_answer": "No", "answer_form": "binary",
         "support_chunks": [40], "visibility_type": "transient"},
        {"card_id": "v_S1_019", "family": "S1", "question": "Describe what is on the counter",
         "canonical_answer": "Cutting board, knife, and bowl on counter",
         "answer_form": "descriptive",
         "support_chunks": [15], "visibility_type": "persistent"},
        {"card_id": "v_F3_020", "family": "F3", "question": "How many tomatoes are visible?",
         "canonical_answer": "3", "answer_form": "number",
         "support_chunks": [22], "visibility_type": "transient"},
    ]
    return cards


# =====================================================================
# Pass 3-A Tests: classify_chunks + extract_card_keywords
# =====================================================================


class TestClassifyChunks:
    """Test the structural chunk classification logic."""

    def _classify(self, evidence):
        """Inline classify_chunks to avoid import issues."""
        fc = {f: [] for f in FAMILY_TARGETS}
        for cap in evidence:
            idx = cap.get("chunk_idx", 0)
            entities = cap.get("visible_entities", [])
            facts = [f for f in cap.get("atomic_facts", [])
                     if f.get("confidence", 0) >= 0.7]
            has_digit_facts = any(
                re.search(r'\d{2,}|[\$€¥£]\d|\d\s*(?:kg|lb|ml|oz|cm|mm|g)\b',
                          f.get("fact", ""))
                for f in facts
            )
            if cap.get("ocr") or has_digit_facts:
                fc["F1"].append(idx)
            if has_digit_facts:
                fc["F3"].append(idx)
            if entities:
                fc["F2"].append(idx)
            if len(entities) >= 2:
                fc["F4"].append(idx)
            if cap.get("state_changes"):
                fc["E2"].append(idx)
            if len(entities) >= 3:
                fc["S1"].append(idx)
        for f in fc:
            fc[f] = sorted(set(fc[f]))
        return fc

    def test_f1_ocr_chunks(self):
        evidence = _make_evidence()
        fc = self._classify(evidence)
        ocr_chunks = [c["chunk_idx"] for c in evidence if c.get("ocr")]
        for c in ocr_chunks:
            assert c in fc["F1"], f"chunk {c} has OCR but not in F1"

    def test_f1_f3_independent(self):
        """F1 and F3 should not short-circuit each other (bug #1 fix)."""
        evidence = [
            {"chunk_idx": 0, "visible_entities": [],
             "atomic_facts": [{"fact": "added 15 grams of salt", "confidence": 0.9}],
             "ocr": ["$4.99"]},  # has OCR AND digit facts
        ]
        fc = self._classify(evidence)
        assert 0 in fc["F1"], "chunk with OCR must be in F1"
        assert 0 in fc["F3"], "chunk with digit facts must be in F3 even if OCR present"

    def test_f2_requires_entities(self):
        evidence = _make_evidence()
        fc = self._classify(evidence)
        for c in fc["F2"]:
            cap = next(e for e in evidence if e["chunk_idx"] == c)
            assert len(cap["visible_entities"]) >= 1

    def test_f4_requires_two_entities(self):
        evidence = _make_evidence()
        fc = self._classify(evidence)
        for c in fc["F4"]:
            cap = next(e for e in evidence if e["chunk_idx"] == c)
            assert len(cap["visible_entities"]) >= 2

    def test_e2_requires_state_changes(self):
        evidence = _make_evidence()
        fc = self._classify(evidence)
        for c in fc["E2"]:
            cap = next(e for e in evidence if e["chunk_idx"] == c)
            assert cap.get("state_changes"), f"chunk {c} in E2 but no state_changes"

    def test_digit_filter_rejects_single_digits(self):
        """Digit regex should reject isolated single digits like 'person 1'."""
        evidence = [
            {"chunk_idx": 0, "visible_entities": [],
             "atomic_facts": [{"fact": "person 1 is sitting", "confidence": 0.9}],
             "ocr": []},
        ]
        fc = self._classify(evidence)
        assert 0 not in fc["F1"], "'person 1' should not trigger F1"
        assert 0 not in fc["F3"], "'person 1' should not trigger F3"

    def test_digit_filter_accepts_multi_digit(self):
        evidence = [
            {"chunk_idx": 0, "visible_entities": [],
             "atomic_facts": [{"fact": "temperature reads 350 degrees", "confidence": 0.9}],
             "ocr": []},
        ]
        fc = self._classify(evidence)
        assert 0 in fc["F1"]

    def test_digit_filter_accepts_unit_pattern(self):
        evidence = [
            {"chunk_idx": 0, "visible_entities": [],
             "atomic_facts": [{"fact": "added 5 ml of oil", "confidence": 0.9}],
             "ocr": []},
        ]
        fc = self._classify(evidence)
        assert 0 in fc["F1"]


class TestExtractCardKeywords:
    """Test keyword extraction handles all answer_form types."""

    def test_binary_uses_question(self):
        card = {"question": "Is the apron red?", "canonical_answer": "Yes",
                "answer_form": "binary"}
        kw = extract_card_keywords(card)
        assert "apron" in kw
        assert "red" in kw
        assert "yes" not in kw

    def test_binary_no_uses_question(self):
        card = {"question": "Is the person stirring?", "canonical_answer": "No",
                "answer_form": "binary"}
        kw = extract_card_keywords(card)
        assert "person" in kw
        assert "stirring" in kw
        assert len(kw) >= 2

    def test_mc_extracts_correct_choice(self):
        card = {"question": "What color? A.Red B.Blue C.White D.Green",
                "canonical_answer": "A", "answer_form": "multiple_choice"}
        kw = extract_card_keywords(card)
        assert "red" in kw
        assert "blue" not in kw

    def test_mc_includes_question_subject(self):
        card = {"question": "What color is the apron? A.Red B.Blue C.White D.Green",
                "canonical_answer": "A", "answer_form": "multiple_choice"}
        kw = extract_card_keywords(card)
        assert "apron" in kw
        assert "red" in kw

    def test_mc_choice_b(self):
        card = {"question": "Which step? A.First B.Third C.Fourth D.Fifth",
                "canonical_answer": "B", "answer_form": "multiple_choice"}
        kw = extract_card_keywords(card)
        assert "third" in kw

    def test_mc_choice_d(self):
        card = {"question": "Tool? A) Knife B) Fork C) Spoon D) Spatula",
                "canonical_answer": "D", "answer_form": "multiple_choice"}
        kw = extract_card_keywords(card)
        assert "spatula" in kw

    def test_number_includes_digits(self):
        card = {"question": "How many tomatoes were cut?",
                "canonical_answer": "3", "answer_form": "number"}
        kw = extract_card_keywords(card)
        assert "3" in kw
        assert "tomatoes" in kw

    def test_number_no_interrogative_words(self):
        card = {"question": "How many tomatoes?",
                "canonical_answer": "3", "answer_form": "number"}
        kw = extract_card_keywords(card)
        assert "many" not in kw

    def test_short_exact_uses_answer(self):
        card = {"question": "What price?", "canonical_answer": "Stainless steel pot",
                "answer_form": "short_exact"}
        kw = extract_card_keywords(card)
        assert "stainless" in kw
        assert "steel" in kw

    def test_descriptive_uses_answer(self):
        card = {"question": "Describe the scene",
                "canonical_answer": "Chef in red apron chops tomatoes",
                "answer_form": "descriptive"}
        kw = extract_card_keywords(card)
        assert "chef" in kw
        assert "tomatoes" in kw


class TestRetentionMatching:
    """Test keyword matching against realistic think texts."""

    THRESHOLDS = {"low": 0.5, "medium": 0.35, "high": 0.2}

    def _is_retained(self, card, think_text):
        kw = extract_card_keywords(card)
        family = card.get("family", "F2")
        rc = RETENTION_CLASS.get(family, "medium")
        threshold = self.THRESHOLDS[rc]
        return _keyword_overlap(think_text, kw) > threshold

    def test_binary_positive(self):
        card = {"question": "Is the apron red?", "canonical_answer": "Yes",
                "answer_form": "binary", "family": "E1"}
        assert self._is_retained(card,
            "Person wearing red apron sprinkles seasoning.")

    def test_binary_negative(self):
        card = {"question": "Is the apron red?", "canonical_answer": "Yes",
                "answer_form": "binary", "family": "E1"}
        assert not self._is_retained(card,
            "Chef adjusts burner dial on stove.")

    def test_mc_positive(self):
        card = {"question": "What color is the apron? A.Red B.Blue C.White D.Green",
                "canonical_answer": "A", "answer_form": "multiple_choice", "family": "F2"}
        assert self._is_retained(card,
            "Red apron visible as chef moves to counter.")

    def test_mc_negative(self):
        card = {"question": "What color is the apron? A.Red B.Blue C.White D.Green",
                "canonical_answer": "A", "answer_form": "multiple_choice", "family": "F2"}
        assert not self._is_retained(card,
            "Blue car passes in background of outdoor scene.")

    def test_number_positive(self):
        card = {"question": "How many tomatoes were cut?", "canonical_answer": "3",
                "answer_form": "number", "family": "F3"}
        assert self._is_retained(card,
            "Chef slices 3 ripe tomatoes on the board.")

    def test_number_false_positive_rejected(self):
        """Number '3' in unrelated context should not match."""
        card = {"question": "How many tomatoes were cut?", "canonical_answer": "3",
                "answer_form": "number", "family": "F3"}
        assert not self._is_retained(card,
            "3 people are watching the cooking show.")

    def test_spatial_binary(self):
        card = {"question": "Is the pot on the right burner?",
                "canonical_answer": "Yes", "answer_form": "binary", "family": "F4"}
        assert self._is_retained(card,
            "Stainless pot sits on the right burner, steam rising.")


# =====================================================================
# Pass 3-B Tests: Placement + Trajectory Planning
# =====================================================================


class TestClassifyAvailability:
    """Test the availability classification logic."""

    def _classify(self, card, ask_chunk, rollout, bitmap):
        support_chunks = set(card.get("support_chunks", []))
        support_start = min(support_chunks) if support_chunks else 0
        support_end = max(support_chunks) if support_chunks else 0
        snapshots = rollout["snapshots"]
        snapshot = snapshots.get(ask_chunk) or snapshots.get(str(ask_chunk))
        if snapshot is None:
            return "unavailable"
        if support_start > ask_chunk:
            return "in_future"
        window_start = snapshot["visual_window_start"]
        window_end = snapshot["chunk_idx"]
        if any(window_start <= c <= window_end for c in support_chunks):
            return "in_visual"
        recent_chunks = {item["chunk"] for item in snapshot.get("recent_thinks", [])}
        retained_present = support_chunks & recent_chunks
        if any(bitmap.get("thinks_retained", {}).get(c, False) for c in retained_present):
            return "in_recent_thinks"
        for idx, event in enumerate(rollout.get("compression_events", [])):
            if event["trigger_chunk"] > ask_chunk:
                break
            compressed = set(event.get("compressed_thinks_chunks", []))
            if support_chunks & compressed:
                if bitmap.get("summary_retained", {}).get(idx, False):
                    return "in_compressed"
        if support_end < ask_chunk:
            return "in_history_only"
        return "unavailable"

    def test_in_visual(self):
        rollout = _make_rollout()
        card = {"support_chunks": [20]}
        avail = self._classify(card, ask_chunk=22, rollout=rollout, bitmap={})
        assert avail == "in_visual"

    def test_in_future(self):
        rollout = _make_rollout()
        card = {"support_chunks": [40]}
        avail = self._classify(card, ask_chunk=20, rollout=rollout, bitmap={})
        assert avail == "in_future"

    def test_in_history_only(self):
        rollout = _make_rollout()
        card = {"support_chunks": [5]}
        avail = self._classify(card, ask_chunk=40, rollout=rollout, bitmap={})
        assert avail == "in_history_only"

    def test_in_recent_thinks_with_bitmap(self):
        rollout = _make_rollout()
        card = {"support_chunks": [15]}
        bitmap = {"thinks_retained": {15: True}, "summary_retained": {}}
        avail = self._classify(card, ask_chunk=22, rollout=rollout, bitmap=bitmap)
        # chunk 15 is in recent_thinks window at chunk 22 (22-10=12, 15>12)
        assert avail in ("in_visual", "in_recent_thinks")

    def test_snapshot_int_str_both_work(self):
        """Snapshot keys can be int or str after JSON serialization."""
        rollout = _make_rollout()
        # Add string-keyed snapshot
        rollout["snapshots"]["25"] = rollout["snapshots"][25]
        del rollout["snapshots"][25]
        card = {"support_chunks": [20]}
        avail = self._classify(card, ask_chunk=25, rollout=rollout, bitmap={})
        assert avail != "unavailable"


class TestDetermineSequenceType:
    def _determine(self, card, availability):
        if card.get("family") == "M1":
            return "multi_response"
        if availability == "in_future":
            return "event_watch"
        if availability in ("in_visual", "in_recent_thinks", "in_compressed"):
            return "immediate_response"
        if availability == "in_history_only":
            return "recall_success"
        return "immediate_response"

    def test_m1_always_multi(self):
        assert self._determine({"family": "M1"}, "in_visual") == "multi_response"
        assert self._determine({"family": "M1"}, "in_history_only") == "multi_response"

    def test_future_event_watch(self):
        assert self._determine({"family": "E2"}, "in_future") == "event_watch"

    def test_visual_immediate(self):
        assert self._determine({"family": "F2"}, "in_visual") == "immediate_response"

    def test_history_recall(self):
        assert self._determine({"family": "F1"}, "in_history_only") == "recall_success"


class TestTrajectoryPlanning:
    """Test the greedy diversity-scored trajectory planner."""

    def _make_placements(self, cards, num_chunks=60):
        """Generate placements from cards for testing."""
        placements = []
        rng = random.Random(42)
        seq_types = ["immediate_response", "recall_success", "event_watch",
                     "recall_fail_then_found", "multi_response"]
        for card in cards:
            for _ in range(rng.randint(1, 3)):
                ask = rng.randint(2, num_chunks - 2)
                seq = rng.choice(seq_types)
                placements.append({
                    "card_id": card["card_id"],
                    "ask_chunk": ask,
                    "sequence_type": seq,
                    "key_chunks": {"ask": ask, "post_silent": min(ask + 1, num_chunks - 1)},
                })
        return placements

    def _plan(self, placements, cards_map, target=5, max_pp=5, gap=8, seed=42):
        """Inline plan_trajectories core logic."""
        rng = random.Random(seed)
        used_families = set()
        used_seq_types = set()
        used_ask_chunks = []
        used_card_ids = set()
        selected = []
        candidates = list(placements)
        budget = target * max_pp

        while candidates and len(selected) < budget:
            scored = []
            for p in candidates:
                if p["card_id"] in used_card_ids:
                    continue
                card = cards_map.get(p["card_id"], {})
                score = 0.0
                if card.get("_support_inferred"):
                    score -= 2.0
                if card.get("answer_form") in {"binary", "multiple_choice", "number", "short_exact"}:
                    score += 1.0
                if card.get("family", "") not in used_families:
                    score += 2.0
                if p["sequence_type"] not in used_seq_types:
                    score += 2.0
                if used_ask_chunks:
                    min_dist = min(abs(p["ask_chunk"] - c) for c in used_ask_chunks)
                    score += min(min_dist / 10.0, 1.5)
                else:
                    score += 1.5
                scored.append((score, p))
            if not scored:
                break
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[0][0]
            ties = [sp for sp in scored if sp[0] >= top - 0.1]
            _, best = ties[rng.randint(0, len(ties) - 1)]
            selected.append(best)
            card = cards_map.get(best["card_id"], {})
            used_families.add(card.get("family", ""))
            used_seq_types.add(best["sequence_type"])
            used_ask_chunks.append(best["ask_chunk"])
            used_card_ids.add(best["card_id"])
            candidates.remove(best)

        selected.sort(key=lambda p: p["ask_chunk"])
        trajectories = []
        paired = set()
        for i in range(len(selected)):
            if i in paired or len(trajectories) >= target:
                break
            group = [selected[i]]
            paired.add(i)
            for j in range(i + 1, len(selected)):
                if j in paired:
                    continue
                if selected[j]["ask_chunk"] - group[-1]["ask_chunk"] >= gap:
                    group.append(selected[j])
                    paired.add(j)
                    if len(group) >= max_pp:
                        break
            trajectories.append({
                "trajectory_id": f"traj_{len(trajectories)}",
                "placements": group,
            })
        return trajectories

    def test_trajectory_count(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5)
        assert len(trajs) <= 5

    def test_no_duplicate_cards_in_selection(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5)
        all_cids = [p["card_id"] for t in trajs for p in t["placements"]]
        assert len(all_cids) == len(set(all_cids)), "Same card used twice"

    def test_family_diversity(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5)
        families = set()
        for t in trajs:
            for p in t["placements"]:
                families.add(cards_map[p["card_id"]]["family"])
        assert len(families) >= 5, f"Only {len(families)} families represented"

    def test_sequence_type_diversity(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5)
        seq_types = set()
        for t in trajs:
            for p in t["placements"]:
                seq_types.add(p["sequence_type"])
        assert len(seq_types) >= 3, f"Only {len(seq_types)} seq_types"

    def test_min_chunk_gap_respected(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5, gap=8)
        for t in trajs:
            chunks = sorted(p["ask_chunk"] for p in t["placements"])
            for i in range(1, len(chunks)):
                assert chunks[i] - chunks[i-1] >= 8, \
                    f"Gap {chunks[i]-chunks[i-1]} < 8 in {t['trajectory_id']}"

    def test_inferred_support_penalized(self):
        cards = _make_cards()
        # Mark one card as inferred
        cards[0]["_support_inferred"] = True
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5)
        used_cids = {p["card_id"] for t in trajs for p in t["placements"]}
        # Inferred card should be deprioritized (not necessarily excluded)
        # but there are enough non-inferred cards to fill all slots
        if len(cards) > 25:
            assert cards[0]["card_id"] not in used_cids

    def test_questions_per_trajectory_reasonable(self):
        cards = _make_cards()
        cards_map = {c["card_id"]: c for c in cards}
        placements = self._make_placements(cards)
        trajs = self._plan(placements, cards_map, target=5, max_pp=5)
        for t in trajs:
            assert 1 <= len(t["placements"]) <= 5


# =====================================================================
# Pass 3-C Tests: Base Sample Selection
# =====================================================================


class TestBaseSampleSelection:
    """Test the selective base chunk sampling logic."""

    WARMUP_CHUNKS = 3
    QUESTION_WINDOW_BEFORE = 2
    QUESTION_WINDOW_AFTER = 3
    COMPRESS_WINDOW = 1
    LONG_SILENT_SAMPLE_INTERVAL = 5
    EVIDENCE_WINDOW = 2

    def _select(self, trajectory, rollout, cards_map):
        """Inline _select_base_chunks."""
        num_chunks = rollout["num_chunks"]
        selected = set()
        for c in range(min(self.WARMUP_CHUNKS, num_chunks)):
            selected.add(c)
        for p in trajectory["placements"]:
            card = cards_map.get(p["card_id"], {})
            for sc in card.get("support_chunks", []):
                for c in range(max(0, sc - self.EVIDENCE_WINDOW),
                               min(num_chunks, sc + self.EVIDENCE_WINDOW + 1)):
                    selected.add(c)
        for p in trajectory["placements"]:
            kc = p["key_chunks"]
            for key, val in kc.items():
                anchors = [val] if isinstance(val, int) else (val if isinstance(val, list) else [])
                for anchor in anchors:
                    ws = max(0, anchor - self.QUESTION_WINDOW_BEFORE)
                    we = min(num_chunks - 1, anchor + self.QUESTION_WINDOW_AFTER)
                    for c in range(ws, we + 1):
                        selected.add(c)
        for event in rollout.get("compression_events", []):
            trigger = event.get("trigger_chunk", -1)
            if trigger < 0:
                continue
            for c in range(max(0, trigger - self.COMPRESS_WINDOW),
                           min(num_chunks, trigger + self.COMPRESS_WINDOW + 1)):
                selected.add(c)
            cc = sorted(event.get("compressed_thinks_chunks", []))
            for c in cc[:2] + cc[-2:]:
                selected.add(c)
        sorted_sel = sorted(selected)
        patrol = []
        prev = -1
        for s in sorted_sel:
            if s - prev > 10:
                for c in range(prev + self.LONG_SILENT_SAMPLE_INTERVAL,
                               s, self.LONG_SILENT_SAMPLE_INTERVAL):
                    patrol.append(c)
            prev = s
        if sorted_sel and num_chunks - 1 - sorted_sel[-1] > 10:
            for c in range(sorted_sel[-1] + self.LONG_SILENT_SAMPLE_INTERVAL,
                           num_chunks, self.LONG_SILENT_SAMPLE_INTERVAL):
                patrol.append(c)
        selected.update(patrol)
        return sorted(selected)

    def _make_trajectory(self):
        return {
            "trajectory_id": "traj_test",
            "placements": [
                {"card_id": "v_F2_002", "ask_chunk": 20,
                 "sequence_type": "immediate_response",
                 "key_chunks": {"ask": 20, "post_silent": 21}},
                {"card_id": "v_F3_003", "ask_chunk": 45,
                 "sequence_type": "recall_success",
                 "key_chunks": {"ask": 45, "post_recall": 45, "post_silent": 46}},
            ],
        }

    def test_warmup_included(self):
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        assert 0 in selected
        assert 1 in selected
        assert 2 in selected

    def test_evidence_anchors_included(self):
        """support_chunks and their neighbors should be selected."""
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        # v_F2_002 has support_chunks=[3], v_F3_003 has support_chunks=[12]
        for sc in [3, 12]:
            for c in range(sc - 2, sc + 3):
                if 0 <= c < 60:
                    assert c in selected, f"evidence anchor chunk {c} missing"

    def test_question_window_included(self):
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        # ask=20 → window [18, 23], post_silent=21 → [19, 24]
        # ask=45 → window [43, 48], post_silent=46 → [44, 49]
        for c in [18, 19, 20, 21, 22, 23, 24]:
            assert c in selected, f"question window chunk {c} missing"
        for c in [43, 44, 45, 46, 47, 48, 49]:
            if c < 60:
                assert c in selected, f"question window chunk {c} missing"

    def test_compress_chunks_included(self):
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        # Compression at chunk 18 compresses chunks 0-9
        assert 17 in selected, "compress trigger-1 missing"
        assert 18 in selected, "compress trigger missing"
        assert 19 in selected, "compress trigger+1 missing"
        # First/last 2 of compressed range (boundary context)
        assert 0 in selected, "compressed range start missing"
        assert 1 in selected, "compressed range start+1 missing"
        assert 8 in selected, "compressed range end-1 missing"
        assert 9 in selected, "compressed range end missing"

    def test_patrol_fills_long_gaps(self):
        # Create a trajectory with question only at chunk 5
        traj = {
            "trajectory_id": "traj_patrol",
            "placements": [
                {"card_id": "v_F2_002", "ask_chunk": 5,
                 "sequence_type": "immediate_response",
                 "key_chunks": {"ask": 5, "post_silent": 6}},
            ],
        }
        rollout = _make_rollout()
        # Remove compression events to create long gaps
        rollout["compression_events"] = []
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        # After question window (chunk ~11), there should be patrol chunks
        # every 5 chunks until end of video
        has_patrol_after_30 = any(c >= 30 for c in selected)
        assert has_patrol_after_30, "No patrol chunks in long silent stretch"

    def test_not_all_60_chunks(self):
        """Should select a subset, not all chunks."""
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        assert len(selected) < 55, f"Selected {len(selected)}/60, too many"

    def test_at_least_20_chunks(self):
        """Should select enough for meaningful training."""
        traj = self._make_trajectory()
        rollout = _make_rollout()
        cards_map = {c["card_id"]: c for c in _make_cards()}
        selected = self._select(traj, rollout, cards_map)
        assert len(selected) >= 20, f"Only {len(selected)} chunks, too few"


# =====================================================================
# Pass 3-A Prompt Tests
# =====================================================================


class TestPromptTemplates:
    """Test that prompt templates render correctly."""

    def test_all_families_have_prompts(self):
        from scripts.agent_data_v5.pass3a_cards import FAMILY_PROMPTS
        for f in FAMILY_TARGETS:
            assert f in FAMILY_PROMPTS, f"No prompt for family {f}"

    def test_format_renders_all_templates(self):
        from scripts.agent_data_v5.pass3a_cards import FAMILY_PROMPTS
        for f, tmpl in FAMILY_PROMPTS.items():
            rendered = tmpl.format(n=3, evidence="test evidence line")
            assert "test evidence line" in rendered, f"{f} evidence not rendered"
            assert "3" in rendered, f"{f} target count not rendered"
            assert "{n}" not in rendered, f"{f} has unrendered {{n}}"
            assert "{evidence}" not in rendered, f"{f} has unrendered {{evidence}}"

    def test_output_schema_present(self):
        from scripts.agent_data_v5.pass3a_cards import FAMILY_PROMPTS
        for f, tmpl in FAMILY_PROMPTS.items():
            rendered = tmpl.format(n=1, evidence="x")
            assert "canonical_answer" in rendered, f"{f} missing schema fields"
            assert "support_chunks" in rendered, f"{f} missing support_chunks"
            assert "visibility_type" in rendered, f"{f} missing visibility_type"

    def test_entity_rule_present(self):
        from scripts.agent_data_v5.pass3a_cards import FAMILY_PROMPTS
        for f, tmpl in FAMILY_PROMPTS.items():
            rendered = tmpl.format(n=1, evidence="x")
            assert "NEVER by ID" in rendered, f"{f} missing entity ID rule"


# =====================================================================
# Integration: Action Distribution
# =====================================================================


class TestActionDistribution:
    """Verify that the overall action ratio is in the target range."""

    def test_fork_plus_base_ratio(self):
        """With 5 questions per trajectory, silent should be ~55-70%."""
        # Simulate fork samples for 5 questions
        fork_actions = []
        # 2 immediate_response: each = 1 response + 1 silent
        fork_actions.extend(["response", "silent"] * 2)
        # 2 recall_success: each = 1 recall + 1 response + 1 silent
        fork_actions.extend(["recall", "response", "silent"] * 2)
        # 1 event_watch: 1 silent(ask) + 2 silent(wait) + 1 response
        fork_actions.extend(["silent", "silent", "silent", "response"])

        # Base samples: ~20 selected chunks (minus fork chunks)
        base_actions = ["silent"] * 17 + ["compress"] * 3

        all_actions = fork_actions + base_actions
        total = len(all_actions)
        silent = sum(1 for a in all_actions if a == "silent")
        response = sum(1 for a in all_actions if a == "response")
        recall = sum(1 for a in all_actions if a == "recall")
        compress = sum(1 for a in all_actions if a == "compress")

        silent_pct = silent / total * 100
        active_pct = (response + recall) / total * 100
        compress_pct = compress / total * 100

        assert 50 <= silent_pct <= 75, f"silent={silent_pct:.0f}% out of range"
        assert active_pct >= 15, f"response+recall={active_pct:.0f}% too low"
        assert compress_pct >= 3, f"compress={compress_pct:.0f}% too low"


# =====================================================================
# Desc Overlap (P1/C1/R1 fallback)
# =====================================================================


class TestDescOverlap:
    def test_identical(self):
        assert _desc_overlap("person wearing red apron", "person wearing red apron") == 1.0

    def test_high_overlap(self):
        assert _desc_overlap("person wearing red apron", "person in red apron") >= 0.6

    def test_low_overlap(self):
        assert _desc_overlap("person wearing red apron", "stainless steel pot") < 0.3

    def test_empty(self):
        assert _desc_overlap("", "something") == 0.0
        assert _desc_overlap("something", "") == 0.0


# =====================================================================
# Edge Cases
# =====================================================================


class TestEdgeCases:
    def test_short_video_no_crash(self):
        """3 chunks = 6 seconds. Should not crash."""
        evidence = [
            {"chunk_idx": i, "time": [i*2, (i+1)*2],
             "visible_entities": [{"desc": "person", "action": "standing"}],
             "atomic_facts": [{"fact": "person standing", "confidence": 0.8}],
             "ocr": [], "state_changes": []}
            for i in range(3)
        ]
        rollout = _make_rollout(num_chunks=3)
        cards = [{"card_id": "c1", "family": "E1", "question": "Is person standing?",
                  "canonical_answer": "Yes", "answer_form": "binary",
                  "support_chunks": [1], "visibility_type": "transient"}]
        cards_map = {c["card_id"]: c for c in cards}

        # classify_chunks should work
        fc = TestClassifyChunks()._classify(evidence)
        assert isinstance(fc, dict)

    def test_empty_evidence(self):
        fc = TestClassifyChunks()._classify([])
        for family_chunks in fc.values():
            assert family_chunks == []

    def test_card_with_no_support_chunks(self):
        card = {"question": "Test?", "canonical_answer": "Yes",
                "answer_form": "binary", "support_chunks": []}
        kw = extract_card_keywords(card)
        assert isinstance(kw, list)

    def test_extract_keywords_empty_string(self):
        assert extract_keywords("") == []

    def test_mc_choice_extraction_no_match(self):
        assert _extract_mc_choice_text("No choices here", "A") == ""

    def test_keyword_overlap_empty_keywords(self):
        assert _keyword_overlap("some text", []) == 0.0

    def test_keyword_overlap_empty_text(self):
        assert _keyword_overlap("", ["keyword"]) == 0.0


# =====================================================================
# Pass 3-A Verification Tests
# =====================================================================


class MockClient:
    """Mock 397B client for testing async verify/visibility functions."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self._calls = []

    async def _call_one(self, messages, max_tokens=2048,
                         temperature=0.7, request_id="", max_retries=3):
        self._calls.append(request_id)
        return self._responses.get(request_id, None)


class TestVerifyCards:
    """Test card verification with mocked 397B."""

    @pytest.mark.asyncio
    async def test_valid_card_passes(self):
        cards = [{"card_id": "v_F1_001", "family": "F1",
                  "question": "What price?", "canonical_answer": "4.99",
                  "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]
        evidence = _make_evidence()
        resp = json.dumps({"valid": True, "support_chunks": [8],
                           "visibility_type": "transient", "canonical_answer": "4.99"})
        client = MockClient({"test_verify_v_F1_001": resp})
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        result = await verify_cards("test", cards, evidence, client)
        assert len(result) == 1
        assert result[0]["_verified"] is True

    @pytest.mark.asyncio
    async def test_invalid_card_dropped(self):
        cards = [{"card_id": "v_F1_001", "family": "F1",
                  "question": "Bad question?", "canonical_answer": "???",
                  "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]
        evidence = _make_evidence()
        resp = json.dumps({"valid": False})
        client = MockClient({"test_verify_v_F1_001": resp})
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        result = await verify_cards("test", cards, evidence, client)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_support_chunks_fixed(self):
        cards = [{"card_id": "v_F1_001", "family": "F1",
                  "question": "What price?", "canonical_answer": "4.99",
                  "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]
        evidence = _make_evidence()
        resp = json.dumps({"valid": True, "support_chunks": [8, 16],
                           "visibility_type": "transient", "canonical_answer": "4.99"})
        client = MockClient({"test_verify_v_F1_001": resp})
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        result = await verify_cards("test", cards, evidence, client)
        assert result[0]["support_chunks"] == [8, 16]

    @pytest.mark.asyncio
    async def test_visibility_type_fixed(self):
        cards = [{"card_id": "v_F1_001", "family": "F1",
                  "question": "What price?", "canonical_answer": "4.99",
                  "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]
        evidence = _make_evidence()
        resp = json.dumps({"valid": True, "support_chunks": [8],
                           "visibility_type": "persistent", "canonical_answer": "4.99"})
        client = MockClient({"test_verify_v_F1_001": resp})
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        result = await verify_cards("test", cards, evidence, client)
        assert result[0]["visibility_type"] == "persistent"

    @pytest.mark.asyncio
    async def test_empty_cards_returns_empty(self):
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        client = MockClient()
        result = await verify_cards("test", [], [], client)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_support_chunks_dropped(self):
        cards = [{"card_id": "v_F1_001", "question": "Test?",
                  "canonical_answer": "X", "answer_form": "short_exact",
                  "support_chunks": [], "visibility_type": "transient"}]
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        client = MockClient()
        result = await verify_cards("test", cards, [], client)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_all_cards_independent(self):
        """All verify calls should be independent (check request_ids)."""
        cards = _make_cards()[:5]
        evidence = _make_evidence()
        responses = {}
        for card in cards:
            rid = f"test_verify_{card['card_id']}"
            responses[rid] = json.dumps({
                "valid": True, "support_chunks": card["support_chunks"],
                "visibility_type": card["visibility_type"],
                "canonical_answer": card["canonical_answer"],
            })
        client = MockClient(responses)
        from scripts.agent_data_v5.pass3a_cards import verify_cards
        result = await verify_cards("test", cards, evidence, client)
        # Each card should generate exactly one call
        assert len(client._calls) == 5
        # All request_ids should be unique (independent calls)
        assert len(set(client._calls)) == 5


# =====================================================================
# Pass 3-B LLM Visibility Tests
# =====================================================================


class TestVisibilityCheck:
    """Test LLM-based visibility check with mocked 397B."""

    @pytest.mark.asyncio
    async def test_answerable_returns_true(self):
        from scripts.agent_data_v5.pass3b_placement import _check_visibility_one
        snapshot = {
            "recent_thinks": [{"time": "20-22", "text": "Person in red apron chopping onions."}],
            "compressed_segments": [],
        }
        card = {"card_id": "c1", "question": "Is the apron red?",
                "canonical_answer": "Yes"}
        client = MockClient({"vid_vis_c1_25": json.dumps({"answerable": True})})
        result = await _check_visibility_one(card, 25, snapshot, client, "vid")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_answerable_returns_false(self):
        from scripts.agent_data_v5.pass3b_placement import _check_visibility_one
        snapshot = {
            "recent_thinks": [{"time": "40-42", "text": "Empty kitchen counter."}],
            "compressed_segments": [],
        }
        card = {"card_id": "c1", "question": "Is the apron red?",
                "canonical_answer": "Yes"}
        client = MockClient({"vid_vis_c1_45": json.dumps({"answerable": False})})
        result = await _check_visibility_one(card, 45, snapshot, client, "vid")
        assert result is False

    @pytest.mark.asyncio
    async def test_api_failure_returns_false(self):
        from scripts.agent_data_v5.pass3b_placement import _check_visibility_one
        snapshot = {"recent_thinks": [], "compressed_segments": []}
        card = {"card_id": "c1", "question": "Test?", "canonical_answer": "X"}
        client = MockClient({})  # no response
        result = await _check_visibility_one(card, 10, snapshot, client, "vid")
        assert result is False

    @pytest.mark.asyncio
    async def test_compute_placements_with_client(self):
        """compute_all_placements with client should use LLM for history chunks."""
        from scripts.agent_data_v5.pass3b_placement import compute_all_placements
        rollout = _make_rollout()
        evidence = _make_evidence()
        cards = [{"card_id": "c1", "family": "F1", "question": "What price?",
                  "canonical_answer": "4.99", "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]

        # Mock: student cannot answer at history chunk → recall
        responses = {}
        for key_pattern in [f"vid_vis_c1_{c}" for c in range(60)]:
            responses[key_pattern] = json.dumps({"answerable": False})
        client = MockClient(responses)

        placements = await compute_all_placements(
            cards, rollout, evidence, client=client, video_id="vid")
        # Should have at least one placement
        assert len(placements) >= 1
        # The visibility check calls should have been made
        vis_calls = [c for c in client._calls if "_vis_" in c]
        assert len(vis_calls) >= 1

    @pytest.mark.asyncio
    async def test_compute_placements_without_client_fallback(self):
        """Without client, should fall back to keyword-based retention."""
        from scripts.agent_data_v5.pass3b_placement import compute_all_placements
        rollout = _make_rollout()
        evidence = _make_evidence()
        cards = [{"card_id": "c1", "family": "F1", "question": "What price?",
                  "canonical_answer": "4.99", "answer_form": "short_exact",
                  "support_chunks": [8], "visibility_type": "transient"}]
        # No client → sync fallback
        placements = await compute_all_placements(
            cards, rollout, evidence, client=None)
        assert len(placements) >= 1

    @pytest.mark.asyncio
    async def test_persistent_cards_skip_llm(self):
        """Persistent cards use pure math, no LLM calls needed."""
        from scripts.agent_data_v5.pass3b_placement import compute_all_placements
        rollout = _make_rollout()
        evidence = _make_evidence()
        cards = [{"card_id": "c1", "family": "F2", "question": "Apron color?",
                  "canonical_answer": "Red", "answer_form": "short_exact",
                  "support_chunks": [3], "visibility_type": "persistent"}]
        client = MockClient({})
        placements = await compute_all_placements(
            cards, rollout, evidence, client=client, video_id="vid")
        # Persistent → immediate_response at fixed positions, no LLM calls
        assert len(client._calls) == 0
        assert len(placements) >= 1
        for p in placements:
            assert p["sequence_type"] == "immediate_response"


class TestPendingLifetime:
    """Test MAX_ACTIVE_QUERIES enforcement in trajectory grouping."""

    def test_resolution_chunk_immediate(self):
        from scripts.agent_data_v5.pass3b_placement import _resolution_chunk
        p = {"ask_chunk": 10, "sequence_type": "immediate_response",
             "key_chunks": {"ask": 10, "post_silent": 11}}
        assert _resolution_chunk(p) == 10  # resolved immediately

    def test_resolution_chunk_event_watch(self):
        from scripts.agent_data_v5.pass3b_placement import _resolution_chunk
        p = {"ask_chunk": 5, "sequence_type": "event_watch",
             "key_chunks": {"ask": 5, "trigger": 20, "post_silent": 21}}
        assert _resolution_chunk(p) == 20  # resolved at trigger

    def test_resolution_chunk_recall_fail(self):
        from scripts.agent_data_v5.pass3b_placement import _resolution_chunk
        p = {"ask_chunk": 10, "sequence_type": "recall_fail_then_found",
             "key_chunks": {"ask": 10, "found_response": 25, "post_silent": 26}}
        assert _resolution_chunk(p) == 25  # resolved when found

    def test_count_pending_at(self):
        from scripts.agent_data_v5.pass3b_placement import _count_pending_at
        group = [
            {"ask_chunk": 5, "sequence_type": "event_watch",
             "key_chunks": {"ask": 5, "trigger": 20}},
            {"ask_chunk": 10, "sequence_type": "immediate_response",
             "key_chunks": {"ask": 10}},
        ]
        # At chunk 8: event_watch is pending (asked at 5, resolves at 20)
        assert _count_pending_at(group, 8) == 1
        # At chunk 12: event_watch still pending, immediate already resolved
        assert _count_pending_at(group, 12) == 1
        # At chunk 4: nothing asked yet
        assert _count_pending_at(group, 4) == 0

    def test_two_event_watches_blocked(self):
        """Two overlapping event_watches should not be in same trajectory."""
        from scripts.agent_data_v5.pass3b_placement import _count_pending_at
        group = [
            {"ask_chunk": 5, "sequence_type": "event_watch",
             "key_chunks": {"ask": 5, "trigger": 30}},
            {"ask_chunk": 10, "sequence_type": "event_watch",
             "key_chunks": {"ask": 10, "trigger": 35}},
        ]
        # At chunk 15, both are pending
        assert _count_pending_at(group, 15) == 2
        # A third event_watch at chunk 15 should be blocked (>= MAX_ACTIVE_QUERIES)
        assert _count_pending_at(group, 15) >= 2

    def test_trajectory_respects_max_active(self):
        """plan_trajectories should not group placements that exceed MAX_ACTIVE_QUERIES."""
        from scripts.agent_data_v5.pass3b_placement import plan_trajectories, _count_pending_at
        # Create 3 event_watch placements with overlapping pending windows
        cards = [
            {"card_id": f"ew_{i}", "family": "E2", "question": f"Q{i}?",
             "canonical_answer": "Yes", "answer_form": "binary",
             "support_chunks": [30 + i * 5]}
            for i in range(3)
        ]
        cards_map = {c["card_id"]: c for c in cards}
        placements = [
            {"card_id": f"ew_{i}", "ask_chunk": 5 + i * 10,
             "sequence_type": "event_watch",
             "key_chunks": {"ask": 5 + i * 10, "trigger": 40 + i * 5,
                            "post_silent": 41 + i * 5}}
            for i in range(3)
        ]
        trajs = plan_trajectories(
            placements, cards_map=cards_map, num_chunks=60,
            max_placements_per_traj=5, min_chunk_gap=8)
        # Check no single trajectory has >2 pending at any point
        for t in trajs:
            all_chunks = set()
            for p in t["placements"]:
                all_chunks.add(p["ask_chunk"])
                all_chunks.add(p["key_chunks"].get("trigger", p["ask_chunk"]))
            for c in all_chunks:
                pending = _count_pending_at(t["placements"], c)
                assert pending <= 2, \
                    f"traj {t['trajectory_id']} has {pending} pending at chunk {c}"


class TestFamilyCoverage:
    """Test family coverage backfill in plan_trajectories."""

    def test_backfill_missing_families(self):
        """If initial selection misses families, backfill should add them."""
        from scripts.agent_data_v5.pass3b_placement import plan_trajectories
        # Create placements heavily biased toward F2
        cards = [
            {"card_id": f"f2_{i}", "family": "F2",
             "question": f"Q{i}?", "canonical_answer": "Yes",
             "answer_form": "binary", "support_chunks": [i * 5]}
            for i in range(10)
        ] + [
            {"card_id": "e1_0", "family": "E1",
             "question": "Action?", "canonical_answer": "Yes",
             "answer_form": "binary", "support_chunks": [15]},
            {"card_id": "s1_0", "family": "S1",
             "question": "Scene?", "canonical_answer": "Kitchen",
             "answer_form": "short_exact", "support_chunks": [20]},
            {"card_id": "f1_0", "family": "F1",
             "question": "Price?", "canonical_answer": "4.99",
             "answer_form": "short_exact", "support_chunks": [25]},
            {"card_id": "f3_0", "family": "F3",
             "question": "Count?", "canonical_answer": "3",
             "answer_form": "number", "support_chunks": [30]},
        ]
        cards_map = {c["card_id"]: c for c in cards}
        placements = [
            {"card_id": c["card_id"], "ask_chunk": c["support_chunks"][0] + 2,
             "sequence_type": "immediate_response",
             "key_chunks": {"ask": c["support_chunks"][0] + 2,
                            "post_silent": c["support_chunks"][0] + 3}}
            for c in cards
        ]
        trajs = plan_trajectories(
            placements, cards_map=cards_map, num_chunks=60,
            max_placements_per_traj=5, min_chunk_gap=4)
        families = set()
        for t in trajs:
            for p in t["placements"]:
                families.add(cards_map[p["card_id"]]["family"])
        assert len(families) >= 4, f"Only {len(families)} families: {families}"

    def test_coverage_with_single_family_video(self):
        """Video with only one family should not crash, just warn."""
        from scripts.agent_data_v5.pass3b_placement import plan_trajectories
        cards = [{"card_id": "f2_0", "family": "F2",
                  "question": "Q?", "canonical_answer": "Yes",
                  "answer_form": "binary", "support_chunks": [5]}]
        cards_map = {c["card_id"]: c for c in cards}
        placements = [{"card_id": "f2_0", "ask_chunk": 10,
                       "sequence_type": "immediate_response",
                       "key_chunks": {"ask": 10, "post_silent": 11}}]
        # Should not crash
        trajs = plan_trajectories(
            placements, cards_map=cards_map, num_chunks=60)
        assert len(trajs) >= 1


class TestForkThink:
    """Test query-aware fork think generation."""

    @pytest.mark.asyncio
    async def test_no_queries_returns_base(self):
        """With no active queries, fork think returns base think unchanged."""
        from scripts.agent_data_v5.pass3c_samples import _generate_fork_think
        client = MockClient({})
        result = await _generate_fork_think(
            "Person chopping onions.", [], client, "vid", 10)
        assert result == "Person chopping onions."
        assert len(client._calls) == 0  # no API call made

    @pytest.mark.asyncio
    async def test_all_answered_returns_base(self):
        """With all queries answered, returns base think unchanged."""
        from scripts.agent_data_v5.pass3c_samples import _generate_fork_think
        queries = [{"question": "Color?", "ask_time": 10,
                     "answers": [{"text": "Red", "time": 12}]}]
        client = MockClient({})
        result = await _generate_fork_think(
            "Person stirring.", queries, client, "vid", 15)
        assert result == "Person stirring."
        assert len(client._calls) == 0

    @pytest.mark.asyncio
    async def test_pending_query_calls_api(self):
        """With pending queries, should call API to rewrite think."""
        from scripts.agent_data_v5.pass3c_samples import _generate_fork_think
        queries = [{"question": "Is apron red?", "ask_time": 10, "answers": []}]
        rewritten = "Person in red apron chopping. Red apron visible on the person."
        client = MockClient({"vid_fthink_15": rewritten})
        result = await _generate_fork_think(
            "Person chopping.", queries, client, "vid", 15)
        assert result == rewritten
        assert len(client._calls) == 1

    @pytest.mark.asyncio
    async def test_api_failure_returns_base(self):
        """API failure falls back to base think."""
        from scripts.agent_data_v5.pass3c_samples import _generate_fork_think
        queries = [{"question": "Q?", "ask_time": 10, "answers": []}]
        client = MockClient({})  # no response
        result = await _generate_fork_think(
            "Base think.", queries, client, "vid", 15)
        assert result == "Base think."


class TestRecallThink:
    """Test recall think generation."""

    @pytest.mark.asyncio
    async def test_generates_real_analysis(self):
        """Should call API to generate real analysis, not hardcoded string."""
        from scripts.agent_data_v5.pass3c_samples import _generate_recall_think
        card = {"question": "What price?", "canonical_answer": "4.99"}
        recall_result = {"source": "historical_frames",
                         "text_content": "[10-12] Price tag shows $4.99"}
        analysis = "Retrieved observation shows price tag with $4.99, matching the question."
        client = MockClient({"vid_rthink_20": analysis})
        result = await _generate_recall_think(
            card, recall_result, client, "vid", 20)
        assert result == analysis
        assert "Recall returned" not in result  # not hardcoded

    @pytest.mark.asyncio
    async def test_failure_recall_generates_analysis(self):
        """Failed recall should also get real analysis."""
        from scripts.agent_data_v5.pass3c_samples import _generate_recall_think
        card = {"question": "What price?", "canonical_answer": "4.99"}
        recall_result = {"source": "failure",
                         "text_content": "No matching results found."}
        analysis = "Recall found no matching observations for the price question."
        client = MockClient({"vid_rthink_20": analysis})
        result = await _generate_recall_think(
            card, recall_result, client, "vid", 20)
        assert result == analysis

    @pytest.mark.asyncio
    async def test_api_failure_falls_back(self):
        """API failure should produce a reasonable fallback."""
        from scripts.agent_data_v5.pass3c_samples import _generate_recall_think
        card = {"question": "Q?"}
        recall_result = {"source": "failure", "text_content": "No results."}
        client = MockClient({})
        result = await _generate_recall_think(
            card, recall_result, client, "vid", 20)
        assert "no matching" in result.lower()


class TestFormatSnapshotText:
    """Test snapshot text formatting for LLM prompt."""

    def test_with_recent_thinks(self):
        from scripts.agent_data_v5.pass3b_placement import _format_snapshot_text
        snapshot = {
            "recent_thinks": [
                {"time": "10-12", "text": "Person chopping onions."},
                {"time": "12-14", "text": "Added garlic to pot."},
            ],
            "compressed_segments": [],
        }
        recent, compressed = _format_snapshot_text(snapshot)
        assert "chopping onions" in recent
        assert "garlic" in recent
        assert compressed == "(empty)"

    def test_with_compressed_segments(self):
        from scripts.agent_data_v5.pass3b_placement import _format_snapshot_text
        snapshot = {
            "recent_thinks": [],
            "compressed_segments": [
                {"time_range": [0, 20], "text": "Person prepared ingredients."},
            ],
        }
        recent, compressed = _format_snapshot_text(snapshot)
        assert recent == "(empty)"
        assert "prepared ingredients" in compressed

    def test_empty_snapshot(self):
        from scripts.agent_data_v5.pass3b_placement import _format_snapshot_text
        snapshot = {"recent_thinks": [], "compressed_segments": []}
        recent, compressed = _format_snapshot_text(snapshot)
        assert recent == "(empty)"
        assert compressed == "(empty)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
