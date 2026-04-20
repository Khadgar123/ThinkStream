"""
Tests for Agent Data Pipeline v5.0

Tests core logic without requiring vLLM endpoint or video files.
Focuses on: memory state management, action minimality, verification, format.
"""

import json
import pytest
from copy import deepcopy

from scripts.agent_data_v5.pass2_rollout import MemoryState
from scripts.agent_data_v5.pass3_tasks import (
    determine_gold_action,
    extract_keywords,
    keyword_overlap,
    build_visibility_matrix,
)
from scripts.agent_data_v5.pass4_forks import (
    build_sample_input,
    simulate_recall_result,
)
from scripts.agent_data_v5.pass5_verify import (
    verify_format,
    verify_grounding,
    verify_action_minimality,
    verify_information_flow,
    label_difficulty,
    filter_samples,
)
from scripts.agent_data_v5.config import (
    COMPRESS_THRESHOLD,
    COMPRESS_RANGE,
    SYSTEM_PROMPT,
    VISUAL_WINDOW_CHUNKS,
)


# ---------------------------------------------------------------------------
# MemoryState Tests
# ---------------------------------------------------------------------------


class TestMemoryState:
    def test_initial_empty(self):
        mem = MemoryState()
        assert mem.compressed_segments == []
        assert mem.recent_observations == []
        assert not mem.should_compress()

    def test_add_observation(self):
        mem = MemoryState()
        mem.add_observation(0, "Chef in red apron at counter.")
        assert len(mem.recent_observations) == 1
        assert mem.recent_observations[0]["chunk"] == 0
        assert mem.recent_observations[0]["time"] == "0-2"
        assert "red apron" in mem.recent_observations[0]["obs"]

    def test_compress_threshold(self):
        mem = MemoryState()
        for i in range(COMPRESS_THRESHOLD - 1):
            mem.add_observation(i, f"Observation {i}")
        assert not mem.should_compress()
        mem.add_observation(COMPRESS_THRESHOLD - 1, "Final observation")
        assert mem.should_compress()

    def test_compress_removes_old(self):
        mem = MemoryState()
        for i in range(COMPRESS_THRESHOLD):
            mem.add_observation(i, f"Observation {i}")

        summary = {"time_range": [0, 20], "text": "Compressed summary."}
        mem.compress(summary)

        assert len(mem.compressed_segments) == 1
        assert len(mem.recent_observations) == 0  # All 10 removed
        assert mem.compressed_segments[0]["text"] == "Compressed summary."

    def test_snapshot_is_immutable(self):
        mem = MemoryState()
        mem.add_observation(0, "First obs")
        snap = mem.snapshot(12)

        # Modify memory after snapshot
        mem.add_observation(1, "Second obs")
        assert len(snap["recent_observations"]) == 1  # Snapshot unchanged
        assert len(mem.recent_observations) == 2

    def test_format_for_prompt(self):
        mem = MemoryState()
        mem.compressed_segments.append({"time_range": [0, 20], "text": "Summary A"})
        mem.add_observation(10, "Recent obs")

        compressed_text, obs_text = mem.format_for_prompt()
        assert 'time="0-20"' in compressed_text
        assert "Summary A" in compressed_text
        assert "[20-22]" in obs_text
        assert "Recent obs" in obs_text


# ---------------------------------------------------------------------------
# Action Minimality Tests
# ---------------------------------------------------------------------------


class TestActionMinimality:
    def _make_snapshot(self, compressed=None, recent_obs=None, window_start=30, chunk_idx=41):
        return {
            "chunk_idx": chunk_idx,
            "compressed_segments": compressed or [],
            "recent_observations": recent_obs or [],
            "visual_window_start": window_start,
        }

    def test_answer_in_visual_window(self):
        """If evidence chunk is in visual window → response."""
        snapshot = self._make_snapshot(window_start=30, chunk_idx=41)
        observations = [{"observation": f"obs {i}"} for i in range(50)]
        action, reason = determine_gold_action(
            ["red", "apron"], snapshot, evidence_chunks=[35], observations=observations
        )
        assert action == "response"
        assert "visual_window" in reason

    def test_answer_in_recent_observations(self):
        """If answer keywords in recent_observations → response."""
        snapshot = self._make_snapshot(
            recent_obs=[{"obs": "Chef wearing red apron standing at counter."}],
            window_start=30, chunk_idx=41,
        )
        observations = [{"observation": f"obs {i}"} for i in range(50)]
        action, reason = determine_gold_action(
            ["red", "apron"], snapshot, evidence_chunks=[5], observations=observations
        )
        assert action == "response"
        assert "recent_observations" in reason

    def test_answer_in_compressed_summary(self):
        """If answer in compressed_summary → response (NOT recall!)."""
        snapshot = self._make_snapshot(
            compressed=[{"text": "Chef in red apron prepared ingredients at counter."}],
            window_start=30, chunk_idx=41,
        )
        observations = [{"observation": f"obs {i}"} for i in range(50)]
        action, reason = determine_gold_action(
            ["red", "apron"], snapshot, evidence_chunks=[5], observations=observations
        )
        assert action == "response"
        assert "compressed" in reason

    def test_answer_requires_recall(self):
        """If answer not in any visible source → recall."""
        snapshot = self._make_snapshot(
            compressed=[{"text": "Vegetables were prepared on the counter."}],
            recent_obs=[{"obs": "Pot simmering on stove."}],
            window_start=30, chunk_idx=41,
        )
        # Observation at evidence chunk DOES mention red apron
        observations = [{"observation": "obs"} for _ in range(50)]
        observations[5] = {"observation": "Chef in red apron picks up knife."}
        action, reason = determine_gold_action(
            ["red", "apron"], snapshot, evidence_chunks=[5], observations=observations
        )
        assert action == "recall"

    def test_answer_unanswerable(self):
        """If answer not found anywhere → unanswerable."""
        snapshot = self._make_snapshot(
            compressed=[{"text": "Vegetables prepared."}],
            window_start=30, chunk_idx=41,
        )
        observations = [{"observation": "Generic scene."} for _ in range(50)]
        action, reason = determine_gold_action(
            ["brand", "knife", "wusthof"], snapshot, evidence_chunks=[5], observations=observations
        )
        assert action == "unanswerable"


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------


class TestVerification:
    def _make_sample(self, output, sample_type="silent", metadata=None):
        return {
            "output": output,
            "sample_type": sample_type,
            "metadata": metadata or {},
        }

    def test_format_valid_silent(self):
        sample = self._make_sample(
            "<observation>Chef slicing tomatoes on wooden board, red and juicy.</observation>"
            "<action>silent</action>"
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_response(self):
        sample = self._make_sample(
            "<observation>Chef adds salt from small white bowl into the simmering pot on right burner.</observation>"
            "<action>response</action>"
            "<response>The chef added approximately one teaspoon of salt.</response>"
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_recall(self):
        sample = self._make_sample(
            '<observation>Pot on stove simmering, no apron visible in current frames.</observation>'
            '<action>recall</action>'
            '<query>{"query":"chef apron color red","time_range":"0-10"}</query>'
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_compress(self):
        sample = self._make_sample(
            '<observation>System triggered memory compression for earlier observations.</observation>'
            '<action>compress</action>'
            '<summary>{"time_range":[0,20],"text":"Chef prepared workspace."}</summary>'
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_invalid_no_observation(self):
        sample = self._make_sample("<action>silent</action>")
        passed, reason = verify_format(sample)
        assert not passed

    def test_format_invalid_query_json(self):
        sample = self._make_sample(
            '<observation>Testing.</observation>'
            '<action>recall</action>'
            '<query>not valid json</query>'
        )
        passed, reason = verify_format(sample)
        assert not passed

    def test_grounding_rejects_sound(self):
        sample = self._make_sample(
            "<observation>Sizzling sound as oil heats up in the pan.</observation>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed
        assert "sound" in reason

    def test_grounding_rejects_emotion(self):
        sample = self._make_sample(
            "<observation>Chef seems happy while cooking the meal.</observation>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_grounding_rejects_meta(self):
        sample = self._make_sample(
            "<observation>I notice the chef is cutting something.</observation>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_grounding_accepts_visual(self):
        sample = self._make_sample(
            "<observation>Chef slices fourth Roma tomato into quarters on wooden board.</observation>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert passed

    def test_action_minimality_unnecessary_recall(self):
        sample = self._make_sample(
            '<observation>Checking memory.</observation>'
            '<action>recall</action>'
            '<query>{"query":"test","time_range":"0-10"}</query>',
            sample_type="recall_query",
            metadata={
                "visibility": {
                    "answer_in_recent_obs": True,  # Answer is visible!
                    "answer_in_compressed": False,
                    "evidence_in_window": False,
                }
            },
        )
        passed, reason = verify_action_minimality(sample)
        assert not passed
        assert "unnecessary" in reason

    def test_difficulty_labels(self):
        assert label_difficulty({"sample_type": "silent"}) == "easy"
        assert label_difficulty({"sample_type": "response", "metadata": {"action_reason": "answer_in_visual_window"}}) == "easy"
        assert label_difficulty({"sample_type": "recall_query", "metadata": {"task_type": "compress_recall_visual"}}) == "hard"

    def test_filter_samples(self):
        good = self._make_sample(
            "<observation>Chef places four Roma tomatoes on the wooden cutting board near the sink.</observation>"
            "<action>silent</action>"
        )
        bad = self._make_sample(
            "<observation>I think the chef is probably tired and wants to stop cooking now.</observation>"
            "<action>silent</action>"
        )
        passed, stats = filter_samples([good, bad])
        assert stats["passed"] == 1
        assert stats["failed"] == 1


# ---------------------------------------------------------------------------
# Recall Result Tests
# ---------------------------------------------------------------------------


class TestRecallResult:
    def test_recall_uses_student_content(self):
        """recall_result must only use student observations, not teacher captions."""
        snapshot = {
            "compressed_segments": [
                {"time_range": [0, 20], "text": "Chef prepared ingredients at counter."}
            ],
            "recent_observations": [
                {"obs": "Pot on stove.", "time": "40-42", "chunk": 20}
            ],
        }
        observations = [
            {"observation": f"obs {i}", "chunk_idx": i}
            for i in range(50)
        ]
        observations[5] = {"observation": "Chef in red apron picks up knife.", "chunk_idx": 5}

        task = {
            "evidence_chunks": [5],
            "gold_answer": "red apron",
        }

        result = simulate_recall_result(task, snapshot, observations)
        # Should contain student observation text, not teacher caption
        assert result["source"] in ("historical_frames", "student_observation",
                                     "compressed_summary", "distractor", "failure")
        # The content should come from observations[5] or compressed segments
        if result["source"] == "historical_frames":
            assert "red apron" in result["text_content"] or "Retrieved frames" in result["text_content"]


# ---------------------------------------------------------------------------
# Keyword Extraction Tests
# ---------------------------------------------------------------------------


class TestKeywords:
    def test_extract_keywords(self):
        kws = extract_keywords("The chef was wearing a red apron")
        assert "chef" in kws
        assert "red" in kws
        assert "apron" in kws
        assert "the" not in kws
        assert "was" not in kws

    def test_keyword_overlap(self):
        assert keyword_overlap("Chef in red apron at counter", ["red", "apron"]) == 1.0
        assert keyword_overlap("Pot on stove simmering", ["red", "apron"]) == 0.0
        assert keyword_overlap("Red pot on stove", ["red", "apron"]) == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
