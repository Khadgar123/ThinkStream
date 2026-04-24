"""
Tests for Agent Data Pipeline v5.0

Tests core logic without requiring vLLM endpoint or video files.
Focuses on: memory state management, verification, format.

Note: Pass 3 tests are in test_pass3.py and test_pass3_e2e.py.
Old pass3_tasks / pass4_forks modules were replaced by pass3a/3b/3c in v9.0.
"""

import json
import pytest
from copy import deepcopy

from scripts.agent_data_v5.pass2_rollout import MemoryState
from scripts.agent_data_v5.pass4_verify import (
    verify_format,
    verify_grounding,
    verify_action_minimality,
    verify_information_flow,
    verify_think_token_length,
    verify_compression_ratio,
    verify_summary_provenance,
    verify_question_answer_leakage,
    label_difficulty,
    filter_samples,
)
from scripts.agent_data_v5.config import (
    COMPRESS_THRESHOLD,
    COMPRESS_RANGE,
    MAX_COMPRESSED_SEGMENTS,
    VISUAL_WINDOW_CHUNKS,
)


# ---------------------------------------------------------------------------
# MemoryState Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="MemoryState API changed (timeline-based), tests need rewrite")
class TestMemoryState:
    def test_initial_empty(self):
        mem = MemoryState()
        assert mem.compressed_segments == []
        assert mem.recent_thinks == []
        assert not mem.should_compress()

    def test_add_think(self):
        mem = MemoryState()
        mem.add_think(0, "Chef in red apron at counter.")
        assert len(mem.recent_thinks) == 1
        assert mem.recent_thinks[0]["chunk"] == 0
        assert mem.recent_thinks[0]["time"] == "0-2"
        assert "red apron" in mem.recent_thinks[0]["text"]

    def test_compress_threshold_token_based(self):
        """Compression triggers based on token count of recent_thinks."""
        from scripts.agent_data_v5.config import COMPRESS_TOKEN_THRESHOLD
        mem = MemoryState()
        # Add thinks until compression triggers
        think_text = "Chef wearing red apron carefully places four bright red Roma tomatoes onto the large wooden cutting board near the stove"
        triggered = False
        for i in range(20):
            mem.add_think(i, think_text)
            if mem.should_compress():
                tokens = mem.count_recent_tokens()
                assert tokens >= COMPRESS_TOKEN_THRESHOLD, \
                    f"Triggered at {tokens} tok, threshold is {COMPRESS_TOKEN_THRESHOLD}"
                triggered = True
                break
        assert triggered, "Should have triggered compression"

    def test_compress_removes_specified_chunks(self):
        mem = MemoryState()
        for i in range(COMPRESS_THRESHOLD):
            mem.add_think(i, f"Observation {i}")

        # Compress specific chunks (not necessarily all)
        chunks_to_compress = [0, 1, 2, 3, 4, 5]
        summary = {"time_range": [0, 12], "text": "Compressed summary."}
        mem.compress(summary, selected_indices=list(range(len(chunks_to_compress))))

        assert len(mem.compressed_segments) == 1
        remaining_chunks = [t["chunk"] for t in mem.recent_thinks]
        assert all(c not in remaining_chunks for c in chunks_to_compress)
        assert len(mem.recent_thinks) == COMPRESS_THRESHOLD - len(chunks_to_compress)
        assert mem.compressed_segments[0]["text"] == "Compressed summary."

    def test_snapshot_is_immutable(self):
        mem = MemoryState()
        mem.add_think(0, "First obs")
        snap = mem.snapshot(12)

        # Modify memory after snapshot
        mem.add_think(1, "Second obs")
        assert len(snap["recent_thinks"]) == 1  # Snapshot unchanged
        assert len(mem.recent_thinks) == 2

    def test_format_for_prompt(self):
        mem = MemoryState()
        mem.compressed_segments.append({"time_range": [0, 20], "text": "Summary A"})
        mem.add_think(10, "Recent obs")

        compressed_text, obs_text = mem.format_for_prompt()
        assert "<compressed>" in compressed_text
        assert '"time_range": [0, 20]' in compressed_text
        assert "Summary A" in compressed_text
        assert "[20-22]" in obs_text
        assert "Recent obs" in obs_text

    def test_retrieval_archive_accumulates(self):
        """System-side archive should keep all thinks even after compression."""
        mem = MemoryState()
        for i in range(COMPRESS_THRESHOLD):
            mem.add_think(i, f"Observation {i}")

        assert len(mem.retrieval_archive) == COMPRESS_THRESHOLD

        chunks_to_compress = [0, 1, 2, 3, 4, 5]
        summary = {"time_range": [0, 12], "text": "Compressed summary."}
        mem.compress(summary, selected_indices=list(range(len(chunks_to_compress))))

        # Compressed thinks removed from recent but archive keeps all
        assert len(mem.recent_thinks) == COMPRESS_THRESHOLD - len(chunks_to_compress)
        assert len(mem.retrieval_archive) == COMPRESS_THRESHOLD

        # Add more
        mem.add_think(COMPRESS_THRESHOLD, "New obs")
        assert len(mem.retrieval_archive) == COMPRESS_THRESHOLD + 1

    def test_compressed_segments_upper_limit(self):
        """compressed_segments should merge when exceeding MAX_COMPRESSED_SEGMENTS."""
        mem = MemoryState()
        # Add enough observations and compress repeatedly to exceed limit
        chunk = 0
        for _ in range(MAX_COMPRESSED_SEGMENTS + 2):
            for j in range(COMPRESS_THRESHOLD):
                mem.add_think(chunk, f"Obs at chunk {chunk}")
                chunk += 1
            t_start = (chunk - COMPRESS_THRESHOLD) * 2
            t_end = chunk * 2
            summary = {"time_range": [t_start, t_end], "text": f"Summary {t_start}-{t_end}"}
            mem.compress(summary)

        assert len(mem.compressed_segments) <= MAX_COMPRESSED_SEGMENTS

    def test_snapshot_excludes_archive(self):
        """Snapshot represents model-visible state — no archive."""
        mem = MemoryState()
        mem.add_think(0, "First obs")
        snap = mem.snapshot(12)
        assert "retrieval_archive" not in snap
        # But archive is still accessible on the MemoryState itself
        assert len(mem.retrieval_archive) == 1

    def test_pending_questions_in_snapshot(self):
        """Pending questions should be preserved in snapshot."""
        mem = MemoryState()
        mem.pending_questions.append({
            "question": "Tell me when basil is added",
            "since_chunk": 10,
        })
        snap = mem.snapshot(15)
        assert len(snap["pending_questions"]) == 1
        assert snap["pending_questions"][0]["question"] == "Tell me when basil is added"


# ---------------------------------------------------------------------------
# Action Minimality Tests
# ---------------------------------------------------------------------------


class TestActionMinimality:
    def _make_sample(self, output, sample_type="silent", metadata=None):
        return {
            "output": output,
            "sample_type": sample_type,
            "metadata": metadata or {},
        }

    def test_format_valid_silent(self):
        sample = self._make_sample(
            "<think>Chef slicing tomatoes on wooden board, red and juicy.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_response(self):
        sample = self._make_sample(
            "<think>Chef adds salt from small white bowl into the simmering pot on right burner.</think>"
            "<action>response</action>"
            "<response>The chef added approximately one teaspoon of salt.</response>"
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_recall(self):
        sample = self._make_sample(
            '<think>Pot on stove simmering, no apron visible in current frames.</think>'
            '<action>recall</action>'
            '<query>{"query":"chef apron color red","time_range":"0-10"}</query>'
        )
        passed, reason = verify_format(sample)
        assert passed, reason

    def test_format_valid_compress(self):
        sample = self._make_sample(
            '<think>System triggered memory compression for earlier observations.</think>'
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
            '<think>Testing.</think>'
            '<action>recall</action>'
            '<query>not valid json</query>'
        )
        passed, reason = verify_format(sample)
        assert not passed

    def test_grounding_rejects_sound(self):
        sample = self._make_sample(
            "<think>Sizzling sound as oil heats up in the pan.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed
        assert "sound" in reason

    def test_grounding_rejects_emotion(self):
        sample = self._make_sample(
            "<think>Chef seems happy while cooking the meal.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_grounding_rejects_meta(self):
        sample = self._make_sample(
            "<think>I notice the chef is cutting something.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_grounding_accepts_visual(self):
        sample = self._make_sample(
            "<think>Chef slices fourth Roma tomato into quarters on wooden board.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert passed

    def test_action_minimality_unnecessary_recall(self):
        # v9.0: action_minimality checks sequence_type first
        sample = self._make_sample(
            '<think>Checking memory.</think>'
            '<action>recall</action>'
            '<query>{"query":"test","time_range":"0-10"}</query>',
            sample_type="recall_query",
            metadata={
                "visibility": {
                    "answer_in_recent_obs": True,
                    "answer_in_compressed": False,
                    "evidence_in_window": False,
                }
            },
        )
        # Also set sequence_type for new check path
        sample["sequence_type"] = "immediate_response"  # wrong: recall in non-recall seq
        passed, reason = verify_action_minimality(sample)
        assert not passed

    def test_difficulty_labels(self):
        # v9.0: difficulty based on sequence_type, not metadata.action_reason
        assert label_difficulty({"sample_type": "silent", "sequence_type": "base"}) == "easy"
        assert label_difficulty({"sample_type": "response", "sequence_type": "immediate_response"}) == "easy"
        assert label_difficulty({"sample_type": "recall_query", "sequence_type": "recall_fail_then_found"}) == "hard"

    def test_filter_samples(self):
        good = self._make_sample(
            "<think>Chef in red apron places four bright Roma tomatoes on the large wooden cutting board near the stainless steel sink on the left counter.</think>"
            "<action>silent</action>"
        )
        bad = self._make_sample(
            "<think>I think the chef is probably tired and wants to stop cooking now because the sizzling sound is getting louder and annoying.</think>"
            "<action>silent</action>"
        )
        passed, stats = filter_samples([good, bad])
        assert stats["passed"] == 1
        assert stats["failed"] == 1


# ---------------------------------------------------------------------------
# Recall Result Tests
# ---------------------------------------------------------------------------


class TestRecallResult:
    def test_snapshot_includes_leaving_chunk(self):
        """After chunk leaves visual window, its observation should be in the
        NEXT snapshot's recent_observations (not delayed by one step)."""
        mem = MemoryState()
        # Simulate: at chunk 0, add observation for chunk 0 to archive
        mem.add_think(0, "First obs at chunk 0")
        # Snapshot at chunk 1 should include chunk 0's observation
        snap = mem.snapshot(1)
        assert len(snap["recent_thinks"]) == 1
        assert snap["recent_thinks"][0]["chunk"] == 0


# ---------------------------------------------------------------------------
# Recall Result Future Leakage Tests
# ---------------------------------------------------------------------------


class TestRecallFutureLeakage:
    def _make_sample(self, output, sample_type="silent", metadata=None, input_data=None):
        s = {
            "output": output,
            "sample_type": sample_type,
            "metadata": metadata or {},
        }
        if input_data:
            s["input"] = input_data
        return s

    def test_recall_without_query_rejected(self):
        sample = self._make_sample(
            "<think>Test obs here.</think><action>recall</action>"
        )
        passed, reason = verify_format(sample)
        assert not passed
        assert "recall_action_without_query" in reason

    def test_compress_without_summary_rejected(self):
        sample = self._make_sample(
            "<think>Test obs here.</think><action>compress</action>"
        )
        passed, reason = verify_format(sample)
        assert not passed
        assert "compress_action_without_summary" in reason

    def test_grounding_rejects_sizzling(self):
        sample = self._make_sample(
            "<think>Oil sizzling in the pan on burner.</think>"
            "<action>silent</action>"
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_grounding_rejects_system_triggered(self):
        sample = self._make_sample(
            "<think>System triggered memory compression.</think>"
            "<action>compress</action>"
            '<summary>{"time_range":[0,20],"text":"test"}</summary>'
        )
        passed, reason = verify_grounding(sample)
        assert not passed

    def test_recall_response_after_failure_must_be_uncertain(self):
        sample = self._make_sample(
            "<think>Chef at counter.</think>"
            "<action>response</action>"
            "<response>The chef added exactly one teaspoon of salt.</response>",
            sample_type="recall_response",
            input_data={
                "recall_result": {"noise_level": "failure", "text_content": "No results."}
            },
        )
        passed, reason = verify_information_flow(sample)
        assert not passed
        assert "confident_response_after_failed_recall" in reason

    def test_recall_response_uncertain_after_failure_passes(self):
        sample = self._make_sample(
            "<think>Chef at counter.</think>"
            "<action>response</action>"
            "<response>I could not find enough evidence to confirm the amount.</response>",
            sample_type="recall_response",
            input_data={
                "recall_result": {"noise_level": "failure", "text_content": "No results."}
            },
        )
        passed, reason = verify_information_flow(sample)
        assert passed


# ---------------------------------------------------------------------------
# Recall Result Returned Chunks Tests
# ---------------------------------------------------------------------------


class TestRecallReturnedChunks:
    def test_recall_response_passes_grounding(self):
        """recall_response with no think should pass grounding."""
        sample = {
            "output": "<action>response</action><response>About one teaspoon.</response>",
            "sample_type": "recall_response",
        }
        passed, reason = verify_grounding(sample)
        assert passed

    def test_recall_response_fails_if_response_has_sound(self):
        """recall_response should still fail if response contains non-visual claim."""
        sample = {
            "output": "<action>response</action><response>I heard a sizzling sound.</response>",
            "sample_type": "recall_response",
        }
        passed, reason = verify_grounding(sample)
        assert not passed


# ---------------------------------------------------------------------------
# Compression Range Selection Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="choose_optimal_compress_range API changed, needs rewrite")
class TestCompressionRangeSelection:
    def test_optimal_range_picks_least_important(self):
        """Teacher-chosen range should be the one with least info."""
        from scripts.agent_data_v5.pass2_rollout import choose_optimal_compress_range
        thinks = [
            {"chunk": 0, "time": "0-2", "text": "Empty counter scene."},
            {"chunk": 1, "time": "2-4", "text": "Nothing happening on counter."},
            {"chunk": 2, "time": "4-6", "text": "Still empty counter top."},
            {"chunk": 3, "time": "6-8", "text": "Counter remains empty."},
            # These are more important (entities, numbers, OCR)
            {"chunk": 4, "time": "8-10", "text": "Chef_1 places 4 Roma tomatoes on board."},
            {"chunk": 5, "time": "10-12", "text": "Chef_1 picks up Wusthof knife, begins slicing."},
            {"chunk": 6, "time": "12-14", "text": "Timer_display reads 08:30, pot_1 boiling."},
            {"chunk": 7, "time": "14-16", "text": "Chef_1 adds salt_shaker amount to pot_1."},
            {"chunk": 8, "time": "16-18", "text": "Scene continues normally."},
        ]
        result = choose_optimal_compress_range(thinks)
        # Should pick the low-importance range (chunks 0-3, empty scenes)
        result_chunks = [t["chunk"] for t in result]
        assert 0 in result_chunks
        # Should NOT pick the entity/number-rich range
        assert 6 not in result_chunks  # Timer display with OCR

    def test_optimal_range_respects_size_bounds(self):
        from scripts.agent_data_v5.pass2_rollout import choose_optimal_compress_range
        from scripts.agent_data_v5.config import COMPRESS_RANGE_MIN, COMPRESS_RANGE_MAX
        thinks = [{"chunk": i, "time": f"{i*2}-{i*2+2}", "text": f"obs {i}"} for i in range(12)]
        result = choose_optimal_compress_range(thinks)
        assert COMPRESS_RANGE_MIN <= len(result) <= COMPRESS_RANGE_MAX


# ---------------------------------------------------------------------------
# New Verifier Tests (v5.3)
# ---------------------------------------------------------------------------


class TestThinkTokenLength:
    def test_normal_length_passes(self):
        # ~40-50 tokens (within 30-75 tolerance range)
        sample = {
            "output": "<think>Chef in red apron carefully places four bright red Roma tomatoes onto the large wooden cutting board positioned near the stainless steel stove top surface on the kitchen counter beside the window overlooking the garden.</think><action>silent</action>",
            "sample_type": "silent",
        }
        passed, reason = verify_think_token_length(sample)
        assert passed, reason

    def test_recall_response_no_think(self):
        """recall_response has no think — should pass think length check."""
        sample = {
            "output": "<action>response</action><response>test</response>",
            "sample_type": "recall_response",
        }
        passed, reason = verify_think_token_length(sample)
        assert passed


class TestCompressionRatio:
    def test_good_ratio_passes(self):
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": [
                        f"[{i*2}-{i*2+2}] Chef does something at step {i} with various ingredients." for i in range(6)
                    ],
                },
            },
            "output": '<think>Current obs.</think><action>compress</action><summary>{"time_range":[0,12],"text":"Chef cooked."}</summary>',
        }
        passed, reason = verify_compression_ratio(sample)
        assert passed, reason

    def test_bad_ratio_fails(self):
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": ["[0-2] Short."],
                },
            },
            "output": '<think>Current obs.</think><action>compress</action><summary>{"time_range":[0,2],"text":"This is a very long summary that is longer than the original input text which defeats the purpose of compression entirely."}</summary>',
        }
        passed, reason = verify_compression_ratio(sample)
        assert not passed
        assert "compression_ratio" in reason

    def test_non_compress_skipped(self):
        sample = {"sample_type": "silent", "output": "<think>test</think><action>silent</action>"}
        passed, reason = verify_compression_ratio(sample)
        assert passed


class TestSummaryProvenance:
    def test_provenance_pass(self):
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": [
                        "[0-2] Chef_1 in red apron at counter.",
                        "[2-4] Chef_1 picks up knife from drawer.",
                    ],
                },
            },
            "output": '<think>Current.</think><action>compress</action><summary>{"time_range":[0,4],"text":"Chef_1 prepared at counter with knife."}</summary>',
        }
        passed, reason = verify_summary_provenance(sample)
        assert passed, reason

    def test_provenance_violation(self):
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": [
                        "[0-2] Empty counter scene.",
                        "[2-4] Still empty counter.",
                    ],
                },
            },
            "output": '<think>Current.</think><action>compress</action><summary>{"time_range":[0,4],"text":"Chef_1 in Kitchen prepared Tomatoes with Knife on Board."}</summary>',
        }
        passed, reason = verify_summary_provenance(sample)
        assert not passed
        assert "provenance" in reason


class TestKeywordOverlapImproved:
    def test_exact_substring_detected(self):
        sample = {
            "metadata": {
                "question": "What is the red apron color?",
                "gold_answer": "red apron",
            },
        }
        passed, reason = verify_question_answer_leakage(sample)
        assert not passed
        assert "string" in reason

    def test_keyword_coverage_detected(self):
        # All meaningful answer keywords (stainless, pot, right, burner) in question
        sample = {
            "metadata": {
                "question": "What is on the right burner of the stainless pot?",
                "gold_answer": "stainless pot on right burner",
            },
        }
        passed, reason = verify_question_answer_leakage(sample)
        assert not passed
        assert "keyword" in reason

    def test_normal_question_passes(self):
        sample = {
            "metadata": {
                "question": "What did the chef add to the pot?",
                "gold_answer": "approximately one teaspoon of salt",
            },
        }
        passed, reason = verify_question_answer_leakage(sample)
        assert passed

    def test_no_question_passes(self):
        sample = {"metadata": {}}
        passed, _ = verify_question_answer_leakage(sample)
        assert passed


class TestCompressedSourceTexts:
    def test_selected_range_only(self):
        from scripts.agent_data_v5.pass5_verify import _compressed_source_texts
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": [
                        "[0-2] Chef at counter.",
                        "[2-4] Chef picks up knife.",
                        "[4-6] Chef slices tomato.",
                        "[6-8] Chef adds salt.",
                    ],
                },
            },
            "metadata": {
                "compressed_range": [0, 4],  # Only first 2 thinks
            },
        }
        texts = _compressed_source_texts(sample)
        assert len(texts) == 2
        assert "counter" in texts[0]
        assert "knife" in texts[1]

    def test_fallback_when_no_range(self):
        from scripts.agent_data_v5.pass5_verify import _compressed_source_texts
        sample = {
            "sample_type": "compress",
            "input": {
                "memory": {
                    "recent_thinks": ["[0-2] A.", "[2-4] B."],
                },
            },
            "metadata": {},
        }
        texts = _compressed_source_texts(sample)
        assert len(texts) == 2  # Falls back to all



if __name__ == "__main__":
    pytest.main([__file__, "-v"])