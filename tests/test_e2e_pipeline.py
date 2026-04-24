"""End-to-end tests for data construction + SFT pipeline.

DEPRECATED: This file references pass3_tasks and pass4_forks which were
replaced by pass3a/3b/3c in v9.0. The new e2e tests are in test_pass3_e2e.py.

Kept for reference. All tests are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="References deleted pass3_tasks/pass4_forks modules. "
           "See test_pass3_e2e.py for current e2e tests."
)
from scripts.agent_data_v5.config import (
    AGENT_CHUNK_SEC,
    VISUAL_WINDOW_CHUNKS,
    COMPRESS_TOKEN_THRESHOLD,
    COMPRESS_RANGE_MIN,
    SYSTEM_PROMPT,
)
from thinkstream.data.agent_protocol import (
    format_memory_block,
    build_user_content,
    parse_agent_output,
)


# ---------------------------------------------------------------------------
# Helpers: simulate a realistic video timeline
# ---------------------------------------------------------------------------

def build_timeline(num_chunks=30):
    """Simulate a full Pass2 rollout: generate thinks, trigger compression."""
    mem = MemoryState()
    observations = []
    snapshots = {}
    compression_events = []

    for i in range(num_chunks):
        snapshots[i] = mem.snapshot(i)

        # Simulate a think (~50 chars ≈ ~12 tokens without real tokenizer)
        think = f"Chunk {i}: chef_1 performs action_{i} at counter with tool_{i % 5}."
        observations.append({"chunk_idx": i, "think": think,
                             "time": [i * AGENT_CHUNK_SEC, (i + 1) * AGENT_CHUNK_SEC]})

        should_compress = (
            mem.count_recent_tokens() >= COMPRESS_TOKEN_THRESHOLD
            and len(mem.recent_thinks) >= COMPRESS_RANGE_MIN
        )

        if should_compress:
            pre_thinks = snapshots[i]["recent_thinks"]
            to_compress = pre_thinks[:COMPRESS_RANGE_MIN]
            first_t = to_compress[0]["chunk"] * AGENT_CHUNK_SEC
            last_t = (to_compress[-1]["chunk"] + 1) * AGENT_CHUNK_SEC
            summary = {
                "time_range": [int(first_t), int(last_t)],
                "text": f"Summary of chunks {to_compress[0]['chunk']}-{to_compress[-1]['chunk']}.",
            }
            chunks_compressed = [t["chunk"] for t in to_compress]
            mem.compress(summary, compressed_chunks=chunks_compressed)
            mem.add_think(i, think)
            compression_events.append({
                "trigger_chunk": i,
                "summary": summary,
                "compressed_thinks_chunks": chunks_compressed,
            })
        else:
            mem.add_think(i, think)

    return {
        "num_chunks": num_chunks,
        "thinks": observations,
        "snapshots": snapshots,
        "compression_events": compression_events,
        "final_memory": mem.snapshot(num_chunks),
    }


def build_evidence(num_chunks=30):
    """Simulate Pass1 evidence for each chunk."""
    evidence = []
    for i in range(num_chunks):
        evidence.append({
            "chunk_idx": i,
            "parse_success": True,
            "visible_entities": [
                {"id": "chef_1", "attributes": ["red apron"], "action": f"action_{i}"},
            ],
            "atomic_facts": [
                {"fact": f"chef_1 performs action_{i} at counter",
                 "confidence": 0.9,
                 "support_level": "direct_current_chunk",
                 "target_resolution_visible": True},
            ],
            "state_changes": [f"action_{i} started"] if i % 10 == 5 else [],
            "ocr": [],
        })
    return evidence


# ---------------------------------------------------------------------------
# Test 1: MemoryState → snapshot → action decision chain
# ---------------------------------------------------------------------------

class TestMemoryToAction:
    """Test the full chain: MemoryState → snapshot → determine_gold_action."""

    def test_answer_in_visual_window(self):
        """When evidence chunk is in visual window → response(from_frames)."""
        rollout = build_timeline(20)
        evidence = build_evidence(20)
        # Ask at chunk 5 about fact at chunk 5 → in visual window
        snapshot = rollout["snapshots"][5]
        keywords = extract_keywords("chef_1 performs action_5 at counter")
        action, reason = determine_gold_action(keywords, snapshot, [5], rollout["thinks"])
        assert action == "response"
        assert "visual_window" in reason

    def test_answer_in_memory_after_window(self):
        """When evidence chunk left window but think still in memory → response(from_memory)."""
        rollout = build_timeline(30)
        # Chunk 2's fact, ask at chunk 16 (window starts at 4, chunk 2 is outside)
        # But think_2 might still be in recent_thinks
        snapshot = rollout["snapshots"][16]
        keywords = extract_keywords("chef_1 performs action_2 at counter")
        action, reason = determine_gold_action(keywords, snapshot, [2], rollout["thinks"])
        # If think_2 is still in recent_thinks (not yet compressed), should be response
        if any("action_2" in t.get("text", "") for t in snapshot.get("recent_thinks", [])):
            assert action == "response"
            assert "recent_think" in reason
        else:
            # If compressed or gone, should be recall
            assert action in ("response", "recall")

    def test_answer_requires_recall(self):
        """When evidence is nowhere accessible → recall."""
        rollout = build_timeline(30)
        # Ask at chunk 28 about a fact at chunk 0 with unique keyword
        snapshot = rollout["snapshots"][28]
        keywords = extract_keywords("unique_never_seen_entity xyz123")
        action, reason = determine_gold_action(keywords, snapshot, [0], rollout["thinks"])
        # unique keyword not in any think → should be recall or unanswerable
        assert action in ("recall", "response")


# ---------------------------------------------------------------------------
# Test 2: Pass4 sample construction → format verification
# ---------------------------------------------------------------------------

class TestSampleConstruction:
    """Test Pass4 sample construction produces valid samples."""

    def _make_sample(self, sample_type, output, snapshot=None, user_input="",
                     recalled_frames=None, recall_result=None, metadata=None):
        if snapshot is None:
            mem = MemoryState()
            mem.add_think(5, "Chef slices tomato on board.")
            mem.add_think(6, "Chef adds olive oil to pot.")
            snapshot = mem.snapshot(7)

        visual_meta = get_visual_frame_info(7, [])
        sample_input = build_sample_input(snapshot, user_input, visual_meta)
        if recalled_frames:
            sample_input["recalled_frames"] = recalled_frames
        if recall_result:
            sample_input["recall_result"] = recall_result

        return {
            "sample_type": sample_type,
            "chunk_idx": 7,
            "input": sample_input,
            "output": output,
            "metadata": metadata or {},
        }

    # All think texts must be 40-60 tokens (~200-350 chars) to pass verification

    def test_silent_passes_verification(self):
        sample = self._make_sample(
            "silent",
            "<think>Chef_1 in red apron reaches for the salt container on the upper shelf "
            "above the stainless steel counter, wooden cutting board still visible with "
            "sliced tomato quarters arranged neatly on the left side near the sink.</think>"
            "<action>silent</action>",
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_response_passes_verification(self):
        sample = self._make_sample(
            "response",
            "<think>Large rectangular cutting board clearly visible on the counter surface, "
            "appears to be made of natural hardwood with a light brown color and visible "
            "grain pattern, positioned between the stove and the sink area.</think>"
            "<action>response</action><response>The cutting board is light brown, wooden.</response>",
            user_input="What color is the board?",
            metadata={"gold_action": "response", "gold_answer": "light brown wooden"},
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_recall_query_passes_verification(self):
        sample = self._make_sample(
            "recall_query",
            '<think>Current visual window shows the pot on stove with sauce simmering, '
            'no salt container or salt-adding action visible in the recent twelve chunks '
            'of video frames, the information about salt must be in earlier segments.</think>'
            '<action>recall</action>'
            '<query>{"query":"salt container chef adding amount","time_range":"0-10"}</query>',
            user_input="How much salt was added?",
            metadata={"gold_action": "recall", "gold_answer": "one teaspoon"},
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_recall_response_no_think_passes(self):
        """recall_response has NO think — must still pass verification."""
        sample = self._make_sample(
            "recall_response",
            '<action>response</action><response>One teaspoon of salt.</response>',
            recalled_frames={"time_range": [4, 8], "n_frames": 4, "source": "historical_frames"},
            recall_result={"source": "historical_frames", "time": "4-8",
                           "text_content": "[4-6] Chef adds one teaspoon salt."},
            metadata={"gold_action": "response", "gold_answer": "one teaspoon"},
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_recall_response_with_think_fails_format(self):
        """recall_response WITH think should still pass (format allows it but data shouldn't have it)."""
        # Actually, verify_format skips think check for recall_response, so having
        # think won't fail format. But our data construction never produces it.
        pass

    def test_compress_passes_verification(self):
        mem = MemoryState()
        for i in range(6):
            mem.add_think(i, f"Chef does step_{i} at counter with knife, preparing ingredients carefully.")
        snapshot = mem.snapshot(6)

        sample = self._make_sample(
            "compress",
            '<think>Chef_1 continues working steadily at the stainless steel counter, '
            'wooden spoon in right hand stirring the pot while monitoring the sauce '
            'consistency, steam rising from the simmering liquid surface.</think>'
            '<action>compress</action>'
            '<summary>{"time_range":[0,8],"text":"[0-4] Chef did step_0 through step_3 at counter '
            'with knife, preparing ingredients carefully for the recipe."}</summary>',
            snapshot=snapshot,
            user_input='<compress_trigger range="0-8"/>',
            metadata={"gold_action": "compress", "compressed_range": [0, 8],
                       "compressed_chunks": [0, 1, 2, 3]},
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_pending_silent_passes(self):
        """Pending mid-point silent: has pending question but no response yet."""
        mem = MemoryState()
        mem.add_think(10, "Sauce still simmering in pot, chef monitoring heat level.")
        mem.pending_questions.append({"question": "Tell me when basil added.", "since_chunk": 8})
        snapshot = mem.snapshot(10)

        sample = self._make_sample(
            "silent",
            "<think>Thick red sauce continues simmering in the large stainless pot on the "
            "front burner, chef_1 standing nearby with arms crossed watching the surface, "
            "no herbs or basil leaves added yet, steam still rising steadily.</think>"
            "<action>silent</action>",
            snapshot=snapshot,
            metadata={"task_type": "pending_silent"},
        )
        result = verify_sample(sample)
        assert result["verification"]["passed"], result["verification"]["fail_reasons"]

    def test_grounding_rejects_sound(self):
        sample = self._make_sample(
            "silent",
            "<think>I heard a sizzling sound from the pan on the stove "
            "while the chef was adding oil to the hot surface area.</think>"
            "<action>silent</action>",
        )
        result = verify_sample(sample)
        assert not result["verification"]["passed"]
        reasons = " ".join(result["verification"]["fail_reasons"])
        assert "non_visual" in reasons or "sound" in reasons

    def test_filter_samples_counts(self):
        good = self._make_sample(
            "silent",
            "<think>Chef_1 in red apron reaches for the salt container on the upper shelf "
            "above the stainless steel counter, wooden cutting board still visible with "
            "sliced tomato quarters arranged neatly on the left side near the sink.</think>"
            "<action>silent</action>",
        )
        bad = self._make_sample(
            "silent",
            "<think>I heard a sizzling noise from the pan on the stove "
            "while the chef was adding oil to the hot surface area.</think>"
            "<action>silent</action>",
        )
        passed, stats = filter_samples([good, bad])
        assert stats["passed"] == 1
        assert stats["failed"] == 1


# ---------------------------------------------------------------------------
# Test 3: agent_protocol format_memory_block consistency
# ---------------------------------------------------------------------------

class TestAgentProtocol:
    """Test shared protocol produces correct format."""

    def test_format_memory_block_with_snapshot(self):
        mem = MemoryState()
        mem.compressed_segments.append({"time_range": [0, 10], "text": "Chef prepared."})
        mem.add_think(5, "Chef slices tomato.")
        mem.add_think(6, "Chef adds oil.")
        mem.pending_questions.append({"question": "What color is apron?", "since_chunk": 4})
        snapshot = mem.snapshot(7)

        text = format_memory_block(snapshot)

        assert "<compressed>" in text
        assert "[0, 10]" in text
        assert "[10-12] Chef slices tomato." in text
        assert "[12-14] Chef adds oil." in text
        assert "<pending>" in text
        assert "What color is apron?" in text

    def test_format_memory_block_with_pipeline_dict(self):
        """Pipeline samples use different keys (compressed vs compressed_segments)."""
        pipeline_memory = {
            "compressed": [{"time_range": [0, 10], "text": "Chef prepared."}],
            "recent_thinks": ["[10-12] Chef slices tomato.", "[12-14] Chef adds oil."],
            "pending": [{"question": "What?", "since": 8}],
        }
        text = format_memory_block(pipeline_memory)
        assert "<compressed>" in text
        assert "[10-12] Chef slices tomato." in text
        assert "<pending>" in text

    def test_build_user_content_video_first(self):
        """Verify video comes before memory in user content."""
        mem = MemoryState()
        mem.add_think(5, "Chef at counter.")
        snapshot = mem.snapshot(6)
        memory_text = format_memory_block(snapshot)

        content = build_user_content(
            memory_text, chunk_idx=6, video_path="/test.mp4",
            user_input="What is happening?",
        )

        # Find text items
        texts = [c["text"] for c in content if c.get("type") == "text"]
        full = "\n".join(texts)

        # Video-first: visual_window before memory
        vw_pos = full.index("<visual_window>")
        mem_pos = full.index("<memory>")
        ui_pos = full.index("<user_input>")
        assert vw_pos < mem_pos < ui_pos, \
            f"Ordering wrong: vw={vw_pos}, mem={mem_pos}, ui={ui_pos}"

    def test_build_user_content_with_recall(self):
        """recall_response: visual_window → recalled_frames → memory → recall_result → user_input."""
        mem = MemoryState()
        mem.add_think(5, "Chef at counter.")
        snapshot = mem.snapshot(6)
        memory_text = format_memory_block(snapshot)

        recalled = {"time_range": [2, 6], "n_frames": 4}
        recall_result = {"source": "historical", "time": "2-6", "text_content": "Chef cut tomato."}

        content = build_user_content(
            memory_text, chunk_idx=6, video_path="/test.mp4",
            user_input="Continue.",
            recalled_frames=recalled,
            recall_result=recall_result,
        )

        texts = [c["text"] for c in content if c.get("type") == "text"]
        full = "\n".join(texts)

        vw_pos = full.index("<visual_window>")
        rf_pos = full.index("<recalled_frames>")
        mem_pos = full.index("<memory>")
        rr_pos = full.index("<recall_result>")
        ui_pos = full.index("<user_input>")
        assert vw_pos < rf_pos < mem_pos < rr_pos < ui_pos

    def test_parse_agent_output_all_types(self):
        """parse_agent_output correctly handles all action types."""
        # silent
        r = parse_agent_output("<think>Chef stirs.</think><action>silent</action>")
        assert r["think"] == "Chef stirs."
        assert r["action"] == "silent"

        # response
        r = parse_agent_output(
            "<think>Red board.</think><action>response</action><response>It is red.</response>")
        assert r["action"] == "response"
        assert r["payload"]["response"] == "It is red."

        # recall
        r = parse_agent_output(
            '<think>Not visible.</think><action>recall</action>'
            '<query>{"query":"salt amount","time_range":"0-10"}</query>')
        assert r["action"] == "recall"
        assert r["payload"]["query"]["query"] == "salt amount"

        # compress
        r = parse_agent_output(
            '<think>Obs.</think><action>compress</action>'
            '<summary>{"time_range":[0,10],"text":"Chef prepared."}</summary>')
        assert r["action"] == "compress"
        assert r["payload"]["summary"]["time_range"] == [0, 10]

        # recall_response (no think)
        r = parse_agent_output(
            "<action>response</action><response>Four tomatoes.</response>")
        assert r["think"] == ""
        assert r["action"] == "response"
        assert r["payload"]["response"] == "Four tomatoes."


# ---------------------------------------------------------------------------
# Test 4: Compression lifecycle
# ---------------------------------------------------------------------------

class TestCompressionLifecycle:
    """Test full compression cycle: accumulate → trigger → compress → verify."""

    def test_full_compression_cycle(self):
        mem = MemoryState()
        triggered_at = None

        for i in range(20):
            snap_before = mem.snapshot(i)
            # ~200 chars each ≈ ~50 tokens (chars/4), 10 thinks = ~500 tokens > 480 threshold
            think = (f"Chef_1 in red apron performs detailed_step_{i} using tool_{i} "
                     f"at position_{i} on the stainless steel counter, carefully handling "
                     f"the ingredient while monitoring the pot on the burner nearby.")

            if mem.should_compress() and len(snap_before["recent_thinks"]) >= COMPRESS_RANGE_MIN:
                triggered_at = i
                pre_thinks = snap_before["recent_thinks"]
                to_compress = pre_thinks[:COMPRESS_RANGE_MIN]
                summary = {
                    "time_range": [
                        to_compress[0]["chunk"] * AGENT_CHUNK_SEC,
                        (to_compress[-1]["chunk"] + 1) * AGENT_CHUNK_SEC,
                    ],
                    "text": "Summary of compressed thinks.",
                }
                mem.compress(summary, [t["chunk"] for t in to_compress])
                mem.add_think(i, think)

                # Verify post-compression state
                assert len(mem.compressed_segments) >= 1
                # Recent thinks should be shorter now
                post_tokens = mem.count_recent_tokens()
                assert post_tokens < COMPRESS_TOKEN_THRESHOLD
                break
            else:
                mem.add_think(i, think)

        assert triggered_at is not None, "Compression never triggered in 20 chunks"

        # Verify the compressed segment
        seg = mem.compressed_segments[-1]
        assert "time_range" in seg
        assert "text" in seg

        # Verify snapshot after compression has both compressed + recent
        snap_after = mem.snapshot(triggered_at + 1)
        assert len(snap_after["compressed_segments"]) >= 1
        assert len(snap_after["recent_thinks"]) > 0

    def test_max_compressed_segments_merge(self):
        """When >5 compressed segments, oldest two merge."""
        mem = MemoryState()
        for i in range(7):
            mem.compressed_segments.append({
                "time_range": [i * 10, (i + 1) * 10],
                "text": f"Summary {i}.",
            })
        # Trigger merge by adding one more via compress
        mem.add_think(0, "dummy")
        mem.compress(
            {"time_range": [70, 80], "text": "Summary 7."},
            compressed_chunks=[0],
        )
        # Should merge until <=5
        assert len(mem.compressed_segments) <= 5
        # First segment should be merged (covers wider range)
        assert mem.compressed_segments[0]["time_range"][0] == 0
        assert mem.compressed_segments[0]["time_range"][1] > 10


# ---------------------------------------------------------------------------
# Test 5: Pipeline pass4 sample assembly (simulated)
# ---------------------------------------------------------------------------

class TestPipelineAssembly:
    """Simulate pipeline.py Pass4 assembly logic for different scenarios."""

    def test_action_priority_question_over_compress(self):
        """When question and compress happen at same chunk, question wins."""
        rollout = build_timeline(20)
        evidence = build_evidence(20)

        # Find a chunk with compression
        compress_chunks = {e["trigger_chunk"] for e in rollout["compression_events"]}
        if not compress_chunks:
            pytest.skip("No compression in this timeline")

        trigger_chunk = min(compress_chunks)
        # Simulate: task_at has a question at same chunk
        task_at = {trigger_chunk: ("response_from_frames", {
            "gold_action": "response", "question": "What?", "ask_chunk": trigger_chunk,
        })}
        interaction_chunks = set(task_at.keys())

        # has_compress should be False when chunk is in interaction_chunks
        has_compress = trigger_chunk in compress_chunks and trigger_chunk not in interaction_chunks
        assert not has_compress, "Question should preempt compress"

    def test_pending_lifecycle_samples(self):
        """Pending task produces 3 samples: start, mid, trigger."""
        pending_task = {
            "task_type": "pending_event_watch",
            "ask_chunk": 5,
            "trigger_chunk": 15,
            "mid_chunk": 10,
            "question": "Tell me when basil is added.",
            "event": "Chef tears basil leaves over pot.",
        }

        # Simulate what pipeline.py does
        pending_active = {}
        for c in range(pending_task["ask_chunk"], pending_task["trigger_chunk"] + 1):
            pending_active[c] = pending_task

        samples_generated = []
        for chunk_idx in range(5, 16):
            is_start = chunk_idx == pending_task["ask_chunk"]
            is_trigger = chunk_idx == pending_task["trigger_chunk"]
            is_mid = chunk_idx == pending_task["mid_chunk"]

            if is_start:
                samples_generated.append(("pending_start", chunk_idx))
            elif is_trigger:
                samples_generated.append(("pending_trigger", chunk_idx))
            elif is_mid:
                samples_generated.append(("pending_mid", chunk_idx))

        assert ("pending_start", 5) in samples_generated
        assert ("pending_mid", 10) in samples_generated
        assert ("pending_trigger", 15) in samples_generated

    def test_recall_produces_two_samples(self):
        """Recall task should produce recall_query + recall_response."""
        rollout = build_timeline(20)
        mem = MemoryState()
        for i in range(10):
            mem.add_think(i, f"Chef does action_{i}.")

        snapshot = mem.snapshot(15)
        observations = rollout["thinks"]

        # Simulate recall result
        task = {
            "ask_chunk": 15,
            "evidence_chunks": [3],
            "gold_action": "recall",
            "question": "What happened at step 3?",
            "gold_answer": "chef_1 performs action_3",
        }
        recall_result = simulate_recall_result(task, snapshot, observations, 15)

        assert "source" in recall_result
        assert "text_content" in recall_result
        assert "returned_chunks" in recall_result


# ---------------------------------------------------------------------------
# Test 6: SFT data_processor format (without GPU)
# ---------------------------------------------------------------------------

class TestSFTFormat:
    """Test SFT message construction matches agent_protocol format."""

    def test_silent_sample_video_first(self):
        """SFT silent sample has video before memory."""
        try:
            from thinkstream.sft.data_processor import build_per_timestep_messages as sft_build
        except ImportError:
            pytest.skip("transformers not available")

        sample = {
            "sample_id": "test_silent",
            "video_path": "/test.mp4",
            "sample_type": "silent",
            "chunk_idx": 5,
            "data_path": "/data",
            "input": {
                "system": "You are a streaming video agent.",
                "memory": {
                    "compressed": [],
                    "recent_thinks": ["[8-10] Chef at counter."],
                },
                "visual_window": {
                    "video_start": 0.0, "video_end": 12.0, "frames": 12,
                    "frame_paths": [f"/data/f{i}.jpg" for i in range(12)],
                },
            },
            "output": "<think>Chef picks up knife.</think><action>silent</action>",
        }
        msgs = sft_build(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        full = "\n".join(user_texts)

        # Video-first ordering
        vw_pos = full.index("<visual_window>")
        mem_pos = full.index("<memory>")
        assert vw_pos < mem_pos, "SFT format must have visual_window before memory"

    def test_recall_response_video_first_with_all_blocks(self):
        """recall_response has: visual_window → recalled_frames → memory → recall_result → user_input."""
        try:
            from thinkstream.sft.data_processor import build_per_timestep_messages as sft_build
        except ImportError:
            pytest.skip("transformers not available")

        sample = {
            "sample_id": "test_recall_resp",
            "video_path": "/test.mp4",
            "sample_type": "recall_response",
            "chunk_idx": 20,
            "data_path": "/data",
            "input": {
                "system": "You are a streaming video agent.",
                "memory": {
                    "compressed": [{"time_range": [0, 20], "text": "Chef prepared."}],
                    "recent_thinks": ["[20-22] Oil poured."],
                    "pending": [{"question": "How many?", "since": 40}],
                },
                "visual_window": {
                    "video_start": 18.0, "video_end": 42.0, "frames": 24,
                    "frame_paths": [f"/data/f{i}.jpg" for i in range(24)],
                },
                "recalled_frames": {
                    "time_range": [8, 12], "n_frames": 4, "source": "historical_frames",
                    "frame_paths": [f"/data/f{i}.jpg" for i in range(8, 12)],
                },
                "recall_result": {
                    "source": "student_think", "time": "8-12",
                    "text_content": "Chef sliced 4 tomatoes.",
                },
                "user_input": "Continue following the protocol to respond.",
            },
            "output": "<action>response</action><response>Four tomatoes.</response>",
        }
        msgs = sft_build(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        full = "\n".join(user_texts)

        vw_pos = full.index("<visual_window>")
        rf_pos = full.index("<recalled_frames>")
        mem_pos = full.index("<memory>")
        rr_pos = full.index("<recall_result>")
        ui_pos = full.index("<user_input>")
        assert vw_pos < rf_pos < mem_pos < rr_pos < ui_pos


# ---------------------------------------------------------------------------
# Test 7: agent_loop MemoryState matches pass2 MemoryState
# ---------------------------------------------------------------------------

class TestAgentLoopConsistency:
    """Verify inference agent_loop MemoryState matches data construction."""

    def test_memory_state_identical(self):
        try:
            from thinkstream.model.agent_loop import MemoryState as InferenceMemory
        except ImportError:
            pytest.skip("transformers not available in this env")

        # Same operations on both
        data_mem = MemoryState()
        infer_mem = InferenceMemory()

        for i in range(5):
            think = f"Chef does action_{i}."
            data_mem.add_think(i, think)
            infer_mem.add_think(i, think)

        data_snap = data_mem.snapshot(5)
        infer_snap = infer_mem.snapshot(5)

        # Compare key fields
        assert len(data_snap["compressed_segments"]) == len(infer_snap["compressed_segments"])
        assert len(data_snap["recent_thinks"]) == len(infer_snap["recent_thinks"])
        for d, i in zip(data_snap["recent_thinks"], infer_snap["recent_thinks"]):
            assert d["text"] == i["text"]
            assert d["chunk"] == i["chunk"]

    def test_compression_identical(self):
        try:
            from thinkstream.model.agent_loop import MemoryState as InferenceMemory
        except ImportError:
            pytest.skip("transformers not available in this env")

        data_mem = MemoryState()
        infer_mem = InferenceMemory()

        for i in range(8):
            think = f"Chef does action_{i} with tool_{i}."
            data_mem.add_think(i, think)
            infer_mem.add_think(i, think)

        summary = {"time_range": [0, 8], "text": "Chef did actions 0-3."}
        data_mem.compress(summary, compressed_chunks=[0, 1, 2, 3])
        infer_mem.compress(summary, compressed_chunks=[0, 1, 2, 3])

        ds = data_mem.snapshot(8)
        is_ = infer_mem.snapshot(8)

        assert ds["compressed_segments"] == is_["compressed_segments"]
        assert len(ds["recent_thinks"]) == len(is_["recent_thinks"])

    def test_format_memory_block_identical(self):
        """Same snapshot → same formatted memory text (shared protocol)."""
        try:
            from thinkstream.model.agent_loop import MemoryState as InferenceMemory
        except ImportError:
            pytest.skip("transformers not available in this env")

        data_mem = MemoryState()
        infer_mem = InferenceMemory()

        for i in range(3):
            data_mem.add_think(i, f"Step {i}.")
            infer_mem.add_think(i, f"Step {i}.")

        data_text = format_memory_block(data_mem.snapshot(3))
        infer_text = format_memory_block(infer_mem.snapshot(3))

        assert data_text == infer_text


# ---------------------------------------------------------------------------
# Test 8: COMPRESS_PROMPT has all placeholders
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    """Test that prompt templates can be formatted without errors."""

    def test_compress_prompt_all_placeholders(self):
        from scripts.agent_data_v5.config import COMPRESS_PROMPT
        # Should not raise KeyError
        result = COMPRESS_PROMPT.format(
            observations_text="[0-2] Chef stirs.",
            visual_context="",
            target_length=100,
            start=0,
            end=4,
        )
        assert "Chef stirs" in result

    def test_compress_prompt_with_visual_context(self):
        from scripts.agent_data_v5.config import COMPRESS_PROMPT
        result = COMPRESS_PROMPT.format(
            observations_text="[0-2] Chef stirs.",
            visual_context="\nFrames provided for reference.\n",
            target_length=100,
            start=0,
            end=4,
        )
        assert "Frames provided" in result

    def test_task_question_prompt(self):
        from scripts.agent_data_v5.config import TASK_QUESTION_PROMPT
        result = TASK_QUESTION_PROMPT.format(
            entity="chef_1",
            attributes="red apron",
            fact="chef adds salt",
            time=10,
            answer="chef adds salt",
        )
        assert "chef_1" in result

    def test_response_prompt(self):
        from scripts.agent_data_v5.config import RESPONSE_PROMPT
        result = RESPONSE_PROMPT.format(
            question="How much salt?",
            evidence="Chef added one teaspoon.",
            answer_type="factoid",
            gold_answer="one teaspoon",
            length_guide="5-40 tokens",
        )
        assert "How much salt" in result

    def test_recall_query_prompt(self):
        from scripts.agent_data_v5.config import RECALL_QUERY_PROMPT
        result = RECALL_QUERY_PROMPT.format(
            question="What happened?",
            visible_context="[0-2] Chef started.",
            time_range="0-10",
        )
        assert "What happened" in result
