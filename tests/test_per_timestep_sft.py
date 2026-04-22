"""Tests for per-timestep SFT data pipeline.

Validates message construction, label masking, and sample weight
without requiring GPU or real video files.
"""

import json
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Test fixtures: mock pipeline samples
# ---------------------------------------------------------------------------

def _make_silent_sample():
    return {
        "sample_id": "vid001_t10_silent_0",
        "video_id": "vid001",
        "video_path": "/data/videos/vid001.mp4",
        "sample_type": "silent",
        "chunk_idx": 5,
        "phase": "1",
        "data_path": "/data",
        "input": {
            "system": "You are a streaming video agent.",
            "memory": {
                "compressed": [],
                "recent_thinks": [
                    "[6-8] Chef places cutting board on counter.",
                    "[8-10] Chef picks up knife from drawer.",
                ],
            },
            "visual_window": {
                "video_start": 0.0,
                "video_end": 12.0,
                "frames": 12,
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(12)],
            },
        },
        "output": "<think>Chef begins slicing red tomato on board.</think><action>silent</action>",
    }


def _make_response_sample():
    return {
        "sample_id": "vid001_t20_response_1",
        "video_id": "vid001",
        "video_path": "/data/videos/vid001.mp4",
        "sample_type": "response",
        "chunk_idx": 10,
        "phase": "1",
        "data_path": "/data",
        "input": {
            "system": "You are a streaming video agent.",
            "memory": {
                "compressed": [
                    {"time_range": [0, 10], "text": "Chef prepared workspace, got knife."},
                ],
                "recent_thinks": [
                    "[10-12] Tomato sliced into quarters.",
                    "[12-14] Garlic minced on board.",
                ],
            },
            "visual_window": {
                "video_start": 0.0,
                "video_end": 22.0,
                "frames": 22,
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(22)],
            },
            "user_input": "What color is the cutting board?",
        },
        "output": "<think>Wooden cutting board visible, light brown color.</think>"
                  "<action>response</action><response>The cutting board is light brown, wooden.</response>",
    }


def _make_recall_query_sample():
    return {
        "sample_id": "vid001_t40_recall_2",
        "video_id": "vid001",
        "video_path": "/data/videos/vid001.mp4",
        "sample_type": "recall_query",
        "chunk_idx": 20,
        "phase": "2",
        "data_path": "/data",
        "input": {
            "system": "You are a streaming video agent.",
            "memory": {
                "compressed": [
                    {"time_range": [0, 20], "text": "Chef sliced tomatoes and garlic."},
                ],
                "recent_thinks": [
                    "[20-22] Oil poured into pot.",
                    "[22-24] Garlic added to oil.",
                ],
            },
            "visual_window": {
                "video_start": 18.0,
                "video_end": 42.0,
                "frames": 24,
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(24)],
            },
            "user_input": "How many tomatoes were sliced?",
        },
        "output": '<think>No tomatoes visible in current window.</think>'
                  '<action>recall</action>'
                  '<query>{"query":"tomato slicing count","time_range":"0-20"}</query>',
    }


def _make_recall_response_sample():
    return {
        "sample_id": "vid001_t40_post_recall",
        "video_id": "vid001",
        "video_path": "/data/videos/vid001.mp4",
        "sample_type": "recall_response",
        "chunk_idx": 20,
        "phase": "2",
        "data_path": "/data",
        "input": {
            "system": "You are a streaming video agent.",
            "memory": {
                "compressed": [
                    {"time_range": [0, 20], "text": "Chef sliced tomatoes and garlic."},
                ],
                "recent_thinks": [
                    "[20-22] Oil poured into pot.",
                    "[22-24] Garlic added to oil.",
                ],
                "pending": [
                    {"question": "How many tomatoes were sliced?", "since": 40, "type": "awaiting_recall_response"},
                ],
            },
            "visual_window": {
                "video_start": 18.0,
                "video_end": 42.0,
                "frames": 24,
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(24)],
            },
            "recalled_frames": {
                "time_range": [8, 12],
                "n_frames": 4,
                "source": "historical_frames",
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(8, 12)],
            },
            "recall_result": {
                "source": "student_think",
                "time": "8-12",
                "text_content": "Retrieved: [8-10] Chef slices 4 Roma tomatoes on board.",
            },
            "user_input": "Continue following the protocol to respond.",
        },
        "output": "<action>response</action><response>Four Roma tomatoes were sliced.</response>",
    }


def _make_compress_sample():
    return {
        "sample_id": "vid001_t44_compress",
        "video_id": "vid001",
        "video_path": "/data/videos/vid001.mp4",
        "sample_type": "compress",
        "chunk_idx": 22,
        "phase": "C1",
        "data_path": "/data",
        "input": {
            "system": "You are a streaming video agent.",
            "memory": {
                "compressed": [
                    {"time_range": [0, 10], "text": "Chef prepared workspace."},
                ],
                "recent_thinks": [
                    "[10-12] Tomato sliced into quarters.",
                    "[12-14] Garlic minced on board.",
                    "[14-16] Oil poured into stainless pot.",
                    "[16-18] Garlic added to hot oil.",
                    "[18-20] Tomato quarters placed into pot.",
                ],
            },
            "visual_window": {
                "video_start": 20.0,
                "video_end": 46.0,
                "frames": 24,
                "frame_paths": [f"/data/frames/vid001/f{i:04d}.jpg" for i in range(24)],
            },
            "user_input": '<compress_trigger>{"range":[10,20]}</compress_trigger>',
        },
        "output": '<think>Chef stirs pot with wooden spoon.</think>'
                  '<action>compress</action>'
                  '<summary>{"time_range":[10,20],"text":"[10-14] Tomato quartered, garlic minced. '
                  '[14-20] Oil heated, garlic browned, tomatoes added to pot."}</summary>',
    }


# ---------------------------------------------------------------------------
# Tests: message construction
# ---------------------------------------------------------------------------

class TestBuildMessages:
    """Test build_per_timestep_messages constructs correct message structure."""

    def test_silent_sample_structure(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_silent_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_memory_block_present(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_silent_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        # Memory is now AFTER visual_window (video-first layout v3.0)
        memory_texts = [t for t in user_texts if "<memory>" in t]
        assert len(memory_texts) == 1
        assert "</memory>" in memory_texts[0]
        assert "[6-8] Chef places cutting board" in memory_texts[0]

    def test_visual_window_header_has_current_time(self):
        """P0-3: visual_window must contain current_time."""
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_silent_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        vw_text = [t for t in user_texts if "<visual_window>" in t][0]
        assert "current_time" in vw_text

    def test_video_entry_uses_frame_paths(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_silent_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        video_entries = [c for c in msgs[1]["content"] if c.get("type") == "video"]
        assert len(video_entries) == 1
        assert isinstance(video_entries[0]["video"], list)  # frame_paths = list

    def test_response_has_user_input(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_response_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        user_input_texts = [t for t in user_texts if "<user_input>" in t]
        assert len(user_input_texts) == 1
        assert "What color" in user_input_texts[0]

    def test_recall_response_has_all_blocks(self):
        """v3.0 ordering: visual_window → recalled_frames → memory → recall_result → user_input."""
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_recall_response_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        full_text = "\n".join(user_texts)

        assert "<memory>" in full_text
        assert "<visual_window>" in full_text
        assert "<recalled_frames>" in full_text
        assert "<recall_result>" in full_text
        assert "<user_input>" in full_text
        assert "<pending>" in full_text

        # v3.0 ordering: visual_window before memory, memory before user_input
        vw_pos = full_text.index("<visual_window>")
        rf_pos = full_text.index("<recalled_frames>")
        mem_pos = full_text.index("<memory>")
        rr_pos = full_text.index("<recall_result>")
        ui_pos = full_text.index("<user_input>")
        assert vw_pos < rf_pos, "visual_window must precede recalled_frames"
        assert rf_pos < mem_pos, "recalled_frames must precede memory"
        assert mem_pos < rr_pos, "memory must precede recall_result"
        assert rr_pos < ui_pos, "recall_result must precede user_input (P0-1)"

    def test_recall_response_has_two_video_entries(self):
        """Visual window + recalled frames = 2 video entries."""
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_recall_response_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        video_entries = [c for c in msgs[1]["content"] if c.get("type") == "video"]
        assert len(video_entries) == 2

    def test_compress_has_trigger(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_compress_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        full_text = "\n".join(user_texts)
        assert "<compress_trigger>" in full_text

    def test_assistant_output_matches(self):
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_silent_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        asst = msgs[2]
        assert asst["content"][0]["text"] == sample["output"]

    def test_compressed_segment_uses_json_inside_tags(self):
        """P0-7: Approach B — JSON inside tags, not XML attributes."""
        from thinkstream.sft.data_processor import build_per_timestep_messages

        sample = _make_response_sample()
        msgs = build_per_timestep_messages(sample, Path("/data"))

        user_texts = [c["text"] for c in msgs[1]["content"] if c.get("type") == "text"]
        memory_text = user_texts[0]
        # Should contain <compressed>{...json...}</compressed>
        assert "<compressed>{" in memory_text
        assert "}</compressed>" in memory_text
        # Should NOT contain XML attributes like <compressed time="0-10">
        assert 'time="' not in memory_text.split("<compressed>")[1].split("</compressed>")[0]


class TestActionWeights:
    """Test per-sample loss weight assignment."""

    def test_all_sample_types_have_weights(self):
        from thinkstream.sft.data_processor import ACTION_WEIGHTS

        expected_types = ["silent", "response", "recall_query", "recall_response", "compress", "merge_compress"]
        for t in expected_types:
            assert t in ACTION_WEIGHTS, f"Missing weight for {t}"

    def test_rare_actions_weighted_higher(self):
        from thinkstream.sft.data_processor import ACTION_WEIGHTS

        assert ACTION_WEIGHTS["recall_query"] > ACTION_WEIGHTS["silent"]
        assert ACTION_WEIGHTS["compress"] > ACTION_WEIGHTS["silent"]


class TestSpecialTokens:
    """Test special token lists."""

    def test_wrapper_tags_paired(self):
        """Wrapper tags (memory, compressed, etc.) should have closing tags.
        Action value tokens (<silent>, <response>) are standalone, not paired.
        """
        from thinkstream.sft.data_processor import SPECIAL_TOKENS_AGENT

        # Tags that wrap content and must be paired
        wrapper_tags = [
            "<think>", "<action>", "<query>", "<recall_result>",
            "<memory>", "<compressed>", "<pending>",
            "<visual_window>", "<recalled_frames>", "<user_input>",
            "<summary>", "<compress_trigger>", "<response>",
        ]
        closing = [t for t in SPECIAL_TOKENS_AGENT if t.startswith("</")]

        for tag in wrapper_tags:
            close_tag = f"</{tag[1:]}"
            assert close_tag in closing, f"Wrapper tag {tag} has no matching closing tag"


class TestMemoryFormat:
    """Test memory block formatting."""

    def test_empty_memory(self):
        from thinkstream.sft.data_processor import _format_memory_block

        text = _format_memory_block({"compressed": [], "recent_thinks": []})
        assert text == ""

    def test_compressed_json_format(self):
        from thinkstream.sft.data_processor import _format_memory_block

        mem = {
            "compressed": [{"time_range": [0, 10], "text": "Chef prepared."}],
            "recent_thinks": ["[10-12] Slicing tomato."],
        }
        text = _format_memory_block(mem)
        assert "<compressed>" in text
        assert '"time_range": [0, 10]' in text
        assert "[10-12] Slicing tomato." in text

    def test_pending_format(self):
        from thinkstream.sft.data_processor import _format_memory_block

        mem = {
            "compressed": [],
            "recent_thinks": [],
            "pending": [{"question": "When is basil added?", "since": 24}],
        }
        text = _format_memory_block(mem)
        assert "<pending>" in text
        assert '"since": 24' in text
        assert "basil" in text
