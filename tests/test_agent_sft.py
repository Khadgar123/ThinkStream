"""Tests for the Agent SFT data pipeline.

Validates data format, message building, label masking, filtering,
and all episode types (R1, RC1, R3, RC7) for the 3-action agent protocol.

Usage:
    python -m pytest tests/test_agent_sft.py -v
    python tests/test_agent_sft.py            # standalone
"""

import json
import random
import re
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import the SFT data processor; flag if unavailable.
# The module has heavy deps (transformers, torch, qwen_vl_utils, torchcodec)
# that may not be installed locally.
_HAS_SFT_PROCESSOR = False
try:
    # Stub truly optional deps first
    for _mod_name in [
        "qwen_vl_utils",
        "torchcodec",
        "torchcodec.decoders",
    ]:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)
    if not hasattr(sys.modules["qwen_vl_utils"], "process_vision_info"):
        sys.modules["qwen_vl_utils"].process_vision_info = MagicMock()
    if not hasattr(sys.modules["torchcodec.decoders"], "VideoDecoder"):
        sys.modules["torchcodec.decoders"].VideoDecoder = MagicMock()

    from thinkstream.data.stream_data_processor import (
        _build_agent_messages,
        find_assistant_spans,
    )
    _HAS_SFT_PROCESSOR = True
except (ImportError, Exception):
    pass

import pytest
_requires_sft_processor = pytest.mark.skipif(
    not _HAS_SFT_PROCESSOR,
    reason="thinkstream.data.stream_data_processor not importable (missing ML deps)",
)

from scripts.agent_data_pipeline.generate_data import (
    assemble_episode,
    _assemble_r3,
    _assemble_rc7,
    generate_triplets,
    filter_episode,
    _check_query_leakage,
    _build_recall_result,
    _extract_query_keywords,
    _make_episode,
    _FORMAT_RE,
    AGENT_SYSTEM_PROMPT,
    AGENT_CHUNK_SEC,
)


# =====================================================================
# Test fixtures
# =====================================================================


def make_segments(n=20, seg_sec=4):
    """Create mock segments for a video."""
    return [
        {
            "segment_id": f"seg_{i:03d}",
            "start_sec": i * seg_sec,
            "end_sec": (i + 1) * seg_sec,
            "action": f"Action in segment {i}",
            "entities": [f"entity_{i}"],
            "visual_details": [
                {"entity": f"entity_{i}", "attributes": f"attr_{i}"}
            ],
            "ocr": "none" if i % 5 != 0 else f"text_{i}",
            "change": "state changed" if i > 0 else "video start",
            "frame_paths": [
                f"/frames/frame_{i * 4 + j:06d}.jpg" for j in range(4)
            ],
        }
        for i in range(n)
    ]


def make_r1_task(video_id="test_vid"):
    """Create a simple R1 (immediate answer) task."""
    return {
        "task_id": f"{video_id}_task_r1",
        "video_id": video_id,
        "task_type": "R1",
        "question": "What is the person doing?",
        "ask_segment": "seg_005",
        "ask_time_sec": 20.0,
        "answer_time_sec": 20.0,
        "expected_answer": "cooking",
        "natural_response": "The person is cooking.",
        "answer_type": "entity",
        "need_recall": False,
        "think_at_ask": "The current frames show the person cooking.",
    }


def make_s3r2_task(video_id="test_vid"):
    """Create a delayed answer (S3_R2) task."""
    return {
        "task_id": f"{video_id}_task_s3r2",
        "video_id": video_id,
        "task_type": "S3_R2",
        "question": "Will the person add salt next?",
        "ask_segment": "seg_005",
        "ask_time_sec": 20.0,
        "answer_segment": "seg_008",
        "answer_time_sec": 32.0,
        "expected_answer": "yes",
        "natural_response": "Yes, the person just added salt.",
        "answer_type": "yesno",
        "need_recall": False,
        "think_at_ask": "User asked about salt. Haven't seen it yet.",
    }


def make_rc1_task(video_id="test_vid"):
    """Create a recall task (RC1)."""
    return {
        "task_id": f"{video_id}_task_rc1",
        "video_id": video_id,
        "task_type": "RC1",
        "question": "What color was the apron at the beginning?",
        "support_segment": "seg_002",
        "ask_segment": "seg_015",
        "ask_time_sec": 60.0,
        "answer_time_sec": 60.0,
        "support_time_sec": 8.0,
        "expected_answer": "red",
        "natural_response": "The apron was red.",
        "answer_type": "slot",
        "need_recall": True,
        "think_at_ask": "Cannot see the apron in current window, need to recall.",
        "think_after_recall": "Retrieved the apron info, it was red.",
        "query_candidates": [
            {
                "query": "chef apron color",
                "time_bias": "past_far",
                "target": "entity",
                "topk": 3,
            }
        ],
    }


def make_r3_task(video_id="test_vid"):
    """Create a progressive answer (R3) task."""
    return {
        "task_id": f"{video_id}_task_r3",
        "video_id": video_id,
        "task_type": "R3",
        "question": "What ingredients are being added?",
        "ask_segment": "seg_005",
        "ask_time_sec": 20.0,
        "response_segments": ["seg_005", "seg_008", "seg_011"],
        "response_times_sec": [20.0, 32.0, 44.0],
        "partial_answers": ["Adding oil", "Now adding garlic", "Adding onions"],
        "expected_answer": "oil, garlic, onions",
        "natural_response": "Oil, garlic, and onions were added.",
        "answer_type": "entity",
        "answer_time_sec": 44.0,
        "need_recall": False,
        "think_at_ask": "See oil being added.",
    }


def make_rc7_task(video_id="test_vid"):
    """Create a multi-turn follow-up recall (RC7) task."""
    return {
        "task_id": f"{video_id}_task_rc7",
        "video_id": video_id,
        "task_type": "RC7",
        "question": "Earlier you said something was added — how much salt?",
        "base_question": "What did the chef add to the soup?",
        "base_answer": "Salt and pepper",
        "base_ask_segment": "seg_003",
        "base_ask_time_sec": 12.0,
        "support_segment": "seg_003",
        "support_time_sec": 12.0,
        "ask_segment": "seg_015",
        "ask_time_sec": 60.0,
        "answer_time_sec": 60.0,
        "expected_answer": "two pinches",
        "natural_response": "It was about two pinches of salt.",
        "answer_type": "slot",
        "need_recall": True,
        "think_at_ask": "The earlier conversation is outside my window.",
        "think_after_recall": "Retrieved earlier conversation, can answer.",
        "query_candidates": [
            {
                "query": "salt added soup amount",
                "time_bias": "past_far",
                "target": "entity",
                "topk": 3,
            }
        ],
    }


# =====================================================================
# Helper
# =====================================================================


def _get_user_video_times(episode):
    """Extract all (video_start, video_end) from user messages."""
    times = []
    for msg in episode.get("messages", []):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "video":
                    times.append((part["video_start"], part["video_end"]))
    return times


def _get_assistant_actions(episode):
    """Extract action type from each assistant message."""
    actions = []
    for msg in episode.get("messages", []):
        if msg["role"] == "assistant":
            c = msg["content"]
            if "<action>silent</action>" in c:
                actions.append("silent")
            elif "<action>response</action>" in c:
                actions.append("response")
            elif "<action>recall</action>" in c:
                actions.append("recall")
    return actions


# =====================================================================
# 1. Data format tests
# =====================================================================


class TestDataFormat:
    """Verify pipeline output data format."""

    def test_r1_episode_structure(self):
        segments = make_segments()
        task = make_r1_task()
        ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")

        assert ep["protocol_version"] == "3action"
        assert ep["task_type"] == "R1"

        msgs = ep["messages"]
        assert msgs[0]["role"] == "system"
        # After system: alternating user/assistant (with possible recall injection)
        for i in range(1, len(msgs) - 1, 2):
            assert msgs[i]["role"] == "user", f"msg[{i}] should be user"

    def test_user_messages_structured_content(self):
        """User video messages must use list content with explicit times."""
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task(), make_r3_task()]:
            ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")
            for msg in ep["messages"]:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    video_parts = [
                        p for p in msg["content"] if p.get("type") == "video"
                    ]
                    assert len(video_parts) == 1
                    vp = video_parts[0]
                    assert "video_start" in vp, "Missing video_start"
                    assert "video_end" in vp, "Missing video_end"
                    assert vp["video_end"] > vp["video_start"]

    def test_no_question_tags(self):
        """Messages must NOT contain <question> tags."""
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task(), make_r3_task(), make_rc7_task()]:
            ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")
            for msg in ep["messages"]:
                c = msg.get("content", "")
                if isinstance(c, str):
                    assert "<question>" not in c, f"Found <question> in: {c[:80]}"
                elif isinstance(c, list):
                    for part in c:
                        if part.get("type") == "text":
                            assert "<question>" not in part["text"]

    def test_no_video_string_placeholder(self):
        """Messages must NOT use legacy '<video>' string placeholder."""
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task()]:
            ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")
            for msg in ep["messages"]:
                if msg["role"] == "user":
                    c = msg["content"]
                    if isinstance(c, str):
                        assert "<video>" not in c

    def test_assistant_format_compliance(self):
        """All assistant messages must match <think>...<action>... format."""
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task(), make_r3_task(), make_rc7_task()]:
            ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")
            for msg in ep["messages"]:
                if msg["role"] == "assistant":
                    assert _FORMAT_RE.search(msg["content"]), (
                        f"Format mismatch: {msg['content'][:80]}"
                    )

    def test_canonical_answer_present(self):
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        ca = ep["canonical_answer"]
        assert "answer_type" in ca
        assert ca["value"]["answer"]

    def test_episode_jsonl_serializable(self):
        """Episode must be JSON-serializable for JSONL output."""
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task(), make_r3_task(), make_rc7_task()]:
            ep = assemble_episode(task, segments, 80.0, "/test/video.mp4")
            line = json.dumps(ep, ensure_ascii=False)
            parsed = json.loads(line)
            assert parsed["episode_id"] == ep["episode_id"]


# =====================================================================
# 2. Video time range tests
# =====================================================================


class TestVideoTimeRanges:

    def test_video_times_ascending(self):
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        times = _get_user_video_times(ep)
        starts = [t[0] for t in times]
        assert starts == sorted(starts), f"Non-ascending: {starts}"

    def test_video_times_contiguous_r1(self):
        """R1 episode should have contiguous 2s chunks."""
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        times = _get_user_video_times(ep)
        for i in range(1, len(times)):
            gap = times[i][0] - times[i - 1][1]
            assert abs(gap) < 0.01, f"Gap at chunk {i}: {gap}"

    def test_rc7_has_time_gap(self):
        """RC7 episode must have a time gap between base and follow-up."""
        segments = make_segments()
        ep = assemble_episode(make_rc7_task(), segments, 80.0, "/v.mp4")
        times = _get_user_video_times(ep)
        assert len(times) >= 4, f"Too few video chunks: {len(times)}"

        # Find the largest gap
        max_gap = 0
        for i in range(1, len(times)):
            gap = times[i][0] - times[i - 1][1]
            max_gap = max(max_gap, gap)
        assert max_gap > 0.1, (
            f"RC7 should have a time gap, max_gap={max_gap}, times={times}"
        )

    def test_video_end_within_duration(self):
        """All video_end values should be <= video duration."""
        duration = 80.0
        segments = make_segments()
        for task in [make_r1_task(), make_rc1_task(), make_r3_task()]:
            ep = assemble_episode(task, segments, duration, "/v.mp4")
            times = _get_user_video_times(ep)
            for vs, ve in times:
                assert ve <= duration + 0.01, f"video_end {ve} > duration {duration}"

    def test_chunk_duration_is_2s(self):
        """Each chunk should be exactly AGENT_CHUNK_SEC (2s) wide."""
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        times = _get_user_video_times(ep)
        for vs, ve in times:
            assert abs((ve - vs) - AGENT_CHUNK_SEC) < 0.01, (
                f"Chunk duration {ve - vs} != {AGENT_CHUNK_SEC}"
            )


# =====================================================================
# 3. Action & role tests (LlamaFactory agentic convention)
# =====================================================================


class TestAgenticActions:

    def test_r1_has_response(self):
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)
        assert "response" in actions

    def test_recall_episode_actions(self):
        """RC1: should have silent → recall → (recall_result) → response."""
        segments = make_segments()
        ep = assemble_episode(make_rc1_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)

        assert "recall" in actions
        assert "response" in actions
        recall_idx = actions.index("recall")
        resp_after = [i for i, a in enumerate(actions) if a == "response" and i > recall_idx]
        assert resp_after, "Response must follow recall"

    def test_recall_result_is_user_role(self):
        """Recall result must be user role (no gradient / LlamaFactory observation)."""
        segments = make_segments()
        ep = assemble_episode(make_rc1_task(), segments, 80.0, "/v.mp4")
        for msg in ep["messages"]:
            if msg["role"] == "system":
                continue  # system prompt mentions <recall_result> in rules
            if isinstance(msg.get("content", ""), str) and "<recall_result>" in msg["content"]:
                assert msg["role"] == "user", "recall_result must be user role"

    def test_s3r2_has_waiting_silents(self):
        """S3_R2: silent chunks between ask and delayed answer."""
        segments = make_segments()
        ep = assemble_episode(make_s3r2_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)

        # Should have silents before the response
        resp_idx = next(i for i, a in enumerate(actions) if a == "response")
        silents_before = [i for i, a in enumerate(actions) if a == "silent" and i < resp_idx]
        assert len(silents_before) >= 1, "S3_R2 needs waiting silents"

    def test_r3_multiple_responses(self):
        segments = make_segments()
        ep = assemble_episode(make_r3_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)
        resp_count = actions.count("response")
        assert resp_count >= 2, f"R3 needs >= 2 responses, got {resp_count}"

    def test_r3_silents_between_responses(self):
        segments = make_segments()
        ep = assemble_episode(make_r3_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)
        # Find consecutive response pair and check for silent between
        resp_indices = [i for i, a in enumerate(actions) if a == "response"]
        assert len(resp_indices) >= 2
        # Between first two responses there should be at least one silent
        between = actions[resp_indices[0] + 1 : resp_indices[1]]
        assert "silent" in between, "R3 needs silents between responses"

    def test_rc7_has_base_qa_then_recall(self):
        """RC7: base response → gap → recall → response."""
        segments = make_segments()
        ep = assemble_episode(make_rc7_task(), segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)

        assert "recall" in actions
        resp_indices = [i for i, a in enumerate(actions) if a == "response"]
        recall_idx = actions.index("recall")

        # At least one response before recall (base Q&A)
        before = [i for i in resp_indices if i < recall_idx]
        assert before, "RC7 needs base response before recall"

        # At least one response after recall
        after = [i for i in resp_indices if i > recall_idx]
        assert after, "RC7 needs response after recall"

    def test_rc7_base_question_in_messages(self):
        segments = make_segments()
        task = make_rc7_task()
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")

        base_q = task["base_question"]
        found = False
        for msg in ep["messages"]:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "text" and base_q in part["text"]:
                        found = True
        assert found, "RC7 should include base question"


# =====================================================================
# 4. Triplet binding tests
# =====================================================================


class TestTriplets:

    def test_triplet_from_recall_task(self):
        triplets = generate_triplets([make_rc1_task()])
        assert len(triplets) == 1
        tri = triplets[0]

        assert tri["recall"]["task_type"] == "RC1"
        assert tri["control"]["task_type"] == "R1"
        assert tri["control"]["need_recall"] is False
        assert tri["false_negative"]["task_type"] == "R7"
        assert tri["false_negative"]["need_recall"] is False

    def test_rc7_no_triplet(self):
        """RC7 is excluded from triplet binding."""
        triplets = generate_triplets([make_rc7_task()])
        assert len(triplets) == 0

    def test_simple_task_no_triplet(self):
        triplets = generate_triplets([make_r1_task()])
        assert len(triplets) == 0

    def test_false_negative_has_recall_phrasing(self):
        triplets = generate_triplets([make_rc1_task()])
        fn = triplets[0]["false_negative"]
        q = fn["question"].lower()
        recall_words = ["earlier", "before", "previously", "a while ago", "at the beginning"]
        assert any(w in q for w in recall_words), f"FN question lacks recall phrasing: {q}"

    def test_control_ask_time_near_support(self):
        """Control episode asks when evidence is still visible."""
        task = make_rc1_task()
        triplets = generate_triplets([task])
        ctrl = triplets[0]["control"]
        assert ctrl["ask_time_sec"] < task["ask_time_sec"]
        assert abs(ctrl["ask_time_sec"] - task["support_time_sec"]) < 10


# =====================================================================
# 5. Filtering tests
# =====================================================================


class TestFiltering:

    def test_valid_episode_passes(self):
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        ok, reason = filter_episode(ep)
        assert ok, f"Should pass, got: {reason}"

    def test_too_few_messages_fails(self):
        ep = {"messages": [{"role": "system", "content": "hi"}], "canonical_answer": {"value": {"answer": "x"}}}
        ok, reason = filter_episode(ep)
        assert not ok
        assert reason == "too_few_messages"

    def test_empty_answer_fails(self):
        segments = make_segments()
        task = make_r1_task()
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        ep["canonical_answer"]["value"]["answer"] = ""
        ok, reason = filter_episode(ep)
        assert not ok
        assert reason == "empty_answer"

    def test_bad_format_fails(self):
        ep = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "no format"},
            ],
            "canonical_answer": {"value": {"answer": "x"}},
        }
        ok, reason = filter_episode(ep)
        assert not ok
        assert "format_mismatch" in reason

    def test_all_task_types_pass_filter(self):
        """All standard task types should produce filterable episodes."""
        segments = make_segments()
        for task_fn in [make_r1_task, make_s3r2_task, make_rc1_task, make_r3_task, make_rc7_task]:
            task = task_fn()
            ep = assemble_episode(task, segments, 80.0, "/v.mp4")
            ok, reason = filter_episode(ep)
            assert ok, f"{task['task_type']} failed filter: {reason}"


# =====================================================================
# 6. Action distribution tests
# =====================================================================


class TestActionDistribution:

    def test_action_counts_balanced(self):
        """Mixed task set should produce all three action types."""
        segments = make_segments()
        tasks = [make_r1_task(), make_rc1_task(), make_r3_task(), make_rc7_task()]

        counts = Counter()
        for task in tasks:
            ep = assemble_episode(task, segments, 80.0, "/v.mp4")
            for a in _get_assistant_actions(ep):
                counts[a] += 1

        total = sum(counts.values())
        assert total > 0
        assert "silent" in counts
        assert "response" in counts
        assert "recall" in counts
        # Silent should dominate
        assert counts["silent"] > counts["response"], (
            f"silent ({counts['silent']}) should > response ({counts['response']})"
        )


# =====================================================================
# 7. _build_agent_messages tests (mock video I/O)
# =====================================================================


@_requires_sft_processor
class TestBuildAgentMessages:
    """Test _build_agent_messages with mocked video duration."""

    def _build(self, item):
        """Call _build_agent_messages with mocked _get_duration."""
        with patch(
            "thinkstream.data.stream_data_processor._get_duration",
            return_value=80.0,
        ), patch(
            "thinkstream.data.stream_data_processor._make_abs_paths",
            side_effect=lambda base, f: f,
        ):
            return _build_agent_messages(item, Path("."))

    def test_structured_content_passthrough(self):
        """Structured video content should be preserved with abs path."""
        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video_start": 10.0, "video_end": 12.0},
                        {"type": "text", "text": "question?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "<think>think</think><action>response</action><response>ans</response>",
                },
            ],
        }
        result = self._build(item)
        msgs = result["messages"]

        # User message should have resolved video path
        user_msg = msgs[1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        vid_part = [p for p in user_msg["content"] if p["type"] == "video"][0]
        assert vid_part["video"] == "/test/video.mp4"
        assert vid_part["video_start"] == 10.0
        assert vid_part["video_end"] == 12.0
        assert vid_part["nframes"] == 2  # FRAMES_PER_CHUNK

    def test_legacy_string_format(self):
        """Legacy '<video>' strings should be converted to structured."""
        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "<video>\n<question>\nWhat?\n</question>"},
                {
                    "role": "assistant",
                    "content": "<think>t</think><action>silent</action>",
                },
            ],
        }
        result = self._build(item)
        user_msg = result["messages"][1]
        assert isinstance(user_msg["content"], list)
        vid = [p for p in user_msg["content"] if p["type"] == "video"]
        assert len(vid) == 1
        # <question> tags should be stripped
        text_parts = [p for p in user_msg["content"] if p["type"] == "text"]
        for tp in text_parts:
            assert "<question>" not in tp["text"]

    def test_recall_result_kept_as_string(self):
        """Recall result user messages stay as plain strings."""
        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video_start": 0, "video_end": 2}],
                },
                {
                    "role": "assistant",
                    "content": '<think>t</think><action>recall</action><query>{"query":"x"}</query>',
                },
                {
                    "role": "user",
                    "content": "<recall_result>result</recall_result>\nContinue.",
                },
                {
                    "role": "assistant",
                    "content": "<think>t</think><action>response</action><response>a</response>",
                },
            ],
        }
        result = self._build(item)
        recall_msg = result["messages"][3]
        assert recall_msg["role"] == "user"
        assert isinstance(recall_msg["content"], str)
        assert "<recall_result>" in recall_msg["content"]

    def test_video_meta_covers_all_chunks(self):
        """video_meta should span the full time range of video chunks."""
        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video_start": 10.0, "video_end": 12.0}],
                },
                {"role": "assistant", "content": "<think>t</think><action>silent</action>"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video_start": 50.0, "video_end": 52.0}],
                },
                {"role": "assistant", "content": "<think>t</think><action>silent</action>"},
            ],
        }
        result = self._build(item)
        vm = result["video_meta"]
        assert vm["total_start"] == 10.0
        assert vm["total_end"] == 52.0
        assert vm["num_chunks"] == 2

    def test_assistant_wrapped_in_list(self):
        """String assistant content should be wrapped in list format."""
        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video_start": 0, "video_end": 2}],
                },
                {
                    "role": "assistant",
                    "content": "<think>t</think><action>silent</action>",
                },
            ],
        }
        result = self._build(item)
        asst = result["messages"][2]
        assert isinstance(asst["content"], list)
        assert asst["content"][0]["type"] == "text"

    def test_videos_field_fallback(self):
        """Should fall back to 'videos' list if 'video_path' is missing."""
        item = {
            "videos": ["/test/video.mp4"],
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video_start": 0, "video_end": 2}],
                },
                {"role": "assistant", "content": "<think>t</think><action>silent</action>"},
            ],
        }
        result = self._build(item)
        assert result["video_meta"]["abs_path"] == "/test/video.mp4"


# =====================================================================
# 8. find_assistant_spans tests
# =====================================================================


@_requires_sft_processor
class TestFindAssistantSpans:
    """Test label masking logic (without real tokenizer)."""

    def test_spans_found_for_mock_ids(self):
        """Verify span detection with mock token IDs."""

        # Create a mock tokenizer
        mock_tok = MagicMock()
        # assistant=100, <|im_end|>=101
        mock_tok.convert_tokens_to_ids.return_value = [100, 101]

        # Simulate: ... | assistant | \n | content | <|im_end|> | \n | ...
        ids = [0, 0, 100, 99, 50, 51, 52, 101, 99, 0]
        #                      ^start=4          ^end=9 (101+1=8, +1=9)
        spans = find_assistant_spans(ids, mock_tok)
        assert len(spans) == 1
        start, end = spans[0]
        assert start == 4  # pos+2
        assert end == 9  # im_end+2

    def test_multiple_spans(self):
        mock_tok = MagicMock()
        mock_tok.convert_tokens_to_ids.return_value = [100, 101]

        # Two assistant turns
        ids = [100, 99, 50, 51, 101, 99, 100, 99, 60, 61, 101, 99]
        spans = find_assistant_spans(ids, mock_tok)
        assert len(spans) == 2

    def test_no_spans_if_no_assistant(self):
        mock_tok = MagicMock()
        mock_tok.convert_tokens_to_ids.return_value = [100, 101]

        ids = [0, 1, 2, 3, 4]
        spans = find_assistant_spans(ids, mock_tok)
        assert len(spans) == 0


# =====================================================================
# 9. Edge case tests
# =====================================================================


class TestEdgeCases:

    def test_short_video_duration(self):
        """Episode with duration shorter than recent window."""
        segments = make_segments(n=5, seg_sec=4)  # 20s video
        task = make_r1_task()
        task["ask_time_sec"] = 10.0
        task["answer_time_sec"] = 10.0
        ep = assemble_episode(task, segments, 20.0, "/v.mp4")
        ok, reason = filter_episode(ep)
        assert ok, f"Short video should still work: {reason}"

    def test_ask_at_video_start(self):
        """Question at t=0 should work."""
        segments = make_segments()
        task = make_r1_task()
        task["ask_time_sec"] = 0.0
        task["answer_time_sec"] = 0.0
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        actions = _get_assistant_actions(ep)
        assert "response" in actions

    def test_answer_near_video_end(self):
        """Answer near the end of video."""
        segments = make_segments()
        task = make_r1_task()
        task["ask_time_sec"] = 78.0
        task["answer_time_sec"] = 78.0
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        times = _get_user_video_times(ep)
        assert all(ve <= 80.0 for _, ve in times)

    def test_multiple_videos_same_format(self):
        """Multiple episodes should produce consistent format."""
        segments = make_segments()
        tasks = [make_r1_task(), make_rc1_task(), make_r3_task()]
        for task in tasks:
            ep = assemble_episode(task, segments, 80.0, "/v.mp4")
            for msg in ep["messages"]:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part["type"] == "video":
                            assert isinstance(part["video_start"], (int, float))
                            assert isinstance(part["video_end"], (int, float))


# =====================================================================
# 10. Regression tests
# =====================================================================


class TestRegressions:

    def test_rc7_recall_after_gap(self):
        """RC7 recall action must appear in the follow-up window, not base."""
        segments = make_segments()
        task = make_rc7_task()
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")

        # Find the recall action and its preceding user message time
        msgs = ep["messages"]
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant" and "<action>recall</action>" in msg["content"]:
                # The user message before this should be in the follow-up window
                user_before = msgs[i - 1]
                assert user_before["role"] == "user"
                if isinstance(user_before["content"], list):
                    for part in user_before["content"]:
                        if part["type"] == "video":
                            # Should be near ask_time (60s), not base_ask_time (12s)
                            assert part["video_start"] >= 40.0, (
                                f"Recall should be in follow-up window, "
                                f"got video_start={part['video_start']}"
                            )

    def test_format_regex_matches_all_actions(self):
        """The format regex should match all three action types."""
        cases = [
            "<think>obs</think><action>silent</action>",
            "<think>ans</think><action>response</action><response>yes</response>",
            '<think>recall</think><action>recall</action><query>{"query":"x"}</query>',
        ]
        for c in cases:
            assert _FORMAT_RE.search(c), f"Regex should match: {c}"

    def test_format_regex_rejects_bad(self):
        bad = [
            "just text",
            "<action>silent</action>",  # missing think
            "<think>t</think><action>unknown</action>",  # bad action
        ]
        for c in bad:
            assert not _FORMAT_RE.search(c), f"Regex should reject: {c}"

    def test_rc7_video_end_clamped_to_duration(self):
        """RC7 video_end must not exceed video duration (bug fix regression)."""
        segments = make_segments(n=20, seg_sec=4)
        task = make_rc7_task()
        # Use a short duration so the follow-up window hits the end
        duration = 62.0
        ep = assemble_episode(task, segments, duration, "/v.mp4")
        times = _get_user_video_times(ep)
        for vs, ve in times:
            assert ve <= duration, (
                f"video_end {ve} > duration {duration}"
            )

    def test_s3r2_question_not_repeated_in_answer_chunk(self):
        """S3_R2: question should only appear in the ask chunk, not answer chunk."""
        segments = make_segments()
        task = make_s3r2_task()
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        question_count = 0
        for msg in ep["messages"]:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "text" and task["question"] in part["text"]:
                        question_count += 1
        assert question_count == 1, (
            f"Question should appear exactly once, got {question_count}"
        )

    def test_ce_weight_agent_format(self):
        """CE weight: <silent> token absent in agent data → uniform weights."""
        # In agent 3-action format, <action>silent</action> does NOT use <silent> token.
        # Verify that the assistant messages never contain <silent> as a standalone token.
        segments = make_segments()
        for task_fn in [make_r1_task, make_rc1_task, make_r3_task]:
            ep = assemble_episode(task_fn(), segments, 80.0, "/v.mp4")
            for msg in ep["messages"]:
                if msg["role"] == "assistant":
                    c = msg["content"]
                    # <silent> should NOT appear as a standalone token
                    # Only <action>silent</action> should appear
                    silent_positions = [
                        i for i in range(len(c))
                        if c[i:].startswith("<silent>")
                    ]
                    for pos in silent_positions:
                        # <silent> only valid if preceded by something other than <action>
                        # In agent format it should NEVER appear
                        assert False, (
                            f"<silent> token found in agent format: {c[:80]}"
                        )

    def test_response_token_appears_in_agent_format(self):
        """CE weight: <response> token DOES appear in agent data."""
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")
        found_response_tag = False
        for msg in ep["messages"]:
            if msg["role"] == "assistant" and "<response>" in msg["content"]:
                found_response_tag = True
        assert found_response_tag, "Agent data should have <response> tag"

    def test_r3_partial_answers_padding(self):
        """R3: if fewer partial_answers than response_times, should still work."""
        segments = make_segments()
        task = make_r3_task()
        # Remove one partial answer to test padding
        task["partial_answers"] = ["Adding oil", "Now adding garlic"]
        # 3 response times but only 2 partial answers
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        # Should still produce valid episode (empty 3rd response)
        ok, reason = filter_episode(ep)
        assert ok, f"R3 with missing partial answer should pass: {reason}"


# =====================================================================
# 11. P0 Bug-fix regression tests
# =====================================================================


class TestP0Fixes:
    """Regression tests for P0 bug fixes."""

    # ── P0-1: Historical think stripping ──

    def test_historical_think_stripped(self):
        """Historical assistant <think> should be stripped in SFT input."""
        if not _HAS_SFT_PROCESSOR:
            pytest.skip("ML deps not available")

        item = {
            "video_path": "/test/video.mp4",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [
                    {"type": "video", "video_start": 0, "video_end": 2},
                ]},
                {"role": "assistant",
                 "content": "<think>The person wears a RED apron.</think><action>silent</action>"},
                {"role": "user", "content": [
                    {"type": "video", "video_start": 2, "video_end": 4},
                ]},
                {"role": "assistant",
                 "content": "<think>Price tag shows 29.99.</think><action>silent</action>"},
                {"role": "user", "content": [
                    {"type": "video", "video_start": 4, "video_end": 6},
                ]},
                {"role": "assistant",
                 "content": "<think>Need to recall the apron color.</think><action>recall</action>"
                            '<query>{"query":"apron color"}</query>'},
            ],
        }
        with patch(
            "thinkstream.data.stream_data_processor._get_duration",
            return_value=80.0,
        ), patch(
            "thinkstream.data.stream_data_processor._make_abs_paths",
            side_effect=lambda base, f: f,
        ):
            result = _build_agent_messages(item, Path("."))

        msgs = result["messages"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(asst_msgs) == 3

        # First two (historical) should have empty thinks
        for m in asst_msgs[:2]:
            text = m["content"][0]["text"]
            assert "<think></think>" in text, f"Historical think not stripped: {text}"
            assert "RED apron" not in text
            assert "29.99" not in text

        # Last one (current target) keeps full think
        last_text = asst_msgs[2]["content"][0]["text"]
        assert "Need to recall" in last_text

    # ── P0-2: Query answer leakage ──

    def test_no_answer_in_query_fallback(self):
        """Fallback query must NOT contain the expected answer."""
        segments = make_segments()
        task = make_rc1_task()
        task["query_candidates"] = []  # Force fallback
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")

        for msg in ep["messages"]:
            if msg["role"] == "assistant" and "<query>" in msg.get("content", ""):
                content = msg["content"]
                # "red" is the expected answer — should NOT be in query
                query_match = re.search(r"<query>(.*?)</query>", content, re.DOTALL)
                if query_match:
                    assert task["expected_answer"].lower() not in query_match.group(1).lower(), (
                        f"Answer leaked into query: {query_match.group(1)}"
                    )

    def test_extract_query_keywords(self):
        assert "color" in _extract_query_keywords("What color was the apron?")
        assert "apron" in _extract_query_keywords("What color was the apron?")
        result = _extract_query_keywords("What is it?")
        assert len(result) > 0  # Should produce something even for short questions

    # ── P0-3: Causal violation ──

    def test_silent_think_is_generic(self):
        """Silent chunks must NOT use segment action text (causal violation)."""
        segments = make_segments()
        ep = assemble_episode(make_r1_task(), segments, 80.0, "/v.mp4")

        for msg in ep["messages"]:
            if msg["role"] == "assistant" and "<action>silent</action>" in msg["content"]:
                c = msg["content"]
                # Must be generic placeholder, not segment-specific
                assert "Observing" in c or "Waiting" in c, (
                    f"Silent think should be generic, got: {c[:80]}"
                )
                # Should NOT contain segment action descriptions
                for seg in segments:
                    assert seg["action"] not in c, (
                        f"Segment action leaked into silent think: {c[:80]}"
                    )

    # ── P0-4: Format regex strictness ──

    def test_format_regex_rejects_silent_with_response(self):
        bad = "<think>t</think><action>silent</action><response>x</response>"
        assert not _FORMAT_RE.search(bad), "silent+response should be rejected"

    def test_format_regex_rejects_response_without_content(self):
        bad = "<think>t</think><action>response</action>"
        assert not _FORMAT_RE.search(bad), "response without content should be rejected"

    def test_format_regex_rejects_recall_without_query(self):
        bad = "<think>t</think><action>recall</action>"
        assert not _FORMAT_RE.search(bad), "recall without query should be rejected"

    # ── P1-1: Recall result noise ──

    def test_recall_result_noise_distribution(self):
        """_build_recall_result should produce 70/20/5/5 distribution."""
        random.seed(42)
        segments = make_segments()
        support = segments[2]
        task = make_rc1_task()

        oracle = 0
        top3 = 0
        distractor = 0
        failure = 0
        n = 2000
        for _ in range(n):
            text, found = _build_recall_result(support, segments, task)
            if found and text.count("<item") == 1:
                oracle += 1
            elif found and text.count("<item") > 1:
                top3 += 1
            elif not found and "Not found" in text:
                failure += 1
            else:
                distractor += 1

        # Allow ±5% tolerance
        assert 0.60 < oracle / n < 0.80, f"oracle={oracle/n:.2f}"
        assert 0.12 < top3 / n < 0.28, f"top3={top3/n:.2f}"
        assert distractor / n < 0.12, f"distractor={distractor/n:.2f}"
        assert failure / n < 0.12, f"failure={failure/n:.2f}"

    def test_recall_result_failure_case(self):
        """When support_seg is None, should always return failure."""
        segments = make_segments()
        task = make_rc1_task()
        text, found = _build_recall_result(None, segments, task)
        assert not found
        assert "Not found" in text

    # ── P1-2: Filter query leakage check ──

    def test_query_leakage_detected(self):
        """Filter should reject episodes where query contains the answer."""
        ep = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "video", "video_start": 0, "video_end": 2}]},
                {"role": "assistant",
                 "content": '<think>t</think><action>recall</action><query>{"query":"red apron"}</query>'},
                {"role": "user", "content": "<recall_result>...</recall_result>\nContinue."},
                {"role": "assistant",
                 "content": "<think>t</think><action>response</action><response>Red.</response>"},
            ],
            "canonical_answer": {"answer_type": "slot", "value": {"answer": "red"}},
        }
        assert _check_query_leakage(ep), "Should detect answer 'red' in query"

    def test_query_leakage_filter_rejects(self):
        """filter_episode should reject episodes with query leakage."""
        ep = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "video", "video_start": 0, "video_end": 2}]},
                {"role": "assistant",
                 "content": '<think>t</think><action>recall</action><query>{"query":"red apron"}</query>'},
                {"role": "user", "content": "<recall_result>...</recall_result>\nContinue."},
                {"role": "assistant",
                 "content": "<think>t</think><action>response</action><response>Red.</response>"},
            ],
            "canonical_answer": {"answer_type": "slot", "value": {"answer": "red"}},
        }
        ok, reason = filter_episode(ep)
        assert not ok
        assert reason == "query_contains_answer"

    def test_clean_query_passes_filter(self):
        """Episodes with clean queries should pass filter."""
        segments = make_segments()
        task = make_rc1_task()
        # Ensure task has proper query_candidates (not fallback)
        ep = assemble_episode(task, segments, 80.0, "/v.mp4")
        ok, reason = filter_episode(ep)
        assert ok, f"Clean query should pass: {reason}"


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
