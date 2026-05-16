"""Unit tests for StreamingRolloutEngine (verl adapter around the streaming engine).

Uses a stub engine that mimics StreamingWindowInferenceEngine's public surface
so tests run on CPU without flash_attn / CUDA Graph dependencies.

Usage::

    python tests/test_streaming_rollout.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkstream.models.streaming_rollout import (  # noqa: E402
    StreamingRolloutEngine,
    TurnResult,
    _infer_stop_reason,
)
from verl.workers.rollout.streaming_rollout import _inspect_streaming_delta_text  # noqa: E402


# ---------------------------------------------------------------------------
# Stub engine — mimics the slice of StreamingWindowInferenceEngine the adapter
# actually calls (generate, reset, reset_to_prefix + a couple of attributes).
# ---------------------------------------------------------------------------


class _StubEngine:
    """Stub that records calls and returns canned tokens + log_probs."""

    def __init__(self, batch_size=1, eos_id=1002, windowed=True):
        self.batch_size = batch_size
        self.primary_eos_token_id = eos_id
        self.calls = []
        self.reset_count = 0
        self.partial_reset_count = 0
        self._has_window = windowed
        # For begin_trajectory / reset_to_prefix dispatch.
        if windowed:
            self._window_count = torch.zeros(batch_size, dtype=torch.long)

    def generate(
        self,
        *,
        input_ids,
        position_ids,
        attention_mask=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        num_generations=1,
        max_new_tokens=128,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        repetition_penalty=1.0,
        sample=None,
        sample_kwargs=None,
        return_log_probs=False,
        turn_kind=None,
        recall_kv_policy=None,
        delete_previous_assistant_kv=False,
    ):
        # Record call for assertions
        self.calls.append({
            "input_ids_shape": tuple(input_ids.shape),
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "num_generations": num_generations,
            "return_log_probs": return_log_probs,
            "has_sample_cb": sample is not None,
            "sample_kwargs": dict(sample_kwargs or {}),
            "turn_kind": turn_kind,
            "recall_kv_policy": recall_kv_policy,
            "delete_previous_assistant_kv": delete_previous_assistant_kv,
        })
        # Synth output: 3 tokens, last is EOS; log_probs are dummy floats.
        effective_bsz = input_ids.shape[0] * num_generations
        tokens = [
            torch.tensor([100, 200, self.primary_eos_token_id], dtype=torch.long)
            for _ in range(effective_bsz)
        ]
        if return_log_probs:
            log_probs = [
                torch.tensor([-0.1, -0.2, -0.05], dtype=torch.float32)
                for _ in range(effective_bsz)
            ]
            return tokens, log_probs
        return tokens

    def reset(self):
        self.reset_count += 1

    def reset_to_prefix(
        self, *, keep_lengths, next_start_pos, keep_window_count=None
    ):
        self.partial_reset_count += 1
        self.last_keep_lengths = keep_lengths
        self.last_next_start_pos = next_start_pos
        self.last_keep_window_count = keep_window_count


class GenerateTurnTests(unittest.TestCase):
    def test_returns_one_result_per_effective_batch_row(self):
        eng = _StubEngine(batch_size=2)
        adapter = StreamingRolloutEngine(eng)
        out = adapter.generate_turn(
            input_ids=torch.zeros((1, 5), dtype=torch.long),
            position_ids=torch.zeros((1, 5), dtype=torch.long),
            sampling_params={"max_new_tokens": 32, "top_k": 10, "top_p": 0.8},
            num_generations=2,
            return_log_probs=True,
        )
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertIsInstance(r, TurnResult)
            self.assertEqual(len(r.token_ids), 3)
            self.assertEqual(len(r.log_probs), 3)
            self.assertEqual(r.stop_reason, "eos")
            self.assertEqual(r.diagnostics["max_new_tokens"], 32)

    def test_sampling_params_forwarded(self):
        eng = _StubEngine()
        adapter = StreamingRolloutEngine(eng)
        adapter.generate_turn(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            sampling_params={
                "max_tokens": 64,  # also accepted as alias
                "top_k": 5,
                "top_p": 0.7,
                "temperature": 0.6,
                "repetition_penalty": 1.1,
            },
            return_log_probs=False,
        )
        self.assertEqual(len(eng.calls), 1)
        call = eng.calls[0]
        self.assertEqual(call["max_new_tokens"], 64)
        self.assertEqual(call["top_k"], 5)
        self.assertEqual(call["top_p"], 0.7)
        self.assertEqual(call["temperature"], 0.6)
        self.assertEqual(call["return_log_probs"], False)

    def test_no_log_probs_when_disabled(self):
        eng = _StubEngine()
        adapter = StreamingRolloutEngine(eng)
        out = adapter.generate_turn(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            return_log_probs=False,
        )
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].log_probs)
        self.assertEqual(out[0].stop_reason, "eos")

    def test_sample_callback_passed_through(self):
        eng = _StubEngine()
        adapter = StreamingRolloutEngine(eng)

        def cb(**kw):
            return kw["next_token"]

        adapter.generate_turn(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            sample_callback=cb,
            sample_callback_kwargs={"foo": "bar"},
        )
        self.assertTrue(eng.calls[0]["has_sample_cb"])

    def test_turn_kind_and_recall_policy_forwarded(self):
        eng = _StubEngine()
        adapter = StreamingRolloutEngine(eng)
        adapter.generate_turn(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            sampling_params={
                "recall_kv_policy": "next_turn",
                "delete_previous_recall_toolcall_kv": True,
            },
            turn_kind="post_recall",
        )
        call = eng.calls[0]
        self.assertEqual(call["turn_kind"], "post_recall")
        self.assertEqual(call["recall_kv_policy"], "next_turn")
        self.assertEqual(call["sample_kwargs"]["turn_kind"], "post_recall")
        self.assertTrue(call["delete_previous_assistant_kv"])


class TrajectoryBoundaryTests(unittest.TestCase):
    def test_begin_trajectory_routes_to_windowed_engine(self):
        eng = _StubEngine(windowed=True)
        adapter = StreamingRolloutEngine(eng)
        keep_len = torch.tensor([10], dtype=torch.long)
        new_pos = torch.tensor([[10]], dtype=torch.long)
        keep_w = torch.tensor([2], dtype=torch.long)
        adapter.begin_trajectory(
            keep_lengths=keep_len,
            next_start_pos=new_pos,
            keep_window_count=keep_w,
        )
        self.assertEqual(eng.partial_reset_count, 1)
        self.assertTrue(torch.equal(eng.last_keep_lengths, keep_len))
        self.assertTrue(torch.equal(eng.last_next_start_pos, new_pos))
        self.assertTrue(torch.equal(eng.last_keep_window_count, keep_w))

    def test_begin_trajectory_drops_window_kwarg_for_base_engine(self):
        eng = _StubEngine(windowed=False)
        adapter = StreamingRolloutEngine(eng)
        keep_len = torch.tensor([10], dtype=torch.long)
        new_pos = torch.tensor([[10]], dtype=torch.long)
        adapter.begin_trajectory(
            keep_lengths=keep_len,
            next_start_pos=new_pos,
            keep_window_count=torch.tensor([2], dtype=torch.long),  # ignored
        )
        self.assertEqual(eng.partial_reset_count, 1)
        # Last_keep_window_count attribute set by stub, but base engine
        # shouldn't receive it.
        self.assertIsNone(eng.last_keep_window_count)

    def test_reset_full(self):
        eng = _StubEngine()
        adapter = StreamingRolloutEngine(eng)
        adapter.reset()
        self.assertEqual(eng.reset_count, 1)


class StopReasonInferenceTests(unittest.TestCase):
    def test_eos(self):
        self.assertEqual(
            _infer_stop_reason([100, 200, 1002], primary_eos_token_id=1002, max_new_tokens=10),
            "eos",
        )

    def test_length(self):
        self.assertEqual(
            _infer_stop_reason([1] * 10, primary_eos_token_id=1002, max_new_tokens=10),
            "length",
        )

    def test_other(self):
        self.assertEqual(
            _infer_stop_reason([1, 2, 3], primary_eos_token_id=1002, max_new_tokens=10),
            "other",
        )

    def test_empty(self):
        self.assertEqual(
            _infer_stop_reason([], primary_eos_token_id=1002, max_new_tokens=10),
            "empty",
        )


class TrueKVDeltaInspectionTests(unittest.TestCase):
    def test_post_recall_delta_allows_recall_tool_response(self):
        text = (
            "<|im_start|>tool\n"
            "<tool_response>\n"
            "The recall tool returned historical video frames for t=1-3."
            "<|vision_start|><|video_pad|><|vision_end|>"
            "\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(
            _inspect_streaming_delta_text(text, turn_kind="post_recall"),
            [],
        )

    def test_next_streaming_delta_rejects_recall_evidence_and_tool_call(self):
        text = (
            "<|im_start|>user\n"
            "<t=4>"
            "<tool_response>The recall tool returned historical video frames for t=1-3.</tool_response>"
            "<tool_call>{}</tool_call>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        flags = _inspect_streaming_delta_text(
            text,
            turn_kind="streaming",
            after_post_recall=True,
        )
        self.assertIn("tool_response_in_non_post_recall_delta", flags)
        self.assertIn("tool_call_in_non_post_recall_delta", flags)
        self.assertIn("post_recall_evidence_leaked_to_next_streaming_delta", flags)
        self.assertIn("post_recall_tool_call_leaked_to_next_streaming_delta", flags)

    def test_delta_rejects_repeated_system_and_tool_schema(self):
        text = (
            "<|im_start|>system\n# Tools\n<tools>{}</tools><|im_end|>\n"
            "<|im_start|>user\n<t=1><t=2><|im_end|>"
        )
        flags = _inspect_streaming_delta_text(text, turn_kind="streaming")
        self.assertIn("repeated_system_prompt", flags)
        self.assertIn("repeated_tool_schema", flags)
        self.assertIn("multiple_current_timestamps", flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
