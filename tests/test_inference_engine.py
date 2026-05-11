"""Unit tests for the streaming inference engine extensions.

Covers Wave 3.1 (log_probs return, tool_call_stop_sample) and Wave 3.2
(reset_to_prefix partial-reset API). Does NOT exercise CUDA Graph / flash-attn
paths — those need a real GPU model. Tests target the pure-Python wrappers
and per-step bookkeeping.

Usage::

    python tests/test_inference_engine.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ToolCallStopSampleTests(unittest.TestCase):
    """tool_call_stop_sample forces EOS the step after </tool_call>."""

    def setUp(self):
        from thinkstream.models.inference import tool_call_stop_sample

        self.fn = tool_call_stop_sample
        self.tool_close = 1001
        self.eos = 1002
        self.pad = 0

    def _make_state(self, batch_size, max_steps, prev_tokens_per_row):
        # generated_tokens[b, t] = the token produced at step t for row b
        gen = torch.full((batch_size, max_steps), self.pad, dtype=torch.long)
        for b, prev in enumerate(prev_tokens_per_row):
            for t, tok in enumerate(prev):
                gen[b, t] = tok
        gen_len = torch.tensor([len(p) for p in prev_tokens_per_row], dtype=torch.long)
        return gen, gen_len

    def test_step_0_passes_through(self):
        # No history at step 0 → never force EOS
        gen, gen_len = self._make_state(2, 4, [[], []])
        sampled = torch.tensor([[42], [43]], dtype=torch.long)
        out = self.fn(
            next_token=sampled,
            logits=torch.zeros(2, 5),
            step=0,
            generated_tokens=gen,
            generated_length=gen_len,
            tool_call_close_token_id=self.tool_close,
            eos_token_id=self.eos,
        )
        self.assertTrue(torch.equal(out, sampled))

    def test_forces_eos_after_tool_close(self):
        # Row 0: prev step emitted </tool_call> → next must be EOS
        # Row 1: prev step emitted normal token → next unchanged
        gen, gen_len = self._make_state(
            2, 4, [[100, self.tool_close], [100, 200]]
        )
        sampled = torch.tensor([[42], [43]], dtype=torch.long)
        out = self.fn(
            next_token=sampled,
            logits=torch.zeros(2, 5),
            step=2,
            generated_tokens=gen,
            generated_length=gen_len,
            tool_call_close_token_id=self.tool_close,
            eos_token_id=self.eos,
        )
        self.assertEqual(out[0, 0].item(), self.eos)
        self.assertEqual(out[1, 0].item(), 43)

    def test_inner_callback_runs_first(self):
        # Inner callback rewrites a token; outer wrapper then applies its rule
        gen, gen_len = self._make_state(
            2, 4, [[100, self.tool_close], [100, 200]]
        )
        sampled = torch.tensor([[42], [43]], dtype=torch.long)

        def inner(next_token, **_):
            # Force everything to 999 — outer should override row 0 to EOS but
            # leave row 1 at 999.
            return torch.full_like(next_token, 999)

        out = self.fn(
            next_token=sampled,
            logits=torch.zeros(2, 5),
            step=2,
            generated_tokens=gen,
            generated_length=gen_len,
            tool_call_close_token_id=self.tool_close,
            eos_token_id=self.eos,
            inner=inner,
        )
        self.assertEqual(out[0, 0].item(), self.eos)
        self.assertEqual(out[1, 0].item(), 999)


class ResetToPrefixSemanticsTests(unittest.TestCase):
    """reset_to_prefix shrinks cache_seqlens to the kept prefix length."""

    def _make_stub_engine(self, batch_size, num_layers, initial_lengths):
        """Tiny stub that exposes the same cache + next_start_pos surface that
        reset_to_prefix relies on, without bringing up the full CUDA engine.
        """
        from thinkstream.models.inference import StreamingInferenceEngine

        class _StubCache:
            def __init__(self, num_layers, batch_size, initial):
                self.cache_seqlens = torch.zeros(
                    (num_layers, batch_size), dtype=torch.int32
                )
                for layer in range(num_layers):
                    self.cache_seqlens[layer] = torch.tensor(
                        initial, dtype=torch.int32
                    )
                self.batch_size = batch_size

            def adjust_seqlens(self, delta, layer_idx=None):
                # Mirror StreamingCache.adjust_seqlens contract.
                assert delta.shape[0] == self.batch_size
                if layer_idx is None:
                    self.cache_seqlens += delta.unsqueeze(0)
                else:
                    self.cache_seqlens[layer_idx] += delta

        class _StubDecoder:
            def __init__(self, num_layers, batch_size, initial):
                self.cache = _StubCache(num_layers, batch_size, initial)

        stub = StreamingInferenceEngine.__new__(StreamingInferenceEngine)
        stub.batch_size = batch_size
        stub.decoder = _StubDecoder(num_layers, batch_size, initial_lengths)
        stub.next_start_pos = torch.tensor([[999], [999]], dtype=torch.long)
        return stub

    def test_truncates_to_keep_length(self):
        stub = self._make_stub_engine(
            batch_size=2, num_layers=3, initial_lengths=[100, 200]
        )
        keep = torch.tensor([10, 30], dtype=torch.long)
        new_pos = torch.tensor([[10], [30]], dtype=torch.long)
        from thinkstream.models.inference import StreamingInferenceEngine

        StreamingInferenceEngine.reset_to_prefix(
            stub, keep_lengths=keep, next_start_pos=new_pos
        )
        for layer in range(3):
            self.assertEqual(stub.decoder.cache.cache_seqlens[layer, 0].item(), 10)
            self.assertEqual(stub.decoder.cache.cache_seqlens[layer, 1].item(), 30)
        self.assertTrue(torch.equal(stub.next_start_pos, new_pos))

    def test_grow_back_is_a_noop_check(self):
        # Calling reset_to_prefix with keep_lengths > current is permitted by
        # adjust_seqlens semantics; the engine treats it as the target. We
        # do not zero-out KV here, so this is "use at your own risk". Test
        # that it at least mechanically sets the seqlens.
        stub = self._make_stub_engine(
            batch_size=2, num_layers=2, initial_lengths=[50, 50]
        )
        keep = torch.tensor([80, 50], dtype=torch.long)
        new_pos = torch.tensor([[80], [50]], dtype=torch.long)
        from thinkstream.models.inference import StreamingInferenceEngine

        StreamingInferenceEngine.reset_to_prefix(
            stub, keep_lengths=keep, next_start_pos=new_pos
        )
        self.assertEqual(stub.decoder.cache.cache_seqlens[0, 0].item(), 80)
        self.assertEqual(stub.decoder.cache.cache_seqlens[0, 1].item(), 50)


class WindowedResetTests(unittest.TestCase):
    """StreamingWindowInferenceEngine.reset_to_prefix also rewinds the video
    sliding-window bookkeeping tensors.
    """

    def _make_windowed_stub(self, batch_size, window_size, init_count):
        from thinkstream.models.inference import StreamingWindowInferenceEngine

        class _StubCache:
            def __init__(self, num_layers, batch_size, initial):
                self.cache_seqlens = torch.full(
                    (num_layers, batch_size), initial, dtype=torch.int32
                )
                self.batch_size = batch_size

            def adjust_seqlens(self, delta, layer_idx=None):
                if layer_idx is None:
                    self.cache_seqlens += delta.unsqueeze(0)
                else:
                    self.cache_seqlens[layer_idx] += delta

        class _StubDecoder:
            def __init__(self):
                self.cache = _StubCache(num_layers=2, batch_size=batch_size, initial=200)

        stub = StreamingWindowInferenceEngine.__new__(StreamingWindowInferenceEngine)
        stub.batch_size = batch_size
        stub.video_flex_window_size = window_size
        stub.decoder = _StubDecoder()
        stub.device = torch.device("cpu")
        stub.next_start_pos = torch.tensor([[999]] * batch_size, dtype=torch.long)
        stub._window_starts = torch.tensor(
            [[10, 20, 30, 40], [11, 21, 31, 41]], dtype=torch.long
        )
        stub._window_ends = torch.tensor(
            [[15, 25, 35, 45], [16, 26, 36, 46]], dtype=torch.long
        )
        stub._window_count = torch.tensor(init_count, dtype=torch.long)
        return stub

    def test_default_clears_window(self):
        stub = self._make_windowed_stub(batch_size=2, window_size=4, init_count=[4, 4])
        keep = torch.tensor([5, 5], dtype=torch.long)
        new_pos = torch.tensor([[5], [5]], dtype=torch.long)
        from thinkstream.models.inference import StreamingWindowInferenceEngine

        StreamingWindowInferenceEngine.reset_to_prefix(
            stub, keep_lengths=keep, next_start_pos=new_pos
        )
        # All window state zeroed since keep_window_count omitted defaults to 0.
        self.assertTrue(torch.equal(stub._window_count, torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.all(stub._window_starts == 0))
        self.assertTrue(torch.all(stub._window_ends == 0))

    def test_keep_partial_window(self):
        stub = self._make_windowed_stub(batch_size=2, window_size=4, init_count=[4, 3])
        keep_len = torch.tensor([50, 30], dtype=torch.long)
        keep_w = torch.tensor([2, 1], dtype=torch.long)
        new_pos = torch.tensor([[50], [30]], dtype=torch.long)
        from thinkstream.models.inference import StreamingWindowInferenceEngine

        StreamingWindowInferenceEngine.reset_to_prefix(
            stub,
            keep_lengths=keep_len,
            next_start_pos=new_pos,
            keep_window_count=keep_w,
        )
        # Row 0 keeps indices [0, 1], zeros [2, 3].
        self.assertTrue(torch.equal(
            stub._window_starts[0], torch.tensor([10, 20, 0, 0], dtype=torch.long)
        ))
        self.assertTrue(torch.equal(
            stub._window_ends[0], torch.tensor([15, 25, 0, 0], dtype=torch.long)
        ))
        # Row 1 keeps index [0] only.
        self.assertTrue(torch.equal(
            stub._window_starts[1], torch.tensor([11, 0, 0, 0], dtype=torch.long)
        ))
        self.assertTrue(torch.equal(stub._window_count, keep_w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
