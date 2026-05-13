"""Targeted tests for PerTimestepDataCollator's video_mask emission.

The streaming_attention path needs a per-token ``video_mask`` so the
patched lce_forward can build the FlexAttention sliding-window block
mask. video_mask is keyed off the ``<|video_pad|>`` token id. We don't
load a real Qwen tokenizer here — a tiny stub keeps the test fast and
avoids the network dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
try:
    import pytest
except ImportError:  # pragma: no cover - local minimal env fallback
    class _PytestFallback:
        @staticmethod
        def skip(message, allow_module_level=False):
            raise SystemExit(message)

    pytest = _PytestFallback()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from thinkstream.sft.data_processor import PerTimestepDataCollator
except ImportError as e:
    # Some local envs have a transformers <-> tokenizers version skew that
    # blocks `import transformers` entirely. Skip cleanly so the rest of
    # the suite still runs.
    pytest.skip(
        f"thinkstream.sft.data_processor import failed ({e}); collator "
        f"tests need a working transformers install.",
        allow_module_level=True,
    )


class _StubTokenizer:
    """Minimal tokenizer surface for the collator: pad token + token-id lookup."""

    pad_token_id = 0
    model_max_length = 32

    def __init__(self, video_token_id: int = 7):
        self._video_token_id = video_token_id

    def convert_tokens_to_ids(self, tok):
        if tok == "<|video_pad|>":
            return self._video_token_id
        return -1


def _make_instance(input_ids):
    ids = torch.tensor(input_ids, dtype=torch.long)
    n = ids.numel()
    labels = ids.clone()
    # position_ids must be [1, 3, L] for Qwen3-VL MROPE; we use a degenerate
    # but shape-correct placeholder. The collator only concats along axis 1.
    position_ids = torch.zeros(1, 3, n, dtype=torch.long)
    return {
        "input_ids": ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "position_ids": position_ids,
    }


def _make_instance_with_recall(input_ids, recall_positions):
    inst = _make_instance(input_ids)
    recall = torch.zeros_like(inst["input_ids"], dtype=torch.bool)
    for pos in recall_positions:
        recall[0, pos] = True
    inst["recall_video_mask"] = recall
    inst["recall_kv_mask"] = recall.clone()
    return inst


def test_video_mask_not_emitted_by_default():
    """Default behaviour preserves the flash_attention_2 contract: no video_mask."""
    tok = _StubTokenizer()
    collator = PerTimestepDataCollator(tok)  # emit_video_mask defaults to False
    batch = collator([
        _make_instance([1, 2, 7, 7, 3]),
        _make_instance([4, 7, 7, 7, 5]),
    ])
    assert "video_mask" not in batch
    print("[OK] video_mask_not_emitted_by_default")


def test_video_mask_emitted_when_requested():
    tok = _StubTokenizer(video_token_id=7)
    collator = PerTimestepDataCollator(tok, emit_video_mask=True)
    batch = collator([
        _make_instance([1, 2, 7, 7, 3]),
        _make_instance([4, 7, 7, 7, 5]),
    ])
    assert "video_mask" in batch
    vm = batch["video_mask"]
    assert vm.dtype == torch.bool
    assert vm.shape == batch["input_ids"].shape
    # Sample 0: positions 2, 3 are video pads; rest are text.
    assert vm[0].tolist() == [False, False, True, True, False]
    # Sample 1: positions 1, 2, 3 are video pads.
    assert vm[1].tolist() == [False, True, True, True, False]
    print("[OK] video_mask_emitted_when_requested")


def test_recall_video_mask_emitted_when_requested():
    tok = _StubTokenizer(video_token_id=7)
    collator = PerTimestepDataCollator(tok, emit_video_mask=True)
    batch = collator([
        _make_instance_with_recall([1, 7, 7, 2, 7, 3], [4]),
        _make_instance([4, 7, 7, 5]),
    ])
    assert "recall_video_mask" in batch
    assert "recall_kv_mask" in batch
    assert batch["recall_video_mask"].tolist() == [
        [False, False, False, False, True, False],
        [False, False, False, False, False, False],
    ]
    assert torch.equal(batch["recall_kv_mask"], batch["recall_video_mask"])
    print("[OK] recall_video_mask_emitted_when_requested")


def test_video_mask_handles_padding_uniformly():
    """Padded positions get input_id=pad_token_id (0) which != video_token_id,
    so they correctly stay False in video_mask."""
    tok = _StubTokenizer(video_token_id=7)
    collator = PerTimestepDataCollator(tok, emit_video_mask=True)
    batch = collator([
        _make_instance([1, 7, 7]),          # length 3
        _make_instance([4, 7, 7, 7, 5]),    # length 5 — sets max
    ])
    vm = batch["video_mask"]
    assert vm.shape == (2, 5)
    # Sample 0 padded with 0s at positions 3, 4 → False.
    assert vm[0].tolist() == [False, True, True, False, False]
    assert vm[1].tolist() == [False, True, True, True, False]
    print("[OK] video_mask_handles_padding_uniformly")


def test_video_mask_token_id_unresolvable_warns_and_skips():
    """If the tokenizer can't resolve <|video_pad|>, emit_video_mask is a
    no-op (logged) rather than crashing — protects smoke tests on stub
    tokenizers."""
    class _NoVidTokenizer(_StubTokenizer):
        def convert_tokens_to_ids(self, tok):
            return -1  # not in vocab
    collator = PerTimestepDataCollator(_NoVidTokenizer(), emit_video_mask=True)
    batch = collator([_make_instance([1, 2, 3])])
    assert "video_mask" not in batch
    print("[OK] video_mask_token_id_unresolvable_warns_and_skips")


if __name__ == "__main__":
    test_video_mask_not_emitted_by_default()
    test_video_mask_emitted_when_requested()
    test_recall_video_mask_emitted_when_requested()
    test_video_mask_handles_padding_uniformly()
    test_video_mask_token_id_unresolvable_warns_and_skips()
    print("\nall PerTimestepDataCollator video_mask tests passed")
