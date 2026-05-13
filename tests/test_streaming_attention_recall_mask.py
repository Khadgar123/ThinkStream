"""Recall-sidecar semantics for the SFT streaming attention mask."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkstream.models.streaming_attention import (  # noqa: E402
    generate_video_sliding_window_mask_mod,
)


def _allowed(mask_mod, q: int, k: int) -> bool:
    return bool(mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(q),
        torch.tensor(k),
    ).item())


def test_recall_video_does_not_advance_ordinary_window():
    # Ordinary video blocks: [1,2] and [7,8].
    # Recall sidecar block: [4,5], between ordinary block 1 and ordinary block 2.
    video = torch.tensor([[
        False, True, True, False, True, True, False, True, True, False,
    ]])
    recall = torch.tensor([[
        False, False, False, False, True, True, False, False, False, False,
    ]])
    attn = torch.ones_like(video, dtype=torch.bool)
    mask_mod = generate_video_sliding_window_mask_mod(
        video,
        attn,
        window_size_n=1,
        recall_video_mask=recall,
    )

    # Text immediately after recall can use the recalled visual evidence.
    assert _allowed(mask_mod, 6, 4)
    # After the next ordinary block arrives, recalled evidence is gone from KV.
    assert not _allowed(mask_mod, 9, 4)
    # The second ordinary block is still current and visible.
    assert _allowed(mask_mod, 9, 7)
    # The first ordinary block falls out with a one-block visual window.
    assert not _allowed(mask_mod, 9, 1)


if __name__ == "__main__":
    test_recall_video_does_not_advance_ordinary_window()
    print("PASS test_streaming_attention_recall_mask")
