"""Unit tests for action-class loss balancing.

Validates:
1. Inverse-frequency class weights (CASIA-style) sum-balance silent vs response.
2. Focal loss reduces to (weighted) CE when gamma=0 and auto_alpha=False.
3. Focal loss upweights low-confidence predictions relative to CE.
4. Both paths leave non-action tokens at weight=1.0.

Run:
  python tests/test_losses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkstream.sft.losses import (  # noqa: E402
    ACTION_TOKEN_NAMES,
    _find_subsequence_anchors,
    compute_inverse_frequency_weights,
    find_tool_name_anchors,
    focal_loss_per_token,
)


# Synthetic vocab:
#   0 = <pad>, 1 = <silent>, 2 = <response>, 3 = </response>, 4 = </think>,
#   5-9 = arbitrary content tokens.
VOCAB_SIZE = 10
SILENT_ID = 1
RESPONSE_ID = 2
RESPONSE_CLOSE_ID = 3
IGNORE = -100

ACTION_IDS = {
    "<silent>": SILENT_ID,
    "<response>": RESPONSE_ID,
}


def test_inverse_frequency_balances_silent_response():
    # 9 silent labels, 1 response label, rest ignored.
    labels = torch.full((1, 12), IGNORE, dtype=torch.long)
    labels[0, 0:9] = SILENT_ID
    labels[0, 9] = RESPONSE_ID
    # labels[0, 10:12] stay -100

    w = compute_inverse_frequency_weights(labels, ACTION_IDS, ignore_index=IGNORE)

    # Sanity: shape match, non-action position has weight 1.0
    assert w.shape == labels.shape
    assert torch.allclose(w[0, 10:12], torch.ones(2))

    silent_w = w[0, 0].item()
    response_w = w[0, 9].item()

    # Inverse frequency: 9 silent + 1 response, 2 classes
    # silent_w = 10 / (2 * 9 + eps)  ≈ 0.555
    # response_w = 10 / (2 * 1 + eps) ≈ 5.0
    assert 0.5 < silent_w < 0.65, f"silent_w expected ~0.555, got {silent_w}"
    assert 4.5 < response_w < 5.5, f"response_w expected ~5.0, got {response_w}"

    # Total contribution should be balanced: silent has 9 tokens, response 1.
    # Sum of weights per class should be roughly equal.
    silent_total = silent_w * 9
    response_total = response_w * 1
    assert abs(silent_total - response_total) / max(silent_total, response_total) < 0.1, (
        f"class totals not balanced: silent={silent_total}, response={response_total}"
    )

    print(f"[OK] inverse_freq: silent_w={silent_w:.3f}, response_w={response_w:.3f}")


def test_response_close_is_not_action_balanced():
    """Only the response-open token is an action decision.

    The closing tag is a formatting target and must stay ordinary CE weight,
    otherwise one response turn contributes two action anchors while one
    silent turn contributes one.
    """
    assert "</response>" not in ACTION_TOKEN_NAMES
    assert "<answer>" not in ACTION_TOKEN_NAMES
    assert "</answer>" not in ACTION_TOKEN_NAMES

    labels = torch.full((1, 12), IGNORE, dtype=torch.long)
    labels[0, 0:9] = SILENT_ID
    labels[0, 9] = RESPONSE_ID
    labels[0, 10] = RESPONSE_CLOSE_ID

    w = compute_inverse_frequency_weights(labels, ACTION_IDS, ignore_index=IGNORE)

    assert torch.isclose(w[0, 10], torch.tensor(1.0)), (
        f"</response> should remain ordinary CE weight, got {w[0, 10].item()}"
    )
    assert w[0, 9] > w[0, 0], "rare <response> start token should be upweighted"


def test_inverse_frequency_clamps_extreme():
    """1 silent vs 10000 response triggers the inverse-frequency ceiling.

    Inverse-frequency weight for the rare class is
    ``total / (num_classes * count + eps)`` ≈ ``10001 / (2 * 1) ≈ 5000``,
    which must be clamped to ``ceil_weight`` (default 20.0). The frequent
    class weight is ≈ 0.5 — well within range.
    """
    n_rare, n_common = 1, 10000
    seq_len = n_rare + n_common
    labels = torch.full((1, seq_len), IGNORE, dtype=torch.long)
    labels[0, 0] = SILENT_ID
    labels[0, 1:1 + n_common] = RESPONSE_ID

    w = compute_inverse_frequency_weights(
        labels, ACTION_IDS, ignore_index=IGNORE, ceil_weight=20.0,
    )
    silent_w = w[0, 0].item()
    response_w = w[0, 1].item()
    assert silent_w == 20.0, (
        f"ceiling 20.0 not applied to rare class; got silent_w={silent_w}"
    )
    assert response_w < 1.0, (
        f"frequent-class weight should be < 1.0; got response_w={response_w}"
    )
    print(f"[OK] clamp_extreme: silent_w={silent_w:.3f}, response_w={response_w:.4f}")


def test_focal_reduces_to_weighted_ce_when_gamma_zero():
    # With gamma=0 and auto_alpha=False, focal == standard CE per token.
    torch.manual_seed(0)
    logits = torch.randn(2, 6, VOCAB_SIZE, requires_grad=False)
    labels = torch.tensor([
        [5, 6, SILENT_ID, 7, IGNORE, IGNORE],
        [5, RESPONSE_ID, 7, 8, IGNORE, IGNORE],
    ])

    focal = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=0.0,
        auto_alpha=False,
        ignore_index=IGNORE,
    )

    # Compare to manual CE (also masking ignore positions to 0).
    flat_logits = logits.reshape(-1, VOCAB_SIZE)
    flat_labels = labels.reshape(-1)
    safe_labels = flat_labels.masked_fill(flat_labels == IGNORE, 0)
    ce = F.cross_entropy(flat_logits, safe_labels, reduction="none")
    ce = ce * (flat_labels != IGNORE).to(ce.dtype)

    diff = (focal - ce).abs().max().item()
    assert diff < 1e-5, f"focal != CE at gamma=0, max diff={diff}"
    print(f"[OK] focal_gamma0_equals_ce: max_diff={diff:.2e}")


def test_focal_emphasizes_low_confidence():
    """Two tokens with the same label but different prediction confidence.
    Focal should give the low-confidence one a higher loss relative to CE."""

    logits = torch.zeros(1, 2, VOCAB_SIZE)
    target = 5
    # Token 0: very confident (logit for target very high)
    logits[0, 0, target] = 10.0
    # Token 1: low confidence (logit for target average)
    logits[0, 1, target] = 0.5

    labels = torch.tensor([[target, target]])

    focal_g2 = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=2.0,
        auto_alpha=False,
        ignore_index=IGNORE,
    )
    focal_g0 = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=0.0,
        auto_alpha=False,
        ignore_index=IGNORE,
    )

    # At gamma=2 the high-confidence token's loss is suppressed much more.
    # Specifically: ratio focal_g2 / focal_g0 should be much smaller for the
    # confident token than for the unconfident token.
    ratio_confident = (focal_g2[0] / focal_g0[0].clamp_min(1e-12)).item()
    ratio_unconfident = (focal_g2[1] / focal_g0[1].clamp_min(1e-12)).item()
    assert ratio_unconfident > ratio_confident * 100, (
        "focal modulation did not differentiate confidence: "
        f"confident_ratio={ratio_confident:.3e}, unconfident_ratio={ratio_unconfident:.3e}"
    )
    print(
        f"[OK] focal_emphasizes_low_conf: confident_ratio={ratio_confident:.3e}, "
        f"unconfident_ratio={ratio_unconfident:.3e}"
    )


def test_focal_auto_alpha_balances_classes():
    """When auto_alpha=True, the rare action token should be upweighted
    relative to a CE-equivalent baseline."""
    logits = torch.randn(1, 12, VOCAB_SIZE)
    labels = torch.full((1, 12), IGNORE, dtype=torch.long)
    labels[0, 0:9] = SILENT_ID    # majority
    labels[0, 9] = RESPONSE_ID    # rare

    focal_with_alpha = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=0.0,                 # isolate alpha effect
        auto_alpha=True,
        ignore_index=IGNORE,
    )
    focal_no_alpha = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=0.0,
        auto_alpha=False,
        ignore_index=IGNORE,
    )

    # auto_alpha should upweight the rare response token (position 9) and
    # downweight the silent tokens (positions 0-8).
    silent_ratio = (focal_with_alpha[0] / focal_no_alpha[0].clamp_min(1e-12)).item()
    response_ratio = (focal_with_alpha[9] / focal_no_alpha[9].clamp_min(1e-12)).item()
    assert response_ratio > silent_ratio, (
        f"auto_alpha did not upweight rare class: "
        f"silent_ratio={silent_ratio:.3f}, response_ratio={response_ratio:.3f}"
    )
    print(
        f"[OK] focal_auto_alpha: silent_ratio={silent_ratio:.3f}, "
        f"response_ratio={response_ratio:.3f}"
    )


def test_find_subsequence_anchors_basic():
    seq = [5, 6, 9, 8, 9, 8, 7]
    needle = [9, 8]
    # Two matches: at index 2 and 4
    matches = _find_subsequence_anchors(seq, needle)
    assert matches == [2, 4], f"expected [2, 4], got {matches}"
    # With mask: disable position 4
    mask = [True] * len(seq)
    mask[4] = False
    matches = _find_subsequence_anchors(seq, needle, search_mask=mask)
    assert matches == [2], f"expected [2], got {matches}"
    print(f"[OK] find_subsequence_anchors")


def test_tool_name_anchors_integrated_with_inverse_freq():
    """Simulate a batch with one compress tool_call (2-token "compress" span)
    and one response (single <response> token). Verify both classes are
    detected and balanced."""

    # Synthetic vocab adds tool name pieces:
    #   1 = <silent>, 2 = <response>, 5-6 = "compress" BPE pieces, 7 = "recall"
    COMP_A, COMP_B = 5, 6
    RECALL = 7

    # Sample 0: assistant emits "compress" (2 tokens) once
    # Sample 1: assistant emits <response> twice
    labels = torch.full((2, 10), IGNORE, dtype=torch.long)
    labels[0, 2] = COMP_A
    labels[0, 3] = COMP_B
    labels[1, 4] = RESPONSE_ID
    labels[1, 7] = RESPONSE_ID

    tool_seqs = {"compress": [COMP_A, COMP_B], "recall": [RECALL]}

    anchors = find_tool_name_anchors(labels, tool_seqs, ignore_index=IGNORE)
    assert anchors["compress"] == [(0, 2)], f"compress anchors: {anchors['compress']}"
    assert anchors["recall"] == [], f"recall anchors should be empty: {anchors['recall']}"

    w = compute_inverse_frequency_weights(
        labels=labels,
        action_token_ids=ACTION_IDS,
        tool_name_sequences=tool_seqs,
        ignore_index=IGNORE,
    )

    # Compress count = 1, response count = 2.
    # Total = 3, num_classes = 2.
    # compress_w = 3 / (2 * 1 + eps) ≈ 1.5
    # response_w = 3 / (2 * 2 + eps) ≈ 0.75
    compress_anchor_w = w[0, 2].item()
    compress_body_w = w[0, 3].item()  # body (second token of "compress")
    response_w = w[1, 4].item()

    assert 1.3 < compress_anchor_w < 1.7, f"compress anchor w={compress_anchor_w}"
    # Body token (second piece) should NOT be re-weighted, stays 1.0
    assert abs(compress_body_w - 1.0) < 1e-5, f"compress body w={compress_body_w} (should be 1.0)"
    assert 0.6 < response_w < 0.85, f"response w={response_w}"

    print(
        f"[OK] tool_name + inverse_freq: compress_anchor={compress_anchor_w:.3f}, "
        f"compress_body={compress_body_w:.3f}, response={response_w:.3f}"
    )


def test_tool_name_anchors_focal_loss():
    """Focal loss with tool name detection: confirm auto_alpha upweights
    rare tool-name anchor."""
    COMP_A, COMP_B = 5, 6
    # 8 response tokens, 1 compress (anchor at position 0)
    labels = torch.full((1, 12), IGNORE, dtype=torch.long)
    labels[0, 0] = COMP_A
    labels[0, 1] = COMP_B
    labels[0, 2:10] = RESPONSE_ID

    # Random logits
    torch.manual_seed(42)
    logits = torch.randn(1, 12, VOCAB_SIZE)

    tool_seqs = {"compress": [COMP_A, COMP_B]}

    focal_with_tool = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        tool_name_sequences=tool_seqs,
        gamma=0.0,
        auto_alpha=True,
        ignore_index=IGNORE,
    )
    focal_no_tool = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        tool_name_sequences=None,
        gamma=0.0,
        auto_alpha=True,
        ignore_index=IGNORE,
    )

    # compress anchor (position 0) should be upweighted in the WITH-tool variant
    comp_with = focal_with_tool[0].item()
    comp_without = focal_no_tool[0].item()
    assert comp_with > comp_without * 1.5, (
        f"tool name detection did not upweight compress anchor: "
        f"with_tool={comp_with:.4f}, without_tool={comp_without:.4f}"
    )

    # Body (position 1) should be unaffected by tool name detection
    body_with = focal_with_tool[1].item()
    body_without = focal_no_tool[1].item()
    assert abs(body_with - body_without) < 1e-5, (
        f"body token weight changed unexpectedly: with={body_with}, without={body_without}"
    )

    print(
        f"[OK] tool_name + focal: compress_with={comp_with:.4f}, "
        f"compress_without={comp_without:.4f}"
    )


def test_focal_ignore_index_contributes_zero():
    logits = torch.randn(1, 4, VOCAB_SIZE)
    labels = torch.tensor([[5, IGNORE, IGNORE, 6]])

    flat = focal_loss_per_token(
        logits=logits,
        labels=labels,
        action_token_ids=ACTION_IDS,
        gamma=2.0,
        auto_alpha=True,
        ignore_index=IGNORE,
    )
    flat = flat.reshape(1, 4)
    assert flat[0, 1].item() == 0.0, f"ignore pos 1 should be 0, got {flat[0, 1]}"
    assert flat[0, 2].item() == 0.0, f"ignore pos 2 should be 0, got {flat[0, 2]}"
    print(f"[OK] focal_ignore_index_zero")


if __name__ == "__main__":
    test_inverse_frequency_balances_silent_response()
    test_inverse_frequency_clamps_extreme()
    test_focal_reduces_to_weighted_ce_when_gamma_zero()
    test_focal_emphasizes_low_confidence()
    test_focal_auto_alpha_balances_classes()
    test_find_subsequence_anchors_basic()
    test_tool_name_anchors_integrated_with_inverse_freq()
    test_tool_name_anchors_focal_loss()
    test_focal_ignore_index_contributes_zero()
    print("\n✅ all action_class_loss tests passed")
