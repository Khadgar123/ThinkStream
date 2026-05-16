"""Action-start loss balancing for streaming SFT.

The only class-discriminative text tokens we balance are action-start anchors:
``</Silence>`` and ``</Response>``. Response body tokens stay ordinary CE
targets so the model learns content without over-weighting answer text as the
action decision.

Two optional mechanisms are available via ``data_args.action_class_loss_mode``:

- ``"inverse_freq"`` (inverse-frequency-weighted): compute per-class token weight = total / (2 *
  per_class_count) in the data collator, multiply into ``token_loss_weight``.
  Loss path stays standard CE. Cheap, fully compatible with Liger fused CE.

- ``"focal"`` (focal-modulation): apply focal modulation (1 - p_target)^gamma *
  CE on assistant tokens, with optional auto-alpha class rebalancing.
  Replaces the trainer's CE path. Needs explicit logits → no Liger fused CE.

Both share the same set of action-start tokens registered via
``ACTION_TOKEN_NAMES``. Tokens outside this set get weight = 1.0 and are
unaffected. Tool names (``compress``/``recall``) are optional anchors handled
separately inside ``<tool_call>`` spans.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# Action-start tokens whose imbalance we explicitly correct. ``<think>`` /
# ``</think>`` appear on every assistant turn and are not class-discriminative.
# The response marker is single-token style (Streamo-like) and has no paired
# close tag, so there is only one response-class vote per response turn.
ACTION_TOKEN_NAMES: Tuple[str, ...] = (
    "</Silence>",
    "</Response>",
)

# Tool names are NOT single tokens under default Qwen BPE — they typically
# decode to multi-piece sequences. We detect them as token spans and treat
# the FIRST token of each span as the discriminative anchor (same role as
# </Silence>/</Response> action-start tokens above).
TOOL_NAME_NAMES: Tuple[str, ...] = (
    "compress",
    "recall",
)


def resolve_action_token_ids(
    tokenizer,
    extra_action_names: Sequence[str] = (),
) -> dict:
    """Resolve action-start token text → token id. Falls back gracefully when a
    token is multi-piece BPE'd (returns None for that key)."""
    out = {}
    names: List[str] = list(ACTION_TOKEN_NAMES) + list(extra_action_names)
    for name in names:
        ids = tokenizer(name, add_special_tokens=False).input_ids
        out[name] = ids[0] if len(ids) == 1 else None
    return out


def resolve_tool_name_token_sequences(
    tokenizer,
    tool_names: Sequence[str] = TOOL_NAME_NAMES,
) -> dict:
    """Tokenize each tool name to its BPE id sequence.

    Used for span detection inside ``<tool_call>{"name": "..."}</tool_call>``
    regions. The first token of each match becomes the class anchor.

    Returns: {tool_name: [ids...]}.

    In a Qwen-style tool_call JSON body the tool name appears as
    ``..."name": "compress"...``. The opening ``"`` is its own token, so
    ``compress`` tokenizes the same way as the bare word. We therefore
    tokenize the bare string directly. Empty results are mapped to ``[]``
    so the caller's "if not needle: continue" check skips them safely.
    """
    out = {}
    for name in tool_names:
        try:
            ids = tokenizer(name, add_special_tokens=False).input_ids
        except Exception:
            ids = []
        out[name] = list(ids or [])
    return out


def _find_subsequence_anchors(
    seq: List[int],
    needle: List[int],
    search_mask: Optional[List[bool]] = None,
) -> List[int]:
    """Return positions where ``seq`` matches ``needle`` (as a contiguous
    substring). Returned positions are the index of the FIRST needle token
    (the anchor). ``search_mask`` is an optional same-length boolean list;
    when given, the match's first token must be at a position where the
    mask is True (used to scope detection to label-valid regions).
    """
    n, m = len(seq), len(needle)
    if m == 0 or n < m:
        return []
    out = []
    # Naive scan; m is small (1-4 tokens), n bounded by sample len (a few K).
    for i in range(n - m + 1):
        if search_mask is not None and not search_mask[i]:
            continue
        ok = True
        for j in range(m):
            if seq[i + j] != needle[j]:
                ok = False
                break
        if ok:
            out.append(i)
    return out


def _compute_tool_call_span_mask(
    seq: List[int],
    base_mask: List[bool],
    open_ids: Optional[List[int]],
    close_ids: Optional[List[int]],
) -> List[bool]:
    """Refine ``base_mask`` to only positions inside ``<tool_call>...</tool_call>``
    blocks, so tool-name anchor matching can't be tricked by occurrences in
    ``<think>`` or response text.

    If either marker sequence is missing/empty, returns ``base_mask`` unchanged.
    The returned mask is the intersection of ``base_mask`` and the union of all
    ``[open_end, close_start)`` half-open ranges.
    """
    if not open_ids or not close_ids:
        return base_mask
    n = len(seq)
    refined = [False] * n
    open_positions = _find_subsequence_anchors(seq, open_ids, search_mask=base_mask)
    if not open_positions:
        return refined
    # Find every close marker — search across the full sequence (not just
    # base_mask) so a marker emitted at the very end (last token = im_end)
    # still terminates the span.
    close_positions = _find_subsequence_anchors(seq, close_ids, search_mask=None)
    close_iter = iter(close_positions)
    next_close = next(close_iter, None)
    for open_pos in open_positions:
        body_start = open_pos + len(open_ids)
        # Advance to first close marker AFTER this body_start.
        while next_close is not None and next_close < body_start:
            next_close = next(close_iter, None)
        if next_close is None:
            # No matching close; treat tail as the body.
            body_end = n
        else:
            body_end = next_close
        for j in range(body_start, body_end):
            if 0 <= j < n and base_mask[j]:
                refined[j] = True
    return refined


def find_tool_name_anchors(
    labels: torch.Tensor,
    tool_name_sequences: dict,
    ignore_index: int = -100,
    tool_call_open_ids: Optional[List[int]] = None,
    tool_call_close_ids: Optional[List[int]] = None,
) -> dict:
    """Locate the anchor positions (first token of each tool-name span) inside
    ``<tool_call>...</tool_call>`` regions of ``labels``.

    Args:
        labels: [B, L]
        tool_name_sequences: {name: [token_ids...]} from
            resolve_tool_name_token_sequences
        tool_call_open_ids, tool_call_close_ids: token sequences for the
            ``<tool_call>`` and ``</tool_call>`` markers. When BOTH provided,
            matching is restricted to positions strictly between them — so
            literal occurrences of "compress" / "recall" inside ``<think>``
            or response text are not counted as anchors. When either is None
            or empty, falls back to the label-valid region (legacy behavior).

    Returns: {name: List[(batch_idx, position)]}
    """
    out: dict = {name: [] for name in tool_name_sequences}
    valid = labels.ne(ignore_index)
    for b in range(labels.shape[0]):
        seq = labels[b].tolist()
        base_mask = valid[b].tolist()
        mask = _compute_tool_call_span_mask(
            seq, base_mask, tool_call_open_ids, tool_call_close_ids
        )
        for name, needle in tool_name_sequences.items():
            if not needle:
                continue
            anchors = _find_subsequence_anchors(seq, needle, search_mask=mask)
            for pos in anchors:
                out[name].append((b, pos))
    return out


def resolve_tool_call_marker_ids(tokenizer) -> "tuple[List[int], List[int]]":
    """Return ``(open_ids, close_ids)`` for the Qwen ``<tool_call>`` markers.

    Uses bare tokenization without special-token addition. Returns empty
    lists if the tokenizer cannot produce a sequence for either marker —
    callers should treat the empty case as "no tool-call span constraint".
    """
    try:
        open_ids = tokenizer("<tool_call>", add_special_tokens=False).input_ids
        close_ids = tokenizer("</tool_call>", add_special_tokens=False).input_ids
    except Exception:
        return [], []
    return list(open_ids or []), list(close_ids or [])


def compute_inverse_frequency_weights(
    labels: torch.Tensor,
    action_token_ids: dict,
    tool_name_sequences: Optional[dict] = None,
    ignore_index: int = -100,
    floor_weight: float = 0.05,
    ceil_weight: float = 20.0,
    eps: float = 1e-3,
    tool_call_open_ids: Optional[List[int]] = None,
    tool_call_close_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Return per-token weights from inverse class frequency.

    inverse-frequency-weighted:
        weight[c] = total_action_anchors / (num_classes * count[c] + eps)

    Action anchors come from two sources:
      1. Single-token action-start ids (e.g. </Silence>, </Response>)
      2. Tool-name spans from ``tool_name_sequences`` (e.g. "compress",
         "recall"); the first token of each match is the anchor.

    Tokens whose label is not in any anchor set get weight 1.0, so the
    standard CE on think/wrapping/text content is preserved.
    Resulting tensor is clamped to ``[floor_weight, ceil_weight]``.

    Shape matches ``labels``. Designed to be multiplied into the existing
    ``token_loss_weight`` (rather than replace it).
    """
    weights = torch.ones_like(labels, dtype=torch.float32)
    valid_mask = labels.ne(ignore_index)

    # ── Pass 1: count anchors by class ──
    counts: dict = {}
    # Single-token anchors
    for name, tid in action_token_ids.items():
        if tid is None:
            continue
        n = int(((labels == tid) & valid_mask).sum().item())
        counts[name] = n

    # Multi-token anchors (tool names): count = number of matches inside
    # ``<tool_call>...</tool_call>`` regions (when markers provided).
    tool_anchors: dict = {}
    if tool_name_sequences:
        tool_anchors = find_tool_name_anchors(
            labels,
            tool_name_sequences,
            ignore_index=ignore_index,
            tool_call_open_ids=tool_call_open_ids,
            tool_call_close_ids=tool_call_close_ids,
        )
        for name, positions in tool_anchors.items():
            counts[name] = len(positions)

    total = sum(counts.values())
    if total == 0:
        return weights  # no anchors in this batch

    num_classes = max(1, sum(1 for c in counts.values() if c > 0))

    # ── Pass 2: apply per-class weight to anchor positions ──
    # Single-token anchors apply to every position where label matches.
    for name, tid in action_token_ids.items():
        if tid is None:
            continue
        c = counts.get(name, 0)
        if c == 0:
            continue
        w = float(total) / (num_classes * c + eps)
        weights[labels == tid] = w

    # Multi-token anchors: weight applied to the FIRST token of each span
    # only. Body tokens of the tool_call (arguments JSON) keep weight 1.0
    # because they are content, not class-discriminative.
    for name, positions in tool_anchors.items():
        c = counts.get(name, 0)
        if c == 0:
            continue
        w = float(total) / (num_classes * c + eps)
        for (b, pos) in positions:
            weights[b, pos] = w

    return weights.clamp_(floor_weight, ceil_weight)


def focal_loss_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_ids: dict,
    tool_name_sequences: Optional[dict] = None,
    gamma: float = 2.0,
    auto_alpha: bool = True,
    ignore_index: int = -100,
    extra_token_weight: Optional[torch.Tensor] = None,
    tool_call_open_ids: Optional[List[int]] = None,
    tool_call_close_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """focal-modulation focal loss with optional auto-alpha class rebalancing.

    Returns a [batch * seq_len_shift] flat tensor of per-token loss values.
    The trainer's existing aggregation (sum / scatter / per-sample mean)
    consumes this without further modification.

    Args:
        logits: [batch, seq_len, vocab_size] — already shifted to predict
            labels[..., 1:]; caller is expected to pass the shifted form.
        labels: [batch, seq_len] — shifted to match logits. ``ignore_index``
            entries contribute zero loss.
        action_token_ids: from ``resolve_action_token_ids``.
        gamma: focal exponent. 0.0 disables focal modulation (standard CE).
        auto_alpha: when True, scales action-start loss by per-batch inverse
            frequency (akin to auto-alpha class rebalancing). Non-action tokens get
            alpha=1.0.
        extra_token_weight: optional [batch, seq_len] multiplier. Applied on
            top of focal + alpha so existing logic composes.

    Output:
        flat_loss [N] where N = batch * seq_len. Positions with label =
        ignore_index are zero. Aggregation up to the trainer.
    """
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)

    valid_mask = flat_labels.ne(ignore_index)
    # Replace ignore_index with 0 to keep gather safe; we mask out at the end.
    safe_labels = flat_labels.masked_fill(~valid_mask, 0)

    # Standard CE per token (no reduction).
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    if gamma != 0.0:
        with torch.no_grad():
            p_target = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1).exp()
            focal_term = (1.0 - p_target).clamp_(0.0, 1.0).pow(gamma)
    else:
        focal_term = torch.ones_like(nll)

    # Alpha per token: 1.0 for non-action, inverse-frequency for action.
    alpha = torch.ones_like(nll)
    if auto_alpha:
        # ── Pass 1: count anchors by class (single-token + tool-name spans) ──
        counts: dict = {}

        for name, tid in action_token_ids.items():
            if tid is None:
                continue
            counts[name] = int(((flat_labels == tid) & valid_mask).sum().item())

        tool_anchors: dict = {}
        if tool_name_sequences:
            # Use 2D labels for span detection (inside tool_call regions when
            # markers provided), then map to flat positions.
            tool_anchors_2d = find_tool_name_anchors(
                labels,
                tool_name_sequences,
                ignore_index=ignore_index,
                tool_call_open_ids=tool_call_open_ids,
                tool_call_close_ids=tool_call_close_ids,
            )
            for name, positions in tool_anchors_2d.items():
                counts[name] = len(positions)
                # Flatten (b, pos) → b*seq_len + pos
                tool_anchors[name] = [b * seq_len + p for (b, p) in positions]

        total_action = sum(counts.values())
        if total_action > 0:
            num_classes = max(1, sum(1 for c in counts.values() if c > 0))

            # Single-token anchors
            for name, tid in action_token_ids.items():
                if tid is None:
                    continue
                c = counts.get(name, 0)
                if c == 0:
                    continue
                a = float(total_action) / (num_classes * c + 1e-3)
                a = max(0.05, min(20.0, a))
                alpha = torch.where(
                    flat_labels == tid, torch.full_like(alpha, a), alpha
                )

            # Multi-token tool-name anchors (apply alpha to first token only)
            for name, flat_positions in tool_anchors.items():
                c = counts.get(name, 0)
                if c == 0 or not flat_positions:
                    continue
                a = float(total_action) / (num_classes * c + 1e-3)
                a = max(0.05, min(20.0, a))
                idx = torch.tensor(flat_positions, device=alpha.device, dtype=torch.long)
                alpha.index_fill_(0, idx, a)

    flat_loss = focal_term * alpha * nll

    if extra_token_weight is not None:
        flat_w = extra_token_weight.reshape(-1).to(dtype=flat_loss.dtype, device=flat_loss.device)
        if flat_w.numel() == flat_loss.numel():
            flat_loss = flat_loss * flat_w

    # Zero out ignored positions.
    flat_loss = flat_loss * valid_mask.to(flat_loss.dtype)
    return flat_loss


def aggregate_per_sample(
    flat_loss: torch.Tensor,
    flat_labels: torch.Tensor,
    flat_weight: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reduce flat per-token loss to per-sample mean.

    Mirrors the existing ``_per_sample_ce_loss`` aggregation so the trainer
    can keep using it as a drop-in replacement. Returns [batch] tensor.
    """
    sample_index = torch.arange(batch_size, device=flat_loss.device).repeat_interleave(seq_len)
    valid = flat_labels.ne(ignore_index)

    if flat_weight is None:
        flat_weight = torch.ones_like(flat_loss)

    flat_weight = flat_weight * valid.to(flat_weight.dtype)

    loss_sum = torch.zeros(batch_size, device=flat_loss.device, dtype=flat_loss.dtype)
    denom = torch.zeros(batch_size, device=flat_loss.device, dtype=flat_loss.dtype)
    loss_sum = loss_sum.scatter_add(0, sample_index, flat_loss)
    denom = denom.scatter_add(0, sample_index, flat_weight)
    return loss_sum / denom.clamp_min(1.0)
