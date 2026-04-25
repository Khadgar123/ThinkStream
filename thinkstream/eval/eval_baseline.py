"""
Baseline model evaluation utilities (streaming mode).

Runs vanilla VLMs on benchmarks using the SAME streaming video delivery
as ThinkStream (chunk-by-chunk, same visual window, same KV cache) but
WITHOUT think/action protocol. The model just sees frames arrive and
answers the question directly when asked.

This ensures a fair comparison: both models see the same temporal
progression of video frames with the same visual window constraints.
The only difference is ThinkStream's trained agent protocol.

Two modes:
  1. streaming (default): uses StreamingWindowInferenceEngine, same as
     ThinkStream eval. Requires model loaded via MODEL_CLS (streaming
     attention patches). Uses baseline_sample_restricted to skip
     think/action and directly pick the answer.
  2. batch: loads all frames at once (unfair, but useful for debugging).
"""

import sys
from pathlib import Path

import torch

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_common import (
    add_common_args,
    build_results,
    load_model_and_processor,
    mcq_predict_streaming,
    save_results,
    setup_distributed,
    cleanup_distributed,
)


# ---------------------------------------------------------------------------
# Baseline sample function (no think/action protocol)
# ---------------------------------------------------------------------------


def baseline_sample_restricted(
    next_token: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    generated_tokens: torch.Tensor,
    generated_length: torch.Tensor,
    restricted_token_ids: list,
    eos_token_id: int,
    **kwargs,
) -> torch.Tensor:
    """Baseline restricted sampling: directly pick the best option token.

    No think budget, no <think>/<response>/<silent> protocol.
    Step 0: pick argmax over restricted_token_ids from logits.
    Step 1+: force EOS.

    This means the model generates exactly ONE token (the answer) and stops.
    Same interface as think_budget_sample_restricted for compatibility with
    streaming_video_chat's sample parameter.
    """
    device = next_token.device

    if step == 0:
        # First token: pick the best option directly from logits
        restricted_ids = torch.tensor(
            restricted_token_ids, device=device, dtype=torch.long
        )
        restricted_logits = logits[:, restricted_ids]  # [B, R]
        top1_local = restricted_logits.argmax(dim=-1)  # [B]
        top1_token = restricted_ids[top1_local]  # [B]
        return top1_token.unsqueeze(1)
    else:
        # After first token: force EOS
        return torch.full_like(next_token, eos_token_id)
