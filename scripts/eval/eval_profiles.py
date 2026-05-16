"""Eval-time context profiles — 16k (default / SFT-aligned) vs 32k (extended).

The two profiles differ ONLY in eval-side caps that don't affect model
distribution: model_max_length, queries-history cap, recall metadata cap,
max_new_tokens. They do NOT change SFT-baked constants
(VISUAL_WINDOW_CHUNKS=8, RECENT_THINKS_TOKEN_BUDGET=4000,
COMPRESS_TOKEN_THRESHOLD=3200, MAX_COMPRESSED_SEGMENTS=5,
merged-segment cap=280) — those would require pass2 re-rollout + SFT
retrain.

v12.5 (2026-04-29) — chunk semantics changed 2s → 1s/chunk. v12.15
uses an 8-chunk visual window (16 frames @ 2fps) so text-memory horizon
stays above visual horizon.

v12.15 (2026-05-12) — visual window is reduced to 8 chunks (16 frames @
2fps) so prompt-side visual context matches the video-KV eviction window.

Per-profile token-budget breakdown (worst-case at the most-loaded chunk):

    Component         | 16k profile  | 32k profile  | Notes
    ------------------|--------------|--------------|----------------------------
    system+tools      |   ~400       |   ~400       | V12 protocol + tool schema
    visual frames     | ~3760 (16fr) | ~3760 (16fr) | 8 chunks × 470 tok/chunk
                      |              |              | (= 16 frames @ ~235 tok ea).
                      |              |              | SFT-baked.
    compressed segs   | 5×280=1400   | 5×280=1400   | MAX_SEGMENTS=5, SUMMARY cap
                      |              |              | 280; both SFT-baked.
    recent_thinks     | ~3500 peak   | ~3500 peak   | Compress fires at 3200 tok;
                      |              |              | peak after a 100-tok think
                      |              |              | before next-step compress.
                      |              |              | SFT-baked.
    queries           |  3 × ~50=150 |  3 × ~50=150 | EVAL-SIDE: capped in
                      |              |              | format_queries_block.
    recall_result     |  ~80         |  ~80         | Metadata only.
    user_input        |   ~50        |   ~50        |
    assistant output  |  256         |  512         | EVAL-SIDE max_new_tokens.
    ------------------|--------------|--------------|----------------------------
    SUBTOTAL          | ~10,166      | ~11,572      |
    model_max_length  | 16,384       | 32,768       |
    HEADROOM          | ~6,218       | ~21,196      |

Compression behaviour (BOTH profiles, since the trigger is SFT-baked):

  - recent_thinks accumulates until the runtime memory budget fires.
  - The system injects a bare <compress_trigger/>. The trigger is a boolean
    memory-pressure event, not a gold action label and not a range hint.
  - Model writes a compress tool_call with its selected time_range and summary.
  - The selected thinks are removed from recent_thinks, replaced by a
    summary segment appended to compressed_segments.
  - When compressed_segments > 5, the OLDEST TWO are merged into one
    segment (also capped at 200 tok). This means after 5+ compress
    events, the earliest segment may represent 8+, 16+, 32+ original
    thinks.

Recall behaviour (BOTH profiles):

  - Model emits a recall <tool_call>{"name":"recall", ...}</tool_call>.
  - Retriever samples up to 4 chunks uniformly from the requested historical
    start_time/end_time interval in the raw retrieval_archive.
  - Returns a Qwen tool-response turn with a short plain status line plus
    recalled visual evidence, then emits its final response.

When to use which profile:

  * 16k (default): SFT-aligned. Use unless you observe overflow truncation
    in eval logs after the v9.4.2 pixel/queries/recall fixes. Closest to
    training distribution.
  * 32k: when long videos (>500 chunks) push the cumulative non-visual
    zones past the 16k headroom. Requires
    Qwen3-VL with at least 32k native context (most variants OK; check
    config.json `max_position_embeddings` and `rope_scaling`). Slightly
    more OOD than 16k since we're feeding longer position ids than SFT
    saw.

Usage:
    from scripts.eval.eval_profiles import apply_profile, EVAL_PROFILES
    cfg = apply_profile("32k")        # mutates agent_protocol globals
    # cfg["model_max_length"] etc. for tokenizer/CLI use
"""
from typing import Dict


EVAL_PROFILES: Dict[str, Dict] = {
    "16k": {
        # Tokenizer / model
        "model_max_length": 16384,
        # v12.5: 128 → 256 (longer answers possible under 4000-tok memory)
        "max_new_tokens_default": 256,
        # agent_protocol caps. recall_text_max_chars is legacy/no-op now that
        # recall_result is metadata-only.
        "query_history_policy": "recent_k",
        "queries_history_cap": 3,
        "recall_text_max_chars": 1600,
        # For the comparison report
        "subtotal_tokens_estimate": 8454,
        "headroom_tokens_estimate": 7930,
    },
    "32k": {
        "model_max_length": 32768,
        "max_new_tokens_default": 512,
        "query_history_policy": "recent_k",
        "queries_history_cap": 3,
        "recall_text_max_chars": 3000,
        "subtotal_tokens_estimate": 9860,
        "headroom_tokens_estimate": 22908,
    },
}


def apply_profile(name: str) -> Dict:
    """Apply the named profile's eval-side caps to agent_protocol globals.

    Returns the profile dict so callers can pull model_max_length /
    max_new_tokens for their tokenizer / generate config.

    Raises KeyError if `name` not in EVAL_PROFILES.

    Idempotent: calling repeatedly with the same name is a no-op (no
    monotonic state to corrupt).
    """
    cfg = EVAL_PROFILES[name]
    # Late import to avoid circular issues if eval scripts probe profiles
    # before importing agent_protocol.
    from thinkstream.data import agent_protocol
    agent_protocol.QUERY_HISTORY_POLICY = cfg.get(
        "query_history_policy",
        agent_protocol.QUERY_HISTORY_POLICY,
    )
    agent_protocol.QUERIES_HISTORY_CAP = cfg["queries_history_cap"]
    agent_protocol.RECALL_TEXT_MAX_CHARS = cfg["recall_text_max_chars"]
    return cfg


def describe_profile(name: str) -> str:
    """Human-readable single-line description for logging at startup."""
    cfg = EVAL_PROFILES[name]
    return (f"profile={name}: max_len={cfg['model_max_length']}, "
            f"query_policy={cfg.get('query_history_policy', 'recent_k')}, "
            f"queries_cap={cfg['queries_history_cap']}, "
            f"recall_metadata_only=True, "
            f"max_new_tokens={cfg['max_new_tokens_default']} | "
            f"~{cfg['subtotal_tokens_estimate']} tok subtotal, "
            f"~{cfg['headroom_tokens_estimate']} tok headroom")
