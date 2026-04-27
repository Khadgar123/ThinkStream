"""Dataset registry for the per-timestep agent.

Production: `stream_agent_p5` (= all train samples mixed). Used by both
SFT (PHASE=mixed) and GDPO RL (default DATASET in grpo_train.sh).

Ablation-only diagnostic splits: `stream_agent_p1`, `stream_agent_p2`,
`stream_agent_c1`. These break the production dataset down by sample
category (basic / recall / compress) so you can train or eval on a
single category for ablations. Per the v9.2 paper survey
(8/8 same-era 2026 works use single-stage SFT), these are NOT a
training curriculum — production is a single SFT pass on `_p5`.

The old `stream_agent_c2` (model-self-pick range SFT) was removed in
v11: range exploration moved to RL stage and is supervised by the
`overflow_pen` reward in `thinkstream/trainer/grpo.py`.
"""

from pathlib import Path

# Base directory for pipeline output.
# Resolves relative to project root (ThinkStream/), not CWD.
import os
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # thinkstream/sft/ → ThinkStream/
_AGENT_DATA_DIR = Path(
    os.environ.get("AGENT_DATA_DIR", str(_PROJECT_ROOT / "data" / "agent_v5" / "final"))
)


def _agent_path(filename: str) -> str:
    return str(_AGENT_DATA_DIR / filename)


DATASET_REGISTRY = {
    # ─── Production ─────────────────────────────────────────────────
    # All train samples mixed. Legacy alias — kept for backward compat.
    # New runs should prefer `stream_agent_sft` (SFT-only) and
    # `stream_agent_rl` (RL-only, held out from SFT) so the GDPO stage
    # cannot reward-hack via memorization on SFT-seen prompts.
    "stream_agent_p5": {
        "annotation_path": _agent_path("phase5_train.jsonl"),
        "data_path": "./",
    },
    # Same content as p5, kept as an alias for explicit "everything" loads.
    "stream_agent_all": {
        "annotation_path": _agent_path("train.jsonl"),
        "data_path": "./",
    },

    # ─── SFT / RL split (v11.1, 2026-04-27) ─────────────────────────
    # train.jsonl is split by video_id (~80/20) into disjoint pools.
    # See data/agent_v5/final/split_manifest.json for the seed and the
    # full list of video_ids per side. The SFT pool is what the
    # per-timestep SFT trainer should consume in production; the RL
    # pool is the held-out prompt set for GDPO/GRPO.
    "stream_agent_sft": {
        "annotation_path": _agent_path("train_sft.jsonl"),
        "data_path": "./",
    },
    "stream_agent_rl": {
        "annotation_path": _agent_path("train_rl.jsonl"),
        "data_path": "./",
    },

    # ─── Ablation-only diagnostic splits ─────────────────────────────
    # Per-category subsets of train samples, for category-specific eval
    # or ablation. Do NOT chain into a curriculum — see module docstring.
    "stream_agent_p1": {
        # Basic silent + response samples only.
        "annotation_path": _agent_path("phase1_train.jsonl"),
        "data_path": "./",
    },
    "stream_agent_p2": {
        # Recall samples (recall_query / recall_response / recall_silent)
        # and query-aware silent/response.
        "annotation_path": _agent_path("phase2_train.jsonl"),
        "data_path": "./",
    },
    "stream_agent_c1": {
        # Compress samples (system trigger + teacher gold range).
        "annotation_path": _agent_path("c1_train.jsonl"),
        "data_path": "./",
    },

    # ─── Eval / held-out sets ────────────────────────────────────────
    "stream_agent_val": {
        "annotation_path": _agent_path("val.jsonl"),
        "data_path": "./",
    },
    "stream_agent_test": {
        "annotation_path": _agent_path("test.jsonl"),
        "data_path": "./",
    },
}


def data_list(dataset_names: list) -> list:
    """Resolve dataset names to config dicts.

    Supports sampling: "stream_agent_p1%50" = 50% of phase 1 data.
    """
    result = []
    for name in dataset_names:
        name = name.strip()
        if not name:
            continue

        sampling_rate = 1.0
        if "%" in name:
            name, rate_str = name.split("%")
            sampling_rate = float(rate_str) / 100.0

        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )

        entry = dict(DATASET_REGISTRY[name])
        entry["sampling_rate"] = sampling_rate
        result.append(entry)

    return result
