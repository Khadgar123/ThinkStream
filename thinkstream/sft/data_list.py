"""Dataset registry for ThinkStream SFT/eval data.

Canonical SFT inputs are pass5 LLaMA-Factory/DeepEyes-style ShareGPT
`*_messages.jsonl` files. Canonical RL inputs are verl parquets built from
`*_trajectories.jsonl` by scripts/agent_data_v5/build_verl_parquet.py.

Older phase/category entries remain only for archived ablations. Do not use
them as a staged curriculum; production SFT is a single pass over
`stream_agent_sft`.
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
    """Resolve a final/-relative filename to a full path.

    v12.5 (2026-04-29): falls back to .gz when the uncompressed file
    is missing — the new pass4 trajectory files (~145MB raw) are
    committed as .gz to fit GitHub's 100MB limit. read_jsonl in
    data_processor.py reads .gz transparently.
    """
    p = _AGENT_DATA_DIR / filename
    if p.exists():
        return str(p)
    gz = _AGENT_DATA_DIR / (filename + ".gz")
    if gz.exists():
        return str(gz)
    # Caller will hit FileNotFoundError on attempt — that's the correct
    # behavior so config typos surface clearly.
    return str(p)


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

    # ─── v12.4/v12.5 trajectory + flat datasets ──────────────────────
    # New canonical inputs, produced by `python -m
    # scripts.agent_data_v5.pass4`. The OLD per-step files above remain
    # for backward compat (1,635 each, post-MAX_SAMPLES_PER_VIDEO=15
    # density cap). New datasets preserve all 47,289 verified samples
    # from pass3e (no post-cap drop), organized by trajectory.
    #
    # v12.6: SFT trainer ingests `*_messages.jsonl` (LLaMA-Factory
    # ShareGPT format) produced by pass5_messages.py from
    # `train_sft_trajectories.jsonl`. Each row = one chunk's snapshot
    # rendered as messages. The flat `*_full.jsonl` form is kept here as
    # `stream_agent_sft_full` for backward compat with archived ablations
    # but the canonical entry is `stream_agent_sft`.
    "stream_agent_sft": {
        "annotation_path": _agent_path("train_sft_messages.jsonl"),
        "data_path": "./",
    },
    "stream_agent_val": {
        "annotation_path": _agent_path("val_messages.jsonl"),
        "data_path": "./",
    },
    "stream_agent_test": {
        "annotation_path": _agent_path("test_messages.jsonl"),
        "data_path": "./",
    },
    # Legacy flat format — pre pass5_messages converter
    "stream_agent_sft_full": {
        "annotation_path": _agent_path("train_sft_full.jsonl"),
        "data_path": "./",
    },

    # RL trainer + streaming benchmark eval ingest `*_trajectories.jsonl`
    # — one row per trajectory, with `questions`, `gold_action_per_chunk`,
    # full `samples` list. Consumed by `_calc_rewards_v12_trajectory` for
    # multi-question per-ask scoring.
    "stream_agent_rl_traj": {
        "annotation_path": _agent_path("train_rl_trajectories.jsonl"),
        "data_path": "./",
    },
    "stream_agent_val_traj": {
        "annotation_path": _agent_path("val_trajectories.jsonl"),
        "data_path": "./",
    },
    "stream_agent_test_traj": {
        "annotation_path": _agent_path("test_trajectories.jsonl"),
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
