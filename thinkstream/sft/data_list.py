"""Dataset registry for ThinkStream SFT/eval data.

Canonical SFT inputs are pass5 trajectory
`*_trajectory.jsonl` files. Canonical RL inputs are verl parquets built from
`*_trajectories.jsonl` by scripts/agent_data/build_verl_parquet.py.

The registry intentionally exposes only trajectory datasets so SFT, RL, and
eval cannot silently mix incompatible prompt layouts.
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # thinkstream/sft/ → ThinkStream/


def _resolve_final_dir() -> Path:
    """Resolve the canonical final/ directory for one generated batch.

    Preferred:
      THINKSTREAM_DATA_ROOT=data/agent_v5/batch2

    Backward compatible:
      AGENT_DATA_DIR may point either at the batch root or directly at final/.
      THINKSTREAM_BATCH=batch2 expands to data/agent_v5/batch2/final.
    """
    explicit_final = os.environ.get("THINKSTREAM_FINAL_DIR")
    if explicit_final:
        p = Path(explicit_final).expanduser()
        return p if p.is_absolute() else _PROJECT_ROOT / p

    root_env = os.environ.get("THINKSTREAM_DATA_ROOT") or os.environ.get("AGENT_DATA_DIR")
    if root_env:
        root = Path(root_env).expanduser()
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        return root if root.name == "final" else root / "final"

    batch = os.environ.get("THINKSTREAM_BATCH", "").strip()
    if batch:
        return _PROJECT_ROOT / "data" / "agent_v5" / batch / "final"
    return _PROJECT_ROOT / "data" / "agent_v5" / "final"


_AGENT_DATA_DIR = _resolve_final_dir()


def _agent_batch_root() -> Path:
    """Return the generated batch root that owns final/rendered outputs."""
    p = _AGENT_DATA_DIR
    if p.name == "final":
        return p.parent
    if p.parent.name == "rendered":
        return p.parents[1]
    return p


def _agent_frames_root() -> str:
    return str(_agent_batch_root() / "frames")


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
    # RL trainer and streaming eval ingest trajectory JSONL/parquets with
    # `questions`, `gold_action_per_chunk`, and full sample metadata.
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

    # ─── Multi-turn trajectory SFT (pass5 trajectory renderer) ──────
    # Each row is one trajectory (between two compress events) carrying a
    # full multi-turn ``messages`` list + inline ``tools`` schema. Consumed
    # by the same WeightedSFTTrainer; ``preprocess_trajectory_sample`` detects
    # ``trajectory_type`` and relaxes the per-row assistant-turn count.
    # Produced by ``scripts.agent_data.pass5.convert_dir(...)``; pipeline
    # The pipeline writes these by default under ``rendered/trajectory``.
    "stream_agent_trajectory_train": {
        # pass4 emits ``train_sft_trajectories.jsonl`` for the SFT split (its
        # canonical naming is ``<split>_trajectories.jsonl`` with split
        # ``train_sft``). pass5.convert_dir strips ``_trajectories`` →
        # ``train_sft_trajectory.jsonl``. Do NOT shorten to
        # ``train_trajectory.jsonl`` — that file is never produced.
        "annotation_path": _agent_path("train_sft_trajectory.jsonl"),
        "data_path": _agent_frames_root(),
    },
    "stream_agent_trajectory_val": {
        "annotation_path": _agent_path("val_trajectory.jsonl"),
        "data_path": _agent_frames_root(),
    },
    "stream_agent_trajectory_test": {
        "annotation_path": _agent_path("test_trajectory.jsonl"),
        "data_path": _agent_frames_root(),
    },
}


def data_list(dataset_names: list) -> list:
    """Resolve dataset names to config dicts.

    Supports sampling: "stream_agent_trajectory_train%50" = 50% of SFT data.
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
