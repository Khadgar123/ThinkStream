"""Phase-based dataset registry for per-timestep agent SFT.

Each phase has its own JSONL file produced by the data construction pipeline.
Training scripts select phase via --dataset_use, e.g. "stream_agent_p1".
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


# Phase-specific datasets
DATASET_REGISTRY = {
    # Phase 1: protocol alignment (silent + response)
    "stream_agent_p1": {
        "annotation_path": _agent_path("phase1_train.jsonl"),
        "data_path": "./",
    },
    # Phase 2: recall learning
    "stream_agent_p2": {
        "annotation_path": _agent_path("phase2_train.jsonl"),
        "data_path": "./",
    },
    # C1: system-specified compression
    "stream_agent_c1": {
        "annotation_path": _agent_path("c1_train.jsonl"),
        "data_path": "./",
    },
    # C2: model-selected compression range
    "stream_agent_c2": {
        "annotation_path": _agent_path("c2_train.jsonl"),
        "data_path": "./",
    },
    # Phase 5: mixed training (all action types)
    "stream_agent_p5": {
        "annotation_path": _agent_path("phase5_train.jsonl"),
        "data_path": "./",
    },
    # Full dataset (all phases, for debugging)
    "stream_agent_all": {
        "annotation_path": _agent_path("train.jsonl"),
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
