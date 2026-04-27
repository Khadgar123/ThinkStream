"""Offline SFT/RL split for an existing train.jsonl.

Use this on the cluster (or any host that has data/agent_v5/final/train.jsonl
but not the *_sft / *_rl variants) to deterministically regenerate:

    data/agent_v5/final/train_sft.jsonl
    data/agent_v5/final/train_rl.jsonl
    data/agent_v5/final/split_manifest.json

The split is by video_id (atomic), seed=42, RL = 20% of train videos.
The 5 `recall_silent` outliers are dropped from the SFT pool to keep the
ClassBalancedDistributedSampler weight ratio bounded (was 107x with
those samples → 9.5x without).

This produces byte-identical files to what `pipeline.py` writes at the
end of a full run, given the same train.jsonl input. It does NOT
re-process video frames or hit any model — pure jsonl manipulation.

Usage:
    python -m scripts.agent_data_v5.split_train_sft_rl
    python -m scripts.agent_data_v5.split_train_sft_rl --rl-frac 0.25
    AGENT_DATA_DIR=/cluster/path/data/agent_v5/final \\
        python -m scripts.agent_data_v5.split_train_sft_rl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

DEFAULT_FINAL_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "agent_v5" / "final"
)
DEFAULT_SEED = 42
DEFAULT_RL_FRAC = 0.20
DROP_SAMPLE_TYPES = {"recall_silent"}  # see module docstring


def split_train(
    final_dir: Path,
    seed: int = DEFAULT_SEED,
    rl_frac: float = DEFAULT_RL_FRAC,
    drop_sample_types: set = DROP_SAMPLE_TYPES,
) -> dict:
    train_path = final_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run pipeline.py first or "
            f"override AGENT_DATA_DIR / --final-dir."
        )

    rows: List[dict] = [json.loads(l) for l in train_path.open()]
    n_before = len(rows)
    rows = [r for r in rows if r.get("sample_type") not in drop_sample_types]
    n_dropped = n_before - len(rows)

    # Deterministic split: sorted video_ids → shuffle → first 80% SFT, rest RL.
    video_ids = sorted({r["video_id"] for r in rows})
    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)
    n_sft = int(len(shuffled) * (1 - rl_frac))
    sft_vids = set(shuffled[:n_sft])
    rl_vids = set(shuffled[n_sft:])
    assert sft_vids.isdisjoint(rl_vids)

    sft_rows = [r for r in rows if r["video_id"] in sft_vids]
    rl_rows = [r for r in rows if r["video_id"] in rl_vids]

    sft_path = final_dir / "train_sft.jsonl"
    rl_path = final_dir / "train_rl.jsonl"
    manifest_path = final_dir / "split_manifest.json"

    sft_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft_rows)
    )
    rl_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rl_rows)
    )
    manifest = {
        "seed": seed,
        "rl_frac": rl_frac,
        "drop_sample_types": sorted(drop_sample_types),
        "sft_videos": sorted(sft_vids),
        "rl_videos": sorted(rl_vids),
        "counts": {
            "input_train_rows": n_before,
            "dropped_rows": n_dropped,
            "sft_videos": len(sft_vids),
            "rl_videos": len(rl_vids),
            "sft_samples": len(sft_rows),
            "rl_samples": len(rl_rows),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--final-dir",
        type=Path,
        default=Path(os.environ.get("AGENT_DATA_DIR", str(DEFAULT_FINAL_DIR))),
        help="Directory containing train.jsonl (also where outputs land).",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--rl-frac", type=float, default=DEFAULT_RL_FRAC)
    args = ap.parse_args()

    manifest = split_train(args.final_dir, seed=args.seed, rl_frac=args.rl_frac)
    counts = manifest["counts"]
    print(f"Wrote split into {args.final_dir}/")
    print(
        f"  train_sft.jsonl: {counts['sft_videos']} videos / "
        f"{counts['sft_samples']} samples"
    )
    print(
        f"  train_rl.jsonl:  {counts['rl_videos']} videos / "
        f"{counts['rl_samples']} samples"
    )
    if counts["dropped_rows"]:
        print(f"  dropped {counts['dropped_rows']} sample_type∈{DROP_SAMPLE_TYPES}")


if __name__ == "__main__":
    main()
