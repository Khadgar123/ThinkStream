"""Select 30 test videos from catalog via stratified sampling."""

import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "data" / "video_catalog_30s_plus.csv.gz"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "agent_v5_test"
NUM_VIDEOS = 30
MIN_DUR = 60
MAX_DUR = 180
SEED = 42


def main():
    random.seed(SEED)

    groups = defaultdict(list)
    with gzip.open(CATALOG, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration_sec"])
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            path = Path(row["video_path"])
            if not path.exists():
                continue
            groups[row["dataset"]].append({
                "video_id": path.stem,
                "video_path": str(path),
                "duration_sec": dur,
                "dataset": row["dataset"],
            })

    print(f"Found {sum(len(v) for v in groups.values())} eligible videos across {len(groups)} datasets")
    for ds, vids in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {ds}: {len(vids)}")

    # Stratified: round-robin from each dataset
    selected = []
    group_keys = sorted(groups.keys())
    random.shuffle(group_keys)
    per_group = max(1, NUM_VIDEOS // len(group_keys))

    for key in group_keys:
        pool = groups[key]
        random.shuffle(pool)
        take = min(per_group, len(pool), NUM_VIDEOS - len(selected))
        selected.extend(pool[:take])

    # Fill remaining
    if len(selected) < NUM_VIDEOS:
        remaining = [v for k in group_keys for v in groups[k] if v not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:NUM_VIDEOS - len(selected)])

    selected = selected[:NUM_VIDEOS]
    print(f"\nSelected {len(selected)} videos:")
    ds_counts = defaultdict(int)
    for v in selected:
        ds_counts[v["dataset"]] += 1
    for ds, cnt in sorted(ds_counts.items()):
        print(f"  {ds}: {cnt}")

    # Save registry
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = OUTPUT_DIR / "video_registry.jsonl"
    with open(registry_path, "w") as f:
        for v in selected:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"\nSaved to {registry_path}")


if __name__ == "__main__":
    main()
