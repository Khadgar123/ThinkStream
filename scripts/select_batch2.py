"""
Select batch 2 videos: 400 stratified by subdataset × duration.

Reads:  data/video_catalog_30s_plus.csv
        data/agent_v5/video_registry.jsonl   (batch 1, to exclude)
Writes: data/agent_v5/batch2_videos.jsonl    (same schema as batch 1)

Strategy: per-stratum reproducible random sample (seed=42).
- Excludes videos used by streamo, batch 1, or invalid (codec/duration).
- Stratification by subdataset (or filename for Koala_raw) AND duration bucket.
"""

import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "data" / "video_catalog_30s_plus.csv"
BATCH1 = PROJECT_ROOT / "data" / "agent_v5" / "video_registry.jsonl"
OUT = PROJECT_ROOT / "data" / "agent_v5" / "batch2_videos.jsonl"

SEED = 42
TARGET = 400

# Per-stratum quotas (subdataset → count)
STRATA = {
    "VideoMind-Dataset/cosmo_cap":            70,
    "VideoMind-Dataset/internvid_vtime":      50,
    "VideoMind-Dataset/charades_sta":         25,
    "VideoMind-Dataset/didemo":               15,
    "VideoMind-Dataset/coin":                 20,
    "VideoMind-Dataset/activitynet":          20,
    "VideoMind-Dataset/qvhighlights":         10,
    "VideoMind-Dataset/queryd":               15,
    "VideoMind-Dataset/hirest":               15,
    "LLaVA-Video-178K/2_3_m_youtube_v0_1":    50,
    "LLaVA-Video-178K/30_60_s_youtube_v0_1":  10,
    "LLaVA-Video-178K/30_60_s_academic_v0_1": 10,
    "tarsier2_unzip/VATEX":                   18,
    "tarsier2_unzip/Charades":                12,
    "tarsier2_unzip/WebVid-10M_part-1":       10,
    "tarsier2_unzip/WebVid-10M_part-2":       10,
    "Koala_raw":                              10,   # match-by-dataset (subdataset=filename)
    "Koala":                                  10,
    "how_to_step":                            10,
    "how_to_caption":                         10,
}
assert sum(STRATA.values()) == TARGET, f"strata sum={sum(STRATA.values())} != {TARGET}"

# Within each stratum, oversample shorter then truncate to enforce mix
DURATION_MIX_PCT = {
    "30-60":   0.12,
    "60-120":  0.30,
    "120-240": 0.35,
    "240-600": 0.18,
    "600+":    0.05,
}


def duration_bucket(d: float) -> str:
    if d < 60:    return "30-60"
    if d < 120:   return "60-120"
    if d < 240:   return "120-240"
    if d < 600:   return "240-600"
    return "600+"


def is_valid_codec(c: str) -> bool:
    c = (c or "").lower()
    return c in ("h264", "hevc", "h265", "vp9", "av1") or c.startswith("h26")


def video_id_from_path(path: str, dataset: str) -> str:
    """Match batch 1's id convention: filename stem (no extension)."""
    name = Path(path).name
    return name.rsplit(".", 1)[0] if "." in name else name


def stratum_key(dataset: str, sub: str) -> str:
    """Unify Koala_raw / Koala / how_to_*: subdataset is per-file there."""
    if dataset in ("Koala_raw", "Koala", "how_to_step", "how_to_caption", "how_to_captio.mp4"):
        return dataset
    return f"{dataset}/{sub}"


def main():
    rng = random.Random(SEED)

    # --- Load batch 1 exclusion set ---
    excluded_paths = set()
    excluded_ids = set()
    if BATCH1.exists():
        with open(BATCH1) as f:
            for line in f:
                v = json.loads(line)
                excluded_paths.add(v["video_path"])
                excluded_ids.add(v["video_id"])
    print(f"[exclude] batch 1 videos: {len(excluded_paths)}")

    # --- Bucket the catalog by (stratum, duration_bucket) ---
    by_stratum: dict = defaultdict(lambda: defaultdict(list))
    skipped_reasons = Counter()
    with open(CATALOG, newline="") as f:
        for r in csv.DictReader(f):
            path = r["video_path"]
            if path in excluded_paths:
                skipped_reasons["batch1_dup"] += 1
                continue
            if r.get("used_in_streamo") == "1":
                skipped_reasons["streamo_used"] += 1
                continue
            if r.get("used_in_thinkstream") == "1":
                skipped_reasons["thinkstream_flag"] += 1
                continue
            try:
                dur = float(r["duration_sec"])
            except (ValueError, TypeError):
                skipped_reasons["bad_duration"] += 1
                continue
            if dur < 30 or dur > 1800:
                skipped_reasons["duration_out_of_range"] += 1
                continue
            try:
                w = int(r["width"]); h = int(r["height"])
                if w * h < 320 * 240:
                    skipped_reasons["resolution_too_low"] += 1
                    continue
            except (ValueError, TypeError):
                pass  # missing res — accept
            if not is_valid_codec(r.get("codec", "")):
                skipped_reasons["bad_codec"] += 1
                continue

            sub = r.get("subdataset", "")
            stratum = stratum_key(r["dataset"], sub)
            if stratum not in STRATA:
                continue  # not a target stratum
            by_stratum[stratum][duration_bucket(dur)].append({
                "video_path": path,
                "video_id": video_id_from_path(path, r["dataset"]),
                "duration_sec": round(dur, 2),
                "dataset": r["dataset"],
                "subdataset": sub,
            })

    print(f"[catalog] candidates per stratum:")
    for s, q in STRATA.items():
        total = sum(len(v) for v in by_stratum[s].values())
        bucket_sizes = {b: len(by_stratum[s][b]) for b in DURATION_MIX_PCT}
        print(f"  {s:48s} total={total:6d} target={q:3d}   buckets={bucket_sizes}")
    print(f"[catalog] skipped: {dict(skipped_reasons)}")

    # --- Sample within each stratum, respecting duration mix ---
    selected = []
    for stratum, quota in STRATA.items():
        target_per_bucket = {b: max(1, round(quota * pct)) for b, pct in DURATION_MIX_PCT.items()}
        # Adjust so they sum to quota
        diff = quota - sum(target_per_bucket.values())
        # Distribute diff into the largest buckets
        sorted_buckets = sorted(target_per_bucket.items(), key=lambda x: -x[1])
        i = 0
        while diff != 0 and sorted_buckets:
            b, _ = sorted_buckets[i % len(sorted_buckets)]
            target_per_bucket[b] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1

        picked = []
        unmet_carryover = 0
        for bucket, want in target_per_bucket.items():
            pool = list(by_stratum[stratum][bucket])
            rng.shuffle(pool)
            take = pool[:want]
            picked.extend(take)
            unmet_carryover += max(0, want - len(take))

        # Fill carryover from any bucket in the same stratum
        if unmet_carryover > 0:
            taken_ids = {p["video_id"] for p in picked}
            spare = [
                v for b in by_stratum[stratum].values() for v in b
                if v["video_id"] not in taken_ids
            ]
            rng.shuffle(spare)
            picked.extend(spare[:unmet_carryover])

        # Safety cap to quota
        picked = picked[:quota]
        selected.extend(picked)

    # If totals fall short of TARGET (some stratum exhausted), top up from the
    # largest remaining pools.
    if len(selected) < TARGET:
        chosen = {v["video_id"] for v in selected}
        leftover = []
        for stratum_buckets in by_stratum.values():
            for bucket_list in stratum_buckets.values():
                leftover.extend(v for v in bucket_list if v["video_id"] not in chosen)
        rng.shuffle(leftover)
        need = TARGET - len(selected)
        selected.extend(leftover[:need])

    # --- Write output (same schema as batch 1 video_registry) ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for v in selected:
            f.write(json.dumps({
                "video_id":     v["video_id"],
                "video_path":   v["video_path"],
                "duration_sec": v["duration_sec"],
                "dataset":      v["dataset"],
            }, ensure_ascii=False) + "\n")

    # --- Distribution summary ---
    print(f"\n[output] {OUT}  ({len(selected)} videos)")
    by_ds = Counter(v["dataset"] for v in selected)
    by_sub = Counter(stratum_key(v["dataset"], v["subdataset"]) for v in selected)
    by_bucket = Counter(duration_bucket(v["duration_sec"]) for v in selected)
    print(f"\n[summary] by dataset:")
    for k, n in by_ds.most_common():
        print(f"  {k:20s} {n:4d}")
    print(f"\n[summary] by stratum (subdataset):")
    for k, n in by_sub.most_common():
        print(f"  {k:48s} {n:4d}  (target {STRATA.get(k,'?')})")
    print(f"\n[summary] by duration bucket:")
    for b in DURATION_MIX_PCT:
        n = by_bucket[b]
        print(f"  {b:8s} {n:4d}  ({100*n/len(selected):.1f}%)")

    durations = [v["duration_sec"] for v in selected]
    durations.sort()
    print(f"\n[summary] duration: min={durations[0]:.0f}s, "
          f"p50={durations[len(durations)//2]:.0f}s, "
          f"p95={durations[int(len(durations)*.95)]:.0f}s, "
          f"max={durations[-1]:.0f}s")


if __name__ == "__main__":
    main()
