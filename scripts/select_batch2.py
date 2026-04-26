"""
Select batch 2 videos: 400 stratified for ThinkStream training.

v2: Lifts the conservative `used_in_streamo` exclusion (it only marks dataset
overlap with a sibling project — the videos themselves aren't contaminated)
and re-balances strata to favor procedural / instructional content, the
content type that best exercises ThinkStream's 14 question families and the
streaming agent's memory + recall + compress mechanisms.

Also reserves 100 disjoint videos as a held-out eval split.

Reads:  data/video_catalog_30s_plus.csv
        data/agent_v5/video_registry.jsonl   (batch 1, excluded)
Writes: data/agent_v5/batch2_videos.jsonl    (400 train videos)
        data/agent_v5/eval_videos.jsonl      (100 held-out videos)
"""

import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "data" / "video_catalog_30s_plus.csv"
BATCH1 = PROJECT_ROOT / "data" / "agent_v5" / "video_registry.jsonl"
OUT_TRAIN = PROJECT_ROOT / "data" / "agent_v5" / "batch2_videos.jsonl"
OUT_EVAL = PROJECT_ROOT / "data" / "agent_v5" / "eval_videos.jsonl"

SEED = 42
TRAIN_TARGET = 400
EVAL_TARGET = 100

# Procedural / instructional → universal training value:
# they exercise F1 (OCR / recipe text), F4 (spatial layouts), E1/E2 (action +
# state change), P1 (procedure), C1 (compare before/after), R1 (chef reappears),
# F5 (repetitive stir/chop), F6 (predictable next step), M1 (multi-step
# commentary), N1 (plausible-but-absent), and S1 (multi-entity scenes).
#
# We weight these heavily; cap auto-captioned / generic web video, since they
# add quantity but not the structural variety the agent needs.
STRATA_TRAIN = {
    # ── Tier S: procedural / instructional ──
    "how_to_step":                            80,   # NEW (was streamo-locked)
    "Koala_raw":                              60,
    "Koala":                                  50,   # NEW (was streamo-locked)
    "VideoMind-Dataset/coin":                 40,   # COIN: explicit step structure
    "how_to_caption":                         30,   # NEW (was streamo-locked)
    "VideoMind-Dataset/hirest":               30,   # tutorial highlights
    # ── Tier A: activity / daily action (F5 / E1 friendly) ──
    "VideoMind-Dataset/charades_sta":         25,
    "tarsier2_unzip/Charades":                15,
    "VideoMind-Dataset/activitynet":          20,
    # ── Tier B: long YouTube — natural streaming length ──
    "LLaVA-Video-178K/2_3_m_youtube_v0_1":    20,
    "LLaVA-Video-178K/1_2_m_youtube_v0_1":    10,
    # ── Tier C: time-grounded / multi-entity ──
    "VideoMind-Dataset/didemo":               10,
    "VideoMind-Dataset/queryd":               10,
}
assert sum(STRATA_TRAIN.values()) == TRAIN_TARGET, \
    f"strata sum={sum(STRATA_TRAIN.values())} != {TRAIN_TARGET}"

# Eval split: same domain mix as training so eval distribution matches train,
# but disjoint videos.
STRATA_EVAL = {
    "how_to_step":                            20,
    "Koala_raw":                              15,
    "Koala":                                  15,
    "VideoMind-Dataset/coin":                 10,
    "how_to_caption":                         10,
    "VideoMind-Dataset/hirest":                8,
    "VideoMind-Dataset/charades_sta":          5,
    "VideoMind-Dataset/activitynet":           5,
    "LLaVA-Video-178K/2_3_m_youtube_v0_1":     5,
    "tarsier2_unzip/Charades":                 4,
    "VideoMind-Dataset/didemo":                3,
}
assert sum(STRATA_EVAL.values()) == EVAL_TARGET, \
    f"eval strata sum={sum(STRATA_EVAL.values())} != {EVAL_TARGET}"

# Duration mix favors lengths where the streaming agent's memory mechanisms
# actually trigger (compression at ~30 chunks ≈ 60s; recall meaningful at
# >12 chunks separation ≈ 24s).
DURATION_MIX_PCT = {
    "30-60":    0.10,   # quick warm-up samples; not enough for compress
    "60-120":   0.25,   # compression begins to matter
    "120-240":  0.40,   # main trajectory band
    "240-600":  0.20,   # long recall / multi-compress
    "600+":     0.05,   # extreme cases for memory-overflow training
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


def video_id_from_path(path: str) -> str:
    name = Path(path).name
    return name.rsplit(".", 1)[0] if "." in name else name


def stratum_key(dataset: str, sub: str) -> str:
    """Per-file datasets bucket by dataset name; others by `dataset/sub`."""
    if dataset in ("Koala_raw", "Koala", "how_to_step", "how_to_caption", "how_to_captio.mp4"):
        return dataset
    return f"{dataset}/{sub}"


def load_candidate_pool(excluded_paths):
    """Bucket the catalog into (stratum → bucket → [video])."""
    pool = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    target_strata = set(STRATA_TRAIN) | set(STRATA_EVAL)
    with open(CATALOG, newline="") as f:
        for r in csv.DictReader(f):
            path = r["video_path"]
            if path in excluded_paths:
                skipped["batch1_dup"] += 1
                continue
            # NOTE: we no longer drop streamo-flagged videos. Streamo just
            # used these in a sibling project — videos themselves are clean.
            if r.get("used_in_thinkstream") == "1":
                skipped["thinkstream_already"] += 1
                continue
            try:
                dur = float(r["duration_sec"])
            except (ValueError, TypeError):
                skipped["bad_duration"] += 1
                continue
            if dur < 30 or dur > 1800:
                skipped["duration_oor"] += 1
                continue
            try:
                if int(r["width"]) * int(r["height"]) < 320 * 240:
                    skipped["res_low"] += 1
                    continue
            except (ValueError, TypeError):
                pass
            if not is_valid_codec(r.get("codec", "")):
                skipped["bad_codec"] += 1
                continue
            stratum = stratum_key(r["dataset"], r.get("subdataset", ""))
            if stratum not in target_strata:
                continue
            pool[stratum][duration_bucket(dur)].append({
                "video_path": path,
                "video_id": video_id_from_path(path),
                "duration_sec": round(dur, 2),
                "dataset": r["dataset"],
                "subdataset": r.get("subdataset", ""),
            })
    return pool, skipped


def sample_strata(pool, strata, rng, claimed: set):
    """Sample per-stratum with duration-mix targets. claimed is mutated."""
    selected = []
    for stratum, quota in strata.items():
        # split quota across duration buckets
        per_bucket = {b: max(1, round(quota * pct)) for b, pct in DURATION_MIX_PCT.items()}
        diff = quota - sum(per_bucket.values())
        sorted_b = sorted(per_bucket.items(), key=lambda x: -x[1])
        i = 0
        while diff != 0 and sorted_b:
            b = sorted_b[i % len(sorted_b)][0]
            per_bucket[b] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1

        picked = []
        carry = 0
        for bucket, want in per_bucket.items():
            avail = [v for v in pool[stratum][bucket] if v["video_id"] not in claimed]
            rng.shuffle(avail)
            take = avail[:want]
            picked.extend(take)
            carry += max(0, want - len(take))
        # carry-over from same stratum
        if carry > 0:
            taken = {v["video_id"] for v in picked}
            spare = [
                v for b in pool[stratum].values() for v in b
                if v["video_id"] not in taken and v["video_id"] not in claimed
            ]
            rng.shuffle(spare)
            picked.extend(spare[:carry])

        picked = picked[:quota]
        for v in picked:
            claimed.add(v["video_id"])
        selected.extend(picked)

    # global fill if we are below target
    return selected


def topup_to_target(selected, pool, target, rng, claimed):
    if len(selected) >= target:
        return selected[:target]
    leftover = []
    for stratum_buckets in pool.values():
        for vs in stratum_buckets.values():
            leftover.extend(v for v in vs if v["video_id"] not in claimed)
    rng.shuffle(leftover)
    need = target - len(selected)
    extras = leftover[:need]
    for v in extras:
        claimed.add(v["video_id"])
    selected.extend(extras)
    return selected


def write_jsonl(out_path, videos):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for v in videos:
            f.write(json.dumps({
                "video_id":     v["video_id"],
                "video_path":   v["video_path"],
                "duration_sec": v["duration_sec"],
                "dataset":      v["dataset"],
            }, ensure_ascii=False) + "\n")


def report(name, videos):
    by_ds = Counter(v["dataset"] for v in videos)
    by_sub = Counter(stratum_key(v["dataset"], v["subdataset"]) for v in videos)
    by_bk = Counter(duration_bucket(v["duration_sec"]) for v in videos)
    durs = sorted(v["duration_sec"] for v in videos)
    print(f"\n=== {name}: {len(videos)} videos ===")
    print(f"  duration: min={durs[0]:.0f}s, p50={durs[len(durs)//2]:.0f}s, "
          f"p95={durs[int(len(durs)*.95)]:.0f}s, max={durs[-1]:.0f}s")
    print(f"  buckets:")
    for b in DURATION_MIX_PCT:
        n = by_bk[b]
        print(f"    {b:8s} {n:3d}  ({100*n/len(videos):.1f}%)")
    print(f"  by stratum:")
    for k, n in by_sub.most_common():
        print(f"    {k:48s} {n:3d}")
    print(f"  by dataset:")
    for k, n in by_ds.most_common():
        print(f"    {k:25s} {n:4d}")


def main():
    rng = random.Random(SEED)

    # Exclude batch 1
    excluded = set()
    if BATCH1.exists():
        with open(BATCH1) as f:
            for line in f:
                excluded.add(json.loads(line)["video_path"])
    print(f"[exclude] batch 1: {len(excluded)}")

    pool, skipped = load_candidate_pool(excluded)
    print(f"[catalog] skip reasons: {dict(skipped)}")

    # Sanity: print pool sizes
    print(f"\n[pool sizes per stratum]")
    for s in set(STRATA_TRAIN) | set(STRATA_EVAL):
        total = sum(len(b) for b in pool[s].values())
        bsizes = {b: len(pool[s][b]) for b in DURATION_MIX_PCT}
        print(f"  {s:48s} total={total:>7d}  buckets={bsizes}")

    claimed = set()
    # Sample TRAIN first (priority), then EVAL on what remains
    train_videos = sample_strata(pool, STRATA_TRAIN, rng, claimed)
    train_videos = topup_to_target(train_videos, pool, TRAIN_TARGET, rng, claimed)
    eval_videos = sample_strata(pool, STRATA_EVAL, rng, claimed)
    eval_videos = topup_to_target(eval_videos, pool, EVAL_TARGET, rng, claimed)

    write_jsonl(OUT_TRAIN, train_videos)
    write_jsonl(OUT_EVAL, eval_videos)

    print(f"\n[output] {OUT_TRAIN}")
    print(f"[output] {OUT_EVAL}")

    report("BATCH 2 (train)", train_videos)
    report("EVAL (held-out)", eval_videos)

    overlap = {v["video_id"] for v in train_videos} & {v["video_id"] for v in eval_videos}
    assert not overlap, f"train/eval overlap: {overlap}"
    print(f"\n[check] train/eval overlap: 0  ✓")


if __name__ == "__main__":
    main()
