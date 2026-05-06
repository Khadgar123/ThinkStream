#!/usr/bin/env python
"""Build video-disjoint batch1 SFT/DAgger experiment splits.

This script creates a reproducible experiment directory under batch1 with
three training stages that do not share videos:

  warmup/  - natural rows plus balanced-row baseline for single-step SFT
  dagger/  - clean anchor rows + trajectories for post-warmup DAgger rollout
  direct/  - clean anchor rows + trajectories for base-policy DAgger rollout

Each stage writes both `train_sft_messages_natural.jsonl` (preferred with
class-loss weights) and `train_sft_messages.jsonl` (legacy deterministic
over/under-sampled baseline).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_RATIOS = {
    "silent": 0.35,
    "response": 0.25,
    "recall": 0.25,
    "compress": 0.15,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_videos(video_ids: List[str], *, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    vids = sorted(video_ids)
    rng.shuffle(vids)
    n = len(vids)
    n_warmup = round(n * 0.60)
    n_dagger = round(n * 0.20)
    return {
        "warmup": sorted(vids[:n_warmup]),
        "dagger": sorted(vids[n_warmup:n_warmup + n_dagger]),
        "direct": sorted(vids[n_warmup + n_dagger:]),
    }


def count_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = Counter(r.get("sample_type", "?") for r in rows)
    by_ver = Counter(bool((r.get("verification") or {}).get("passed", True)) for r in rows)
    return {
        "n": len(rows),
        "videos": len({r.get("video_id") for r in rows}),
        "by_type": dict(sorted(by_type.items())),
        "verification_passed": dict(sorted((str(k), v) for k, v in by_ver.items())),
    }


def balance_rows(
    rows: List[Dict[str, Any]],
    *,
    target_total: int,
    ratios: Dict[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("sample_type", "?")].append(row)

    out: List[Dict[str, Any]] = []
    types = list(ratios)
    assigned = 0
    for st in types:
        pool = by_type.get(st, [])
        if not pool:
            continue
        if st == types[-1]:
            k = target_total - assigned
        else:
            k = int(round(target_total * ratios[st]))
            assigned += k
        if k <= 0:
            continue
        if len(pool) >= k:
            chosen = rng.sample(pool, k)
        else:
            chosen = [rng.choice(pool) for _ in range(k)]
        for i, row in enumerate(chosen):
            item = dict(row)
            meta = dict(item.get("metadata") or {})
            meta["ratio_experiment"] = {
                "source_sample_type": st,
                "resample_index": i,
                "source_pool_size": len(pool),
            }
            item["metadata"] = meta
            out.append(item)

    rng.shuffle(out)
    return out


def copy_eval_messages(source_rendered: Path, target_dir: Path) -> None:
    for name in ("val_messages.jsonl", "test_messages.jsonl"):
        src = source_rendered / name
        dst = target_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-root", default="data/agent_v5/batch1")
    p.add_argument("--source-rendered", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--warmup-total", type=int, default=8192)
    p.add_argument("--stage-total", type=int, default=4096)
    p.add_argument(
        "--ratios",
        default="silent=0.35,response=0.25,recall=0.25,compress=0.15",
        help="Comma-separated sample_type=ratio list.",
    )
    args = p.parse_args()

    batch_root = Path(args.batch_root)
    source_rendered = Path(args.source_rendered) if args.source_rendered else batch_root / "rendered" / "video_meta_all"
    out_root = Path(args.out_root) if args.out_root else batch_root / "experiments" / "ratio_dagger_v1"
    ratios = {
        k.strip(): float(v)
        for part in args.ratios.split(",")
        for k, v in [part.split("=", 1)]
    }
    s = sum(ratios.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {s}")

    messages = read_jsonl(source_rendered / "train_sft_messages.jsonl")
    trajectories = read_jsonl(batch_root / "final" / "train_sft_trajectories.jsonl")
    traj_by_video = {tr["video_id"]: tr for tr in trajectories}
    message_videos = {r["video_id"] for r in messages}
    splits = split_videos(sorted(message_videos), seed=args.seed)

    manifest: Dict[str, Any] = {
        "batch_root": str(batch_root),
        "source_rendered": str(source_rendered),
        "out_root": str(out_root),
        "seed": args.seed,
        "ratios": ratios,
        "splits": {k: {"n_videos": len(v), "videos": v} for k, v in splits.items()},
        "outputs": {},
    }

    for stage, videos in splits.items():
        stage_dir = out_root / stage
        video_set = set(videos)
        stage_rows = [r for r in messages if r.get("video_id") in video_set]
        stage_traj = [traj_by_video[v] for v in videos if v in traj_by_video]
        target_total = args.warmup_total if stage == "warmup" else args.stage_total
        balanced = balance_rows(
            stage_rows,
            target_total=target_total,
            ratios=ratios,
            seed=args.seed + {"warmup": 11, "dagger": 23, "direct": 37}[stage],
        )

        write_jsonl(stage_dir / "train_sft_messages_natural.jsonl", stage_rows)
        write_jsonl(stage_dir / "train_sft_messages.jsonl", balanced)
        write_jsonl(stage_dir / "train_sft_trajectories.jsonl", stage_traj)
        copy_eval_messages(source_rendered, stage_dir)

        manifest["outputs"][stage] = {
            "dir": str(stage_dir),
            "natural": count_rows(stage_rows),
            "balanced": count_rows(balanced),
            "trajectories": len(stage_traj),
        }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))
    print(f"Wrote {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
