"""Rebuild final/pass4/pass5 inputs from cached pass3c samples.

This utility is for pass4/pass5-only iterations where pass3 teacher outputs
should be reused even if stage-version checks consider earlier caches stale.
It reads raw samples_3c plus pass1/pass2/task-card caches directly, renders and
tags samples, applies the current split policy, and writes final/*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from scripts.agent_data.pass3e_verify import tag_samples
from scripts.agent_data.pipeline import (
    _balanced_video_buckets,
    assign_phase,
)
from scripts.agent_data.render_samples import render_video_samples


logger = logging.getLogger(__name__)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_videos(path: Path, limit: int = 0) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_samples(path: Path) -> List[Dict]:
    obj = _read_json(path)
    if isinstance(obj, dict):
        obj = obj.get("samples") or []
    if not isinstance(obj, list):
        raise ValueError(f"unsupported sample cache shape: {path}")
    return obj


def _frame_paths(data_dir: Path, video_id: str) -> List[str]:
    frame_dir = data_dir / "frames" / video_id
    if not frame_dir.exists():
        return []
    return [str(p) for p in sorted(frame_dir.glob("*.jpg"))]


def _thin_to_first_trajectory(samples: List[Dict]) -> List[Dict]:
    by_vid: Dict[str, List[Dict]] = {}
    for sample in samples:
        by_vid.setdefault(str(sample.get("video_id") or ""), []).append(sample)
    kept: List[Dict] = []
    for _vid, vsamps in by_vid.items():
        traj_ids = sorted({str(s.get("trajectory_id") or "0") for s in vsamps})
        first_traj = traj_ids[0] if traj_ids else "0"
        for sample in vsamps:
            tid = str(sample.get("trajectory_id") or "")
            sample_type = str(sample.get("sample_type") or sample.get("action") or "")
            if (
                not tid
                or tid == first_traj
                or sample.get("sequence_type") == "base"
                or sample_type in {"recall", "compress"}
            ):
                kept.append(sample)
    return kept


def rebuild(data_dir: Path, videos_jsonl: Path, *, num_videos: int, seed: int) -> Dict:
    videos = _read_videos(videos_jsonl, num_videos)
    if not videos:
        raise RuntimeError(f"no videos loaded from {videos_jsonl}")

    rendered_samples: List[Dict] = []
    raw_video_count = 0
    for video in videos:
        vid = str(video.get("video_id") or "")
        if not vid:
            continue
        sample_path = data_dir / "samples_3c" / f"{vid}.json"
        rollout_path = data_dir / "rollout" / f"{vid}.json"
        evidence_path = data_dir / "evidence_1b" / f"{vid}.json"
        cards_path = data_dir / "task_cards" / f"{vid}.json"
        if not sample_path.exists():
            continue
        missing = [
            str(p) for p in (rollout_path, evidence_path, cards_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(f"{vid}: missing required cache(s): {missing}")

        raw_samples = _load_samples(sample_path)
        for sample in raw_samples:
            sample.setdefault("video_id", vid)
            sample.setdefault("video_path", video.get("video_path", ""))
        rollout = _read_json(rollout_path)
        cards = _read_json(cards_path)
        cards_map = {str(card.get("card_id")): card for card in cards}
        rendered = render_video_samples(
            raw_samples,
            rollout,
            str(video.get("video_path") or ""),
            vid,
            cards_map,
            all_frame_paths=_frame_paths(data_dir, vid),
        )
        rendered_samples.extend(rendered)
        raw_video_count += 1

    if not rendered_samples:
        raise RuntimeError("render produced no samples")
    logger.info("Rendered %d samples from %d videos", len(rendered_samples), raw_video_count)

    evidence_map = {
        str(video.get("video_id")): _read_json(data_dir / "evidence_1b" / f"{video.get('video_id')}.json")
        for video in videos
        if video.get("video_id") and (data_dir / "evidence_1b" / f"{video.get('video_id')}.json").exists()
    }
    tagged_samples, stats = tag_samples(rendered_samples, evidence_map=evidence_map)
    if not tagged_samples:
        raise RuntimeError("verification/tagging produced no samples")
    logger.info(
        "Verification: %d/%d passed (%.1f%%)",
        stats.get("passed", 0),
        stats.get("total", 0),
        float(stats.get("pass_rate", 0.0)) * 100.0,
    )

    verified_dir = data_dir / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    for old in verified_dir.glob("*.json"):
        old.unlink()
    verified_by_vid: Dict[str, List[Dict]] = {}
    for sample in tagged_samples:
        verified_by_vid.setdefault(str(sample.get("video_id") or "unknown"), []).append(sample)
    for vid, samples in verified_by_vid.items():
        (verified_dir / f"{vid}.json").write_text(
            json.dumps(
                {"samples": samples, "stats": {"video_id": vid, "count": len(samples)}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for i, sample in enumerate(tagged_samples):
        sample["sample_id"] = f"{sample.get('video_id', 'unk')}_{sample.get('action', 'unk')}_{i}"
        sample["phase"] = assign_phase(sample)

    video_ids = sorted({str(s.get("video_id") or "") for s in tagged_samples if s.get("video_id")})
    split_buckets, split_balance_audit = _balanced_video_buckets(
        video_ids,
        tagged_samples,
        seed=seed,
    )
    train_vids = split_buckets["train_sft"] | split_buckets["train_rl"]
    train_samples = [s for s in tagged_samples if s.get("video_id") in train_vids]
    train_sft_samples = [s for s in tagged_samples if s.get("video_id") in split_buckets["train_sft"]]
    train_rl_samples = [s for s in tagged_samples if s.get("video_id") in split_buckets["train_rl"]]
    val_samples = _thin_to_first_trajectory([
        s for s in tagged_samples if s.get("video_id") in split_buckets["val"]
    ])
    test_samples = _thin_to_first_trajectory([
        s for s in tagged_samples if s.get("video_id") in split_buckets["test"]
    ])

    final_dir = data_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for old in final_dir.glob("*"):
        if old.is_file():
            old.unlink()

    split_counts = {
        "train": _write_jsonl(final_dir / "train.jsonl", train_samples),
        "train_sft": _write_jsonl(final_dir / "train_sft.jsonl", train_sft_samples),
        "train_rl": _write_jsonl(final_dir / "train_rl.jsonl", train_rl_samples),
        "val": _write_jsonl(final_dir / "val.jsonl", val_samples),
        "test": _write_jsonl(final_dir / "test.jsonl", test_samples),
    }
    logger.info("Final split counts: %s", split_counts)

    family_counts = Counter(
        (sample.get("metadata") or {}).get("family", "")
        for sample in tagged_samples
        if (sample.get("metadata") or {}).get("family")
    )
    seq_counts = Counter(str(sample.get("sequence_type") or "") for sample in tagged_samples)
    base_role_counts = Counter(str(sample.get("base_role") or "") for sample in tagged_samples)
    phase_counts = Counter(str(sample.get("phase") or "") for sample in train_samples)
    pipeline_stats = dict(stats)
    pipeline_stats.update({
        "train_count": len(train_samples),
        "train_sft_count": len(train_sft_samples),
        "train_rl_count": len(train_rl_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "video_counts": {name: len(vids) for name, vids in split_buckets.items()},
        "phase_counts": dict(phase_counts),
        "legacy_phase_files_emitted": False,
        "split_by_video": True,
        "split_balance_audit": split_balance_audit,
        "global_family_distribution": dict(family_counts),
        "global_sequence_type_distribution": dict(seq_counts),
        "global_base_role_distribution": dict(base_role_counts),
        "rebuilt_from": "samples_3c_cache",
    })
    (final_dir / "pipeline_stats.json").write_text(
        json.dumps(pipeline_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (final_dir / "_version").write_text("rebuilt_from_samples_cache\n", encoding="utf-8")
    return pipeline_stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--videos-jsonl", type=Path, required=True)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = rebuild(
        args.data_dir,
        args.videos_jsonl,
        num_videos=args.num_videos,
        seed=args.seed,
    )
    print(json.dumps({
        "train_sft_count": stats.get("train_sft_count"),
        "train_rl_count": stats.get("train_rl_count"),
        "val_count": stats.get("val_count"),
        "test_count": stats.get("test_count"),
        "split_balance_audit": stats.get("split_balance_audit"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
