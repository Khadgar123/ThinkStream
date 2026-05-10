"""Create a non-destructive SFT/RL/eval/test scheme from trajectory banks.

The generated directory is a standalone data root for training scripts:

- final/train_sft_trajectories.jsonl
- final/train_rl_trajectories.jsonl
- final/val_trajectories.jsonl
- final/test_trajectories.jsonl
- rendered/video_meta_standard_query_last/{train_sft,val,test}_messages.jsonl
- rendered/video_meta_standard_query_last/{train_rl,val}_rl_multi_q.parquet
- reports/distribution.{json,md}

No source files are modified. The split is trajectory/video-disjoint and uses
the already-generated teacher trajectories, so it can re-balance data without
re-running pass1/pass2/pass3.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_FRAME_PROTOCOL = "video_meta"
CANONICAL_RENDER_LAYOUT = "standard_query_last"
CANONICAL_RENDER_DIRNAME = f"{CANONICAL_FRAME_PROTOCOL}_{CANONICAL_RENDER_LAYOUT}"
SUPPORTED_RENDER_LAYOUTS = ["standard_query_last"]

DEFAULT_SPLIT_COUNTS = {
    "train_sft": 150,
    "dagger_source": 0,
    "train_rl": 175,
    "val": 50,
    "test": 50,
}

SPLIT_ORDER = ("train_sft", "dagger_source", "train_rl", "val", "test")
BALANCED_CANDIDATE_POOL = 512


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _video_id(obj: dict) -> str:
    return str(obj.get("video_id") or obj.get("id") or obj.get("video") or "")


def _resolve_source_from_bank(bank: Path) -> Path:
    manifest = bank / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"trajectory bank manifest not found: {manifest}")
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    source = Path(obj.get("source_data_dir") or "")
    if not source:
        raise ValueError(f"missing source_data_dir in {manifest}")
    return source


def _resolve_bank_from_batch(batch: Path) -> Path:
    bank = batch / "trajectory_bank"
    if not bank.exists():
        raise FileNotFoundError(f"trajectory bank not found under batch root: {bank}")
    return bank


def _relative_symlink(src: Path, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel, target_is_directory=src.is_dir())


def _link_if_exists(src: Path, dst: Path, *, force: bool) -> None:
    if src.exists():
        _relative_symlink(src, dst, force=force)


def _source_key(source: Path) -> str:
    name = source.name or "source"
    parent = source.parent.name
    return name if not parent else f"{parent}_{name}"


def _load_video_stats(bank: Path) -> List[Dict[str, Any]]:
    path = bank / "video_stats.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    source = _resolve_source_from_bank(bank)
    rows = list(_read_jsonl(path))
    if not rows:
        raise ValueError(f"empty video stats: {path}")
    traj_stats = bank / "trajectory_stats.jsonl"
    if traj_stats.exists():
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for row in _read_jsonl(traj_stats):
            by_video.setdefault(_video_id(row), []).append(row)
        for row in rows:
            sample_types = Counter()
            gold_actions = Counter()
            question_types = Counter()
            answer_forms = Counter()
            categories = Counter()
            skills = Counter()
            for traj in by_video.get(_video_id(row), []):
                sample_types.update(traj.get("sample_types") or {})
                gold_actions.update(traj.get("gold_actions") or {})
                question_types.update(traj.get("question_types") or {})
                answer_forms.update(traj.get("answer_forms") or {})
                categories.update(traj.get("categories") or {})
                skills.update(traj.get("skills") or {})
            if sample_types:
                row["sample_types"] = dict(sample_types)
            if gold_actions:
                row["gold_actions"] = dict(gold_actions)
            if question_types:
                row["question_types"] = dict(question_types)
            if answer_forms:
                row["answer_forms"] = dict(answer_forms)
            if categories:
                row["categories"] = dict(categories)
            if skills:
                row["skills"] = dict(skills)
    for row in rows:
        row["_bank"] = str(bank)
        row["_source"] = str(source)
        row["_source_key"] = _source_key(source)
    return rows


def _load_multi_video_stats(
    banks: Sequence[Path],
    *,
    dedupe_video_id: str = "error",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    seen_bank: Dict[str, str] = {}
    for bank in banks:
        for row in _load_video_stats(bank):
            vid = _video_id(row)
            if not vid:
                continue
            if vid in seen:
                if dedupe_video_id == "error":
                    raise ValueError(
                        f"duplicate video_id across banks: {vid} in {seen_bank[vid]} and {bank}"
                    )
                if dedupe_video_id == "first":
                    continue
                if dedupe_video_id == "last":
                    rows[seen[vid]] = row
                    seen_bank[vid] = str(bank)
                    continue
                raise ValueError(f"unknown dedupe_video_id policy: {dedupe_video_id}")
            seen[vid] = len(rows)
            seen_bank[vid] = str(bank)
            rows.append(row)
    if not rows:
        raise ValueError("no video stats loaded")
    return rows


def _trajectory_map_from_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    missing: List[str] = []
    for row in rows:
        vid = _video_id(row)
        paths = row.get("trajectory_paths") or []
        if not paths:
            missing.append(vid)
            continue
        # The exported bank currently stores one trajectory per video. Keep the
        # first path so split counts remain video-level and deterministic.
        path = Path(paths[0]).expanduser()
        if not path.is_absolute():
            path = Path(str(row.get("_bank") or ".")).parent / path
        if not path.exists():
            missing.append(f"{vid}:{path}")
            continue
        traj = json.loads(path.read_text(encoding="utf-8"))
        out[vid] = traj
    if missing:
        raise ValueError(f"missing trajectory bank files: {missing[:5]}")
    return out


def _weighted_take(
    rows: Sequence[Dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    mode: str,
) -> List[Dict[str, Any]]:
    pool = list(rows)
    selected: List[Dict[str, Any]] = []
    if count > len(pool):
        raise ValueError(f"requested {count} rows, only {len(pool)} available")

    def weight(row: Dict[str, Any]) -> float:
        recall = float(row.get("n_recall_calls") or 0)
        compress = float(row.get("n_compress_events") or 0)
        questions = float(row.get("n_questions") or 0)
        if mode == "train_sft":
            return 1.0 + 1.6 * recall + 0.8 * compress + 0.1 * questions
        if mode == "train_rl":
            return 1.0 + 1.2 * recall + 1.0 * compress + 0.1 * questions
        return 1.0

    for _ in range(count):
        weights = [max(0.001, weight(r)) for r in pool]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        idx = len(pool) - 1
        for i, w in enumerate(weights):
            acc += w
            if acc >= pick:
                idx = i
                break
        selected.append(pool.pop(idx))
    return selected


def _render_dir_name(protocol: str, render_layout: str) -> str:
    if protocol == CANONICAL_FRAME_PROTOCOL and render_layout == CANONICAL_RENDER_LAYOUT:
        return CANONICAL_RENDER_DIRNAME
    return f"{protocol}_{render_layout}"


def _feature_vector(row: Dict[str, Any]) -> Counter:
    """Video-level distribution features used for split balancing."""
    c = Counter()
    for key, prefix in (
        ("families", "family"),
        ("question_types", "question_type"),
        ("answer_forms", "answer_form"),
        ("categories", "category"),
        ("skills", "skill"),
        ("sample_types", "sample_type"),
        ("gold_actions", "gold_action"),
        ("availability", "availability"),
    ):
        values = Counter(row.get(key) or {})
        total = sum(values.values()) or 1
        for name, value in values.items():
            if name:
                c[f"{prefix}:{name}"] = float(value) / total
    n_questions = max(1.0, float(row.get("n_questions") or 0))
    recall = float(row.get("n_recall_calls") or 0)
    compress = float(row.get("n_compress_events") or 0)
    c["has_recall"] = 1.0 if recall > 0 else 0.0
    c["has_compress"] = 1.0 if compress > 0 else 0.0
    c["recall_per_question"] = min(2.0, recall / n_questions)
    c["compress_per_question"] = min(2.0, compress / n_questions)
    c["questions_per_video"] = min(16.0, n_questions) / 16.0
    return c


def _mean_feature_vector(rows: Sequence[Dict[str, Any]]) -> Counter:
    out = Counter()
    if not rows:
        return out
    for row in rows:
        out.update(_feature_vector(row))
    scale = 1.0 / float(len(rows))
    for key in list(out):
        out[key] *= scale
    return out


def _feature_distance(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    total = 0.0
    for key in keys:
        if key.startswith("family:"):
            weight = 2.0
        elif key.startswith(("question_type:", "sample_type:", "gold_action:")):
            weight = 1.5
        else:
            weight = 1.0
        total += weight * abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0)))
    return total


def _balanced_take(
    rows: Sequence[Dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    target: Counter,
    mode: str,
) -> List[Dict[str, Any]]:
    """Greedy subset selection that keeps family/recall/compress ratios close."""
    pool = list(rows)
    pool_features = [_feature_vector(row) for row in pool]
    selected: List[Dict[str, Any]] = []
    if count > len(pool):
        raise ValueError(f"requested {count} rows, only {len(pool)} available")

    running = Counter()
    for _ in range(count):
        best_idx = 0
        best_score = float("inf")
        denom = float(len(selected) + 1)
        if len(pool) > BALANCED_CANDIDATE_POOL:
            candidate_indices = rng.sample(range(len(pool)), BALANCED_CANDIDATE_POOL)
        else:
            candidate_indices = range(len(pool))
        for idx in candidate_indices:
            row = pool[idx]
            cand = Counter(running)
            cand.update(pool_features[idx])
            avg = Counter({k: v / denom for k, v in cand.items()})
            score = _feature_distance(avg, target)
            score += rng.random() * 1e-6
            if score < best_score:
                best_idx = idx
                best_score = score
        chosen = pool.pop(best_idx)
        chosen_features = pool_features.pop(best_idx)
        selected.append(chosen)
        running.update(chosen_features)
    return selected


def _build_splits(
    rows: List[Dict[str, Any]],
    *,
    counts: Dict[str, int],
    seed: int,
    train_allocation: str = "balanced",
) -> Dict[str, List[Dict[str, Any]]]:
    need = sum(counts.values())
    if need > len(rows):
        raise ValueError(f"requested {need} videos, bank has {len(rows)}")
    rng = random.Random(seed)
    remaining = list(rows)
    rng.shuffle(remaining)

    splits: Dict[str, List[Dict[str, Any]]] = {}
    target = _mean_feature_vector(rows)

    if train_allocation == "balanced":
        # Holdout first keeps eval/test close to the global distribution, then
        # train splits are filled with the same target plus a slight value
        # preference for recall/compress-heavy trajectories.
        for name in ("test", "val", "dagger_source", "train_rl", "train_sft"):
            if counts.get(name, 0) <= 0:
                splits[name] = []
                continue
            selected = _balanced_take(
                remaining,
                counts[name],
                rng=rng,
                target=target,
                mode=name,
            )
            selected_ids = {_video_id(r) for r in selected}
            remaining = [r for r in remaining if _video_id(r) not in selected_ids]
            splits[name] = selected
        return splits

    # Holdout first keeps eval/test closest to the global distribution.
    for name in ("test", "val"):
        if counts.get(name, 0) <= 0:
            splits[name] = []
            continue
        selected = _weighted_take(
            remaining,
            counts[name],
            rng=rng,
            mode=name,
        )
        selected_ids = {_video_id(r) for r in selected}
        remaining = [r for r in remaining if _video_id(r) not in selected_ids]
        splits[name] = selected

    if train_allocation == "weighted":
        # Historical behavior: each train split is sampled independently with
        # its own weight function. This makes SFT high-value but can
        # leave RL with fewer recall/compress trajectories.
        order = ("dagger_source", "train_sft", "train_rl")
        for name in order:
            if counts.get(name, 0) <= 0:
                splits[name] = []
                continue
            selected = _weighted_take(
                remaining,
                counts[name],
                rng=rng,
                mode=name,
            )
            selected_ids = {_video_id(r) for r in selected}
            remaining = [r for r in remaining if _video_id(r) not in selected_ids]
            splits[name] = selected
        return splits

    if train_allocation != "stratified":
        raise ValueError(f"unknown train_allocation={train_allocation!r}")

    def score(row: Dict[str, Any]) -> float:
        return (
            2.4 * float(row.get("n_recall_calls") or 0)
            + 1.2 * float(row.get("n_compress_events") or 0)
            + 0.15 * float(row.get("n_questions") or 0)
        )

    train_names = tuple(
        name for name in ("train_rl", "dagger_source", "train_sft")
        if counts.get(name, 0) > 0
    )
    if not train_names:
        return splits
    sorted_remaining = sorted(
        remaining,
        key=lambda r: (score(r), int(r.get("n_samples") or 0)),
        reverse=True,
    )
    cycle: List[str] = []
    count_gcd = reduce(gcd, (counts[name] for name in train_names))
    count_gcd = max(1, count_gcd)
    for name in train_names:
        reps = max(1, counts[name] // count_gcd)
        cycle.extend([name] * reps)

    alloc = {name: [] for name in train_names}
    cursor = 0
    for row in sorted_remaining:
        placed = False
        for _ in range(len(cycle)):
            name = cycle[cursor % len(cycle)]
            cursor += 1
            if len(alloc[name]) < counts[name]:
                alloc[name].append(row)
                placed = True
                break
        if not placed:
            break
    for name in train_names:
        if len(alloc[name]) != counts[name]:
            raise ValueError(
                f"stratified allocation filled {name} with {len(alloc[name])}, "
                f"expected {counts[name]}"
            )
        splits[name] = alloc[name]
    return splits


def _counter_sum(rows: Sequence[Dict[str, Any]], key: str) -> Counter:
    c = Counter()
    for row in rows:
        c.update(row.get(key) or {})
    return c


def _split_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_types = _counter_sum(rows, "sample_types")
    gold_actions = _counter_sum(rows, "gold_actions")
    families = _counter_sum(rows, "families")
    availability = _counter_sum(rows, "availability")
    question_types = _counter_sum(rows, "question_types")
    answer_forms = _counter_sum(rows, "answer_forms")
    categories = _counter_sum(rows, "categories")
    skills = _counter_sum(rows, "skills")
    sources = Counter(str(r.get("_source_key") or "unknown") for r in rows)
    n_samples = sum(int(r.get("n_samples") or 0) for r in rows)
    n_questions = sum(int(r.get("n_questions") or 0) for r in rows)
    n_recall = sum(int(r.get("n_recall_calls") or 0) for r in rows)
    n_compress = sum(int(r.get("n_compress_events") or 0) for r in rows)
    return {
        "videos": len(rows),
        "samples": n_samples,
        "questions": n_questions,
        "recall_calls": n_recall,
        "compress_events": n_compress,
        "sample_types": dict(sample_types),
        "gold_actions": dict(gold_actions),
        "sample_type_ratio": {
            k: round(v / n_samples, 6) for k, v in sorted(sample_types.items()) if n_samples
        },
        "families": dict(families),
        "availability": dict(availability),
        "question_types": dict(question_types),
        "answer_forms": dict(answer_forms),
        "categories": dict(categories),
        "skills": dict(skills),
        "sources": dict(sources),
        "recall_per_video": round(n_recall / len(rows), 3) if rows else 0.0,
        "compress_per_video": round(n_compress / len(rows), 3) if rows else 0.0,
        "questions_per_video": round(n_questions / len(rows), 3) if rows else 0.0,
    }


def _manifest_lookup_for_sources(sources: Sequence[Path]) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for source in sources:
        selected_manifest = source / "selected_videos.jsonl"
        if not selected_manifest.exists():
            continue
        skey = _source_key(source)
        for row in _read_jsonl(selected_manifest):
            vid = _video_id(row)
            if vid:
                out[(skey, vid)] = row
    return out


def _link_frames_for_selected(out: Path, rows: Sequence[Dict[str, Any]], *, force: bool) -> None:
    frames_root = out / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        vid = _video_id(row)
        source = Path(str(row.get("_source") or ""))
        if not vid or not source:
            continue
        src = source / "frames" / vid
        if src.exists():
            _relative_symlink(src, frames_root / vid, force=force)


def _link_rollouts_for_selected(out: Path, rows: Sequence[Dict[str, Any]], *, force: bool) -> None:
    rollout_root = out / "rollout"
    rollout_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        vid = _video_id(row)
        source = Path(str(row.get("_source") or ""))
        if not vid or not source:
            continue
        src = source / "rollout" / f"{vid}.json"
        if src.exists():
            _relative_symlink(src, rollout_root / f"{vid}.json", force=force)


def _write_split_manifests(
    *,
    sources: Sequence[Path],
    banks: Sequence[Path],
    out: Path,
    splits: Dict[str, List[Dict[str, Any]]],
    force: bool,
) -> Dict[str, int]:
    by_source_video = _manifest_lookup_for_sources(sources)

    counts = {}
    split_dir = out / "splits"
    all_rows: List[dict] = []
    for split, rows in splits.items():
        manifest_rows = []
        for row in rows:
            vid = _video_id(row)
            source = Path(str(row.get("_source") or ""))
            skey = str(row.get("_source_key") or _source_key(source))
            manifest = dict(by_source_video.get((skey, vid)) or {})
            if not manifest:
                manifest = {"video_id": vid, "video_path": row.get("video_path", "")}
            manifest["source_data_dir"] = str(source)
            manifest["source_key"] = skey
            manifest["trajectory_bank"] = row.get("_bank", "")
            manifest["scheme_split"] = split
            manifest["trajectory_bank_stats"] = {
                "n_questions": row.get("n_questions", 0),
                "n_recall_calls": row.get("n_recall_calls", 0),
                "n_compress_events": row.get("n_compress_events", 0),
            }
            manifest_rows.append(manifest)
            all_rows.append(manifest)
        counts[f"{split}_videos"] = _write_jsonl(split_dir / f"{split}_videos.jsonl", manifest_rows)
    counts["selected_videos"] = _write_jsonl(out / "selected_videos.jsonl", all_rows)
    _link_frames_for_selected(out, [row for rows in splits.values() for row in rows], force=force)
    bank_dir = out / "trajectory_bank_sources"
    for bank in banks:
        source = _resolve_source_from_bank(bank)
        _link_if_exists(bank, bank_dir / _source_key(source), force=force)
    _link_rollouts_for_selected(
        out,
        [row for rows in splits.values() for row in rows],
        force=force,
    )
    return counts


def _write_trajectory_files(
    *,
    out: Path,
    splits: Dict[str, List[Dict[str, Any]]],
    trajectories: Dict[str, dict],
) -> Dict[str, int]:
    final = out / "final"
    counts: Dict[str, int] = {}
    mapping = {
        "train_sft": "train_sft_trajectories.jsonl",
        "dagger_source": "train_sft_dagger_source_trajectories.jsonl",
        "train_rl": "train_rl_trajectories.jsonl",
        "val": "val_trajectories.jsonl",
        "test": "test_trajectories.jsonl",
    }
    for split, filename in mapping.items():
        if split not in splits:
            continue
        rows = []
        missing = []
        for stat in splits[split]:
            vid = _video_id(stat)
            traj = trajectories.get(vid)
            if not traj:
                missing.append(vid)
                continue
            rows.append(traj)
        if missing:
            raise ValueError(f"missing source trajectories for {split}: {missing[:5]}")
        counts[filename] = _write_jsonl(final / filename, rows)

    return counts


def _render_messages(
    *,
    final_dir: Path,
    out: Path,
    protocols: Sequence[str],
    render_layout: str,
    no_balance_sft: bool,
) -> Dict[str, int]:
    from scripts.agent_data_v5.pass5_messages import convert, write_dataset_info

    counts: Dict[str, int] = {}
    for protocol in protocols:
        rendered = out / "rendered" / _render_dir_name(protocol, render_layout)
        rendered.mkdir(parents=True, exist_ok=True)
        split_specs = [
            ("train_sft_trajectories.jsonl", "train_sft_messages.jsonl", True),
            ("val_trajectories.jsonl", "val_messages.jsonl", False),
            ("test_trajectories.jsonl", "test_messages.jsonl", False),
        ]
        for src_name, dst_name, is_sft in split_specs:
            balance = is_sft and not no_balance_sft
            result = convert(
                final_dir / src_name,
                rendered / dst_name,
                is_trajectory=True,
                base_path=Path.cwd(),
                data_dir=out,
                limit=None,
                balance_sft=balance,
                frame_protocol=protocol,
                render_layout=render_layout,
            )
            counts[f"rendered/{rendered.name}/{dst_name}"] = int(result.get("ok", 0))
        write_dataset_info(rendered, ["train_sft", "val", "test"])
        _write_json(
            rendered / "render_manifest.json",
            {
                "generated_by": "make_training_scheme.py",
                "source_final_dir": str(final_dir),
                "output_dir": str(rendered),
                "frame_protocol": protocol,
                "render_layout": render_layout,
                "splits": ["train_sft", "val", "test"],
            },
        )
    return counts


def _build_rl_parquets(
    *,
    final_dir: Path,
    out: Path,
    protocols: Sequence[str],
    render_layout: str,
) -> Dict[str, int]:
    import pandas as pd

    from scripts.agent_data_v5.build_verl_parquet import _iter_rows_multi_q

    counts: Dict[str, int] = {}
    specs = [
        ("train_rl_trajectories.jsonl", "train_rl_multi_q.parquet", False),
        ("val_trajectories.jsonl", "val_rl_multi_q.parquet", False),
        ("test_trajectories.jsonl", "test_rl_multi_q.parquet", False),
        ("train_rl_trajectories.jsonl", "train_rl_multi_q_segment_cache.parquet", True),
        ("val_trajectories.jsonl", "val_rl_multi_q_segment_cache.parquet", True),
        ("test_trajectories.jsonl", "test_rl_multi_q_segment_cache.parquet", True),
    ]
    for protocol in protocols:
        rendered = out / "rendered" / _render_dir_name(protocol, render_layout)
        rendered.mkdir(parents=True, exist_ok=True)
        for src_name, dst_name, include_student_cache in specs:
            rows = list(
                _iter_rows_multi_q(
                    final_dir / src_name,
                    max_questions_per_traj=16,
                    frame_protocol=protocol,
                    render_layout=render_layout,
                    include_student_cache=include_student_cache,
                )
            )
            if not rows:
                raise ValueError(f"no RL rows from {src_name}")
            pd.DataFrame(rows).to_parquet(rendered / dst_name, index=False)
            counts[f"rendered/{rendered.name}/{dst_name}"] = len(rows)
    return counts


def _write_report(
    *,
    out: Path,
    sources: Sequence[Path],
    banks: Sequence[Path],
    seed: int,
    train_allocation: str,
    splits: Dict[str, List[Dict[str, Any]]],
    manifest_counts: Dict[str, int],
    trajectory_counts: Dict[str, int],
    message_counts: Dict[str, int],
    parquet_counts: Dict[str, int],
    protocols: Sequence[str],
    render_layout: str,
) -> Dict[str, Any]:
    global_rows = [row for rows in splits.values() for row in rows]
    split_summaries = {name: _split_summary(rows) for name, rows in splits.items()}
    report = {
        "sources": [str(x) for x in sources],
        "banks": [str(x) for x in banks],
        "out": str(out),
        "seed": seed,
        "train_allocation": train_allocation,
        "protocols": list(protocols),
        "render_layout": render_layout,
        "canonical_render_dir": _render_dir_name(CANONICAL_FRAME_PROTOCOL, render_layout),
        "video_counts": {name: len(rows) for name, rows in splits.items()},
        "global_selected": _split_summary(global_rows),
        "splits": split_summaries,
        "row_counts": {
            "manifests": manifest_counts,
            "trajectories": trajectory_counts,
            "messages": message_counts,
            "parquets": parquet_counts,
        },
        "canonical_paths": {
            "sft_messages": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "train_sft_messages.jsonl")
                for p in protocols
            },
            "dagger_source_trajectories": str(out / "final" / "train_sft_dagger_source_trajectories.jsonl"),
            "dagger_output_messages": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "train_sft_dagger_messages.jsonl")
                for p in protocols
            },
            "rl_train_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "train_rl_multi_q.parquet")
                for p in protocols
            },
            "rl_train_segment_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "train_rl_multi_q_segment_cache.parquet")
                for p in protocols
            },
            "rl_val_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "val_rl_multi_q.parquet")
                for p in protocols
            },
            "rl_val_segment_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "val_rl_multi_q_segment_cache.parquet")
                for p in protocols
            },
            "rl_test_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "test_rl_multi_q.parquet")
                for p in protocols
            },
            "rl_test_segment_parquet": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "test_rl_multi_q_segment_cache.parquet")
                for p in protocols
            },
            "eval_messages": {
                p: str(out / "rendered" / _render_dir_name(p, render_layout) / "val_messages.jsonl")
                for p in protocols
            },
        },
        "suggested_loss_ratios": {
            "teacher_sft": "silent=0.35,response=0.25,recall=0.25,compress=0.15",
        },
    }
    reports = out / "reports"
    _write_json(reports / "distribution.json", report)
    _write_json(out / "scheme.json", report)

    lines = [
        "# ThinkStream Training Scheme",
        "",
        f"- sources: `{', '.join(str(x) for x in sources)}`",
        f"- banks: `{', '.join(str(x) for x in banks)}`",
        f"- output: `{out}`",
        f"- seed: `{seed}`",
        f"- train allocation: `{train_allocation}`",
        f"- protocols: `{', '.join(protocols)}`",
        f"- render layout: `{render_layout}`",
        "",
        "## Split Summary",
        "",
        "| split | videos | samples | silent | response | recall | compress | recall/video | compress/video |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("train_sft", "dagger_source", "train_rl", "val", "test"):
        if name not in split_summaries:
            continue
        s = split_summaries[name]
        st = s["sample_types"]
        lines.append(
            "| {name} | {videos} | {samples} | {silent} | {response} | {recall} | {compress} | {rpv} | {cpv} |".format(
                name=name,
                videos=s["videos"],
                samples=s["samples"],
                silent=st.get("silent", 0),
                response=st.get("response", 0),
                recall=st.get("recall", 0),
                compress=st.get("compress", 0),
                rpv=s["recall_per_video"],
                cpv=s["compress_per_video"],
            )
        )
    render_dir = _render_dir_name(CANONICAL_FRAME_PROTOCOL, render_layout)
    lines.extend([
        "",
        "## Trainable Files",
        "",
        f"- SFT: `rendered/{render_dir}/train_sft_messages.jsonl`",
        "- DAgger source: `final/train_sft_dagger_source_trajectories.jsonl`",
        f"- DAgger output target: `rendered/{render_dir}/train_sft_dagger_messages.jsonl`",
        f"- Eval: `rendered/{render_dir}/val_messages.jsonl`, `rendered/{render_dir}/test_messages.jsonl`",
        f"- RL full-video: `rendered/{render_dir}/train_rl_multi_q.parquet`",
        f"- RL/agent eval: `rendered/{render_dir}/val_rl_multi_q.parquet`, `rendered/{render_dir}/test_rl_multi_q.parquet`",
        f"- RL segment: `rendered/{render_dir}/train_rl_multi_q_segment_cache.parquet`",
        "- RL source trajectories: `final/train_rl_trajectories.jsonl`",
        "",
    ])
    (reports / "distribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="", help="Batch root. Inferred from --bank when omitted.")
    parser.add_argument("--bank", default="data/agent_v5/batch3/trajectory_bank")
    parser.add_argument(
        "--banks",
        nargs="+",
        default=None,
        help="One or more trajectory banks. When set, --source is ignored.",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="One or more batch roots; each must contain trajectory_bank/.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--sft-videos", type=int, default=DEFAULT_SPLIT_COUNTS["train_sft"])
    parser.add_argument("--dagger-videos", type=int, default=DEFAULT_SPLIT_COUNTS["dagger_source"])
    parser.add_argument("--rl-videos", type=int, default=DEFAULT_SPLIT_COUNTS["train_rl"])
    parser.add_argument("--val-videos", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test-videos", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument(
        "--frame-protocols",
        nargs="+",
        default=[CANONICAL_FRAME_PROTOCOL],
        choices=[CANONICAL_FRAME_PROTOCOL],
    )
    parser.add_argument(
        "--render-layout",
        default=CANONICAL_RENDER_LAYOUT,
        choices=SUPPORTED_RENDER_LAYOUTS,
        help=(
            "Prompt layout for rendered messages and RL parquets. "
            "standard_query_last is the only supported training layout."
        ),
    )
    parser.add_argument("--no-render", action="store_true", help="Only write split trajectories/reports.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip RL parquet generation.")
    parser.add_argument("--no-balance-sft", action="store_true", help="Disable pass5 SFT silent downsampling.")
    parser.add_argument(
        "--train-allocation",
        choices=["balanced", "weighted", "stratified"],
        default="balanced",
        help=(
            "balanced keeps family/recall/compress ratios close across "
            "SFT/RL/eval/test; weighted and stratified are archived."
        ),
    )
    parser.add_argument(
        "--dedupe-video-id",
        choices=["error", "first", "last"],
        default="error",
        help=(
            "Policy for duplicate video_id across banks. Default preserves the "
            "previous fail-fast behavior; use last to prefer later --banks entries."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batches and args.banks:
        raise ValueError("pass either --batches or --banks, not both")

    if args.batches:
        raw_banks = []
        for raw_batch in args.batches:
            batch = Path(raw_batch).expanduser()
            if not batch.is_absolute():
                batch = Path.cwd() / batch
            raw_banks.append(str(_resolve_bank_from_batch(batch)))
    else:
        raw_banks = args.banks if args.banks else [args.bank]
    banks: List[Path] = []
    for raw in raw_banks:
        bank = Path(raw).expanduser()
        if not bank.is_absolute():
            bank = Path.cwd() / bank
        banks.append(bank)
    if args.batches or args.banks or len(banks) > 1:
        sources = [_resolve_source_from_bank(bank) for bank in banks]
    else:
        source = Path(args.source).expanduser() if args.source else _resolve_source_from_bank(banks[0])
        if not source.is_absolute():
            source = Path.cwd() / source
        sources = [source]
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"{out} already exists; pass --force to update")
    out.mkdir(parents=True, exist_ok=True)

    counts = {
        "train_sft": args.sft_videos,
        "dagger_source": args.dagger_videos,
        "train_rl": args.rl_videos,
        "val": args.val_videos,
        "test": args.test_videos,
    }
    stats_rows = _load_multi_video_stats(
        banks,
        dedupe_video_id=args.dedupe_video_id,
    )
    splits = _build_splits(
        stats_rows,
        counts=counts,
        seed=args.seed,
        train_allocation=args.train_allocation,
    )
    trajectories = _trajectory_map_from_stats([row for rows in splits.values() for row in rows])
    manifest_counts = _write_split_manifests(
        sources=sources,
        banks=banks,
        out=out,
        splits=splits,
        force=args.force,
    )
    trajectory_counts = _write_trajectory_files(
        out=out,
        splits=splits,
        trajectories=trajectories,
    )

    protocols = list(dict.fromkeys(args.frame_protocols))
    message_counts: Dict[str, int] = {}
    parquet_counts: Dict[str, int] = {}
    if not args.no_render:
        message_counts = _render_messages(
            final_dir=out / "final",
            out=out,
            protocols=protocols,
            render_layout=args.render_layout,
            no_balance_sft=args.no_balance_sft,
        )
    if not args.no_parquet:
        parquet_counts = _build_rl_parquets(
            final_dir=out / "final",
            out=out,
            protocols=protocols,
            render_layout=args.render_layout,
        )

    report = _write_report(
        out=out,
        sources=sources,
        banks=banks,
        seed=args.seed,
        train_allocation=args.train_allocation,
        splits=splits,
        manifest_counts=manifest_counts,
        trajectory_counts=trajectory_counts,
        message_counts=message_counts,
        parquet_counts=parquet_counts,
        protocols=protocols,
        render_layout=args.render_layout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
