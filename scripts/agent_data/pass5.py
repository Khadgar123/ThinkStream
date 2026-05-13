"""Pass5 — render pass4 trajectory records as multi-turn JSONL.

CLI driver around ``pass5_splitter.render_trajectory_record_to_rows``.
Reads pass4 ``final/<split>_trajectories.jsonl`` files and writes rows
(one per sub-trajectory after splitting on compress events) to
``final/<split>_trajectory.jsonl`` (or a user-specified path).

Design intent:
- Independent of legacy pass5_messages.py (different output schema)
- Pure function pipeline: read JSONL → render → write JSONL
- No model calls, no GPU
- Idempotent: re-running overwrites cleanly

Usage:
    python -m scripts.agent_data.pass5 \\
        --input data/agent_v5/final/train_trajectories.jsonl \\
        --output data/agent_v5/final/train_trajectory.jsonl \\
        --frames-root data/agent_v5/frames

    # Or process all splits in a directory:
    python -m scripts.agent_data.pass5 \\
        --input-dir data/agent_v5/final \\
        --output-dir data/agent_v5/final_trajectory \\
        --frames-root data/agent_v5/frames
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.agent_data.pass5_splitter import (  # noqa: E402
    render_trajectory_record_to_rows,
)
from thinkstream.data.agent_protocol import (  # noqa: E402
    chunk_frame_indices,
    project_frame_filename,
)


logger = logging.getLogger("pass5")


# ---------------------------------------------------------------------------
# Frame resolver
# ---------------------------------------------------------------------------

# Matches the extracted frame layout used by pass1a/RL/eval: 2 frames per
# chunk, sequentially numbered starting at chunk_idx * FRAMES_PER_CHUNK + 1.
FRAMES_PER_CHUNK = 2
FRAME_NAME_PATTERN = "frame_{frame_idx:06d}.jpg"


@dataclass
class FrameResolverConfig:
    """How to map (video_id, chunk_idx) → list of frame paths."""
    frames_root: Optional[Path] = None      # e.g. data/agent_v5/frames
    frames_per_chunk: int = FRAMES_PER_CHUNK
    pattern: str = FRAME_NAME_PATTERN
    # When True, returned frame paths are absolute (prefixed with frames_root
    # if given). When False, paths are relative to frames_root so downstream
    # tokenisation/dataloading can resolve them.
    absolute: bool = False


def _frame_name_for_source_index(frame_index: int, pattern: str) -> str:
    """Map a zero-based source frame index to the configured JPEG name."""
    if pattern == FRAME_NAME_PATTERN:
        return project_frame_filename(frame_index)
    return pattern.format(frame_idx=int(frame_index) + 1)


def make_frame_resolver(
    video_id: str,
    config: FrameResolverConfig,
) -> Callable[[int], List[str]]:
    """Build a per-video frame resolver matching the v5 frame layout."""
    if config.frames_root is None:
        # Pure pattern: caller is responsible for resolving the prefix later.
        def _resolve_pattern_only(chunk_idx: int) -> List[str]:
            return [
                _frame_name_for_source_index(idx, config.pattern)
                for idx in chunk_frame_indices(chunk_idx, config.frames_per_chunk)
            ]
        return _resolve_pattern_only

    root = config.frames_root / video_id

    def _resolve(chunk_idx: int) -> List[str]:
        paths = []
        for idx in chunk_frame_indices(chunk_idx, config.frames_per_chunk):
            name = _frame_name_for_source_index(idx, config.pattern)
            full = root / name
            if config.absolute:
                paths.append(str(full))
            else:
                # Relative to frames_root (matches legacy pass5 convention).
                paths.append(str(Path(video_id) / name))
        return paths
    return _resolve


def make_sample_aware_resolver(
    video_id: str,
    config: FrameResolverConfig,
    samples_by_chunk: Optional[Dict[int, Dict]] = None,
) -> Callable[[int], List[str]]:
    """Resolver that prefers explicit ``frame_paths`` from each sample's
    ``input.visual_window.frame_paths`` (legacy pass3 format), falling back
    to the pattern resolver otherwise.
    """
    fallback = make_frame_resolver(video_id, config)
    samples_by_chunk = samples_by_chunk or {}

    def _resolve(chunk_idx: int) -> List[str]:
        sample = samples_by_chunk.get(chunk_idx)
        if sample:
            inp = sample.get("input") or {}
            vw = inp.get("visual_window") or {}
            paths = vw.get("frame_paths")
            if paths:
                return list(paths)
        return fallback(chunk_idx)
    return _resolve


# ---------------------------------------------------------------------------
# JSONL streaming I/O
# ---------------------------------------------------------------------------

def iter_pass4_records(input_path: Path) -> Iterator[Dict]:
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed JSONL line: %s", e)


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

@dataclass
class ConversionStats:
    n_records_in: int = 0
    n_rows_out: int = 0
    n_from_start: int = 0
    n_from_compress: int = 0
    n_compact_memory_update: int = 0
    n_compress_events: int = 0
    n_questions: int = 0
    n_chunks_total: int = 0

    def to_dict(self) -> Dict:
        return {
            "records_in": self.n_records_in,
            "rows_out": self.n_rows_out,
            "from_start": self.n_from_start,
            "from_compress": self.n_from_compress,
            "compact_memory_update": self.n_compact_memory_update,
            "compress_events": self.n_compress_events,
            "questions": self.n_questions,
            "chunks_total": self.n_chunks_total,
            "avg_chunks_per_row": (
                self.n_chunks_total / self.n_rows_out if self.n_rows_out else 0.0
            ),
        }


def convert_file(
    input_path: Path,
    output_path: Path,
    frames_root: Optional[Path] = None,
    limit: Optional[int] = None,
    log_every: int = 100,
) -> ConversionStats:
    """Read one pass4 JSONL → emit v2 JSONL. Returns stats."""
    stats = ConversionStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fr_config = FrameResolverConfig(frames_root=frames_root)

    with output_path.open("w", encoding="utf-8") as out_f:
        for rec_idx, record in enumerate(iter_pass4_records(input_path)):
            if limit is not None and stats.n_records_in >= limit:
                break
            stats.n_records_in += 1

            video_id = record.get("video_id", "unknown")
            samples_by_chunk = {
                int(s.get("chunk_idx", 0)): s for s in (record.get("samples") or [])
            }
            resolver = make_sample_aware_resolver(
                video_id, fr_config, samples_by_chunk,
            )

            try:
                rows = render_trajectory_record_to_rows(record, resolver)
            except Exception as exc:
                logger.error(
                    "render failed for %s/%s: %s",
                    video_id, record.get("trajectory_id", "?"), exc,
                )
                continue

            for row in rows:
                stats.n_rows_out += 1
                if row["trajectory_type"] == "from_start":
                    stats.n_from_start += 1
                elif row["trajectory_type"] == "from_compress":
                    stats.n_from_compress += 1
                elif row["trajectory_type"] == "compact_memory_update":
                    stats.n_compact_memory_update += 1
                if row["compress_event"] is not None:
                    stats.n_compress_events += 1
                stats.n_questions += len(row["questions_in_segment"])
                stats.n_chunks_total += int(row["n_chunks"])
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (rec_idx + 1) % log_every == 0:
                logger.info(
                    "  ... processed %d records → %d rows (compress=%d)",
                    stats.n_records_in, stats.n_rows_out, stats.n_compress_events,
                )

    logger.info(
        "[%s] %d records → %d rows (start=%d, from_compress=%d, compact=%d, "
        "compress_events=%d, questions=%d, avg_chunks=%.1f) → %s",
        input_path.name,
        stats.n_records_in, stats.n_rows_out,
        stats.n_from_start, stats.n_from_compress, stats.n_compact_memory_update,
        stats.n_compress_events, stats.n_questions,
        stats.to_dict()["avg_chunks_per_row"],
        output_path.name,
    )
    return stats


#: Splits that pass5.convert_dir should render into per-trajectory SFT
#: JSONL. RL splits (``train_rl_*``) are excluded — RL consumes pass4's
#: ``*_trajectories.jsonl`` directly via ``build_verl_parquet.py`` and
#: rendering them here just burns disk for files no downstream component
#: reads.
SFT_TRAJECTORY_SPLITS = frozenset({"train_sft", "val", "test"})


def convert_dir(
    input_dir: Path,
    output_dir: Path,
    frames_root: Optional[Path] = None,
    limit: Optional[int] = None,
    splits: Optional[frozenset] = None,
) -> Dict[str, Dict]:
    """Process every ``*_trajectories.jsonl`` under input_dir.

    ``splits`` defaults to ``SFT_TRAJECTORY_SPLITS`` — only the SFT /
    eval / test splits get rendered. Pass an explicit ``frozenset`` to
    override (e.g. ``frozenset({"my_custom_split"})`` for ad-hoc work).
    """
    allowed_splits = splits if splits is not None else SFT_TRAJECTORY_SPLITS
    manifest: Dict[str, Dict] = {}
    sources = sorted(input_dir.glob("*_trajectories.jsonl"))
    if not sources:
        logger.warning("no *_trajectories.jsonl files under %s", input_dir)
        return manifest
    for src in sources:
        split = src.stem.replace("_trajectories", "")
        if split not in allowed_splits:
            logger.info(
                "pass5.convert_dir: skipping %s (split=%r not in %s)",
                src.name, split, sorted(allowed_splits),
            )
            continue
        dst = output_dir / f"{split}_trajectory.jsonl"
        stats = convert_file(src, dst, frames_root=frames_root, limit=limit)
        manifest[split] = stats.to_dict()
    # Write a manifest summary alongside the outputs.
    manifest_path = output_dir / "_pass5_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("manifest saved → %s", manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input", type=Path,
        help="single pass4 trajectories.jsonl input",
    )
    src_group.add_argument(
        "--input-dir", type=Path,
        help="directory containing pass4 *_trajectories.jsonl files",
    )

    out_group = ap.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--output", type=Path, help="single v2 jsonl output path",
    )
    out_group.add_argument(
        "--output-dir", type=Path, help="directory for per-split v2 outputs",
    )

    ap.add_argument(
        "--frames-root", type=Path, default=None,
        help="optional root for frame path resolution (defaults to "
             "leaving frame paths in their legacy form from pass3)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap records processed (smoke test)",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )

    # Mode A: single file
    if args.input:
        if not args.output:
            ap.error("--input requires --output (use --output-dir for batch)")
        convert_file(
            args.input, args.output,
            frames_root=args.frames_root, limit=args.limit,
        )
        return

    # Mode B: directory
    if args.input_dir:
        if not args.output_dir:
            ap.error("--input-dir requires --output-dir")
        convert_dir(
            args.input_dir, args.output_dir,
            frames_root=args.frames_root, limit=args.limit,
        )


if __name__ == "__main__":
    main()
