#!/usr/bin/env python3
"""Rerender legacy pass2 compression rows as compact-memory SFT rows.

The May-15 mixed backup contains old pass2 rollouts:

  summary = {"time_range": [start, end_exclusive], "text": "..."}

and pass3c samples rendered those as obsolete ``compress`` tool calls.  The
current SFT renderer expects standalone compact-memory rows:

  system(compact prompt) -> user(OLD_MEMORY + NEW_CAPTIONS) -> assistant(<m>...)

This script does not call a model and does not mutate rollout.  It reconstructs
the compact-memory training row from pass2's own rollout state, then rerenders
``final/*_trajectories.jsonl`` into ``rendered/trajectory/*_trajectory.jsonl``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.pass5 import (  # noqa: E402
    ConversionStats,
    FrameResolverConfig,
    make_sample_aware_resolver,
)
from scripts.agent_data.pass5_splitter import (  # noqa: E402
    DEFAULT_QUERY_INJECTION_POLICY,
    normalize_query_injection_policy,
    render_trajectory_record_to_rows,
)


SPLITS = ("train_sft", "val", "test")
M_LINE_RE = re.compile(r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>', re.S | re.I)


def _inc(stats: Dict[str, int], key: str, n: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + int(n)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip(), quote=False)


def _range_from_item(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    chunks = item.get("source_chunks") or item.get("chunks") or []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)) and chunks:
        try:
            vals = sorted(int(c) for c in chunks)
            return vals[0], vals[-1]
        except (TypeError, ValueError):
            pass
    tr = item.get("time_range") or item.get("time") or []
    if isinstance(tr, Sequence) and not isinstance(tr, (str, bytes)) and len(tr) >= 2:
        try:
            start = int(tr[0])
            end = int(tr[1])
        except (TypeError, ValueError):
            return None
        if end > start:
            # Legacy pass2 stores half-open [start, end).  Existing pass5
            # already rendered this as start-(end-1), matching source_chunks.
            end -= 1
        return start, max(start, end)
    return None


def _segments_to_mlines(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments or []:
        rng = _range_from_item(seg)
        text = _safe_text(seg.get("text") or seg.get("summary") or seg.get("think"))
        if rng is None or not text:
            continue
        start, end = rng
        lines.append(f'<m t="{start}-{end}">{text}</m>')
    return "\n".join(lines)


def _chunks_from_event(event: Dict[str, Any]) -> List[int]:
    summary = event.get("summary") or {}
    for key in ("compressed_raw_think_chunks", "compressed_thinks_chunks", "selected_indices"):
        vals = event.get(key) or []
        if vals:
            try:
                return sorted(set(int(c) for c in vals))
            except (TypeError, ValueError):
                pass
    vals = summary.get("source_chunks") or []
    if vals:
        try:
            return sorted(set(int(c) for c in vals))
        except (TypeError, ValueError):
            pass
    tr = summary.get("time_range") or []
    if isinstance(tr, Sequence) and not isinstance(tr, (str, bytes)) and len(tr) >= 2:
        try:
            start = int(tr[0])
            end = int(tr[1])
            return list(range(start, max(start, end)))
        except (TypeError, ValueError):
            pass
    return []


def _think_map(rollout: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in rollout.get("thinks") or []:
        try:
            chunk = int(item.get("chunk_idx"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("think") or item.get("text") or "").strip()
        if text:
            out[chunk] = text
    return out


def _caption_block(rollout: Dict[str, Any], chunks: Sequence[int]) -> str:
    thinks = _think_map(rollout)
    lines = []
    for chunk in sorted(set(int(c) for c in chunks)):
        text = _safe_text(thinks.get(chunk, ""))
        if text:
            lines.append(f'  <c t="{chunk}">{text}</c>')
    return "\n".join(lines)


def _snapshot_for_event(rollout: Dict[str, Any], trigger: int, *, post: bool) -> Dict[str, Any]:
    snapshots = rollout.get("snapshots") or {}
    keys = [trigger + 1, trigger] if post else [trigger, trigger - 1]
    for key in keys:
        if str(key) in snapshots and isinstance(snapshots[str(key)], dict):
            return snapshots[str(key)]
        if key in snapshots and isinstance(snapshots[key], dict):
            return snapshots[key]
    if post and isinstance(rollout.get("final_memory"), dict):
        return rollout["final_memory"]
    return {}


def _event_summary_segment(event: Dict[str, Any]) -> Dict[str, Any]:
    summary = event.get("summary") or {}
    out = dict(summary)
    chunks = _chunks_from_event(event)
    if chunks:
        out["source_chunks"] = chunks
        out["time_range"] = [min(chunks), max(chunks) + 1]
    return out


def _post_memory_text(rollout: Dict[str, Any], event: Dict[str, Any]) -> str:
    trigger = int(event.get("trigger_chunk") or 0)
    post_snapshot = _snapshot_for_event(rollout, trigger, post=True)
    text = _segments_to_mlines(post_snapshot.get("compressed_segments") or [])
    if text:
        return text

    pre_snapshot = _snapshot_for_event(rollout, trigger, post=False)
    segments = list(pre_snapshot.get("compressed_segments") or [])
    segments.append(_event_summary_segment(event))
    return _segments_to_mlines(segments)


def _covered_range(chunks: Sequence[int]) -> Tuple[int, int]:
    vals = sorted(set(int(c) for c in chunks))
    if not vals:
        return 0, 0
    return vals[0], vals[-1]


def _memory_update_input(rollout: Dict[str, Any], event: Dict[str, Any]) -> str:
    trigger = int(event.get("trigger_chunk") or 0)
    chunks = _chunks_from_event(event)
    start, end = _covered_range(chunks)
    pre_snapshot = _snapshot_for_event(rollout, trigger, post=False)
    old_lines = _segments_to_mlines(pre_snapshot.get("compressed_segments") or [])
    captions = _caption_block(rollout, chunks)
    old_body = f"  {old_lines.replace(chr(10), chr(10) + '  ')}" if old_lines else ""
    caption_body = captions or "  (no source captions found in pass2 rollout)"
    return (
        "OLD_MEMORY:\n"
        "<MEM>\n"
        f"{old_body}\n"
        "</MEM>\n\n"
        "NEW_CAPTIONS:\n"
        "<NEW_CAPTIONS>\n"
        f"{caption_body}\n"
        "</NEW_CAPTIONS>\n\n"
        f"Covered latest span: t={start}-{end}\n"
        "Coverage check: preserve useful OLD_MEMORY and cover the listed "
        "NEW_CAPTIONS using their real timestamps.\n"
        "Return only compact-memory XML lines:\n"
        '<m t="start-end">one concise event or state.</m>\n'
        "Do not output NEW_MEMORY:, markdown, prose, analysis, or any text "
        "outside the <m> lines."
    )


def _memory_entries_from_mlines(text: str) -> List[Dict[str, Any]]:
    entries = []
    for m in M_LINE_RE.finditer(text or ""):
        start = int(m.group(1))
        end = int(m.group(2) if m.group(2) is not None else m.group(1))
        entries.append({
            "t": f"{start}-{end}",
            "text": html.unescape(re.sub(r"\s+", " ", m.group(3)).strip()),
        })
    return entries


class RolloutCache:
    def __init__(self, batch_root: Path):
        self.batch_root = batch_root
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def get(self, video_id: str) -> Optional[Dict[str, Any]]:
        if video_id not in self._cache:
            path = self.batch_root / "rollout" / f"{video_id}.json"
            self._cache[video_id] = _load_json(path) if path.exists() else None
        return self._cache[video_id]


def _event_map(rollout: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for event in rollout.get("compression_events") or []:
        try:
            out[int(event.get("trigger_chunk"))] = event
        except (TypeError, ValueError):
            continue
    return out


def _gold_chunks_from_mlines(mem_text: str) -> List[int]:
    chunks: List[int] = []
    for entry in _memory_entries_from_mlines(mem_text):
        try:
            start, end = [int(x) for x in str(entry.get("t", "")).split("-", 1)]
        except (TypeError, ValueError):
            continue
        chunks.extend(range(start, end + 1))
    return sorted(set(chunks))


def _repair_compress_sample(
    sample: Dict[str, Any],
    rollout: Optional[Dict[str, Any]],
    stats: Dict[str, int],
    *,
    add_metadata: bool,
) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(sample, dict) or sample.get("sample_type") != "compress":
        return sample, False
    _inc(stats, "compress_seen")
    if not rollout:
        _inc(stats, "missing_rollout")
        return sample, False
    try:
        chunk = int(sample.get("chunk_idx") or 0)
    except (TypeError, ValueError):
        chunk = 0
    event = _event_map(rollout).get(chunk)
    if event is None:
        _inc(stats, "missing_event")
        return sample, False
    mem_text = _post_memory_text(rollout, event).strip()
    if not mem_text:
        _inc(stats, "empty_mem_text")
        return sample, False

    update_input = _memory_update_input(rollout, event)
    entries = _memory_entries_from_mlines(mem_text)
    chunks = _gold_chunks_from_mlines(mem_text)
    ns = dict(sample)
    ns.update({
        "output": mem_text,
        "memory_update_input": update_input,
        "gold_caption": mem_text,
        "gold_compress_chunks": chunks,
        "gold_memory_entries": entries,
        "memory_update_mode": "compact_mem",
        "user_input": "",
        "recall_result": None,
        "base_role": ns.get("base_role") or "compress_action",
        "inter_chunk": True,
    })
    inp = ns.get("input")
    if isinstance(inp, dict):
        new_inp = dict(inp)
        new_inp["memory_update_input"] = update_input
        ns["input"] = new_inp
    if add_metadata or isinstance(ns.get("metadata"), dict):
        meta = dict(ns.get("metadata") or {})
        meta.update({
            "task_type": "compact_memory_update",
            "memory_update_input": update_input,
            "memory_update_mode": "compact_mem",
            "gold_memory_entries": entries,
            "gold_compress_chunks": chunks,
            "legacy_pass2_rerender": True,
        })
        ns["metadata"] = meta
    _inc(stats, "compress_repaired")
    _inc(stats, "compress_augmented")
    return ns, True


def _augment_record(record: Dict[str, Any], rollouts: RolloutCache, stats: Dict[str, int]) -> Dict[str, Any]:
    video_id = str(record.get("video_id") or "")
    rollout = rollouts.get(video_id)
    if not rollout:
        _inc(stats, "missing_rollout")
        return record
    if not _event_map(rollout):
        return record

    out = dict(record)
    new_samples = []
    changed = False
    for sample in record.get("samples") or []:
        ns, did_change = _repair_compress_sample(
            sample,
            rollout,
            stats,
            add_metadata=True,
        )
        new_samples.append(ns)
        changed = changed or did_change
    if changed:
        out["samples"] = new_samples
        _inc(stats, "records_augmented")
    return out


def repair_json_samples_file(
    path: Path,
    batch_root: Path,
    *,
    write: bool,
    add_metadata: bool,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "files_seen": 1,
        "files_changed": 0,
        "samples_seen": 0,
        "compress_seen": 0,
        "compress_repaired": 0,
        "missing_rollout": 0,
        "missing_event": 0,
        "empty_mem_text": 0,
    }
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        samples = data["samples"]
        container = "dict"
    elif isinstance(data, list):
        samples = data
        container = "list"
    else:
        _inc(stats, "unsupported_json")
        return stats

    rollouts = RolloutCache(batch_root)
    default_video_id = path.stem
    new_samples: List[Dict[str, Any]] = []
    changed = False
    rollout_cache_by_vid: Dict[str, Optional[Dict[str, Any]]] = {}
    for sample in samples:
        _inc(stats, "samples_seen")
        video_id = str(sample.get("video_id") or default_video_id) if isinstance(sample, dict) else default_video_id
        if video_id not in rollout_cache_by_vid:
            rollout_cache_by_vid[video_id] = rollouts.get(video_id)
        ns, did_change = _repair_compress_sample(
            sample,
            rollout_cache_by_vid.get(video_id),
            stats,
            add_metadata=add_metadata,
        )
        new_samples.append(ns)
        changed = changed or did_change

    if changed and write:
        if container == "dict":
            out = dict(data)
            out["samples"] = new_samples
        else:
            out = new_samples
        _write_json_atomic(path, out)
        _inc(stats, "files_changed")
    elif changed:
        _inc(stats, "files_changed")
    return stats


def repair_final_jsonl_file(
    path: Path,
    batch_root: Path,
    *,
    write: bool,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "files_seen": 1,
        "files_changed": 0,
        "records_seen": 0,
        "records_changed": 0,
        "samples_seen": 0,
        "compress_seen": 0,
        "compress_repaired": 0,
        "missing_rollout": 0,
        "missing_event": 0,
        "empty_mem_text": 0,
    }
    rollouts = RolloutCache(batch_root)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out_f = tmp.open("w", encoding="utf-8") if write else None
    changed_any = False
    try:
        for record in _iter_jsonl(path):
            _inc(stats, "records_seen")
            record_changed = False
            if isinstance(record.get("samples"), list):
                video_id = str(record.get("video_id") or "")
                rollout = rollouts.get(video_id) if video_id else None
                new_samples = []
                for sample in record.get("samples") or []:
                    _inc(stats, "samples_seen")
                    ns, did_change = _repair_compress_sample(
                        sample,
                        rollout,
                        stats,
                        add_metadata=True,
                    )
                    new_samples.append(ns)
                    record_changed = record_changed or did_change
                if record_changed:
                    record = dict(record)
                    record["samples"] = new_samples
            else:
                _inc(stats, "samples_seen")
                video_id = str(record.get("video_id") or "")
                ns, record_changed = _repair_compress_sample(
                    record,
                    rollouts.get(video_id) if video_id else None,
                    stats,
                    add_metadata=True,
                )
                record = ns
            if record_changed:
                changed_any = True
                _inc(stats, "records_changed")
            if out_f is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()
    if write:
        if changed_any:
            tmp.replace(path)
            _inc(stats, "files_changed")
        elif tmp.exists():
            tmp.unlink()
    elif changed_any:
        _inc(stats, "files_changed")
    return stats


def _merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        if isinstance(value, int):
            _inc(dst, key, value)


def repair_sources(batch_root: Path, *, write: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for rel, add_metadata in (("samples_3c", False), ("verified", True)):
        dir_path = batch_root / rel
        totals: Dict[str, int] = {}
        if dir_path.exists():
            for path in sorted(dir_path.glob("*.json")):
                _merge_counts(
                    totals,
                    repair_json_samples_file(
                        path,
                        batch_root,
                        write=write,
                        add_metadata=add_metadata,
                    ),
                )
        report[rel] = totals

    final_dir = batch_root / "final"
    totals = {}
    if final_dir.exists():
        for path in sorted(final_dir.glob("*.jsonl")):
            _merge_counts(totals, repair_final_jsonl_file(path, batch_root, write=write))
    report["final"] = totals
    return report


def restore_missing_from_backup(
    batch_root: Path,
    backup_batch_root: Path,
    *,
    write: bool,
    subdirs: Sequence[str],
) -> Dict[str, int]:
    stats: Dict[str, int] = {"files_seen": 0, "files_copied": 0, "files_existing": 0}
    if not backup_batch_root.exists():
        stats["missing_backup_batch"] = 1
        return stats
    for rel in subdirs:
        src = backup_batch_root / rel
        dst = batch_root / rel
        if not src.exists():
            continue
        for path in sorted(src.rglob("*")):
            if path.is_dir():
                continue
            _inc(stats, "files_seen")
            target = dst / path.relative_to(src)
            if target.exists():
                _inc(stats, "files_existing")
                continue
            _inc(stats, "files_copied")
            if write:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return stats


def rerender_split(
    batch_root: Path,
    split: str,
    *,
    write: bool,
    limit_records: Optional[int],
    query_injection_policy: str,
) -> Dict[str, Any]:
    src = batch_root / "final" / f"{split}_trajectories.jsonl"
    dst = batch_root / "rendered" / "trajectory" / f"{split}_trajectory.jsonl"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    rollouts = RolloutCache(batch_root)
    fr_config = FrameResolverConfig(frames_root=None)
    cstats = ConversionStats()
    repair_stats = {
        "records_augmented": 0,
        "compress_augmented": 0,
        "missing_rollout": 0,
        "missing_event": 0,
        "empty_mem_text": 0,
    }
    if not src.exists():
        return {"missing": str(src)}
    if write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_f = tmp.open("w", encoding="utf-8")
    else:
        out_f = None
    try:
        for idx, record in enumerate(_iter_jsonl(src)):
            if limit_records is not None and idx >= limit_records:
                break
            cstats.n_records_in += 1
            record = _augment_record(record, rollouts, repair_stats)
            video_id = str(record.get("video_id") or "unknown")
            samples_by_chunk = {
                int(s.get("chunk_idx", 0)): s
                for s in (record.get("samples") or [])
                if isinstance(s, dict)
            }
            resolver = make_sample_aware_resolver(video_id, fr_config, samples_by_chunk)
            rows = render_trajectory_record_to_rows(
                record,
                resolver,
                query_injection_policy=query_injection_policy,
            )
            for row in rows:
                cstats.n_rows_out += 1
                if row["trajectory_type"] == "from_start":
                    cstats.n_from_start += 1
                elif row["trajectory_type"] == "from_compress":
                    cstats.n_from_compress += 1
                elif row["trajectory_type"] == "compact_memory_update":
                    cstats.n_compact_memory_update += 1
                if row.get("compress_event") is not None:
                    cstats.n_compress_events += 1
                cstats.n_questions += len(row.get("questions_in_segment") or [])
                cstats.n_chunks_total += int(row.get("n_chunks") or 0)
                if out_f is not None:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()
    if write:
        tmp.replace(dst)
    return {
        "source": str(src),
        "output": str(dst),
        "wrote": bool(write),
        "conversion": cstats.to_dict(),
        "repair": repair_stats,
    }


def _batch_roots(root: Path, names: Sequence[str]) -> List[Path]:
    if names:
        return [root / name if not Path(name).is_absolute() else Path(name) for name in names]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("batch"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--batches", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--query-injection-policy", default=DEFAULT_QUERY_INJECTION_POLICY)
    parser.add_argument(
        "--sync-sources",
        action="store_true",
        help="Repair samples_3c, verified, and final JSONL sources before rerendering.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Optional backup root used only to copy files missing from the destination batch.",
    )
    parser.add_argument(
        "--restore-missing",
        action="store_true",
        help="Copy missing non-rollout pipeline files from --backup-root/batchN before repair.",
    )
    parser.add_argument(
        "--restore-subdirs",
        nargs="*",
        default=["placements", "task_cards", "samples_3c", "verified", "final"],
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    backup_root = None
    if args.backup_root is not None:
        backup_root = args.backup_root.expanduser()
        if not backup_root.is_absolute():
            backup_root = PROJECT_ROOT / backup_root
    query_policy = normalize_query_injection_policy(args.query_injection_policy)
    report: Dict[str, Any] = {
        "root": str(root),
        "write": bool(args.write),
        "query_injection_policy": query_policy,
        "batches": {},
    }
    for batch_root in _batch_roots(root, args.batches):
        if not batch_root.exists():
            report["batches"][str(batch_root)] = {"missing": True}
            continue
        batch_report: Dict[str, Any] = {}
        if args.restore_missing:
            if backup_root is None:
                batch_report["restore_missing"] = {"error": "--backup-root is required"}
            else:
                batch_report["restore_missing"] = restore_missing_from_backup(
                    batch_root,
                    backup_root / batch_root.name,
                    write=bool(args.write),
                    subdirs=args.restore_subdirs,
                )
        if args.sync_sources:
            batch_report["sync_sources"] = repair_sources(batch_root, write=bool(args.write))
        split_report = {}
        for split in args.splits:
            split_report[split] = rerender_split(
                batch_root,
                split,
                write=bool(args.write),
                limit_records=args.limit_records,
                query_injection_policy=query_policy,
            )
        batch_report["rerendered_splits"] = split_report
        report["batches"][str(batch_root)] = batch_report

    if args.report_out:
        report_out = args.report_out.expanduser()
        if not report_out.is_absolute():
            report_out = PROJECT_ROOT / report_out
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_out"] = str(report_out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
