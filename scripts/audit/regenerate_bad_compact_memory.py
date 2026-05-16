#!/usr/bin/env python3
"""Regenerate compact-memory rollouts that have hard timestamp defects.

This is intentionally narrower than a full batch retrofit.  It detects rollout
files whose compact <m t="..."> entries have parse/line-count failures,
overlapping closed intervals, or intervals that reach the current trigger
chunk, then replays only those videos with a stricter timestamp prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data import pass2_rollout as pass2  # noqa: E402
from scripts.agent_data import retrofit_compact_memory as retrofit  # noqa: E402
from scripts.agent_data.retrofit_compact_memory import (  # noqa: E402
    DEFAULT_MODEL,
    _json_dump,
    _json_load,
    replay_rollout_compact,
)


DEFAULT_ENDPOINTS = (
    "http://10.16.10.172:8000/v1",
    "http://10.16.11.160:8000/v1",
)

STRICT_TIMESTAMP_RULES = """

Hard timestamp rules for this regeneration:
- For this regeneration, return exactly 5 <m> lines. Do not return 1, 2, 3, 4, 6, or more than 6 lines.
- Treat every t value as an inclusive chunk index, not a half-open boundary.
- The largest output timestamp must be <= the Covered latest span end.
- A caption <c t="92"> covers exactly chunk 92; write 92-92 unless t=93 is also present in the input.
- Output ranges must be chronological and non-overlapping. Adjacent ranges must advance by one chunk: 0-5 then 6-10, never 0-5 then 5-10.
- Do not duplicate the same span or nest a small span inside a larger span.
- Use only integer timestamps grounded in OLD_MEMORY ranges or NEW_CAPTIONS t values.
"""


def _install_strict_prompt() -> None:
    if "Hard timestamp rules for this regeneration" in pass2.COMPACT_MEMORY_UPDATE_SYSTEM_PROMPT:
        return
    pass2.COMPACT_MEMORY_UPDATE_SYSTEM_PROMPT += STRICT_TIMESTAMP_RULES
    pass2.COMPACT_MEMORY_UPDATE_PROMPT += """

Additional timestamp check before final answer:
- Return exactly 5 <m> lines for this regeneration.
- All <m> ranges are closed intervals.
- No output end timestamp is greater than the Covered latest span end.
- No two output ranges overlap or duplicate each other.
"""
    pass2.PASS_CONFIG["pass2_rollout"]["temperature"] = 0.0


def _entries_to_text(entries: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for entry in entries:
        tr = entry.get("time_range") or [0, 0]
        lines.append(f'  <m t="{int(tr[0])}-{int(tr[1])}">{str(entry.get("text") or "").strip()}</m>')
    return "\n".join(lines)


def _set_entry_range(entry: Dict[str, Any], start: int, end: int) -> None:
    entry["time_range"] = [int(start), int(end)]
    entry["source_chunks"] = list(range(int(start), int(end) + 1))
    entry["compact_memory"] = True


def _repair_summary_ranges(summary: Dict[str, Any], *, max_end: int) -> Dict[str, Any]:
    """Repair closed-interval defects while keeping vLLM-generated text."""
    entries_in = summary.get("entries") or []
    if not entries_in:
        return summary
    changed = False
    entries = deepcopy(entries_in)
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        tr = entry.get("time_range") or []
        if not (isinstance(tr, list) and len(tr) >= 2):
            continue
        try:
            start = int(tr[0])
            end = int(tr[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
            changed = True
        if end > max_end:
            end = max_end
            changed = True
        if start > max_end:
            start = max_end
            changed = True
        if start < 0:
            start = 0
            changed = True
        if not normalized:
            if start <= end:
                _set_entry_range(entry, start, end)
                normalized.append(entry)
            continue

        prev = normalized[-1]
        prev_start, prev_end = [int(x) for x in prev["time_range"]]
        if start <= prev_end:
            # Prefer splitting the previous span before this span. This keeps
            # both vLLM-generated lines when the model nested one range inside
            # another, e.g. 5-12 followed by 8-12.
            if prev_start <= start - 1:
                _set_entry_range(prev, prev_start, start - 1)
                changed = True
            elif prev_end + 1 <= max_end and end > prev_end:
                start = prev_end + 1
                changed = True
            elif prev_end < max_end:
                start = prev_end + 1
                end = max(start, min(end, max_end))
                changed = True
            else:
                changed = True
                continue
        if start <= end:
            _set_entry_range(entry, start, end)
            normalized.append(entry)

    # Final monotonic pass in case a previous shrink made another edge case.
    repaired: List[Dict[str, Any]] = []
    prev_end: Optional[int] = None
    for entry in normalized:
        start, end = [int(x) for x in entry["time_range"]]
        end = min(end, max_end)
        if prev_end is not None and start <= prev_end:
            start = prev_end + 1
            changed = True
        if start > max_end or start > end:
            changed = True
            continue
        _set_entry_range(entry, start, end)
        repaired.append(entry)
        prev_end = end

    if not changed or len(repaired) < 4:
        return summary
    out = dict(summary)
    out["entries"] = repaired
    out["time_range"] = [
        min(int(e["time_range"][0]) for e in repaired),
        max(int(e["time_range"][1]) for e in repaired),
    ]
    out["source_chunks"] = sorted(set(c for e in repaired for c in e.get("source_chunks", [])))
    out["text"] = _entries_to_text(repaired)
    out["parse_success"] = True
    out["n_entries"] = len(repaired)
    out["range_repaired"] = True
    out["range_repair_reason"] = "closed_interval_overlap_or_future_boundary"
    return out


def _install_range_repair() -> None:
    original = retrofit.parse_compress_result

    def parse_with_range_repair(raw: Optional[str], meta: Dict[str, Any]) -> Dict[str, Any]:
        summary = original(raw, meta)
        if meta.get("task_type") != "compact_memory_update":
            return summary
        tr = meta.get("time_range") or []
        if not (isinstance(tr, list) and len(tr) >= 2):
            return summary
        try:
            max_end = int(tr[1])
        except (TypeError, ValueError):
            return summary
        return _repair_summary_ranges(summary, max_end=max_end)

    retrofit.parse_compress_result = parse_with_range_repair


def _event_ranges(event: Dict[str, Any]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    summary = event.get("summary") or {}
    for entry in summary.get("entries") or []:
        tr = entry.get("time_range") or []
        if not (isinstance(tr, list) and len(tr) >= 2):
            continue
        try:
            start = int(tr[0])
            end = int(tr[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return sorted(ranges)


def bad_event_reasons(event: Dict[str, Any]) -> List[str]:
    summary = event.get("summary") or {}
    reasons = set()
    entries = summary.get("entries") or []
    n_entries = len(entries)
    if not summary.get("parse_success"):
        reasons.add("parse_fail")
    if n_entries and not (4 <= n_entries <= 6):
        reasons.add("bad_line_count")

    trigger = event.get("trigger_chunk")
    trigger_i: Optional[int]
    try:
        trigger_i = int(trigger) if trigger is not None else None
    except (TypeError, ValueError):
        trigger_i = None

    prev_end: Optional[int] = None
    for start, end in _event_ranges(event):
        if start < 0:
            reasons.add("negative_time")
        if trigger_i is not None and end >= trigger_i:
            reasons.add("future_or_current_chunk")
        if prev_end is not None and start <= prev_end:
            reasons.add("overlap")
        prev_end = max(prev_end if prev_end is not None else end, end)
    return sorted(reasons)


def rollout_bad_events(rollout: Dict[str, Any]) -> List[Dict[str, Any]]:
    bad: List[Dict[str, Any]] = []
    for idx, event in enumerate(rollout.get("compression_events") or []):
        reasons = bad_event_reasons(event)
        if not reasons:
            continue
        bad.append({
            "event_index": idx,
            "trigger_chunk": event.get("trigger_chunk"),
            "reasons": reasons,
            "ranges": _event_ranges(event),
        })
    return bad


def _iter_rollouts(batch_roots: Sequence[Path]) -> Iterable[Tuple[int, Path, Path]]:
    for batch_root in batch_roots:
        name = batch_root.name
        batch_idx = int(name.replace("batch", "")) if name.startswith("batch") and name[5:].isdigit() else -1
        rollout_dir = batch_root / "rollout"
        for path in sorted(rollout_dir.glob("*.json")):
            if path.name in {"_retrofit_stats.json"}:
                continue
            yield batch_idx, batch_root, path


def find_targets(
    batch_roots: Sequence[Path],
    *,
    limit: Optional[int] = None,
    videos: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    video_set = {str(v) for v in (videos or []) if str(v)}
    targets: List[Dict[str, Any]] = []
    for batch_idx, batch_root, path in _iter_rollouts(batch_roots):
        rollout = _json_load(path)
        video_id = str(rollout.get("video_id") or path.stem)
        if video_set and video_id not in video_set and path.stem not in video_set:
            continue
        bad = rollout_bad_events(rollout)
        if not bad:
            continue
        targets.append({
            "batch": batch_idx,
            "batch_root": str(batch_root),
            "path": str(path),
            "video_id": video_id,
            "bad_events_before": bad,
            "n_events": len(rollout.get("compression_events") or []),
        })
        if limit is not None and len(targets) >= int(limit):
            break
    return targets


async def _replay_one(
    target: Dict[str, Any],
    *,
    endpoints: Sequence[str],
    model: str,
    semaphores: Dict[str, asyncio.Semaphore],
    executor: ThreadPoolExecutor,
    timeout: int,
    max_attempts: int,
    target_index: int,
) -> Dict[str, Any]:
    src_path = Path(target["path"])
    last_error = None
    attempts: List[Dict[str, Any]] = []
    for attempt in range(max(1, int(max_attempts))):
        endpoint = endpoints[(target_index + attempt) % len(endpoints)]
        started = time.time()
        try:
            rollout = _json_load(src_path)
            new_rollout = await replay_rollout_compact(
                rollout,
                api_base=endpoint,
                model=model,
                sem=semaphores[endpoint],
                executor=executor,
                timeout=timeout,
                dry_run=False,
                protected_intervals=[],
            )
            bad_after = rollout_bad_events(new_rollout)
            attempts.append({
                "attempt": attempt + 1,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 3),
                "bad_after": bad_after,
                "n_events": len(new_rollout.get("compression_events") or []),
            })
            if not bad_after:
                return {
                    **target,
                    "ok": True,
                    "endpoint": endpoint,
                    "attempts": attempts,
                    "new_rollout": new_rollout,
                }
            last_error = f"post_validation_failed:{len(bad_after)}"
        except Exception as exc:  # noqa: BLE001 - report and retry on the other node.
            last_error = repr(exc)
            attempts.append({
                "attempt": attempt + 1,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 3),
                "error": last_error,
            })
    return {
        **target,
        "ok": False,
        "endpoint": None,
        "attempts": attempts,
        "error": last_error,
    }


async def replay_targets(
    targets: Sequence[Dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    max_concurrent_per_endpoint: int,
    timeout: int,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    semaphores = {
        endpoint: asyncio.Semaphore(max(1, int(max_concurrent_per_endpoint)))
        for endpoint in endpoints
    }
    executor = ThreadPoolExecutor(max_workers=max(1, int(max_concurrent_per_endpoint) * len(endpoints)))
    try:
        tasks = [
            _replay_one(
                target,
                endpoints=endpoints,
                model=model,
                semaphores=semaphores,
                executor=executor,
                timeout=timeout,
                max_attempts=max_attempts,
                target_index=idx,
            )
            for idx, target in enumerate(targets)
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def _copy_backup(src_path: Path, backup_root: Path) -> Path:
    rel = src_path.relative_to(PROJECT_ROOT)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst


def apply_results(results: Sequence[Dict[str, Any]], *, backup_root: Path) -> List[Dict[str, Any]]:
    applied: List[Dict[str, Any]] = []
    for result in results:
        if not result.get("ok"):
            continue
        src_path = Path(result["path"])
        backup_path = _copy_backup(src_path, backup_root)
        _json_dump(src_path, result["new_rollout"])
        applied.append({
            "path": str(src_path),
            "backup_path": str(backup_path),
            "video_id": result.get("video_id"),
            "batch": result.get("batch"),
            "endpoint": result.get("endpoint"),
        })
    return applied


def _summarize_targets(targets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_batch = Counter()
    reasons = Counter()
    bad_events = 0
    for target in targets:
        by_batch[str(target.get("batch"))] += 1
        for event in target.get("bad_events_before") or []:
            bad_events += 1
            reasons.update(event.get("reasons") or [])
    return {
        "files": len(targets),
        "bad_events": bad_events,
        "files_by_batch": dict(sorted(by_batch.items())),
        "reasons": dict(sorted(reasons.items())),
    }


def _summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    endpoint_files = Counter(str(r.get("endpoint")) for r in ok)
    endpoint_events = Counter()
    for r in ok:
        endpoint_events[str(r.get("endpoint"))] += int(r.get("n_events") or 0)
    return {
        "ok_files": len(ok),
        "failed_files": len(failed),
        "endpoint_files": dict(sorted(endpoint_files.items())),
        "endpoint_events": dict(sorted(endpoint_events.items())),
        "failed": [
            {
                "path": r.get("path"),
                "video_id": r.get("video_id"),
                "error": r.get("error"),
                "attempts": r.get("attempts"),
            }
            for r in failed
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", default=["9", "11"])
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent-per-endpoint", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--videos", nargs="*", default=None, help="Optional video ids/stems to target.")
    parser.add_argument("--report", default=None)
    parser.add_argument("--apply", action="store_true", help="Overwrite original rollout files after validation.")
    parser.add_argument("--repair-ranges", action="store_true", help="Post-parse repair closed-interval overlap/future boundaries.")
    args = parser.parse_args()

    endpoint_hosts = []
    for endpoint in args.endpoints:
        host = endpoint.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        endpoint_hosts.append(host)
    no_proxy = ",".join(dict.fromkeys(endpoint_hosts + ["localhost", "127.0.0.1"]))
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), no_proxy]))
    os.environ["no_proxy"] = ",".join(filter(None, [os.environ.get("no_proxy", ""), no_proxy]))

    _install_strict_prompt()
    if args.repair_ranges:
        _install_range_repair()
    batch_roots = []
    for value in args.batches:
        raw = str(value)
        if raw.startswith("batch"):
            batch_roots.append(PROJECT_ROOT / "data" / "agent_v5" / raw)
        elif raw.isdigit():
            batch_roots.append(PROJECT_ROOT / "data" / "agent_v5" / f"batch{raw}")
        else:
            path = Path(raw).expanduser()
            batch_roots.append(path if path.is_absolute() else PROJECT_ROOT / path)

    targets = find_targets(batch_roots, limit=args.limit, videos=args.videos)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report) if args.report else PROJECT_ROOT / "data" / "agent_v5" / "audits" / f"regenerate_bad_compact_memory_{stamp}.json"
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    results = asyncio.run(replay_targets(
        targets,
        endpoints=list(args.endpoints),
        model=args.model,
        max_concurrent_per_endpoint=args.max_concurrent_per_endpoint,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )) if targets else []

    applied: List[Dict[str, Any]] = []
    backup_root = PROJECT_ROOT / "data" / "agent_v5" / "audits" / f"bad_compact_memory_backup_{stamp}"
    if args.apply:
        applied = apply_results(results, backup_root=backup_root)

    compact_results = []
    for result in results:
        out = {k: v for k, v in result.items() if k != "new_rollout"}
        compact_results.append(out)

    report = {
        "generated_at": stamp,
        "strict_prompt": True,
        "range_repair": bool(args.repair_ranges),
        "model": args.model,
        "endpoints": list(args.endpoints),
        "applied": bool(args.apply),
        "backup_root": str(backup_root) if applied else None,
        "target_summary": _summarize_targets(targets),
        "result_summary": _summarize_results(results),
        "applied_files": applied,
        "results": compact_results,
    }
    _json_dump(report_path, report)
    print(json.dumps({
        "report": str(report_path),
        "target_summary": report["target_summary"],
        "result_summary": report["result_summary"],
        "applied_files": len(applied),
        "backup_root": report["backup_root"],
    }, ensure_ascii=False, indent=2))
    return 0 if not report["result_summary"]["failed_files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
