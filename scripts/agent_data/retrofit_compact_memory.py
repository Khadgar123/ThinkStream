#!/usr/bin/env python3
"""Retrofit existing batch data to compact <MEM> memory updates.

This script is for already-built batches where pass1/pass3 artifacts exist and
we only want to rerun pass2-style memory compression. It:

1. Replays pass2 from existing rollout ``thinks`` with the new compact-memory
   trigger and teacher prompt.
2. Writes new rollout JSONs.
3. Rewrites existing samples/verified rows without rerunning pass3 LLM calls:
   every row receives the new pass2 memory snapshot, old compress rows are
   removed, and new <MEM> compress rows are inserted at the new trigger chunks.
4. Optionally rebuilds deterministic final trajectory files and the multi-turn
   pass5 trajectory render.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.pass2_rollout import (  # noqa: E402
    MemoryState,
    build_compress_request,
    parse_compress_result,
    plan_compact_memory_boundaries,
    plan_compact_memory_intervals,
)
from scripts.agent_data.pass3c_samples import _memory_from_snapshot  # noqa: E402
from scripts.agent_data.pass4 import _build_trajectory_record, _read_video_ids  # noqa: E402
from scripts.agent_data.pass5 import convert_dir as convert_trajectory_dir  # noqa: E402


LOGGER = logging.getLogger("retrofit_compact_memory")
DEFAULT_MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
DEFAULT_API_BASE = "http://10.16.12.175:8000/v1"


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _parse_batches(values: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        if re.fullmatch(r"\d+\-\d+", raw):
            start, end = [int(x) for x in raw.split("-", 1)]
            for idx in range(start, end + 1):
                roots.append(PROJECT_ROOT / "data" / "agent_v5" / f"batch{idx}")
        elif re.fullmatch(r"\d+", raw):
            roots.append(PROJECT_ROOT / "data" / "agent_v5" / f"batch{int(raw)}")
        else:
            p = Path(raw).expanduser()
            roots.append(p if p.is_absolute() else PROJECT_ROOT / p)
    return roots


def _teacher_post(
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    choice = parsed["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"teacher response truncated: usage={parsed.get('usage')}")
    msg = choice.get("message") or {}
    return (msg.get("content") or msg.get("reasoning") or "").strip()


async def _call_teacher_async(
    sem: asyncio.Semaphore,
    api_base: str,
    model: str,
    request: Dict[str, Any],
    *,
    executor: Optional[ThreadPoolExecutor],
    timeout: int,
    dry_run: bool,
) -> Optional[str]:
    if dry_run:
        return None
    async with sem:
        loop = asyncio.get_running_loop()
        fn = partial(
            _teacher_post,
            api_base,
            model,
            request["messages"],
            max_tokens=int(request["max_tokens"]),
            temperature=float(request["temperature"]),
            timeout=timeout,
        )
        return await loop.run_in_executor(executor, fn)


def _think_map(rollout: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in rollout.get("thinks") or []:
        try:
            c = int(item.get("chunk_idx"))
        except (TypeError, ValueError):
            continue
        out[c] = str(item.get("think") or item.get("text") or "").strip()
    return out


async def replay_rollout_compact(
    rollout: Dict[str, Any],
    *,
    api_base: str,
    model: str,
    sem: asyncio.Semaphore,
    executor: Optional[ThreadPoolExecutor],
    timeout: int,
    dry_run: bool,
    protected_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    protected_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    video_id = str(rollout.get("video_id") or rollout.get("id") or "")
    thinks_by_chunk = _think_map(rollout)
    if rollout.get("num_chunks") is not None:
        num_chunks = int(rollout.get("num_chunks") or 0)
    else:
        num_chunks = max(thinks_by_chunk.keys(), default=-1) + 1

    memory = MemoryState()
    snapshots: Dict[str, Any] = {}
    compression_events: List[Dict[str, Any]] = []
    trigger_intervals: List[int] = []
    last_trigger = 0
    planned_intervals = plan_compact_memory_intervals(num_chunks)
    raw_planned_boundaries = plan_compact_memory_boundaries(num_chunks)
    protected_intervals = list(protected_intervals or [])
    qa_records = list(protected_records or [])
    if not qa_records:
        qa_records = [
            {"start": int(lo), "end": int(hi), "span": int(hi) - int(lo) + 1}
            for lo, hi in protected_intervals
        ]
    effective_protected, blocked_long_qa = _split_qa_boundary_records(qa_records)
    planned_boundaries = _adjust_boundaries_for_qa(
        raw_planned_boundaries,
        num_chunks=num_chunks,
        protected_intervals=effective_protected,
    )
    planned_boundary_set = set(planned_boundaries)

    for chunk_idx in range(num_chunks):
        snapshots[str(chunk_idx)] = memory.snapshot(chunk_idx)
        schedule_ready = chunk_idx in planned_boundary_set
        if schedule_ready:
            # Balanced retrofit uses precomputed boundaries. Token diagnostics
            # are only needed on emitted events; doing this every chunk turns
            # large-batch replay into a local tokenizer bottleneck.
            raw_trigger_diag = memory.compress_trigger_diagnostic()
            trigger_diag = dict(raw_trigger_diag)
            trigger_diag.update({
                "triggered": True,
                "mode": "compact_memory_update_balanced",
                "schedule_ready": True,
                "planned_intervals": planned_intervals,
                "planned_boundaries": planned_boundaries,
                "raw_planned_boundaries": raw_planned_boundaries,
                "qa_protected_intervals": protected_intervals,
                "qa_effective_protected_intervals": effective_protected,
                "qa_blocked_long_intervals": blocked_long_qa,
                "reason": "balanced_boundary",
            })
            pre_timeline = snapshots[str(chunk_idx)]["timeline"]
            req = build_compress_request(pre_timeline, memory, video_id, chunk_idx)
            if req is not None:
                started = time.time()
                try:
                    raw = await _call_teacher_async(
                        sem,
                        api_base,
                        model,
                        req,
                        executor=executor,
                        timeout=timeout,
                        dry_run=dry_run,
                    )
                    err = None
                except Exception as exc:  # keep replay moving; fallback is deterministic
                    raw = None
                    err = str(exc)
                summary = parse_compress_result(raw, req["_meta"])
                selected_indices = req["_meta"]["selected_indices"]
                if summary.get("compact_memory_update") and summary.get("entries"):
                    memory.replace_with_compact_entries(summary)
                else:
                    memory.compress(summary, selected_indices=selected_indices)
                post_tokens = memory.count_recent_tokens()
                trigger_intervals.append(chunk_idx - last_trigger)
                last_trigger = chunk_idx
                event = {
                    "trigger_chunk": chunk_idx,
                    "summary": summary,
                    "selected_indices": selected_indices,
                    "compressed_thinks_chunks": req["_meta"].get("chunks", []),
                    "compressed_raw_think_chunks": req["_meta"].get("raw_think_chunks", []),
                    "memory_update_input": req["messages"][-1]["content"],
                    "old_memory_text": req["_meta"].get("old_memory_text", ""),
                    "new_captions_text": req["_meta"].get("new_captions_text", ""),
                    "teacher_policy": req["_meta"].get("teacher_policy", {}),
                    "trigger_diagnostic": trigger_diag,
                    "compact_memory_update": bool(summary.get("compact_memory_update")),
                    "post_compress_tokens": post_tokens,
                    "teacher_latency_sec": round(time.time() - started, 3),
                }
                if err:
                    event["teacher_error"] = err
                compression_events.append(event)

        think_text = thinks_by_chunk.get(chunk_idx, "")
        if think_text:
            memory.add_think(chunk_idx, think_text)

    out = deepcopy(rollout)
    out["snapshots"] = snapshots
    out["compression_events"] = compression_events
    out["final_memory"] = memory.snapshot(num_chunks)
    out["compact_memory_retrofit"] = {
        "enabled": True,
        "dry_run": bool(dry_run),
        "n_compressions": len(compression_events),
        "trigger_intervals": trigger_intervals,
    }
    out["compact_memory_plan"] = {
        "enabled": True,
        "policy": "balanced_full_video",
        "intervals": planned_intervals,
        "boundaries": planned_boundaries,
        "raw_boundaries": raw_planned_boundaries,
        "qa_protected_intervals": protected_intervals,
        "qa_effective_protected_intervals": effective_protected,
        "qa_blocked_long_intervals": blocked_long_qa,
    }
    return out


def _qa_protected_records_from_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return chunk-boundary ranges that must not contain a compact boundary.

    A boundary at chunk B is between B-1 and B. It is unsafe when a query was
    already asked and at least one answer is still pending, i.e. ask < B <=
    max(answer_chunks). This keeps compression outside question-answer spans.
    """
    by_card: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        card_id = str(sample.get("card_id") or (sample.get("metadata") or {}).get("card_id") or "")
        if not card_id:
            continue
        meta = sample.get("metadata") or {}
        rec = by_card.setdefault(card_id, {"card_id": card_id, "ask": None, "answers": []})
        ask = meta.get("ask_chunk")
        if ask is not None:
            try:
                ask_i = int(ask)
                rec["ask"] = ask_i if rec["ask"] is None else min(int(rec["ask"]), ask_i)
            except (TypeError, ValueError):
                pass
        if not rec.get("question"):
            rec["question"] = (
                meta.get("question")
                or sample.get("question")
                or sample.get("query")
                or ""
            )
        if sample.get("sample_type") in {"response", "recall_response"} or (
            sample.get("sample_type") == "recall" and sample.get("action") == "response"
        ):
            try:
                rec["answers"].append(int(sample.get("chunk_idx")))
            except (TypeError, ValueError):
                pass
        for emit in meta.get("per_emit_answers") or []:
            if isinstance(emit, dict) and emit.get("chunk") is not None:
                try:
                    rec["answers"].append(int(emit.get("chunk")))
                except (TypeError, ValueError):
                    pass
    protected: List[Dict[str, Any]] = []
    for rec in by_card.values():
        ask = rec.get("ask")
        answers = rec.get("answers") or []
        if ask is None or not answers:
            continue
        end = max(int(x) for x in answers)
        if end > int(ask):
            start = int(ask) + 1
            protected.append({
                "card_id": rec.get("card_id", ""),
                "ask": int(ask),
                "start": start,
                "end": end,
                "span": end - start + 1,
                "answer_chunks": sorted(set(int(x) for x in answers)),
                "question": str(rec.get("question") or "")[:240],
            })
    protected.sort(key=lambda r: (int(r["start"]), int(r["end"]), str(r.get("card_id", ""))))
    return protected


def _qa_protected_intervals_from_samples(samples: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    return [
        (int(r["start"]), int(r["end"]))
        for r in _qa_protected_records_from_samples(samples)
    ]


def _split_qa_boundary_records(
    records: Sequence[Dict[str, Any]],
    *,
    max_protected_span: int = 36,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Keep short QA spans as boundary guards; mark long spans for review.

    Existing batches were made before compact-memory boundaries existed. Some
    questions are asked near t=1 and answered hundreds of seconds later. If
    those long spans move every compact boundary after the answer, pass2
    compresses 200+ captions at once and destroys the target 25-36 chunk
    distribution. Those questions should be reviewed or re-placed; they should
    not control the pass2 compression schedule.
    """
    effective: List[Tuple[int, int]] = []
    blocked: List[Dict[str, Any]] = []
    for rec in records:
        start = int(rec.get("start", 0))
        end = int(rec.get("end", 0))
        span = int(rec.get("span") or (end - start + 1))
        if start <= 0 or end < start:
            continue
        if span <= int(max_protected_span):
            effective.append((start, end))
        else:
            out = dict(rec)
            out["reason"] = "qa_span_too_long_for_compact_boundary"
            out["max_protected_span"] = int(max_protected_span)
            blocked.append(out)
    return effective, blocked


def _is_boundary_protected(boundary: int, intervals: Sequence[Tuple[int, int]]) -> bool:
    b = int(boundary)
    return any(int(lo) <= b <= int(hi) for lo, hi in intervals)


def _adjust_boundaries_for_qa(
    boundaries: Sequence[int],
    *,
    num_chunks: int,
    protected_intervals: Sequence[Tuple[int, int]],
    min_len: int = 25,
    max_len: int = 36,
) -> List[int]:
    """Move compact boundaries out of QA spans, preferring after-answer moves."""
    adjusted: List[int] = []
    for raw in sorted(int(b) for b in boundaries if 0 < int(b) < int(num_chunks)):
        b = raw
        if _is_boundary_protected(b, protected_intervals):
            containing = [
                (lo, hi) for lo, hi in protected_intervals
                if int(lo) <= b <= int(hi)
            ]
            hi = max(int(x[1]) for x in containing)
            b = min(int(num_chunks), hi + 1)
        if b <= 0 or b >= int(num_chunks):
            continue
        if adjusted and b - adjusted[-1] < int(min_len):
            # The prior boundary already created the shorter side. Drop this
            # one rather than creating a tiny trajectory.
            continue
        adjusted.append(b)

    # Remove any boundary still protected after overlapping-interval shifts.
    adjusted = [
        b for b in adjusted
        if not _is_boundary_protected(b, protected_intervals)
    ]
    # Drop final tiny tail where possible; short tails are worse than one
    # slightly long final segment.
    if adjusted and int(num_chunks) - adjusted[-1] < int(min_len):
        adjusted.pop()
    return adjusted


def _load_verified_samples_for_video(batch_root: Path, video_id: str) -> List[Dict[str, Any]]:
    for root_name in ("verified", "samples_3c"):
        path = batch_root / root_name / f"{video_id}.json"
        if not path.exists():
            continue
        blob = _json_load(path)
        if isinstance(blob, dict):
            return list(blob.get("samples") or [])
        if isinstance(blob, list):
            return list(blob)
    return []


def _sample_sort_key(sample: Dict[str, Any]) -> Tuple[int, int]:
    order = {"compress": -1, "recall_query": 0, "recall_response": 1, "response": 2, "silent": 4, "recall_silent": 5, "recall": 1}
    return int(sample.get("chunk_idx", 0)), order.get(str(sample.get("sample_type") or ""), 6)


def _mem_text_to_entries(mem_text: str) -> List[Dict[str, Any]]:
    entries = []
    body_match = re.search(r"<MEM>\s*(.*?)\s*</MEM>", mem_text or "", re.S | re.I)
    if not body_match:
        return entries
    for m in re.finditer(r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>', body_match.group(1), re.S | re.I):
        start = int(m.group(1))
        end = int(m.group(2) if m.group(2) is not None else m.group(1))
        text = re.sub(r"\s+", " ", m.group(3)).strip()
        entries.append({"t": f"{start}-{end}", "text": text})
    return entries


def _verification_pass() -> Dict[str, Any]:
    return {"passed": True, "checks": {}, "difficulty": "medium", "fail_reasons": []}


def _new_compress_sample(
    event: Dict[str, Any],
    *,
    template: Optional[Dict[str, Any]],
    snapshot: Dict[str, Any],
    video_id: str,
    include_input: bool,
) -> Dict[str, Any]:
    chunk_idx = int(event.get("trigger_chunk", 0))
    summary = event.get("summary") or {}
    mem_text = str(summary.get("text") or "").strip()
    base = deepcopy(template or {})
    sample = {
        "chunk_idx": chunk_idx,
        "sample_type": "compress",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": base.get("trajectory_id") or f"{video_id}_traj0",
        "card_id": "",
        "sequence_type": "compress_event",
        "action": "compress",
        "output": mem_text,
        "queries": deepcopy(base.get("queries") or []),
        "user_input": "",
        "memory_update_input": str(event.get("memory_update_input") or "").strip(),
        "recall_result": None,
        "base_role": "compress_action",
        "inter_chunk": True,
        "v12_inter_chunk": True,
        "gold_caption": mem_text,
        "gold_compress_chunks": list(event.get("compressed_raw_think_chunks") or []),
        "gold_memory_entries": _mem_text_to_entries(mem_text),
        "memory_update_mode": "compact_mem",
        "video_id": video_id,
    }
    if base.get("video_path"):
        sample["video_path"] = base.get("video_path")
    metadata = dict(base.get("metadata") or {})
    metadata.update({
        "gold_action": "compress",
        "task_type": "compact_memory_update",
        "gold_answer": "",
        "canonical_answer": "",
        "gold_compress_chunks": list(event.get("compressed_raw_think_chunks") or []),
        "gold_memory_entries": _mem_text_to_entries(mem_text),
        "memory_update_mode": "compact_mem",
    })
    sample["metadata"] = metadata
    if include_input:
        sample["input"] = {
            "system": "",
            "memory": _memory_from_snapshot(snapshot),
            "user_input": "",
            "memory_update_input": str(event.get("memory_update_input") or "").strip(),
        }
        sample["verification"] = _verification_pass()
    return sample


def _apply_snapshot_to_sample(sample: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(sample)
    if isinstance(out.get("input"), dict):
        out["input"] = deepcopy(out["input"])
        out["input"]["memory"] = _memory_from_snapshot(snapshot)
    return out


def migrate_sample_file(
    source_path: Path,
    rollout: Dict[str, Any],
    dest_path: Path,
    *,
    verified_shape: bool,
) -> Dict[str, int]:
    blob = _json_load(source_path)
    if verified_shape:
        samples = list(blob.get("samples") or [])
    else:
        samples = list(blob or [])
    video_id = str(rollout.get("video_id") or (samples[0].get("video_id") if samples else source_path.stem))
    snapshots = rollout.get("snapshots") or {}
    events_by_chunk = {
        int(e.get("trigger_chunk")): e
        for e in (rollout.get("compression_events") or [])
        if e.get("trigger_chunk") is not None
    }
    noncompress = [s for s in samples if s.get("sample_type") != "compress"]
    by_chunk_template: Dict[int, Dict[str, Any]] = {}
    for s in noncompress:
        by_chunk_template.setdefault(int(s.get("chunk_idx", 0)), s)
    fallback_template = noncompress[0] if noncompress else (samples[0] if samples else {})

    out_samples: List[Dict[str, Any]] = []
    inserted = set()
    for sample in sorted(noncompress, key=_sample_sort_key):
        chunk_idx = int(sample.get("chunk_idx", 0))
        if chunk_idx in events_by_chunk and chunk_idx not in inserted:
            snap = snapshots.get(str(chunk_idx)) or snapshots.get(chunk_idx) or {}
            out_samples.append(_new_compress_sample(
                events_by_chunk[chunk_idx],
                template=by_chunk_template.get(chunk_idx) or fallback_template,
                snapshot=snap,
                video_id=video_id,
                include_input=verified_shape,
            ))
            inserted.add(chunk_idx)
        snap = snapshots.get(str(chunk_idx)) or snapshots.get(chunk_idx) or {}
        out_samples.append(_apply_snapshot_to_sample(sample, snap))

    for chunk_idx, event in sorted(events_by_chunk.items()):
        if chunk_idx in inserted:
            continue
        snap = snapshots.get(str(chunk_idx)) or snapshots.get(chunk_idx) or {}
        out_samples.append(_new_compress_sample(
            event,
            template=by_chunk_template.get(chunk_idx) or fallback_template,
            snapshot=snap,
            video_id=video_id,
            include_input=verified_shape,
        ))

    out_samples = sorted(out_samples, key=_sample_sort_key)
    if verified_shape:
        stats = dict(blob.get("stats") or {})
        stats["compact_memory_retrofit"] = {
            "source": str(source_path),
            "samples": len(out_samples),
            "compress": sum(1 for s in out_samples if s.get("sample_type") == "compress"),
        }
        _json_dump(dest_path, {"samples": out_samples, "stats": stats})
    else:
        _json_dump(dest_path, out_samples)
    return {
        "samples": len(out_samples),
        "compress": sum(1 for s in out_samples if s.get("sample_type") == "compress"),
    }


async def replay_batch_rollouts(
    batch_root: Path,
    *,
    out_suffix: str,
    api_base: str,
    model: str,
    max_concurrent: int,
    timeout: int,
    dry_run: bool,
    limit_videos: Optional[int],
    protect_qa_boundaries: bool = False,
) -> Dict[str, Any]:
    src_dir = batch_root / "rollout"
    dst_dir = batch_root / f"rollout_{out_suffix}"
    files = sorted(src_dir.glob("*.json"))
    if limit_videos:
        files = files[:limit_videos]
    sem = asyncio.Semaphore(max(1, int(max_concurrent)))
    executor = None if dry_run else ThreadPoolExecutor(max_workers=max(1, int(max_concurrent)))
    stats = {"videos": 0, "compressions": 0, "failed": 0, "intervals": []}

    async def _one(path: Path) -> None:
        try:
            rollout = _json_load(path)
            video_id = str(rollout.get("video_id") or path.stem)
            protected = (
                _qa_protected_intervals_from_samples(
                    _load_verified_samples_for_video(batch_root, video_id)
                )
                if protect_qa_boundaries
                else []
            )
            new_rollout = await replay_rollout_compact(
                rollout,
                api_base=api_base,
                model=model,
                sem=sem,
                executor=executor,
                timeout=timeout,
                dry_run=dry_run,
                protected_intervals=protected,
            )
            _json_dump(dst_dir / path.name, new_rollout)
            cm = new_rollout.get("compact_memory_retrofit") or {}
            stats["videos"] += 1
            stats["compressions"] += int(cm.get("n_compressions", 0))
            stats["intervals"].extend(cm.get("trigger_intervals") or [])
        except Exception as exc:
            stats["failed"] += 1
            LOGGER.exception("failed replay %s: %s", path, exc)

    try:
        await asyncio.gather(*[_one(p) for p in files])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    _json_dump(dst_dir / "_retrofit_stats.json", stats)
    return stats


async def replay_many_batch_rollouts(
    batch_roots: Sequence[Path],
    *,
    out_suffix: str,
    api_base: str,
    model: str,
    max_concurrent: int,
    timeout: int,
    dry_run: bool,
    limit_videos: Optional[int],
    protect_qa_boundaries: bool = False,
) -> Dict[str, Any]:
    """Replay pass2 compact memory for all videos across all batches.

    One coroutine owns one video and executes its compression calls
    sequentially. The shared semaphore only limits how many videos are inside a
    teacher request at once, so dependencies within a video are preserved while
    videos across batches run in parallel.
    """
    sem = asyncio.Semaphore(max(1, int(max_concurrent)))
    executor = None if dry_run else ThreadPoolExecutor(max_workers=max(1, int(max_concurrent)))
    stats_by_batch: Dict[str, Dict[str, Any]] = {}
    jobs: List[Tuple[Path, Path, Path]] = []
    for batch_root in batch_roots:
        src_dir = batch_root / "rollout"
        dst_dir = batch_root / f"rollout_{out_suffix}"
        files = sorted(src_dir.glob("*.json"))
        if limit_videos:
            files = files[:limit_videos]
        stats_by_batch[str(batch_root)] = {
            "videos": 0,
            "compressions": 0,
            "failed": 0,
            "intervals": [],
            "output_dir": str(dst_dir),
        }
        for path in files:
            jobs.append((batch_root, dst_dir, path))

    async def _one(batch_root: Path, dst_dir: Path, path: Path) -> None:
        key = str(batch_root)
        try:
            rollout = _json_load(path)
            video_id = str(rollout.get("video_id") or path.stem)
            protected = (
                _qa_protected_intervals_from_samples(
                    _load_verified_samples_for_video(batch_root, video_id)
                )
                if protect_qa_boundaries
                else []
            )
            new_rollout = await replay_rollout_compact(
                rollout,
                api_base=api_base,
                model=model,
                sem=sem,
                executor=executor,
                timeout=timeout,
                dry_run=dry_run,
                protected_intervals=protected,
            )
            _json_dump(dst_dir / path.name, new_rollout)
            cm = new_rollout.get("compact_memory_retrofit") or {}
            stats_by_batch[key]["videos"] += 1
            stats_by_batch[key]["compressions"] += int(cm.get("n_compressions", 0))
            stats_by_batch[key]["intervals"].extend(cm.get("trigger_intervals") or [])
        except Exception as exc:
            stats_by_batch[key]["failed"] += 1
            LOGGER.exception("failed replay %s: %s", path, exc)

    try:
        await asyncio.gather(*[_one(*job) for job in jobs])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    for batch_root in batch_roots:
        key = str(batch_root)
        out_dir = batch_root / f"rollout_{out_suffix}"
        _json_dump(out_dir / "_retrofit_stats.json", stats_by_batch.get(key, {}))
    return stats_by_batch


def migrate_batch_samples(
    batch_root: Path,
    *,
    out_suffix: str,
    limit_videos: Optional[int],
) -> Dict[str, Any]:
    rollout_dir = batch_root / f"rollout_{out_suffix}"
    stats: Dict[str, Any] = {"videos": 0, "samples_3c": {}, "verified": {}}
    rollout_files = sorted(p for p in rollout_dir.glob("*.json") if not p.name.startswith("_"))
    if limit_videos:
        rollout_files = rollout_files[:limit_videos]
    for rpath in rollout_files:
        rollout = _json_load(rpath)
        vid = str(rollout.get("video_id") or rpath.stem)
        src_sample = batch_root / "samples_3c" / f"{vid}.json"
        src_verified = batch_root / "verified" / f"{vid}.json"
        if src_sample.exists():
            st = migrate_sample_file(
                src_sample,
                rollout,
                batch_root / f"samples_3c_{out_suffix}" / src_sample.name,
                verified_shape=False,
            )
            stats["samples_3c"][vid] = st
        if src_verified.exists():
            st = migrate_sample_file(
                src_verified,
                rollout,
                batch_root / f"verified_{out_suffix}" / src_verified.name,
                verified_shape=True,
            )
            stats["verified"][vid] = st
        stats["videos"] += 1
    _json_dump(batch_root / f"verified_{out_suffix}" / "_retrofit_stats.json", stats)
    return stats


def _emit_split_from_verified(
    split_name: str,
    video_ids: Sequence[str],
    verified_dir: Path,
    out_path: Path,
) -> Dict[str, Any]:
    rows = []
    action_counts: Dict[str, int] = {}
    missing = 0
    for vid in sorted(video_ids):
        path = verified_dir / f"{vid}.json"
        if not path.exists():
            missing += 1
            continue
        blob = _json_load(path)
        samples = blob.get("samples") or []
        by_traj: Dict[str, List[Dict[str, Any]]] = {}
        for sample in samples:
            by_traj.setdefault(sample.get("trajectory_id") or "unknown_traj", []).append(sample)
        for tid, tsamples in sorted(by_traj.items()):
            rec = _build_trajectory_record(vid, tid, tsamples)
            rows.append(rec)
            for act, count in rec.get("stats", {}).get("actions", {}).items():
                action_counts[act] = action_counts.get(act, 0) + int(count)
    n = _write_jsonl(out_path, rows)
    return {
        "split": split_name,
        "videos_in_split": len(video_ids),
        "videos_missing": missing,
        "trajectories": n,
        "actions": action_counts,
        "output_path": str(out_path),
    }


def rebuild_final_and_messages(
    batch_root: Path,
    *,
    out_suffix: str,
    render_protocol: str,
    render_layout: str,
) -> Dict[str, Any]:
    source_final = batch_root / "final"
    final_dir = batch_root / f"final_{out_suffix}"
    verified_dir = batch_root / f"verified_{out_suffix}"
    final_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train_sft": _read_video_ids(source_final / "train_sft.jsonl"),
        "train_rl": _read_video_ids(source_final / "train_rl.jsonl"),
        "val": _read_video_ids(source_final / "val.jsonl"),
        "test": _read_video_ids(source_final / "test.jsonl"),
    }
    manifest = {
        "generated_by": "retrofit_compact_memory.py",
        "source_final_dir": str(source_final),
        "source_verified_dir": str(verified_dir),
        "splits": {},
    }
    for split, vids in splits.items():
        manifest["splits"][split] = _emit_split_from_verified(
            split,
            sorted(vids),
            verified_dir,
            final_dir / f"{split}_trajectories.jsonl",
        )
    _json_dump(final_dir / "trajectories_manifest.json", manifest)

    trajectory_rendered_dir = batch_root / "rendered" / f"trajectory_{out_suffix}"
    trajectory_manifest = convert_trajectory_dir(
        input_dir=final_dir,
        output_dir=trajectory_rendered_dir,
        frames_root=None,
    )
    _json_dump(trajectory_rendered_dir / "render_manifest.json", {
        "generated_by": "retrofit_compact_memory.py",
        "source_final_dir": str(final_dir),
        "trajectory_output_dir": str(trajectory_rendered_dir),
        "frame_protocol": render_protocol,
        "render_layout": render_layout,
        "trajectory_manifest": trajectory_manifest,
    })
    return {
        "final_dir": str(final_dir),
        "trajectory_rendered_dir": str(trajectory_rendered_dir),
        "trajectory_manifest": trajectory_manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", default=["1-9"], help="Batch roots or batch indices/ranges, e.g. 1 2 3 or 1-9.")
    parser.add_argument("--out-suffix", default="compact_mem_v1")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not call teacher; use deterministic fallback <MEM> summaries.")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--render-protocol", default="video_meta")
    parser.add_argument("--render-layout", default="standard_query_last")
    parser.add_argument(
        "--protect-qa-boundaries",
        action="store_true",
        help=(
            "Move compact boundaries out of short ask->answer spans. Default "
            "is off for production retrofit so long-running QA does not skew "
            "the balanced 25-36 chunk compact schedule."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    batch_roots = _parse_batches(args.batches)
    all_stats: Dict[str, Any] = {}
    pass2_only = not args.skip_rollout and args.skip_samples and args.skip_final
    if pass2_only:
        existing_roots = [p for p in batch_roots if p.exists()]
        missing_roots = [p for p in batch_roots if not p.exists()]
        for p in missing_roots:
            LOGGER.warning("skip missing batch root: %s", p)
        rollout_stats = asyncio.run(replay_many_batch_rollouts(
            existing_roots,
            out_suffix=args.out_suffix,
            api_base=args.api_base,
            model=args.model,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            dry_run=args.dry_run,
            limit_videos=args.limit_videos,
            protect_qa_boundaries=args.protect_qa_boundaries,
        ))
        print(json.dumps({k: {"rollout": v} for k, v in rollout_stats.items()}, ensure_ascii=False, indent=2))
        return 0

    for batch_root in batch_roots:
        if not batch_root.exists():
            LOGGER.warning("skip missing batch root: %s", batch_root)
            continue
        LOGGER.info("processing %s", batch_root)
        bstats: Dict[str, Any] = {}
        if not args.skip_rollout:
            bstats["rollout"] = asyncio.run(replay_batch_rollouts(
                batch_root,
                out_suffix=args.out_suffix,
                api_base=args.api_base,
                model=args.model,
                max_concurrent=args.max_concurrent,
                timeout=args.timeout,
                dry_run=args.dry_run,
                limit_videos=args.limit_videos,
                protect_qa_boundaries=args.protect_qa_boundaries,
            ))
        if not args.skip_samples:
            bstats["samples"] = migrate_batch_samples(
                batch_root,
                out_suffix=args.out_suffix,
                limit_videos=args.limit_videos,
            )
        if not args.skip_final:
            bstats["final"] = rebuild_final_and_messages(
                batch_root,
                out_suffix=args.out_suffix,
                render_protocol=args.render_protocol,
                render_layout=args.render_layout,
            )
        all_stats[str(batch_root)] = bstats
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
