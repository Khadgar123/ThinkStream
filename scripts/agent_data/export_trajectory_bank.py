"""Export canonical per-trajectory files from an existing agent_v5 batch.

This is a post-pass utility. It does not change the pass1-pass5 pipeline or
any final split files; it reads the already-rendered trajectory JSONL files and
the pass2 rollout cache, then writes one self-contained JSON file per
(video_id, trajectory_id). The output is meant to be a "trajectory bank" that
later sampling scripts can assemble into SFT/RL/eval/test splits without
re-running teacher generation.

Inputs:
  <data_dir>/final/{train_sft,train_rl,val,test}_trajectories.jsonl
  <data_dir>/rollout/{video_id}.json

Outputs:
  <data_dir>/trajectory_bank/trajectories/{safe_video_id}__{safe_tid}.json
  <data_dir>/trajectory_bank/trajectory_stats.jsonl
  <data_dir>/trajectory_bank/video_stats.jsonl
  <data_dir>/trajectory_bank/manifest.json

Usage:
  THINKSTREAM_DATA_ROOT=data/agent_v5/batch3 \\
    python -m scripts.agent_data.export_trajectory_bank
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from .config import DATA_ROOT as DEFAULT_DATA_DIR
except Exception:
    DEFAULT_DATA_DIR = Path("data/agent_v5")

SCHEMA_VERSION = "trajectory_bank.v1"
DEFAULT_SPLITS = ("train_sft", "train_rl", "val", "test")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.@-]+")
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def _resolve_data_dir(raw: Optional[str]) -> Path:
    p = Path(raw).expanduser() if raw else DEFAULT_DATA_DIR
    if not p.is_absolute():
        p = Path.cwd() / p
    if p.name == "final":
        p = p.parent
    return p.resolve()


def _safe_name(raw: str) -> str:
    cleaned = _SAFE_RE.sub("_", str(raw)).strip("._")
    return cleaned[:180] or "unknown"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _extract_tool_call(text: str) -> Dict[str, Any]:
    m = _TOOL_JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_recall_query(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured recall query from a merged v12 recall sample."""
    turn1 = sample.get("v12_assistant_turn_1") or ""
    tool = _extract_tool_call(turn1)
    args = tool.get("arguments") if isinstance(tool, dict) else None
    if isinstance(args, dict):
        return {
            "query": str(args.get("query", "")).strip(),
            "time_range": args.get("time_range", ""),
        }
    return {}


def _extract_compress_tool(sample: Dict[str, Any]) -> Dict[str, Any]:
    tool = _extract_tool_call(sample.get("output") or "")
    args = tool.get("arguments") if isinstance(tool, dict) else None
    if not isinstance(args, dict):
        return {}
    return {
        "time_range": args.get("time_range", []),
        "text": args.get("text", ""),
    }


def _rollout_think_map(rollout: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for t in rollout.get("thinks") or []:
        try:
            c = int(t.get("chunk_idx"))
        except Exception:
            continue
        out[c] = {
            "chunk_idx": c,
            "time": t.get("time"),
            "think": t.get("think", t.get("text", "")),
            "source": t.get("source", ""),
        }
    return out


def _raw_thinks_for_chunks(
    think_map: Dict[int, Dict[str, Any]],
    chunks: Iterable[Any],
) -> List[Dict[str, Any]]:
    out = []
    for raw_c in chunks or []:
        try:
            c = int(raw_c)
        except Exception:
            continue
        if c in think_map:
            out.append(deepcopy(think_map[c]))
    return out


def _compression_events(
    traj: Dict[str, Any],
    rollout: Dict[str, Any],
    think_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_trigger = {
        int(e.get("trigger_chunk")): e
        for e in rollout.get("compression_events") or []
        if e.get("trigger_chunk") is not None
    }
    out = []
    for s in traj.get("samples") or []:
        if s.get("sample_type") != "compress" and s.get("action") != "compress":
            continue
        chunk_idx = int(s.get("chunk_idx", -1))
        event = by_trigger.get(chunk_idx, {})
        summary = event.get("summary") or {}
        source_chunks = (
            summary.get("source_chunks")
            or event.get("compressed_source_chunks")
            or event.get("compressed_thinks_chunks")
            or (s.get("metadata") or {}).get("gold_compress_chunks")
            or []
        )
        raw_chunks = event.get("compressed_raw_think_chunks") or source_chunks
        tool_summary = _extract_compress_tool(s)
        out.append({
            "trigger_chunk": chunk_idx,
            "tool_time_range": tool_summary.get("time_range", []),
            "summary": {
                "time_range": summary.get("time_range", tool_summary.get("time_range", [])),
                "text": summary.get("text", tool_summary.get("text", "")),
                "source_chunks": list(source_chunks),
            },
            "raw_think_chunks": list(raw_chunks),
            "raw_thinks": _raw_thinks_for_chunks(think_map, raw_chunks),
            "teacher_policy": event.get("teacher_policy", {}),
            "post_compress_tokens": event.get("post_compress_tokens"),
            "sample_output": s.get("output", ""),
        })
    return out


def _recall_events(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for s in traj.get("samples") or []:
        if s.get("sample_type") != "recall":
            continue
        rr = s.get("recall_result") or {}
        q = _extract_recall_query(s)
        out.append({
            "chunk_idx": int(s.get("chunk_idx", -1)),
            "card_id": s.get("card_id", ""),
            "action": s.get("action", ""),
            "sequence_type": s.get("sequence_type", ""),
            "query": q,
            "returned_chunks": list(rr.get("returned_chunks") or []),
            "result_time": rr.get("time", ""),
            "answer": _extract_answer(
                s.get("v12_assistant_turn_2") or s.get("output") or ""
            ),
            "assistant_turn_1": s.get("v12_assistant_turn_1", ""),
            "assistant_turn_2": s.get("v12_assistant_turn_2", s.get("output", "")),
        })
    return out


def _question_records(
    traj: Dict[str, Any],
    recall_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recalls_by_card: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    answers_by_card: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recall_events:
        recalls_by_card[r.get("card_id", "")].append(r)
    for s in traj.get("samples") or []:
        cid = s.get("card_id", "")
        if not cid:
            continue
        if s.get("sample_type") in {"response", "recall"}:
            text = s.get("v12_assistant_turn_2") or s.get("output") or ""
            ans = _extract_answer(text)
            if ans:
                answers_by_card[cid].append({
                    "chunk_idx": int(s.get("chunk_idx", -1)),
                    "sample_type": s.get("sample_type", ""),
                    "action": s.get("action", ""),
                    "answer": ans,
                    "output": text,
                })
    out = []
    for q in traj.get("questions") or []:
        cid = q.get("card_id", "")
        row = deepcopy(q)
        row["recall_events"] = recalls_by_card.get(cid, [])
        row["answer_events"] = answers_by_card.get(cid, [])
        out.append(row)
    return out


def _interval_stats(values: List[int]) -> Dict[str, Any]:
    if len(values) < 2:
        return {"count": len(values), "diffs": []}
    diffs = [b - a for a, b in zip(values, values[1:])]
    return {
        "count": len(values),
        "diffs": diffs,
        "min": min(diffs),
        "max": max(diffs),
        "mean": round(sum(diffs) / len(diffs), 3),
        "median": statistics.median(diffs),
    }


def _trajectory_stats(
    split: str,
    traj: Dict[str, Any],
    questions: List[Dict[str, Any]],
    recall_events: List[Dict[str, Any]],
    compression_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    samples = traj.get("samples") or []
    actions = Counter(s.get("sample_type", "?") for s in samples)
    gold_actions = Counter(s.get("action", "?") for s in samples)
    families = Counter(q.get("family", "") for q in questions if q.get("family"))
    availability = Counter(q.get("availability", "") for q in questions if q.get("availability"))
    question_types = Counter(
        q.get("question_type", "") for q in questions if q.get("question_type")
    )
    answer_forms = Counter(
        q.get("answer_form", "") for q in questions if q.get("answer_form")
    )
    categories = Counter(q.get("category", "") for q in questions if q.get("category"))
    skills = Counter(q.get("skill", "") for q in questions if q.get("skill"))
    ask_chunks = sorted(
        int(q.get("ask_chunk"))
        for q in questions
        if isinstance(q.get("ask_chunk"), int) and q.get("ask_chunk") >= 0
    )
    n_compress = len(compression_events)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "video_id": traj.get("video_id", ""),
        "trajectory_id": traj.get("trajectory_id", ""),
        "video_path": traj.get("video_path", ""),
        "n_samples": len(samples),
        "n_questions": len(questions),
        "n_recall_calls": len(recall_events),
        "n_recall_response": sum(1 for r in recall_events if r.get("action") == "response"),
        "n_recall_silent": sum(1 for r in recall_events if r.get("action") == "silent"),
        "n_compress_events": n_compress,
        "n_extra_compressions": max(0, n_compress - 1),
        "has_multiple_compressions": n_compress > 1,
        "families": dict(families),
        "availability": dict(availability),
        "question_types": dict(question_types),
        "answer_forms": dict(answer_forms),
        "categories": dict(categories),
        "skills": dict(skills),
        "sample_types": dict(actions),
        "gold_actions": dict(gold_actions),
        "ask_chunk_interval": _interval_stats(ask_chunks),
        "ask_chunks": ask_chunks,
        "recall_chunks": [r.get("chunk_idx") for r in recall_events],
        "compress_trigger_chunks": [c.get("trigger_chunk") for c in compression_events],
        "chunk_idx_min": (traj.get("stats") or {}).get("chunk_idx_min"),
        "chunk_idx_max": (traj.get("stats") or {}).get("chunk_idx_max"),
    }


def _build_record(
    *,
    data_dir: Path,
    split: str,
    traj: Dict[str, Any],
    rollout: Dict[str, Any],
    include_samples: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    think_map = _rollout_think_map(rollout)
    recalls = _recall_events(traj)
    compressions = _compression_events(traj, rollout, think_map)
    questions = _question_records(traj, recalls)
    stats = _trajectory_stats(split, traj, questions, recalls, compressions)
    record = {
        "schema_version": SCHEMA_VERSION,
        "batch_root": str(data_dir),
        "split": split,
        "video_id": traj.get("video_id", ""),
        "trajectory_id": traj.get("trajectory_id", ""),
        "video_path": traj.get("video_path", ""),
        "protocol_version": traj.get("protocol_version", ""),
        "questions": questions,
        "gold_action_per_chunk": traj.get("gold_action_per_chunk", {}),
        "recall_events": recalls,
        "compression_events": compressions,
        "pass2_thinks": [
            deepcopy(think_map[c]) for c in sorted(think_map)
        ],
        "stats": stats,
    }
    if include_samples:
        record["samples"] = traj.get("samples") or []
    return record, stats


def export_bank(
    data_dir: Path,
    out_dir: Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    *,
    include_samples: bool = True,
) -> Dict[str, Any]:
    final_dir = data_dir / "final"
    rollout_dir = data_dir / "rollout"
    if not final_dir.exists():
        raise FileNotFoundError(f"final dir not found: {final_dir}")
    if not rollout_dir.exists():
        raise FileNotFoundError(f"rollout dir not found: {rollout_dir}")

    traj_out_dir = out_dir / "trajectories"
    if traj_out_dir.exists():
        for old_path in traj_out_dir.glob("*.json"):
            old_path.unlink()
    stats_rows: List[Dict[str, Any]] = []
    records_written = 0
    skipped_missing_rollout = 0
    split_counts = Counter()

    for split in splits:
        path = final_dir / f"{split}_trajectories.jsonl"
        if not path.exists():
            logger.warning("missing trajectory split: %s", path)
            continue
        for traj in _iter_jsonl(path):
            vid = str(traj.get("video_id", ""))
            tid = str(traj.get("trajectory_id", ""))
            rollout_path = rollout_dir / f"{vid}.json"
            if not rollout_path.exists():
                skipped_missing_rollout += 1
                logger.warning("[%s/%s] missing rollout: %s", vid, tid, rollout_path)
                continue
            rollout = _read_json(rollout_path)
            record, stats = _build_record(
                data_dir=data_dir,
                split=split,
                traj=traj,
                rollout=rollout,
                include_samples=include_samples,
            )
            filename = f"{_safe_name(vid)}__{_safe_name(tid)}.json"
            rel_path = Path("trajectories") / filename
            record["bank_path"] = str(out_dir / rel_path)
            stats["bank_path"] = str(out_dir / rel_path)
            _write_json(out_dir / rel_path, record)
            stats_rows.append(stats)
            records_written += 1
            split_counts[split] += 1

    stats_path = out_dir / "trajectory_stats.jsonl"
    _append_jsonl(stats_path, stats_rows)

    video_rows = _video_stats(stats_rows)
    video_stats_path = out_dir / "video_stats.jsonl"
    _append_jsonl(video_stats_path, video_rows)

    manifest = _manifest(
        data_dir=data_dir,
        out_dir=out_dir,
        stats_rows=stats_rows,
        video_rows=video_rows,
        split_counts=split_counts,
        skipped_missing_rollout=skipped_missing_rollout,
        include_samples=include_samples,
    )
    _write_json(out_dir / "manifest.json", manifest)
    logger.info(
        "trajectory bank: %d records -> %s", records_written, out_dir
    )
    return manifest


def _video_stats(stats_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in stats_rows:
        by_video[row.get("video_id", "")].append(row)
    out = []
    for vid, rows in sorted(by_video.items()):
        families = Counter()
        availability = Counter()
        question_types = Counter()
        answer_forms = Counter()
        categories = Counter()
        skills = Counter()
        sample_types = Counter()
        gold_actions = Counter()
        splits = Counter()
        for r in rows:
            families.update(r.get("families") or {})
            availability.update(r.get("availability") or {})
            question_types.update(r.get("question_types") or {})
            answer_forms.update(r.get("answer_forms") or {})
            categories.update(r.get("categories") or {})
            skills.update(r.get("skills") or {})
            sample_types.update(r.get("sample_types") or {})
            gold_actions.update(r.get("gold_actions") or {})
            splits[r.get("split", "")] += 1
        n_compress = sum(int(r.get("n_compress_events", 0)) for r in rows)
        out.append({
            "schema_version": SCHEMA_VERSION,
            "video_id": vid,
            "video_path": rows[0].get("video_path", "") if rows else "",
            "splits": dict(splits),
            "n_trajectories": len(rows),
            "n_samples": sum(int(r.get("n_samples", 0)) for r in rows),
            "n_questions": sum(int(r.get("n_questions", 0)) for r in rows),
            "n_recall_calls": sum(int(r.get("n_recall_calls", 0)) for r in rows),
            "n_compress_events": n_compress,
            "n_extra_compressions": sum(int(r.get("n_extra_compressions", 0)) for r in rows),
            "has_multiple_compressions": n_compress > 1,
            "families": dict(families),
            "availability": dict(availability),
            "question_types": dict(question_types),
            "answer_forms": dict(answer_forms),
            "categories": dict(categories),
            "skills": dict(skills),
            "sample_types": dict(sample_types),
            "gold_actions": dict(gold_actions),
            "trajectory_paths": [r.get("bank_path", "") for r in rows],
        })
    return out


def _manifest(
    *,
    data_dir: Path,
    out_dir: Path,
    stats_rows: List[Dict[str, Any]],
    video_rows: List[Dict[str, Any]],
    split_counts: Counter,
    skipped_missing_rollout: int,
    include_samples: bool,
) -> Dict[str, Any]:
    families = Counter()
    availability = Counter()
    question_types = Counter()
    answer_forms = Counter()
    categories = Counter()
    skills = Counter()
    sample_types = Counter()
    gold_actions = Counter()
    for row in stats_rows:
        families.update(row.get("families") or {})
        availability.update(row.get("availability") or {})
        question_types.update(row.get("question_types") or {})
        answer_forms.update(row.get("answer_forms") or {})
        categories.update(row.get("categories") or {})
        skills.update(row.get("skills") or {})
        sample_types.update(row.get("sample_types") or {})
        gold_actions.update(row.get("gold_actions") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts.agent_data.export_trajectory_bank",
        "source_data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "include_samples": include_samples,
        "counts": {
            "trajectories": len(stats_rows),
            "videos": len(video_rows),
            "splits": dict(split_counts),
            "skipped_missing_rollout": skipped_missing_rollout,
            "questions": sum(int(r.get("n_questions", 0)) for r in stats_rows),
            "recall_calls": sum(int(r.get("n_recall_calls", 0)) for r in stats_rows),
            "compress_events": sum(int(r.get("n_compress_events", 0)) for r in stats_rows),
            "multi_compress_trajectories": sum(
                1 for r in stats_rows if r.get("has_multiple_compressions")
            ),
        },
        "families": dict(families),
        "availability": dict(availability),
        "question_types": dict(question_types),
        "answer_forms": dict(answer_forms),
        "categories": dict(categories),
        "skills": dict(skills),
        "sample_types": dict(sample_types),
        "gold_actions": dict(gold_actions),
        "files": {
            "trajectory_dir": str(out_dir / "trajectories"),
            "trajectory_stats": str(out_dir / "trajectory_stats.jsonl"),
            "video_stats": str(out_dir / "video_stats.jsonl"),
            "manifest": str(out_dir / "manifest.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Batch root containing final/ and rollout/. Defaults to THINKSTREAM_DATA_ROOT.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir. Default: <data-dir>/trajectory_bank.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Trajectory splits to export.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Do not embed full rendered samples in each trajectory file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    data_dir = _resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "trajectory_bank"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    export_bank(
        data_dir=data_dir,
        out_dir=out_dir.resolve(),
        splits=args.splits,
        include_samples=not args.no_samples,
    )


if __name__ == "__main__":
    main()
