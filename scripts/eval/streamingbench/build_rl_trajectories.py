#!/usr/bin/env python3
"""Build StreamingBench rows for the ThinkStream recurrent RL evaluator.

This converts mjuicem/StreamingBench CSV rows into the same multi-question
trajectory JSONL/parquet schema used by scripts/eval/ovo/run_rl_recurrent_eval.sh.
It intentionally includes Proactive Output rows, which the vLLM base runner
skips because they are not multiple-choice questions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkstream.eval.prompt_contract import build_streaming_query_meta  # noqa: E402


TIME_RE = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)$")
QUESTION_RE = re.compile(r"^(?P<family>.+?)_sample_(?P<sample>\d+)_")
SAMPLE_RE = re.compile(r"sample_(?P<sample>\d+)")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BenchRow:
    idx: int
    csv_name: str
    question_id: str
    family: str
    sample_id: int
    task_type: str
    question: str
    ask_sec: int
    answer_sec: int
    answer: str
    options: List[str]
    temporal_clue_type: str
    frames_required: str
    source_video: Path
    video_rel: str


def _parse_time(value: str) -> int:
    text = str(value or "").strip()
    match = TIME_RE.match(text)
    if not match:
        raise ValueError(f"bad time value: {value!r}")
    h = int(match.group("h") or 0)
    m = int(match.group("m") or 0)
    s = int(match.group("s") or 0)
    return h * 3600 + m * 60 + s


def _parse_options(value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(str(value or ""))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _question_family_sample(qid: str, task_type: str, fallback_idx: int) -> Tuple[str, int]:
    match = QUESTION_RE.match(str(qid or ""))
    if match:
        return match.group("family"), int(match.group("sample"))
    nums = re.findall(r"\d+", str(qid or ""))
    sample_id = int(nums[-1]) if nums else fallback_idx
    return task_type or "StreamingBench", sample_id


class VideoResolver:
    def __init__(self, video_root: Path):
        self.video_root = video_root
        self._by_sample: Dict[int, List[Path]] = defaultdict(list)
        for path in sorted(video_root.glob("**/sample_*/*.mp4")):
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            match = SAMPLE_RE.search(str(path))
            if match:
                self._by_sample[int(match.group("sample"))].append(path)

    def resolve(self, *, family: str, task_type: str, sample_id: int) -> Optional[Path]:
        candidates = self._by_sample.get(int(sample_id), [])
        if not candidates:
            return None
        family_norm = _normalize(family)
        task_norm = _normalize(task_type)
        best: Optional[Tuple[int, Path]] = None
        for path in candidates:
            parent_norm = _normalize(" ".join(path.parts))
            score = 0
            if family_norm and family_norm in parent_norm:
                score += 1000
            if task_norm and task_norm in parent_norm:
                score += 500
            for token in re.findall(r"[A-Za-z]+", family + " " + task_type):
                if _normalize(token) in parent_norm:
                    score += 1
            if best is None or score > best[0]:
                best = (score, path)
        if best and best[0] > 0:
            return best[1]
        return candidates[0] if len(candidates) == 1 else None


def _frame_dir_exists(frames_root: Path, video_rel: str) -> bool:
    vp = Path(video_rel)
    candidates = [
        frames_root / vp.parent,
        frames_root / vp.with_suffix(""),
        frames_root / vp.stem,
        frames_root / vp.with_suffix("").name,
    ]
    return any(c.exists() and any(c.glob("frame_*.jpg")) for c in candidates)


def load_rows(csv_dir: Path, video_root: Path, frames_root: Path) -> Tuple[List[BenchRow], Dict[str, Any]]:
    resolver = VideoResolver(video_root)
    rows: List[BenchRow] = []
    skipped = Counter()
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw_i, raw in enumerate(reader):
                task_type = str(raw.get("task_type") or "").strip()
                qid = str(raw.get("question_id") or "").strip()
                family, sample_id = _question_family_sample(qid, task_type, raw_i + 1)
                try:
                    ask_sec = _parse_time(str(raw.get("time_stamp") or ""))
                except ValueError:
                    skipped["bad_time"] += 1
                    continue
                answer_sec = ask_sec
                answer = str(raw.get("answer") or "").strip()
                options = _parse_options(str(raw.get("options") or ""))
                if not answer:
                    answer = str(raw.get("ground_truth_output") or "").strip()
                    try:
                        answer_sec = _parse_time(str(raw.get("ground_truth_time_stamp") or ""))
                    except ValueError:
                        answer_sec = ask_sec
                if not answer:
                    skipped["missing_answer"] += 1
                    continue
                video = resolver.resolve(family=family, task_type=task_type, sample_id=sample_id)
                if video is None:
                    skipped["missing_video"] += 1
                    continue
                try:
                    video_rel = video.relative_to(video_root).as_posix()
                except ValueError:
                    video_rel = video.as_posix()
                if not _frame_dir_exists(frames_root, video_rel):
                    skipped["missing_frames"] += 1
                    continue
                if options and answer not in set(LETTERS[: len(options)]):
                    skipped["bad_mc_answer"] += 1
                    continue
                rows.append(BenchRow(
                    idx=len(rows),
                    csv_name=csv_path.name,
                    question_id=qid,
                    family=family,
                    sample_id=sample_id,
                    task_type=task_type,
                    question=str(raw.get("question") or "").strip(),
                    ask_sec=int(ask_sec),
                    answer_sec=int(max(ask_sec, answer_sec)),
                    answer=answer,
                    options=options,
                    temporal_clue_type=str(raw.get("temporal_clue_type") or "").strip(),
                    frames_required=str(raw.get("frames_required") or "").strip(),
                    source_video=video,
                    video_rel=video_rel,
                ))
    return rows, {"skipped": dict(skipped)}


def _strip_option_label(option: str) -> str:
    return re.sub(r"^\s*[A-E][\).:]\s*", "", str(option or "")).strip()


def _answer_form(row: BenchRow) -> str:
    if row.options:
        return "multiple_choice"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", row.answer.strip()):
        return "number"
    return "short_exact"


def _support_start(row: BenchRow, window_sec: int) -> int:
    if row.temporal_clue_type == "Concurrent":
        return row.ask_sec
    if row.temporal_clue_type == "Subsequent":
        return row.ask_sec
    return max(0, row.ask_sec - int(window_sec))


def _question_payload(row: BenchRow, *, support_window_sec: int) -> Dict[str, Any]:
    form = _answer_form(row)
    q: Dict[str, Any] = {
        "card_id": row.question_id or f"streamingbench:{row.idx}",
        "family": row.task_type,
        "family_name": row.task_type,
        "category": row.csv_name.removesuffix(".csv"),
        "skill": "streamingbench",
        "question": row.question,
        "ask_chunk": int(row.ask_sec),
        "ask_chunks": [int(row.ask_sec)],
        "answer_chunks": [int(row.answer_sec)],
        "per_emit_answers": [{"chunk": int(row.answer_sec), "value": row.answer}],
        "gold_answer": row.answer,
        "answer_form": form,
        "support_chunks": list(range(_support_start(row, support_window_sec), row.answer_sec + 1)),
        "streamingbench_csv": row.csv_name,
        "streamingbench_question_id": row.question_id,
        "streamingbench_family": row.family,
        "streamingbench_sample_id": row.sample_id,
        "streamingbench_task_type": row.task_type,
        "streamingbench_temporal_clue_type": row.temporal_clue_type,
        "streamingbench_frames_required": row.frames_required,
        "streamingbench_ask_sec": row.ask_sec,
        "streamingbench_answer_sec": row.answer_sec,
        "open_until": int(row.answer_sec),
    }
    if row.options:
        q["options"] = list(row.options)
        q["correct_option"] = row.answer
        idx = LETTERS.index(row.answer) if row.answer in LETTERS else -1
        if 0 <= idx < len(row.options):
            q["correct_answer_text"] = _strip_option_label(row.options[idx])
    meta = build_streaming_query_meta(q, answer_form=form)
    for key in ("answer_instruction", "answer_style"):
        if key in meta:
            q[key] = meta[key]
    return q


def _bounds(row: BenchRow, *, support_window_sec: int, min_span: int, max_span: int, post_context: int) -> Tuple[int, int, bool]:
    min_start = _support_start(row, support_window_sec)
    end = int(row.answer_sec) + int(post_context)
    if row.temporal_clue_type == "Subsequent":
        start = max(0, row.ask_sec - 1)
    else:
        start = max(0, min(min_start, end - max_span + 1))
    if start == row.ask_sec and start > 0:
        start -= 1
    if end - start + 1 < min_span:
        end = start + min_span - 1
    return int(start), int(end), bool(end - start + 1 > max_span)


def _trajectory(row: BenchRow, *, support_window_sec: int, min_span: int, max_span: int, post_context: int) -> Dict[str, Any]:
    q = _question_payload(row, support_window_sec=support_window_sec)
    start, end, span_exceeded = _bounds(
        row,
        support_window_sec=support_window_sec,
        min_span=min_span,
        max_span=max_span,
        post_context=post_context,
    )
    tid = (
        f"streamingbench__{Path(row.video_rel).with_suffix('').as_posix().replace('/', '__')}"
        f"__{row.idx:05d}"
    )
    return {
        "trajectory_id": tid,
        "video_id": tid,
        "source_video_path": row.video_rel,
        "video_path": row.video_rel,
        "segment_start_chunk": start,
        "segment_end_chunk": end,
        "questions": [q],
        "gold_action_per_chunk": {str(int(row.answer_sec)): "response"},
        "offline_compress_chunks": [],
        "samples": [],
        "stats": {
            "n_chunks_covered": end + 1,
            "chunk_idx_max": end,
            "n_questions": 1,
        },
        "ovo_split_meta": {
            "benchmark": "StreamingBench",
            "source_csv": row.csv_name,
            "task_type": row.task_type,
            "temporal_clue_type": row.temporal_clue_type,
            "frames_required": row.frames_required,
            "sample_id": row.sample_id,
            "question_id": row.question_id,
            "segment_start_chunk": start,
            "segment_end_chunk": end,
            "span_exceeded_soft_limit": span_exceeded,
            "benchmark_track": "strict_window_with_future_wait",
        },
    }


def _sample_rows(rows: List[BenchRow], *, per_task: int, seed: int, limit: int) -> List[BenchRow]:
    rng = random.Random(seed)
    by_task: Dict[str, List[BenchRow]] = defaultdict(list)
    for row in rows:
        by_task[row.task_type].append(row)
    selected: List[BenchRow] = []
    for task in sorted(by_task):
        candidates = sorted(
            by_task[task],
            key=lambda r: (
                r.answer_sec - r.ask_sec,
                r.csv_name,
                r.sample_id,
                r.question_id,
            ),
        )
        if per_task > 0 and len(candidates) > per_task:
            head = candidates[: max(per_task * 3, per_task)]
            candidates = rng.sample(head, per_task)
        selected.extend(candidates[:per_task] if per_task > 0 else candidates)
    selected = sorted(selected, key=lambda r: (r.task_type, r.csv_name, r.sample_id, r.question_id))
    return selected[:limit] if limit > 0 else selected


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(jsonl_path: Path, parquet_path: Path, *, max_questions_per_traj: int) -> int:
    import pandas as pd
    from scripts.agent_data.build_verl_parquet import _iter_rows_multi_q

    rows = list(
        _iter_rows_multi_q(
            jsonl_path,
            max_questions_per_traj=max_questions_per_traj,
            frame_protocol="video_meta",
            render_layout="standard_query_last",
            include_student_cache=False,
        )
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return len(rows)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rows, load_summary = load_rows(args.csv_dir, args.video_root, args.frames_root)
    selected = _sample_rows(
        rows,
        per_task=max(0, int(args.sample_per_task_type)),
        seed=int(args.seed),
        limit=max(0, int(args.limit)),
    )
    trajectories = [
        _trajectory(
            row,
            support_window_sec=int(args.support_window_sec),
            min_span=int(args.min_span_chunks),
            max_span=int(args.max_span_chunks),
            post_context=int(args.post_context_chunks),
        )
        for row in selected
    ]
    out_jsonl = Path(args.out_jsonl)
    _write_jsonl(out_jsonl, trajectories)

    summary: Dict[str, Any] = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "trajectories": len(trajectories),
        "questions": sum(len(t.get("questions") or []) for t in trajectories),
        "tasks": dict(Counter(r.task_type for r in selected)),
        "csv": dict(Counter(r.csv_name for r in selected)),
        "temporal_clue_type": dict(Counter(r.temporal_clue_type for r in selected)),
        "answer_form": dict(Counter(_answer_form(r) for r in selected)),
        "span_over_max": sum(
            1 for t in trajectories if (t.get("ovo_split_meta") or {}).get("span_exceeded_soft_limit")
        ),
        "max_span": max(
            (
                int(t["segment_end_chunk"]) - int(t["segment_start_chunk"]) + 1
                for t in trajectories
            ),
            default=0,
        ),
        "sample_per_task_type": int(args.sample_per_task_type),
        "limit": int(args.limit),
        **load_summary,
    }
    if args.out_parquet:
        summary["parquet_rows"] = _write_parquet(
            out_jsonl,
            Path(args.out_parquet),
            max_questions_per_traj=1,
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-parquet", default="")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--sample-per-task-type", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--support-window-sec", type=int, default=32)
    ap.add_argument("--min-span-chunks", type=int, default=25)
    ap.add_argument("--max-span-chunks", type=int, default=45)
    ap.add_argument("--post-context-chunks", type=int, default=2)
    args = ap.parse_args()

    summary = build(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary_out:
        p = Path(args.summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
