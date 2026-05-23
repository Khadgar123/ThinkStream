#!/usr/bin/env python3
"""Export structured statistics for agent_v5 batches.

The final trajectory JSONL files are very large because every line contains
all rendered samples. This exporter parses only the top-level metadata,
questions, and stats fields so that it can build paper/plot friendly tables
without materializing prompts, memories, or frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


QUESTION_TYPE_FIELDS = [
    "category",
    "skill",
    "family",
    "family_name",
    "task_family",
    "task_subtype",
    "slot_group",
    "slot_subtype",
    "temporal_bucket",
    "timing_type",
    "benchmark_source",
    "benchmark_task",
    "ovo_task",
    "target_ovo_task",
    "temporal_role",
    "support_policy",
    "card_support_policy",
    "question_way",
    "evidence_type",
    "answer_behavior",
    "answer_form",
    "question_type",
    "availability",
    "required_answer_mode",
    "answer_mode_reason",
]

BOOLEAN_TYPE_FIELDS = [
    "ours_unique",
    "recall_eligible",
    "state_memory_required",
]

LIST_TYPE_FIELDS = [
    "allowed_support_policies",
    "legal_answer_modes",
    "forbidden_answer_modes",
]

TEXT_FIELDS = [
    "question",
    "canonical_answer",
    "correct_answer_text",
    "correct_option",
    "gold_answer",
    "answer_style",
    "answer_instruction",
    "question_goal",
    "placement_hint",
    "readable_task_name",
    "legacy_family_id",
]

QUESTION_DETAIL_FIELDS = [
    "options",
    "accepted_answers",
    "per_emit_answers",
    "ask_chunks",
    "expected_answer_chunks",
    "missing_answer_chunks",
    "gold_compress_chunks",
]

CANDIDATE_DETAIL_FIELDS = [
    "evidence_window",
    "grounding_frames",
    "operation",
    "question_style",
    "recall_query",
    "slot_id",
]

TRAJECTORY_RE = re.compile(
    r'^\{"video_id": "(?P<video_id>(?:[^"\\]|\\.)*)", '
    r'"trajectory_id": "(?P<trajectory_id>(?:[^"\\]|\\.)*)", '
    r'"video_path": "(?P<video_path>(?:[^"\\]|\\.)*)", '
    r'"card_id": "(?P<card_id>(?:[^"\\]|\\.)*)", '
    r'"protocol_version": "(?P<protocol_version>(?:[^"\\]|\\.)*)"'
)


def parse_batches(value: str) -> list[str]:
    batches: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.replace("batch", ""))
            end = int(end_s.replace("batch", ""))
            batches.extend(f"batch{i}" for i in range(start, end + 1))
        else:
            idx = part.replace("batch", "")
            batches.append(f"batch{int(idx)}")
    return batches


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def json_loads_string(value: str) -> str:
    return json.loads(f'"{value}"')


def split_name(path: Path) -> str:
    return path.name.replace("_trajectories.jsonl", "")


def source_dataset(video_path: str) -> str:
    marker = "/data/datasets/"
    if marker in video_path:
        return video_path.split(marker, 1)[1].split("/", 1)[0]
    return "unknown"


def source_relpath(video_path: str) -> str:
    marker = "/data/datasets/"
    if marker in video_path:
        return video_path.split(marker, 1)[1]
    return video_path


def extract_top_level_json_value(line: str, key: str, *, from_right: bool = False) -> Any:
    needle = f'"{key}": '
    pos = line.rfind(needle) if from_right else line.find(needle)
    if pos < 0:
        return None
    idx = pos + len(needle)
    end = find_json_value_end(line, idx)
    return json.loads(line[idx:end])


def find_json_value_end(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isspace():
        idx += 1
    first = text[idx]
    if first in "[{":
        stack = [first]
        idx += 1
        in_string = False
        escape = False
        while idx < len(text):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    opener = stack.pop()
                    if (opener, ch) not in {("[", "]"), ("{", "}")}:
                        raise ValueError(f"mismatched JSON brackets near index {idx}")
                    if not stack:
                        return idx + 1
            idx += 1
        raise ValueError("unterminated JSON value")
    if first == '"':
        idx += 1
        escape = False
        while idx < len(text):
            ch = text[idx]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                return idx + 1
            idx += 1
        raise ValueError("unterminated JSON string")
    while idx < len(text) and text[idx] not in ",}\n\r":
        idx += 1
    return idx


def frame_info(batch_dir: Path, video_id: str) -> dict[str, Any]:
    frame_dir = batch_dir / "frames" / video_id
    info: dict[str, Any] = {
        "frame_dir": str(frame_dir),
        "frame_dir_exists": frame_dir.is_dir(),
        "frame_count": "",
        "frame_fps": "",
        "frame_duration_sec": "",
    }
    if not frame_dir.is_dir():
        return info

    fps_path = frame_dir / ".fps"
    fps = None
    if fps_path.exists():
        try:
            fps = float(fps_path.read_text().strip())
        except ValueError:
            fps = None

    count = 0
    with os.scandir(frame_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith("frame_"):
                count += 1

    info["frame_count"] = count
    if fps:
        info["frame_fps"] = fps
        info["frame_duration_sec"] = round(count / fps, 3)
    return info


def distribution(values: Iterable[float]) -> dict[str, Any]:
    vals = sorted(v for v in values if v != "" and v is not None)
    if not vals:
        return {}
    n = len(vals)

    def pct(q: float) -> Any:
        return vals[int((n - 1) * q)]

    return {
        "count": n,
        "min": vals[0],
        "p25": pct(0.25),
        "median": pct(0.50),
        "mean": round(sum(vals) / n, 3),
        "p75": pct(0.75),
        "max": vals[-1],
    }


def safe_ratio(numerator: Any, denominator: Any) -> Any:
    if numerator == "" or denominator == "" or denominator in (0, 0.0):
        return ""
    return round(float(numerator) / float(denominator), 6)


def time_bucket(value: Any, width: int) -> str:
    if value == "":
        return ""
    start = int(float(value) // width) * width
    return f"{start}-{start + width}"


def duration_bucket(value: Any) -> str:
    if value == "":
        return ""
    seconds = float(value)
    if seconds < 60:
        return "0-60"
    if seconds < 120:
        return "60-120"
    if seconds < 180:
        return "120-180"
    if seconds < 300:
        return "180-300"
    return "300+"


def position_decile(time_sec: Any, duration_sec: Any) -> Any:
    ratio = safe_ratio(time_sec, duration_sec)
    if ratio == "":
        return ""
    return min(9, max(0, int(float(ratio) * 10)))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/agent_v5"))
    parser.add_argument("--batches", default="1-10")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agent_v5/statistics/batch1_10_structured"),
    )
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    args = parser.parse_args()

    root = args.root
    batches = parse_batches(args.batches)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    video_agg: dict[str, dict[str, Any]] = {}

    count_tables: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    selected_card_ids: set[str] = set()
    selected_batch_card_ids: set[tuple[str, str]] = set()
    duplicate_video_tracker: Counter[str] = Counter()

    # First pass: final trajectories and actual questions.
    for batch in batches:
        batch_dir = root / batch
        final_dir = batch_dir / "final"
        for path in sorted(final_dir.glob("*_trajectories.jsonl")):
            split = split_name(path)
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.strip("\x00\r\n ")
                    if not line:
                        continue
                    header = TRAJECTORY_RE.match(line)
                    if not header:
                        raise ValueError(f"could not parse trajectory header: {path}:{line_no}")
                    groups = header.groupdict()
                    video_id = json_loads_string(groups["video_id"])
                    trajectory_id = json_loads_string(groups["trajectory_id"])
                    video_path = json_loads_string(groups["video_path"])
                    first_card_id = json_loads_string(groups["card_id"])
                    protocol_version = json_loads_string(groups["protocol_version"])
                    questions = extract_top_level_json_value(line, "questions")
                    stats = extract_top_level_json_value(line, "stats", from_right=True) or {}
                    actions = stats.get("actions") or {}
                    frame = frame_info(batch_dir, video_id)
                    src_dataset = source_dataset(video_path)
                    dataset_counts[src_dataset] += 1

                    n_questions = int(stats.get("n_questions") or len(questions or []))
                    n_samples = int(stats.get("n_samples") or sum(actions.values()))
                    n_chunks = int(stats.get("n_chunks_covered") or 0)
                    chunk_duration_sec = round(n_chunks * args.chunk_sec, 3)
                    frame_duration_sec = frame.get("frame_duration_sec", "")
                    video_duration_sec = (
                        frame_duration_sec if frame_duration_sec != "" else chunk_duration_sec
                    )
                    split_counts[split] += 1
                    batch_counts[batch] += 1
                    action_counts.update(actions)
                    duplicate_video_tracker[video_id] += 1

                    trow = {
                        "batch": batch,
                        "split": split,
                        "line_no": line_no,
                        "video_id": video_id,
                        "trajectory_id": trajectory_id,
                        "first_card_id": first_card_id,
                        "protocol_version": protocol_version,
                        "video_path": video_path,
                        "source_dataset": src_dataset,
                        "source_relpath": source_relpath(video_path),
                        "n_questions": n_questions,
                        "n_samples": n_samples,
                        "n_chunks_covered": n_chunks,
                        "chunk_sec": args.chunk_sec,
                        "chunk_duration_sec": chunk_duration_sec,
                        "video_duration_sec": video_duration_sec,
                        "duration_bucket_sec": duration_bucket(video_duration_sec),
                        "questions_per_min": safe_ratio(n_questions * 60, video_duration_sec),
                        "silent_actions": actions.get("silent", 0),
                        "response_actions": actions.get("response", 0),
                        "compress_actions": actions.get("compress", 0),
                        "recall_actions": actions.get("recall", 0),
                        **frame,
                    }
                    trajectory_rows.append(trow)

                    agg = video_agg.setdefault(
                        video_id,
                        {
                            "video_id": video_id,
                            "video_path": video_path,
                            "source_dataset": src_dataset,
                            "source_relpath": source_relpath(video_path),
                            "batches": set(),
                            "splits": set(),
                            "trajectory_count": 0,
                            "actual_question_count": 0,
                            "candidate_card_count": 0,
                            "n_samples": 0,
                            "silent_actions": 0,
                            "response_actions": 0,
                            "compress_actions": 0,
                            "recall_actions": 0,
                            "n_chunks_covered_max": 0,
                            "chunk_duration_sec_max": 0,
                            "frame_count_max": 0,
                            "frame_fps": frame.get("frame_fps", ""),
                            "frame_duration_sec_max": 0,
                            "frame_dir_any": frame.get("frame_dir", ""),
                        },
                    )
                    agg["batches"].add(batch)
                    agg["splits"].add(split)
                    agg["trajectory_count"] += 1
                    agg["actual_question_count"] += n_questions
                    agg["n_samples"] += n_samples
                    agg["silent_actions"] += actions.get("silent", 0)
                    agg["response_actions"] += actions.get("response", 0)
                    agg["compress_actions"] += actions.get("compress", 0)
                    agg["recall_actions"] += actions.get("recall", 0)
                    agg["n_chunks_covered_max"] = max(agg["n_chunks_covered_max"], n_chunks)
                    agg["chunk_duration_sec_max"] = max(agg["chunk_duration_sec_max"], chunk_duration_sec)
                    if isinstance(frame.get("frame_count"), int):
                        agg["frame_count_max"] = max(agg["frame_count_max"], frame["frame_count"])
                    if isinstance(frame.get("frame_duration_sec"), float):
                        agg["frame_duration_sec_max"] = max(
                            agg["frame_duration_sec_max"], frame["frame_duration_sec"]
                        )

                    for q_idx, q in enumerate(questions or []):
                        card_id = q.get("card_id", "")
                        if card_id:
                            selected_card_ids.add(card_id)
                            selected_batch_card_ids.add((batch, card_id))
                        for field in QUESTION_TYPE_FIELDS:
                            count_tables[f"question.{field}"][str(q.get(field, ""))] += 1
                        for field in BOOLEAN_TYPE_FIELDS:
                            count_tables[f"question.{field}"][str(bool(q.get(field)))] += 1

                        ask_chunk = q.get("ask_chunk")
                        answer_chunks = q.get("answer_chunks") or q.get("expected_answer_chunks") or []
                        support_chunks = q.get("support_chunks") or []
                        per_emit_answers = q.get("per_emit_answers") or []
                        first_answer_chunk = min(answer_chunks) if answer_chunks else ""
                        support_start = min(support_chunks) if support_chunks else ""
                        support_end = max(support_chunks) if support_chunks else ""
                        verification = q.get("verification") or {}
                        ask_time_sec = (
                            round(ask_chunk * args.chunk_sec, 3) if ask_chunk is not None else ""
                        )
                        first_answer_time_sec = (
                            round(first_answer_chunk * args.chunk_sec, 3)
                            if first_answer_chunk != ""
                            else ""
                        )
                        support_start_time_sec = (
                            round(support_start * args.chunk_sec, 3)
                            if support_start != ""
                            else ""
                        )
                        support_end_time_sec = (
                            round(support_end * args.chunk_sec, 3) if support_end != "" else ""
                        )
                        ask_decile = position_decile(ask_time_sec, video_duration_sec)
                        count_tables["question.source_dataset"][src_dataset] += 1
                        count_tables["question.split"][split] += 1
                        count_tables["question.options_count"][str(len(q.get("options") or []))] += 1
                        count_tables["question.emit_count"][str(len(per_emit_answers))] += 1
                        count_tables["question.support_chunk_count"][str(len(support_chunks))] += 1
                        count_tables["question.ask_time_bucket_30s"][
                            time_bucket(ask_time_sec, 30)
                        ] += 1
                        count_tables["question.ask_position_decile"][str(ask_decile)] += 1
                        if q.get("correct_option", ""):
                            count_tables["question.correct_option"][str(q.get("correct_option"))] += 1

                        qrow = {
                            "batch": batch,
                            "split": split,
                            "video_id": video_id,
                            "source_dataset": src_dataset,
                            "source_relpath": source_relpath(video_path),
                            "video_path": video_path,
                            "trajectory_id": trajectory_id,
                            "question_index": q_idx,
                            "card_id": card_id,
                            "n_chunks_covered": n_chunks,
                            "chunk_duration_sec": chunk_duration_sec,
                            "frame_duration_sec": frame_duration_sec,
                            "video_duration_sec": video_duration_sec,
                            "duration_bucket_sec": duration_bucket(video_duration_sec),
                            "ask_chunk": ask_chunk if ask_chunk is not None else "",
                            "ask_time_sec": ask_time_sec,
                            "ask_time_bucket_30s": time_bucket(ask_time_sec, 30),
                            "ask_time_norm": safe_ratio(ask_time_sec, video_duration_sec),
                            "ask_position_decile": ask_decile,
                            "answer_chunks": answer_chunks,
                            "first_answer_chunk": first_answer_chunk,
                            "first_answer_time_sec": first_answer_time_sec,
                            "first_answer_time_norm": safe_ratio(
                                first_answer_time_sec, video_duration_sec
                            ),
                            "answer_delay_chunks": first_answer_chunk - ask_chunk
                            if first_answer_chunk != "" and ask_chunk is not None
                            else "",
                            "support_chunks": support_chunks,
                            "support_start_chunk": support_start,
                            "support_end_chunk": support_end,
                            "support_start_time_sec": support_start_time_sec,
                            "support_end_time_sec": support_end_time_sec,
                            "support_start_time_norm": safe_ratio(
                                support_start_time_sec, video_duration_sec
                            ),
                            "support_end_time_norm": safe_ratio(
                                support_end_time_sec, video_duration_sec
                            ),
                            "ask_minus_support_end_chunks": ask_chunk - support_end
                            if ask_chunk is not None and support_end != ""
                            else "",
                            "support_chunk_count": len(support_chunks),
                            "emit_count": len(per_emit_answers),
                            "options_count": len(q.get("options") or []),
                            "verification_passed": verification.get("passed", ""),
                            "verification_fail_reasons": verification.get("fail_reasons", []),
                        }
                        for field in (
                            QUESTION_TYPE_FIELDS
                            + BOOLEAN_TYPE_FIELDS
                            + LIST_TYPE_FIELDS
                            + TEXT_FIELDS
                            + QUESTION_DETAIL_FIELDS
                        ):
                            qrow[field] = q.get(field, "")
                        question_rows.append(qrow)

    duplicate_video_ids = {video_id for video_id, count in duplicate_video_tracker.items() if count > 1}
    for row in trajectory_rows:
        row["duplicate_video_id"] = row["video_id"] in duplicate_video_ids

    # Second pass: candidate cards and selected flags.
    candidate_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for batch in batches:
        task_card_dir = root / batch / "task_cards"
        for path in sorted(task_card_dir.glob("*.json")):
            video_id = path.stem
            cards = json.loads(path.read_text(encoding="utf-8"))
            video_meta = video_agg.get(video_id, {})
            candidate_source_dataset = video_meta.get("source_dataset", "")
            candidate_source_relpath = video_meta.get("source_relpath", "")
            if video_id in video_agg:
                video_agg[video_id]["candidate_card_count"] += len(cards)
            for card_idx, card in enumerate(cards):
                card_id = card.get("card_id", "")
                for field in QUESTION_TYPE_FIELDS:
                    candidate_type_counts[f"candidate.{field}"][str(card.get(field, ""))] += 1
                for field in BOOLEAN_TYPE_FIELDS:
                    candidate_type_counts[f"candidate.{field}"][str(bool(card.get(field)))] += 1
                candidate_type_counts["candidate.source_dataset"][candidate_source_dataset] += 1
                candidate_type_counts["candidate.options_count"][
                    str(len(card.get("options") or []))
                ] += 1
                candidate_type_counts["candidate.support_chunk_count"][
                    str(len(card.get("support_chunks") or []))
                ] += 1
                candidate_type_counts["candidate.gold_emit_count"][
                    str(len(card.get("gold_emits") or []))
                ] += 1
                if card.get("correct_option", ""):
                    candidate_type_counts["candidate.correct_option"][
                        str(card.get("correct_option"))
                    ] += 1
                crow = {
                    "batch": batch,
                    "video_id": video_id,
                    "source_dataset": candidate_source_dataset,
                    "source_relpath": candidate_source_relpath,
                    "candidate_index": card_idx,
                    "card_id": card_id,
                    "selected_in_same_batch": (batch, card_id) in selected_batch_card_ids,
                    "selected_in_any_batch": card_id in selected_card_ids,
                    "support_chunks": card.get("support_chunks", []),
                    "support_chunk_count": len(card.get("support_chunks") or []),
                    "gold_emits": card.get("gold_emits", []),
                    "gold_emit_count": len(card.get("gold_emits") or []),
                    "options_count": len(card.get("options") or []),
                }
                for field in (
                    QUESTION_TYPE_FIELDS
                    + BOOLEAN_TYPE_FIELDS
                    + LIST_TYPE_FIELDS
                    + TEXT_FIELDS
                    + QUESTION_DETAIL_FIELDS
                    + CANDIDATE_DETAIL_FIELDS
                ):
                    crow[field] = card.get(field, "")
                candidate_rows.append(crow)

    for key, counter in candidate_type_counts.items():
        count_tables[key].update(counter)

    video_rows: list[dict[str, Any]] = []
    for video_id, agg in sorted(video_agg.items()):
        row = dict(agg)
        row["batches"] = sorted(row["batches"])
        row["splits"] = sorted(row["splits"])
        row["duplicate_video_id"] = video_id in duplicate_video_ids
        row["candidate_selected_ratio"] = (
            round(row["actual_question_count"] / row["candidate_card_count"], 6)
            if row["candidate_card_count"]
            else ""
        )
        row["duration_bucket_sec"] = duration_bucket(row["frame_duration_sec_max"])
        row["questions_per_min"] = safe_ratio(
            row["actual_question_count"] * 60, row["frame_duration_sec_max"]
        )
        row["candidate_cards_per_min"] = safe_ratio(
            row["candidate_card_count"] * 60, row["frame_duration_sec_max"]
        )
        video_rows.append(row)

    count_rows: list[dict[str, Any]] = []
    for name, counter in sorted(count_tables.items()):
        entity, field = name.split(".", 1)
        for value, count in counter.most_common():
            count_rows.append({"entity": entity, "field": field, "value": value, "count": count})
    for split, count in split_counts.most_common():
        count_rows.append({"entity": "trajectory", "field": "split", "value": split, "count": count})
    for batch, count in batch_counts.most_common():
        count_rows.append({"entity": "trajectory", "field": "batch", "value": batch, "count": count})
    for dataset, count in dataset_counts.most_common():
        count_rows.append({"entity": "trajectory", "field": "source_dataset", "value": dataset, "count": count})
    for action, count in action_counts.most_common():
        count_rows.append({"entity": "action", "field": "action", "value": action, "count": count})

    selection_rate_rows: list[dict[str, Any]] = []
    candidate_fields_for_rates = sorted(
        name.split(".", 1)[1] for name in count_tables if name.startswith("candidate.")
    )
    for field in candidate_fields_for_rates:
        candidate_counter = count_tables.get(f"candidate.{field}", Counter())
        selected_counter = count_tables.get(f"question.{field}", Counter())
        for value, candidate_count in candidate_counter.most_common():
            selected_count = selected_counter.get(value, 0)
            selection_rate_rows.append(
                {
                    "field": field,
                    "value": value,
                    "candidate_count": candidate_count,
                    "selected_count": selected_count,
                    "selection_rate": safe_ratio(selected_count, candidate_count),
                }
            )

    trajectory_fields = [
        "batch",
        "split",
        "line_no",
        "video_id",
        "duplicate_video_id",
        "trajectory_id",
        "first_card_id",
        "protocol_version",
        "source_dataset",
        "source_relpath",
        "video_path",
        "n_questions",
        "n_samples",
        "n_chunks_covered",
        "chunk_sec",
        "chunk_duration_sec",
        "video_duration_sec",
        "duration_bucket_sec",
        "questions_per_min",
        "frame_count",
        "frame_fps",
        "frame_duration_sec",
        "frame_dir_exists",
        "frame_dir",
        "silent_actions",
        "response_actions",
        "compress_actions",
        "recall_actions",
    ]
    video_fields = [
        "video_id",
        "duplicate_video_id",
        "source_dataset",
        "source_relpath",
        "video_path",
        "batches",
        "splits",
        "trajectory_count",
        "actual_question_count",
        "candidate_card_count",
        "candidate_selected_ratio",
        "duration_bucket_sec",
        "questions_per_min",
        "candidate_cards_per_min",
        "n_samples",
        "n_chunks_covered_max",
        "chunk_duration_sec_max",
        "frame_count_max",
        "frame_fps",
        "frame_duration_sec_max",
        "frame_dir_any",
        "silent_actions",
        "response_actions",
        "compress_actions",
        "recall_actions",
    ]
    question_fields = [
        "batch",
        "split",
        "video_id",
        "source_dataset",
        "source_relpath",
        "video_path",
        "trajectory_id",
        "question_index",
        "card_id",
        "n_chunks_covered",
        "chunk_duration_sec",
        "frame_duration_sec",
        "video_duration_sec",
        "duration_bucket_sec",
        "ask_chunk",
        "ask_time_sec",
        "ask_time_bucket_30s",
        "ask_time_norm",
        "ask_position_decile",
        "answer_chunks",
        "first_answer_chunk",
        "first_answer_time_sec",
        "first_answer_time_norm",
        "answer_delay_chunks",
        "support_chunks",
        "support_start_chunk",
        "support_end_chunk",
        "support_start_time_sec",
        "support_end_time_sec",
        "support_start_time_norm",
        "support_end_time_norm",
        "ask_minus_support_end_chunks",
        "support_chunk_count",
        "emit_count",
        "options_count",
        "verification_passed",
        "verification_fail_reasons",
        *QUESTION_TYPE_FIELDS,
        *BOOLEAN_TYPE_FIELDS,
        *LIST_TYPE_FIELDS,
        *TEXT_FIELDS,
        *QUESTION_DETAIL_FIELDS,
    ]
    candidate_fields = [
        "batch",
        "video_id",
        "source_dataset",
        "source_relpath",
        "candidate_index",
        "card_id",
        "selected_in_same_batch",
        "selected_in_any_batch",
        "support_chunks",
        "support_chunk_count",
        "gold_emits",
        "gold_emit_count",
        "options_count",
        *QUESTION_TYPE_FIELDS,
        *BOOLEAN_TYPE_FIELDS,
        *LIST_TYPE_FIELDS,
        *TEXT_FIELDS,
        *QUESTION_DETAIL_FIELDS,
        *CANDIDATE_DETAIL_FIELDS,
    ]

    write_csv(out_dir / "trajectories.csv", trajectory_rows, trajectory_fields)
    write_csv(out_dir / "videos.csv", video_rows, video_fields)
    write_csv(out_dir / "questions.csv", question_rows, question_fields)
    write_csv(out_dir / "candidate_task_cards.csv", candidate_rows, candidate_fields)
    write_csv(out_dir / "counts_long.csv", count_rows, ["entity", "field", "value", "count"])
    write_csv(
        out_dir / "selection_rates.csv",
        selection_rate_rows,
        ["field", "value", "candidate_count", "selected_count", "selection_rate"],
    )

    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for row in question_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "root": str(root),
        "batches": batches,
        "chunk_sec": args.chunk_sec,
        "files": {
            "trajectories_csv": str(out_dir / "trajectories.csv"),
            "videos_csv": str(out_dir / "videos.csv"),
            "questions_csv": str(out_dir / "questions.csv"),
            "questions_jsonl": str(out_dir / "questions.jsonl"),
            "candidate_task_cards_csv": str(out_dir / "candidate_task_cards.csv"),
            "counts_long_csv": str(out_dir / "counts_long.csv"),
            "selection_rates_csv": str(out_dir / "selection_rates.csv"),
        },
        "totals": {
            "trajectories": len(trajectory_rows),
            "unique_videos": len(video_rows),
            "duplicate_video_ids": len(duplicate_video_ids),
            "actual_questions": len(question_rows),
            "unique_actual_question_card_ids": len(selected_card_ids),
            "candidate_task_cards": len(candidate_rows),
            "actions": dict(action_counts),
        },
        "split_counts": dict(split_counts),
        "batch_counts": dict(batch_counts),
        "source_dataset_counts": dict(dataset_counts),
        "distributions": {
            "questions_per_trajectory": distribution(row["n_questions"] for row in trajectory_rows),
            "chunk_duration_sec": distribution(row["chunk_duration_sec"] for row in trajectory_rows),
            "frame_duration_sec": distribution(row["frame_duration_sec"] for row in trajectory_rows),
            "questions_per_min": distribution(row["questions_per_min"] for row in trajectory_rows),
            "ask_time_sec": distribution(row["ask_time_sec"] for row in question_rows),
            "ask_time_norm": distribution(row["ask_time_norm"] for row in question_rows),
            "answer_delay_chunks": distribution(row["answer_delay_chunks"] for row in question_rows),
            "support_chunk_count": distribution(row["support_chunk_count"] for row in question_rows),
            "candidate_selected_ratio": distribution(
                row["candidate_selected_ratio"] for row in video_rows
            ),
        },
        "top_counts": {
            name: dict(counter.most_common(50)) for name, counter in sorted(count_tables.items())
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# agent_v5 batch1-10 structured statistics

Generated from `{root}` batches `{','.join(batches)}`.

Files:
- `videos.csv`: one row per unique video id; source dataset, duration, split membership, actual/candidate question counts.
- `trajectories.csv`: one row per final trajectory; split, source, frame-derived duration, chunk duration, and action counts.
- `questions.csv`: one row per actual question that entered final trajectories; includes ask time, support/answer chunks, all taxonomy fields, and answer metadata.
- `questions.jsonl`: same information as `questions.csv`, easier to load with Python.
- `candidate_task_cards.csv`: one row per candidate card from `task_cards`; includes whether it was selected into final trajectories.
- `counts_long.csv`: tidy count table for quick plotting (`entity`, `field`, `value`, `count`).
- `selection_rates.csv`: candidate-to-selected ratios by taxonomy/source/answer fields.
- `summary.json`: totals, distributions, and top count tables.

Duration notes:
- `frame_duration_sec` is computed as extracted frame count divided by the `.fps` value in each frame directory.
- `chunk_duration_sec` is computed from final trajectory chunks with `chunk_sec={args.chunk_sec}`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"Wrote structured statistics to {out_dir}")


if __name__ == "__main__":
    main()
