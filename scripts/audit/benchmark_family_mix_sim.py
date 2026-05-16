"""Audit benchmark question mix and simulate current Pass3A slot mix.

This script is intentionally read-only. It reports:
- OVO-Bench answer forms, temporal buckets, and subtypes.
- StreamingBench answer forms, temporal buckets, and subtypes.
- Current ThinkStream family/slot mix over local batch evidence files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.agent_data.pass3_slot_planner import balanced_family_targets, build_pass3_slot_plan
from scripts.agent_data.pass3a_cards import PASS3A_TARGETS_BY_FAMILY
from scripts.agent_data.placement.design import (
    CURRENT_DIRECT_TARGET_FRACTION,
    FUTURE_DELAYED_TARGET_FRACTION,
    MULTI_ANSWER_TARGET_FRACTION,
    OURS_TARGET_FRACTION,
    PAST_STATE_DIRECT_TARGET_FRACTION,
    RECALL_TARGET_FRACTION,
    UNANSWERABLE_TARGET_FRACTION,
)
from scripts.agent_data.placement.llm_prompts import FAMILY_RULES, family_taxonomy
from scripts.audit.plan_pass3_slots_batch import _parse_batches


OVO_RT = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
OVO_BT = {"EPM", "ASI", "HLD"}
OVO_FT = {"REC", "SSR", "CRR"}


STREAM_TASK_TO_WAY = {
    "Object Perception": "object_perception",
    "Action Perception": "action_recognition",
    "Text-Rich Understanding": "text_readout",
    "Attribute Perception": "attribute_perception",
    "Spatial Understanding": "spatial_relation",
    "Counting": "visible_count",
    "Event Understanding": "event_understanding",
    "Causal Reasoning": "causal_reasoning",
    "Prospective Reasoning": "future_prediction",
    "Clips Summarize": "scene_summary",
    "Sequential Question Answering": "sequential_reference",
    "Emotion Recognition": "emotion_context",
    "Multimodal Alignment": "multimodal_alignment",
    "Scene Understanding": "scene_understanding",
    "Source Discrimination": "source_discrimination",
    "Misleading Context Recognition": "misleading_context",
    "Anomaly Context Understanding": "anomaly_context",
    "Proactive Output": "proactive_output",
    "Active Output": "proactive_output",
}


LIFECYCLE_TO_BUCKET = {
    "current_direct": "current_direct",
    "current_future_cue": "current_future_prediction",
    "future_delayed": "future_delayed",
    "past_visual_candidate": "past_single_candidate",
    "past_or_delayed_clue_candidate": "past_or_delayed_clue",
    "small_unanswerable_negative": "past_unanswerable",
    "state_memory_direct": "global_state_memory",
    "multi_local_or_global_count": "multi_count",
    "multi_local_status": "multi_status",
    "multi_local_sufficiency": "multi_sufficiency",
    "multi_local_live_narration": "multi_live_narration",
}


def pct(counter: Counter) -> Dict[str, str]:
    total = sum(counter.values()) or 1
    return {k: f"{v} ({v / total:.2%})" for k, v in counter.most_common()}


def answer_space_bucket(answer_form: str) -> str:
    af = str(answer_form or "")
    if af in {"short_exact", "descriptive", "number_or_short_exact"}:
        return "short_text"
    if af in {"number", "number_multi"}:
        return "number"
    if af in {"binary", "binary_multi", "binary_probe"}:
        return "binary"
    if af == "multiple_choice":
        return "multiple_choice"
    return af or "unknown"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ovo_answer_form(row: Dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    if task == "REC":
        return "number_multi"
    if task in {"SSR", "CRR"}:
        return "binary_multi" if task == "CRR" else "binary_probe"
    options = row.get("options")
    if isinstance(options, list) and options:
        return "multiple_choice"
    return "unknown"


def ovo_temporal_bucket(row: Dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    if task == "REC":
        return "multi_from_start_count"
    if task == "SSR":
        return "current_status_probe"
    if task == "CRR":
        return "future_delayed_sufficiency"
    if task == "FPD":
        return "current_future_prediction"
    if task == "HLD":
        return "past_unanswerable"
    if task in OVO_BT:
        return "past_visual_recall_candidate"
    if task in OVO_RT:
        return "current_direct"
    return "unknown"


def audit_ovo(path: Path) -> Dict[str, Any]:
    rows = load_json(path)
    task = Counter()
    answer_form = Counter()
    answer_space = Counter()
    temporal = Counter()
    subtype = Counter()
    response_units = Counter()
    response_answer_space = Counter()
    binary_label = Counter()
    number_label = Counter()
    multi_emits = []
    for row in rows:
        t = str(row.get("task") or "")
        af = ovo_answer_form(row)
        answer_space_name = answer_space_bucket(af)
        task[t] += 1
        answer_form[af] += 1
        answer_space[answer_space_name] += 1
        temporal[ovo_temporal_bucket(row)] += 1
        subtype[f"{t}:{ovo_temporal_bucket(row)}:{af}"] += 1
        if t in {"REC", "SSR", "CRR"}:
            infos = row.get("test_info") or []
            n = len(infos)
            response_units[t] += n
            response_answer_space[answer_space_name] += n
            if t in {"SSR", "CRR"}:
                for item in infos:
                    binary_label["Yes" if int(item.get("type", 0)) == 1 else "No"] += 1
            elif t == "REC":
                for item in infos:
                    number_label[str(item.get("count", ""))] += 1
            if n:
                multi_emits.append(n)
        else:
            response_units[t] += 1
            response_answer_space[answer_space_name] += 1
    return {
        "n": len(rows),
        "task": pct(task),
        "answer_form": pct(answer_form),
        "answer_space_question": pct(answer_space),
        "answer_space_response_units": pct(response_answer_space),
        "binary_label_response_units": pct(binary_label),
        "number_label_response_units": pct(number_label),
        "temporal": pct(temporal),
        "subtype": pct(subtype),
        "response_units_by_task": dict(response_units.most_common()),
        "multi_emit_mean": round(sum(multi_emits) / max(len(multi_emits), 1), 3),
        "multi_emit_p50": sorted(multi_emits)[len(multi_emits) // 2] if multi_emits else 0,
    }


def parse_options(raw: str) -> List[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def stream_answer_form(row: Dict[str, str]) -> str:
    if parse_options(row.get("options", "")):
        return "multiple_choice"
    out = str(row.get("ground_truth_output") or row.get("answer") or "").strip()
    return "number_or_short_exact" if out else "unknown"


def stream_answer_space(row: Dict[str, str]) -> str:
    if parse_options(row.get("options", "")):
        return "multiple_choice"
    out = str(row.get("ground_truth_output") or row.get("answer") or "").strip()
    if not out:
        return "unknown"
    return "number" if out.isdigit() else "short_text"


def stream_temporal_bucket(row: Dict[str, str]) -> str:
    clue = str(row.get("temporal_clue_type") or "").strip()
    if clue == "Concurrent":
        return "current_direct"
    if clue == "Prior":
        return "past_context"
    if clue == "Subsequent":
        return "future_delayed"
    return "unknown"


def audit_streaming(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    by_file: Dict[str, int] = {}
    for csv_path in sorted(path.glob("*.csv")):
        file_rows = list(csv.DictReader(csv_path.open(newline="")))
        rows.extend(file_rows)
        by_file[csv_path.name] = len(file_rows)
    answer_form = Counter()
    answer_space = Counter()
    temporal = Counter()
    frames = Counter()
    task_type = Counter()
    subtype = Counter()
    for row in rows:
        af = stream_answer_form(row)
        answer_space_name = stream_answer_space(row)
        tb = stream_temporal_bucket(row)
        tt = str(row.get("task_type") or "")
        way = STREAM_TASK_TO_WAY.get(tt, tt or "unknown")
        answer_form[af] += 1
        answer_space[answer_space_name] += 1
        temporal[tb] += 1
        frames[str(row.get("frames_required") or "")] += 1
        task_type[tt] += 1
        subtype[f"{way}:{tb}:{af}:{row.get('frames_required') or ''}"] += 1
    return {
        "n": len(rows),
        "by_file": by_file,
        "answer_form": pct(answer_form),
        "answer_space_question": pct(answer_space),
        "answer_space_response_units": pct(answer_space),
        "temporal": pct(temporal),
        "frames_required": pct(frames),
        "task_type": pct(task_type),
        "subtype": pct(subtype),
    }


def iter_evidence_files(root: Path, batches: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for batch in batches:
        bdir = root / batch
        edir = bdir / "evidence_1b"
        if not edir.exists():
            edir = bdir / "evidence_1a"
        if edir.exists():
            paths.extend(sorted(edir.glob("*.json")))
    return paths


def audit_ours(root: Path, batches_raw: str, *, mode: str, limit: int) -> Dict[str, Any]:
    paths = iter_evidence_files(root, _parse_batches(batches_raw))
    if limit:
        paths = paths[:limit]
    family = Counter()
    family_name = Counter()
    answer_form = Counter()
    answer_space = Counter()
    response_answer_space = Counter()
    question_way = Counter()
    lifecycle = Counter()
    bucket = Counter()
    task_family = Counter()
    task_subtype = Counter()
    timing_type = Counter()
    slot_group = Counter()
    slot_subtype = Counter()
    temporal_bucket = Counter()
    answer_behavior = Counter()
    style = Counter()
    ours = Counter()
    slots_per_video = []
    missing = 0
    for path in paths:
        try:
            evidence = load_json(path)
        except Exception:
            missing += 1
            continue
        if not isinstance(evidence, list) or not evidence:
            missing += 1
            continue
        video_id = path.stem
        targets = (
            balanced_family_targets(video_id, mode)
            if mode == "balanced"
            else {f: int(PASS3A_TARGETS_BY_FAMILY.get(f, 1)) for f in FAMILY_RULES}
        )
        slots = build_pass3_slot_plan(evidence, video_id, targets)
        slots_per_video.append(len(slots))
        for slot in slots:
            fam = str(slot.get("family") or "")
            rule = FAMILY_RULES.get(fam, {})
            af = str(slot.get("answer_form") or rule.get("answer_form") or "")
            answer_space_name = answer_space_bucket(af)
            tax = family_taxonomy(fam)
            family[fam] += 1
            family_name[str(tax.get("family_name") or fam)] += 1
            answer_form[af] += 1
            answer_space[answer_space_name] += 1
            response_answer_space[answer_space_name] += max(1, len(slot.get("answer_chunks") or []))
            question_way[str(slot.get("question_way") or "")] += 1
            lc = str(slot.get("lifecycle") or "")
            lifecycle[lc] += 1
            bucket[LIFECYCLE_TO_BUCKET.get(lc, lc or "unknown")] += 1
            task_family[str(slot.get("task_family") or slot.get("slot_group") or "")] += 1
            task_subtype[str(slot.get("task_subtype") or slot.get("slot_subtype") or "")] += 1
            timing_type[str(slot.get("timing_type") or slot.get("temporal_bucket") or "")] += 1
            slot_group[str(slot.get("slot_group") or "")] += 1
            slot_subtype[str(slot.get("slot_subtype") or "")] += 1
            temporal_bucket[str(slot.get("temporal_bucket") or "")] += 1
            answer_behavior[str(slot.get("answer_behavior") or "")] += 1
            style[str(slot.get("question_style") or "")] += 1
            ours["ours_unique" if tax.get("ours_unique") else "benchmark_like"] += 1
    return {
        "videos": len(paths),
        "missing_or_empty": missing,
        "slots_total": sum(slots_per_video),
        "slots_per_video_mean": round(sum(slots_per_video) / max(len(slots_per_video), 1), 3),
        "target_knobs": {
            "current_direct": CURRENT_DIRECT_TARGET_FRACTION,
            "past_state_direct": PAST_STATE_DIRECT_TARGET_FRACTION,
            "recall": RECALL_TARGET_FRACTION,
            "future_delayed": FUTURE_DELAYED_TARGET_FRACTION,
            "multi_answer": MULTI_ANSWER_TARGET_FRACTION,
            "unanswerable": UNANSWERABLE_TARGET_FRACTION,
            "ours_unique": OURS_TARGET_FRACTION,
        },
        "bucket": pct(bucket),
        "task_family": pct(task_family),
        "task_subtype": pct(task_subtype),
        "timing_type": pct(timing_type),
        "slot_group": pct(slot_group),
        "slot_subtype": pct(slot_subtype),
        "temporal_bucket": pct(temporal_bucket),
        "answer_behavior": pct(answer_behavior),
        "lifecycle": pct(lifecycle),
        "family": pct(family),
        "family_name": pct(family_name),
        "question_way": pct(question_way),
        "answer_form": pct(answer_form),
        "answer_space_question": pct(answer_space),
        "answer_space_response_units": pct(response_answer_space),
        "question_style": pct(style),
        "ours_mix": pct(ours),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovo", default="/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json")
    parser.add_argument("--streaming", default="/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/StreamingBench")
    parser.add_argument("--root", default="data/agent_v5")
    parser.add_argument("--batches", default="1-11")
    parser.add_argument("--mode", choices=["balanced", "full"], default="balanced")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    report = {
        "ovo": audit_ovo(Path(args.ovo)),
        "streamingbench": audit_streaming(Path(args.streaming)),
        "ours_slot_sim": audit_ours(Path(args.root), args.batches, mode=args.mode, limit=args.limit),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
