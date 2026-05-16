#!/usr/bin/env python3
"""Summarize benchmark question ways and evidence/timing types.

This is read-only and intentionally lexical. It is a planning/audit aid for
pass3 card prompts, not a ground-truth evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List


QUESTION_WAY_PATTERNS = [
    ("proactive_output", re.compile(r"^\s*when\b.*\boutput\b", re.I)),
    ("sequential_reference", re.compile(r"previous question|mentioned in the previous|first question", re.I)),
    ("repeated_count", re.compile(r"\bhow many\b|\bcount\b|\bso far\b", re.I)),
    ("text_readout", re.compile(r"\b(text|word|logo|number|sign|label|shown on|displayed)\b", re.I)),
    ("spatial_relation", re.compile(r"\bwhere\b|\bposition\b|\bleft\b|\bright\b|\bin relation to\b", re.I)),
    ("causal_intent", re.compile(r"\bwhy\b|\bcause\b|\bpurpose\b|\bintend\b|\btrying to\b", re.I)),
    ("future_prediction", re.compile(r"\bnext\b|\bwill\b|\blikely\b|\babout to\b", re.I)),
    ("person_identity_interaction", re.compile(r"^\s*who\b|\bperson\b|\binteracting\b", re.I)),
    ("emotion_context", re.compile(r"\bemotion\b|\bmood\b|\bfeeling\b|\bexcited\b|\bhappy\b|\banxious\b|\bflustered\b", re.I)),
    ("temporal_order", re.compile(r"\bbefore\b|\bafter\b|\bfirst\b|\bthen\b|\bprior\b|\bearlier\b|\blater\b|\bwhen\b|\bwhile\b", re.I)),
    ("current_status_probe", re.compile(r"\bcurrent(?:ly)?\b|\bright now\b|\bnow\b|\blatest\b|\bvisible\b", re.I)),
]


def _pct(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {str(k): round(v / total * 100.0, 2) for k, v in counter.most_common()}


def _way(question: str, *, task: str = "", temporal: str = "") -> str:
    q = str(question or "")
    task_l = str(task or "").lower()
    if task == "HLD":
        return "unanswerable_absence"
    if task == "REC" or "counting" in task_l:
        return "repeated_count"
    if task == "SSR":
        return "current_status_probe"
    if task == "CRR":
        return "evidence_sufficiency_probe"
    if task == "OCR" or "text-rich" in task_l:
        return "text_readout"
    if task == "ACR" or "action" in task_l:
        return "action_recognition"
    if task == "FPD" or "prospective" in task_l:
        return "future_prediction"
    if "summarize" in task_l or "context" in task_l:
        return "scene_summary"
    for name, pat in QUESTION_WAY_PATTERNS:
        if pat.search(q):
            return name
    if temporal == "Subsequent":
        return "proactive_output"
    return "object_attribute"


def _evidence_type(way: str) -> str:
    return {
        "object_attribute": "object_attribute_visual",
        "person_identity_interaction": "person_relation_visual",
        "action_recognition": "action_event_visual",
        "text_readout": "text_ocr_visual",
        "spatial_relation": "spatial_relation_visual",
        "temporal_order": "temporal_order_visual",
        "causal_intent": "causal_context_visual",
        "future_prediction": "future_cue_visual",
        "proactive_output": "future_trigger_visual",
        "repeated_count": "repeated_event_stream",
        "current_status_probe": "status_probe_stream",
        "evidence_sufficiency_probe": "status_probe_stream",
        "unanswerable_absence": "absence_unanswerable",
        "sequential_reference": "person_relation_visual",
        "emotion_context": "emotion_context_visual",
        "scene_summary": "global_context_memory",
    }.get(way, "object_attribute_visual")


def _timing_for_ovo(task: str, question: str) -> str:
    if task in {"REC", "SSR", "CRR"}:
        return "multi_answer"
    if task == "FPD":
        return "current_direct"
    q = str(question or "").lower()
    if re.search(r"\bbefore\b|\bafter\b|\bwhen\b|\bwhile\b|\bprevious|earlier|did i|was\b|were\b", q):
        return "past_visual_recall_candidate"
    return "current_direct"


_PAST_WORDS = re.compile(
    r"\bbefore\b|\bafter\b|\bwhen\b|\bwhile\b|previous|earlier|"
    r"just said|did i|was\b|were\b|had\b",
    re.I,
)
_RECENT_OR_CURRENT_WORDS = re.compile(
    r"\bcurrent(?:ly)?\b|\bright now\b|\bnow\b|\bvisible\b|"
    r"just occurred|just now",
    re.I,
)


def _ovo_eval_timing_bucket(task: str, question: str) -> str:
    """Benchmark-facing timing bucket for expanded OVO evaluation rows.

    OVO has no recall labels. ``past_history_unknown_gap`` means the question is
    about prior evidence, but the JSON does not expose enough evidence timing to
    decide whether our agent should answer from memory or call recall.
    """
    if task in {"REC", "SSR"}:
        return "multi_answer_state"
    if task == "CRR":
        return "future_delayed_multi_probe"
    if task == "FPD":
        return "current_direct_future_cue"
    if task in {"EPM", "ASI", "HLD"}:
        return "past_history_unknown_gap"
    if _PAST_WORDS.search(str(question or "")):
        return "past_history_unknown_gap"
    return "current_direct"


def _expand_ovo_eval_rows(row: Dict) -> List[Dict]:
    task = str(row.get("task") or "")
    infos = row.get("test_info") if task in {"REC", "SSR", "CRR"} else None
    if not isinstance(infos, list) or not infos:
        return [row]
    expanded = []
    for i, info in enumerate(infos):
        item = dict(row)
        item["eval_index"] = i
        item["eval_info"] = info
        expanded.append(item)
    return expanded


def _nested_pct(counter_by_key: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
    return {str(k): _pct(v) for k, v in sorted(counter_by_key.items())}


def audit_ovo(path: Path) -> Dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    task = Counter()
    way = Counter()
    evidence = Counter()
    timing = Counter()
    examples = defaultdict(list)
    for row in rows:
        t = str(row.get("task") or "")
        q = str(row.get("question") or "")
        w = _way(q, task=t)
        task[t] += 1
        way[w] += 1
        evidence[_evidence_type(w)] += 1
        timing[_timing_for_ovo(t, q)] += 1
        if len(examples[w]) < 3:
            examples[w].append(q)
    eval_task = Counter()
    eval_way = Counter()
    eval_timing = Counter()
    eval_timing_by_task = defaultdict(Counter)
    eval_timing_by_way = defaultdict(Counter)
    for row in rows:
        for item in _expand_ovo_eval_rows(row):
            t = str(item.get("task") or "")
            q = str(item.get("question") or "")
            w = _way(q, task=t)
            b = _ovo_eval_timing_bucket(t, q)
            eval_task[t] += 1
            eval_way[w] += 1
            eval_timing[b] += 1
            eval_timing_by_task[t][b] += 1
            eval_timing_by_way[w][b] += 1
    return {
        "n": len(rows),
        "task_pct": _pct(task),
        "question_way_pct": _pct(way),
        "evidence_type_pct": _pct(evidence),
        "timing_proxy_pct": _pct(timing),
        "eval_rows": {
            "n": sum(eval_task.values()),
            "task_pct": _pct(eval_task),
            "question_way_pct": _pct(eval_way),
            "timing_bucket_pct": _pct(eval_timing),
            "timing_by_task_pct": _nested_pct(eval_timing_by_task),
            "timing_by_question_way_pct": _nested_pct(eval_timing_by_way),
        },
        "examples": dict(examples),
    }


def _sec(ts: str) -> int:
    parts = [int(x) for x in str(ts).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


def _streamingbench_timing_bucket(row: Dict, question_way: str) -> str:
    """Benchmark-facing timing bucket for StreamingBench rows.

    ``temporal_clue_type`` is the dataset's real clue direction. It still does
    not expose exact evidence timestamps for Prior rows, so long-gap recall is
    reported as unknown-gap rather than as ground-truth recall.
    """
    temporal = str(row.get("temporal_clue_type") or "")
    question = str(row.get("question") or "")
    task_type = str(row.get("task_type") or "")
    if temporal == "Concurrent":
        return "current_direct"
    if temporal == "Subsequent":
        return "future_delayed"
    if question_way == "sequential_reference":
        return "past_direct_response_history"
    if _RECENT_OR_CURRENT_WORDS.search(question):
        return "past_direct_recent_or_current"
    if (
        task_type in {
            "Misleading Context Recognition",
            "Anomaly Context Understanding",
            "Source Discrimination",
            "Scene Understanding",
        }
        or question_way == "scene_summary"
    ):
        return "past_global_context_unknown_gap"
    return "past_visual_unknown_gap"


def audit_streamingbench(root: Path) -> Dict:
    files = sorted(root.glob("*.csv"))
    file_counts = Counter()
    task = Counter()
    temporal = Counter()
    frames = Counter()
    way = Counter()
    evidence = Counter()
    timing = Counter()
    timing_by_task = defaultdict(Counter)
    timing_by_way = defaultdict(Counter)
    examples = defaultdict(list)
    proactive_delays: List[int] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                q = row.get("question", "")
                tt = row.get("task_type", "")
                tmp = row.get("temporal_clue_type", "")
                w = _way(q, task=tt, temporal=tmp)
                file_counts[path.name] += 1
                task[tt] += 1
                temporal[tmp] += 1
                frames[row.get("frames_required", "")] += 1
                way[w] += 1
                evidence[_evidence_type(w)] += 1
                b = _streamingbench_timing_bucket(row, w)
                timing[b] += 1
                timing_by_task[tt][b] += 1
                timing_by_way[w][b] += 1
                if len(examples[w]) < 3:
                    examples[w].append(q)
                if "ground_truth_time_stamp" in row and row.get("ground_truth_time_stamp"):
                    proactive_delays.append(_sec(row["ground_truth_time_stamp"]) - _sec(row.get("time_stamp", "0")))
    delay = {}
    if proactive_delays:
        xs = sorted(proactive_delays)
        delay = {
            "n": len(xs),
            "mean": round(mean(xs), 2),
            "p50": median(xs),
            "p90": xs[int(0.9 * (len(xs) - 1))],
            "max": max(xs),
        }
    return {
        "n": sum(file_counts.values()),
        "file_pct": _pct(file_counts),
        "task_type_pct": _pct(task),
        "temporal_pct": _pct(temporal),
        "frames_required_pct": _pct(frames),
        "question_way_pct": _pct(way),
        "evidence_type_pct": _pct(evidence),
        "timing_bucket_pct": _pct(timing),
        "timing_by_task_pct": _nested_pct(timing_by_task),
        "timing_by_question_way_pct": _nested_pct(timing_by_way),
        "proactive_delay_sec": delay,
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ovo", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json"))
    parser.add_argument("--streamingbench", type=Path, default=Path("/home/tione/notebook/gaozhenkun/hzh/data/mjuicem/StreamingBench/StreamingBench"))
    parser.add_argument("--out", type=Path, default=Path("data/agent_v5/audits/benchmark_question_taxonomy.json"))
    args = parser.parse_args()
    report = {
        "ovo": audit_ovo(args.ovo),
        "streamingbench": audit_streamingbench(args.streamingbench),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
