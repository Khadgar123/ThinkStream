#!/usr/bin/env python3
"""Static simulation for active-query injection windows.

This audit does not call a model. It uses pass4 gold questions/samples and
compares deterministic query injection policies:

- always inject at ask_chunk
- optionally repeat inside a small direct window
- always force-inject at gold answer chunks
- always restore the still-open query at from-compress segment boundaries

The goal is to decide a rendering policy without relying on an undertrained
checkpoint's rollout behavior.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_data.pass5_splitter import (  # noqa: E402
    is_compress_sample,
    split_samples_by_compress,
)
from thinkstream.data.agent_protocol import (  # noqa: E402
    canonical_answer_instruction,
    format_queries_block,
)


@dataclass(frozen=True)
class Question:
    card_id: str
    family: str
    family_name: str
    question_type: str
    availability: str
    ours_unique: bool
    ask: int
    answers: Tuple[int, ...]
    support: Tuple[int, ...]
    question: str
    options: Tuple[str, ...]
    answer_form: str
    answer_style: str

    @property
    def final(self) -> int:
        return max(self.answers) if self.answers else self.ask

    def active_at(self, chunk: int) -> bool:
        return self.ask <= chunk <= self.final

    def prompt_dict(self, emitted: Sequence[Tuple[int, str]] = ()) -> Dict[str, Any]:
        return {
            "question": self.question,
            "ask_time": self.ask,
            "options": list(self.options),
            "answer_form": self.answer_form,
            "answer_style": self.answer_style,
            "answer_instruction": canonical_answer_instruction({
                "answer_form": self.answer_form,
                "answer_style": self.answer_style,
            }),
            "answers": [
                {"time": int(t), "text": str(text)}
                for t, text in emitted
            ],
        }


def _as_int_list(value: Any) -> List[int]:
    out: List[int] = []
    for item in value or []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(x for x in out if x >= 0))


def _questions(record: Dict[str, Any]) -> List[Question]:
    out: List[Question] = []
    for idx, raw in enumerate(record.get("questions") or []):
        try:
            ask = int(raw.get("ask_chunk"))
        except (TypeError, ValueError):
            continue
        if ask < 0:
            continue
        answers = _as_int_list(raw.get("answer_chunks") or raw.get("expected_answer_chunks"))
        if not answers:
            answers = [ask]
        question = str(raw.get("question") or "").strip()
        if not question:
            continue
        out.append(Question(
            card_id=str(raw.get("card_id") or f"q{idx}"),
            family=str(raw.get("family") or ""),
            family_name=str(raw.get("family_name") or ""),
            question_type=str(raw.get("question_type") or ""),
            availability=str(raw.get("availability") or ""),
            ours_unique=bool(raw.get("ours_unique")),
            ask=ask,
            answers=tuple(answers),
            support=tuple(_as_int_list(raw.get("support_chunks"))),
            question=question,
            options=tuple(str(x) for x in (raw.get("options") or [])),
            answer_form=str(raw.get("answer_form") or ""),
            answer_style=str(raw.get("answer_style") or ""),
        ))
    return sorted(out, key=lambda q: (q.ask, q.final, q.card_id))


def _sample_sort_key(sample: Dict[str, Any]) -> Tuple[int, int]:
    order = {
        "compress": -1,
        "recall_query": 0,
        "recall_response": 1,
        "recall": 1,
        "response": 2,
        "silent": 4,
        "recall_silent": 5,
    }
    return int(sample.get("chunk_idx", 0)), (
        -1 if is_compress_sample(sample) else order.get(str(sample.get("sample_type") or ""), 6)
    )


def _segments(samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    return split_samples_by_compress(sorted(samples, key=_sample_sort_key))


def _visual_chunks_and_boundaries(record: Dict[str, Any]) -> Tuple[List[int], Set[int]]:
    chunks: List[int] = []
    boundaries: Set[int] = set()
    for seg_idx, seg in enumerate(_segments(record.get("samples") or [])):
        stream = seg[:-1] if seg and is_compress_sample(seg[-1]) else seg
        stream = [s for s in stream if not is_compress_sample(s)]
        if not stream:
            continue
        first = int(stream[0].get("chunk_idx", 0))
        if seg_idx > 0:
            boundaries.add(first)
        for s in stream:
            chunks.append(int(s.get("chunk_idx", 0)))
    return sorted(set(chunks)), boundaries


def _answer_texts(record: Dict[str, Any]) -> Dict[Tuple[str, int], str]:
    texts: Dict[Tuple[str, int], str] = {}
    for q in record.get("questions") or []:
        card_id = str(q.get("card_id") or "")
        for emit in q.get("per_emit_answers") or []:
            if not isinstance(emit, dict):
                continue
            try:
                chunk = int(emit.get("chunk"))
            except (TypeError, ValueError):
                continue
            value = str(emit.get("value") or "").strip()
            if value:
                texts[(card_id, chunk)] = value
    return texts


def _forced_injection_chunks(q: Question, strategy: str, boundaries: Set[int]) -> Set[int]:
    active = set(range(q.ask, q.final + 1))
    chunks: Set[int] = {q.ask}
    if strategy in {"current_ask_boundary", "ask_once"}:
        pass
    elif strategy == "ask_answer":
        chunks.update(q.answers)
    elif strategy == "answer_only":
        chunks = set(q.answers)
    elif strategy == "ask_window4_answer":
        chunks.update(q.answers)
        chunks.update(range(q.ask, min(q.final, q.ask + 3) + 1))
    elif strategy == "ask_window8_answer":
        chunks.update(q.answers)
        chunks.update(range(q.ask, min(q.final, q.ask + 7) + 1))
    elif strategy == "periodic8_answer":
        chunks.update(q.answers)
        c = q.ask
        while c <= q.final:
            chunks.add(c)
            c += 8
    elif strategy == "support_answer":
        chunks.update(q.answers)
        chunks.update(c for c in q.support if q.ask <= c <= q.final)
    elif strategy == "every_active":
        chunks.update(active)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    chunks.update(c for c in boundaries if q.ask < c <= q.final)
    return chunks & active


def _block_chars(q: Question, prior_answers: Sequence[Tuple[int, str]]) -> int:
    return len(format_queries_block([q.prompt_dict(prior_answers)]))


def simulate_file(path: Path, strategies: Sequence[str], limit: Optional[int]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "path": str(path),
        "records": 0,
        "questions": 0,
        "visual_turns": 0,
        "response_turns": 0,
        "active_turns": 0,
        "boundary_open_questions": 0,
        "max_open_overlap": 0,
        "families": Counter(),
        "availability": Counter(),
        "question_type": Counter(),
        "ours_unique": Counter(),
        "strategies": {},
    }
    per_strategy = {
        name: {
            "query_injected_turns": 0,
            "query_block_chars": 0,
            "response_with_injection": 0,
            "response_with_recent8_injection": 0,
            "active_silent_with_injection": 0,
            "active_silent_turns": 0,
            "max_gap_to_response": 0,
            "gaps_to_response": [],
            "repeats_per_question": [],
            "family_response_gaps": defaultdict(list),
        }
        for name in strategies
    }

    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats["records"] >= limit:
                break
            record = json.loads(line)
            questions = _questions(record)
            if not questions:
                continue
            visual_chunks, boundaries = _visual_chunks_and_boundaries(record)
            if not visual_chunks:
                continue
            answer_texts = _answer_texts(record)
            stats["records"] += 1
            stats["questions"] += len(questions)
            stats["visual_turns"] += len(visual_chunks)
            stats["families"].update(q.family for q in questions)
            stats["availability"].update(q.availability for q in questions)
            stats["question_type"].update(q.question_type for q in questions)
            stats["ours_unique"].update("ours_unique" if q.ours_unique else "ovo_like" for q in questions)

            for c in visual_chunks:
                active_qs = [q for q in questions if q.active_at(c)]
                stats["max_open_overlap"] = max(stats["max_open_overlap"], len(active_qs))
                if active_qs:
                    stats["active_turns"] += 1
                stats["response_turns"] += sum(1 for q in active_qs if c in q.answers)
            for q in questions:
                stats["boundary_open_questions"] += sum(1 for b in boundaries if q.ask < b <= q.final)

            for name, s in per_strategy.items():
                injection_by_q = {
                    q.card_id: _forced_injection_chunks(q, name, boundaries)
                    for q in questions
                }
                for q in questions:
                    s["repeats_per_question"].append(len(injection_by_q[q.card_id]))
                    prior: List[Tuple[int, str]] = []
                    for c in visual_chunks:
                        if not q.active_at(c):
                            continue
                        injected = c in injection_by_q[q.card_id]
                        is_answer = c in q.answers
                        if injected:
                            s["query_injected_turns"] += 1
                            s["query_block_chars"] += _block_chars(q, prior)
                        if is_answer:
                            last = max((x for x in injection_by_q[q.card_id] if x <= c), default=None)
                            gap = 999 if last is None else c - last
                            s["max_gap_to_response"] = max(s["max_gap_to_response"], gap)
                            s["gaps_to_response"].append(gap)
                            s["family_response_gaps"][q.family].append(gap)
                            s["response_with_injection"] += int(injected)
                            s["response_with_recent8_injection"] += int(gap <= 8)
                            value = answer_texts.get((q.card_id, c))
                            if value:
                                prior.append((c, value))
                        else:
                            s["active_silent_turns"] += 1
                            s["active_silent_with_injection"] += int(injected)

    for name, s in per_strategy.items():
        total_resp = max(1, stats["response_turns"])
        active_silent = max(1, s["active_silent_turns"])
        gaps = s["gaps_to_response"]
        repeats = s["repeats_per_question"]
        fam_gap_summary = {}
        for fam, vals in sorted(s["family_response_gaps"].items()):
            if vals:
                fam_gap_summary[fam] = {
                    "n": len(vals),
                    "avg": round(sum(vals) / len(vals), 3),
                    "p95": sorted(vals)[int(0.95 * (len(vals) - 1))],
                    "max": max(vals),
                }
        stats["strategies"][name] = {
            "_response_with_injection": s["response_with_injection"],
            "_response_with_recent8_injection": s["response_with_recent8_injection"],
            "_active_silent_with_injection": s["active_silent_with_injection"],
            "_active_silent_turns": s["active_silent_turns"],
            "query_injected_turns": s["query_injected_turns"],
            "query_injected_per_visual_turn": s["query_injected_turns"] / max(1, stats["visual_turns"]),
            "avg_query_block_chars_per_visual_turn": s["query_block_chars"] / max(1, stats["visual_turns"]),
            "response_injection_coverage": s["response_with_injection"] / total_resp,
            "response_recent8_coverage": s["response_with_recent8_injection"] / total_resp,
            "active_silent_injection_rate": s["active_silent_with_injection"] / active_silent,
            "avg_gap_to_response": (sum(gaps) / len(gaps)) if gaps else 0.0,
            "p95_gap_to_response": sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else 0,
            "max_gap_to_response": s["max_gap_to_response"],
            "avg_repeats_per_question": (sum(repeats) / len(repeats)) if repeats else 0.0,
            "max_repeats_per_question": max(repeats) if repeats else 0,
            "family_response_gaps": fam_gap_summary,
        }
    for key in ["families", "availability", "question_type", "ours_unique"]:
        stats[key] = dict(stats[key])
    return stats


def _merge(results: Sequence[Dict[str, Any]], strategies: Sequence[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "files": len(results),
        "records": sum(r["records"] for r in results),
        "questions": sum(r["questions"] for r in results),
        "visual_turns": sum(r["visual_turns"] for r in results),
        "response_turns": sum(r["response_turns"] for r in results),
        "active_turns": sum(r["active_turns"] for r in results),
        "boundary_open_questions": sum(r["boundary_open_questions"] for r in results),
        "max_open_overlap": max((r["max_open_overlap"] for r in results), default=0),
        "families": dict(sum((Counter(r["families"]) for r in results), Counter())),
        "availability": dict(sum((Counter(r["availability"]) for r in results), Counter())),
        "question_type": dict(sum((Counter(r["question_type"]) for r in results), Counter())),
        "ours_unique": dict(sum((Counter(r["ours_unique"]) for r in results), Counter())),
        "strategies": {},
    }
    for name in strategies:
        injected = sum(r["strategies"][name]["query_injected_turns"] for r in results)
        chars_per_turn_num = sum(
            r["strategies"][name]["avg_query_block_chars_per_visual_turn"] * r["visual_turns"]
            for r in results
        )
        repeats_num = sum(
            r["strategies"][name]["avg_repeats_per_question"] * r["questions"]
            for r in results
        )
        resp_cov = (
            sum(r["strategies"][name]["_response_with_injection"] for r in results)
            / max(1, merged["response_turns"])
        )
        recent_cov = (
            sum(r["strategies"][name]["_response_with_recent8_injection"] for r in results)
            / max(1, merged["response_turns"])
        )
        silent_den = sum(r["strategies"][name]["_active_silent_turns"] for r in results)
        silent_rate = (
            sum(r["strategies"][name]["_active_silent_with_injection"] for r in results)
            / max(1, silent_den)
        )
        avg_gap = statistics.fmean(
            r["strategies"][name]["avg_gap_to_response"] for r in results
        ) if results else 0.0
        p95_gap = max((r["strategies"][name]["p95_gap_to_response"] for r in results), default=0)
        max_gap = max((r["strategies"][name]["max_gap_to_response"] for r in results), default=0)
        merged["strategies"][name] = {
            "query_injected_turns": injected,
            "query_injected_per_visual_turn": injected / max(1, merged["visual_turns"]),
            "avg_query_block_chars_per_visual_turn": chars_per_turn_num / max(1, merged["visual_turns"]),
            "response_injection_coverage": resp_cov,
            "response_recent8_coverage": recent_cov,
            "active_silent_injection_rate": silent_rate,
            "avg_gap_to_response_mean_by_file": avg_gap,
            "p95_gap_to_response_max_by_file": p95_gap,
            "max_gap_to_response": max_gap,
            "avg_repeats_per_question": repeats_num / max(1, merged["questions"]),
        }
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/query_injection_matrix/forced_window_static.json"))
    ap.add_argument("--limit-per-file", type=int, default=None)
    ap.add_argument(
        "--strategies",
        default="current_ask_boundary,ask_answer,ask_window4_answer,ask_window8_answer,periodic8_answer,support_answer,every_active",
    )
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    paths = [p for p in args.paths if p.exists()]
    results = [simulate_file(p, strategies, args.limit_per_file) for p in paths]
    payload = {
        "mode": "forced_query_window_static",
        "strategies": strategies,
        "merged": _merge(results, strategies),
        "files": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["merged"], ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(args.out), "files": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
