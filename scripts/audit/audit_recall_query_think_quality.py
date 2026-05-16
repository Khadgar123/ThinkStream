#!/usr/bin/env python3
"""Audit pass3c recall query and post-recall think quality.

This is intentionally schema-tolerant: it can scan stale v11/v12 samples and
current v12.77 samples, then report old protocol residues, answer-space leaks,
and post-recall template repetition.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ANSWER_RE = re.compile(r"</Response>\s*(.*?)\s*$", re.DOTALL | re.IGNORECASE)
LEGACY_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+")

GENERIC_THINK_PATTERNS = [
    "the recalled frames provide the historical evidence needed",
    "historical evidence needed for this pending question",
    "i compare that retrieved moment",
    "give the grounded answer",
    "the recalled evidence resolves",
    "the returned history is enough",
    "the historical visual evidence answers",
]

QUERY_FORBIDDEN_GENERIC = {
    "answer",
    "option",
    "choice",
    "correct",
    "letter",
    "yes",
    "no",
}

NUMBER_WORDS = {
    "0": {"zero"},
    "1": {"one"},
    "2": {"two"},
    "3": {"three"},
    "4": {"four"},
    "5": {"five"},
    "6": {"six"},
    "7": {"seven"},
    "8": {"eight"},
    "9": {"nine"},
    "10": {"ten"},
}


def _iter_json_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".json":
            yield p
        elif p.is_dir():
            yield from sorted(p.glob("*.json"))


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        samples = obj.get("samples") or obj.get("data") or []
        if isinstance(samples, list):
            return [x for x in samples if isinstance(x, dict)]
    return []


def _extract_think(text: str) -> str:
    m = THINK_RE.search(text or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = LEGACY_ANSWER_RE.search(text or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _extract_tool_query(text: str) -> Dict[str, Any]:
    m = TOOL_RE.search(text or "")
    if not m:
        return {}
    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    args = obj.get("arguments") if isinstance(obj, dict) else {}
    return args if isinstance(args, dict) else {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _compact(text: str) -> str:
    return "".join(ch for ch in _norm(text) if ch.isalnum())


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(str(text or "")) if len(t) >= 2]


def _answer_text(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    for key in ("gold_answer", "canonical_answer", "correct_answer_text", "sft_answer"):
        value = sample.get(key) or meta.get(key)
        if value:
            return str(value).strip()
    return _extract_answer(str(sample.get("v12_assistant_turn_2") or sample.get("output") or ""))


def _answer_leaks(answer: str, text: str, *, answer_form: str = "") -> bool:
    answer = str(answer or "").strip()
    text = str(text or "")
    if not answer or not text or answer.lower() == "unable to answer":
        return False
    if len(answer) >= 2 and re.search(re.escape(answer), text, re.IGNORECASE):
        return True
    answer_n = _norm(answer)
    if answer_n in {"yes", "no"}:
        return re.search(rf"(?<!\w){re.escape(answer_n)}(?!\w)", text, re.IGNORECASE) is not None
    if re.fullmatch(r"\d+", answer_n):
        if re.search(rf"(?<!\w){re.escape(answer_n)}(?!\w)", text):
            return True
        return any(
            re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text, re.IGNORECASE)
            for w in NUMBER_WORDS.get(answer_n, set())
        )
    if answer_form == "multiple_choice" and len(answer) == 1 and answer.upper() in "ABCDE":
        return re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", text) is not None
    return bool(_compact(answer) and len(_compact(answer)) >= 2 and _compact(answer) in _compact(text))


def _ngram_key(text: str, n: int = 6) -> str:
    toks = _tokens(text)
    return " ".join(toks[:n])


def audit(paths: List[str], *, max_examples: int = 8) -> Dict[str, Any]:
    counters = Counter()
    query_counter = Counter()
    think_prefix_counter = Counter()
    examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def add_example(kind: str, file: Path, sample: Dict[str, Any], text: str) -> None:
        if len(examples[kind]) >= max_examples:
            return
        examples[kind].append({
            "file": str(file),
            "sample_id": str(sample.get("sample_id") or sample.get("trajectory_id") or ""),
            "card_id": str(sample.get("card_id") or ""),
            "chunk_idx": str(sample.get("chunk_idx") or ""),
            "text": text[:260],
        })

    for file in _iter_json_files(paths):
        samples = _load_samples(file)
        counters["files"] += 1
        counters["samples"] += len(samples)
        for sample in samples:
            if sample.get("sample_type") != "recall" and "v12_assistant_turn_1" not in sample:
                continue
            counters["recall_samples"] += 1
            turn1 = str(sample.get("v12_assistant_turn_1") or "")
            turn2 = str(sample.get("v12_assistant_turn_2") or sample.get("output") or "")
            query = _extract_tool_query(turn1)
            query_text = str(query.get("query") or "")
            answer = _answer_text(sample)
            meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
            answer_form = str(sample.get("answer_form") or meta.get("answer_form") or "")
            think = _extract_think(turn2)

            if "<answer>" in turn2.lower() or "</answer>" in turn2.lower():
                counters["legacy_answer_tag"] += 1
                add_example("legacy_answer_tag", file, sample, turn2)
            if "</Response>" not in turn2 and "<answer>" not in turn2.lower() and "</Silence>" not in turn2:
                counters["missing_current_terminal"] += 1
                add_example("missing_current_terminal", file, sample, turn2)
            if query_text:
                query_counter[query_text.lower()] += 1
                if _answer_leaks(answer, query_text, answer_form=answer_form):
                    counters["query_answer_leak"] += 1
                    add_example("query_answer_leak", file, sample, query_text)
                q_tokens = set(_tokens(query_text))
                if q_tokens & QUERY_FORBIDDEN_GENERIC:
                    counters["query_generic_answer_space_word"] += 1
                    add_example("query_generic_answer_space_word", file, sample, query_text)
            else:
                counters["missing_recall_query"] += 1
                add_example("missing_recall_query", file, sample, turn1)

            if think:
                prefix = _ngram_key(think)
                think_prefix_counter[prefix] += 1
                low = think.lower()
                if any(p in low for p in GENERIC_THINK_PATTERNS):
                    counters["generic_post_recall_think"] += 1
                    add_example("generic_post_recall_think", file, sample, think)
                if _answer_leaks(answer, think, answer_form=answer_form):
                    counters["think_answer_leak"] += 1
                    add_example("think_answer_leak", file, sample, think)
                if len(_tokens(think)) > 28:
                    counters["think_too_long"] += 1
                    add_example("think_too_long", file, sample, think)
            else:
                counters["missing_post_recall_think"] += 1
                add_example("missing_post_recall_think", file, sample, turn2)

    recall_n = max(counters["recall_samples"], 1)
    return {
        "counts": dict(counters),
        "rates": {
            key + "_pct": round(value * 100.0 / recall_n, 3)
            for key, value in counters.items()
            if key not in {"files", "samples", "recall_samples"}
        },
        "top_queries": query_counter.most_common(20),
        "top_think_prefixes": think_prefix_counter.most_common(20),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="sample JSON files or directories")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.paths, max_examples=args.max_examples)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
