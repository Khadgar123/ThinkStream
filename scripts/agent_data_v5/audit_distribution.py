"""Audit v12 data-construction quality and distribution.

The audit is intentionally format-tolerant:
  - flat sample JSONL, e.g. final/train_sft.jsonl
  - trajectory JSONL, e.g. final/train_rl_trajectories.jsonl
  - verified/*.json directories

It checks the acceptance axes that matter for ThinkStream data quality:
question timing, selected question composition, silent/response/recall rates,
MC answer balance, multi-answer coverage, and open-ended/non-MCQ coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


_HEURISTICS = [
    ("yes_no", re.compile(r"^(yes|no)\b\.?$", re.IGNORECASE)),
    ("mc_letter", re.compile(r"^[A-E]\.?$")),
    ("mc_drift", re.compile(r"^[A-E]\.\s+\w+", re.IGNORECASE)),
    ("number", re.compile(r"^-?\d+(\.\d+)?$")),
]
_ANSWER_RE = re.compile(r"<(?:response|answer)>(.*?)</(?:response|answer)>", re.DOTALL)


def classify_response_text(text: str) -> str:
    """Heuristic form classification of answer content."""
    t = (text or "").strip()
    if not t:
        return "_empty"
    for label, pat in _HEURISTICS:
        if pat.match(t):
            return label
    if len(t) <= 30:
        return "short_exact_or_entity"
    return "descriptive"


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / max(1, d), 2)


def _stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 2),
        "median": round(statistics.median(s), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "p25": round(_percentile(s, 25), 2),
        "p75": round(_percentile(s, 75), 2),
    }


def _percentile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_xs) - 1)
    if f == c:
        return float(sorted_xs[f])
    return float(sorted_xs[f] + (k - f) * (sorted_xs[c] - sorted_xs[f]))


def _answer_text(sample: Dict) -> str:
    out = sample.get("v12_assistant_turn_2") or sample.get("output", "") or ""
    m = _ANSWER_RE.search(out)
    return m.group(1).strip() if m else ""


def _sample_action(sample: Dict) -> str:
    action = sample.get("action") or (sample.get("metadata") or {}).get("gold_action")
    if action:
        return str(action)
    st = sample.get("sample_type", "")
    if st in ("response", "recall", "recall_response"):
        return "response"
    if st == "compress":
        return "compress"
    return "silent"


def _is_recall_sample(sample: Dict) -> bool:
    st = sample.get("sample_type", "")
    return st in ("recall", "recall_query", "recall_response", "recall_silent") or bool(
        sample.get("recall_result")
    )


class Accumulator:
    def __init__(self, path: Path):
        self.path = path
        self.n_rows = 0
        self.n_samples = 0
        self.n_trajectories = 0
        self.n_recall_samples = 0
        self.sample_types = Counter()
        self.actions = Counter()
        self.response_text = Counter()
        self.response_text_by_form = Counter()
        self.questions: Dict[Tuple[str, str, str], Dict] = {}
        self.ask_by_traj: Dict[Tuple[str, str], set] = defaultdict(set)

    def add_sample(self, sample: Dict, *, parent_vid: str = "", parent_tid: str = "") -> None:
        self.n_samples += 1
        st = str(sample.get("sample_type", "?"))
        action = _sample_action(sample)
        self.sample_types[st] += 1
        self.actions[action] += 1
        if _is_recall_sample(sample):
            self.n_recall_samples += 1

        meta = sample.get("metadata") or {}
        vid = str(sample.get("video_id") or parent_vid or "")
        tid = str(sample.get("trajectory_id") or parent_tid or "")
        cid = str(sample.get("card_id") or "")
        if cid and meta.get("question"):
            key = (vid, tid, cid)
            q = self.questions.setdefault(key, {
                "video_id": vid,
                "trajectory_id": tid,
                "card_id": cid,
                "family": meta.get("family", ""),
                "answer_form": meta.get("answer_form", ""),
                "question_type": meta.get("question_type", ""),
                "mechanism": meta.get("availability", "") or sample.get("sequence_type", ""),
                "ask_chunk": meta.get("ask_chunk", -1),
                "answer_chunks": set(),
                "per_emit_answers": list(meta.get("per_emit_answers") or []),
                "correct_option": meta.get("correct_option", ""),
            })
            ask = q.get("ask_chunk", -1)
            if isinstance(ask, int) and ask >= 0:
                self.ask_by_traj[(vid, tid)].add(ask)
            if action == "response":
                chunk = int(sample.get("chunk_idx", -1))
                if chunk >= 0:
                    q["answer_chunks"].add(chunk)

        if action == "response":
            af = meta.get("answer_form", "") or "_blank"
            cls = classify_response_text(_answer_text(sample))
            self.response_text[cls] += 1
            self.response_text_by_form[(af, cls)] += 1

    def add_question(self, q: Dict, *, video_id: str, trajectory_id: str) -> None:
        cid = str(q.get("card_id") or "")
        if not cid:
            return
        key = (str(video_id), str(trajectory_id), cid)
        rec = self.questions.setdefault(key, {
            "video_id": str(video_id),
            "trajectory_id": str(trajectory_id),
            "card_id": cid,
            "family": q.get("family", ""),
            "answer_form": q.get("answer_form", ""),
            "question_type": q.get("question_type", ""),
            "mechanism": q.get("availability", ""),
            "ask_chunk": q.get("ask_chunk", -1),
            "answer_chunks": set(),
            "per_emit_answers": list(q.get("per_emit_answers") or []),
            "correct_option": q.get("correct_option", ""),
        })
        for field in ("family", "answer_form", "question_type", "mechanism", "correct_option"):
            if not rec.get(field) and q.get(field):
                rec[field] = q.get(field)
        if not rec.get("per_emit_answers") and q.get("per_emit_answers"):
            rec["per_emit_answers"] = list(q.get("per_emit_answers") or [])
        ask = q.get("ask_chunk", -1)
        if isinstance(ask, int) and ask >= 0:
            rec["ask_chunk"] = ask
            self.ask_by_traj[(str(video_id), str(trajectory_id))].add(ask)
        for c in q.get("answer_chunks") or []:
            try:
                ci = int(c)
            except Exception:
                continue
            rec["answer_chunks"].add(ci)

    def add_row(self, row: Dict) -> None:
        self.n_rows += 1
        if isinstance(row.get("questions"), list) and isinstance(row.get("samples"), list):
            self.n_trajectories += 1
            vid = str(row.get("video_id", ""))
            tid = str(row.get("trajectory_id", ""))
            for q in row.get("questions") or []:
                self.add_question(q, video_id=vid, trajectory_id=tid)
            for s in row.get("samples") or []:
                self.add_sample(s, parent_vid=vid, parent_tid=tid)
        else:
            self.add_sample(row)

    def report(self) -> Dict:
        q_records = list(self.questions.values())
        q_forms = Counter(q.get("answer_form") or "_blank" for q in q_records)
        q_types = Counter(q.get("question_type") or "_blank" for q in q_records)
        families = Counter(q.get("family") or "_blank" for q in q_records)
        mechs = Counter(q.get("mechanism") or "_blank" for q in q_records)
        mc_correct = Counter(
            q.get("correct_option") for q in q_records
            if q.get("answer_form") == "multiple_choice" and q.get("correct_option")
        )
        ask_chunks = [
            int(q.get("ask_chunk"))
            for q in q_records
            if isinstance(q.get("ask_chunk"), int) and q.get("ask_chunk") >= 0
        ]
        q_intervals: List[int] = []
        for asks in self.ask_by_traj.values():
            seq = sorted(asks)
            q_intervals.extend(seq[i + 1] - seq[i] for i in range(len(seq) - 1))
        multi_answer = [
            q for q in q_records
            if len(q.get("per_emit_answers") or []) > 1
            or len(q.get("answer_chunks") or []) > 1
            or q.get("question_type") == "multi_emit"
        ]
        answer_delays: List[int] = []
        for q in q_records:
            ask = q.get("ask_chunk", -1)
            if not isinstance(ask, int) or ask < 0:
                continue
            for c in sorted(q.get("answer_chunks") or []):
                answer_delays.append(int(c) - ask)
        n_response = self.actions.get("response", 0)
        n_silent = self.actions.get("silent", 0)
        n_compress = self.actions.get("compress", 0)
        n_recall = self.n_recall_samples
        non_mc = sum(v for k, v in q_forms.items() if k != "multiple_choice")
        return {
            "path": str(self.path),
            "n_rows": self.n_rows,
            "n_total": self.n_samples,
            "n_trajectories": self.n_trajectories,
            "sample_type": dict(self.sample_types),
            "action": dict(self.actions),
            "rates": {
                "silent_rate": _pct(n_silent, self.n_samples),
                "response_rate": _pct(n_response, self.n_samples),
                "recall_rate": _pct(n_recall, self.n_samples),
                "compress_rate": _pct(n_compress, self.n_samples),
            },
            "questions": {
                "n_questions": len(q_records),
                "by_answer_form": dict(q_forms.most_common()),
                "by_question_type": dict(q_types.most_common()),
                "by_family": dict(families.most_common()),
                "by_mechanism": dict(mechs.most_common()),
                "non_mc_questions": non_mc,
                "non_mc_rate": _pct(non_mc, len(q_records)),
                "multi_answer_questions": len(multi_answer),
                "multi_answer_rate": _pct(len(multi_answer), len(q_records)),
                "mc_correct_option": {k: mc_correct.get(k, 0) for k in ["A", "B", "C", "D", "E"]},
            },
            "timing": {
                "ask_chunk": _stats([float(x) for x in ask_chunks]),
                "q_interval_chunks": _stats([float(x) for x in q_intervals]),
                "answer_delay_chunks": _stats([float(x) for x in answer_delays]),
            },
            "response_text_classified": dict(self.response_text.most_common()),
            "response_text_by_form": {
                f"meta={k[0]}/text={k[1]}": v
                for k, v in self.response_text_by_form.most_common(30)
            },
        }


def audit_jsonl(path: Path) -> Dict:
    acc = Accumulator(path)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            acc.add_row(json.loads(line))
    return acc.report()


def audit_verified_dir(root: Path) -> Dict:
    acc = Accumulator(root)
    n_files = 0
    for fp in sorted(root.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        n_files += 1
        try:
            d = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        for s in d.get("samples", []) or []:
            acc.add_sample(s, parent_vid=fp.stem)
    report = acc.report()
    report["n_files"] = n_files
    return report


def diagnose(report: Dict) -> List[str]:
    flags: List[str] = []
    n_total = report.get("n_total", 0)
    if n_total == 0:
        return ["BLOCKER: no samples loaded"]

    rates = report.get("rates", {})
    if rates.get("response_rate", 0) <= 0:
        flags.append("BLOCKER: 0 response samples")
    if rates.get("silent_rate", 0) < 55 or rates.get("silent_rate", 0) > 97:
        flags.append(
            f"SILENT RATE: {rates.get('silent_rate', 0):.2f}% outside loose 55-97% sanity band"
        )
    if rates.get("response_rate", 0) < 2 or rates.get("response_rate", 0) > 35:
        flags.append(
            f"RESPONSE RATE: {rates.get('response_rate', 0):.2f}% outside loose 2-35% sanity band"
        )
    if rates.get("recall_rate", 0) == 0:
        flags.append("RECALL: 0 recall-shaped samples")

    qs = report.get("questions", {})
    n_q = qs.get("n_questions", 0)
    if n_q == 0:
        flags.append("BLOCKER: no questions detected")
    if n_q and qs.get("non_mc_rate", 0) < 15:
        flags.append(
            f"OPEN/NON-MCQ LOW: {qs.get('non_mc_rate', 0):.2f}% non-MCQ questions; target >= 15%"
        )
    if n_q and qs.get("multi_answer_questions", 0) == 0:
        flags.append("MULTI-ANSWER: no multi-emit / one-question-many-answer cases detected")

    mc = qs.get("mc_correct_option", {})
    total_mc = sum(mc.values()) if isinstance(mc, dict) else 0
    if total_mc:
        pcts = [mc.get(k, 0) / total_mc * 100 for k in ["A", "B", "C", "D"]]
        spread = max(pcts) - min(pcts)
        if spread > 10:
            flags.append(f"MC BALANCE: correct-option spread {spread:.1f}pp > 10pp")

    qint = (report.get("timing") or {}).get("q_interval_chunks") or {}
    if qint.get("n", 0):
        mean_interval = qint.get("mean", 0)
        if mean_interval < 5 or mean_interval > 25:
            flags.append(
                f"QUESTION TIMING: mean interval {mean_interval} chunks outside loose 5-25 band"
            )

    text_class = report.get("response_text_classified", {})
    n_resp_text = sum(text_class.values()) if isinstance(text_class, dict) else 0
    if n_resp_text:
        drift = text_class.get("mc_drift", 0)
        if drift / max(1, n_resp_text) > 0.02:
            flags.append(
                f"MC DRIFT: {drift}/{n_resp_text} responses look like 'A. text' instead of clean letters"
            )
    return flags


def _print_counter(title: str, data: Dict[str, int], total: int, limit: int = 30) -> None:
    print(f"\n  {title}:")
    for k, v in list(data.items())[:limit]:
        print(f"    {str(k):>28s}: {v:>7d}  ({_pct(int(v), total):5.2f}%)")


def print_report(report: Dict, flags: List[str]) -> None:
    print(f"  rows: {report.get('n_rows')}  samples: {report.get('n_total')}  "
          f"trajectories: {report.get('n_trajectories')}")
    print(f"  rates: {report.get('rates')}")
    _print_counter("sample_type", report.get("sample_type", {}), report.get("n_total", 0))
    _print_counter("action", report.get("action", {}), report.get("n_total", 0))

    qs = report.get("questions", {})
    print(f"\n  questions: {qs.get('n_questions', 0)}")
    print(f"    non_mc_rate={qs.get('non_mc_rate', 0):.2f}%  "
          f"multi_answer_rate={qs.get('multi_answer_rate', 0):.2f}%")
    _print_counter("question answer_form", qs.get("by_answer_form", {}), qs.get("n_questions", 0))
    _print_counter("question_type", qs.get("by_question_type", {}), qs.get("n_questions", 0))
    _print_counter("family", qs.get("by_family", {}), qs.get("n_questions", 0))
    _print_counter("mechanism", qs.get("by_mechanism", {}), qs.get("n_questions", 0))
    print(f"\n  MC correct option: {qs.get('mc_correct_option', {})}")
    print(f"\n  timing: {report.get('timing')}")

    if report.get("response_text_classified"):
        _print_counter(
            "response text classified",
            report.get("response_text_classified", {}),
            sum(report.get("response_text_classified", {}).values()),
        )

    print("\n--- Diagnostics ---")
    if not flags:
        print("  All checks pass")
    for f in flags:
        print(f"  {f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/agent_v5")
    parser.add_argument(
        "--verified", action="store_true",
        help="Audit verified/*.json instead of final/train_sft.jsonl",
    )
    parser.add_argument(
        "--trajectories", action="store_true",
        help="Audit final/train_rl_trajectories.jsonl by default",
    )
    parser.add_argument("--file", default=None, help="Specific JSONL to audit")
    parser.add_argument("--output", default=None, help="Save report as JSON")
    args = parser.parse_args()

    root = Path(args.data_root)
    print("=" * 70)
    print(f"ThinkStream data quality audit — root: {root}")
    print("=" * 70)

    if args.file:
        path = Path(args.file)
        report = audit_jsonl(path)
        print(f"\n--- JSONL: {path} ---")
    elif args.verified:
        path = root / "verified"
        report = audit_verified_dir(path)
        print(f"\n--- verified/: {path} ---")
    elif args.trajectories:
        path = root / "final" / "train_rl_trajectories.jsonl"
        report = audit_jsonl(path)
        print(f"\n--- trajectories: {path} ---")
    else:
        path = root / "final" / "train_sft.jsonl"
        report = audit_jsonl(path)
        print(f"\n--- final SFT: {path} ---")

    flags = diagnose(report)
    print_report(report, flags)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(
            {"report": report, "flags": flags},
            indent=2, ensure_ascii=False,
        ))
        print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
