"""Audit data-pipeline distribution at every stage.

Run after pass3c (or pass3d) to verify the answer-form mix matches the
v12.0 design intent. Usage:

    python -m scripts.agent_data_v5.audit_distribution
        # Audits final/train_sft.jsonl
    python -m scripts.agent_data_v5.audit_distribution --verified
        # Audits verified/*.json (pass4 audit, not the SFT source)

Design targets (v12.0 — see FAMILY_TARGETS in pass3a_cards.py):
    Card-level MC ≥ 65%
    Response-sample MC ≥ 5% (after multi-probe expansion + pass3d sampling)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Bag of regexes for heuristic answer-form classification when metadata
# is missing/wrong (catches "A. eggplant" drift).
_HEURISTICS = [
    ("yes_no", re.compile(r"^(yes|no)\b\.?$", re.IGNORECASE)),
    ("mc_letter", re.compile(r"^[A-D]\.?$")),
    ("mc_drift", re.compile(r"^[A-D]\.\s+\w+", re.IGNORECASE)),  # "A. eggplant"
    ("number", re.compile(r"^-?\d+(\.\d+)?$")),
]


def classify_response_text(text: str) -> str:
    """Heuristic form classification of <response>X</response> content."""
    t = (text or "").strip()
    if not t:
        return "_empty"
    for label, pat in _HEURISTICS:
        if pat.match(t):
            return label
    if len(t) <= 30:
        return "short_exact_or_entity"
    return "descriptive"


def audit_jsonl(path: Path) -> Dict:
    sample_types = Counter()
    metadata_form_by_st = Counter()
    response_classified = Counter()
    n_total = 0
    n_response_with_text = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            s = json.loads(line)
            st = s.get("sample_type", "?")
            sample_types[st] += 1

            af = (s.get("metadata") or {}).get("answer_form", "") or "_blank"
            metadata_form_by_st[(st, af)] += 1

            # Heuristic re-classification of response text — catches
            # MC drift ("A. eggplant" misclassified as descriptive)
            if st in ("response", "recall_response"):
                out = s.get("output", "") or s.get("v12_assistant_turn_2", "")
                m = re.search(r"<(?:response|answer)>(.*?)</(?:response|answer)>",
                              out, re.DOTALL)
                if m:
                    n_response_with_text += 1
                    cls = classify_response_text(m.group(1))
                    response_classified[(af, cls)] += 1

    return {
        "path": str(path),
        "n_total": n_total,
        "sample_type": dict(sample_types),
        "metadata_form_by_st": {f"{k[0]}/{k[1]}": v for k, v in
                                sorted(metadata_form_by_st.items(),
                                       key=lambda x: -x[1])[:30]},
        "response_text_classified": {f"meta={k[0]}/text={k[1]}": v
                                     for k, v in
                                     sorted(response_classified.items(),
                                            key=lambda x: -x[1])[:30]},
        "n_response_with_text": n_response_with_text,
    }


def audit_verified_dir(root: Path) -> Dict:
    sample_types = Counter()
    n_total = 0
    n_files = 0
    af_by_st = Counter()

    for fp in sorted(root.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        n_files += 1
        with fp.open() as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue
        for s in d.get("samples", []):
            n_total += 1
            st = s.get("sample_type", "?")
            af = (s.get("metadata") or {}).get("answer_form", "") or "_blank"
            sample_types[st] += 1
            af_by_st[(st, af)] += 1

    return {
        "n_files": n_files,
        "n_total": n_total,
        "sample_type": dict(sample_types),
        "metadata_form_by_st": {f"{k[0]}/{k[1]}": v for k, v in
                                sorted(af_by_st.items(),
                                       key=lambda x: -x[1])[:25]},
    }


def diagnose(report: Dict) -> List[str]:
    """Generate human-readable diagnostic flags from a report."""
    flags = []
    n_total = report.get("n_total", 0)
    if n_total == 0:
        flags.append("EMPTY: no samples loaded")
        return flags

    st = report.get("sample_type", {})
    n_resp = st.get("response", 0) + st.get("recall_response", 0)
    if n_resp == 0:
        flags.append("BLOCKER: 0 response samples — model has nothing to learn the answering behavior")

    # Look at meta_form/text classification for response samples
    text_class = report.get("response_text_classified", {})
    n_mc_letter = sum(v for k, v in text_class.items() if "mc_letter" in k)
    n_mc_drift = sum(v for k, v in text_class.items() if "mc_drift" in k)
    n_yes_no = sum(v for k, v in text_class.items() if "yes_no" in k)
    n_number = sum(v for k, v in text_class.items() if "number" in k)
    n_descriptive = sum(v for k, v in text_class.items() if "descriptive" in k)
    total_resp = report.get("n_response_with_text", 0)

    if total_resp > 0:
        mc_clean_rate = n_mc_letter / total_resp
        mc_drift_rate = n_mc_drift / total_resp

        if mc_clean_rate < 0.05:
            flags.append(
                f"MC CLEAN: only {n_mc_letter} ({100*mc_clean_rate:.1f}%) "
                f"response samples emit a clean letter — target >= 5%"
            )
        if mc_drift_rate > 0.02:
            flags.append(
                f"MC DRIFT: {n_mc_drift} ({100*mc_drift_rate:.1f}%) "
                f"response samples emit 'A. blah' form — pass3c "
                f"_normalize_exact_form_answer not applied; re-render data"
            )
        if n_yes_no / max(1, total_resp) > 0.40:
            flags.append(
                f"BINARY OVER: {n_yes_no} ({100*n_yes_no/total_resp:.1f}%) "
                f"yes/no — F7 multi-probe likely overflowing"
            )
        if n_descriptive / max(1, total_resp) > 0.55:
            flags.append(
                f"DESC OVER: {n_descriptive} ({100*n_descriptive/total_resp:.1f}%) "
                f"descriptive — may include misclassified MC drift"
            )

    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="data/agent_v5",
        help="Path to data/agent_v5/ root",
    )
    parser.add_argument(
        "--verified", action="store_true",
        help="Audit verified/*.json (pass4 audit) instead of final/train_sft.jsonl",
    )
    parser.add_argument(
        "--file", default=None,
        help="Specific JSONL to audit (overrides default final/train_sft.jsonl)",
    )
    parser.add_argument("--output", default=None, help="Save report as JSON")
    args = parser.parse_args()

    root = Path(args.data_root)

    print("=" * 70)
    print(f"v12.0 data distribution audit — root: {root}")
    print("=" * 70)

    if args.file:
        path = Path(args.file)
        report = audit_jsonl(path)
        print(f"\n--- JSONL: {path} ---")
    elif args.verified:
        path = root / "verified"
        report = audit_verified_dir(path)
        print(f"\n--- verified/ : {path} ---")
    else:
        path = root / "final" / "train_sft.jsonl"
        report = audit_jsonl(path)
        print(f"\n--- final SFT: {path} ---")

    print(f"  total samples: {report.get('n_total')}")
    print(f"\n  sample_type:")
    for k, v in sorted(report.get("sample_type", {}).items(),
                       key=lambda x: -x[1]):
        pct = 100 * v / max(1, report["n_total"])
        print(f"    {k:>20s}: {v:>6d}  ({pct:.1f}%)")

    print(f"\n  metadata.answer_form by sample_type (top):")
    for k, v in report.get("metadata_form_by_st", {}).items():
        print(f"    {k:>40s}: {v}")

    if "response_text_classified" in report:
        print(f"\n  Heuristic re-classification of <response> text "
              f"(catches MC drift):")
        for k, v in report["response_text_classified"].items():
            print(f"    {k:>50s}: {v}")
        print(f"  (n_response_with_text = {report['n_response_with_text']})")

    flags = diagnose(report)
    print(f"\n--- Diagnostics ---")
    if not flags:
        print("  ✓ All checks pass")
    for f in flags:
        print(f"  ⚠ {f}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(
            {"report": report, "flags": flags},
            indent=2, ensure_ascii=False
        ))
        print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
