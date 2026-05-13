#!/usr/bin/env python
"""Compact comparison table for OVO recurrent/base JSON summaries."""
import argparse
import json
from pathlib import Path


def _pct(num, den):
    return float(num) / float(den) if den else 0.0


def _agent_diag(summary):
    health = summary.get("health") or {}
    if health:
        answer = health.get("answer") or {}
        recall = health.get("recall") or {}
        comp = health.get("compression") or {}
        runtime = health.get("format_runtime") or {}
        return {
            "recall": recall.get("events", 0),
            "recall_hit": recall.get("support_hit_rate", 0.0),
            "comp": comp.get("events", 0),
            "comp_succ": comp.get("success_rate", 0.0),
            "stable": runtime.get("stable_think_pair_rate", 0.0),
            "acc_wr": recall.get("acc_with_recall", 0.0),
            "acc_nr": recall.get("acc_without_recall", 0.0),
            "early": answer.get("early_rate", 0.0),
            "late": answer.get("late_rate", 0.0),
            "missing": answer.get("missing_rate", 0.0),
            "acc_no_early": answer.get("no_early_acc", summary.get("overall_no_early", 0.0)),
            "acc_no_late": answer.get("no_late_acc", summary.get("overall_no_late", 0.0)),
            "acc_on_time": answer.get("on_time_acc", summary.get("overall_on_time", 0.0)),
        }

    diag = summary.get("diagnostics") or {}
    by_task = summary.get("by_task") or summary.get("per_task") or {}
    recall = sum(v.get("recall_events", 0) for v in diag.values())
    recall_hits = 0.0
    recall_den = 0
    comp = sum(v.get("compress_events", 0) for v in diag.values())
    comp_succ_num = 0.0
    comp_succ_den = 0
    stable = sum(v.get("stable_think_pairs", 0) for v in diag.values())
    wr_n = wr_c = nr_n = nr_c = 0
    early_n = late_n = missing_n = timing_den = 0.0
    no_early_n = no_late_n = on_time_n = 0.0
    for task, v in diag.items():
        n = v.get("recall_events", 0)
        recall_hits += v.get("recall_support_hit_rate", 0.0) * n
        recall_den += n
        c = v.get("compress_events", 0)
        comp_succ_num += v.get("compress_success_rate", 0.0) * c
        comp_succ_den += c
        wr_n += v.get("n_with_recall", 0)
        wr_c += v.get("acc_with_recall", 0.0) * v.get("n_with_recall", 0)
        nr_n += v.get("n_without_recall", 0)
        nr_c += v.get("acc_without_recall", 0.0) * v.get("n_without_recall", 0)
        probes = (by_task.get(task) or {}).get("n", 0)
        timing_den += probes
        early_n += v.get("response_early_rate", 0.0) * probes
        late_n += v.get("response_late_rate", 0.0) * probes
        missing_n += v.get("response_missing_rate", 0.0) * probes
        no_early_n += v.get("acc_no_early", 0.0) * probes
        no_late_n += v.get("acc_no_late", 0.0) * probes
        on_time_n += v.get("acc_on_time", 0.0) * probes
    return {
        "recall": recall,
        "recall_hit": _pct(recall_hits, recall_den),
        "comp": comp,
        "comp_succ": _pct(comp_succ_num, comp_succ_den),
        "stable": stable,
        "acc_wr": _pct(wr_c, wr_n),
        "n_wr": wr_n,
        "acc_nr": _pct(nr_c, nr_n),
        "n_nr": nr_n,
        "early": _pct(early_n, timing_den),
        "late": _pct(late_n, timing_den),
        "missing": _pct(missing_n, timing_den),
        "acc_no_early": summary.get(
            "overall_no_early",
            _pct(no_early_n, timing_den),
        ),
        "acc_no_late": summary.get(
            "overall_no_late",
            _pct(no_late_n, timing_den),
        ),
        "acc_on_time": summary.get(
            "overall_on_time",
            _pct(on_time_n, timing_den),
        ),
    }


def _overall_value(summary):
    overall = summary.get("overall", 0.0)
    if isinstance(overall, dict):
        return float(
            overall.get("trajectory_mean_correct_question_weighted")
            or overall.get("avg")
            or overall.get("score_mean")
            or 0.0
        )
    return float(overall or 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsons", nargs="+")
    args = p.parse_args()
    rows = []
    for path in args.jsons:
        with open(path) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        cat = summary.get("category", {})
        diag = _agent_diag(summary)
        name = data.get("run_name") or Path(path).stem
        rows.append({
            "name": name,
            "overall": _overall_value(summary),
            "rt": (cat.get("RT") or {}).get("avg", 0.0),
            "bt": (cat.get("BT") or {}).get("avg", 0.0),
            "ft": (cat.get("FT") or {}).get("avg", 0.0),
            **diag,
        })

    print(
        f"{'run':<36} {'all':>6} {'RT':>6} {'BT':>6} {'FT':>6} "
        f"{'noE':>6} {'noL':>6} {'onT':>6} "
        f"{'rec':>5} {'r_hit':>6} {'acc+r':>7} {'acc-r':>7} "
        f"{'early':>6} {'late':>6} {'miss':>6} "
        f"{'comp':>5} {'c_ok':>6} {'stable':>7}"
    )
    print("-" * 156)
    for r in sorted(rows, key=lambda x: x["name"]):
        print(
            f"{r['name'][:36]:<36} "
            f"{r['overall']:>6.3f} {r['rt']:>6.3f} {r['bt']:>6.3f} {r['ft']:>6.3f} "
            f"{r['acc_no_early']:>6.3f} {r['acc_no_late']:>6.3f} "
            f"{r['acc_on_time']:>6.3f} "
            f"{int(r['recall']):>5} {r['recall_hit']:>6.3f} "
            f"{r['acc_wr']:>7.3f} {r['acc_nr']:>7.3f} "
            f"{r['early']:>6.3f} {r['late']:>6.3f} {r['missing']:>6.3f} "
            f"{int(r['comp']):>5} {r['comp_succ']:>6.3f} {r['stable']:>7.3f}"
        )


if __name__ == "__main__":
    main()
