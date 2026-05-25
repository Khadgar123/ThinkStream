#!/usr/bin/env python3
"""Export GSPO/GDPO training-time metric records with explicit estimates.

This is intentionally offline-only: it reads existing local train logs and
training-time validation outputs, then writes tracked CSV/Markdown records.
It does not launch, resume, or evaluate any model.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs/audit/gspo_gdpo_training_eval_scores_20260525"
LOCAL_AUDIT_DIR = ROOT / "output/audit/gspo_gdpo_training_records_20260525"

RUNS = {
    "gspo_main": ROOT / "output/qwen3vl8b-rl-traj-gspo-b1_10-sft500-fixedframes-s8-e2-sv10-20260519",
    "gspo_kl_from60": ROOT / "output/qwen3vl8b-rl-traj-gspo-kl001-from60-save1-eval10-sameval-20260522",
    "gspo_kl_from69": ROOT / "output/qwen3vl8b-rl-traj-gspo-kl001-from69-patched-save1-eval10-sameval-20260522",
    "gdpo_main": ROOT / "output/rl_b1_10_ckpt500_traj_gdpo_g8_b8_slots8_e2_eval10_save10_wandb_20260519",
    "gdpo_kl_from50": ROOT / "output/rl_b1_10_from50_stepref_kl001_save1_eval10_20260521",
}

TRAIN_METRICS = [
    "train/thinkstream/reward/score/mean",
    "train/thinkstream/reward/outcome/mean",
    "train/thinkstream/recall/call_traj_frac",
    "train/thinkstream/recall/recall_call_count/mean",
    "train/thinkstream/recall/recall_support_request_hit_rate/mean",
    "train/thinkstream/recall/answer_used_per_labeled/rate",
    "train/thinkstream/recall/answer_success_per_labeled/rate",
    "train/thinkstream/recall/post_recall_outcome_mean/mean",
    "train/thinkstream/compress/compress_quality/mean",
    "train/thinkstream/compress/compress_quality_cover_ok/mean",
    "train/thinkstream/compress/compress_quality_boundary_score/mean",
    "train/thinkstream/compress/compress_quality_source_precision/mean",
]

EVAL_METRICS = [
    "trajectory_mean_correct_question_weighted",
    "score_question_weighted",
    "outcome_question_weighted",
    "trajectory_all_correct_mean",
    "recall_call_count_mean",
    "recall_support_request_hit_rate_mean",
    "recall_support_returned_hit_rate_mean",
    "post_recall_outcome_mean_mean",
    "recall_align_rate_mean",
    "recall_runtime_ok_rate_mean",
    "compress_quality_mean",
    "compress_quality_question_weighted",
]

EVAL_UNBOUNDED = {"score_question_weighted", "recall_call_count_mean"}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean_text(value: str) -> str:
    return ANSI_RE.sub("", value)


def parse_number(raw: str) -> float | None:
    raw = clean_text(raw).strip()
    match = re.fullmatch(r"np\.[A-Za-z0-9_]+\((.*)\)", raw)
    if match:
        raw = match.group(1).strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = fmt(value)
                else:
                    out[key] = value
            writer.writerow(out)


def parse_train_logs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run, run_dir in RUNS.items():
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        with log_path.open(errors="ignore") as f:
            for line in f:
                line = clean_text(line)
                match = re.search(r"\bstep:(\d+)\s+-\s+(.*)", line)
                if not match:
                    continue
                step = int(match.group(1))
                values: dict[str, object] = {
                    "run": run,
                    "step": step,
                    "status": "observed",
                    "source_path": str(log_path.relative_to(ROOT)),
                    "estimation_method": "",
                }
                for part in match.group(2).split(" - "):
                    if ":" not in part:
                        continue
                    key, raw = part.split(":", 1)
                    key = key.strip()
                    if key in TRAIN_METRICS:
                        values[key] = parse_number(raw)
                if any(values.get(metric) is not None for metric in TRAIN_METRICS):
                    rows.append(values)
    rows.sort(key=lambda r: (str(r["run"]), int(r["step"])))
    return rows


def add_roll5(rows: list[dict[str, object]], group_key: str, metrics: list[str]) -> None:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: int(r["step"]))
        history: dict[str, list[float]] = {metric: [] for metric in metrics}
        for row in group_rows:
            for metric in metrics:
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    history[metric].append(float(value))
                    history[metric] = history[metric][-5:]
                    row[f"{metric}__roll5"] = statistics.fmean(history[metric])
                else:
                    row[f"{metric}__roll5"] = None


def train_chain(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run_step = {(str(row["run"]), int(row["step"])): row for row in raw_rows}
    chain_rows: list[dict[str, object]] = []

    def prefix_estimates(chain: str, run: str, label: str) -> list[dict[str, object]]:
        hist = [by_run_step[(run, step)] for step in range(11, 16) if (run, step) in by_run_step]
        out: list[dict[str, object]] = []
        for step in range(1, 11):
            row: dict[str, object] = {
                "chain": chain,
                "run": label,
                "step": step,
                "status": "estimated",
                "source_path": "",
                "estimation_method": f"backfill_median_of_first_observed_training_steps_11_15_from_{run}; no new run/eval launched",
            }
            for metric in TRAIN_METRICS:
                vals = [float(h[metric]) for h in hist if isinstance(h.get(metric), (int, float))]
                row[metric] = statistics.median(vals) if vals else None
            out.append(row)
        return out

    chain_rows.extend(prefix_estimates("gspo_preferred", "gspo_main", "gspo_estimated_prefix"))
    for step in range(11, 61):
        row = by_run_step.get(("gspo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gspo_preferred"})
    for step in range(61, 70):
        row = by_run_step.get(("gspo_kl_from60", step)) or by_run_step.get(("gspo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gspo_preferred"})

    gspo_est: dict[str, object] = {
        "chain": "gspo_preferred",
        "run": "gspo_estimated_after_from60",
        "step": 70,
        "status": "estimated",
        "source_path": "",
        "estimation_method": "median_of_gspo_preferred_training_steps_65_69; no new run/eval launched",
    }
    gspo_hist = [row for row in chain_rows if row["chain"] == "gspo_preferred" and int(row["step"]) < 70]
    for metric in TRAIN_METRICS:
        vals = [
            float(row[metric])
            for row in sorted(gspo_hist, key=lambda r: int(r["step"]))[-5:]
            if isinstance(row.get(metric), (int, float))
        ]
        gspo_est[metric] = statistics.median(vals) if vals else None
    chain_rows.append(gspo_est)

    chain_rows.extend(prefix_estimates("gdpo_preferred", "gdpo_main", "gdpo_estimated_prefix"))
    for step in range(11, 51):
        row = by_run_step.get(("gdpo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gdpo_preferred"})
    for step in range(51, 81):
        row = by_run_step.get(("gdpo_kl_from50", step)) or by_run_step.get(("gdpo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gdpo_preferred"})

    chain_rows.sort(key=lambda r: (str(r["chain"]), int(r["step"])))
    add_roll5(chain_rows, "chain", TRAIN_METRICS)
    return chain_rows


def load_validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in read_csv(LOCAL_AUDIT_DIR / "validation_overall.csv"):
        row: dict[str, object] = {
            "run": raw["run"],
            "step": int(raw["step"]),
            "rows": int(float(raw["rows"])),
            "questions": int(float(raw["questions"])),
            "source_path": raw["source_path"],
            "status": "observed",
            "estimation_method": "",
        }
        for metric in EVAL_METRICS:
            row[metric] = parse_number(raw.get(metric, ""))
        rows.append(row)
    rows.sort(key=lambda r: (str(r["run"]), int(r["step"])))
    return rows


def nearest_observed(rows: list[dict[str, object]], metric: str, step: int) -> tuple[float | None, str]:
    candidates = [
        (abs(int(row["step"]) - step), int(row["step"]), float(row[metric]))
        for row in rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not candidates:
        return None, ""
    _, source_step, value = min(candidates)
    return value, f"nearest_observed_same_run_step_{source_step}"


def fill_eval_row_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run"])].append(row)
    filled: list[dict[str, object]] = []
    for row in rows:
        out = dict(row)
        missing_methods = []
        for metric in EVAL_METRICS:
            if isinstance(out.get(metric), (int, float)):
                continue
            value, method = nearest_observed(by_run[str(row["run"])], metric, int(row["step"]))
            out[metric] = value
            if value is not None:
                missing_methods.append(f"{metric}:{method}")
        if missing_methods:
            out["status"] = "mixed_observed_estimated"
            out["estimation_method"] = "; ".join(missing_methods)
        filled.append(out)
    return filled


def extrapolate_eval_metric(rows: list[dict[str, object]], metric: str, next_step: int) -> tuple[float | None, str]:
    observed = [
        row
        for row in sorted(rows, key=lambda r: int(r["step"]))
        if int(row["step"]) < next_step and isinstance(row.get(metric), (int, float))
    ]
    if len(observed) >= 2:
        last = observed[-1]
        prev = observed[-2]
        value = float(last[metric]) + 0.5 * (float(last[metric]) - float(prev[metric]))
        method = f"half_linear_extrapolation_from_steps_{prev['step']}_{last['step']}"
    elif observed:
        value = float(observed[-1][metric])
        method = f"carry_forward_step_{observed[-1]['step']}"
    else:
        return None, ""
    if metric not in EVAL_UNBOUNDED:
        value = max(0.0, min(1.0, value))
    return value, method


def eval_chain(filled_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run_step = {(str(row["run"]), int(row["step"])): row for row in filled_rows}
    chain_rows: list[dict[str, object]] = []

    for step in [10, 20, 30, 40, 50, 60]:
        row = by_run_step.get(("gspo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gspo_preferred"})

    gspo_rows = [row for row in chain_rows if row["chain"] == "gspo_preferred"]
    gspo_est: dict[str, object] = {
        "chain": "gspo_preferred",
        "run": "gspo_estimated_after_from60",
        "step": 70,
        "rows": "",
        "questions": "",
        "source_path": "",
        "status": "estimated",
    }
    methods = []
    for metric in EVAL_METRICS:
        value, method = extrapolate_eval_metric(gspo_rows, metric, 70)
        gspo_est[metric] = value
        if method:
            methods.append(f"{metric}:{method}")
    gspo_est["estimation_method"] = (
        "no GSPO restart validation/generations for step70; "
        + "; ".join(methods)
    )
    chain_rows.append(gspo_est)

    for step in [10, 20, 30, 40, 50]:
        row = by_run_step.get(("gdpo_main", step))
        if row:
            chain_rows.append({**row, "chain": "gdpo_preferred"})
    for step in [60, 70, 80]:
        row = by_run_step.get(("gdpo_kl_from50", step))
        if row:
            chain_rows.append({**row, "chain": "gdpo_preferred"})

    chain_rows.sort(key=lambda r: (str(r["chain"]), int(r["step"])))
    return chain_rows


def long_rows(
    rows: list[dict[str, object]],
    metrics: list[str],
    *,
    include_roll5: bool = False,
    chain_or_run_key: str = "run",
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        for metric in metrics:
            status = row.get("status", "")
            method = row.get("estimation_method", "")
            value = row.get(metric)
            if value is None:
                continue
            entry: dict[str, object] = {
                chain_or_run_key: row.get(chain_or_run_key, ""),
                "run": row.get("run", ""),
                "step": row.get("step", ""),
                "metric": metric,
                "value": value,
                "status": status,
                "estimation_method": method,
                "source_path": row.get("source_path", ""),
            }
            if include_roll5:
                entry["roll5"] = row.get(f"{metric}__roll5")
            out.append(entry)
    return out


def method_for_metric(row: dict[str, object], metric: str) -> str:
    method = str(row.get("estimation_method", ""))
    for part in method.split("; "):
        prefix = f"{metric}:"
        if part.startswith(prefix):
            return part[len(prefix) :]
    return method


def estimated_values(train_chain_rows: list[dict[str, object]], eval_chain_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in train_chain_rows:
        if row.get("status") != "estimated":
            continue
        for metric in TRAIN_METRICS:
            rows.append(
                {
                    "table": "training_preferred_chain",
                    "chain": row["chain"],
                    "step": row["step"],
                    "metric": metric,
                    "value": row.get(metric),
                    "method": row.get("estimation_method", ""),
                }
            )
    for row in eval_chain_rows:
        status = str(row.get("status", ""))
        if status == "estimated":
            metrics = EVAL_METRICS
        elif status == "mixed_observed_estimated":
            method = str(row.get("estimation_method", ""))
            metrics = [metric for metric in EVAL_METRICS if f"{metric}:" in method]
        else:
            continue
        for metric in metrics:
            rows.append(
                {
                    "table": "training_time_validation_eval_chain",
                    "chain": row["chain"],
                    "step": row["step"],
                    "metric": metric,
                    "value": row.get(metric),
                    "method": method_for_metric(row, metric),
                }
            )
    return rows


def metric_plan_rows() -> list[dict[str, str]]:
    return [
        {
            "dimension": "Training Reward / Outcome",
            "data_file": "training_preferred_chain_long.csv",
            "x": "step",
            "y_metrics": "train/thinkstream/reward/score/mean; train/thinkstream/reward/outcome/mean",
            "smoothing": "roll5",
            "conclusion_hint": "No stable monotonic rise; use rolling mean to show noise.",
        },
        {
            "dimension": "Recall Usage and Request Hit",
            "data_file": "training_preferred_chain_long.csv",
            "x": "step",
            "y_metrics": "call_traj_frac; recall_call_count/mean; recall_support_request_hit_rate/mean",
            "smoothing": "roll5",
            "conclusion_hint": "Usage/request-hit are clearer than returned-hit for whether recall is being used.",
        },
        {
            "dimension": "Recall Effectiveness",
            "data_file": "training_preferred_chain_long.csv",
            "x": "step",
            "y_metrics": "answer_used_per_labeled/rate; answer_success_per_labeled/rate; post_recall_outcome_mean/mean",
            "smoothing": "roll5",
            "conclusion_hint": "Track use and successful use; avoid overclaiming answer_success_per_used if usage shifts.",
        },
        {
            "dimension": "Compact Memory Quality",
            "data_file": "training_preferred_chain_long.csv",
            "x": "step",
            "y_metrics": "compress_quality/mean; cover_ok/mean; boundary_score/mean; source_precision/mean",
            "smoothing": "roll5",
            "conclusion_hint": "These are the strongest memory-quality curves, especially compress_quality and cover_ok.",
        },
        {
            "dimension": "Training-Time Eval Accuracy",
            "data_file": "training_time_validation_eval_chain_long.csv",
            "x": "checkpoint step",
            "y_metrics": "trajectory_mean_correct_question_weighted; score_question_weighted; outcome_question_weighted",
            "smoothing": "none",
            "conclusion_hint": "This is the train-time validation/eval curve, not LVB/OVO/StreamingBench benchmark eval.",
        },
    ]


def first_last(rows: list[dict[str, object]], chain: str, metric: str) -> tuple[float | None, float | None]:
    vals = [
        (int(row["step"]), float(row[metric]))
        for row in rows
        if row.get("chain") == chain and row.get("status") != "estimated" and isinstance(row.get(metric), (int, float))
    ]
    if not vals:
        return None, None
    vals.sort()
    return vals[0][1], vals[-1][1]


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def write_readme(
    train_raw: list[dict[str, object]],
    train_pref: list[dict[str, object]],
    eval_pref: list[dict[str, object]],
) -> None:
    gspo_eval70 = next(row for row in eval_pref if row["chain"] == "gspo_preferred" and int(row["step"]) == 70)
    gdpo_eval80 = next(row for row in eval_pref if row["chain"] == "gdpo_preferred" and int(row["step"]) == 80)
    gspo_score_first, gspo_score_last = first_last(train_pref, "gspo_preferred", "train/thinkstream/reward/score/mean")
    gdpo_score_first, gdpo_score_last = first_last(train_pref, "gdpo_preferred", "train/thinkstream/reward/score/mean")
    gspo_mem_first, gspo_mem_last = first_last(train_pref, "gspo_preferred", "train/thinkstream/compress/compress_quality/mean")
    gdpo_mem_first, gdpo_mem_last = first_last(train_pref, "gdpo_preferred", "train/thinkstream/compress/compress_quality/mean")

    lines = [
        "# GSPO/GDPO Training-Time Eval Score Record 2026-05-25",
        "",
        "Scope: existing local training logs plus training-time validation/generation outputs only. "
        "This record intentionally excludes LVB, OVO, and StreamingBench benchmark eval results.",
        "",
        "No new training run, restart, or model eval was launched for this export.",
        "",
        "## Files",
        "- `training_raw_metrics_wide.csv`: observed per-step train metrics parsed from every available run log.",
        "- `training_preferred_chain_wide.csv`: preferred restart-aware train curve with GSPO step70 estimated.",
        "- `training_preferred_chain_long.csv`: long-format plotting table for reward/outcome/recall/memory metrics.",
        "- `training_time_validation_eval_chain_wide.csv`: training-time validation/eval checkpoint curve.",
        "- `training_time_validation_eval_chain_long.csv`: long-format validation/eval plotting table.",
        "- `estimated_metric_values.csv`: every explicitly estimated value and the method used.",
        "- `plot_metric_plan.csv`: recommended 4-5 plot groupings and metric choices.",
        "- `validation_by_category_observed.csv`, `validation_by_task_observed.csv`: observed validation breakdowns.",
        "",
        "## Restart-Aware Chains",
        "- `gspo_preferred`: estimated train steps 1-10, `gspo_main` steps 11-60, then `gspo_kl_from60` steps 61-69, plus estimated step70.",
        "- `gdpo_preferred`: estimated train steps 1-10, `gdpo_main` steps 11-50, then `gdpo_kl_from50` steps 51-80.",
        "",
        "## Estimation Policy",
        "- Missing per-metric values inside an observed validation row are filled from the nearest observed checkpoint in the same run.",
        "- Missing training steps 1-10 are backfilled from the median of the first observed training steps 11-15.",
        "- Missing GSPO training step70 is estimated as the median of GSPO preferred training steps 65-69.",
        "- Missing GSPO validation/eval step70 is estimated by half-step linear extrapolation from validation steps 50 and 60.",
        "- All estimated rows are marked `estimated`; mixed observed rows are marked `mixed_observed_estimated`.",
        "",
        "## Key Values",
        f"- Observed train rows exported: {len(train_raw)}.",
        f"- Preferred chain train rows exported: {len(train_pref)}, including "
        f"{sum(1 for row in train_pref if row.get('status') == 'estimated')} estimated rows.",
        f"- GSPO estimated validation/eval step70 accuracy: {pct(float(gspo_eval70['trajectory_mean_correct_question_weighted']))}; "
        f"score: {float(gspo_eval70['score_question_weighted']):.3f}.",
        f"- GDPO preferred validation/eval step80 accuracy: {pct(float(gdpo_eval80['trajectory_mean_correct_question_weighted']))}; "
        f"score: {float(gdpo_eval80['score_question_weighted']):.3f}.",
        f"- GSPO train score first/last observed in preferred chain: {gspo_score_first:.3f} -> {gspo_score_last:.3f}.",
        f"- GDPO train score first/last observed in preferred chain: {gdpo_score_first:.3f} -> {gdpo_score_last:.3f}.",
        f"- GSPO compact memory quality first/last observed: {gspo_mem_first:.3f} -> {gspo_mem_last:.3f}.",
        f"- GDPO compact memory quality first/last observed: {gdpo_mem_first:.3f} -> {gdpo_mem_last:.3f}.",
        "",
        "## Plot Recommendations",
        "1. Training Reward / Outcome: plot `train/thinkstream/reward/score/mean` and "
        "`train/thinkstream/reward/outcome/mean` with `roll5`; conclusion should say no stable monotonic rise.",
        "2. Recall Usage and Request Hit: plot `call_traj_frac`, `recall_call_count/mean`, "
        "`recall_support_request_hit_rate/mean`.",
        "3. Recall Effectiveness: plot `answer_used_per_labeled/rate`, `answer_success_per_labeled/rate`, "
        "and optionally `post_recall_outcome_mean/mean`.",
        "4. Compact Memory Quality: plot `compress_quality/mean`, `cover_ok/mean`, "
        "`boundary_score/mean`, `source_precision/mean`.",
        "5. Training-Time Eval Accuracy: plot `trajectory_mean_correct_question_weighted` from "
        "`training_time_validation_eval_chain_long.csv`.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = parse_train_logs()
    add_roll5(train_raw, "run", TRAIN_METRICS)
    train_pref = train_chain(train_raw)

    validation_raw = load_validation_rows()
    validation_filled = fill_eval_row_metrics(validation_raw)
    validation_pref = eval_chain(validation_filled)

    train_wide_fields = [
        "chain",
        "run",
        "step",
        "status",
        "source_path",
        "estimation_method",
        *TRAIN_METRICS,
        *[f"{metric}__roll5" for metric in TRAIN_METRICS],
    ]
    eval_wide_fields = [
        "chain",
        "run",
        "step",
        "rows",
        "questions",
        "status",
        "source_path",
        "estimation_method",
        *EVAL_METRICS,
    ]

    write_csv(OUT_DIR / "training_raw_metrics_wide.csv", train_wide_fields[1:], train_raw)
    write_csv(OUT_DIR / "training_preferred_chain_wide.csv", train_wide_fields, train_pref)
    write_csv(
        OUT_DIR / "training_preferred_chain_long.csv",
        ["chain", "run", "step", "metric", "value", "roll5", "status", "estimation_method", "source_path"],
        long_rows(train_pref, TRAIN_METRICS, include_roll5=True, chain_or_run_key="chain"),
    )
    write_csv(OUT_DIR / "training_time_validation_eval_observed_filled_wide.csv", eval_wide_fields[1:], validation_filled)
    write_csv(OUT_DIR / "training_time_validation_eval_chain_wide.csv", eval_wide_fields, validation_pref)
    write_csv(
        OUT_DIR / "training_time_validation_eval_chain_long.csv",
        ["chain", "run", "step", "metric", "value", "status", "estimation_method", "source_path"],
        long_rows(validation_pref, EVAL_METRICS, chain_or_run_key="chain"),
    )
    write_csv(
        OUT_DIR / "estimated_metric_values.csv",
        ["table", "chain", "step", "metric", "value", "method"],
        estimated_values(train_pref, validation_pref),
    )
    write_csv(
        OUT_DIR / "plot_metric_plan.csv",
        ["dimension", "data_file", "x", "y_metrics", "smoothing", "conclusion_hint"],
        metric_plan_rows(),
    )

    for name in ["inventory.csv", "validation_by_category.csv", "validation_by_task.csv"]:
        src = LOCAL_AUDIT_DIR / name
        if src.exists():
            dst_name = name
            if name.startswith("validation_by_"):
                dst_name = name.replace(".csv", "_observed.csv")
            text = src.read_text().replace("\r\n", "\n").replace("\r", "\n")
            (OUT_DIR / dst_name).write_text(text)

    write_readme(train_raw, train_pref, validation_pref)
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
