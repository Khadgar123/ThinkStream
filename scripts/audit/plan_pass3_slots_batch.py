"""Build external Pass3A slot-plan files for batch-balanced refreshes.

The generated directory can be injected into pass3A with:

  THINKSTREAM_PASS3A_SLOT_PLAN_DIR=data/agent_v5/audits/slot_plans_v1295/slots

This keeps batch-level family/style/lifecycle decisions outside the LLM. The
LLM receives fixed slots and only writes the natural question/options.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from scripts.agent_data.pass3_slot_planner import (
    balanced_family_targets,
    build_pass3_slot_plan,
    pass3a_batch_source_row_targets,
    pass3a_response_row_budget,
    pass3a_source_slot_requirements_from_rows,
    slot_plan_summary,
)


def _parse_batches(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            out.extend(f"batch{i}" for i in range(int(lo_s), int(hi_s) + 1))
        elif part.startswith("batch"):
            out.append(part)
        else:
            out.append(f"batch{int(part)}")
    return out


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iter_video_rows(root: Path, batches: Iterable[str]) -> List[Dict]:
    rows: List[Dict] = []
    for batch in batches:
        batch_dir = root / batch
        # Prefer task_cards so the plan covers exactly the refreshable set when
        # old cards already exist. Fall back to evidence files for fresh runs.
        card_dir = batch_dir / "task_cards"
        if card_dir.exists():
            for path in sorted(card_dir.glob("*.json")):
                rows.append({"batch": batch, "video_id": path.stem})
            continue
        evidence_dir = batch_dir / "evidence_1b"
        if not evidence_dir.exists():
            evidence_dir = batch_dir / "evidence_1a"
        for path in sorted(evidence_dir.glob("*.json")):
            rows.append({"batch": batch, "video_id": path.stem})
    return rows


def _evidence_path(root: Path, batch: str, video_id: str) -> Path:
    batch_dir = root / batch
    p = batch_dir / "evidence_1b" / f"{video_id}.json"
    if p.exists():
        return p
    return batch_dir / "evidence_1a" / f"{video_id}.json"


def _balanced_targets(video_id: str, mode: str) -> Dict[str, int]:
    """Back-compat alias; the policy lives in pass3_slot_planner."""
    return balanced_family_targets(video_id, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/agent_v5")
    parser.add_argument("--batches", default="1-11")
    parser.add_argument("--out-dir", default="data/agent_v5/audits/slot_plans_v1295")
    parser.add_argument("--mode", choices=["full", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_dir)
    slot_dir = out_root / "slots"
    slot_dir.mkdir(parents=True, exist_ok=True)

    rows = _iter_video_rows(root, _parse_batches(args.batches))
    if args.limit:
        rows = rows[: int(args.limit)]

    loaded_rows: List[Dict] = []
    for row in rows:
        batch = str(row["batch"])
        video_id = str(row["video_id"])
        evidence = _load_json(_evidence_path(root, batch, video_id))
        if not isinstance(evidence, list) or not evidence:
            loaded_rows.append({**row, "evidence": None, "num_chunks": 0})
            continue
        num_chunks = max(
            int(cap.get("chunk_idx", i))
            for i, cap in enumerate(evidence)
            if isinstance(cap, dict)
        ) + 1
        loaded_rows.append({
            **row,
            "evidence": evidence,
            "num_chunks": num_chunks,
        })

    batch_source_rows = pass3a_batch_source_row_targets({
        str(row["video_id"]): int(row["num_chunks"])
        for row in loaded_rows
        if int(row.get("num_chunks") or 0) > 0
    })

    counters = Counter()
    family = Counter()
    answer_form = Counter()
    response_unit_answer_form = Counter()
    question_way = Counter()
    lifecycle = Counter()
    task_family = Counter()
    task_subtype = Counter()
    timing_type = Counter()
    slot_group = Counter()
    slot_subtype = Counter()
    temporal_bucket = Counter()
    answer_behavior = Counter()
    support_bin = Counter()
    slot_source = Counter()
    slot_keep_reason = Counter()
    slot_plan_reason = Counter()
    planned_response_rows = 0
    planned_source_rows = Counter()
    planned_source_slots = Counter()
    per_video_slots: List[int] = []
    per_video_plan: List[Dict] = []

    for row in loaded_rows:
        batch = str(row["batch"])
        video_id = str(row["video_id"])
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            counters["missing_evidence"] += 1
            continue
        source_row_targets = batch_source_rows.get(video_id) or {}
        if source_row_targets:
            response_row_budget = pass3a_response_row_budget(int(row["num_chunks"]))
            planned_response_rows += response_row_budget
            planned_source_rows.update(source_row_targets)
            planned_source_slots.update(pass3a_source_slot_requirements_from_rows(source_row_targets))
            per_video_plan.append({
                "batch": batch,
                "video_id": video_id,
                "num_chunks": int(row["num_chunks"]),
                "target_response_rows": int(response_row_budget),
                "target_response_rate": round(
                    response_row_budget / max(1, int(row["num_chunks"])),
                    4,
                ),
                "source_rows": dict(source_row_targets),
                "source_slots": pass3a_source_slot_requirements_from_rows(source_row_targets),
            })
        targets = _balanced_targets(video_id, args.mode)
        slot_audit = Counter()
        slots = build_pass3_slot_plan(
            evidence,
            video_id,
            targets,
            audit=slot_audit,
            source_row_targets=source_row_targets,
            seed=int(args.seed),
        )
        slot_plan_reason.update(slot_audit)
        if not slots:
            counters["empty_slots"] += 1
            continue
        summary = slot_plan_summary(slots)
        family.update(summary["family"])
        answer_form.update(summary["answer_form"])
        for slot in slots:
            af = str(slot.get("answer_form") or "")
            response_unit_answer_form[af] += max(1, len(slot.get("answer_chunks") or []))
        question_way.update(summary["question_way"])
        lifecycle.update(summary["lifecycle"])
        slot_source.update(summary["slot_source"])
        slot_keep_reason.update(summary["slot_keep_reason"])
        task_family.update(summary["task_family"])
        task_subtype.update(summary["task_subtype"])
        timing_type.update(summary["timing_type"])
        slot_group.update(summary["slot_group"])
        slot_subtype.update(summary["slot_subtype"])
        temporal_bucket.update(summary["temporal_bucket"])
        answer_behavior.update(summary["answer_behavior"])
        support_bin.update(summary["support_bin"])
        per_video_slots.append(len(slots))
        (slot_dir / f"{video_id}.json").write_text(
            json.dumps({
                "batch": batch,
                "video_id": video_id,
                "mode": args.mode,
                "slots": slots,
            }, ensure_ascii=False, indent=2) + "\n"
        )
        counters["ok"] += 1

    report = {
        "rows": len(rows),
        "slot_dir": str(slot_dir),
        "counters": dict(counters),
        "slots_total": int(sum(per_video_slots)),
        "slots_per_video_mean": round(sum(per_video_slots) / max(len(per_video_slots), 1), 3),
        "family": dict(family.most_common()),
        "answer_form": dict(answer_form.most_common()),
        "response_unit_answer_form": dict(response_unit_answer_form.most_common()),
        "question_way": dict(question_way.most_common()),
        "lifecycle": dict(lifecycle.most_common()),
        "slot_source": dict(slot_source.most_common()),
        "planned_response_rows": int(planned_response_rows),
        "planned_source_rows": dict(planned_source_rows.most_common()),
        "planned_source_slots": dict(planned_source_slots.most_common()),
        "per_video_plan_count": len(per_video_plan),
        "slot_keep_reason": dict(slot_keep_reason.most_common()),
        "slot_plan_keep_drop_reason": dict(slot_plan_reason.most_common()),
        "task_family": dict(task_family.most_common()),
        "task_subtype": dict(task_subtype.most_common()),
        "timing_type": dict(timing_type.most_common()),
        "slot_group": dict(slot_group.most_common()),
        "slot_subtype": dict(slot_subtype.most_common()),
        "temporal_bucket": dict(temporal_bucket.most_common()),
        "answer_behavior": dict(answer_behavior.most_common()),
        "support_bin_top": dict(support_bin.most_common(30)),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out_root / "per_video_plan.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in per_video_plan)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
