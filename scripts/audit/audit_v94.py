"""v9.4 dry-run audit — comprehensive distribution report.

Single-command summary of: cards × family × form, placements × tier × seq,
samples × family × form × tier, verify pass rate per family.

Usage:
    python -m scripts.audit.audit_v94                    # default dirs
    python -m scripts.audit.audit_v94 --root data/agent_v5
    python -m scripts.audit.audit_v94 --report /tmp/v94_report.txt
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_data_v5.pass3a_cards import FAMILY_TARGETS
from scripts.agent_data_v5.pass3d_select import FAMILY_TO_OVO, OVO_TASK_QUOTA


# Expected distribution targets (used to flag anomalies).
TARGET_FORM_PCT = {
    "multiple_choice": 0.55,  # ≥
    "binary":          0.15,  # ≤
    "number":          0.10,  # ~
    "short_exact":     0.10,  # ~
    "descriptive":     0.10,  # ~
}

TARGET_TIER_PCT = {
    # Across transient placements only — persistent counted separately.
    "easy_in_visual":          0.30,
    "medium_in_compressed":    0.25,
    "hard_history_only":       0.15,
}


def _bar(pct: float, width: int = 20) -> str:
    n = int(pct * width)
    return "█" * n + "·" * (width - n)


def _fmt_pct(n: int, total: int, width: int = 6) -> str:
    if total == 0:
        return f"{0:>{width}.1%}"
    return f"{n / total:>{width}.1%}"


def _section(title: str, lines):
    out = ["", "─" * 78, f"  {title}", "─" * 78]
    out.extend(lines)
    return "\n".join(out)


def audit_cards(cards_dir: Path) -> str:
    """task_cards/{video_id}.json — list of card dicts."""
    families = Counter()
    forms = Counter()
    by_family_form: defaultdict = defaultdict(Counter)
    visibility = Counter()
    n_files = 0
    n_cards = 0

    for f in sorted(cards_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            cards = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        for c in cards:
            fam = c.get("family", "?")
            form = c.get("answer_form", "?")
            families[fam] += 1
            forms[form] += 1
            by_family_form[fam][form] += 1
            visibility[c.get("visibility_type", "?")] += 1
            n_cards += 1

    if n_cards == 0:
        return _section("PASS 3-A — Cards", ["  (no cards found)"])

    lines = [
        f"  Videos with cards: {n_files}",
        f"  Total cards:       {n_cards}",
        f"  Mean cards/video:  {n_cards / max(n_files, 1):.1f}",
        "",
        "  Per family (target / actual):",
    ]
    for fam in sorted(FAMILY_TARGETS.keys()):
        tgt = FAMILY_TARGETS[fam]
        n = families.get(fam, 0)
        avg = n / max(n_files, 1)
        gap = avg / tgt if tgt > 0 else 1.0
        flag = "" if 0.5 <= gap <= 1.6 else ("  LOW" if gap < 0.5 else "  HIGH")
        lines.append(f"    {fam:4s}  target={tgt}  avg={avg:5.2f}  total={n:4d}{flag}")

    lines.append("")
    lines.append("  answer_form distribution:")
    for form in ("multiple_choice", "binary", "number", "short_exact", "descriptive"):
        n = forms.get(form, 0)
        pct = n / n_cards
        target = TARGET_FORM_PCT.get(form, 0)
        bar = _bar(pct)
        lines.append(f"    {form:18s} {n:5d}  {pct:5.1%}  {bar}  target={target:.0%}")

    lines.append("")
    lines.append("  Form × family heatmap (% of family):")
    forms_show = ["multiple_choice", "binary", "number", "short_exact", "descriptive"]
    head = "    fam  | " + " | ".join(f"{f[:8]:>8s}" for f in forms_show)
    lines.append(head)
    lines.append("    " + "-" * (len(head) - 4))
    for fam in sorted(by_family_form.keys()):
        total_f = sum(by_family_form[fam].values())
        if total_f == 0:
            continue
        row = f"    {fam:4s} | " + " | ".join(
            f"{by_family_form[fam].get(f, 0) / total_f * 100:6.0f}%  " for f in forms_show
        )
        lines.append(row)

    lines.append("")
    lines.append(f"  visibility_type: persistent={visibility.get('persistent',0)} "
                 f"transient={visibility.get('transient',0)}")

    return _section("PASS 3-A — Cards", lines)


def _family_from_card_id(card_id: str) -> str:
    """Card IDs are formatted as {video}_{FAMILY}_{nnn}. Extract family from
    the second-to-last underscore-segment. Returns '?' on parse failure."""
    if not card_id:
        return "?"
    parts = card_id.rsplit("_", 2)  # ["video", "FAMILY", "nnn"] (or shorter)
    if len(parts) >= 2:
        return parts[-2]
    return "?"


def audit_placements(placements_dir: Path) -> str:
    """placements/{video_id}.json — {placements: [...], trajectories: [...]}"""
    n_files = 0
    n_placements = 0
    tiers = Counter()
    seqs = Counter()
    family_x_tier: defaultdict = defaultdict(Counter)
    untagged = 0

    for f in sorted(placements_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        for p in d.get("placements", []):
            n_placements += 1
            tier = p.get("difficulty_tier")
            if not tier:
                untagged += 1
                tier = "UNTAGGED"
            tiers[tier] += 1
            seqs[p.get("sequence_type", "?")] += 1
            # Family may not be denormalized on legacy placements (pre-v9.1);
            # fall back to parsing the card_id.
            fam = p.get("family") or _family_from_card_id(p.get("card_id", ""))
            family_x_tier[fam][tier] += 1

    if n_placements == 0:
        return _section("PASS 3-B — Placements", ["  (no placements found)"])

    lines = [
        f"  Videos with placements: {n_files}",
        f"  Total placements:       {n_placements}",
        f"  Mean placements/video:  {n_placements / max(n_files, 1):.1f}",
        f"  UNTAGGED placements:    {untagged}  (must be 0 after v9.4 rerun)",
        "",
        "  difficulty_tier distribution:",
    ]
    for tier in ("easy_in_visual", "medium_in_compressed", "hard_history_only",
                 "persistent_spread", "event_watch", "multi_response", "UNTAGGED"):
        n = tiers.get(tier, 0)
        pct = n / n_placements
        bar = _bar(pct)
        target = TARGET_TIER_PCT.get(tier)
        target_str = f"  target={target:.0%}" if target else ""
        lines.append(f"    {tier:24s} {n:5d}  {pct:5.1%}  {bar}{target_str}")

    lines.append("")
    lines.append("  sequence_type distribution:")
    for seq, n in seqs.most_common():
        lines.append(f"    {seq:30s} {n:5d}  {_fmt_pct(n, n_placements)}")

    lines.append("")
    lines.append("  Family × tier (placements):")
    tier_cols = ["easy_in_visual", "medium_in_compressed", "hard_history_only",
                 "persistent_spread", "event_watch", "multi_response", "UNTAGGED"]
    head = "    fam  | " + " | ".join(f"{t[:7]:>7s}" for t in tier_cols)
    lines.append(head)
    lines.append("    " + "-" * (len(head) - 4))
    for fam in sorted(family_x_tier.keys()):
        row = f"    {fam:4s} | " + " | ".join(
            f"{family_x_tier[fam].get(t, 0):>7d}" for t in tier_cols
        )
        lines.append(row)

    return _section("PASS 3-B — Placements", lines)


def audit_samples_3c(samples_dir: Path) -> str:
    n_files = 0
    n_samples = 0
    sample_types = Counter()
    seqs = Counter()
    base_roles = Counter()
    actions = Counter()

    for f in sorted(samples_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            samples = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        for s in samples:
            n_samples += 1
            sample_types[s.get("sample_type", "?")] += 1
            seqs[s.get("sequence_type", "?")] += 1
            br = s.get("base_role")
            if br:
                base_roles[br] += 1
            actions[s.get("action", "?")] += 1

    if n_samples == 0:
        return _section("PASS 3-C — Samples", ["  (no samples found)"])

    lines = [
        f"  Videos with samples: {n_files}",
        f"  Total samples:       {n_samples}",
        f"  Mean samples/video:  {n_samples / max(n_files, 1):.1f}",
        "",
        "  action distribution:",
    ]
    for act, n in actions.most_common():
        lines.append(f"    {act:20s} {n:6d}  {_fmt_pct(n, n_samples)}")

    lines.append("")
    lines.append("  sample_type distribution:")
    for st, n in sample_types.most_common():
        lines.append(f"    {st:20s} {n:6d}  {_fmt_pct(n, n_samples)}")

    lines.append("")
    lines.append("  sequence_type distribution:")
    for sq, n in seqs.most_common():
        lines.append(f"    {sq:30s} {n:6d}  {_fmt_pct(n, n_samples)}")

    if base_roles:
        lines.append("")
        lines.append("  base_role distribution:")
        for br, n in base_roles.most_common():
            lines.append(f"    {br:25s} {n:6d}  {_fmt_pct(n, n_samples)}")

    return _section("PASS 3-C — Samples", lines)


def audit_verified(verified_dir: Path) -> str:
    n_files = 0
    n_total = 0
    n_passed = 0
    fail_reasons = Counter()
    by_family_pass: defaultdict = defaultdict(lambda: [0, 0])  # [passed, total]
    by_form_pass: defaultdict = defaultdict(lambda: [0, 0])
    by_tier_pass: defaultdict = defaultdict(lambda: [0, 0])

    for f in sorted(verified_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        for s in d.get("samples", []):
            v = s.get("verification") or {}
            passed = bool(v.get("passed", False))
            n_total += 1
            if passed:
                n_passed += 1
            for r in v.get("fail_reasons", []) or []:
                fail_reasons[r.split(":")[0]] += 1

            md = s.get("metadata", {}) or {}
            fam = md.get("family") or s.get("family") or "?"
            form = md.get("answer_form") or "?"
            # tier is on the placement, not always plumbed to sample;
            # try a few common spots.
            tier = (md.get("difficulty_tier")
                    or s.get("difficulty_tier")
                    or "?")

            by_family_pass[fam][1] += 1
            by_form_pass[form][1] += 1
            by_tier_pass[tier][1] += 1
            if passed:
                by_family_pass[fam][0] += 1
                by_form_pass[form][0] += 1
                by_tier_pass[tier][0] += 1

    if n_total == 0:
        return _section("PASS 4 — Verified", ["  (no verified samples found)"])

    lines = [
        f"  Videos with verified output: {n_files}",
        f"  Total samples:               {n_total}",
        f"  Passed:                      {n_passed}  ({n_passed/n_total:.1%})",
        f"  Failed:                      {n_total - n_passed}",
        "",
        "  Top fail reasons:",
    ]
    for r, n in fail_reasons.most_common(12):
        lines.append(f"    {r:40s} {n:6d}  {_fmt_pct(n, n_total)}")

    lines.append("")
    lines.append("  Pass rate by family:")
    for fam in sorted(by_family_pass.keys()):
        p, t = by_family_pass[fam]
        rate = p / max(t, 1)
        flag = "" if rate >= 0.7 else "  LOW"
        bar = _bar(rate, 15)
        lines.append(f"    {fam:6s}  {p:5d}/{t:5d}  {rate:5.1%}  {bar}{flag}")

    lines.append("")
    lines.append("  Pass rate by answer_form:")
    for form in ("multiple_choice", "binary", "number", "short_exact", "descriptive"):
        if form not in by_form_pass:
            continue
        p, t = by_form_pass[form]
        rate = p / max(t, 1)
        bar = _bar(rate, 15)
        lines.append(f"    {form:18s}  {p:5d}/{t:5d}  {rate:5.1%}  {bar}")

    lines.append("")
    lines.append("  Pass rate by difficulty_tier:")
    for tier in sorted(by_tier_pass.keys()):
        if tier == "?":
            continue
        p, t = by_tier_pass[tier]
        rate = p / max(t, 1)
        bar = _bar(rate, 15)
        lines.append(f"    {tier:25s}  {p:5d}/{t:5d}  {rate:5.1%}  {bar}")

    return _section("PASS 4 — Verified", lines)


def audit_ovo_coverage(verified_dir: Path) -> str:
    """Project the verified samples onto OVO_TASK_QUOTA targets."""
    by_ovo_task: Counter = Counter()
    for f in sorted(verified_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for s in d.get("samples", []):
            v = s.get("verification") or {}
            if not v.get("passed", False):
                continue
            md = s.get("metadata", {}) or {}
            fam = md.get("family") or s.get("family") or ""
            tasks = FAMILY_TO_OVO.get(fam, [])
            if isinstance(tasks, str):
                tasks = [tasks]
            for t in tasks:
                by_ovo_task[t] += 1

    if not by_ovo_task:
        return _section("OVO Task Coverage Projection", ["  (no verified samples)"])

    lines = ["  Projected OVO task coverage (passed samples × FAMILY_TO_OVO):"]
    for task in sorted(OVO_TASK_QUOTA.keys()):
        have = by_ovo_task.get(task, 0)
        need = OVO_TASK_QUOTA[task]
        gap = have - need
        flag = "" if gap >= 0 else f"  GAP {-gap}"
        bar = _bar(min(have / max(need, 1), 1.0), 20)
        lines.append(f"    {task:5s}  have={have:5d}  quota={need:5d}  {bar}{flag}")
    return _section("OVO Task Coverage Projection", lines)


def audit_pipeline_stats(stats_path: Path) -> str:
    """final/pipeline_stats.json — real verify pass/fail rates from the
    last run. verified/ files only keep PASSED samples after filter_samples,
    so verified pass-rate from disk is trivially 100%; pipeline_stats has
    the pre-filter input/output counts and per-reason fail breakdown."""
    if not stats_path.exists():
        return _section("pipeline_stats.json", ["  (not found — pipeline did not finalize)"])
    try:
        d = json.load(open(stats_path))
    except Exception as e:
        return _section("pipeline_stats.json", [f"  parse error: {e}"])

    lines = []
    pre = d.get("pre_verify_total") or d.get("input_samples") or d.get("total_samples")
    post = d.get("verified_passed") or d.get("passed") or d.get("final_samples")
    if pre and post:
        lines.append(f"  Pre-verify samples:  {pre}")
        lines.append(f"  Post-verify passed:  {post}  ({post/pre:.1%})")
    if "fail_reasons" in d:
        lines.append("")
        lines.append("  Top fail reasons (pre-filter):")
        for r, n in sorted(d["fail_reasons"].items(), key=lambda x: -x[1])[:15]:
            lines.append(f"    {r:40s} {n:6d}")
    if "global_family_distribution" in d:
        lines.append("")
        lines.append("  global_family_distribution (post-filter):")
        for f, n in sorted(d["global_family_distribution"].items()):
            lines.append(f"    {f:5s}  {n:6d}")
    if "action_distribution" in d:
        lines.append("")
        lines.append("  action_distribution (post-filter):")
        for a, n in sorted(d["action_distribution"].items()):
            lines.append(f"    {a:15s}  {n:6d}")
    return _section("pipeline_stats.json (real pass rate)", lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/agent_v5",
                   help="agent_v5 data root")
    p.add_argument("--report", default=None,
                   help="write report to file (default: stdout only)")
    args = p.parse_args()

    root = Path(args.root)
    sections = []

    sections.append(f"v9.4 DRY-RUN AUDIT — {root.resolve()}")
    sections.append("=" * 78)

    sections.append(audit_cards(root / "task_cards"))
    sections.append(audit_placements(root / "placements"))
    sections.append(audit_samples_3c(root / "samples_3c"))
    sections.append(audit_verified(root / "verified"))
    sections.append(audit_pipeline_stats(root / "final" / "pipeline_stats.json"))
    sections.append(audit_ovo_coverage(root / "verified"))

    out = "\n".join(sections)
    print(out)
    if args.report:
        Path(args.report).write_text(out)
        print(f"\n[written to {args.report}]")


if __name__ == "__main__":
    main()
