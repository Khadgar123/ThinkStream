#!/usr/bin/env python
"""Project the per-video question / placement distribution from current
config (no real data needed — pure constants math).

Outputs:
  1. Per-family card / placement counts (pre-cap, post-cap).
  2. Tier breakdown (easy_in_visual / medium_in_compressed / hard_history_only
     / event_watch / multi_response / persistent_spread / pn1).
  3. answer_form distribution.
  4. OVO 12-task share.
  5. "Needs recall" rate — the metric that drives whether RL learns to recall.
  6. Silent-rate projection (radius-0, naive).

All numbers are ESTIMATES based on per-family targets + average yield;
real numbers should match within ±15%.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants mirrored from scripts/agent_data_v5/{config,pass3a_cards,pass3b}
# ---------------------------------------------------------------------------

import os as _os
PROFILE = _os.environ.get("SIM_PROFILE", "v12.8").lower()  # v12.5 / v12.6 / v12.7 / v12.8

if PROFILE == "v12.5":
    FAMILY_TARGETS = {
        "F1": 3, "F2": 2, "S1": 2, "F3": 1, "R1": 1, "CR4": 2, "F4": 3,
        "E1": 1, "M1": 2, "E2": 3, "C1": 1, "CR1": 1, "CR5": 2, "P1": 2,
        "CR2": 1, "F7": 1, "F5": 1, "F6": 1, "N1": 2, "CR3": 1, "CR6": 1,
        "CR7": 1, "PN1": 44,
    }
    PN1_PER_SEC_CAP = None
    TIER_BONUS_T2 = 0.0
    TIER_BONUS_T3 = 0.0
    MAX_TRAJ = 5
    MAX_Q_PER_TRAJ = 5
elif PROFILE == "v12.6":
    FAMILY_TARGETS = {
        "F1": 3, "F2": 2, "S1": 2, "F3": 1, "R1": 1, "CR4": 2, "F4": 3,
        "E1": 1, "M1": 2, "E2": 3, "C1": 1, "CR1": 1, "CR5": 2, "P1": 2,
        "CR2": 1, "F7": 1, "F5": 1, "F6": 2, "N1": 4, "CR3": 1, "CR6": 1,
        "CR7": 1, "PN1": 44,
    }
    PN1_PER_SEC_CAP = 0.10
    TIER_BONUS_T2 = 1.0
    TIER_BONUS_T3 = 2.0
    MAX_TRAJ = 5
    MAX_Q_PER_TRAJ = 5
elif PROFILE == "v12.7":
    FAMILY_TARGETS = {
        "F1": 3, "F2": 2, "S1": 2, "F3": 1, "R1": 1, "CR4": 2, "F4": 3,
        "E1": 1, "M1": 2, "E2": 3, "C1": 1, "CR1": 1, "CR5": 2, "P1": 2,
        "CR2": 1, "F7": 1, "F5": 1, "F6": 2, "N1": 4, "CR3": 1, "CR6": 1,
        "CR7": 1, "PN1": 44,
    }
    PN1_PER_SEC_CAP = 0.06
    TIER_BONUS_T2 = 1.0
    TIER_BONUS_T3 = 2.0
    MAX_TRAJ = 1
    MAX_Q_PER_TRAJ = 12
else:  # v12.8 — silent-leaning balanced config
    FAMILY_TARGETS = {
        "F1": 4, "F2": 2, "S1": 2, "F3": 1, "R1": 1, "CR4": 2, "F4": 3,
        "E1": 1, "M1": 2, "E2": 3, "C1": 1, "CR1": 1, "CR5": 2, "P1": 2,
        "CR2": 1, "F7": 1, "F5": 1, "F6": 2, "N1": 4, "CR3": 1, "CR6": 1,
        "CR7": 1, "PN1": 44,
    }
    PN1_PER_SEC_CAP = 0.04
    TIER_BONUS_T2 = 1.5
    TIER_BONUS_T3 = 2.5
    MAX_TRAJ = 1
    MAX_Q_PER_TRAJ = 8

# OVO 12-task share (canonical, sorted desc)
OVO_TASK_SHARE = {
    "EPM": 0.181, "HLD": 0.113, "OJR": 0.112, "STU": 0.109, "OCR": 0.091,
    "ASI": 0.090, "ATR": 0.071, "ACR": 0.066, "FPD": 0.062, "REC": 0.050,
    "CRR": 0.029, "SSR": 0.026,
}

# Family → OVO task (from pass3a_cards.py FAMILY_TARGETS comments)
FAMILY_OVO = {
    "F1": "OCR", "F2": "ATR", "S1": "ATR", "F3": "OJR", "R1": "OJR",
    "CR4": "OJR", "F4": "STU", "E1": "ACR", "M1": "ACR", "E2": "EPM",
    "C1": "EPM", "CR1": "EPM", "CR5": "CRR", "P1": "ASI", "CR2": "ASI",
    "F7": "SSR", "F5": "REC", "F6": "FPD", "N1": "HLD",
    "CR3": None, "CR6": None, "CR7": None,  # uncategorized reasoning
    "PN1": "narration",  # not in OVO
}

# Family → answer_form (from FAMILY_PROMPTS in pass3a_cards.py)
FAMILY_FORM = {
    "F1": "short_exact", "F2": "multiple_choice", "S1": "descriptive",
    "F3": "number", "R1": "multiple_choice", "CR4": "multiple_choice",
    "F4": "multiple_choice", "E1": "multiple_choice", "M1": "descriptive",
    "E2": "binary_or_mc",  # E2 has both binary + MC variants
    "C1": "multiple_choice", "CR1": "multiple_choice", "CR5": "descriptive",
    "P1": "multiple_choice", "CR2": "multiple_choice", "F7": "binary",
    "F5": "number", "F6": "multiple_choice", "N1": "multiple_choice",
    "CR3": "multiple_choice", "CR6": "multiple_choice", "CR7": "multiple_choice",
    "PN1": "descriptive",
}

# Family → visibility class (persistent vs transient).
# Persistent = answer doesn't change over time; placed as 3 spread, all "easy".
# Transient  = event-driven; gets 3-tier placement (T1+T2+T3).
PERSISTENT = {"F1", "S1", "CR3", "CR6", "M1"}
# Multi-probe families produce more placements per card (multi-tier same Q).
MULTI_PROBE_PROBES = {
    "F7": 5,  # binary multi-probe across step_chunk
    "F5": 4,  # counting multi-probe (count grows)
    "CR5": 3,  # multi-response reasoning
}

# pass3b density caps (profile-driven)
MAX_TRAJECTORIES_PER_VIDEO = MAX_TRAJ
MAX_QUESTIONS_PER_TRAJECTORY = MAX_Q_PER_TRAJ
NON_PN1_CAP = MAX_TRAJECTORIES_PER_VIDEO * MAX_QUESTIONS_PER_TRAJECTORY

# Final render cap (after pass3c base sample addition).
if PROFILE == "v12.8":
    MAX_SAMPLES_PER_VIDEO = 30
elif PROFILE == "v12.7":
    MAX_SAMPLES_PER_VIDEO = 25
else:
    MAX_SAMPLES_PER_VIDEO = 15

# Pass3a verify yield (from comments — average across families).
VERIFY_YIELD = 0.88

# Tier 3 only emits if video long enough (line 835: hist_hi >= num_chunks*2//3).
# Approximate gate: T3 dropped on videos < ~120s (need >= 2 compress events
# past support_end, first compress at chunk 53).
def t3_emit_prob(duration_sec: int) -> float:
    if duration_sec < 90:
        return 0.0
    if duration_sec < 120:
        return 0.3
    if duration_sec < 180:
        return 0.7
    return 1.0

# T2 always emits if video > ~80s (needs 1 compress past support).
def t2_emit_prob(duration_sec: int) -> float:
    if duration_sec < 60:
        return 0.0
    if duration_sec < 90:
        return 0.5
    return 1.0

# PN1 candidate yield depends on novelty events ~ duration.
# 44 cap, but on short videos novelty events are limited.
def pn1_actual(duration_sec: int) -> int:
    # Linear with duration up to cap; ~0.25 events/sec novel.
    base = min(int(duration_sec * 0.25), int(FAMILY_TARGETS["PN1"] * VERIFY_YIELD))
    if PN1_PER_SEC_CAP is not None:
        return min(base, max(2, int(duration_sec * PN1_PER_SEC_CAP)))
    return base


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(duration_sec: int) -> dict:
    """Project per-video counts at given video duration."""
    chunks = duration_sec  # 1s/chunk
    pT2 = t2_emit_prob(duration_sec)
    pT3 = t3_emit_prob(duration_sec)

    # Step 1: build a flat list of placements, each tagged (family, tier).
    # Aggregations derive from this list.
    placements = []  # list of (family, tier, count)

    pn1_n = pn1_actual(duration_sec)
    placements.append(("PN1", "pn1", pn1_n))

    for fam, n_cards in FAMILY_TARGETS.items():
        if fam == "PN1":
            continue
        verified = n_cards * VERIFY_YIELD

        if fam in MULTI_PROBE_PROBES:
            placements.append((fam, "multi_response", verified * MULTI_PROBE_PROBES[fam]))
        elif fam in PERSISTENT:
            placements.append((fam, "persistent_spread", verified * 3))
        elif fam in {"CR2", "CR4"}:
            placements.append((fam, "medium_in_compressed", verified * pT2))
            placements.append((fam, "hard_history_only", verified * pT3))
        else:
            placements.append((fam, "easy_in_visual", verified * 1))
            placements.append((fam, "medium_in_compressed", verified * pT2))
            placements.append((fam, "hard_history_only", verified * pT3))

        if fam == "E2":
            placements.append((fam, "event_watch", verified))
        if fam == "M1":
            placements.append((fam, "multi_response", verified))

    # Step 2: pass3b density cap on non-PN1 with TIER-WEIGHTED SURVIVAL.
    # Each placement's selection weight = base × tier_bonus. Survival probability
    # for placement i = NON_PN1_CAP × weight_i / sum(weights). Models the
    # +tier_bonus added to _score_placement for medium/hard tiers.
    tier_bonus = {
        "easy_in_visual": 0.0,
        "medium_in_compressed": TIER_BONUS_T2,
        "hard_history_only": TIER_BONUS_T3,
        "event_watch": 1.5,        # rare_seq bonus baked in
        "multi_response": 0.5,
        "persistent_spread": 0.0,
        "pn1": 0.0,
    }
    # Scoring approximation: relative probability ~ (1 + bonus); higher bonus
    # → higher selection rate. Normalized so total survivors = NON_PN1_CAP.
    non_pn1 = [(fam, tier, cnt) for fam, tier, cnt in placements if fam != "PN1"]
    weighted = [(fam, tier, cnt, cnt * (1.0 + tier_bonus.get(tier, 0.0)))
                for fam, tier, cnt in non_pn1]
    total_weight = sum(w for _, _, _, w in weighted)
    total_count = sum(c for _, _, c, _ in weighted)

    capped = []
    if total_count > NON_PN1_CAP and total_weight > 0:
        for fam, tier, cnt, w in weighted:
            survived = NON_PN1_CAP * (w / total_weight)
            capped.append((fam, tier, min(cnt, survived)))
    else:
        capped = [(fam, tier, cnt) for fam, tier, cnt, _ in weighted]
    # Re-add PN1 (bypasses cap)
    capped.append(("PN1", "pn1", pn1_n))

    # Step 3: MAX_SAMPLES_PER_VIDEO=15 cap on RENDERED.
    total_placed = sum(c for _, _, c in capped)
    scale_render = min(1.0, MAX_SAMPLES_PER_VIDEO / total_placed) if total_placed > 0 else 1.0
    rendered = [(fam, tier, c * scale_render) for fam, tier, c in capped]

    # Aggregate
    placements_by_tier = defaultdict(float)
    placements_by_family = defaultdict(float)
    placements_by_form = defaultdict(float)
    placements_by_ovo = defaultdict(float)
    for fam, tier, c in rendered:
        placements_by_tier[tier] += c
        placements_by_family[fam] += c
        placements_by_form[FAMILY_FORM[fam]] += c
        if FAMILY_OVO[fam]:
            placements_by_ovo[FAMILY_OVO[fam]] += c

    rendered_by_tier = dict(placements_by_tier)
    rendered_total = sum(rendered_by_tier.values())

    # "Needs recall" classification:
    # - hard_history_only: 100% needs recall
    # - medium_in_compressed: ~50% (LLM-checked, half-half answerable from
    #   compressed vs requires recall)
    # - event_watch: 0% (asked before evidence, not recall-style)
    # - everything else: 0%
    needs_recall = (
        rendered_by_tier.get("hard_history_only", 0) * 1.0 +
        rendered_by_tier.get("medium_in_compressed", 0) * 0.5
    )

    # Silent rate — TWO different metrics, both reported.
    #
    # A. RUNTIME silent rate = fraction of CHUNKS where the model emits a
    #    silent action at inference time. The MAX_SAMPLES_PER_VIDEO=15 cap
    #    only affects which samples land in train.jsonl, NOT the runtime
    #    trajectory. So runtime active chunks = pass3b-capped placements
    #    (non-PN1 ≤ 25, PN1 = pn1_n), NOT the 15 rendered samples.
    #
    # B. TRAINING-DATA silent rate = fraction of train.jsonl samples with
    #    sample_type="silent". Depends on round-robin in pipeline.py:872+
    #    over (family, action) buckets. Approximate as:
    #      silent_in_15 / 15 ≈ (n_silent_buckets) / (n_total_buckets) × 15
    #    Empirically ~30-40% post-bucket.
    runtime_active_chunks = sum(c for fam, tier, c in capped)
    silent_rate_runtime = max(0.0, (chunks - runtime_active_chunks) / chunks)
    # Crude approximation for training-data silent: round-robin keeps ~5/15
    # silent samples (warmup + patrol + question_window) regardless of cap.
    silent_rate_training = 5.0 / MAX_SAMPLES_PER_VIDEO  # ≈ 33%
    silent_rate = silent_rate_runtime  # legacy field

    # Single-trajectory question density (assuming 1 trajectory/video).
    # Number of QA placements (excluding PN1) = min(non-PN1 total, NON_PN1_CAP).
    # When MAX_TRAJ=1, all non-PN1 placements live in 1 traj → q_per_traj = capped.
    n_qa_in_traj = min(total_count, NON_PN1_CAP) / max(MAX_TRAJECTORIES_PER_VIDEO, 1)
    # Time interval between asks within a trajectory
    if n_qa_in_traj > 1:
        # Trajectory spans the full video; questions evenly distributed.
        q_interval_sec = duration_sec / n_qa_in_traj
    else:
        q_interval_sec = float('inf')

    # Response vs Silent ratio (training data view) — bucket-aware approximation
    # of pipeline.py:872 round-robin.
    #
    # Buckets: pass3c emits per-(family, action) and per-base-role samples.
    # Round-robin cycles each bucket; the per-bucket cursor controls which
    # samples survive when total candidates > cap.
    # Approximate bucket counts per video:
    #   response_buckets = #unique non-PN1 families with placements + 1 (PN1)
    #   silent_buckets   = 5 fixed base roles (warmup, patrol, question_window
    #                      silent, evidence_anchor, compress_boundary)
    #   compress_buckets = ~1 (compression events)
    placements_pre_cap = sum(c for fam, tier, c in capped)  # incl PN1
    families_with_placements = len({fam for fam, tier, c in capped
                                     if fam != "PN1" and c >= 0.3})
    response_buckets = max(1, families_with_placements) + (1 if pn1_n > 0 else 0)
    silent_buckets = 5  # warmup / patrol / question_window_silent / evidence_anchor / compress_boundary
    compress_buckets = 1
    total_buckets = response_buckets + silent_buckets + compress_buckets
    # Each bucket fills proportionally up to cap; respect supply per bucket.
    # Approximate supply: response = placements_pre_cap (split across buckets),
    # silent = 5 buckets × ~5 samples each = 25 candidates,
    # compress = 1 bucket × ~3 candidates.
    response_supply = placements_pre_cap
    silent_supply_per_bucket = 5
    silent_supply = silent_buckets * silent_supply_per_bucket
    compress_supply = 3
    total_supply = response_supply + silent_supply + compress_supply
    if total_supply > MAX_SAMPLES_PER_VIDEO:
        # Round-robin allocates picks per bucket cycle. Each bucket gets
        # ~ MAX_SAMPLES_PER_VIDEO / total_buckets picks; cap by supply.
        picks_per_bucket = MAX_SAMPLES_PER_VIDEO / total_buckets
        response_in_train = min(response_supply, picks_per_bucket * response_buckets)
        silent_in_train = min(silent_supply, picks_per_bucket * silent_buckets)
        compress_in_train = min(compress_supply, picks_per_bucket * compress_buckets)
        # Re-normalize residual back to cap if any bucket underfilled
        actual = response_in_train + silent_in_train + compress_in_train
        if actual < MAX_SAMPLES_PER_VIDEO:
            residual = MAX_SAMPLES_PER_VIDEO - actual
            # distribute residual to buckets with remaining supply
            response_room = response_supply - response_in_train
            silent_room = silent_supply - silent_in_train
            if response_room + silent_room > 0:
                share_r = response_room / (response_room + silent_room) if (response_room + silent_room) else 0
                response_in_train += residual * share_r
                silent_in_train += residual * (1 - share_r)
    else:
        response_in_train = response_supply
        silent_in_train = silent_supply
        compress_in_train = compress_supply
    response_silent_ratio_train = (
        response_in_train / (response_in_train + silent_in_train)
        if (response_in_train + silent_in_train) else 0.0
    )

    return {
        "duration_sec": duration_sec,
        "chunks": chunks,
        "pn1_actual": pn1_n,
        "non_pn1_total_pre_cap": total_count,
        "non_pn1_capped": min(total_count, NON_PN1_CAP),
        "total_placed_pre_render_cap": total_placed,
        "rendered_total": rendered_total,
        "render_scale": scale_render,
        "by_tier": dict(rendered_by_tier),
        "by_form": dict(placements_by_form),
        "by_ovo": dict(placements_by_ovo),
        "by_family": dict(placements_by_family),
        "needs_recall": needs_recall,
        "needs_recall_rate": needs_recall / rendered_total if rendered_total else 0.0,
        "silent_rate": silent_rate,
        "silent_rate_runtime": silent_rate_runtime,
        "silent_rate_training": silent_rate_training,
        "runtime_active_chunks": runtime_active_chunks,
        # New v12.7 metrics
        "n_qa_in_traj": n_qa_in_traj,
        "q_interval_sec": q_interval_sec,
        "response_in_train": response_in_train,
        "silent_in_train": silent_in_train,
        "response_silent_ratio_train": response_silent_ratio_train,
    }


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def print_report():
    durations = [60, 90, 120, 150, 180, 240, 320]

    sims = {d: simulate(d) for d in durations}

    # Header
    print("=" * 90)
    print(" v12.5 Question / Placement Distribution Simulator")
    print("=" * 90)
    print()
    print(f"FAMILY_TARGETS sum (non-PN1) = {sum(v for k, v in FAMILY_TARGETS.items() if k != 'PN1')} cards/video")
    print(f"PN1 target                   = {FAMILY_TARGETS['PN1']} candidates/video")
    print(f"pass3b cap (non-PN1)         = {NON_PN1_CAP} placements/video")
    print(f"MAX_SAMPLES_PER_VIDEO        = {MAX_SAMPLES_PER_VIDEO}")
    print(f"VERIFY_YIELD                 = {VERIFY_YIELD:.0%}")
    print()

    # Section 1: per-tier breakdown
    print("─" * 90)
    print("1. RENDERED PLACEMENTS BY TIER (after pass3b + MAX_SAMPLES_PER_VIDEO)")
    print("─" * 90)
    tiers = ["easy_in_visual", "medium_in_compressed", "hard_history_only",
             "event_watch", "multi_response", "persistent_spread", "pn1"]
    print(f"{'duration':>10}  " + "  ".join(f"{t[:8]:>9}" for t in tiers) + "    total")
    for d, s in sims.items():
        row = f"{d:>5}s  ({s['chunks']:>3}c) "
        for t in tiers:
            row += f"  {s['by_tier'].get(t, 0):>9.1f}"
        row += f"   {s['rendered_total']:>6.1f}"
        print(row)

    # Section 2: needs-recall rate
    print()
    print("─" * 90)
    print("2. NEEDS-RECALL RATE  (T2 × 0.5 + T3 × 1.0; PN1 / T1 / persistent = 0)")
    print("─" * 90)
    print(f"{'duration':>10}  {'rendered':>10}  {'needs_recall':>15}  {'rate':>8}")
    for d, s in sims.items():
        print(f"{d:>5}s      {s['rendered_total']:>10.1f}  {s['needs_recall']:>15.2f}  {fmt_pct(s['needs_recall_rate']):>8}")

    # Section 3: by answer_form
    print()
    print("─" * 90)
    print("3. BY answer_form")
    print("─" * 90)
    forms = ["multiple_choice", "binary", "binary_or_mc", "number", "short_exact", "descriptive"]
    print(f"{'duration':>10}  " + "  ".join(f"{f[:11]:>11}" for f in forms))
    for d, s in sims.items():
        row = f"{d:>5}s   "
        for f in forms:
            row += f"  {s['by_form'].get(f, 0):>11.1f}"
        print(row)

    # Section 4: by OVO task
    print()
    print("─" * 90)
    print("4. BY OVO TASK (relative share — should mirror OVO_TASK_SHARE for non-PN1)")
    print("─" * 90)
    print(f"{'task':<12}  {'OVO target':>10}  " + "  ".join(f"{d}s" for d in durations))
    for task in OVO_TASK_SHARE:
        target = OVO_TASK_SHARE[task]
        row = f"{task:<12}  {target*100:>9.1f}%  "
        for d in durations:
            s = sims[d]
            non_pn1 = sum(v for k, v in s["by_ovo"].items() if k != "narration")
            share = (s["by_ovo"].get(task, 0) / non_pn1 * 100) if non_pn1 else 0
            row += f"  {share:>4.1f}%"
        print(row)

    # Section 5: PN1 / silent
    print()
    print("─" * 90)
    print("5. PN1 + SILENT RATE")
    print("─" * 90)
    print(f"{'duration':>10}  {'PN1':>5}  {'active_ck':>10}  {'silent_RT':>10}  {'silent_TRAIN':>12}")
    print(f"{'(target)':>10}  {'':>5}  {'~30%':>10}  {'~70%':>10}  {'~30-40%':>12}")
    for d, s in sims.items():
        print(f"{d:>5}s    "
              f"  {s['pn1_actual']:>5}  "
              f"  {s['runtime_active_chunks']:>8.1f}  "
              f"  {fmt_pct(s['silent_rate_runtime']):>10}  "
              f"  {fmt_pct(s['silent_rate_training']):>12}")
    print()
    print("  silent_RT     = runtime per-chunk silent rate (chunks - placements) / chunks")
    print("  silent_TRAIN  = training-data silent rate, train.jsonl sample_type=silent share")
    print(f"  pass3b cap    = ≤{NON_PN1_CAP} non-PN1 placements/video; PN1 capped {PN1_PER_SEC_CAP}/sec")

    # Section 7: single-trajectory ask cadence + response/silent
    print()
    print("─" * 90)
    print(f"7. PER-TRAJECTORY METRICS (MAX_TRAJ={MAX_TRAJ}, MAX_Q/TRAJ={MAX_Q_PER_TRAJ})")
    print("─" * 90)
    print(f"{'duration':>10}  {'qa/traj':>8}  {'q_interval':>11}  {'q_per_min':>10}  "
          f"{'resp':>5}  {'silent':>7}  {'resp:silent':>13}")
    for d, s in sims.items():
        qpm = (60.0 / s['q_interval_sec']) if s['q_interval_sec'] != float('inf') else 0.0
        ratio = (
            f"{s['response_in_train']:.1f}:{s['silent_in_train']:.1f}"
        )
        print(f"{d:>5}s     "
              f"  {s['n_qa_in_traj']:>8.1f}"
              f"  {s['q_interval_sec']:>9.1f}s"
              f"  {qpm:>9.2f}/m"
              f"  {s['response_in_train']:>5.1f}"
              f"  {s['silent_in_train']:>7.1f}"
              f"  {ratio:>13}")
    print()
    print("  qa/traj         = QA placements (non-PN1) per trajectory")
    print("  q_interval      = average seconds between asks within trajectory")
    print("  q_per_min       = same in q/min (compare OVO 0.6 / StreamingBench 1.0)")
    print("  resp:silent     = train.jsonl response (incl PN1) vs silent sample ratio")

    # Section 6: per-family table
    print()
    print("─" * 90)
    print("6. PER-FAMILY RENDERED COUNT (median 150s video as representative)")
    print("─" * 90)
    s150 = simulate(150)
    print(f"{'family':<8}  {'OVO':<14}  {'form':<16}  {'cards':>6}  {'rendered':>10}")
    for fam in sorted(FAMILY_TARGETS, key=lambda f: -FAMILY_TARGETS[f]):
        print(f"{fam:<8}  {str(FAMILY_OVO[fam] or '—'):<14}  {FAMILY_FORM[fam]:<16}  "
              f"{FAMILY_TARGETS[fam]:>6}  {s150['by_family'].get(fam, 0):>10.2f}")

    # Final verdict
    print()
    print("=" * 90)
    print(" KEY TAKEAWAYS")
    print("=" * 90)
    s_med = sims[150]
    print(f"  Median (150s) video rendered samples : {s_med['rendered_total']:.1f}")
    print(f"  Of which need recall (T2/T3 weighted): {s_med['needs_recall']:.1f} ({fmt_pct(s_med['needs_recall_rate'])})")
    print(f"  Silent rate (raw)                    : {fmt_pct(s_med['silent_rate'])}")
    print()
    print("  Recall-required questions are concentrated in T2/T3 placements.")
    print("  Short videos drop T3 entirely → recall rate near zero on <90s.")
    print("  PN1 dominates rendered samples on long videos → drowns recall signal.")


if __name__ == "__main__":
    print_report()
