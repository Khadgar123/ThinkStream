"""Simulator: load existing pass1-2 outputs, run v2 pass3 logic
deterministically (no LLM), report 4 distributions:

  1. Card distribution by family / answer_form / question_type
  2. MC option-balance (per-letter counts)
  3. Trajectory mechanism distribution: silent_then_response / direct /
     recall_demo / multi_emit + recall noise type breakdown
  4. Per-video silent rate (fraction of chunks where gold == silent)

Usage:
  python -m scripts.agent_data_v5.v2.simulate \
    --evidence-dir data/agent_v5/evidence_1b \
    --rollout-dir  data/agent_v5/rollout       # optional, for num_chunks
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..stable_hash import stable_seed
from .cards import generate_cards
from .design import (
    AGENT_CHUNK_SEC,
    MAX_QUESTIONS_PER_TRAJECTORY,
    MAX_TRAJECTORIES_PER_VIDEO,
    Card,
    Placement,
    Sample,
    adaptive_q_count,
    assign_recall_noise,
    is_response_kind,
    is_silent_kind,
    place_card,
    render_video_samples,
    select_trajectory,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_evidence(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def num_chunks_from(evidence: List[Dict], rollout_path: Path = None) -> int:
    if rollout_path and rollout_path.exists():
        try:
            data = json.loads(rollout_path.read_text())
            n = data.get("num_chunks")
            if n:
                return int(n)
        except Exception:
            pass
    if evidence:
        return max(c.get("chunk_idx", 0) for c in evidence) + 1
    return 0


def compression_event_chunks_from(rollout_path: Path) -> List[int]:
    """Extract trigger chunks from rollout's compression events (for compress_silent)."""
    if not rollout_path or not rollout_path.exists():
        return []
    try:
        data = json.loads(rollout_path.read_text())
        return [int(e.get("trigger_chunk", -1))
                for e in data.get("compression_events", [])
                if e.get("trigger_chunk", -1) >= 0]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------


def simulate_one_video(
    video_id: str,
    evidence: List[Dict],
    num_chunks: int,
    seed: int = 42,
    compression_event_chunks: Optional[List[int]] = None,
) -> Dict:
    """Run v2 pipeline for one video. Returns aggregated metrics + raw samples."""
    rng = random.Random(stable_seed(seed, video_id, modulo=1_000_000))

    cards = generate_cards(evidence, video_id, seed=seed)

    # Generate ALL candidate placements (multiple tiers per card)
    placements_by_card: Dict[str, List[Placement]] = {}
    all_placements: List[Placement] = []
    for card in cards:
        plcs = place_card(card, num_chunks, rng)
        placements_by_card[card.card_id] = plcs
        all_placements.extend(plcs)

    # ADAPTIVE: q-count scales with video length (6-14 instead of fixed 10)
    target_q = adaptive_q_count(num_chunks)
    trajectory_placements = select_trajectory(
        cards, placements_by_card, num_chunks, rng, max_q=target_q,
    )

    # Render samples using ONLY the selected trajectory placements
    selected_pbc: Dict[str, List[Placement]] = {}
    for p in trajectory_placements:
        selected_pbc.setdefault(p.card_id, []).append(p)

    assign_recall_noise(trajectory_placements, rng)
    # NEW: pass evidence (for patrol stratification) + compression events
    samples = render_video_samples(
        cards, selected_pbc, num_chunks,
        evidence=evidence, rng=rng,
        compression_event_chunks=compression_event_chunks,
    )

    return {
        "video_id": video_id,
        "num_chunks": num_chunks,
        "target_q": target_q,
        "cards": cards,
        "all_candidate_placements": all_placements,    # before selection
        "placements": trajectory_placements,           # after selection
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict]) -> Dict:
    out: Dict = {}

    # ── 1. Card distribution ─────────────────────────────────────────────
    family_count: Counter = Counter()
    answer_form_count: Counter = Counter()
    qtype_count: Counter = Counter()
    cards_per_video: List[int] = []
    for r in results:
        cards = r["cards"]
        cards_per_video.append(len(cards))
        for c in cards:
            family_count[c.family] += 1
            answer_form_count[c.answer_form] += 1
            qtype_count[c.question_type] += 1
    out["cards"] = {
        "total_cards": sum(family_count.values()),
        "videos": len(results),
        "cards_per_video": {
            "mean": round(statistics.mean(cards_per_video), 1) if cards_per_video else 0,
            "median": int(statistics.median(cards_per_video)) if cards_per_video else 0,
            "min": min(cards_per_video) if cards_per_video else 0,
            "max": max(cards_per_video) if cards_per_video else 0,
        },
        "by_family": dict(family_count.most_common()),
        "by_answer_form": dict(answer_form_count.most_common()),
        "by_question_type": dict(qtype_count.most_common()),
    }

    # ── 2. MC option balance ─────────────────────────────────────────────
    mc_correct: Counter = Counter()
    for r in results:
        for c in r["cards"]:
            if c.answer_form == "multiple_choice" and c.correct_option:
                mc_correct[c.correct_option] += 1
    total_mc = sum(mc_correct.values())
    out["mc_balance"] = {
        "total_mc_cards": total_mc,
        "by_correct_option": {k: mc_correct.get(k, 0) for k in ["A", "B", "C", "D"]},
        "by_correct_option_pct": {
            k: round(mc_correct.get(k, 0) / max(total_mc, 1) * 100, 1)
            for k in ["A", "B", "C", "D"]
        },
    }

    # ── 3. Trajectory mechanism distribution ─────────────────────────────
    mech_count: Counter = Counter()
    placements_per_video: List[int] = []
    recall_noise: Counter = Counter()
    silent_then_response_lead: List[int] = []
    direct_gap: List[int] = []
    recall_demo_gap: List[int] = []
    # NEW: trajectory shape — questions per traj, q-interval, traj count
    questions_per_traj: List[int] = []
    q_intervals_chunks: List[float] = []
    q_intervals_seconds: List[float] = []
    for r in results:
        plcs = r["placements"]
        placements_per_video.append(len(plcs))
        # trajectory shape (1 traj per video by design)
        n_q = len(plcs)
        questions_per_traj.append(n_q)
        if n_q >= 2:
            ask_chunks = sorted(p.ask_chunk for p in plcs)
            diffs = [ask_chunks[i + 1] - ask_chunks[i] for i in range(len(ask_chunks) - 1)]
            mean_diff_c = sum(diffs) / len(diffs)
            q_intervals_chunks.append(mean_diff_c)
            q_intervals_seconds.append(mean_diff_c * AGENT_CHUNK_SEC)
        for p in plcs:
            mech_count[p.mechanism] += 1
            for noise in p.recall_at.values():
                recall_noise[noise] += 1
            # gap stats — find this card's emit chunk
            card = next((c for c in r["cards"] if c.card_id == p.card_id), None)
            if not card or not card.gold_emits:
                continue
            if card.question_type == "single_emit":
                emit = card.gold_emits[0].chunk
                gap = p.ask_chunk - emit
                if p.mechanism == "silent_then_response":
                    silent_then_response_lead.append(-gap)  # positive lead
                elif p.mechanism == "direct":
                    direct_gap.append(gap)
                elif p.mechanism == "recall_demo":
                    recall_demo_gap.append(gap)
    out["trajectory_shape"] = {
        "trajectories_per_video": MAX_TRAJECTORIES_PER_VIDEO,
        "max_questions_per_trajectory": MAX_QUESTIONS_PER_TRAJECTORY,
        "questions_per_trajectory": _stats(questions_per_traj),
        "q_interval_chunks": _stats_float(q_intervals_chunks),
        "q_interval_seconds": _stats_float(q_intervals_seconds),
    }
    out["mechanism"] = {
        "total_placements": sum(mech_count.values()),
        "placements_per_video_mean": round(
            statistics.mean(placements_per_video), 1) if placements_per_video else 0,
        "by_mechanism": dict(mech_count.most_common()),
        "by_mechanism_pct": {
            k: round(v / max(sum(mech_count.values()), 1) * 100, 1)
            for k, v in mech_count.items()
        },
        "recall_noise_distribution": dict(recall_noise),
        "recall_noise_pct": {
            k: round(v / max(sum(recall_noise.values()), 1) * 100, 1)
            for k, v in recall_noise.items()
        } if recall_noise else {},
        "silent_then_response_lead_chunks": _stats(silent_then_response_lead),
        "direct_gap_chunks": _stats(direct_gap),
        "recall_demo_gap_chunks": _stats(recall_demo_gap),
    }

    # ── 4. Per-video AND per-trajectory silent rate ──────────────────────
    # Two metrics:
    #   (a) per-video: ALL cards merged into one trajectory (lower bound)
    #   (b) per-trajectory: 1 placement = 1 trajectory (production-realistic upper bound)
    per_video_silent_rate: List[float] = []
    per_traj_silent_rate: List[float] = []
    sample_kind_count: Counter = Counter()
    silent_rate_by_video: Dict[str, float] = {}
    per_traj_by_mech: Dict[str, List[float]] = defaultdict(list)
    cards_by_id_global: Dict[str, Card] = {}
    for r in results:
        for c in r["cards"]:
            cards_by_id_global[c.card_id] = c

    for r in results:
        n_chunks = r["num_chunks"]
        if n_chunks == 0:
            continue
        # (a) per-video: merge all cards
        n_silent = sum(1 for s in r["samples"] if is_silent_kind(s.sample_kind))
        n_response = sum(1 for s in r["samples"] if is_response_kind(s.sample_kind))
        for s in r["samples"]:
            sample_kind_count[s.sample_kind] += 1
        rate = n_silent / max(n_silent + n_response, 1)
        per_video_silent_rate.append(rate)
        silent_rate_by_video[r["video_id"]] = round(rate, 3)

        # (b) per-trajectory: 1 placement → trajectory of length num_chunks
        # Each chunk that's NOT in this placement's window = patrol silent.
        for p in r["placements"]:
            n_active_response = sum(
                1 for (k, _) in p.chunk_actions.values() if k == "response"
            )
            n_active_silent = sum(
                1 for (k, _) in p.chunk_actions.values() if k == "silent"
            )
            n_patrol = n_chunks - len(p.chunk_actions)
            traj_silent = n_active_silent + n_patrol
            traj_response = n_active_response
            tr = traj_silent / max(traj_silent + traj_response, 1)
            per_traj_silent_rate.append(tr)
            per_traj_by_mech[p.mechanism].append(tr)

    out["silent_rate"] = {
        "videos": len(per_video_silent_rate),
        "per_video_all_merged": {
            "mean": round(statistics.mean(per_video_silent_rate), 3) if per_video_silent_rate else 0,
            "median": round(statistics.median(per_video_silent_rate), 3) if per_video_silent_rate else 0,
            "p10": round(_percentile(per_video_silent_rate, 10), 3) if per_video_silent_rate else 0,
            "p90": round(_percentile(per_video_silent_rate, 90), 3) if per_video_silent_rate else 0,
            "min": round(min(per_video_silent_rate), 3) if per_video_silent_rate else 0,
            "max": round(max(per_video_silent_rate), 3) if per_video_silent_rate else 0,
        },
        "per_trajectory_one_placement": {
            "n_trajectories": len(per_traj_silent_rate),
            "mean": round(statistics.mean(per_traj_silent_rate), 3) if per_traj_silent_rate else 0,
            "median": round(statistics.median(per_traj_silent_rate), 3) if per_traj_silent_rate else 0,
            "p10": round(_percentile(per_traj_silent_rate, 10), 3) if per_traj_silent_rate else 0,
            "p90": round(_percentile(per_traj_silent_rate, 90), 3) if per_traj_silent_rate else 0,
            "by_mechanism_mean": {
                k: round(statistics.mean(v), 3) for k, v in per_traj_by_mech.items()
            },
        },
        "sample_kind_breakdown_video_level": dict(sample_kind_count.most_common()),
        "extremes_video_level": {
            "lowest_5": sorted(silent_rate_by_video.items(), key=lambda x: x[1])[:5],
            "highest_5": sorted(silent_rate_by_video.items(), key=lambda x: -x[1])[:5],
        },
    }

    return out


def _stats(xs: List[int]) -> Dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 1),
        "median": int(statistics.median(xs)),
        "min": min(xs),
        "max": max(xs),
        "p25": int(_percentile(xs, 25)),
        "p75": int(_percentile(xs, 75)),
    }


def _stats_float(xs: List[float]) -> Dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 2),
        "median": round(statistics.median(xs), 2),
        "min": round(min(xs), 2),
        "max": round(max(xs), 2),
        "p25": round(_percentile(xs, 25), 2),
        "p75": round(_percentile(xs, 75), 2),
    }


def _percentile(xs: List, p: float) -> float:
    s = sorted(xs)
    if not s:
        return 0
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def print_report(agg: Dict, n_videos: int) -> None:
    print()
    print("=" * 76)
    print(f"V2 PIPELINE SIMULATION REPORT  (videos analyzed: {n_videos})")
    print("=" * 76)

    # 1. Cards
    cd = agg["cards"]
    print()
    print("─" * 76)
    print(f"1. CARD DISTRIBUTION  ({cd['total_cards']} cards across {cd['videos']} videos)")
    print("─" * 76)
    print(f"  Cards per video: mean={cd['cards_per_video']['mean']} "
          f"median={cd['cards_per_video']['median']} "
          f"min={cd['cards_per_video']['min']} "
          f"max={cd['cards_per_video']['max']}")
    print()
    print("  By family:")
    total_cards = cd["total_cards"]
    for fam, n in cd["by_family"].items():
        pct = round(n / max(total_cards, 1) * 100, 1)
        bar = "█" * int(pct / 2)
        print(f"    {fam:5s} {n:5d}  {pct:5.1f}%  {bar}")
    print()
    print("  By answer_form:")
    for af, n in cd["by_answer_form"].items():
        pct = round(n / max(total_cards, 1) * 100, 1)
        bar = "█" * int(pct / 2)
        print(f"    {af:18s} {n:5d}  {pct:5.1f}%  {bar}")
    print()
    print("  By question_type:")
    for qt, n in cd["by_question_type"].items():
        pct = round(n / max(total_cards, 1) * 100, 1)
        bar = "█" * int(pct / 2)
        print(f"    {qt:18s} {n:5d}  {pct:5.1f}%  {bar}")

    # 2. MC balance
    mc = agg["mc_balance"]
    print()
    print("─" * 76)
    print(f"2. MULTIPLE-CHOICE BALANCE  ({mc['total_mc_cards']} MC cards)")
    print("─" * 76)
    if mc["total_mc_cards"]:
        for letter in ["A", "B", "C", "D"]:
            n = mc["by_correct_option"][letter]
            pct = mc["by_correct_option_pct"][letter]
            bar = "█" * int(pct / 2)
            print(f"  Correct = {letter}: {n:4d}  {pct:5.1f}%  {bar}")
        diff_pct = max(mc["by_correct_option_pct"].values()) - min(mc["by_correct_option_pct"].values())
        verdict = "BALANCED" if diff_pct <= 5.0 else "MILDLY SKEWED" if diff_pct <= 10 else "IMBALANCED"
        print(f"  → max-min spread: {diff_pct:.1f}pp ({verdict}; ideal: each ≈ 25%)")
    else:
        print("  (no MC cards generated)")

    # 3a. Trajectory shape
    ts = agg["trajectory_shape"]
    print()
    print("─" * 76)
    print(f"3a. TRAJECTORY SHAPE  (production caps: "
          f"{ts['trajectories_per_video']} traj/video, "
          f"{ts['max_questions_per_trajectory']} q/traj)")
    print("─" * 76)
    qpt = ts["questions_per_trajectory"]
    print(f"  Questions per trajectory: mean={qpt['mean']} median={qpt['median']} "
          f"min={qpt['min']} max={qpt['max']} (p25={qpt['p25']} p75={qpt['p75']})")
    qic = ts["q_interval_chunks"]
    qis = ts["q_interval_seconds"]
    print(f"  Q-interval (chunks):     mean={qic['mean']} median={qic['median']} "
          f"min={qic['min']} max={qic['max']} (p25={qic['p25']} p75={qic['p75']})")
    print(f"  Q-interval (seconds):    mean={qis['mean']}s median={qis['median']}s "
          f"min={qis['min']}s max={qis['max']}s (p25={qis['p25']}s p75={qis['p75']}s)")
    print(f"  Industry reference: LiveChat/MMDuet 7-15s; OVOBench ~100s/q (sparse)")

    # 3b. Mechanism
    mech = agg["mechanism"]
    print()
    print("─" * 76)
    print(f"3b. TRAJECTORY MECHANISM  ({mech['total_placements']} placements, "
          f"avg {mech['placements_per_video_mean']}/video)")
    print("─" * 76)
    print()
    print("  By mechanism:")
    for k, v in mech["by_mechanism"].items():
        pct = mech["by_mechanism_pct"][k]
        bar = "█" * int(pct / 2)
        print(f"    {k:25s} {v:5d}  {pct:5.1f}%  {bar}")
    print()
    print("  Recall scheduling:")
    if mech["recall_noise_distribution"]:
        for k, v in mech["recall_noise_distribution"].items():
            pct = mech["recall_noise_pct"].get(k, 0)
            hint = (
                "recall_demo hit" if k == "oracle" else
                "recall_demo noisy hit" if k == "noisy" else
                "recall+silent wait state"
            )
            print(f"    {k:10s} {v:5d}  {pct:5.1f}%  ({hint}; no terminal failure)")
    else:
        print("    (no recall placements → no scheduling samples)")
    print()
    print("  Gap distributions (chunks):")
    print(f"    silent_then_response lead time: {mech['silent_then_response_lead_chunks']}")
    print(f"    direct gap (ask − emit):        {mech['direct_gap_chunks']}")
    print(f"    recall_demo gap (ask − emit):   {mech['recall_demo_gap_chunks']}")

    # 4. Silent rate
    sr = agg["silent_rate"]
    print()
    print("─" * 76)
    print(f"4. SILENT RATE  ({sr['videos']} videos)")
    print("─" * 76)
    print()
    pv = sr["per_video_all_merged"]
    print(f"  (a) PER-VIDEO (all cards merged into one mega-trajectory):")
    print(f"      mean={pv['mean']:.3f}  median={pv['median']:.3f}  "
          f"p10={pv['p10']:.3f}  p90={pv['p90']:.3f}  "
          f"min={pv['min']:.3f}  max={pv['max']:.3f}")
    pt = sr["per_trajectory_one_placement"]
    print()
    print(f"  (b) PER-TRAJECTORY (1 placement = 1 trajectory; production-realistic):")
    print(f"      n={pt['n_trajectories']}  mean={pt['mean']:.3f}  "
          f"median={pt['median']:.3f}  p10={pt['p10']:.3f}  p90={pt['p90']:.3f}")
    print()
    print(f"      by mechanism (mean silent rate per traj):")
    for mech, m in pt["by_mechanism_mean"].items():
        print(f"        {mech:25s} {m:.3f}")
    print()
    print("  Sample-kind breakdown (video level, ALL cards merged):")
    total_samples = sum(sr["sample_kind_breakdown_video_level"].values())
    for k, v in sr["sample_kind_breakdown_video_level"].items():
        pct = round(v / max(total_samples, 1) * 100, 1)
        bar = "█" * int(pct / 2)
        print(f"    {k:18s} {v:6d}  {pct:5.1f}%  {bar}")
    print()
    print("  Lowest-silent videos (video level):")
    for vid, r in sr["extremes_video_level"]["lowest_5"]:
        print(f"    {vid:15s}  silent={r:.3f}")
    print("  Highest-silent videos (video level):")
    for vid, r in sr["extremes_video_level"]["highest_5"]:
        print(f"    {vid:15s}  silent={r:.3f}")
    print()
    print("=" * 76)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True, type=Path)
    ap.add_argument("--rollout-dir", type=Path, default=None,
                    help="Optional. Used only for num_chunks lookup.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit videos for quick test. 0 = all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    evidence_files = sorted(args.evidence_dir.glob("*.json"))
    if args.limit:
        evidence_files = evidence_files[: args.limit]
    if not evidence_files:
        print(f"No evidence files found in {args.evidence_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Simulating {len(evidence_files)} videos…", file=sys.stderr)

    results = []
    for ef in evidence_files:
        video_id = ef.stem
        try:
            evidence = load_evidence(ef)
        except Exception as e:
            print(f"  skip {video_id}: load error {e}", file=sys.stderr)
            continue
        rollout_path = (args.rollout_dir / f"{video_id}.json") if args.rollout_dir else None
        n_chunks = num_chunks_from(evidence, rollout_path)
        if n_chunks == 0:
            continue
        compress_chunks = compression_event_chunks_from(rollout_path) if rollout_path else []
        try:
            r = simulate_one_video(
                video_id, evidence, n_chunks,
                seed=args.seed,
                compression_event_chunks=compress_chunks,
            )
            results.append(r)
        except Exception as e:
            print(f"  error {video_id}: {e}", file=sys.stderr)

    print(f"Aggregating {len(results)} videos…", file=sys.stderr)
    agg = aggregate(results)
    print_report(agg, n_videos=len(results))

    if args.save_json:
        # Strip dataclasses for serialization
        def _serialize(o):
            if hasattr(o, "__dict__"):
                return {k: _serialize(v) for k, v in o.__dict__.items()}
            if isinstance(o, list):
                return [_serialize(x) for x in o]
            if isinstance(o, dict):
                return {k: _serialize(v) for k, v in o.items()}
            return o
        args.save_json.write_text(json.dumps(_serialize(agg), indent=2, default=str))
        print(f"\nSaved aggregate to {args.save_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
