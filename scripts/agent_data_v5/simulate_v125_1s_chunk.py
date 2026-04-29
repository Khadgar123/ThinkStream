"""End-to-end simulator for v12.5 1s/chunk + 16K context final config.

Reads existing batch1 evidence_1a (2s/chunk) and projects to the new 1s/chunk
semantics by treating each existing chunk as TWO consecutive 1s chunks (the
second copy inherits evidence — a NOVELTY UPPER BOUND on PN1 candidates;
real 1s frames have ~half this density).

Runs the full classify_chunks + plan_trajectories flow under v12.5 constants
and reports:
  - per-batch video length distribution
  - num_chunks/PN1/QA/trajectories per video
  - silent rate (combined and per-source)
  - tier-2/tier-3 ask placement seconds (verifying the bug fix)
  - per-family coverage and OVO task coverage
  - 16K context budget breakdown
  - compress event timing per video bucket
  - memory horizon comparison
  - recall sample yield projection

Run:
  python scripts/agent_data_v5/simulate_v125_1s_chunk.py 60
"""

import json
import sys
from pathlib import Path
from collections import Counter
from statistics import median, mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.agent_data_v5 import config
from scripts.agent_data_v5.pass3a_cards import classify_chunks, FAMILY_TARGETS
from scripts.agent_data_v5.pass3b_placement import plan_trajectories


# ── v12.5 constants snapshot ───────────────────────────────────────────────

V125 = {
    "AGENT_CHUNK_SEC": config.AGENT_CHUNK_SEC,
    "FPS": config.FPS,
    "FRAMES_PER_CHUNK": config.FRAMES_PER_CHUNK,
    "VISUAL_WINDOW_CHUNKS": config.VISUAL_WINDOW_CHUNKS,
    "RECENT_THINKS_TOKEN_BUDGET": config.RECENT_THINKS_TOKEN_BUDGET,
    "COMPRESS_TOKEN_THRESHOLD": config.COMPRESS_TOKEN_THRESHOLD,
    "COMPRESS_HYSTERESIS_THRESHOLD": config.COMPRESS_HYSTERESIS_THRESHOLD,
    "COMPRESS_RANGE_MIN": config.COMPRESS_RANGE_MIN,
    "COMPRESS_RANGE_MAX": config.COMPRESS_RANGE_MAX,
    "COMPRESS_REMOVE_TOKENS": config.COMPRESS_REMOVE_TOKENS,
    "OBSERVATION_AVG_TOKENS": config.OBSERVATION_AVG_TOKENS,
    "THINK_TOKENS": config.THINK_TOKENS,
    "MAX_COMPRESSED_SEGMENTS": config.MAX_COMPRESSED_SEGMENTS,
    "SUMMARY_TOKENS_MAX": config.SUMMARY_TOKENS_MAX,
    "MAX_TRAJECTORIES_PER_VIDEO": config.MAX_TRAJECTORIES_PER_VIDEO,
    "MAX_QUESTIONS_PER_TRAJECTORY": config.MAX_QUESTIONS_PER_TRAJECTORY,
    "MAX_SAMPLES_PER_VIDEO": config.MAX_SAMPLES_PER_VIDEO,
}


def double_chunks(evidence_2s):
    """Project a 2s/chunk evidence list to 1s/chunk by duplicating each chunk."""
    new_evidence = []
    for old in evidence_2s:
        old_idx = old.get("chunk_idx", 0)
        for sub in (0, 1):
            ch = dict(old)
            ch["chunk_idx"] = old_idx * 2 + sub
            t0, t1 = old.get("time", [old_idx * 2, old_idx * 2 + 2])
            ch["time"] = [t0 + sub, t0 + sub + 1]
            new_evidence.append(ch)
    return new_evidence


def fake_card_for_chunk(family, chunk_idx, card_id):
    return {
        "card_id": card_id,
        "family": family,
        "answer_form": "descriptive" if family in ("S1", "M1", "PN1", "CR5") else "binary",
        "support_chunks": [chunk_idx],
        "key_chunks": {"ask": chunk_idx},
        "canonical_answer": f"ans_{card_id}",
        "question": f"q_{card_id}",
        "sequence_type": "evidence_then_ask",
        "answer_type": "factoid",
    }


def fake_placements(family_chunks, cards_map):
    out = []
    for fam, chunks in family_chunks.items():
        for ck in chunks:
            cid = f"{fam}_c{ck}"
            cards_map[cid] = fake_card_for_chunk(fam, ck, cid)
            out.append({
                "card_id": cid,
                "family": fam,
                "ask_chunk": ck,
                "key_chunks": {"ask": ck},
                "sequence_type": cards_map[cid]["sequence_type"],
                "answer_form": cards_map[cid]["answer_form"],
                "answer_type": "factoid",
            })
    return out


def silent_rate(num_chunks, ask_chunks, radius):
    if num_chunks == 0:
        return 1.0
    active = [False] * num_chunks
    for ac in ask_chunks:
        for c in range(max(0, ac - radius), min(num_chunks, ac + radius + 1)):
            active[c] = True
    return sum(1 for a in active if not a) / num_chunks


def expected_compress_events(num_chunks):
    """How many compress events fire on a video of given chunk count."""
    trig_at = V125["COMPRESS_TOKEN_THRESHOLD"] // V125["OBSERVATION_AVG_TOKENS"]
    evict_per = V125["COMPRESS_REMOVE_TOKENS"] // V125["OBSERVATION_AVG_TOKENS"]
    if num_chunks < trig_at:
        return 0
    return 1 + max(0, (num_chunks - trig_at) // evict_per)


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_v125_constants():
    section("v12.5 FINAL CONSTANTS (1s/chunk + 16K context)")
    rows = [
        ("AGENT_CHUNK_SEC",      V125["AGENT_CHUNK_SEC"], "1s per chunk"),
        ("FPS",                  V125["FPS"], "2 fps (= FRAMES_PER_CHUNK / chunk_sec)"),
        ("FRAMES_PER_CHUNK",     V125["FRAMES_PER_CHUNK"], "2 frames per chunk"),
        ("VISUAL_WINDOW_CHUNKS", V125["VISUAL_WINDOW_CHUNKS"], "16s visual context (32 frames)"),
        ("OBSERVATION_AVG_TOKENS", V125["OBSERVATION_AVG_TOKENS"], "matches THINK prompt midpoint"),
        ("THINK_TOKENS",         V125["THINK_TOKENS"], "soft target range"),
        ("RECENT_THINKS_TOKEN_BUDGET", V125["RECENT_THINKS_TOKEN_BUDGET"],
            f"{V125['RECENT_THINKS_TOKEN_BUDGET'] // V125['OBSERVATION_AVG_TOKENS']}s text horizon"),
        ("COMPRESS_TOKEN_THRESHOLD", V125["COMPRESS_TOKEN_THRESHOLD"],
            f"first compress at chunk ~{V125['COMPRESS_TOKEN_THRESHOLD'] // V125['OBSERVATION_AVG_TOKENS']}"),
        ("COMPRESS_RANGE", f"[{V125['COMPRESS_RANGE_MIN']}, {V125['COMPRESS_RANGE_MAX']}]",
            "min/max thinks evicted per compress"),
        ("COMPRESS_REMOVE_TOKENS", V125["COMPRESS_REMOVE_TOKENS"],
            f"~{V125['COMPRESS_REMOVE_TOKENS'] // V125['OBSERVATION_AVG_TOKENS']} thinks per compress"),
        ("MAX_TRAJECTORIES_PER_VIDEO", V125["MAX_TRAJECTORIES_PER_VIDEO"], ""),
        ("MAX_QUESTIONS_PER_TRAJECTORY", V125["MAX_QUESTIONS_PER_TRAJECTORY"], ""),
        ("MAX_SAMPLES_PER_VIDEO", V125["MAX_SAMPLES_PER_VIDEO"], "per-video sample cap"),
    ]
    for name, val, note in rows:
        print(f"  {name:32s} = {str(val):14s}  {note}")


def print_tier_offsets():
    """Test the new chunk_when_compressed-based tier 2/3 placement."""
    from scripts.agent_data_v5.pass3b_placement import (
        chunk_when_compressed, FIRST_COMPRESS_CHUNK,
        EVICT_THINKS_PER_COMPRESS,
    )
    section("TIER 2/3 OFFSETS (compression-aware, post-bug-fix)")
    VW = V125["VISUAL_WINDOW_CHUNKS"]
    print(f"FIRST_COMPRESS_CHUNK = {FIRST_COMPRESS_CHUNK}, "
          f"EVICT_THINKS_PER_COMPRESS = {EVICT_THINKS_PER_COMPRESS}")
    print()
    print(f"{'support_end':>12} | {'evict_at':>10} | {'tier2_lo':>10} | "
          f"{'tier2_hi':>10} | {'tier3_lo':>10} | check")
    for support_end in [4, 8, 24, 30, 50, 80]:
        evict_at = chunk_when_compressed(support_end)
        comp_lo = max(support_end + VW + 1, evict_at + 5)
        comp_hi = comp_lo + EVICT_THINKS_PER_COMPRESS
        hist_lo = max(support_end + VW + 1,
                      evict_at + EVICT_THINKS_PER_COMPRESS + 5)
        # Verify tier 2 is past first compress AND past evict point
        ok2 = comp_lo >= evict_at + 1 and comp_lo > support_end + VW
        ok3 = hist_lo >= evict_at + EVICT_THINKS_PER_COMPRESS
        status = "✓" if (ok2 and ok3) else "✗"
        print(f"{support_end:>12} | {evict_at:>10} | {comp_lo:>10} | "
              f"{comp_hi:>10} | {hist_lo:>10} | {status}")
    print()
    print("Reading: 'support_end=8 → evicted at chunk 53 (first compress); ")
    print("          tier 2 ask at chunk 58 (past evict + 5); tier 3 at 83'")


def main(n_videos=60):
    print_v125_constants()
    print_tier_offsets()

    section(f"END-TO-END SIMULATION ({n_videos} batch1 videos, projected to 1s/chunk)")
    print(f"NOTE: simulator doubles 2s evidence chunks → over-counts PN1 ~2×.")
    print(f"      Real 1s frames will produce ~half the PN1 density shown below.")

    ev_dir = config.EVIDENCE_1A_DIR
    files = sorted([f for f in ev_dir.iterdir() if f.suffix == ".json"
                    and not f.name.startswith("_")])[:n_videos]

    per_video = []
    fam_count = Counter()
    pn1_only_silent_rates = []
    qa_only_silent_rates = []
    combined_silent_rates = []
    question_gap_seconds = []
    traj_counts = []
    chunk_counts_new = []
    pn1_counts = []
    qa_placement_counts = []
    compress_events_per_video = []

    for fp in files:
        try:
            old_evidence = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(old_evidence, list) or not old_evidence:
            continue
        new_evidence = double_chunks(old_evidence)
        num_chunks = len(new_evidence)
        chunk_counts_new.append(num_chunks)
        compress_events_per_video.append(expected_compress_events(num_chunks))

        family_chunks = classify_chunks(new_evidence)
        capped = {}
        for fam, chunks in family_chunks.items():
            tgt = FAMILY_TARGETS.get(fam, 0)
            if tgt and chunks:
                capped[fam] = list(chunks)[:tgt]
        fam_count.update({f: len(c) for f, c in capped.items()})
        pn1_n = len(capped.get("PN1", []))
        pn1_counts.append(pn1_n)

        cards_map = {}
        placements = fake_placements(capped, cards_map)
        try:
            trajectories = plan_trajectories(
                placements, cards_map=cards_map,
                num_chunks=num_chunks, seed=42,
            )
        except Exception as e:
            print(f"  WARN plan_trajectories failed for {fp.stem}: {e}")
            continue
        traj_counts.append(len(trajectories))

        all_asks, qa_asks, pn1_asks = [], [], []
        for t in trajectories:
            for p in t["placements"]:
                fam = p.get("family", "?")
                ck = p["ask_chunk"]
                all_asks.append(ck)
                (pn1_asks if fam == "PN1" else qa_asks).append(ck)
        qa_placement_counts.append(len(qa_asks))

        s_combined = silent_rate(num_chunks, all_asks, radius=6)
        s_qa = silent_rate(num_chunks, qa_asks, radius=6)
        s_pn1 = silent_rate(num_chunks, pn1_asks, radius=6)
        combined_silent_rates.append(s_combined)
        qa_only_silent_rates.append(s_qa)
        pn1_only_silent_rates.append(s_pn1)

        sorted_asks = sorted(set(all_asks))
        gaps = [sorted_asks[i + 1] - sorted_asks[i]
                for i in range(len(sorted_asks) - 1)]
        question_gap_seconds.extend(gaps)

        per_video.append({
            "video": fp.stem,
            "num_chunks": num_chunks,
            "pn1": pn1_n,
            "qa_placements": len(qa_asks),
            "total_placements": len(all_asks),
            "trajectories": len(trajectories),
            "silent_rate_combined": s_combined,
            "compress_events": expected_compress_events(num_chunks),
        })

    section("PER-VIDEO METRICS")
    rows = [
        ("num_chunks (1s = sec)", chunk_counts_new),
        ("PN1 candidates", pn1_counts),
        ("QA placements (post-greedy)", qa_placement_counts),
        ("trajectories/video", traj_counts),
        ("compress events expected", compress_events_per_video),
    ]
    for name, vals in rows:
        print(f"  {name:30s}  median={median(vals):.1f}  mean={mean(vals):.1f}  "
              f"min={min(vals)}  max={max(vals)}")

    section("SILENT RATE (active = ±6 chunks around any ask = ~13s active window)")
    print(f"  combined (PN1 + QA):  median={median(combined_silent_rates):.3f}  "
          f"mean={mean(combined_silent_rates):.3f}")
    print(f"  QA only:              median={median(qa_only_silent_rates):.3f}  "
          f"mean={mean(qa_only_silent_rates):.3f}")
    print(f"  PN1 only:             median={median(pn1_only_silent_rates):.3f}  "
          f"mean={mean(pn1_only_silent_rates):.3f}")
    print(f"  Target band: 0.65-0.70")
    print(f"  ⚠  Simulator over-counts PN1 ~2× → expect real silent rate ~+15-20pp higher")

    if question_gap_seconds:
        section("QUESTION GAP (seconds between consecutive asks; combined PN1+QA)")
        sgs = sorted(question_gap_seconds)
        print(f"  median={median(sgs):.1f}s  mean={mean(sgs):.1f}s  "
              f"p10={sgs[len(sgs)//10]:.0f}s  p90={sgs[9*len(sgs)//10]:.0f}s")

    section("SILENT RATE BY VIDEO LENGTH")
    buckets = [(60, 120), (120, 180), (180, 360), (360, 9999)]
    for lo, hi in buckets:
        bucket = [(v, s) for v, s in zip(per_video, combined_silent_rates)
                  if lo <= v["num_chunks"] < hi]
        if not bucket:
            continue
        rates = [s for _, s in bucket]
        plac = [v["total_placements"] for v, _ in bucket]
        ces = [v["compress_events"] for v, _ in bucket]
        print(f"  [{lo:3d}-{hi:4d}s]  n={len(bucket):3d}  "
              f"silent_med={median(rates):.3f}  "
              f"placements_med={median(plac):.1f}  "
              f"compress_events_med={median(ces):.1f}")

    section("FAMILY COVERAGE (per-video target → realized)")
    sorted_fam = sorted(fam_count.items(), key=lambda x: -x[1])
    n = len(per_video)
    for fam, total in sorted_fam:
        target = FAMILY_TARGETS.get(fam, 0)
        avg = total / n if n else 0
        rate = avg / target if target else 0.0
        bar = "█" * int(rate * 20)
        print(f"  {fam:5s}  target={target:3d}  realized_avg={avg:5.2f}  "
              f"yield={rate:.2f} {bar}")

    section("OVOBench TASK COVERAGE (avg cards/video by task)")
    OVO_MAP = {
        "OCR": ["F1"], "ATR": ["F2", "S1"], "OJR": ["F3", "R1", "CR4"],
        "STU": ["F4"], "ACR": ["E1", "M1"], "EPM": ["E2", "C1", "CR1"],
        "CRR": ["CR1", "CR5"], "ASI": ["P1", "CR2"], "SSR": ["F7"],
        "REC": ["F5"], "FPD": ["F6"], "HLD": ["N1"],
    }
    ovo_targets = {  # OVO ratio × ~50 cards/video
        "OCR": 4.5, "ATR": 3.5, "OJR": 5.6, "STU": 5.5, "ACR": 3.3,
        "EPM": 9.0, "CRR": 1.5, "ASI": 4.5, "SSR": 1.3, "REC": 2.5,
        "FPD": 3.1, "HLD": 5.6,
    }
    for ovo, fams in OVO_MAP.items():
        avg = sum(fam_count[f] for f in fams) / n if n else 0
        target = ovo_targets.get(ovo, 0)
        status = "✓" if avg >= target * 0.5 else "⚠ under-covered"
        print(f"  {ovo:4s}  fams={','.join(fams):20s}  avg/video={avg:.2f}  "
              f"OVO-implied target≈{target:.1f}  {status}")

    section("16K CONTEXT BUDGET (worst-case at most-loaded chunk)")
    sys_t = config.SYSTEM_PROMPT_TOKENS
    visual_t = config.VISUAL_WINDOW_TOKENS
    recall_v = config.RECALL_VISION_TOKENS
    cseg_t = config.MAX_COMPRESSED_SEGMENTS * config.SUMMARY_TOKENS_MAX
    thinks_t = config.RECENT_THINKS_TOKEN_BUDGET
    queries_t = 8 * 50
    recall_text = 400
    output_t = 256
    total = sys_t + visual_t + recall_v + cseg_t + thinks_t + queries_t + recall_text + output_t
    headroom = 16384 - total
    rows = [
        ("system + tools", sys_t),
        ("visual_window (32 fr × 64 tok)", visual_t),
        ("recall_vision (4 fr)", recall_v),
        ("compressed_segments (5×280)", cseg_t),
        ("recent_thinks budget (4000)", thinks_t),
        ("past queries (8 × 50)", queries_t),
        ("recall_result text", recall_text),
        ("assistant output", output_t),
    ]
    for name, val in rows:
        print(f"  {name:35s}  {val:5d}")
    print(f"  {'─' * 35}  ─────")
    print(f"  {'subtotal':35s}  {total:5d}")
    print(f"  {'16K headroom':35s}  {headroom:5d}")
    if headroom >= 5000:
        print(f"  ✓ headroom comfortable")

    section("MEMORY HORIZON")
    visual_horizon_s = V125["VISUAL_WINDOW_CHUNKS"] * V125["AGENT_CHUNK_SEC"]
    text_thinks = V125["RECENT_THINKS_TOKEN_BUDGET"] // V125["OBSERVATION_AVG_TOKENS"]
    text_horizon_s = text_thinks * V125["AGENT_CHUNK_SEC"]
    print(f"  visual_window: {V125['VISUAL_WINDOW_CHUNKS']} chunks × "
          f"{V125['AGENT_CHUNK_SEC']}s = {visual_horizon_s}s "
          f"({V125['VISUAL_WINDOW_CHUNKS']*V125['FRAMES_PER_CHUNK']} frames)")
    print(f"  text_memory:   ~{text_thinks} thinks × {V125['AGENT_CHUNK_SEC']}s = "
          f"~{text_horizon_s}s observation history")
    ratio = text_horizon_s / max(visual_horizon_s, 1)
    print(f"  ratio:         text/visual = {ratio:.2f}× "
          f"({'✓ text > visual' if ratio > 1 else '❌ FAIL'})")

    section("RECALL SAMPLE YIELD (compress + outside-window)")
    total_compress = sum(compress_events_per_video)
    print(f"  videos simulated: {n}")
    print(f"  total compress events expected: {total_compress}")
    print(f"  avg compress events/video: {mean(compress_events_per_video):.2f}")
    too_short = sum(1 for v in per_video
                    if v["num_chunks"] < V125["COMPRESS_TOKEN_THRESHOLD"]
                    // V125["OBSERVATION_AVG_TOKENS"])
    print(f"  videos too-short for compress: {too_short}/{n}")
    print(f"  recall samples projected: ~{int(total_compress * 0.5)} compress-recall + "
          f"outside-window (videos ≥17s)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(n_videos=n)
