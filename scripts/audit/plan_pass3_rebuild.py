"""Plan a pass3 refresh from existing batch task cards.

This is a read-only control-loop script. It re-runs the current pass3B
selection logic on already-generated pass3A cards, validates the resulting
question lifecycle, and writes a per-video plan:

- rerun_3bc_only: existing cards are usable; rebuild placements/samples/final.
- rerun_3a_then_3bc: cards are semantically stale or too sparse.
- inspect: pass3B would produce invalid lifecycle/timing under current code.

No teacher calls are made here.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.agent_data.pass3a_cards import dict_to_card, _verify_card_layers
from scripts.agent_data.pass3b_placement import (
    _drop_degraded_recall_placements,
    _placement_crosses_compress_boundary,
    _refine_selected_recall_with_rollout,
)
from scripts.agent_data.stable_hash import stable_seed
from scripts.agent_data.placement import design


F7_STALE_RE = re.compile(r"\b(has|have|happened|yet|by now|so far)\b", re.I)
F7_CURRENT_RE = re.compile(r"\b(currently|right now|at this moment|now)\b", re.I)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _iter_batches(root: Path, batches: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for batch in batches:
        batch_dir = root / batch
        if batch_dir.exists():
            yield batch, batch_dir


def _video_index(root: Path, batch: str, batch_dir: Path) -> Dict[str, Dict]:
    rows: List[Dict] = []
    suffix = batch.removeprefix("batch")
    for path in (
        root / f"batch{suffix}_videos.jsonl",
        batch_dir / "selected_videos.jsonl",
        batch_dir / "video_registry.jsonl",
    ):
        rows.extend(_load_jsonl(path))
    out: Dict[str, Dict] = {}
    for row in rows:
        vid = str(row.get("video_id") or "")
        if vid and vid not in out:
            out[vid] = row
    return out


def _cards_from_data(data) -> Tuple[List[design.Card], Dict[str, Dict]]:
    raw_cards = data.get("cards") or data.get("task_cards") or [] if isinstance(data, dict) else data or []
    cards: List[design.Card] = []
    cards_map: Dict[str, Dict] = {}
    for raw in raw_cards:
        if not isinstance(raw, dict):
            continue
        try:
            card = dict_to_card(raw)
        except Exception:
            continue
        cards.append(card)
        cards_map[card.card_id] = raw
    return cards, cards_map


def _fallback_num_chunks(cards: List[design.Card]) -> int:
    max_chunk = 0
    for card in cards:
        chunks = list(card.grounding_frames or []) + [e.chunk for e in card.gold_emits]
        for raw in chunks:
            try:
                max_chunk = max(max_chunk, int(raw))
            except (TypeError, ValueError):
                continue
    return max(60, max_chunk + 24)


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(x) for x in values)

    def q(frac: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(frac * (len(ordered) - 1))))
        return round(ordered[idx], 4)

    return {
        "min": round(ordered[0], 4),
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(ordered[-1], 4),
        "avg": round(sum(ordered) / len(ordered), 4),
        "n": len(ordered),
    }


def _response_chunks(p: design.Placement) -> List[int]:
    return sorted(
        int(c)
        for c, action in p.chunk_actions.items()
        if action and str(action[0]) == "response"
    )


def _placement_span(p: design.Placement) -> Optional[Tuple[int, int]]:
    chunks = [int(c) for c in p.chunk_actions.keys()]
    if not chunks:
        return None
    return min(chunks), max(chunks)


def _support_gap(card: design.Card, response_chunk: int) -> Optional[int]:
    support: List[int] = []
    for raw in card.grounding_frames or []:
        try:
            support.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not support:
        return None
    return int(response_chunk) - max(support)


def _is_stale_f7(card: design.Card) -> bool:
    if str(card.family or "") != "F7":
        return False
    question = str(card.question or "")
    return bool(F7_STALE_RE.search(question)) or not bool(F7_CURRENT_RE.search(question))


def _semantic_card_failures(
    *,
    cards_map: Dict[str, Dict],
    evidence: List[Dict],
) -> Counter:
    evidence_by_chunk = {
        int(cap.get("chunk_idx")): cap
        for cap in evidence
        if isinstance(cap, dict) and isinstance(cap.get("chunk_idx"), int)
    }
    failures: Counter = Counter()
    for raw in cards_map.values():
        if not isinstance(raw, dict):
            continue
        verdict = _verify_card_layers(raw, evidence_by_chunk)
        if verdict != "PASS":
            family = str(raw.get("family", ""))
            failures[f"{family}:{verdict}"] += 1
    return failures


def _build_selected(
    *,
    video_id: str,
    cards: List[design.Card],
    cards_map: Dict[str, Dict],
    rollout: Dict,
    evidence: List[Dict],
    pipeline_seed: int,
    with_rollout_refine: bool,
) -> Tuple[List[design.Placement], Counter]:
    num_chunks = int(rollout.get("num_chunks") or _fallback_num_chunks(cards))
    place_rng = random.Random(stable_seed(42, video_id, modulo=1_000_000))
    rejected: Counter = Counter()
    placements_by_card: Dict[str, List[design.Placement]] = {}

    for card in cards:
        placements = design.place_card(card, num_chunks, place_rng)
        placements = design.refine_placements_with_evidence(card, placements, evidence)
        good: List[design.Placement] = []
        for placement in placements:
            ok, reason = design.placement_timing_verdict(card, placement)
            if ok:
                good.append(placement)
            else:
                rejected[reason] += 1
        if good:
            placements_by_card[card.card_id] = good

    compression_boundaries = [
        int(e.get("trigger_chunk"))
        for e in (rollout or {}).get("compression_events", [])
        if e.get("trigger_chunk") is not None
    ]
    filtered_by_card: Dict[str, List[design.Placement]] = {}
    for cid, placements in placements_by_card.items():
        for placement in placements:
            if _placement_crosses_compress_boundary(placement, compression_boundaries):
                rejected["crosses_compress_boundary"] += 1
                continue
            filtered_by_card.setdefault(cid, []).append(placement)

    select_seed = stable_seed(pipeline_seed * 10_000, video_id, modulo=0x1000000)
    select_rng = random.Random(select_seed)
    selected = design.select_trajectory(
        cards,
        filtered_by_card,
        num_chunks,
        select_rng,
        max_q=design.adaptive_q_count(num_chunks),
    )
    cards_by_id = {c.card_id: c for c in cards}
    design.assign_recall_noise(selected, select_rng, cards_by_id=cards_by_id)
    if with_rollout_refine:
        _refine_selected_recall_with_rollout(
            selected,
            cards_map,
            rollout,
            video_id=video_id,
        )
        selected, dropped_degraded = _drop_degraded_recall_placements(selected)
        if dropped_degraded:
            rejected["dropped_degraded_recall_demo"] += dropped_degraded
    return selected, rejected


def _validate_selected(
    *,
    cards_by_id: Dict[str, design.Card],
    selected: List[design.Placement],
) -> Counter:
    issues: Counter = Counter()
    spans: List[Tuple[int, int, str]] = []
    for p in selected:
        card = cards_by_id.get(p.card_id)
        responses = _response_chunks(p)
        span = _placement_span(p)
        if span:
            spans.append((span[0], span[1], p.card_id))
        if not responses:
            issues["placement_without_response"] += 1
            continue
        if card and card.question_type == "single_emit" and len(responses) != 1:
            issues["single_emit_not_one_answer"] += 1
        if card and card.question_type == "multi_emit" and len(responses) < 2:
            issues["multi_emit_too_few_answers"] += 1
        recall_responses = [
            c for c in responses
            if int(c) in {int(k) for k in p.recall_at.keys()}
        ]
        policy = str(getattr(p, "support_policy", "") or "")
        first_response = responses[0]
        gap = _support_gap(card, first_response) if card else None
        if p.mechanism == "recall_demo":
            if not recall_responses:
                issues["recall_demo_without_recall_response"] += 1
            if policy != design.SUPPORT_HISTORICAL_VISUAL_RECALL:
                issues["recall_demo_wrong_policy"] += 1
            if gap is not None and gap <= design.VISUAL_WINDOW_CHUNKS:
                issues["recall_demo_support_too_recent"] += 1
        elif p.mechanism == "direct":
            if recall_responses:
                issues["direct_has_recall"] += 1
            if first_response != int(p.ask_chunk):
                issues["direct_not_answer_at_ask"] += 1
            if gap is not None and gap > design.VISUAL_WINDOW_CHUNKS:
                issues["direct_support_too_old"] += 1
        elif p.mechanism == "memory_direct":
            if recall_responses:
                issues["memory_direct_has_recall"] += 1
            if policy != design.SUPPORT_HISTORICAL_STATE_MEMORY:
                issues["memory_direct_wrong_policy"] += 1
        elif p.mechanism == "silent_then_response":
            if first_response <= int(p.ask_chunk):
                issues["future_wait_not_future"] += 1
            if policy != design.SUPPORT_FUTURE_CURRENT_CUE:
                issues["future_wait_wrong_policy"] += 1
        elif p.mechanism == "multi_emit":
            if card and card.question_type != "multi_emit":
                issues["multi_emit_wrong_card_type"] += 1
    spans.sort()
    for prev, cur in zip(spans, spans[1:]):
        if cur[0] <= prev[1]:
            issues["overlapping_question_spans"] += 1
    return issues


def analyze(args: argparse.Namespace) -> Dict:
    root = Path(args.root)
    batches = [x.strip() for x in args.batches.split(",") if x.strip()]
    totals = Counter()
    decisions = Counter()
    families = Counter()
    mechanisms = Counter()
    policies = Counter()
    answer_forms = Counter()
    question_types = Counter()
    issue_counts = Counter()
    rejected_counts = Counter()
    by_batch = Counter()
    q_counts: List[int] = []
    ask_gaps: List[int] = []
    ask_positions: List[float] = []
    recall_support_gaps: List[int] = []
    direct_support_gaps: List[int] = []
    future_wait_leads: List[int] = []
    multi_emit_spans: List[int] = []
    decision_rows: List[Dict] = []

    for batch, batch_dir in _iter_batches(root, batches):
        task_dir = batch_dir / "task_cards"
        rollout_dir = batch_dir / "rollout"
        evidence_dir = batch_dir / "evidence_1b"
        video_index = _video_index(root, batch, batch_dir)
        if not task_dir.exists():
            continue
        for task_path in sorted(task_dir.glob("*.json")):
            if args.limit_videos and totals["videos"] >= int(args.limit_videos):
                break
            video_id = task_path.stem
            card_data = _load_json(task_path)
            rollout = _load_json(rollout_dir / f"{video_id}.json") or {}
            evidence = _load_json(evidence_dir / f"{video_id}.json") or []
            cards, cards_map = _cards_from_data(card_data)
            if not cards:
                continue
            num_chunks = int(rollout.get("num_chunks") or _fallback_num_chunks(cards))
            selected, rejected = _build_selected(
                video_id=video_id,
                cards=cards,
                cards_map=cards_map,
                rollout=rollout,
                evidence=evidence,
                pipeline_seed=int(args.seed),
                with_rollout_refine=bool(args.with_rollout_refine),
            )
            cards_by_id = {c.card_id: c for c in cards}
            issues = _validate_selected(cards_by_id=cards_by_id, selected=selected)
            semantic_issues = _semantic_card_failures(
                cards_map=cards_map,
                evidence=evidence if isinstance(evidence, list) else [],
            )
            issue_counts.update(semantic_issues)
            issue_counts.update(issues)
            rejected_counts.update(rejected)

            target_q = design.adaptive_q_count(num_chunks)
            low_floor = max(2, min(4, target_q // 2))
            if issues:
                decision = "inspect"
            elif semantic_issues or len(selected) < low_floor:
                decision = "rerun_3a_then_3bc"
            else:
                decision = "rerun_3bc_only"
            decisions[decision] += 1
            by_batch[batch] += len(selected)
            totals["videos"] += 1
            totals["cards"] += len(cards)
            totals["questions"] += len(selected)
            q_counts.append(len(selected))

            ask_chunks = sorted(int(p.ask_chunk) for p in selected)
            if len(ask_chunks) >= 2:
                ask_gaps.extend(b - a for a, b in zip(ask_chunks, ask_chunks[1:]))
            if num_chunks > 0:
                ask_positions.extend(c / max(1, num_chunks - 1) for c in ask_chunks)

            for p in selected:
                card = cards_by_id.get(p.card_id)
                if not card:
                    continue
                policy = p.support_policy or design.infer_card_policy_fields(card)["support_policy"]
                families[card.family] += 1
                mechanisms[p.mechanism] += 1
                policies[policy] += 1
                answer_forms[card.answer_form] += 1
                question_types[card.question_type] += 1
                totals["recall_q"] += int(p.mechanism == "recall_demo")
                totals["direct_q"] += int(p.mechanism == "direct")
                totals["current_direct_q"] += int(design._is_current_direct_placement(p))
                totals["state_probe_q"] += int(design._is_state_probe_placement(p, card))
                totals["ours_q"] += int(design._is_ours_card(card))
                totals["ovo_regular_q"] += int(not design._is_ours_card(card))
                responses = _response_chunks(p)
                totals["response_rows"] += len(responses)
                totals["recall_response_rows"] += sum(
                    1 for c in responses if int(c) in {int(k) for k in p.recall_at.keys()}
                )
                if responses:
                    gap = _support_gap(card, responses[0])
                    if gap is not None:
                        if p.mechanism == "recall_demo":
                            recall_support_gaps.append(gap)
                        elif p.mechanism == "direct":
                            direct_support_gaps.append(gap)
                if p.mechanism == "silent_then_response" and responses:
                    future_wait_leads.append(int(responses[0]) - int(p.ask_chunk))
                if p.mechanism == "multi_emit" and responses:
                    multi_emit_spans.append(max(responses) - min(responses))

            row = {
                "batch": batch,
                "video_id": video_id,
                "video_path": str((video_index.get(video_id) or {}).get("video_path", "")),
                "num_chunks": num_chunks,
                "n_cards": len(cards),
                "target_q": target_q,
                "n_questions": len(selected),
                "decision": decision,
                "issues": dict(issues),
                "semantic_issues": dict(semantic_issues),
                "rejected": dict(rejected),
                "ask_chunks": ask_chunks,
                "families": Counter(
                    cards_by_id[p.card_id].family
                    for p in selected
                    if p.card_id in cards_by_id
                ),
                "mechanisms": Counter(p.mechanism for p in selected),
            }
            row["families"] = dict(row["families"])
            row["mechanisms"] = dict(row["mechanisms"])
            decision_rows.append(row)

    question_total = max(1, totals["questions"])
    summary = {
        "settings": {
            "root": str(root),
            "batches": batches,
            "seed": int(args.seed),
            "recall_target": design.RECALL_TARGET_FRACTION,
            "recall_max": design.RECALL_MAX_FRACTION,
            "ours_target": design.OURS_TARGET_FRACTION,
            "ours_max": design.OURS_MAX_FRACTION,
            "state_probe_target": design.STATE_PROBE_TARGET_FRACTION,
            "state_probe_max": design.STATE_PROBE_MAX_FRACTION,
            "current_direct_target": design.CURRENT_DIRECT_TARGET_FRACTION,
            "with_rollout_refine": bool(args.with_rollout_refine),
        },
        "totals": dict(totals),
        "ratios": {
            key: round(totals[key] / question_total, 4)
            for key in (
                "recall_q",
                "direct_q",
                "current_direct_q",
                "state_probe_q",
                "ours_q",
                "ovo_regular_q",
            )
        },
        "decisions": dict(decisions),
        "families": families.most_common(),
        "mechanisms": mechanisms.most_common(),
        "support_policies": policies.most_common(),
        "answer_forms": answer_forms.most_common(),
        "question_types": question_types.most_common(),
        "issues": issue_counts.most_common(),
        "rejected": rejected_counts.most_common(),
        "by_batch_questions": dict(by_batch),
        "per_video_questions": _quantiles(q_counts),
        "ask_gap_chunks": _quantiles(ask_gaps),
        "ask_position_ratio": _quantiles(ask_positions),
        "recall_support_gap_chunks": _quantiles(recall_support_gaps),
        "direct_support_gap_chunks": _quantiles(direct_support_gaps),
        "future_wait_lead_chunks": _quantiles(future_wait_leads),
        "multi_emit_span_chunks": _quantiles(multi_emit_spans),
    }
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        with (out_dir / "video_decisions.jsonl").open("w") as f:
            for row in decision_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for decision in ("rerun_3bc_only", "rerun_3a_then_3bc", "inspect"):
            with (out_dir / f"{decision}_videos.jsonl").open("w") as f:
                for row in decision_rows:
                    if row["decision"] != decision:
                        continue
                    if row.get("video_path"):
                        f.write(json.dumps({
                            "video_id": row["video_id"],
                            "video_path": row["video_path"],
                        }, ensure_ascii=False) + "\n")
        summary["out_dir"] = str(out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/agent_v5")
    parser.add_argument(
        "--batches",
        default="batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument(
        "--with-rollout-refine",
        action="store_true",
        help=(
            "Also run pass3B's rollout recall refinement. This matches "
            "production more closely but is much slower; use it for strict "
            "spot checks after the fast full-batch plan."
        ),
    )
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
