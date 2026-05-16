"""Simulate pass3 placement selection ratios without rewriting data.

This is a fast control loop for tuning OVO-vs-ours, direct-vs-recall, and
state/probe proportions on existing task_cards. It intentionally avoids pass3c
LLM calls and does not write generated samples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from scripts.agent_data.pass3a_cards import dict_to_card
from scripts.agent_data.pass3b_placement import _refine_selected_recall_with_rollout
from scripts.agent_data.placement import design


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(frac: float) -> float:
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, max(0, round(frac * (len(ordered) - 1))))
        return round(float(ordered[idx]), 4)

    return {
        "min": round(float(ordered[0]), 4),
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(float(ordered[-1]), 4),
        "avg": round(sum(float(x) for x in ordered) / len(ordered), 4),
        "n": len(ordered),
    }


def _patch_thresholds(args: argparse.Namespace) -> None:
    mapping = {
        "recall_target": "RECALL_TARGET_FRACTION",
        "recall_max": "RECALL_MAX_FRACTION",
        "ours_target": "OURS_TARGET_FRACTION",
        "ours_max": "OURS_MAX_FRACTION",
        "state_probe_target": "STATE_PROBE_TARGET_FRACTION",
        "state_probe_max": "STATE_PROBE_MAX_FRACTION",
        "current_direct_target": "CURRENT_DIRECT_TARGET_FRACTION",
        "past_state_direct_target": "PAST_STATE_DIRECT_TARGET_FRACTION",
        "past_state_direct_max": "PAST_STATE_DIRECT_MAX_FRACTION",
    }
    for arg_name, attr in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(design, attr, float(value))
    if args.ours_min is not None:
        design.OURS_MIN_QUESTIONS = int(args.ours_min)
    if args.current_direct_min is not None:
        design.CURRENT_DIRECT_MIN_QUESTIONS = int(args.current_direct_min)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iter_video_files(root: Path, batches: Iterable[str]) -> Iterable[Tuple[str, Path, Path, Path]]:
    for batch in batches:
        batch_dir = root / batch
        task_dir = batch_dir / "task_cards"
        rollout_dir = batch_dir / "rollout"
        evidence_dir = batch_dir / "evidence_1b"
        if not task_dir.exists():
            continue
        for task_path in sorted(task_dir.glob("*.json")):
            video_id = task_path.stem
            yield batch, task_path, rollout_dir / f"{video_id}.json", evidence_dir / f"{video_id}.json"


def _cards_from_data(data) -> Tuple[List[design.Card], Dict[str, Dict]]:
    if isinstance(data, dict):
        raw_cards = data.get("cards") or data.get("task_cards") or []
    else:
        raw_cards = data or []
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
        for chunk in list(card.grounding_frames or []) + [e.chunk for e in card.gold_emits]:
            try:
                max_chunk = max(max_chunk, int(chunk))
            except (TypeError, ValueError):
                continue
    return max(60, max_chunk + 24)


def simulate(args: argparse.Namespace) -> Dict:
    _patch_thresholds(args)
    root = Path(args.root)
    batches = [x.strip() for x in args.batches.split(",") if x.strip()]
    rng_seed = int(args.seed)

    totals = Counter()
    families = Counter()
    mechanisms = Counter()
    policies = Counter()
    answer_forms = Counter()
    answer_spaces = Counter()
    response_answer_spaces = Counter()
    binary_response_labels = Counter()
    number_response_labels = Counter()
    question_types = Counter()
    by_batch = Counter()
    rejected = Counter()
    q_counts: List[int] = []
    ask_gaps: List[int] = []
    ask_positions: List[float] = []
    trajectory_spans: List[int] = []
    recall_support_gaps: List[int] = []
    direct_support_gaps: List[int] = []
    examples = []

    for batch, task_path, rollout_path, evidence_path in _iter_video_files(root, batches):
        if args.limit_videos and totals["videos_seen"] >= int(args.limit_videos):
            break
        card_data = _load_json(task_path)
        if card_data is None:
            continue
        rollout = _load_json(rollout_path) or {}
        evidence = (_load_json(evidence_path) or []) if args.use_evidence_refine else []
        cards, cards_map = _cards_from_data(card_data)
        if not cards:
            continue
        num_chunks = int(rollout.get("num_chunks") or _fallback_num_chunks(cards))
        rng = random.Random(rng_seed)

        placements_by_card: Dict[str, List[design.Placement]] = {}
        for card in cards:
            placements = design.place_card(card, num_chunks, rng)
            if evidence:
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

        selected = design.select_trajectory(
            cards,
            placements_by_card,
            num_chunks,
            rng,
            max_q=design.adaptive_q_count(num_chunks),
        )
        cards_by_id = {c.card_id: c for c in cards}
        design.assign_recall_noise(selected, rng, cards_by_id=cards_by_id)
        if args.with_rollout_refine and rollout:
            _refine_selected_recall_with_rollout(
                selected,
                cards_map,
                rollout,
                video_id=task_path.stem,
            )

        totals["videos_seen"] += 1
        totals["cards"] += len(cards)
        totals["questions"] += len(selected)
        q_counts.append(len(selected))
        ask_chunks = sorted(int(p.ask_chunk) for p in selected)
        if len(ask_chunks) >= 2:
            ask_gaps.extend(b - a for a, b in zip(ask_chunks, ask_chunks[1:]))
            trajectory_spans.append(max(ask_chunks) - min(ask_chunks))
        if num_chunks > 0:
            ask_positions.extend(round(c / max(1, num_chunks - 1), 4) for c in ask_chunks)
        by_batch[batch] += len(selected)
        for placement in selected:
            card = cards_by_id.get(placement.card_id)
            if not card:
                continue
            policy = placement.support_policy or design.infer_card_policy_fields(card)["support_policy"]
            totals["recall_q"] += int(placement.mechanism == "recall_demo")
            totals["direct_q"] += int(placement.mechanism == "direct")
            totals["current_direct_q"] += int(design._is_current_direct_placement(placement))
            totals["state_probe_q"] += int(design._is_state_probe_placement(placement, card))
            totals["ours_q"] += int(design._is_ours_card(card))
            totals["ovo_regular_q"] += int(not design._is_ours_card(card))
            totals["response_rows"] += sum(
                1
                for kind, _value in placement.chunk_actions.values()
                if kind == "response"
            )
            totals["recall_response_rows"] += sum(
                1
                for c, (kind, _value) in placement.chunk_actions.items()
                if kind == "response" and int(c) in placement.recall_at
            )
            families[card.family] += 1
            mechanisms[placement.mechanism] += 1
            policies[policy] += 1
            answer_forms[card.answer_form] += 1
            answer_bucket = design._answer_form_bucket(card)
            answer_spaces[answer_bucket] += 1
            response_count = 0
            for kind, value in placement.chunk_actions.values():
                if kind != "response":
                    continue
                response_count += 1
                if answer_bucket == "binary":
                    binary_response_labels[str(value).strip()] += 1
                elif answer_bucket == "number":
                    number_response_labels[str(value).strip()] += 1
            response_answer_spaces[answer_bucket] += response_count
            question_types[card.question_type] += 1
            support = []
            for raw in card.grounding_frames or []:
                try:
                    support.append(int(raw))
                except (TypeError, ValueError):
                    continue
            if support:
                gap = int(placement.ask_chunk) - max(support)
                if placement.mechanism == "recall_demo":
                    recall_support_gaps.append(gap)
                if placement.mechanism == "direct":
                    direct_support_gaps.append(gap)
            if len(examples) < int(args.examples):
                examples.append({
                    "batch": batch,
                    "video_id": task_path.stem,
                    "family": card.family,
                    "mechanism": placement.mechanism,
                    "support_policy": policy,
                    "ask_chunk": int(placement.ask_chunk),
                    "question": card.question[:160],
                })

    question_total = max(1, totals["questions"])
    ratios = {
        key: round(totals[key] / question_total, 4)
        for key in (
            "recall_q",
            "direct_q",
            "current_direct_q",
            "state_probe_q",
            "ours_q",
            "ovo_regular_q",
        )
    }
    return {
        "settings": {
            "recall_target": design.RECALL_TARGET_FRACTION,
            "recall_max": design.RECALL_MAX_FRACTION,
            "ours_target": design.OURS_TARGET_FRACTION,
            "ours_max": design.OURS_MAX_FRACTION,
            "state_probe_target": design.STATE_PROBE_TARGET_FRACTION,
            "state_probe_max": design.STATE_PROBE_MAX_FRACTION,
            "current_direct_target": design.CURRENT_DIRECT_TARGET_FRACTION,
            "past_state_direct_target": design.PAST_STATE_DIRECT_TARGET_FRACTION,
            "past_state_direct_max": design.PAST_STATE_DIRECT_MAX_FRACTION,
            "binary_target": design.BINARY_TARGET_FRACTION,
            "binary_max": design.BINARY_MAX_FRACTION,
            "number_target": design.NUMBER_TARGET_FRACTION,
            "number_max": design.NUMBER_MAX_FRACTION,
            "short_text_target": design.SHORT_TEXT_TARGET_FRACTION,
            "short_text_max": design.SHORT_TEXT_MAX_FRACTION,
        },
        "totals": dict(totals),
        "ratios": ratios,
        "families": families.most_common(),
        "mechanisms": mechanisms.most_common(),
        "support_policies": policies.most_common(),
        "answer_forms": answer_forms.most_common(),
        "answer_spaces_question": answer_spaces.most_common(),
        "answer_spaces_response_units": response_answer_spaces.most_common(),
        "binary_response_labels": binary_response_labels.most_common(),
        "number_response_labels": number_response_labels.most_common(),
        "question_types": question_types.most_common(),
        "per_video_questions": _quantiles(q_counts),
        "ask_gap_chunks": _quantiles(ask_gaps),
        "ask_position_ratio": _quantiles(ask_positions),
        "trajectory_ask_span_chunks": _quantiles(trajectory_spans),
        "recall_support_gap_chunks": _quantiles(recall_support_gaps),
        "direct_support_gap_chunks": _quantiles(direct_support_gaps),
        "by_batch": dict(by_batch),
        "rejected": rejected.most_common(),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/agent_v5")
    parser.add_argument("--batches", default="batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-rollout-refine", action="store_true")
    parser.add_argument("--use-evidence-refine", action="store_true")
    parser.add_argument("--recall-target", type=float)
    parser.add_argument("--recall-max", type=float)
    parser.add_argument("--ours-target", type=float)
    parser.add_argument("--ours-max", type=float)
    parser.add_argument("--ours-min", type=int)
    parser.add_argument("--state-probe-target", type=float)
    parser.add_argument("--state-probe-max", type=float)
    parser.add_argument("--current-direct-target", type=float)
    parser.add_argument("--current-direct-min", type=int)
    parser.add_argument("--past-state-direct-target", type=float)
    parser.add_argument("--past-state-direct-max", type=float)
    args = parser.parse_args()
    print(json.dumps(simulate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
