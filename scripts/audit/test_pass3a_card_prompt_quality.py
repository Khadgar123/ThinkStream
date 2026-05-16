"""Audit Pass3A card-prompt quality across vLLM endpoints.

The audit sends the current slot-constrained card prompt for every family to
each endpoint, parses the JSON cards, validates slot/chunk consistency, runs
the normal pass3A verifier, and writes per-family examples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.agent_data.pass3_slot_planner import (
    apply_slot_metadata_to_card,
    build_pass3_slot_plan,
    card_matches_planned_slot,
    group_slots_by_family,
)
from scripts.agent_data.pass3a_cards import (
    PASS3A_TARGETS_BY_FAMILY,
    _normalize_card_in_place,
    _verify_card_layers,
)
from scripts.agent_data.placement.llm_prompts import (
    FAMILY_RULES,
    card_generation_prompt,
    family_taxonomy,
    parse_card_response,
)
from scripts.agent_data_pipeline.vllm_client import VLLMClient


DEFAULT_ENDPOINTS = (
    "4card=http://10.16.18.9:8000/v1",
    "8card_a=http://10.16.12.175:8000/v1",
    "8card_b=http://10.16.10.172:8000/v1",
)


def _load_json(path: Path):
    return json.loads(path.read_text())


async def _model_for_endpoint(api_base: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(api_base.rstrip("/") + "/models")
        resp.raise_for_status()
        data = resp.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models at {api_base}")
    return str(models[0]["id"])


def _parse_endpoint(raw: str) -> Tuple[str, str]:
    if "=" in raw:
        name, url = raw.split("=", 1)
        return name.strip(), url.strip().rstrip("/")
    url = raw.strip().rstrip("/")
    return url.replace("http://", "").replace(":", "_").replace("/", "_"), url


def _evidence_by_chunk(evidence: List[Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for cap in evidence:
        if isinstance(cap, dict) and isinstance(cap.get("chunk_idx"), int):
            out[int(cap["chunk_idx"])] = cap
    return out


def _card_quality_flags(card: Dict) -> List[str]:
    flags: List[str] = []
    question = str(card.get("question") or "")
    if len(question) < 8:
        flags.append("short_question")
    if any(marker in question for marker in ("A)", "B)", "Options:", "Answer with")):
        flags.append("question_contains_rendering")
    family = str(card.get("family") or "")
    emits = card.get("gold_emits") or []
    if family in {"F5", "F7"} and not (6 <= len(emits) <= 9):
        flags.append(f"{family.lower()}_emit_count_not_6_9")
    if family == "CRR1" and not (3 <= len(emits) <= 5):
        flags.append("crr_emit_count_not_3_5")
    if card.get("answer_form") == "multiple_choice":
        opts = card.get("options") or []
        if len(opts) != len(set(str(o).strip().lower() for o in opts)):
            flags.append("duplicate_options")
        co = str(card.get("correct_option") or "")
        if not co or co < "A" or co > "E":
            flags.append("bad_correct_option")
    return flags


async def _audit_one(
    *,
    endpoint_name: str,
    client: VLLMClient,
    video_id: str,
    family: str,
    evidence: List[Dict],
    evidence_index: Dict[int, Dict],
    slots: List[Dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> Dict:
    prompt = card_generation_prompt(
        family,
        evidence,
        target_n=len(slots) if slots else int(PASS3A_TARGETS_BY_FAMILY.get(family, 1)),
        planned_slots=slots,
    )
    result = {
        "endpoint": endpoint_name,
        "video_id": video_id,
        "family": family,
        "slots": len(slots),
        "ok": False,
        "parse_n": 0,
        "pass_n": 0,
        "slot_match_n": 0,
        "reject": {},
        "quality_flags": {},
        "cards": [],
        "error": "",
    }
    try:
        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=f"prompt_quality_{endpoint_name}_{video_id}_{family}",
            enable_thinking=enable_thinking,
        )
    except Exception as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
        return result

    cards = parse_card_response(raw or "", family)
    result["parse_n"] = len(cards)
    rejects = Counter()
    flags_by_card: Dict[str, List[str]] = {}
    for i, card in enumerate(cards):
        card["card_id"] = f"{video_id}_{family}_audit_{endpoint_name}_{i}"
        card.update(family_taxonomy(family))
        apply_slot_metadata_to_card(card, slots)
        card.setdefault("recall_query", None)
        _normalize_card_in_place(card)
        slot_match = card_matches_planned_slot(card, slots)
        if slot_match:
            result["slot_match_n"] += 1
        verdict = "slot_plan_mismatch" if not slot_match else _verify_card_layers(card, evidence_index)
        if verdict == "PASS":
            result["pass_n"] += 1
        else:
            rejects[verdict] += 1
        flags = _card_quality_flags(card)
        if flags:
            flags_by_card[str(card.get("slot_id") or card.get("card_id") or i)] = flags
        result["cards"].append({
            "slot_id": card.get("slot_id", ""),
            "question": card.get("question", ""),
            "answer_form": card.get("answer_form", ""),
            "canonical_answer": card.get("canonical_answer", ""),
            "options": card.get("options"),
            "correct_option": card.get("correct_option"),
            "gold_emits": card.get("gold_emits"),
            "grounding_frames": card.get("grounding_frames"),
            "question_way": card.get("question_way"),
            "evidence_type": card.get("evidence_type"),
            "support_policy": card.get("support_policy"),
            "legacy_family_id": card.get("legacy_family_id"),
            "task_family": card.get("task_family"),
            "task_subtype": card.get("task_subtype"),
            "timing_type": card.get("timing_type"),
            "readable_task_name": card.get("readable_task_name"),
            "slot_group": card.get("slot_group"),
            "slot_subtype": card.get("slot_subtype"),
            "temporal_bucket": card.get("temporal_bucket"),
            "benchmark_source": card.get("benchmark_source"),
            "answer_behavior": card.get("answer_behavior"),
            "verdict": verdict,
            "quality_flags": flags,
        })
    result["reject"] = dict(rejects)
    result["quality_flags"] = flags_by_card
    result["ok"] = bool(slots) and result["pass_n"] == len(slots)
    return result


def _summarize(results: List[Dict]) -> Dict:
    endpoint = defaultdict(Counter)
    family = defaultdict(Counter)
    rejects = Counter()
    flags = Counter()
    for row in results:
        ep = row["endpoint"]
        fam = row["family"]
        endpoint[ep]["families"] += 1
        endpoint[ep]["slots"] += int(row.get("slots", 0))
        endpoint[ep]["parse_n"] += int(row.get("parse_n", 0))
        endpoint[ep]["pass_n"] += int(row.get("pass_n", 0))
        endpoint[ep]["slot_match_n"] += int(row.get("slot_match_n", 0))
        endpoint[ep]["ok_families"] += int(bool(row.get("ok")))
        endpoint[ep]["errors"] += int(bool(row.get("error")))
        family[fam]["endpoints"] += 1
        family[fam]["slots"] += int(row.get("slots", 0))
        family[fam]["pass_n"] += int(row.get("pass_n", 0))
        family[fam]["slot_match_n"] += int(row.get("slot_match_n", 0))
        family[fam]["ok_endpoints"] += int(bool(row.get("ok")))
        for key, value in (row.get("reject") or {}).items():
            rejects[key] += int(value)
        for card_flags in (row.get("quality_flags") or {}).values():
            for flag in card_flags:
                flags[flag] += 1
    return {
        "endpoint": {k: dict(v) for k, v in endpoint.items()},
        "family": {k: dict(v) for k, v in sorted(family.items())},
        "rejects": dict(rejects.most_common()),
        "quality_flags": dict(flags.most_common()),
    }


async def _run(args: argparse.Namespace) -> None:
    evidence_path = Path(args.evidence)
    evidence = _load_json(evidence_path)
    if not isinstance(evidence, list) or not evidence:
        raise SystemExit(f"bad evidence file: {evidence_path}")
    video_id = args.video_id or evidence_path.stem
    evidence_index = _evidence_by_chunk(evidence)

    families = [
        f.strip() for f in args.families.split(",") if f.strip()
    ] if args.families else list(FAMILY_RULES.keys())
    family_targets = {
        family: int(PASS3A_TARGETS_BY_FAMILY.get(family, 1))
        for family in families
        if family in FAMILY_RULES
    }
    slots_by_family = group_slots_by_family(
        build_pass3_slot_plan(evidence, video_id, family_targets, seed=int(args.seed))
    )

    endpoint_specs = [_parse_endpoint(x) for x in (args.endpoint or DEFAULT_ENDPOINTS)]
    clients = {}
    for name, api_base in endpoint_specs:
        model = await _model_for_endpoint(api_base)
        clients[name] = VLLMClient(
            api_base=api_base,
            model=model,
            max_concurrent=max(1, int(args.per_endpoint_concurrent)),
            timeout=float(args.timeout),
        )

    sem = asyncio.Semaphore(max(1, int(args.total_concurrent)))

    async def wrapped(name: str, family: str) -> Dict:
        async with sem:
            return await _audit_one(
                endpoint_name=name,
                client=clients[name],
                video_id=video_id,
                family=family,
                evidence=evidence,
                evidence_index=evidence_index,
                slots=slots_by_family.get(family, []),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                enable_thinking=bool(args.enable_thinking),
            )

    results = await asyncio.gather(
        *(wrapped(name, family) for name in clients for family in families)
    )
    report = {
        "video_id": video_id,
        "evidence": str(evidence_path),
        "endpoints": {
            name: {"api_base": api_base, "model": clients[name].model}
            for name, api_base in endpoint_specs
        },
        "families": families,
        "summary": _summarize(results),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("video_id", "evidence", "endpoints", "summary")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="data/agent_v5/batch8/evidence_1b/KxljcLB1Kfs.json")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--endpoint", action="append", default=[], help="name=http://host:port/v1")
    parser.add_argument("--families", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--per-endpoint-concurrent", type=int, default=8)
    parser.add_argument("--total-concurrent", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", default="data/agent_v5/audits/pass3a_card_prompt_quality_3nodes.json")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
