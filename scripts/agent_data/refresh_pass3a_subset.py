"""Refresh pass3A task cards for a selected subset of videos.

Input is normally produced by scripts/audit/plan_pass3_rebuild.py:

  python scripts/agent_data/refresh_pass3a_subset.py \
    --plan-jsonl data/agent_v5/audits/pass3_rebuild_plan_fast/video_decisions.jsonl \
    --decisions inspect,rerun_3a_then_3bc \
    --api-base http://...:8000/v1 \
    --model /path/to/model \
    --max-concurrent 1024

The script only rewrites task_cards/{video_id}.json for selected videos.
Downstream placements/samples/final should then be rebuilt with
--force_rerun_from 3b.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set

from scripts.agent_data.pass3a_cards import (
    PASS3A_TARGETS_BY_FAMILY,
    _card_to_dict,
    _generate_via_heuristic,
    _load_external_slot_plan,
    _normalize_card_in_place,
    _verify_card_for_mode,
    generate_cards,
    verify_cards,
)
from scripts.agent_data.config import PASS_CONFIG, VLLM_MODEL
from scripts.agent_data.placement.llm_prompts import (
    FAMILY_RULES,
    card_generation_prompt,
    family_taxonomy,
    parse_card_response,
)
from scripts.agent_data.pass3_slot_planner import (
    apply_slot_metadata_to_card,
    build_pass3_slot_plan,
    card_matches_planned_slot,
    filter_pass3_slot_plan,
    group_slots_by_family,
    pass3a_batch_source_row_targets,
)
from scripts.agent_data_pipeline.vllm_client import VLLMClient


logger = logging.getLogger(__name__)
ALLOW_HEURISTIC_FALLBACK = (
    os.environ.get("THINKSTREAM_PASS3A_ALLOW_HEURISTIC_FALLBACK", "")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
ENABLE_SLOT_PLANNING = (
    os.environ.get("THINKSTREAM_PASS3A_ENABLE_SLOT_PLANNING", "1")
    .strip()
    .lower()
    not in {"0", "false", "no", "off"}
)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _evidence_num_chunks(evidence: List[Dict]) -> int:
    chunks = [
        int(cap.get("chunk_idx", i))
        for i, cap in enumerate(evidence or [])
        if isinstance(cap, dict)
    ]
    return max(chunks, default=0) + 1


def _raw_cards_from_task_data(data) -> List[Dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        raw = data.get("cards") or data.get("task_cards") or []
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            return [x for x in raw.values() if isinstance(x, dict)]
    return []


def _iter_plan_rows(path: Path, decisions: Set[str], batches: Set[str]) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if decisions and str(row.get("decision", "")) not in decisions:
                continue
            if batches and str(row.get("batch", "")) not in batches:
                continue
            if not row.get("video_id") or not row.get("batch"):
                continue
            rows.append(row)
    return rows


def _iter_all_selected_rows(root: Path, batches: Set[str]) -> List[Dict]:
    rows: List[Dict] = []
    for batch in sorted(batches):
        batch_dir = root / batch
        list_path = batch_dir / "selected_videos.jsonl"
        if not list_path.exists():
            list_path = batch_dir / "video_registry.jsonl"
        if not list_path.exists():
            logger.warning("[%s] no selected_videos.jsonl/video_registry.jsonl", batch)
            continue
        with list_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                video_id = str(item.get("video_id") or item.get("id") or "").strip()
                if not video_id:
                    continue
                rows.append({
                    "batch": batch,
                    "video_id": video_id,
                    "decision": "rerun_3a_then_3bc",
                })
    return rows


async def _refresh_one(
    *,
    row: Dict,
    root: Path,
    client: VLLMClient,
    dry_run: bool,
    families: Set[str],
    seed: int,
    source_row_targets: Dict[str, int],
) -> Dict:
    batch = str(row["batch"])
    video_id = str(row["video_id"])
    batch_dir = root / batch
    evidence_path = batch_dir / "evidence_1b" / f"{video_id}.json"
    out_path = batch_dir / "task_cards" / f"{video_id}.json"
    evidence = _load_json(evidence_path)
    if not isinstance(evidence, list) or not evidence:
        return {
            "batch": batch,
            "video_id": video_id,
            "ok": False,
            "reason": "missing_evidence_1b",
        }
    if dry_run:
        return {
            "batch": batch,
            "video_id": video_id,
            "ok": True,
            "dry_run": True,
            "families": sorted(families),
            "evidence_chunks": len(evidence),
            "out_path": str(out_path),
        }
    if families:
        existing = _raw_cards_from_task_data(_load_json(out_path))
        replacement = await _generate_family_subset(
            video_id=video_id,
            evidence=evidence,
            client=client,
            families=families,
            seed=seed,
        )
        cards = [
            c for c in existing
            if str(c.get("family", "")) not in families
        ] + replacement
    else:
        cards = await generate_cards(
            video_id,
            evidence,
            client,
            seed=seed,
            source_row_targets=source_row_targets,
        )
        cards = await verify_cards(video_id, cards, evidence, client)
    if not cards:
        return {
            "batch": batch,
            "video_id": video_id,
            "ok": False,
            "reason": "no_verified_cards",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n")
    return {
        "batch": batch,
        "video_id": video_id,
        "ok": True,
        "n_cards": len(cards),
        "families": sorted(families),
        "out_path": str(out_path),
    }


async def _generate_family_subset(
    *,
    video_id: str,
    evidence: List[Dict],
    client: VLLMClient,
    families: Set[str],
    seed: int,
) -> List[Dict]:
    cfg = PASS_CONFIG.get("pass3a", {})
    max_tokens = int(cfg.get("max_tokens", 16384))
    temperature = float(cfg.get("temperature", 0.7))
    enable_thinking = cfg.get("thinking", False)
    evidence_by_chunk = {
        int(cap.get("chunk_idx")): cap
        for cap in evidence
        if isinstance(cap, dict) and isinstance(cap.get("chunk_idx"), int)
    }
    fallback_by_family: Dict[str, List[Dict]] = {}
    if ALLOW_HEURISTIC_FALLBACK:
        for card in _generate_via_heuristic(evidence, video_id, seed):
            raw = _card_to_dict(card)
            if _verify_card_for_mode(raw, evidence_by_chunk) == "PASS":
                fallback_by_family.setdefault(card.family, []).append(raw)
    slot_by_family: Dict[str, List[Dict]] = {}
    if ENABLE_SLOT_PLANNING:
        external_slots = [
            slot for slot in _load_external_slot_plan(video_id)
            if str(slot.get("family", "")) in families
        ]
        external_slots = filter_pass3_slot_plan(evidence, external_slots)
        if external_slots:
            slot_by_family = group_slots_by_family(external_slots)
        else:
            family_targets = {
                family: int(PASS3A_TARGETS_BY_FAMILY.get(family, 1))
                for family in families
                if family in FAMILY_RULES
            }
            slot_by_family = group_slots_by_family(
                build_pass3_slot_plan(evidence, video_id, family_targets, seed=seed)
            )

    async def one_family(family: str) -> List[Dict]:
        if family not in FAMILY_RULES:
            return []
        planned_slots = slot_by_family.get(family, [])
        target_n = len(planned_slots) if planned_slots else int(PASS3A_TARGETS_BY_FAMILY.get(family, 1))
        prompt = card_generation_prompt(
            family,
            evidence,
            target_n=target_n,
            planned_slots=planned_slots,
        )
        try:
            raw = await client._call_one(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=f"{video_id}_3a_refresh_{family}",
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning("[%s] refresh %s failed: %s", video_id, family, exc)
            raw = ""
        cards = parse_card_response(raw or "", family) if raw else []
        out: List[Dict] = []
        for i, card in enumerate(cards):
            card["card_id"] = f"{video_id}_{family}_{seed:04d}_{i}"
            card.update(family_taxonomy(family))
            apply_slot_metadata_to_card(card, planned_slots)
            card.setdefault("recall_query", None)
            _normalize_card_in_place(card)
            if _verify_card_for_mode(card, evidence_by_chunk) != "PASS":
                continue
            if planned_slots and not card_matches_planned_slot(card, planned_slots):
                continue
            else:
                out.append(card)
        if out:
            return out
        return list(fallback_by_family.get(family, []))

    generated: List[Dict] = []
    for family_cards in await asyncio.gather(*(one_family(f) for f in sorted(families))):
        generated.extend(family_cards)
    return generated


async def _run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    decisions = {x.strip() for x in args.decisions.split(",") if x.strip()}
    batches = {x.strip() for x in args.batches.split(",") if x.strip()}
    families = {x.strip() for x in args.families.split(",") if x.strip()}
    if args.all_selected:
        if not batches:
            raise SystemExit("--all-selected requires --batches")
        rows = _iter_all_selected_rows(root, batches)
    else:
        if not args.plan_jsonl:
            raise SystemExit("--plan-jsonl is required unless --all-selected is set")
        rows = _iter_plan_rows(Path(args.plan_jsonl), decisions, batches)
    if args.limit:
        rows = rows[: int(args.limit)]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "refresh_pass3a_subset: rows=%d decisions=%s batches=%s dry_run=%s",
        len(rows),
        sorted(decisions),
        sorted(batches) if batches else "all",
        args.dry_run,
    )
    if not rows:
        report_path.write_text("")
        return

    source_row_targets_by_key: Dict[str, Dict[str, int]] = {}
    if not families:
        video_num_chunks: Dict[str, int] = {}
        for row in rows:
            batch = str(row.get("batch", ""))
            video_id = str(row.get("video_id", ""))
            evidence = _load_json(root / batch / "evidence_1b" / f"{video_id}.json")
            if isinstance(evidence, list) and evidence:
                video_num_chunks[f"{batch}/{video_id}"] = _evidence_num_chunks(evidence)
        source_row_targets_by_key = pass3a_batch_source_row_targets(video_num_chunks)

    client = VLLMClient(
        api_base=args.api_base,
        model=args.model,
        max_concurrent=max(1, int(args.max_concurrent)),
        timeout=float(args.timeout),
    )
    video_sem = asyncio.Semaphore(max(1, int(args.video_concurrent)))

    async def wrapped(row: Dict) -> Dict:
        async with video_sem:
            try:
                return await _refresh_one(
                    row=row,
                    root=root,
                    client=client,
                    dry_run=bool(args.dry_run),
                    families=families,
                    seed=int(args.seed),
                    source_row_targets=source_row_targets_by_key.get(
                        f"{row.get('batch', '')}/{row.get('video_id', '')}",
                        {},
                    ),
                )
            except Exception as exc:
                return {
                    "batch": row.get("batch", ""),
                    "video_id": row.get("video_id", ""),
                    "ok": False,
                    "reason": exc.__class__.__name__,
                    "error": str(exc),
                }

    ok = 0
    ok_batches: Set[str] = set()
    with report_path.open("w") as report:
        for result in await asyncio.gather(*(wrapped(row) for row in rows)):
            ok += int(bool(result.get("ok")))
            if result.get("ok") and result.get("batch"):
                ok_batches.add(str(result["batch"]))
            report.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info("refresh_pass3a_subset complete: ok=%d/%d report=%s", ok, len(rows), report_path)
    if ok == len(rows) and not args.dry_run and not families and (
        args.all_selected or args.write_version
    ):
        from scripts.agent_data.cache_version import STAGE_DIRS, write_stage_version

        original_task_cards_dir = STAGE_DIRS["3a"]
        try:
            for batch in sorted(ok_batches):
                STAGE_DIRS["3a"] = root / batch / "task_cards"
                write_stage_version("3a")
                logger.info("[%s] wrote pass3a version marker", batch)
        finally:
            STAGE_DIRS["3a"] = original_task_cards_dir
    if ok < len(rows) and not args.allow_partial:
        raise SystemExit(
            f"pass3A refresh incomplete: ok={ok}/{len(rows)}; "
            f"inspect {report_path} or rerun with --allow-partial"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/agent_v5")
    parser.add_argument("--plan-jsonl", default="")
    parser.add_argument(
        "--all-selected",
        action="store_true",
        help=(
            "Refresh every video listed in each requested batch's "
            "selected_videos.jsonl/video_registry.jsonl instead of reading "
            "a rebuild plan."
        ),
    )
    parser.add_argument(
        "--write-version",
        action="store_true",
        help=(
            "Write the pass3a cache version marker after a successful subset "
            "refresh. Use only when the subset completes a batch whose other "
            "task_cards are already from the same pass3a version."
        ),
    )
    parser.add_argument("--decisions", default="inspect,rerun_3a_then_3bc")
    parser.add_argument("--batches", default="")
    parser.add_argument(
        "--families",
        default="",
        help=(
            "Optional comma-separated family subset to regenerate and merge "
            "into existing task_cards, e.g. F7. Empty means regenerate all "
            "families for each selected video."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default=VLLM_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=1024)
    parser.add_argument("--video-concurrent", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=5400.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail the process when some selected videos fail refresh.",
    )
    parser.add_argument(
        "--report",
        default="data/agent_v5/audits/refresh_pass3a_subset_report.jsonl",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
