"""PASS 2B rescue rollout from existing pass1 evidence.

This is intentionally NOT wired into pipeline.py.

Use case:
  pass2 observation rollouts can become stale-repeat loops when the teacher
  sees full historical text memory plus the sliding visual window. Existing
  pass1a/pass1b evidence is per-chunk, current-only, and was generated
  independently. This rescue path converts that current-only evidence into
  pass2-compatible think notes, then sweeps the timeline once to build memory
  snapshots and compression events.

The output schema matches pass2_rollout.py:
  {video_id, num_chunks, thinks, compression_events, snapshots, final_memory}

Recommended usage for an old batch:
  THINKSTREAM_DATA_ROOT=data/agent_v5/batch1 \
    python -m scripts.agent_data_v5.pass2b_rescue_from_pass1 \
      --output-dir data/agent_v5/batch1/rollout_pass2b

To replace rollout cache for downstream pass3, pass --output-dir .../rollout
explicitly after inspecting the generated audit stats.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .cache_version import STAGE_VERSIONS
from .config import (
    AGENT_CHUNK_SEC,
    COMPRESS_HYSTERESIS_THRESHOLD,
    COMPRESS_RANGE_MIN,
    DATA_ROOT,
    PASS_CONFIG,
    VLLM_MODEL,
)
from .pass2_rollout import (
    MemoryState,
    _fallback_compress_text,
    build_compress_request,
    parse_compress_result,
)

logger = logging.getLogger(__name__)


PREFERRED_THINK_FIELDS = (
    "think",
    "think_note",
    "observation_think",
    "caption",
    "dense_caption",
)


def _clean_text(value: object, *, max_chars: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("fact") or value.get("text") or value.get("desc") or ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    # Frame tags and XML-ish protocol tags are routing metadata, never target text.
    text = text.replace("<frame", "frame").replace("/>", "")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def _join_limited(items: Iterable[str], *, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _entity_phrase(entity: object) -> str:
    if not isinstance(entity, dict):
        return _clean_text(entity)
    desc = _clean_text(entity.get("desc") or entity.get("id") or "", max_chars=220)
    action = _clean_text(entity.get("action") or "", max_chars=100)
    pos = _clean_text(entity.get("position") or "", max_chars=80)
    if not desc:
        return ""
    phrase = desc
    if action and action.lower() != "static":
        phrase = f"{phrase} {action}"
    if pos:
        phrase = f"{phrase} ({pos})"
    return phrase


def _fact_text(fact: object) -> str:
    if isinstance(fact, dict):
        return _clean_text(fact.get("fact") or fact.get("text") or fact)
    return _clean_text(fact)


def _cap_words(text: str, *, max_words: int = 95) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words]).rstrip(" ,;:")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


def _sentence_case(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] if text[0].islower() else text


def build_think_from_pass1_evidence(cap: Dict) -> str:
    """Convert one pass1 evidence chunk into a current-only think note.

    Prefer an explicit pass1 think/caption field if a future pass1 version
    provides it. Existing batch1/batch2 pass1 JSON lacks that field, so this
    function deterministically renders visible entities, facts, state changes,
    OCR, and spatial relations. No previous chunk is consulted.
    """
    for field in PREFERRED_THINK_FIELDS:
        text = _clean_text(cap.get(field), max_chars=900)
        if text:
            return _cap_words(text)

    entities = _join_limited(
        (_entity_phrase(e) for e in cap.get("visible_entities") or []),
        limit=4,
    )
    facts = _join_limited(
        (_fact_text(f) for f in cap.get("atomic_facts") or []),
        limit=3,
    )
    changes = _join_limited((cap.get("state_changes") or []), limit=2)
    ocr = _join_limited((cap.get("ocr") or []), limit=3)
    spatial = _clean_text(cap.get("spatial") or "", max_chars=260)

    sentences: List[str] = []
    if entities:
        sentences.append("The current frames show " + "; ".join(entities) + ".")
    if facts:
        for fact in facts:
            fact_sentence = _sentence_case(fact).rstrip(" .")
            if fact_sentence:
                sentences.append(fact_sentence + ".")
    if changes:
        sentences.append("A visible state change is " + "; ".join(changes) + ".")
    if ocr:
        sentences.append("Visible text reads " + "; ".join(ocr) + ".")
    if spatial:
        if spatial[-1] not in ".!?":
            spatial += "."
        sentences.append(spatial)

    if not sentences:
        return "The current chunk has little discernible visual content."
    return _cap_words(" ".join(sentences))


def _normalize_evidence(evidence: Sequence[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for i, cap in enumerate(evidence or []):
        if not isinstance(cap, dict):
            continue
        item = dict(cap)
        try:
            chunk_idx = int(item.get("chunk_idx", i))
        except (TypeError, ValueError):
            chunk_idx = i
        item["chunk_idx"] = chunk_idx
        if "time" not in item:
            item["time"] = [
                chunk_idx * AGENT_CHUNK_SEC,
                (chunk_idx + 1) * AGENT_CHUNK_SEC,
            ]
        facts = []
        for fact in item.get("atomic_facts") or []:
            if isinstance(fact, dict):
                f = dict(fact)
                f.setdefault("confidence", 1.0)
                facts.append(f)
            else:
                facts.append({"fact": str(fact), "confidence": 1.0})
        item["atomic_facts"] = facts
        normalized.append(item)
    normalized.sort(key=lambda x: int(x.get("chunk_idx", 0)))
    return normalized


def _deterministic_summary(meta: Dict) -> Dict:
    return {
        "time_range": meta["time_range"],
        "text": _fallback_compress_text(meta),
        "parse_success": True,
        "source": "pass2b_deterministic",
    }


async def _compress_summary(
    *,
    client,
    comp_request: Dict,
    mode: str,
) -> Dict:
    effective_mode = "vllm" if mode == "auto" and client is not None else mode
    if effective_mode == "auto":
        effective_mode = "deterministic"

    if effective_mode == "deterministic":
        return _deterministic_summary(comp_request["_meta"])

    if effective_mode != "vllm":
        raise ValueError(f"Unsupported compression mode: {mode}")

    try:
        raw = await client._call_one(
            messages=comp_request["messages"],
            max_tokens=comp_request["max_tokens"],
            temperature=comp_request["temperature"],
            request_id=comp_request["id"],
            enable_thinking=bool(PASS_CONFIG["pass2_rollout"].get("thinking", False)),
        )
    except Exception as exc:
        logger.warning(
            "pass2b compression call failed for %s: %s; using deterministic fallback",
            comp_request.get("id", "?"),
            exc,
        )
        return _deterministic_summary(comp_request["_meta"])

    summary = parse_compress_result(raw, comp_request["_meta"])
    if not summary.get("parse_success"):
        summary["source"] = "pass2b_vllm_fallback"
    return summary


async def run_pass2b_single_video(
    *,
    video_id: str,
    evidence: Sequence[Dict],
    client=None,
    compress_mode: str = "auto",
) -> Dict:
    """Build a pass2-compatible rollout from current-only pass1 evidence."""
    caps = _normalize_evidence(evidence)
    num_chunks = (max((int(c.get("chunk_idx", 0)) for c in caps), default=-1) + 1)

    memory = MemoryState()
    thinks: List[Dict] = []
    compression_events: List[Dict] = []
    snapshots: Dict[int, Dict] = {}

    evidence_by_chunk = {int(c.get("chunk_idx", i)): c for i, c in enumerate(caps)}

    for chunk_idx in range(num_chunks):
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)
        pre_action_timeline = snapshots[chunk_idx]["timeline"]
        pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]
        should_compress_now = (
            compress_mode != "none"
            and memory.should_compress()
            and len(pre_action_thinks) >= COMPRESS_RANGE_MIN
        )

        cap = evidence_by_chunk.get(chunk_idx, {"chunk_idx": chunk_idx})
        think_text = build_think_from_pass1_evidence(cap)
        thinks.append({
            "chunk_idx": chunk_idx,
            "time": [
                chunk_idx * AGENT_CHUNK_SEC,
                (chunk_idx + 1) * AGENT_CHUNK_SEC,
            ],
            "think": think_text,
            "source": "pass1_evidence",
        })

        if should_compress_now:
            comp_request = build_compress_request(
                pre_action_timeline,
                memory,
                video_id,
                chunk_idx,
                evidence=list(caps),
                frame_paths=None,
            )
            if comp_request is not None:
                summary = await _compress_summary(
                    client=client,
                    comp_request=comp_request,
                    mode=compress_mode,
                )
                selected_indices = comp_request["_meta"]["selected_indices"]
                memory.compress(summary, selected_indices=selected_indices)
                memory.add_think(chunk_idx, think_text)

                post_compress_tokens = memory.count_recent_tokens()
                compression_events.append({
                    "trigger_chunk": chunk_idx,
                    "summary": summary,
                    "selected_indices": selected_indices,
                    "compressed_thinks_chunks": comp_request["_meta"].get("chunks", []),
                    "teacher_policy": comp_request["_meta"].get("teacher_policy", {}),
                    "hysteresis_ok": post_compress_tokens <= COMPRESS_HYSTERESIS_THRESHOLD,
                    "post_compress_tokens": post_compress_tokens,
                    "source": "pass2b_rescue_from_pass1",
                })
                continue

        memory.add_think(chunk_idx, think_text)

    return {
        "video_id": video_id,
        "num_chunks": num_chunks,
        "thinks": thinks,
        "compression_events": compression_events,
        "snapshots": snapshots,
        "final_memory": memory.snapshot(num_chunks),
        "rollout_source": "pass2b_rescue_from_pass1",
        "compression_mode": compress_mode,
    }


def load_pass1_evidence(video_id: str, evidence_dir: Path) -> Optional[List[Dict]]:
    path = evidence_dir / f"{video_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def save_rollout_json(video_id: str, rollout: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    path.write_text(json.dumps(rollout, ensure_ascii=False, indent=2))


def _iter_video_ids(
    *,
    evidence_dir: Path,
    videos_jsonl: Optional[Path],
    explicit_video_ids: Sequence[str],
    limit: int,
) -> List[str]:
    if explicit_video_ids:
        ids = [str(v) for v in explicit_video_ids]
    elif videos_jsonl:
        ids = []
        with videos_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                vid = row.get("video_id")
                if vid:
                    ids.append(str(vid))
    else:
        ids = sorted(p.stem for p in evidence_dir.glob("*.json"))

    if limit > 0:
        ids = ids[:limit]
    return ids


async def run_many(args: argparse.Namespace) -> Dict:
    data_root = Path(args.data_root)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    if evidence_dir is None:
        preferred = data_root / "evidence_1b"
        evidence_dir = preferred if preferred.exists() else data_root / "evidence_1a"
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "rollout_pass2b"

    video_ids = _iter_video_ids(
        evidence_dir=evidence_dir,
        videos_jsonl=Path(args.videos_jsonl) if args.videos_jsonl else None,
        explicit_video_ids=args.video_id or [],
        limit=int(args.limit or 0),
    )
    if not video_ids:
        raise RuntimeError(f"No video ids found from {evidence_dir}")

    client = None
    if args.compress_mode in {"auto", "vllm"} and args.api_base:
        from scripts.agent_data_pipeline.vllm_client import VLLMClient

        client = VLLMClient(
            api_base=args.api_base,
            model=args.model,
            max_concurrent=max(1, int(args.max_concurrent)),
            timeout=5400.0,
        )
    elif args.compress_mode == "vllm":
        raise RuntimeError("--compress-mode vllm requires --api-base")

    semaphore = asyncio.Semaphore(max(1, int(args.max_concurrent)))
    stats = {"videos": 0, "thinks": 0, "compressions": 0, "skipped": 0}

    async def _one(video_id: str) -> None:
        out_path = output_dir / f"{video_id}.json"
        if out_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            return
        evidence = load_pass1_evidence(video_id, evidence_dir)
        if evidence is None:
            logger.warning("[%s] missing pass1 evidence in %s", video_id, evidence_dir)
            stats["skipped"] += 1
            return
        async with semaphore:
            rollout = await run_pass2b_single_video(
                video_id=video_id,
                evidence=evidence,
                client=client,
                compress_mode=args.compress_mode,
            )
        save_rollout_json(video_id, rollout, output_dir)
        stats["videos"] += 1
        stats["thinks"] += len(rollout.get("thinks") or [])
        stats["compressions"] += len(rollout.get("compression_events") or [])

    await asyncio.gather(*[_one(vid) for vid in video_ids])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_version").write_text(STAGE_VERSIONS.get("2", "pass2b"))
    (output_dir / "pass2b_stats.json").write_text(
        json.dumps({
            **stats,
            "evidence_dir": str(evidence_dir),
            "output_dir": str(output_dir),
            "compress_mode": args.compress_mode,
        }, indent=2, ensure_ascii=False)
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pass2-compatible rollout JSON from existing pass1 evidence."
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--videos-jsonl", default="")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compress-mode",
        choices=["auto", "deterministic", "vllm", "none"],
        default="auto",
        help="auto uses vLLM when --api-base is provided, otherwise deterministic.",
    )
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model", default=VLLM_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = asyncio.run(run_many(args))
    logger.info(
        "pass2b complete: videos=%d thinks=%d compressions=%d skipped=%d",
        stats["videos"],
        stats["thinks"],
        stats["compressions"],
        stats["skipped"],
    )


if __name__ == "__main__":
    main()
