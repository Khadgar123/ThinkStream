#!/usr/bin/env python3
"""Rerun the known nonblank-but-empty batch9 evidence chunks.

This is intentionally narrow: it re-annotates selected pass1a chunks, reruns
pass1b for the affected videos so entity ids/state changes stay consistent,
and optionally writes the updated evidence files back into the batch root.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.config import FRAMES_PER_CHUNK, HIRES_MM_PROCESSOR_KWARGS, PASS_CONFIG
from scripts.agent_data.pass1a_evidence import build_evidence_request, parse_evidence_result
from scripts.agent_data.pass1b_enrich import run_pass1b
from scripts.agent_data_pipeline.vllm_client import VLLMClient


DEFAULT_TARGETS = {
    "25864270@N05_6450836691_0465712195": [52, 64],
    "gog9Nyh9Gro": [39],
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _frame_paths(batch_root: Path, video_id: str) -> List[str]:
    frame_dir = batch_root / "frames" / video_id
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames under {frame_dir}")
    return [str(p) for p in paths]


def _by_chunk(items: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in items:
        try:
            out[int(item.get("chunk_idx"))] = item
        except (TypeError, ValueError):
            continue
    return out


def _summary(item: Dict[str, Any] | None) -> Dict[str, Any]:
    if not item:
        return {"missing": True}
    return {
        "parse_success": bool(item.get("parse_success")),
        "think_len": len(str(item.get("think") or "")),
        "entity_count": len(item.get("visible_entities") or []),
        "fact_count": len(item.get("atomic_facts") or []),
        "state_change_count": len(item.get("state_changes") or []),
        "think_preview": str(item.get("think") or "")[:160],
    }


async def _annotate_chunk(
    *,
    client: VLLMClient,
    video_id: str,
    all_frame_paths: List[str],
    chunk_idx: int,
) -> Dict[str, Any]:
    request = build_evidence_request(chunk_idx, all_frame_paths, video_id)
    enable_thinking = bool(PASS_CONFIG["pass1a"].get("thinking", True))

    async def call_once(temperature: float, suffix: str = "") -> Dict[str, Any]:
        raw = await client._call_one(
            messages=request["messages"],
            max_tokens=request["max_tokens"],
            temperature=temperature,
            request_id=request["id"] + suffix,
            enable_thinking=enable_thinking,
            mm_processor_kwargs=HIRES_MM_PROCESSOR_KWARGS,
        )
        parsed = parse_evidence_result(raw, request["_meta"])
        parsed["chunk_idx"] = chunk_idx
        parsed["video_id"] = video_id
        return parsed

    caption = await call_once(float(request["temperature"]))
    if caption.get("_silent_empty") or not caption.get("parse_success"):
        retry = await call_once(0.7, "_retry")
        if retry.get("parse_success"):
            caption = retry
        else:
            caption["_retry_failed"] = True
    return caption


def _parse_targets(values: List[str]) -> Dict[str, List[int]]:
    if not values:
        return dict(DEFAULT_TARGETS)
    targets: Dict[str, List[int]] = {}
    for value in values:
        video_id, raw_chunks = value.split(":", 1)
        chunks = [int(x) for x in raw_chunks.split(",") if x.strip()]
        targets.setdefault(video_id, []).extend(chunks)
    return {k: sorted(set(v)) for k, v in targets.items()}


async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    batch_root = args.batch_root
    targets = _parse_targets(args.target)
    client_1a = VLLMClient(
        args.api_base,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )
    client_1b = VLLMClient(
        args.api_base,
        model=args.model,
        max_concurrent=max(1, min(args.max_concurrent, 4)),
        timeout=args.timeout,
    )

    report: Dict[str, Any] = {
        "batch_root": str(batch_root),
        "api_base": args.api_base,
        "model": args.model,
        "write": bool(args.write),
        "videos": {},
    }

    for video_id, chunks in targets.items():
        path_1a = batch_root / "evidence_1a" / f"{video_id}.json"
        path_1b = batch_root / "evidence_1b" / f"{video_id}.json"
        if not path_1a.exists():
            raise FileNotFoundError(path_1a)

        evidence_1a = _load_json(path_1a)
        before_1a = _by_chunk(evidence_1a)
        frames = _frame_paths(batch_root, video_id)
        num_chunks = len(frames) // FRAMES_PER_CHUNK
        for chunk_idx in chunks:
            if chunk_idx < 0 or chunk_idx >= num_chunks:
                raise ValueError(f"{video_id} chunk {chunk_idx} outside 0..{num_chunks - 1}")

        rerun = await asyncio.gather(*[
            _annotate_chunk(
                client=client_1a,
                video_id=video_id,
                all_frame_paths=frames,
                chunk_idx=chunk_idx,
            )
            for chunk_idx in chunks
        ])
        replacement = {int(item["chunk_idx"]): item for item in rerun}
        failed_chunks = [
            c for c in chunks
            if not replacement.get(c, {}).get("parse_success")
        ]
        if failed_chunks and not args.allow_failed_write:
            report["videos"][video_id] = {
                "num_frames": len(frames),
                "num_chunks": num_chunks,
                "skipped_write_reason": (
                    "pass1a rerun failed parse_success for chunks "
                    + ",".join(str(c) for c in failed_chunks)
                ),
                "chunks": {
                    str(c): {
                        "before_1a": _summary(before_1a.get(c)),
                        "after_1a": _summary(replacement.get(c)),
                    }
                    for c in chunks
                },
            }
            continue
        updated_1a = [
            replacement.get(int(item.get("chunk_idx", -1)), item)
            for item in evidence_1a
        ]
        updated_1b = await run_pass1b(updated_1a, client_1b, video_id)
        after_1b = _by_chunk(updated_1b)

        if args.write:
            _write_json(path_1a, updated_1a)
            _write_json(path_1b, updated_1b)

        report["videos"][video_id] = {
            "num_frames": len(frames),
            "num_chunks": num_chunks,
            "chunks": {
                str(c): {
                    "before_1a": _summary(before_1a.get(c)),
                    "after_1a": _summary(replacement.get(c)),
                    "after_1b": _summary(after_1b.get(c)),
                }
                for c in chunks
            },
        }

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=Path("data/agent_v5/batch9"))
    parser.add_argument("--api-base", default="http://10.16.12.175:8000/v1")
    parser.add_argument(
        "--model",
        default="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
    )
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5400.0)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="video_id:chunk,chunk. Defaults to the three known bad chunks.",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--allow-failed-write",
        action="store_true",
        help="Write files even when a targeted pass1a rerun still fails parsing.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    report = asyncio.run(main_async(args))
    out = args.report
    if out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = args.batch_root / "audits" / f"rerun_bad_chunks_{stamp}.json"
    _write_json(out, report)
    print(json.dumps({"report": str(out), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
