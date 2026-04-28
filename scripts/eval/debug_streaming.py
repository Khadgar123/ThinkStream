#!/usr/bin/env python
"""Streaming-eval debug mode — pinpoint context-overflow root cause.

Runs the streaming agent on ONE sample with debug=True and prints a
human-readable per-chunk zone breakdown. Use this when overflow is
suspected: the report shows exactly which zone (compressed_segments /
recent_thinks / queries / recall_result / visual / recalled_frames) is
inflating the prompt, and how memory state evolves chunk by chunk.

Usage:
    python scripts/eval/debug_streaming.py \\
        --ckpt output/agent-sft \\
        --test_jsonl data/agent_v5/final/test.jsonl \\
        --video_root /path/to/videos \\
        --frames_root data/agent_v5/frames \\
        --sample_idx 0 \\
        [--profile 16k|32k] [--retriever bm25|hybrid]

Output: per-chunk table of zone tokens + memory delta + flagged overflows,
plus a final summary listing the top-5 worst chunks by total estimated
prompt size.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor, AutoTokenizer

from thinkstream.model.agent_loop import (
    StreamingAgentLoop,
    make_generate_fn,
    AGENT_CHUNK_SEC,
)
from thinkstream.model.retrieval import make_retriever
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import (
    register_special_tokens,
    update_processor_pixels,
)
from scripts.eval.ovo.eval_full import (
    detect_model_class, reset_visual_index,
)
from scripts.eval.test_set_agent import (
    extract_gold, extract_question, gold_kind,
    resolve_video_path,
)
from scripts.eval.eval_profiles import apply_profile, describe_profile


# ─── Pretty printers ────────────────────────────────────────────────────────


def _zone_str(zones: Dict, name: str) -> str:
    z = zones.get(name) or {"text_tokens": 0, "n_frames": 0, "est_video": 0}
    parts = []
    if z.get("text_tokens"):
        parts.append(f"text={z['text_tokens']}")
    if z.get("n_frames"):
        parts.append(f"{z['n_frames']}fr×~200≈{z['est_video']}")
    return ", ".join(parts) if parts else "—"


def print_step_trace(trace: Dict, max_length: int):
    """Print one chunk's debug trace in a compact, human-readable way."""
    chunk = trace["chunk_idx"]
    zones = trace["zones"]
    total = zones.get("_total_estimated", 0)
    pct = 100 * total / max_length

    flag = ""
    if pct >= 90:
        flag = "  🚨 OVERFLOW IMMINENT"
    elif pct >= 75:
        flag = "  ⚠️  HIGH"

    print(f"\n┌─ chunk {chunk}  total={total}/{max_length} ({pct:.0f}%){flag}")

    # Question / trigger
    if trace.get("user_question"):
        print(f"│ user_question: {trace['user_question'][:80]!r}")
    if trace.get("compress_trigger_text"):
        print(f"│ compress_trigger: {trace['compress_trigger_text']}")

    # Memory before
    mb = trace["memory_before"]
    print(f"│ memory_BEFORE: comp_segs={mb['compressed_segments']} "
          f"({mb['compressed_text_tokens']} tok) | "
          f"recent={mb['recent_thinks']} ({mb['recent_thinks_tokens']} tok) | "
          f"queries={mb['queries_total']} (pending={mb['queries_pending']})")

    # Per-zone tokens
    print(f"│ zones:")
    for zname in ("system", "visual_window", "recalled_frames",
                  "memory", "queries", "recall_result", "user_input", "other"):
        s = _zone_str(zones, zname)
        if s == "—":
            continue
        print(f"│   {zname:<18s} {s}")
    txt = zones.get("_total_text_tokens", 0)
    vid = zones.get("_total_est_video", 0)
    print(f"│   {'TOTAL':<18s} text={txt} + video~={vid} = {total}")

    # Output / parse
    print(f"│ output: {trace['model_output_chars']} chars, "
          f"action={trace['parsed_action']!r}, "
          f"format_ok={trace['format_ok']}, "
          f"compress_ok={trace['compress_succeeded']}")

    # Memory after
    ma = trace["memory_after"]
    print(f"│ memory_AFTER:  comp_segs={ma['compressed_segments']} | "
          f"recent={ma['recent_thinks']} ({ma['recent_thinks_tokens']} tok) | "
          f"queries={ma['queries_total']} (pending={ma['queries_pending']})")

    # Flag specific anomalies
    if zones.get("memory", {}).get("text_tokens", 0) > 2000:
        print(f"│ ⚠️  memory text >2000 tok — segments may be uncapped or thinks bloated")
    if zones.get("queries", {}).get("text_tokens", 0) > 1500:
        print(f"│ ⚠️  queries text >1500 tok — cap not applied?")
    if zones.get("visual_window", {}).get("est_video", 0) > 8000:
        print(f"│ ⚠️  visual_window estimate >8000 — pixel budget too high?")
    if zones.get("recalled_frames", {}).get("n_frames", 0) > 12:
        print(f"│ ⚠️  recalled_frames >12 — recall returned an unusual amount")

    print("└─")


def print_summary(traces: List[Dict], max_length: int):
    """Top-N worst chunks + per-zone aggregates."""
    if not traces:
        return
    # Sort by total estimated tokens
    sorted_traces = sorted(traces, key=lambda t: t["zones"].get("_total_estimated", 0),
                           reverse=True)
    print()
    print("=" * 78)
    print(f"SUMMARY  ({len(traces)} chunks walked)")
    print("=" * 78)

    print("\nTop 5 chunks by total prompt estimate:")
    print(f"  {'chunk':>5}  {'total':>6}  {'%cap':>5}  "
          f"{'text':>5}  {'video':>6}  action")
    for t in sorted_traces[:5]:
        z = t["zones"]
        total = z.get("_total_estimated", 0)
        pct = 100 * total / max_length
        print(f"  {t['chunk_idx']:>5d}  {total:>6d}  {pct:>4.0f}%  "
              f"{z.get('_total_text_tokens', 0):>5d}  "
              f"{z.get('_total_est_video', 0):>6d}  {t['parsed_action']!r}")

    # Per-zone max across run
    print("\nMax per-zone text tokens across all chunks (audits cap effectiveness):")
    zone_max = {}
    for t in traces:
        for k, v in t["zones"].items():
            if not isinstance(v, dict):
                continue
            zone_max[k] = max(zone_max.get(k, 0), v.get("text_tokens", 0))
    for zname in ("system", "visual_window", "recalled_frames",
                  "memory", "queries", "recall_result", "user_input"):
        if zname not in zone_max or zone_max[zname] == 0:
            continue
        print(f"  {zname:<20s} {zone_max[zname]:>5d} tok")

    # Format violations
    n_fmt_err = sum(1 for t in traces if not t["format_ok"])
    n_compress_attempts = sum(1 for t in traces if t["compress_trigger_text"])
    n_compress_ok = sum(1 for t in traces
                        if t["compress_trigger_text"] and t["compress_succeeded"])
    print(f"\nFormat violations: {n_fmt_err}/{len(traces)} chunks")
    print(f"Compress attempts: {n_compress_attempts}, "
          f"succeeded: {n_compress_ok}/{n_compress_attempts}")

    # Overflow warning
    n_over = sum(1 for t in traces
                 if t["zones"].get("_total_estimated", 0) >= 0.9 * max_length)
    if n_over:
        print(f"\n🚨 {n_over} chunks above 90% of model_max_length ({max_length})")
    else:
        print(f"\n✅ All chunks below 90% of model_max_length ({max_length})")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default="data/agent_v5/frames")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index into test.jsonl scorable subset (sorted by file order)")
    p.add_argument("--profile", default="16k", choices=["16k", "32k"])
    p.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--siglip_path", default="google/siglip-base-patch16-224")
    p.add_argument("--compress_mode", default="system", choices=["system", "self"])
    p.add_argument("--max_results", type=int, default=4)
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Override walk length (default: ask_chunk + 60)")
    p.add_argument("--dump_json", default=None,
                   help="Write full trace dump to this JSON file for post-mortem")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    # Apply profile
    cfg = apply_profile(args.profile)
    print(describe_profile(args.profile))
    max_length = cfg["model_max_length"]

    # Load model + processor + tokenizer
    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt}")
    model = Cls.from_pretrained(
        args.ckpt, dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    register_special_tokens(processor, model_type)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and \
            hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    print(f"Building retriever: kind={args.retriever}")
    retriever = make_retriever(
        kind=args.retriever, siglip_path=args.siglip_path,
        alpha=args.alpha, max_results=args.max_results, device="cuda",
    )

    # Build loop with debug=True
    loop = StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer, processor=processor, model_type=model_type,
        min_pixels=100352, max_pixels=150528,
        max_new_tokens=cfg["max_new_tokens_default"],
        retriever=retriever, compress_mode=args.compress_mode,
        frames_root=args.frames_root, video_root=args.video_root,
    )
    loop.debug = True   # turn on per-step trace

    # Pick sample
    samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    scorable = []
    for s in samples:
        if s.get("sample_type") not in ("response", "recall_response"):
            continue
        gold = extract_gold(s)
        if gold_kind(gold) in ("yes_no", "int", "letter"):
            scorable.append((s, gold))
    if not scorable:
        print("No scorable samples in test.jsonl"); return
    if args.sample_idx >= len(scorable):
        print(f"--sample_idx {args.sample_idx} out of range "
              f"(only {len(scorable)} scorable)"); return
    sample, gold = scorable[args.sample_idx]
    question = extract_question(sample)
    if not question:
        print("Sample has no question"); return
    video_path = resolve_video_path(sample, args.video_root)
    if not video_path or not Path(video_path).exists():
        print(f"Video not found: {video_path}"); return
    ask_chunk = int(sample.get("chunk_idx", 0))

    print()
    print("=" * 78)
    print(f"SAMPLE")
    print("=" * 78)
    print(f"  sample_id   : {sample.get('sample_id') or sample.get('trajectory_id')}")
    print(f"  video_path  : {video_path}")
    print(f"  ask_chunk   : {ask_chunk}")
    print(f"  question    : {question[:120]!r}")
    print(f"  gold_answer : {gold!r} (kind={gold_kind(gold)})")
    print(f"  profile     : {args.profile} (max_len={max_length})")
    print(f"  retriever   : {args.retriever}")
    print(f"  compress    : {args.compress_mode}")

    # Walk
    max_chunk = args.max_chunks if args.max_chunks else ask_chunk + 60
    print(f"\nWalking chunks 0..{max_chunk}...")
    loop.reset()
    reset_visual_index(loop.retriever)

    traces: List[Dict] = []
    response_text = None
    response_chunk = None

    for chunk_idx in range(max_chunk + 1):
        q = question if chunk_idx == ask_chunk else None
        try:
            result = loop.step(chunk_idx=chunk_idx, video_path=video_path,
                               user_question=q)
        except Exception as e:
            print(f"\n💥 chunk {chunk_idx} EXCEPTION: {type(e).__name__}: {e}")
            if chunk_idx <= ask_chunk:
                break
            continue

        trace = result.get("debug_trace")
        if trace:
            traces.append(trace)
            print_step_trace(trace, max_length)

        # Track response
        action = result.get("action")
        if action == "response" and not response_text:
            response_text = result.get("payload", {}).get("response", "")
            response_chunk = chunk_idx
        elif action == "recall" and not response_text:
            fa = result.get("final_action")
            fp = result.get("final_payload") or {}
            if fa == "response":
                response_text = fp.get("response", "")
                response_chunk = chunk_idx

        if chunk_idx >= ask_chunk and response_text:
            break

    print_summary(traces, max_length)

    print()
    print("=" * 78)
    print("RESPONSE")
    print("=" * 78)
    print(f"  gold: {gold!r}")
    print(f"  pred: {response_text!r} (at chunk {response_chunk})")

    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "profile": args.profile,
                "sample": {"id": sample.get("sample_id"),
                           "ask_chunk": ask_chunk,
                           "question": question, "gold": gold},
                "response": response_text,
                "response_chunk": response_chunk,
                "traces": traces,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full trace to {args.dump_json}")


if __name__ == "__main__":
    main()
