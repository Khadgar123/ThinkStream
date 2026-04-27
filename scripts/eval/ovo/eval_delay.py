"""Delay-aware OVO eval — uses ORIGINAL ovo_bench_new.json (NOT formatted).

Why a separate script:
  The formatted version (ovo-bench-formatted.jsonl) collapses every FT
  test_info point into an independent "ask + answer at video_end" record,
  which destroys the delay structure CRR was designed to test:

      ask_time:  297s  (question is posed)
      clue_time: 308s  (evidence appears in video)
      test_info probes: 297, 302, 310, 318, 338
                        type=0 type=0 type=1 type=1 type=1

  i.e., agent should *remember* the question for 11s without answering
  Yes, then answer Yes once evidence arrives. Our SFT explicitly teaches
  this (silent + active queries), but the formatted eval never exercises
  it — every probe is a fresh single-shot QA. This script does it right.

What it does:
  For each CRR sample:
    1. Find ask_chunk = ask_time / AGENT_CHUNK_SEC
    2. Find max_chunk = last test_info.realtime / AGENT_CHUNK_SEC
    3. Run agent_loop chunks 0..max_chunk with retriever indexing
    4. Inject the question at ask_chunk only
    5. Record agent's action+payload at every chunk
    6. At each test_info.realtime, score against type:
         type=0 (before evidence): correct iff agent did NOT say Yes
                                    (lenient: silent OK, "No" OK)
                                    (strict OVO: must explicitly say "No")
         type=1 (after evidence):  correct iff agent said Yes
                                    (response with content starting "Y")
    7. Aggregate strict + lenient accuracy, plus false_positive rate
       (type=0 probes where agent said Yes — the real failure mode).

Hybrid retrieval:
  Pass --retriever hybrid to use BM25 + dense visual scoring (SigLIP).
  Pass --retriever bm25 (default) for the existing baseline. The
  retriever is built once and shared across CRR samples; chunk
  embeddings are cleared between samples (they're per-video).

Usage:
    python scripts/eval/ovo/eval_delay.py \\
        --ckpt output/agent-sft \\
        --benchmark_json /path/to/ovo_bench_new.json \\
        --video_root /path/to/videos \\
        --task CRR \\
        --retriever hybrid \\
        --alpha 0.5 \\
        --n 20
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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


# ─── Helpers ─────────────────────────────────────────────────────────────────


def detect_model_class(ckpt: str):
    name = ckpt.lower()
    basename = Path(ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls, "qwen2.5vl"
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls, "qwen3vl"


def is_yes(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if t.startswith("yes") or t.startswith("y "):
        return True
    if t == "y":
        return True
    return False


def is_no(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if t.startswith("no") or t.startswith("n "):
        return True
    if t == "n":
        return True
    return False


def build_crr_question(sample: dict) -> str:
    """Format CRR question with Yes/No instruction. Mirrors OVO formatter."""
    q = sample["question"]
    return (
        f"{q}\n"
        f"Return \"Yes\" if the action described has happened in the visible "
        f"video so far; otherwise return \"No\"."
    )


def resolve_video_path(video_field: str, video_root: str) -> str:
    p = Path(video_field)
    if p.is_absolute():
        return str(p)
    return str(Path(video_root) / video_field)


# ─── Per-sample eval ─────────────────────────────────────────────────────────


def eval_one_crr(sample, model, processor, tokenizer, model_type,
                 retriever, video_root, max_new_tokens=128,
                 min_pixels=100352*2, max_pixels=100352*4,
                 compress_mode="system"):
    """Run agent through CRR sample respecting ask_time. Return per-probe records."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    ask_time = float(sample["ask_time"])
    test_info = sample["test_info"]
    last_probe_time = max(float(t["realtime"]) for t in test_info)
    max_chunk = int(last_probe_time / AGENT_CHUNK_SEC) + 1
    ask_chunk = int(ask_time / AGENT_CHUNK_SEC)

    question = build_crr_question(sample)

    generate_fn = make_generate_fn(model, processor, model_type=model_type)
    loop = StreamingAgentLoop(
        generate_fn=generate_fn,
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        retriever=retriever,
        compress_mode=compress_mode,
    )
    loop.reset()
    # Ensure retriever is also reset between samples (clears visual index)
    if hasattr(retriever, "chunk_embeddings"):
        retriever.chunk_embeddings.clear()

    # chunk_idx → (action, response_text, raw_output)
    per_chunk = {}
    for chunk_idx in range(max_chunk):
        q = question if chunk_idx == ask_chunk else None
        try:
            result = loop.step(chunk_idx=chunk_idx, video_path=video_path, user_question=q)
        except Exception as e:
            per_chunk[chunk_idx] = ("error", "", str(e))
            continue

        action = result.get("action", "?")
        payload = result.get("payload", {}) or {}
        if action == "recall":
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload", {}) or {}
            response_text = final_payload.get("response", "")
            per_chunk[chunk_idx] = (final_action, response_text, "")
        elif action == "response":
            per_chunk[chunk_idx] = ("response", payload.get("response", ""), "")
        elif action == "silent":
            per_chunk[chunk_idx] = ("silent", "", "")
        elif action == "compress":
            per_chunk[chunk_idx] = ("compress", "", "")
        else:
            per_chunk[chunk_idx] = (action, "", "")

    # Score probes
    probe_records = []
    for probe in test_info:
        t = float(probe["realtime"])
        ptype = probe["type"]
        chunk_idx = int(t / AGENT_CHUNK_SEC)
        action, resp, _ = per_chunk.get(chunk_idx, ("missing", "", ""))

        said_yes = is_yes(resp) if action == "response" else False
        said_no = is_no(resp) if action == "response" else False
        was_silent = action == "silent"

        # Strict OVO: type=0 → must say "No"; type=1 → must say "Yes"
        if ptype == 0:
            strict_correct = said_no
            lenient_correct = was_silent or said_no  # silent OK as "wait"
            false_positive = said_yes  # the real failure mode
        else:  # type == 1
            strict_correct = said_yes
            lenient_correct = said_yes
            false_positive = False

        probe_records.append({
            "realtime": t,
            "chunk_idx": chunk_idx,
            "type": ptype,
            "action": action,
            "response": resp,
            "strict_correct": bool(strict_correct),
            "lenient_correct": bool(lenient_correct),
            "false_positive": bool(false_positive),
        })

    return {
        "id": sample.get("id"),
        "video": sample.get("video"),
        "ask_time": ask_time,
        "clue_time": sample.get("clue_time"),
        "n_probes": len(probe_records),
        "probes": probe_records,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="SFT or RL ckpt dir")
    p.add_argument("--benchmark_json", required=True,
                   help="Path to ORIGINAL ovo_bench_new.json (not formatted)")
    p.add_argument("--video_root", required=True, help="Prefix joined to relative video paths")
    p.add_argument("--task", default="CRR",
                   help="Filter to this task only. CRR has explicit ask_time/clue_time. "
                        "Other tasks won't have ask_time and will be skipped.")
    p.add_argument("--n", type=int, default=None, help="Max samples (default: all)")
    p.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Hybrid alpha: 1.0=pure BM25, 0.0=pure visual")
    p.add_argument(
        "--compress_mode",
        default="system",
        choices=["system", "self"],
        help=(
            "How <action>compress</action> is triggered. 'system' (SFT "
            "ckpt): when memory.should_compress() fires, system inserts "
            "<compress_trigger range='X-Y'/> with FIFO range — model only "
            "writes the <summary>. 'self' (RL ckpt post-GDPO): system never "
            "inserts a trigger; the model autonomously decides when to "
            "compress and which range to summarize. Pure-SFT under 'self' "
            "will likely never compress and overflow on long videos."
        ),
    )
    p.add_argument("--siglip_path", default="google/siglip-base-patch16-224")
    p.add_argument("--max_results", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    # Load ckpt
    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.ckpt)
    register_special_tokens(processor, model_type)
    processor = update_processor_pixels(processor, DataArguments())

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=16384, padding_side="right", use_fast=False
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    # Build retriever (shared across samples; visual index cleared per sample)
    print(f"Building retriever: kind={args.retriever}, alpha={args.alpha}")
    retriever = make_retriever(
        kind=args.retriever,
        siglip_path=args.siglip_path,
        alpha=args.alpha,
        max_results=args.max_results,
        device="cuda",
    )

    # Load benchmark
    with open(args.benchmark_json) as f:
        all_samples = json.load(f)

    samples = [s for s in all_samples if s.get("task") == args.task]
    if args.task == "CRR":
        samples = [s for s in samples if "ask_time" in s and "test_info" in s]
    if args.n:
        samples = samples[: args.n]
    print(f"Filtered to {len(samples)} {args.task} samples with delay structure")

    # Run
    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            r = eval_one_crr(
                s, model, processor, tokenizer, model_type,
                retriever, args.video_root,
                max_new_tokens=args.max_new_tokens,
                compress_mode=args.compress_mode,
            )
            if r is None:
                continue
            results.append(r)
            if (i + 1) % 5 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{i+1}/{len(samples)}] {rate:.2f} samples/min — "
                      f"latest probes correct (strict): "
                      f"{sum(p['strict_correct'] for p in r['probes'])}/{r['n_probes']}")
        except Exception as e:
            print(f"Sample {i} (id={s.get('id')}) failed: {type(e).__name__}: {e}")

    if not results:
        print("No successful samples.")
        return

    # Aggregate
    by = defaultdict(lambda: {"n": 0, "strict": 0, "lenient": 0, "fp": 0})
    for r in results:
        for p in r["probes"]:
            for key in (f"type{p['type']}", "_all"):
                by[key]["n"] += 1
                by[key]["strict"] += int(p["strict_correct"])
                by[key]["lenient"] += int(p["lenient_correct"])
                by[key]["fp"] += int(p["false_positive"])

    print()
    h = f"{'bucket':<10}  {'n':>5}  {'strict':>8}  {'lenient':>8}  {'false_pos':>10}"
    print(h)
    print("-" * len(h))
    for k in sorted(by.keys()):
        v = by[k]
        if v["n"] == 0:
            continue
        print(f"{k:<10}  {v['n']:>5}  "
              f"{v['strict']/v['n']:>8.3f}  "
              f"{v['lenient']/v['n']:>8.3f}  "
              f"{v['fp']/v['n']:>10.3f}")
    print()
    n_samples = len(results)
    n_probes = sum(r["n_probes"] for r in results)
    print(f"{n_samples} samples, {n_probes} probes ({n_probes/n_samples:.1f} probes/sample)")

    out_path = args.out
    if not out_path:
        out_path = (
            f"{args.ckpt}/eval/ovo_delay_{args.task}_"
            f"{args.retriever}_{args.compress_mode}.json"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "task": args.task,
            "compress_mode": args.compress_mode,
            "retriever": {"kind": args.retriever, "alpha": args.alpha,
                          "siglip_path": args.siglip_path if args.retriever == "hybrid" else None},
            "n_samples": n_samples,
            "n_probes": n_probes,
            "by_bucket": {k: dict(v) for k, v in by.items()},
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
