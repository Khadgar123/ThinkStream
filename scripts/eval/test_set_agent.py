#!/usr/bin/env python
"""SFT/RL streaming-agent eval on our test.jsonl.

This is the trajectory-walk evaluator: load an SFT or RL checkpoint, walk the
agent through the video chunk-by-chunk via StreamingAgentLoop.step(), and
score the response emitted at the gold ask_chunk. Mirrors scripts/eval/ovo/
eval_full.py but reads our test.jsonl format.

Why not test_set_sft_gen.py: that script does ONE-SHOT model.generate() with
the input prompt, which feeds the model a pre-built memory state. The
agent-loop path actually walks the timeline, accumulates memory itself, and
exercises think/silent/recall/compress — i.e. the full streaming behaviour
the model was trained on. If the streaming machinery is broken (memory
overflow, recall mis-firing, compression mis-routing) one-shot generation
won't catch it; this script does.

Recall backend (--retriever):
  bm25    — pure lexical (fast, no vision encoder).
  hybrid  — bm25 ⊕ siglip vision ranking (alpha-weighted).
Compress mode (--compress_mode):
  system  — system emits <compress_trigger range="..."> with FIFO range when
            recent_thinks tokens exceed threshold. Model writes summary.
            (Matches SFT training data exactly.)
  self    — no trigger; model is expected to autonomously emit <action>compress
            </action><summary>{...}</summary>. RL-only (pure SFT is OOD here).

Logged per sample:
  action emitted at ask_chunk (or first response after), the response text,
  whether recall fired during the walk, how many compress events fired,
  and whether the response matched gold.

Usage:
    python scripts/eval/test_set_agent.py \\
        --ckpt output/agent-sft \\
        --test_jsonl data/agent_v5/final/test.jsonl \\
        --video_root /path/to/videos \\
        --frames_root data/agent_v5/frames \\
        --retriever hybrid --compress_mode system \\
        [--n 200]
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
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
    update_processor_pixels,
)
from scripts.eval.ovo.eval_full import (
    detect_model_class,
    extract_letter, extract_int, is_yes, is_no,
    reset_visual_index,
)


GOLD_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL)


# ─── test.jsonl helpers (shared in spirit with test_set_base.py) ─────────────


def extract_gold(sample):
    out = sample.get("output", "")
    m = GOLD_RE.search(out)
    return m.group(1).strip() if m else None


def gold_kind(gold):
    if gold is None:
        return None
    g = gold.strip()
    if g in ("Yes", "No"):
        return "yes_no"
    if g.isdigit():
        return "int"
    if len(g) == 1 and g.isalpha():
        return "letter"
    return "descriptive"


def extract_question(sample):
    """User-facing question for this chunk: prefer input.user_input, else the
    most recent unanswered query, else the most recent query of any kind.

    v9.4.2: reject whitespace-only user_input (a string of spaces is truthy
    in Python and would emit a blank <user_input/> tag — OOD vs SFT).
    """
    inp = sample.get("input", {})
    ui = inp.get("user_input")
    if ui and ui.strip():
        return ui
    queries = inp.get("queries", []) or []
    for q in reversed(queries):
        if not q.get("answers"):
            qtext = q.get("question") or ""
            if qtext.strip():
                return qtext
    if queries:
        qtext = queries[-1].get("question") or ""
        if qtext.strip():
            return qtext
    return None


def score(pred_text, gold, kind):
    if not pred_text or not gold:
        return False
    pred = pred_text.strip()
    if kind == "yes_no":
        if is_yes(pred):
            return gold == "Yes"
        if is_no(pred):
            return gold == "No"
        return False
    if kind == "int":
        v = extract_int(pred)
        return v is not None and str(v) == gold
    if kind == "letter":
        let = extract_letter(pred)
        return let is not None and let.upper() == gold.upper()
    return False  # descriptive — not auto-scored


def resolve_video_path(sample, video_root):
    vp = sample.get("video_path", "")
    if not vp:
        return None
    p = Path(vp)
    if p.is_absolute():
        return str(p)
    if video_root:
        return str(Path(video_root) / vp)
    return str(Path(sample.get("data_path") or ".") / vp)


# ─── Agent walk + scoring ────────────────────────────────────────────────────


def walk_and_score(sample, loop, video_path, ask_chunk,
                   max_extra=2, scoring="strict", num_chunks_video=None,
                   support_chunks=None):
    """Walk agent through video chunks, injecting question at ask_chunk.

    Scoring modes:
      strict (default): walk to ask_chunk + max_extra; only count responses
        in [ask_chunk, ask_chunk+max_extra].
      lenient: walk to end of video (or ask_chunk+60); count any response
        at chunk ≥ ask_chunk regardless of how late.

    BUG FIX (v9.4.2): premature responses (chunk < ask_chunk — model
    spuriously answered before being asked) USED to be captured as the
    "answer" for scoring, contaminating accuracy. Now we ONLY accept
    responses at chunk ≥ ask_chunk; premature responses are counted in
    telemetry but don't pollute the prediction.

    Telemetry returned (for stream-eval stats):
      n_compress_events           : how many <compress_trigger> fired
      compress_thinks_at_trigger  : list of len(recent_thinks) at each trigger
      compress_chunks_per_event   : list of len(compressed_chunks) per trigger
                                    (always ~4 = COMPRESS_RANGE_MIN)
      n_recall                    : how many <action>recall</action> fired
      recall_events               : per-recall {chunk_idx, returned_chunks,
                                                hit_support}
      response_offset_chunks      : response_chunk - ask_chunk (None if none)
      n_premature_responses       : responses at chunk < ask_chunk
                                    (DO NOT count toward accuracy)
      n_late_responses            : responses at chunk > ask_chunk + 2

    Args:
        support_chunks: gold support chunks for THIS sample (used to compute
            recall_hit_support). None disables that metric.
    """
    question = extract_question(sample)
    if not question:
        return None

    if scoring == "lenient":
        max_chunk = num_chunks_video - 1 if num_chunks_video else ask_chunk + 60
        max_chunk = max(max_chunk, ask_chunk + 1)
    else:
        max_chunk = ask_chunk + max_extra

    loop.reset()
    # v9.4.2 BUG FIX: reset hybrid retriever's per-video visual index too.
    # Without this, siglip chunk_embeddings from PRIOR samples persist into
    # the current sample's recall pool, causing cross-video leakage that
    # inflates rec_hit and contaminates accuracy on whichever sample
    # happens to share keywords/visuals with a previous trajectory.
    # eval_full.py:eval_mcq already does this (line 324); we missed it here.
    reset_visual_index(loop.retriever)

    actions = []
    response_chunk = None
    response_text = ""
    compress_thinks_at_trigger: List[int] = []
    compress_chunks_per_event: List[int] = []
    recall_events: List[Dict] = []
    n_premature_responses = 0
    n_late_responses = 0
    support_set = set(support_chunks or [])
    # v9.4.2 extra telemetry
    prompt_text_tokens_per_step: List[int] = []
    think_tokens_per_step: List[int] = []
    n_format_violations = 0
    n_compress_attempts = 0       # = how many times the trigger fired
    n_compress_succeeded = 0       # subset where model's <summary> was valid

    # Compress-quality telemetry (the additional metrics)
    compress_chunk_count: Dict[int, int] = {}   # chunk_idx → times rolled into a summary
    n_partial_compress = 0  # events where < COMPRESS_RANGE_MIN thinks were compressed
    n_step_errors = 0  # bounded step errors (e.g. video EOF in lenient mode)
    for chunk_idx in range(max_chunk + 1):
        q = question if chunk_idx == ask_chunk else None
        try:
            result = loop.step(chunk_idx=chunk_idx, video_path=video_path,
                               user_question=q)
        except Exception as e:
            # v9.4.2: lenient mode walks ask_chunk + 60 chunks regardless of
            # video duration → step() can throw on out-of-bounds frames near
            # the end. We tolerate up to 3 consecutive errors AFTER ask_chunk
            # (likely video EOF) before bailing; before ask_chunk a single
            # error means a real video-decode failure → bail immediately.
            actions.append(("error", str(e)[:80]))
            if chunk_idx < ask_chunk:
                break
            n_step_errors += 1
            if n_step_errors >= 3:
                break
            continue

        # Compress telemetry (set by agent_loop when trigger fired this step)
        ct = result.get("compress_telemetry")
        if ct:
            compress_thinks_at_trigger.append(ct["thinks_count_at_trigger"])
            compressed_chunks = ct.get("compressed_chunks") or []
            compress_chunks_per_event.append(len(compressed_chunks))
            n_compress_attempts += 1
            if result.get("compress_succeeded"):
                n_compress_succeeded += 1
            # Track which chunks have been rolled into a summary (multi-count
            # = the chunk has been re-compressed via merge in compressed_segments).
            for c in compressed_chunks:
                compress_chunk_count[int(c)] = compress_chunk_count.get(int(c), 0) + 1
            # Partial compress: model's <summary> covered fewer chunks than
            # the trigger's range. Symptom of summary writing the wrong
            # time_range — bookkeeping here so streaming eval can flag it.
            from thinkstream.model.agent_loop import COMPRESS_RANGE_MIN
            if 0 < len(compressed_chunks) < COMPRESS_RANGE_MIN:
                n_partial_compress += 1
        # Per-step extras
        if result.get("prompt_text_token_count") is not None:
            prompt_text_tokens_per_step.append(result["prompt_text_token_count"])
        if result.get("think_token_count") is not None:
            think_tokens_per_step.append(result["think_token_count"])
        if not result.get("format_ok", True):
            n_format_violations += 1

        action = result.get("action", "?")
        payload = result.get("payload") or {}

        if action == "recall":
            returned = result.get("recall_returned_chunks", []) or []
            # Detect query schema (with_time_range vs keyword_only) and
            # compute per-recall hit fraction (|returned ∩ gold| / |gold|).
            q = (payload or {}).get("query") or {}
            schema = "with_time_range" if isinstance(q, dict) and q.get("time_range") \
                else "keyword_only"
            if support_set:
                returned_set = set(returned)
                hit_frac = len(returned_set & support_set) / max(len(support_set), 1)
            else:
                hit_frac = None
            recall_events.append({
                "chunk_idx": chunk_idx,
                "returned_chunks": list(returned),
                "hit_support": (bool(support_set & set(returned))
                                if support_set else None),
                "schema": schema,
                "hit_fraction": hit_frac,
            })
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload") or {}
            actions.append((final_action, final_payload.get("response", "")[:80]))
            if final_action == "response":
                if chunk_idx < ask_chunk:
                    n_premature_responses += 1
                elif not response_text:
                    response_chunk = chunk_idx
                    response_text = final_payload.get("response", "")
                    if chunk_idx > ask_chunk + 2:
                        n_late_responses += 1
        elif action == "response":
            actions.append(("response", payload.get("response", "")[:80]))
            if chunk_idx < ask_chunk:
                # Premature — ignore for accuracy, count for telemetry.
                n_premature_responses += 1
            elif not response_text:
                response_chunk = chunk_idx
                response_text = payload.get("response", "")
                if chunk_idx > ask_chunk + 2:
                    n_late_responses += 1
        elif action == "compress":
            actions.append(("compress", ""))
        else:
            actions.append((action, ""))

        # Early exit only after the question was asked AND we accepted a response
        if chunk_idx >= ask_chunk and response_text:
            break

    n_steps = max(1, max_chunk + 1)  # at least one step in the loop
    return {
        "question": question,
        "ask_chunk": ask_chunk,
        "response_chunk": response_chunk,
        "response": response_text,
        "response_offset_chunks": (response_chunk - ask_chunk
                                   if response_chunk is not None else None),
        "actions": actions,
        # Compress / recall telemetry
        "n_compress_events": len(compress_thinks_at_trigger),
        "compress_thinks_at_trigger": compress_thinks_at_trigger,
        "compress_chunks_per_event": compress_chunks_per_event,
        "n_recall": len(recall_events),
        "recall_events": recall_events,
        "n_premature_responses": n_premature_responses,
        "n_late_responses": n_late_responses,
        # v9.4.2 extras (per-trajectory)
        "prompt_text_tokens_per_step": prompt_text_tokens_per_step,
        "think_tokens_per_step": think_tokens_per_step,
        "n_format_violations": n_format_violations,
        "n_steps": len(actions),
        "n_compress_attempts": n_compress_attempts,
        "n_compress_succeeded": n_compress_succeeded,
        # Compress-quality extras
        "compress_chunk_count": dict(compress_chunk_count),
        "n_chunks_revisited": sum(1 for c in compress_chunk_count.values() if c > 1),
        "n_partial_compress": n_partial_compress,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default="data/agent_v5/frames")
    p.add_argument("--n", type=int, default=200,
                   help="Max samples to evaluate (after filtering scorable)")
    p.add_argument("--retriever", default="hybrid", choices=["bm25", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--siglip_path", default="google/siglip-base-patch16-224")
    p.add_argument("--use_agent_vision", action="store_true",
                   help="Reuse the agent model's own vision tower for hybrid "
                        "retrieval (no SigLIP). Default off.")
    p.add_argument("--compress_mode", default="system",
                   choices=["system", "self"])
    p.add_argument("--max_results", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_extra_chunks", type=int, default=2,
                   help="How many chunks past ask_chunk to wait for response "
                        "(strict mode only)")
    p.add_argument("--profile", default="16k", choices=["16k", "32k"],
                   help="Eval context profile (see scripts/eval/eval_profiles.py "
                        "for full token-budget breakdown). 16k (default): "
                        "SFT-aligned, queries_cap=8, recall=800char, "
                        "max_new_tokens=128. 32k: extended for Qwen3-VL's "
                        "native context, queries_cap=24, recall=3000char, "
                        "max_new_tokens=256.")
    p.add_argument("--scoring", default="strict", choices=["strict", "lenient"],
                   help="strict: response must come within max_extra_chunks "
                        "of ask_chunk. lenient: walk full video, take any "
                        "response after ask_chunk — measures correctness "
                        "independent of response-timing.")
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    # Apply eval profile FIRST — must run before agent_loop / agent_protocol
    # are used so the module-level caps reflect the requested profile.
    from scripts.eval.eval_profiles import apply_profile, describe_profile
    profile_cfg = apply_profile(args.profile)
    print(describe_profile(args.profile))
    if args.max_new_tokens == 128 and args.profile == "32k":
        # User accepted profile default; honour the profile's max_new_tokens
        args.max_new_tokens = profile_cfg["max_new_tokens_default"]

    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt}")
    model = Cls.from_pretrained(
        args.ckpt, dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=profile_cfg["model_max_length"],
        padding_side="right", use_fast=False,
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    print(f"Building retriever: kind={args.retriever}, alpha={args.alpha}, "
          f"vision_source={'agent' if args.use_agent_vision else 'siglip'}")
    retriever = make_retriever(
        kind=args.retriever, siglip_path=args.siglip_path,
        alpha=args.alpha, max_results=args.max_results, device="cuda",
        agent_model=model if args.use_agent_vision else None,
        agent_processor=processor if args.use_agent_vision else None,
    )

    # v9.4.2: SFT-aligned pixel budget (see eval_full.py for rationale —
    # the prior 200704/401408 values doubled visual-frame token cost
    # vs SFT training and overflowed model_max_length).
    loop = StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer, processor=processor, model_type=model_type,
        min_pixels=100352, max_pixels=150528,
        max_new_tokens=args.max_new_tokens,
        retriever=retriever, compress_mode=args.compress_mode,
        frames_root=args.frames_root, video_root=args.video_root,
    )

    # Filter scorable samples
    samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    scorable = []
    for s in samples:
        if s.get("sample_type") not in ("response", "recall_response"):
            continue
        gold = extract_gold(s)
        kind = gold_kind(gold)
        if kind in ("yes_no", "int", "letter"):
            scorable.append((s, gold, kind))
    if args.n and 0 < args.n < len(scorable):
        scorable = scorable[: args.n]
    print(f"Filtered {len(scorable)} scorable samples (yes_no / int / letter)")

    results = []
    skipped = 0
    t0 = time.time()
    # Per-kind aggregates including v9.4.2 telemetry buckets
    def _empty_bucket():
        return {
            "n": 0, "correct": 0,
            "n_recall": 0, "n_compress_events": 0,
            "n_premature": 0, "n_late": 0,
            "n_no_response": 0,
            # Lists for distribution stats (avg/p50/p95)
            "compress_thinks_at_trigger": [],
            "compress_chunks_per_event": [],
            "response_offset_chunks": [],
            "n_recall_with_support_known": 0,
            "n_recall_hit_support": 0,
            # v9.4.2 extras
            "prompt_tokens_per_step": [],   # all steps across all samples
            "think_tokens_per_step": [],
            "n_format_violations": 0,
            "total_steps": 0,
            "n_compress_attempts": 0,
            "n_compress_succeeded": 0,
            # Recall — per-schema split + fractional hit-rate distribution
            "n_recall_schema_with_time_range": 0,
            "n_recall_schema_with_time_range_hit": 0,
            "n_recall_schema_keyword_only": 0,
            "n_recall_schema_keyword_only_hit": 0,
            "recall_hit_fractions": [],
            # Compress quality
            "n_chunks_revisited": 0,
            "n_partial_compress": 0,
        }
    by = defaultdict(_empty_bucket)

    for i, (s, gold, kind) in enumerate(scorable):
        try:
            video_path = resolve_video_path(s, args.video_root)
            if not video_path or not Path(video_path).exists():
                skipped += 1
                continue
            ask_chunk = int(s.get("chunk_idx", 0))
            # If the sample carries support_chunks (v9.x metadata), use them
            # to score recall hit-rate. Older samples without metadata get
            # None and we skip the recall-acc metric for them.
            metadata = s.get("metadata") or {}
            support_chunks = metadata.get("support_chunks") or s.get("support_chunks")
            walk = walk_and_score(
                s, loop, video_path, ask_chunk,
                max_extra=args.max_extra_chunks,
                scoring=args.scoring,
                num_chunks_video=None,
                support_chunks=support_chunks,
            )
            if walk is None:
                skipped += 1
                continue
            correct = score(walk["response"], gold, kind)
            for k in (kind, "_all"):
                bucket = by[k]
                bucket["n"] += 1
                bucket["correct"] += int(correct)
                bucket["n_recall"] += walk["n_recall"]
                bucket["n_compress_events"] += walk["n_compress_events"]
                bucket["n_premature"] += walk["n_premature_responses"]
                bucket["n_late"] += walk["n_late_responses"]
                if walk["response_chunk"] is None:
                    bucket["n_no_response"] += 1
                else:
                    bucket["response_offset_chunks"].append(
                        walk["response_offset_chunks"])
                bucket["compress_thinks_at_trigger"].extend(
                    walk["compress_thinks_at_trigger"])
                bucket["compress_chunks_per_event"].extend(
                    walk["compress_chunks_per_event"])
                # Recall hit-rate (only when support_chunks known)
                for ev in walk["recall_events"]:
                    if ev["hit_support"] is not None:
                        bucket["n_recall_with_support_known"] += 1
                        if ev["hit_support"]:
                            bucket["n_recall_hit_support"] += 1
                        # Per-schema split: lets us see whether
                        # with_time_range queries actually retrieve
                        # better than keyword_only ones (the whole point
                        # of training the dual schema).
                        sch = ev.get("schema", "?")
                        bucket[f"n_recall_schema_{sch}"] += 1
                        if ev["hit_support"]:
                            bucket[f"n_recall_schema_{sch}_hit"] += 1
                        if ev.get("hit_fraction") is not None:
                            bucket["recall_hit_fractions"].append(ev["hit_fraction"])
                # v9.4.2 extras
                bucket["prompt_tokens_per_step"].extend(walk["prompt_text_tokens_per_step"])
                bucket["think_tokens_per_step"].extend(walk["think_tokens_per_step"])
                bucket["n_format_violations"] += walk["n_format_violations"]
                bucket["total_steps"] += walk["n_steps"]
                bucket["n_compress_attempts"] += walk["n_compress_attempts"]
                bucket["n_compress_succeeded"] += walk["n_compress_succeeded"]
                # Compress-quality extras
                bucket["n_chunks_revisited"] += walk["n_chunks_revisited"]
                bucket["n_partial_compress"] += walk["n_partial_compress"]
            results.append({
                "idx": i,
                "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                "kind": kind, "gold": gold,
                "ask_chunk": ask_chunk,
                "response_chunk": walk["response_chunk"],
                "response": walk["response"][:300],
                "correct": correct,
                "response_offset_chunks": walk["response_offset_chunks"],
                "n_recall": walk["n_recall"],
                "n_compress_events": walk["n_compress_events"],
                "n_premature": walk["n_premature_responses"],
                "n_late": walk["n_late_responses"],
                "compress_thinks_at_trigger": walk["compress_thinks_at_trigger"],
                "recall_events": walk["recall_events"],
                "actions": walk["actions"],
            })
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{i+1}/{len(scorable)}] {rate:.2f} samp/sec")
        except Exception as e:
            print(f"Sample {i} failed: {type(e).__name__}: {e}")
            skipped += 1

    if not results:
        print("No successful runs.")
        return

    # Helper for percentile/avg of small lists.
    def _stats(xs):
        if not xs:
            return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0}
        s = sorted(xs)
        n = len(s)
        return {
            "n": n,
            "avg": sum(s) / n,
            "p50": s[n // 2],
            "p95": s[min(n - 1, int(n * 0.95))],
            "max": s[-1],
        }

    print()
    # Block 1: accuracy + compress + recall + timing
    print("=" * 80)
    print("ACCURACY / COMPRESS / RECALL / TIMING")
    print("=" * 80)
    header = (f"{'kind':<10}  {'n':>5}  {'acc':>6}  "
              f"{'cmp/s':>5}  {'cmp_thk':>7}  {'cmp_ok':>6}  "
              f"{'rec/s':>5}  {'rec_hit':>7}  "
              f"{'off':>5}  {'late':>4}  {'pre':>3}  {'noresp':>6}")
    print(header); print("-" * len(header))
    for k in sorted(by.keys()):
        v = by[k]
        if v["n"] == 0:
            continue
        cmp_thk = _stats(v["compress_thinks_at_trigger"])["avg"]
        off_avg = _stats(v["response_offset_chunks"])["avg"]
        rec_hit = (v["n_recall_hit_support"] / v["n_recall_with_support_known"]
                   if v["n_recall_with_support_known"] else float("nan"))
        rec_hit_str = f"{rec_hit:>7.2f}" if rec_hit == rec_hit else "      —"
        cmp_ok = (v["n_compress_succeeded"] / v["n_compress_attempts"]
                  if v["n_compress_attempts"] else float("nan"))
        cmp_ok_str = f"{cmp_ok:>6.2f}" if cmp_ok == cmp_ok else "     —"
        print(f"{k:<10}  {v['n']:>5}  {v['correct']/v['n']:>6.3f}  "
              f"{v['n_compress_events']/v['n']:>5.1f}  {cmp_thk:>7.1f}  {cmp_ok_str}  "
              f"{v['n_recall']/v['n']:>5.1f}  {rec_hit_str}  "
              f"{off_avg:>5.1f}  {v['n_late']:>4d}  {v['n_premature']:>3d}  "
              f"{v['n_no_response']:>6d}")

    # Block 2: prompt-token distribution, think-len, format violations
    print()
    print("=" * 80)
    print("PROMPT LENGTH / THINK CHATTINESS / FORMAT")
    print("=" * 80)
    header2 = (f"{'kind':<10}  {'pt_avg':>7}  {'pt_p50':>7}  {'pt_p95':>7}  "
               f"{'pt_max':>7}  {'th_avg':>7}  {'th_p95':>7}  {'fmt_err%':>9}")
    print(header2); print("-" * len(header2))
    for k in sorted(by.keys()):
        v = by[k]
        if v["n"] == 0:
            continue
        pt = _stats(v["prompt_tokens_per_step"])
        th = _stats(v["think_tokens_per_step"])
        fmt_err_pct = (100 * v["n_format_violations"] / max(v["total_steps"], 1))
        print(f"{k:<10}  {pt['avg']:>7.0f}  {pt['p50']:>7.0f}  "
              f"{pt['p95']:>7.0f}  {pt['max']:>7.0f}  "
              f"{th['avg']:>7.1f}  {th['p95']:>7.1f}  {fmt_err_pct:>8.1f}%")

    # Block 3: recall schema split + compress quality
    print()
    print("=" * 80)
    print("RECALL SCHEMA / COMPRESS QUALITY")
    print("=" * 80)
    header3 = (f"{'kind':<10}  {'rec_w_tr':>8}  {'hit_w_tr':>8}  "
               f"{'rec_kw':>6}  {'hit_kw':>6}  {'hit_frac':>8}  "
               f"{'cmp_par':>7}  {'revisit':>7}")
    print(header3); print("-" * len(header3))
    for k in sorted(by.keys()):
        v = by[k]
        if v["n"] == 0:
            continue
        rwt = v["n_recall_schema_with_time_range"]
        hwt = (v["n_recall_schema_with_time_range_hit"] / rwt
               if rwt else float("nan"))
        rkw = v["n_recall_schema_keyword_only"]
        hkw = (v["n_recall_schema_keyword_only_hit"] / rkw
               if rkw else float("nan"))
        frac = _stats(v["recall_hit_fractions"])["avg"] if v["recall_hit_fractions"] else float("nan")
        hwt_s = f"{hwt:>8.2f}" if hwt == hwt else "       —"
        hkw_s = f"{hkw:>6.2f}" if hkw == hkw else "     —"
        frac_s = f"{frac:>8.2f}" if frac == frac else "       —"
        print(f"{k:<10}  {rwt:>8d}  {hwt_s}  {rkw:>6d}  {hkw_s}  {frac_s}  "
              f"{v['n_partial_compress']:>7d}  {v['n_chunks_revisited']:>7d}")

    print()
    print("Legend:")
    print("  cmp/s    = avg compress events per sample")
    print("  cmp_thk  = avg # recent_thinks at compress trigger (target ~5-6)")
    print("  cmp_ok   = compress success rate (trigger fired AND model wrote valid <summary>)")
    print("  rec/s    = avg recall events per sample")
    print("  rec_hit  = recall hit rate (returned at least one gold support_chunk)")
    print("  off      = avg response_chunk - ask_chunk (0 = on-time)")
    print("  late     = # responses at chunk > ask_chunk + 2")
    print("  pre      = # premature responses (chunk < ask_chunk; not in acc)")
    print("  noresp   = # samples with no response by max_chunk")
    print("  pt_*     = prompt TEXT-only token count per step (visual ~4700 not included);")
    print("             pt_max should stay below model_max_length - ~5000 visual budget")
    print("  th_*     = <think> token count per step (SFT trained 25-130; high = chatty drift)")
    print("  fmt_err% = format violation rate (no <think>/<action>, malformed payload)")
    print("  rec_w_tr = # recalls whose query carried a time_range")
    print("  hit_w_tr = hit-rate among with_time_range queries")
    print("  rec_kw   = # recalls whose query was keyword-only")
    print("  hit_kw   = hit-rate among keyword-only queries")
    print("  hit_frac = avg |returned ∩ gold| / |gold| across all recalls")
    print("  cmp_par  = # compress events where summary covered <COMPRESS_RANGE_MIN chunks")
    print("  revisit  = # chunks that were rolled into a summary >1 time (re-compression)")
    print(f"\nSkipped: {skipped}")

    out = args.out or (
        f"{args.ckpt}/eval/test_agent/"
        f"{args.compress_mode}_{args.retriever}_{args.scoring}_{args.profile}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "compress_mode": args.compress_mode,
            "retriever": {"kind": args.retriever, "alpha": args.alpha},
            "scoring": args.scoring,
            "profile": args.profile,
            "profile_cfg": profile_cfg,
            "n_samples": len(results),
            "n_skipped": skipped,
            "by_kind": {k: dict(v) for k, v in by.items()},
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
