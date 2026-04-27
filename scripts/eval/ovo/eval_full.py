"""Unified OVO-Bench eval on the ORIGINAL ovo_bench_new.json.

Why this script vs eval_ovo.py:
  eval_ovo.py reads ovo-bench-formatted.jsonl, where each FT test_info
  point was pre-expanded into an independent (ask, answer at video_end)
  record. That destroys the ask_time / clue_time / test_info delay
  structure that CRR was specifically designed to test, and makes
  REC's "running count" semantics indistinguishable from a single-shot
  QA at the last probe time.

  This script reads the original 1,640-record ovo_bench_new.json and
  dispatches per task family:

    BT / RT (EPM ASI HLD OCR ACR ATR STU FPD OJR)
       - Single realtime, MCQ A/B/C/D.
       - ask_chunk = realtime / 2; one probe; score first letter vs gt.

    FT-REC
       - test_info has multiple {realtime, count} probes (cumulative count).
       - Question active from chunk 0; probe at each test_info.realtime
         and regex-extract integer.

    FT-SSR
       - Each test_info entry asks about a specific step at a specific
         time (different probes can ask about different steps). Treated
         as independent single-shot Yes/No queries (no shared ask_time).

    FT-CRR
       - Has ask_time AND clue_time. Question is asked once at ask_time;
         the agent must wait for evidence (clue_time) before saying Yes.
         test_info type=0 = before clue (expect No / silent),
         type=1 = after clue (expect Yes). This is the only OVO task
         with genuine ask-vs-answer delay structure.

  All tasks share the same agent loop with the same compress_mode and
  retriever — so a single eval run gives directly comparable per-task
  numbers.

Reports:
  Per task: accuracy
  Per category (RT / BT / FT): mean of task accuracies
  Overall: mean of category averages (matches OVO paper Table 2)
  CRR specifically: also reports type=0 / type=1 / fp_rate breakdown
                    (the delay-sensitive metric)

Usage:
    python scripts/eval/ovo/eval_full.py \\
        --ckpt output/agent-sft \\
        --benchmark_json /path/to/ovo_bench_new.json \\
        --video_root /path/to/videos \\
        --compress_mode system \\
        --retriever hybrid \\
        [--tasks CRR,SSR,REC] [--n 30]
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


# ─── Task taxonomy (mirrors official constant.py) ────────────────────────────

RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
BT_TASKS = {"EPM", "ASI", "HLD"}
FT_TASKS = {"REC", "SSR", "CRR"}
ALL_TASKS = RT_TASKS | BT_TASKS | FT_TASKS


# ─── Detect ckpt model class ─────────────────────────────────────────────────


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


def resolve_video_path(video_field, video_root):
    p = Path(video_field)
    return str(p) if p.is_absolute() else str(Path(video_root) / video_field)


# ─── Per-task prompt builders (match OVO official prompt families) ───────────

def build_mcq_question(sample):
    """BT/RT MCQ prompt: question + options + 'Answer with a single letter.'"""
    options = sample.get("options", [])
    lines = [sample["question"]]
    for i, opt in enumerate(options):
        lines.append(f"{chr(65+i)}. {opt}")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def build_rec_question(sample):
    """REC prompt asks for cumulative integer count."""
    activity = sample.get("activity", "perform the action")
    return (
        f"You're watching a video where people may perform a certain action "
        f"repetitively. The performer is referred to as 'they'.\n"
        f"How many times have they {activity} so far?\n"
        f"Your response type should be INT, for example, 0/1/2/3."
    )


def build_ssr_question(step_text):
    """SSR prompt asks Yes/No about a specific step."""
    return (
        f"You're watching a tutorial video which contains a sequence of steps. "
        f"The following is one step from the procedure:\n\n{step_text}\n\n"
        f"Your task is to decide: Is the person in the video currently "
        f"carrying out this step?\n"
        f"Return \"Yes\" if they are; return \"No\" if not."
    )


def build_crr_question(sample):
    """CRR prompt asks Yes/No about whether a described action has occurred."""
    return (
        f"{sample['question']}\n"
        f"Return \"Yes\" if the action described has happened in the visible "
        f"video so far; otherwise return \"No\"."
    )


# ─── Answer extraction & scoring ─────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-Da-d])\b")
_INT_RE = re.compile(r"\d+")


def extract_letter(text):
    if not text:
        return None
    t = text.strip()
    # First A-D letter (word boundary or first char)
    if t and t[0].upper() in "ABCD":
        return t[0].upper()
    m = _LETTER_RE.search(t)
    return m.group(1).upper() if m else None


def extract_int(text):
    if not text:
        return None
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def is_yes(text):
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("yes") or t.startswith("y ") or t == "y"


def is_no(text):
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("no") or t.startswith("n ") or t == "n"


# ─── Agent runner: shared streaming loop ─────────────────────────────────────


def run_agent(loop, video_path, ask_chunks, max_chunk):
    """Run agent through chunks 0..max_chunk, injecting questions per ask_chunks.

    ask_chunks: dict {chunk_idx: question_text} — question(s) to inject at
                specific chunks. Multiple injections are supported (e.g.,
                SSR injects fresh per probe).

    Returns: dict {chunk_idx: (action, response_text)}
    """
    per_chunk = {}
    for chunk_idx in range(max_chunk + 1):
        q = ask_chunks.get(chunk_idx)
        try:
            result = loop.step(chunk_idx=chunk_idx, video_path=video_path, user_question=q)
        except Exception as e:
            per_chunk[chunk_idx] = ("error", str(e))
            continue
        action = result.get("action", "?")
        payload = result.get("payload") or {}
        if action == "recall":
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload") or {}
            per_chunk[chunk_idx] = (final_action, final_payload.get("response", ""))
        elif action == "response":
            per_chunk[chunk_idx] = ("response", payload.get("response", ""))
        elif action == "silent":
            per_chunk[chunk_idx] = ("silent", "")
        elif action == "compress":
            per_chunk[chunk_idx] = ("compress", "")
        else:
            per_chunk[chunk_idx] = (action, "")
    return per_chunk


def make_loop(model, processor, tokenizer, model_type, retriever,
              compress_mode, max_new_tokens, frames_root=None, video_root=None):
    return StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=100352*2,
        max_pixels=100352*4,
        max_new_tokens=max_new_tokens,
        retriever=retriever,
        compress_mode=compress_mode,
        frames_root=frames_root,
        video_root=video_root,
    )


def reset_visual_index(retriever):
    if hasattr(retriever, "chunk_embeddings"):
        retriever.chunk_embeddings.clear()


# ─── Per-task evaluators ─────────────────────────────────────────────────────


def eval_mcq(sample, loop, retriever, video_root):
    """BT / RT: single-realtime MCQ. ask once, score the response at that chunk."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    realtime = float(sample["realtime"])
    ask_chunk = int(realtime / AGENT_CHUNK_SEC)
    # Run a couple of chunks past ask_chunk in case model was silent and
    # responded next chunk (rare but happens).
    max_chunk = ask_chunk + 2

    question = build_mcq_question(sample)
    loop.reset()
    reset_visual_index(retriever)
    per_chunk = run_agent(loop, video_path, {ask_chunk: question}, max_chunk)

    # Find first response at or after ask_chunk
    pred_letter = None
    response_chunk = None
    for c in range(ask_chunk, max_chunk + 1):
        action, resp = per_chunk.get(c, ("missing", ""))
        if action == "response" and resp:
            pred_letter = extract_letter(resp)
            response_chunk = c
            break

    gt_idx = sample["gt"]
    gt_letter = chr(65 + gt_idx)
    correct = pred_letter == gt_letter

    return {
        "task": sample["task"],
        "id": sample.get("id"),
        "probes": [{
            "realtime": realtime,
            "ask_chunk": ask_chunk,
            "response_chunk": response_chunk,
            "gt": gt_letter,
            "pred": pred_letter,
            "correct": correct,
        }],
    }


def eval_rec(sample, loop, retriever, video_root):
    """FT-REC: cumulative integer count. Inject question at chunk 0,
    score at each test_info probe."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    test_info = sample["test_info"]
    last_probe = max(float(t["realtime"]) for t in test_info)
    max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1

    question = build_rec_question(sample)
    loop.reset()
    reset_visual_index(retriever)
    per_chunk = run_agent(loop, video_path, {0: question}, max_chunk)

    probes = []
    for probe in test_info:
        t = float(probe["realtime"])
        gt_count = int(probe["count"])
        c = int(t / AGENT_CHUNK_SEC)
        # Find response at this chunk or the closest preceding one
        action, resp = per_chunk.get(c, ("missing", ""))
        # If silent at exact probe, fall back to last response before it
        if action != "response":
            for back in range(c, -1, -1):
                a, r = per_chunk.get(back, ("missing", ""))
                if a == "response" and r:
                    action, resp = a, r
                    break
        pred_count = extract_int(resp)
        probes.append({
            "realtime": t,
            "chunk_idx": c,
            "gt": gt_count,
            "pred": pred_count,
            "correct": pred_count == gt_count,
        })
    return {"task": "REC", "id": sample.get("id"), "probes": probes}


def eval_ssr(sample, loop, retriever, video_root):
    """FT-SSR: each test_info entry asks about a specific step at a specific
    time. Treat each as independent (inject fresh question per probe)."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    test_info = sample["test_info"]
    last_probe = max(float(t["realtime"]) for t in test_info)
    max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1

    # Inject question at each probe's chunk (with that probe's step)
    ask_chunks = {}
    probes_meta = []
    for probe in test_info:
        t = float(probe["realtime"])
        c = int(t / AGENT_CHUNK_SEC)
        step = probe.get("step", "")
        ask_chunks[c] = build_ssr_question(step)
        probes_meta.append({"realtime": t, "chunk_idx": c, "type": probe["type"]})

    loop.reset()
    reset_visual_index(retriever)
    per_chunk = run_agent(loop, video_path, ask_chunks, max_chunk)

    probes = []
    for meta in probes_meta:
        c = meta["chunk_idx"]
        ptype = meta["type"]
        action, resp = per_chunk.get(c, ("missing", ""))
        said_yes = is_yes(resp) if action == "response" else False
        said_no = is_no(resp) if action == "response" else False
        was_silent = action == "silent"
        if ptype == 0:
            strict = said_no
            lenient = was_silent or said_no
            fp = said_yes
        else:
            strict = said_yes
            lenient = said_yes
            fp = False
        probes.append({
            "realtime": meta["realtime"],
            "chunk_idx": c,
            "type": ptype,
            "action": action,
            "response": resp,
            "strict_correct": strict,
            "lenient_correct": lenient,
            "false_positive": fp,
        })
    return {"task": "SSR", "id": sample.get("id"), "probes": probes}


def eval_crr(sample, loop, retriever, video_root):
    """FT-CRR: ask once at ask_time; probe at each test_info time. Delay-sensitive."""
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None

    ask_time = float(sample["ask_time"])
    test_info = sample["test_info"]
    last_probe = max(float(t["realtime"]) for t in test_info)
    ask_chunk = int(ask_time / AGENT_CHUNK_SEC)
    max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1

    question = build_crr_question(sample)
    loop.reset()
    reset_visual_index(retriever)
    per_chunk = run_agent(loop, video_path, {ask_chunk: question}, max_chunk)

    probes = []
    for probe in test_info:
        t = float(probe["realtime"])
        ptype = probe["type"]
        c = int(t / AGENT_CHUNK_SEC)
        action, resp = per_chunk.get(c, ("missing", ""))
        said_yes = is_yes(resp) if action == "response" else False
        said_no = is_no(resp) if action == "response" else False
        was_silent = action == "silent"
        if ptype == 0:
            strict = said_no
            lenient = was_silent or said_no
            fp = said_yes
        else:
            strict = said_yes
            lenient = said_yes
            fp = False
        probes.append({
            "realtime": t,
            "chunk_idx": c,
            "type": ptype,
            "action": action,
            "response": resp,
            "strict_correct": strict,
            "lenient_correct": lenient,
            "false_positive": fp,
        })
    return {
        "task": "CRR", "id": sample.get("id"),
        "ask_time": ask_time, "clue_time": sample.get("clue_time"),
        "probes": probes,
    }


def dispatch_eval(sample, loop, retriever, video_root):
    task = sample.get("task")
    if task in BT_TASKS or task in RT_TASKS:
        return eval_mcq(sample, loop, retriever, video_root)
    if task == "REC":
        return eval_rec(sample, loop, retriever, video_root)
    if task == "SSR":
        return eval_ssr(sample, loop, retriever, video_root)
    if task == "CRR":
        return eval_crr(sample, loop, retriever, video_root)
    return None


# ─── Reporting ───────────────────────────────────────────────────────────────


def aggregate(results):
    """Build per-task / per-category / overall accuracy. Matches OVO Table 2."""
    by_task = defaultdict(lambda: {"n": 0, "correct": 0,
                                    "fp_n": 0, "fp": 0,
                                    "type0_n": 0, "type0_strict": 0, "type0_lenient": 0,
                                    "type1_n": 0, "type1_strict": 0, "type1_lenient": 0})
    for r in results:
        task = r["task"]
        for p in r["probes"]:
            by_task[task]["n"] += 1
            # MCQ / REC use 'correct'; FT-Y/N uses strict_correct
            if "correct" in p:
                by_task[task]["correct"] += int(p["correct"])
            else:
                by_task[task]["correct"] += int(p["strict_correct"])
                # delay-specific buckets
                if p.get("type") == 0:
                    by_task[task]["type0_n"] += 1
                    by_task[task]["type0_strict"] += int(p["strict_correct"])
                    by_task[task]["type0_lenient"] += int(p["lenient_correct"])
                    by_task[task]["fp"] += int(p.get("false_positive", False))
                    by_task[task]["fp_n"] += 1
                elif p.get("type") == 1:
                    by_task[task]["type1_n"] += 1
                    by_task[task]["type1_strict"] += int(p["strict_correct"])
                    by_task[task]["type1_lenient"] += int(p["lenient_correct"])

    # Category averages: mean of task accuracies (matches OVO paper)
    def cat_avg(tasks):
        accs = []
        for t in tasks:
            v = by_task.get(t)
            if v and v["n"] > 0:
                accs.append(v["correct"] / v["n"])
        return sum(accs) / len(accs) if accs else 0.0, len(accs)

    rt_avg, rt_n = cat_avg(RT_TASKS)
    bt_avg, bt_n = cat_avg(BT_TASKS)
    ft_avg, ft_n = cat_avg(FT_TASKS)
    overall = (rt_avg + bt_avg + ft_avg) / max(1, sum(1 for n in [rt_n, bt_n, ft_n] if n > 0))

    return {
        "by_task": dict(by_task),
        "category": {
            "RT": {"avg": rt_avg, "n_tasks": rt_n},
            "BT": {"avg": bt_avg, "n_tasks": bt_n},
            "FT": {"avg": ft_avg, "n_tasks": ft_n},
        },
        "overall": overall,
    }


def print_report(agg):
    print()
    print(f"{'task':<8}  {'n':>6}  {'acc':>7}  {'fp_rate':>9}  notes")
    print("-" * 70)
    for task in sorted(agg["by_task"]):
        v = agg["by_task"][task]
        if v["n"] == 0:
            continue
        acc = v["correct"] / v["n"]
        fp = (v["fp"] / v["fp_n"]) if v["fp_n"] > 0 else 0.0
        notes = ""
        if task in ("CRR", "SSR"):
            t0 = v["type0_n"]
            t1 = v["type1_n"]
            t0a = (v["type0_strict"] / t0) if t0 else 0.0
            t1a = (v["type1_strict"] / t1) if t1 else 0.0
            notes = f"t0_strict={t0a:.3f}({t0}) t1_strict={t1a:.3f}({t1})"
        print(f"{task:<8}  {v['n']:>6}  {acc:>7.3f}  {fp:>9.3f}  {notes}")

    print()
    print(f"Real-Time Visual Perception (RT): {agg['category']['RT']['avg']:.3f} "
          f"({agg['category']['RT']['n_tasks']} tasks)")
    print(f"Backward Tracing (BT):            {agg['category']['BT']['avg']:.3f} "
          f"({agg['category']['BT']['n_tasks']} tasks)")
    print(f"Forward Active Responding (FT):   {agg['category']['FT']['avg']:.3f} "
          f"({agg['category']['FT']['n_tasks']} tasks)")
    print(f"OVERALL (mean of categories):     {agg['overall']:.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmark_json", required=True,
                   help="Path to ORIGINAL ovo_bench_new.json")
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default=None,
                   help="Root dir with pre-extracted 1fps frames (skip online decode)")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated subset (e.g., CRR,SSR,REC). Default: all 12.")
    p.add_argument("--n_per_task", type=int, default=None,
                   help="Cap samples per task (for quick smoke-tests)")
    p.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--siglip_path", default="google/siglip-base-patch16-224")
    p.add_argument("--compress_mode", default="system", choices=["system", "self"])
    p.add_argument("--max_results", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    # Load model
    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    register_special_tokens(processor, model_type)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=16384, padding_side="right", use_fast=False
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    print(f"Building retriever: kind={args.retriever}, alpha={args.alpha}")
    retriever = make_retriever(
        kind=args.retriever, siglip_path=args.siglip_path,
        alpha=args.alpha, max_results=args.max_results, device="cuda",
    )

    loop = make_loop(model, processor, tokenizer, model_type, retriever,
                     args.compress_mode, args.max_new_tokens,
                     frames_root=args.frames_root, video_root=args.video_root)

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)

    task_filter = set(t.strip() for t in args.tasks.split(",")) if args.tasks else ALL_TASKS

    # Group by task to apply per-task caps and report progress
    by_task = defaultdict(list)
    for s in all_samples:
        if s.get("task") in task_filter:
            by_task[s["task"]].append(s)
    if args.n_per_task:
        for t in by_task:
            by_task[t] = by_task[t][: args.n_per_task]

    total_samples = sum(len(v) for v in by_task.values())
    print(f"Running on {total_samples} samples across {len(by_task)} tasks: "
          f"{sorted(by_task.keys())}")

    results = []
    t0 = time.time()
    done = 0
    for task in sorted(by_task.keys()):
        for sample in by_task[task]:
            try:
                r = dispatch_eval(sample, loop, retriever, args.video_root)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"[{task} id={sample.get('id')}] failed: {type(e).__name__}: {e}")
            done += 1
            if done % 10 == 0:
                rate = done / max(1e-6, time.time() - t0)
                print(f"[{done}/{total_samples}] {rate*60:.1f} samples/min")

    if not results:
        print("No successful samples.")
        return

    agg = aggregate(results)
    print_report(agg)

    out_path = args.out or (
        f"{args.ckpt}/eval/ovo_full/full_{args.compress_mode}_{args.retriever}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "compress_mode": args.compress_mode,
            "retriever": {"kind": args.retriever, "alpha": args.alpha,
                          "siglip_path": args.siglip_path if args.retriever == "hybrid" else None},
            "tasks_evaluated": sorted(by_task.keys()),
            "n_samples": len(results),
            "summary": agg,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
