#!/usr/bin/env python
"""OVO-Bench base eval — VLM-only (no streaming agent, no recall, no compress).

Mirrors scripts/eval/test_set_base.py for the OVO benchmark. Used to measure
"what does a plain Qwen3-VL get on OVO when given N frames of video?" — the
ceiling/baseline a streaming agent must beat.

Two video-context modes (--mode):
  offline    — uniformly sample --max_frames frames from [0, ask_realtime].
               "Offline ceiling": full information up to the question.
  streaming  — uniformly sample --max_frames frames from
               [ask_realtime - visual_window_sec, ask_realtime].
               Apples-to-apples vs the streaming agent's visual_window.

For FT (REC/SSR/CRR) tasks we evaluate at EACH test_info probe time. The model
sees only frames up to the probe time (not future). For RT/BT (MCQ) tasks we
evaluate once at sample.realtime.

Usage:
    python scripts/eval/ovo/base.py \\
        --ckpt Qwen/Qwen3-VL-8B-Instruct \\
        --benchmark_json /path/to/ovo_bench_new.json \\
        --video_root /path/to/videos \\
        --frames_root data/agent_v5/frames \\
        --mode streaming \\
        --max_frames 24
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
from transformers import AutoProcessor

from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels
from scripts.eval.ovo.eval_full import (
    RT_TASKS, BT_TASKS, FT_TASKS, ALL_TASKS,
    detect_model_class,
    build_mcq_question, build_rec_question, build_ssr_question, build_crr_question,
    extract_letter, extract_int, is_yes, is_no,
    resolve_video_path,
)


AGENT_CHUNK_SEC = 2.0
DEFAULT_VISUAL_WINDOW_SEC = 24.0  # 12 chunks × 2s, matches agent_loop default


# ─── Frame sampling ──────────────────────────────────────────────────────────


def sample_frame_paths(frames_root: Path, video_basename: str,
                       t_start: float, t_end: float, n_frames: int):
    """Pick at most n_frames frame paths from [t_start, t_end].

    Frames are pre-extracted at 1 fps (extract_frames.py): file names look
    like {basename}/{idx:06d}.jpg where idx is the second offset (0-based).
    """
    frame_dir = Path(frames_root) / video_basename
    if not frame_dir.exists():
        return None
    all_frames = sorted(frame_dir.glob("*.jpg"))
    if not all_frames:
        return None
    # Filter to [t_start, t_end] by frame index (== second offset).
    in_range = []
    for fp in all_frames:
        try:
            idx = int(fp.stem)
        except ValueError:
            continue
        if t_start <= idx <= t_end:
            in_range.append(fp)
    if not in_range:
        return None
    if len(in_range) <= n_frames:
        return [str(f) for f in in_range]
    # Even sampling
    step = (len(in_range) - 1) / (n_frames - 1) if n_frames > 1 else 1
    indices = [round(i * step) for i in range(n_frames)]
    return [str(in_range[i]) for i in indices]


def build_messages(frame_paths, question):
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a helpful video understanding assistant. Watch "
                    "the video carefully and answer based on observations. "
                    "If the question is yes/no, answer Yes or No. If it asks "
                    "for a count, answer with the integer. If it is multiple "
                    "choice, answer with a single letter A/B/C/D."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": question},
            ],
        },
    ]


# ─── Per-task evaluators (no agent_loop) ─────────────────────────────────────


def eval_one_probe(model, processor, pad_id,
                   frame_paths, question, max_new_tokens):
    """Single VLM forward. Returns the decoded text."""
    messages = build_messages(frame_paths, question)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = gen[0, prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _frames_for_probe(frames_root, video_path, mode, ask_t,
                      visual_window_sec, max_frames):
    """Pick frame range for a probe at time ask_t (seconds)."""
    basename = Path(video_path).stem
    if mode == "streaming":
        t_start = max(0.0, ask_t - visual_window_sec)
        t_end = ask_t
    else:  # offline
        t_start = 0.0
        t_end = ask_t
    return sample_frame_paths(frames_root, basename, t_start, t_end, max_frames)


def _strict_letter(text: str):
    """Strict format check for MC: response.strip() must be exactly one of A-D."""
    if not text:
        return None
    s = text.strip()
    return s if s in ("A", "B", "C", "D") else None


def _strict_yes_no(text: str):
    if not text:
        return None
    s = text.strip()
    return s if s in ("Yes", "No") else None


def _strict_int(text: str):
    if not text:
        return None
    s = text.strip()
    return s if s.isdigit() else None


def eval_mcq_base(sample, model, processor, pad_id, video_root, frames_root,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient"):
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None
    realtime = float(sample["realtime"])
    fp = _frames_for_probe(frames_root, video_path, mode, realtime,
                           visual_window_sec, max_frames)
    if not fp:
        return None
    question = build_mcq_question(sample)
    text = eval_one_probe(model, processor, pad_id, fp, question, max_new_tokens)
    pred = _strict_letter(text) if scoring == "strict" else extract_letter(text)
    gt = chr(65 + sample["gt"])
    return {
        "task": sample["task"], "id": sample.get("id"),
        "probes": [{
            "realtime": realtime, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
        }],
    }


def eval_rec_base(sample, model, processor, pad_id, video_root, frames_root,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient"):
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None
    question = build_rec_question(sample)
    probes = []
    for probe in sample["test_info"]:
        ask_t = float(probe["realtime"])
        fp = _frames_for_probe(frames_root, video_path, mode, ask_t,
                               visual_window_sec, max_frames)
        if not fp:
            continue
        text = eval_one_probe(model, processor, pad_id, fp, question, max_new_tokens)
        if scoring == "strict":
            s = _strict_int(text)
            pred = int(s) if s else None
        else:
            pred = extract_int(text)
        gt = int(probe["count"])
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def _yes_no_pred(text, scoring):
    if scoring == "strict":
        return _strict_yes_no(text)
    if is_yes(text):
        return "Yes"
    if is_no(text):
        return "No"
    return None


def eval_ssr_base(sample, model, processor, pad_id, video_root, frames_root,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient"):
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None
    probes = []
    for probe in sample["test_info"]:
        ask_t = float(probe["realtime"])
        fp = _frames_for_probe(frames_root, video_path, mode, ask_t,
                               visual_window_sec, max_frames)
        if not fp:
            continue
        question = build_ssr_question(probe.get("step", ""))
        text = eval_one_probe(model, processor, pad_id, fp, question, max_new_tokens)
        gt = "Yes" if probe.get("type") == 1 else "No"
        pred = _yes_no_pred(text, scoring)
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def eval_crr_base(sample, model, processor, pad_id, video_root, frames_root,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient"):
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        return None
    question = build_crr_question(sample)
    probes = []
    for probe in sample["test_info"]:
        ask_t = float(probe["realtime"])
        fp = _frames_for_probe(frames_root, video_path, mode, ask_t,
                               visual_window_sec, max_frames)
        if not fp:
            continue
        text = eval_one_probe(model, processor, pad_id, fp, question, max_new_tokens)
        gt = "Yes" if probe.get("type") == 1 else "No"
        pred = _yes_no_pred(text, scoring)
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "type": probe.get("type"),
            "raw": text[:200], "correct": pred == gt,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def dispatch(sample, **kw):
    task = sample.get("task")
    if task in RT_TASKS or task in BT_TASKS:
        return eval_mcq_base(sample, **kw)
    if task == "REC":
        return eval_rec_base(sample, **kw)
    if task == "SSR":
        return eval_ssr_base(sample, **kw)
    if task == "CRR":
        return eval_crr_base(sample, **kw)
    return None


# ─── Aggregation ─────────────────────────────────────────────────────────────


def aggregate(results):
    by_task = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        for p in r.get("probes", []):
            by_task[r["task"]]["n"] += 1
            by_task[r["task"]]["correct"] += int(bool(p.get("correct")))
    out = {}
    for t, v in by_task.items():
        out[t] = {"n": v["n"], "acc": v["correct"] / max(v["n"], 1)}
    # Category averages (mean of per-task accs in each category)
    def cat_avg(tasks):
        accs = [out[t]["acc"] for t in tasks if t in out]
        return {"avg": sum(accs) / max(len(accs), 1), "n_tasks": len(accs)}
    return {
        "per_task": out,
        "category": {
            "RT": cat_avg(RT_TASKS),
            "BT": cat_avg(BT_TASKS),
            "FT": cat_avg(FT_TASKS),
        },
        "overall": (cat_avg(RT_TASKS)["avg"] + cat_avg(BT_TASKS)["avg"]
                    + cat_avg(FT_TASKS)["avg"]) / 3,
    }


def print_report(agg):
    print()
    print(f"{'task':<6}  {'n':>6}  {'acc':>8}")
    print("-" * 25)
    for t in sorted(agg["per_task"].keys()):
        v = agg["per_task"][t]
        print(f"{t:<6}  {v['n']:>6}  {v['acc']:>8.3f}")
    print()
    for cat in ("RT", "BT", "FT"):
        v = agg["category"][cat]
        print(f"{cat}: {v['avg']:.3f}  ({v['n_tasks']} tasks)")
    print(f"OVERALL: {agg['overall']:.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default="data/agent_v5/frames",
                   help="Pre-extracted 1fps frames root (extract_frames.py output)")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated subset of OVO tasks (default: all 12)")
    p.add_argument("--n_per_task", type=int, default=None)
    p.add_argument("--mode", default="streaming",
                   choices=["offline", "streaming"])
    p.add_argument("--max_frames", type=int, default=24,
                   help="Frame budget. Streaming default: 24 (fixed window). "
                        "Offline sweep: 64 / 128 / 256 / 512 / 1024.")
    p.add_argument("--visual_window_sec", type=float, default=DEFAULT_VISUAL_WINDOW_SEC,
                   help="Window size for streaming mode (default: 24s = 12 chunks)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--scoring", default="lenient", choices=["lenient", "strict"],
                   help="lenient (default): first-matching-token wins. strict: "
                        "response.strip() must EQUAL the expected token "
                        "(A-D / Yes-No / digit). Strict tells you whether "
                        "the model natively follows the requested format.")
    p.add_argument("--profile", default="16k", choices=["16k", "32k"],
                   help="Eval context profile (parity with agent eval). For "
                        "base eval the only effect is metadata stamping in "
                        "the output JSON — base VLM doesn't use queries/recall "
                        "caps. Match the agent profile when comparing.")
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    Cls, _ = detect_model_class(args.ckpt)
    print(f"[mode={args.mode} max_frames={args.max_frames}] Loading {Cls.__name__} from {args.ckpt}")
    model = Cls.from_pretrained(
        args.ckpt, dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)
    task_filter = (set(t.strip() for t in args.tasks.split(",")) if args.tasks
                   else ALL_TASKS)
    by_task = defaultdict(list)
    for s in all_samples:
        if s.get("task") in task_filter:
            by_task[s["task"]].append(s)
    if args.n_per_task:
        for t in by_task:
            by_task[t] = by_task[t][:args.n_per_task]

    total = sum(len(v) for v in by_task.values())
    print(f"Running base eval on {total} samples across {len(by_task)} tasks: "
          f"{sorted(by_task.keys())}")

    kw = dict(
        model=model, processor=processor, pad_id=pad_id,
        video_root=args.video_root, frames_root=args.frames_root,
        mode=args.mode, visual_window_sec=args.visual_window_sec,
        max_frames=args.max_frames, max_new_tokens=args.max_new_tokens,
        scoring=args.scoring,
    )

    results = []
    t0 = time.time()
    done = 0
    for task in sorted(by_task.keys()):
        for sample in by_task[task]:
            try:
                r = dispatch(sample, **kw)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"[{task} id={sample.get('id')}] {type(e).__name__}: {e}")
            done += 1
            if done % 10 == 0:
                rate = done / max(1e-6, time.time() - t0)
                print(f"[{done}/{total}] {rate*60:.1f} samples/min")

    if not results:
        print("No successful samples.")
        return

    agg = aggregate(results)
    print_report(agg)

    out = args.out or (
        f"{args.ckpt}/eval/ovo_base/"
        f"{args.mode}_{args.max_frames}_{args.scoring}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "ckpt": args.ckpt, "mode": args.mode,
            "max_frames": args.max_frames,
            "visual_window_sec": args.visual_window_sec,
            "scoring": args.scoring,
            "profile": args.profile,
            "tasks_evaluated": sorted(by_task.keys()),
            "n_samples": len(results),
            "summary": agg, "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
