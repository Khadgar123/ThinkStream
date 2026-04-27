#!/usr/bin/env python
"""Fair base-model eval on our test.jsonl — two video-context modes.

Extracts the user question and gold answer from each test sample,
asks the base model the question with visual context, and scores the
output via OVO-style prefix/substring match.

Two modes (--mode):
  offline    — full video [0, video_end] uniformly sampled to 64 frames.
               This is the base-VLM ceiling: "what's the best a strong
               offline VLM can do given full information?"
  streaming  — only the visual_window slice the streaming agent sees,
               i.e. the last visual_window_chunks*agent_chunk_sec
               seconds before the decision (default 12*2=24 sec, 24
               frames @ 1fps). This is the apples-to-apples baseline:
               "given the SAME visual context our agent has at decision
               time, what does base produce?" — the relevant comparison
               for measuring what the agent protocol adds.

Only scores Yes/No, integer, and single-letter responses (587 of 1,600
test samples after filtering response/recall_response). Descriptive
gold responses (323 samples) are skipped because reliable scoring
requires an LLM judge — out of scope for a quick base baseline.

Single-GPU usage:
    python scripts/eval/test_set_base.py \\
        --ckpt Qwen/Qwen3-VL-8B-Instruct \\
        --test_jsonl data/agent_v5/final/test.jsonl \\
        --mode streaming \\
        --n 200
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


GOLD_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL)


def detect_model_class(ckpt: str):
    name = ckpt.lower()
    basename = Path(ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls
    if "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls


def extract_gold(sample):
    out = sample.get("output", "")
    if not out and "messages" in sample:
        last = sample["messages"][-1]
        c = last.get("content", "")
        if isinstance(c, list):
            out = (c[0] if c else {}).get("text", "")
        else:
            out = c
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
    """Pull the user-facing question relevant to this chunk.

    Priority: input.user_input on this chunk → most recent unanswered
    query → most recent query of any kind. Returns None if no question
    is found (then the sample is skipped).
    """
    inp = sample.get("input", {})
    ui = inp.get("user_input")
    if ui:
        return ui
    queries = inp.get("queries", []) or []
    for q in reversed(queries):
        if not q.get("answers"):
            return q.get("question")
    if queries:
        return queries[-1].get("question")
    return None


def score(pred_text, gold, kind):
    if not pred_text or not gold:
        return False
    pred = pred_text.strip()
    if kind == "yes_no":
        # Whichever of "yes"/"no" appears first in the response
        lower = pred.lower()
        i_yes, i_no = lower.find("yes"), lower.find("no")
        i_yes = i_yes if i_yes >= 0 else 10**9
        i_no = i_no if i_no >= 0 else 10**9
        if i_yes == i_no == 10**9:
            return False
        return ("Yes" if i_yes < i_no else "No") == gold
    if kind == "int":
        m = re.search(r"\d+", pred)
        return bool(m) and m.group() == gold
    if kind == "letter":
        return pred[:1].upper() == gold.upper()
    return False  # descriptive: not scored


def resolve_video_path(sample, video_root):
    vp = sample.get("video_path", "")
    if not vp:
        return None
    p = Path(vp)
    if p.is_absolute():
        return str(p)
    if video_root:
        return str(Path(video_root) / vp)
    base = Path(sample.get("data_path") or ".")
    return str(base / vp)


def build_messages(video_input, question, is_frame_paths=False):
    """Build messages for base model eval.

    Args:
        video_input: either a video file path (offline mode) or a list of
            pre-extracted frame paths (streaming mode).
        question: user question text.
        is_frame_paths: if True, video_input is a list of frame paths and
            we pass it directly to the processor (no online decoding).
    """
    video_content: dict
    if is_frame_paths:
        video_content = {
            "type": "video",
            "video": video_input,
        }
    else:
        video_content = {
            "type": "video",
            "video": video_input,
        }
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful video understanding assistant. Watch the "
                        "video carefully and answer questions based on what you observe. "
                        "If the question is yes/no, answer with Yes or No. If the "
                        "question asks for a count, answer with the integer."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": question},
            ],
        },
    ]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument(
        "--mode",
        default="streaming",
        choices=["offline", "streaming"],
        help="offline = full video [0, video_end]; streaming = only the "
        "visual_window slice the agent sees at decision time.",
    )
    p.add_argument("--n", type=int, default=200, help="Max samples to evaluate")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Override frame count. Defaults: offline=64, streaming=24",
    )
    p.add_argument(
        "--visual_window_chunks",
        type=int,
        default=12,
        help="Window length for --mode streaming (chunks of agent_chunk_sec).",
    )
    p.add_argument("--agent_chunk_sec", type=float, default=2.0)
    p.add_argument("--video_root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    if args.max_frames is None:
        args.max_frames = 64 if args.mode == "offline" else (
            args.visual_window_chunks * 2
        )

    Cls = detect_model_class(args.ckpt)
    print(f"[mode={args.mode}] Loading {Cls.__name__} from {args.ckpt} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()

    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # Filter test.jsonl to scorable response samples
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
    print(f"Filtered {len(scorable)} scorable samples (Yes/No, integer, letter)")

    results = []
    skipped = 0
    t0 = time.time()
    root_path = Path(args.test_jsonl).resolve().parent.parent.parent.parent  # .../ThinkStream
    for i, (s, gold, kind) in enumerate(scorable):
        try:
            chunk_idx = int(s.get("chunk_idx", 0))
            decision_end = (chunk_idx + 1) * args.agent_chunk_sec
            question = extract_question(s)
            if not question:
                skipped += 1
                continue

            if args.mode == "offline":
                video_path = resolve_video_path(s, args.video_root)
                if not video_path:
                    skipped += 1
                    continue
                # Infer pre-extracted frame directory from video basename
                video_basename = Path(video_path).stem
                frame_dir = root_path / "data" / "agent_v5" / "frames" / video_basename
                if not frame_dir.exists():
                    # fallback: infer from sample's existing frame_paths
                    inp = s.get("input", {})
                    vw = inp.get("visual_window", {})
                    frame_paths = vw.get("frame_paths", [])
                    if frame_paths:
                        frame_dir = root_path / Path(frame_paths[0]).parent
                    else:
                        skipped += 1
                        continue
                all_frames = sorted(frame_dir.glob("*.jpg"))
                if not all_frames:
                    skipped += 1
                    continue
                n_frames = len(all_frames)
                if n_frames > args.max_frames:
                    indices = [int(i * (n_frames - 1) / (args.max_frames - 1)) for i in range(args.max_frames)]
                    sampled = [str(all_frames[i]) for i in indices]
                else:
                    sampled = [str(f) for f in all_frames]
                messages = build_messages(sampled, question, is_frame_paths=True)
            else:  # streaming: use pre-extracted frame_paths from visual_window
                inp = s.get("input", {})
                vw = inp.get("visual_window", {})
                frame_paths = vw.get("frame_paths", [])
                if not frame_paths:
                    skipped += 1
                    continue
                # Resolve relative paths to absolute
                abs_paths = []
                for fp in frame_paths:
                    p = Path(fp)
                    if p.is_absolute():
                        abs_paths.append(str(p))
                    else:
                        abs_paths.append(str(root_path / fp))
                messages = build_messages(abs_paths, question, is_frame_paths=True)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {
                k: v.to(model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            new_tokens = gen[0, prompt_len:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            correct = score(text, gold, kind)
            if args.mode == "offline":
                vw_start, vw_end = 0.0, float(decision_end)
            else:
                vw_start = float(decision_end - args.visual_window_chunks * args.agent_chunk_sec)
                vw_end = float(decision_end)
            results.append({
                "idx": i,
                "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                "sample_type": s.get("sample_type"),
                "kind": kind,
                "gold": gold,
                "video_window": [vw_start, vw_end],
                "pred": text[:300],
                "correct": correct,
            })
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{i+1}/{len(scorable)}] {rate:.2f} samples/sec")
        except Exception as e:
            print(f"Sample {i} failed: {type(e).__name__}: {e}")
            skipped += 1
            continue

    if not results:
        print("No successful runs.")
        return

    by = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        for k in (r["kind"], "_all"):
            by[k]["n"] += 1
            by[k]["correct"] += int(r["correct"])

    print()
    header = f"{'kind':<15}  {'n':>5}  {'accuracy':>10}"
    print(header)
    print("-" * len(header))
    for k in sorted(by.keys()):
        v = by[k]
        if v["n"] == 0:
            continue
        print(f"{k:<15}  {v['n']:>5}  {v['correct']/v['n']:>10.3f}")
    print()
    print(f"Skipped: {skipped} (missing video / missing question / errors)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "mode": args.mode,
                "test_jsonl": args.test_jsonl,
                "n_samples": len(results),
                "n_skipped": skipped,
                "by_kind": {k: dict(v) for k, v in by.items()},
                "samples": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
