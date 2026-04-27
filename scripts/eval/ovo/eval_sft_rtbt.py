"""Single-step agent-protocol eval for RT/BT MCQ tasks.

Matches SFT training distribution: each sample is one independent forward pass
with pre-extracted frames and agent-protocol message format.

No streaming loop, no context accumulation, no compression drift.

Usage:
    python scripts/eval/ovo/eval_sft_rtbt.py \
        --ckpt output/agent-sft/checkpoint-100 \
        --benchmark_json /path/to/ovo_bench_new.json \
        --video_root /path/to/videos \
        --frames_root /path/to/frames \
        [--tasks EPM,ASI,HLD,OCR,ACR,ATR,STU,FPD,OJR]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor, AutoTokenizer

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    SYSTEM_PROMPT,
    VISUAL_WINDOW_CHUNKS,
    build_user_content,
)
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import (
    register_special_tokens,
    update_processor_pixels,
)
from thinkstream.model.agent_loop import make_generate_fn


def make_loop(model, processor, model_type):
    return make_generate_fn(model, processor, model_type=model_type)


def _get_frame_paths(video_path, chunk_idx, frames_root, video_root):
    """Build frame_paths for the current visual_window from pre-extracted frames."""
    if not frames_root:
        return None
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC

    vp = Path(video_path)
    if video_root:
        try:
            rel = vp.relative_to(Path(video_root))
            stem = rel.with_suffix("")
            frame_dir = Path(frames_root) / stem
        except ValueError:
            frame_dir = Path(frames_root) / vp.with_suffix("")
    else:
        frame_dir = Path(frames_root) / vp.with_suffix("")

    if not frame_dir.exists():
        return None

    start_frame = int(video_start) + 1
    end_frame = int(video_end) + 1
    frame_paths = []
    for i in range(start_frame, end_frame + 1):
        fp = frame_dir / f"frame_{i:06d}.jpg"
        if fp.exists():
            frame_paths.append(str(fp))

    n_expected = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK
    if len(frame_paths) < max(1, n_expected // 2):
        return None
    return frame_paths


def _build_mcq_question(sample):
    """Build question text matching SFT training distribution.

    SFT data uses open-ended questions without option lists.
    We keep the raw question and do text→option matching at parse time.
    """
    return sample["question"]


def _extract_letter(text, options=None):
    """Extract predicted choice from model output.

    First tries to find a single letter A-E. If that fails and options are
    provided, performs text matching against option strings (the model was
    trained on open-ended answers, not MCQ letters).
    """
    if not text:
        return None
    t = text.strip()
    # 1) Direct letter match
    if t and t[0].upper() in "ABCDE" and len(t) == 1:
        return t[0].upper()
    import re
    m = re.search(r"\b([A-Ea-e])\b", t)
    if m:
        return m.group(1).upper()

    # 2) Text matching against options (training distribution is open-ended)
    if options:
        t_lower = t.lower()
        # Exact or substring match: option text in output or output in option text
        best_idx = None
        best_score = 0
        for idx, opt in enumerate(options):
            opt_lower = opt.lower()
            if t_lower == opt_lower:
                return chr(65 + idx)
            if opt_lower in t_lower or t_lower in opt_lower:
                # Prefer longer matches to avoid false positives on short words
                score = len(opt_lower) if opt_lower in t_lower else len(t_lower)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is not None:
            return chr(65 + best_idx)

    return None


RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
BT_TASKS = {"EPM", "ASI", "HLD"}
ALL_MCQ = RT_TASKS | BT_TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default=None)
    p.add_argument("--tasks", default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # detect model class (copied from eval_full.py)
    name = args.ckpt.lower()
    basename = Path(args.ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        model_type = "qwen3vl"
    elif "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        model_type = "qwen3vl"
    elif "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        model_type = "qwen2.5vl"
    else:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        model_type = "qwen3vl"
    print(f"Loading {Cls.__name__} from {args.ckpt} ...")
    try:
        import flash_attn
        _has_fa = True
    except ImportError:
        _has_fa = False
    model_kwargs = {}
    if not args.no_bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if _has_fa:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Cls.from_pretrained(args.ckpt, **model_kwargs)
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

    generate_fn = make_loop(model, processor, model_type)

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)

    task_filter = set(t.strip() for t in args.tasks.split(",")) if args.tasks else ALL_MCQ

    samples = [s for s in all_samples if s.get("task") in task_filter]
    print(f"Evaluating {len(samples)} RT/BT samples across tasks: {sorted(set(s['task'] for s in samples))}")

    results = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        video_path = str(Path(args.video_root) / sample["video"])
        realtime = float(sample["realtime"])
        ask_chunk = int(realtime / AGENT_CHUNK_SEC)

        frame_paths = _get_frame_paths(video_path, ask_chunk, args.frames_root, args.video_root)

        question = _build_mcq_question(sample)
        user_content = build_user_content(
            memory_text="",
            chunk_idx=ask_chunk,
            video_path=video_path,
            user_input=question,
            queries=None,
            min_pixels=100352 * 2,
            max_pixels=100352 * 4,
            frame_paths=frame_paths,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

        try:
            output_text = generate_fn(
                messages=messages,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"[{idx}] error: {e}")
            output_text = ""

        # Parse response (try agent protocol first, then fallback)
        import re
        resp_match = re.search(r"<response>(.*?)</response>", output_text, re.DOTALL)
        if resp_match:
            resp_text = resp_match.group(1).strip()
        else:
            # Fallback: take text after </think> or last think
            think_end = output_text.rfind("</think>")
            if think_end >= 0:
                resp_text = output_text[think_end + len("</think>"):].strip()
            else:
                resp_text = output_text.strip()

        pred_letter = _extract_letter(resp_text, options=sample.get("options"))
        gt_idx = sample.get("gt", 0)
        gt_letter = chr(65 + gt_idx)
        correct = pred_letter == gt_letter

        results.append({
            "task": sample["task"],
            "id": sample.get("id"),
            "realtime": realtime,
            "ask_chunk": ask_chunk,
            "gt": gt_letter,
            "pred": pred_letter,
            "correct": correct,
            "raw_output": output_text,
            "parsed_response": resp_text,
        })

        if (idx + 1) % 10 == 0:
            rate = (idx + 1) / max(1e-6, time.time() - t0)
            print(f"[{idx + 1}/{len(samples)}] {rate * 60:.1f} samples/min")

    # Aggregate
    by_task = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        by_task[r["task"]]["n"] += 1
        by_task[r["task"]]["correct"] += int(r["correct"])

    rt_accs = []
    bt_accs = []
    print("\nPer-task results:")
    for task in sorted(by_task):
        v = by_task[task]
        acc = v["correct"] / v["n"]
        print(f"  {task}: {v['correct']}/{v['n']} = {acc:.3f}")
        if task in RT_TASKS:
            rt_accs.append(acc)
        elif task in BT_TASKS:
            bt_accs.append(acc)

    print(f"\nRT avg: {sum(rt_accs) / len(rt_accs) if rt_accs else 0:.3f}")
    print(f"BT avg: {sum(bt_accs) / len(bt_accs) if bt_accs else 0:.3f}")

    out_path = args.out or f"{args.ckpt}/eval/ovo_rtbt_single_step.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "tasks_evaluated": sorted(task_filter),
            "n_samples": len(results),
            "rt_avg": sum(rt_accs) / len(rt_accs) if rt_accs else 0,
            "bt_avg": sum(bt_accs) / len(bt_accs) if bt_accs else 0,
            "by_task": dict(by_task),
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
