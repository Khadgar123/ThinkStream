#!/usr/bin/env python
"""Generative eval of SFT checkpoint on test set — answer correctness.

Loads an SFT checkpoint, runs model.generate() on each test sample,
extracts the v12 <answer> payload from the generated text, and scores it
against the gold answer with the same form-aware matcher as RL/eval
(binary, MC letter/full text, number, short exact, descriptive).

This is the L3 (generative) counterpart to test_set_eval.py's L1+L2
(teacher-forced action accuracy). It answers: "when the model actually
generates outputs, does the <response> content match the gold answer?"

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_set_sft_gen.py \\
        --ckpt output/agent-sft/checkpoint-100 \\
        --test_jsonl data/agent_v5/final/test.jsonl \\
        --n 0
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
from transformers import AutoTokenizer

from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import (
    _resolve_video_paths,
    update_processor_pixels,
)
from scripts.agent_data_v5.pass5_messages import (
    build_messages as build_per_timestep_messages,
)
from thinkstream.data.agent_protocol import (
    normalize_frame_protocol,
    normalize_render_layout,
    tools_for_turn,
)
from thinkstream.trainer.outcome_match import score_outcome_by_form
from scripts.eval.processor_loader import load_processor_for_checkpoint

GOLD_RE = re.compile(
    r"<(?P<tag>answer|response)>(?P<answer>.*?)</(?P=tag)>", re.DOTALL
)
RESPONSE_RE = re.compile(
    r"<(?P<tag>answer|response)>(?P<answer>.*?)</(?P=tag)>", re.DOTALL
)


def collect_video_metadata(messages):
    metas = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                meta = item.get("video_metadata")
                if isinstance(meta, dict):
                    metas.append({
                        k: v for k, v in meta.items()
                        if k != "do_sample_frames"
                    })
    return metas


def _messages_text(messages):
    parts = []
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _tool_mode_for_prompt(sample, messages):
    mode = sample.get("tool_schema_mode")
    if mode:
        return str(mode)
    if sample.get("v12_inter_chunk"):
        return "compress"
    if any(m.get("role") == "assistant" for m in messages) and "<recall_result>" in _messages_text(messages):
        return "recall_response"
    return "streaming"


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


def extract_gold(sample):
    out = sample.get("output", "")
    if not out and "messages" in sample:
        for msg in reversed(sample.get("messages") or []):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                out = "".join(
                    str(item.get("text") or "")
                    for item in content
                    if isinstance(item, dict)
                )
            elif isinstance(content, str):
                out = content
            if out:
                break
    m = GOLD_RE.search(out)
    if not m:
        return None
    answer = m.group("answer").strip()
    return answer or None


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


def score(pred_text, gold, kind):
    if not pred_text or not gold:
        return False
    pred = pred_text.strip()
    if kind == "yes_no":
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
    return score_outcome_by_form(
        pred,
        gold_answer=gold,
        answer_form="descriptive",
    ) >= 0.5


def _sample_answer_form(sample, kind):
    meta = sample.get("metadata") or {}
    form = sample.get("answer_form") or meta.get("answer_form") or ""
    if form:
        return form
    return {
        "yes_no": "binary",
        "int": "number",
        "letter": "multiple_choice",
        "descriptive": "descriptive",
    }.get(kind, "descriptive")


def score_sample(pred_text, sample, gold, kind):
    if not pred_text or not gold:
        return False
    meta = sample.get("metadata") or {}
    answer_form = _sample_answer_form(sample, kind)
    options = sample.get("options") or meta.get("options") or []
    correct_option = (
        sample.get("correct_option")
        if sample.get("correct_option") is not None
        else meta.get("correct_option", "")
    )
    if answer_form == "multiple_choice" and not options and kind == "letter":
        return score(pred_text, gold, kind)
    return score_outcome_by_form(
        pred_text,
        options=options,
        correct_option=correct_option,
        gold_answer=gold,
        answer_form=answer_form,
    ) >= 0.5


def extract_response_from_generation(text):
    m = RESPONSE_RE.search(text)
    if m:
        return m.group("answer").strip()
    # Fallback for models that omit <action>/<response> tags and output
    # <think>...</think>responseNo  or  <think>...</think>response 12
    think_end = text.rfind("</think>")
    if think_end >= 0:
        after = text[think_end + len("</think>"):].strip()
        # Look for explicit action + response tags even if earlier regex missed
        am = re.search(r"<action>(.*?)</action>", after)
        if am:
            if am.group(1).strip() == "response":
                rm = re.search(r"<response>(.*?)</response>", after)
                if rm:
                    return rm.group(1).strip()
        # Simplified format: "response" + answer directly after think
        if after.lower().startswith("response"):
            return after[len("response"):].strip()
    return text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--n", type=int, default=200, help="Max samples to evaluate (0 = all)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--frame-protocol",
        default=None,
        choices=["ts_image", "video_meta"],
        help="Used only when --test_jsonl is a flat row file rendered on the fly.",
    )
    p.add_argument(
        "--render-layout",
        default=None,
        help="Used only when --test_jsonl is a flat row file rendered on the fly.",
    )
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    render_layout = normalize_render_layout(args.render_layout)

    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()

    processor = load_processor_for_checkpoint(args.ckpt)
    data_args = DataArguments(dataset_use="", model_type=model_type, max_sample_tokens=12000)
    processor = update_processor_pixels(processor, data_args)
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # Load and filter samples
    samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))

    scorable = []
    for s in samples:
        if s.get("sample_type") not in ("response", "recall_response", "recall"):
            continue
        gold = extract_gold(s)
        kind = gold_kind(gold)
        if gold is not None and kind is not None:
            scorable.append(s)

    if args.n and 0 < args.n < len(scorable):
        scorable = scorable[: args.n]
    print(f"Filtered {len(scorable)} form-aware scorable samples")

    root_path = Path(args.test_jsonl).resolve().parent.parent.parent.parent
    results = []
    skipped = 0
    t0 = time.time()

    for i, s in enumerate(scorable):
        try:
            # Build messages from input only (no assistant output). Prefer the
            # pass5 LLaMA-Factory/DeepEyes messages already stored in current
            # datasets; fall back to the canonical pass5 builder for flat rows.
            messages = (
                _resolve_video_paths(s["messages"], root_path)
                if "messages" in s
                else build_per_timestep_messages(
                    s,
                    root_path,
                    frame_protocol=frame_protocol,
                    render_layout=render_layout,
                )
            )
            # Drop only the target assistant. For recall_answer rows the
            # previous assistant recall tool_call is part of the runtime prompt.
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]

            template_kwargs = dict(
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                do_sample_frames=False,
            )
            video_metadata = collect_video_metadata(messages)
            if video_metadata:
                template_kwargs["video_metadata"] = video_metadata
            tools = tools_for_turn(_tool_mode_for_prompt(s, messages))
            if tools is not None:
                template_kwargs["tools"] = tools
            inputs = processor.apply_chat_template(messages, **template_kwargs)
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

            pred_response = extract_response_from_generation(text)
            gold = extract_gold(s)
            kind = gold_kind(gold)
            answer_form = _sample_answer_form(s, kind)
            correct = score_sample(pred_response, s, gold, kind)

            results.append({
                "idx": i,
                "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                "sample_type": s.get("sample_type"),
                "kind": kind,
                "answer_form": answer_form,
                "gold": gold,
                "pred_raw": text[:500],
                "pred_response": pred_response[:300],
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
    print(f"Skipped: {skipped} (errors)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "test_jsonl": args.test_jsonl,
                "frame_protocol": frame_protocol,
                "n_samples": len(results),
                "n_skipped": skipped,
                "by_kind": {k: dict(v) for k, v in by.items()},
                "samples": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
