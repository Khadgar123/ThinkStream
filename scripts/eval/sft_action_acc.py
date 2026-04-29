#!/usr/bin/env python
"""L3: generative action-accuracy eval on val.jsonl.

Unlike the in-loop teacher-forced eval (eval/action_accuracy in wandb),
this script runs real model.generate() with greedy decoding and parses
the output text. The teacher-forced metric is an OPTIMISTIC upper
bound; this script gives the inference-time accuracy you actually
deploy with.

Run after SFT completes (or on any saved checkpoint):
    python scripts/eval/sft_action_acc.py \\
        --ckpt output/agent-sft \\
        --val data/agent_v5/final/val.jsonl \\
        --n 200 \\
        --out output/agent-sft/eval_gen_action.json

Reports per-sample-type:
    - action_accuracy:  pred action keyword matches gold (silent /
      response / recall / compress)
    - post_continued:   model produced more content after </action>
      (for silent samples this should be FALSE; for response/recall/
      compress this should be TRUE)
    - silent_eos_rate:  silent samples that correctly stopped
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

from thinkstream.sft.data_processor import (
    register_special_tokens,
    build_per_timestep_messages_v12 as build_per_timestep_messages,
    update_processor_pixels,
)
from thinkstream.sft.argument import DataArguments


ACTION_RE = re.compile(r"<action>\s*([a-zA-Z_]+)\s*</action>", re.DOTALL)


def parse_output(text: str) -> dict:
    """Extract action keyword + post-action continuation flag.

    silent samples: gold = no payload after </action>
    response/recall/compress: gold = <response>/<query>/<summary> after
    """
    m = ACTION_RE.search(text)
    action = m.group(1).strip() if m else None
    after = text[m.end():] if m else ""

    has_response = "<response>" in after
    has_query = "<query>" in after
    has_summary = "<summary>" in after
    post_continued = has_response or has_query or has_summary

    return {
        "action": action,
        "post_continued": post_continued,
        "post_type": (
            "response" if has_response
            else "query" if has_query
            else "summary" if has_summary
            else "eos"
        ),
    }


def gold_action(sample: dict) -> str:
    """Extract gold action from sample's gold output."""
    out = sample.get("output", "")
    if not out and "messages" in sample:
        # v5 format: last message is assistant
        last = sample["messages"][-1]
        out = (last.get("content") or [{}])[0].get("text", "") if isinstance(last.get("content"), list) else last.get("content", "")
    m = ACTION_RE.search(out)
    return m.group(1).strip() if m else None


def load_model(ckpt: str, bf16: bool = True):
    name = ckpt.lower()
    if "qwen3" in name or "qwen-3" in name or "qwen_3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        # default to Qwen3-VL — most SFT runs use it
        from transformers import Qwen3VLForConditionalGeneration as Cls

    print(f"Loading {Cls.__name__} from {ckpt} ...")
    model = Cls.from_pretrained(
        ckpt,
        dtype=torch.bfloat16 if bf16 else None,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="SFT checkpoint dir")
    p.add_argument("--val", required=True, help="Path to val.jsonl")
    p.add_argument("--n", type=int, default=200, help="Max samples to evaluate")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--out", default=None, help="Output JSON path")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    model = load_model(args.ckpt, bf16=not args.no_bf16)
    processor = AutoProcessor.from_pretrained(args.ckpt)
    # Special tokens are usually saved with the SFT ckpt, but harmless to
    # ensure registration so missing ones don't blow up tokenization.
    register_special_tokens(processor, "qwen3vl")
    # Mirror SFT pixel/fps config so visual features match training distribution.
    processor = update_processor_pixels(processor, DataArguments())

    samples = []
    with open(args.val) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break
    print(f"Loaded {len(samples)} samples from {args.val}")

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id or eos_id

    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            base_path = Path(s.get("data_path") or ".")
            if "messages" in s:
                # Drop assistant turn so the model has to produce it
                msgs = s["messages"][:-1]
            else:
                full = build_per_timestep_messages(s, base_path)
                msgs = full[:-1]

            inputs = processor.apply_chat_template(
                msgs,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,         # greedy: deterministic, matches argmax semantics
                    pad_token_id=pad_id,
                )
            new_tokens = gen[0, prompt_len:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)

            pred = parse_output(text)
            ga = gold_action(s)

            results.append({
                "idx": i,
                "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                "sample_type": s.get("sample_type", "?"),
                "gold_action": ga,
                "pred_action": pred["action"],
                "action_correct": pred["action"] == ga and pred["action"] is not None,
                "post_continued": pred["post_continued"],
                "post_type": pred["post_type"],
                "raw_pred": text[:600],
            })

            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{i+1}/{len(samples)}] {rate:.2f} samples/sec")
        except Exception as e:
            print(f"Sample {i} failed: {type(e).__name__}: {e}")
            continue

    if not results:
        print("No successful generations.")
        return

    # Aggregate per sample_type + grand total
    by = defaultdict(lambda: {"n": 0, "action_correct": 0, "post_continued": 0})
    for r in results:
        for key in (r["sample_type"], "_all"):
            by[key]["n"] += 1
            by[key]["action_correct"] += int(r["action_correct"])
            by[key]["post_continued"] += int(r["post_continued"])

    print()
    header = f"{'sample_type':<25}  {'n':>5}  {'action_acc':>10}  {'post_continued':>14}"
    print(header)
    print("-" * len(header))
    for st in sorted(by.keys()):
        b = by[st]
        if b["n"] == 0:
            continue
        aa = b["action_correct"] / b["n"]
        pc = b["post_continued"] / b["n"]
        print(f"{st:<25}  {b['n']:>5}  {aa:>10.3f}  {pc:>14.3f}")

    sil = by.get("silent")
    if sil and sil["n"] > 0:
        eos_rate = 1 - sil["post_continued"] / sil["n"]
        wrong = sil["post_continued"]
        print(f"\nsilent_eos_rate (gen): {eos_rate:.3f}  ({sil['n']-wrong}/{sil['n']} stopped correctly)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "ckpt": args.ckpt,
                    "val": args.val,
                    "n_samples": len(results),
                    "by_sample_type": {k: dict(v) for k, v in by.items()},
                    "samples": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
