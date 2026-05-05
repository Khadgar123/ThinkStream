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
    - action_accuracy:  v12 terminal action matches gold (silent /
      response / recall / compress). v12 maps <answer></answer> to silent,
      non-empty <answer> to response, and tool calls to recall/compress.
    - post_continued:   non-silent terminal emitted
    - silent_eos_rate:  silent samples that correctly stopped
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from thinkstream.sft.data_processor import (
    _resolve_video_paths,
    update_processor_pixels,
)
from thinkstream.sft.argument import DataArguments
from scripts.agent_data_v5.pass5_messages import (
    build_messages as build_per_timestep_messages,
)
from thinkstream.data.agent_protocol import (
    normalize_frame_protocol,
    parse_agent_output_v12,
)
from scripts.eval.processor_loader import load_processor_for_checkpoint


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
                    meta = {k: v for k, v in meta.items() if k != "do_sample_frames"}
                    metas.append(meta)
    return metas


def parse_output(text: str) -> dict:
    """Extract the v12 terminal action from generated text."""
    parsed = parse_agent_output_v12(text)
    kind = parsed.get("kind")
    if kind == "answer":
        action = "response" if (parsed.get("answer_text") or "").strip() else "silent"
    elif kind in {"recall", "compress"}:
        action = kind
    else:
        action = None

    return {
        "action": action,
        "post_continued": action not in (None, "silent"),
        "post_type": action or "format_error",
        "format_error": parsed.get("format_error"),
    }


def gold_action(sample: dict) -> str:
    """Extract gold action from sample's gold output."""
    out = sample.get("output", "")
    if not out and "messages" in sample:
        # v5 format: last message is assistant
        last = sample["messages"][-1]
        out = (last.get("content") or [{}])[0].get("text", "") if isinstance(last.get("content"), list) else last.get("content", "")
    return parse_output(out)["action"]


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
    p.add_argument(
        "--frame-protocol",
        default=None,
        choices=["ts_image", "video_meta"],
        help="Used only when --val is a flat row file rendered on the fly.",
    )
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()
    frame_protocol = normalize_frame_protocol(args.frame_protocol)

    model = load_model(args.ckpt, bf16=not args.no_bf16)
    processor = load_processor_for_checkpoint(args.ckpt)
    # Special tokens are usually saved with the SFT ckpt, but harmless to
    # ensure registration so missing ones don't blow up tokenization.
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
                msgs = _resolve_video_paths(s["messages"][:-1], base_path)
            else:
                full = build_per_timestep_messages(
                    s, base_path, frame_protocol=frame_protocol
                )
                msgs = full[:-1]

            template_kwargs = dict(
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                do_sample_frames=False,
            )
            video_metadata = collect_video_metadata(msgs)
            if video_metadata:
                template_kwargs["video_metadata"] = video_metadata
            inputs = processor.apply_chat_template(msgs, **template_kwargs)
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
                    "frame_protocol": frame_protocol,
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
