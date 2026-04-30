#!/usr/bin/env python
"""Run trainer.evaluate() on the held-out test set with a saved SFT ckpt.

Reuses the WeightedSFTTrainer eval path, so the metrics match wandb's
in-loop eval byte-for-byte:
    eval_loss              token-span weighted loss (same formula as train)
    eval/action_accuracy   teacher-forced argmax of action keyword
    eval/action_acc_<cls>  per sample_type breakdown
    eval/post_action_acc_<cls>  transition token correctness
    eval/silent_eos_rate   silent samples that correctly stop after </action>

This is the L1+L2 measurement on test.jsonl. For the L3 generative
counterpart (real model.generate, parses <action> from output),
pair this with `scripts/eval/sft_action_acc.py --val test.jsonl`.

Single-GPU usage:
    python scripts/eval/test_set_eval.py \\
        --ckpt output/agent-sft \\
        --dataset stream_agent_test

Multi-GPU (recommended for 1,600-sample test set):
    torchrun --nproc_per_node=8 scripts/eval/test_set_eval.py \\
        --ckpt output/agent-sft \\
        --dataset stream_agent_test
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from thinkstream.sft.argument import DataArguments, TrainingArguments
from thinkstream.sft.data_processor import (
    PerTimestepDataset,
    PerTimestepDataCollator,
    update_processor_pixels,
)
from thinkstream.sft.trainer import WeightedSFTTrainer


def detect_model_class(ckpt_path: str):
    """Pick Qwen3-VL / Qwen3-VL-MoE / Qwen2.5-VL based on ckpt path."""
    name = ckpt_path.lower()
    basename = Path(ckpt_path.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        return Qwen3VLMoeForConditionalGeneration, "qwen3vl"
    if "qwen3" in name:
        return Qwen3VLForConditionalGeneration, "qwen3vl"
    if "qwen2.5" in name or "qwen-2.5" in name:
        return Qwen2_5_VLForConditionalGeneration, "qwen2.5vl"
    # Default to Qwen3-VL — most ThinkStream SFT runs use it.
    return Qwen3VLForConditionalGeneration, "qwen3vl"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="SFT checkpoint dir")
    p.add_argument(
        "--dataset",
        default="stream_agent_test",
        help="Dataset name from data_list registry. Default: stream_agent_test (1,600 video-disjoint samples).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on test set size for a quick smoke test.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_sample_tokens", type=int, default=12000)
    p.add_argument("--output_json", default=None, help="Where to dump metrics dict")
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    Cls, model_type = detect_model_class(args.ckpt)
    bf16 = not args.no_bf16

    rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    if is_main:
        print(f"Loading {Cls.__name__} from {args.ckpt} ...")

    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if bf16 else None,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt,
        model_max_length=16384,
        padding_side="right",
        use_fast=False,
    )
    # Keep tokenizer in sync with processor's added vocab
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    # Mirror SFT default DataArguments — visual pixels, fps, etc.
    data_args = DataArguments(
        dataset_use="",
        model_type=model_type,
        max_sample_tokens=args.max_sample_tokens,
    )
    processor = update_processor_pixels(processor, data_args)

    # Build the test PerTimestepDataset directly (bypasses make_per_timestep_data_module)
    if is_main:
        print(f"Building test dataset from registry name: {args.dataset}")
    test_dataset = PerTimestepDataset(
        processor,
        data_args,
        dataset_use_override=args.dataset,
        max_samples=args.max_samples,
    )

    output_dir = Path(args.ckpt) / "eval" / f"test_{args.dataset}"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=args.batch_size,
        bf16=bf16,
        report_to="none",
        model_max_length=16384,
        do_train=False,
        do_eval=True,
    )

    collator = PerTimestepDataCollator(tokenizer)
    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        data_collator=collator,
    )

    if is_main:
        print(f"Running eval on {len(test_dataset)} samples ...")
    metrics = trainer.evaluate()

    if is_main:
        print("\n=== Test set metrics ===")
        for k in sorted(metrics):
            v = metrics[k]
            if isinstance(v, (int, float)):
                print(f"  {k:<45s}  {float(v):.4f}")
            else:
                print(f"  {k:<45s}  {v}")

        out_path = Path(args.output_json) if args.output_json else (output_dir / "metrics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in metrics.items()}, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
