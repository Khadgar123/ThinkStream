"""Evaluate a checkpoint on the test set.

Usage:
    torchrun --nproc_per_node=8 thinkstream/sft/eval_test.py \
        --checkpoint_dir output/agent-sft/checkpoint-100 \
        --dataset stream_agent_test \
        --batch_size 8
"""

import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from thinkstream.sft.trainer import WeightedSFTTrainer
from thinkstream.sft.data_processor import (
    make_per_timestep_data_module,
    PerTimestepDataset,
    PerTimestepDataCollator,
)
from thinkstream.sft.argument import ModelArguments, DataArguments, TrainingArguments
from dataclasses import dataclass, field


@dataclass
class EvalArguments:
    checkpoint_dir: str = field(default="output/agent-sft/checkpoint-100")
    dataset: str = field(default="stream_agent_test")
    batch_size: int = field(default=8)
    max_samples: int = field(default=None)


def rank0_print(*args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvalArguments))
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    checkpoint_dir = eval_args.checkpoint_dir

    # Override data args for test eval
    data_args.dataset_use = eval_args.dataset
    data_args.eval_dataset_use = None
    data_args.model_type = "qwen3vl"

    rank0_print(f"Loading checkpoint from: {checkpoint_dir}")
    rank0_print(f"Evaluating on: {eval_args.dataset}")

    # Load model from checkpoint
    name_lower = checkpoint_dir.lower()
    if "qwen3" in name_lower and "moe" in name_lower:
        model_cls = Qwen3VLMoeForConditionalGeneration
    elif "qwen3" in name_lower:
        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in name_lower:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        # Default to Qwen3VL based on project usage
        model_cls = Qwen3VLForConditionalGeneration

    model = model_cls.from_pretrained(
        checkpoint_dir,
        attn_implementation="flash_attention_2",
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    processor = AutoProcessor.from_pretrained(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Build test dataset
    test_dataset = PerTimestepDataset(
        processor,
        data_args,
        dataset_use_override=eval_args.dataset,
        max_samples=eval_args.max_samples,
    )
    rank0_print(f"Test samples: {len(test_dataset)}")

    collator = PerTimestepDataCollator(tokenizer)

    # Override training args for eval-only
    training_args.per_device_eval_batch_size = eval_args.batch_size
    training_args.dataloader_num_workers = 4

    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=collator,
    )

    rank0_print("Running evaluation...")
    metrics = trainer.evaluate()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("\n=== Test Set Evaluation Results ===")
        for key in sorted(metrics.keys()):
            if key.startswith("eval/"):
                val = metrics[key]
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")
        print("====================================")


if __name__ == "__main__":
    main()
