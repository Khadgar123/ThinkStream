"""Per-timestep agent SFT training entry point.

Based on Qwen3-VL official finetune, adapted for ThinkStream.
Supports Qwen2.5-VL and Qwen3-VL (including MoE variants).

Usage (production):
    PHASE=mixed bash scripts/sft_per_timestep.sh
    # or directly:
    torchrun --nproc_per_node=8 thinkstream/sft/train.py \
        --model_name_or_path Qwen/Qwen3-VL-8B \
        --dataset_use stream_agent_p5 \
        --output_dir output/agent-mixed
"""

import os
import logging
import pathlib
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

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from thinkstream.sft.trainer import WeightedSFTTrainer
from thinkstream.sft.data_processor import (
    make_per_timestep_data_module,
    register_special_tokens,
    smart_init_special_token_embeddings,
    SPECIAL_TOKENS_AGENT,
)
from thinkstream.sft.argument import ModelArguments, DataArguments, TrainingArguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    """Configure which components are trainable."""
    for n, p in model.visual.named_parameters():
        p.requires_grad = model_args.tune_mm_vision

    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = model_args.tune_mm_mlp

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ── Load model (auto-detect: Qwen2.5-VL or Qwen3-VL) ──
    name_lower = model_args.model_name_or_path.lower()
    model_basename = Path(model_args.model_name_or_path.rstrip("/")).name.lower()

    if "qwen3" in name_lower and "a" in model_basename:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in name_lower:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in name_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError(
            f"Unsupported model: {model_args.model_name_or_path}. "
            f"Only Qwen2.5-VL and Qwen3-VL are supported."
        )

    rank0_print(f"Model: {model_args.model_name_or_path} ({model.__class__.__name__})")
    rank0_print(f"Model type: {data_args.model_type}")

    # ── Processor + special tokens ──
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    register_special_tokens(processor, data_args.model_type)

    # Resize embeddings for added special tokens (must be before DeepSpeed init)
    model.resize_token_embeddings(len(processor.tokenizer))

    # v11.4: smart-init the new special-token embeddings from natural-word
    # equivalents. Without this, HF's default mean-init produces tiny-
    # magnitude embeddings that lose at sampling time to well-trained
    # natural English ("response" the word beats <action> the structural
    # token at logit comparison). The first v11.3 SFT run produced
    # `</think>responseThe video...` in free generation — root cause was
    # this cold-start. See docs/v11.3_sft_run_postmortem.md.
    if data_args.model_type == "qwen3vl":
        _agent_tags = [t for t in SPECIAL_TOKENS_AGENT
                       if t not in ("<think>", "</think>")]
    else:
        _agent_tags = list(SPECIAL_TOKENS_AGENT)
    smart_init_special_token_embeddings(model, processor, _agent_tags)

    model.config.use_cache = False

    # ── Gradient checkpointing ──
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    # ── Tokenizer (separate from processor, needed by Trainer for saving) ──
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Sync special tokens added to processor's tokenizer
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    # ── Trainable parameters ──
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType

        rank0_print("LoRA enabled")
        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    # ── Data module ──
    data_module = make_per_timestep_data_module(processor, data_args)

    rank0_print(f"Train samples: {len(data_module['train_dataset'])}")
    if data_module.get("eval_dataset") is not None:
        rank0_print(f"Eval samples: {len(data_module['eval_dataset'])}")

    # ── Train ──
    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("Checkpoint found, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ── Save ──
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
