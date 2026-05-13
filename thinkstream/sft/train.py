"""Trajectory-mixed agent SFT training entry point.

Based on Qwen3-VL official finetune, adapted for ThinkStream.
Supports Qwen2.5-VL and Qwen3-VL (including MoE variants).

Usage (production):
    bash scripts/sft_trajectory.sh
    # or directly:
    torchrun --nproc_per_node=8 thinkstream/sft/train.py \
        --model_name_or_path Qwen/Qwen3-VL-8B \
        --dataset_use stream_agent_trajectory_train \
        --output_dir output/agent-trajectory-sft
"""

import os
import logging
import pathlib
import sys
from pathlib import Path

import torch
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
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
from thinkstream.sft.data_processor import make_trajectory_data_module
from thinkstream.sft.args import ModelArguments, DataArguments, TrainingArguments
from thinkstream.data.agent_protocol import AGENT_SPECIAL_TOKENS
# Patch lce_forward to accept video_mask and build the FlexAttention
# block mask (no-op when attn_implementation != "streaming_attention").
# Importing the module triggers the patch.
from thinkstream.models import patch as _ts_models_patch  # noqa: F401
from thinkstream.models.streaming_attention import register_streaming_attention

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


def _register_agent_special_tokens(tokenizer) -> int:
    """Register canonical agent tags as indivisible special tokens."""
    existing_vocab = tokenizer.get_vocab()
    already_special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    to_add = [
        tok for tok in AGENT_SPECIAL_TOKENS
        if tok not in existing_vocab or tok not in already_special
    ]
    if not to_add:
        return 0
    return int(tokenizer.add_special_tokens({
        "additional_special_tokens": list(AGENT_SPECIAL_TOKENS),
    }))


def _resize_and_init_new_embeddings(model, old_vocab_size: int, new_vocab_size: int):
    """Resize LM embeddings and initialize newly added tags from old means."""
    in_emb = model.get_input_embeddings()
    if in_emb is None or not hasattr(in_emb, "weight"):
        return
    old_embedding_size = int(in_emb.weight.shape[0])
    target_size = max(new_vocab_size, old_embedding_size)
    if target_size > old_embedding_size:
        model.resize_token_embeddings(target_size)
        in_emb = model.get_input_embeddings()
    init_start = int(old_vocab_size)
    init_end = int(new_vocab_size)
    if init_end <= init_start:
        return
    with torch.no_grad():
        if in_emb is not None and hasattr(in_emb, "weight"):
            weight = in_emb.weight
            mean = weight[:init_start].mean(dim=0, keepdim=True)
            weight[init_start:init_end].copy_(mean)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, "weight"):
            weight = out_emb.weight
            if weight.shape[0] < target_size:
                model.resize_token_embeddings(target_size)
                out_emb = model.get_output_embeddings()
                weight = out_emb.weight
            mean = weight[:init_start].mean(dim=0, keepdim=True)
            if weight.shape[0] >= init_end:
                weight[init_start:init_end].copy_(mean)


class ProcessorSaveCallback(transformers.TrainerCallback):
    """Save Qwen VL processor assets into every Trainer checkpoint."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if not getattr(args, "should_save", True):
            return control
        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        if checkpoint_dir.is_dir():
            self.processor.save_pretrained(checkpoint_dir)
        return control


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
    attn_implementation = os.environ.get(
        "THINKSTREAM_ATTN_IMPLEMENTATION",
        attn_implementation,
    )
    # Register the FlexAttention "streaming_attention" backend before model
    # construction so HF can resolve it if the caller selects it. The
    # backend is only ATTACHED when the model's AttentionInterface routes
    # to it (i.e. the user passed attn_implementation="streaming_attention").
    register_streaming_attention()
    name_lower = model_args.model_name_or_path.lower()
    model_basename = Path(model_args.model_name_or_path.rstrip("/")).name.lower()
    model_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config_model_type = str(getattr(model_config, "model_type", "") or "").lower()
    config_arches = [
        str(a).lower() for a in (getattr(model_config, "architectures", None) or [])
    ]
    config_text = " ".join([config_model_type, *config_arches])

    if (
        "qwen3vlmoe" in config_text
        or "qwen3_vl_moe" in config_text
        or ("qwen3" in name_lower and "a" in model_basename)
    ):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif (
        "qwen3vlforconditionalgeneration" in config_text
        or "qwen3_vl" in config_text
        or "qwen3" in name_lower
    ):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif (
        "qwen2_5_vl" in config_text
        or "qwen2.5" in name_lower
        or "qwen2_5" in name_lower
    ):
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

    vision_config = getattr(model.config, "vision_config", None)
    vision_attn_implementation = os.environ.get(
        "THINKSTREAM_VISION_ATTN_IMPLEMENTATION",
        "flash_attention_2",
    )
    if attn_implementation == "streaming_attention" and vision_config is not None:
        # Match the reference SFT/GRPO setup: streaming_attention is only for
        # language tokens carrying video_block_mask; the vision tower does not
        # receive that mask and should use ordinary vision attention.
        vision_config._attn_implementation = vision_attn_implementation

    rank0_print(f"Model: {model_args.model_name_or_path} ({model.__class__.__name__})")
    rank0_print(f"Model type: {data_args.model_type}")
    rank0_print(f"Attention: {attn_implementation}")
    if vision_config is not None:
        rank0_print(
            "Vision attention: "
            f"{getattr(vision_config, '_attn_implementation', None)}"
        )

    # Video sliding-window size for streaming_attention. Aligns with
    # SLIDING_WINDOW_CHUNKS in pass5_splitter so SFT mask matches the
    # runtime engine's KV-eviction window. Stored on text config so the
    # patched lce_forward can read it via model.config.video_flex_window_size.
    if attn_implementation == "streaming_attention":
        # Imported lazily to avoid circular deps and keep the data-only
        # construction path lightweight.
        from scripts.agent_data.pass5_splitter import SLIDING_WINDOW_CHUNKS
        window_size = int(
            os.environ.get("THINKSTREAM_VIDEO_FLEX_WINDOW_SIZE", SLIDING_WINDOW_CHUNKS)
        )
        model.config.video_flex_window_size = window_size
        rank0_print(f"video_flex_window_size: {window_size}")

    # ── Processor ──
    # v12 hard-requires Qwen3-VL because Qwen2.5-VL's bundled chat_template
    # silently drops `tools=`.
    if data_args.model_type != "qwen3vl":
        raise RuntimeError(
            f"v12 protocol REQUIRES Qwen3-VL (model_type=qwen3vl); "
            f"got model_type={data_args.model_type!r}. Qwen2.5-VL's chat "
            f"template has no tools support."
        )
    processor_path = (
        os.environ.get("THINKSTREAM_PROCESSOR_PATH")
        or os.environ.get("PROCESSOR_MODEL")
        or model_args.model_name_or_path
    )
    processor = AutoProcessor.from_pretrained(
        processor_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    if processor_path != model_args.model_name_or_path:
        rank0_print(f"Processor: {processor_path}")
    old_vocab_size = len(processor.tokenizer)
    n_added = _register_agent_special_tokens(processor.tokenizer)
    new_vocab_size = len(processor.tokenizer)
    _resize_and_init_new_embeddings(model, old_vocab_size, new_vocab_size)
    rank0_print(
        "[v12] agent special tokens registered: "
        f"added={n_added}, vocab={old_vocab_size}->{new_vocab_size}, "
        f"tokens={list(AGENT_SPECIAL_TOKENS)}"
    )

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
        processor_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    _register_agent_special_tokens(tokenizer)
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
    data_module = make_trajectory_data_module(
        processor,
        data_args,
        emit_video_mask=(attn_implementation == "streaming_attention"),
    )

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
    # Stash data_args so trainer.compute_loss can read action_class_loss_mode
    # and friends (DataArguments fields). HF Trainer doesn't pass DataArguments
    # in by default; this is the conventional escape hatch.
    trainer.data_args = data_args
    trainer.add_callback(ProcessorSaveCallback(processor))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("Checkpoint found, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ── Save ──
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
        for checkpoint_dir in Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
            if checkpoint_dir.is_dir():
                processor.save_pretrained(checkpoint_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
