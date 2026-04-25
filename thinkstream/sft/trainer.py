"""Per-timestep agent SFT trainer.

Only keeps what we actually use:
- WeightedSFTTrainer: per-sample loss weighting by action type
- create_optimizer: per-component learning rate (vision_tower_lr, mm_projector_lr)
- print_trainable_parameters: debugging utility

Removed from Qwen3-VL official finetune:
- flash_attention_forward / varlen attention (data_flatten=False, not needed)
- qwen2vl_forward (Qwen2-VL not supported)
- replace_qwen2_vl_attention_class (only for data_flatten mode)
"""

import torch
from transformers import Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Per-sample weighted loss Trainer
# ---------------------------------------------------------------------------

class WeightedSFTTrainer(Trainer):
    """Trainer that supports per-sample loss weighting.

    If the batch contains 'sample_weights', weight per-sample loss
    before reduction. Otherwise fall back to standard loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        token_loss_weight = inputs.pop("token_loss_weight", None)

        outputs = model(**inputs)

        if (sample_weights is None or sample_weights.numel() == 0) and token_loss_weight is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            B, L, V = shift_logits.shape
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(B * L, V), shift_labels.view(B * L)
            ).view(B, L)

            valid_mask = (shift_labels != IGNORE_INDEX).float()

            # v9: per-token weight (span-based: think/action/response/query/summary)
            if token_loss_weight is not None:
                tw = token_loss_weight[..., 1:].contiguous().to(per_token_loss.device)
                effective_mask = valid_mask * tw
            else:
                effective_mask = valid_mask

            per_sample_loss = (
                (per_token_loss * effective_mask).sum(dim=1)
                / effective_mask.sum(dim=1).clamp(min=1)
            )

            if sample_weights is not None and sample_weights.numel() > 0:
                sample_weights = sample_weights.to(per_sample_loss.device)
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            else:
                loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Per-component learning rate optimizer
# ---------------------------------------------------------------------------

def create_optimizer(self):
    """Create optimizer with optional per-component learning rates.

    Supports:
    - vision_tower_lr: separate LR for ViT parameters
    - mm_projector_lr: separate LR for merger/projector parameters
    - Standard weight decay grouping (bias excluded)

    If neither vision_tower_lr nor mm_projector_lr is set, falls back
    to standard 2-group optimizer (decay vs no-decay).
    """
    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        if self.args.mm_projector_lr and self.args.vision_tower_lr:
            # 3 component groups × 2 (decay/no-decay) = 6 groups
            projector_params = [n for n, _ in opt_model.named_parameters() if "merger" in n]
            vision_params = [n for n, _ in opt_model.named_parameters() if "visual" in n]

            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n not in projector_params
                            and n not in vision_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n not in projector_params
                            and n not in vision_params and p.requires_grad],
                 "weight_decay": 0.0},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in vision_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": self.args.vision_tower_lr},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in vision_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": self.args.vision_tower_lr},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in projector_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": self.args.mm_projector_lr},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in projector_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": self.args.mm_projector_lr},
            ]
        elif self.args.mm_projector_lr:
            projector_params = [n for n, _ in opt_model.named_parameters() if "merger" in n]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n not in projector_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n not in projector_params and p.requires_grad],
                 "weight_decay": 0.0},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in projector_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": self.args.mm_projector_lr},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in projector_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": self.args.mm_projector_lr},
            ]
        else:
            # Standard: decay vs no-decay
            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and p.requires_grad],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and p.requires_grad],
                 "weight_decay": 0.0},
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# ---------------------------------------------------------------------------
# Debugging utilities
# ---------------------------------------------------------------------------

def _print_trainable_visual(self) -> None:
    trainable = [i for i, b in enumerate(self.blocks) if all(p.requires_grad for p in b.parameters())]
    frozen = [i for i, b in enumerate(self.blocks) if not all(p.requires_grad for p in b.parameters())]
    merger_trainable = any(p.requires_grad for p in self.merger.parameters())
    print(f"Vision: trainable blocks={trainable or 'None'}, frozen={frozen or 'None'}, merger={merger_trainable}")


def _print_trainable_llm(self) -> None:
    trainable = [i for i, l in enumerate(self.language_model.layers) if any(p.requires_grad for p in l.parameters())]
    embed = any(p.requires_grad for p in self.language_model.embed_tokens.parameters())
    print(f"LLM: trainable layers={len(trainable)}/{len(list(self.language_model.layers))}, embed={embed}")


# ---------------------------------------------------------------------------
# Apply monkey patches
# ---------------------------------------------------------------------------

Trainer.create_optimizer = create_optimizer

Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = _print_trainable_visual
Qwen2_5_VLModel.print_trainable_parameters = _print_trainable_llm
Qwen3VLVisionModel.print_trainable_parameters = _print_trainable_visual
Qwen3VLModel.print_trainable_parameters = _print_trainable_llm
Qwen3VLMoeVisionModel.print_trainable_parameters = _print_trainable_visual
Qwen3VLMoeModel.print_trainable_parameters = _print_trainable_llm
