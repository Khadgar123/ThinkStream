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
import torch.distributed as dist
from collections import defaultdict
from pathlib import Path
from typing import Dict
from transformers import Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)

from thinkstream.trainer.audit import AuditWriter, resolve_audit_dir
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit_step = self._init_audit_writers()
        self._reset_eval_accumulator()

    def _init_audit_writers(self):
        """Open <audit_dir>/sft_step.jsonl + sft_sample.jsonl. Rank-0 only."""
        audit_dir = resolve_audit_dir(
            getattr(self.args, "audit_log_dir", None),
            self.args.output_dir,
        )
        if audit_dir is None:
            self._audit_step_writer = None
            self._audit_sample_writer = None
            return None
        self._audit_step_writer = AuditWriter(audit_dir / "sft_step.jsonl")
        self._audit_sample_writer = AuditWriter(audit_dir / "sft_sample.jsonl")
        self._audit_every = max(1, int(getattr(self.args, "audit_log_every", 1)))
        return 0

    def _get_train_sampler(self, train_dataset=None):
        """Override default sampler to enable class-balanced sampling.

        With --class_balanced_sampler (default True), build a distributed
        WeightedRandomSampler so action types like recall/compress/response
        are not drowned by silent (~70% of mixed-phase data).
        """
        ds = train_dataset if train_dataset is not None else self.train_dataset
        use_balanced = getattr(self.args, "class_balanced_sampler", False)
        if ds is None or not use_balanced:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        from thinkstream.sft.data_processor import ClassBalancedDistributedSampler
        try:
            sample_types = [s.get("sample_type", "silent") for s in ds.samples]
        except AttributeError:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        sampler = ClassBalancedDistributedSampler(
            sample_types=sample_types,
            num_samples=len(sample_types),
            seed=self.args.seed,
            smoothing=getattr(self.args, "class_balance_smoothing", 0.7),
        )
        if self._audit_step_writer is not None:
            try:
                self._audit_step_writer.write({
                    "event": "class_balanced_sampler_init",
                    "n_total": len(sample_types),
                    "per_rank": len(sampler),
                    "world_size": sampler.world_size,
                    "class_weights": {
                        k: round(v, 3)
                        for k, v in sampler._cls_weights_summary.items()
                    },
                })
            except Exception:
                pass
        return sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        token_loss_weight = inputs.pop("token_loss_weight", None)
        sample_meta = inputs.pop("sample_meta", None)
        eval_meta = inputs.pop("eval_meta", None)
        # Keep input_ids handy for argmax-vs-gold accumulation during eval
        eval_input_ids = inputs["input_ids"] if not self.model.training else None

        outputs = model(**inputs)

        per_sample_loss_for_audit = None
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
            per_sample_loss_for_audit = per_sample_loss.detach()

            if sample_weights is not None and sample_weights.numel() > 0:
                sample_weights = sample_weights.to(per_sample_loss.device)
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            else:
                loss = per_sample_loss.mean()

        # ── Eval-time accuracy accumulation (teacher-forced argmax) ──
        # Done before audit because audit guard requires model.training=True
        # and this branch is the eval path.
        if not self.model.training and eval_meta is not None and eval_input_ids is not None:
            try:
                self._accumulate_eval_argmax(outputs.logits, eval_input_ids, eval_meta)
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).debug("eval argmax accum skipped: %s", e)

        # ── Audit log: per-step aggregate + per-sample breakdown ──
        # Skip during eval — model.training=False means HF Trainer is in
        # evaluate(); writing those rows would interleave eval-loss into
        # the train audit stream and double-count _audit_step.
        if self._audit_step_writer is not None and self.model.training:
            try:
                self._write_sft_audit(
                    loss=loss,
                    per_sample_loss=per_sample_loss_for_audit,
                    sample_weights=sample_weights,
                    token_loss_weight=token_loss_weight,
                    labels=inputs.get("labels"),
                    sample_meta=sample_meta,
                )
            except Exception as e:
                # Never let audit logging break training
                import logging as _logging
                _logging.getLogger(__name__).debug("audit log skipped: %s", e)

        return (loss, outputs) if return_outputs else loss

    def _write_sft_audit(
        self, *, loss, per_sample_loss, sample_weights, token_loss_weight,
        labels, sample_meta,
    ) -> None:
        if self._audit_step is None:
            return
        self._audit_step += 1
        if self._audit_step % self._audit_every != 0:
            return

        step_record: dict = {
            "step": self.state.global_step,
            "audit_step": self._audit_step,
            "epoch": self.state.epoch,
            "lr": self._current_lr(),
            "loss": float(loss.detach().item()) if torch.is_tensor(loss) else float(loss),
        }
        if per_sample_loss is not None:
            psl = per_sample_loss.float().cpu()
            step_record["per_sample_loss"] = {
                "mean": float(psl.mean()),
                "min": float(psl.min()),
                "max": float(psl.max()),
                "n": int(psl.numel()),
            }
        if sample_weights is not None and sample_weights.numel() > 0:
            sw = sample_weights.float().cpu()
            step_record["sample_weights"] = {
                "mean": float(sw.mean()),
                "min": float(sw.min()),
                "max": float(sw.max()),
            }
        if labels is not None:
            unmasked = (labels != IGNORE_INDEX).sum(dim=-1).float().cpu()
            step_record["unmasked_tokens"] = {
                "mean": float(unmasked.mean()),
                "max": float(unmasked.max()),
            }
        self._audit_step_writer.write(step_record)

        # Per-sample stream — needs sample_meta from collator
        if (
            self._audit_sample_writer is not None
            and sample_meta
            and per_sample_loss is not None
        ):
            psl_list = per_sample_loss.float().cpu().tolist()
            sw_list = (
                sample_weights.float().cpu().tolist()
                if sample_weights is not None and sample_weights.numel() > 0
                else [1.0] * len(psl_list)
            )
            for i, meta in enumerate(sample_meta):
                if i >= len(psl_list):
                    break
                self._audit_sample_writer.write({
                    "step": self.state.global_step,
                    "sample_id": meta.get("sample_id"),
                    "video_id": meta.get("video_id"),
                    "chunk_idx": meta.get("chunk_idx"),
                    "sample_type": meta.get("sample_type"),
                    "action": meta.get("action"),
                    "sequence_type": meta.get("sequence_type"),
                    "loss": psl_list[i],
                    "weight": sw_list[i] if i < len(sw_list) else 1.0,
                })

    def _current_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return -1.0

    # -----------------------------------------------------------------
    # Eval-time argmax accuracy (teacher-forced) — see data_processor's
    # _extract_eval_positions. Tracks per-class:
    #   - action_match / action_total  → eval/action_acc[_<class>]
    #   - post_match   / post_total    → eval/post_action_acc[_<class>]
    # silent_eos_rate is just post_action_acc filtered to silent samples.
    # -----------------------------------------------------------------

    def _reset_eval_accumulator(self):
        self._eval_acc = {
            "action_match": defaultdict(int),
            "action_total": defaultdict(int),
            "post_match":   defaultdict(int),
            "post_total":   defaultdict(int),
        }

    def _accumulate_eval_argmax(self, logits, input_ids, eval_meta) -> None:
        """Teacher-forced argmax match at known structural positions."""
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (B, L)
            B, L = preds.shape
            for b, meta in enumerate(eval_meta):
                if not meta:
                    continue
                stype = meta.get("sample_type", "?") or "?"

                kw_positions = meta.get("action_keyword_positions") or []
                if kw_positions:
                    self._eval_acc["action_total"][stype] += 1
                    self._eval_acc["action_total"]["_all"] += 1
                    all_match = True
                    for p in kw_positions:
                        # logits[p-1] predicts token at position p
                        if p <= 0 or p >= L:
                            all_match = False
                            break
                        if preds[b, p - 1].item() != int(input_ids[b, p].item()):
                            all_match = False
                            break
                    if all_match:
                        self._eval_acc["action_match"][stype] += 1
                        self._eval_acc["action_match"]["_all"] += 1

                pp = meta.get("post_action_position")
                if pp is not None and 0 < pp < L:
                    self._eval_acc["post_total"][stype] += 1
                    self._eval_acc["post_total"]["_all"] += 1
                    if preds[b, pp - 1].item() == int(input_ids[b, pp].item()):
                        self._eval_acc["post_match"][stype] += 1
                        self._eval_acc["post_match"]["_all"] += 1

    def _all_reduce_eval_acc(self) -> None:
        """Sum per-rank counters across DDP world. No-op if not distributed."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        device = next(self.model.parameters()).device
        # Union of keys across ranks (each rank may have seen different sample types)
        local_keys = set()
        for d in self._eval_acc.values():
            local_keys.update(d.keys())
        # Gather union across ranks
        gathered: list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, sorted(local_keys))
        all_keys = sorted({k for ks in gathered for k in ks})
        if not all_keys:
            return
        # Pack 4 counters × len(all_keys) into one tensor for a single all_reduce
        n = len(all_keys)
        buf = torch.zeros(4 * n, dtype=torch.long, device=device)
        bands = ["action_match", "action_total", "post_match", "post_total"]
        for bi, band in enumerate(bands):
            d = self._eval_acc[band]
            for ki, k in enumerate(all_keys):
                buf[bi * n + ki] = int(d.get(k, 0))
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        for bi, band in enumerate(bands):
            self._eval_acc[band] = defaultdict(int)
            for ki, k in enumerate(all_keys):
                self._eval_acc[band][k] = int(buf[bi * n + ki].item())

    def _finalize_eval_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # Overall action accuracy
        tot_a = self._eval_acc["action_total"].get("_all", 0)
        if tot_a > 0:
            out["eval/action_accuracy"] = (
                self._eval_acc["action_match"].get("_all", 0) / tot_a
            )
        # Per-class action accuracy
        for stype, tot in self._eval_acc["action_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/action_acc_{stype}"] = (
                self._eval_acc["action_match"].get(stype, 0) / tot
            )
        # Per-class post-action transition accuracy (silent_eos_rate is
        # the silent slice — surfaced as its own metric for visibility).
        for stype, tot in self._eval_acc["post_total"].items():
            if stype == "_all" or tot == 0:
                continue
            acc = self._eval_acc["post_match"].get(stype, 0) / tot
            out[f"eval/post_action_acc_{stype}"] = acc
            if stype == "silent":
                out["eval/silent_eos_rate"] = acc
        return out

    def evaluate(self, *args, **kwargs):
        """Wrap HF eval to inject custom argmax-accuracy metrics into wandb.

        Trainer.evaluate() already calls self.log(metrics) inside before
        returning, so we add our extras AFTER super() and re-log them so
        wandb picks up `eval/action_accuracy`, `eval/silent_eos_rate`, etc.
        """
        self._reset_eval_accumulator()
        metrics = super().evaluate(*args, **kwargs)
        self._all_reduce_eval_acc()
        extra = self._finalize_eval_metrics()
        if extra:
            # Stamp with the same global_step the eval loop just used so
            # the extras show up on the same wandb x-axis tick as eval_loss.
            self.log(extra)
            metrics.update(extra)
        return metrics


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
